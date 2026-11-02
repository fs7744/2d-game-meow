"""项目内的**姿势集**与**成片集**（落在画布项目的文件夹下）。

**为什么不进用户资料库（`library/poses.json`）**：那是"跨项目复用的单个姿势"，
存的是官方 pose JSON + 一张缩略图。而这里要存的是**一整条动作抽出来的 N 帧**：
它属于某一次动作采样的产物，跟项目、跟那条动画绑定，换项目复用它没意义，
而且 N 张 768² 的人偶图塞进 poses.json 会把库文件撑爆。

目录（`assets/projects/<项目 id>/` 下，和 project.json、产物图在一起）：

    <项目>/
    ├── project.json
    ├── poses/<姿势集名>/           ← 人偶编辑器「存入姿势集」
    │   ├── set.json                （帧表：序号 / 时间 / 朝向 / 官方 pose JSON / 文件名）
    │   ├── f_001.png …
    │   └── f_008.png
    └── pose_frames/<成片集名>/      ← 角色迁移「批量生成」
        ├── render.json             （每帧的候选表 + 用户选了哪一个）
        ├── f_001_c01.png …         （**候选永不删除**）
        └── f_008_c02.png

**候选只增不删**是硬要求：用户重新生成某一帧之后，可能觉得上一版更好。
所以每一帧存的是 `candidates[]` + `selected`（下标），重新生成 = 追加一个候选
并选中它，旧文件原地留着，点一下就能选回去。

**名字不能重复**：同项目下同名会覆盖前一次的产物，而用户的心智是"再存一份"，
不是"替换"。所以创建时撞名直接报 409，由前端提示换个名字。
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflows import characters
from workflows import paths
from workflows import projectstore

#: 项目目录下的两个子目录
POSES_DIR = "poses"
RENDERS_DIR = "pose_frames"
#: 元数据文件名
SET_FILE = "set.json"
RENDER_FILE = "render.json"

#: 单次存入的帧数上限。动作抽帧给的是 2~200 的量级，再多就该拆成几条了。
MAX_FRAMES = 240
#: 单帧 PNG 的 base64 上限（768² 白底人偶约 60~120KB，给到 4MB 是防呆）
MAX_FRAME_B64 = 4 * 1024 * 1024
#: 一帧的候选上限（每张约 25s，再多就等太久了）
MAX_CANDIDATES = 8

_LOCK = threading.RLock()
#: 创建时间戳。**必须严格递增**：同一秒内连存两个集时，`created` 相同会让
#: "新的在前"的排序退化成目录遍历顺序，用户看到的列表顺序会莫名跳。
_LAST_STAMP = 0


def _stamp() -> int:
    global _LAST_STAMP
    now = int(time.time() * 1000)
    if now <= _LAST_STAMP:
        now = _LAST_STAMP + 1
    _LAST_STAMP = now
    return now


class PoseSetError(RuntimeError):
    """可读错误（API 转 400/409）。"""


class PoseSetExists(PoseSetError):
    """同项目下已有同名文件夹。"""


# ---------------------------------------------------------------- 基础


def _project_root(project: str) -> Path:
    """项目目录。

    **不调 `generate.scope_dir`**：那个模块会 import torch（几十秒），而这里只是
    拼路径、建目录。清洗规则仍走 `projectstore.safe_scope` 这唯一一份实现，
    否则产物和 `project.json` 会因两套规则算出不同目录名而分家。
    """
    d = paths.PROJECTS_DIR / projectstore.safe_scope(project)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sub(project: str, kind: str, name: str) -> Path:
    """`<项目>/<kind>/<name>`，并挡住路径穿越。"""
    root = _project_root(project)
    safe = characters.ensure_safe_name(name, "名字")
    target = Path(characters.safe_join_under(str(root), kind, safe))
    return target


def _write_json(path: Path, doc: Dict[str, Any]) -> None:
    """原子写（半截 JSON 会让整个集打不开）。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _url(path: Path) -> str:
    """磁盘路径 → `/files/...`（前端 <img> 直接用）。"""
    try:
        rel = path.resolve().relative_to(paths.ASSETS.resolve())
    except (ValueError, OSError):
        return ""
    return "/files/" + rel.as_posix()


def _decode_png(raw: str) -> bytes:
    """dataURL 或裸 base64 → PNG 字节。"""
    s = str(raw or "").strip()
    if not s:
        raise PoseSetError("帧数据是空的")
    if s.startswith("data:"):
        comma = s.find(",")
        if comma < 0:
            raise PoseSetError("帧数据的 dataURL 格式不对")
        s = s[comma + 1:]
    if len(s) > MAX_FRAME_B64:
        raise PoseSetError(f"单帧图片太大（base64 超过 {MAX_FRAME_B64 // 1024}KB）")
    try:
        data = base64.b64decode(s, validate=False)
    except (ValueError, TypeError) as exc:
        raise PoseSetError("帧数据不是合法的 base64") from exc
    if not data.startswith(b"\x89PNG"):
        raise PoseSetError("帧数据不是 PNG")
    return data


def _exists_name(root: Path, name: str) -> bool:
    d = root / name
    return d.is_dir() and any(d.iterdir())


# ---------------------------------------------------------------- 姿势集


def save_pose_set(project: str, name: str, frames: List[Dict[str, Any]],
                  source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """新建一个姿势集文件夹（**同名直接报错，不覆盖**）。

    `frames`：`[{t, yaw, pose, png}]`，`png` 是 dataURL / base64。
    """
    if not isinstance(frames, list) or not frames:
        raise PoseSetError("没有要存的帧")
    if len(frames) > MAX_FRAMES:
        raise PoseSetError(f"一次最多存 {MAX_FRAMES} 帧（收到 {len(frames)}）")

    target = _sub(project, POSES_DIR, name)
    if target.exists() and any(target.iterdir()):
        raise PoseSetExists(f"项目里已经有同名的姿势集「{name}」，换个名字")

    parsed = []
    for i, item in enumerate(frames, 1):
        if not isinstance(item, dict):
            raise PoseSetError(f"第 {i} 帧的格式不对")
        parsed.append({
            "index": i,
            "t": float(item.get("t") or 0),
            "yaw": int(item.get("yaw") or 0),
            "pose": item.get("pose"),
            "png": _decode_png(item.get("png")),
        })

    with _LOCK:
        # 先写到一个临时目录，全部成功了再改名 —— 中途失败不会留下半套帧
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            entries = []
            for item in parsed:
                fn = f"f_{item['index']:03d}.png"
                (tmp / fn).write_bytes(item["png"])
                entries.append({
                    "index": item["index"],
                    "t": item["t"],
                    "yaw": item["yaw"],
                    "pose": item["pose"],
                    "file": fn,
                })
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            os.replace(tmp, target)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        doc = {
            "name": target.name,
            "created": _stamp(),
            "source": source or {},
            "frames": entries,
        }
        _write_json(target / SET_FILE, doc)
        return _set_summary(target, doc)


def _set_summary(folder: Path, doc: Dict[str, Any]) -> Dict[str, Any]:
    frames = doc.get("frames") or []
    cover = ""
    if frames:
        cover = _url(folder / str(frames[0].get("file") or ""))
    return {
        "name": doc.get("name") or folder.name,
        "count": len(frames),
        "cover": cover,
        "created": int(doc.get("created") or 0),
        "source": doc.get("source") or {},
    }


def list_pose_sets(project: str) -> List[Dict[str, Any]]:
    """项目里的全部姿势集（新的在前）。"""
    root = _project_root(project) / POSES_DIR
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        doc = _read_json(d / SET_FILE)
        if not doc.get("frames"):
            continue
        out.append(_set_summary(d, doc))
    out.sort(key=lambda x: x.get("created") or 0, reverse=True)
    return out


def load_pose_set(project: str, name: str) -> Dict[str, Any]:
    """一个姿势集的帧表（带 URL）。**不回传 pose JSON**：前端只用来出图和翻看，
    一份 17 根骨的 JSON 乘 30 帧会把响应撑到几百 KB，而它只在重新编辑时才需要。
    """
    folder = _sub(project, POSES_DIR, name)
    doc = _read_json(folder / SET_FILE)
    if not doc.get("frames"):
        raise PoseSetError(f"没有这个姿势集：{name}")
    frames = []
    for f in doc["frames"]:
        frames.append({
            "index": int(f.get("index") or 0),
            "t": float(f.get("t") or 0),
            "yaw": int(f.get("yaw") or 0),
            "url": _url(folder / str(f.get("file") or "")),
        })
    out = _set_summary(folder, doc)
    out["frames"] = frames
    return out


# ---------------------------------------------------------------- 成片集


def create_render(project: str, name: str, pose_set: str,
                  prompt: str = "", params: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """新建一个成片集文件夹（同名报错），帧表先建好空的。"""
    src = load_pose_set(project, pose_set)          # 不存在会抛
    target = _sub(project, RENDERS_DIR, name)
    if target.exists() and any(target.iterdir()):
        raise PoseSetExists(f"项目里已经有同名的成片集「{name}」，换个名字")

    doc = {
        "name": target.name,
        "created": _stamp(),
        "poseSet": src["name"],
        "prompt": prompt,
        "params": params or {},
        "frames": [
            {"index": f["index"], "source": f["url"], "candidates": [], "selected": -1}
            for f in src["frames"]
        ],
    }
    with _LOCK:
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / RENDER_FILE, doc)
    return load_render(project, name)


def load_render(project: str, name: str) -> Dict[str, Any]:
    """成片集详情：每帧的候选（带 URL）+ 当前选中。"""
    folder = _sub(project, RENDERS_DIR, name)
    doc = _read_json(folder / RENDER_FILE)
    if not doc:
        raise PoseSetError(f"没有这个成片集：{name}")
    frames = []
    for f in doc.get("frames") or []:
        cands = []
        for c in f.get("candidates") or []:
            cands.append({
                "index": int(c.get("index") or 0),
                "url": _url(folder / str(c.get("file") or "")),
                "at": int(c.get("at") or 0),
                "seed": c.get("seed"),
            })
        sel = int(f.get("selected") if f.get("selected") is not None else -1)
        frames.append({
            "index": int(f.get("index") or 0),
            "source": f.get("source") or "",
            "candidates": cands,
            "selected": sel,
            "selectedIndex": cands[sel]["index"] if 0 <= sel < len(cands) else -1,
        })
    return {
        "name": doc.get("name") or folder.name,
        "poseSet": doc.get("poseSet") or "",
        "prompt": doc.get("prompt") or "",
        "created": int(doc.get("created") or 0),
        "frames": frames,
        "done": sum(1 for f in frames if f["selectedIndex"] >= 0),
        "total": len(frames),
    }


def list_renders(project: str) -> List[Dict[str, Any]]:
    root = _project_root(project) / RENDERS_DIR
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        doc = _read_json(d / RENDER_FILE)
        if not doc:
            continue
        frames = doc.get("frames") or []
        cover = ""
        for f in frames:
            cands = f.get("candidates") or []
            sel = int(f.get("selected") if f.get("selected") is not None else -1)
            if 0 <= sel < len(cands):
                cover = _url(d / str(cands[sel].get("file") or ""))
                break
        out.append({
            "name": doc.get("name") or d.name,
            "poseSet": doc.get("poseSet") or "",
            "created": int(doc.get("created") or 0),
            "total": len(frames),
            "done": sum(1 for f in frames
                        if int(f.get("selected") if f.get("selected") is not None else -1) >= 0),
            "cover": cover,
        })
    out.sort(key=lambda x: x.get("created") or 0, reverse=True)
    return out


def _frame_entry(doc: Dict[str, Any], index: int) -> Dict[str, Any]:
    for f in doc.get("frames") or []:
        if int(f.get("index") or 0) == int(index):
            return f
    raise PoseSetError(f"成片集里没有第 {index} 帧")


def add_candidate(project: str, name: str, index: int, src: Path,
                  seed: Optional[int] = None) -> Dict[str, Any]:
    """把生成好的一张图**追加**成该帧的候选，并选中它。旧候选原地保留。"""
    folder = _sub(project, RENDERS_DIR, name)
    with _LOCK:
        doc = _read_json(folder / RENDER_FILE)
        if not doc:
            raise PoseSetError(f"没有这个成片集：{name}")
        frame = _frame_entry(doc, index)
        cands = frame.setdefault("candidates", [])
        if len(cands) >= MAX_CANDIDATES:
            raise PoseSetError(
                f"第 {index} 帧已经有 {len(cands)} 个候选（上限 {MAX_CANDIDATES}）")
        n = int(cands[-1]["index"]) + 1 if cands else 1
        fn = f"f_{int(index):03d}_c{n:02d}.png"
        dst = folder / fn
        try:
            shutil.move(str(src), dst)
        except OSError:
            shutil.copyfile(str(src), dst)
            try:
                os.remove(src)
            except OSError:
                pass
        # 落盘时顺手记下尺寸，前端排网格时不用等图加载
        size = [0, 0]
        try:
            from PIL import Image
            with Image.open(dst) as im:
                size = [im.width, im.height]
        except Exception:                      # noqa: BLE001
            pass
        cands.append({
            "index": n, "file": fn, "at": int(time.time() * 1000),
            "seed": seed, "size": size,
        })
        frame["selected"] = len(cands) - 1
        _write_json(folder / RENDER_FILE, doc)
    return load_render(project, name)


def select_candidate(project: str, name: str, index: int, candidate: int
                     ) -> Dict[str, Any]:
    """选第 N 个候选。**任何旧候选都还能选回来**（这就是不删它们的理由）。"""
    folder = _sub(project, RENDERS_DIR, name)
    with _LOCK:
        doc = _read_json(folder / RENDER_FILE)
        if not doc:
            raise PoseSetError(f"没有这个成片集：{name}")
        frame = _frame_entry(doc, index)
        cands = frame.get("candidates") or []
        pos = next((i for i, c in enumerate(cands)
                    if int(c.get("index") or 0) == int(candidate)), None)
        if pos is None:
            raise PoseSetError(f"第 {index} 帧没有第 {candidate} 个候选")
        frame["selected"] = pos
        _write_json(folder / RENDER_FILE, doc)
    return load_render(project, name)


def frame_source_path(project: str, pose_set: str, index: int) -> str:
    """取姿势集里第 index 帧的**磁盘路径**（生成时当条件图用）。"""
    folder = _sub(project, POSES_DIR, pose_set)
    doc = _read_json(folder / SET_FILE)
    for f in doc.get("frames") or []:
        if int(f.get("index") or 0) == int(index):
            p = folder / str(f.get("file") or "")
            if p.is_file():
                return str(p)
    raise PoseSetError(f"姿势集「{pose_set}」里没有第 {index} 帧")


def frame_indexes(project: str, pose_set: str) -> List[int]:
    folder = _sub(project, POSES_DIR, pose_set)
    doc = _read_json(folder / SET_FILE)
    return [int(f.get("index") or 0) for f in doc.get("frames") or []]
