"""用户资料库：提示词收藏 / 姿势库 / 界面偏好 —— 全部落在服务端磁盘。

**为什么必有这一层**（用户明确要求：localStorage 里一个字都不留）：

- 浏览器存储会在「换浏览器 / 清站点数据 / 无痕窗口」时凭空消失，而提示词收藏和
  姿势库是用户攒了很久的资产，丢一次就再也找不回来；
- 同一个姿势要在十个角色上复用 —— 它属于**用户**而不是某个项目，跟项目 JSON
  放一起会让每个工程复制一份；
- 落盘后能备份、能拷走换机器、能用 git 管版本。

文件布局（`library/` 目录，和 `templates/` 同级 —— 都是跨项目共享的用户资源）：

    library/
    ├── prompts.json   提示词收藏（正向 + 负向）
    ├── poses.json     姿势库（官方 pose JSON + 预览图）
    └── prefs.json     界面偏好（蒙版颜色等）

**写入原子**（临时文件 + `os.replace`）：收藏是高频操作，半截 JSON 会让整个库
打不开，不能冒这个险。

**脏数据是常态**：手改过、老版本写过、别的工具生成过。所以读一律容错 ——
文件坏了当作空库返回，不让一个坏文件把整个应用拖到 500。
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflows import paths

PROMPTS_FILE = "prompts.json"
POSES_FILE = "poses.json"
PREFS_FILE = "prefs.json"

#: 上限。再多人肉也翻不完，也顺手防止一句话 bug 把文件写到几百 MB。
MAX_PROMPTS = 500
MAX_POSES = 200
#: 姿势预览图的 base64 上限（约 1.5MB）。预览是可选的，超了就弃掉，
#: 不能让一条姿势把整个库文件撑爆。
MAX_PREVIEW_BYTES = 1_500_000

_LOCK = threading.RLock()


# ---------------------------------------------------------------- 基础读写


def _path(name: str) -> Path:
    return paths.LIBRARY_DIR / name


def _read_list(name: str) -> List[Dict[str, Any]]:
    p = _path(name)
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return []
    if isinstance(doc, dict):            # 老格式可能被包了一层 {"items": [...]}
        doc = doc.get("items")
    return doc if isinstance(doc, list) else []


def _read_dict(name: str) -> Dict[str, Any]:
    p = _path(name)
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _write(name: str, doc: Any) -> None:
    """原子写一个 JSON 文件。"""
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    with _LOCK:
        os.replace(str(tmp), str(p))


def _now_id(prefix: str) -> str:
    return f"{prefix}{int(time.time() * 1000):x}{secrets.token_hex(2)}"


def _norm_angle(deg: Any) -> int:
    """角度 → 0~359 的整数。姿势库里的朝向是渲染机位，负数/浮点没有意义。"""
    try:
        n = int(round(float(deg)))
    except (TypeError, ValueError):
        return 0
    return ((n % 360) + 360) % 360


# ---------------------------------------------------------------- 提示词收藏


def load_prompts() -> List[Dict[str, Any]]:
    """读提示词收藏，按收藏时间倒序。"""
    out: List[Dict[str, Any]] = []
    for x in _read_list(PROMPTS_FILE):
        if not isinstance(x, dict):
            continue
        text = str(x.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "id": str(x.get("id") or text),
            "text": text,
            "negative": str(x.get("negative") or "").strip(),
            "at": int(x.get("at") or 0),
        })
    out.sort(key=lambda p: p.get("at") or 0, reverse=True)
    return out


def _save_prompts(items: List[Dict[str, Any]]) -> None:
    _write(PROMPTS_FILE, items[:MAX_PROMPTS])


def _prompt_key(text: str, negative: str) -> str:
    return f"{text.strip()}\u0000{negative.strip()}"


def has_prompt(text: str, negative: str = "") -> bool:
    k = _prompt_key(text, negative)
    return any(_prompt_key(p["text"], p.get("negative", "")) == k
               for p in load_prompts())


def add_prompt(text: str, negative: str = "",
               items: Optional[List[Dict[str, Any]]] = None
               ) -> List[Dict[str, Any]]:
    """收藏一条提示词。正向+负向都一样才算重复（重复则原样返回）。"""
    cur = items if items is not None else load_prompts()
    t = (text or "").strip()
    if not t:
        return cur
    n = (negative or "").strip()
    k = _prompt_key(t, n)
    if any(_prompt_key(p.get("text", ""),
                       p.get("negative", "")) == k for p in cur):
        return cur
    out = ([{"id": _now_id("p"), "text": t, "negative": n,
             "at": int(time.time() * 1000)}] + cur)[:MAX_PROMPTS]
    _save_prompts(out)
    return out


def remove_prompt(pid: str, items: Optional[List[Dict[str, Any]]] = None
                  ) -> List[Dict[str, Any]]:
    cur = items if items is not None else load_prompts()
    out = [p for p in cur if p.get("id") != pid]
    if len(out) != len(cur):
        _save_prompts(out)
    return out


# ---------------------------------------------------------------- 姿势库


def load_poses() -> List[Dict[str, Any]]:
    """读姿势库，按存入时间倒序。

    `pose` 是官方 PoseStudio 的 `getPose()` JSON —— 骨骼欧拉角 + IK effector +
    极点。**我们不解析也不改它**：格式归官方版本号管，解析了就等于承诺"永远跟上
    它的数据结构"，而存 JSON 的目的是精确还原，中间插一道只会丢精度。
    """
    out: List[Dict[str, Any]] = []
    for x in _read_list(POSES_FILE):
        if not isinstance(x, dict):
            continue
        pose = x.get("pose")
        if not isinstance(pose, (dict, list)):
            continue
        prev = x.get("preview")
        if isinstance(prev, str) and len(prev) > MAX_PREVIEW_BYTES:
            prev = ""                      # 预览图过大就丢掉，别撑爆文件
        out.append({
            "id": str(x.get("id") or x.get("name") or "pose"),
            "name": str(x.get("name") or "未命名").strip() or "未命名",
            "pose": pose,
            "yaw": _norm_angle(x.get("yaw")),
            "preview": prev or "",
            "at": int(x.get("at") or 0),
        })
    out.sort(key=lambda p: p.get("at") or 0, reverse=True)
    return out


def _save_poses(items: List[Dict[str, Any]]) -> None:
    _write(POSES_FILE, items[:MAX_POSES])


def add_pose(name: str, pose: Any, yaw: Any = 0, preview: str = "",
             items: Optional[List[Dict[str, Any]]] = None
             ) -> List[Dict[str, Any]]:
    """存一条姿势。**同名覆盖**：用户改完胳膊想原地更新，不该翻出两条。"""
    cur = items if items is not None else load_poses()
    nm = (name or "").strip() or "姿势"
    if not isinstance(pose, (dict, list)):
        raise ValueError("pose 必须是官方 getPose() 的 JSON 对象")
    pv = preview if isinstance(preview, str) else ""
    if len(pv) > MAX_PREVIEW_BYTES:
        pv = ""
    item = {
        "id": _now_id("z"), "name": nm, "pose": pose,
        "yaw": _norm_angle(yaw), "preview": pv,
        "at": int(time.time() * 1000),
    }
    # 同名 → **原地替换**（保留原 id 与位置），否则插到最前
    for i, x in enumerate(cur):
        if x.get("name") == nm:
            cur = list(cur)
            cur[i] = {**item, "id": x.get("id") or item["id"],
                      "at": int(time.time() * 1000)}
            out = cur
            break
    else:
        out = [item] + cur
    _save_poses(out)
    return out


def add_poses(batch: List[Dict[str, Any]],
              items: Optional[List[Dict[str, Any]]] = None
              ) -> List[Dict[str, Any]]:
    """一次存多条（从一条动画抽出来的 N 帧）。

    **一次落盘**：逐条调 `add_pose` 会写 N 次文件，一条 walk 抽 16 帧就是 16 次
    整库重写（每帧还带一张预览图），没必要。这里按顺序合并完只写一次。

    规则和 `add_pose` 一致：同名**原地覆盖**（保留原 id），否则插到最前。
    """
    cur = list(items if items is not None else load_poses())
    now = int(time.time() * 1000)
    for k, raw in enumerate(batch or []):
        if not isinstance(raw, dict):
            continue
        pose = raw.get("pose")
        if not isinstance(pose, (dict, list)):
            raise ValueError("pose 必须是官方 getPose() 的 JSON 对象")
        nm = str(raw.get("name") or "").strip() or "姿势"
        pv = raw.get("preview") if isinstance(raw.get("preview"), str) else ""
        if len(pv) > MAX_PREVIEW_BYTES:
            pv = ""
        item = {
            "id": _now_id("z"), "name": nm, "pose": pose,
            "yaw": _norm_angle(raw.get("yaw")), "preview": pv,
            "at": now + k,
        }
        for i, x in enumerate(cur):
            if x.get("name") == nm:
                cur[i] = {**item, "id": x.get("id") or item["id"]}
                break
        else:
            cur.insert(0, item)
    _save_poses(cur)
    return cur


def rename_pose(pid: str, name: str,
                items: Optional[List[Dict[str, Any]]] = None
                ) -> List[Dict[str, Any]]:
    cur = items if items is not None else load_poses()
    nm = (name or "").strip()
    if not nm:
        return cur
    # 名字被别人占了 → 不动。硬改名会凭空造出两条同名姿势，删的时候分不清是谁。
    if any(x.get("name") == nm and x.get("id") != pid for x in cur):
        return cur
    out = [{**x, "name": nm} if x.get("id") == pid else x for x in cur]
    if any(a is not b for a, b in zip(cur, out)):
        _save_poses(out)
    return out


def remove_pose(pid: str, items: Optional[List[Dict[str, Any]]] = None
                ) -> List[Dict[str, Any]]:
    cur = items if items is not None else load_poses()
    out = [p for p in cur if p.get("id") != pid]
    if len(out) != len(cur):
        _save_poses(out)
    return out


# ---------------------------------------------------------------- 偏好


def load_prefs() -> Dict[str, Any]:
    return {str(k): v for k, v in _read_dict(PREFS_FILE).items()}


def save_prefs(patch: Dict[str, Any]) -> Dict[str, Any]:
    """合并写偏好（只覆盖传进来的键）。"""
    cur = load_prefs()
    cur.update({str(k): v for k, v in (patch or {}).items()})
    _write(PREFS_FILE, cur)
    return cur


# ---------------------------------------------------------------- 迁移


def seed_library(prompts: Optional[List[Dict[str, Any]]] = None,
                 poses: Optional[List[Dict[str, Any]]] = None,
                 prefs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把浏览器 localStorage 里的旧数据搬进磁盘（**只补不覆盖**）。

    为什么是"补"：用户可能在两个浏览器里都攒过东西，来回切的时候哪边都不能丢。
    返回迁移后的完整库，前端拿到就能直接替换本地状态。
    """
    if isinstance(prompts, list):
        cur = load_prompts()
        for x in reversed(prompts):
            if isinstance(x, dict) and str(x.get("text") or "").strip():
                cur = add_prompt(str(x.get("text")),
                                 str(x.get("negative") or ""), cur)
    if isinstance(poses, list):
        cur_p = load_poses()
        for x in reversed(poses):
            if isinstance(x, dict) and isinstance(x.get("pose"), (dict, list)):
                cur_p = add_pose(str(x.get("name") or "姿势"), x.get("pose"),
                                 x.get("yaw"), x.get("preview") or "", cur_p)
    if isinstance(prefs, dict) and prefs:
        cur_prefs = load_prefs()
        # 已有的不动（**当前这份是用户在用的**，旧的是别处漂来的备份）
        for k, v in prefs.items():
            cur_prefs.setdefault(str(k), v)
        _write(PREFS_FILE, cur_prefs)
    return snapshot()


def snapshot() -> Dict[str, Any]:
    """整个库一次给全：冷启动只需一次请求。"""
    return {
        "prompts": load_prompts(),
        "poses": load_poses(),
        "prefs": load_prefs(),
    }
