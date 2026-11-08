"""角色中心「试玩验收台」的只读资产扫描。

设计约束（很重要，别改坏）：

* **不新增任何落盘数据**。帧全部来自现有 VNCCS 目录契约
  ``assets/characters/<角色>/Sprites/<服装>/<表情>/*.png``，
  fps / loop / pivot 从已有的 ``Atlas/atlas.json`` 读。没有 atlas 就给默认值，
  并在返回值里标明 ``source="default"``，不假装是验收过的值。
* **不假装已有方向资产**。老角色目录只有「服装 × 表情」两维；方向帧由
  ``posegen`` 按 ``set.json`` 的 direction 落成
  ``sprite_<direction>_NNNN.png``，也能来自画布项目里的多机位姿势集、或
  ``atlas.json`` 里带 direction 的动画组。三处都如实报告，缺的仍要列出来。
* 纯函数 ``collect_from`` 接受 roots，便于单测用 tmp_path，不依赖全局状态。

视角模式（与前端 ``web/src/store/playtest.ts`` 保持同一张表）：

===================  ==========================================================
topdown-4            俯视四方向 down / up / left / right
topdown-8            俯视八方向，多四个对角；双键向量合成
side-2               横板左右两个朝向 + 水平镜像派生，带地面 / 重力
===================  ==========================================================

八方向的 yaw 取自 ``web/src/components/PoseStudio.tsx`` 的 ``YAWS``
（0 正面 / 45 右前 / 90 右侧 / 135 右后 / 180 背面 / 225 左后 /
270 左侧 / 315 左前）。角度与方向名的对应按「相机绕 Y 顺时针」约定，
**第一次真机出八方向帧时要校准一次**，别把这个映射当成官方定义 —— 官方
VNCCS 没有原生八方向功能，见 ``docs/角色中心-剩余工作与做法.md`` §8。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import characters
from . import motiontemplates
from .directions import (
    DIRECTIONS, YAW_MAP, MIRROR_FROM, direction_keys,
    direction_from_name, validate_direction,
)
from .paths import ASSETS, PROJECTS_DIR

#: 归档目录（整批替换时旧帧进 V1/），枚举帧时必须跳过。
_ARCHIVE_RE = re.compile(r"^v\d+$", re.IGNORECASE)
_FRAME_TAIL = re.compile(r"(\d+)(?=\D*$)")

DEFAULT_FRAME_RATE = 8.0
DEFAULT_LOOP = True
DEFAULT_PIVOT = {"x": 0.5, "y": 1.0}


def _frame_sort_key(path: Path) -> tuple[int, str]:
    m = _FRAME_TAIL.search(path.stem)
    return (int(m.group(1)) if m else 0, path.name)


def _frame_files(folder: Path) -> List[Path]:
    """只取直接子文件里的 PNG，跳过 V1/V2 归档目录。"""
    if not folder.is_dir():
        return []
    out = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
        and not p.name.startswith(".")
    ]
    return sorted(out, key=_frame_sort_key)


def _asset_url(path: Path, assets_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(Path(assets_root).resolve())
    except (ValueError, OSError):
        return ""
    return "/files/" + "/".join(rel.parts)


def _read_atlas(root: Path) -> Dict[str, Any]:
    """读现有 atlas.json。没有 / 坏了就返回默认值并标 source。"""
    file = Path(root) / "Atlas" / "atlas.json"
    out: Dict[str, Any] = {
        "exists": False,
        "frameRate": DEFAULT_FRAME_RATE,
        "loop": DEFAULT_LOOP,
        "pivot": dict(DEFAULT_PIVOT),
        "canvas": [0, 0],
        "directions": [],
        "source": "default",
    }
    if not file.is_file():
        return out
    try:
        meta = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        out["broken"] = True
        return out
    info = meta.get("meta") or {}
    animations = meta.get("animations") or {}
    first = next(iter(animations.values()), {}) if isinstance(animations, dict) else {}
    canvas = info.get("canvas") or {}
    pivot = info.get("pivot") or {}
    dirs = {
        validate_direction(a.get("direction"))
        for a in (animations.values() if isinstance(animations, dict) else [])
        if isinstance(a, dict)
    }
    dirs.discard("")
    out.update({
        "exists": True,
        "frameRate": float(first.get("frameRate", DEFAULT_FRAME_RATE) or DEFAULT_FRAME_RATE),
        "loop": bool(first.get("loop", DEFAULT_LOOP)),
        "pivot": {
            "x": float(pivot.get("x", DEFAULT_PIVOT["x"])),
            "y": float(pivot.get("y", DEFAULT_PIVOT["y"])),
        },
        "canvas": [int(canvas.get("w", 0) or 0), int(canvas.get("h", 0) or 0)],
        "directions": sorted(dirs),
        "source": "atlas",
    })
    return out


def _direction_assets(root: Path) -> Dict[str, Any]:
    """扫角色目录里已经落盘的方向帧。

    认两种形态（都由 ``directions.py`` 的同一张白名单判定）：
      * ``Sprites/<服装>/<表情>/<方向>/*.png``（目录即方向）；
      * ``sprite_<方向>_0001.png``（posegen 按方向出帧时用的文件名）。

    归档 ``V1/`` 里的旧帧不算数 —— 整批替换后那已经是上一版素材。
    """
    present: set[str] = set()
    sprites = Path(root) / "Sprites"
    if not sprites.is_dir():
        return {"present": [], "convention": "none"}
    for path in sprites.rglob("*"):
        rel = path.relative_to(sprites)
        # 跳过 V1/V2 归档子树，以及隐藏的原子发布临时目录
        if any(_ARCHIVE_RE.match(p) or p.startswith(".") for p in rel.parts):
            continue
        key = (validate_direction(path.name) if path.is_dir()
               else direction_from_name(path.name))
        if key:
            present.add(key)
    return {
        "present": sorted(present),
        "convention": "directory-or-filename" if present else "none",
    }


def _project_direction_assets(projects_root: Path | None) -> Dict[str, Any]:
    """扫画布项目的姿势集：``set.json`` 里带 ``direction`` 的帧就是方向资产。

    人偶节点按 YAWS 多机位抽帧后，方向信息落在
    ``assets/projects/<项目>/poses/<集>/set.json`` 的每一帧上，
    而不是文件名里。角色中心验收台要把这些方向算作「已补出」，
    否则用户明明抽了八方向，验收台还一直报缺 8 向。
    """
    empty: Dict[str, Any] = {"present": [], "sets": [], "templates": []}
    if not projects_root:
        return empty
    base = Path(projects_root)
    if not base.is_dir():
        return empty
    present: set[str] = set()
    sets: List[Dict[str, Any]] = []
    templates: Dict[str, Dict[str, Any]] = {}
    for set_file in sorted(base.glob("*/poses/*/set.json")):
        try:
            doc = json.loads(set_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
        tid = str(source.get("templateId") or "")
        # 抽帧面板选的模板留在 set.json.source 里：验收台要能说清
        # 「这些方向帧是按哪套动作模板、多少 fps 抽的」，并给出成本估算。
        if tid and tid not in templates:
            tpl = motiontemplates.get_template(tid)
            if tpl:
                pb = motiontemplates.playback(tpl)
                templates[tid] = {
                    "id": tid,
                    "name": tpl["name"],
                    "viewMode": tpl["viewMode"],
                    "directions": tpl["directions"],
                    "actionCount": len(tpl["actions"]),
                    "fps": pb["fps"],
                    "loop": pb["loop"],
                    "estimate": motiontemplates.estimate(tpl),
                    "project": set_file.parents[2].name,
                    "poseSet": set_file.parent.name,
                }
        frames = doc.get("frames") or []
        dirs = sorted({
            str(f["direction"]) for f in frames
            if isinstance(f, dict) and f.get("direction")
        })
        if not dirs:
            continue
        present.update(dirs)
        sets.append({
            "project": set_file.parents[2].name,
            "name": set_file.parent.name,
            "directions": dirs,
            "frames": len(frames),
            "path": str(set_file.parent),
            "templateId": tid,
        })
    return {"present": sorted(present), "sets": sets,
            "templates": list(templates.values())}


def _readiness(mode: str, present: Iterable[str], clip_count: int) -> Dict[str, Any]:
    have = set(present)
    required = direction_keys(mode)
    missing: List[str] = []
    mirrored: List[str] = []
    for key in required:
        if key in have:
            continue
        src = MIRROR_FROM.get(key)
        if mode == "side-2" and src and src in have:
            mirrored.append(key)
        else:
            missing.append(key)
    if clip_count == 0:
        note = "这个角色还没有精灵帧：先回步骤 ③ 出素材。"
    elif not missing:
        note = "方向资产齐全，可以按当前视角模式验收并导出。"
    elif have & set(required):
        note = (f"已有 {len(have & set(required))}/{len(required)} 个方向帧；"
                "剩下的按白模机位补出，或先用同一组帧占位试玩。")
    else:
        note = ("现有角色目录只有「服装 × 表情」两维，没有方向维；"
                "可用同一组帧占位试玩，但要进游戏需按模板补出各方向帧。")
    return {
        "ready": clip_count > 0 and not missing,
        "required": required,
        "missing": missing,
        "mirrored": mirrored,
        "yawMap": {k: YAW_MAP[k] for k in required},
        "hasFrames": clip_count > 0,
        "note": note,
    }


def collect_from(
    root: Path,
    name: str,
    assets_root: Path | None = None,
    projects_root: Path | None = None,
) -> Dict[str, Any]:
    """扫一个角色目录，产出试玩页需要的全部数据（纯函数、只读）。

    ``projects_root`` 传入时额外扫画布项目的多机位姿势集，把已抽出的方向
    算进就绪判定；不传（单测 / 隔离环境）就只看角色目录本身。
    """
    base = Path(root)
    assets = Path(assets_root) if assets_root is not None else ASSETS
    atlas = _read_atlas(base)
    clips: List[Dict[str, Any]] = []
    sprites = base / "Sprites"
    if sprites.is_dir():
        # 默认服装 / 默认表情排最前：试玩页默认选中第一组，别一打开就落到
        # 某个冷门换装上。目录大小写两种都要认（官方 Neutral / 我们 neutral）。
        costumes = sorted(
            (p for p in sprites.iterdir() if p.is_dir()),
            key=lambda p: (0 if p.name in ("Naked", "Original") else 1, p.name.lower()),
        )
        for costume_dir in costumes:
            emotions = sorted(
                (p for p in costume_dir.iterdir() if p.is_dir()),
                key=lambda p: (0 if p.name.lower() == "neutral" else 1, p.name.lower()),
            )
            for emo_dir in emotions:
                if _ARCHIVE_RE.match(emo_dir.name):
                    continue
                frames = _frame_files(emo_dir)
                if not frames:
                    continue
                # 方向帧按方向分组：八方向的 8 组帧绝不能串成一条动画播，
                # 否则人在原地疯狂转圈。老目录没有方向帧时 by_dir 为空，
                # frames 保持原来的「整组帧」语义，行为一字不变。
                by_dir: Dict[str, List[str]] = {}
                for p in frames:
                    key = direction_from_name(p.name)
                    if key:
                        by_dir.setdefault(key, []).append(_asset_url(p, assets))
                default_dir = next(
                    (d for d in DIRECTIONS["topdown-8"] if by_dir.get(d)), "")
                clips.append({
                    "id": f"{costume_dir.name}/{emo_dir.name}",
                    "label": f"{costume_dir.name} · {emo_dir.name}",
                    "costume": costume_dir.name,
                    "emotion": emo_dir.name,
                    "frames": (by_dir.get(default_dir) if default_dir else None)
                              or [_asset_url(p, assets) for p in frames],
                    "directionFrames": by_dir,
                    "directions": sorted(by_dir),
                    "defaultDirection": default_dir,
                    "totalFrames": len(frames),
                    "frameRate": atlas["frameRate"],
                    "loop": atlas["loop"],
                })
    found = _direction_assets(base)
    proj = _project_direction_assets(projects_root)
    # atlas.json 里带 direction 的动画组也算「已经打包好的方向资产」，
    # 但不覆盖角色目录本身的判定（convention 仍然只描述落盘帧）。
    all_present = sorted(
        set(found["present"]) | set(proj["present"]) | set(atlas["directions"]))
    return {
        "character": name,
        "clips": clips,
        "clipCount": len(clips),
        "frameCount": sum(c["totalFrames"] for c in clips),
        "atlas": atlas,
        "directions": {
            "present": all_present,
            "characterDirs": found["present"],
            "projectDirs": proj["present"],
            "atlasDirs": atlas["directions"],
            "poseSets": proj["sets"],
            "templates": proj["templates"],
            "convention": found["convention"],
        },
        "viewModes": {
            mode: _readiness(mode, all_present, len(clips))
            for mode in DIRECTIONS
        },
    }


def collect(name: str) -> Dict[str, Any]:
    """按角色名扫描（供 app.py 调用）。"""
    return collect_from(
        Path(characters.character_dir(name)), name,
        projects_root=PROJECTS_DIR,
    )
