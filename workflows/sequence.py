"""角色中心「序列帧动画」的归档目录（3D 人偶 → 抽帧 → 参考图批量帧）。

和「拆部件 / 骨骼动画」同一条规则：产物按**生命阶段**分目录，老数据不动。

    assets/characters/<角色>/Sequence[/s_<阶段>]/
        studio.json                 人偶编辑器的当前状态（pose / yaw / scene）
        poses/<姿势集名>/            抽帧存下来的整组人偶帧（set.json + f_*.png）
        pose_frames/<成片集名>/       批量生成的成片（render.json + 候选 PNG）
        exports/<名字>/               Godot 工程包（逐帧 PNG + .tres + .tscn）
        exports/<名字>.zip
        mannequin_<时间戳>.png        「导出人偶图」存下来的一张

**为什么不复用画布项目目录**：这一页的产物属于"这个角色的这个阶段"，不是某个
画布工程的临时素材；混在 `assets/projects/<id>/` 里，换画布就找不到了，也没法
在阶段页里按阶段列出历史。姿势集/成片集的实现仍复用 `workflows/poseset.py`
（同一套候选语义），这里只提供根目录与 studio 状态。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from workflows import characters
from workflows import paths
from workflows import stages as stages_mod

DIR_NAME = "Sequence"
STUDIO_FILE = "studio.json"
MANNEQUIN_PREFIX = "mannequin_"
#: studio.json 的上限。pose JSON + 场景 tab 通常几 KB，给到 4MB 是防呆。
MAX_STUDIO_BYTES = 4 * 1024 * 1024


class SequenceError(RuntimeError):
    """可读错误（API 转 400）。"""


def root(character: str, stage: str = "") -> Path:
    """这个角色 / 阶段的序列帧根目录（会建出来）。

    默认阶段用 `Sequence/` 根，其它阶段 `Sequence/s_<id>/` —— 与 `Parts/`、
    `RigAnim/` 完全一致，阶段切换时产物不会混在一起。
    """
    name = characters.ensure_safe_name(character, "角色名")
    base = paths.CHARACTERS_DIR / name / DIR_NAME
    sub = stages_mod.stage_dir_name(stage) if stage else ""
    d = base / sub if sub else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_studio(character: str, stage: str = "") -> Dict[str, Any]:
    """读人偶编辑器的状态；没有就返回空对象（老角色 / 第一次进来）。"""
    p = root(character, stage) / STUDIO_FILE
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def save_studio(character: str, stage: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """原子写 studio.json。只收 JSON 可序列化的字典（调用方负责 shape）。"""
    if not isinstance(doc, dict):
        raise SequenceError("studio 必须是对象")
    raw = json.dumps(doc, ensure_ascii=False)
    if len(raw.encode("utf-8")) > MAX_STUDIO_BYTES:
        raise SequenceError("人偶状态太大（超过 4MB）")
    p = root(character, stage) / STUDIO_FILE
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(raw, encoding="utf-8")
    os.replace(tmp, p)
    return doc


def save_mannequin(character: str, stage: str, data: bytes) -> Dict[str, Any]:
    """「导出人偶图」：白底灰人偶 PNG 收进本阶段，返回 URL。"""
    if not data:
        raise SequenceError("空文件")
    if not data.startswith(b"\x89PNG"):
        raise SequenceError("人偶图必须是 PNG")
    import time
    import uuid

    d = root(character, stage)
    name = f"{MANNEQUIN_PREFIX}{int(time.time())}_{uuid.uuid4().hex[:4]}.png"
    out = d / name
    out.write_bytes(data)
    rel = out.resolve().relative_to(paths.ASSETS.resolve()).as_posix()
    return {"path": str(out), "url": f"/files/{rel}", "name": name}


def _url(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(paths.ASSETS.resolve())
    except (ValueError, OSError):
        return ""
    return "/files/" + "/".join(rel.parts)


def list_mannequins(character: str, stage: str = "") -> list:
    """最近导出的人偶图（新的在前，最多 12 张）。"""
    d = root(character, stage)
    out = []
    for p in sorted(d.glob(f"{MANNEQUIN_PREFIX}*.png"), reverse=True)[:12]:
        out.append({"name": p.name, "url": _url(p), "mtime": int(p.stat().st_mtime)})
    return out


def summary(character: str, stage: str = "") -> Dict[str, Any]:
    """这一页顶部的概览：几套姿势集 / 几套成片 / 最近的人偶图。"""
    from workflows import poseset

    return {
        "root": DIR_NAME,
        "character": character,
        "stage": stage or "default",
        "poseSets": poseset.list_pose_sets("", character=character, stage=stage),
        "renders": poseset.list_renders("", character=character, stage=stage),
        "mannequins": list_mannequins(character, stage),
    }
