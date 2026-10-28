"""LoRA 注册表：扫描 models/lora/*.safetensors，合并人工元数据。

为什么需要元数据：Civitai 下下来的 safetensors 里只有权重，没有「触发词」和
「推荐强度」。触发词不知道就等于白装（实测 xuer 的触发词是 `xuer martial arts`，
不写这个词它只输出普通照片）。所以元数据落在同目录的 `loras.json`，
以文件名为键，缺失时用保守默认值。

清单是**可选**的：文件存在就能用，元数据只是改良。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

LORA_DIR = "lora"
META_FILE = "loras.json"

_DEFAULT: Dict[str, Any] = {
    "label": None,
    "trigger": "",
    "scale": 0.8,
    "preset": None,
    "note": "",
    "source": "",
    # 选了这个 LoRA 后节点该跟着改的开关。实测像素 LoRA 都会在背景上画
    # 彩色/纹理底，不抠图就没法用，所以允许注册表声明「必须抠图」。
    "matte": None,        # None=不改动用户的勾选；True/False=强制
    "clip_skip": None,    # None=用节点当前值；2=多数动漫/像素 LoRA 的训练设置
}


def scan(models_root: str | Path) -> List[Dict[str, Any]]:
    """列出可用 LoRA（按 label 排序）。路径相对于 models 根目录。"""
    root = Path(models_root)
    d = root / LORA_DIR
    if not d.is_dir():
        return []

    meta: Dict[str, Any] = {}
    mf = d / META_FILE
    if mf.exists():
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    out: List[Dict[str, Any]] = []
    for f in sorted(d.glob("*.safetensors")):
        info = dict(_DEFAULT)
        info.update(meta.get(f.stem) or {})
        info["name"] = f.stem
        info["label"] = info["label"] or f.stem
        info["path"] = f"{LORA_DIR}/{f.name}"
        info["size_mb"] = round(f.stat().st_size / 2**20, 1)
        out.append(info)

    return sorted(out, key=lambda x: (x["label"] or "").lower())


def resolve(models_root: str | Path, name: str | None) -> str | None:
    """把 LoRA 名（stem 或相对路径）解析成绝对路径；空值返回 None。"""
    if not name:
        return None
    root = Path(models_root)
    cand = root / name if "/" in name or "\\" in name else root / LORA_DIR / f"{name}.safetensors"
    if cand.exists():
        return str(cand)
    # 也接受直接给绝对路径
    p = Path(name)
    return str(p) if p.exists() else None
