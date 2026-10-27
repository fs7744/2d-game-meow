"""确定性后处理：像素化、调色板量化、逐帧对齐、精灵图打包。

原则（沿用 pixelforge 的实测结论）：
1. 像素感必须由后处理保证，不能只靠提示词——模型只会画出「看起来像像素」的糊图。
2. 逐帧生成的角色会自己重新构图，必须按源帧锚点做**整体整数平移**拉回，
   否则落格后整段动画抖动。只平移、不缩放（缩放会抹掉下蹲/抬腿的真实大小变化）。
3. 打包装箱时单元格尺寸取全帧最大值，角色按「脚底中心」对齐。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


# ---------------- 像素化 / 调色板 ----------------

def pixelate(img: Image.Image, block: int) -> Image.Image:
    """按 block 像素栅格吸附：先降采样再最近邻放大回原尺寸。"""
    if block <= 1:
        return img
    w, h = img.size
    small = img.resize((max(1, w // block), max(1, h // block)), Image.LANCZOS)
    return small.resize((w, h), Image.NEAREST)


def quantize_palette(img: Image.Image, colors: int = 16) -> Image.Image:
    """压到固定色数（默认 16 色），保留 alpha。PIL 的中位切分，确定性可复现。"""
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    pal = rgba.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
    out = pal.convert("RGBA")
    out.putalpha(alpha.point(lambda a: 255 if a > 127 else 0))  # 硬边 alpha
    return out


def trim_alpha(img: Image.Image, pad: int = 0) -> Image.Image:
    """裁到 alpha 包围盒，四周留 pad。"""
    rgba = img.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if not bbox:
        return rgba
    l, t, r, b = bbox
    l, t = max(0, l - pad), max(0, t - pad)
    r, b = min(rgba.width, r + pad), min(rgba.height, b + pad)
    return rgba.crop((l, t, r, b))


# ---------------- 逐帧对齐 ----------------

def anchor_of(img: Image.Image) -> Tuple[float, float, int]:
    """返回 (脚底中心 x, 脚底 y, 高度)。锚点定义：alpha 包围盒底边中点。"""
    bbox = img.convert("RGBA").getchannel("A").getbbox()
    if not bbox:
        return img.width / 2, float(img.height), 0
    l, t, r, b = bbox
    return (l + r) / 2, float(b), b - t


def align_to(src: Image.Image, ref: Image.Image) -> Image.Image:
    """把 src 整体整数平移到 ref 的脚底锚点上（不改尺寸、不缩放）。"""
    sx, sy, _ = anchor_of(src)
    rx, ry, _ = anchor_of(ref)
    dx, dy = int(round(rx - sx)), int(round(ry - sy))
    if dx == 0 and dy == 0:
        return src
    canvas = Image.new("RGBA", src.size, (0, 0, 0, 0))
    canvas.paste(src, (dx, dy))
    return canvas


def align_sequence(frames: Sequence[Image.Image]) -> List[Image.Image]:
    """以首帧为锚，把后续帧拉回同一落脚点（抑制模型自构图带来的上下漂移）。"""
    if not frames:
        return []
    ref = frames[0]
    return [frames[0]] + [align_to(f, ref) for f in frames[1:]]


# ---------------- 精灵图打包 ----------------

def pack_sheet(
    frames: Iterable[Image.Image],
    out_png: str | Path,
    cols: int = 8,
    sidecar: dict | None = None,
) -> dict:
    """等距网格打包，单元格按最大帧尺寸 + 底部中心对齐。返回索引信息。"""
    frames = [f.convert("RGBA") for f in frames]
    if not frames:
        raise ValueError("没有可打包的帧")
    cw = max(f.width for f in frames)
    ch = max(f.height for f in frames)
    cols = max(1, min(cols, len(frames)))
    rows = (len(frames) + cols - 1) // cols

    sheet = Image.new("RGBA", (cw * cols, ch * rows), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        cx, cy = i % cols, i // cols
        cell = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        cell.paste(f, ((cw - f.width) // 2, ch - f.height))  # 底边对齐
        sheet.paste(cell, (cx * cw, cy * ch))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)

    info = {
        "sheet": str(out_png),
        "cell_w": cw,
        "cell_h": ch,
        "cols": cols,
        "rows": rows,
        "count": len(frames),
        "frame_index": [{"i": i, "col": i % cols, "row": i // cols} for i in range(len(frames))],
        **(sidecar or {}),
    }
    Path(str(out_png) + ".json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info
