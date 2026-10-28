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

import cv2
import numpy as np
from PIL import Image


# ---------------- 像素化 / 调色板 ----------------

def pixelate(img: Image.Image, block: int, resample=Image.BOX) -> Image.Image:
    """按 block 像素栅格吸附：先降采样再最近邻放大回原尺寸。

    降采样用 BOX（面积平均）而不是 LANCZOS：LANCZOS 有负瓣，会在色块交界处
    振铃出「过冲色」（深色线旁边冒一圈更亮的边），像素化之后再量化，这些过冲色
    会变成额外的脏色阶。BOX 是纯平均，不造假色。实测同一张图的唯一色数下降、
    锐度上升（见 tools/ab_quality.py）。
    """
    if block <= 1:
        return img
    w, h = img.size
    small = img.resize((max(1, w // block), max(1, h // block)), resample)
    return small.resize((w, h), Image.NEAREST)


def quantize_palette(img: Image.Image, colors: int = 16) -> Image.Image:
    """压到固定色数（默认 16 色），保留 alpha。PIL 的中位切分，确定性可复现。"""
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    pal = rgba.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
    out = pal.convert("RGBA")
    out.putalpha(alpha.point(lambda a: 255 if a > 127 else 0))  # 硬边 alpha
    return out


def clean_isolated(img: Image.Image, passes: int = 1) -> Image.Image:
    """去孤立点：把「上下左右四个邻居颜色都跟自己不同」的像素处理掉。

    这是像素画的经典清洗步骤，专治一类特定问题：**角色身上/轮廓外的零星杂色**。
    实测 SDXL 的像素风出图，量化到 32 色后角色周围会冒出一粒粒红、青、橙的点。
    它们大多是**主体连通域的一部分**（实测 1024² 图上 25 个前景块，最小的也有
    2368px，根本没有孤立小块），所以面积过滤和形态学开运算都拿它们没办法 ——
    那两种手段只动 alpha，而这里要动的是**颜色**。

    分两类处理：
    - 主体内部的杂色粒：改成上邻颜色（沿扫描线方向连续）
    - 飘在外面、四邻皆透明、且 5×5 邻域只有它自己的：直接删掉

    5×5 计数这一条不能省：像素画的 1px 斜向轮廓四邻同样全透明，只看 4 邻域会
    把整条对角线吃掉（实测就是这样，所以加了这层保护）。
    """
    if passes <= 0:
        return img
    arr = np.asarray(img.convert("RGBA")).copy()
    h, w = arr.shape[:2]
    if h < 3 or w < 3:
        return Image.fromarray(arr, "RGBA")

    k5 = np.ones((5, 5), np.float32)
    for _ in range(passes):
        c, a = arr[..., :3], arr[..., 3]
        m = a[1:-1, 1:-1] > 0

        up_fg, dn_fg = a[:-2, 1:-1] > 0, a[2:, 1:-1] > 0
        lf_fg, rt_fg = a[1:-1, :-2] > 0, a[1:-1, 2:] > 0

        # 与「同为前景」的邻居同色的数量
        same = np.zeros((h - 2, w - 2), dtype=np.int8)
        same[m] += (np.all(c[1:-1, 1:-1][m] == c[:-2, 1:-1][m], axis=-1) & up_fg[m])
        same[m] += (np.all(c[1:-1, 1:-1][m] == c[2:, 1:-1][m], axis=-1) & dn_fg[m])
        same[m] += (np.all(c[1:-1, 1:-1][m] == c[1:-1, :-2][m], axis=-1) & lf_fg[m])
        same[m] += (np.all(c[1:-1, 1:-1][m] == c[1:-1, 2:][m], axis=-1) & rt_fg[m])

        iso = (same == 0) & m
        if not iso.any():
            break

        cnt = cv2.filter2D((a > 0).astype(np.float32), -1, k5,
                           borderType=cv2.BORDER_CONSTANT)
        outside = iso & ~up_fg & ~dn_fg & ~lf_fg & ~rt_fg & (cnt[1:-1, 1:-1] <= 1.5)
        rest = iso & ~outside

        # 写回：统一用整图尺寸的 mask，避免链式切片索引的赋值歧义
        del_mask = np.zeros((h, w), bool)
        del_mask[1:-1, 1:-1] = outside
        if del_mask.any():
            arr[..., 3][del_mask] = 0

        if rest.any():
            # 上邻优先；上邻是透明就退回左邻
            arr[1:-1, 1:-1, :3][rest] = arr[:-2, 1:-1, :3][rest]
            no_up = rest & ~up_fg
            arr[1:-1, 1:-1, :3][no_up] = arr[1:-1, :-2, :3][no_up]

    return Image.fromarray(arr, "RGBA")


def foreground_blobs(img: Image.Image, min_px: int = 400) -> int:
    """抠图后前景连通域个数（只数面积 >= min_px 的块）。

    >1 通常意味着「画面里挤了多个主体」：SD1.5 画小体型怪物（史莱姆、小动物）时
    很爱复制一份，负向词压不住。数出来写进 meta，产线可以据此退图，而不是等
    打包成精灵图之后才用肉眼发现。只在抠图后有意义（未抠图时全图是一个域）。

    注意这是个**下界**：u2net 只分割「显著主体」，画面里挤了十几个小怪物时它
    往往只留一个（实测史莱姆图案墙 → mask 里只剩 1 块）。所以 1 不代表干净，
    但 >1 基本确实脏。
    """
    alpha = np.asarray(img.convert("RGBA").getchannel("A"))
    if alpha.min() == 255:
        return 1
    fg = (alpha > 127).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    return int(sum(1 for a in stats[1:, cv2.CC_STAT_AREA] if a >= min_px))


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
