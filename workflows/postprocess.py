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

def _pixelate_grid(img: Image.Image, block: int) -> Image.Image | None:
    """按**检测到的网格**采样像素化；检测不可信时返回 None（由调用方回退 BOX）。

    为什么值得多这一步：AI 出的"像素风"网格几乎不会是整数倍（实测 6.3px 这种），
    直接 BOX 降采样会把格子边界切在色块中间，混出脏色。这里先检测网格、每格取
    中位色、再最近邻放大回原尺寸——每格必然单色，这正是"像素画"与"低分辨率图"
    的区别。`perfect_pixel` 那套已经在 wpx 节点上验证过（每格单色率 100%）。

    **只在检测值与请求值接近时采用**（±30%）：差太远说明这张图本来不是那个
    像素密度，硬套会把细节抹平，不如老实回退。

    **前提要说清楚**：网格对齐只对"已经是像素画"的输入有意义（wpx 节点那条路）。
    主链路的输入是连续调的 AI 图，**根本没有可检测的网格**——实测 768² 的 W1
    产物、以及"6px 块的随机噪声图"都检测不出，直接回退。所以这里先用色数做
    便宜的前置判断：连续调图有成千上万色，立刻返回，不付检测的时间。
    """
    if block < 2:
        return None
    # 便宜的前置判断：像素画色数很少（本项目产物量化到 8~32 色），
    # 超过 512 色几乎不可能是网格化像素画，直接跳过昂贵检测。
    if img.convert("RGB").getcolors(maxcolors=512) is None:
        return None
    try:
        from . import pixelfy as pxf
    except Exception:                       # noqa: BLE001  依赖缺失就回退
        return None
    try:
        r = pxf.refine_one(img.convert("RGB"), grid_snap=True, palette=None)
    except Exception:                       # noqa: BLE001
        return None
    meta = r.get("meta") or {}
    if meta.get("gridFailed"):
        return None
    gw, gh = (meta.get("grid") or [0, 0])[:2]
    # `detect_grid_scale` 返回的是**格数**（横/纵各多少格），不是每格多少像素。
    # 768² + block=6 → 期望 128 格；用它和 block 直接比是拿错单位（踩过）。
    want_w, want_h = max(1, img.width // block), max(1, img.height // block)
    lo_w, hi_w = want_w * 0.7, want_w * 1.3
    lo_h, hi_h = want_h * 0.7, want_h * 1.3
    if not (lo_w <= gw <= hi_w and lo_h <= gh <= hi_h):
        return None
    small = r["image"].convert("RGB")
    if small.width < 1 or small.height < 1:
        return None
    up = small.resize(img.size, Image.NEAREST)
    if img.mode == "RGBA":
        # alpha 沿用 BOX（先降后升），与老路径保持一致：不在这里改 alpha 语义
        a = img.getchannel("A").resize(small.size, Image.BOX).resize(img.size, Image.NEAREST)
        up = up.convert("RGBA")
        up.putalpha(a)
    return up


def _block_median(img: Image.Image, block: int) -> Image.Image:
    """按块取**中位数**而不是平均值。

    为什么比平均好：BOX 平均会把背景色按比例混进边缘块，轮廓外面挂一圈灰边
    （看着像抗锯齿）。实测同一张图并排对比，中位数取色的边缘更硬、色块更纯，
    更接近手工像素画（对比图 assets/_sample_cmp.png）。

    alpha 不能取中位数——那会给出半透明块。用「块内多数像素是否不透明」判，
    保证轮廓是硬的（像素素材要的就是硬边）。
    """
    import numpy as np

    rgba = np.asarray(img.convert("RGBA"))
    h, w = rgba.shape[:2]
    hh, ww = (h // block) * block, (w // block) * block
    grid = rgba[:hh, :ww].reshape(hh // block, block, ww // block, block, 4)
    rgb = np.median(grid[..., :3].astype(np.float32), axis=(1, 3)).astype(np.uint8)
    solid = (grid[..., 3] > 127).mean(axis=(1, 3)) >= 0.5
    out = np.dstack([rgb, (solid * 255).astype(np.uint8)])
    out = np.repeat(np.repeat(out, block, axis=0), block, axis=1)
    if out.shape[0] != h or out.shape[1] != w:      # 右下不足一块的边
        pad = np.zeros((h, w, 4), np.uint8)
        pad[: out.shape[0], : out.shape[1]] = out
        out = pad
    return Image.fromarray(out, "RGBA")


def pixelate(img: Image.Image, block: int, resample=Image.BOX,
             grid_align: bool = False, sample: str = "box") -> Image.Image:
    """按 block 像素栅格吸附：先降采样再最近邻放大回原尺寸。

    降采样用 BOX（面积平均）而不是 LANCZOS：LANCZOS 有负瓣，会在色块交界处
    振铃出「过冲色」（深色线旁边冒一圈更亮的边），像素化之后再量化，这些过冲色
    会变成额外的脏色阶。BOX 是纯平均，不造假色。实测同一张图的唯一色数下降、
    锐度上升（见 tools/ab_quality.py）。

    `grid_align=True` 时先试**网格对齐采样**（见 _pixelate_grid）：
    主链路以前只有 BOX，而项目自己更好的实现锁在 wpx 节点里，等于"好工具没接到
    产线上"。检测不可信会自动回退，所以打开它不改变稳定性。
    """
    if block <= 1:
        return img
    if grid_align:
        aligned = _pixelate_grid(img, block)
        if aligned is not None:
            return aligned
    if sample == "median":
        try:
            return _block_median(img, block)
        except Exception:                  # noqa: BLE001  异常就退回 BOX，别让出图失败
            pass
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
