"""局部 AI 重绘的图像运算层（纯 PIL/numpy，不碰模型）。

这一层是「画得像不像」与「只改该改的地方」的分工：
**模型负责像不像，本模块负责只改该改的地方、且看不出来改过。**

四条硬约束，都是从 pixelforge 的实测结论搬过来的：

1. **贴回的羽化只向内**：羽化权重 = 模糊(蒙版) × 二值蒙版，区域外的权重恒为 0，
   所以「选区外的像素一个都不变」是**逐像素可验证**的承诺，不是嘴上说说。
2. **接缝色彩校正只作用在边界带**：对整块选区做均值/方差匹配，会把用户要的改动
   一起抹平（"把衣服改成红色"被拉回原来的蓝）。校正的本意是消接缝，
   所以只在边界向内的一条带上做。
3. **色板吸附只吸「够近」的颜色**：像素素材重画后必然带来新杂色，把颜色数压回
   源图色板就没有拼贴感；但差得远的新颜色必须保留，否则"要改成红色"会被吸成灰。
4. **蒙版先外扩再取包围盒**：用户涂的区域往往紧贴要改的东西，模型没有"往哪长"
   的余地；外扩一圈 + 带上上下文，模型才知道该往哪个方向补。

所有函数都是确定性的，可以单独测试（见 tests/test_repaint.py）。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

# 蒙版判有效的阈值：前端画的是纯白 + 全透明，中间值只来自缩放插值
MASK_THRESH = 127
# 送模型前的尺寸对齐：VAE 8× + patch 2× = 16 的倍数
ALIGN = 16


# --------------------------------------------------------------------------- #
# 透明底处理
# --------------------------------------------------------------------------- #

def flatten_rgb(img: Image.Image, bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """RGBA → RGB，透明区域填成指定底色（默认白）。

    为什么不能直接 convert("RGB")：那样透明区会变成**黑色**（PIL 的默认），
    精灵素材转过去等于给模型一张黑底图——模型会把背景也当成内容重画，
    出来的选区周围会带一圈脏边。填白才符合"透明 = 什么都没有"的语义，
    而且白底是模型最常见的假设。

    为什么必须转 RGB：VAE 只有 3 通道，RGBA 进去会变成 4 通道，
    UNet 首层按 4 通道卷积 → 激活尺寸爆炸（实测 512² 直接 OOM）。
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    base = Image.new("RGBA", img.size, (bg[0], bg[1], bg[2], 255))
    base.alpha_composite(img)
    return base.convert("RGB")


# --------------------------------------------------------------------------- #
# 蒙版运算
# --------------------------------------------------------------------------- #

def load_mask(img: Image.Image | str) -> Image.Image:
    """任意输入 → 单通道 L 蒙版（>127 为重绘区）。"""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        # 有 alpha 的蒙版：按 alpha 判有效（前端导出的蒙版是白 + 透明）
        a = np.asarray(img.getchannel("A"))
        if a.min() < MASK_THRESH:
            return Image.fromarray(a, "L")
        return img.convert("L")
    return img.convert("L")


def mask_bbox(mask: Image.Image, thresh: int = MASK_THRESH
              ) -> Optional[Tuple[int, int, int, int]]:
    """蒙版的包围盒 (x0, y0, x1, y1)，右下开区间；空蒙版返回 None。"""
    a = np.asarray(mask.convert("L")) > thresh
    if not a.any():
        return None
    ys = np.nonzero(a.any(axis=1))[0]
    xs = np.nonzero(a.any(axis=0))[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def mask_coverage(mask: Image.Image, thresh: int = MASK_THRESH) -> float:
    a = np.asarray(mask.convert("L")) > thresh
    return round(float(a.mean()), 4) if a.size else 0.0


def dilate(mask: Image.Image, px: int) -> Image.Image:
    """二值膨胀（MaxFilter）。

    为什么用 PIL 的 MaxFilter 而不是 cv2：cv2 只在 GPU 依赖里（可选的），
    这个函数必须永远可用。MaxFilter 的核是方的，px 形参按「边长的一半」算，
    视觉上和圆形核差别可以忽略。
    """
    px = max(0, int(px))
    if px <= 0:
        return mask
    k = px * 2 + 1
    out = mask
    while k > 1:
        step = min(k, 9)
        if step % 2 == 0:
            step -= 1
        if step < 3:
            break
        out = out.filter(ImageFilter.MaxFilter(step))
        k -= step - 1
    return out


def feather(mask: Image.Image, px: int) -> Image.Image:
    """羽化：模糊半径 ≥1 才做（0 = 硬边）。"""
    px = int(px)
    if px <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(max(0.5, px / 2.0)))


def expand_box(box: Tuple[int, int, int, int], pad: int,
               size: Tuple[int, int], align: int = ALIGN
               ) -> Tuple[int, int, int, int]:
    """包围盒向外扩 pad，再按 align 对齐并夹到图内。"""
    x0, y0, x1, y1 = box
    pad = max(0, int(pad))
    x0 -= pad
    y0 -= pad
    x1 += pad
    y1 += pad
    W, H = size
    if align > 1:
        x0 = (x0 // align) * align
        y0 = (y0 // align) * align
        x1 = -((-x1) // align) * align
        y1 = -((-y1) // align) * align
    x0 = max(0, min(x0, W - align))
    y0 = max(0, min(y0, H - align))
    x1 = max(x0 + align, min(x1, W))
    y1 = max(y0 + align, min(y1, H))
    return int(x0), int(y0), int(x1), int(y1)


def fit_size(size: Tuple[int, int], *, lo: int = 512,
             hi: int = 1024, align: int = ALIGN) -> Tuple[int, int]:
    """把要送模型的尺寸规整成「16 的倍数，长边落在 [lo, hi]」。

    小区域会被放大：模型在 512 上画的细节比在 128 上画得好，而放大再缩回
    （LANCZOS）丢的信息远少于让模型在极小画布上凑合。
    """
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    long_side = max(w, h)
    scale = 1.0
    if long_side < lo:
        scale = lo / long_side
    elif long_side > hi:
        scale = hi / long_side
    w2 = int(round(w * scale / align)) * align
    h2 = int(round(h * scale / align)) * align
    return max(align, w2), max(align, h2)


# --------------------------------------------------------------------------- #
# 贴回：羽化 + 接缝带色彩校正
# --------------------------------------------------------------------------- #

def band_weight(mask: Image.Image, band: int) -> np.ndarray:
    """接缝带权重（0~1，float32）：蒙版边界**向内** band px 的过渡带。

    做法：blur(mask) 与 mask 相减——只有边界附近非零；取绝对值归一化后
    再乘回蒙版本身，保证只在选区内生效。区域中心权重≈0，所以中心的新颜色
    不会被拉走。
    """
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    if band <= 0:
        return np.zeros_like(m)
    blur = np.asarray(feather(mask, band * 2).convert("L"), dtype=np.float32) / 255.0
    w = np.clip(np.abs(m - blur) * 2.2, 0.0, 1.0) * m
    return w.astype(np.float32)


def color_transfer(gen: np.ndarray, src: np.ndarray, weight: np.ndarray,
                   strength: float) -> np.ndarray:
    """按权重把 gen 的通道均值/方差校正到 src（Reinhard 的局部版）。

    gen/src: (H,W,4) float32；weight: (H,W) float32；strength: 0~1。
    统计只在 weight>0.3 的像素上做——它们就是接缝带。
    """
    if strength <= 0:
        return gen
    sel = weight > 0.3
    if not sel.any():
        return gen
    out = gen.copy()
    for c in range(3):                       # alpha 不动：透明关系不能靠均值修
        g = gen[..., c][sel]
        s = src[..., c][sel]
        mg, sg = float(g.mean()), float(g.std())
        ms, ss = float(s.mean()), float(s.std())
        if sg < 1e-3:
            continue
        ratio = min(2.5, max(0.4, ss / sg))  # 方差比夹紧：防伪影被放大成噪点
        pred = (gen[..., c] - mg) * ratio + ms
        k = (weight * strength).astype(np.float32)
        out[..., c] = gen[..., c] * (1 - k) + pred * k
    return np.clip(out, 0, 255)


def composite(src: Image.Image, gen: Image.Image, mask: Image.Image, *,
              feather_px: int = 4, hard: bool = False,
              match_color: bool = True, match_strength: float = 0.45
              ) -> Image.Image:
    """把 gen 按蒙版贴回 src（尺寸必须已对齐）。

    羽化**只向内**（羽化后的权重再乘一次二值蒙版）：这样「选区外的像素一个都不
    变」是可以逐像素验证的硬承诺。向外羽化虽然过渡更软，但会把模型内容推到用户
    没涂的地方——那正是局部重绘最不该发生的事。

    alpha 也按同一权重混合：素材图带透明时，新区域的透明关系由模型结果决定。
    """
    if gen.size != src.size:
        gen = gen.resize(src.size, Image.LANCZOS)
    s = np.asarray(src.convert("RGBA"), dtype=np.float32)
    g = np.asarray(gen.convert("RGBA"), dtype=np.float32)

    inside = (np.asarray(mask.convert("L")) > MASK_THRESH).astype(np.float32)
    if hard:
        a = inside
        w = np.zeros_like(a)
    else:
        soft = np.asarray(feather(mask, feather_px).convert("L"), dtype=np.float32) / 255.0
        a = soft * inside
        w = (band_weight(mask, max(1, int(feather_px) * 3)) if match_color
             else np.zeros_like(a))
    if match_color and not hard:
        g = color_transfer(g, s, w, match_strength)

    a3 = a[..., None]
    out = s * (1 - a3) + g * a3
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


# --------------------------------------------------------------------------- #
# 色板吸附（像素素材专用）
# --------------------------------------------------------------------------- #

def extract_palette(img: Image.Image, colors: int = 32) -> np.ndarray:
    """取源图色板（(N,3) int32）。用 PIL 的中位切分，确定性可复现。"""
    q = img.convert("RGB").quantize(colors=max(2, int(colors)),
                                    method=Image.MEDIANCUT, dither=Image.NONE)
    pal = q.getpalette() or []
    arr = np.asarray(pal, dtype=np.int32).reshape(-1, 3)
    return arr[:max(2, int(colors))]


def snap_palette(img: Image.Image, src: Image.Image, region: Image.Image, *,
                 colors: int = 32, tol: int = 30
                 ) -> Tuple[Image.Image, np.ndarray, float]:
    """把重绘区里「接近已有颜色」的杂色吸回源图色板。

    **不是全量量化**。像素素材重画后常带一层新色（模型不受控），把颜色数压回
    源图色板就没有拼贴感；但如果用户就是要把这块改成红色、源图里根本没有红，
    全量量化会把红色按最近距离吸成灰——「要的功能被后处理吃掉」。
    所以只吸附**距离够近**（tol 以内，按 R2G4B3 加权）的颜色，差得远的新颜色
    原样保留。返回的比例就是「吸附了多少像素」。
    """
    pal = extract_palette(src, colors)
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3).astype(np.int32)

    nearest = np.empty_like(flat)
    dist = np.empty(len(flat), dtype=np.float32)
    block = 65536
    for s in range(0, len(flat), block):
        chunk = flat[s:s + block]
        d = ((chunk[:, None, 0] - pal[None, :, 0]) ** 2 * 2
             + (chunk[:, None, 1] - pal[None, :, 1]) ** 2 * 4
             + (chunk[:, None, 2] - pal[None, :, 2]) ** 2 * 3)
        idx = np.argmin(d, axis=1)
        nearest[s:s + block] = pal[idx]
        dist[s:s + block] = d[np.arange(len(chunk)), idx]

    # 容差换算成加权距离：per-channel tol 时，三项平方和 ≈ tol²×(2+4+3)
    thr = float(tol) ** 2 * 9.0
    take = dist <= thr
    reg = np.asarray(region.convert("L")).reshape(-1) > MASK_THRESH
    take &= reg
    out = flat.copy()
    out[take] = nearest[take]
    changed = float(take.sum()) / max(1, int(reg.sum()))
    rgb = Image.fromarray(out.astype(np.uint8).reshape(h, w, 3), "RGB").convert("RGBA")
    rgb.putalpha(img.convert("RGBA").getchannel("A"))
    return rgb, pal, round(changed, 3)


# --------------------------------------------------------------------------- #
# 选区裁剪（送模型前的准备）
# --------------------------------------------------------------------------- #

def crop_region(src: Image.Image, mask: Image.Image, *, expand: int = 8,
                context: float = 0.35, align: int = ALIGN
                ) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """裁出「选区 + 上下文」，返回 (裁剪图, 裁剪蒙版, 原图上的框)。

    步骤（照 pixelforge）：
      1. 蒙版外扩 expand px——给模型留「往哪长」的余地；
      2. 取外扩后的包围盒，再按 context 比例向外带一圈上下文；
      3. 框按 align 对齐（VAE 8× + patch 2× 的硬约束），否则管线会静默改尺寸。
    """
    work = dilate(mask, expand)
    box = mask_bbox(work)
    if box is None:
        box = mask_bbox(mask)
    if box is None:
        raise ValueError("蒙版是空的：先用笔刷涂出要重绘的区域")
    x0, y0, x1, y1 = box
    ctx = int(round(max(x1 - x0, y1 - y0) * max(0.0, min(float(context), 2.0))))
    final = expand_box((x0, y0, x1, y1), ctx, src.size, align)
    return src.crop(final), mask.crop(final), final
