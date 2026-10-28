"""参考图预处理：给 IP-Adapter / ControlNet 用的「干净」参考图。

业界共识（本项目实测印证）：
  - 参考图要**主体居中、背景干净**。IP-Adapter 用 CLIP Vision 编码参考图，
    背景会一起被编码进去——黑底会被当成角色的暗部，身份特征被污染。
  - **像素画缩放必须用最近邻**。LANCZOS/BILINEAR 会把像素块糊成渐变，
    编码出来的特征就丢了「这是像素画」这个信息。
  - 参考图与生成图**尺寸比例要接近**，否则主体在编码特征里的占比不对。

为什么这个模块值得单独抽出来：wpc / wss 两条链路都要「把一张可能是透明底、
可能是低分辨率像素画的角色图，整理成模型能正确理解的参考图」。两处各写一遍
必然走样——历史上 wpc 就因为直接 `convert("RGB")` 让透明区变黑（实测 66.5% 的
参考图是纯黑背景），在这里统一收口。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

# 参考图底色：白。模型最常见的假设是「白底 = 什么都没有」，
# 黑色在像素画语境里是描边色，用黑底会被误读成角色的暗部。
REF_BG = (255, 255, 255)


def flatten_rgb(img: Image.Image, bg: Tuple[int, int, int] = REF_BG) -> Image.Image:
    """RGBA → RGB，透明区填成指定底色（默认白）。

    为什么不能直接 convert("RGB")：那样透明区会变成**黑色**（PIL 的默认），
    像素素材转过去等于给模型一张黑底图——模型会把背景也当成内容重画。
    填白才符合「透明 = 什么都没有」的语义，而且白底是模型最常见的假设。
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    base = Image.new("RGBA", img.size, (bg[0], bg[1], bg[2], 255))
    base.alpha_composite(img)
    return base.convert("RGB")


def is_pixel_art(img: Image.Image) -> bool:
    """启发式判断是不是像素画（决定缩放用最近邻还是 LANCZOS）。

    **先说清楚这件事的本质**：从一张 64×64 的小图区分「像素画」和
    「低分辨率平涂插画」是**信息不足**的——两者都可能是 3~4 个颜色、
    几个亮度档、硬边。实测下面这些判据全都不可靠：

      - 「边缘跳变率」：像素画的圆形边缘是斜向的，水平跳变率低 → 漏判
      - 「最近邻放大后是否变化」：最近邻对任何图都不产生新颜色 → 恒为真
      - 「色数少 + 大色块」：卡通渲染也满足 → 误判
      - 「2×2 块单色率」：渐变图是阶梯状的、卡通边缘很细 → 误判
      - 「跳变周期性」：水平渐变图也是周期性的 → 误判
      - 「亮度直方图中间层占比」：绿色的史莱姆正好落在中间层 → 误判
      - 「亮度直方图稀疏性」：平涂卡通的亮度档也少 → 误判

    所以这里**不追求判准**，只保证「判错的后果足够小」：
    判错只是缩放的插值方式不同（最近邻 vs LANCZOS），
    不会毁掉结果。真正要紧的是 `prepare_reference` 里的
    「透明区填白」和「主体居中」——那两个是确定的改进。

    判据：**亮度直方图的稀疏性 + 硬边倾向**。
    像素画的亮度是「离散尖峰」（骑士 4 档、史莱姆 3 档），
    抗锯齿渲染是「连续谱」（横跨 30+ 档）。
    """
    rgb = flatten_rgb(img)
    w, h = rgb.size
    if w < 8 or h < 8:
        return False
    a = np.asarray(rgb, dtype=np.int16)

    counts = rgb.getcolors(maxcolors=1 << 20) or []
    if not counts or len(counts) > 256:
        return False

    # 用 float 算亮度：int16 的 299/587/114 加权会溢出成负数
    luma = ((a[..., 0] * 299.0 + a[..., 1] * 587.0 + a[..., 2] * 114.0) / 1000.0)
    luma = np.clip(luma, 0, 255).astype(np.int32)
    hist = np.bincount(luma.ravel(), minlength=256).astype(np.float32)
    total = float(hist.sum())
    if total <= 0:
        return False
    hist /= total
    # 64 个亮度桶（每桶 4 个亮度值），统计有效档数
    effective = int((hist.reshape(64, 4).sum(axis=1) >= 0.001).sum())
    # 像素画的亮度档很少；抗锯齿渲染横跨很多档
    return effective <= 12


def prepare_reference(img: Image.Image, size: int = 512,
                      pad_ratio: float = 0.12,
                      bg: Tuple[int, int, int] = REF_BG) -> Image.Image:
    """把任意角色图整理成模型能正确理解的参考图。

    步骤：
      1. 透明区填底色（不是黑）
      2. 裁到主体包围盒（去掉大片留白，让主体在特征里占比合理）
      3. **等比缩放**到目标尺寸 —— 像素画用最近邻，其余用 LANCZOS
      4. 贴到 `size` 画布中央，四周留白

    第 4 步的留白很关键：IP-Adapter 的 CLIP Vision 是在 224² 上做 14×14
    patch 切分的，主体如果顶满整张图，patch 里就只有角色没有环境，
    编码出来的特征会偏向「局部纹理」而不是「整体身份」。
    """
    src = flatten_rgb(img, bg)
    # 裁到主体：非背景色的包围盒
    arr = np.asarray(src, dtype=np.int16)
    diff = np.abs(arr - np.array(bg, dtype=np.int16)).max(axis=2)
    mask = diff > 16
    if mask.any():
        ys, xs = np.nonzero(mask)
        l, t, r, b = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        src = src.crop((l, t, r, b))

    # 等比缩放：像素画优先最近邻（保留像素块），否则 LANCZOS
    resample = Image.NEAREST if is_pixel_art(img) else Image.LANCZOS
    # 主体最多占画布的 (1 - 2*pad_ratio)，留一点边
    usable = size * (1.0 - 2.0 * max(0.0, min(0.4, pad_ratio)))
    scale = usable / max(src.width, src.height)
    w = max(1, int(round(src.width * scale)))
    h = max(1, int(round(src.height * scale)))
    src = src.resize((w, h), resample)

    # 贴到中央画布
    canvas = Image.new("RGB", (size, size), bg)
    ox = (size - w) // 2
    oy = (size - h) // 2
    canvas.paste(src, (ox, oy))
    return canvas
