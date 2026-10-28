"""局部 AI 重绘（w5）的图像运算层测试。

全是纯 PIL/numpy，不跑模型，秒级完成。
重点钉住两条硬承诺：
  1. **选区外的像素一个都不变**（羽化只向内）
  2. **色板吸附只吸够近的颜色**（用户要的新颜色不能被吃掉）
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from workflows import repaint as rp


def _mask(size=(256, 256), box=(80, 80, 176, 176)) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    return m


def _src(size=(256, 256)) -> Image.Image:
    im = Image.new("RGBA", size, (30, 60, 120, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((80, 80, 176, 176), fill=(200, 40, 40, 255))
    d.rectangle((0, 0, 255, 20), fill=(10, 10, 10, 255))
    return im


# --------------------------------------------------------------------------- #
# 蒙版运算
# --------------------------------------------------------------------------- #

def test_mask_bbox_and_coverage():
    m = _mask()
    # PIL 的 rectangle 两端都包含：(80,80,176,176) 覆盖 97×97 像素
    assert rp.mask_bbox(m) == (80, 80, 177, 177)
    assert rp.mask_coverage(m) == pytest.approx(97 * 97 / (256 * 256), abs=1e-3)
    assert rp.mask_bbox(Image.new("L", (64, 64), 0)) is None


def test_load_mask_uses_alpha_when_present():
    """前端导出的蒙版是「白 + 透明」，所以 alpha 才是有效区。"""
    m = Image.new("RGBA", (32, 32), (255, 255, 255, 0))
    ImageDraw.Draw(m).rectangle((4, 4, 12, 12), fill=(255, 255, 255, 255))
    out = rp.load_mask(m)
    assert out.mode == "L"
    assert rp.mask_bbox(out) == (4, 4, 13, 13)


def test_dilate_grows_mask():
    m = _mask()
    d = rp.dilate(m, 4)
    box = rp.mask_bbox(d)
    assert box[0] < 80 and box[1] < 80 and box[2] > 176 and box[3] > 176


def test_fit_size_aligns_to_16_and_keeps_in_range():
    """送模型的尺寸必须是 16 的倍数，且长边落在 [lo, hi]。"""
    w, h = rp.fit_size((100, 60))
    assert w % 16 == 0 and h % 16 == 0
    assert max(w, h) == 512
    w2, h2 = rp.fit_size((4000, 3000))
    assert max(w2, h2) == 1024
    assert w2 % 16 == 0 and h2 % 16 == 0


def test_expand_box_clamps_inside_image():
    box = rp.expand_box((0, 0, 256, 256), 50, (256, 256))
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 256
    assert 0 <= y0 < y1 <= 256


def test_crop_region_includes_context():
    src = _src()
    crop, cmask, box = rp.crop_region(src, _mask(), expand=8, context=0.35)
    # 框要比选区大（带上了上下文）
    assert box[0] < 80 and box[2] > 176
    assert crop.size == (box[2] - box[0], box[3] - box[1])
    assert cmask.size == crop.size


# --------------------------------------------------------------------------- #
# 贴回：选区外逐点不变（核心承诺）
# --------------------------------------------------------------------------- #

def test_composite_leaves_outside_pixels_identical():
    """选区外的像素一个都不变——这是局部重绘的立命之本。"""
    src = _src()
    m = _mask()
    gen = Image.new("RGBA", src.size, (0, 255, 0, 255))   # 全绿，差异极大

    out = rp.composite(src, gen, m, feather_px=4)
    a = np.asarray(src).astype(int)
    b = np.asarray(out).astype(int)
    outside = np.asarray(m) <= rp.MASK_THRESH
    assert (a[outside] == b[outside]).all(), "选区外有像素变了"


def test_composite_changes_inside():
    src = _src()
    m = _mask()
    gen = Image.new("RGBA", src.size, (0, 255, 0, 255))
    out = rp.composite(src, gen, m, feather_px=4)
    a = np.asarray(src).astype(int)
    b = np.asarray(out).astype(int)
    inside = np.asarray(m) > rp.MASK_THRESH
    assert not (a[inside] == b[inside]).all(), "选区里应该被改了"


def test_composite_hard_edge_is_binary():
    """硬边模式：选区内的像素应该被**完全替换**，不是半透明混合。"""
    src = _src()
    m = _mask()
    gen = Image.new("RGBA", src.size, (0, 255, 0, 255))
    out = rp.composite(src, gen, m, feather_px=4, hard=True)
    assert out.getpixel((128, 128))[:3] == (0, 255, 0)


def test_match_color_does_not_drag_whole_region_back():
    """色彩校正只作用在边界带，区域中心的新颜色不能被拉回原色。

    这条是实测结论：早期对整块选区做均值/方差匹配，「把衣服改成红色」
    会被拉回原来的蓝——校正的本意是消接缝，不该把用户要的改动一起抹平。
    """
    src = _src()                              # 选区里是红色 (200,40,40)
    m = _mask()
    gen = Image.new("RGBA", src.size, (0, 0, 255, 255))   # 模型给了蓝色
    out = rp.composite(src, gen, m, feather_px=4, match_color=True)
    center = out.getpixel((128, 128))[:3]
    # 中心应该还是偏蓝（没被拉回红）
    assert center[2] > center[0], center


# --------------------------------------------------------------------------- #
# 色板吸附
# --------------------------------------------------------------------------- #

def test_snap_palette_keeps_far_colors():
    """差得远的新颜色必须保留——否则「要改成红色」会被吸成灰。"""
    src = Image.new("RGB", (64, 64), (10, 10, 10))       # 源图只有近黑
    region = Image.new("L", (64, 64), 255)
    img = Image.new("RGBA", (64, 64), (230, 20, 20, 255))  # 纯红
    out, pal, changed = rp.snap_palette(img, src, region, colors=8, tol=30)
    assert out.getpixel((10, 10))[:3] == (230, 20, 20), "红色不该被吸走"
    assert changed == 0.0


def test_snap_palette_absorbs_near_colors():
    """够近的杂色要吸回源图色板——这是消除拼贴感的手段。"""
    src = Image.new("RGB", (64, 64), (10, 10, 10))
    region = Image.new("L", (64, 64), 255)
    img = Image.new("RGBA", (64, 64), (14, 12, 11, 255))   # 与源色很接近
    out, pal, changed = rp.snap_palette(img, src, region, colors=8, tol=30)
    assert out.getpixel((10, 10))[:3] == (10, 10, 10)
    assert changed > 0.9


def test_snap_palette_only_touches_region():
    """区域外连 ±1 的抖动都不该出现。"""
    src = Image.new("RGB", (64, 64), (10, 10, 10))
    region = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(region).rectangle((0, 0, 31, 63), fill=255)
    img = Image.new("RGBA", (64, 64), (14, 12, 11, 255))
    out, pal, changed = rp.snap_palette(img, src, region, colors=8, tol=30)
    assert out.getpixel((50, 10))[:3] == (14, 12, 11), "选区外被改了"
    assert out.getpixel((10, 10))[:3] == (10, 10, 10)


def test_band_weight_is_zero_outside_mask():
    m = _mask()
    w = rp.band_weight(m, 12)
    outside = np.asarray(m) <= rp.MASK_THRESH
    assert np.allclose(w[outside], 0.0)


def test_rgba_must_be_converted_before_vae():
    """RGBA 图不能直接进 VAE——会变成 4 通道，UNet 首层卷积直接 OOM。

    实测：512² 的 RGBA 进 StableDiffusionImg2ImgPipeline，报
    "Tried to allocate 36.00 GiB"（UNet 按 4 通道卷积，激活尺寸爆炸）。
    所以送模型前必须 convert("RGB")。
    """
    from diffusers import VaeImageProcessor

    p = VaeImageProcessor(vae_scale_factor=8)
    assert p.preprocess(Image.new("RGB", (64, 64), (1, 2, 3))).shape[1] == 3
    assert p.preprocess(Image.new("RGBA", (64, 64), (1, 2, 3, 4))).shape[1] == 4


def test_flatten_rgb_fills_transparent_with_white():
    """透明区必须填白，不能变黑。

    直接 convert("RGB") 会把透明区变成黑色——精灵素材转过去等于给模型
    一张黑底图，模型会把背景也当内容重画，出来的选区周围带一圈脏边。
    """
    im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((8, 8, 20, 20), fill=(200, 40, 40, 255))
    out = rp.flatten_rgb(im)
    assert out.mode == "RGB"
    assert out.getpixel((2, 2)) == (255, 255, 255), "透明区应填白"
    assert out.getpixel((14, 14)) == (200, 40, 40), "实体像素应保持"
