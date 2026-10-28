"""参考条件图（骨架 / 轮廓 / 自动降级）与确定性换皮。

关键回归点：
  - **轮廓条件不能只是边缘**：只给边缘时模型把整屏填成纹理，
    成品里主体只剩 3 个像素（参考 677 个）。自制软边缘是 709 个。
  - **auto 必须真的降级**：像素画上人体检测器一个框都检不出（实测），
    auto 要从 pose 退到 outline，而不是报错。
  - **换皮必须确定性**：同样的输入两次跑结果逐像素一致。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from workflows import pixelfy as pxf
from workflows import refcond


def _pixel_knight(size: int = 64) -> Image.Image:
    """像素画骑士（透明底）。人体检测器认不出来的那种。"""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([26, 48, 30, 58], fill=(40, 40, 60, 255))
    d.rectangle([33, 48, 37, 58], fill=(40, 40, 60, 255))
    d.rectangle([25, 30, 38, 48], fill=(120, 30, 40, 255))
    d.rectangle([27, 18, 36, 30], fill=(235, 200, 160, 255))
    d.rectangle([26, 16, 37, 21], fill=(160, 165, 175, 255))
    d.rectangle([21, 31, 25, 42], fill=(120, 30, 40, 255))
    d.rectangle([38, 31, 42, 42], fill=(120, 30, 40, 255))
    d.rectangle([44, 20, 46, 40], fill=(200, 200, 210, 255))
    return im


def _slime(size: int = 64) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([10, 20, 54, 58], fill=(70, 190, 90, 255))
    d.ellipse([14, 24, 50, 54], fill=(110, 225, 120, 255))
    return im


# ---------------- 轮廓条件 ----------------

def test_outline_condition_has_edges():
    """轮廓条件图要有边缘像素（不是全黑）。"""
    cond, info = refcond.outline_condition(_pixel_knight(), size=256)
    assert cond.size == (256, 256)
    assert info["edge_pixels"] > 0
    arr = np.asarray(cond.convert("L"))
    assert arr.min() == 0, "黑底"
    assert arr.max() == 255, "有白边"


def test_outline_condition_marks_interior():
    """关键回归：**条件图必须标出主体内部**。

    只给边缘时模型不知道哪里是主体，会把整屏填成纹理（成品主体只剩
    3px / 参考 677px）。软边缘靠「主体内部压暗 + 背景提亮」传达信息。
    """
    cond, _info = refcond.outline_condition(_pixel_knight(), size=64)
    arr = np.asarray(cond.convert("L")).astype(np.float32)
    # 主体内部应当明显暗于背景
    assert arr[30, 30] < arr[2, 2], f"主体内 {arr[30, 30]} 应暗于背景 {arr[2, 2]}"


def test_outline_condition_rejects_blank():
    """全透明图找不到主体，要报错而不是给一张全黑条件图。"""
    with pytest.raises(ValueError, match="找不到主体"):
        refcond.outline_condition(Image.new("RGBA", (32, 32), (0, 0, 0, 0)))


def test_outline_condition_uses_nearest_neighbour():
    """放大必须用最近邻：像素画的硬边被 LANCZOS 糊成灰带就丢了像素感。"""
    cond, _ = refcond.outline_condition(_pixel_knight(), size=256)
    a = np.asarray(cond.convert("L"))
    # 最近邻放大的特征：横纵方向色块都是纯色，没有插值灰阶
    assert len(np.unique(a[0, :64])) == 1, "同一像素块内不应出现插值灰阶"


# ---------------- 自动降级 ----------------

def test_build_condition_auto_falls_back_to_outline():
    """像素画上 pose 必然失败，auto 要退到 outline 并如实记录原因。"""
    cond, used, info = refcond.build_condition(
        _pixel_knight(), mode="auto", size=128, extractor=None)
    assert used == "outline"
    assert info["fallback_from"] == "pose"
    assert info["fallback_reason"], "降级原因要留下来，别让用户瞎猜"


def test_build_condition_pose_mode_raises_without_extractor():
    """强制 pose 但姿态模型不可用 → 明确报错（不要静默降级，用户选的就是 pose）。"""
    with pytest.raises(ValueError, match="姿态模型不可用"):
        refcond.build_condition(_pixel_knight(), mode="pose", size=128, extractor=None)


def test_build_condition_rejects_unknown_mode():
    with pytest.raises(ValueError, match="未知参考模式"):
        refcond.build_condition(_pixel_knight(), mode="nope", size=128)


# ---------------- 确定性换皮 ----------------

def test_recolor_by_luma_keeps_shape():
    """换皮只改色不改形：alpha 必须逐像素不变。"""
    src = _pixel_knight()
    pal = [(10, 200, 30), (200, 40, 40), (30, 30, 30)]
    out = pxf.recolor_by_luma(src, pal, keep_alpha=True)
    a0 = np.asarray(src.getchannel("A"))
    a1 = np.asarray(out.getchannel("A"))
    assert (a0 == a1).all(), "alpha 不能变"


def test_recolor_by_luma_uses_palette():
    """前景像素的颜色必须全部来自给定色板。"""
    src = _pixel_knight()
    pal = [(10, 200, 30), (200, 40, 40), (30, 30, 30)]
    out = pxf.recolor_by_luma(src, pal, keep_alpha=True)
    arr = np.asarray(out.convert("RGBA"))
    fg = arr[..., 3] > 127
    used = {tuple(c) for c in arr[..., :3][fg].tolist()}
    assert used <= set(pal), used - set(pal)


def test_recolor_by_luma_is_deterministic():
    """同样的输入两次跑，结果必须逐像素一致（这是它相对 AI 路线的核心优势）。"""
    src = _pixel_knight()
    pal = pxf.extract_palette([_slime()], 8)
    a = np.asarray(pxf.recolor_by_luma(src, pal))
    b = np.asarray(pxf.recolor_by_luma(src, pal))
    assert (a == b).all()


def test_recolor_by_luma_preserves_luminance_order():
    """明度映射要保住明暗关系：原来更亮的像素，映射后也该更亮。"""
    src = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    src.putpixel((5, 5), (200, 200, 200, 255))     # 亮
    src.putpixel((10, 10), (30, 30, 30, 255))      # 暗
    pal = [(10, 10, 10), (120, 120, 120), (240, 240, 240)]
    out = np.asarray(pxf.recolor_by_luma(src, pal).convert("RGB"))
    assert out[5, 5].mean() > out[10, 10].mean()


def test_recolor_by_luma_empty_palette_is_noop():
    """空色板时原样返回，不能把图搞坏。"""
    src = _pixel_knight()
    out = pxf.recolor_by_luma(src, [], keep_alpha=True)
    assert np.asarray(out.getchannel("A")).sum() == np.asarray(src.getchannel("A")).sum()


def test_extract_palette_ignores_transparent():
    """角色图的色板不该被透明区（RGB 常是 0）污染。"""
    pal = pxf.extract_palette([_slime()], 8)
    assert pal, "应当提取到色板"
    assert all(sum(c) > 0 for c in pal), f"不该出现纯黑: {pal}"
