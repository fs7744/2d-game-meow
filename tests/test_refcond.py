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


# --------------------------------------------------------------------------- #
# 像素化的质量增强（这批是「像素生成质量差」的修复）
# --------------------------------------------------------------------------- #

def test_palette_excludes_transparent_pixels():
    """色板提取必须排除透明像素。

    **这是「像素生成质量差」的头号根因**：抠图后的图透明区 RGB 是纯黑
    （convert("RGB") 把 alpha=0 变成 0,0,0），而角色图前景常常只占 16%。
    不排除的话色板被「背景黑 + 抠图边缘白」吃掉大半色位——实测一张
    1024² 动漫立绘转像素后整张只剩 12 色、几乎全白，主体真正的银发蓝甲
    一个都没进色板。
    """
    from workflows import pixelfy as pxf

    # 造一张「16% 前景 + 84% 纯黑透明区」的图
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, 50, 50], fill=(200, 30, 40, 255))     # 红
    d.rectangle([10, 50, 50, 90], fill=(30, 60, 200, 255))     # 蓝
    a = np.asarray(im)
    fg_ratio = float((a[..., 3] > 127).mean())
    assert 0.1 < fg_ratio < 0.45, fg_ratio

    pal = pxf.extract_palette([im], 8)
    # 色板里必须出现前景的红和蓝，而不是被背景黑占满
    has_red = any(r > 150 and g < 120 and b < 120 for r, g, b in pal)
    has_blue = any(b > 150 and r < 120 for r, g, b in pal)
    assert has_red, f"色板里没有前景红：{pal}"
    assert has_blue, f"色板里没有前景蓝：{pal}"


def test_palette_all_transparent_falls_back():
    """全透明的图不能把整批色板拖空（要退回全部像素，而不是空色板）。"""
    from workflows import pixelfy as pxf

    blank = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    pal = pxf.extract_palette([blank], 4)
    # 退化成单一色是合理的（全透明图本来就没信息），但不能是空列表
    assert len(pal) >= 1, "全透明图不该产出空色板"


def test_pre_enhance_boosts_contrast():
    """对比度预增强要真的拉开色差（不然量化后是一坨灰）。

    实测同一张动漫立绘转 192²/32 色：不开时红披风偏暗红、金发偏土黄、
    整体发灰；开 1.0 后鲜红 + 亮黄。业界把这一步列为 photo-to-pixel 的第一步。
    """
    from workflows import pixelfy as pxf

    im = Image.new("RGBA", (64, 64), (120, 60, 60, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, 50, 50], fill=(160, 80, 80, 255))
    a0 = np.asarray(pxf.prep_enhance(im, 0.0).convert("RGB")).astype(float)
    a1 = np.asarray(pxf.prep_enhance(im, 1.0).convert("RGB")).astype(float)
    # 增强后色彩离散度应该变大（std 上升）
    assert a1.std() > a0.std(), (a0.std(), a1.std())
    # alpha 必须原样保留
    assert (np.asarray(pxf.prep_enhance(im, 1.0).getchannel("A"))
            == np.asarray(im.getchannel("A"))).all()


def test_edge_rebuild_keeps_palette_size():
    """轮廓描边不能破坏色板约束。

    **踩过的坑**：第一版按 0.7 直接乘暗，色数从 32 涨到 45~49——像素画的
    硬约束是「颜色数 ≤ 色板」，凭空多出中间色等于破功。现在改成从色板里
    找更暗的一档，色数严格不变。
    """
    from workflows import pixelfy as pxf

    im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, 24, 24], fill=(200, 200, 200, 255))
    d.rectangle([10, 10, 22, 22], fill=(80, 120, 200, 255))
    pal = [(200, 200, 200), (80, 120, 200), (40, 60, 100), (10, 10, 10)]
    before = len({tuple(c) for c in np.asarray(im.convert("RGB")).reshape(-1, 3).tolist()})
    out = pxf.enhance_edges(im, pal, outline_only=True)
    colors = {tuple(c) for c in np.asarray(out.convert("RGB")).reshape(-1, 3).tolist()}
    # 描边后所有颜色必须仍在色板内
    assert colors <= {tuple(c) for c in pal} | {(0, 0, 0)}, colors
    assert len(colors) <= max(before, len(pal)), "描边不该产生新颜色"


def test_edge_rebuild_outline_only_protects_face():
    """默认只描外轮廓——对所有内部结构边描边会把脸糊成一块灰。

    实测：同一张角色图，全边界描边后脸部五官糊成灰块；只描轮廓时脸保持
    干净、盔甲与披风的边界照样立起来了。像素精灵的立体感主要来自外轮廓。
    """
    from workflows import pixelfy as pxf

    # 一块前景，内部有「脸」一样的细碎色块
    im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([4, 4, 28, 28], fill=(230, 200, 180, 255))
    for i in range(6):
        d.rectangle([10 + i, 12, 10 + i, 18], fill=(60, 40, 40, 255))
    pal = [(230, 200, 180), (60, 40, 40), (20, 20, 20)]

    outline = pxf.enhance_edges(im, pal, outline_only=True)
    inner = pxf.enhance_edges(im, pal, outline_only=False, threshold=10)

    # 脸部中心（内部）在 outline_only 下必须保持原色
    a_o = np.asarray(outline)
    a_i = np.asarray(inner)
    center = (16, 14)      # 内部区域
    assert tuple(a_o[center[1], center[0], :3]) != (20, 20, 20), "轮廓描边不该动内部"
    # 而全边界描边会动到内部（说明两个模式确实不同）
    changed = (a_o[..., :3] != a_i[..., :3]).any(axis=2).sum()
    assert changed > 0, "两种模式应当有差异"
