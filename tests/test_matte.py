"""抠图 / 去白边：连通域语义、混色边清除、alpha 二值、裁边。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from workflows import matte as M


def _sprite(halo=None, bg=(255, 255, 255), hole=False) -> Image.Image:
    im = Image.new("RGB", (128, 128), bg)
    d = ImageDraw.Draw(im)
    if halo:
        d.rectangle([38, 38, 90, 90], fill=halo)      # 2px 混色边
        d.rectangle([40, 40, 88, 88], fill=(200, 30, 30))
    else:
        d.rectangle([40, 40, 88, 88], fill=(200, 30, 30))
    if hole:
        d.rectangle([58, 58, 70, 70], fill=bg)        # 与背景同色但不连通
    return im


def _alpha(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"))[..., 3]


def test_estimate_bg_from_border():
    assert M.estimate_bg_color(_sprite()) == (255, 255, 255)
    assert M.estimate_bg_color(_sprite(bg=(12, 34, 56))) == (12, 34, 56)


def test_matte_makes_background_transparent():
    out = M.matte(_sprite())
    a = _alpha(out)
    assert a[0, 0] == 0 and a[127, 127] == 0
    assert a[50, 50] == 255


def test_matte_alpha_is_binary():
    a = _alpha(M.matte(_sprite(halo=(255, 190, 190), hole=True)))
    assert set(np.unique(a).tolist()) <= {0, 255}


def test_matte_keeps_enclosed_same_color_region():
    """角色内部与背景同色的洞必须保留——只抠与画面边界连通的区域。"""
    a = _alpha(M.matte(_sprite(hole=True)))
    assert a[64, 64] == 255, "内部白洞被误抠掉了"


def test_fringe_removal_reduces_halo():
    im = _sprite(halo=(255, 190, 190), hole=True)
    without = M.matte(im, fringe=False)
    with_ = M.matte(im, fringe=True)
    # 混色边颜色偏白：开启去边后，边缘上「接近背景」的像素应当减少
    def whitish(rgba):
        arr = np.asarray(rgba.convert("RGBA")).astype(int)
        edge = (arr[..., 3] > 0)
        near_white = (np.abs(arr[..., :3] - 255).max(axis=2) < 80) & edge
        return int(near_white.sum())

    assert whitish(with_) < whitish(without), "去白边没有效果"
    assert M.has_fringe(with_, (255, 255, 255)) == 0


def test_custom_bg_color():
    im = _sprite(bg=(0, 120, 0))
    a = _alpha(M.matte(im, bg=(0, 120, 0)))
    assert a[0, 0] == 0 and a[50, 50] == 255


def test_tolerance_controls_aggressiveness():
    im = _sprite(halo=(250, 250, 250))
    loose = _alpha(M.matte(im, tol=120))       # 宽容差：把近白边也当背景
    strict = _alpha(M.matte(im, tol=8))
    assert (loose == 0).sum() >= (strict == 0).sum()


def test_trim_crops_to_content():
    im = M.matte(_sprite())
    t = M.trim(im)
    assert t.width < 128 and t.height < 128
    assert t.getchannel("A").getbbox() == (0, 0, t.width, t.height)


def test_matte_on_generated_frame(root):
    """真实生成图也要能抠干净（背景是模型自己画的，不是纯白）。

    素材用 `assets/_qwen_verify/char.png`（tools/verify_qwen_workflows.py 生成的
    白猫骑士立绘）。**不现场生成**：Qwen 一张 24 步要 25s，而这条测的是抠图，
    与生成无关；素材不在就跳过，别让单测变成几分钟。
    """
    src = root / "assets" / "_qwen_verify" / "char.png"
    if not src.is_file():
        import pytest
        pytest.skip("缺素材 assets/_qwen_verify/char.png（先跑 tools/verify_qwen_workflows.py）")
    raw = Image.open(src).convert("RGBA")
    out = M.matte(raw)
    a = _alpha(out)
    assert (a == 0).mean() > 0.08, f"透明区域太少，抠图没生效: {(a == 0).mean():.2f}"
    assert (a == 0).mean() < 0.95, "几乎全透明，抠过头了"
    assert set(np.unique(a).tolist()) <= {0, 255}
# ---------------- matte_auto 的后端选择与回退 ----------------

def test_auto_without_model_uses_color(tmp_path):
    info = {}
    out = M.matte_auto(_sprite(), "auto", tmp_path, info=info)
    assert info == {"backend": "color", "fallback": False}
    assert out.getchannel("A").getbbox() is not None


def test_auto_falls_back_when_u2net_blanks(monkeypatch, tmp_path):
    """u2net 交出一张几乎全透明的图时必须回退色键。

    实测草稿档（256²/8 步）出图会让 isnet 只留下 0.44% 前景——抠图「成功」，
    下游贴边检测/主体计数/像素化全拿到空数据，最后产出一张空精灵，而且不报错。
    """
    (tmp_path / M.AI_MODEL).write_bytes(b"stub")          # 让 ai_available 为真
    blank = Image.new("RGBA", (128, 128), (0, 0, 0, 0))

    monkeypatch.setattr(M, "_AI_SESSIONS", {})
    monkeypatch.setattr(M, "matte_ai", lambda img, d, **kw: blank.copy())

    info = {}
    out = M.matte_auto(_sprite(), "auto", tmp_path, info=info)
    assert info == {"backend": "color", "fallback": True}
    assert M.fg_ratio(out) > 0.05, "回退后必须真的抠出东西"


def test_auto_keeps_u2net_when_sane(monkeypatch, tmp_path):
    """u2net 结果合理时不要多此一举（结果对象应原样返回）。"""
    (tmp_path / M.AI_MODEL).write_bytes(b"stub")
    ok = M.matte(_sprite())

    monkeypatch.setattr(M, "_AI_SESSIONS", {})
    monkeypatch.setattr(M, "matte_ai", lambda img, d, **kw: ok)

    info = {}
    out = M.matte_auto(_sprite(), "auto", tmp_path, info=info)
    assert out is ok
    assert info == {"backend": "u2net", "fallback": False}


def test_forced_backends_do_not_fall_back(monkeypatch, tmp_path):
    """强制 u2net（用户显式选择）不该被偷偷换掉，否则调参结果没法复现。"""
    (tmp_path / M.AI_MODEL).write_bytes(b"stub")
    blank = Image.new("RGBA", (128, 128), (0, 0, 0, 0))

    monkeypatch.setattr(M, "_AI_SESSIONS", {})
    monkeypatch.setattr(M, "matte_ai", lambda img, d, **kw: blank)

    info = {}
    out = M.matte_auto(_sprite(), "u2net", tmp_path, info=info)
    assert info == {"backend": "u2net", "fallback": False}
    assert M.fg_ratio(out) == 0.0


# ---------------- Qwen 原生透明：别再重抠一遍 ----------------
#
# 现场问题：底模是 Qwen-Image-2.1、图本来就带 alpha，用户又勾了「抠图」，
# 于是 isnet 把白毛角色的额头/脸颊当背景咬掉 —— 看到的就是「抠图把脸抠没了」。

def _rgba_with_alpha() -> Image.Image:
    """干净的 alpha：主体实心、四周全透明。"""
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([30, 24, 98, 100], fill=(235, 235, 235, 255))
    return im


def test_flatten_white_does_not_blacken_transparent_area():
    """透明区必须合成到白底：convert("RGB") 会留下 VAE 噪声色，分割模型会被带偏。"""
    im = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([2, 2, 5, 5], fill=(200, 30, 30, 255))
    flat = M.flatten_white(im)
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)
    assert flat.getpixel((3, 3)) == (200, 30, 30)
    # 本来就不透明的图不该被改动
    opaque = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
    assert M.flatten_white(opaque).getpixel((0, 0)) == (1, 2, 3)


def test_alpha_trust_accepts_clean_and_rejects_fog():
    ok, obs = M.alpha_trust(_rgba_with_alpha())
    assert ok and obs["mid"] < 0.12

    # 雾状 alpha：大片中间值（提示词里写了 white background 时 Qwen 就出这个）
    fog = Image.new("RGBA", (128, 128), (200, 200, 200, 120))
    ok2, obs2 = M.alpha_trust(fog)
    assert not ok2 and obs2["mid"] > 0.5

    # 全不透明：没有 alpha 可言
    assert M.alpha_trust(Image.new("RGBA", (16, 16), (9, 9, 9, 255)))[0] is False


def test_matte_auto_reuses_native_alpha():
    """带可信 alpha 的图，auto 后端应原样返回（不再跑分割）。"""
    src = _rgba_with_alpha()
    info = {}
    out = M.matte_auto(src, "auto", None, info=info)
    assert info["backend"] == "alpha" and info["alpha_reused"] is True
    assert np.array_equal(np.asarray(out.getchannel("A")),
                          np.asarray(src.getchannel("A")))


def test_matte_auto_still_recuts_when_alpha_is_fog():
    """alpha 不可信时必须真的重抠，不能拿雾当成品交出去。"""
    fog = Image.new("RGBA", (128, 128), (200, 200, 200, 120))
    ImageDraw.Draw(fog).rectangle([30, 30, 90, 90], fill=(200, 30, 30, 255))
    info = {}
    out = M.matte_auto(fog, "color", None, info=info)
    assert info["backend"] == "color"
    a = np.asarray(out.getchannel("A"))
    assert set(np.unique(a)).issubset({0, 255})       # 色键给硬边


def test_fill_holes_fills_interior_but_not_open_gap():
    """主体内部的洞要补回实心；通到边界/外部的空隙不能补。"""
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 108, 108], fill=(200, 30, 30, 255))   # 主体
    d.rectangle([52, 52, 76, 76], fill=(0, 0, 0, 0))           # 内部洞
    out = M.fill_holes(im)
    a = np.asarray(out.getchannel("A"))
    assert a[64, 64] == 255                                    # 洞被补上

    # 「两腿之间」的缺口通到主体下边界之外 → 不该被填
    legs = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(legs)
    d2.rectangle([20, 20, 56, 110], fill=(0, 0, 255, 255))
    d2.rectangle([72, 20, 108, 110], fill=(0, 0, 255, 255))
    out2 = M.fill_holes(legs)
    a2 = np.asarray(out2.getchannel("A"))
    assert a2[64, 64] == 0                                     # 中间仍透明


# ---------------- Qwen 原生透明：别再重抠一遍 ----------------
#
# 现场问题：底模是 Qwen-Image-2.1、图本来就带 alpha，用户又勾了「抠图」，
# 于是 isnet 把白毛角色的额头/脸颊当背景咬掉 —— 看到的就是「抠图把脸抠没了」。

def _rgba_with_alpha() -> Image.Image:
    """干净的 alpha：主体实心、四周全透明。"""
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([30, 24, 98, 100], fill=(235, 235, 235, 255))
    return im


def test_flatten_white_does_not_blacken_transparent_area():
    """透明区必须合成到白底：convert("RGB") 会留下 VAE 噪声色，分割模型会被带偏。"""
    im = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([2, 2, 5, 5], fill=(200, 30, 30, 255))
    flat = M.flatten_white(im)
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)
    assert flat.getpixel((3, 3)) == (200, 30, 30)
    # 本来就不透明的图不该被改动
    opaque = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
    assert M.flatten_white(opaque).getpixel((0, 0)) == (1, 2, 3)


def test_alpha_trust_accepts_clean_and_rejects_fog():
    ok, obs = M.alpha_trust(_rgba_with_alpha())
    assert ok and obs["mid"] < 0.12

    # 雾状 alpha：大片中间值（提示词里写了 white background 时 Qwen 就出这个）
    fog = Image.new("RGBA", (128, 128), (200, 200, 200, 120))
    ok2, obs2 = M.alpha_trust(fog)
    assert not ok2 and obs2["mid"] > 0.5

    # 全不透明：没有 alpha 可言
    assert M.alpha_trust(Image.new("RGBA", (16, 16), (9, 9, 9, 255)))[0] is False


def test_matte_auto_reuses_native_alpha():
    """带可信 alpha 的图，auto 后端应原样返回（不再跑分割）。"""
    src = _rgba_with_alpha()
    info = {}
    out = M.matte_auto(src, "auto", None, info=info)
    assert info["backend"] == "alpha" and info["alpha_reused"] is True
    assert np.array_equal(np.asarray(out.getchannel("A")),
                          np.asarray(src.getchannel("A")))


def test_matte_auto_still_recuts_when_alpha_is_fog():
    """alpha 不可信时必须真的重抠，不能拿雾当成品交出去。"""
    fog = Image.new("RGBA", (128, 128), (200, 200, 200, 120))
    ImageDraw.Draw(fog).rectangle([30, 30, 90, 90], fill=(200, 30, 30, 255))
    info = {}
    out = M.matte_auto(fog, "color", None, info=info)
    assert info["backend"] == "color"
    a = np.asarray(out.getchannel("A"))
    assert set(np.unique(a)).issubset({0, 255})       # 色键给硬边


def test_fill_holes_fills_interior_but_not_open_gap():
    """主体内部的洞要补回实心；通到边界/外部的空隙不能补。"""
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 108, 108], fill=(200, 30, 30, 255))   # 主体
    d.rectangle([52, 52, 76, 76], fill=(0, 0, 0, 0))           # 内部洞
    out = M.fill_holes(im)
    a = np.asarray(out.getchannel("A"))
    assert a[64, 64] == 255                                    # 洞被补上

    # 「两腿之间」的缺口通到主体下边界之外 → 不该被填
    legs = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(legs)
    d2.rectangle([20, 20, 56, 110], fill=(0, 0, 255, 255))
    d2.rectangle([72, 20, 108, 110], fill=(0, 0, 255, 255))
    out2 = M.fill_holes(legs)
    a2 = np.asarray(out2.getchannel("A"))
    assert a2[64, 64] == 0                                     # 中间仍透明
