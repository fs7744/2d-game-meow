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


def test_matte_on_generated_frame(eng):
    """真实生成图也要能抠干净（背景是模型自己选的纯色，不是纯白）。"""
    from workflows import generate as g

    r = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=8,
                  seed=21, preset="sprite", tag="mattegen")
    raw = Image.open(r["file"]).convert("RGBA")
    out = M.matte(raw)
    a = _alpha(out)
    assert (a == 0).mean() > 0.08, f"透明区域太少，抠图没生效: {(a == 0).mean():.2f}"
    assert (a == 0).mean() < 0.95, "几乎全透明，抠过头了"
    assert set(np.unique(a).tolist()) <= {0, 255}
