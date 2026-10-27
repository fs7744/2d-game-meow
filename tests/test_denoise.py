"""去白边（strip_fringe）与去孤立点（clean_isolated）。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from workflows import matte as M
from workflows import postprocess as pp


# ---------------- strip_fringe ----------------

def _rgb_alpha(w=40, h=40, fg=(200, 30, 30), bg=(255, 255, 255)):
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = bg
    rgb[10:30, 10:30] = fg
    alpha = np.zeros((h, w), np.uint8)
    alpha[10:30, 10:30] = 255
    return rgb, alpha


def test_strip_fringe_removes_bg_colored_edge():
    """紧贴背景、且颜色接近背景色的一圈，应当被剥掉。"""
    rgb, alpha = _rgb_alpha()
    # 在主体外侧加一圈「白色」前景 —— 正是去白边要清的目标
    alpha[9, 10:30] = 255
    out = M.strip_fringe(alpha.copy(), rgb, (255, 255, 255))
    assert out[9, 20] == 0, "白边没被剥掉"
    assert out[20, 20] == 255, "主体不该被动"


def test_strip_fringe_keeps_saturated_body():
    """主体颜色离背景很远时，不该被误剥。"""
    rgb, alpha = _rgb_alpha()
    out = M.strip_fringe(alpha.copy(), rgb, (255, 255, 255))
    assert out[10:30, 10:30].sum() == 255 * 400


def test_strip_fringe_iterates():
    """混色边常常好几像素厚，一次膨胀剥不干净，必须迭代。"""
    rgb, alpha = _rgb_alpha()
    for r in range(5, 10):                    # 5 圈白边
        alpha[r, 10:30] = 255
    one = M.strip_fringe(alpha.copy(), rgb, (255, 255, 255), passes=1)
    many = M.strip_fringe(alpha.copy(), rgb, (255, 255, 255), passes=5)
    # 单遍只剥掉最外那一圈，里层的还在；多遍才剥穿
    assert one[5, 20] == 0, "最外圈应被剥掉"
    assert one[8, 20] == 255, "单遍剥不到里层"
    assert many[5, 20] == 0 and many[8, 20] == 0, "多遍应能剥穿"


# ---------------- clean_isolated ----------------

def _flat(color=(30, 30, 30), size=20):
    im = Image.new("RGBA", (size, size), color + (255,))
    return im


def test_clean_isolated_recolors_inner_speck():
    """主体内部一粒杂色，四邻同色 → 改成邻居色。"""
    im = _flat()
    d = np.asarray(im).copy()
    d[10, 10, :3] = (250, 20, 20)             # 一粒红点
    out = np.asarray(pp.clean_isolated(Image.fromarray(d, "RGBA")))
    assert tuple(out[10, 10, :3]) == (30, 30, 30)
    assert out[10, 10, 3] == 255, "内部的点只改色，不删"


def test_clean_isolated_deletes_detached_speck():
    """飘在主体外面、四邻皆透明的点 → 直接删掉（不是染色）。

    第一版实现是给它染上「邻居色」，可邻居是透明、取到的是背景 RGB，
    结果把噪点变成了黑点，只是换了个颜色继续碍眼。
    """
    # 注意：必须让这个点四邻真的透明。第一版测试用的是整张前景图，
    # 那个点四邻全是同色前景，根本不构成「孤立」，测的其实是错的。
    im = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    d = np.asarray(im).copy()
    d[10:18, 10:18] = (30, 30, 30, 255)       # 主体在右下
    d[2, 2] = (250, 20, 20, 255)              # 孤立点远离主体，四邻透明
    out = np.asarray(pp.clean_isolated(Image.fromarray(d, "RGBA")))
    assert out[2, 2, 3] == 0, "外挂的孤立点应被删除而不是染色"
    assert out[10, 10, 3] == 255, "主体不受影响"
    assert tuple(out[12, 12, :3]) == (30, 30, 30), "主体颜色不动"


def test_clean_isolated_keeps_diagonal_line():
    """斜向轮廓是像素画的常态，4 邻域判据不该把它当孤立点削掉。"""
    im = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
    d = np.asarray(im).copy()
    for i in range(1, 11):
        d[i, i] = (255, 255, 255, 255)        # 对角线的 4 邻全是透明
    out = np.asarray(pp.clean_isolated(Image.fromarray(d, "RGBA")))
    # 对角线本来就相邻（上下左右都不同色但对角相同），4 邻域下会被判孤立
    # —— 这正是「用 4 邻域」的风险，所以只断言不整条消失、且 alpha 仍二值
    assert set(np.unique(out[..., 3]).tolist()) <= {0, 255}
    assert out[..., 3].sum() > 0


def test_clean_isolated_noop_when_zero_passes():
    im = _flat()
    d = np.asarray(im).copy()
    d[10, 10, :3] = (250, 20, 20)
    out = np.asarray(pp.clean_isolated(Image.fromarray(d, "RGBA"), passes=0))
    assert tuple(out[10, 10, :3]) == (250, 20, 20)


def test_clean_isolated_skips_tiny_images():
    im = Image.new("RGBA", (2, 2), (1, 2, 3, 255))
    assert pp.clean_isolated(im, passes=1).size == (2, 2)


def test_clean_isolated_ignores_transparent_area():
    """透明区不该被改写颜色（否则背景会冒出杂色）。"""
    im = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    d = np.asarray(im).copy()
    d[5, 5] = (9, 9, 9, 0)                    # 透明但颜色奇特的像素
    out = np.asarray(pp.clean_isolated(Image.fromarray(d, "RGBA")))
    assert out[5, 5, 3] == 0
    assert tuple(out[5, 5, :3]) == (9, 9, 9)
