"""参考图预处理（refprep）与帧级评分（framescore）。

关键回归点：
  - **透明区必须填白**：直接 convert("RGB") 会让透明区变黑。实测一张
    64×64 的像素角色图有 **66.5% 是纯黑背景**，IP-Adapter 会把黑色
    当成角色的暗部，身份特征被污染。
  - **主体要裁出来居中**：CLIP Vision 在 224² 上切 14×14 patch，主体
    顶满画面时 patch 里只有局部纹理，编码不出整体身份。
  - **帧级评分与主体类型无关**：像素角色、怪物、机甲全都检不出关节，
    所以不能用 pose 分数，必须有 silhouette / identity 这套。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from workflows import framescore
from workflows import refprep


def _pixel_char(size: int = 64) -> Image.Image:
    """像素角色（透明底）。"""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([size // 6, size // 3, size * 5 // 6, size * 9 // 10], fill=(70, 190, 90, 255))
    d.ellipse([size // 4, size // 3 + 4, size * 3 // 4, size * 9 // 10 - 4],
              fill=(110, 225, 120, 255))
    return im


def _opaque_char(size: int = 64) -> Image.Image:
    """不透明的像素角色（白底）。"""
    im = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse([size // 6, size // 3, size * 5 // 6, size * 9 // 10], fill=(70, 190, 90, 255))
    return im


# ---------------- refprep ----------------

def test_flatten_rgb_fills_transparent_with_white():
    """透明区必须填白，不能是黑。"""
    im = _pixel_char()
    out = refprep.flatten_rgb(im)
    arr = np.asarray(out)
    assert tuple(arr[0, 0]) == (255, 255, 255), "透明区应当是白底"
    # 直接 convert("RGB") 会变黑——这正是要修的 bug
    assert tuple(np.asarray(im.convert("RGB"))[0, 0]) == (0, 0, 0)


def test_prepare_reference_removes_black_background():
    """预处理后的参考图不该有大片纯黑背景。

    实测修复前：66.5% 的角色参考图是纯黑，IP-Adapter 身份被污染。
    """
    im = _pixel_char()
    out = refprep.prepare_reference(im, size=512)
    arr = np.asarray(out)
    black_ratio = float((arr.sum(axis=2) == 0).mean())
    assert black_ratio < 0.01, f"纯黑占比 {black_ratio:.1%}，应当接近 0"


def test_prepare_reference_centers_subject():
    """主体要居中——CLIP Vision 靠 14×14 patch 编码，居中才编码得准。"""
    # 主体偏在右下角
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([90, 90, 120, 120], fill=(200, 40, 40, 255))
    out = refprep.prepare_reference(im, size=256)
    arr = np.asarray(out).astype(np.int16)
    mask = np.abs(arr - 255).max(axis=2) > 16
    assert mask.any(), "主体不该被裁没"
    ys, xs = np.nonzero(mask)
    cx = (xs.min() + xs.max()) / 2 / 256.0
    cy = (ys.min() + ys.max()) / 2 / 256.0
    assert abs(cx - 0.5) < 0.15, f"水平中心 {cx:.2f} 应接近 0.5"
    assert abs(cy - 0.5) < 0.15, f"垂直中心 {cy:.2f} 应接近 0.5"


def test_prepare_reference_output_size():
    out = refprep.prepare_reference(_pixel_char(), size=512)
    assert out.size == (512, 512)
    assert out.mode == "RGB"


def test_prepare_reference_keeps_pixels_crisp():
    """像素画缩放必须用最近邻：糊了就不是像素画了。"""
    im = _pixel_char()
    out = refprep.prepare_reference(im, size=512)
    arr = np.asarray(out)
    # 主体区域内应当只有少数几种颜色（每格一色，没被插值）
    fg = arr[np.abs(arr.astype(np.int16) - 255).max(axis=2) > 16]
    assert len(set(map(tuple, fg.reshape(-1, 3).tolist()))) <= 8, "像素被插值了"


def test_is_pixel_art_recognises_pixel_material():
    """像素画应当被认出来（启发式，只影响插值方式）。"""
    assert refprep.is_pixel_art(_pixel_char()) is True


def test_is_pixel_art_rejects_smooth_image():
    """平滑插画不该被当成像素画。"""
    yy, xx = np.mgrid[0:64, 0:64]
    smooth = np.stack([(np.sin(xx / 7 + c) * 60 + np.cos(yy / 9) * 50 + 128)
                        for c in range(3)], axis=2).clip(0, 255).astype(np.uint8)
    assert refprep.is_pixel_art(Image.fromarray(smooth)) is False


# ---------------- framescore ----------------

def test_silhouette_iou_identical_is_one():
    """同一张图跟自己比，IoU 必须是 1。"""
    im = _pixel_char()
    assert framescore.silhouette_iou(im, im) == pytest.approx(1.0, abs=0.02)


def test_silhouette_iou_different_is_low():
    """形状完全不同的两张图，IoU 必须低。"""
    a = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(a).rectangle([0, 0, 30, 30], fill=(255, 0, 0, 255))
    b = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(b).rectangle([34, 34, 63, 63], fill=(255, 0, 0, 255))
    assert framescore.silhouette_iou(a, b) < 0.3


def test_identity_similarity_same_colour_high():
    """同色角色的身份相似度应当高。"""
    a = _pixel_char()
    b = _pixel_char()
    assert framescore.identity_similarity(a, b) > 0.95


def test_identity_similarity_different_colour_low():
    """不同色的角色，身份相似度应当明显低。"""
    a = _pixel_char()
    b = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(b).ellipse([10, 20, 54, 58], fill=(200, 40, 40, 255))
    assert framescore.identity_similarity(a, b) < 0.9


def test_identity_similarity_empty_is_zero():
    """空图不能崩，也不能给出虚高的分。"""
    empty = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    assert framescore.identity_similarity(empty, _pixel_char()) == 0.0


def test_evaluate_scores_good_frame_higher():
    """贴合参考帧 + 身份接近的帧，分数应当更高。"""
    ref = _pixel_char()
    good = _pixel_char()
    # 坏帧：形状不同（右上角一块）、颜色不同
    bad = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(bad).rectangle([50, 0, 63, 13], fill=(200, 40, 40, 255))
    char = _pixel_char()
    ev_good = framescore.evaluate(good, ref, char)
    ev_bad = framescore.evaluate(bad, ref, char)
    assert ev_good["score"] > ev_bad["score"]
    assert ev_good["passed"] is True
    assert ev_bad["passed"] is False


def test_evaluate_detects_cut_off_edges():
    """主体贴边（被切断）要被扣分并判不合格。"""
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([0, 0, 20, 20], fill=(70, 190, 90, 255))
    ev = framescore.evaluate(im, _pixel_char(), _pixel_char())
    assert ev["edge"], "左边贴边应当被报出来"
    assert ev["passed"] is False


def test_pick_best_returns_highest_score():
    """挑候选要挑分数最高的。"""
    img_a = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    img_b = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    cands = [(img_a, {"score": 1.0}), (img_b, {"score": 5.0})]
    img, ev, i = framescore.pick_best(cands)
    assert i == 1 and ev["score"] == 5.0 and img is img_b


def test_pick_best_prefers_first_on_tie():
    """同分时取第一个（保证可复现：候选按 seed 递增生成）。"""
    a = Image.new("RGBA", (8, 8))
    b = Image.new("RGBA", (8, 8))
    img, ev, i = framescore.pick_best([(a, {"score": 3.0}), (b, {"score": 3.0})])
    assert i == 0 and img is a


def test_pick_best_empty_raises():
    with pytest.raises(ValueError, match="没有候选"):
        framescore.pick_best([])


def test_sequence_consistency_reports_drift():
    """整段序列的一致性指标要能算出来。"""
    frames = [_pixel_char() for _ in range(3)]
    res = framescore.sequence_consistency(frames)
    assert "identity_drift" in res and "silhouette_drift" in res
    assert res["identity_drift"] >= 0.0


def test_sequence_consistency_single_frame():
    """单帧没有「漂移」，不该崩。"""
    res = framescore.sequence_consistency([_pixel_char()])
    assert res["identity_drift"] == 0.0
