"""帧质量评分（workflows/quality.py）：纯图像度量，不需要生成模型。"""
from __future__ import annotations

from PIL import Image, ImageDraw

from workflows import quality


def _sprite(size: int = 64, box=(16, 8, 48, 60)) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle(box, fill=(200, 40, 40, 255))
    return im


def test_coverage_counts_alpha_only():
    assert 0.2 < quality.coverage(_sprite()) < 0.8
    # 没抠图的 RGB 图 alpha 全 255 -> 1.0，代表「这个信号不可用」
    assert quality.coverage(Image.new("RGB", (32, 32), (1, 2, 3))) == 1.0


def test_has_alpha():
    assert quality.has_alpha(_sprite()) is True
    assert quality.has_alpha(Image.new("RGB", (8, 8), (0, 0, 0))) is False


def test_centroid_and_drift():
    a = _sprite(box=(4, 4, 20, 20))
    b = _sprite(box=(40, 40, 56, 56))
    ca, cb = quality.centroid(a), quality.centroid(b)
    assert ca and cb and ca[0] < cb[0] and ca[1] < cb[1]
    assert quality.drift(a, a) == 0.0          # 自己跟自己不漂
    assert quality.drift(a, b) > 0.3           # 换到右下角必然漂


def test_drift_of_empty_frame_is_worst():
    empty = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    assert quality.drift(_sprite(32), empty) == 1.0


def test_edge_touch_detected():
    # 贴到底边的帧会被判「脚被裁掉」，这是精灵对齐的硬伤
    im = _sprite(box=(10, 10, 40, 64))
    assert "bottom" in quality.evaluate(im)["edge"]
    # 居中不贴边的帧不该有任何边
    assert quality.evaluate(_sprite())["edge"] == []


def test_edges_ignored_without_alpha():
    """没抠图时四条边全是前景，贴边判据必须整体失效，否则每帧都被判被裁。"""
    assert quality.evaluate(Image.new("RGB", (32, 32), (9, 9, 9)))["edge"] == []


class _FakePose:
    """假装姿态模型：固定返回 n 个可见关节。"""

    def __init__(self, n: int) -> None:
        self.n = n

    def visible_points(self, img, thr: float = 0.25):   # noqa: ARG002
        return {i: (0.5, 0.5) for i in range(self.n)}


def test_gate_needs_extractor():
    ev = quality.evaluate(_sprite(), extractor=None, min_points=8)
    assert ev["points"] == 0
    assert ev["passed"] is False, "没有姿态模型时必须表示「无法判定」，不能装成通过"


def test_gate_passes_with_enough_points():
    ev = quality.evaluate(_sprite(), extractor=_FakePose(12), min_points=8)
    assert ev["points"] == 12 and ev["passed"] is True


def test_gate_fails_on_broken_anatomy():
    ev = quality.evaluate(_sprite(), extractor=_FakePose(3), min_points=8)
    assert ev["passed"] is False


def test_more_points_scores_higher():
    """排序要靠分数：关节多的帧必须排在前面（挑帧就是比这个）。"""
    good = quality.evaluate(_sprite(), extractor=_FakePose(15))
    bad = quality.evaluate(_sprite(), extractor=_FakePose(4))
    assert good["score"] > bad["score"]


def test_cropped_frame_loses_to_whole_frame():
    """同样 12 个关节，被裁掉脚的那帧要扣分（贴合精灵对齐需求）。"""
    whole = quality.evaluate(_sprite(), extractor=_FakePose(12))
    cut = quality.evaluate(_sprite(box=(10, 10, 40, 64)), extractor=_FakePose(12))
    assert whole["score"] > cut["score"]
    assert cut["passed"] is False
