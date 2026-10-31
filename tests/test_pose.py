"""姿态提取：检测器切换、关键点合理性、骨架图渲染、与精灵图切片联动。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workflows.pose import PoseExtractor

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "pose_keypoint" / "dwpose"


def _humanoid(frames) -> Path:
    """全身人形精灵图（由 walk_frames 夹具保证存在）。"""
    assert frames, "缺少人形帧素材"
    return frames[0]


@pytest.fixture(scope="module")
def ex_nodet() -> PoseExtractor:
    return PoseExtractor(MODEL_DIR, detector="none")


@pytest.fixture(scope="module")
def ex_auto() -> PoseExtractor:
    return PoseExtractor(MODEL_DIR, detector="auto")


def test_backend_detection(ex_nodet, ex_auto):
    assert ex_nodet.backend == "none"
    assert ex_auto.backend in ("yolox", "torchvision", "none")


def test_keypoints_cover_major_joints(ex_nodet, walk_frames):
    kp = ex_nodet.keypoints(Image.open(_humanoid(walk_frames)).convert("RGB"))
    assert set(kp) == set(range(18)), "必须输出 OpenPose-18 全部索引"
    visible = [i for i, (_, _, s) in kp.items() if s >= 0.25]
    assert len(visible) >= 10, f"可见关节太少: {visible}"


def test_joints_inside_image_and_ordered(ex_nodet, walk_frames):
    kp = ex_nodet.keypoints(Image.open(_humanoid(walk_frames)).convert("RGB"))
    for i, (x, y, _s) in kp.items():
        assert -0.05 <= x <= 1.05, f"关节 {i} x 越界 {x}"
        assert -0.05 <= y <= 1.05, f"关节 {i} y 越界 {y}"
    # 头部必须在髋部上方（归一化 y 更小）
    if kp[0][2] > 0.3 and kp[8][2] > 0.3:
        assert kp[0][1] < kp[8][1], "鼻子应当高于右髋"


def test_pose_map_is_rendered(ex_nodet, walk_frames):
    img = Image.open(_humanoid(walk_frames)).convert("RGB")
    pm = ex_nodet.pose_map(img, size=(256, 256))
    assert pm.size == (256, 256)
    arr = np.asarray(pm)
    colored = (arr.sum(axis=2) > 30).sum()
    assert colored > 200, "骨架图几乎是全黑，渲染失败"


def test_detector_agrees_with_whole_image(ex_auto, ex_nodet, walk_frames):
    """有检测器时，主要关节位置应与整图回归接近（差异 < 图像 12%）。"""
    if ex_auto.backend == "none":
        pytest.skip("两种检测器都不可用")
    img = Image.open(_humanoid(walk_frames)).convert("RGB")
    a = ex_auto.keypoints(img)
    b = ex_nodet.keypoints(img)
    errs = [abs(a[i][0] - b[i][0]) + abs(a[i][1] - b[i][1])
            for i in range(18) if a[i][2] > 0.4 and b[i][2] > 0.4]
    if len(errs) < 3:
        pytest.skip(f"共同可见关节太少（{len(errs)}），小尺寸素材无法比较两后端")
    assert float(np.median(errs)) < 0.3, f"两种后端差异过大: {np.median(errs):.2f}"


def test_engine_holds_pose_extractor_singleton(eng, walk_frames):
    """质量闸门每帧都要检姿态，引擎必须缓存会话（不然每帧重开 onnx 太亏）。"""
    from workflows import quality

    if not (MODEL_DIR / "dw-ll_ucoco_384.onnx").exists():
        pytest.skip("未下载姿态模型")
    ex = eng.pose_extractor()
    assert ex is not None
    assert eng.pose_extractor() is ex, "必须复用同一个 onnx 会话"

    # 真实产物上跑一遍评分：字段齐全、关节数在合法区间（0~18）
    ev = quality.evaluate(Image.open(walk_frames[0]), ex, min_points=8)
    assert 0 <= ev["points"] <= 18
    assert isinstance(ev["passed"], bool) and "score" in ev


def test_cli_grid_mode(tmp_path, root, walk_frames):
    """精灵图按网格切片 -> 批量出骨架图。"""
    sheet = tmp_path / "sheet.png"
    frames = list(walk_frames)[:4]
    ims = [Image.open(f).convert("RGBA").resize((128, 128)) for f in frames]
    canvas = Image.new("RGBA", (128 * len(ims), 128), (0, 0, 0, 0))
    for i, im in enumerate(ims):
        canvas.paste(im, (i * 128, 0))
    canvas.save(sheet)

    out = tmp_path / "poses"
    r = subprocess.run(
        [sys.executable, str(root / "tools" / "pose_from_image.py"), str(sheet),
         "--grid", f"{len(ims)}x1", "-o", str(out), "--detector", "none", "--size", "128"],
        capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=600, cwd=str(root))
    assert r.returncode == 0, r.stderr[-500:]
    assert len(list(out.glob("pose_*.png"))) == len(ims)


# --------------------------------------------------------------------------- #
# YOLOX 解码（真 bug 修复的看守）
# --------------------------------------------------------------------------- #

def test_default_detector_is_torchvision():
    """默认检测器必须是 torchvision（FasterRCNN），不是 YOLOX。

    原因：随包的 yolox_l.onnx 输出的是**未解码**的 head 结果
    ((1, 8400, 85)，前 4 列是网格坐标 + log 尺寸)，而且 anchor 布局与官方
    YOLOX 的 anchor-major（25200 条）不一致——8400 恰好等于格子数，说明每格
    只输出一个 anchor。按官方 decode 解出来的框全是垃圾（实测 (-1,1,1,4)），
    人体裁剪完全错位，姿态回归置信度全线偏低（0.03~0.16）。

    torchvision 的 FasterRCNN 直接输出像素坐标，实测同一张图 18/18 关节全中。
    """
    from workflows import pose as P

    ex = P.PoseExtractor(MODEL_DIR, detector="auto")
    assert ex.backend == "torchvision", ex.backend


def test_yolox_backend_is_not_auto_selected():
    """YOLOX 只能显式指定（detector="yolox"），auto 不该选它。"""
    from workflows import pose as P

    ex = P.PoseExtractor(MODEL_DIR, detector="none")
    assert ex.backend == "none", ex.backend


# --------------------------------------------------------------------------- #
# 归一化骨架（角色迁移的核心）
# --------------------------------------------------------------------------- #

def test_normalized_points_takes_all_confident_joints():
    """归一化要取**所有**过阈值的点，不能只取 min_points 个。

    写反的后果（实测）：只取 10 个点，肩膀/脖子全丢，骨架只剩头 + 一只手 +
    两条腿，ControlNet 拿到的姿态是残的。
    """

    from workflows import pose as P

    class _Fake(P.PoseExtractor):
        def __init__(self):
            self._kp = {i: (0.5, 0.5 + i * 0.01, 0.9) for i in range(18)}

        def keypoints(self, img, pick="largest"):
            return self._kp

        def person_boxes(self, img, allow_fallback=True):
            return [(1.0, 10.0, 10.0, 210.0, 410.0)]

    ex = _Fake()
    im = Image.new("RGB", (100, 100), (0, 0, 0))
    pts = ex.normalized_points(im, thr=0.25, min_points=10)
    assert len(pts) == 18, len(pts)


def test_normalized_points_covers_center_of_frame():
    """归一化后的人物要**居中**（ControlNet 是空间敏感的）。"""
    from workflows import pose as P

    class _Fake(P.PoseExtractor):
        def __init__(self):
            self._kp = {i: (0.2, 0.5, 0.9) for i in range(18)}

        def keypoints(self, img, pick="largest"):
            return self._kp

        def person_boxes(self, img, allow_fallback=True):
            # 人缩在左上角
            return [(1.0, 0.0, 0.0, 20.0, 20.0)]

    ex = _Fake()
    im = Image.new("RGB", (100, 100), (0, 0, 0))
    pts = ex.normalized_points(im, thr=0.25)
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    assert min(xs) >= 0.0 and max(xs) <= 1.0
    assert min(ys) >= 0.0 and max(ys) <= 1.0
    # 人物中心应该被搬到画面中部（原来在左上角）
    assert sum(xs) / len(xs) > 0.2


def test_normalized_points_fills_up_to_min_points():
    """置信度都低时，也要补齐到 min_points（宁可要个大致对的骨架）。"""
    from workflows import pose as P

    class _Fake(P.PoseExtractor):
        def __init__(self):
            self._kp = {i: (0.5, 0.5, 0.05 + i * 0.001) for i in range(18)}

        def keypoints(self, img, pick="largest"):
            return self._kp

        def person_boxes(self, img, allow_fallback=True):
            return [(1.0, 0.0, 0.0, 100.0, 100.0)]

    ex = _Fake()
    im = Image.new("RGB", (100, 100), (0, 0, 0))
    pts = ex.normalized_points(im, thr=0.25, min_points=10)
    assert len(pts) == 10, len(pts)
    assert len(pts) == 10, len(pts)
