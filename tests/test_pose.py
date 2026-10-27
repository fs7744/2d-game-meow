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


def _humanoid(assets) -> Path:
    """找一张全身人形精灵；没有就用骨架帧反推一张。"""
    cands = sorted(assets.glob("walk*_frames/frame_000*.png"))
    if cands:
        return cands[0]
    pytest.skip("没有可用的全身人形测试图")


@pytest.fixture(scope="module")
def ex_nodet() -> PoseExtractor:
    return PoseExtractor(MODEL_DIR, detector="none")


@pytest.fixture(scope="module")
def ex_auto() -> PoseExtractor:
    return PoseExtractor(MODEL_DIR, detector="auto")


def test_backend_detection(ex_nodet, ex_auto):
    assert ex_nodet.backend == "none"
    assert ex_auto.backend in ("yolox", "torchvision", "none")


def test_keypoints_cover_major_joints(ex_nodet, assets):
    kp = ex_nodet.keypoints(Image.open(_humanoid(assets)).convert("RGB"))
    assert set(kp) == set(range(18)), "必须输出 OpenPose-18 全部索引"
    visible = [i for i, (_, _, s) in kp.items() if s >= 0.25]
    assert len(visible) >= 10, f"可见关节太少: {visible}"


def test_joints_inside_image_and_ordered(ex_nodet, assets):
    kp = ex_nodet.keypoints(Image.open(_humanoid(assets)).convert("RGB"))
    for i, (x, y, s) in kp.items():
        assert -0.05 <= x <= 1.05, f"关节 {i} x 越界 {x}"
        assert -0.05 <= y <= 1.05, f"关节 {i} y 越界 {y}"
    # 头部必须在髋部上方（归一化 y 更小）
    if kp[0][2] > 0.3 and kp[8][2] > 0.3:
        assert kp[0][1] < kp[8][1], "鼻子应当高于右髋"


def test_pose_map_is_rendered(ex_nodet, assets):
    img = Image.open(_humanoid(assets)).convert("RGB")
    pm = ex_nodet.pose_map(img, size=(256, 256))
    assert pm.size == (256, 256)
    arr = np.asarray(pm)
    colored = (arr.sum(axis=2) > 30).sum()
    assert colored > 200, "骨架图几乎是全黑，渲染失败"


def test_detector_agrees_with_whole_image(ex_auto, ex_nodet, assets):
    """有检测器时，主要关节位置应与整图回归接近（差异 < 图像 12%）。"""
    if ex_auto.backend == "none":
        pytest.skip("两种检测器都不可用")
    img = Image.open(_humanoid(assets)).convert("RGB")
    a = ex_auto.keypoints(img)
    b = ex_nodet.keypoints(img)
    errs = [abs(a[i][0] - b[i][0]) + abs(a[i][1] - b[i][1])
            for i in range(18) if a[i][2] > 0.4 and b[i][2] > 0.4]
    assert errs, "没有共同可见关节，无法比较"
    assert float(np.median(errs)) < 0.24, f"两种后端差异过大: {np.median(errs):.2f}"


def test_cli_grid_mode(tmp_path, root):
    """精灵图按网格切片 -> 批量出骨架图。"""
    sheet = tmp_path / "sheet.png"
    frames = sorted((root / "assets").glob("walk*_frames/frame_000*.png"))[:4]
    if len(frames) < 2:
        pytest.skip("缺少切片素材")
    ims = [Image.open(f).convert("RGBA").resize((128, 128)) for f in frames]
    canvas = Image.new("RGBA", (128 * len(ims), 128), (0, 0, 0, 0))
    for i, im in enumerate(ims):
        canvas.paste(im, (i * 128, 0))
    canvas.save(sheet)

    out = tmp_path / "poses"
    r = subprocess.run(
        [sys.executable, str(root / "tools" / "pose_from_image.py"), str(sheet),
         "--grid", f"{len(ims)}x1", "-o", str(out), "--detector", "none", "--size", "128"],
        capture_output=True, text=True, timeout=600, cwd=str(root))
    assert r.returncode == 0, r.stderr[-500:]
    assert len(list(out.glob("pose_*.png"))) == len(ims)
