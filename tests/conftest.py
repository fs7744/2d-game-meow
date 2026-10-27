"""pytest 公共夹具。"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ASSETS = ROOT / "assets"


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def assets() -> Path:
    ASSETS.mkdir(exist_ok=True)
    return ASSETS


@pytest.fixture(scope="session")
def eng(root):
    """整套测试共用一个 Engine，避免反复加载权重。"""
    from workflows.engine import Engine

    return Engine(root / "models")


@pytest.fixture(scope="session")
def pose_frames(assets) -> list[Path]:
    """确保有骨架帧可测；没有就现场生成。"""
    frames = sorted(assets.glob("_pose_walk_*.png"))
    if len(frames) < 4:
        sys.path.insert(0, str(ROOT))
        from tools.make_pose import main as gen_pose

        gen_pose()
        frames = sorted(assets.glob("_pose_walk_*.png"))
    return frames


@pytest.fixture(scope="session")
def walk_frames(eng, assets) -> list[Path]:
    """一套人形精灵帧。素材丢了就现场生成（约 5s），避免测试因缺素材而 skip。"""
    d = assets / "_fixture_walk_frames"
    have = sorted(d.glob("frame_*.png")) if d.exists() else []
    if len(have) >= 4:
        return have

    from workflows import generate as g

    poses = sorted(assets.glob("_pose_walk_*.png"))[:4]
    if len(poses) < 4:
        from tools.make_pose import main as gen_pose

        gen_pose()
        poses = sorted(assets.glob("_pose_walk_*.png"))[:4]
    r = g.poseswap(eng, pose_images=poses, prompt="white cat knight, red cape, front view",
                   steps=12, seed=999, width=256, height=256, cols=4, tag="_fixture_walk")
    return sorted(Path(r["frames_dir"]).glob("frame_*.png"))


@pytest.fixture(scope="session")
def walk_frames(eng, assets) -> list[Path]:
    """一套人形精灵帧。素材丢了就现场生成（约 5s），避免测试因缺素材而 skip。"""
    d = assets / "_fixture_walk_frames"
    have = sorted(d.glob("frame_*.png")) if d.exists() else []
    if len(have) >= 4:
        return have

    from workflows import generate as g

    poses = sorted(assets.glob("_pose_walk_*.png"))[:4]
    if len(poses) < 4:
        from tools.make_pose import main as gen_pose

        gen_pose()
        poses = sorted(assets.glob("_pose_walk_*.png"))[:4]
    r = g.poseswap(eng, pose_images=poses, prompt="white cat knight, red cape, front view",
                   steps=12, seed=999, width=256, height=256, cols=4, tag="_fixture_walk")
    return sorted(Path(r["frames_dir"]).glob("frame_*.png"))
