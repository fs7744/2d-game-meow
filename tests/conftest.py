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
