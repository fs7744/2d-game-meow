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


def _fixture_frames() -> list[Path]:
    """上一次跑测试留下的夹具帧。poseswap 的目录名带时间戳，必须用 glob。"""
    found = [p for d in ASSETS.glob("_fixture_walk*_frames") for p in d.glob("frame_*.png")]
    return sorted(found)


@pytest.fixture(scope="session")
def walk_frames(eng, assets) -> list[Path]:
    """一套人形精灵帧。素材丢了就现场生成（约 5s），避免测试因缺素材而 skip。"""
    have = _fixture_frames()
    if len(have) >= 4:
        return have

    from workflows import generate as g

    poses = sorted(assets.glob("_pose_walk_*.png"))[:4]
    if len(poses) < 4:
        from tools.make_pose import main as gen_pose

        gen_pose()
        poses = sorted(assets.glob("_pose_walk_*.png"))[:4]
    g.poseswap(eng, pose_images=poses, prompt="white cat knight, red cape, front view",
               steps=12, seed=999, width=256, height=256, cols=4, tag="_fixture_walk",
               gate=False)          # 夹具要的是「一定有 4 帧」，不是「尽量好」
    have = _fixture_frames()
    if len(have) < 4:              # pragma: no cover
        raise AssertionError("夹具帧生成失败")
    return have


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient。训练测试也要用，所以放在 conftest 共用。"""
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c
