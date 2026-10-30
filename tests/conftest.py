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
    """整套测试共用一个 Engine，避免反复加载权重。

    **必须缓存实例**：session 级 fixture 只是缓存「函数结果」，
    但这里每次调用都会 `Engine(...)` 新建一个——引擎是管线的持有者，
    新实例 = 新的管线缓存 = 新的显存占用。
    实测不缓存时连续跑测试，峰值显存能到 14.9GB（单跑只有 5.78GB）。
    """
    from workflows.engine import Engine

    if not hasattr(eng, "_instance"):
        eng._instance = Engine(root / "models")
    return eng._instance


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
    """上一次跑测试留下的夹具帧。poseswap 的目录名带时间戳，必须用 glob。

    **必须递归找**：产物落在「项目目录」下（assets/projects/default/），
    不是 assets 根——这是项目作用域隔离之后的行为。以前这里用 `ASSETS.glob`
    只在根下找，于是每次跑测试都以为"没有夹具"、重新生成一遍
    （4 帧 × 一次 poseswap），还会在生成后仍然报「夹具帧生成失败」。
    """
    found = [p for d in ASSETS.rglob("_fixture_walk*_frames")
             for p in d.glob("frame_*.png")]
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


# ---------------- SD 归档后的统一跳过 ----------------
#
# SD 权重已移到 `models/_archive/sd/`（见 README「底模」）。历史上有一批用例
# 直接 `txt2img(model="sd15")`，权重不在就抛
# `FileNotFoundError: 权重缺失: .../checkpoints/sd15`——**这不是回归**，
# 而是「这台机器上跑不了 SD」。逐条加 skipif 有 40 多条，容易漏，
# 于是统一在这里把它转成 skip（带明确原因），权重搬回来就自动恢复执行。
#
# 只匹配 `权重缺失` + SD 路径：Qwen 权重缺失仍然会正常失败，
# 不会被这条规则掩盖。

_SD_HINTS = ("checkpoints\\sd15", "checkpoints/sd15",
             "checkpoints\\sdxl", "checkpoints/sdxl",
             "controlnet", "ipadapter", "animatediff", "models\\lora")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None:
        err = call.excinfo.value
        if isinstance(err, FileNotFoundError) and "权重缺失" in str(err) \
                and any(h in str(err) for h in _SD_HINTS):
            rep.outcome = "skipped"
            rep.longrepr = (f"SD 权重已归档到 models/_archive/sd/"
                            f"（搬回后自动恢复）：{err}")
