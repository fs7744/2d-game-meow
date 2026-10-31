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

#: **测试产物根目录**。测试期间的一切生成物（产物 PNG、索引、上传、蒙版）
#: 都落在这里，不进 `assets/`。
#:
#: 为什么要隔离：`assets/` 是**项目要用的运行时目录**——索引、上传、用户项目的
#: 产物都在那儿。以前 pytest 直接往 `assets/` 里写，跑一轮就多出几十个
#: `apiwbg_*.png` / `w1_*.png`，把真正的交付物淹没，删项目时还会误删。
#:
#: 必须在**导入 workflows 之前**设好：`workflows.paths.ASSETS` 是模块级常量，
#: import 时求值。conftest 在 collection 阶段加载，早于所有测试模块的 import，
#: 所以这里的时序是对的。
TEST_OUT = ROOT / "tests" / "_out"
os.environ["MEOW_ASSETS_DIR"] = str(TEST_OUT / "assets")

#: **模板根也要隔离**。模板以前漏了这一步：`templates/*.mproj` 是跨项目共享的
#: 用户资源，测试里的 `save_template` 直接往里写，于是跑一轮就在用户的模板列表里
#: 多出 `API_34266.mproj` / `t_io_d4cdd081.mproj` 这类垃圾（名字带随机后缀，
#: 用户根本分不清哪些是自己的）。实盘检查过一次：`templates/` 里 5 个文件全是
#: 测试残留，没有一个是用户建的。
os.environ["MEOW_TEMPLATES_DIR"] = str(TEST_OUT / "templates")

from workflows import paths                                    # noqa: E402

#: 资产根（测试期间指向 tests/_out/assets）
ASSETS = paths.ASSETS


@pytest.fixture(scope="session", autouse=True)
def _assert_isolated():
    """**开场先确认真的隔离了**，否则一条测试都不许跑。

    为什么要有这条：`paths.ASSETS` 是 import 时求值的常量，隔离完全依赖
    「conftest 早于所有模块 import」这个时序。哪天有人在 `conftest` 之前
    import 了 `workflows`（比如加个 `conftest` 上层的 `conftest.py`、
    或把 `paths` 提到环境变量赋值前），隔离就静默失效——测试照样全绿，
    但产物全落进用户的 `assets/`，而且没人会发现。
    这里把它变成**硬失败**：跑不动比悄悄写脏好。
    """
    out = TEST_OUT.resolve()
    for name, p in (("MEOW_ASSETS_DIR", paths.ASSETS),
                    ("MEOW_TEMPLATES_DIR", paths.TEMPLATES_DIR)):
        assert out in Path(p).resolve().parents, (
            f"{name} 没有指向 tests/_out（实际 {p}）——测试会写脏用户的真实目录。"
            f"检查 conftest 里两个环境变量是否在 import workflows 之前设置。")


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def assets() -> Path:
    """测试期间的资产根。**不是**仓库里的 `assets/`。"""
    ASSETS.mkdir(parents=True, exist_ok=True)
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
