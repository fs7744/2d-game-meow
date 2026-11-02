"""项目内的姿势集 / 成片集（`assets/projects/<项目>/poses|pose_frames/`）。

**为什么要单独测这一层**：它是"动作抽帧 → 批量出片 → 挑候选"这条链的地基，
而且有三条**必须在存储层就钉住**的语义：

1. **同名拒绝**（不是覆盖）：用户的心智是"再存一份"，静默替换会让人以为
   上一次的还在。
2. **候选只增不删**：重新生成只是追加，旧的必须还能选回来。
3. **名字不能跳出项目目录**：名字直接当文件夹名，`../` 之类的必须挡掉。

生成（`posebatch.run_batch`）不在这里测 —— 那条要真跑 Qwen，按项目约定
不在常规测试里开（见 README 的测试子集表）。
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from workflows import paths
from workflows import poseset

#: 1×1 的 PNG（最小的合法图，足够验证"字节原样落盘"）
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)
PNG_DATA_URL = "data:image/png;base64," + PNG_B64

PROJECT = "ptest_poseset"


@pytest.fixture(autouse=True)
def _clean():
    d = paths.PROJECTS_DIR / PROJECT
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    yield


def _frames(n: int = 3):
    return [{"t": i * 0.25, "yaw": 90, "pose": {"bones": {"x": i}}, "png": PNG_DATA_URL}
            for i in range(1, n + 1)]


def test_pose_set_lands_in_project_folder():
    one = poseset.save_pose_set(PROJECT, "走路", _frames(3),
                                source={"clip": "Walk", "count": 3})
    assert one["name"] == "走路" and one["count"] == 3
    assert one["cover"].startswith("/files/projects/")

    folder = paths.PROJECTS_DIR / PROJECT / poseset.POSES_DIR / "走路"
    assert folder.is_dir(), "必须落在项目的 poses/<名字>/ 下"
    assert (folder / poseset.SET_FILE).is_file()
    names = sorted(p.name for p in folder.glob("*.png"))
    assert names == ["f_001.png", "f_002.png", "f_003.png"]
    # 字节原样落盘（不是存了张空图）
    assert (folder / "f_001.png").read_bytes() == base64.b64decode(PNG_B64)

    detail = poseset.load_pose_set(PROJECT, "走路")
    assert [f["index"] for f in detail["frames"]] == [1, 2, 3]
    assert all(f["url"].endswith(".png") for f in detail["frames"])


def test_pose_set_duplicate_name_is_rejected():
    poseset.save_pose_set(PROJECT, "走路", _frames(2))
    with pytest.raises(poseset.PoseSetExists):
        poseset.save_pose_set(PROJECT, "走路", _frames(5))
    # 原文件没被覆盖（还是 2 帧）
    assert len(poseset.load_pose_set(PROJECT, "走路")["frames"]) == 2
    # 换个名字就正常
    poseset.save_pose_set(PROJECT, "走路 2", _frames(5))
    assert [s["name"] for s in poseset.list_pose_sets(PROJECT)] == ["走路 2", "走路"]


def test_pose_set_rejects_unsafe_name():
    for bad in ("../逃出去", "a/b", "..", ""):
        with pytest.raises((ValueError, poseset.PoseSetError)):
            poseset.save_pose_set(PROJECT, bad, _frames(1))
    assert not (paths.PROJECTS_DIR / PROJECT).joinpath("..").resolve().joinpath(
        "逃出去").exists()


def test_pose_set_rejects_bad_frame_bytes():
    with pytest.raises(poseset.PoseSetError):
        poseset.save_pose_set(PROJECT, "坏帧", [{"png": "data:image/png;base64,bm90YXBuZw=="}])
    # 失败不留半套目录
    assert not (paths.PROJECTS_DIR / PROJECT / poseset.POSES_DIR / "坏帧").exists()


def test_candidates_only_append_never_delete():
    poseset.save_pose_set(PROJECT, "走", _frames(2))
    poseset.create_render(PROJECT, "成品", "走", prompt="a cat")

    # 造两张"生成结果"（占位字节即可，add_candidate 只搬文件）
    tmp = paths.PROJECTS_DIR / PROJECT
    a = tmp / "wpcb_A.png"
    b = tmp / "wpcb_B.png"
    a.write_bytes(base64.b64decode(PNG_B64))
    b.write_bytes(base64.b64decode(PNG_B64) + b"\x00")

    r1 = poseset.add_candidate(PROJECT, "成品", 1, a, seed=111)
    frame = next(f for f in r1["frames"] if f["index"] == 1)
    assert len(frame["candidates"]) == 1 and frame["selected"] == 0
    assert frame["selectedIndex"] == 1

    r2 = poseset.add_candidate(PROJECT, "成品", 1, b, seed=222)
    frame = next(f for f in r2["frames"] if f["index"] == 1)
    assert len(frame["candidates"]) == 2, "重生成要**追加**，不能替换"
    assert frame["selected"] == 1, "新出的自动选中"

    folder = paths.PROJECTS_DIR / PROJECT / poseset.RENDERS_DIR / "成品"
    assert (folder / "f_001_c01.png").is_file(), "旧候选必须还在（用户可能更喜欢它）"
    assert (folder / "f_001_c02.png").is_file()

    # 选回旧的
    r3 = poseset.select_candidate(PROJECT, "成品", 1, 1)
    frame = next(f for f in r3["frames"] if f["index"] == 1)
    assert frame["selected"] == 0 and frame["selectedIndex"] == 1
    assert (folder / "f_001_c02.png").is_file(), "选回去也不该删掉另一个"


def test_render_frames_follow_pose_set():
    poseset.save_pose_set(PROJECT, "跳", _frames(4))
    one = poseset.create_render(PROJECT, "跳成品", "跳")
    assert [f["index"] for f in one["frames"]] == [1, 2, 3, 4]
    assert all(f["selectedIndex"] == -1 for f in one["frames"])
    assert one["done"] == 0 and one["total"] == 4

    # 同名拒绝
    with pytest.raises(poseset.PoseSetExists):
        poseset.create_render(PROJECT, "跳成品", "跳")

    # 帧源路径能拿到（批量生成时当条件图）
    p = poseset.frame_source_path(PROJECT, "跳", 3)
    assert Path(p).is_file() and p.endswith("f_003.png")
    assert poseset.frame_indexes(PROJECT, "跳") == [1, 2, 3, 4]


def test_candidate_cap_and_bad_index():
    poseset.save_pose_set(PROJECT, "跑", _frames(1))
    poseset.create_render(PROJECT, "跑成品", "跑")
    tmp = paths.PROJECTS_DIR / PROJECT
    for i in range(poseset.MAX_CANDIDATES):
        f = tmp / f"wpcb_{i}.png"
        f.write_bytes(base64.b64decode(PNG_B64))
        poseset.add_candidate(PROJECT, "跑成品", 1, f)
    f = tmp / "wpcb_over.png"
    f.write_bytes(base64.b64decode(PNG_B64))
    with pytest.raises(poseset.PoseSetError):
        poseset.add_candidate(PROJECT, "跑成品", 1, f)
    with pytest.raises(poseset.PoseSetError):
        poseset.add_candidate(PROJECT, "跑成品", 99, f)
    with pytest.raises(poseset.PoseSetError):
        poseset.select_candidate(PROJECT, "跑成品", 1, 99)


def test_list_renders_reports_progress_and_cover():
    poseset.save_pose_set(PROJECT, "走", _frames(2))
    poseset.create_render(PROJECT, "成品", "走")
    tmp = paths.PROJECTS_DIR / PROJECT
    f = tmp / "wpcb_x.png"
    f.write_bytes(base64.b64decode(PNG_B64))
    poseset.add_candidate(PROJECT, "成品", 1, f)

    items = poseset.list_renders(PROJECT)
    assert len(items) == 1
    assert items[0]["done"] == 1 and items[0]["total"] == 2
    assert items[0]["cover"].endswith("f_001_c01.png")
