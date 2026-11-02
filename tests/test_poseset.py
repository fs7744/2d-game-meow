"""项目内的姿势集 / 成片集（`assets/projects/<项目>/poses|pose_frames/`）。

**为什么要单独测这一层**：它是"动作抽帧 → 批量出片 → 挑候选"这条链的地基，
而且有三条**必须在存储层就钉住**的语义：

1. **同名拒绝**（不是覆盖）：用户的心智是"再存一份"，静默替换会让人以为
   上一次的还在。
2. **候选只增不删**：重新生成只是追加，旧的必须还能选回来。
3. **名字不能跳出项目目录**：名字直接当文件夹名，`../` 之类的必须挡掉。

生成（`posebatch.run_batch`）在这里测**编排**，但把 `generate.pose_clone`
换成一个假的落盘函数 —— 真跑 Qwen 要 GPU，按项目约定不在常规测试里开
（见 README 的测试子集表）。这样能钉住"帧表有没有走对、候选有没有追加上"。
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


def _distinct_png(level: int, side: int = 8) -> str:
    """一张**内容不同**的小图（base64）。

    GIF 测试必须用不同的帧：Pillow 写 GIF 时会把连续的相同画面合并成一帧，
    四张一样的 1×1 会变成一个单帧 GIF —— 那不是导出坏了，是优化。"""
    from PIL import Image

    import io

    buf = io.BytesIO()
    Image.new("RGB", (side, side), (level, level, level)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _distinct_png(level: int, side: int = 8) -> str:
    """一张**内容不同**的小图（base64）。

    GIF 测试必须用不同的帧：Pillow 写 GIF 时会把连续的相同画面合并成一帧，
    四张一样的 1×1 会变成一个单帧 GIF —— 那不是导出坏了，是优化。"""
    from PIL import Image

    import io

    buf = io.BytesIO()
    Image.new("RGB", (side, side), (level, level, level)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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


def test_gif_export_for_review():
    """把一套帧合成 GIF（团队审核用）。"""
    from PIL import Image

    poseset.save_pose_set(PROJECT, "走路", [
        {"t": i * 0.2, "yaw": 0, "pose": {}, "png": _distinct_png(40 * i)}
        for i in range(1, 5)
    ])
    info = poseset.export_gif(PROJECT, poseset.POSES_DIR, "走路")
    assert info["frames"] == 4 and info["skipped"] == 0
    assert info["url"].startswith("/files/projects/")
    out = paths.PROJECTS_DIR / PROJECT / poseset.POSES_DIR / "走路" / poseset.GIF_FILE
    assert out.is_file() and out.read_bytes()[:4] in (b"GIF8", b"GIF9")
    with Image.open(out) as im:
        assert im.format == "GIF"
        assert getattr(im, "n_frames", 1) == 4, "GIF 应该有 4 帧"
        # 帧时长：Pillow 把它放在每帧的 info 里，seek 到第 0 帧再读
        im.seek(0)
        # **GIF 只有 10ms 精度**：要的是 125，文件里是 120，返回的也是 120
        assert info["frame_ms"] == 120, info["frame_ms"]
        assert im.info.get("duration") == info["frame_ms"], im.info
        assert im.info.get("loop") == 0, "GIF 要循环播放（loop=0）"


def test_gif_skips_unselected_render_frames():
    """成片集只导**当前选中的候选**，没选中的帧跳过并报数。"""
    poseset.save_pose_set(PROJECT, "跑", _frames(3))
    poseset.create_render(PROJECT, "跑成片", "跑")
    tmp = paths.PROJECTS_DIR / PROJECT
    for i in range(3):
        p = tmp / f"wpcb_g{i}.png"
        p.write_bytes(base64.b64decode(PNG_B64))
        if i < 2:                      # 只给前两帧出图，第三帧故意留空
            poseset.add_candidate(PROJECT, "跑成片", i + 1, p)

    info = poseset.export_gif(PROJECT, poseset.RENDERS_DIR, "跑成片")
    assert info["frames"] == 2, "只有两帧有选中的候选"
    assert info["skipped"] == 1, "没出图的帧要报出来，不能静默少帧"

    # 一帧都没有 → 明确报错（不是给一个空 GIF）
    poseset.create_render(PROJECT, "空成片", "跑")
    with pytest.raises(poseset.PoseSetError):
        poseset.export_gif(PROJECT, poseset.RENDERS_DIR, "空成片")


def test_run_batch_requires_pose_set():
    """`run_batch` 靠 `params["pose_set"]` 去 `poses/<名字>/set.json` 找源帧。

    2026-11-02 这里漏传过：`/pose_renders/{name}/run` 只传了 `render`，
    于是每帧都抛「缺成片集名或姿势集名」，而 `render.json` 已经建好 ——
    界面上的表现就是用户说的「点全部生成，出来一堆空格子但没有图」。
    """
    from workflows import posebatch

    poseset.save_pose_set(PROJECT, "走路", _frames(2))
    poseset.create_render(PROJECT, "走路成片", "走路")
    with pytest.raises(posebatch.PoseBatchError):
        posebatch.run_batch(None, {"project": PROJECT, "render": "走路成片",
                                   "char_image": "char.png"})


def test_run_batch_appends_candidates(monkeypatch, tmp_path):
    """一帧一候选：候选**追加**进 render.json，`bg` 原样透传到 pose_clone。

    `bg` 那条断言是"透明底"的下游保证：面板以前把非 dark 的底色一律映射成
    white，用户选了透明底却出白图。
    """
    from workflows import generate, posebatch

    poseset.save_pose_set(PROJECT, "跑", _frames(2))
    poseset.create_render(PROJECT, "跑成片", "跑")

    seen = []

    def fake_clone(eng, *, pose_image, char_image, seed=-1, scope="", tag="", **kw):
        out = tmp_path / f"gen_{len(seen)}.png"
        out.write_bytes(base64.b64decode(PNG_B64))
        seen.append({"pose": pose_image, "seed": seed, "bg": kw.get("bg")})
        return {"file": str(out), "meta": {"seed": seed}}

    monkeypatch.setattr(generate, "pose_clone", fake_clone)
    res = posebatch.run_batch(None, {
        "project": PROJECT, "render": "跑成片", "pose_set": "跑",
        "char_image": "char.png", "bg": "transparent", "engine": "wpc",
    })

    assert res["images"] == 2 and res["failed"] == []
    doc = poseset.load_render(PROJECT, "跑成片")
    assert doc["done"] == 2
    assert all(len(f["candidates"]) == 1 for f in doc["frames"])
    assert all(f["selectedIndex"] == 1 for f in doc["frames"]), "新候选自动选中"
    # 透传：条件图用的是姿势集里的帧文件，底色没被改写成 white
    assert [Path(m["pose"]).name for m in seen] == ["f_001.png", "f_002.png"]
    assert {m["bg"] for m in seen} == {"transparent"}


def test_run_batch_defaults_to_vnccs(monkeypatch, tmp_path):
    """引擎跟着节点走，**默认 VNCCS**。

    批量以前写死 `pose_clone`（wpc），而「角色迁移」节点默认是 VNCCS ——
    界面选 VNCCS、实际跑 wpc。实测 wpc 遇到「站立角色图 + 灰人偶」会整段照抄
    角色图的站姿（12 帧帧间差异 0.08/255，肉眼就是"和原图一模一样"），
    同一批走 VNCCS 帧间差异 28.4。这条用例盯住"别再退回 wpc"。
    """
    from workflows import generate, posebatch

    poseset.save_pose_set(PROJECT, "踢", _frames(1))
    poseset.create_render(PROJECT, "踢成片", "踢")

    calls = []

    def fake_vnccs(eng, *, pose_image, **kw):
        out = tmp_path / "v.png"
        out.write_bytes(base64.b64decode(PNG_B64))
        calls.append(kw)
        return {"file": str(out), "meta": {}}

    def boom(*a, **kw):                       # wpc 不该被调用
        raise AssertionError("默认引擎应该是 VNCCS，不该走 pose_clone")

    monkeypatch.setattr(generate, "vnccs_pose", fake_vnccs)
    monkeypatch.setattr(generate, "pose_clone", boom)

    res = posebatch.run_batch(None, {
        "project": PROJECT, "render": "踢成片", "pose_set": "踢",
        "char_image": "char.png", "bg": "transparent", "steps": 16,
    })
    assert res["engine"] == "wvnccs" and res["images"] == 1
    assert calls and calls[0]["transparent"] is True, "透明底要转到 VNCCS 的 transparent"
    assert calls[0]["steps"] >= 20, "VNCCS 的 LoRA 按 25 步训的，步数不能压到 16"
