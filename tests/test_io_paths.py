"""输入路径：视频抽帧、精灵图切片（含空格子）、CLI 错误处理、非 8 倍数尺寸。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def _frames(frames, n=8):
    """人形帧素材（由 walk_frames 夹具保证存在）。"""
    got = list(frames)[:n]
    assert len(got) >= 4, f"帧素材不足：{len(got)}"
    return [Image.open(f).convert("RGB").resize((256, 256)) for f in got]


def _make_avi(path: Path, frames) -> None:
    import cv2

    h, w = frames[0].size[1], frames[0].size[0]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (w, h))
    assert vw.isOpened(), "OpenCV 打不开 MJPG 编码器"
    for im in frames:
        vw.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    vw.release()


def test_video_frames_extraction(tmp_path, walk_frames):
    frames = _frames(walk_frames)
    vid = tmp_path / "walk.avi"
    _make_avi(vid, frames)

    from tools.pose_from_image import frames_from_video

    got = list(frames_from_video(vid, step=2))
    assert len(got) == len(range(0, len(frames), 2))
    idx, im = got[0]
    assert idx == 0 and im.size == frames[0].size


def test_cli_on_video(tmp_path, walk_frames):
    frames = _frames(walk_frames)
    vid = tmp_path / "walk.avi"
    _make_avi(vid, frames)
    out = tmp_path / "poses"
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "pose_from_image.py"), str(vid),
         "--step", "2", "-o", str(out), "--detector", "none", "--size", "128"],
        capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=900, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr[-400:]
    assert len(list(out.glob("pose_*.png"))) == len(range(0, len(frames), 2))


def test_sheet_skips_blank_cells(walk_frames):
    from tools.pose_from_image import frames_from_sheet

    frames = _frames(walk_frames)[:2]
    cw, ch = frames[0].size
    sheet = Image.new("RGBA", (cw * 3, ch), (0, 0, 0, 0))
    sheet.paste(frames[0].convert("RGBA"), (0, 0))
    # 中间空一格的方块（半透明填充，不是全空），第三格放第二帧
    ImageDraw.Draw(sheet).rectangle([cw + 10, 10, cw * 2 - 10, ch - 10], fill=(0, 0, 0, 255))
    sheet.paste(frames[1].convert("RGBA"), (cw * 2, 0))

    got = list(frames_from_sheet(sheet, cols=3, rows=1))
    assert len(got) == 3, "空格子只应跳过 alpha 与亮度都为 0 的格子"


def test_sheet_drops_fully_empty_cell(tmp_path):
    from tools.pose_from_image import frames_from_sheet

    sheet = Image.new("RGBA", (256, 128), (0, 0, 0, 0))
    ImageDraw.Draw(sheet).rectangle([10, 10, 118, 118], fill=(200, 30, 30, 255))
    got = list(frames_from_sheet(sheet, cols=2, rows=1))
    assert len(got) == 1


def test_cli_missing_file_exits_2(tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "pose_from_image.py"),
         str(tmp_path / "nope.png"), "-o", str(tmp_path / "o")],
        capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300, cwd=str(ROOT))
    assert r.returncode == 2
    assert "找不到输入" in r.stdout


# ---------------- 非 8 倍数尺寸 ----------------

def test_inpaint_non_multiple_of_8(eng, assets):
    """250×250 不是 8 的倍数：内部要按 248 推理再还原，尺寸必须回到 250。

    **两条策略的承诺不同，选区外的断言也不同**：

    · ``strategy="whole"``（蒙版贴回）：选区外**逐点不变**（差恒为 0）。
    · ``strategy="mask"``（默认，模型自己看蒙版）：走 VAE 解码，蒙版外有
      约 19 级像素偏差（docstring 里写明了，视觉上完全一致）。
      这是这条用例以前失败的原因——断言按 whole 写，跑的却是 mask。

    所以两条都跑，各自断言各自的契约。
    """
    src = assets / "_t_odd_src.png"
    mask = assets / "_t_odd_mask.png"
    im = Image.new("RGB", (250, 250), (40, 80, 120))
    ImageDraw.Draw(im).rectangle([80, 80, 160, 160], fill=(200, 40, 40))
    im.save(src)
    m = Image.new("L", (250, 250), 0)
    ImageDraw.Draw(m).rectangle([120, 120, 200, 200], fill=255)
    m.save(mask)

    from workflows import generate as g

    before = np.asarray(Image.open(src).convert("RGB")).astype(int)
    msk = np.asarray(Image.open(mask).convert("L"))

    # ---- 贴回策略：选区外逐点不变 ----
    r = g.inpaint(eng, image=src, mask=mask, prompt="green block", steps=8, seed=3,
                  tag="odd", strategy="whole")
    out = Image.open(r["file"])
    assert out.size == (250, 250), f"尺寸被改了: {out.size}"
    after = np.asarray(out.convert("RGB")).astype(int)
    assert np.abs(after - before).max(axis=2)[msk == 0].max() == 0

    # ---- 蒙版策略：尺寸同样要还原，偏差在可接受范围内 ----
    r2 = g.inpaint(eng, image=src, mask=mask, prompt="green block", steps=8, seed=3,
                   tag="odd", strategy="mask")
    out2 = Image.open(r2["file"])
    assert out2.size == (250, 250), f"尺寸被改了: {out2.size}"
    after2 = np.asarray(out2.convert("RGB")).astype(int)
    diff = int(np.abs(after2 - before).max(axis=2)[msk == 0].max())
    # VAE 解码的固有误差；实测 19。留出余量但不放水（>32 说明溢出到选区外了）
    assert diff <= 32, f"蒙版外偏差过大：{diff}"
    assert r2["meta"]["outside_max_diff"] == diff


def test_pack_sheet_with_more_cols_than_frames(tmp_path):
    from workflows import postprocess as pp

    frames = [Image.new("RGBA", (32, 32), (i * 30, 0, 0, 255)) for i in range(3)]
    info = pp.pack_sheet(frames, tmp_path / "s.png", cols=8)
    assert (info["cols"], info["rows"]) == (3, 1), "cols 应被夹到帧数"


def test_gif_input(tmp_path, walk_frames):
    """GIF 也走同一个 VideoCapture 分支（cv2 默认后端支持）。"""
    frames = _frames(walk_frames)
    gif = tmp_path / "walk.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=120, loop=0)

    from tools.pose_from_image import frames_from_video

    got = list(frames_from_video(gif, step=1))
    if not got:
        pytest.skip("当前 OpenCV 后端读不了 GIF（不是缺陷，改用 AVI/MP4）")
    assert len(got) >= 2


def test_check_env_reports_json():
    import json

    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_env.py")],
        capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300, cwd=str(ROOT))
    data = json.loads(r.stdout)
    for key in ("python", "torch", "libs", "weights", "missing"):
        assert key in data
    assert data["torch"]["available"] is True
    assert isinstance(data["missing"], list)


def test_apply_lora_missing_file_raises(eng, tmp_path):
    pipe = eng.txt2img_pipe()
    with pytest.raises(Exception):
        eng.apply_lora(pipe, tmp_path / "nope.safetensors")


# ---------------- 测试不能写脏用户的目录 ----------------
#
# 背景：这块踩过两次坑，都不是"报错"，而是**静默写进用户的真实目录**：
#   1. `templates/` 漏了隔离 → 用户的模板列表里多出 5 个 `API_34266.mproj`
#      这类随机后缀的垃圾（实盘检查过，全是 `test_projects.py` 的残留）；
#   2. 拿真实 `assets/` 跑 headless 浏览器做界面验证 → 前端首屏"没有项目就新建
#      一个"，于是每连一次就多一个空项目，还会改掉 `workspace.json` 的 current。
# 所以这里把「测试/验证的写入范围」钉成不变量。

def test_test_dirs_are_outside_user_dirs():
    """pytest 期间的资产根与模板根都必须在 `tests/_out/` 下。

    这条和 `conftest.py` 里的 autouse 看守夹具是双保险：夹具在 session 开场拦住
    "全都跑完才发现写脏"，这条在静态上把契约写清楚（改坏了会看到具体断言）。
    """
    from workflows import paths

    out = (ROOT / "tests" / "_out").resolve()
    for name, p in (("ASSETS", paths.ASSETS), ("TEMPLATES_DIR", paths.TEMPLATES_DIR)):
        p = Path(p).resolve()
        assert out in p.parents, f"paths.{name} = {p}，不在 {out} 之下"
    # 顺便钉住"没指向用户目录"这个更直白的表述
    assert Path(paths.ASSETS).resolve() != (ROOT / "assets").resolve()
    assert Path(paths.TEMPLATES_DIR).resolve() != (ROOT / "templates").resolve()


def test_sandbox_server_points_at_tests_out():
    """界面验证用的沙盒实例必须指向 `tests/_out/ui/`，不是真 assets。

    只验证 `sandbox_dirs()` 算出来的两个路径——**不启动服务、也不改本进程环境变量**
    （改了会影响后续用例：`paths` 已经在 import 时求值，但子进程会读到）。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "serve_sandbox", ROOT / "tools" / "serve_sandbox.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dirs = mod.sandbox_dirs()
    assert set(dirs) == {"MEOW_ASSETS_DIR", "MEOW_TEMPLATES_DIR"}
    ui = (ROOT / "tests" / "_out" / "ui").resolve()
    for key, val in dirs.items():
        assert ui in Path(val).resolve().parents, f"{key} = {val}，不在 {ui} 之下"
