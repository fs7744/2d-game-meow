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
    """250×250 不是 8 的倍数：内部要按 248 推理再还原，选区外必须逐点不变。"""
    src = assets / "_t_odd_src.png"
    mask = assets / "_t_odd_mask.png"
    im = Image.new("RGB", (250, 250), (40, 80, 120))
    ImageDraw.Draw(im).rectangle([80, 80, 160, 160], fill=(200, 40, 40))
    im.save(src)
    m = Image.new("L", (250, 250), 0)
    ImageDraw.Draw(m).rectangle([120, 120, 200, 200], fill=255)
    m.save(mask)

    from workflows import generate as g

    r = g.inpaint(eng, image=src, mask=mask, prompt="green block", steps=8, seed=3, tag="odd")
    out = Image.open(r["file"])
    assert out.size == (250, 250), f"尺寸被改了: {out.size}"

    before = np.asarray(Image.open(src).convert("RGB")).astype(int)
    after = np.asarray(out.convert("RGB")).astype(int)
    msk = np.asarray(Image.open(mask).convert("L"))
    assert np.abs(after - before).max(axis=2)[msk == 0].max() == 0


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


def test_apply_lora_none_is_idempotent(eng):
    pipe = eng.txt2img_pipe()
    eng.apply_lora(pipe, None)
    eng.apply_lora(pipe, None)
    assert eng.stats()["lora"] is None


def test_apply_lora_missing_file_raises(eng, tmp_path):
    pipe = eng.txt2img_pipe()
    with pytest.raises(Exception):
        eng.apply_lora(pipe, tmp_path / "nope.safetensors")
