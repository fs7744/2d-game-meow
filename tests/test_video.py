"""W5 视频驱动：帧格式兼容、落地、模型路径校验、真实生成（默认跳过）。"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from workflows import video


def test_as_image_handles_numpy_and_pil():
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    a = video.as_image(arr)
    assert isinstance(a, Image.Image) and a.mode == "RGB" and a.size == (32, 32)

    b = video.as_image(np.random.rand(32, 32, 3))         # float 0~1
    assert b.mode == "RGB" and np.asarray(b).dtype == np.uint8

    c = video.as_image(Image.new("RGBA", (8, 8), (1, 2, 3, 255)))
    assert c.mode == "RGB"


def test_save_mp4_and_frames(tmp_path):
    frames = [Image.new("RGB", (64, 48), (i * 10, 60, 120)) for i in range(5)]
    mp4 = video.save_mp4(frames, tmp_path / "c.mp4", fps=8)
    assert mp4.exists() and mp4.stat().st_size > 0

    paths = video.save_frames(frames, tmp_path / "frames")
    assert len(paths) == 5 and all(p.exists() for p in paths)
    assert Image.open(paths[0]).size == (64, 48)


def test_save_mp4_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        video.save_mp4([], tmp_path / "x.mp4")


def test_wan_model_dir_reports_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("WAN_MODEL_DIR", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError) as exc:
        video.wan_model_dir()
    assert "wan2.2-ti2v-5b" in str(exc.value)


def test_wan_model_dir_resolves_default(monkeypatch, tmp_path):
    fake = tmp_path / "wan"
    fake.mkdir()
    (fake / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WAN_MODEL_DIR", str(fake))
    assert video.wan_model_dir() == fake


@pytest.mark.slow
def test_wan_t2v_smoke(tmp_path):
    """真跑一次 Wan（约 30s + 25s 加载）。设置 RUN_WAN=1 才执行。"""
    if os.environ.get("RUN_WAN") != "1":
        pytest.skip("设 RUN_WAN=1 才跑真实视频生成")
    eng = video.VideoEngine()
    frames, meta = eng.generate("a pixel art cat, flat background", num_frames=9,
                                steps=6, height=256, width=256)
    assert len(frames) == 9
    assert meta["frames"] == 9 and meta["peak_vram_gb"] < 24
    assert all(isinstance(f, Image.Image) for f in frames)
    eng.free()
