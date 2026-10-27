"""游戏动效导出：GIF/APNG 帧数、索引几何、Godot 资源格式、确定性。"""
from __future__ import annotations

import json
import zipfile

from PIL import Image

from workflows import anim


def _frames(n=4, size=(64, 64)):
    out = []
    for i in range(n):
        im = Image.new("RGBA", size, (0, 0, 0, 0))
        im.paste((i * 40, 80, 160, 255), (8 + i * 4, 12, 40 + i * 4, 52))
        out.append(im)
    return out


def test_gif_frame_count_and_size(tmp_path):
    frames = _frames(4)
    p = anim.save_gif(frames, tmp_path / "a.gif", fps=8)
    im = Image.open(p)
    assert getattr(im, "n_frames", 1) == 4
    assert im.size == (64, 64)


def test_apng_frame_count(tmp_path):
    p = anim.save_apng(_frames(3), tmp_path / "a.png", fps=10)
    im = Image.open(p)
    assert getattr(im, "n_frames", 1) == 3


def test_gif_scale_is_nearest(tmp_path):
    p = anim.save_gif(_frames(1), tmp_path / "s.gif", fps=8, scale=2)
    assert Image.open(p).size == (128, 128)


def test_engine_index_geometry(tmp_path):
    frames = _frames(5)
    sheet = __import__("workflows.postprocess", fromlist=["pack_sheet"]).pack_sheet(
        frames, tmp_path / "sheet.png", cols=3)
    idx = anim.engine_index(frames, sheet, name="walk", fps=12, loop=False)
    assert idx["count"] == 5 and idx["grid"] == {"cols": 3, "rows": 2}
    assert idx["frames"][3] == {"i": 3, "x": 0, "y": sheet["cell_h"],
                                "w": sheet["cell_w"], "h": sheet["cell_h"]}
    assert idx["filter"] == "nearest" and idx["loop"] is False


def test_export_godot_produces_all_artifacts(tmp_path):
    r = anim.export_godot(_frames(4), tmp_path, anim_name="walk", fps=8, cols=4)
    for key in ("sheet", "tres", "tscn", "json", "gif", "apng", "zip"):
        from pathlib import Path
        assert Path(r[key]).exists(), f"{key} 没生成"
    assert zipfile.ZipFile(r["zip"]).namelist()
    idx = json.loads(open(r["json"], encoding="utf-8").read())
    assert idx["name"] == "walk" and idx["count"] == 4
    assert idx["godot"]["texture_filter"] == 1


def test_godot_tres_regions_are_sequential(tmp_path):
    r = anim.export_godot(_frames(4), tmp_path, anim_name="walk", cols=2)
    tres = open(r["tres"], encoding="utf-8").read()
    cell_w, cell_h = r["index"]["cell"]["w"], r["index"]["cell"]["h"]
    assert f"Rect2(0, 0, {cell_w}, {cell_h})" in tres
    assert f"Rect2({cell_w}, 0, {cell_w}, {cell_h})" in tres
    assert f"Rect2(0, {cell_h}, {cell_w}, {cell_h})" in tres
    assert tres.count('[sub_resource type="AtlasTexture"') == 4
    assert '"speed": 10.0' in tres or '"speed":' in tres


def test_godot_tscn_pixel_art_settings(tmp_path):
    r = anim.export_godot(_frames(3), tmp_path, anim_name="run", cols=3, fps=12)
    tscn = open(r["tscn"], encoding="utf-8").read()
    assert "texture_filter = 1" in tscn          # 最近邻，像素不糊
    assert 'autoplay = "run"' in tscn
    assert "type=\"AnimatedSprite2D\"" in tscn
    cell_h = r["index"]["cell"]["h"]
    assert f"offset = Vector2(0, {-cell_h / 2:g})" in tscn


def test_export_is_deterministic(tmp_path):
    a = anim.export_godot(_frames(3), tmp_path / "a", anim_name="walk", cols=3)
    b = anim.export_godot(_frames(3), tmp_path / "b", anim_name="walk", cols=3)
    assert open(a["tres"], encoding="utf-8").read() == open(b["tres"], encoding="utf-8").read()
    assert open(a["tscn"], encoding="utf-8").read() == open(b["tscn"], encoding="utf-8").read()


def test_export_rejects_empty():
    import pytest

    with pytest.raises(ValueError):
        anim.export_godot([], "/tmp/none", anim_name="x")
