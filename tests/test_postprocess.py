"""后处理：对齐、打包、量化、裁边。纯 CPU，不碰模型。"""
from __future__ import annotations

from PIL import Image, ImageDraw

from workflows import postprocess as pp


def _frame(y: int, h: int = 70, w: int = 48) -> Image.Image:
    im = Image.new("RGBA", (128, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([40, y, 40 + w, y + h], fill=(220, 60, 60, 255))
    return im


def test_anchor_is_bottom_center_of_alpha():
    f = _frame(40)
    bbox = f.getchannel("A").getbbox()          # PIL 的 right/bottom 是开区间
    x, y, h = pp.anchor_of(f)
    assert y == bbox[3]                         # 脚底 = alpha 包围盒底边
    assert h == bbox[3] - bbox[1]
    assert abs(x - (bbox[0] + bbox[2]) / 2) < 1  # 水平居中


def test_align_sequence_kills_vertical_drift():
    frames = [_frame(40 + i * 4) for i in range(8)]
    before = [pp.anchor_of(f)[1] for f in frames]
    after = [pp.anchor_of(f)[1] for f in pp.align_sequence(frames)]
    assert max(before) - min(before) == 28      # 原始漂移
    assert max(after) - min(after) == 0          # 对齐后归零


def test_align_preserves_size_and_pixels():
    a, b = _frame(40), _frame(70)
    out = pp.align_to(b, a)
    assert out.size == b.size
    # 只做整数平移：像素总量（非透明像素数）不变
    assert sum(1 for p in out.getdata() if p[3] > 0) == sum(1 for p in b.getdata() if p[3] > 0)


def test_pack_sheet_geometry(tmp_path):
    frames = [_frame(40 + i * 3) for i in range(7)]
    info = pp.pack_sheet(frames, tmp_path / "s.png", cols=3)
    assert (info["cols"], info["rows"], info["count"]) == (3, 3, 7)
    assert info["cell_w"] == 128 and info["cell_h"] == 160
    sheet = Image.open(info["sheet"])
    assert sheet.size == (384, 480)
    # sidecar 必须落盘且能读回
    import json
    from pathlib import Path

    assert json.loads(Path(info["sheet"] + ".json").read_text("utf-8"))["count"] == 7


def test_pack_sheet_bottom_aligns_cells(tmp_path):
    tall = Image.new("RGBA", (128, 160), (0, 0, 0, 0))
    ImageDraw.Draw(tall).rectangle([50, 20, 78, 159], fill=(0, 200, 0, 255))
    short = Image.new("RGBA", (128, 160), (0, 0, 0, 0))
    ImageDraw.Draw(short).rectangle([50, 100, 78, 139], fill=(0, 0, 255, 255))
    info = pp.pack_sheet([tall, short], tmp_path / "b.png", cols=2)
    sheet = Image.open(info["sheet"]).convert("RGBA")
    # 两格底边都贴单元格底部：第 159 行有像素，第 140~158 行只有第一格
    assert sheet.getpixel((60, 159))[3] > 0
    assert sheet.getpixel((60 + 128, 145))[3] == 0


def test_quantize_limits_colors():
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i in range(40):                      # 40 种颜色
        d.rectangle([i, 0, i, 63], fill=(i * 6, 255 - i * 6, i * 3, 255))
    q = pp.quantize_palette(im, colors=16)
    colors = {p[:3] for p in q.getdata() if p[3] > 0}
    assert len(colors) <= 16


def test_quantize_alpha_is_binary():
    im = Image.new("RGBA", (16, 16), (10, 20, 30, 100))   # 半透明
    q = pp.quantize_palette(im, colors=8)
    assert all(p[3] in (0, 255) for p in q.getdata())


def test_pixelate_blocks_are_uniform():
    im = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(im).ellipse([8, 8, 56, 56], fill=(0, 0, 0))
    p = pp.pixelate(im, 4)
    # 每个 4x4 块内部应当完全同色
    for by in range(0, 64, 4):
        for bx in range(0, 64, 4):
            block = {p.getpixel((bx + dx, by + dy)) for dx in range(4) for dy in range(4)}
            assert len(block) == 1


def test_trim_alpha_crops_to_bbox():
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([30, 20, 69, 79], fill=(255, 0, 0, 255))
    t = pp.trim_alpha(im, pad=2)
    assert t.size == (44, 64)                 # 40x60 + 2px 边


def test_pixelate_keeps_blocks_single_color():
    """像素化后每个 block×block 格子必须是单色，否则放大看就是脏点。

    BOX（面积平均）不会像 LANCZOS 那样在色块交界处振铃出过冲色；振铃色再被量化，
    就变成额外的脏色阶（实测同一张图 32 色 vs 30 色）。
    """
    import numpy as np

    im = Image.new("RGB", (64, 64), (250, 250, 250))
    ImageDraw.Draw(im).rectangle([16, 16, 47, 47], fill=(30, 60, 200))
    out = np.asarray(pp.pixelate(im, 8).convert("RGB"))
    cells = out.reshape(8, 8, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 64, 3)
    assert (cells == cells[:, :1, :]).all()


def test_foreground_blobs_counts_separated_subjects():
    one = _frame(40)
    assert pp.foreground_blobs(one) == 1
    two = one.copy()
    ImageDraw.Draw(two).rectangle([90, 40, 120, 110], fill=(60, 200, 60, 255))
    assert pp.foreground_blobs(two) == 2
    # 小块不算主体，避免把噪点当成第二个角色（两块分别 3360px / 2201px）
    assert pp.foreground_blobs(two, min_px=2500) == 1
    # 没抠图时全图是一个域
    assert pp.foreground_blobs(Image.new("RGB", (64, 64), (1, 2, 3))) == 1
