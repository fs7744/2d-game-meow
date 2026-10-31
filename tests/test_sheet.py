"""精灵图网格检测与切分（纯确定性，不跑模型）。

这组用例锁住踩过的坑：
  - 「找完全空行空列」会把 4×1 判成 10×3（帧贴边时没有空行）
  - 「单轴评分」会把 4×1 判成 6×1（剑是独立连通域，被切到别的格子）
  - 「投影谷值」会把纵向 1×4 判成 2×1（横向怎么切都不跨界）
"""
from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from workflows import sheet as sheet_mod


def _knight(i: int, size: int = 64) -> Image.Image:
    """一帧像素骑士（size×size，透明底）。i 决定腿的相位。"""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    u = size // 64                      # 缩放系数
    d.rectangle([26 * u, 48 * u, 30 * u, 58 * u], fill=(40, 40, 60, 255))
    d.rectangle([33 * u, 48 * u, 37 * u, 58 * u], fill=(40, 40, 60, 255))
    d.rectangle([25 * u, 30 * u, 38 * u, 48 * u], fill=(120, 30, 40, 255))
    d.rectangle([27 * u, 18 * u, 36 * u, 30 * u], fill=(235, 200, 160, 255))
    d.rectangle([26 * u, 16 * u, 37 * u, 21 * u], fill=(160, 165, 175, 255))
    d.rectangle([21 * u, 31 * u, 25 * u, 42 * u], fill=(120, 30, 40, 255))
    d.rectangle([38 * u, 31 * u, 42 * u, 42 * u], fill=(120, 30, 40, 255))
    # 剑：独立的连通域（这是「单轴评分」判据被击穿的元凶）
    d.rectangle([44 * u, 20 * u, 46 * u, 40 * u], fill=(200, 200, 210, 255))
    return im


def _sheet(cols: int, rows: int, cell: int = 64, gap: int = 0) -> Image.Image:
    im = Image.new("RGBA", (cols * cell + (cols - 1) * gap,
                            rows * cell + (rows - 1) * gap), (0, 0, 0, 0))
    n = 0
    for r in range(rows):
        for c in range(cols):
            im.paste(_knight(n, cell), (c * (cell + gap), r * (cell + gap)))
            n += 1
    return im


@pytest.mark.parametrize("cols,rows", [(4, 1), (2, 2), (3, 2), (1, 4), (8, 1)])
def test_detect_grid_finds_uniform_sheets(cols, rows):
    """等宽等高的精灵图应当被正确识别。"""
    got = sheet_mod.detect_grid(_sheet(cols, rows))
    assert got == (cols, rows), got


def test_detect_grid_with_gaps():
    """帧之间留透明缝是游戏素材的常态，也要能识别。"""
    got = sheet_mod.detect_grid(_sheet(4, 1, gap=3))
    assert got == (4, 1), got


def test_detect_grid_single_frame_returns_none():
    """只有一帧时不该瞎猜——返回 None 让调用方要求用户手填。"""
    assert sheet_mod.detect_grid(_knight(0)) is None


def test_detect_grid_white_background():
    """不透明底（截图/白底素材）也要能用四角背景色近似。"""
    im = _sheet(4, 1).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.paste(im, (0, 0), im)
    assert sheet_mod.detect_grid(bg) == (4, 1)


def test_detect_grid_does_not_confuse_columns_with_rows():
    """纵向 1×4 不能被误判成 2×1（横向切分永远「不跨界」，单看一轴分不出来）。"""
    got = sheet_mod.detect_grid(_sheet(1, 4))
    assert got == (1, 4), got


def test_split_sheet_cuts_cells():
    """切分结果：数量、顺序、尺寸都要对。"""
    cells = sheet_mod.split_sheet(_sheet(4, 1), 4, 1)
    assert len(cells) == 4
    assert all(c[2].size == (64, 64) for c in cells)
    assert [c[0] for c in cells] == [0, 1, 2, 3]      # 列号递增
    assert all(c[1] == 0 for c in cells)              # 都在第一行


def test_split_sheet_keeps_position_for_2x2():
    """2×2 切分要保留 (列, 行) 位置，否则重新打包会乱序。"""
    cells = sheet_mod.split_sheet(_sheet(2, 2), 2, 2)
    assert [(c[0], c[1]) for c in cells] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_split_sheet_skips_empty_cells():
    """空格子（透明的）要被跳过，不能产出全透明帧。"""
    im = Image.new("RGBA", (128, 64), (0, 0, 0, 0))
    im.paste(_knight(0), (0, 0))
    cells = sheet_mod.split_sheet(im, 2, 1)
    assert len(cells) == 1


def test_split_sheet_trim_crops_to_content():
    """trim=True 时裁到内容包围盒。"""
    im = Image.new("RGBA", (128, 64), (0, 0, 0, 0))
    im.paste(_knight(0), (0, 0))
    cells = sheet_mod.split_sheet(im, 2, 1, trim=True)
    assert len(cells) == 1
    assert cells[0][2].width < 64, cells[0][2].size


def test_split_sheet_rejects_grid_beyond_image():
    """格子小于 1px 时必须报错，而不是产出 0 像素的碎片。"""
    with pytest.raises(ValueError, match="超出图片尺寸"):
        sheet_mod.split_sheet(_sheet(1, 1), 128, 128)   # 64px 图切 128 格 → 每格 0px


def test_foreground_mask_uses_alpha():
    """透明底素材：alpha 通道即前景。"""
    im = _knight(0)
    m = sheet_mod.foreground_mask(im)
    assert m.dtype == bool and m.any()
    assert not m[0, 0], "左上角应当是透明背景"


def test_foreground_mask_empty_image():
    """全透明图：前景为空（不抛异常，交给调用方判断）。"""
    m = sheet_mod.foreground_mask(Image.new("RGBA", (32, 32), (0, 0, 0, 0)))
    assert not m.any()
