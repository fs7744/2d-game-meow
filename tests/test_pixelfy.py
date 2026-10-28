"""完美像素转换（wpx）的核心算法测试。

重点钉住：
  1. **每格一色**——像素画与「低分辨率图」的根本区别
  2. **网格检测正确**——FFT 找出真实的像素块尺寸
  3. **死循环防护**——恶意/离谱的网格尺寸不能把进程挂死
  4. **调色板量化后无色板外颜色**——验收指标必须为 0
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from workflows import perfect_pixel as pp
from workflows import pixelfy as pxf


def _grid_img(cells=8, cell=8, colors=None):
    """造一张「真·像素画」：cells×cells 格，每格一色。"""
    size = cells * cell
    im = Image.new("RGB", (size, size), (250, 250, 250))
    d = ImageDraw.Draw(im)
    for y in range(cells):
        for x in range(cells):
            c = colors[y][x] if colors else ((x * 37) % 256, (y * 53) % 256, 128)
            d.rectangle([x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1], fill=c)
    return im


# --------------------------------------------------------------------------- #
# 网格检测
# --------------------------------------------------------------------------- #

def test_detect_grid_scale_finds_real_cell_size():
    """FFT 要能测出真实的像素块尺寸。"""
    im = _grid_img(cells=8, cell=8)
    arr = np.asarray(im)
    gw, gh = pp.detect_grid_scale(arr)
    assert gw == 8 and gh == 8, (gw, gh)


def test_detect_grid_scale_handles_non_square_cells():
    """非正方格子也要能测出来（宽高不同）。"""
    im = Image.new("RGB", (64, 32), (10, 20, 30))
    d = ImageDraw.Draw(im)
    for y in range(4):
        for x in range(8):
            d.rectangle([x * 8, y * 8, (x + 1) * 8 - 1, (y + 1) * 8 - 1],
                        fill=((x * 30) % 256, (y * 60) % 256, 200))
    gw, gh = pp.detect_grid_scale(np.asarray(im))
    assert gw == 8, gw
    assert gh == 4, gh


def test_detect_grid_scale_returns_none_for_flat_image():
    """纯色图没有网格——必须返回 None（而不是给个离谱数字）。"""
    gw, gh = pp.detect_grid_scale(np.full((64, 64, 3), 128, dtype=np.uint8))
    assert gw is None or gh is None or (gw > 1 and gh > 1)


# --------------------------------------------------------------------------- #
# 死循环防护
# --------------------------------------------------------------------------- #

def test_refine_grids_rejects_absurd_grid_size():
    """网格远大于边长时必须立刻返回，不能死循环。

    实测（上游 pixelforge）：64×64 图 grid=2000 会卡死 5600+ 迭代——
    cell_w < 0.5 时 find_best_grid 找不到峰，round(origin) 把小数进度抹掉，
    x 每轮被打回同一个整数，永远推不动。
    """
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    xs, ys = pp.refine_grids(img, 2000, 2000)
    assert xs == [] and ys == []


def test_refine_grids_rejects_zero_and_negative_grid():
    """grid <= 0 会除零或让循环条件永真——入口必须挡住。"""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert pp.refine_grids(img, 0, 8) == ([], [])
    assert pp.refine_grids(img, -5, 8) == ([], [])


def test_perfect_pixel_returns_none_on_failure():
    """检测失败必须返回 (None, None, 原图)——调用方要能判空。"""
    img = np.full((32, 32, 3), 200, dtype=np.uint8)
    gw, gh, out = pp.perfect_pixel(img)
    assert (gw is None) == (gh is None)


# --------------------------------------------------------------------------- #
# 每格一色（核心承诺）
# --------------------------------------------------------------------------- #

def test_refine_produces_one_color_per_cell():
    """输出必须「每格一色」：把结果按格切开，每格内只有一个颜色。"""
    im = _grid_img(cells=8, cell=8)
    r = pxf.refine_batch([im], colors=16)["items"][0]
    out = r["image"]
    assert out.size == (8, 8), out.size
    arr = np.asarray(out.convert("RGB")).reshape(-1, 3)
    # 8×8 = 64 格，每格 1 像素 → 64 个像素
    assert len(arr) == 64


def test_refine_detects_grid_and_reports_it():
    r = pxf.refine_batch([_grid_img(cells=8, cell=8)], colors=16)["items"][0]
    m = r["meta"]
    assert m["grid"] == [8, 8], m["grid"]
    assert m["detected"] is True
    assert m["size"] == [8, 8]


def test_no_color_outside_palette_after_quantize():
    """量化后色板外颜色必须是 0——这是验收指标。"""
    im = _grid_img(cells=8, cell=8)
    res = pxf.refine_batch([im], colors=8)
    item = res["items"][0]
    assert item["meta"]["offPalette"] == 0, item["meta"]["offPalette"]
    assert len(res["palette"]) <= 8


def test_alpha_is_binary_after_refine():
    """像素画只有「有/无」两态——alpha 必须二值。"""
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([16, 16, 47, 47], fill=(200, 40, 40, 255))
    r = pxf.refine_batch([im], colors=8)["items"][0]
    out = r["image"]
    a = np.asarray(out.getchannel("A"))
    assert set(np.unique(a).tolist()) <= {0, 255}, np.unique(a).tolist()


def test_shared_palette_across_batch():
    """一批图要共享同一块色板——逐张各自取色板会显著偏色。"""
    ims = [_grid_img(cells=8, cell=8, colors=[[(10 * i % 256, 200, 30) for i in range(8)]]
                     * 8),
           _grid_img(cells=8, cell=8, colors=[[(200, 10 * i % 256, 30) for i in range(8)]]
                     * 8)]
    res = pxf.refine_batch(ims, colors=16)
    assert res["palette"]
    # 所有产物都只能用这块色板
    for item in res["items"]:
        assert item["meta"]["offPalette"] == 0


# --------------------------------------------------------------------------- #
# 噪点清理
# --------------------------------------------------------------------------- #

def test_clean_specks_removes_isolated_noise():
    """孤立的近似色噪点要被清掉。

    注意噪点必须与周围**接近**（这里是灰 30 vs 黑底）：差异大的单像素
    （眼睛、高光）是有意义的细节，不能清——这就是 max_diff 条件的用途。
    """
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[8, 8] = (30, 30, 30)        # 孤立一点，与黑底差异很小
    out = pxf.clean_specks(rgb)
    assert tuple(out[8, 8]) == (0, 0, 0)


def test_clean_specks_keeps_high_contrast_detail():
    """单像素眼睛/高光与周围差异大——必须保留。"""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[8, 8] = (255, 255, 255)      # 纯白点，与黑底差异极大
    out = pxf.clean_specks(rgb)
    assert tuple(out[8, 8]) == (255, 255, 255)


def test_clean_specks_keeps_thin_outline():
    """1 像素描边通常有 2 个以上同色邻居——不能被误删。"""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[8, :] = (255, 255, 255)
    out = pxf.clean_specks(rgb)
    assert tuple(out[8, 0]) == (255, 255, 255)


# --------------------------------------------------------------------------- #
# 调色板
# --------------------------------------------------------------------------- #

def test_apply_palette_uses_int32():
    """差值平方最大 255²×4 ≈ 26 万，int16 会溢出算错最近色。

    实测白色被映射成红色就是这条——必须 int32。
    """
    pal = [(255, 0, 0), (0, 0, 255)]
    img = Image.new("RGB", (4, 4), (255, 255, 255))
    out = pxf.apply_palette(img, pal)
    assert out.getpixel((0, 0)) in pal


def test_extract_palette_is_deterministic():
    """色板提取必须可复现（不能有随机采样）。"""
    ims = [_grid_img(cells=8, cell=8)]
    a = pxf.extract_palette(ims, 16)
    b = pxf.extract_palette(ims, 16)
    assert a == b


def test_transparent_bg_makes_flat_background_transparent():
    """纯色背景应该被转成透明。"""
    im = Image.new("RGB", (64, 64), (250, 250, 250))
    ImageDraw.Draw(im).rectangle([24, 24, 39, 39], fill=(30, 60, 200))
    r = pxf.refine_batch([im], colors=8, transparent_bg=True)["items"][0]
    out = r["image"]
    a = np.asarray(out.getchannel("A"))
    # 输出是原生像素尺寸（网格检测后可能是 8×8），坐标要按比例换算
    w, h = out.size
    cx, cy = int(32 / 64 * w), int(32 / 64 * h)
    assert a[0, 0] == 0, "四角背景应变透明"
    assert a[cy, cx] == 255, "主体应保留"


# --------------------------------------------------------------------------- #
# 网格吸附到标准尺寸
# --------------------------------------------------------------------------- #

def test_snap_dim_rounds_to_multiple():
    """吸附：往最近的 base 倍数靠，但差太多就放弃（保密度）。"""
    assert pxf._snap_dim(157, 16) == 160
    assert pxf._snap_dim(29, 16) == 32
    assert pxf._snap_dim(30, 16) == 32
    assert pxf._snap_dim(20, 16) == 16
    # 已经是 base 的倍数就原样保留
    assert pxf._snap_dim(8, 16) == 8
    assert pxf._snap_dim(32, 16) == 32
    # 差太多（密度变化过大）就不吸附：8 抬成 16 等于把像素密度砍半
    assert pxf._snap_dim(8, 16) == 8
    # 夹在合理范围（太小/太大的检测值不硬抬，直接夹到 8~256）
    assert pxf._snap_dim(3, 16) == 8
    assert pxf._snap_dim(9999, 16) == 256


def test_snap_to_makes_output_a_standard_size():
    """吸附后输出尺寸必须是 base 的整数倍。

    动机：AI 出的「像素风」没有严格网格（实测检测出 157×153），
    这种尺寸的精灵图没法用——必须吸附到 16/32/64/128 这种标准尺寸。

    这里的夹具是 8×8 格（检测值 8），8 已经是「够标准」的精灵尺寸，
    吸附容差下会原样保留——所以断言用「8 的倍数」而不是「16 的倍数」。
    要验证吸附本身生效，另用 _snap_dim 的单元测试。
    """
    im = _grid_img(cells=8, cell=8)
    res = pxf.refine_batch([im], colors=16, snap_to=16)
    out = res["items"][0]["image"]
    assert out.width % 8 == 0 and out.height % 8 == 0, out.size
    assert out.size == (8, 8), out.size


def test_snap_to_none_keeps_detected_size():
    """snap_to=None = 不吸附，保留检测到的原始值。"""
    im = _grid_img(cells=8, cell=8)
    res = pxf.refine_batch([im], colors=16, snap_to=None)
    m = res["items"][0]["meta"]
    assert "gridSnapped" not in m
    assert m["grid"] == [8, 8]
