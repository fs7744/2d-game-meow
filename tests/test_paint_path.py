"""「只勾透明背景、不勾像素化」时必须走**柔边**路径。

背景（2026-11-02 用户反馈）：非像素画风的出图也带像素感。查下来契约层有两条
把模型往像素画上拽（已在 `test_presets.py` 覆盖），后处理层还有一条：

  transparent=True 且 alpha 起雾时，`_postprocess` 一律 `_harden_alpha(240)`，
  把整条中间过渡带砍成 0/255 —— 边缘就是台阶状锯齿。像素画要这个（硬边是风格），
  插画不要（抗锯齿边缘恰恰是"不是像素画"的证据）。

所以这里钉住三件事：
  1. 插画档走 `sanitize_alpha`（只清 <16 / >240），中间过渡**原样保留**；
  2. 像素化档仍走 `_harden_alpha`（0/255），旧行为不变；
  3. 形态学开运算只在像素化档执行 —— 插画档下 1~2px 的细突起（发丝、挂饰）
     不能被削掉。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from workflows import generate as G


def _foggy_rgba(w: int = 40, h: int = 40) -> Image.Image:
    """一张「实心主体 + 雾状背景 + 中间过渡带」的 RGBA。

    alpha 分布刻意对上实测的 Qwen 雾状 alpha：主体 255、背景 ~230、
    轮廓上留一圈 200 的过渡（这正是抗锯齿边缘被砍成台阶的地方）。
    """
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 180, 100, 90
    arr[..., 3] = 230                 # 背景：雾
    arr[10:30, 10:30, 3] = 255        # 主体：实心
    arr[9, 10:30, 3] = 200            # 上边缘过渡带
    arr[30, 10:30, 3] = 200           # 下边缘过渡带
    return Image.fromarray(arr, "RGBA")


def _run(img: Image.Image, *, pixel: int, palette: int):
    info: dict = {}
    out, _edge, _subjects = G._postprocess(
        img, None, matte=False, matte_backend="auto", denoise=1,
        pixel=pixel, palette=palette, clean_iso=1, info=info, transparent=True)
    return out, info


def test_paint_path_keeps_soft_alpha_edges():
    """插画档：过渡带的 alpha 必须留着，不能被二值化。"""
    out, info = _run(_foggy_rgba(), pixel=0, palette=0)
    a = np.asarray(out.convert("RGBA").getchannel("A"))
    assert info.get("alpha_softened") is True, info
    assert info.get("alpha_hardened") is None, f"插画档不该硬切: {info}"
    # 过渡带（第 9 行）原样是 200，硬切会变 0
    assert int(a[9, 20]) == 200, f"过渡带被砍了: {a[9, 20]}"
    assert int(a[20, 20]) == 255, "主体不该被动"
    # 背景仍是 230：sanitize_alpha 只清 <16 与 >240，不碰中间
    assert int(a[35, 20]) == 230


def test_pixel_path_still_hardens_alpha():
    """像素化档：维持旧的硬切行为（像素画就是要 0/255）。"""
    out, info = _run(_foggy_rgba(), pixel=0, palette=32)
    a = np.asarray(out.convert("RGBA").getchannel("A"))
    assert info.get("alpha_hardened") is True, info
    assert info.get("alpha_softened") is None
    assert int(a[9, 20]) == 0, "像素档的过渡带应被硬切成 0"
    assert int(a[20, 20]) == 255


def test_paint_path_does_not_erode_thin_details():
    """插画档不做形态学开运算：1px 的细突起（发丝 / 挂饰）要保住。"""
    arr = np.zeros((40, 40, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 200, 60, 60
    arr[..., 3] = 230
    arr[10:30, 10:30, 3] = 255
    arr[30:38, 20, 3] = 255       # 1px 宽、8px 长的细须（像一缕发丝）
    img = Image.fromarray(arr, "RGBA")

    out, _ = _run(img, pixel=0, palette=0)
    a = np.asarray(out.convert("RGBA").getchannel("A"))
    kept = int((a[30:38, 20] == 255).sum())
    assert kept >= 7, f"插画档把细须削掉了（只剩 {kept}/8 px）"


@pytest.mark.parametrize("transparent,expected", [(True, 0), (False, 1)])
def test_matte_node_still_cuts(transparent: bool, expected: int):
    """`matte=True`（用户显式勾了抠图）的行为不受本次改动影响：
    不透明 → 真的去抠；透明 → 模型自带 alpha，不再重复抠一遍。"""
    import types

    info: dict = {}
    calls = []

    def _fake_matte(rgba, backend, matte_dir, info=None):
        calls.append(backend)
        return rgba

    import workflows.matte as M
    orig = M.matte_auto
    M.matte_auto = _fake_matte
    try:
        G._postprocess(_foggy_rgba(), types.SimpleNamespace(matte_dir="."),
                       matte=True, matte_backend="auto",
                       denoise=1, pixel=0, palette=0, clean_iso=1,
                       info=info, transparent=transparent)
    finally:
        M.matte_auto = orig
    assert len(calls) == expected, f"transparent={transparent} 时抠图次数不对: {calls}"
