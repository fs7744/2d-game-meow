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


# ---------------- 白底预乘边缘：保留过渡带 ≠ 保留白色 ----------------
#
# 用户的第二份反馈：「你看这边缘这头发，你肯定抠图了」。上一版只把 alpha 硬切
# 换成了「保留过渡带」，但**过渡带的 RGB 是白底合成出来的**——实测一张 Qwen
# 透明图：过渡带（16≤α≤240）RGB 亮度 203，实心区 128。保留它就是保留白色，
# 深色背景上就是一圈白雾边。
#
# 正解是 unpremultiply：C_fg = (C_obs - 255*(1-α)) / α。实测把 203 拉回 125.3，
# 与实心区 128.2 吻合。

def _white_matte(w=40, h=40):
    """前半张是「白底预乘」的半透明边缘，后半张是实心主体。"""
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 180, 100, 90
    arr[15:35, :, 3] = 128                    # 大片半透明（白底混合）
    # 白底预乘：α=0.5 时 RGB = fg*0.5 + 255*0.5 ≈ (218, 178, 173)
    arr[15:35, :, 0], arr[15:35, :, 1], arr[15:35, :, 2] = 218, 178, 173
    arr[35:40, 10:30, 3] = 255                # 实心主体
    return Image.fromarray(arr, "RGBA")


def test_defringe_pulls_white_back_to_foreground():
    """白污染必须被还原：过渡带亮度要落回实心区附近。"""
    out = G._defringe_white_matte(_white_matte())
    a = np.asarray(out).astype(np.float32)
    al, rgb = a[..., 3], a[..., :3]
    mid = (al >= 16) & (al <= 240)
    solid = al >= 241
    assert mid.any() and solid.any()
    delta = abs(float(rgb[mid].mean()) - float(rgb[solid].mean()))
    assert delta < 12, f'过渡带亮度 {rgb[mid].mean():.1f} 没拉回实心区 {rgb[solid].mean():.1f}'


def test_defringe_leaves_clean_edges_alone():
    """过渡带本来就正确（无白污染）时不许动 —— 动了会让边缘发暗。"""
    arr = np.zeros((40, 40, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 60, 40, 40
    arr[15:35, :, 3] = 128
    arr[35:40, 10:30, 3] = 255
    src = Image.fromarray(arr, "RGBA")
    out = G._defringe_white_matte(src)
    assert np.array_equal(np.asarray(src), np.asarray(out)), '没有白污染却改了像素'


def test_soft_path_also_defringes(monkeypatch):
    """端到端：插画档走软边时，白雾边必须一起去掉。"""
    out, info = _run(_white_matte(), pixel=0, palette=0)
    assert info.get('alpha_softened') is True, info
    a = np.asarray(out.convert('RGBA')).astype(np.float32)
    al, rgb = a[..., 3], a[..., :3]
    mid = (al >= 16) & (al <= 240)
    solid = al >= 241
    assert mid.any()
    delta = abs(float(rgb[mid].mean()) - float(rgb[solid].mean()))
    assert delta < 12, f'软边档没去白边：过渡带 {rgb[mid].mean():.1f} vs 实心 {rgb[solid].mean():.1f}'
    # alpha 本身仍是渐变的（不能顺手又硬切了）：128 这个中间值必须还在
    assert 128 in set(np.unique(al).tolist()), '过渡带被二值化了'


# ---------------- 没勾抠图就**绝不**调分割模型 ----------------
#
# 用户第三份反馈：「我没选抠图，你为啥要主动扣？」。原来的 `_postprocess` 在
# 「整张 alpha 都是雾」时会自己调 `matte_auto`（u2net），后果是头发边缘被啃、
# 轮廓外一圈 mask 肥边。跑不跑分割模型是用户的选择，不该替他决定。

def _all_foggy(w=40, h=40):
    """整张都是雾：没有任何 >=240 的实心像素，alpha_trust 必然不通过。"""
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 200, 150, 140
    arr[..., 3] = 128
    return Image.fromarray(arr, "RGBA")


def test_no_matte_never_runs_segmentation():
    """整张雾 + 没勾抠图 → 一个像素都不许交给分割模型。"""
    import types
    import workflows.matte as M

    calls = []

    def _forbidden(*a, **k):
        calls.append(1)
        raise AssertionError('没勾抠图却调了 matte_auto（分割模型）')

    orig = M.matte_auto
    M.matte_auto = _forbidden
    info: dict = {}
    try:
        G._postprocess(_all_foggy(), types.SimpleNamespace(matte_dir='.'),
                       matte=False, matte_backend='auto', denoise=1,
                       pixel=0, palette=0, clean_iso=1, info=info,
                       transparent=True)
    finally:
        M.matte_auto = orig
    assert calls == [], '没勾抠图却跑了分割模型'
    assert info.get('alpha_missing') is True, f'没标记"没出透明底": {info}'
    assert info.get('alpha_recut') is None


def test_explicit_matte_still_runs_segmentation_on_fog():
    """显式勾了抠图 → 该抠就抠（这是用户的选择）。"""
    import types
    import workflows.matte as M

    calls = []

    def _fake(rgba, backend, matte_dir, **k):
        calls.append(backend)
        return rgba

    orig = M.matte_auto
    M.matte_auto = _fake
    try:
        G._postprocess(_all_foggy(), types.SimpleNamespace(matte_dir='.'),
                       matte=True, matte_backend='auto', denoise=1,
                       pixel=0, palette=0, clean_iso=1, info={},
                       transparent=False)
    finally:
        M.matte_auto = orig
    assert calls == ['auto'], '勾了抠图却没跑'


def test_clean_alpha_still_defringes_rgb():
    """**alpha 可信 ≠ RGB 干净**。

    `alpha_trust` 只看 alpha 的数值分布（有没有真透明区、雾多不多），看不出
    过渡带的颜色是"白底预乘"出来的。实测一张 `alpha_reused=True` 的图：
    过渡带亮度 237.8、实心区 206.4，87% 的过渡像素接近纯白 —— 深色背景上
    就是一整圈白边。所以复用 alpha 时 **RGB 仍要去白底**。
    """
    # 造一张「alpha 结构干净（有真透明区、雾很少）但 RGB 是白底预乘」的图
    # 尺寸要够大：`_defringe_white_matte` 在过渡带样本 < 64 时不动手
    # （防"整张图只有零星几个抗锯齿点"被误判成白污染）。
    arr = np.zeros((100, 100, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 120, 90, 80
    arr[30:70, 30:70, 3] = 255          # 实心主体（40×40）
    # 上下各 1px 过渡带（白底预乘，α≈0.5），共 80 像素 ≥ 64
    for row in (29, 70):
        arr[row, 30:70, 3] = 128
        arr[row, 30:70, 0:3] = (188, 172, 167)
    img = Image.fromarray(arr, "RGBA")

    ok, obs = __import__('workflows.matte', fromlist=['x']).alpha_trust(img)
    assert ok, f'这张图本该 alpha 可信：{obs}'

    info: dict = {}
    out, _e, _s = G._postprocess(img, None, matte=False, matte_backend='auto',
                                 denoise=1, pixel=0, palette=0, clean_iso=1,
                                 info=info, transparent=True)
    assert info.get('alpha_reused') is True, info
    a = np.asarray(out.convert('RGBA')).astype(np.float32)
    al, rgb = a[..., 3], a[..., :3]
    mid = (al >= 16) & (al <= 240)
    solid = al >= 241
    assert mid.any() and solid.any()
    delta = abs(float(rgb[mid].mean()) - float(rgb[solid].mean()))
    assert delta < 25, f'alpha 复用了但没去白底：过渡带 {rgb[mid].mean():.1f} vs 实心 {rgb[solid].mean():.1f}'


def test_defringe_can_be_disabled_for_raw_output():
    """`raw_output=True` = **模型输出原样**（用户要"原生"时用这个开关）。

    只允许 `sanitize_alpha` 那点清理（<16 清零、>240 拉满），过渡带的 RGB
    一个数都不许改 —— 否则"原生"就成了空话。
    """
    src = _white_matte()
    info: dict = {}
    out, _e, _s = G._postprocess(src, None, matte=False, matte_backend='auto',
                                 denoise=1, pixel=0, palette=0, clean_iso=1,
                                 info=info, transparent=True, raw_output=True)
    a = np.asarray(out.convert('RGBA')).astype(np.float32)
    b = np.asarray(src.convert('RGBA')).astype(np.float32)
    mid = (a[..., 3] >= 16) & (a[..., 3] <= 240)
    assert mid.any(), '过渡带没了，这个测试就没意义'
    delta = abs(float(a[..., :3][mid].mean()) - float(b[..., :3][mid].mean()))
    assert delta < 1.0, f'关掉去白边后 RGB 仍被改了：{delta:.2f}'


def test_defringe_does_not_touch_alpha():
    """去白边**只改 RGB**：alpha 分布前后必须逐值一致。"""
    src = _white_matte()
    out, _e, _s = G._postprocess(src, None, matte=False, matte_backend='auto',
                                 denoise=1, pixel=0, palette=0, clean_iso=1,
                                 info={}, transparent=True)
    a1 = np.asarray(src.convert('RGBA'))[..., 3]
    a2 = np.asarray(out.convert('RGBA'))[..., 3]
    # sanitize_alpha 会把 >240 拉满 / <16 归零，先把这一步算进预期
    expect = np.where(a1 > 240, 255, np.where(a1 < 16, 0, a1)).astype(np.uint8)
    assert np.array_equal(expect, a2), 'alpha 被去白边改动了'


def test_defringe_can_be_disabled_for_raw_output():
    """`raw_output=True` = **模型输出原样**（用户要"原生"时用这个开关）。

    只允许 `sanitize_alpha` 那点清理（<16 清零、>240 拉满），过渡带的 RGB
    一个数都不许改 —— 否则"原生"就成了空话。
    """
    src = _white_matte()
    info: dict = {}
    out, _e, _s = G._postprocess(src, None, matte=False, matte_backend='auto',
                                 denoise=1, pixel=0, palette=0, clean_iso=1,
                                 info=info, transparent=True, raw_output=True)
    a = np.asarray(out.convert('RGBA')).astype(np.float32)
    b = np.asarray(src.convert('RGBA')).astype(np.float32)
    mid = (a[..., 3] >= 16) & (a[..., 3] <= 240)
    assert mid.any(), '过渡带没了，这个测试就没意义'
    delta = abs(float(a[..., :3][mid].mean()) - float(b[..., :3][mid].mean()))
    assert delta < 1.0, f'关掉去白边后 RGB 仍被改了：{delta:.2f}'


def test_defringe_does_not_touch_alpha():
    """去白边**只改 RGB**：alpha 分布前后必须逐值一致。"""
    src = _white_matte()
    out, _e, _s = G._postprocess(src, None, matte=False, matte_backend='auto',
                                 denoise=1, pixel=0, palette=0, clean_iso=1,
                                 info={}, transparent=True)
    a1 = np.asarray(src.convert('RGBA'))[..., 3]
    a2 = np.asarray(out.convert('RGBA'))[..., 3]
    # sanitize_alpha 会把 >240 拉满 / <16 归零，先把这一步算进预期
    expect = np.where(a1 > 240, 255, np.where(a1 < 16, 0, a1)).astype(np.uint8)
    assert np.array_equal(expect, a2), 'alpha 被去白边改动了'
