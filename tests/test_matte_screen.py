"""纯色幕抠像（移植自 VNCCS `nodes/chroma_screen_matte.py`）。

**它零模型依赖**（只 import torch），所以这组测试能真跑、不需要 GPU 或权重：
合成一张"纯色幕 + 中间一块灰人偶"的图，验证幕布被抠干净、主体保住、颜色不偏。
算法是官方调过的，我们一行没改，测试的目的是**钉住移植没把它搬坏**。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from workflows import characters as ch
from workflows import matte_screen as ms
from workflows import paths

torch = pytest.importorskip("torch")


def _screen_image(color=(0, 255, 0), fg=(115, 115, 115), size=256) -> Image.Image:
    """纯色幕布 + 中间一块前景（模拟出图后的立绘）。"""
    im = Image.new("RGB", (size, size), color)
    for y in range(size // 3, size * 2 // 3):
        for x in range(size // 3, size * 2 // 3):
            im.putpixel((x, y), fg)
    return im


def test_matte_removes_screen_and_keeps_subject(tmp_path):
    src = tmp_path / "green.png"
    _screen_image().save(src)
    out, err = ms.matte_image_file(str(src))
    assert err is None and out and os.path.isfile(out)

    got = Image.open(out)
    assert got.mode == "RGBA", "抠完必须是 RGBA（带 alpha 的精灵才有意义）"
    # 幕布（角落）应该全透
    assert got.getpixel((3, 3))[3] < 8, got.getpixel((3, 3))
    # 主体（中心）应该全不透明，且**颜色没被去溢色改掉**
    assert got.getpixel((128, 128))[3] > 247, got.getpixel((128, 128))
    assert got.getpixel((128, 128))[:3] == (115, 115, 115), got.getpixel((128, 128))


def test_matte_works_for_blue_and_magenta(tmp_path):
    """蓝幕 / 品红幕同样要能抠（官方是按边框取样的，不限定绿幕）。"""
    for color in ((0, 0, 255), (255, 0, 255)):
        src = tmp_path / f"key{color[0]}_{color[2]}.png"
        _screen_image(color=color).save(src)
        out, err = ms.matte_image_file(str(src))
        assert err is None, err
        got = Image.open(out)
        assert got.getpixel((3, 3))[3] < 8, f"{color} 幕布没抠干净"
        assert got.getpixel((128, 128))[3] > 247


def test_matte_output_keeps_size_and_sits_beside_original(tmp_path):
    src = tmp_path / "sprite.png"
    _screen_image(size=192).save(src)
    out, _ = ms.matte_image_file(str(src))
    assert Path(out).name == "sprite_matte.png", "另存一份，不能覆盖原图"
    assert Image.open(out).size == (192, 192), "尺寸不能变"
    assert os.path.isfile(src), "原图必须还在（抠错了能重来）"


def test_matte_missing_file_returns_error_not_raise():
    out, err = ms.matte_image_file("/绝对不存在的/图.png")
    assert out is None and err, "失败要返回原因，不能抛（抠图是增强不是依赖）"


def test_auto_matte_depends_on_key_color():
    """白底/灰底**不自动抠**：幕布色按边框取，白毛白甲会被一起判成背景。"""
    assert ms.should_auto_matte("green") is True
    assert ms.should_auto_matte("blue") is True
    assert ms.should_auto_matte("white") is False
    assert ms.should_auto_matte("grey") is False
    # 用户明确给了就以用户为准
    assert ms.should_auto_matte("white", True) is True
    assert ms.should_auto_matte("green", False) is False


def test_publish_can_matte_and_keeps_both(tmp_path):
    """publish 时开抠像：原图与 `_matte` 两份都进角色目录。"""
    if ch.character_exists("Matte"):
        ch.delete_character("Matte")
    ch.create_character("Matte")
    try:
        src = tmp_path / "gen.png"
        _screen_image().save(src)
        r = ch.publish_image("Matte", str(src), matte=True)
        assert Path(r["path"]).is_file(), "原图要收进来"
        assert r["matte"] and Path(r["matte"]).is_file(), "抠图也要收进来"
        assert not r["matte_error"], r["matte_error"]
        assert Path(r["matte"]).name.endswith("_matte.png")
        # 没开抠像时就不该有 matte 字段的内容
        r2 = ch.publish_image("Matte", str(src), matte=False)
        assert r2["matte"] == ""
    finally:
        ch.delete_character("Matte")


def test_official_module_is_intact():
    """搬过来的官方实现要还在（签名就是那份契约）。"""
    import inspect

    sig = inspect.signature(ms.chroma_screen_matte.screen_matte)
    for name in ("tolerance", "softness", "despill_strength", "edge_width",
                 "matte_cleanup", "foreground_recover", "edge_decontaminate",
                 "edge_choke", "output_mode"):
        assert name in sig.parameters, f"官方参数被改没了：{name}"
    # 唯一允许的改动：不再 import comfy
    src = Path(ms.chroma_screen_matte.__file__).read_text(encoding="utf-8")
    assert "import comfy" not in src, "不该还依赖 ComfyUI"
