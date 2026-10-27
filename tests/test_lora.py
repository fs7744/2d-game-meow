"""LoRA 注册表 + 去碎屑 + 贴边检测。"""
from __future__ import annotations

import json

import pytest
from PIL import Image, ImageDraw

from workflows import lora as L
from workflows import matte as M
from workflows import presets as P


# ---------------- LoRA 注册表 ----------------

def _fake_lora(root, name: str, size: int = 1024):
    d = root / "lora"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.safetensors"
    p.write_bytes(b"\0" * size)
    return p


def test_scan_empty_when_no_dir(tmp_path):
    assert L.scan(tmp_path) == []


def test_scan_reads_metadata(tmp_path):
    _fake_lora(tmp_path, "wuxia_v2")
    _fake_lora(tmp_path, "other")
    (tmp_path / "lora" / "loras.json").write_text(
        json.dumps({"wuxia_v2": {"label": "武侠", "trigger": "wuxia", "scale": 0.75}}),
        encoding="utf-8")

    items = {x["name"]: x for x in L.scan(tmp_path)}
    assert items["wuxia_v2"]["label"] == "武侠"
    assert items["wuxia_v2"]["trigger"] == "wuxia"
    assert items["wuxia_v2"]["scale"] == 0.75
    assert items["wuxia_v2"]["path"] == "lora/wuxia_v2.safetensors"
    # 没写元数据的用默认，但仍可用
    assert items["other"]["label"] == "other"
    assert items["other"]["scale"] == 0.8


def test_scan_survives_broken_json(tmp_path):
    _fake_lora(tmp_path, "x")
    (tmp_path / "lora" / "loras.json").write_text("{ broken", encoding="utf-8")
    items = L.scan(tmp_path)
    assert len(items) == 1 and items[0]["name"] == "x"


def test_resolve_accepts_name_and_path(tmp_path):
    p = _fake_lora(tmp_path, "wuxia_v2")
    assert L.resolve(tmp_path, "wuxia_v2") == str(p)
    assert L.resolve(tmp_path, "lora/wuxia_v2.safetensors") == str(p)
    assert L.resolve(tmp_path, str(p)) == str(p)
    assert L.resolve(tmp_path, None) is None
    assert L.resolve(tmp_path, "nope") is None


# ---------------- 去碎屑 ----------------

def _figure_with_debris():
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 150, 180], fill=(30, 30, 30, 255))     # 主体，面积 111*141
    d.rectangle([180, 120, 192, 140], fill=(200, 20, 20, 255))  # 印章（小）
    d.point([(20, 190)], fill=(1, 2, 3, 255))                   # 一粒屑
    return im


def test_despeckle_drops_seal_and_dust():
    out = M.despeckle(_figure_with_debris())
    a = out.getchannel("A")
    assert a.getpixel((190, 130)) == 0, "印章应被清掉"
    assert a.getpixel((20, 190)) == 0, "碎屑应被清掉"
    assert a.getpixel((100, 100)) == 255, "主体必须保留"


def test_despeckle_keeps_single_subject_untouched():
    im = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([10, 10, 40, 40], fill=(9, 9, 9, 255))
    assert M.despeckle(im).getchannel("A").getbbox() == (10, 10, 41, 41)


def test_despeckle_noop_when_empty():
    im = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    assert M.despeckle(im).getchannel("A").getbbox() is None


# ---------------- 贴边检测 ----------------

def test_edge_stats_clean_frame_is_zero():
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([20, 20, 80, 80], fill=(1, 1, 1, 255))
    assert M.edge_touch(im) == []
    assert max(M.edge_stats(im).values()) == 0.0


def test_edge_stats_flags_cut_feet():
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    # 前景一直画到最后一行的 60%，等价于脚被裁
    ImageDraw.Draw(im).rectangle([20, 20, 80, 99], fill=(1, 1, 1, 255))
    st = M.edge_stats(im)
    assert st["bottom"] > 0.5
    assert "bottom" in M.edge_touch(im)


def test_edge_touch_ignores_hairline_contact():
    """发丝蹭到边一两个像素是正常构图，不该算「被裁」。"""
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 60, 60], fill=(1, 1, 1, 255))
    d.point([(50, 0)], fill=(1, 1, 1, 255))       # 顶部单点接触，占比 1%
    assert M.edge_touch(im) == []
    assert M.edge_stats(im)["top"] == pytest.approx(0.01, abs=1e-6)


# ---------------- 武侠契约 ----------------

def test_wuxia_preset_exists_and_differs_from_sprite():
    pos, neg = P.build("swordsman", None, "wuxia")
    assert "ink painting" in pos
    # 武侠契约刻意不写 pixel art：实测 LoRA 会压过风格词，两头不到位
    assert "pixel art" not in pos
    assert "seal" in neg and "signature" in neg, "武侠 LoRA 会盖印章，必须压"
    assert "cut off head" in neg

    sprite_pos, _ = P.build("swordsman", None, "sprite")
    assert "pixel art" in sprite_pos


def test_wuxia_preset_keeps_budget():
    pos, neg = P.build("Wuxia, swordsman, holding sword, front view", None, "wuxia")
    assert P.rough_tokens(pos) <= 77
    assert P.rough_tokens(neg) <= 77


def test_unknown_preset_still_raises():
    with pytest.raises(ValueError):
        P.build("x", None, "nope")
