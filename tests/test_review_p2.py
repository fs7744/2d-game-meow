"""评审修复的 P2/P3 回归测试：高清修复、少步蒸馏、抠图模型选择、网格对齐、资产索引。"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from workflows import generate as g
from workflows import matte as M
from workflows import postprocess as pp
from workflows.assetdb import AssetIndex

# SD 权重已归档：这两条要真跑 SD 的 hires / 蒸馏 LoRA（恢复权重后自动生效）
from pathlib import Path as _P

_SD_WEIGHTS = (_P(__file__).resolve().parents[1] / "models" / "checkpoints"
               / "sd15" / "model_index.json").is_file()
requires_sd = pytest.mark.skipif(
    not _SD_WEIGHTS,
    reason="SD 权重已归档到 models/_archive/sd/（恢复后自动恢复）")


# ---------------- 高清修复（Q6：512 出结构 → 放大补细节）----------------

@requires_sd
def test_hires_fix_doubles_size_and_reports(eng):
    r = g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=3,
                  seed=5, preset="none", pixel=0, hires=2.0, hires_steps=3,
                  tag="hires_test")
    out = Image.open(r["file"])
    assert out.size == (512, 512), out.size
    assert r["meta"]["hires"] == 2.0
    assert r["meta"]["hires_steps"] == 3
    assert r["meta"]["width"] == 512, "meta 要记录实际尺寸"


@requires_sd
def test_hires_off_by_default(eng):
    r = g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=3,
                  seed=5, preset="none", tag="nohires")
    assert r["meta"]["hires"] is None
    assert Image.open(r["file"]).size == (256, 256)


# ---------------- 少步蒸馏（Q3）----------------

@requires_sd
def test_distill_missing_gives_actionable_error(eng):
    """没有权重时必须报清楚「放哪个文件」，而不是晦涩的 diffusers 异常。"""
    with pytest.raises(FileNotFoundError, match="蒸馏 LoRA 缺失"):
        g.txt2img(eng, prompt="x", width=256, height=256, steps=2, seed=1,
                  preset="none", distill="lcm", tag="distill_err")


def test_distill_unknown_name_rejected(eng):
    with pytest.raises(ValueError, match="未知蒸馏方案"):
        eng.apply_distill(object(), "magic")


# ---------------- 抠图：BiRefNet 优先 + 软 alpha ----------------

def test_pick_model_prefers_birefnet(tmp_path):
    assert M.pick_model(tmp_path) is None
    (tmp_path / M.AI_MODEL).write_bytes(b"x")
    assert M.pick_model(tmp_path)[1] == "isnet-general-use"
    (tmp_path / "birefnet-general.onnx").write_bytes(b"x")
    assert M.pick_model(tmp_path)[1] == "birefnet-general", "有 BiRefNet 就该优先用"
    assert M.ai_available(tmp_path) is True


def test_matte_auto_forwards_alpha_mode_to_color_backend(tmp_path):
    """alpha_mode 只有分割后端认；色键后端不能因为多收了参数就 TypeError。"""
    icon = Image.new("RGB", (64, 64), (255, 255, 255))
    icon.paste((20, 20, 40), (20, 20, 44, 44))
    out = M.matte_auto(icon, "color", tmp_path, alpha_mode="soft")
    assert out.mode == "RGBA"


def test_matte_ai_soft_keeps_half_transparency(monkeypatch, tmp_path):
    """soft 模式必须保住半透明边（高清插画的发丝/纱）。"""
    half = Image.new("RGBA", (32, 32), (10, 20, 30, 100))
    monkeypatch.setattr(M, "load_ai_session", lambda *a, **k: object())
    monkeypatch.setitem(__import__("sys").modules, "rembg",
                        type("m", (), {"remove": staticmethod(lambda img, **k: half)}))
    soft = M.matte_ai(Image.new("RGB", (32, 32), "white"), tmp_path, alpha_mode="soft")
    hard = M.matte_ai(Image.new("RGB", (32, 32), "white"), tmp_path, alpha_mode="binary")
    assert set(np.unique(np.asarray(soft.getchannel("A"))).tolist()) == {100}
    assert set(np.unique(np.asarray(hard.getchannel("A"))).tolist()) <= {0, 255}


# ---------------- 主链路网格对齐像素化（Q4）----------------

def test_pixelate_grid_aligns_when_input_is_pixel_art():
    """已经是像素画的输入 → 每格单色且与真值一致；连续调图则回退 BOX。"""
    rng = np.random.default_rng(0)
    pal = np.array([(20, 30, 40), (200, 60, 60), (240, 240, 220), (60, 160, 80)],
                   dtype=np.uint8)
    small = Image.fromarray(pal[rng.integers(0, 4, (128, 128))], "RGB")
    big = small.resize((768, 768), Image.NEAREST)          # 6px 格

    out = pp.pixelate(big, 6, grid_align=True)
    a = np.asarray(out.convert("RGB"))
    cells = a.reshape(128, 6, 128, 6, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 36, 3)
    assert (cells == cells[:, :1, :]).all(), "每格必须单色"
    assert np.array_equal(a[::6, ::6], np.asarray(small)), "每格取样要与真值一致"

    photo = Image.fromarray(rng.integers(0, 255, (768, 768, 3), dtype=np.uint8), "RGB")
    # 连续调没有可检测网格 → 必须回退到 BOX 路径（不报错、尺寸不变）
    assert pp.pixelate(photo, 6, grid_align=True).size == photo.size


# ---------------- 资产索引（M7：不再每次全量解析）----------------

def _entry(name: str, scope: str = "default") -> dict:
    return {"file": str(name), "scope": scope, "workflow": "w1",
            "created": "2026-10-28T00:00:00"}


def test_asset_index_is_incremental(tmp_path):
    (tmp_path / "index.jsonl").write_text("", encoding="utf-8")
    ix = AssetIndex(tmp_path)
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    with (tmp_path / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_entry(tmp_path / "a.png")) + "\n")
    assert ix.sync() == 1
    assert ix.sync() == 0, "没有新行时不该重复入库"

    with (tmp_path / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_entry(tmp_path / "b.png")) + "\n")
    assert ix.sync() == 1

    items = ix.recent(limit=10)
    assert [i["workflow"] for i in items] == ["w1", "w1"]
    assert ix.recent(limit=10, project="nope") == []


def test_asset_index_skips_vanished_files(tmp_path):
    """文件被手工删掉后不再返回（否则前端拿到 404 链接）。"""
    (tmp_path / "index.jsonl").write_text("", encoding="utf-8")
    ix = AssetIndex(tmp_path)
    p = tmp_path / "gone.png"
    p.write_bytes(b"x")
    with (tmp_path / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_entry(p)) + "\n")
    ix.sync()
    assert len(ix.recent(limit=5)) == 1
    p.unlink()
    assert ix.recent(limit=5) == []


def test_asset_index_truncated_log_rebuilds(tmp_path):
    """jsonl 被重建（手工删过）时要整体重建，而不是按旧偏移读到文件末尾。"""
    (tmp_path / "index.jsonl").write_text("", encoding="utf-8")
    ix = AssetIndex(tmp_path)
    (tmp_path / "a.png").write_bytes(b"x")
    with (tmp_path / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_entry(tmp_path / "a.png")) + "\n")
    ix.sync()
    (tmp_path / "index.jsonl").write_text(
        json.dumps(_entry(tmp_path / "a.png")) + "\n", encoding="utf-8")
    assert ix.sync() >= 0
    assert len(ix.recent(limit=5)) == 1


def test_sync_survives_corrupt_line(tmp_path):
    """索引里混进坏字节时，只跳过坏行，不能让整个清单接口挂掉。

    实测线上索引里出现过一段 2990 字节的 NUL（并发追加交错），
    `json.loads(bytes)` 会把它误判成 UTF-32 并抛 UnicodeDecodeError——
    那不是 JSONDecodeError，没兜住就是 500。
    """
    (tmp_path / "index.jsonl").write_bytes(b"")
    ix = AssetIndex(tmp_path)
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"x")

    blob = b""
    blob += json.dumps(_entry(tmp_path / "a.png")).encode() + b"\n"
    blob += b"\x00" * 2990 + b"\n"                     # 损坏行
    blob += json.dumps(_entry(tmp_path / "b.png")).encode() + b"\n"
    (tmp_path / "index.jsonl").write_bytes(blob)

    assert ix.sync() == 2, "两条正常记录都要入索引"
    assert len(ix.recent(limit=10)) == 2


def test_append_line_is_single_write(tmp_path):
    """追加一条只产生一行，且不破坏已有内容。"""
    from workflows.assetdb import append_line

    p = tmp_path / "index.jsonl"
    append_line(p, {"file": "x", "scope": "default"})
    append_line(p, {"file": "y", "scope": "default"})
    lines = p.read_bytes().decode("utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(x)["scope"] == "default" for x in lines)
    assert p.read_bytes().endswith(b"\n")


def test_sync_survives_corrupt_line(tmp_path):
    """索引里混进坏字节时，只跳过坏行，不能让整个清单接口挂掉。

    实测线上索引里出现过一段 2990 字节的 NUL（并发追加交错），
    `json.loads(bytes)` 会把它误判成 UTF-32 并抛 UnicodeDecodeError——
    那不是 JSONDecodeError，没兜住就是 500。
    """
    (tmp_path / "index.jsonl").write_bytes(b"")
    ix = AssetIndex(tmp_path)
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"x")

    blob = b""
    blob += json.dumps(_entry(tmp_path / "a.png")).encode() + b"\n"
    blob += b"\x00" * 2990 + b"\n"                     # 损坏行
    blob += json.dumps(_entry(tmp_path / "b.png")).encode() + b"\n"
    (tmp_path / "index.jsonl").write_bytes(blob)

    assert ix.sync() == 2, "两条正常记录都要入索引"
    assert len(ix.recent(limit=10)) == 2


def test_append_line_is_single_write(tmp_path):
    """追加一条只产生一行，且不破坏已有内容。"""
    from workflows.assetdb import append_line

    p = tmp_path / "index.jsonl"
    append_line(p, {"file": "x", "scope": "default"})
    append_line(p, {"file": "y", "scope": "default"})
    lines = p.read_bytes().decode("utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(x)["scope"] == "default" for x in lines)
    assert p.read_bytes().endswith(b"\n")
