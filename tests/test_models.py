"""底模与能力表：Qwen-only 之后只有 qwen-image-2.1 一个底模。

SD 时代这个文件测的是「sd15 / sdxl / 动漫底模各自该用多大尺寸、clip_skip 怎么换算、
refiner 的 embeds 维度对不对」——那些代码已经删了（见 docs/PLAN-QWEN-ONLY.md §6）。
现在要钉住的是：**只剩一个底模时，接口仍然自洽**。
"""
from __future__ import annotations

import pytest

from workflows import generate as g
from workflows.engine import DEFAULT_MODEL


def test_available_models_is_qwen_only(eng):
    have = eng.available_models()
    assert have == [DEFAULT_MODEL], have


def test_unknown_model_rejected(eng):
    with pytest.raises(ValueError, match="只有 qwen-image-2.1"):
        eng.txt2img_pipe("sd3")


def test_models_endpoint(client, eng):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] == [DEFAULT_MODEL]
    assert set(body["arch"].values()) == {"qwen21"}

    rec = body["recommend"][DEFAULT_MODEL]
    # 768² / 24 步 / 不像素化：每一条都有实测依据（PLAN §2.6 的步数梯度）
    assert rec["width"] == 768 and rec["height"] == 768
    assert rec["pixel"] == 0 and rec["palette"] == 0
    assert rec["steps"] >= 20
    assert 1 <= rec["max_candidates"] <= 6

    caps = body["caps"][DEFAULT_MODEL]
    assert caps["transparent"] is True and caps["instruction"] is True
    assert caps["lora"] is False                      # Qwen 不吃 SD LoRA
    assert len(caps["workflows"]) == 10
    assert set(caps["approx"]) == {"inpaint", "ai_repaint", "poseswap", "sprite_swap"}
    # 已经没有 refiner 了：前端不该再看到这个开关
    assert body["refiner"] is False


def test_arch_is_always_qwen21(eng):
    assert eng.arch(DEFAULT_MODEL) == "qwen21"
    assert eng.is_xl(DEFAULT_MODEL) is False


def test_txt2img_model_echoed(eng, monkeypatch, tmp_path):
    """meta 里要如实记底模名——产物可追溯靠它。"""
    from PIL import Image

    monkeypatch.setattr(g, "ASSETS", tmp_path)
    monkeypatch.setattr(g, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(g, "_qwen_call",
                        lambda *a, **kw: Image.new("RGBA", (kw["width"], kw["height"]),
                                                   (200, 30, 30, 255)))
    r = g.txt2img(eng, prompt="blue slime", width=64, height=64, steps=8,
                  seed=1, preset="none", tag="model_echo")
    assert r["meta"]["model"] == DEFAULT_MODEL
