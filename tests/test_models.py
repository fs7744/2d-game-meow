"""底模自选：SD1.5 / SDXL。"""
from __future__ import annotations

import time

import pytest

from workflows.engine import Engine
from workflows import generate as g


def test_available_models_lists_roots(root, eng):
    from workflows.engine import BASE_MODELS

    have = eng.available_models()
    assert "sd15" in have                       # 产线主链路必须有
    for m in have:
        assert m in BASE_MODELS


def test_unknown_model_rejected(eng):
    with pytest.raises(ValueError, match="未知底模"):
        eng.txt2img_pipe("sd3")


def test_models_endpoint(client, eng):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "sd15" in body["available"]

    # 前端靠这个端点决定下拉内容：没下载的不能出现，否则选了却跑不动
    for m in body["available"]:
        assert (eng.root / "checkpoints" / m / "model_index.json").exists(), m


def test_txt2img_model_echoed(eng):
    r = g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=6,
                  seed=1, preset="none", tag="model_sd15")
    assert r["meta"]["model"] == "sd15"


def test_w4_and_w3_refuse_sdxl(eng):
    """W3 局部重绘与 W4 姿态驱动只有 SD1.5 权重，选错必须明确报错而不是出废图。"""
    with pytest.raises(ValueError, match="只支持 SD1.5"):
        eng.inpaint_pipe("sdxl")
    with pytest.raises(ValueError, match="只支持 SD1.5"):
        eng.pose_pipe(model="sdxl")


@pytest.mark.slow
def test_sdxl_generates_when_present(eng, root):
    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 权重")

    t0 = time.time()
    r = g.txt2img(eng, prompt="pixel art game sprite, white cat knight, red cape, katana, front view",
                  width=1024, height=1024, steps=20, cfg=6.5, seed=4242,
                  preset="sprite", pixel=8, palette=32, matte=True,
                  model="sdxl", tag="sdxl_test")
    assert r["meta"]["model"] == "sdxl"
    assert r["meta"]["seconds"] > 0
    print(f"  [sdxl] {time.time() - t0:.1f}s peak={r['meta']['peak_vram_gb']}GB")
    eng.free()
