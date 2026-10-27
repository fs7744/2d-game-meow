"""底模自选：SD1.5 / SDXL。"""
from __future__ import annotations

import time

import pytest
from PIL import Image, ImageDraw

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


@pytest.mark.slow
def test_w3_sdxl_uses_base_weights(eng, assets, root):
    """W3 也支持 SDXL：直接用 base 权重跑 StableDiffusionXLInpaintPipeline。

    SDXL 没有官方 inpaint 微调版，diffusers 的 XL inpaint 管线能吃 4 通道 UNet
    （把 mask+被遮图拼进 latent 只在 9 通道 UNet 上发生），所以不需要额外权重。
    这里验证三件事：能建管线、选区外逐点不变、meta 记对了型号。
    """
    import numpy as np

    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 底模")

    src = assets / "_test_w3_sdxl_src.png"
    mask = assets / "_test_w3_sdxl_mask.png"
    im = Image.new("RGB", (768, 768), (30, 60, 90))
    ImageDraw.Draw(im).rectangle([200, 200, 400, 400], fill=(200, 30, 30))
    im.save(src)
    m = Image.new("L", (768, 768), 0)
    ImageDraw.Draw(m).rectangle([350, 350, 550, 550], fill=255)
    m.save(mask)

    eng.free()                       # 先腾显存，SDXL 各管线加起来容易顶到 24G
    t0 = time.time()
    r = g.inpaint(eng, image=src, mask=mask, prompt="green grass block", steps=10,
                  cfg=6.0, seed=7, model="sdxl", tag="w3_sdxl_test")
    assert r["meta"]["model"] == "sdxl"

    before = np.asarray(Image.open(src).convert("RGB")).astype(int)
    after = np.asarray(Image.open(r["file"]).convert("RGB")).astype(int)
    outside = np.asarray(Image.open(mask).convert("L")) == 0
    assert np.abs(after - before).max(axis=2)[outside].max() == 0, "选区外被改动了"
    print(f"  [w3 sdxl] {time.time() - t0:.1f}s peak={r['meta']['peak_vram_gb']}GB")
    eng.free()


def test_w4_sdxl_needs_its_own_controlnet(eng):
    """SD1.5 与 SDXL 的 ControlNet 形状不同，绝不能混用。

    有 SDXL 版 OpenPose 就应当能建管线；没有则必须报「SDXL 版 ControlNet 缺失」，
    而不是悄悄拿 SD1.5 的权重去跑（那会直接 shape 不匹配崩掉）。
    """
    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 底模")
    try:
        pipe = eng.pose_pipe(model="sdxl", with_ipadapter=False)
        assert pipe is not None
    except FileNotFoundError as exc:
        assert "SDXL" in str(exc), str(exc)


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


@pytest.mark.slow
def test_w4_sdxl_end_to_end(eng, pose_frames):
    """W4 + SDXL：骨架锁全身 + SDXL 画质。出帧即算通过（构图质量靠肉眼看）。"""
    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 底模")
    if not (eng.root / "controlnet" / "sdxl_openpose" / "config.json").exists():
        pytest.skip("未下载 SDXL 版 OpenPose ControlNet")

    from workflows import generate as g

    r = g.poseswap(eng, pose_images=list(pose_frames)[:2],
                   prompt="pixel art game sprite, wuxia swordsman, red robe, holding sword",
                   steps=12, seed=999, width=1024, height=1024, cols=2,
                   pixel=8, palette=32, matte=True, preset="pixel",
                   model="sdxl", tag="w4_sdxl_test")
    assert r["frames"] == 2
    assert r["model"] == "sdxl"       # poseswap 把 meta 展开在返回字典里
    eng.free()


@pytest.mark.slow
def test_sdxl_ip_adapter_loads(eng):
    """SDXL 的 IP-Adapter 与 SD1.5 共享 ViT-H image encoder。

    这里最容易踩的坑：HF 上 sdxl_models/image_encoder 是 ViT-bigG（hidden 1664 /
    projection 1280），而 ip-adapter_sdxl_vit-h 要的是 ViT-H（1280 / 1024）。
    拿错会报 mat1 and mat2 shapes cannot be multiplied。
    """
    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 底模")
    models = eng.root / "ipadapter" / "models"
    if not (models / "ip-adapter_sdxl_vit-h.safetensors").exists():
        pytest.skip("未下载 SDXL 版 IP-Adapter 权重")

    base, sub, name = eng.ipa_weight("sdxl")
    assert sub == "models" and name == "ip-adapter_sdxl_vit-h.safetensors"
    # 与 SD1.5 共用同一个 image_encoder（都是 ViT-H）
    assert base == str(eng.root / "ipadapter")
    b2, s2, _ = eng.ipa_weight("sd15")
    assert (base, sub) == (b2, s2)


@pytest.mark.slow
def test_sdxl_controlnet_kinds(eng):
    """SDXL 的三条 ControlNet（openpose / canny / tile）按 model 分开取。"""
    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 底模")
    cn_root = eng.root / "controlnet"
    for kind in ("openpose", "canny", "tile"):
        if not (cn_root / f"sdxl_{kind}" / "config.json").exists():
            pytest.skip(f"未下载 SDXL 版 {kind}")
        cn = eng.controlnet(kind, "sdxl")
        assert cn is not None
    # SD1.5 与 SDXL 的 ControlNet 必须缓存在不同 key 下
    assert any(k.endswith(":sdxl") for k in eng._ctrl)
    assert not any(k == "openpose" for k in eng._ctrl if "sdxl" in k)
    eng.free()
