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

    # 每个底模都要带推荐参数：SDXL 推荐 768 而不是 1024。
    # 实测 1024² 反而更差（角色占比大 → 更易顶边，可交付率 92%→75%），
    # 前端若按"SDXL 就该 1024"填表单，用户会稳定拿到更差的结果。
    #
    # **例外：动漫底模**（底模名在 Engine.MODEL_RECOMMEND 里的）是反过来的——
    # 它们是在 Danbooru 高分辨率插画上训的，1024² 才出细节；而且
    # pixel/palette 必须是 0：动漫立绘要保画质，像素化是给精灵的，
    # 把插画压成 6px 格子等于毁掉它。
    from workflows.engine import Engine as _E

    rec = body["recommend"]
    for m in body["available"]:
        r = rec[m]
        if m in _E.MODEL_RECOMMEND:
            assert r["width"] == 1024 and r["height"] == 1024, r
            assert r["pixel"] == 0 and r["palette"] == 0, r
            # 动漫底模要带上 clip_skip 与 preset，前端靠它们自动填表单
            assert r["clip_skip"] == 2 and r["preset"] == "anime", r
        else:
            assert r["width"] == 768 and r["height"] == 768, r
            assert r["pixel"] > 1 and r["palette"] > 1
        # 候选上限也要带出来：前端拿它当输入框 max。sdxl@1024 批量 2 会占满
        # 24GB，不报 OOM 但耗时翻 3.5 倍——这个上限得由后端给，不能让用户猜。
        assert 1 <= r["max_candidates"] <= 6, r


def test_txt2img_model_echoed(eng):
    r = g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=6,
                  seed=1, preset="none", tag="model_sd15")
    assert r["meta"]["model"] == "sd15"


def test_clip_skip_on_sdxl_uses_penultimate_by_default(eng):
    """SDXL 的 clip_skip 语义与 SD1.5 不同，记进 meta 的是**换算后**的档位。

    以前这里断言「SDXL 上 clip_skip 不生效、记成 1」——那是旧行为。
    现在 SDXL 也支持 clip_skip，但要做换算（见 generate._text_kwargs）：

        diffusers 的 SDXL encode_prompt：
            clip_skip is None  → hidden_states[-2]      ← 默认 = 倒数第二层
            else               → hidden_states[-(clip_skip + 2)]

    也就是说 **SDXL 默认取的就是倒数第二层**，那已经等价于动漫底模
    (Illustrious/Animagine) 推荐的 "clip skip 2"。所以：
      · 用户给 1 或 2 → 都是默认档（penultimate），meta 记 2
      · 用户给 3      → 真多跳一层
    不换算的话，直接把 2 传进去会跳到 -4 层，出图一片灰雾（实测过）。
    """
    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 底模")
    r = g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=4, seed=1,
                  preset="none", model="sdxl", clip_skip=2, tag="cs_xl")
    assert r["meta"]["clip_skip"] == 2
    assert r["meta"]["model"] == "sdxl"


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


# ---------------- SDXL Refiner：embeds 维度与「精修不糊」----------------

def test_refiner_embeds_match_unet_cross_attention_dim(eng):
    """refiner 的 cross-attn embeds 必须正好等于 UNet 的 cross_attention_dim。

    踩过的坑：写成 `CLIP-L(768) + OpenCLIP 前 512 = 1280`——维度凑对了，
    但**语义完全错位**。refiner 拿到「半段 CLIP-L + 半段 OpenCLIP」的拼接
    向量，按错误语义去噪，出图糊成一团灰雾（实测拉普拉斯方差 1826 → 480，
    白底线稿直接变灰底）。正确做法是只取 text_encoder_2 的完整 1280 维隐层：
    SDXL refiner 的 `cross_attention_dim=1280` 与 OpenCLIP bigG 的 hidden
    size 是同一个数，不是巧合。

    这条断言只看维度的话抓不住 bug（拼出来的也是 1280），所以同时断言
    **与 OpenCLIP 隐层逐元素相等**——拼接向量不可能等于它。
    """
    if not eng.has_refiner():
        pytest.skip("未下载 SDXL refiner")
    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 底模")

    import torch as _t

    base = eng.txt2img_pipe("sdxl")
    rp = eng.refiner_pipe()
    pos, neg, pos_pool, neg_pool = eng.refiner_embeds("a cat knight", "blurry")

    dim = rp.unet.config.cross_attention_dim
    assert pos.shape == (1, 77, dim), pos.shape
    assert neg.shape == (1, 77, dim), neg.shape
    # pooled 走 text_embeds，也是 1280
    assert pos_pool.shape == (1, 1280), pos_pool.shape
    assert neg_pool.shape == (1, 1280), neg_pool.shape

    # 必须是 OpenCLIP 隐层本身，不能是拼接出来的
    tok2, enc2 = base.tokenizer_2, base.text_encoder_2
    ids = tok2("a cat knight", padding="max_length",
               max_length=tok2.model_max_length, truncation=True,
               return_tensors="pt").input_ids.to(enc2.device)
    with _t.no_grad():
        ref = enc2(ids, output_hidden_states=True).hidden_states[-2]
    assert _t.allclose(pos, ref, atol=1e-4), "cross-attn embeds 不等于 OpenCLIP 隐层（又拼了？）"


def test_refiner_strength_maps_from_start_ratio():
    """「精修起点」→ 精修幅度 = 1 − 起点，且有安全上下限。

    为什么用 strength 而不是 denoising_start：denoising_start 是「底模提前停」
    的两段式参数，这里底模已经跑满，再传 0.8 等于把成品重新加噪到 80% 再只走
    4 步——实测糊成灰雾。精修本质是低幅度 img2img，对应 strength。
    """
    for end, want in ((0.8, 0.2), (0.9, 0.1), (0.5, 0.5)):
        got = max(0.03, min(0.60, 1.0 - float(end)))
        assert abs(got - want) < 1e-6, (end, got)
    # 极端值要夹住：起点 0.99 不能让 strength 归零（否则等于没跑）
    assert max(0.03, min(0.60, 1.0 - 0.99)) == 0.03
    assert max(0.03, min(0.60, 1.0 - 0.1)) == 0.60


def test_refiner_output_is_not_mush(eng):
    """回归：勾了精修不能把图弄糊。

    实测基线（768²/20 步）：
      不精修     拉普拉斯方差 ~1826
      修复前精修   ~727，且整体像素差 28（糊 + 变色）
      修复后精修   ~1616，像素差 ~8

    阈值定在「方差 ≥ 不精修的 70% 且像素差 ≤ 20」——修复前的 727/28 会被
    两条同时拦下，修复后（1616/8）通过。用相对值而不是绝对值，换显卡/换
    权重也能用。
    """
    if not eng.has_refiner():
        pytest.skip("未下载 SDXL refiner")
    if "sdxl" not in eng.available_models():
        pytest.skip("未下载 SDXL 底模")

    import numpy as np
    from PIL import Image

    def lap_var(im):
        a = np.asarray(im.convert("L"), dtype=np.float32)
        r = (4 * a - np.roll(a, 1, 0) - np.roll(a, -1, 0)
             - np.roll(a, 1, 1) - np.roll(a, -1, 1))
        return float(r[1:-1, 1:-1].var())

    # **步数不能太低**：strength 0.2 × 8 步只剩 1 步，改动量太小，
    # 条件错了也看不出糊（第一版用 8 步就是这么漏过去的）。20 步 → 4 步够暴露。
    kw = dict(prompt="a cat knight, red cape", negative=None, preset="none",
              width=512, height=512, steps=20, cfg=6.5, seed=4242,
              model="sdxl", clip_skip=2, matte=False)
    off = g.txt2img(eng, refiner=False, tag="rf_off", **kw)
    on = g.txt2img(eng, refiner=True, refiner_end=0.8, refiner_steps=20,
                   tag="rf_on", **kw)

    a = Image.open(off["file"])
    b = Image.open(on["file"])
    mine, theirs = lap_var(a), lap_var(b)
    diff = float(np.abs(np.asarray(a.convert("RGB"), dtype=np.int16)
                        - np.asarray(b.convert("RGB"), dtype=np.int16)).mean())

    assert theirs >= mine * 0.70, f"精修把图弄糊了：{mine:.0f} → {theirs:.0f}"
    assert diff <= 20, f"精修改动过大（应当小幅）：{diff:.1f}"
    m = on["meta"]
    assert m["refiner"] is True
    assert m["refiner_strength"] == 0.2
    assert m["refiner_steps_run"] >= 1
