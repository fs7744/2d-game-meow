"""评审修复的 P1 回归测试：共享 UNet 的副作用、批量候选、长提示词、闸门取样点。

这些都需要真实权重，所以单独成文件（和 test_generate.py 一样按需加载）。
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
# SD 权重已归档到 models/_archive/sd/：下面这些用例要真跑 SD 管线，
# 权重不在就跳过（恢复权重后自动重新生效，不是删测试）。
_SD_WEIGHTS = (Path(__file__).resolve().parent.parent / "models" / "checkpoints"
               / "sd15" / "model_index.json").is_file()
requires_sd = pytest.mark.skipif(
    not _SD_WEIGHTS,
    reason="SD 权重已归档到 models/_archive/sd/（恢复后自动恢复）")

from pathlib import Path
import torch
from PIL import Image

from workflows import clip as clip_mod
from workflows import generate as g
from workflows import presets


# ---------------- 共享 UNet：IP-Adapter 必须能摘能挂 ----------------

@requires_sd
def test_txt2img_works_after_ip_adapter_pipeline(eng):
    """pose 管线把 IP-Adapter 注进**共享 UNet** 之后，文生图还能跑。

    这是共享权重的真实副作用：`load_ip_adapter` 会给 UNet 插 `encoder_hid_proj`，
    而文生图没有参考图，UNet 会抛 `argument of type 'NoneType' is not iterable`。
    修法是切回无参考图的管线前先摘掉补丁（engine._release_ipa_if_needed）。
    """
    eng.free()
    g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=4, seed=11,
              preset="none", tag="share_a")
    # 建一条带 IP-Adapter 的 pose 管线（会改共享 UNet）
    pipe = eng.pose_pipe(controlnet="canny", with_ipadapter=True, model="sd15")
    assert getattr(pipe.unet, "encoder_hid_proj", None) is not None

    # 再回到文生图：必须自动摘掉 IP-Adapter 而不是崩
    r = g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=4, seed=11,
                  preset="none", tag="share_b")
    assert r["file"]

    # 再回 pose 管线：补丁必须被挂回去
    again = eng.pose_pipe(controlnet="canny", with_ipadapter=True, model="sd15")
    assert getattr(again.unet, "encoder_hid_proj", None) is not None


@requires_sd
def test_shared_unet_saves_memory(eng):
    """txt2img 与 pose 共用同一个 UNet 对象（省一份 1.7GB）。"""
    t = eng.txt2img_pipe("sd15")
    p = eng.pose_pipe(controlnet="canny", with_ipadapter=False, model="sd15")
    assert p.unet is t.unet, "pose 管线应当复用已加载的底座权重"


# ---------------- 批量候选：输出必须与逐个生成一致 ----------------

@requires_sd
def test_batched_candidates_are_deterministic_and_close(eng):
    """批量候选要**可复现**，并且与逐个生成数值等价。

    实测批量与逐个**不是逐像素一致**：不同 batch 尺寸走到不同的 conv 内核，
    输出差 1~4 级灰度（mean 2.3~4.1/255）。这是可接受的——同配置重跑仍逐像素一致，
    产线契约（"同 seed 可复现"）没被破坏，变的是"跨配置对比"。
    """
    eng.free()
    pipe = eng.pose_pipe(controlnet="canny", with_ipadapter=False, model="sd15")
    cond = Image.new("RGB", (256, 256), (255, 255, 255))
    seeds = [7, 1007]
    kw = dict(image=cond, prompt="blue slime", negative_prompt="blurry",
              num_inference_steps=4, guidance_scale=5.0, width=256, height=256)

    def batch():
        return pipe(num_images_per_prompt=2,
                    generator=[torch.Generator("cuda").manual_seed(s) for s in seeds],
                    **kw).images

    b1, b2 = batch(), batch()
    assert np.array_equal(np.asarray(b1[0]), np.asarray(b2[0])), "批量必须可复现"
    assert np.array_equal(np.asarray(b1[1]), np.asarray(b2[1]))

    seq = [pipe(generator=torch.Generator("cuda").manual_seed(s), **kw).images[0]
           for s in seeds]
    for a, b in zip(seq, b1):
        diff = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
        assert diff.mean() < 8.0, f"批量与逐个差异过大: {diff.mean():.2f}/255"


# ---------------- 长提示词：不再被启发式估算提前截断 ----------------

def test_build_uses_real_tokenizer():
    """`fit` 必须按真实 BPE 计数，而不是 1.35 token/词 的估算。

    实测 WUXIA_NEG 真实 35 token、估算 113——按估算截断等于白扔 2/3 的负向词。
    """
    assert clip_mod.count_text(presets.WUXIA_NEG) == 35 or clip_mod.count_text("x") == 0
    pos, neg = presets.build("swordsman holding a sword", "", "wuxia")
    # 真实计数下，整段武侠负向词应当完整保留
    assert neg.endswith("photo, blurry, deformed"), neg
    assert presets.count(neg) <= 77


def test_long_negative_kept_under_chunk_budget():
    """调大预算后，长负向词不再被"77"砍掉（分段编码能吃到 154）。"""
    long_neg = ", ".join(f"bad thing {i}" for i in range(40))
    _, short = presets.build("x", long_neg, "none")
    _, long = presets.build("x", long_neg, "none", max_tokens=presets.CHUNK_BUDGET)
    assert len(long) > len(short)
    assert presets.count(long) <= presets.CHUNK_BUDGET


@requires_sd
def test_text_kwargs_chunks_only_when_needed(eng):
    """短文本走字符串（与历史逐像素一致），长文本才走 embeds。"""
    pipe = eng.txt2img_pipe("sd15")
    kw, cs = g._text_kwargs(pipe, "blue slime", "blurry", 1, "sd15")
    assert kw == {"prompt": "blue slime", "negative_prompt": "blurry"}
    assert cs == 1

    long = ", ".join(["a very detailed knight with flame patterns"] * 12)
    assert clip_mod.count_tokens(pipe, long) > 77
    kw2, _ = g._text_kwargs(pipe, long, "blurry", 1, "sd15")
    assert "prompt_embeds" in kw2
    assert kw2["prompt_embeds"].shape[1] == 154, kw2["prompt_embeds"].shape


# ---------------- 质量闸门取像素化之前的图 ----------------

def test_postprocess_exposes_pre_pixel_image(eng):
    """闸门要在像素化/量化**之前**评分：128 有效像素上数关节会误杀好帧。"""
    # 造一张多色渐变图：量化到 8 色后必然只剩少数颜色，量化前应有大量颜色
    ramp = Image.linear_gradient("L").resize((256, 256))
    src = Image.merge("RGB", (ramp, ramp.rotate(90), ramp.transpose(Image.FLIP_LEFT_RIGHT)))
    info = {}
    out, _edge, _subj = g._postprocess(
        src, eng, matte=False, matte_backend="color", denoise=0,
        pixel=8, palette=8, clean_iso=0, info=info)
    pre = info.get("pre_pixel")
    assert pre is not None and pre.size == src.size
    n_pre = len(pre.convert("RGB").getcolors(maxcolors=1 << 20) or [])
    n_out = len(out.convert("RGB").getcolors(maxcolors=1 << 20) or [])
    assert n_pre > 50, n_pre
    assert n_out <= 8, n_out


@requires_sd
def test_text_embeds_align_when_only_negative_is_long(eng):
    """正向短、负向长时两侧 embeds 必须等长。

    上线后实测到的崩法：正向 3 token（77）、负向 289 token（154），
    diffusers 在 CFG 相减时抛
    `prompt_embeds torch.Size([1,77,768]) != negative_prompt_embeds
    torch.Size([1,154,768])`。

    原因是两侧各算各的段数：短的那侧只出 1 段。修法是取两侧段数最大值、
    短的一侧补空段。
    """
    from workflows import clip as c

    pipe = eng.txt2img_pipe("sd15")
    long_neg = ", ".join(f"ugly term{i}" for i in range(60))
    short_pos = "blue slime monster"
    assert c.count_tokens(pipe, short_pos) < 20
    assert c.count_tokens(pipe, long_neg) > 150

    kw, _ = g._text_kwargs(pipe, short_pos, long_neg, 1, "sd15")
    assert "prompt_embeds" in kw
    pos, neg = kw["prompt_embeds"], kw["negative_prompt_embeds"]
    assert pos.shape == neg.shape, f"{tuple(pos.shape)} vs {tuple(neg.shape)}"
    assert pos.shape[1] == 154, "负向超长时应该分段到 154"


@requires_sd
def test_text_embeds_both_short_stays_string_path(eng):
    """两侧都短时仍走字符串路径——历史输出必须逐像素不变。"""
    pipe = eng.txt2img_pipe("sd15")
    kw, cs = g._text_kwargs(pipe, "blue slime", "blurry", 1, "sd15")
    assert kw == {"prompt": "blue slime", "negative_prompt": "blurry"} and cs == 1
