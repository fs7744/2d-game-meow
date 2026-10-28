"""提示词契约与下载器过滤规则（纯逻辑，不碰模型/网络）。"""
from __future__ import annotations

from workflows import presets


def test_preset_appends_suffix_and_negative():
    pos, neg = presets.build("blue slime", None, "sprite")
    assert pos.startswith("blue slime,")
    assert "flat plain background" in pos
    assert "pattern" in neg and "watermark" in neg


def test_preset_keeps_user_negative():
    _, neg = presets.build("x", "my own negative", "sprite")
    assert "my own negative" in neg
    # 预设里最脏的词仍排在最前（77 token 截断时优先保留）
    assert neg.index("pattern") < neg.index("my own negative")


def test_preset_none_is_passthrough():
    pos, neg = presets.build("just this", "", "none")
    assert pos == "just this" and neg == ""


def test_unknown_preset_raises():
    try:
        presets.build("x", None, "nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知 preset 必须报错")


def test_sprite_suffix_fits_clip_budget():
    """正向后缀必须给主体描述留余量：SD1.5 只有 77 token。"""
    pos, _ = presets.build("blue slime monster", None, "sprite")
    assert presets.rough_tokens(pos) < 50, presets.rough_tokens(pos)
    assert presets.rough_tokens(presets.SPRITE_SUFFIX) < 40


def test_rough_tokens_handles_cjk():
    assert presets.rough_tokens("蓝色史莱姆") > 0
    assert presets.rough_tokens("") == 0


# ---------------- 下载器过滤 ----------------

def test_fetch_pick_filters_junk():
    from tools.fetch_modelscope import pick

    files = [
        ("safety_checker/model.safetensors", 1_200_000_000),
        ("unet/diffusion_pytorch_model.safetensors", 3_000_000_000),
        ("unet/diffusion_pytorch_model.bin", 6_000_000_000),
        ("v1-5-pruned.ckpt", 7_000_000_000),
        ("model_index.json", 400),
    ]
    kept = pick(files, {"include": [], "exclude": []})
    names = [p for p, _ in kept]
    assert "safety_checker/model.safetensors" not in names
    assert "unet/diffusion_pytorch_model.bin" not in names      # 有同名 safetensors
    assert "v1-5-pruned.ckpt" not in names
    assert "unet/diffusion_pytorch_model.safetensors" in names


def test_fetch_pick_prefers_fp16():
    from tools.fetch_modelscope import pick

    files = [
        ("unet/diffusion_pytorch_model.safetensors", 3_000_000_000),
        ("unet/diffusion_pytorch_model.fp16.safetensors", 1_500_000_000),
    ]
    names = [p for p, _ in pick(files, {"include": [], "exclude": []})]
    assert names == ["unet/diffusion_pytorch_model.fp16.safetensors"]


def test_fetch_pick_include_and_exclude():
    from tools.fetch_modelscope import pick

    files = [("control_v11p_sd15_openpose.pth", 1),
             ("control_v11p_sd15_canny.pth", 1),
             ("control_v11p_sd15_depth.pth", 1)]
    rules = {"include": ["openpose", "canny"], "exclude": ["canny"]}
    names = [p for p, _ in pick(files, rules)]
    assert names == ["control_v11p_sd15_openpose.pth"]


def test_long_prompt_keeps_suffix():
    """用户主体描述写超了要截主体，绝不能把契约后缀丢掉。

    老实现是 prompt + suffix 直接交给 CLIP，超出 77 token 时被静默丢弃的恰好是
    后缀（扁平背景 / 硬描边），画出来背景就脏了 —— 这正是「提示词越长越难看」
    的真实原因。
    """
    long_prompt = ", ".join(["a very detailed knight with flame patterns"] * 8)
    pos, _ = presets.build(long_prompt, None, "sprite")
    assert pos.endswith(presets.SPRITE_SUFFIX)
    assert presets.rough_tokens(pos) <= 77


def test_fit_never_returns_empty_for_comma_free_text():
    """整段没有逗号且超预算时，按词截断，不能返回空串（等于没有负向词）。"""
    text = " ".join(["ugly"] * 200)
    out = presets.fit(text, budget=20)
    assert out and out.startswith("ugly")
    assert presets.rough_tokens(out) <= 20


def test_fit_drops_tail_sections():
    text = ", ".join(f"word{i}" for i in range(40))
    out = presets.fit(text, budget=20)
    assert "word0" in out and "word39" not in out
