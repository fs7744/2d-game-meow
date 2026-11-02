"""提示词契约（`workflows/presets.py`）。

核心契约：**拼接，不截断**。用户在框里写多少，就原样交给模型多少——
模型侧（Qwen3-VL + diffusers 的 encode_prompt）本来就没有 truncation，
SD 时代那套「按 token 预算砍」已经删了（见 `presets.build` 的 docstring）。
"""
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
    # 预设的词仍排在用户词前面：顺序即优先级，用户看到的顺序要稳定
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


def test_empty_prompt_yields_suffix_only():
    """没写提示词时不能拼出一个前导逗号。"""
    pos, _ = presets.build("", None, "sprite")
    assert pos == presets.SPRITE_SUFFIX
    assert not pos.startswith(",")


# ---------------- 不截断（这条是产品契约，退化过两次） ----------------

def test_long_prompt_is_delivered_whole():
    """**一个字都不能砍**：用户写 1000 字就给 1000 字。

    历史上这里有过两版截断：先是按 77 token 硬砍（CLIP 时代），后来改成
    154（`CHUNK_BUDGET`）。两次都错在同一个地方——**我们替用户做减法**。
    Qwen 的 tokenizer 上限 262144、管线只 padding 不 truncation，
    模型能吃多长就吃多长；写长是有理由的（官方示例就是几百字中文详述）。
    """
    body = "，".join([f"第{i}个细节是红披风配银色肩甲" for i in range(60)])
    pos, _ = presets.build(body, None, "sprite")
    assert body in pos, "用户描述被改写了"
    assert pos.endswith(presets.SPRITE_SUFFIX), "契约后缀要留在末尾"
    assert len(pos) > len(body)


def test_long_negative_is_delivered_whole():
    """负向词同理：不截。它只在 cfg>1 时才会发出去，用户写长也不该被砍。"""
    negs = ", ".join(f"bad thing {i}" for i in range(50))
    _, neg = presets.build("x", negs, "sprite")
    assert negs in neg


def test_suffix_is_kept_short():
    """后缀是**系统写的**那句，要保持短：堆长了会盖过用户自己写的内容。"""
    for name in ("sprite", "pixel", "portrait", "wuxia", "anime"):
        suffix, _ = presets.PRESETS[name]
        # 当前最长的是 sprite（28 词，含 FRAMING 的留白约束）。32 是上限，
        # 不是目标值——加词前先问自己"这句是不是产线硬要求"。
        assert len(suffix.split()) <= 32, (name, len(suffix.split()), suffix)


# ---------------- 不像素化时的契约（2026-11-02） ----------------
#
# 用户报「没勾像素化、只选透明背景，出图仍有像素感」。根子有两处，都在契约层：
#   1. sprite 档写着 "16-bit pixel art, hard outline, flat colors" —— 没勾像素化
#      却还挂着这个档，等于直接要求模型画像素画；
#   2. sprite 的负向词压 "anti-aliasing" —— 描边不能抗锯齿，边缘就是台阶状锯齿，
#      远看同样像低分辨率像素画。
# 所以 `pixelate=False` 时必须换成 `SPRITE_SUFFIX_PAINT` / `SPRITE_NEG_PAINT`。

def test_paint_contract_has_no_pixel_style_words():
    """不像素化：正向不得出现 pixel art / hard outline / flat colors。"""
    pos, _ = presets.build("韩式漫画风格的女剑士", None, "sprite", pixelate=False)
    lowered = pos.lower()
    for bad in ("pixel art", "hard outline", "flat colors", "16-bit"):
        assert bad not in lowered, f"不像素化却带了像素风格词 {bad!r}: {pos}"
    # 构图约束要留着：单主体 / 全身 / 居中 / 干净背景仍是产线要求
    assert "single character" in lowered and "full body" in lowered
    assert "centered" in lowered


def test_paint_contract_keeps_anti_aliasing_allowed():
    """不像素化：负向词不能压 anti-aliasing（插画要的就是抗锯齿边缘）。"""
    _, neg = presets.build("x", None, "sprite", pixelate=False)
    assert "anti-aliasing" not in neg.lower(), f"插画被压了抗锯齿: {neg}"


def test_pixelate_true_keeps_pixel_style_words():
    """像素化：原契约一个字都不能少（像素精灵就是要硬边、要压抗锯齿）。"""
    pos, neg = presets.build("x", None, "sprite", pixelate=True)
    assert pos.endswith(presets.SPRITE_SUFFIX)
    assert "16-bit pixel art" in pos
    assert "anti-aliasing" in neg


def test_paint_contract_leaves_other_presets_alone():
    """只有 sprite 档有像素风格词；anime/portrait/wuxia 本来就没有，不该被动。"""
    for name in ("anime", "portrait", "wuxia"):
        a = presets.build("x", None, name, pixelate=True)
        b = presets.build("x", None, name, pixelate=False)
        assert a == b, f"{name} 档被 pixelate 改动了：{a} != {b}"


def test_build_default_still_pixelates():
    """默认参数保持旧行为（pixelate=True）——老调用方不受影响。"""
    pos, _ = presets.build("x", None, "sprite")
    assert pos.endswith(presets.SPRITE_SUFFIX)
