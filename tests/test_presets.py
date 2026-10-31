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
