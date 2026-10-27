"""提示词契约（preset）。

背景：SD1.5 裸提示词会自作主张加背景纹理、渐变、多余角色；只写 "pixel art" 得到的
是「看起来像像素」的糊图。这里把「产线要求」固化成后缀与负向词，用户只描述主体。

硬约束：SD1.5 的 CLIP 只有 77 token，超出部分会被**静默丢弃**——
所以正向后缀必须短，且把最重要的词放前面（背景干净 > 画风 > 构图）。
负向词同理，最脏的那几个词放最前。

用法：txt2img(prompt="blue slime monster", preset="sprite")
"""
from __future__ import annotations

# 正向后缀：约 28 token，留足余量给主体描述
SPRITE_SUFFIX = (
    "single character, full body, centered, flat plain background, "
    "16-bit pixel art, hard outline, flat colors"
)

# 负向词：按「污染程度」排序，前面被保留的概率最高
SPRITE_NEG = (
    # 最脏的放最前：格纹背景是像素化之后最刺眼的败笔
    "checkerboard, grid, lattice, tile pattern, pattern, texture, "
    "text, letters, chinese characters, sign, watermark, "
    "gradient, background scenery, multiple characters, "
    "photo, 3d render, blurry, anti-aliasing, extra limbs, cropped"
)

PORTRAIT_SUFFIX = "single character, full body, centered, plain background, clean lineart"
PORTRAIT_NEG = (
    "background scenery, multiple characters, text, watermark, "
    "blurry, photo, 3d render, cropped, deformed"
)

PRESETS = {
    "sprite": (SPRITE_SUFFIX, SPRITE_NEG),
    "portrait": (PORTRAIT_SUFFIX, PORTRAIT_NEG),
    "none": ("", ""),
}


def rough_tokens(text: str) -> int:
    """粗估 token 数：英文按 1.35 token/词，中文按 1 字 1.6 token。"""
    words = len(text.split())
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    return int(words * 1.35 + cjk * 1.6)


def build(prompt: str, negative: str | None, preset: str = "sprite") -> tuple[str, str]:
    if preset not in PRESETS:
        raise ValueError(f"未知 preset: {preset}（可选 {', '.join(PRESETS)}）")
    suffix, neg = PRESETS[preset]
    pos = f"{prompt.strip()}, {suffix}" if suffix else prompt
    negs = ", ".join(x for x in (neg, (negative or "").strip()) if x)
    return pos, negs
