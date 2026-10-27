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

# 武侠 / 国风立绘。刻意**不写 pixel art**：实测武侠类 LoRA 权重会压过风格词，
# 强行写 pixel art 只会让画面在「插画」和「像素」之间两头不到位。
# 像素化交给后处理（pixelate + palette），比让 SD 直接画像素稳得多。
WUXIA_SUFFIX = (
    # 实测对比：加 "feet visible, standing" 会让人物缩到画面中间、衣纹糊成一团；
    # 加 "clean lineart" 才能让宽袍的衣褶保住。裁切问题交给 edge_touch 报告，
    # 不靠提示词祈祷。
    "full body head to toe, centered, plain white background, "
    "chinese ink painting, clean lineart"
)
# 顺序即优先级：CLIP 只有 77 token，实测这串写到 113 token 尾部全被静默丢弃。
# 所以只留最关键的 13 段，宁少不滥。
WUXIA_NEG = (
    # 1 裁切（最致命：头脚出画面就不能当立绘）
    "cropped, cut off head, cut off feet, "
    # 2 印章题字（武侠 LoRA 的通病）
    "signature, seal, text, watermark, "
    # 3 水墨纸纹（抠图抠不净的主因）
    "ink splatter, paper texture, background scenery, "
    # 4 质量兜底
    "photo, blurry, deformed"
)

# 像素 LoRA 专用：**不要**再写 "pixel art"——LoRA 已经给了风格，重复要求会让画面
# 在「插画」和「像素」之间打架（实测 px_illustrious 套 sprite 契约会出彩色怪物）。
# 这里只约束构图与背景，像素化交给后处理（pixel=4 + palette=24）。
PIXEL_SUFFIX = "full body, standing, centered, plain white background, single character"
PIXEL_NEG = (
    "multiple characters, background scenery, grid, frame, border, text, watermark, "
    "photo, blurry, deformed, cropped"
)

PRESETS = {
    "sprite": (SPRITE_SUFFIX, SPRITE_NEG),
    "pixel": (PIXEL_SUFFIX, PIXEL_NEG),
    "portrait": (PORTRAIT_SUFFIX, PORTRAIT_NEG),
    "wuxia": (WUXIA_SUFFIX, WUXIA_NEG),
    "none": ("", ""),
}


def rough_tokens(text: str) -> int:
    """粗估 token 数：英文按 1.35 token/词，中文按 1 字 1.6 token。"""
    words = len(text.split())
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    return int(words * 1.35 + cjk * 1.6)


def fit(text: str, budget: int = 77) -> str:
    """按逗号段截断到 token 预算内。

    为什么不靠模型自己截：SD1.5 超出 77 token 是**静默丢弃**，不会报错。负向词
    往往写很长（实测武侠契约写到 113），末尾那几条等于没写。这里显式按段累加，
    保证「排在前面的重要词」一定进得去，并把被砍掉的部分暴露出来。
    """
    if rough_tokens(text) <= budget:
        return text
    out: list[str] = []
    total = 0
    for part in (x.strip() for x in text.split(",")):
        if not part:
            continue
        cost = rough_tokens(part) + 1                      # 逗号
        if total + cost > budget:
            break
        out.append(part)
        total += cost
    return ", ".join(out)


def build(prompt: str, negative: str | None, preset: str = "sprite") -> tuple[str, str]:
    if preset not in PRESETS:
        raise ValueError(f"未知 preset: {preset}（可选 {', '.join(PRESETS)}）")
    suffix, neg = PRESETS[preset]
    pos = f"{prompt.strip()}, {suffix}" if suffix else prompt
    negs = ", ".join(x for x in (neg, (negative or "").strip()) if x)
    return pos, fit(negs)
