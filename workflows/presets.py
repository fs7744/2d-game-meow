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


def _cut_words(text: str, budget: int) -> str:
    """整段都塞不下时按词截断。没有逗号的超长提示词不能直接丢成空串。"""
    keep: list[str] = []
    used = 0
    for w in text.split():
        cost = rough_tokens(w) + 1
        if used + cost > budget:
            break
        keep.append(w)
        used += cost
    return " ".join(keep)


def fit(text: str, budget: int = 77) -> str:
    """按逗号段截断到 token 预算内。

    为什么不靠模型自己截：SD1.5 超出 77 token 是**静默丢弃**，不会报错。负向词
    往往写很长（实测武侠契约写到 113），末尾那几条等于没写。这里显式按段累加，
    保证「排在前面的重要词」一定进得去，并把被砍掉的部分暴露出来。

    两个边界情况必须处理，否则「截断」会比不截断更糟：
    - 只有一段却没有逗号（用户直接粘了一长串）→ 按词截，不能返回空串；
    - 第一段就超预算 → 同上，至少要留下开头那几个词。
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
            if not out:                                    # 一句都没进去：退化成按词截
                tail = _cut_words(part, budget)
                if tail:
                    out.append(tail)
            break
        out.append(part)
        total += cost
    return ", ".join(out)


def build(prompt: str, negative: str | None, preset: str = "sprite",
          budget: int = 77) -> tuple[str, str]:
    """用户描述 + 契约后缀，两侧都在 77 token 预算内。

    **预算先留给后缀**：后缀（扁平背景 / 硬描边 / 全身）才是产线契约，用户主体
    描述写超了应该截主体，而不是把契约末尾丢掉——老实现是直接 `prompt + suffix`
    交给 CLIP，超长时被静默丢弃的恰好是后缀，画出来背景就脏了。
    """
    if preset not in PRESETS:
        raise ValueError(f"未知 preset: {preset}（可选 {', '.join(PRESETS)}）")
    suffix, neg = PRESETS[preset]
    user = (prompt or "").strip().strip(",")
    if suffix:
        room = max(8, budget - rough_tokens(suffix) - 1)
        user = fit(user, room)
        pos = f"{user}, {suffix}" if user else suffix
    else:
        pos = fit(user, budget)
    negs = ", ".join(x for x in (neg, (negative or "").strip()) if x)
    return pos, fit(negs, budget)
