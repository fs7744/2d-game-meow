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
# 留白词：**效果没有被证明，保留是因为没有可观测的副作用**。
# 两次各 6 张的对照里，贴边率是 50% 与 67%（基线 67%）——差异落在噪声范围内。
# 同期也试过 "feet visible, standing"（WUXIA 注释里记过它会让人物缩到中间），
# 贴边仍是 67%、多主体反而升到 33%，同样无效。
# 也就是说：**SD1.5 base 的裁切不是提示词能救的**，方向在换基座和候选+闸门。
FRAMING = "small figure in frame, ample empty space around, margin at top and bottom"

SPRITE_SUFFIX = (
    "single character, full body, centered, flat plain background, "
    f"16-bit pixel art, hard outline, flat colors, {FRAMING}"
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
# 顺序即优先级：越靠前被保留的概率越高。
# **注意**：这里曾经写着"实测这串写到 113 token 尾部全被丢弃"——那是 `rough_tokens`
# 的估算值。用真实 CLIPTokenizer 数出来是 **35 token**，根本没超 77。真正被搞坏的
# 不是"超长被丢"，而是"按高估值提前截断"：估算器把 35 当成 113，于是 `fit()` 在
# 77 的预算里只留下了前面几条。现在 `fit()` 用真实 BPE 计数（见 presets.count）。
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
PIXEL_SUFFIX = (
    f"full body, standing, centered, plain white background, single character, "
    f"{FRAMING}"
)
PIXEL_NEG = (
    "multiple characters, background scenery, grid, frame, border, text, watermark, "
    "photo, blurry, deformed, cropped"
)

# ---- 动漫（anime）----
#
# 面向 **SDXL 动漫底模**（Illustrious / Animagine）。业界对这些模型的共识是：
#
#   · 用 **danbooru tag**（1girl / solo / long hair / blue eyes…），不是自然语言。
#     Illustrious 是在 Danbooru 上重训的，tag 到概念的映射很准，同义改写反而变差。
#   · **质量词只要两个**——`masterpiece, best quality`。堆更多（amazing quality /
#     absurdres / very aesthetic…）不会更好，实测会引入偏差与伪影。
#   · **负向词要短**。写一大串"通用负面词"在这些模型上是负优化：它们已经把
#     低质量样本排除了，再堆词只会压制正常特征。
#   · **clip_skip=2** 是硬性设置（见 generate._text_kwargs，SDXL 现已支持）。
#   · CFG 5~7（v0.1 用 6.5）、步数 20~28、1024²。
#
# 与 sprite/pixel 契约的区别：那两个是**给像素精灵**的（扁平背景 + 硬描边）；
# 这里要的是**动漫插画质感**——赛璐璐/厚涂都行，但人物结构必须正确。
# 背景仍要求简单，因为游戏素材要抠图（matte）。
ANIME_SUFFIX = (
    "masterpiece, best quality, "
    # 风格必须显式给——Illustrious v0.1 官方就注明「the model is not
    # aesthetically pre-tuned」。实测三档对比：
    #   不给风格词      → 偏写实厚涂，不是日系动漫
    #   flat colors+cel shading → **赛璐璐平涂，全身也最完整**
    #   anime style+clean lineart → 线条最锐利
    # 两个都给，实测能同时拿到"色块分明 + 线条干净"。
    "anime, flat colors, cel shading, clean lineart, "
    # 背景要简单：游戏素材下游要抠图（matte）
    "simple background, plain background, centered"
)
# 顺序即优先级。**只写这几条**——动漫底模堆负面词是负优化。
ANIME_NEG = (
    "worst quality, low quality, "
    "bad anatomy, bad hands, extra digits, "
    "text, watermark, signature, "
    "multiple views, blurry"
)

PRESETS = {
    "sprite": (SPRITE_SUFFIX, SPRITE_NEG),
    "pixel": (PIXEL_SUFFIX, PIXEL_NEG),
    "portrait": (PORTRAIT_SUFFIX, PORTRAIT_NEG),
    "wuxia": (WUXIA_SUFFIX, WUXIA_NEG),
    "anime": (ANIME_SUFFIX, ANIME_NEG),
    "none": ("", ""),
}


def rough_tokens(text: str) -> int:
    """粗估 token 数：英文按 1.35 token/词，中文按 1 字 1.6 token。

    **只作为兜底**。它对逗号分隔的英文串高估约 3 倍（实测 WUXIA_NEG 估算 113、
    真实 35），据此截断会把本来塞得下的词白白砍掉。有 tokenizer 时一律走
    `count`（真实 BPE 计数），估算法只在没下权重/没装 transformers 时用。
    """
    words = len(text.split())
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    return int(words * 1.35 + cjk * 1.6)


def count(text: str) -> int:
    """token 数；拿不到 tokenizer 时退回启发式（宁可粗也不能崩）。

    SD 时代这里走 `workflows.clip` 的真实 BPE 计数。**那个模块已经删了**：
    Qwen 的文本编码器是 Qwen3-VL，CLIP 的 77 token 预算对它没有意义，
    契约后缀的作用也从「别被截断」变成「给模型一句构图约束」。
    所以现在恒走启发式——保留这个函数是因为 `build(fit=...)` 还要用它。
    """
    return rough_tokens(text)


def _cut_words(text: str, budget: int) -> str:
    """整段都塞不下时按词截断。没有逗号的超长提示词不能直接丢成空串。"""
    keep: list[str] = []
    used = 0
    for w in text.split():
        cost = count(w) + 1
        if used + cost > budget:
            break
        keep.append(w)
        used += cost
    return " ".join(keep)


def fit(text: str, budget: int = 77) -> str:
    """按逗号段截断到 token 预算内。

    为什么不靠模型自己截：SD1.5 超出 77 token 是**静默丢弃**，不会报错。负向词
    往往写很长，末尾那几条等于没写。这里显式按段累加，保证「排在前面的重要词」
    一定进得去，并把被砍掉的部分暴露出来。

    两个边界情况必须处理，否则「截断」会比不截断更糟：
    - 只有一段却没有逗号（用户直接粘了一长串）→ 按词截，不能返回空串；
    - 第一段就超预算 → 同上，至少要留下开头那几个词。
    """
    if count(text) <= budget:
        return text
    out: list[str] = []
    total = 0
    for part in (x.strip() for x in text.split(",")):
        if not part:
            continue
        cost = count(part) + 1                              # 逗号
        if total + cost > budget:
            if not out:                                    # 一句都没进去：退化成按词截
                tail = _cut_words(part, budget)
                if tail:
                    out.append(tail)
            break
        out.append(part)
        total += cost
    return ", ".join(out)


# 分段编码能塞下的总预算（77 × 2）。见 workflows/clip.py 的 encode_long：
# 超过 77 token 的部分不再被丢弃，而是切成第二段参与交叉注意力。
CHUNK_BUDGET = 154


def build(prompt: str, negative: str | None, preset: str = "sprite",
          budget: int = 77, max_tokens: int = 0) -> tuple[str, str]:
    """用户描述 + 契约后缀，两侧都在 token 预算内。

    **预算先留给后缀**：后缀（扁平背景 / 硬描边 / 全身）才是产线契约，用户主体
    描述写超了应该截主体，而不是把契约末尾丢掉——老实现是直接 `prompt + suffix`
    交给 CLIP，超长时被静默丢弃的恰好是后缀，画出来背景就脏了。

    `max_tokens`：调大预算，配合 `clip.embeds` 的分段编码使用（见 CHUNK_BUDGET）。
    默认 0 表示维持 77 的硬上限，老行为逐字不变。
    """
    if preset not in PRESETS:
        raise ValueError(f"未知 preset: {preset}（可选 {', '.join(PRESETS)}）")
    if max_tokens and max_tokens > budget:
        budget = int(max_tokens)
    suffix, neg = PRESETS[preset]
    user = (prompt or "").strip().strip(",")
    if suffix:
        room = max(8, budget - count(suffix) - 1)
        user = fit(user, room)
        pos = f"{user}, {suffix}" if user else suffix
    else:
        pos = fit(user, budget)
    negs = ", ".join(x for x in (neg, (negative or "").strip()) if x)
    return pos, fit(negs, budget)
