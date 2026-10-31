"""提示词契约（preset）。

背景：裸提示词会自作主张加背景纹理、渐变、多余角色；只写 "pixel art" 得到的
是「看起来像像素」的糊图。这里把「产线要求」固化成后缀与负向词，用户只描述主体。

**长度：不截断，用户写多少给多少。**
SD 时代 CLIP 只有 77 token、超出部分**静默丢弃**，所以那时必须把后缀压到极短、
还得按预算砍用户描述。现在底模是 Qwen-Image（编码器 Qwen3-VL，tokenizer 的
`model_max_length` 是 262144，diffusers 的 `encode_prompt()` 只 padding、
不 truncation），**模型侧没这个限制**，`build()` 也就一行都不砍。

剩下的只是成本问题：prompt 越长，DiT 的 joint attention 序列越长、越慢越吃显存。
这个取舍交给用户（见 `build()` 的 docstring）。

**语言**：后缀是英文常量；用户正文**中英文都行**（Qwen 的编码器支持中文）。
两者拼一起时模型能分清哪句是契约——这也是后缀保持英文的一个理由。

用法：txt2img(prompt="蓝色史莱姆怪物", preset="sprite")
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
# 顺序即优先级：越脏的词越靠前（万一以后又要按预算取舍，先保这些）。
#
# 这里记着一段教训：曾经的 `rough_tokens` 估算器把这条 **35 token** 的串估成
# 113，于是按预算截断时只留下前几条——**估算偏差比"超长"本身更伤**。
# 现在全程不截断（见 build 的 docstring），这段留作"别再加回估算式截断"的提醒。
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


def build(prompt: str, negative: str | None,
          preset: str = "sprite") -> tuple[str, str]:
    """用户描述 + 契约后缀。**一个字都不截**。

    **为什么不截**：截断是我们的判断，不是模型的限制。Qwen-Image 的编码器是
    Qwen3-VL，tokenizer 的 `model_max_length` 是 262144，diffusers 的
    `pipeline_qwenimage21.encode_prompt()` 只做 `padding=True`、**没有
    truncation**——模型能吃多长就吃多长。SD 时代那套「77 token 必须砍」的约束
    （CLIP 超长静默丢弃）随管线一起作废了。

    代价是**越长的提示词越慢**：prompt token 会进 DiT 的 joint attention 序列，
    和图像 token 一起算注意力。但这个代价该由用户自己权衡——他写 300 字是有
    理由的（Qwen 官方示例本身就是几百字的中文详述），不该被我们悄悄砍掉一半。
    界面上只提示"越长越慢"，不替他做减法。

    顺序上后缀仍放最后：用户写的描述在前、系统补的契约在后，读起来清楚
    "哪些是自己写的"；模型对末尾约束通常也更敏感。
    """
    if preset not in PRESETS:
        raise ValueError(f"未知 preset: {preset}（可选 {', '.join(PRESETS)}）")
    suffix, neg = PRESETS[preset]
    user = (prompt or "").strip().strip(",")
    if user and suffix:
        pos = f"{user}, {suffix}"
    else:
        pos = user or suffix
    negs = ", ".join(x for x in (neg, (negative or "").strip()) if x)
    return pos, negs
