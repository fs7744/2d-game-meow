"""换装（Outfit Change）与换脸（Face Swap）——都是「两图 + 结构化指令」。

## 为什么能这么做

Qwen-Image-2.1 的多图条件是**同一个 vision context**：两张图一起进模型，
由模型自己分配「谁是身份、谁是服装／谁是场景、谁是脸」。所以不需要任何
专用权重，只要把**指令写清楚每张图负责什么**。

管线会在 prompt 前面自动插 `<image1>...<imageN>` 占位符
（`QwenImage21Pipeline._get_qwen_prompt_embeds`），所以**提示词里可以直接
用 `<image1>` / `<image2>` 指代**——这比"第一张图""第二张图"这种自然语言
描述更稳，是官方支持的多图引用语法。

## 换装 vs 换脸的差别

| | 图 1（基底） | 图 2（参考） | 要保留 | 要替换 |
|---|---|---|---|---|
| 换装 | 人物照 | 服装 | 身份/姿势/背景 | 衣服的剪裁、颜色、材质 |
| 换脸 | 场景人物 | 身份肖像 | 姿势/服装/背景 | 脸、发型、胡须 |

两者的 prompt 结构其实一样，只是"保留什么/替换什么"反过来。

## 游戏素材价值

- **换装**：同一角色出不同装备/皮肤（商店套装、等级外观），游戏里高频需求
- **换脸**：同一套动作/立绘换主角脸（换主角、捏脸结果的落图）

## 实测（见 docs/PLAN-QWEN-ONLY.md §2.16）

768²/16 步约 14s；换装后服装特征来自图 2、人还是图 1 的；
换脸后脸像图 2、姿势与场景是图 1 的。
"""

from __future__ import annotations


from PIL import Image

#: 换装：图 1 是人物，图 2 是服装
OUTFIT_PROMPT = (
    "Dress the character from <image1> in the outfit from <image2>. "
    "Keep the identity, face, hairstyle, body pose and background of <image1> "
    "unchanged. Match the garment's silhouette, color, material and details "
    "to <image2>. Output a single coherent image."
)

#: 换脸：图 1 是场景，图 2 是身份
FACE_PROMPT = (
    "Replace the head and face of the person in <image1> with the person from "
    "<image2>. Keep the pose, body, clothing, camera framing and background of "
    "<image1> unchanged. Transfer the facial features, hairstyle and facial hair "
    "from <image2>. Output a single coherent image."
)


def _ref_slots(n: int) -> str:
    """把「图 2 起」的槽位写成一句自然语言：`<image2>`、`<image2> and <image3>`…"""
    tags = [f"<image{i}>" for i in range(2, n + 2)]
    if len(tags) == 1:
        return tags[0]
    return ", ".join(tags[:-1]) + " and " + tags[-1]


def multi_swap_prompt(kind: str, n_refs: int) -> str:
    """多张参考图时的指令。

    为什么要单独写而不是简单拼：多给几张不说明白，模型会以为要**换几张不同的
    衣服**（实测会出现上下身拼装）。所以显式说"这些是同一件服装/同一个人的
    不同角度，请合成为一个"。
    """
    slots = _ref_slots(n_refs)
    if kind == "outfit":
        return (
            f"Dress the character from <image1> in the outfit shown in {slots}. "
            f"All of {slots} are the SAME garment seen from different angles or "
            f"at different levels of detail — combine them into one consistent "
            f"outfit, do not blend several different costumes. "
            f"Keep the identity, face, hairstyle, body pose and background of "
            f"<image1> unchanged. Output a single coherent image."
        )
    return (
        f"Replace the head and face of the person in <image1> with the person "
        f"shown in {slots}. All of {slots} are the SAME person from different "
        f"angles — combine them into one consistent face, do not blend several "
        f"different people. Keep the pose, body, clothing, camera framing and "
        f"background of <image1> unchanged. Output a single coherent image."
    )


def swap_prompt(kind: str, extra: str = "", n_refs: int = 1) -> str:
    """组指令：内置的"谁管什么"模板 + 用户的补充要求。

    内置部分不要删——它是这个功能能用的原因。用户的补充接在后面，
    用来加细节（比如"外套改成皮夹克"）。

    `n_refs` 是参考图张数（不含基底图）。为 1 时用最初的双图模板，
    大于 1 时换成多图模板（显式声明"同一件服装/同一个人"）。
    """
    if n_refs > 1:
        base = multi_swap_prompt(kind, n_refs)
    else:
        base = OUTFIT_PROMPT if kind == "outfit" else FACE_PROMPT
    extra = (extra or "").strip()
    return f"{base} {extra}" if extra else base


def fit_pair(base: Image.Image, ref: Image.Image,
             size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    """两张参考图都缩到目标尺寸。

    Qwen 的条件图会被 `output_resolution` 统一编码，但**源图尺寸不一致**
    会让模型对"哪张是主体"产生混淆（尤其一张竖一张横时）。统一尺寸更稳。
    """
    return (base.convert("RGBA").resize(size, Image.LANCZOS),
            ref.convert("RGBA").resize(size, Image.LANCZOS))


def describe(kind: str) -> str:
    """给前端/日志用的一句话说明。"""
    return ("换装：把图 2 的服装穿到图 1 的人物身上，保留其身份与姿势"
            if kind == "outfit"
            else "换脸：把图 2 的脸/发型换到图 1 的人物上，保留其姿势与场景")
