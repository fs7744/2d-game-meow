"""角色中心的出图参数与提示词组装（移植自 VNCCS 的 QI2 路径）。

**"以官方为准"具体落在哪**：造角色的 prompt 不是随便一句描述，官方是**结构化字段**
拼出来的（`character_creator_v2.py:1124` `_structured_character_prompt`），
每个字段一行 `Key: value`。这套写法是他们调过 LoRA 之后定下来的——自己另写一句
自然语言，风格 LoRA 就吃不准。所以这里逐字照搬：

| 这里 | 官方 |
|---|---|
| `QI2_COMPOSITION_PROMPT` | `character_creator_v2.py:263` |
| `QI2_NATURAL_FRAMING` | `:192`（portrait / cowboy_shot / full_body） |
| `QI2_ALPHA_BACKGROUND_PROMPT` | `:175` |
| `QI2_STYLE_REFERENCE_HEADING` | `:176` |
| `qi2_fields()` | `:1101` `_qi2_character_fields` |
| `structured_prompt()` | `:1124` `_structured_character_prompt` |
| `generation_resolution()` | `:977` `get_generation_resolution` |
| `QI2_DEFAULTS` | `vnccs_control_center.py:2073`（steps 25 / cfg 3.0 / euler / simple） |

**没有移植的两块**（都要别的模型，见 docs/角色中心-VNCCS移植评估.md §4.1）：
- `_qi2_prompt_rewriter.txt` 的字段扩写（要 LLM）；
- Prompt Enhancement 后的 `natural_prompt`。
所以这里产出的是**未扩写**的结构化 prompt，字段值原样保留。等 gguf 到位，
在 `structured_prompt()` 之后接一层扩写即可，接口不用变。
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from workflows import paths
from workflows import qwen_text
from workflows import textlabels

log = logging.getLogger(__name__)

#: 官方：`character_creator_v2.py:263`
QI2_COMPOSITION_PROMPT = (
    "Create one image of exactly one character in a single continuous view. "
    "The character appears once and occupies most of the image height. "
    "No character sheet, collage, panels, insets, separate body-part studies "
    "or additional views."
)
#: 官方：`:175`
QI2_ALPHA_BACKGROUND_PROMPT = "transparent background with alpha channel"
#: 官方：`:176`
QI2_STYLE_REFERENCE_HEADING = "Visual style reference (rendering only):"
#: 官方：`:192`（只搬了 QI2 用的三个取景；portrait 走 image_type，其余走 framing）
QI2_NATURAL_FRAMING = {
    "portrait": (
        "Portrait: a close-up of the character's head and shoulders. "
        "Keep the complete head and hair visible and make the face large in the image. "
        "Crop just below the shoulders; the waist, hips and legs stay outside the image."
    ),
    "cowboy_shot": (
        "Cowboy Shot (cowboy_shot): use a tight head-to-upper-thigh crop. "
        "Keep the complete head and hair visible, with the top of the hair just below "
        "the top image edge. Fill the entire image height with this cropped view of "
        "the character. The bottom image edge must cut across the upper thighs, "
        "just below the hips, so the character fills the frame. "
        "Do not zoom out to fit the complete head-to-toe figure or leave empty space "
        "below the crop."
    ),
    "full_body": (
        "full-length view: show the character's complete body from head to toe, "
        "including both feet and shoes. The whole figure fills the image height, "
        "head near the top edge and feet near the bottom edge. Do not crop at the "
        "knees, thighs, waist or ankles, and do not turn it into a bust or "
        "half-body shot."
    ),
}
#: 官方：`:364-366`（latent 9:16，步长 8）
LATENT_ASPECT_WIDTH = 9
LATENT_ASPECT_HEIGHT = 16
LATENT_DIMENSION_STEP = 8

# ---------------------------------------------------------------- 中文提示词
#
# Qwen-Image 2.1 的文本编码器就是 Qwen3-VL，**中文是原生能力**（官方模型卡也
# 把中英双语当卖点）。官方 VNCCS 模板全英文，是因为它要喂 ComfyUI 的 CLIP、
# 而且风格库/标签库本身就是英文 trait；我们是同一套权重，没有理由不让人用中文。
#
# 做法：结构词（取景/背景/字段名）中文化，属性值走译表；画风段整段引用画风库
# 自己的视觉定义原文（见 style_block）——它是画风数据，不是结构词。
# Prompt Enhancement 中文模式照跑：qwen_text 有中文 system prompt。

QI2_COMPOSITION_PROMPT_ZH = (
    "只画一个角色，单一连续视角；角色只出现一次，并占据画面的大部分高度。"
    "不要角色设定集、拼贴、分格、局部特写或额外的多视角。"
)
QI2_NATURAL_FRAMING_ZH = {
    "portrait": (
        "头像：从头顶到肩部的近景。完整露出头部与头发，脸在画面里尽量大；"
        "肩膀以下（腰、胯、腿）不入画。"
    ),
    "cowboy_shot": (
        "半身：头顶到大腿上部。完整露出头部与头发，头顶贴近画面上边缘；"
        "下边缘压在大腿上部（胯部略下）。不要缩到全身，下方不要留空。"
    ),
    "full_body": (
        "全身：从头到脚完整入画，包括双脚与鞋子。人物占满画面高度，"
        "头顶贴近上边缘、脚贴近下边缘；不要在膝盖、大腿、腰或脚踝处裁切，"
        "不要画成半身或胸像。"
    ),
}
#: 出图底色 → 中文说法（值是官方 background_color，原样英文落盘）
BG_COLOR_ZH = {
    "green": "绿", "blue": "蓝", "white": "白", "grey": "灰", "gray": "灰",
    "magenta": "品红",
}
#: 性别值（官方 sex 只有 female / male）→ 中文
_SEX_ZH = {"female": "女", "male": "男"}
#: 画质词与默认服装：标签库不收录这些整段词，单独给几条常用的
_AESTHETICS_ZH = {
    "masterpiece": "杰作",
    "best quality": "最佳质量",
    "high quality": "高质量",
    "simple outfit": "简约服装",
}

#: 官方 QI2 采样默认（`vnccs_control_center.py:2073`、`:117`）
QI2_DEFAULTS = {
    "steps": 25,
    "cfg": 3.0,
    "sampler": "euler",
    "scheduler": "simple",
    "target_size": 1024,
}

_STYLE_CACHE: Optional[Dict[str, Any]] = None
_PRESET_CACHE: Optional[Dict[str, Any]] = None


def _preset_key(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").lower().split())


def _load_json(name: str, cache: str) -> Optional[Dict[str, Any]]:
    """读随项目发行的官方数据（web/public/vnccs/character_template/）。

    放在那里是为了**前端也能直接 fetch 同一份**（风格选择器要用），
    不复制第二份——复制就会有天对不上。
    """
    path = os.path.join(str(paths.VNCCS_TEMPLATE_DIR), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def style_catalog() -> Dict[str, Any]:
    global _STYLE_CACHE
    if _STYLE_CACHE is None:
        _STYLE_CACHE = _load_json("character_styles.json", "styles") or {}
    return _STYLE_CACHE


def preset_catalog() -> Dict[str, Any]:
    global _PRESET_CACHE
    if _PRESET_CACHE is None:
        _PRESET_CACHE = _load_json("character_presets_v2.json", "presets") or {}
    return _PRESET_CACHE


def style_prompt_of(style_id: str) -> str:
    """风格 id → 官方那条三段式 prompt（`Style: / Reference: / Visual description:`）。"""
    if not style_id:
        return ""
    for group in style_catalog().get("groups") or []:
        for item in group.get("styles") or []:
            if item.get("id") == style_id:
                return str(item.get("prompt") or "")
    return ""


def race_features(value: str) -> str:
    """种族 → 官方的物种特征提示（`nodes/character_presets.py:26` `race_features`）。

    官方的做法是**不改动用户写的字段**，只在旁边补一句"典型物种特征"，
    因为用户手写的特征要能覆盖默认。
    """
    source = str(value or "").strip()
    presets: Dict[str, Any] = {}
    for preset in ((preset_catalog().get("tags") or {}).get("races") or []):
        for alias in [preset.get("tag"), preset.get("label"),
                      *(preset.get("synonyms") or [])]:
            presets[_preset_key(alias)] = preset
    descriptions: List[str] = []
    seen = set()
    for token in source.split(","):
        token = token.strip().strip("()[]")
        token = re.sub(r":\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*$", "", token)
        preset = presets.get(_preset_key(token))
        if preset and preset.get("tag") not in seen:
            seen.add(preset.get("tag"))
            descriptions.append(f'{preset.get("label")}: {preset.get("prompt")}')
    if not descriptions:
        return ""
    return ("Typical species features (explicit character traits override these "
            "defaults): " + " ".join(descriptions))


def _framing_key(info: Dict[str, Any]) -> str:
    """官方：`character_creator_v2.py:1057` `_character_framing_key`

    ⚠️ **默认值跟官方不同**：官方没写 framing 时按 `cowboy_shot`（半身到大腿），
    那是给「角色卡/头像」用的。我们做的是游戏素材，立绘要能切精灵图，
    半身图切出来脚都没有。所以这里**默认 full_body**，只有用户显式选了
    portrait / cowboy_shot 才降级。
    """
    if str(info.get("image_type", "")).strip().lower() == "portrait":
        return "portrait"
    key = str(info.get("framing", "") or "").strip().lower()
    if key in ("portrait", "头像"):
        return "portrait"
    if key in ("cowboy_shot", "cowboy", "half_body", "半身"):
        return "cowboy_shot"
    return "full_body"


def _clothing_prompt(info: Dict[str, Any]) -> str:
    """衣服：角色自己的 additional 里没写就用默认的一套（官方 `_character_clothing_prompt`）。"""
    for key in ("clothing", "costume", "outfit"):
        value = str(info.get(key, "") or "").strip()
        if value:
            return value
    return "simple outfit"


def prompt_language(info: Dict[str, Any]) -> str:
    """出图提示词的语言：`zh` / `en`（**默认中文**）。

    存在 `character_info.prompt_language` 里，跟着角色走（不是全局设置）——
    同一个库里可以有中文角色和英文角色；没写过的老角色按中文处理。
    """
    value = str((info or {}).get("prompt_language") or "").strip().lower()
    return "en" if value in ("en", "en-us", "english") else "zh"


def pe_eligible(info: Dict[str, Any], enhance: bool) -> bool:
    """要不要跑 Prompt Enhancement —— **只看用户有没有点**。

    以前这里还按语言拦一道（中文模式跳过），现在中文有自己的 system prompt，
    改写器直接输出中文描述，不需要再绕。
    """
    return bool(enhance)


def qi2_fields(info: Dict[str, Any]) -> Dict[str, str]:
    """character_info → QI2 结构化字段（官方 `character_creator_v2.py:1101`）。"""
    background = str(info.get("background_color", "") or "").strip()
    framing_key = _framing_key(info)
    zh = prompt_language(info) == "zh"
    if zh:
        bg = ("透明背景（带 alpha 通道）" if background.lower() == "transparent"
              else (f"纯{BG_COLOR_ZH.get(background.lower(), background)}色背景"
                    if background else "简单背景"))
    else:
        bg = (QI2_ALPHA_BACKGROUND_PROMPT if background.lower() == "transparent"
              else (f"solid {background} background" if background else "simple background"))
    framing = ((QI2_NATURAL_FRAMING_ZH if zh else QI2_NATURAL_FRAMING)[framing_key])
    traits = {
        key: str(info.get(key, "") or "")
        # breast_size 官方 `_qi2_character_fields` 漏了，于是结构化 prompt 里
        # `Breast size:` 永远是空行（官方 bug，我们补上）。
        for key in ("race", "skin_color", "body", "face", "hair", "eyes",
                    "breast_size", "additional_details")
    }
    clothing = _clothing_prompt(info)
    aesthetics = str(info.get("aesthetics", "masterpiece") or "")
    if zh:
        # 中文模式：属性值也译成中文（`silver hair` → 银发）。译表来自
        # `web/src/labels.ts`（见 tools/build_labels_zh.py），查不到就保留原值。
        traits = {k: textlabels.translate_traits(v) for k, v in traits.items()}
        clothing = _AESTHETICS_ZH.get(clothing, clothing)
        aesthetics = _AESTHETICS_ZH.get(aesthetics, aesthetics)
    fields = {
        "gender": str(info.get("sex", "female") or "female"),
        "age": (f"{int(info.get('age', 18) or 18)} 岁" if zh
                else f"{int(info.get('age', 18) or 18)} years old"),
        **traits,
        # 物种特征那句是官方英文数据（"Elf: pointed ears..."），中文模式不拼，
        # 免得整段提示词里突然蹦一句英文；用户自己的标签已经带了这些特征。
        # 官方物种特征那句（"Elf: pointed ears…"）以前中文模式直接置空，
        # 理由是"免得蹦一句英文"，但用户只填了「精灵」一个词时，尖耳朵这些
        # 视觉特征就真的没了。跟画风段同理：**官方数据原文照搬**，标签中文化，
        # 信息完整比"整段纯中文"重要。
        "race_features": race_features(info.get("race", "")),
        "custom_style": str(info.get("custom_style") or ""),
        "clothing": clothing,
        "expression": "面无表情" if zh else "expressionless",
        "framing": (f"单角色；{framing}" if zh else f"single character; {framing}"),
        "background": bg,
        "aesthetics": aesthetics,
        "lora_prompt": str(info.get("lora_prompt", "") or ""),
    }
    return {k: str(v or "").strip() for k, v in fields.items()}


def _split_style(style_reference: str) -> Tuple[str, str, str]:
    """官方三段式 → (画风名, 参考作品, 视觉描述)。对不上格式就整段当描述。"""
    text = str(style_reference or "").strip()
    lines = text.splitlines()
    if (len(lines) == 3 and lines[0].startswith("Style: ")
            and lines[1].startswith("Reference: ")
            and lines[2].startswith("Visual description: ")):
        return (lines[0][len("Style: "):], lines[1][len("Reference: "):],
                lines[2][len("Visual description: "):])
    if len(lines) > 1 and lines[0].startswith("Style: "):
        return lines[0][len("Style: "):], "", " ".join(lines[1:])
    return "", "", text


def style_block(style_reference: str, lang: str = "zh") -> str:
    """**画风段**：内容只来自画风库自己那一条定义，公共段一个风格字都不掺。

    以前中文模式只留一句「画风：Q 版动漫」，官方那段 Visual description
    被整段丢掉 —— Q 版最要紧的「大头压倒小身体、圆润线条、夸张大眼」
    全没了，出图自然不像 Q 版；其它 357 条画风同样被削平成名字。

    现在把画风库自己的描述**原文照搬**：它是画风的**数据**，不是官方英文
    结构词（结构标签中文化即可）。Qwen3-VL 中英双语原生支持，而这些描述
    本来就是风格库配套用词，自己改写或硬译反而失真 —— 用户要的是
    「画风由自己影响」，那就得把它自己的定义完整交给模型。

    没选画风时返回空串：**公共段绝不替用户决定画风**。
    en 模式也返回空串（官方三段式由 structured_prompt 原样负责，不重复）。
    """
    if lang != "zh":
        return ""
    name, reference, description = _split_style(style_reference)
    name = name.strip().rstrip(".")
    description = str(description or "").strip()
    reference = str(reference or "").strip()
    if not name and not description:
        return ""
    label = textlabels.style_label(name) if name else ""
    if name:
        title = f"【画风】{label}" + (f"（{name}）" if label != name else "")
    else:
        title = "【画风】自定义描述"
    lines = [title,
             "下面这段是所选画风的完整视觉定义，严格按它作画，不要混入其他画风的特征："]
    if description:
        lines.append(description)
    if reference:
        lines.append(f"参考源流：{reference}")
    return "\n".join(lines)


def style_line_zh(style_reference: str) -> str:
    """兼容旧调用：中文模式现在给的是完整画风段，不再是一句画风名。"""
    return style_block(style_reference, "zh")


def structured_prompt(fields: Dict[str, str], style_reference: str = "",
                      lang: str = "en") -> str:
    """拼出官方那个结构化 prompt（官方 `character_creator_v2.py:1124`）。

    `style_reference` 是风格库里那条三段式文本，官方在这里把它拆成
    Style / Style reference / Style description 三行。

    `lang="zh"` 时**整段都是中文**：结构词、取景短文、属性值（走
    `workflows/textlabels.py` 的译表），连风格块也只留中文画风名 ——
    官方那段英文 Style/Reference/Visual description 是给英文提示词用的，
    直接塞进中文 prompt 只会让模型看到两种语言混排。
    """
    style, reference, description = _split_style(style_reference)

    # 属性行：**空的就不写**。官方会把 `Skin color: ` 这种空行也打出来，
    # 中文界面看着像缺了个词；胸围更是官方漏填的字段（见 qi2_fields 注释）。
    female = fields.get("gender") == "female"
    rows_zh = [
        ("性别", _SEX_ZH.get(fields.get("gender", ""), fields.get("gender", ""))),
        ("年龄", fields.get("age", "")),
        ("种族", fields.get("race", "")),
        ("物种特征", fields.get("race_features", "")),
        ("面部特征", ", ".join(v for v in (fields.get("face", ""),
                                          fields.get("expression", "")) if v)),
        ("头发", fields.get("hair", "")),
        ("眼睛", fields.get("eyes", "")),
        ("肤色", fields.get("skin_color", "")),
        ("胸围", fields.get("breast_size", "") if female else ""),
        ("体型", fields.get("body", "")),
        ("服装", fields.get("clothing", "")),
        ("附加细节", fields.get("additional_details", "")),
        ("LoRA 触发词", fields.get("lora_prompt", "")),
        ("自定义画风要求", fields.get("custom_style", "")),
    ]
    rows_en = [
        ("Gender", fields.get("gender", "")),
        ("Age", fields.get("age", "")),
        ("Race", ", ".join(v for v in (fields.get("race", ""),
                                      fields.get("race_features", "")) if v)),
        ("Facial features", ", ".join(v for v in (fields.get("face", ""),
                                                 fields.get("expression", "")) if v)),
        ("Hair", fields.get("hair", "")),
        ("Eyes", fields.get("eyes", "")),
        ("Skin color", fields.get("skin_color", "")),
        ("Breast size", fields.get("breast_size", "") if female else ""),
        ("Build", fields.get("body", "")),
        ("Clothes", fields.get("clothing", "")),
        ("Additional details", fields.get("additional_details", "")),
        ("LoRA triggers", fields.get("lora_prompt", "")),
        ("Custom style", fields.get("custom_style", "")),
    ]

    if lang == "zh":
        # **公共段不写任何风格词**：以前这里写死「画风：… or 默认日式动画风格」
        # 还有一句「按这个画风的笔触、轮廓、配色」——前者在没选画风/译表查不到
        # 时会强加一个日式动画风格，后者是泛化的风格指令，都对具体画风是干扰。
        # 画风全部交给末尾那个独立段落（style_block），它只引用画风库自己的定义。
        head = ["请画一张角色立绘。"]
        if fields.get("aesthetics"):
            head.append(f"画质要求：{fields['aesthetics']}")
        head.extend([
            "画面里只有一个角色。",
            f"取景：{QI2_COMPOSITION_PROMPT_ZH} {fields.get('framing', '')}",
            f"背景：{fields.get('background', '')}",
        ])
        block = style_block(style_reference, "zh")
        if block:
            # 放在**属性行之前**，跟官方「风格段在前」的顺序一致。放末尾会被
            # PE 的 LoRA 切行连带切掉（改写时就看不到画风定义了）；PE 成功后
            # 官方还会在末尾再追加一次风格参考，权重并不吃亏。
            head.extend(["", block, ""])
        head.extend(f"{label}：{value}" for label, value in rows_zh if value)
        return "\n".join(head)

    prompt = [
        f"Draw an image of a character in the following style: {style}",
        f"Style reference: {reference}",
        f"Style description: {description}",
        f"Additional style parameters: {fields.get('aesthetics', '')}",
        "The image contains one character.",
        f"Framing: {QI2_COMPOSITION_PROMPT} {fields.get('framing', '')}",
        f"Background: {fields.get('background', '')}",
    ]
    prompt.extend(f"{label}: {value}" for label, value in rows_en if value)
    return "\n".join(prompt)


def character_prompt(info: Dict[str, Any], style_id: str = "") -> str:
    """一条龙：character_info + 风格 id → 官方结构的 prompt（语言跟着角色走）。"""
    style = style_prompt_of(style_id or str(info.get("style", "") or ""))
    return structured_prompt(qi2_fields(info), style, prompt_language(info))


# ---------------------------------------------------------------- Prompt Enhancement
#
# 官方 `character_creator_v2.py:1271-1317` `generate_creator_qi2_prompt`。
# 用的是**图像模型自己的 Qwen3-VL 文本编码器**（workflows/qwen_text.py），
# 不需要任何新权重。整条流程逐句照搬，包括两个容易被漏掉的细节：
#   · 送进 PE 的 request 要**先切掉 `LoRA triggers:` 那一行**（:1285）；
#   · 风格参考是**在 PE 之后**再追加一次（:1315），不是只在结构化 prompt 里。


def _strip_unit_prompt_weights(value: str) -> str:
    """官方 `character_creator_v2.py:493`：去掉 `:1.0` 这种无效权重。"""
    return re.sub(r":\s*1\.0+\b", "", str(value or ""))


def _qi2_json_result(generated: str) -> Optional[Dict[str, Any]]:
    """官方 `character_creator_v2.py:1015`：把最终那段 JSON 抠出来。

    **必须解析**：官方随包的 t2i system prompt 结尾写着
    `Return one strictly valid JSON object ... {"rewritten_prompt": ..., "wh_ratio": ...}`
    —— 也就是说模型**按要求就该回 JSON**。以前我们见到 `{` 就判失败，
    等于每次 PE 都白跑（日志没有报错，前端只显示「未扩写」）。

    `<think>...</think>` 之前的内容整段丢掉；没有闭合的 think 直接算失败。
    """
    text = str(generated or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif "<think>" in text:
        return None
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            parsed, _end = decoder.raw_decode(text[start:])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and ("fields" in parsed or "rewritten_prompt" in parsed):
            return parsed
    return None


def _qi2_rewritten_prompt(generated: str, original: str) -> str:
    """官方 `:1034`：取 JSON 里的 `rewritten_prompt`，取不到再按纯文本兜。

    **这里跟官方有一处刻意不同**：官方是失败时返回 `original`（:1042），
    我们返回空串让调用方统一走"退回原文"。原因是拿返回值跟 request 比来判断
    "有没有改写成功"会有**假阳性**——`.strip()` 一去尾空格就不相等了
    （实测：`Additional details: ` 这种结尾带空格的行，比出来是"改写了"）。
    """
    parsed = _qi2_json_result(generated)
    if parsed is not None:
        rewritten = parsed.get("rewritten_prompt")
        if isinstance(rewritten, str) and rewritten.strip():
            return rewritten.strip()
    text = str(generated or "").strip()
    if not text or any(m in text for m in ("<think>", "{", "```")):
        return ""
    return text


def enhance_character_prompt(
    pipe: Any, info: Dict[str, Any], style_id: str = "",
    enhance: bool = True,
) -> Dict[str, str]:
    """造角色的完整 prompt：结构化 → PE → 补透明底 → 追加风格参考与 LoRA 触发词。

    返回 `{prompt, raw}`，`raw` 是未扩写版——**PE 失败时调用方要用 raw**，
    增强不能成为出图的必要条件。
    """
    style = style_prompt_of(style_id or str(info.get("style") or ""))
    lang = prompt_language(info)
    fields = {k: _strip_unit_prompt_weights(v) for k, v in qi2_fields(info).items()}
    original = structured_prompt(fields, style, lang)
    if not enhance or pipe is None:
        return {"prompt": original, "raw": original, "applied": "0"}

    # PE 的请求体：只删 LoRA 触发词那一行（官方 :1285）。
    # 不能用 rsplit(marker)[0] —— 那样会把 LoRA **之后的所有行**一起砍掉；
    # 画风段/自定义画风一旦排在后面，改写时就看不见了。
    marker = "LoRA 触发词：" if lang == "zh" else "LoRA triggers:"
    request = "\n".join(line for line in original.splitlines()
                        if not line.startswith(marker)).rstrip()
    generated = qwen_text.enhance(pipe, request, kind="t2i", lang=lang)
    rewritten = _strip_unit_prompt_weights(_qi2_rewritten_prompt(generated, request))
    # 空串 = 改写没成功（见 _qi2_rewritten_prompt 的注释），不要跟 request 比
    applied = bool(rewritten.strip())
    if not applied:
        # 官方 :1303-1306 也是这么兜的：改写没生效就用原 prompt，只记一条日志
        rewritten = original
    else:
        # 透明底：PE 可能把它改写没了，官方 :1308-1312 会补回来
        alpha = ("透明背景" if lang == "zh" else QI2_ALPHA_BACKGROUND_PROMPT)
        if (alpha in request and alpha not in rewritten):
            rewritten = f"{rewritten.rstrip()}\n{alpha}"
        # 风格参考在 PE 之后追加（官方 :1313-1315）：PE 会把画风段一起改写，
        # 所以这里补回**完整**画风定义，别只补一句画风名。
        if style:
            if lang == "zh":
                block = style_block(style, "zh")
                if block:
                    rewritten = f"{rewritten}\n\n{block}"
            else:
                rewritten = f"{rewritten}\n\n{QI2_STYLE_REFERENCE_HEADING}\n{style}"
    lora = str(fields.get("lora_prompt") or "").strip()
    # 改写没生效时 rewritten 就是 original，里面已经有 LoRA 那一行了，
    # 再补一次会重复（重复触发词会把 LoRA 权重拉偏）。
    if lora and applied:
        rewritten = (f"{rewritten}\n\nLoRA 触发词：{lora}" if lang == "zh"
                     else f"{rewritten}\n\nLoRA triggers: {lora}")
    return {"prompt": rewritten, "raw": original, "applied": "1" if applied else "0"}


def expand_character_fields(
    pipe: Any, info: Dict[str, Any], style_reference: str = "",
) -> str:
    """官方 `generate_qi2_prompt`（:1235）：把 character_fields 的 JSON 扩写。

    `aesthetics` 与 `lora_prompt` **不参与扩写**（官方 `QI2_LITERAL_FIELDS`，
    :262）——它们是要原样保留的字面值。
    """
    if pipe is None:
        return ""
    fields = qi2_fields(info)
    pe_fields = {k: v for k, v in fields.items() if k not in QI2_LITERAL_FIELDS}
    request = "character_fields:\n" + json.dumps(pe_fields, ensure_ascii=False)
    lang = prompt_language(info)
    out = qwen_text.generate_text(
        pipe, request, qwen_text.system_prompt_of("fields", lang))
    if not out:
        return ""
    if style_reference:
        if lang == "zh":
            block = style_block(style_reference, "zh")
            if block:
                out = f"{out}\n\n{block}"
        else:
            out = f"{out}\n\n{QI2_STYLE_REFERENCE_HEADING}\n{style_reference}"
    return _strip_unit_prompt_weights(out)


#: 官方 `character_creator_v2.py:262`：这两个字段扩写时要原样保留
QI2_LITERAL_FIELDS = {"aesthetics", "lora_prompt", "custom_style"}


def generation_resolution(target_size: int = 1024) -> Tuple[int, int]:
    """官方 `character_creator_v2.py:977`：按总像素数等比，再对齐到 latent 步长。"""
    target_pixels = float(int(target_size or 1024) * 1024)
    width_unit = LATENT_ASPECT_WIDTH * LATENT_DIMENSION_STEP
    height_unit = LATENT_ASPECT_HEIGHT * LATENT_DIMENSION_STEP
    scale_units = max(1, round(math.sqrt(target_pixels / float(width_unit * height_unit))))
    return int(width_unit * scale_units), int(height_unit * scale_units)


# ---------------------------------------------------------------- 换装（Clothes Designer）
#
# 官方 `clothes_designer.py:413-448` `construct_prompt`。换装**不是"只存描述"**——
# 它是把描述变成一张**换装后的底图**（图生图），这张底图再当后续出姿势的输入。
# 所以 prompt 有三个必须照抄的部分：指令句、服装描述的顺序、背景段。

#: 服装的五个字段，**顺序就是 prompt 里的顺序**（官方 `:434` 写死的）
CLOTHES_FIELDS = ("top", "bottom", "head", "shoes", "face")

#: 官方 `:55-59`：换装只认这三种背景，其余一律当 Green
CLOTHES_BACKGROUNDS = {"Green", "Blue", "Transparent"}

#: 官方 `:442`：负向词里那句 `(naked, nude, ...:2.0)` 是**带权重的**，别当普通词读
CLOTHES_NEGATIVE = "bad quality, worst quality, (naked, nude, nipple, penis, vagina:2.0)"
CLOTHES_NEGATIVE_CLONE = (
    ", background scenery, patterned background, shapes in background, "
    "multicolored background, textured background, gradient background")
CLOTHES_NEGATIVE_BLUE = ", purple background, violet background, gradient background"

#: 官方 QI2 换装采样（`clothes_designer.py:555-556` / `:163`）
CLOTHES_DEFAULTS = {"steps": 25, "cfg": 3.0, "target_size": 1024}

#: 官方 `:59`：透明底这个值的字面量（不是 "Alpha"——那是**输出**用的）
TRANSPARENT_BACKGROUND = "Transparent"


def normalize_background_color(value: Any, default: str = "Transparent") -> str:
    """官方 `:354-356`：`green` / `GREEN` 一律归一到 `Green`/`Blue`/`Transparent`。

    认不出就回默认值。**默认是 `Transparent`**（QI2 原生 alpha）：既不用抠图，
    prompt 里的背景段也有值（否则模型会自己画场景）。透明不支持的模型会在
    `effective_background_color` 里退回 Green。
    """
    text = str(value or default).strip().capitalize()
    return text if text in CLOTHES_BACKGROUNDS else default


def effective_background_color(value: Any, model_kind: str = "qi2") -> str:
    """官方 `:359-363`：QI2 支持真透明；别的模型退回 Green。"""
    background = normalize_background_color(value)
    if background == TRANSPARENT_BACKGROUND and str(model_kind or "").lower() != "qi2":
        return "Green"
    return background


def clothes_background_prompt(background_color: str) -> str:
    """官方 `_clothes_background_prompt:124-135`（**逐字**）。

    为什么把 RGB / hex 都写进 prompt：只说 "green" 模型会画出渐变、墨绿、
    甚至带紫色的绿，抠像时 `tolerance` 一放宽就把衣服一起吃掉了。
    """
    if background_color == TRANSPARENT_BACKGROUND:
        return "Transparent background with alpha channel."
    if background_color == "Blue":
        return ("flat uniform pure blue background, exact RGB (0, 0, 255), "
                "hex #0000FF, no purple, no violet, no gradient")
    return ("flat uniform pure green background, exact RGB (0, 255, 0), "
            "hex #00FF00, no gradient")


def clothes_description(costume: Dict[str, str]) -> str:
    """五个字段按官方顺序用换行拼起来（空值丢掉）。"""
    parts = [str(costume.get(key, "") or "").strip() for key in CLOTHES_FIELDS]
    return "\n".join(p for p in parts if p)


def construct_clothes_prompt(
    costume: Dict[str, str], background_color: Any = None, clone: bool = False,
) -> Tuple[str, str]:
    """官方 `construct_prompt:413-448`：`(positive, negative)` 一次给全。

    `clone=True` 是官方的另一种模式（"照着第 2 张图的衣服穿"），
    此时服装字段不参与，而且要多一串"别画背景纹样"的负向词。
    """
    bg = effective_background_color(background_color)
    background_prompt = clothes_background_prompt(bg)

    if clone:
        positive = f"Dress character to clothes from image 2\n{background_prompt}"
    else:
        desc = clothes_description(costume)
        positive = f"Dress the character:\n{desc}\n{background_prompt}"

    negative = CLOTHES_NEGATIVE
    if clone:
        negative += CLOTHES_NEGATIVE_CLONE
    if bg == "Blue":
        negative += CLOTHES_NEGATIVE_BLUE
    return positive, negative
