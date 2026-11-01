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
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from workflows import paths
from workflows import qwen_text

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
    "full_body": "show the character's complete body from head to toe",
}
#: 官方：`:364-366`（latent 9:16，步长 8）
LATENT_ASPECT_WIDTH = 9
LATENT_ASPECT_HEIGHT = 16
LATENT_DIMENSION_STEP = 8

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
    """官方：`character_creator_v2.py:1057` `_character_framing_key`"""
    if str(info.get("image_type", "")).strip().lower() == "portrait":
        return "portrait"
    return ("full_body" if str(info.get("framing", "") or "").strip().lower()
            == "full_body" else "cowboy_shot")


def _clothing_prompt(info: Dict[str, Any]) -> str:
    """衣服：角色自己的 additional 里没写就用默认的一套（官方 `_character_clothing_prompt`）。"""
    for key in ("clothing", "costume", "outfit"):
        value = str(info.get(key, "") or "").strip()
        if value:
            return value
    return "simple outfit"


def qi2_fields(info: Dict[str, Any]) -> Dict[str, str]:
    """character_info → QI2 结构化字段（官方 `character_creator_v2.py:1101`）。"""
    background = str(info.get("background_color", "") or "").strip()
    framing_key = _framing_key(info)
    fields = {
        "gender": str(info.get("sex", "female") or "female"),
        "age": f"{int(info.get('age', 18) or 18)} years old",
        **{key: str(info.get(key, "") or "")
           for key in ("race", "skin_color", "body", "face", "hair", "eyes",
                       "additional_details")},
        "race_features": race_features(info.get("race", "")),
        "clothing": _clothing_prompt(info),
        "expression": "expressionless",
        "framing": f"single character; {QI2_NATURAL_FRAMING[framing_key]}",
        "background": (QI2_ALPHA_BACKGROUND_PROMPT if background.lower() == "transparent"
                       else (f"solid {background} background" if background
                             else "simple background")),
        "aesthetics": str(info.get("aesthetics", "masterpiece") or ""),
        "lora_prompt": str(info.get("lora_prompt", "") or ""),
    }
    return {k: str(v or "").strip() for k, v in fields.items()}


def structured_prompt(fields: Dict[str, str], style_reference: str = "") -> str:
    """拼出官方那个结构化 prompt（官方 `character_creator_v2.py:1124`）。

    `style_reference` 是风格库里那条三段式文本，官方在这里把它拆成
    Style / Style reference / Style description 三行。
    """
    style, reference, description = "", "", str(style_reference or "").strip()
    lines = description.splitlines()
    if (len(lines) == 3 and lines[0].startswith("Style: ")
            and lines[1].startswith("Reference: ")
            and lines[2].startswith("Visual description: ")):
        style = lines[0][len("Style: "):]
        reference = lines[1][len("Reference: "):]
        description = lines[2][len("Visual description: "):]
    elif len(lines) > 1 and lines[0].startswith("Style: "):
        style = lines[0][len("Style: "):]
        description = " ".join(lines[1:])

    prompt = [
        f"Draw an image of a character in the following style: {style}",
        f"Style reference: {reference}",
        f"Style description: {description}",
        f"Additional style parameters: {fields.get('aesthetics', '')}",
        "The image contains one character.",
        f"Framing: {QI2_COMPOSITION_PROMPT} {fields.get('framing', '')}",
        f"Background: {fields.get('background', '')}",
        f"Gender: {fields.get('gender', '')}",
        f"Age: {fields.get('age', '')}",
        f"Race: {', '.join(v for v in (fields.get('race', ''), fields.get('race_features', '')) if v)}",
        f"Facial features: {', '.join(v for v in (fields.get('face', ''), fields.get('expression', '')) if v)}",
        f"Hair: {fields.get('hair', '')}",
        f"Eyes: {fields.get('eyes', '')}",
        f"Skin color: {fields.get('skin_color', '')}",
    ]
    if fields.get("gender") == "female":
        prompt.append(f"Breast size: {fields.get('breast_size', '')}")
    prompt.append(f"Build: {fields.get('body', '')}")
    prompt.append(f"Clothes: {fields.get('clothing', '')}")
    prompt.extend([
        f"Additional details: {fields.get('additional_details', '')}",
        f"LoRA triggers: {fields.get('lora_prompt', '')}",
    ])
    return "\n".join(prompt)


def character_prompt(info: Dict[str, Any], style_id: str = "") -> str:
    """一条龙：character_info + 风格 id → 官方结构的 prompt。"""
    style = style_prompt_of(style_id or str(info.get("style", "") or ""))
    return structured_prompt(qi2_fields(info), style)


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


def _qi2_rewritten_prompt(generated: str, original: str) -> str:
    """官方 `:1034`：生成结果为空 / 带 think、JSON、代码块的，一律算改写失败。

    **这里跟官方有一处刻意不同**：官方是失败时返回 `original`（:1042），
    我们返回空串让调用方统一走"退回原文"。原因是拿返回值跟 request 比来判断
    "有没有改写成功"会有**假阳性**——`.strip()` 一去尾空格就不相等了
    （实测：`Additional details: ` 这种结尾带空格的行，比出来是"改写了"）。
    """
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
    fields = {k: _strip_unit_prompt_weights(v) for k, v in qi2_fields(info).items()}
    original = structured_prompt(fields, style)
    if not enhance or pipe is None:
        return {"prompt": original, "raw": original, "applied": "0"}

    # PE 的请求体：切掉 LoRA triggers 那一行（官方 :1285）
    request = original.rsplit("\nLoRA triggers:", 1)[0]
    generated = qwen_text.enhance(pipe, request, kind="t2i")
    rewritten = _strip_unit_prompt_weights(_qi2_rewritten_prompt(generated, request))
    # 空串 = 改写没成功（见 _qi2_rewritten_prompt 的注释），不要跟 request 比
    applied = bool(rewritten.strip())
    if not applied:
        # 官方 :1303-1306 也是这么兜的：改写没生效就用原 prompt，只记一条日志
        rewritten = original
    else:
        # 透明底：PE 可能把它改写没了，官方 :1308-1312 会补回来
        if (QI2_ALPHA_BACKGROUND_PROMPT in request.lower()
                and QI2_ALPHA_BACKGROUND_PROMPT not in rewritten.lower()):
            rewritten = f"{rewritten.rstrip()}\n{QI2_ALPHA_BACKGROUND_PROMPT}"
        # 风格参考在 PE 之后原样追加（官方 :1313-1315）
        if style:
            rewritten = (f"{rewritten}\n\n{QI2_STYLE_REFERENCE_HEADING}\n{style}")
    lora = str(fields.get("lora_prompt") or "").strip()
    if lora:
        rewritten = f"{rewritten}\n\nLoRA triggers: {lora}"
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
    out = qwen_text.generate_text(
        pipe, request, qwen_text.system_prompt_of("fields"))
    if not out:
        return ""
    if style_reference:
        out = f"{out}\n\n{QI2_STYLE_REFERENCE_HEADING}\n{style_reference}"
    return _strip_unit_prompt_weights(out)


#: 官方 `character_creator_v2.py:262`：这两个字段扩写时要原样保留
QI2_LITERAL_FIELDS = {"aesthetics", "lora_prompt"}


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


def normalize_background_color(value: Any, default: str = "Green") -> str:
    """官方 `:354-356`：`green` / `GREEN` 一律归一到 `Green`/`Blue`/`Transparent`。

    认不出就回 `Green`——不是"什么都不做"：prompt 里的背景段必须有值，
    否则模型会自己画背景，抠像就废了。
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
