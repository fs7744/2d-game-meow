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


def generation_resolution(target_size: int = 1024) -> Tuple[int, int]:
    """官方 `character_creator_v2.py:977`：按总像素数等比，再对齐到 latent 步长。"""
    target_pixels = float(int(target_size or 1024) * 1024)
    width_unit = LATENT_ASPECT_WIDTH * LATENT_DIMENSION_STEP
    height_unit = LATENT_ASPECT_HEIGHT * LATENT_DIMENSION_STEP
    scale_units = max(1, round(math.sqrt(target_pixels / float(width_unit * height_unit))))
    return int(width_unit * scale_units), int(height_unit * scale_units)


def clothes_prompt(info: Dict[str, Any], costume: Dict[str, str]) -> str:
    """换装的编辑指令（官方 `clothes_designer.py:413-448`：`Dress the character: ...`）。

    官方这里还会走 `qi2_edit_prompt_rewriter.txt` 扩写（要 LLM），我们暂时**原样**。
    """
    parts = [str(v or "").strip() for v in (
        costume.get("head", ""), costume.get("face", ""),
        costume.get("top", ""), costume.get("bottom", ""),
        costume.get("shoes", ""),
    )]
    body = ", ".join(p for p in parts if p)
    return f"Dress the character:\n{body}" if body else "Dress the character:\n"
