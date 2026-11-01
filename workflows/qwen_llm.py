"""本地 LLM（Qwen3.5-4B GGUF）—— 移植 VNCCS 的 `qwen_vl.py` 与 Cloner 旁路。

**它是旁路，不是依赖**：看图反推属性、一句话造角色、服装向导都用它，
但**出图链路一次都不碰**。模型没装 / 依赖缺 / 推理超时，都必须退成
"让用户手填"，而不是把界面卡死。

来源（相对 `ComfyUI_VNCCS-main`）：

| 这里 | 官方 |
|---|---|
| `configure_qwen_text_chat` | `nodes/qwen_vl.py:53-64`（逐字） |
| `parse_wizard_json` | `:89-118`（三级降级） |
| `normalize_character_wizard_traits` | `:121-138` |
| `composite_alpha` / `_background_rgb` | `nodes/character_cloner.py:52-73` |
| `CLONER_SYSTEM_PROMPT` / `CLONER_INSTRUCTION` | `:323-353`（逐字） |
| `analyze_character_image` | `:290-390` |

**跟官方的两处必要差异**（都不是口味问题）：

1. **有超时**。官方 `preview_runtime.py:25-52` 只有一个串行锁，没有超时——
   8B 模型跑一次几十秒，HTTP 挂死时完全查不出原因。这里每次推理都带超时，
   超时返回 `TIMEOUT`，让前端提示"再试一次"。
2. **用完就卸**。默认每次调用结束把模型从显存里放掉（`MEOW_LLM_KEEP=1` 可留驻）：
   4.5GB 常驻跟 Qwen 图像管线抢 24GB，用户切回去出图就可能 OOM。
   代价是每次多 3~5 秒加载，而推理本身是几十秒，占总时间很小。
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from workflows import paths

# ---------------------------------------------------------------- 权重与上下文

#: 官方 `qwen_vl.py:66`（Q8_0，4.5GB）
MODEL_FILENAME = "Qwen3.5-4B-Q8_0.gguf"
#: 官方 `qwen_vl.py:72`（视觉投影器；只做纯文本时可不要）
MMPROJ_FILENAME = "mmproj-F16.gguf"

#: 官方 Cloner 用 4096（`character_cloner.py:296`），Creator 用 6144（`:1832`）
VISION_CTX = 4096
TEXT_CTX = 6144

#: 送进模型的图上限（官方 `:346` 是 **512**，不是 384——384 那个是给
#: Qwen-Image-Edit 编码器的 `vl_size`，两回事）
IMAGE_MAX_SIDE = 512
IMAGE_JPEG_QUALITY = 85

#: 默认推理超时（秒）。官方没有超时；一次反推实测 20~40s，给足余量。
DEFAULT_TIMEOUT = float(os.environ.get("MEOW_LLM_TIMEOUT", "300"))

#: 是否让模型常驻显存（默认**不**，见模块注释第 2 条）
KEEP_LOADED = os.environ.get("MEOW_LLM_KEEP") == "1"

#: GPU 层数。官方是 -1（全部上卡）；显存紧时可设 0 全走 CPU。
N_GPU_LAYERS = int(os.environ.get("MEOW_LLM_GPU_LAYERS", "-1"))

#: 允许从模型带回来、并写进角色属性的字段（官方白名单 `:374-388`）
TRAIT_KEYS = (
    "sex", "age", "race", "skin_color", "hair", "eyes", "face", "body",
    "additional_details", "aesthetics", "nsfw",
)

#: 官方 `character_cloner.py:322`（system 一句）
CLONER_SYSTEM_PROMPT = (
    "You extract visible physical character identity, with no outfit or pose "
    "descriptions. Output valid JSON only."
)

#: 官方 `character_cloner.py:323-353`（user 指令，**逐字**照抄）
CLONER_INSTRUCTION = """Analyze the character in the image and strictly output valid JSON.
Extract visible physical character traits using concise comma-separated tags.
Describe colors and physical traits in your own words; do not choose from presets or a closed list.

Keys:
- sex (string: 'male' or 'female')
- age (int: estimated number)
- race (string: e.g. 'human', 'elf', 'cyborg')
- skin_color (string: describe the actual visible skin color, including unusual or multiple colors)
- hair (string: comma-separated tags for color and style, e.g. 'blue hair, long hair, ponytail')
- eyes (string: comma-separated tags for color and shape, e.g. 'green eyes, tsurime')
- face (string: clearly visible facial features, e.g. 'freckles', 'facial scar', 'makeup')
- body (string: tags for build, e.g. 'slim', 'muscular', 'tall')
- additional_details (string: physical character traits not covered by the other fields, e.g. 'monster arm', 'extra limbs', 'tail', 'wings', 'body markings')
- aesthetics (string: visible art style, e.g. 'anime style, illustration, flat color')
- nsfw (boolean)

Rules:
- Describe only features clearly visible in the image. Do not invent traits or copy the examples.
- Use an empty string for absent, hidden, or uncertain traits. Do not fill a field just to avoid an empty value.
- Determine skin_color from exposed skin, not clothing, background, or assumed human skin tones.
- Preserve unusual skin colors as drawn. Red or pink skin across the face or body is skin_color, not blush.
- Use "pale skin" only for unusually pale/very light skin, never as a generic default.
- Add blush only when distinct localized cheek blush is visible against the surrounding skin color.
- Different colors on exposed body parts can be character traits; do not assume they are gloves or tights.
- additional_details contains only physical character features that do not fit race, skin_color, hair, eyes, face, or body.
- Do not include clothing, footwear, wearable accessories, held objects, pose, actions, facial expressions, camera framing, or background in character trait fields.

Return all keys in a raw JSON object. Do not output the word 'tag' as a value."""


class LlmError(RuntimeError):
    """带官方错误码的 LLM 失败（前端按 code 决定怎么提示）。

    官方那套码（`character_cloner.py:224-402`）：`DEPENDENCY_MISSING` /
    `MODEL_MISSING` / `MODEL_INVALID` / `MMPROJ_MISSING` / `INVALID_RESPONSE` /
    `PARSE_ERROR` / `INFERENCE_ERROR`。我们多一个 `TIMEOUT`（官方没有超时）。
    """

    def __init__(self, code: str, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.raw = raw


# ---------------------------------------------------------------- 权重定位


def model_path() -> str:
    return os.path.join(str(paths.LLM_DIR), MODEL_FILENAME)


def mmproj_path() -> str:
    return os.path.join(str(paths.LLM_DIR), MMPROJ_FILENAME)


def validate_gguf(path: str, label: str = "模型") -> None:
    """官方 `qwen_vl.py:143-154`：文件在、够大、magic 是 `GGUF`。

    **要真校验 magic**：下一个下载到一半的坑（几百 MB 的残file）会让
    llama.cpp 抛一句完全看不出原因的错，用户只会以为"我们代码坏了"。
    """
    if not os.path.isfile(path):
        raise LlmError("MODEL_MISSING", f"{label} 不存在：{path}")
    size = os.path.getsize(path)
    if size < 1024 * 1024:
        raise LlmError("MODEL_INVALID", f"{label} 太小，不像有效的 GGUF（{size} 字节）")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        raise LlmError("MODEL_INVALID", f"{label} 不是有效的 GGUF（magic={magic!r}）")


# ---------------------------------------------------------------- 加载


_TEXT_LLM: Any = None
_VISION_LLM: Any = None
_LOAD_LOCK = threading.Lock()
#: 推理串行锁（官方 `preview_runtime.py:25-52` 也是串行：同一份权重不能并发解码）
_INFER_LOCK = threading.Lock()


def _import_llama() -> Any:
    try:
        import llama_cpp
    except ImportError as exc:
        raise LlmError(
            "DEPENDENCY_MISSING",
            f"没装 llama-cpp-python：{exc}（需要 JamePeng fork，含 Qwen35ChatHandler）",
        ) from exc
    return llama_cpp


def qwen_vl_chat_handler(llama_cpp: Any) -> Any:
    """官方 `qwen_vl.py:27-49`：**只认 `Qwen35ChatHandler`，不许降级**。

    官方拒绝用 `Llava15ChatHandler` 兜底，因为拿它跑 Qwen 的 mmproj 会崩。
    装错 fork 的症状就是这里 `ImportError/AttributeError`，很容易被误判成
    "我们代码写错了"。
    """
    chat_format = getattr(llama_cpp, "llama_chat_format", None)
    if chat_format is None:
        try:
            import llama_cpp.llama_chat_format as chat_format       # noqa: F401
        except Exception:                                          # noqa: BLE001
            chat_format = None
    handler = getattr(chat_format, "Qwen35ChatHandler", None) if chat_format else None
    if handler is None:
        available = [n for n in dir(chat_format) if "Handler" in n] if chat_format else []
        raise LlmError(
            "DEPENDENCY_MISSING",
            "llama-cpp-python 里没有 Qwen35ChatHandler（需要 JamePeng fork）。"
            f"当前可用的 handler：{available}",
        )
    return handler


def configure_qwen_text_chat(llm: Any) -> None:
    """官方 `qwen_vl.py:53-64` **逐字**：从 GGUF metadata 读 chat template，
    头部拼 `{% set enable_thinking = false %}`。

    为什么要拼这一句：模板默认开思考，模型会把 token 全花在 `<think>` 上，
    900 的 `max_tokens` 常常不够，出来就被截断成空。
    """
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        raise LlmError("MODEL_INVALID", "Qwen3.5 GGUF 缺 chat template")
    llm.chat_handler = Jinja2ChatFormatter(
        template="{% set enable_thinking = false %}" + template,
        eos_token="<|im_end|>",
        bos_token="",
    ).to_chat_handler()


def load_llm(vision: bool = True) -> Any:
    """加载（并缓存）本地 LLM。`vision=True` 时带多模态投影器。

    第一次调用会读 4.5GB，约 3~8 秒；之后走缓存。用完是否卸载见
    `KEEP_LOADED`。
    """
    global _TEXT_LLM, _VISION_LLM
    cached = _VISION_LLM if vision else _TEXT_LLM
    if cached is not None:
        return cached
    with _LOAD_LOCK:
        cached = _VISION_LLM if vision else _TEXT_LLM
        if cached is not None:
            return cached
        llama_cpp = _import_llama()
        path = model_path()
        validate_gguf(path, "Qwen3.5 GGUF")
        kwargs: Dict[str, Any] = {
            "model_path": path,
            "n_ctx": VISION_CTX if vision else TEXT_CTX,
            "n_gpu_layers": N_GPU_LAYERS,
            "verbose": False,
        }
        if vision:
            proj = mmproj_path()
            if not os.path.isfile(proj):
                raise LlmError(
                    "MMPROJ_MISSING",
                    f"缺视觉投影器：{proj}（看图反推要多模态）")
            validate_gguf(proj, "视觉投影器")
            handler_cls = qwen_vl_chat_handler(llama_cpp)
            kwargs["chat_handler"] = handler_cls(
                clip_model_path=proj, enable_thinking=False, verbose=False)
        try:
            llm = llama_cpp.Llama(**kwargs)
        except Exception as exc:                                   # noqa: BLE001
            msg = str(exc)
            code = "MMPROJ_MISSING" if "mmproj" in msg.lower() else "MODEL_INVALID"
            raise LlmError(code, f"加载模型失败：{msg}") from exc
        if not vision:
            # 纯文本要自己装 chat handler（官方 `character_creator_v2.py:1832-1839`）
            configure_qwen_text_chat(llm)
        if vision:
            _VISION_LLM = llm
        else:
            _TEXT_LLM = llm
        return llm


def unload_llm() -> None:
    """放掉显存里的模型（默认每次调用后都会调）。"""
    global _TEXT_LLM, _VISION_LLM
    with _LOAD_LOCK:
        _TEXT_LLM = None
        _VISION_LLM = None


def llm_status() -> Dict[str, Any]:
    """给 `/health` 之类的地方看：装没装、权重在不在、有没有加载。"""
    out: Dict[str, Any] = {
        "model": MODEL_FILENAME,
        "path": model_path(),
        "present": os.path.isfile(model_path()),
        "mmproj": os.path.isfile(mmproj_path()),
        "loaded": _VISION_LLM is not None or _TEXT_LLM is not None,
    }
    try:
        _import_llama()
        out["dependency"] = "ok"
    except LlmError as exc:
        out["dependency"] = exc.code
    return out


# ---------------------------------------------------------------- 推理（带超时）


def _infer(fn: Any, timeout: float) -> Any:
    """在子线程里跑推理，主线程最多等 `timeout` 秒。

    **为什么不能直接调**：`create_chat_completion` 是 C++ 里的阻塞调用，
    Python 侧没有取消接口；不套线程的话，一次卡住就把 HTTP 请求一起挂死
    （官方就是这样，只能重启进程）。

    超时后我们把锁**留给还在跑的那个线程**去释放——这样下一次调用会
    `acquire` 失败并立刻回 `TIMEOUT`，而不是又压一个推理上去。
    """
    deadline = max(1.0, float(timeout))
    if not _INFER_LOCK.acquire(timeout=deadline):
        raise LlmError("TIMEOUT", "上一次推理还没结束，稍后再试")
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:                               # noqa: BLE001
            box["error"] = exc
        finally:
            _INFER_LOCK.release()

    thread = threading.Thread(target=target, daemon=True, name="llm-infer")
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        raise LlmError("TIMEOUT", f"推理超过 {int(deadline)}s 还没返回，请重试")
    if "error" in box:
        exc = box["error"]
        if isinstance(exc, LlmError):
            raise exc
        raise LlmError("INFERENCE_ERROR", f"推理失败：{exc}") from exc
    return box.get("value")


# ---------------------------------------------------------------- 解析


def parse_wizard_json(content: str) -> Optional[Dict[str, Any]]:
    """官方 `qwen_vl.py:89-118`：三级降级把模型输出抠成 dict。

    1. `json_repair.loads`（模型常写坏：中文引号、尾逗号、少括号）
    2. 结果是 list（模型把单个对象包成数组）→ 取 `[0]`
    3. 剥 ` ```json ` / ` ``` ` 代码块 → 正则捞 `{...}`
    4. 还不行 → `None`（调用方按 `PARSE_ERROR` 处理，并带上原文）
    """
    data = None
    try:
        import json_repair

        data = json_repair.loads(content)
    except Exception:                                              # noqa: BLE001
        data = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        try:
            text = str(content or "").strip()
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0]
            else:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    text = match.group(0)
            data = json.loads(text.strip())
        except Exception:                                          # noqa: BLE001
            data = None
    return data if isinstance(data, dict) else None


def normalize_character_wizard_traits(data: Dict[str, Any]) -> Dict[str, Any]:
    """官方 `qwen_vl.py:121-138`：把模型的自由输出收敛成能用的属性。

    - 7 个文本字段：list → `", ".join`；非字符串强转；strip
    - `sex` 只认 male/female（模型有时写 "Male" / "man"）
    - `age` clamp 到 `[1, 100]`（模型会写 0 或 300）
    """
    result: Dict[str, Any] = {}
    for key in ("race", "skin_color", "hair", "eyes", "face", "body", "additional_details"):
        value = data.get(key, "")
        if isinstance(value, list):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        elif not isinstance(value, str):
            value = str(value) if value is not None else ""
        result[key] = value.strip()

    sex = str(data.get("sex", "female")).strip().lower()
    result["sex"] = "male" if sex.startswith("m") else "female"
    try:
        age = int(float(data.get("age", 18)))
    except (TypeError, ValueError):
        age = 18
    result["age"] = max(1, min(100, age))
    return result


def filter_traits(data: Dict[str, Any]) -> Dict[str, Any]:
    """官方 `character_cloner.py:374-388`：白名单 + 收敛 + 类型修正。

    白名单是**防注入**：模型偶尔会多吐 `positive_prompt` / `config` 这类键，
    直接塞进角色 config 就能改到别处去。
    """
    picked = {k: v for k, v in data.items() if k in TRAIT_KEYS}
    normalized = normalize_character_wizard_traits(picked)
    for key in set(picked) & set(normalized):
        picked[key] = normalized[key]
    if "aesthetics" in picked:
        value = picked["aesthetics"]
        picked["aesthetics"] = (", ".join(str(i) for i in value)
                                if isinstance(value, list) else str(value or ""))
    if "nsfw" in picked:
        picked["nsfw"] = picked["nsfw"] is True or str(picked["nsfw"]).lower() == "true"
    return picked


# ---------------------------------------------------------------- 图像预处理


def background_rgb(value: str, default: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[int, int, int]:
    """官方 `character_cloner.py:52-63`：green / blue / `#RRGGBB` / 白兜底。"""
    normalized = str(value or "").strip().lower()
    if normalized == "green":
        return (0, 255, 0)
    if normalized == "blue":
        return (0, 0, 255)
    if normalized.startswith("#") and len(normalized) == 7:
        try:
            return tuple(int(normalized[i:i + 2], 16) for i in (1, 3, 5))   # type: ignore[return-value]
        except ValueError:
            pass
    return default


def composite_alpha(image: Image.Image, background_color: str = "") -> Image.Image:
    """官方 `character_cloner.py:65-73`：透明图先压到背景色上再送模型。

    **必须压**：JPEG 没有 alpha，直接 `convert("RGB")` 会把透明区变成黑，
    模型就从"一坨黑边"里读出"黑色头发/黑色轮廓"这种根本没画的东西。
    """
    if image.mode not in ("RGBA", "LA") and not (
            image.mode == "P" and "transparency" in image.info):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    if not np.any(alpha < 255):
        return rgba.convert("RGB")
    bg = Image.new("RGBA", rgba.size, (*background_rgb(background_color), 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def image_data_url(image_path: str, background_color: str = "") -> str:
    """图片 → `data:image/jpeg;base64,...`（官方 `:346-354` 的做法）。

    **不用文件路径**：官方也是塞 data URL——llama.cpp 的 chat handler 只认
    内容块，给路径它不会自己读。
    """
    if not os.path.isfile(image_path):
        raise LlmError("IMAGE_MISSING", f"图不存在：{image_path}")
    with Image.open(image_path) as opened:
        img = composite_alpha(ImageOps.exif_transpose(opened), background_color)
        img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------- 看图反推属性


def analyze_character_image(
    image_path: str, *, background_color: str = "",
    timeout: float = DEFAULT_TIMEOUT, keep_loaded: Optional[bool] = None,
) -> Dict[str, Any]:
    """一张角色图 → 11 个属性字段（官方 `character_cloner.py:290-390`）。

    **只返回字段，不写 config**：官方是先弹给用户看、用户确认再存。
    属性写错了会一路影响后面所有出图，不能默默覆盖。
    """
    data_url = image_data_url(image_path, background_color)
    llm = load_llm(vision=True)
    messages = [
        {"role": "system", "content": CLONER_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": CLONER_INSTRUCTION},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    def call() -> Any:
        # 采样参数照官方 `:357-361`：max_tokens 1024 / temperature 0.2
        return llm.create_chat_completion(
            messages=messages, max_tokens=1024, temperature=0.2)

    started = time.time()
    try:
        response = _infer(call, timeout)
    finally:
        if not (KEEP_LOADED if keep_loaded is None else keep_loaded):
            unload_llm()
    seconds = round(time.time() - started, 2)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError("INFERENCE_ERROR", f"模型返回结构不对：{exc}") from exc
    if not content or not str(content).strip():
        raise LlmError("INVALID_RESPONSE", "模型返回空内容，请重试")
    data = parse_wizard_json(str(content))
    if not isinstance(data, dict):
        raise LlmError(
            "PARSE_ERROR",
            "模型没返回 JSON 对象（原文见 raw），可以重试或手填",
            raw=str(content)[:2000])
    traits = filter_traits(data)
    if not traits:
        raise LlmError(
            "PARSE_ERROR",
            "返回的 JSON 里没有一个认识的属性字段",
            raw=str(content)[:2000])
    return {
        "fields": traits,
        "seconds": seconds,
        "raw": str(content)[:2000],
        "image": image_path,
    }
