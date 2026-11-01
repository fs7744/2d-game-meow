"""Prompt Enhancement：拿**图像模型自己的 Qwen3-VL 文本编码器**当 LLM 用。

**这不是我提的方案，是官方的做法**：VNCCS 的 PE 走 ComfyUI 原生 `TextGenerate`，
喂的就是 QI2 的 CLIP（`character_creator_v2.py:1293`）—— 那个 CLIP 是
Qwen3-VL 8B，跟我们 `engine.py:478` 加载的 `Qwen3VLForConditionalGeneration`
是同一个东西。所以 PE **不需要任何新权重**。

为什么官方这么选（两条都成立）：
- **同分布**：改写出来的措辞，跟图像模型训练时见到的文本一致，风格 LoRA 吃得更准；
- **不重复加载**：编码器本来每次出图都要加载一次来编码 prompt，PE 只是让它
  多干一会儿，不是额外再拉一个模型进来。

采样参数照抄官方 `character_creator_v2.py:159-174`（`QI2_TEXT_GENERATION_DEFAULTS`）。
system prompt 读随项目发行的官方文件 `character_template/qwen_image_21_t2i_system_prompt.txt`。

> ⚠️ **已知代价（还没解决）**：PE 要跑自回归解码，文本编码器得**常驻 GPU**；
> 而我们的管线为了省显存是「编码完就挪走」（`engine.py:439`，两块 fp8 常驻
> 15.5GB 已经压线）。24GB 上跟 transformer 抢是真的，8B 解码 2048 token 也不快。
> 所以每个入口都做了**失败降级**：PE 挂了就退回未扩写的 prompt，不阻断出图。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

from workflows import paths

log = logging.getLogger(__name__)

#: 官方 `character_creator_v2.py:159-174`
QI2_TEXT_DEFAULTS: Dict[str, Any] = {
    "max_length": 2048,
    "temperature": 0.3,
    "top_k": 64,
    "top_p": 0.95,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
    "seed": 0,
    "presence_penalty": 0.0,
}

#: 官方那两份提示词模板（随项目发行，已搬进 web/public/vnccs/character_template/）
SYSTEM_PROMPT_FILES = {
    # 造角色（T2I）：把请求改写成"你看到的画面"的一段英文描述
    "t2i": "qwen_image_21_t2i_system_prompt.txt",
    # 字段扩写：把 character_fields 的 JSON 扩成精确英文描述
    "fields": "qi2_prompt_rewriter.txt",
    # 换装（编辑式）
    "edit": "qi2_edit_prompt_rewriter.txt",
}


def system_prompt_of(kind: str = "t2i") -> str:
    """读官方的 system prompt 文件。文件缺失返回空串（**不自己编一套**）。"""
    name = SYSTEM_PROMPT_FILES.get(kind)
    if not name:
        return ""
    path = os.path.join(str(paths.VNCCS_TEMPLATE_DIR), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        log.warning("官方提示词模板读不到：%s", path)
        return ""


def _tokenizer_of(pipe: Any):
    """Qwen 管线的 tokenizer 在不同版本里挂的位置不一样，两处都试。"""
    tok = getattr(pipe, "tokenizer", None)
    if tok is not None:
        return tok
    proc = getattr(pipe, "processor", None)
    return getattr(proc, "tokenizer", None) if proc is not None else None


class _OnDevice:
    """把文本编码器临时搬到 GPU 上跑生成，用完放回原处。

    **为什么要放回**：管线开了 cpu offload 时，编码器平时住在 CPU（这就是
    我们省显存的办法）。PE 完不还原，后面每次出图都多占 8.8G，
    用户会以为"跑了几次之后怎么变慢了"。
    """

    def __init__(self, module: Any, device: Optional[str] = None):
        self.mod = module
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.origin: Optional[torch.device] = None

    def __enter__(self):
        try:
            self.origin = next(self.mod.parameters()).device
        except StopIteration:                     # 空权重（fp8 骨架）
            self.origin = None
        if str(self.origin) != self.device:
            self.mod.to(self.device)
        return self.mod

    def __exit__(self, *exc):
        try:
            if self.origin is not None and str(self.origin) != str(
                    next(self.mod.parameters()).device):
                self.mod.to(self.origin)
        except StopIteration:
            pass
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return False


def _template(tok: Any, messages: List[Dict[str, str]]) -> str:
    """套 chat 模板。**必须关掉 thinking**：Qwen3 默认会输出一大段思考，
    官方也是显式 `thinking=False`（`qwen_vl.py:53-64` 的 `enable_thinking=false`）。

    不同 transformers 版本对 `enable_thinking` 的支持不一样，不支持就退回去。
    """
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate_text(
    pipe: Any,
    request: str,
    system_prompt: str = "",
    *,
    max_new_tokens: Optional[int] = None,
    **overrides: Any,
) -> str:
    """用管线的文本编码器跑一次文本生成。

    **任何失败都返回空串**，让调用方退回未扩写的 prompt —— PE 是增强，
    不该成为出图的必要条件。
    """
    te = getattr(pipe, "text_encoder", None)
    tok = _tokenizer_of(pipe)
    if te is None or tok is None:
        log.warning("PE 跳过：拿不到 text_encoder / tokenizer")
        return ""
    req = str(request or "").strip()
    if not req:
        return ""

    params = dict(QI2_TEXT_DEFAULTS)
    params.update({k: v for k, v in overrides.items() if v is not None})
    n_new = int(max_new_tokens or params.pop("max_length", 2048))
    # 官方那几个采样参数里的 seed / presence_penalty 我们不用（前者要整套
    # generator，后者 transformers 不一定支持），只留真正会传进 generate 的。
    gen: Dict[str, Any] = {
        "max_new_tokens": n_new,
        "do_sample": True,
        "temperature": float(params.get("temperature", 0.3)),
        "top_k": int(params.get("top_k", 64)),
        "top_p": float(params.get("top_p", 0.95)),
        "min_p": float(params.get("min_p", 0.05)),
        "repetition_penalty": float(params.get("repetition_penalty", 1.05)),
    }

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": req})

    try:
        text = _template(tok, messages)
        with torch.no_grad():
            with _OnDevice(te) as model:
                inputs = tok(text, return_tensors="pt").to(model.device)
                # transformers 版本差异：min_p 是后加的，老版本没有就退掉
                for drop in ((), ("min_p",)):
                    kwargs = {k: v for k, v in gen.items() if k not in drop}
                    try:
                        out = model.generate(**inputs, **kwargs)
                        break
                    except TypeError as exc:
                        if "min_p" not in str(exc) or drop:
                            raise
                        continue
                cut = inputs["input_ids"].shape[-1]
                seq = out[0][cut:] if out.ndim > 1 else out[cut:]
        return tok.decode(seq, skip_special_tokens=True).strip()
    except Exception as exc:                      # noqa: BLE001
        log.warning("Prompt Enhancement 失败（退回原 prompt）：%s", exc)
        return ""


def enhance(
    pipe: Any, prompt: str, kind: str = "t2i", system_prompt: Optional[str] = None,
    **overrides: Any,
) -> str:
    """PE：改写一段 prompt。失败返回空串（**调用方要自己兜底**）。"""
    sp = system_prompt if system_prompt is not None else system_prompt_of(kind)
    return generate_text(pipe, prompt, sp, **overrides)
