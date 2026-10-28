"""CLIP skip：自己算 prompt embeds，不用 pipeline 的 `clip_skip` 参数。

为什么不用 diffusers 那个参数：0.41 的分支写死了
`self.text_encoder.text_model.final_layer_norm`，而本机是 transformers 5.21——
`CLIPTextModel` 被摊平了（`text_model` 这一层不存在，`final_layer_norm` 直接挂在
顶层），只要传 `clip_skip=` 就 AttributeError: 'CLIPTextModel' object has no
attribute 'text_model'。这不是我们的错，但也不能等上游修。

语义按 A1111 / Civitai 模型页的写法（模型页说 "clip skip 2" 就是这里传 2）：
  clip_skip=1  最后一层 —— 默认，等价于不传参数（逐像素一致，有测试兜着）
  clip_skip=2  跳过最后一层 —— 多数动漫 / 像素 LoRA 的训练设置

SD1.5 之外的底模不走这里：SDXL 的 prompt 涉及两个文本编码器（CLIP-L + OpenCLIP-G）
和 pooled embeds，动一处就要动一片，收益却远没有 SD1.5 明显。
"""
from __future__ import annotations

from typing import Any

import torch


def available(pipe: Any) -> bool:
    return getattr(pipe, "text_encoder", None) is not None and hasattr(pipe, "tokenizer")


def encode(pipe: Any, text: str, clip_skip: int = 1) -> torch.Tensor:
    """返回 (1, 77, 768) 的 prompt embeds。clip_skip=1 与管道默认输出一致。"""
    if not available(pipe):
        raise ValueError("该 pipeline 没有可用的 text_encoder/tokenizer")
    clip_skip = max(1, int(clip_skip))
    tok = pipe.tokenizer(
        [text], padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    enc = pipe.text_encoder
    ids = tok.input_ids.to(enc.device)
    with torch.no_grad():
        # hidden_states[0] 是 embedding 输出，[-1] 是最后一层（未过 final LN）
        hidden = enc(ids, output_hidden_states=True).hidden_states
        if clip_skip > len(hidden):
            clip_skip = len(hidden)
        emb = hidden[-clip_skip]
        # 跳过层之后必须补 final LN，否则与「最后一层 + LN」不在同一空间
        norm = getattr(enc, "final_layer_norm", None)
        if norm is None and hasattr(enc, "text_model"):        # transformers 4.x 布局
            norm = enc.text_model.final_layer_norm
        if norm is not None:
            emb = norm(emb)
    return emb.to(enc.dtype)


def embeds(pipe: Any, prompt: str, negative: str, clip_skip: int = 1):
    """(正向 embeds, 负向 embeds)。clip_skip=1 时调用方可直接忽略本模块。"""
    return encode(pipe, prompt, clip_skip), encode(pipe, negative, clip_skip)
