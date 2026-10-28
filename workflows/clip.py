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
    tok = pipe.tokenizer(
        [text], padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    return _forward(pipe, tok.input_ids, clip_skip)


def _forward(pipe: Any, ids: torch.Tensor, clip_skip: int = 1) -> torch.Tensor:
    """给定 (1, L) 的 token id，跑一次文本编码器并取第 -clip_skip 层。

    抽出来是因为**分段编码**要对每一段各跑一次，逻辑必须和单段完全一致，
    否则 clip_skip=2 时两种路径会落在不同的表示空间里。
    """
    clip_skip = max(1, int(clip_skip))
    enc = pipe.text_encoder
    ids = ids.to(enc.device)
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


def count_tokens(pipe: Any, text: str) -> int:
    """**真实** token 数（不含 BOS/EOS）。比按"1.35 token/词"估算准得多。

    `presets.rough_tokens` 是纯字符串启发式，对逗号分隔的英文串**高估约 3 倍**
    （实测 WUXIA_NEG 估算 113、真实 35），于是 `fit()` 会把本来塞得下的词提前砍掉。
    """
    if not available(pipe):
        return 0
    return len(pipe.tokenizer(text, add_special_tokens=False).input_ids)


_TOKENIZER: Any = None
_TOKENIZER_TRIED = False


def count_text(text: str) -> int:
    """不依赖管线的真实计数（给 presets 用）：从本地权重目录现找一个 CLIPTokenizer。

    找不到（没下权重 / 纯 CPU 单测环境）时返回 **0**，调用方据此退回启发式。
    这样做是为了让"提示词预算"这件事在**构造提示词的时刻**就是准的——
    而构造提示词发生在管线加载之前，拿不到 pipe。
    """
    global _TOKENIZER, _TOKENIZER_TRIED
    if not _TOKENIZER_TRIED:
        _TOKENIZER_TRIED = True
        try:
            from pathlib import Path

            from transformers import CLIPTokenizer

            root = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
            for d in sorted(root.glob("*/tokenizer")):
                if (d / "vocab.json").exists() or (d / "tokenizer_config.json").exists():
                    _TOKENIZER = CLIPTokenizer.from_pretrained(str(d))
                    break
        except Exception:                 # noqa: BLE001  缺依赖/缺权重都不该影响出图
            _TOKENIZER = None
    if _TOKENIZER is None:
        return 0
    try:
        return len(_TOKENIZER(text, add_special_tokens=False).input_ids)
    except Exception:                     # noqa: BLE001
        return 0


def needs_chunk(pipe: Any, *texts: str) -> bool:
    """任一段超过 77 token 就要分段编码。没有 tokenizer 时保守返回 False。"""
    if not available(pipe):
        return False
    limit = pipe.tokenizer.model_max_length
    return any(count_tokens(pipe, t) > limit - 2 for t in texts if t)


def split_chunks(pipe: Any, text: str, max_chunks: int = 2) -> list:
    """把文本切成 75 token 的段列表。空文本返回 `[[]]`（一段空）。"""
    limit = pipe.tokenizer.model_max_length           # 77
    step = limit - 2                                  # 留 BOS/EOS
    ids = pipe.tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return [[]]
    return [ids[i:i + step] for i in range(0, len(ids), step)][:max_chunks]


def _encode_chunk(pipe: Any, ids, clip_skip: int) -> torch.Tensor:
    """编码**一段**，补到 77 个位置（BOS + 内容 + EOS + pad）。"""
    limit = pipe.tokenizer.model_max_length
    bos, eos = pipe.tokenizer.bos_token_id, pipe.tokenizer.eos_token_id
    pad = pipe.tokenizer.pad_token_id
    full = [bos] + list(ids) + [eos]
    full = full[:limit] + [pad] * max(0, limit - len(full))
    return _forward(pipe, torch.tensor([full], dtype=torch.long), clip_skip)


def encode_long(pipe: Any, text: str, clip_skip: int = 1,
                max_chunks: int = 2, chunks: int | None = None) -> torch.Tensor:
    """长文本分段编码后拼起来，返回 (1, 77×n, 768)。

    **为什么需要**：SD1.5 的 CLIP 位置编码只有 77 个位置，超出部分被
    **静默丢弃**（不报错、不警告）。本项目的武侠契约自己就注明"写到 113 token
    尾部全被丢弃"——等于最后几条质量兜底词从来没生效过。

    UNet 的交叉注意力对序列长度不敏感，所以把文本按 75 token 切段、各自编码后
    在序列维拼接（A1111 的 "CLIP 77→154" 就是这个做法），模型照样能读到
    第 78~154 个 token。代价是交叉注意力 seq_len 翻倍，512² 下约 +3~5% 耗时。

    **`chunks` 指定段数**（正负对齐用）：文本本身不够长时补**空段**，而不是
    少返回几段。曾经的实现各算各的段数，于是「正向短 + 负向长」得到
    (1,77,768) 与 (1,154,768)，diffusers 在 CFG 相减时直接抛
    `prompt_embeds != negative_prompt_embeds`。
    """
    parts = split_chunks(pipe, text, max_chunks)
    n = max(1, chunks or len(parts))
    embs = [_encode_chunk(pipe, parts[i] if i < len(parts) else [], clip_skip)
            for i in range(n)]
    return torch.cat(embs, dim=1) if n > 1 else embs[0]


def embeds(pipe: Any, prompt: str, negative: str, clip_skip: int = 1,
           max_chunks: int = 2):
    """(正向 embeds, 负向 embeds)。clip_skip=1 时调用方可直接忽略本模块。

    **两侧段数必须一致**：diffusers 的 CFG 是 `pos - neg`，形状必须相同。
    只让超长的一侧分段、另一侧保持 77，会直接抛
    `prompt_embeds torch.Size([1,77,768]) != negative_prompt_embeds
    torch.Size([1,154,768])`——所以这里统一取两侧段数的最大值，短板补空段。
    """
    n_pos = len(split_chunks(pipe, prompt, max_chunks))
    n_neg = len(split_chunks(pipe, negative, max_chunks))
    n = max(n_pos, n_neg)

    # 超过 max_chunks 段能装下的量仍然会被丢——但要说出来。
    # 静默截断是这条链路的老毛病（CLIP 本身就是这样），不该再重复一次。
    limit = pipe.tokenizer.model_max_length
    room = max_chunks * (limit - 2)
    for label, text in (("正向", prompt), ("负向", negative)):
        ntok = count_tokens(pipe, text)
        if ntok > room:
            print(f"[warn] {label}提示词 {ntok} token，超过 {max_chunks} 段能容纳的 "
                  f"{room}，末尾会被丢弃（分段编码上限）")

    if n <= 1:
        return encode(pipe, prompt, clip_skip), encode(pipe, negative, clip_skip)
    return (encode_long(pipe, prompt, clip_skip, max_chunks, chunks=n),
            encode_long(pipe, negative, clip_skip, max_chunks, chunks=n))
