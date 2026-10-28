"""CLIP skip：自实现编码的正确性（clip_skip=1 必须与管道默认逐元素一致）。"""
from __future__ import annotations

import pytest
import torch

from workflows import clip


def test_clip_skip_1_matches_pipeline_default(eng):
    """cs=1 等价于不传参数——不成立的话「保持默认」就是空话，等于偷偷改了画风。"""
    pipe = eng.txt2img_pipe("sd15")
    text = "blue slime monster, flat colors"
    tok = pipe.tokenizer([text], padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    with torch.no_grad():
        plain = pipe.text_encoder(tok.input_ids.to(pipe.text_encoder.device))[0]
    mine = clip.encode(pipe, text, 1)
    assert torch.equal(plain, mine)


def test_clip_skip_2_is_a_different_embedding(eng):
    """跳过最后一层必须真的换向量，否则这个开关就是摆设。"""
    pipe = eng.txt2img_pipe("sd15")
    a, b = clip.encode(pipe, "blue slime monster", 1), clip.encode(pipe, "blue slime monster", 2)
    assert a.shape == b.shape == (1, 77, a.shape[-1])
    assert not torch.equal(a, b)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    assert 0.5 < cos < 0.99, cos


def test_clip_skip_clamps_above_layer_count(eng):
    """写个离谱的值不能崩，夹到层数上限即可。"""
    pipe = eng.txt2img_pipe("sd15")
    deep = clip.encode(pipe, "x", 999)
    assert deep.shape[0] == 1 and torch.isfinite(deep).all()


def test_clip_helper_rejects_pipeline_without_text_encoder():
    class Bare:
        pass

    assert clip.available(Bare()) is False
    with pytest.raises(ValueError):
        clip.encode(Bare(), "x", 2)
