"""Qwen-Image-2.1 的**真·蒙版局部重绘**（masked inpaint）。

## 为什么需要这个模块

Qwen-Image-2.1 的官方 diffusers 管线（`QwenImage21Pipeline`）**没有** mask 参数：
`__call__` 的签名里只有 `prompt / image / negative_prompt / true_cfg_scale /
num_inference_steps / height / width / latents / callback_on_step_end`……
所以「涂哪改哪」在 diffusers 里没有现成入口。

但**这不等于模型做不到**。ComfyUI 的 `Qwen Image 2.1 Inpainting | No-Offset Edits`
工作流用的就是**标准 2.1 权重**（没有专用 inpaint 权重），靠两件事实现：

1. `SetLatentNoiseMask`：初始 latent 里，蒙版内换成噪声
2. 采样循环里的 `denoise_mask`：**每一步**把蒙版外拉回原 latent
   （ComfyUI `comfy/samplers.py:641-642`）
   ```
   out = out * denoise_mask + self.latent_image * latent_mask
   ```

第 2 步是**采样器级**机制，与权重无关——所以在 diffusers 里也能复刻，
用管线现成的 `latents=` 入参 + `callback_on_step_end` 钩子就够。

## 核心公式（flow matching）

蒙版外每一步要还原成「原图在当前噪声级下的样子」：

    x_t = (1 - σ) · x0 + σ · ε

- `x0`：原图的 VAE latent（管线内部已做 mean/std 归一化）
- `ε`：**固定**的那一份初始噪声（必须是同一个，否则轨迹不一致）
- `σ`：当前噪声级（`scheduler.sigmas[i+1]`，步后 latent 已降到这一级）

`σ=1` 时就是纯噪声（管线的正常起点），`σ→0` 时趋于 `x0`——模型会把蒙版外
还原成原图。

## 三个必须踩准的坑（都是实测踩出来的）

1. **pack / unpack 一定要用管线自己的 `_pack_latents` / `_unpack_latents`**。
   Qwen-Image-2.1 的 latents 是 **2×2 patchify** 过的（管线里那句
   `latents.shape[1] // 4` 就是这个意思），手写 `transpose+reshape` 会让
   空间布局错位 → 输出是满屏彩色条纹。踩过一次。
2. **条件图的 latent 必须与管线内部算的那一份逐字节一致**。管线会用
   `output_resolution` 经 `calculate_dimensions` 重算条件图尺寸（不一定是
   你传的 w×h），自己按 w×h 编码会对不上 → 同样输出彩色条纹。踩过一次。
3. **VAE 只吃 RGBA（4 通道）**。给 RGB 会报
   `expected input to have 4 channels, but got 3`。Qwen 的 VAE 是原生透明底
   设计的，不是 bug。

## 实测（本机 RTX 4090 24GB / 768² / 20 步）

- 耗时 ~15s（与普通改图同量级，没有额外开销）
- 蒙版外最大像素差 **19**（不是 0，但视觉上完全一致；剩余差异来自 VAE
  重建本身，ComfyUI 官方也说明"不保证像素级完全相同"）
- 蒙版内按指令执行（实测「盔甲改翠绿」→ 选区变绿、脸与背景不动）
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

# VAE 下采样倍率：Qwen-Image-2.1 的 vae_scale_factor = 16
VAE_SCALE = 16


def _calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    """复刻 diffusers 的 calculate_dimensions（Qwen 管线内部用的那个）。

    必须一致——管线用它决定条件图的实际尺寸，我们自己编码时也要用同一个尺寸，
    否则 latent 网格错位，输出是彩色条纹。
    """
    height = math.sqrt(target_area / ratio)
    width = math.ceil(height * ratio)
    if width % VAE_SCALE != 0:
        width = math.ceil(width / VAE_SCALE) * VAE_SCALE
    if height % VAE_SCALE != 0:
        height = math.ceil(height / VAE_SCALE) * VAE_SCALE
    return int(width), int(height)


def _encode_source_latent(pipe, cond: Image.Image, work: Tuple[int, int],
                          output_resolution: int, seed: int):
    """按管线的规则编码原图，返回 (lat0[1,C,lh,lw], latent 尺寸)。

    复刻管线内部 `prepare_latents` 的那一步，保证与 `input_images_latents`
    逐字节一致。
    """
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
        calculate_dimensions,
    )

    dev = pipe._execution_device
    dtype = pipe.transformer.dtype if hasattr(pipe.transformer, "dtype") else torch.bfloat16
    w, h = work
    iw, ih, _ = calculate_dimensions(output_resolution * output_resolution, w / h)
    vae_img = pipe.image_processor.preprocess(cond, width=iw, height=ih).unsqueeze(2)
    with torch.no_grad():
        lat0 = pipe._encode_vae_image(
            vae_img.to(dev, dtype), torch.Generator("cpu").manual_seed(seed))
    lat0 = lat0.squeeze(2)                       # [1, C, lh, lw]
    return lat0, (lat0.shape[2], lat0.shape[3])


def _mask_to_latent(mask: Image.Image, latent_size: Tuple[int, int],
                    feather: int, device, dtype) -> torch.Tensor:
    """蒙版缩到 latent 尺寸并羽化，返回 [1,1,lh,lw] 的 0~1 张量。

    羽化是必要的：硬边 mask 会在选区边界留下明显接缝。
    """
    lw, lh = latent_size[1], latent_size[0]
    m = mask.convert("L").resize((lw, lh), Image.BILINEAR)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    arr = np.asarray(m).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device, dtype)[None, None]


def masked_inpaint(pipe, *, src: Image.Image, mask: Image.Image,
                   prompt: str, negative: Optional[str] = None,
                   steps: int = 20, cfg: float = 1.0, seed: int = 0,
                   work: Optional[Tuple[int, int]] = None,
                   output_resolution: Optional[int] = None,
                   feather: int = 2) -> Image.Image:
    """ComfyUI 式 no-offset 蒙版局部重绘。

    返回与 `src` 同尺寸的 RGB 图（蒙版外与原图几乎逐点一致）。
    """
    dev = pipe._execution_device
    dtype = torch.bfloat16
    w, h = work or src.size
    # **工作尺寸有下限 768**：Qwen 的 VAE 下采样 16 倍、latent 还要再 patchify，
    # 画布太小就没有足够 token 表达蒙版边缘，"涂哪改哪"会退化成"改哪猜哪"。
    # 实测三档（同一素材、同一提示词、20 步）：
    #   512²（latent 32×32）→ 选区**没被改**（蒙版糊成一个点）
    #   768²（latent 48×48）→ 选区正确变绿，主体与背景完全保持
    #  1024²（latent 64×64）→ 同样正确，但慢 1.5 倍
    # 所以下限取 768。
    w = max(768, (int(w) // 16) * 16)
    h = max(768, (int(h) // 16) * 16)
    out_res = output_resolution or max(w, h)

    # 条件图必须是 RGBA（VAE 只吃 4 通道），透明区填白避免模型把黑块当内容
    cond = src.convert("RGBA")
    if (w, h) != cond.size:
        cond = cond.resize((w, h), Image.LANCZOS)
    bg = Image.new("RGBA", cond.size, (255, 255, 255, 255))
    cond = Image.alpha_composite(bg, cond)

    lat0, (lh, lw) = _encode_source_latent(pipe, cond, (w, h), out_res, seed)
    m_t = _mask_to_latent(mask, (lh, lw), feather, dev, dtype)

    # 初始 latent：flow matching 下 σ_0 = 1，x_1 就是纯噪声
    gen = torch.Generator(dev).manual_seed(int(seed))
    eps = torch.randn((1, lat0.shape[1], lh, lw), generator=gen,
                      device=dev, dtype=dtype)
    packed = pipe._pack_latents(eps, 1, lat0.shape[1], lh, lw)

    def _cb(pipe_, i, tstep, kwargs):
        """每步后把蒙版外拉回「原图在该噪声级下的样子」（ComfyUI denoise_mask）。"""
        lt = kwargs["latents"]
        b, seq, c = lt.shape
        # **必须用管线自己的 unpack/pack**：2.1 的 latents 是 2×2 patchify 过的，
        # 手写 transpose+reshape 会错位（输出满屏彩色条纹）。
        spatial = pipe_._unpack_latents(lt, w, h, pipe_.vae_scale_factor)
        spatial = spatial.reshape(b, c, lh, lw)
        sig = pipe_.scheduler.sigmas
        sigma = float(sig[i + 1]) if i + 1 < len(sig) else 0.0
        kept = (1 - sigma) * lat0 + sigma * eps
        spatial = spatial * m_t + kept * (1 - m_t)
        kwargs["latents"] = pipe_._pack_latents(spatial, b, c, lh, lw)
        return kwargs

    kw = dict(
        prompt=prompt,
        image=cond,
        width=w, height=h,
        num_inference_steps=max(8, min(int(steps), 60)),
        true_cfg_scale=float(cfg),
        generator=torch.Generator(dev).manual_seed(int(seed)),
        output_resolution=out_res,
        latents=packed,
        callback_on_step_end=_cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    # 真 CFG（cfg>1）才需要负向词，否则每步跑两遍 transformer
    if float(cfg) > 1.0 and negative:
        kw["negative_prompt"] = negative

    res = pipe(**kw)
    out = res.images[0].convert("RGB")
    if out.size != src.size:
        out = out.resize(src.size, Image.LANCZOS)
    return out
