"""Qwen-Image-2.1 的 ControlNet Union 分支（alibaba-pai 官方权重）。

## 这是什么

`alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union`（7.2GB）是**专为
Qwen-Image-2.1** 训的控制分支（不是 20B 那个 union）。它：

- 覆盖 **8 种结构条件**：Canny / Depth / Grayscale / HED / Lineart /
  MLSD / Pose / Scribble
- **同时支持蒙版 inpaint**：control 输入的 129 通道 =
  `control latents(64) | mask(1) | masked-image latents(64)`
- 只含 **control 分支**（`control_img_in` + 16 个 `control_blocks`，~7.0GB），
  用 `strict=False` 加载到 base transformer 上，base 保持冻结
- 每 **2 层**注入一次（`control_layers = [0, 2, 4, …, 30]`，16 个注入点），
  通过零初始化的 `before_proj` / `after_proj` 加回主分支

## 为什么能自己实现（不用 VideoX-Fun）

官方的参考实现是 VideoX-Fun 的 `config/qwenimage21/qwenimage21_control.yaml`，
但权重本身是**纯 safetensors**，结构能从 key 名和形状完全推出来：

| key | 形状 | 说明 |
|---|---|---|
| `control_img_in.weight` | [4096, 129] | 129 通道 → hidden |
| `control_blocks.N.attn.to_{q,k,v,out.0}.weight` | [4096, 4096] | 自注意力 |
| `control_blocks.N.attn.norm_{q,k}.weight` | [128] | q/k 的 RMSNorm（head_dim） |
| `control_blocks.N.img_mlp.{proj,gate_layer}.weight` | [12288, 4096] | SwiGLU 上投影 |
| `control_blocks.N.img_mlp.out.weight` | [4096, 12288] | SwiGLU 下投影 |
| `control_blocks.N.before_proj.{weight,bias}` | [4096, 4096] / [4096] | **只有 block 0 有** |
| `control_blocks.N.after_proj.{weight,bias}` | [4096, 4096] / [4096] | 每块都有（零门控） |

3 个关键对齐点（都验证过）：
1. `img_mlp` 的 `proj/gate_layer/out` 命名与 diffusers 的
   `QwenImage21SwiGLUFeedForward` **完全一致**（不是 `gate_up` 那种合并式）
2. `12288 = 4096 × 3` → `mlp_ratio = 3.0`
3. `norm_q/norm_k` 的形状是 `[128]` = `attention_head_dim`，不是 `[4096]`

所以直接用 diffusers 的 `QwenImage21Attention` + `QwenImage21SwiGLUFeedForward`
拼出来即可，**不需要引 VideoX-Fun 的代码**。

## 三个条件怎么组装（129 通道）

```
control_in = cat([ control_latents(64),        # 条件图的 VAE latent
                   mask(1),                    # inpaint 蒙版；纯控制时全 0
                   masked_image_latents(64) ], # 被遮住的图；纯控制时全 0
                 dim=1)                        # → 129
```

- **纯 pose / canny / depth 控制**：mask 与 masked-image 都填 0
  （官方文档：`the pipeline zero-pads the mask / masked-image channels`）
- **inpaint**：control=原图 latent，mask=涂抹区，masked-image=原图（区外置灰）
- **两者可叠加**：同时给控制图和 mask，重绘区会同时听提示词和结构

## 显存

base transformer 我们走 fp8（6.7GB）+ control 分支 7.2GB(bf16) ≈ 14GB，
再加 VAE 与激活——24G 卡上要确认。**这是接进来前必须实测的第一件事。**
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

#: 注入点：每 2 个 transformer block 注入一次（32 层 → 16 个）
CONTROL_LAYERS = list(range(0, 32, 2))

#: control 输入通道数 = 64 控制 latent + 1 mask + 64 遮罩图 latent
CONTROL_IN_DIM = 129


class ControlBlock(nn.Module):
    """一个控制块：自注意力 + SwiGLU FFN。

    刻意复用 diffusers 的组件（`QwenImage21Attention` / `QwenImage21SwiGLUFeedForward`），
    因为它们的参数命名与官方权重**逐字对应**——自己写反而容易在
    `to_out.0` 还是 `to_out`、`gate_layer` 还是 `gate_up` 上错位。
    """

    def __init__(self, dim: int = 4096, num_heads: int = 32,
                 head_dim: int = 128, mlp_ratio: float = 3.0, eps: float = 1e-6,
                 with_before_proj: bool = False):
        super().__init__()
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21Attention, QwenImage21SwiGLUFeedForward,
        )

        self.attn = QwenImage21Attention(
            dim=dim, heads=num_heads, dim_head=head_dim, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(
            hidden_size=dim, mlp_hidden_size=int(dim * mlp_ratio))
        self.after_proj = nn.Linear(dim, dim)
        # **只有 block 0 有 before_proj**（官方权重就是这样，不是每块都有）
        self.before_proj = nn.Linear(dim, dim) if with_before_proj else None


class ControlNetUnionBranch(nn.Module):
    """ControlNet Union 的完整控制分支（16 块 + 输入投影）。

    用法：
    ```python
    branch = ControlNetUnionBranch()
    branch.load_state_dict(sd, strict=True)
    branch.eval()

    # 在 base transformer 的 forward 里，每 2 层注入一次：
    #   hidden = hidden + branch(control_latents, mask, masked_latents)[i]
    ```
    """

    def __init__(self, dim: int = 4096, num_heads: int = 32,
                 head_dim: int = 128, mlp_ratio: float = 3.0,
                 control_in_dim: int = CONTROL_IN_DIM,
                 num_blocks: int = len(CONTROL_LAYERS)):
        super().__init__()
        self.control_img_in = nn.Linear(control_in_dim, dim)
        self.control_blocks = nn.ModuleList([
            ControlBlock(dim, num_heads, head_dim, mlp_ratio,
                         with_before_proj=(i == 0))
            for i in range(num_blocks)
        ])
        self.dim = dim

    def forward(self, control_in: torch.Tensor) -> list[torch.Tensor]:
        """`control_in`：[B, 129, H, W]（latent 空间）→ 16 个 skip 张量。

        每个 skip 的形状与 base transformer 的 hidden_states 一致
        （[B, H*W, dim]），由调用方在对应层 `hidden = hidden + skip`。
        """
        b, c, h, w = control_in.shape
        # 通道优先展平成 token 序列：[B, 129, H*W] → [B, H*W, 129]
        tokens = control_in.reshape(b, c, h * w).transpose(1, 2)
        x = self.control_img_in(tokens)
        first = self.control_blocks[0]
        if first.before_proj is not None:
            x = x + first.before_proj(x)

        skips: list[torch.Tensor] = []
        for blk in self.control_blocks:
            x = x + blk.attn(x)
            x = x + blk.img_mlp(x)
            skips.append(blk.after_proj(x))
        return skips


def expected_keys(dim: int = 4096, num_blocks: int = 16,
                  control_in_dim: int = CONTROL_IN_DIM) -> set[str]:
    """这个模块**应该**产出的 state_dict key 集合。

    与官方权重的 key 对比、或做 `load_state_dict(strict=True)` 前的自检。
    """
    keys = {"control_img_in.weight", "control_img_in.bias"}
    for i in range(num_blocks):
        p = f"control_blocks.{i}."
        keys |= {
            p + "after_proj.weight", p + "after_proj.bias",
            p + "attn.norm_q.weight", p + "attn.norm_k.weight",
            p + "attn.to_q.weight", p + "attn.to_k.weight",
            p + "attn.to_v.weight", p + "attn.to_out.0.weight",
            p + "img_mlp.proj.weight", p + "img_mlp.gate_layer.weight",
            p + "img_mlp.out.weight",
        }
    # before_proj 只有 block 0 有
    keys |= {"control_blocks.0.before_proj.weight",
             "control_blocks.0.before_proj.bias"}
    return keys


# ---------------- 接入 diffusers 的 transformer ----------------

def load_branch(path: str, *, dtype=torch.bfloat16, device: str = "cpu") -> ControlNetUnionBranch:
    """加载官方 control 分支权重。

    `strict=True` 能成功就是结构完全对齐的证明（我们已用随机初始化验证过
    180 个 key 逐一对应）。
    """
    from safetensors.torch import load_file

    sd = load_file(path)
    branch = ControlNetUnionBranch()
    missing, unexpected = branch.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"control 分支权重不匹配：缺 {len(missing)} 个、多 {len(unexpected)} 个。"
            f"前几个：缺={list(missing)[:3]} 多={list(unexpected)[:3]}")
    return branch.to(device=device, dtype=dtype).eval()


def encoder_condition(*, control: "Image.Image | None" = None,
                      mask: "Image.Image | None" = None,
                      masked_image: "Image.Image | None" = None,
                      vae, size: tuple[int, int],
                      device, dtype=torch.bfloat16) -> torch.Tensor:
    """把条件图拼成 control 分支要的 129 通道 latent。

    `129 = control(64) | mask(1) | masked-image(64)`

    - **纯结构控制**（pose/canny/depth…）：只给 `control`，mask 与 masked-image 全 0
      （官方文档：zero-pads the mask / masked-image channels）
    - **inpaint**：`control` = 原图，`mask` = 涂抹区（白=重画），
      `masked_image` = 原图（区外置灰）
    - **两者叠加**：都给，重绘区同时听提示词和结构
    """
    import numpy as np

    w, h = size

    def _encode(img: "Image.Image | None") -> torch.Tensor:
        if img is None:
            return torch.zeros((1, 64, h // 16, w // 16), device=device, dtype=dtype)
        im = img.convert("RGBA").resize((w, h), __import__("PIL.Image", fromlist=["Image"]).LANCZOS)
        t = torch.from_numpy(np.asarray(im)).float() / 127.5 - 1.0
        t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device, dtype)
        with torch.no_grad():
            enc = vae.encode(t)          # **只调一次**（重复调用会白算一遍编码）
            out = enc.latent_dist.mode() if hasattr(enc, "latent_dist") else enc
        if not isinstance(out, torch.Tensor):
            out = out[0] if isinstance(out, (tuple, list)) else out.latents
        if out.dim() == 5:               # Qwen VAE 是视频式 5 维 [B,C,F,H,W]
            out = out.squeeze(2)
        return out

    ctrl = _encode(control)
    masked = _encode(masked_image)

    if mask is None:
        m = torch.zeros((1, 1, h // 16, w // 16), device=device, dtype=dtype)
    else:
        mimg = mask.convert("L").resize((w // 16, h // 16), 1)
        m = torch.from_numpy(np.asarray(mimg)).float().to(device, dtype) / 255.0
        m = m[None, None]

    return torch.cat([ctrl, m, masked], dim=1)   # [1, 129, H/16, W/16]


class ControlInjection:
    """把 control skips 注入 base transformer 的指定层（用 forward hook，不改源码）。

    用法：
    ```python
    inj = ControlInjection(transformer, branch, scale=1.0)
    inj.set_condition(control_latents)     # [1, 129, H/16, W/16]
    ... 跑 pipe(...) ...
    inj.remove()
    ```
    """

    def __init__(self, transformer, branch: ControlNetUnionBranch,
                 scale: float = 1.0):
        self.transformer = transformer
        self.branch = branch
        self.scale = float(scale)
        self._handles: list = []
        self._skips: list[torch.Tensor] | None = None
        # 注入点：每 2 个 block（与官方 control_layers = [0,2,...,30] 一致）
        self.layers = [i for i in CONTROL_LAYERS
                       if i < len(transformer.transformer_blocks)]
        self._install()

    def set_condition(self, control_in: torch.Tensor) -> None:
        """算一次 skips（**不依赖 timestep**，所以整条采样链复用同一份，省算力）。"""
        with torch.no_grad():
            self._skips = self.branch(control_in.to(self.branch.control_img_in.weight.dtype))

    def _install(self) -> None:
        blocks = self.transformer.transformer_blocks
        for k, layer_idx in enumerate(self.layers):
            def _hook(module, inputs, output, _k=k):
                if self._skips is None:
                    return output
                skip = self._skips[_k]
                # 形状防御：skip 的 token 数必须与 hidden 一致，否则说明
                # control 条件图的分辨率与目标 latent 不一致（会静默错位）
                if skip.shape[1] != output.shape[1]:
                    raise RuntimeError(
                        f"control skip 的 token 数 {skip.shape[1]} 与第 {layer_idx} 层 "
                        f"hidden 的 {output.shape[1]} 不一致：条件图尺寸要与目标 latent 对齐")
                return output + skip.to(output.dtype) * self.scale

            self._handles.append(blocks[layer_idx].register_forward_hook(_hook))

    def remove(self) -> None:
        """卸掉所有 hook（一定要在生成结束后调，否则会污染后续无控制的生成）。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._skips = None
