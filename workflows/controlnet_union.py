"""Qwen-Image-2.1 的 ControlNet Union 分支（alibaba-pai 官方权重）。

## 这是什么

`alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union`（7.55GB）是**专为
Qwen-Image-2.1** 训的控制分支（不是 20B 那个 union）。它：

- 覆盖 **8 种结构条件**：Canny / Depth / Grayscale / HED / Lineart /
  MLSD / Pose / Scribble
- **同时支持蒙版 inpaint**：control 输入的 129 通道 =
  `control latents(64) | mask(1) | masked-image latents(64)`
- 只含 **control 分支**（`control_img_in` + 16 个 `control_blocks`，~7.0GB），
  用 `strict=False` 加载到 base transformer 上，base 保持冻结
- 每 **2 层**注入一次（`control_layers = [0, 2, 4, …, 30]`，16 个注入点）

## 它是双流（VACE 风格），不是「一次性跳连」

这一点是最容易做错的（我们第一版就错了，症状是「8 步勉强能用、24 步必崩」）。
官方 ComfyUI 实现（PR #16519，`comfy/ldm/qwen_image21/model.py`）：

```python
def init_stream(self, x, control, prefix_len):
    # control latents fill the target rows, text and reference rows stay zero
    joint = torch.zeros_like(x)
    joint[:, prefix_len:] = self.control_img_in(control)
    return self.control_blocks[0].before_proj(joint) + x

def step(self, index, c, mod, pe, attn_fn, ...):
    c = block(c, mod, pe, attn_fn, ...)      # 完整 block，吃 modulation
    return c, block.after_proj(c)
```

两个含义：

1. **control block 是 base block 的完整副本**（`QwenImage21FunControlBlock`
   直接继承 `QwenImage21TransformerBlock`），吃 timestep 调制（`modulation`）
   和 rotary embedding。参数上没有 norm 权重，是因为 base 的 `img_norm1/2`
   是 `elementwise_affine=False`（无参），modulation 复用 base 的。
2. **每一步都重跑整条 control 流**，所以 skip 随 timestep 变化。

如果按「算一次、整条采样链复用」做，skip 就成了常量偏置：每步叠加一次，
低噪声后期它相对占比越来越大，最后变成「照抄条件图」——实测 24 步 / scale 0.6
直接出骨架本身的彩色线段。改成双流后这个累积效应消失。

## 为什么能自己实现（不用 VideoX-Fun）

权重是**纯 safetensors**，结构能从 key 名和形状完全推出来。数量对账：

```
16 blocks × 9 key (attn 6 + img_mlp 3) = 144
16 blocks × 2 key (after_proj)         =  32
 1 block  × 2 key (before_proj，仅 0)  =   2
control_img_in                          =   2
                                   合计 = 180  ← 与官方逐 key 吻合
```

所以用 diffusers 的 `QwenImage21TransformerBlock` **继承**（不是包一层，
否则 key 会多出 `block.` 前缀）再挂 `before_proj` / `after_proj` 即可，
**不需要引 VideoX-Fun 的代码**。

## 三个条件怎么组装（129 通道）

```
control_in = cat([ control_latents(64),        # 条件图的 VAE latent（已归一化）
                   mask(1),                    # inpaint 蒙版；纯控制时填 0（全重画）
                   masked_image_latents(64) ], # 被遮住的图；纯控制时全 0
                 dim=1)                        # → 129
```

- **纯结构控制**（pose/canny/depth）：只给 `control`，mask 与 masked-image
  都填 **0**（= 全部重画，这是官方 zero-pad 的真实含义）。
  **别填 1**：1 表示"保留原图"，配上空的 masked-image 就等于要求模型
  输出一张空图（实测 RGBA 不透明占比 0.0%，整张全透明）。
- **inpaint**：control = 原图，mask = 涂抹区（白=重画，内部会转成 0），
  masked-image = 原图（区外置灰）
- **两者可叠加**：同时给控制图和 mask，重绘区会同时听提示词和结构

## 有 control 时必须关 KV cache

pipeline 的 `use_kv_cache` 默认 True，它会把文本 token 当 prefix 缓存，
之后每个 block 只处理图像 token（768² 下 2304 个），而 control 链算的是
完整 joint（2331 = 2304 图像 + 27 文本）—— 两者对不上，hint 只能硬塞进
图像段，信号全错。**调用方要显式传 `use_kv_cache=False`**
（官方实现里就有这个判断：`cache_enabled = ... and control_context is None`）。

## 显存（实测 768² / RTX 4090 24GB）

base transformer fp8 走 cpu offload，control 分支 bf16 常驻 7.05GB。
生成时 reserved 7.4GB、peak 18.7GB，余量约 5GB。24G 卡够用，
**不需要**把 control 分支也 fp8 化。

## 必须踩准的坑

1. **条件 latent 要做 mean/std 归一化**：管线的 `_encode_vae_image` 是
   `(x - latents_mean) / latents_std`。漏掉这步，尺度差几倍，
   16 层 skip 直接把 hidden 打成 NaN，出图**全黑**。
2. **`init_stream` 要加回 base 的 hidden**（`+ x`）：只把 control 投影进去
   会让控制流从零起步，与 base 的表示脱节。
3. **注入用 `prepend=True`**（torch≥2.6）：抢在 accelerate 的 cpu-offload
   post-hook 之前执行，此时 output 还在 GPU 上，加法不用搬设备。
4. **控制块要继承而不是组合**：包一层会让 key 多出 `block.` 前缀。
"""

from __future__ import annotations

import torch
from torch import nn

#: 注入点：每 2 个 transformer block 注入一次（32 层 → 16 个）
CONTROL_LAYERS = list(range(0, 32, 2))

#: control 输入通道数 = 64 控制 latent + 1 mask + 64 遮罩图 latent
CONTROL_IN_DIM = 129


def _block_class():
    """动态构造「完整 Transformer block + before_proj / after_proj」的类。

    继承而不是组合：组合会让 state_dict 的 key 变成 `block.attn.to_q.weight`，
    多出 `block.` 前缀，官方权重就对不上。
    """
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21TransformerBlock,
    )

    class ControlBlock(QwenImage21TransformerBlock):
        def __init__(self, dim=4096, num_heads=32, head_dim=128,
                     mlp_ratio=3, eps=1e-6, first=False):
            super().__init__(dim, num_heads, head_dim, mlp_ratio, eps)
            self.after_proj = nn.Linear(dim, dim)
            # **只有 block 0 有 before_proj**（官方权重就是这样）
            self.before_proj = nn.Linear(dim, dim) if first else None

    return ControlBlock


class ControlNetUnionBranch(nn.Module):
    """ControlNet Union 的完整控制分支（双流，VACE 风格）。

    用法（由 `ControlInjection` 驱动，**每个采样步都要跑**）：

    ```python
    branch = ControlNetUnionBranch()
    branch.load_state_dict(sd, strict=True)      # 180 key 全对
    c = branch.init_stream(x, control_tokens)    # 步初：建控制流
    c, skip = branch.step(0, c, mod, pe)         # 每层推进一次
    hidden = hidden + skip
    ```
    """

    def __init__(self, dim: int = 4096, num_heads: int = 32,
                 head_dim: int = 128, mlp_ratio: int = 3, eps: float = 1e-6,
                 control_in_dim: int = CONTROL_IN_DIM,
                 num_blocks: int = len(CONTROL_LAYERS)):
        super().__init__()
        self.control_img_in = nn.Linear(control_in_dim, dim)
        block_cls = _block_class()
        self.control_blocks = nn.ModuleList([
            block_cls(dim, num_heads, head_dim, mlp_ratio, eps, first=(i == 0))
            for i in range(num_blocks)
        ])
        self.dim = dim

    # ---------------- 双流的两个动作 ----------------

    def init_stream(self, x: torch.Tensor, control_tokens: torch.Tensor,
                    image_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """步初：把 control 塞进 joint 序列的**图像位置**，再与 base hidden 相加。

        `x`              : [B, L, dim]   base 的 joint_hidden_states（文本+图像）
        `control_tokens` : [B, HW, 129]  control 条件，已展平成图像 token 数
        `image_pad_mask` : [L] bool，标记 joint 序列里哪些位置是图像 token

        **必须用 mask 散射，不能用前缀切片**。diffusers 里那段注释写得很直白：

        > Text positions of the joint sequence line up, in order, with the
        > non-image positions of the vision-language sequence — the two are
        > **interleaved**, so the mask **cannot be sliced off as a prefix**.

        按 `j[:, n:]` 猜位置会把 control 铺到错的位置上，模型收到的信号全乱，
        表现是"照着条件图画线条人"或干脆输出一张空图（都已实测）。
        """
        b, l, _ = x.shape
        n = control_tokens.shape[1]
        features = self.control_img_in(control_tokens.to(x.dtype)).to(x.dtype)

        joint = torch.zeros_like(x)
        if image_pad_mask is not None:
            mask = image_pad_mask.to(x.device)
            if mask.shape[0] != l:
                raise RuntimeError(
                    f"image_pad_mask 长度 {mask.shape[0]} 与 joint 长度 {l} 不一致；"
                    f"control token 数 {n}，mask 里 True 有 {int(mask.sum())}")
            got = int(mask.sum())
            if got != n:
                raise RuntimeError(
                    f"图像位置 {got} 与 control token 数 {n} 不一致："
                    f"条件图分辨率必须与目标 latent 对齐")
            joint[:, mask] = features
        else:
            # 兜底（拿不到 img_mask 时）：按"文本在前、图像在后"铺。纯文生图
            # 单条件图时成立，但多图/参考图场景会错位，所以只当兜底。
            if n > l:
                raise RuntimeError(
                    f"control token 数 {n} 多于 joint 序列长度 {l}："
                    f"条件图比目标分辨率大")
            joint[:, l - n:] = features
        return self.control_blocks[0].before_proj(joint) + x

    def step(self, index: int, c: torch.Tensor, modulation: torch.Tensor,
             rotary_emb: torch.Tensor | None = None,
             **kw) -> tuple[torch.Tensor, torch.Tensor]:
        """推进一层控制流，返回 (新的控制流, 该层的 skip)。

        `modulation` / `rotary_emb` 与 base block 收到的是**同一份**
        （transformer 里所有 block 共用），所以控制流也在同样的
        timestep 条件与位置编码下演化——这正是它能跟上采样轨迹的原因。
        """
        blk = self.control_blocks[index]
        c = blk(hidden_states=c, modulation=modulation,
                rotary_emb=rotary_emb,
                attention_mask=kw.get("attention_mask"),
                target_token_mask=kw.get("target_token_mask"),
                key_valid=kw.get("key_valid"))
        return c, blk.after_proj(c)


def expected_keys(dim: int = 4096, num_blocks: int = 16,
                  control_in_dim: int = CONTROL_IN_DIM) -> set[str]:
    """这个模块**应该**产出的 state_dict key 集合（与官方 180 key 对账用）。"""
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

def load_branch(path: str, *, dtype=torch.bfloat16,
                device: str = "cpu") -> ControlNetUnionBranch:
    """加载官方 control 分支权重。

    `strict=True` 能成功就是结构完全对齐的证明。
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

    - **纯结构控制**（pose/canny/depth）：只给 `control`，mask 默认全 1
    - **inpaint**：`control` = 原图，`mask` = 涂抹区（白=重画），
      `masked_image` = 原图（区外置灰）
    - **两者叠加**：都给，重绘区同时听提示词和结构
    """
    import numpy as np

    w, h = size

    def _encode(img: "Image.Image | None") -> torch.Tensor:
        if img is None:
            # 官方语义是 zero-pad：填充**归一化空间**的 0（不是像素空间的 0）
            return torch.zeros((1, 64, h // 16, w // 16), device=device, dtype=dtype)
        im = img.convert("RGBA").resize(
            (w, h), __import__("PIL.Image", fromlist=["Image"]).LANCZOS)
        t = torch.from_numpy(np.array(im, copy=True)).float() / 127.5 - 1.0
        t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device, dtype)
        with torch.no_grad():
            enc = vae.encode(t)          # **只调一次**（重复调用会白算一遍编码）
            out = enc.latent_dist.mode() if hasattr(enc, "latent_dist") else enc
        if not isinstance(out, torch.Tensor):
            out = out[0] if isinstance(out, (tuple, list)) else out.latents
        if out.dim() == 5:               # Qwen VAE 是视频式 5 维 [B,C,F,H,W]
            out = out.squeeze(2)
        # **必须做 mean/std 归一化**：管线的 `_encode_vae_image` 就是
        # `(x - latents_mean) / latents_std`（pipeline_qwenimage21.py:448）。
        # 漏掉这一步，尺度差好几倍，出图全黑（已实测踩过）。
        mean = torch.tensor(vae.config.latents_mean).view(
            1, -1, 1, 1).to(out.device, out.dtype)
        std = torch.tensor(vae.config.latents_std).view(
            1, -1, 1, 1).to(out.device, out.dtype)
        return (out - mean) / std

    ctrl = _encode(control)
    masked = _encode(masked_image)

    if mask is None:
        # **纯结构控制必须填 0（全重画）**。
        #
        # 语义来自官方 pipeline：`mask_latent = 1 - mask_condition`，其中
        # `mask_condition` 白=重画。所以这一通道 **1 = 保留原图，0 = 重画**。
        # 填 1 等于告诉模型「整张图保持 `masked_image`」，而纯结构控制时
        # masked_image 是空的 → 模型老老实实输出一张空图。
        #
        # 这个坑很隐蔽：填 1 时"与对照组的像素差"反而更大（40→80），看着像
        # "控制力翻倍"，其实是画面在往空白塌；继续加强度到 1.0 就得到一张
        # 全透明 PNG（实测不透明占比 0.0%）。度量指标在这里会骗人。
        m = torch.zeros((1, 1, h // 16, w // 16), device=device, dtype=dtype)
    else:
        # 显式传入的 mask 按官方约定：白=重画 → 转成 1=保留
        mimg = mask.convert("L").resize((w // 16, h // 16), 1)
        m = 1.0 - torch.from_numpy(np.array(mimg)).float().to(
            device, dtype) / 255.0
        m = m[None, None]

    return torch.cat([ctrl, m, masked], dim=1)   # [1, 129, H/16, W/16]


class ControlInjection:
    """把双流控制分支接到 base transformer 上（forward hook，不改 diffusers 源码）。

    ```python
    inj = ControlInjection(transformer, branch, scale=1.0)
    inj.set_condition(control_latents)     # [1, 129, H/16, W/16]
    ... 跑 pipe(...) ...
    inj.remove()
    ```

    **每一步都会重跑整条控制流**（控制流随 timestep 演化），
    所以不再有「skip 是常量、步数越多累积越狠」的问题。
    """

    def __init__(self, transformer, branch: ControlNetUnionBranch,
                 scale: float = 1.0):
        self.transformer = transformer
        self.branch = branch
        self.scale = float(scale)
        self._handles: list = []
        self._tokens: torch.Tensor | None = None
        self._stream: torch.Tensor | None = None
        #: 诊断：(层号, hidden 绝对值均值, skip 绝对值均值)，只记前 8 次
        self.trace: list[tuple[int, float, float]] = []
        #: 诊断：每个采样步「首层 skip」的绝对值均值——双流的证据：
        #: 它应当随步变化；旧的一次性实现这里是恒定值。
        self.step_trace: list[float] = []
        #: 注入开关：采样过程中可以按步关掉
        self.enabled = True
        #: 诊断：与 hidden 设备不一致的次数（prepend 没生效时会涨）
        self._device_mismatch = 0
        #: 图像位置 mask（由 transformer 的 pre-hook 每步更新）
        self._image_pad_mask: torch.Tensor | None = None
        self._tr_handle = None
        # 注入点：每 2 个 block（与官方 control_layers = [0,2,...,30] 一致）
        self.layers = [i for i in CONTROL_LAYERS
                       if i < len(transformer.transformer_blocks)]
        self._install()
        self._install_mask_probe()

    def set_condition(self, control_in: torch.Tensor) -> None:
        """登记条件（每步复用同一份条件，但控制流每步重建）。

        `control_in`：[B, 129, H, W] → 展平成 [B, HW, 129] 供 `control_img_in`。
        """
        b, c, h, w = control_in.shape
        # (h,w) 顺序的每个空间位置作为一行，与 base 的图像 token 顺序一致
        self._tokens = control_in.reshape(b, c, h * w).transpose(1, 2).contiguous()

    def _install_mask_probe(self) -> None:
        """从 transformer 的入参里抠出 `img_mask`，复算出 joint 序列的图像位置。

        diffusers 内部就是这么算的（`transformer_qwenimage21.py:911`）：

        ```python
        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        ```

        一个图像 slot 展开成 4 个 token，文本 slot 保持 1 个，所以两者是**交错**
        的，不能按前缀切。
        """
        from diffusers.models.transformers import transformer_qwenimage21 as _t

        per_slot = getattr(_t, "_IMG_TOKENS_PER_SLOT", 4)

        def _pre(module, args, kwargs):
            im = kwargs.get("img_mask")
            if im is None:
                return None
            if not torch.is_tensor(im):
                return None
            try:
                im2 = im if im.dim() == 2 else im[None]
                repeats = torch.where(im2, per_slot, 1)[0]
                self._image_pad_mask = torch.repeat_interleave(
                    im2[0], repeats).to(im.device)
            except Exception:                 # noqa: BLE001
                self._image_pad_mask = None
            return None

        try:
            self._tr_handle = self.transformer.register_forward_pre_hook(
                _pre, with_kwargs=True)
        except TypeError:
            self._tr_handle = None

    def _install(self) -> None:
        blocks = self.transformer.transformer_blocks
        for k, layer_idx in enumerate(self.layers):
            # `_li`：layer_idx 是循环变量，必须绑成默认参数，否则闭包捕获
            # 的是循环结束后的值（诊断里所有层会打印成 30）。
            def _hook(module, args, kwargs, output, _k=k, _li=layer_idx):
                if self._tokens is None or not self.enabled:
                    return output
                x = kwargs.get("hidden_states")
                if x is None or not torch.is_tensor(x):
                    x = args[0] if args else None
                mod = kwargs.get("modulation")
                if not torch.is_tensor(x) or mod is None:
                    return output
                if self._tokens.device != x.device:
                    self._tokens = self._tokens.to(x.device)
                if _k == 0:
                    # 每个采样步从这里重建控制流
                    self._stream = self.branch.init_stream(
                        x, self._tokens, self._image_pad_mask)
                self._stream, skip = self.branch.step(
                    _k, self._stream, mod, kwargs.get("rotary_emb"),
                    attention_mask=kwargs.get("attention_mask"),
                    target_token_mask=kwargs.get("target_token_mask"),
                    key_valid=kwargs.get("key_valid"))
                if len(self.trace) < 8:
                    self.trace.append((int(_li),
                                       float(output.abs().mean()),
                                       float(skip.abs().mean())))
                if _k == 0:
                    self.step_trace.append(float(skip.abs().mean()))
                if skip.device != output.device:
                    self._device_mismatch += 1
                    skip = skip.to(output.device)
                return output + skip.to(output.dtype) * self.scale

            # `prepend=True`（torch≥2.6）：抢在 accelerate 的 cpu-offload
            # post-hook 之前执行，此时 output 还在 GPU 上，加法不用搬设备。
            try:
                h = blocks[layer_idx].register_forward_hook(
                    _hook, prepend=True, with_kwargs=True)
            except TypeError:                 # 老 torch 没有 prepend/with_kwargs
                h = blocks[layer_idx].register_forward_hook(_hook)
            self._handles.append(h)

    def remove(self) -> None:
        """卸掉所有 hook（一定要在生成结束后调，否则会污染后续无控制的生成）。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._tr_handle is not None:
            self._tr_handle.remove()
            self._tr_handle = None
        self._stream = None
