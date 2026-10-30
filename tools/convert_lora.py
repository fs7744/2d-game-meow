"""把 ComfyUI 格式的 Qwen-Image LoRA 转成 diffusers 原生格式。

## 为什么需要

Qwen-Image-2.1 的 LoRA 生态大半发在 **ComfyUI 格式**（Comfy-Org、
ausboss、lilylilith 等），而 diffusers 的 `pipe.load_lora_weights()` 只认
**diffusers 原生格式**。两者键名规则不同：

| | ComfyUI | diffusers |
|---|---|---|
| 前缀 | `diffusion_model.` | `transformer.` |
| 下投影 | `.lora_A.weight` | `.lora.down.weight` |
| 上投影 | `.lora_B.weight` | `.lora.up.weight` |

**A/B 的顺序也不能弄反**：ComfyUI 的 `lora_A` 是降维（[rank, in]）、
`lora_B` 是升维（[out, rank]）；diffusers 的 `down` 是降维、`up` 是升维。
所以是 `A→down`、`B→up`，**同名映射，不是交换**。

对照参考（本机实测的头部）：
- ComfyUI：`diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight` [16, 4096]
- diffusers：`transformer.transformer_blocks.0.attn.to_k.lora.down.weight` [16, 4096]

## `img_mlp.gate_up` 必须拆开（2026-10-31 修）

ComfyUI 把 SwiGLU 的 gate 和 up 合成**一个** Linear：

```python
gate, up = self.gate_up(x).chunk(2, dim=-1)      # 权重 [2*intermediate, hidden]
out(silu(gate) * up)
```

diffusers 是**两个独立的** Linear：

```python
out(silu(gate_layer(x)) * proj(x))                # gate_layer / proj 各 [intermediate, hidden]
```

键名对不上，直接按名字映射会让这 64 个键（32 块 × down/up）**静默丢失**——
`load_lora_weights` 只打一句 "unexpected keys found in the model" 就继续了。
实测本机三个 LoRA 各丢 17% 权重：`qwen21_anyangle`、`qwen21_outpaint`、
`vnccs_pose`（64/384）。

拆法是**精确**的，不是近似：LoRA 的 `up`（B，形状 [2*intermediate, rank]）
按行分块，前一半是 gate、后一半是 up（对应 `chunk(2, dim=-1)`）；`down`（A，
形状 [rank, hidden]）两边**共用同一份**，复制成两个键即可。

## 用法

```bash
# 从 ComfyUI 原始文件转
python tools/convert_lora.py <in.safetensors> <out.safetensors>

# 修补"已经转过、但 gate_up 被漏掉"的文件（输入输出可同名）
python tools/convert_lora.py models/lora/vnccs_pose.safetensors \
                            models/lora/vnccs_pose.safetensors
```

脚本同时接受 `diffusion_model.` 与 `transformer.` 两种前缀，所以对已转换过的
文件重跑一遍就能把漏掉的那部分补回去。

跑法：python tools/convert_lora.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

#: ComfyUI 合并版 SwiGLU 的模块名
GATE_UP = ".img_mlp.gate_up."

#: diffusers 里对应的两个模块：`chunk(2, dim=-1)` 的前一半是 gate
SPLIT_TARGETS = ("gate_layer", "proj")


def _strip_prefix(k: str) -> str | None:
    """去掉 ComfyUI / diffusers 的前缀，只留主体。"""
    if k.startswith("diffusion_model."):
        return k[len("diffusion_model."):]
    if k.startswith("transformer."):
        return k[len("transformer."):]
    return None


def _split_gate_up(body: str, v, is_down: bool) -> list[tuple[str, object]]:
    """把 `img_mlp.gate_up` 的一对张量拆成 gate_layer / proj 两组。"""
    head = body.split(GATE_UP)[0]            # transformer_blocks.N
    if is_down:
        # A 两边共用，但**必须 clone**：safetensors 不允许两个键指向同一块内存
        # （会报 "Some tensors share memory"），而且共享内存写盘后容易被后续
        # 修改串味。多出来的体积是 32 块 × rank32 × 4096 × 2B ≈ 16MB，可以接受。
        return [(f"transformer.{head}.img_mlp.{t}.lora.down.weight", v.clone())
                for t in SPLIT_TARGETS]
    if v.shape[0] % 2:
        raise ValueError(
            f"gate_up 的 up 张量第一维 {v.shape[0]} 不是偶数，无法对半分：{body}")
    half = v.shape[0] // 2
    return [
        (f"transformer.{head}.img_mlp.gate_layer.lora.up.weight",
         v[:half].contiguous()),
        (f"transformer.{head}.img_mlp.proj.lora.up.weight",
         v[half:].contiguous()),
    ]


def convert_weights(sd: dict) -> tuple[dict, list[str]]:
    """返回 (转换结果, 跳过的键)。"""
    out: dict = {}
    skipped: list[str] = []
    for k, v in sd.items():
        body = _strip_prefix(k)
        if body is None:
            skipped.append(k)
            continue
        is_down = ".lora_A.weight" in body or ".lora.down.weight" in body
        is_up = ".lora_B.weight" in body or ".lora.up.weight" in body
        if not (is_down or is_up):
            # ComfyUI 有些 LoRA 另带 alpha 标记，diffusers 不需要
            skipped.append(k)
            continue

        if GATE_UP in body:
            for nk, nv in _split_gate_up(body, v, is_down):
                out[nk] = nv
            continue

        if is_down:
            nk = "transformer." + body.replace(".lora_A.weight", ".lora.down.weight") \
                                    .replace(".lora.down.weight", ".lora.down.weight")
        else:
            nk = "transformer." + body.replace(".lora_B.weight", ".lora.up.weight") \
                                    .replace(".lora.up.weight", ".lora.up.weight")
        out[nk] = v
    return out, skipped


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])

    sd = load_file(str(src))
    out, skipped = convert_weights(sd)

    if not out:
        print(f"❌ 没有可转换的键：{src} 看起来不是 ComfyUI 格式的 Qwen LoRA")
        print(f"   头几个键：{list(sd)[:5]}")
        return 1

    n_gate = sum(1 for k in out if "gate_layer" in k or (
        ".img_mlp.proj." in k and ".lora." in k))
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst), metadata={"format": "pt"})

    print(f"✅ {src.name} → {dst.name}")
    print(f"   转换 {len(out)} 个张量（其中 img_mlp 拆分后 {n_gate} 个），"
          f"跳过 {len(skipped)} 个")
    if skipped:
        print(f"   跳过示例：{skipped[:3]}")
    for k, v in out.items():
        if ".lora.down." in k:
            print(f"   抽查 down 形状 {tuple(v.shape)}（应为 [rank, in_features]）")
            break
    for k, v in out.items():
        if ".lora.up." in k:
            print(f"   抽查 up   形状 {tuple(v.shape)}（应为 [out_features, rank]）")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
