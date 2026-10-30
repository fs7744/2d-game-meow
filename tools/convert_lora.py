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

## 用法

```bash
python tools/convert_lora.py <in.safetensors> <out.safetensors>
```

转换后就能直接：
```python
pipe.load_lora_weights("out.safetensors")
pipe.set_adapters(["name"], [0.8])
```

跑法：python tools/convert_lora.py --help
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def convert_key(k: str) -> str | None:
    """ComfyUI 键名 → diffusers 键名。返回 None 表示不认这个键（跳过）。"""
    if not k.startswith("diffusion_model."):
        return None
    k = k[len("diffusion_model."):]
    if ".lora_A.weight" in k:
        return "transformer." + k.replace(".lora_A.weight", ".lora.down.weight")
    if ".lora_B.weight" in k:
        return "transformer." + k.replace(".lora_B.weight", ".lora.up.weight")
    # ComfyUI 有些 LoRA 另带 alpha / alpha 的 rank 标记，diffusers 不需要
    if k.endswith(".alpha"):
        return None
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])

    sd = load_file(str(src))
    out: dict = {}
    skipped: list[str] = []
    for k, v in sd.items():
        nk = convert_key(k)
        if nk is None:
            skipped.append(k)
            continue
        out[nk] = v

    if not out:
        print(f"❌ 没有可转换的键：{src} 看起来不是 ComfyUI 格式的 Qwen LoRA")
        print(f"   头几个键：{list(sd)[:5]}")
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst), metadata={"format": "pt"})

    print(f"✅ {src.name} → {dst.name}")
    print(f"   转换 {len(out)} 个张量，跳过 {len(skipped)} 个")
    if skipped:
        print(f"   跳过示例：{skipped[:3]}")
    # 抽查一个，确认形状没反
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
