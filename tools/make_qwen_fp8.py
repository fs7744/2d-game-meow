"""把 Qwen-Image-2.1 的 text_encoder（Qwen3-VL，bf16 16.3GB）量化成 fp8 缓存。

**为什么要做**：这块权重是整个管线的显存与内存峰值所在——实测文生图
768² 的保留峰值 16.59GB，几乎就是 text_encoder 的 16.3GB 本身
（transformer 走 fp8 后只有 6.7GB，且被 cpu offload 搬走了）。
本机 32GB 内存里它又常驻 16.3GB，加上 transformer 的 6.7GB 就快见底。

量化到 fp8(E4M3) 后：显存 ~8.2GB、内存同样的减少一半，管线可以不再依赖
model_cpu_offload 每一步搬权重。

**逐输出通道 scale**，不是逐张量：LLM 的 q/k/v 与 mlp 各行的动态范围差几个
数量级，逐张量 scale 会被 outlier 拉偏（本项目在 transformer 上踩过）。
布局与 `engine._fp8_linear_cls` 一致：`<prefix>.w_fp8`（fp8 buffer）+
`<prefix>.w_scale`（bf16，形状 [out, 1]），`weight` 留 0 长度 bf16 占位。

用法：
    .venv\\Scripts\\python.exe tools\\make_qwen_fp8.py
    .venv\\Scripts\\python.exe tools\\make_qwen_fp8.py --model models/checkpoints/qwen-image-2.1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FP8_MAX = 448.0          # torch.float8_e4m3fn 的可表示上限


def _log(*a):
    print(*a, flush=True)


def build_skeleton(cfg_dir: Path):
    """按 config 建一个全 meta 的模型，只为了知道哪些 key 属于 nn.Linear。"""
    from accelerate import init_empty_weights

    import importlib

    cfg = json.loads((cfg_dir / "config.json").read_text("utf-8"))
    arch = (cfg.get("architectures") or ["Qwen3VLForConditionalGeneration"])[0]
    mtype = cfg.get("model_type", "qwen3_vl")
    # architectures 里是类名，配置类在 transformers.models.<model_type> 下
    mod = importlib.import_module(f"transformers.models.{mtype}")
    cls = getattr(mod, arch)
    config = cls.config_class.from_pretrained(str(cfg_dir))
    with init_empty_weights():
        model = cls(config)
    return model


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按输出通道缩放量化成 fp8。返回 (w_fp8 uint8 视图, w_scale)。"""
    w = w.to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    q = (w / scale.to(torch.float32)).clamp(-FP8_MAX, FP8_MAX)
    return q.to(torch.float8_e4m3fn), scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/checkpoints/qwen-image-2.1")
    ap.add_argument("--out", default="", help="默认 <model>/text_encoder-fp8")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.model)
    te = root / "text_encoder"
    out = Path(args.out) if args.out else root / "text_encoder-fp8"
    if (out / "model.safetensors").is_file() and not args.force:
        _log(f"已存在：{out/'model.safetensors'}（加 --force 重建）")
        return 0

    index = json.loads((te / "model.safetensors.index.json").read_text("utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    _log(f"{len(weight_map)} 个张量 / {len(shards)} 个分片")

    model = build_skeleton(te)
    linear = {n for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and n}
    _log(f"nn.Linear 模块 {len(linear)} 个")
    del model

    pack: dict[str, torch.Tensor] = {}
    n_q = n_keep = 0
    total_q = total_k = 0
    t0 = time.perf_counter()
    for si, shard in enumerate(shards, 1):
        sd = load_file(str(te / shard), device="cpu")
        for k, v in sd.items():
            prefix = k.rsplit(".", 1)[0]
            name = k.rsplit(".", 1)[1]
            if name == "weight" and prefix in linear and v.dim() == 2:
                q, sc = quantize_weight(v)
                pack[f"{prefix}.w_fp8"] = q
                pack[f"{prefix}.w_scale"] = sc
                n_q += 1
                total_q += v.numel()
            else:
                pack[k] = v.contiguous()
                n_keep += 1
            if name in ("weight", "bias") or k.endswith("scale_inv"):
                total_k += v.numel()
        del sd
        _log(f"  分片 {si}/{len(shards)} 完成，已累计 {len(pack)} 个张量，"
             f"{time.perf_counter() - t0:.0f}s")

    out.mkdir(parents=True, exist_ok=True)
    save_file(pack, str(out / "model.safetensors"))
    del pack
    for f in ("config.json", "generation_config.json"):
        if (te / f).is_file():
            shutil.copy2(te / f, out / f)
    meta = {"quant": "fp8_e4m3_per_out_channel", "linear_count": n_q,
            "kept": n_keep, "source": str(te)}
    (out / "quant.json").write_text(json.dumps(meta, indent=2), "utf-8")
    _log(f"完成：{out/'model.safetensors'} "
         f"{ (out/'model.safetensors').stat().st_size / 2**30:.1f}GB，"
         f"量化 {n_q} 个 Linear，用时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
