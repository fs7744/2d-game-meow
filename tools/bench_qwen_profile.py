"""Qwen-Image-2.1 的耗时拆解：到底慢在哪一段。

**先量再优化**：这一轮不试任何优化项，先把 768²/24 步这一张的时间拆成

    文本编码（encode_prompt ×2） / DiT 前向（每步） / VAE 解码 / 其余（搬运等）

「其余」主要是 `enable_model_cpu_offload` 的权重搬运——文本编码器 8.8G、
transformer 6.7G，每步都要在 CPU↔GPU 之间倒一遍。

用法：
    .venv\\Scripts\\python.exe tools\\bench_qwen_profile.py
    .venv\\Scripts\\python.exe tools\\bench_qwen_profile.py --repeat 3

**必须预热 + 多次取中位数**（本项目铁律：单次调用波动可达 70%）。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from workflows.engine import Engine  # noqa: E402

MODEL = "qwen-image-2.1"
POS = ("a pixel art white cat knight with red cape and golden armor, standing, "
       "full body, plain white background")
NEG = ("blurry, low quality, worst quality, jpeg artifacts, watermark, "
       "signature, text, deformed, extra limbs, bad anatomy, cropped, out of frame")


class Timer:
    """按名字累计耗时（每次调用后同步 CUDA，避免把排队时间算错）。"""

    def __init__(self) -> None:
        self.acc: dict[str, float] = {}

    def wrap(self, obj, name: str, key: str) -> None:
        orig = getattr(obj, name)

        def wrapper(*a, **k):
            t0 = time.perf_counter()
            r = orig(*a, **k)
            torch.cuda.synchronize()
            self.acc[key] = self.acc.get(key, 0.0) + (time.perf_counter() - t0)
            return r

        setattr(obj, name, wrapper)

    def restore(self, obj, name: str, orig) -> None:
        setattr(obj, name, orig)


def one_run(pipe, *, size: int, steps: int, seed: int, timer: Timer | None = None):
    kw = dict(prompt=POS, negative_prompt=NEG, width=size, height=size,
              num_inference_steps=steps, true_cfg_scale=4.0,
              generator=torch.Generator("cuda").manual_seed(seed))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = pipe(**kw)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return out, total, round(torch.cuda.max_memory_reserved() / 2**30, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--offload", default=os.environ.get("MEOW_QWEN_OFFLOAD", "1"),
                    help="1=文本编码器用完搬出显存（默认）；0=两块常驻")
    args = ap.parse_args()

    os.environ["MEOW_QWEN_OFFLOAD"] = args.offload

    eng = Engine("models")
    pipe = eng.txt2img_pipe(MODEL)
    print(f"[环境] MEOW_QWEN_OFFLOAD={args.offload}  "
          f"CUDA {torch.cuda.get_device_name(0)}")

    # ---- 预热：第一次要把权重搬上 GPU，不能计入 ----
    t0 = time.perf_counter()
    one_run(pipe, size=args.size, steps=args.steps, seed=1)
    print(f"[预热] {time.perf_counter() - t0:.1f}s（不计入）")

    totals: list[float] = []
    parts: dict[str, list[float]] = {}
    for i in range(args.repeat):
        timer = Timer()
        orig_encode = pipe.encode_prompt
        orig_tf = pipe.transformer.forward
        orig_vae = pipe.vae.decode
        timer.wrap(pipe, "encode_prompt", "文本编码")
        timer.wrap(pipe.transformer, "forward", "DiT 前向")
        timer.wrap(pipe.vae, "decode", "VAE 解码")
        try:
            _out, total, vram = one_run(pipe, size=args.size, steps=args.steps,
                                        seed=10 + i, timer=timer)
        finally:
            timer.restore(pipe, "encode_prompt", orig_encode)
            timer.restore(pipe.transformer, "forward", orig_tf)
            timer.restore(pipe.vae, "decode", orig_vae)
        totals.append(total)
        for k, v in timer.acc.items():
            parts.setdefault(k, []).append(v)
        rest = total - sum(timer.acc.values())
        parts.setdefault("其余(搬运等)", []).append(rest)
        print(f"  第{i + 1}次 总 {total:.2f}s  峰值 {vram}GB  "
              + "  ".join(f"{k} {v:.2f}s" for k, v in timer.acc.items())
              + f"  其余 {rest:.2f}s")

    med = statistics.median(totals)
    print(f"\n=== 中位数（{args.repeat} 次，{args.size}²/{args.steps} 步）===")
    print(f"总耗时 {med:.2f}s")
    for k, v in parts.items():
        m = statistics.median(v)
        print(f"  {k:12s} {m:6.2f}s  ({m / med * 100:4.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
