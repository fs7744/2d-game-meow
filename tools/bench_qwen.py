"""Qwen-Image-2.1 的单位成本与「条件图分辨率对齐」实测。

三个问题要一次量清（都影响显存与质量）：

  1. 文生图 768² / 24 步的耗时与**保留峰值**（峰值一律看 memory_reserved，
     allocated 会低估，见 README 的度量约定）。
  2. 指令改图时 `output_resolution` 不传（=1024）会怎样：条件图被放大到
     1024² 面积再编码，比目标 768² 多 ~78% 的图 token，既慢又让「参考图的
     尺度」和「要生成的尺度」对不上。
  3. 传 output_resolution = 目标边长后，同样的 seed 出图差多少、快多少。

用法：
    .venv\\Scripts\\python.exe tools\\bench_qwen.py            # 全部
    .venv\\Scripts\\python.exe tools\\bench_qwen.py --quick    # 8 步冒烟

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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from workflows.engine import Engine  # noqa: E402

OUT = Path("assets/_qwen_bench")
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "qwen-image-2.1"
POS = ("a pixel art white cat knight with red cape and golden armor, standing, "
       "full body, plain white background")
NEG = ("blurry, low quality, worst quality, jpeg artifacts, watermark, "
       "signature, text, deformed, extra limbs, bad anatomy, cropped, out of frame")


def _vram() -> float:
    return round(torch.cuda.max_memory_reserved() / 2**30, 2)


def _run(pipe, *, prompt, image=None, size=768, steps=24, seed=5,
         output_resolution=None, label=""):
    """一次前向，返回 (图, 秒, 峰值GB)。"""
    kw = dict(prompt=prompt, negative_prompt=NEG, width=size, height=size,
              num_inference_steps=steps, true_cfg_scale=4.0,
              generator=torch.Generator("cuda").manual_seed(seed))
    if image is not None:
        kw["image"] = image.convert("RGB").resize((size, size), Image.LANCZOS)
    if output_resolution is not None:
        kw["output_resolution"] = int(output_resolution)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    im = pipe(**kw).images[0]
    dt = time.perf_counter() - t
    print(f"  [{label}] {dt:.2f}s  peak_reserved={_vram():.2f}GB", flush=True)
    return im, dt, _vram()


def _diff(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.int16)
    y = np.asarray(b.convert("RGB"), dtype=np.int16)
    return float(np.abs(x - y).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="8 步冒烟（默认 24 步）")
    ap.add_argument("--size", type=int, default=768)
    args = ap.parse_args()
    steps = 8 if args.quick else 24
    size = args.size

    t0 = time.perf_counter()
    eng = Engine("models")
    pipe = eng.txt2img_pipe(MODEL)
    print(f"加载 {time.perf_counter() - t0:.1f}s，峰值 {_vram():.2f}GB", flush=True)

    # 预热：第一次调用含 Triton/卷积内核选择、VAE 首次编译，不计入
    print("[warmup]", flush=True)
    base, _dt, _ = _run(pipe, prompt=POS, size=size, steps=steps, seed=1, label="warmup")

    # 1) 文生图（三次取中位数）
    t1 = [_run(pipe, prompt=POS, size=size, steps=steps, seed=i, label=f"t2i{i}")[1]
          for i in (2, 3, 4)]
    t2i = statistics.median(t1)
    print(f"[1] 文生图 {size}²/{steps}步：中位 {t2i:.2f}s", flush=True)

    # 2) 指令改图，**不传** output_resolution（老行为 = 1024）
    a_img, a_dt, a_vram = _run(pipe, prompt="make the cape blue", image=base,
                              size=size, steps=steps, seed=7, label="edit@1024")
    # 3) 指令改图，output_resolution = 目标边长
    b_img, b_dt, b_vram = _run(pipe, prompt="make the cape blue", image=base,
                              size=size, steps=steps, seed=7,
                              output_resolution=size, label="edit@aligned")
    a_img.save(OUT / "edit_out1024.png")
    b_img.save(OUT / "edit_aligned.png")
    base.save(OUT / "base.png")
    print(f"[2] 改图 条件图 1024²：{a_dt:.2f}s / {a_vram:.2f}GB", flush=True)
    print(f"[3] 改图 条件图对齐 {size}²：{b_dt:.2f}s / {b_vram:.2f}GB", flush=True)
    print(f"[4] 两种条件图分辨率的输出平均像素差 {_diff(a_img, b_img):.2f}/255",
          flush=True)
    print(f"[5] 对齐后省 {100 * (1 - b_dt / a_dt):.0f}% 耗时、"
          f"{100 * (1 - b_vram / a_vram):.0f}% 峰值", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
