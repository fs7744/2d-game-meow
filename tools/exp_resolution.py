"""分辨率实验：精灵帧到底需要 768² 吗？

**前提**：精灵成品最后要缩到 32~128px（PLAN §2.1 的理由是「1024² 的细节在
缩图上根本看不见」）。同样的逻辑往下推一步：**768² 的细节在 64px 的精灵帧上
是不是也大部分看不见？** 如果 512² 出图缩到 64px 一样能用，那每帧的图 token
少 44%，时间也少 44%（DiT 前向与边长平方成正比）。

本脚本对同一批提示词出 512² / 640² / 768² 三档，然后：
  · 量耗时与峰值；
  · 把每张缩到 64px（精灵常用尺寸）拼成对照图，肉眼判「缩完还认不认得出」；
  · 量 DWPose 关节数（人形主体在缩图后还认不认得出）与前景占比。

用法：
    .venv\\Scripts\\python.exe tools/exp_resolution.py
"""
from __future__ import annotations

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

from workflows import paths
from workflows import quality  # noqa: E402
from workflows.engine import Engine  # noqa: E402

MODEL = "qwen-image-2.1"
OUT = paths.tool_out("_res_exp")
OUT.mkdir(parents=True, exist_ok=True)

SIZES = (512, 640, 768)

PROMPTS = [
    ("knight", "a pixel art white cat knight with red cape and golden armor, "
               "standing, full body, plain white background"),
    ("mecha", "a pixel art blue battle mech with a cannon, standing, full body, "
              "plain white background"),
    ("wolf", "a pixel art gray wolf swordsman, standing, full body, "
             "plain white background"),
    ("mage", "a pixel art red hooded mage holding a staff, standing, full body, "
             "plain white background"),
]


def run(pipe, prompt: str, size: int, steps: int, seed: int):
    kw = dict(prompt=prompt, width=size, height=size, num_inference_steps=steps,
              true_cfg_scale=1.0,
              generator=torch.Generator("cuda").manual_seed(seed))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = pipe(**kw)
    torch.cuda.synchronize()
    return out.images[0].convert("RGBA"), time.perf_counter() - t0, \
        round(torch.cuda.max_memory_reserved() / 2**30, 2)


def main() -> int:
    eng = Engine("models")
    pipe = eng.txt2img_pipe(MODEL)
    ex = eng.pose_extractor()

    times: dict[int, list[float]] = {s: [] for s in SIZES}
    vram: dict[int, float] = {}
    for name, prompt in PROMPTS:
        for size in SIZES:
            img, dt, v = run(pipe, prompt, size, 24, 42)
            times[size].append(dt)
            vram[size] = max(vram.get(size, 0), v)
            img.save(OUT / f"{name}_{size}.png")
            ev = quality.evaluate(img, ex)
            a = np.asarray(img.getchannel("A"))
            print(f"[{name:7s} {size}] {dt:5.1f}s  峰值 {v:5.2f}G  "
                  f"关节 {ev['points']:2d}  前景 {(a > 127).mean() * 100:5.1f}%",
                  flush=True)

    print("\n=== 各档中位耗时 ===")
    base = None
    for size in SIZES:
        med = statistics.median(times[size])
        base = med if base is None else base
        print(f"  {size}²  {med:5.2f}s  峰值 {vram[size]:5.2f}G  "
              f"（相对 {size}² 基准 {med / base * 100:.0f}%）")

    # 对照图：上排原图缩到 64px，下排原图缩到 128px
    for sprite_px in (64, 128):
        cols = len(SIZES)
        sheet = Image.new("RGB", (cols * (sprite_px + 8) + 8,
                                  len(PROMPTS) * (sprite_px + 8) + 8), (255, 255, 255))
        for r, (name, _p) in enumerate(PROMPTS):
            for c, size in enumerate(SIZES):
                im = Image.open(OUT / f"{name}_{size}.png").convert("RGBA")
                im = im.resize((sprite_px, sprite_px), Image.LANCZOS)
                bg = Image.new("RGBA", im.size, (110, 110, 110, 255))
                bg.alpha_composite(im)
                sheet.paste(bg.convert("RGB"),
                            (8 + c * (sprite_px + 8), 8 + r * (sprite_px + 8)))
        sheet.save(OUT / f"cmp_{sprite_px}.png")
        print(f"对照图：{OUT / f'cmp_{sprite_px}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
