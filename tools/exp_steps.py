"""步数实验：关掉真 CFG 之后，24 步的下限还成立吗？

**为什么重测**：PLAN §2.6 的结论「低于 16 步出糊块、24 步才完整」是在
**真 CFG 开着**（48 次前向）时测的。关掉真 CFG 后采样路径变了
（只有 24 次前向，没有无条件分支），「需要多少步」可能也变了。
步数是线性省时间的旋钮：24 → 16 步就是 -33%。

用法：
    .venv\\Scripts\\python.exe tools/exp_steps.py
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

from workflows import quality  # noqa: E402
from workflows.engine import Engine  # noqa: E402

MODEL = "qwen-image-2.1"
OUT = Path("assets/_steps_exp")
OUT.mkdir(parents=True, exist_ok=True)

STEPS = (12, 16, 20, 24)

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


def run(pipe, prompt: str, steps: int, seed: int, size: int = 768):
    kw = dict(prompt=prompt, width=size, height=size, num_inference_steps=steps,
              true_cfg_scale=1.0,
              generator=torch.Generator("cuda").manual_seed(seed))
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    out = pipe(**kw)
    torch.cuda.synchronize()
    return out.images[0].convert("RGBA"), time.perf_counter() - t0


def main() -> int:
    eng = Engine("models")
    pipe = eng.txt2img_pipe(MODEL)
    ex = eng.pose_extractor()

    times: dict[int, list[float]] = {s: [] for s in STEPS}
    for name, prompt in PROMPTS:
        for steps in STEPS:
            img, dt = run(pipe, prompt, steps, 42)
            times[steps].append(dt)
            img.save(OUT / f"{name}_{steps}.png")
            ev = quality.evaluate(img, ex)
            # 「缺件」的直接证据：前景连通域数 > 1（头/手与身体分离）
            blobs = int(ev.get("blobs", 0) or 0)
            print(f"[{name:7s} {steps:2d}步] {dt:5.1f}s  关节 {ev['points']:2d}  "
                  f"连通域 {blobs}", flush=True)

    print("\n=== 各档中位耗时 ===")
    for s in STEPS:
        print(f"  {s:2d} 步  {statistics.median(times[s]):5.2f}s")

    for sprite_px in (64, 128):
        sheet = Image.new("RGB", (len(STEPS) * (sprite_px + 8) + 8,
                                  len(PROMPTS) * (sprite_px + 8) + 8), (255, 255, 255))
        for r, (name, _p) in enumerate(PROMPTS):
            for c, steps in enumerate(STEPS):
                im = Image.open(OUT / f"{name}_{steps}.png").convert("RGBA")
                im = im.resize((sprite_px, sprite_px), Image.LANCZOS)
                bg = Image.new("RGBA", im.size, (110, 110, 110, 255))
                bg.alpha_composite(im)
                sheet.paste(bg.convert("RGB"),
                            (8 + c * (sprite_px + 8), 8 + r * (sprite_px + 8)))
        sheet.save(OUT / f"cmp_{sprite_px}.png")
        print("对照图：", OUT / f"cmp_{sprite_px}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
