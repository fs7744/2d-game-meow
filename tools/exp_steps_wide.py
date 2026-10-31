"""步数下限重测（关掉真 CFG 之后）。

**要重测的原因**：PLAN §2.6 的结论「12 步没有头、16 步四肢断、24 步才完整」
是在**真 CFG 开着**时测的（每个 step 跑两遍 transformer）。现在默认关掉真 CFG
（单次前向），采样轨迹变了——初测 12 步的成品与 24 步几乎一样好。
那这条结论就不成立了，必须重测。

**这次测得更宽**：12 个提示词（含武器、坐姿、动物、机甲、透明底等难例）×
4 档步数，统计产线判据（连通域 > 1 = 缺件、关节数、贴边）与目视对照。

用法：
    .venv\\Scripts\\python.exe tools/exp_steps_wide.py
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
OUT = paths.tool_out("_steps_wide")
OUT.mkdir(parents=True, exist_ok=True)

STEPS = (8, 12, 16, 20, 24)

PROMPTS = [
    ("knight", "a pixel art white cat knight with red cape and golden armor, "
               "standing, full body, plain white background"),
    ("mecha", "a pixel art blue battle mech with a cannon, standing, full body, "
              "plain white background"),
    ("wolf", "a pixel art gray wolf swordsman, standing, full body, "
             "plain white background"),
    ("mage", "a pixel art red hooded mage holding a staff, standing, full body, "
             "plain white background"),
    ("dragon", "a pixel art green dragon warrior, standing, full body, "
               "plain white background"),
    ("slime", "a pixel art blue slime monster with big eyes, standing, full body, "
              "plain white background"),
    ("ninja", "a pixel art black ninja with two swords, crouching, full body, "
              "plain white background"),
    ("bird", "a pixel art red parrot standing on a branch, full body, "
             "plain white background"),
    ("robot", "a pixel art small white robot with a round head, standing, "
              "full body, plain white background"),
    ("girl", "a pixel art girl with long pink hair and a blue dress, standing, "
             "full body, plain white background"),
    ("skeleton", "a pixel art skeleton warrior with a rusty sword, standing, "
                 "full body, plain white background"),
    ("ghost", "a pixel art cute ghost floating, full body, plain white background"),
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
    stats: dict[int, dict[str, int]] = {s: {"blobs": 0, "edge": 0, "lowpts": 0}
                                         for s in STEPS}
    for name, prompt in PROMPTS:
        for steps in STEPS:
            img, dt = run(pipe, prompt, steps, 42)
            times[steps].append(dt)
            img.save(OUT / f"{name}_{steps}.png")
            ev = quality.evaluate(img, ex)
            blobs = int(ev.get("blobs", 0) or 0)
            st = stats[steps]
            if blobs > 1:
                st["blobs"] += 1
            if ev["edge"]:
                st["edge"] += 1
            if ev["points"] < 8:
                st["lowpts"] += 1
            print(f"[{name:8s} {steps:2d}步] {dt:5.1f}s 关节 {ev['points']:2d} "
                  f"连通域 {blobs} 贴边 {len(ev['edge'])}", flush=True)

    print("\n=== 汇总（12 个提示词）===")
    print(f"{'步数':>4s} {'中位耗时':>9s} {'缺件(连通域>1)':>14s} {'贴边':>6s} "
          f"{'关节<8':>8s}")
    for s in STEPS:
        st = stats[s]
        print(f"{s:4d} {statistics.median(times[s]):8.2f}s {st['blobs']:14d} "
              f"{st['edge']:6d} {st['lowpts']:8d}")

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
