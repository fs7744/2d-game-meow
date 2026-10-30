"""真 CFG 的 A/B：关掉它能省 41% 时间，画质到底掉不掉。

**背景**（tools/bench_qwen_profile.py 量到的）：
    768²/24 步总耗时 21.95s，其中 DiT 前向 18.84s（86%）。
    而 `true_cfg_scale > 1 且给了负向词` 时，管线**每个 step 跑两遍 transformer**
    （有条件 + 无条件）——24 步就是 48 次前向。
    不给负向词（真 CFG 静默失效）就只剩 24 次前向：21.3s → 12.6s（-41%）。

单张对照看着画质不降（甚至更精细），但单张说明不了问题——
负向词的作用是抑制「多肢体/缺件/糊」这类**偶发**失败。
所以这里用 6 个差别较大的提示词各跑两遍，统计产线判据（单主体/贴边/前景占比）
并拼出对照图供目视。

用法：
    .venv\\Scripts\\python.exe tools/exp_cfg_ab.py
"""
from __future__ import annotations

import os
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
OUT = Path("assets/_cfg_ab")
OUT.mkdir(parents=True, exist_ok=True)

NEG = ("blurry, low quality, worst quality, jpeg artifacts, watermark, "
       "signature, text, deformed, extra limbs, bad anatomy, cropped, out of frame")

PROMPTS = [
    ("knight", "a pixel art white cat knight with red cape and golden armor, "
               "standing, full body, plain white background"),
    ("mecha", "a pixel art blue battle mech with a cannon, standing, full body, "
              "plain white background"),
    ("slime", "a pixel art blue slime monster with big eyes, standing, full body, "
              "plain white background"),
    ("wolf", "a pixel art gray wolf swordsman, standing, full body, "
             "plain white background"),
    ("mage", "a pixel art red hooded mage holding a staff, standing, full body, "
             "plain white background"),
    ("dragon", "a pixel art green dragon warrior, standing, full body, "
               "plain white background"),
]


def run(pipe, prompt: str, *, cfg: float, use_neg: bool, steps: int, seed: int):
    kw = dict(prompt=prompt, width=768, height=768, num_inference_steps=steps,
              true_cfg_scale=cfg,
              generator=torch.Generator("cuda").manual_seed(seed))
    if use_neg:
        kw["negative_prompt"] = NEG
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    out = pipe(**kw)
    torch.cuda.synchronize()
    return out.images[0].convert("RGBA"), time.perf_counter() - t0


def main() -> int:
    eng = Engine("models")
    pipe = eng.txt2img_pipe(MODEL)
    ex = eng.pose_extractor()

    rows = []
    for name, prompt in PROMPTS:
        for tag, cfg, use_neg in (("cfg4", 4.0, True), ("cfg1", 1.0, False)):
            img, dt = run(pipe, prompt, cfg=cfg, use_neg=use_neg, steps=24, seed=42)
            img.save(OUT / f"{name}_{tag}.png")
            a = np.asarray(img.getchannel("A"))
            ev = quality.evaluate(img, ex)
            fg = float((a > 127).mean())
            rows.append((name, tag, dt, fg, len(ev["edge"]), ev["points"],
                         ev["coverage"]))
            print(f"[{name:7s}/{tag}] {dt:5.1f}s  前景 {fg * 100:5.1f}%  "
                  f"贴边 {len(ev['edge'])}  关节 {ev['points']}  "
                  f"覆盖 {ev['coverage']:.2f}", flush=True)

    print("\n=== 汇总 ===")
    for tag in ("cfg4", "cfg1"):
        sel = [r for r in rows if r[1] == tag]
        print(f"{tag}: 平均 {np.mean([r[2] for r in sel]):.1f}s  "
              f"前景均值 {np.mean([r[3] for r in sel]) * 100:.1f}%  "
              f"贴边总数 {sum(r[4] for r in sel)}  "
              f"关节 <8 的帧数 {sum(1 for r in sel if r[5] < 8)}")

    # 对照图：每个提示词一行，左=cfg4、右=cfg1
    tile = 260
    sheet = Image.new("RGB", (2 * (tile + 8) + 8, len(PROMPTS) * (tile + 8) + 8),
                      (255, 255, 255))
    for r, (name, _prompt) in enumerate(PROMPTS):
        for c, tag in enumerate(("cfg4", "cfg1")):
            im = Image.open(OUT / f"{name}_{tag}.png").convert("RGBA")
            im = im.resize((tile, tile), Image.LANCZOS)
            bg = Image.new("RGBA", im.size, (110, 110, 110, 255))
            bg.alpha_composite(im)
            sheet.paste(bg.convert("RGB"), (8 + c * (tile + 8), 8 + r * (tile + 8)))
    sheet.save(OUT / "cmp.png")
    print("对照图：", OUT / "cmp.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
