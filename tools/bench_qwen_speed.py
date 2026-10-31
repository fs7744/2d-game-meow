"""Qwen 提速实验：时间到底花在哪、哪些旋钮真的有用。

**第一轮已经量清**（tools/bench_qwen_profile.py，768²/24 步中位数）：

    总耗时 21.95s = DiT 前向 18.84s (86%) + 文本编码 1.79s (8%)
                        + VAE 解码 1.12s (5%) + 搬运 0.12s (0.5%)

所以**文本编码与显存搬运都不是瓶颈**（之前担心的 fp8 文本编码器只占 8%）。
要提速只能动 DiT 前向本身，而它有一个可疑的结构：

    `true_cfg_scale > 1 且给了负向词` → 管线每个 step 跑**两遍** transformer
    （有条件 + 无条件），24 步就是 48 次前向。

去掉负向词（true_cfg 静默失效）就只剩 24 次前向 → 时间减半。
但画质会掉多少？这是本脚本要回答的问题。

用法：
    .venv\\Scripts\\python.exe tools/bench_qwen_speed.py --stage time
    .venv\\Scripts\\python.exe tools/bench_qwen_speed.py --stage quality
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

from workflows import paths  # noqa: E402
from workflows.engine import Engine  # noqa: E402

MODEL = "qwen-image-2.1"
OUT = paths.tool_out("_speed_exp")
OUT.mkdir(parents=True, exist_ok=True)

POS = ("a pixel art white cat knight with red cape and golden armor, standing, "
       "full body, plain white background")
NEG = ("blurry, low quality, worst quality, jpeg artifacts, watermark, "
       "signature, text, deformed, extra limbs, bad anatomy, cropped, out of frame")

# (名字, true_cfg_scale, steps, 给不给负向词)
CONFIGS = [
    ("cfg4_neg_24", 4.0, 24, True),      # 现状
    ("cfg1_nonneg_24", 1.0, 24, False),  # 去掉真 CFG（单次前向）
    ("cfg1_nonneg_32", 1.0, 32, False),  # 单次前向 + 多补几步
    ("cfg4_neg_16", 4.0, 16, True),      # 少步数（双次前向）
]


def _run(pipe, *, cfg: float, steps: int, use_neg: bool, size: int, seed: int):
    kw = dict(prompt=POS, width=size, height=size,
              num_inference_steps=steps, true_cfg_scale=cfg,
              generator=torch.Generator("cuda").manual_seed(seed))
    if use_neg:
        kw["negative_prompt"] = NEG
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = pipe(**kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out.images[0].convert("RGBA"), dt, \
        round(torch.cuda.max_memory_reserved() / 2**30, 2)


def stage_time(repeat: int, size: int) -> None:
    eng = Engine("models")
    pipe = eng.txt2img_pipe(MODEL)
    _run(pipe, cfg=4.0, steps=8, use_neg=True, size=size, seed=0)   # 预热
    print(f"{'配置':22s} {'中位耗时':>10s} {'峰值':>8s}  相对现状")
    base = None
    for name, cfg, steps, use_neg in CONFIGS:
        ts, vram = [], 0.0
        for i in range(repeat):
            _img, dt, v = _run(pipe, cfg=cfg, steps=steps, use_neg=use_neg,
                               size=size, seed=20 + i)
            ts.append(dt)
            vram = max(vram, v)
        med = statistics.median(ts)
        if base is None:
            base = med
        print(f"{name:22s} {med:9.2f}s {vram:7.2f}G  {med / base * 100:5.0f}%")


def stage_quality(size: int) -> None:
    """把四种配置的成品并排存出来，供目视 + 产线判据比较。"""
    eng = Engine("models")
    pipe = eng.txt2img_pipe(MODEL)
    imgs = []
    for name, cfg, steps, use_neg in CONFIGS:
        img, dt, _v = _run(pipe, cfg=cfg, steps=steps, use_neg=use_neg,
                           size=size, seed=42)
        img.save(OUT / f"{name}.png")
        print(f"{name:22s} {dt:6.2f}s")
        imgs.append(img.resize((320, 320), Image.LANCZOS))
    sheet = Image.new("RGB", (len(imgs) * 328, 320), (255, 255, 255))
    for i, im in enumerate(imgs):
        bg = Image.new("RGBA", im.size, (110, 110, 110, 255))
        bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), (i * 328, 0))
    sheet.save(OUT / "cmp.png")
    print("对照图：", OUT / "cmp.png")

    # 产线判据：前景占比 + 贴边
    from workflows import quality
    ex = eng.pose_extractor()
    print("\n前景占比 / 贴边：")
    for name, _cfg, _steps, _neg in CONFIGS:
        im = Image.open(OUT / f"{name}.png").convert("RGBA")
        a = np.asarray(im.getchannel("A"))
        ev = quality.evaluate(im, ex)
        print(f"  {name:22s} 前景 {(a > 127).mean() * 100:5.1f}%  "
              f"贴边 {len(ev['edge'])}  覆盖 {ev['coverage']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="time", choices=["time", "quality", "both"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--size", type=int, default=768)
    args = ap.parse_args()
    if args.stage in ("time", "both"):
        stage_time(args.repeat, args.size)
    if args.stage in ("quality", "both"):
        stage_quality(args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
