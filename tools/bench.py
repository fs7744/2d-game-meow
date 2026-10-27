"""全量基准：逐工作流、逐分辨率测耗时与峰值显存，输出 Markdown 表。

用法：python tools/bench.py [--steps 25]
结果写 assets/_bench_full.json，并打印可直接贴进 README 的表格。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from workflows import generate as g  # noqa: E402
from workflows.engine import Engine  # noqa: E402

ASSETS = ROOT / "assets"
ROWS: list[dict] = []


def record(name: str, seconds: float, peak: float, note: str = "") -> None:
    ROWS.append({"case": name, "seconds": round(seconds, 2), "peak_gb": round(peak, 2),
                 "note": note})
    print(f"  {name:38s} {seconds:6.2f}s  {peak:5.2f}GB  {note}", flush=True)


def peak() -> float:
    return torch.cuda.max_memory_allocated() / 2**30


def bench_txt(eng: Engine, w: int, h: int, steps: int, n: int = 3) -> None:
    for _ in range(n):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        g.txt2img(eng, prompt="blue slime monster", width=w, height=h, steps=steps,
                  seed=1, tag="bench")
        record(f"W1 txt2img {w}x{h} / {steps}步", time.perf_counter() - t0, peak())


def bench_img2img(eng: Engine, strength: float) -> None:
    src = Image.new("RGB", (512, 512), (240, 240, 240))
    ImageDraw.Draw(src).rectangle([128, 128, 384, 384], fill=(60, 90, 160))
    p = ASSETS / "_bench_src512.png"
    src.save(p)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g.img2img(eng, init_image=p, prompt="blue slime monster", strength=strength,
              steps=25, seed=1, tag="bench")
    record(f"W2 img2img 512² strength={strength}", time.perf_counter() - t0, peak())


def bench_inpaint(eng: Engine) -> None:
    src = ASSETS / "_bench_src512.png"
    # 预热：首次调用含 pipeline 加载，不计入稳态耗时
    if not (ASSETS / "_bench_mask512.png").exists():
        pass
    m = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(m).rectangle([180, 180, 340, 340], fill=255)
    mask = ASSETS / "_bench_mask512.png"
    m.save(mask)
    g.inpaint(eng, image=src, mask=mask, prompt="warmup", steps=4, seed=1, tag="warm")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    g.inpaint(eng, image=src, mask=mask, prompt="green grass block", steps=25, seed=1, tag="bench")
    record("W3 inpaint 512²", time.perf_counter() - t0, peak())


def bench_pose(eng: Engine, frames: list[Path], n: int, ref: Path | None) -> None:
    # 预热：首次调用含 ControlNet / IP-Adapter 加载
    g.poseswap(eng, pose_images=frames[:1], prompt="warmup", ref_image=ref,
               steps=4, seed=1, width=512, height=512, tag="warm")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    r = g.poseswap(eng, pose_images=frames[:n], prompt="white cat knight, red cape",
                   ref_image=ref, ipa_scale=0.85, steps=25, seed=1,
                   width=512, height=512, pixel=4, palette=16, tag="bench")
    total = time.perf_counter() - t0
    record(f"W4 poseswap {n}帧 512²{'（IPA）' if ref else ''}", total, peak(),
           f"{total / n:.2f}s/帧")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()
    ASSETS.mkdir(exist_ok=True)

    t_load = time.perf_counter()
    eng = Engine(ROOT / "models")
    eng.txt2img_pipe()
    record("模型加载 SD1.5（冷启动）", time.perf_counter() - t_load, 0.0)
    print()

    bench_txt(eng, 512, 512, a.steps)
    bench_txt(eng, 512, 768, a.steps)
    bench_txt(eng, 768, 768, a.steps)
    bench_txt(eng, 512, 512, 8, n=2)
    print()
    bench_img2img(eng, 0.45)
    bench_img2img(eng, 0.65)
    bench_inpaint(eng)
    print()

    poses = sorted(ASSETS.glob("_pose_walk_*.png"))
    if not poses:
        from tools.make_pose import main as gen_pose
        gen_pose()
        poses = sorted(ASSETS.glob("_pose_walk_*.png"))
    ref = next(iter(sorted(ASSETS.glob("slime_*.png"))), None)
    bench_pose(eng, poses, 1, None)
    bench_pose(eng, poses, 8, None)
    if ref:
        bench_pose(eng, poses, 8, ref)

    out = ASSETS / "_bench_full.json"
    out.write_text(json.dumps(ROWS, ensure_ascii=False, indent=2), "utf-8")

    print("\n| 用例 | 耗时 | 峰值显存 | 备注 |")
    print("|---|---|---|---|")
    for r in ROWS:
        print(f"| {r['case']} | {r['seconds']}s | {r['peak_gb'] or '—'}GB | {r['note']} |")
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
