"""步数对「姿态贴合度」的影响（w4 / wss 的主战场）。

精灵图那批实验说明 16 步够用，但 w4 的产出是**按骨架摆姿势**的图——
步数少了会不会连姿势也摆不准？这里量的是同一个指标（成品关节 vs 骨架关节
的平均偏差），只是把步数从 24 降到 16 / 12。

顺带测 1024²（立绘档）下的 16 vs 24 步，确认立绘也能降步。

用法：
    .venv\\Scripts\\python.exe tools/exp_steps_pose.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tools.make_pose import render, walk_pose  # noqa: E402
from workflows import paths
from workflows import generate as g  # noqa: E402
from workflows import refprep  # noqa: E402
from workflows.engine import Engine  # noqa: E402

OUT = paths.tool_out("_steps_pose")
OUT.mkdir(parents=True, exist_ok=True)

STEPS = (12, 16, 20, 24)
FRAMES = 3


def pose_deviation(img, eng, kp, size=768):
    cut, _e, _s = g._postprocess(img, eng, matte=True, matte_backend="auto",
                                 denoise=1, pixel=0, palette=0, clean_iso=0)
    flat = Image.new("RGB", cut.size, (255, 255, 255))
    flat.paste(cut, (0, 0), cut)
    pts = eng.pose_extractor().visible_points(flat, thr=0.25)
    diffs = [np.hypot(pts[k][0] - kp[k][0], pts[k][1] - kp[k][1]) * size
             for k in range(14) if k in pts]
    return float(np.mean(diffs)) if diffs else float("nan")


def main() -> int:
    eng = Engine("models")
    char_path = Path("assets/_qwen_verify/char.png")
    if not char_path.is_file():
        print("缺素材 assets/_qwen_verify/char.png")
        return 1
    char = refprep.prepare_reference(
        Image.open(char_path).convert("RGBA"), size=768)
    prompt = "a pixel art white cat knight with red cape and golden armor"

    print("=== 姿态贴合度（768²，w4 的单图条件）===")
    for steps in STEPS:
        devs, times = [], []
        for i in range(FRAMES):
            kp = walk_pose(i / FRAMES)
            cond = g._qwen_cond_image(render(kp, size=(768, 768)))
            instr = ("The images are: first the character reference, then a body "
                     "skeleton showing the target pose. Redraw the same character "
                     "in exactly that pose, full body, centered, "
                     + g._qwen_backdrop("dark") + "." + g.QWEN_FRAME_RULES)
            t0 = time.perf_counter()
            im = g._qwen_call(eng, prompt=instr, negative=g.QWEN_FRAME_NEG,
                              ref_image=[char, cond], width=768, height=768,
                              steps=steps, cfg=1.0, seed=2026 + i,
                              model="qwen-image-2.1", label="w4")
            times.append(time.perf_counter() - t0)
            im.save(OUT / f"pose_{steps}_{i}.png")
            devs.append(pose_deviation(im, eng, kp))
        print(f"[{steps:2d} 步] 偏差均值 {np.mean(devs):5.0f}px  "
              f"逐帧 {[round(d) for d in devs]}  用时 {np.mean(times):.1f}s/帧",
              flush=True)

    print("\n=== 立绘档 1024²（16 vs 24 步）===")
    for steps in (16, 24):
        t0 = time.perf_counter()
        im = g._qwen_call(eng, prompt=prompt, negative=g.QWEN_NEG,
                          width=1024, height=1024, steps=steps, cfg=1.0, seed=42,
                          model="qwen-image-2.1", label="portrait")
        dt = time.perf_counter() - t0
        im.save(OUT / f"portrait_{steps}.png")
        print(f"[{steps} 步] {dt:.1f}s")
    a = Image.open(OUT / "portrait_16.png").convert("RGBA")
    b = Image.open(OUT / "portrait_24.png").convert("RGBA")
    sheet = Image.new("RGB", (2 * 392, 384), (255, 255, 255))
    for i, im in enumerate((a, b)):
        im = im.resize((384, 384), Image.LANCZOS)
        bg = Image.new("RGBA", im.size, (110, 110, 110, 255))
        bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), (i * 392, 0))
    sheet.save(OUT / "portrait_cmp.png")
    print("对照图：", OUT / "portrait_cmp.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
