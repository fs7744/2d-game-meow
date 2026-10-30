"""对比两个 VNCCS LoRA 版本（V1 vs V2）在同一组输入下的表现。

一次加载模型，循环 [LoRA × seed]，避免反复冷启动。
输出 `assets/_vnccs_cmp_<版本>_<seed>.png` 和一张总对比图。

指标看三样（单看像素相似度没意义——姿态变了像素必然不同）：
- **清晰度/头部清晰度**：糊没糊
- **服装配色相似度**：与参考图在主体区域的颜色直方图相关性，
  用来判断"服装有没有漂"
- 人工看姿态与身份

用法: python tools/compare_vnccs_versions.py [--seeds 43,7]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from workflows.engine import Engine

TEMPLATE = ("replace the pose of <image 2> with the pose of <image 1>. "
            "keep the character of <image 2>")
IDENTITY = (
    "The character is a female warrior with long black hair tied in a low ponytail, "
    "fair skin, wearing tight dark blue leather armor with visible stitching lines and "
    "light blue highlight patches on the shoulders and elbows, a dark cloth belt with "
    "hanging skirt panels, matching blue pants with knee pads, and dark grey tall boots "
    "with silver metal buckles. 2D game character sprite, flat cel shading, "
    "clean black outlines, plain white background, full body, detailed."
)


def pad(im: Image.Image, W: int, H: int) -> Image.Image:
    r = min(W / im.width, H / im.height)
    nw, nh = max(1, round(im.width * r)), max(1, round(im.height * r))
    c = Image.new("RGB", (W, H), (255, 255, 255))
    c.paste(im.resize((nw, nh), Image.LANCZOS), ((W - nw) // 2, (H - nh) // 2))
    return c


def hist_sim(a: Image.Image, b: Image.Image, k: int = 32) -> float:
    """两图颜色直方图的相关性（1 = 分布一致）。用来衡量服装配色漂没漂。"""
    def h(im):
        arr = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.float32)
        # 去掉接近白的背景像素，只看主体
        m = arr.max(axis=2) < 240
        if m.sum() < 100:
            m = np.ones_like(m)
        out = []
        for c in range(3):
            hh, _ = np.histogram(arr[..., c][m], bins=k, range=(0, 255), density=True)
            out.append(hh)
        return np.concatenate(out)
    ha, hb = h(a), h(b)
    return float(np.corrcoef(ha, hb)[0, 1])


def sharp(im: Image.Image) -> float:
    g = np.asarray(im.convert("L"), dtype=np.float32)
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
           - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="assets/_cn_none.png")
    ap.add_argument("--pose", default="assets/_pose_walk_07.png")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seeds", default="43,7")
    ap.add_argument("--loras", default="vnccs_pose,vnccs_pose_v2")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    loras = [s.strip() for s in args.loras.split(",") if s.strip()]

    eng = Engine("models")
    pipe = eng.txt2img_pipe("qwen-image-2.1")
    ref = Image.open(args.ref).convert("RGB")
    pose = Image.open(args.pose).convert("RGB")
    W = max(256, int(pose.size[0] * 1.5) // 16 * 16)
    H = max(256, int(pose.size[1] * 1.5) // 16 * 16)
    print(f"画布 {W}x{H}（骨架比例 ×1.5）", flush=True)

    prompt = f"{TEMPLATE}\n{IDENTITY}"
    tiles, rows = [], []
    for name in loras:
        eng.apply_loras(pipe, [{"name": name, "scale": 1.0}])
        print(f"\n=== {name} ===", flush=True)
        for sd in seeds:
            t = time.time()
            out = pipe(prompt=prompt,
                       image=[pad(pose, W, H), pad(ref, W, H)],
                       width=W, height=H, num_inference_steps=args.steps,
                       true_cfg_scale=1.0, negative_prompt="",
                       generator=torch.Generator("cuda").manual_seed(sd),
                       output_resolution=max(W, H)).images[0].convert("RGB")
            p = f"assets/_vnccs_cmp_{name}_{sd}.png"
            out.save(p)
            hs, sh = hist_sim(ref, out), sharp(out)
            print(f"  seed={sd:<4} {time.time()-t:5.1f}s  配色相似={hs:.4f}  "
                  f"清晰度={sh:.1f}  → {p}", flush=True)
            tiles.append(out)
        eng.unload_loras(pipe)

    # 总图：每行一个版本
    cols = len(seeds)
    grid = Image.new("RGB", (W * (cols + 1), H * len(loras)), "white")
    for r, name in enumerate(loras):
        grid.paste(pad(ref, W, H), (0, r * H))          # 每行第一格都是参考图
        for c, sd in enumerate(seeds):
            im = Image.open(f"assets/_vnccs_cmp_{name}_{sd}.png")
            grid.paste(im, ((c + 1) * W, r * H))
    grid.save("assets/_vnccs_versions_grid.png")
    print(f"\n  → assets/_vnccs_versions_grid.png"
          f"（行={','.join(loras)} 列=参考图 + seed {seeds}）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
