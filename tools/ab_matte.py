"""抠图后端 A/B：色键（matte）vs 语义分割（u2net / isnet）。

为什么要有这个工具：像素画对 alpha 边缘极敏感——多留一圈背景色就是「白边」，
抠掉一条 1px 描边就是「缺胳膊」。这种差异肉眼看两张图很难说清，必须量。

量什么：
  mid       中间 alpha（8~247）占比。像素画要硬边，越接近 0 越好。
  cov       前景占比。太低=主体被吃光，太高=背景留着。
  fringe    混色边像素数（matte.has_fringe）。白边的主指标。
  blobs     前景连通块数。>1 说明主体被打碎（白袍角色被色键误伤的典型表现）。
  peri_area 周长/面积。细长结构（剑、触角）被吃掉会让它掉下来。
  IoU       两个 mask 的重合度。低于 0.9 说明两种后端对「什么是主体」分歧很大。

用法：
  python tools/ab_matte.py                 # 现生成 4 个样本（约 30s）
  python tools/ab_matte.py --reuse         # 复用 assets/_ab 里上次的样本
输出 assets/_ab/mt_<name>_{raw,color,u2net}.png 与 mt_sheet.png（raw|color|u2net 三联）。

实测结论（2026-10-28，768²，SD1.5）：
  - 像素 LoRA 出图背景是彩色/纹理 → 色键几乎抠不动（cov 0.90~0.98，IoU 0.13~0.33）。
  - 武侠白袍 + 白底 → 色键把袍子当背景，主体碎成 3 块（IoU 0.63）。
  - 两个后端 alpha **都是二值**（mid=0），所以「u2net 边缘软、不适合像素画」这个
    假设不成立 —— auto 继续默认 u2net 是对的。
  - 但 u2net 会整张抠空（草稿档 256²/8 步实测只剩 0.44%），这才需要回退色键，
    见 matte.matte_auto 的 FG_LO/FG_HI 保护。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows import generate as g                # noqa: E402
from workflows import matte as m                   # noqa: E402
from workflows import postprocess as pp            # noqa: E402
from workflows.engine import Engine                # noqa: E402

OUT = ROOT / "assets" / "_ab"


def metrics(rgba: Image.Image) -> dict:
    import cv2

    a = np.asarray(rgba.convert("RGBA"))[..., 3]
    hard = (a > 127).astype(np.uint8)
    cnts, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    peri = sum(cv2.arcLength(c, True) for c in cnts)
    area = float(hard.sum()) or 1.0
    return {
        "mid": round(float(((a > 8) & (a < 247)).mean()), 4),
        "cov": round(m.fg_ratio(rgba), 4),
        "fringe": m.has_fringe(rgba),
        "blobs": pp.foreground_blobs(rgba),
        "peri_area": round(peri / area, 4),
    }


def iou(x: Image.Image, y: Image.Image) -> float:
    a = np.asarray(x.convert("RGBA"))[..., 3] > 127
    b = np.asarray(y.convert("RGBA"))[..., 3] > 127
    return float((a & b).sum() / max(1, (a | b).sum()))


def tile(items: list[tuple[str, Image.Image]], cell: int = 380) -> Image.Image:
    sheet = Image.new("RGB", (cell * len(items), cell + 20), (30, 30, 33))
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(items):
        bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA").resize((cell, cell), Image.NEAREST), (0, 0))
        sheet.paste(bg.convert("RGB"), (i * cell, 20))
        d.text((i * cell + 5, 5), label, fill=(235, 235, 235))
    return sheet


def build_samples(eng: Engine) -> dict[str, Image.Image]:
    cases: dict[str, Image.Image] = {}
    kw = dict(width=768, height=768, steps=30, cfg=6.5, preset="pixel",
              lora="models/lora/px_illustrious.safetensors", lora_scale=0.75)
    for name, prompt, seed in [("slime", "blue slime monster", 7),
                               ("knight", "white cat knight, red cape, katana", 8)]:
        r = g.txt2img(eng, prompt=prompt, seed=seed, tag=f"_mt_{name}", **kw)
        cases[name] = Image.open(r["file"]).convert("RGBA")

    r = g.txt2img(eng, prompt="white robed swordsman", preset="wuxia", width=768,
                  height=768, steps=30, cfg=6.5, seed=9,
                  lora="models/lora/wuxia_v2.safetensors", lora_scale=0.8, tag="_mt_wuxia")
    cases["wuxia_whiterobe"] = Image.open(r["file"]).convert("RGBA")

    poses = sorted((ROOT / "assets").glob("_pose_walk_*.png"))[:1]
    if poses:
        r = g.poseswap(eng, pose_images=poses, prompt="pixel art knight, red armor",
                       preset="pixel", steps=20, seed=11, width=768, height=768, cols=1,
                       lora="models/lora/px_illustrious.safetensors", lora_scale=0.75,
                       gate=False, tag="_mt_pose")
        cases["pose_frame"] = Image.open(
            sorted(Path(r["frames_dir"]).glob("*.png"))[0]).convert("RGBA")
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="复用 assets/_ab 里的样本")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    eng = Engine(ROOT / "models")
    names = ["slime", "knight", "wuxia_whiterobe", "pose_frame"]
    if args.reuse and all((OUT / f"mt_{n}_raw.png").exists() for n in names):
        cases = {n: Image.open(OUT / f"mt_{n}_raw.png").convert("RGBA") for n in names}
    else:
        cases = build_samples(eng)

    rows = []
    for name, raw in cases.items():
        cm = m.despeckle(m.matte(raw))
        am = m.despeckle(m.matte_ai(raw, eng.matte_dir))
        mc, ma = metrics(cm), metrics(am)
        print(f"{name:18s} color={mc}")
        print(f"{'':18s} u2net={ma}  IoU={iou(cm, am):.3f}")
        raw.save(OUT / f"mt_{name}_raw.png")
        cm.save(OUT / f"mt_{name}_color.png")
        am.save(OUT / f"mt_{name}_u2net.png")
        rows.append(tile([(f"{name} raw", raw), ("color", cm), ("u2net", am)]))

    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)),
                      (18, 18, 20))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(OUT / "mt_sheet.png")
    print("sheet:", OUT / "mt_sheet.png")


if __name__ == "__main__":
    main()
