"""质量 A/B 实验器：同 seed 下对采样参数与后处理做对照，量出「哪一档更好」。

用法：
  python tools/ab_quality.py                 # 采样档 + 后处理档都跑
  python tools/ab_quality.py --only post     # 只测后处理（最省时间，1 张图）

输出 assets/_ab/ 下的 PNG 与一张带标签的 sheet.png（2 行 × N 列），
并打印每个变体的：拉普拉斯方差（锐度）、唯一色数、像素块均匀率。

指标怎么看：
  锐度     拉普拉斯方差，越高越「有棱角」。像素画要的是高，不是低。
  唯一色数 量化后必须 <= palette，超了说明量化没生效。
  块均匀率 每个 block×block 格子里只有 1 种颜色的比例。像素画应当接近 1.0；
           偏低说明像素化被后续步骤（量化抖动/清洗）搅浑了。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows import clip                      # noqa: E402
from workflows import matte as matte_mod          # noqa: E402
from workflows import postprocess as pp          # noqa: E402
from workflows import presets                    # noqa: E402
from workflows.engine import Engine              # noqa: E402

OUT = ROOT / "assets" / "_ab"
PROMPT = "blue slime monster with a tiny gold crown"
SEED = 4242
LORA = "models/lora/px_illustrious.safetensors"


def flatted(img: Image.Image) -> np.ndarray:
    """透明底压到白底再量指标，否则 alpha=0 区域会被当成黑色高对比。"""
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB"))


def metrics(img: Image.Image, block: int = 0) -> dict:
    import cv2

    rgb = flatted(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    solid = arr[..., 3] > 127
    colors = len({tuple(p) for p in arr[solid][:, :3]})

    uni = -1.0
    if block > 1:
        h, w = solid.shape
        hh, ww = h // block * block, w // block * block
        cells = arr[:hh, :ww].reshape(hh // block, block, ww // block, block, 4)
        cells = cells.transpose(0, 2, 1, 3, 4).reshape(-1, block * block, 4)
        # 逐像素比「与块内第一个像素是否完全相同」——用 np.void/view 比会退化成比字节
        uni = float((cells == cells[:, :1, :]).all(axis=(1, 2)).mean())
    return {"sharp": round(sharp, 1), "colors": colors, "uniform": round(uni, 3)}


def gen(pipe, prompt: str, neg: str, cfg: float, clip_skip: int, size: int = 768) -> Image.Image:
    kw = {}
    if clip_skip != 1:      # pipeline 自带的 clip_skip 在本机 transformers 5.x 下会崩
        pe, ne = clip.embeds(pipe, prompt, neg, clip_skip)
        kw = dict(prompt_embeds=pe, negative_prompt_embeds=ne)
    else:
        kw = dict(prompt=prompt, negative_prompt=neg)
    return pipe(width=size, height=size, num_inference_steps=30, guidance_scale=cfg,
                generator=torch.Generator("cuda").manual_seed(SEED), **kw).images[0]


def crop_center(img: Image.Image, size: int = 320) -> Image.Image:
    """按 alpha 包围盒中心裁 size×size 的 1:1 局部——像素风要看真实像素，不能被缩放糊掉。"""
    rgba = img.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    cx, cy = (rgba.width // 2, rgba.height // 2) if not bbox else \
        ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    x = max(0, min(rgba.width - size, cx - size // 2))
    y = max(0, min(rgba.height - size, cy - size // 2))
    return rgba.crop((x, y, x + size, y + size))


def sheet(items: list[tuple[str, Image.Image, dict, str]], cell: int = 320, cols: int = 4):
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cell * cols, (cell + 26) * rows), (28, 28, 30))
    d = ImageDraw.Draw(canvas)
    for i, (label, img, mt, mode) in enumerate(items):
        x, y = (i % cols) * cell, (i // cols) * (cell + 26)
        if mode == "crop":
            tile = crop_center(img, cell)
        else:
            tile = img.convert("RGBA").resize((cell, cell), Image.LANCZOS)
        bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
        bg.alpha_composite(tile.convert("RGBA"), (0, 0))
        canvas.paste(bg.convert("RGB"), (x, y + 26))
        d.text((x + 6, y + 7), label, fill=(235, 235, 235))
        d.text((x + 6, y + cell + 10),
               f"sharp {mt['sharp']}  colors {mt['colors']}  uni {mt['uniform']}",
               fill=(160, 190, 220))
    path = OUT / "sheet.png"
    canvas.save(path)
    return path


def white_ratio(img: Image.Image, band: int = 24, tol: int = 18) -> float:
    """画布四边 band 像素内「接近纯白」的比例。精灵管线的第一硬指标：背景要白。

    传 0（非白底）→ 抠图只能靠 u2net，边缘必糊；色键后端在这种图上直接失效。
    """
    rgb = flatted(img)
    h, w = rgb.shape[:2]
    band = min(band, h // 2, w // 2)
    edge = np.concatenate([
        rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
        rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3),
    ])
    return float(((255 - edge.astype(np.int16)).max(axis=1) <= tol).mean())


def blobs(img: Image.Image, min_px: int = 400) -> int:
    """抠图后的前景连通域个数。>1 基本就是「画面里出现了多个主体/道具」。"""
    import cv2

    fg = (np.asarray(img.convert("RGBA"))[..., 3] > 127).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    return int(sum(1 for a in stats[1:, cv2.CC_STAT_AREA] if a >= min_px))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["post", "sampling", "bg", "all"], default="all")
    ap.add_argument("--size", type=int, default=768)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    eng = Engine(ROOT / "models")
    prompt, neg = presets.build(PROMPT, None, "pixel")
    pipe = eng.txt2img_pipe("sd15")
    eng.apply_lora(pipe, LORA, 0.75)

    items: list[tuple[str, Image.Image, dict, str]] = []

    if args.only in ("sampling", "all"):
        for cfg, cs in [(7.0, 1), (6.0, 1), (6.5, 2), (7.0, 2)]:
            raw = gen(pipe, prompt, neg, cfg, cs, args.size)
            tag = f"cfg{cfg} cs{cs}"
            raw.save(OUT / f"raw_{tag.replace(' ', '_')}.png")
            # 采样档要看「模型本身画得怎么样」，所以量未像素化的原图
            items.append((tag, raw, metrics(raw), "fit"))
            print(f"[sampling] {tag}: {items[-1][2]}")

    if args.only in ("post", "all"):
        raw = gen(pipe, prompt, neg, 6.5, 2, args.size)
        raw = matte_mod.despeckle(matte_mod.matte_auto(raw.convert("RGBA"), "auto",
                                                       eng.matte_dir))
        raw.save(OUT / "raw_for_post.png")
        variants = {
            "LANCZOS+32": lambda im: pp.quantize_palette(
                pp.pixelate(im, 6, resample=Image.LANCZOS), 32),
            "BOX+32": lambda im: pp.quantize_palette(
                pp.pixelate(im, 6, resample=Image.BOX), 32),
            "BOX+24": lambda im: pp.quantize_palette(
                pp.pixelate(im, 6, resample=Image.BOX), 24),
            "BOX+24+clean": lambda im: pp.clean_isolated(
                pp.quantize_palette(pp.pixelate(im, 6, resample=Image.BOX), 24), passes=1),
        }
        for name, fn in variants.items():
            im = fn(raw)
            im.save(OUT / f"post_{name}.png")
            mt = metrics(im, 6)
            items.append((name, im, mt, "crop"))
            print(f"[post] {name}: {mt}")

    if args.only in ("bg", "all"):
        # 精灵管线的第一硬指标是「背景干净且够白」，不然抠图只能靠 u2net，边缘必糊。
        pos, neg0 = presets.build(f"pixelart, {PROMPT}", None, "pixel")
        extra = ", colored background, blue background, gradient background, scenery"
        cases = [
            ("cs1 baseNeg", 6.5, 1, neg0),
            ("cs2 baseNeg", 6.5, 2, neg0),
            ("cs2 +negBg", 6.5, 2, presets.fit(neg0 + extra)),
            ("cs2 +negBg cfg8.5", 8.5, 2, presets.fit(neg0 + extra)),
        ]
        for name, cfg, cs, neg in cases:
            raw = gen(pipe, pos, neg, cfg, cs, args.size)
            cut = matte_mod.despeckle(
                matte_mod.matte_auto(raw.convert("RGBA"), "auto", eng.matte_dir))
            mt = metrics(raw)
            mt["white"] = round(white_ratio(raw), 3)
            mt["blobs"] = blobs(cut)
            raw.save(OUT / f"bg_{name.replace(' ', '_').replace('+', '')}.png")
            items.append((f"{name}  white{mt['white']} blobs{mt['blobs']}", raw, mt, "fit"))
            print(f"[bg] {name}: {mt}")

    print("sheet:", sheet(items))


if __name__ == "__main__":
    main()
