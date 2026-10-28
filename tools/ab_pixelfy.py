"""完美像素转换（wpx）的真实素材验证。

跑法：.venv/Scripts/python.exe tools/ab_pixelfy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows import pixelfy as pxf          # noqa: E402
from workflows import perfect_pixel as pp      # noqa: E402

OUT = ROOT / "assets" / "_ab"
OUT.mkdir(parents=True, exist_ok=True)


def one_color_per_cell(arr: np.ndarray, cell: int) -> float:
    """每格内颜色唯一的比例——像素画的硬指标。"""
    h, w = arr.shape[:2]
    hh, ww = h // cell * cell, w // cell * cell
    if hh == 0 or ww == 0:
        return 0.0
    cells = arr[:hh, :ww].reshape(hh // cell, cell, ww // cell, cell, -1)
    cells = cells.transpose(0, 2, 1, 3, 4).reshape(-1, cell * cell, cells.shape[-1])
    uniq = [len(set(map(tuple, c.tolist()))) for c in cells]
    return float(np.mean([u == 1 for u in uniq]))


def main() -> None:
    real = sorted((ROOT / "assets").glob("_chain_a_*.png"))
    if not real:
        raise SystemExit("缺少真实素材 assets/_chain_a_*.png")
    src = real[0]
    im = Image.open(src).convert("RGBA")
    print(f"源图: {src.name} {im.size}")

    arr = np.asarray(im.convert("RGB"))
    gw, gh = pp.detect_grid_scale(arr)
    print(f"检测网格: {gw}×{gh}")

    panels = [("source", im)]
    for label, kw in [
        ("auto_32", dict(colors=32, snap_to=16)),
        ("auto_16", dict(colors=16, snap_to=16)),
        ("target64", dict(colors=32, target_size=[64, 64])),
        ("nosnap", dict(colors=32, grid_snap=False)),
    ]:
        res = pxf.refine_batch([im], **kw)
        item = res["items"][0]
        out = item["image"]
        m = item["meta"]
        a = np.asarray(out.convert("RGB"))
        cell = max(1, min(out.size) // max(1, m["grid"][0] or 1))
        print(f"[{label}] {out.size} grid={m['grid']} colors={m['colors']} "
              f"offPal={m['offPalette']} alphaBin={m['alphaBinary']} "
              f"oneColorPerCell={one_color_per_cell(a, cell):.3f}")
        panels.append((label, out))

    cell = 300
    sheet = Image.new("RGB", (cell * len(panels), cell + 26), (28, 28, 30))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, (label, im2) in enumerate(panels):
        x = i * cell
        bg = Image.new("RGBA", (cell, cell), (235, 235, 235, 255))
        bg.alpha_composite(im2.convert("RGBA").resize((cell, cell), Image.NEAREST), (0, 0))
        sheet.paste(bg.convert("RGB"), (x, 26))
        d.text((x + 6, 7), f"{label} {im2.size}", fill=(235, 235, 235))
    path = OUT / "pixelfy_sheet.png"
    sheet.save(path)
    print("sheet:", path)


if __name__ == "__main__":
    main()
