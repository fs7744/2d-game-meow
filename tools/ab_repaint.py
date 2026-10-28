"""局部 AI 重绘（w5）的真实验证：像素素材上改局部，看质量与「选区外不变」。

跑法：.venv/Scripts/python.exe tools/ab_repaint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflows import generate as g          # noqa: E402
from workflows.engine import Engine          # noqa: E402

OUT = ROOT / "assets" / "_ab"
OUT.mkdir(parents=True, exist_ok=True)


def make_pair(w=512, h=512):
    """用真实像素素材：像素 LoRA 出的白猫骑士（assets/_chain_a_*）。

    合成色块当源图没有参考价值——模型对纯几何图形没什么可画的，
    量出来的"改了多少"没有画质意义。真实素材才看得出接缝与风格是否延续。
    """
    real = sorted((ROOT / "assets").glob("_chain_a_*.png"))
    if not real:
        raise SystemExit("缺少真实素材 assets/_chain_a_*.png")
    src = OUT / "_rp_src.png"
    im = Image.open(real[0]).convert("RGB")
    im = im.resize((w, h), Image.LANCZOS) if im.size != (w, h) else im
    im.save(src)

    # 蒙版盖在角色躯干中央偏左（披风/身体那块）
    mk = OUT / "_rp_mask.png"
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle(
        [w // 2 - 55, h // 2 - 40, w // 2 + 5, h // 2 + 60], fill=255)
    m.save(mk)
    return src, mk


def main() -> None:
    eng = Engine(ROOT / "models")
    src, mk = make_pair()
    LORA = ROOT / "models" / "lora" / "px_illustrious.safetensors"

    cases = [
        # 羽化 0（硬边）vs 羽化 8（软边）+ 边界色彩校正
        ("hard_st95", dict(lora=str(LORA), lora_scale=0.75, steps=25, strength=0.95,
                           palette_snap=False, feather=0, match_color=False)),
        ("soft_st95", dict(lora=str(LORA), lora_scale=0.75, steps=25, strength=0.95,
                           palette_snap=False, feather=8, match_color=True)),
        ("hard_st10", dict(lora=str(LORA), lora_scale=0.75, steps=25, strength=1.0,
                           palette_snap=False, feather=0, match_color=False)),
        ("soft_st10", dict(lora=str(LORA), lora_scale=0.75, steps=25, strength=1.0,
                           palette_snap=False, feather=8, match_color=True)),
    ]

    results = []
    for tag, kw in cases:
        r = g.ai_repaint(eng, image=src, mask=mk,
                         prompt="golden armor on the chest",
                         preset="pixel", scope="_abtest", tag=tag, **kw)
        out = Path(r["file"])
        a = np.asarray(Image.open(src).convert("RGBA")).astype(int)
        b = np.asarray(Image.open(out).convert("RGBA")).astype(int)
        outside = np.asarray(Image.open(mk)) <= 127
        same = bool((a[outside] == b[outside]).all())
        inside_changed = float((a[~outside] != b[~outside]).any(axis=-1).mean())
        print(f"[{tag}] {out.name}  outside_identical={same}  "
              f"inside_changed={inside_changed:.3f}  "
              f"peak={r['meta']['peak_vram_gb']}GB  {r['meta']['seconds']}s  "
              f"target={r['meta']['target_size']}")
        results.append((tag, out, src, same))

    # 对照图：源 / 蒙版 / 两种结果
    cell = 260
    sheet = Image.new("RGB", (cell * 6, cell + 26), (28, 28, 30))
    d = ImageDraw.Draw(sheet)
    panels = [("source", Image.open(src))]
    mkm = Image.open(mk).convert("L")
    panels.append(("mask", Image.merge("RGBA", (mkm, mkm, mkm, mkm))))
    panels += [(t, Image.open(p)) for t, p, _, _ in results]
    for i, (label, im) in enumerate(panels):
        x = i * cell
        bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA").resize((cell, cell), Image.NEAREST), (0, 0))
        sheet.paste(bg.convert("RGB"), (x, 26))
        d.text((x + 6, 7), label, fill=(235, 235, 235))
    path = OUT / "rp_sheet.png"
    sheet.save(path)
    print("sheet:", path)


if __name__ == "__main__":
    main()
