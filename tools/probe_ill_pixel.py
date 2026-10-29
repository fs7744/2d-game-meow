"""Illustrious 类底模上跑像素 LoRA 的对照工具。

**为什么单独写**：项目原有的 `tools/ab_base.py` 用的是自然语言契约
（preset=pixel：`a knight in silver armor...`），而 Illustrious 是在 danbooru
tag 上训的，不认自然语言构图词——实测同一个 LoRA 挂 sdxl-anime：

    preset=pixel（自然语言）   可交付 17%，贴边 83%
    preset=anime（tag）        可交付 50%，贴边  0%

所以测 Illustrious 必须换 tag 契约，否则量出来的是"契约错了"而不是"LoRA 不行"。
这个工具就是干这件事的。

用法：
    # 单个 LoRA
    python tools/probe_ill_pixel.py --lora pixel_xl15_il
    # 多个 LoRA + 不挂 LoRA 的对照
    python tools/probe_ill_pixel.py --lora pixelxl15,pixeldream --base-line

指标口径与产线一致：有效像素固定 128（1024÷8），抠图后算连通域。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from workflows import generate as g
from workflows import lora as lora_reg
from workflows import pixelfy as pxf
from workflows.engine import Engine

OUT = ROOT / "assets" / "_ill_pixel"

# danbooru tag：Illustrious 的主场。`solo` 压多主体、`full body` 压贴边。
PROMPTS = [
    ("knight", "1girl, solo, knight, silver armor, holding sword, red cape, "
               "full body, simple background"),
    ("mage", "1girl, solo, mage, purple robe, witch hat, holding staff, "
             "full body, simple background"),
    ("slime", "green slime, monster, game boss, simple background"),
]


def purity(img: Image.Image) -> float:
    """硬边占比：相邻像素色差 >32 的，占「所有非零色差」的比例。

    **这个指标才是「像素画 vs 低分辨率插画」的分水岭**：可交付率三者都高
    （只要是全身单主体），但像素纯度差 20 倍——

        项目产线的真像素帧        0.447
        Illustrious 直出（无描边） 0.020

    真像素画是双峰分布（同色块内差 0、块边界差很大）；抗锯齿插画的色差
    集中在中小值（渐变过渡），所以硬边占比极低。
    """
    a = np.asarray(img.convert("RGBA"))
    rgb = a[..., :3].astype(np.int16)
    m = a[..., 3] > 127
    dh = np.abs(np.diff(rgb, axis=1)).max(axis=2)
    dv = np.abs(np.diff(rgb, axis=0)).max(axis=2)
    d = np.concatenate([dh[m[:, :-1] & m[:, 1:]], dv[m[:-1, :] & m[1:, :]]])
    nz = d[d > 0]
    return float((nz > 32).mean()) if len(nz) else 0.0


def one(eng, model: str, pid: str, text: str, seed: int, lora: str,
        scale: float, preset: str, size: int, steps: int) -> dict:
    r = g.txt2img(eng, prompt=text, negative=None, preset=preset,
                  width=size, height=size, steps=steps, cfg=6.5, seed=seed,
                  model=model, clip_skip=2, matte=False,
                  lora=lora or "", lora_scale=scale,
                  tag=f"illpx_{Path(lora).stem if lora else 'nolo':s}_{pid}_{seed}")
    raw = Image.open(r["file"])
    block = max(2, size // 128)          # 与产线一致：有效像素固定 128
    out, edge, subjects = g._postprocess(
        raw, eng, matte=True, matte_backend="auto", denoise=1,
        pixel=block, palette=32, clean_iso=1, info={})
    # **必须补这一步**：`_postprocess` 只做色板量化，没有轮廓描边；
    # 而产线（sprite_swap / topixel）走的是 pixelfy.refine_batch(edge_rebuild=True)。
    # 少了描边，硬边占比会从 ~0.45 掉到 ~0.02，看起来像「LoRA 没用」。
    pal = pxf.extract_palette([out], 32)
    if pal:
        out = pxf.enhance_edges(out, pal, outline_only=True)
    a = np.asarray(out.convert("RGBA"))
    fg = float((a[..., 3] > 127).mean()) if a.shape[-1] == 4 else 0.0
    colors = len({tuple(c) for c in a[..., :3][a[..., 3] > 127].tolist()}) \
        if a.shape[-1] == 4 and (a[..., 3] > 127).any() else 0
    ok = (subjects == 1) and (not edge) and (0.05 <= fg <= 0.60)
    return dict(subj=subjects, edge=len(edge or []), fg=fg, colors=colors,
                purity=purity(out), sec=r["meta"]["seconds"], ok=ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdxl-anime")
    ap.add_argument("--lora", default="", help="逗号分隔的 LoRA 名或路径")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--preset", default="anime", choices=["anime", "pixel"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--base-line", action="store_true",
                    help="额外跑一组不挂 LoRA 的对照")
    a = ap.parse_args()

    names = [x for x in a.lora.split(",") if x] or [""]
    if a.base_line and "" not in names:
        names.insert(0, "")

    eng = Engine(ROOT / "models")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        path = lora_reg.resolve(ROOT / "models", name) if name else ""
        label = name or "(不挂 LoRA)"
        got = []
        for pid, text in PROMPTS:
            for i in range(a.seeds):
                seed = 1000 + i * 7919
                m = one(eng, a.model, pid, text, seed, path, a.lora_scale,
                        a.preset, a.size, a.steps)
                got.append(m)
                print(f"  [{label}] {pid} s{seed}: subj={m['subj']} "
                      f"edge={m['edge']} fg={m['fg']:.2f} colors={m['colors']} "
                      f"硬边={m['purity']:.3f} {'OK' if m['ok'] else '×'} "
                      f"{m['sec']:.1f}s", flush=True)
        n = len(got)
        rows.append(dict(label=label, n=n,
                         ok=sum(x["ok"] for x in got) / n,
                         multi=sum(x["subj"] > 1 for x in got) / n,
                         edge=sum(x["edge"] > 0 for x in got) / n,
                         fg=sum(x["fg"] for x in got) / n,
                         colors=sum(x["colors"] for x in got) / n,
                         purity=sum(x["purity"] for x in got) / n,
                         sec=sum(x["sec"] for x in got) / n))

    print()
    print(f"底模 {a.model} · 契约 {a.preset} · {a.size}²/"
          f"{a.steps} 步 · LoRA 强度 {a.lora_scale}")
    print("| LoRA | 样本 | 可交付↑ | 多主体↓ | 贴边↓ | 前景 | 均色数 | 硬边↑ | 耗时 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['n']} | {r['ok']:.0%} | {r['multi']:.0%} | "
              f"{r['edge']:.0%} | {r['fg']:.2f} | {r['colors']:.0f} | "
              f"{r['purity']:.3f} | {r['sec']:.1f}s |")


if __name__ == "__main__":
    main()
