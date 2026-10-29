"""Qwen-only 可行性验证（先验证再定重写范围）。

A) w5 局部 AI 重绘 → Qwen：裁剪 + 指令改图 + 羽化贴回
   关键指标：选区外逐点不变（蒙版保证）、选区内是否真的按指令改了、耗时
B) Qwen 逐帧出图（w4/wss 的底座）：耗时/帧、关节数（姿态一致性）、
   可交付率三要素（单主体 ∧ 无贴边 ∧ 前景 5~60%）
C) Qwen-Image-Edit 2511 + 20B 组件能否做 mask inpaint（静态核实）
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from tools.make_pose import render, walk_pose  # noqa: E402
from workflows import quality as q  # noqa: E402
from workflows import repaint as rp  # noqa: E402
from workflows.engine import Engine  # noqa: E402
from workflows.generate import QWEN_NEG, _qwen_call  # noqa: E402
from workflows.generate import matte_bg as gen_matte_bg  # noqa: E402

OUT = Path("assets/_qwen_verify")
OUT.mkdir(parents=True, exist_ok=True)
NEG = QWEN_NEG


def log(*a):
    print(*a, flush=True)


def make_char(eng: Engine) -> Image.Image:
    """先用 Qwen 生成一张角色立绘，当后续验证的素材（已有则复用，省 60s）。"""
    cached = OUT / "char.png"
    if cached.is_file():
        log("[A0] 复用已生成的角色立绘")
        return Image.open(cached).convert("RGB")
    t = time.time()
    im = _qwen_call(
        eng, prompt="a pixel art white cat knight with red cape and golden armor, "
                    "standing, full body, plain white background",
        negative=NEG, width=768, height=768, steps=24, cfg=4.0, seed=5,
        model="qwen-image-2.1", label="char")
    log(f"[A0] 角色立绘 {time.time() - t:.1f}s")
    im.convert("RGB").save(OUT / "char.png")
    return im.convert("RGB")


def verify_a(eng: Engine, char: Image.Image) -> dict:
    """A) 局部重绘：Qwen 指令改图 + 蒙版贴回。"""
    src = char.convert("RGBA")
    msk = Image.new("L", src.size, 0)
    d = ImageDraw.Draw(msk)
    # 选中上半身一块（盔甲位置）
    w, h = src.size
    d.rectangle([int(w * 0.34), int(h * 0.22), int(w * 0.66), int(h * 0.46)], fill=255)
    crop, cmask, box = rp.crop_region(src, msk, expand=8, context=0.35)
    log(f"[A] 裁剪 {crop.size} 框={box}")

    t = time.time()
    gen = _qwen_call(
        eng,
        prompt="Edit this image: change the armor to bright emerald green, "
               "keep the character identity, pose and background unchanged.",
        negative=NEG, ref_image=[crop.convert("RGB")],
        width=768, height=768, steps=24, cfg=4.0, seed=5,
        model="qwen-image-2.1", label="w5q")
    dt = time.time() - t
    gen = gen.convert("RGB")
    if gen.size != crop.size:
        gen = gen.resize(crop.size, Image.LANCZOS)
    out = rp.composite(src, gen, msk, feather_px=6)
    out.convert("RGB").save(OUT / "a_local_edit.png")

    # 指标
    a = np.asarray(src.convert("RGB"), dtype=np.int16)
    b = np.asarray(out.convert("RGB"), dtype=np.int16)
    m = np.asarray(msk, dtype=np.uint8) > 127
    outside = ~m
    outside_diff = float(np.abs(a - b)[outside].max()) if outside.any() else 0.0
    inside_diff = float(np.abs(a - b)[m].mean()) if m.any() else 0.0
    log(f"[A] 耗时 {dt:.1f}s | 选区外最大像素差 {outside_diff:.0f} "
        f"| 选区内平均差 {inside_diff:.1f}")
    return {"seconds": round(dt, 1), "outside_max_diff": outside_diff,
            "inside_mean_diff": round(inside_diff, 1)}


def verify_b(eng: Engine, char: Image.Image, extractor) -> dict:
    """B) 逐帧出图：骨架 → 全身（w4/wss 的底座）。"""
    rows = []
    for i in range(3):
        kp = walk_pose(i / 3)
        skel = render(kp, size=(768, 768))
        skel.save(OUT / f"b_skel_{i}.png")
        t = time.time()
        im = _qwen_call(
            eng,
            prompt="The first image is the character reference. The second image is a "
                   "pose skeleton. Redraw the same character in exactly that pose, "
                   "full body, centered, plain white background, pixel art.",
            negative=NEG, ref_image=[char, skel], width=768, height=768,
            steps=24, cfg=4.0, seed=100 + i, model="qwen-image-2.1", label="w4q")
        dt = time.time() - t
        im = im.convert("RGB")
        im.save(OUT / f"b_frame_{i}.png")
        matted.save(OUT / f"b_frame_{i}_matted.png")
        # 先抠图再评估：Qwen 不开透明底时输出是 RGB，alpha 全 255 →
        # coverage 恒为 1.0，"前景 5~60%" 的判据必须对**抠完的**图算。
        matted, minfo = gen_matte_bg(
            eng, init_image=OUT / f"b_frame_{i}.png", matte_backend="auto",
            scope="qwen_verify")
        sc = q.evaluate(matted, extractor)
        rows.append((round(dt, 1), sc))
        log(f"[B] 帧{i}: {dt:.1f}s points={sc['points']} coverage={sc['coverage']} "
            f"edge={len(sc['edge'])} passed={sc['passed']}")
    avg = round(sum(r[0] for r in rows) / len(rows), 1)
    ok = sum(1 for r in rows if r[1]["passed"] and 0.05 <= r[1]["coverage"] <= 0.60)
    log(f"[B] 平均 {avg}s/帧，可交付(通过闸门且前景5~60%) {ok}/{len(rows)}")
    return {"avg_seconds": avg, "deliverable": f"{ok}/{len(rows)}",
            "frames": [{"seconds": s, **m} for s, m in rows]}


def verify_c() -> dict:
    """C) Qwen-Image-Edit 2511 能不能做 mask inpaint。"""
    import json
    import inspect
    from diffusers import QwenImageEditPipeline

    d = Path("D:/code/pixelforge/models/qwen-image-edit-2511")
    idx = json.loads((d / "model_index.json").read_text("utf-8")) if (d / "model_index.json").is_file() else {}
    have = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    sig = list(inspect.signature(QwenImageEditPipeline.__call__).parameters)
    mask_ok = "mask_image" in sig or "mask" in sig
    log(f"[C] edit-2511 组件: {have}")
    log(f"[C] model_index._class_name = {idx.get('_class_name')}")
    log(f"[C] QwenImageEditPipeline 支持 mask: {mask_ok}")
    return {"components": have, "class_name": idx.get("_class_name"),
            "mask_supported": mask_ok}


def main() -> int:
    log("=== C（静态核实） ===")
    c = verify_c()
    eng = Engine("models")
    ex = eng.pose_extractor()
    log("=== A ===")
    char = make_char(eng)
    a = verify_a(eng, char)
    log("=== B ===")
    b = verify_b(eng, char, ex)
    log("=== 汇总 ===")
    log({"A": a, "B": b, "C": c})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
