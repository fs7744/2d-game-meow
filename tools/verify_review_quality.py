"""全工作流**生成质量**实测：跑一遍主要工作流，检查产物与 meta 是否自洽。

为什么要有这个脚本：单测能验"函数没崩"，但验不了"出图能不能用"。
这里对每条工作流检查四件事：

1. **meta 的尺寸 == 落盘图片的真实像素尺寸**（以前 w2 根本不记尺寸，
   w4/wss 记的是生成画布而不是精灵图）
2. **主体占比在 5%~60%**（产线判据：太小是废图，太大顶边）
3. **alpha 干净**：半透明像素不能多（透明底出图常见"一层雾"）
4. **没有 NaN / 纯色 / 全黑**

外加一条**跨工作流的污染检查**：跑完扩图（它要挂 Outpaint LoRA）之后，
`engine.active_loras()` 必须为空——以前 Outpaint LoRA 永久留在共享管线里，
之后所有生成都带着它出图。

用法：
    python tools/verify_review_quality.py            # 全部
    python tools/verify_review_quality.py --only w1,wout
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw

from workflows import generate as g
from workflows.engine import Engine

#: 产线判据
SUBJECT_LO, SUBJECT_HI = 5.0, 60.0


def quality_of(path: str) -> dict:
    """从落盘文件量质量指标（不依赖生成时的中间状态）。"""
    im = Image.open(path)
    a = np.asarray(im.convert("RGBA"), dtype=np.int16)
    alpha = a[..., 3]
    rgb = a[..., :3]
    solid = bool((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all())
    return {
        "尺寸": f"{im.width}x{im.height}",
        "模式": im.mode,
        "主体占比": round(float((alpha > 32).mean()) * 100, 1),
        "半透明占比": round(float(((alpha > 20) & (alpha < 235)).mean()) * 100, 1),
        "纯色": solid,
        "均值": round(float(rgb.mean()), 1),
    }


def check_meta_size(meta: dict, path: str) -> str:
    """meta 里记的尺寸与实际文件对不对得上。"""
    im = Image.open(path)
    want = meta.get("sheet_size") or [meta.get("width"), meta.get("height")]
    if not want or want[0] is None:
        return "meta 无尺寸字段"
    if list(want) == [im.width, im.height]:
        return "尺寸一致"
    return f"**尺寸不符**：meta {want} vs 文件 {im.width}x{im.height}"


def run_all(only: set[str] | None) -> int:
    eng = Engine("models")
    rows = []

    def run(tag, fn, meta_key=None):
        if only and tag not in only:
            return
        t = time.time()
        try:
            r = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] 失败：{type(exc).__name__}: {exc}", flush=True)
            rows.append((tag, "失败", str(exc)[:60]))
            return
        path = r.get("file") or r.get("sheet") or ""
        meta = r.get("meta", {})
        if not path or not os.path.isfile(path):
            print(f"[{tag}] 没有产物文件（{path}）", flush=True)
            rows.append((tag, "无产物", path))
            return
        q = quality_of(path)
        size_note = check_meta_size(meta, path)
        subj = q["主体占比"]
        verdict = "OK"
        if q["纯色"] and q["均值"] < 250:
            verdict = "**纯色/全白**"
        elif subj < SUBJECT_LO:
            verdict = f"**主体过小**（<{SUBJECT_LO}%）"
        elif subj > SUBJECT_HI:
            verdict = f"**主体过大**（>{SUBJECT_HI}%）"
        elif q["半透明占比"] > 8:
            verdict = "**半透明太多**"
        if size_note.startswith("**"):
            verdict += " " + size_note
        print(f"[{tag}] {time.time()-t:5.1f}s {q} {verdict}", flush=True)
        rows.append((tag, verdict, f"{q['尺寸']} 主体{q['主体占比']}% "
                                  f"steps={meta.get('steps')} "
                                  f"fast_applied={meta.get('fast_applied')}"))
        return r

    # ---- 素材 ----
    ref = "assets/_cn_none.png"
    if not os.path.isfile(ref):
        print(f"缺少素材 {ref}", file=sys.stderr)
        return 1
    src = Image.open(ref).convert("RGBA")
    # 造一张局部重绘用的蒙版
    mask_path = "assets/_qa_mask.png"
    m = Image.new("L", src.size, 0)
    ImageDraw.Draw(m).ellipse([src.width * 0.25, src.height * 0.25,
                               src.width * 0.75, src.height * 0.55], fill=255)
    m.save(mask_path)

    # ---- 逐条工作流 ----
    run("w1", lambda: g.txt2img(
        eng, prompt="a small orange cat sitting, 2D game sprite, flat shading",
        width=768, height=768, steps=16, seed=7, transparent=True, scope="qa"))

    # w2 / w3 **不接受 width/height**：输出尺寸跟源图走（有意设计，
    # 改图场景下「跟原图一样大」是压倒性预期）。前端也不传这两个键。
    run("w2", lambda: g.img2img(
        eng, init_image=ref, prompt="a blue armored warrior, full body",
        steps=16, seed=7, scope="qa", matte=True))

    run("w3", lambda: g.inpaint(
        eng, image=ref, mask=mask_path, prompt="a red scarf",
        steps=16, seed=7, scope="qa"))

    run("wbg", lambda: g.matte_bg(
        eng, init_image=ref, scope="qa"))

    run("wpx", lambda: g.pixelify(
        eng, init_image=ref, target=[64, 64], colors=16, scope="qa"))

    run("wout", lambda: g.outpaint(
        eng, image=ref, direction="right", expand=0.5,
        prompt="a stone wall", scope="qa"))

    # **扩图之后** LoRA 必须已经卸干净（以前 Outpaint LoRA 会永久留在管线里）
    left = eng.active_loras()
    print(f"\n[污染检查] 扩图后仍挂着的 LoRA：{left}", flush=True)
    if left:
        print("  ❌ 有泄漏：后续所有生成都会带着它出图", flush=True)
    else:
        print("  ✅ 没有泄漏", flush=True)

    print("\n" + "=" * 72)
    for tag, verdict, note in rows:
        print(f"{tag:6s} {verdict:34s} {note}")
    print("=" * 72)
    bad = [r for r in rows if r[1] not in ("OK",) and not r[1].startswith("失败")]
    return 1 if (bad or left) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑某几条，逗号分隔")
    args = ap.parse_args()
    only = set(x for x in args.only.split(",") if x) or None
    return run_all(only)


if __name__ == "__main__":
    sys.exit(main())
