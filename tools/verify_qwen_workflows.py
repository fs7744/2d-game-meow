"""验证 Qwen-only 的四个新工作流（w3 / w5 / w4 / wss）能跑通且产出可判读。

要的是**数字**，不是「跑完了」：

  E) w3 局部重绘：选区外最大像素差（应恒为 0）+ 选区内平均差（确实改了）
     + 耗时/峰值
  F) w5 局部 AI 重绘：同上，另外记 native/web 的策略与色板吸附比例
  G) w4 姿态序列：3 帧骨架 → DWPose 关节数 / 贴边 / 前景占比 / 可交付率
  H) wss 序列帧替换：3 帧小精灵图 → 帧间质心漂移 / 共享色板色数 / 逐帧耗时

素材全部现场生成或复用 `assets/_qwen_verify/`（避免提交二进制）。
用法：
    .venv\\Scripts\\python.exe tools\\verify_qwen_workflows.py
    .venv\\Scripts\\python.exe tools\\verify_qwen_workflows.py --steps 8   # 冒烟
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image, ImageDraw  # noqa: E402

from tools.make_pose import render, walk_pose  # noqa: E402
from workflows import generate as gen  # noqa: E402
from workflows import quality as q  # noqa: E402
from workflows.engine import Engine  # noqa: E402

OUT = Path("assets/_qwen_verify")
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "qwen-image-2.1"


def log(*a):
    print(*a, flush=True)


def char_image(eng: Engine, steps: int) -> Path:
    p = OUT / "char.png"
    if p.is_file():
        log(f"[0] 复用角色立绘 {p}")
        return p
    r = gen.txt2img(eng, prompt="a pixel art white cat knight with red cape and "
                                "golden armor, standing, full body, plain white background",
                    width=768, height=768, steps=steps, preset="none",
                    pixel=0, palette=0, model=MODEL, scope="qwen_verify")
    log(f"[0] 立绘 {r['meta']['seconds']}s → {r['file']}")
    Image.open(r["file"]).convert("RGB").save(p)
    return p


def mask_halfway(src: Path) -> Path:
    """选中上半身一块（盔甲位置）——w3/w5 共用同一块蒙版。"""
    im = Image.open(src)
    w, h = im.size
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rectangle(
        [int(w * 0.34), int(h * 0.22), int(w * 0.66), int(h * 0.46)], fill=255)
    p = OUT / "mask_armor.png"
    m.save(p)
    return p


def sheet_from_char(src: Path) -> tuple[Path, int, int]:
    """造一张 3×1 的小精灵图（64² 格子），当作「游戏里的原动画」。"""
    im = Image.open(src).convert("RGBA").resize((56, 56), Image.LANCZOS)
    sheet = Image.new("RGBA", (64 * 3, 64), (0, 0, 0, 0))
    for i in range(3):
        sheet.paste(im, (i * 64 + 4, 4 + i * 2), im)
    p = OUT / "anim_sheet.png"
    sheet.save(p)
    return p, 3, 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--skip-frames", action="store_true")
    ap.add_argument("--only", default="",
                    help="只跑某几段，逗号分隔：e,f,g,h（默认全跑）")
    args = ap.parse_args()
    only = {x.strip().lower() for x in args.only.split(",") if x.strip()}
    def want(tag: str) -> bool:
        return not only or tag in only

    t0 = time.perf_counter()
    eng = Engine("models")
    char = char_image(eng, args.steps)
    msk = mask_halfway(char)
    log(f"引擎就绪 {time.perf_counter() - t0:.0f}s")

    # ---- E: w3 局部重绘 ----
    if want("e"):
        e = gen.inpaint(eng, image=char, mask=msk,
                        prompt="bright emerald green armor with gold trim",
                        steps=args.steps, model=MODEL, scope="qwen_verify")
        m = e["meta"]
        log(f"[E] w3 {m['seconds']}s / {m['peak_vram_gb']}GB | "
            f"选区外最大差 {m['outside_max_diff']} | work={m['work_size']}")

    # ---- F: w5 局部 AI 重绘 ----
    if want("f"):
        f = gen.ai_repaint(eng, image=char, mask=msk,
                           prompt="bright emerald green armor with gold trim",
                           steps=args.steps, model=MODEL, palette_snap=True,
                           scope="qwen_verify")
        fm = f["meta"]
        log(f"[F] w5 {fm['seconds']}s / {fm['peak_vram_gb']}GB | "
            f"选区外最大差 {fm['outside_max_diff']} | 色板吸附 {fm['palette_snapped']}")

    if args.skip_frames:
        return 0

    # ---- G: w4 姿态序列 ----
    if want("g"):
        pdir = OUT / "skel"
        pdir.mkdir(exist_ok=True)
        for i in range(args.frames):
            render(walk_pose(i / args.frames), size=(768, 768)).save(
                pdir / f"frame_{i:02d}.png")
        g = gen.poseswap(eng, pose_images=pdir,
                         prompt="a pixel art white cat knight, red cape, "
                                "golden armor, full body", ref_image=char,
                         steps=args.steps, model=MODEL, width=768, height=768,
                         matte=True, candidates=1, gate=True, gate_retries=1,
                         scope="qwen_verify")
        gm = g["meta"]
        frames = [Image.open(p) for p in sorted(Path(g["frames_dir"]).glob("*.png"))]
        ok = 0
        for i, im in enumerate(frames):
            sc = q.evaluate(im.convert("RGBA"), eng.pose_extractor())
            good = sc["passed"] and 0.05 <= sc["coverage"] <= 0.60
            ok += int(good)
            log(f"[G] 帧{i}: points={sc['points']} coverage={sc['coverage']} "
                f"edge={sc['edge']} passed={sc['passed']} 可交付={good} "
                f"blobs={gm['frame_scores'][i].get('blobs')} "
                f"attempts={gm['frame_scores'][i].get('attempts')}")
        log(f"[G] w4 {gm['seconds_total']}s 总 / {gm['per_frame_seconds']} 每帧 | "
            f"可交付 {ok}/{len(frames)}")

    # ---- H: wss 序列帧替换 ----
    if want("h"):
        sheet, cols, rows = sheet_from_char(char)
        h = gen.sprite_swap(eng, sheet_image=sheet, char_image=char, cols=cols,
                            rows=rows, steps=args.steps, model=MODEL,
                            palette=32, keep_alpha=True, scope="qwen_verify")
        hm = h["meta"]
        log(f"[H] wss {hm['seconds_total']}s / {hm['per_frame_seconds']} 每帧 | "
            f"cell={hm['cell']} 色板={hm['palette_size']} | {h['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
