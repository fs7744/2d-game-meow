"""两阶段姿态实验：骨架草图 → 换皮。

**为什么做这个**：B1 测出 w4 姿态贴合的根因是「参考图会抢姿态」——
把角色图和骨架图一起丢进 Qwen 的 vision context 时，模型更相信角色图里的
站姿，于是骨架里的迈步没体现（成品关节 vs 骨架关节平均偏差 108px/768²）。
把角色图整个去掉，偏差降到 67px（腿真的迈开了），但身份漂移——
画出来的不再是那个角色。

**两阶段的想法**：把「定姿态」和「定身份」拆开。

    阶段 1（草图）：只给骨架 + 一句角色描述 → 姿态准、身份是「一个差不多的角色」
    阶段 2（换皮）：草图 + 角色图一起做 img2img → 身份回来，而草图已经把姿态钉住

用「成品关节 vs 骨架关节的平均像素偏差」当主判据（越小姿态越贴），
同时用「与角色参考图的主色一致度」当副判据（越大身份越保得住）。

用法：
    python tools/exp_two_stage.py --frames 3 --steps 24
    python tools/exp_two_stage.py --only two            # 只跑两阶段
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from tools.make_pose import render, walk_pose
from workflows import paths
from workflows import generate as g
from workflows import refprep
from workflows.engine import Engine

OUT = paths.tool_out("_two_stage")


# ---------------------------------------------------------------- 度量

def pose_deviation(img: Image.Image, eng: Engine, kp: dict, size: int = 768) -> tuple[float, float]:
    """(成品踝距, 关节平均偏差px)。

    偏差按「成品检出的关节 vs 骨架关节」的坐标距离算——**只看 DWPose 检出的
    关节数量是不够的**（检到 18 个只能证明画出了像人形的角色）。
    """
    cut, _, _ = g._postprocess(img, eng, matte=True, matte_backend="auto",
                               denoise=1, pixel=0, palette=0, clean_iso=0)
    flat = Image.new("RGB", cut.size, (255, 255, 255))
    flat.paste(cut, (0, 0), cut)
    pts = eng.pose_extractor().visible_points(flat, thr=0.25)
    ankle = (abs(pts[10][0] - pts[13][0]) * size
             if (10 in pts and 13 in pts) else float("nan"))
    diffs = [np.hypot(pts[k][0] - kp[k][0], pts[k][1] - kp[k][1]) * size
             for k in range(14) if k in pts]
    return ankle, (float(np.mean(diffs)) if diffs else float("nan"))


def color_identity(img: Image.Image, ref: Image.Image) -> float:
    """与角色参考图的**配色分布**接近程度（0~1）。

    做法：把两张图的前景像素各自量化成 16 色，比较两张直方图的相似度。

    上一版用「12 色集合是否相同」当判据，结果两两都是 1.00——区分度为零，
    看不出「整身金甲 vs 白袍」这种明显差别。16 色 + 直方图才有区分度。
    """
    def pal(im: Image.Image) -> np.ndarray:
        rgba = im.convert("RGBA")
        a = np.asarray(rgba.getchannel("A"))
        rgb = np.asarray(rgba.convert("RGB")).reshape(-1, 3)
        fg = rgb[a.reshape(-1) > 127]
        if len(fg) == 0:
            return np.zeros(16, dtype=np.float32)
        small = Image.fromarray(fg.reshape(-1, 1, 3)).resize((64, 64), Image.LANCZOS)
        q = small.quantize(colors=16, method=Image.MEDIANCUT)
        hist = np.histogram(np.asarray(q).reshape(-1), bins=16,
                            range=(0, 16))[0].astype(np.float32)
        return hist / max(1.0, hist.sum())
    a, b = pal(img), pal(ref)
    return float(np.minimum(a, b).sum() / max(1e-6, np.maximum(a, b).sum()))


def white_ratio(img: Image.Image) -> float:
    """前景里「接近白」的像素占比（这个角色是白袍，用它当身份探针）。"""
    rgba = img.convert("RGBA")
    a = np.asarray(rgba.getchannel("A"))
    rgb = np.asarray(rgba.convert("RGB")).astype(np.int16)
    fg = a > 127
    if not fg.any():
        return 0.0
    white = (rgb.min(axis=2) > 170) & fg
    return float(white.sum() / fg.sum())


# ---------------------------------------------------------------- 三个变体

def stage1_draft(eng: Engine, cond: Image.Image, prompt: str, *, steps: int,
                 seed: int, size: int) -> Image.Image:
    """阶段 1：只给骨架（+ 角色描述），不喂角色图。"""
    instr = ("This image is a body skeleton showing a pose. Draw " + prompt
             + " in exactly that pose, full body, centered, "
             + g._qwen_backdrop("dark") + "." + g.QWEN_FRAME_RULES)
    return g._qwen_call(eng, prompt=instr, negative=g.QWEN_FRAME_NEG,
                        ref_image=cond, width=size, height=size,
                        steps=steps, seed=seed, model="qwen-image-2.1",
                        label="stage1")


def stage2_reskin(eng: Engine, draft: Image.Image, char: Image.Image,
                  prompt: str, *, steps: int, seed: int, size: int) -> Image.Image:
    """阶段 2：草图 + 角色图一起当条件，做一次「指令改图」。"""
    instr = ("The first image is a rough draft showing the target body pose. "
             "The second image is the character reference: keep the same face, "
             "hairstyle, outfit and colors. Redraw the character from the second "
             "image in exactly the pose of the first image, full body, centered, "
             + g._qwen_backdrop("dark") + "." + g.QWEN_FRAME_RULES)
    return g._qwen_call(eng, prompt=instr, negative=g.QWEN_FRAME_NEG,
                        ref_image=[draft, char], width=size, height=size,
                        steps=steps, seed=seed, model="qwen-image-2.1",
                        label="stage2")


def stage2_reskin_rev(eng: Engine, draft: Image.Image, char: Image.Image,
                      prompt: str, *, steps: int, seed: int, size: int) -> Image.Image:
    """阶段 2 的另一个顺序：**角色图在前、草图在后**。

    实验发现：Qwen 的多图条件里，**第一张图决定「画谁」**——
    角色图在前时身份保得住、姿态却被角色的站姿带走；草图在前时姿态准了，
    身份却漂了（整身变成金甲，白袍没了）。顺序本身就是一个旋钮，
    这里把顺序反过来再测一次。
    """
    instr = ("The first image is the character reference: keep exactly the same "
             "face, hairstyle, outfit, armor and colors. The second image is a "
             "rough draft showing the target body pose. Redraw the character "
             "from the first image in exactly the pose of the second image, "
             "full body, centered, " + g._qwen_backdrop("dark") + "."
             + g.QWEN_FRAME_RULES)
    return g._qwen_call(eng, prompt=instr, negative=g.QWEN_FRAME_NEG,
                        ref_image=[char, draft], width=size, height=size,
                        steps=steps, seed=seed, model="qwen-image-2.1",
                        label="stage2rev")


def baseline(eng: Engine, char: Image.Image, cond: Image.Image, prompt: str,
             *, steps: int, seed: int, size: int) -> Image.Image:
    """现状：角色图 + 骨架图一起给（w4 的做法）。"""
    instr = ("The first image is the character reference: keep the same face, "
             "hairstyle, outfit and colors. The second image is a body skeleton "
             "showing the target pose. Redraw the same character in exactly that "
             "pose, full body, centered, "
             + g._qwen_backdrop("dark") + "." + g.QWEN_FRAME_RULES)
    return g._qwen_call(eng, prompt=instr, negative=g.QWEN_FRAME_NEG,
                        ref_image=[char, cond], width=size, height=size,
                        steps=steps, seed=seed, model="qwen-image-2.1",
                        label="baseline")


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--only", default="", help="只跑某个变体：base / two")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    eng = Engine("models")
    # 角色素材：验证脚本的输出目录（tests/_out/_qwen_verify/）优先
    char_path = paths.tool_out("_qwen_verify") / "char.png"
    if not char_path.is_file():
        char_path = ROOT / "assets" / "_qwen_verify" / "char.png"
    if not char_path.is_file():
        print(f"缺少角色素材 {char_path}（先跑 tools/verify_qwen_workflows.py）")
        return 1
    char = refprep.prepare_reference(
        Image.open(char_path).convert("RGBA"), size=args.size)
    prompt = "a pixel art white cat knight with red cape and golden armor"

    variants = ["base", "two", "rev"] if not args.only else [args.only]
    rows = []
    for i in range(args.frames):
        kp = walk_pose(i / max(1, args.frames))
        cond = g._qwen_cond_image(render(kp, size=(args.size, args.size)))
        seed = 2026 + i

        for v in variants:
            t0 = time.perf_counter()
            if v == "base":
                im = baseline(eng, char, cond, prompt, steps=args.steps,
                              seed=seed, size=args.size)
            else:
                draft = stage1_draft(eng, cond, prompt, steps=args.steps,
                                     seed=seed, size=args.size)
                draft.save(OUT / f"draft_{i}.png")
                if v == "rev":
                    im = stage2_reskin_rev(eng, draft, char, prompt,
                                           steps=args.steps, seed=seed,
                                           size=args.size)
                else:
                    im = stage2_reskin(eng, draft, char, prompt, steps=args.steps,
                                       seed=seed, size=args.size)
            dt = time.perf_counter() - t0
            im.save(OUT / f"{v}_{i}.png")
            ankle, dev = pose_deviation(im, eng, kp, size=args.size)
            ident = color_identity(im, char)
            white = white_ratio(im)
            rows.append((v, i, ankle, dev, ident, dt, white))
            print(f"[{v}{i}] 踝距={ankle:.0f} 偏差={dev:.0f}px "
                  f"身份={ident:.2f} 白占比={white:.2f} {dt:.0f}s", flush=True)

    print("\n=== 汇总（偏差越小越好，身份越大越好）===")
    for v in variants:
        sel = [r for r in rows if r[0] == v]
        if not sel:
            continue
        print(f"{v:5s} 偏差均值 {np.mean([r[3] for r in sel]):.0f}px  "
              f"身份均值 {np.mean([r[4] for r in sel]):.2f}  "
              f"白占比 {np.mean([r[6] for r in sel]):.2f}  "
              f"每帧 {np.mean([r[5] for r in sel]):.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
