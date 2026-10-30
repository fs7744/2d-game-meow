"""实验：把「首帧」当第三张参考图，能不能提高逐帧一致性？

背景：Qwen 逐帧没有 AnimateDiff 的时序注意力，帧与帧之间零耦合。
现在唯一的约束是「同一张角色图 + 各自的骨架图」。假设是：把**首帧成品**
也塞进 vision context（三张图：角色 / 骨架 / 上一帧），模型会继承上一帧的
光影与配色，帧间会稳一些。

对照（同 seed、同骨架、同参数）：
  A 现状：char + skeleton
  B 锚定：char + skeleton + 上一帧（首帧自己跑一次当锚）

指标（都来自 framescore，和产线 meta 同一套）：
  identity_drift    相邻帧身份相似度下降幅度（越小越稳）
  silhouette_drift  相邻帧剪影 IoU（越小动作越连贯）
  joints/coverage   质量闸门口径
单帧成本也要记：B 多花一次锚帧生成。

用法：.venv\\Scripts\\python.exe tools\\exp_frame_anchor.py --frames 3 --steps 24
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image  # noqa: E402

from tools.make_pose import render, walk_pose  # noqa: E402
from workflows import framescore, quality  # noqa: E402
from workflows import generate as g  # noqa: E402
from workflows import refprep  # noqa: E402
from workflows.engine import Engine  # noqa: E402

OUT = Path("assets/_anchor_exp")
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "qwen-image-2.1"


def log(*a):
    print(*a, flush=True)


def run(eng: Engine, mode: str, frames: int, steps: int, seed: int, bg: str):
    """跑一段序列，返回 (帧列表, 指标, 每帧耗时)。"""
    char = refprep.prepare_reference(
        Image.open("assets/_qwen_verify/char.png").convert("RGBA"), size=768)
    instr = ("The first image is the character reference: keep the same face, "
             "hairstyle, outfit and colors. The second image is a body skeleton "
             "showing the target pose. Redraw the same character in exactly that "
             "pose, full body, centered, "
             + g._qwen_backdrop(bg) + "." + g.QWEN_FRAME_RULES)
    outs, times = [], []
    anchor = None
    for i in range(frames):
        cond = g._qwen_cond_image(
            render(walk_pose(i / frames), size=(768, 768)))
        refs = [char, cond]
        if mode == "anchor" and anchor is not None:
            # 三张图：身份 / 姿态 / 上一帧（光影与配色参照）
            refs = [char, cond, anchor]
        t = time.time()
        im = g._qwen_call(eng, prompt=instr, negative=g.QWEN_FRAME_NEG,
                          ref_image=refs, width=768, height=768, steps=steps,
                          seed=seed, model=MODEL, label=f"{mode}{i}")
        times.append(round(time.time() - t, 1))
        im.save(OUT / f"{mode}_{i}.png")
        outs.append(im)
        anchor = im
    return outs, times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bg", default="dark")
    args = ap.parse_args()

    t0 = time.time()
    eng = Engine("models")
    ex = eng.pose_extractor()
    log(f"引擎就绪 {time.time() - t0:.0f}s")

    for mode in ("plain", "anchor"):
        t = time.time()
        frames, times = run(eng, mode, args.frames, args.steps, args.seed, args.bg)
        cons = framescore.sequence_consistency(frames)
        rows = []
        for i, im in enumerate(frames):
            info = {}
            cut, edge, subj = g._postprocess(
                im, eng, matte=True, matte_backend="auto", denoise=1,
                pixel=0, palette=0, clean_iso=0, info=info)
            sc = quality.evaluate(cut, ex)
            rows.append((i, sc["points"], sc["coverage"], len(sc["edge"]),
                         info.get("backend")))
        log(f"[{mode}] 总 {time.time() - t:.0f}s / 每帧 {times}")
        log(f"[{mode}] identity_drift={cons['identity_drift']} "
            f"silhouette_drift={cons['silhouette_drift']} worst={cons['worst_pair']}")
        for r in rows:
            log(f"[{mode}] 帧{r[0]}: 关节={r[1]} 占比={r[2]} 贴边={r[3]} matte={r[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
