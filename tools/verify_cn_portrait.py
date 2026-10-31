"""完全复刻 VideoX-Fun 官方示例配置跑一次 ControlNet（竖版构图）。

对应 `examples/qwenimage21_fun/predict_t2i_control.py`：

| 参数 | 官方值 | 这里 |
|---|---|---|
| sample_size | `[1728, 992]`（H×W，竖版 9:16） | `[1024, 768]`（同等比例，32 对齐） |
| num_inference_steps | 40 | 同 |
| true_cfg_scale | 1.0 | 同 |
| control_context_scale | 1.0 | 同 |
| negative_prompt | `" "` | 同 |
| prompt | 几百字中文详述 | 原样照抄 |
| use_kv_cache | True（但有 control 时失效） | False（显式关） |

之前一直用 768² 方形 + 一行短 prompt，与官方配置差得太远——方形会让人物
撑满画布把头挤出去，短 prompt 会让模型照抄骨架。这个脚本用来验证
"是不是配置差异造成的"。

用法: python tools/verify_cn_portrait.py [--scale 1.0] [--steps 40]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")
from pathlib import Path

# ---- 产物目录：验证脚本的输出**不进 assets/**（那里是项目资产） ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workflows import paths                                  # noqa: E402


def _out_path(name: str) -> Path:
    """验证产物的落盘位置（tests/_out/ 下）。"""
    return paths.tool_out(name)


def _find_asset(name: str) -> Path:
    """找输入素材：先看测试输出目录，再看仓库 assets/（历史素材兼容）。"""
    for base in (paths.TEST_OUT / "legacy_assets", paths.ASSETS, paths.ROOT / "assets"):
        p = base / name
        if p.is_file():
            return p
    return paths.TEST_OUT / "legacy_assets" / name

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image

from workflows.engine import Engine
from workflows import controlnet_union as cnu

W_PATH = "models/controlnet/qwen21_union.safetensors"

# 官方示例的 prompt，逐字照抄（换成竖版人物也成立，正好验证控制）
PROMPT = (
    "画面中央是一位年轻女孩，她拥有一头令人印象深刻的亮紫色长发，发丝在海风中轻盈飘扬，"
    "营造出动感而唯美的效果。她的长发两侧各扎着黑色蝴蝶结发饰，增添了几分可爱与俏皮感。"
    "女孩身穿一袭纯白色无袖连衣裙，裙摆轻盈飘逸，与她清新的气质完美契合。"
    "她的妆容精致自然，淡粉色的唇妆和温柔的眼神流露出恬静优雅的气质。"
    "她单手叉腰，姿态自信从容，目光直视镜头，展现出既甜美又不失个性的魅力。"
    "背景是一片开阔的海景，湛蓝的海水在阳光照射下波光粼粼，闪烁着钻石般的光芒。"
    "天空呈现出清澈的蔚蓝色，点缀着几朵洁白的云朵，营造出晴朗明媚的夏日氛围。"
    "画面前景右下角可见粉紫色的小花丛和绿色植物，为整体构图增添了自然生机和色彩层次。"
    "整张照片色调明亮清新，紫色头发与白色裙装、蓝色海天形成鲜明而和谐的色彩对比。"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default=str(_find_asset("_pose_walk_02.png")))
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--transparent", action="store_true")
    args = ap.parse_args()

    H = args.height // 32 * 32
    Wd = args.width // 32 * 32
    print(f"画布 {Wd}x{H}（竖版，与官方 992x1728 同比例）", flush=True)

    eng = Engine("models")
    pipe = eng.txt2img_pipe("qwen-image-2.1")
    tr = pipe.transformer
    branch = cnu.load_branch(W_PATH, dtype=torch.bfloat16, device="cuda")
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype

    cond = Image.open(args.pose).convert("RGB")
    print(f"条件图 {args.pose} {cond.size} → {Wd}x{H}", flush=True)
    ctrl = cnu.encoder_condition(control=cond, vae=vae, size=(Wd, H),
                                 device="cuda", dtype=vae_dtype)
    print(f"  条件 latent {tuple(ctrl.shape)}", flush=True)

    from workflows.generate import _qwen_prompt, _strip_bg_words
    clean, _ = _strip_bg_words(args.prompt)
    full = _qwen_prompt(clean, args.transparent)

    def flatten(im):
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            return bg
        return im.convert("RGB")

    def run(label, use_control):
        inj = None
        if use_control:
            inj = cnu.ControlInjection(tr, branch, scale=args.scale)
            inj.set_condition(ctrl)
        try:
            t = time.time()
            kw = {"use_kv_cache": False} if inj is not None else {}
            out = pipe(prompt=full, width=Wd, height=H,
                       num_inference_steps=args.steps, true_cfg_scale=1.0,
                       negative_prompt=" ",
                       generator=torch.Generator("cuda").manual_seed(43),
                       output_resolution=max(Wd, H), **kw).images[0]
            dt = time.time() - t
            p = str(_out_path(f"_cnp_{label}.png"))
            img = out if args.transparent else out.convert("RGB")
            img.save(p)
            print(f"  {label}: {dt:.1f}s  → {p}", flush=True)
            if inj is not None and inj._image_pad_mask is not None:
                mk = inj._image_pad_mask
                print(f"    图像位置 {int(mk.sum())}/{len(mk)}", flush=True)
            return out
        finally:
            if inj is not None:
                inj.remove()

    print("\n[1] 无控制", flush=True)
    base = run("none", False)
    print("\n[2] 带控制 scale=" + str(args.scale), flush=True)
    ctl = run("pose", True)

    side = Image.new("RGB", (Wd * 3, H), "white")
    side.paste(flatten(base), (0, 0))
    side.paste(cond.resize((Wd, H)), (Wd, 0))
    side.paste(flatten(ctl), (Wd * 2, 0))
    side.save(str(_out_path("_cnp_compare.png")))
    print("  → " + str(_out_path("_cnp_compare.png")) + "（左=无控制 中=条件图 右=带控制）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
