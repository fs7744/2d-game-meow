"""角色参考图 + 骨架 ControlNet 同时输入。

对应我们真正要用的场景（w4 姿态序列 / wss 序列帧替换）：
- **角色参考图**走 text encoder 的 vision 输入（`image=`），保住外观；
- **骨架图**走 ControlNet 的 129 通道，钉住姿态；
- prompt 里明确要求"严格参考角色图"，否则模型会自由发挥。

两者不冲突：参考图变成 joint 序列里的 vision token（会让序列变长），
骨架走 control 分支，`image_pad_mask` 只标 target 图像 token，数量仍然
等于 control token 数，所以散射不受影响。

用法:
    python tools/verify_cn_ref_pose.py --ref assets/_cn_none.png \
        --pose assets/_pose_walk_07.png --prompt "..."
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

# 描述必须与角色图对得上（用户要求"写的和图一模一样"），并在末尾明确
# 要求严格参考角色图 + 按姿态图摆动作。
DEFAULT_PROMPT = (
    "严格参考角色图的外观，画出同一个角色：一位黑发女性战士，头发向后梳成低马尾，"
    "面容清秀、眉眼细长、肤色白皙。她身穿深蓝色紧身皮甲，皮甲表面有明显的缝线纹路、"
    "关节处有浅蓝色高光块，肩部和手肘有加厚的护甲片。腰部系深色布质腰带，"
    "腰带下垂着数片布质护裙。双腿穿着同色紧身裤，膝盖有护膝，"
    "脚穿深灰色长靴，靴筒上有银灰色金属扣带。全身像，2D 游戏角色立绘，"
    "平涂赛璐璐上色，清晰黑色轮廓线，细节丰富。"
    "现在请让她做出姿态参考图中的动作，保持角色外观完全一致。"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(_find_asset("_cn_none.png")))
    ap.add_argument("--pose", default=str(_find_asset("_pose_walk_07.png")))
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    H = args.height // 32 * 32
    Wd = args.width // 32 * 32
    print(f"画布 {Wd}x{H}", flush=True)

    eng = Engine("models")
    pipe = eng.txt2img_pipe("qwen-image-2.1")
    tr = pipe.transformer
    branch = cnu.load_branch(W_PATH, dtype=torch.bfloat16, device="cuda")
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype

    ref = Image.open(args.ref).convert("RGB")
    pose = Image.open(args.pose).convert("RGB")
    print(f"角色图 {args.ref} {ref.size}", flush=True)
    print(f"骨架图 {args.pose} {pose.size}", flush=True)

    ctrl = cnu.encoder_condition(control=pose, vae=vae, size=(Wd, H),
                                 device="cuda", dtype=vae_dtype)
    print(f"  条件 latent {tuple(ctrl.shape)}", flush=True)

    def flatten(im):
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            return bg
        return im.convert("RGB")

    def run(label, use_ref, use_ctrl):
        inj = None
        if use_ctrl:
            inj = cnu.ControlInjection(tr, branch, scale=args.scale)
            inj.set_condition(ctrl)
        try:
            t = time.time()
            kw = {"use_kv_cache": False} if inj is not None else {}
            if use_ref:
                kw["image"] = ref.resize((Wd, H), Image.LANCZOS)
            out = pipe(prompt=args.prompt, width=Wd, height=H,
                       num_inference_steps=args.steps, true_cfg_scale=1.0,
                       generator=torch.Generator("cuda").manual_seed(43),
                       output_resolution=max(Wd, H), **kw).images[0]
            dt = time.time() - t
            p = str(_out_path(f"_cnr_{label}.png"))
            flatten(out).save(p)
            print(f"  {label:14s} {dt:5.1f}s → {p}", flush=True)
            if inj is not None and inj._image_pad_mask is not None:
                mk = inj._image_pad_mask
                print(f"    图像位置 {int(mk.sum())}/{len(mk)}", flush=True)
            return out
        finally:
            if inj is not None:
                inj.remove()

    # 四组对照：纯参考 / 纯骨架 / 参考+骨架（本命） / 都没有
    print("\n[A] 只给角色图（保外观，姿态自由）", flush=True)
    a = run("ref", True, False)
    print("[B] 只给骨架（钉姿态，外观自由）", flush=True)
    b = run("pose", False, True)
    print("[C] 角色图 + 骨架（本命）", flush=True)
    c = run("ref_pose", True, True)

    tiles = [flatten(a), ref.resize((Wd, H)), flatten(b), flatten(c)]
    side = Image.new("RGB", (Wd * len(tiles), H), "white")
    for i, t in enumerate(tiles):
        side.paste(t, (i * Wd, 0))
    side.save(str(_out_path("_cnr_compare.png")))
    print("\n  → " + str(_out_path("_cnr_compare.png")) + "（左→右：只参考 / 角色图 / 只骨架 / 参考+骨架）",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
