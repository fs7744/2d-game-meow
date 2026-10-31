"""VNCCS 姿态迁移的**透明底专项验证**。

## 为什么单独验

`--transparent` 会往 prompt 里插一段 "keep the background fully transparent"
（`generate.QWEN_ALPHA_HEAD/TAIL`），而 VNCCS 的指令是"按骨架重画角色"。
两者目的不同，之前只在 ControlNet 那条路上撞过枪：控制信号要求"整张结构
实心"，透明指令要求"背景全空"，结果整张图被压成半透明。

所以这里不看像素差（**那个指标在"空白 vs 白底"上会骗人**，填错 mask 时它
反而更大），只看出四个数：

  1. `不透明占比` —— alpha > 128 的像素比例。0% 就是"整张空了"。
  2. `半透明占比` —— 20 < alpha < 235。游戏引擎里这层会被当半透贴图，
     边缘之外出现大片半透明就是"糊了"。
  3. `主体占比` —— 同上口径，产线判据要求 5%~60%。
  4. `边缘像素数` —— alpha 在 1..255 之间的像素，反映抗锯齿是否正常。

## 用法

    python tools/verify_vnccs_alpha.py \
        --ref assets/_cn_none.png --pose assets/_pose_from_pixel.png

输出：`assets/_vnccs_alpha.png`（左=不透明 / 中=透明底铺棋盘 / 右=alpha 通道）
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

import numpy as np
from PIL import Image

from workflows import generate as gen
from workflows.engine import Engine

#: 与 vnccs_pose 默认一致的身份描述（换角色时用 --extra 覆盖）
DEFAULT_EXTRA = (
    "a female warrior with long black hair in a low ponytail, dark blue leather "
    "armor with light blue highlight patches, matching blue pants with knee pads, "
    "dark grey tall boots with silver buckles, 2D game character sprite, "
    "flat cel shading, clean black outlines, full body, detailed"
)


def alpha_stats(im: Image.Image) -> dict:
    """透明底是否"真的可用"的四个数。"""
    a = np.asarray(im.convert("RGBA").getchannel("A"), dtype=np.uint8)
    tot = a.size
    return {
        "不透明占比": round(float((a > 128).sum()) / tot * 100, 2),
        "半透明占比": round(float(((a > 20) & (a < 235)).sum()) / tot * 100, 2),
        "主体占比": round(float((a > 32).sum()) / tot * 100, 2),
        "全透明占比": round(float((a <= 20).sum()) / tot * 100, 2),
        "alpha均值": round(float(a.mean()), 1),
    }


def checker(im: Image.Image) -> Image.Image:
    """把 RGBA 铺到棋盘上，肉眼才看得出"哪些其实是透明的"。"""
    w, h = im.size
    cell = max(8, w // 32)
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    px = bg.load()
    for y in range(h):
        for x in range(w):
            if ((x // cell) + (y // cell)) % 2:
                px[x, y] = (204, 204, 204)
    bg.paste(im.convert("RGB"), (0, 0), im.convert("RGBA").getchannel("A"))
    return bg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(_find_asset("_cn_none.png")))
    ap.add_argument("--pose", default=str(_find_asset("_pose_from_pixel.png")))
    ap.add_argument("--extra", default=DEFAULT_EXTRA)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--lora", type=float, default=1.0)
    ap.add_argument("--out", default=str(_out_path("_vnccs_alpha.png")))
    args = ap.parse_args()

    eng = Engine("models")
    results = {}
    for label, transparent in (("opaque", False), ("transparent", True)):
        t = time.time()
        r = gen.vnccs_pose(
            eng, pose_image=args.pose, char_image=args.ref,
            prompt=args.extra, steps=args.steps, seed=args.seed,
            lora_scale=args.lora, transparent=transparent)
        path = r["file"]
        im = Image.open(path)
        dt = time.time() - t
        st = alpha_stats(im)
        results[label] = (im.convert("RGBA"), st, dt, path)
        print(f"[{label:11s}] {dt:5.1f}s  mode={im.mode}  "
              + "  ".join(f"{k}={v}" for k, v in st.items()), flush=True)

    opq, tr = results["opaque"][0], results["transparent"][0]
    w, h = opq.size
    side = Image.new("RGB", (w * 3, h), "white")
    side.paste(opq.convert("RGB"), (0, 0))
    side.paste(checker(tr), (w, 0))
    # alpha 通道本身也画出来：哪里是 0 一眼能看到
    alpha_vis = Image.merge("RGB", [tr.getchannel("A")] * 3)
    side.paste(alpha_vis, (w * 2, 0))
    side.save(args.out)
    print(f"→ {args.out}（左=不透明 / 中=透明底铺棋盘 / 右=alpha 通道）",
          flush=True)

    st = results["transparent"][1]
    ok = 5.0 <= st["主体占比"] <= 60.0 and st["不透明占比"] > 3.0
    print(f"\n结论: {'✅ 透明底可用' if ok else '❌ 透明底不可用'}"
          f"（主体 {st['主体占比']}%，判据 5~60%；"
          f"不透明 {st['不透明占比']}%）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
