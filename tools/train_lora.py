"""LoRA 训练 CLI。

数据集准备（最容易被忽略、也最决定成败的一步）：

    dataset/
      knight_01.png
      knight_01.txt      <- 可选；没有 .txt 时用 --caption 统一兜底
      knight_02.png
      ...

  - 角色 LoRA：15~30 张同一角色、不同姿势/角度，每张写一句 caption。
    caption 里放一个独特触发词（如 `mk_knight`），别的词只描述这张图的差异。
  - 风格 LoRA：20~40 张同风格图，所有图共用一句 caption。

用法：
    python tools/train_lora.py --data dataset/knight --name mk_knight \
        --caption "mk_knight, a knight in armor, pixel art" --steps 400
    python tools/train_lora.py --data dataset/style --name my_style --caption "my_style, 16-bit" --epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workflows import train as T          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 SD1.5 LoRA")
    ap.add_argument("--data", required=True, help="数据集目录（图片 + 可选同名 .txt）")
    ap.add_argument("--name", required=True, help="输出名（写到 models/lora/<name>.safetensors）")
    ap.add_argument("--caption", default="", help="没有 .txt 时的统一 caption")
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank（角色 8~16，风格 4~8）")
    ap.add_argument("--alpha", type=int, default=0, help="0=等于 rank")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=0, help="0=按样本数与 epochs 自动估算")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=0, help="每 N 步存中间权重（0=不存）")
    ap.add_argument("--center-crop", action="store_true", help="居中裁成正方形（默认等比缩放）")
    a = ap.parse_args()

    try:
        ds = T.Dataset(ROOT / a.data, resolution=a.resolution,
                       caption=a.caption, center_crop=a.center_crop)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[x] {exc}", file=sys.stderr)
        return 2

    steps = a.steps or T.plan_steps(len(ds), epochs=a.epochs, batch_size=a.batch)
    print(f"[i] 样本 {len(ds)} 张，rank {a.rank}，{steps} 步，分辨率 {a.resolution}")
    for p, c in ds.items[:3]:
        print(f"    {p.name}  ->  {c[:70]}")

    last = {"pct": -1}

    def show(ev: dict) -> None:
        if ev.get("stage") != "train" or "step" not in ev:
            return
        pct = int(ev["step"] / ev["total"] * 100)
        if pct == last["pct"]:
            return
        last["pct"] = pct
        print(f"    {ev['step']:>5}/{ev['total']}  loss={ev.get('loss')}  "
              f"{ev.get('elapsed')}s  {ev.get('peak_vram_gb')}GB", flush=True)

    t0 = time.perf_counter()
    info = T.train(ROOT / "models", ROOT / a.data, a.name,
                   caption=a.caption, rank=a.rank, alpha=a.alpha or None,
                   lr=a.lr, steps=steps, batch_size=a.batch,
                   resolution=a.resolution, seed=a.seed,
                   save_every=a.save_every, center_crop=a.center_crop,
                   progress=show)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"[✓] 总耗时 {time.perf_counter() - t0:.1f}s -> {info['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
