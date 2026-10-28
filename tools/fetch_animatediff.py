"""下载 AnimateDiff 的 motion adapter（时序一致性方案用）。

    python tools/fetch_animatediff.py

体积约 1.7GB（fp16）。走 ModelScope——HF 在国内经常连不上，
项目里其它权重的下载脚本也是这个路子（见 fetch_modelscope.py）。

拿不到就跳过：wss 节点会自动退回逐帧方案（engine_mode='frame'），
功能不受影响，只是帧间一致性略差。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "animatediff" / "motion-adapter-v1-5-2"

REPO = "Shanghai_AI_Laboratory/animatediff-motion-adapter-v1-5-2"
FILES = [
    "config.json",
    "configuration.json",
    "diffusion_pytorch_model.fp16.safetensors",
]
BASE = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        out = DEST / f
        if out.exists() and out.stat().st_size > 0:
            print(f"已存在，跳过: {f}  ({out.stat().st_size / 2**20:.1f}MB)")
            continue
        url = BASE.format(repo=REPO, path=f)
        t0 = time.perf_counter()
        print(f"下载 {f} ...", flush=True)
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            n = 0
            with open(out, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    n += len(chunk)
        print(f"  完成 {n / 2**20:.1f}MB，用时 {time.perf_counter() - t0:.0f}s")
    print(f"\n权重目录: {DEST}")
    for p in sorted(DEST.iterdir()):
        print(f"  {p.name}  {p.stat().st_size / 2**20:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
