"""审计所有 LoRA：挂上去到底生不生效。

**为什么不能只看 header**：架构不匹配有两种表现，header 一个都判断不出来。

  1. 报错（`NoMatchingPeftModuleError`）—— 好发现
  2. **静默无效**：`load_lora_weights` 不报错，但 unet 里 LoRA 增量为 0，
     出图和「不挂 LoRA」完全一样。wa n_t2v 那个视频 LoRA 就是这样。

所以判据只能是「挂上之后的产物和不挂差多少」。diff ≈ 0 就是没用。

做法：同一底模 / 同一 seed / 同一 prompt / 4 步 512²（够快，够看出效果），
每个 LoRA 挂到它自己标注的 base 上，量平均像素差。

用法：
    python tools/audit_loras.py            # 全部
    python tools/audit_loras.py --only pixelizer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from workflows import generate as g
from workflows import lora as lora_reg
from workflows.engine import Engine

OUT = ROOT / "assets" / "_lora_audit"
PROMPT = "a knight in silver armor, red cape, holding sword, full body"
# diff 阈值：4 步 512² 下，真生效的 LoRA 通常 5~40；静默无效是 0.0~0.2
EFFECTIVE = 0.8


def diff(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.int16)
    y = np.asarray(b.convert("RGB"), dtype=np.int16)
    return float(np.abs(x - y).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只测名字里含该子串的")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    eng = Engine(ROOT / "models")
    meta = json.loads((ROOT / "models" / "lora" / "loras.json").read_text("utf-8"))

    # 每个底模一张基线（不挂 LoRA）
    base: dict[str, Image.Image] = {}
    for m in ("sd15", "sdxl"):
        if m not in eng.available_models():
            continue
        r = g.txt2img(eng, prompt=PROMPT, negative=None, preset="none",
                      width=512, height=512, steps=4, cfg=4.0, seed=777,
                      model=m, matte=False, tag=f"audit_base_{m}")
        base[m] = Image.open(r["file"])

    rows = []
    for name, cfg in meta.items():
        if a.only and a.only not in name:
            continue
        path = lora_reg.resolve(ROOT / "models", name)
        if not path:
            rows.append(dict(name=name, base="-", diff=None,
                             verdict="缺文件", err=""))
            continue
        model = cfg.get("base") or "sd15"
        if model not in base:
            # 标了 flux/wan 之类本机没有的底模：直接判不可用，不跑
            rows.append(dict(name=name, base=model, diff=None,
                             verdict=f"本机无 {model} 底模", err=""))
            continue
        try:
            r = g.txt2img(eng, prompt=PROMPT, negative=None, preset="none",
                          width=512, height=512, steps=4, cfg=4.0, seed=777,
                          model=model, matte=False, lora=path,
                          lora_scale=a.scale, tag=f"audit_{name[:24]}")
            d = diff(base[model], Image.open(r["file"]))
            rows.append(dict(name=name, base=model, diff=d,
                             verdict="生效" if d >= EFFECTIVE else "静默无效",
                             err=""))
        except Exception as e:                      # noqa: BLE001
            rows.append(dict(name=name, base=model, diff=None,
                             verdict="报错", err=f"{type(e).__name__}: {e}"[:120]))
        print(f"  {name[:44]:44s} {rows[-1]['base']:5s} "
              f"{'' if rows[-1]['diff'] is None else f'diff={rows[-1]['diff']:.1f}'}  "
              f"{rows[-1]['verdict']}", flush=True)

    print()
    bad = [r for r in rows if r["verdict"] != "生效"]
    print(f"共 {len(rows)} 个，生效 {len(rows) - len(bad)}，有问题 {len(bad)}")
    if bad:
        print("\n| LoRA | base | diff | 判定 |")
        print("|---|---|---|---|")
        for r in bad:
            d = "-" if r["diff"] is None else f"{r['diff']:.1f}"
            print(f"| {r['name'][:48]} | {r['base']} | {d} | {r['verdict']} |")
    (OUT / "report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
