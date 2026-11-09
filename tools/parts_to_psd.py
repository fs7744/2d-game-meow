# -*- coding: utf-8 -*-
"""把一份层清单合成为**标准分层 PSD**（必须用 see-through 的 venv 跑）。

为什么单独一个脚本：本项目主环境只装了 fastapi/torch/Qwen 那一套，没有
psd_tools；而 see-through 的 venv 里有 `psd_tools 1.26`（支持 `PSDImage.new`
+ `create_pixel_layer`）。写 PSD 这件事没必要在主环境再装一份，跨进程调用
反而干净：`workflows/partsplit.py` 只依赖「一个能跑的 CLI + 约定的清单格式」。

清单格式（`manifest.partial.json`，由 partsplit 生成）：
    {"canvas": [w, h],
     "layers": [{"name": "头（head）", "file": "layers/head.png",
                 "x": 120, "y": 8}, ...]}

`layers` 必须按**从底到顶**排（先画的在下面）。产出是可被 Photoshop /
Affinity / Krita / GIMP 直接打开的分层 PSD（每层带 alpha、位置正确、层名保留）。

用法：
  <see-through>/.venv/Scripts/python.exe tools/parts_to_psd.py \
      --manifest <...>/manifest.partial.json --out <...>/model.psd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from psd_tools import PSDImage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mf_path = Path(a.manifest).resolve()
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    canvas = mf.get("canvas") or [0, 0]
    w, h = int(canvas[0]), int(canvas[1])
    layers = mf.get("layers") or []
    if w <= 0 or h <= 0 or not layers:
        print(f"[err] 清单不完整：canvas={canvas} layers={len(layers)}", flush=True)
        return 2

    base = mf_path.parent
    # psd-tools 的 size 是 (width, height)
    psd = PSDImage.new(mode="RGBA", size=(w, h), depth=8)
    added = 0
    for layer in layers:
        f = (base / str(layer.get("file") or "")).resolve()
        if not f.is_file():
            print(f"[warn] 缺层文件，跳过：{f}", flush=True)
            continue
        with Image.open(f) as im:
            img = im.convert("RGBA")
        # **层名必须是 ASCII**：psd-tools 写层名走 mac_roman，中文会抛
        # charmap 异常。中文显示名放在 manifest 的 name 里给前端预览用。
        name = str(layer.get("psdName") or layer.get("name") or f.stem)
        name = name.encode("ascii", "ignore").decode() or f.stem
        psd.create_pixel_layer(
            img, name=name,
            top=int(layer.get("y") or 0), left=int(layer.get("x") or 0),
            opacity=255)
        added += 1
    if added == 0:
        print("[err] 一层都没加进去", flush=True)
        return 3
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    psd.save(str(out))
    print(f"[ok] {out} layers={added} size={w}x{h}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
