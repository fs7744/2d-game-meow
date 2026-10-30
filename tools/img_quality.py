"""量化对比生成图的画质（不靠肉眼猜）。

三个指标：

- **清晰度**：灰度拉普拉斯方差，在 alpha>128 的**内部区域**（腐蚀 3px，
  避开边缘过渡带）。这是判断"糊没糊"最灵敏的指标，无控制 vs 带控制
  能差 8 倍。
- **alpha 干净度**：`alpha均值`（理想接近 255）与 `半透明比例`
  （20<alpha<235 的像素占比）。模型自己出的 alpha 很容易整片发虚，
  半透明比例一高，抠进游戏里就是一圈灰边。
- **头部区域清晰度**：取主体包围盒上部 22%，单独算。头丢没丢在这里最明显。

用法:
    python tools/img_quality.py a.png b.png ...
    python tools/img_quality.py assets/_cn_*.png
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image


def _lap_var(g: np.ndarray, m: np.ndarray) -> float:
    """在 mask 内部算拉普拉斯方差（越锐越高）。"""
    if m.sum() < 50:
        return 0.0
    # 4 邻域拉普拉斯，避免依赖 scipy
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
           - g[1:-1, :-2] - g[1:-1, 2:])
    mm = m[1:-1, 1:-1]
    if mm.sum() < 50:
        return 0.0
    return float(lap[mm].var())


def _erode(m: np.ndarray, it: int = 3) -> np.ndarray:
    out = m.copy()
    for _ in range(it):
        e = out.copy()
        e[1:, :] &= out[:-1, :]
        e[:-1, :] &= out[1:, :]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        out = e
    return out


def stats(path: str) -> dict:
    im = Image.open(path)
    a = np.asarray(im.convert("RGBA"), dtype=np.float32)
    al = a[..., 3]
    g = np.asarray(im.convert("L"), dtype=np.float32)
    m = al > 128
    row = {"文件": path.replace("\\", "/").split("/")[-1], "模式": im.mode,
           "尺寸": f"{im.width}x{im.height}"}
    if m.sum() < 200:
        row["主体占比%"] = round(float(m.mean() * 100), 1)
        row["清晰度"] = 0
        row["头部清晰度"] = 0
        row["alpha均值"] = round(float(al.mean()), 1)
        row["半透明%"] = 100.0
        return row

    inner = _erode(m, 3)
    if inner.sum() < 200:
        inner = m
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()
    head_h = max(1, int((y1 - y0) * 0.22))
    head = m.copy()
    head[y0 + head_h:] = False

    row.update({
        "主体占比%": round(float(m.mean() * 100), 1),
        "清晰度": round(_lap_var(g, inner), 1),
        "头部清晰度": round(_lap_var(g, head), 1),
        "alpha均值": round(float(al[m].mean()), 1),
        "半透明%": round(float(((al > 20) & (al < 235)).mean() * 100), 1),
        "亮度": round(float(g[m].mean()), 1),
        "对比度": round(float(g[m].std()), 1),
    })
    return row


def main(argv: list[str]) -> int:
    paths = argv[1:]
    if not paths:
        print(__doc__)
        return 1
    rows = []
    for p in paths:
        try:
            rows.append(stats(p))
        except Exception as e:                    # noqa: BLE001
            print(f"{p}: 读取失败 {e}")
    if not rows:
        return 1
    cols = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in cols:
                cols.append(k)
    w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
