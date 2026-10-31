"""生成标准 OpenPose 骨架图（COCO-18），用于无 DWPose 依赖时的条件图冒烟测试。

OpenPose 的 ControlNet 只认「彩色肢体 + 关节圆点」的画法，颜色编码必须与训练时一致，
否则条件失效（这是 ControlNet 官方仓库明确写死的约定）。
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

# 官方 18 色（BODY_25 之前的 COCO-18 版本）
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]
# 标准 19 条边 -> 颜色索引
LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    (2, 16), (5, 17),
]


def render(kp: dict[int, tuple[float, float]], size=(512, 768), stick: int = 12, dot: int = 8) -> Image.Image:
    """kp: {关节索引: (x, y)}，坐标归一化到 0~1。"""
    w, h = size
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = {k: (v[0] * w, v[1] * h) for k, v in kp.items()}
    for i, (a, b) in enumerate(LIMBS):
        if a in pts and b in pts:
            d.line([pts[a], pts[b]], fill=COLORS[i % len(COLORS)], width=stick)
    for k, (x, y) in pts.items():
        c = COLORS[k % len(COLORS)]
        d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=c)
    return img


def walk_pose(phase: float) -> dict[int, tuple[float, float]]:
    """一个简单的行走姿态族：phase=0 双脚并拢，phase=0.5 迈到最大步幅。"""
    s = math.sin(phase * 2 * math.pi)
    return {
        0: (0.50, 0.10),                       # nose
        1: (0.50, 0.20),                       # neck
        2: (0.42, 0.22), 3: (0.38, 0.36), 4: (0.36 - 0.10 * s, 0.50),
        5: (0.58, 0.22), 6: (0.62, 0.36), 7: (0.64 + 0.10 * s, 0.50),
        8: (0.46, 0.52), 9: (0.44 - 0.06 * s, 0.68), 10: (0.42 - 0.14 * s, 0.86),
        11: (0.54, 0.52), 12: (0.56 + 0.06 * s, 0.68), 13: (0.58 + 0.14 * s, 0.86),
        14: (0.47, 0.09), 15: (0.53, 0.09), 16: (0.46, 0.09), 17: (0.54, 0.09),
    }


def main() -> None:
    # 输出到统一路径模块解析出来的资产根。测试期间 conftest 会把
    # MEOW_ASSETS_DIR 指到 tests/_out/assets，所以这里自动跟着走，
    # 不会往仓库的 assets/ 里丢测试素材。
    from workflows import paths

    out = paths.ASSETS
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(8):
        img = render(walk_pose(i / 8))
        frames.append(img)
        img.save(out / f"_pose_walk_{i:02d}.png")
    print(f"wrote {len(frames)} pose frames -> {out}")


if __name__ == "__main__":
    main()
