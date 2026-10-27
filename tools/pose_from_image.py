"""从图片/视频/GIF/精灵图提骨骼，输出可直接喂 W4 的 OpenPose 条件图。

用法：
    # 单图
    python tools/pose_from_image.py assets/some_sprite.png -o assets/pose/
    # 精灵图（按网格切）
    python tools/pose_from_image.py sheet.png --grid 8x1 -o assets/pose/
    # 视频 / GIF（按步长抽帧）
    python tools/pose_from_image.py walk.mp4 --step 3 -o assets/pose/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from workflows.pose import PoseExtractor  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}


def frames_from_video(path: Path, step: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    idx = out = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield out, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out += 1
        idx += 1
    cap.release()


def frames_from_sheet(path: Path, cols: int, rows: int):
    sheet = Image.open(path).convert("RGBA")
    cw, ch = sheet.width // cols, sheet.height // rows
    n = 0
    for r in range(rows):
        for c in range(cols):
            cell = sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
            if cell.getchannel("A").getbbox() is None and cell.convert("L").getextrema()[1] == 0:
                continue                      # 空格子
            yield n, cell.convert("RGB")
            n += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--out", default="assets/pose")
    ap.add_argument("--grid", default=None, help="精灵图网格，如 8x1")
    ap.add_argument("--step", type=int, default=1, help="视频每 N 帧取一帧")
    ap.add_argument("--size", type=int, default=512, help="输出条件图边长")
    ap.add_argument("--thr", type=float, default=0.25, help="关节置信度阈值")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--detector", default="auto",
                    choices=["auto", "yolox", "torchvision", "none"],
                    help="auto=有 yolox 用 yolox，否则退 torchvision")
    ap.add_argument("--no-detector", action="store_true", help="等价于 --detector none")
    a = ap.parse_args()

    src = Path(a.source)
    if not src.exists():
        print(f"找不到输入: {src}")
        return 2
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = PoseExtractor(ROOT / "models" / "pose_keypoint" / "dwpose",
                       device=a.device, use_detector=None if a.detector != "none" else False,
                       detector=a.detector)
    print(f"检测器: {ex.backend}")

    if src.suffix.lower() in VIDEO_EXT:
        stream = frames_from_video(src, a.step)
    elif a.grid:
        cols, rows = (int(x) for x in a.grid.lower().split("x"))
        stream = frames_from_sheet(src, cols, rows)
    else:
        stream = iter([(0, Image.open(src).convert("RGB"))])

    n, kept = 0, 0
    for i, img in stream:
        kp = ex.keypoints(img)
        vis = sum(1 for _, _, s in kp.values() if s >= a.thr)
        pm = ex.pose_map(img, size=(a.size, a.size), thr=a.thr)
        pm.save(out_dir / f"pose_{i:04d}.png")
        print(f"  frame {i:04d}  可见关节 {vis}/18  -> pose_{i:04d}.png")
        n += 1
        kept += vis >= 6
    print(f"完成：{n} 帧，其中 {kept} 帧骨架可用（>=6 个关节）-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
