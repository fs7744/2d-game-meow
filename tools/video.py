"""W5 视频驱动：文生视频 / 图生视频，落地 mp4 + gif + 帧序列。

用法：
    python tools/video.py t2v "pixel art cat knight walking, flat background" -o assets/clip
    python tools/video.py i2v assets/cat.png "the same character walking" -o assets/clip
    python tools/video.py t2v "..." --frames 49 --steps 20 --size 480 --fps 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from workflows import anim  # noqa: E402
from workflows.video import VideoEngine, save_frames, save_mp4  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["t2v", "i2v"])
    ap.add_argument("prompt")
    ap.add_argument("--image", default=None, help="i2v 的参考图")
    ap.add_argument("-o", "--out", default="assets/clip")
    ap.add_argument("--negative", default="")
    ap.add_argument("--frames", type=int, default=25, help="帧数（内部会夹到 4n+1）")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261027)
    a = ap.parse_args()

    img = None
    if a.mode == "i2v":
        if not a.image:
            print("i2v 必须给 --image")
            return 2
        p = Path(a.image)
        if not p.exists():
            print(f"找不到参考图: {p}")
            return 2
        img = Image.open(p)

    eng = VideoEngine()
    frames, meta = eng.generate(a.prompt, image=img, negative=a.negative,
                                num_frames=a.frames, steps=a.steps,
                                height=a.size, width=a.size, seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_frames(frames, out / "frames")
    mp4 = save_mp4(frames, out / "clip.mp4", fps=a.fps)
    gif = anim.save_gif(frames, out / "clip.gif", fps=a.fps)
    print(f"\n{meta['mode']}: {meta['frames']} 帧 / {meta['seconds']}s "
          f"({meta['seconds_per_frame']}s/帧)，峰值显存 {meta['peak_vram_gb']}GB")
    print(f"产物：{mp4}、{gif}、{out / 'frames'}/")
    print("\n接姿态提取：")
    print(f"  python tools/pose_from_image.py {mp4} --step 2 -o assets/pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
