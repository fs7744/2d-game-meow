"""W5 视频驱动：文生视频 / 图生视频 → 帧序列 → 喂给姿态提取或直接打包。

底座用本机已有的 Wan2.2 TI2V-5B（diffusers WanPipeline 格式，37GB）。
为什么不是 MimicMotion：它权重 ~16GB 要从头下（代理 480KB/s ≈ 9 小时），
而 Wan 已经在盘上且经 pixelforge 实测可跑。MimicMotion 的优势是「真人动作驱动」，
等有明确需求再单独接。

显存现实（24GB 卡）：transformer bf16 约 18.6GB，必须 enable_model_cpu_offload，
峰值约 19GB；同时不要与 SD1.5 的 pipeline 共存（先 free）。
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

DEFAULT_WAN = Path("D:/code/pixelforge/models/wan2.2-ti2v-5b")


def wan_model_dir() -> Path:
    p = Path(os.environ.get("WAN_MODEL_DIR", DEFAULT_WAN))
    if not (p / "model_index.json").exists():
        raise FileNotFoundError(
            f"未找到 Wan2.2 视频底座：{p}。设置环境变量 WAN_MODEL_DIR 指向 "
            f"diffusers 格式的 wan2.2-ti2v-5b 目录。")
    return p


class VideoEngine:
    """懒加载 + 常驻。两种调用形态用两条管线（WanPipeline / WanImageToVideoPipeline）。"""

    def __init__(self, model_dir: str | Path | None = None, device: str = "cuda",
                 offload: bool = True) -> None:
        self.dir = Path(model_dir) if model_dir else wan_model_dir()
        self.device = device
        self.offload = offload
        self._pipes: dict[str, object] = {}

    def pipe(self, mode: str = "t2v"):
        import torch
        from diffusers import WanImageToVideoPipeline, WanPipeline

        if mode in self._pipes:
            return self._pipes[mode]

        cls = WanPipeline if mode == "t2v" else WanImageToVideoPipeline
        t0 = time.perf_counter()
        pipe = cls.from_pretrained(str(self.dir), torch_dtype=torch.bfloat16)
        if self.offload and self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        self._pipes[mode] = pipe
        print(f"[video] {mode} 加载 {time.perf_counter() - t0:.1f}s", flush=True)
        return pipe

    def generate(self, prompt: str, *, image: Optional[Image.Image] = None,
                 negative: str = "", num_frames: int = 25, steps: int = 10,
                 height: int = 384, width: int = 384, guidance: float = 5.0,
                 seed: int = 20261027) -> tuple[List[Image.Image], dict]:
        import torch

        if num_frames % 4 != 1:
            num_frames = (num_frames // 4) * 4 + 1        # Wan 要求 4n+1
        height, width = height // 16 * 16, width // 16 * 16
        mode = "i2v" if image is not None else "t2v"
        pipe = self.pipe(mode)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        kw = dict(prompt=prompt, negative_prompt=negative, height=height, width=width,
                  num_frames=num_frames, num_inference_steps=steps,
                  guidance_scale=guidance,
                  generator=torch.Generator("cpu").manual_seed(seed))
        if image is not None:
            kw["image"] = image.convert("RGB").resize((width, height))
        out = pipe(**kw)
        dt = time.perf_counter() - t0
        # diffusers 的 Wan 管线返回 numpy (T,H,W,3) 而不是 PIL，两种都要能吃
        frames = [as_image(f) for f in out.frames[0]]
        meta = {
            "mode": mode, "seconds": round(dt, 2), "frames": len(frames),
            "seconds_per_frame": round(dt / max(1, len(frames)), 2),
            "size": [width, height], "steps": steps, "seed": seed,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "model": self.dir.name,
        }
        return frames, meta

    def free(self) -> None:
        import gc

        import torch

        self._pipes.clear()
        gc.collect()
        torch.cuda.empty_cache()


# ---------------- 帧序列落地 ----------------

def as_image(frame) -> Image.Image:
    """numpy(uint8/float) 或 PIL 统一转成 RGB Image。"""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8) if arr.max() <= 1.0 \
            else np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

def save_mp4(frames: Sequence[Image.Image], out: str | Path, fps: int = 12) -> Path:
    """用 OpenCV 写 H.264 不保证有编码器，这里用 mp4v（兼容性好）。"""
    import cv2

    if not frames:
        raise ValueError("没有帧")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("OpenCV 打不开 mp4v 编码器")
    for f in frames:
        vw.write(cv2.cvtColor(np.asarray(f.convert("RGB")), cv2.COLOR_RGB2BGR))
    vw.release()
    return out


def save_frames(frames: Sequence[Image.Image], out_dir: str | Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        p = out_dir / f"frame_{i:04d}.png"
        f.save(p)
        paths.append(p)
    return paths
