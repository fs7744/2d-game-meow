"""工作流实现层。统一接口 generate(params) -> dict(asset)。

对应文档 §5：
W1 txt2img / W2 img2img / W3 inpaint / W4 poseswap(+sprite sheet)。
所有函数都是同步阻塞的，由 app.py 的任务队列串行调度（单卡不并发）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

from . import anim as anim_mod
from . import matte as matte_mod
from . import postprocess as pp
from . import presets
from .engine import Engine

ASSETS = Path(__file__).resolve().parent.parent / "assets"

NEG_DEFAULT = (
    "photo, photorealistic, 3d render, blurry, jpeg artifacts, gradient background, "
    "multiple characters, text, watermark, signature, cropped, out of frame, "
    "anti-aliasing, soft edges"
)


def _stamp(tag: str) -> str:
    return f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"


def _save(img: Image.Image, name: str, tags: str, meta: dict) -> Dict[str, Any]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{name}.png"
    img.save(out)
    side = {"file": str(out), "tags": tags, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta}
    Path(str(out) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=2), "utf-8")
    with (ASSETS / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(side, ensure_ascii=False) + "\n")
    return {"file": str(out), "sidecar": str(out) + ".json", "meta": side}


def _timer():
    t0 = time.perf_counter()
    return lambda: round(time.perf_counter() - t0, 2)


# 质量档位（实测，RTX 4090，fp16，DPM++ 2M Karras）：
#   草稿  512² /  8 步  —— 0.30s，只用来验证构图
#   可用  768² / 30 步  —— 1.8s，SD1.5 的甜点：再大就开始重复肢体（1024² 实测出 3~4 个猫头）
#   精修  768² / 45 步 + cfg 6.5 —— 2.6s，细节更稳，代价是慢
# 像素化统一用 pixel=6 / palette=32（768 ÷ 6 = 128 有效像素，正好是精灵尺寸）；
# 512² 配 pixel=4 会把有效像素压到 128 但源信息本来就少，实测明显更糊。
QUALITY_PRESETS = {
    "draft": dict(width=512, height=512, steps=8, cfg=7.0, pixel=0, palette=0),
    "normal": dict(width=768, height=768, steps=30, cfg=7.0, pixel=6, palette=32),
    "fine": dict(width=768, height=768, steps=45, cfg=6.5, pixel=6, palette=32),
}


# ---------------- W1 文生图 ----------------

def txt2img(eng: Engine, *, prompt: str, negative: Optional[str] = None,
            width: int = 768, height: int = 768, steps: int = 30, cfg: float = 7.0,
            seed: int = -1, lora: Optional[str] = None, lora_scale: float = 0.8,
            pixel: int = 0, palette: int = 0, preset: str = "sprite",
            matte: bool = False, matte_backend: str = "auto",
            model: str = "sd15", tag: str = "w1") -> Dict[str, Any]:
    prompt, negative = presets.build(prompt, negative, preset)
    ntok = presets.rough_tokens(prompt)
    if ntok > 77:
        print(f"[warn] 正向提示词约 {ntok} token，超过 SD1.5 的 77 上限，尾部会被截断")
    pipe = eng.txt2img_pipe(model)
    eng.apply_lora(pipe, lora, lora_scale)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    gen = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    img = pipe(prompt=prompt, negative_prompt=negative or NEG_DEFAULT, width=width, height=height,
               num_inference_steps=steps, guidance_scale=cfg, generator=gen).images[0]
    seconds = dt()
    # 贴边检测要在抠图之后、像素化之前做（像素化会改变 alpha 边缘像素）
    edge: list[str] = []
    # 顺序很重要：先抠图（按原始颜色的连通域判断更准），再像素化与量化。
    # 反过来做会把背景噪点量化成格纹，抠图也会因为色阶被压而失效。
    if matte:
        # 保留 alpha：转回 RGB 会把透明区域压成黑色，等于白抠
        img = matte_mod.despeckle(
            matte_mod.matte_auto(img.convert("RGBA"), matte_backend, eng.matte_dir))
        edge = matte_mod.edge_touch(img)
    if pixel > 1:
        img = pp.pixelate(img, pixel)
    if palette:
        img = pp.quantize_palette(img, palette)
    return _save(img, _stamp(tag), "txt2img", {
        "workflow": "w1", "prompt": prompt, "negative": negative or NEG_DEFAULT,
        "width": width, "height": height, "steps": steps, "cfg": cfg, "seed": seed,
        "lora": lora, "lora_scale": lora_scale if lora else None,
        "pixel_block": pixel, "palette": palette, "matte": matte,
        "matte_backend": matte_backend, "edge_touch": edge, "model": model,
        "seconds": seconds, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })


# ---------------- W2 图生图 ----------------

def img2img(eng: Engine, *, init_image: str | Path, prompt: str,
            negative: Optional[str] = None, strength: float = 0.55, steps: int = 25,
            cfg: float = 7.0, seed: int = -1, lora: Optional[str] = None,
            lora_scale: float = 0.8, model: str = "sd15", tag: str = "w2") -> Dict[str, Any]:
    pipe = eng.img2img_pipe(model)
    eng.apply_lora(pipe, lora, lora_scale)
    ref = Image.open(init_image).convert("RGB")
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    img = pipe(prompt=prompt, negative_prompt=negative or NEG_DEFAULT, image=ref,
               strength=strength, num_inference_steps=steps, guidance_scale=cfg,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    seconds = dt()
    return _save(img, _stamp(tag), "img2img", {
        "workflow": "w2", "prompt": prompt, "init_image": str(init_image),
        "strength": strength, "steps": steps, "cfg": cfg, "seed": seed, "lora": lora,
        "seconds": seconds, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })


# ---------------- W3 局部重绘 ----------------

def inpaint(eng: Engine, *, image: str | Path, mask: str | Path, prompt: str,
            negative: Optional[str] = None, strength: float = 0.9, steps: int = 25,
            cfg: float = 7.5, seed: int = -1, tag: str = "w3") -> Dict[str, Any]:
    """只重画 mask 区域，并把结果按羽化蒙版贴回——选区外像素逐点不变。"""
    pipe = eng.inpaint_pipe()
    src = Image.open(image).convert("RGB")
    msk = Image.open(mask).convert("L")
    if msk.size != src.size:
        msk = msk.resize(src.size, Image.NEAREST)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    # 关键：不显式传 width/height 时，diffusers 会用 unet.sample_size（512）覆盖，
    # 输出尺寸就和源图不一致了（贴回时会 "images do not match"）。
    w8, h8 = max(64, src.width // 8 * 8), max(64, src.height // 8 * 8)
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    raw = pipe(prompt=prompt, negative_prompt=negative or NEG_DEFAULT, image=src,
               mask_image=msk, strength=strength, num_inference_steps=steps,
               guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(seed),
               width=w8, height=h8).images[0]
    seconds = dt()
    if raw.size != src.size:
        raw = raw.resize(src.size, Image.LANCZOS)
    composed = composite(src, raw, msk)
    return _save(composed, _stamp(tag), "inpaint", {
        "workflow": "w3", "prompt": prompt, "image": str(image), "mask": str(mask),
        "strength": strength, "steps": steps, "seed": seed, "raw": None,
        "seconds": seconds, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })


def composite(src: Image.Image, gen: Image.Image, mask: Image.Image, feather: int = 3) -> Image.Image:
    """把生成结果按蒙版贴回，**承诺选区外逐点不变**。

    羽化必须「向内」：先做形态学腐蚀（MinFilter）再高斯模糊，模糊的扩散范围才不会
    越过原始蒙版边界。反过来直接模糊会把选区外的像素一起吃掉。
    """
    from PIL import ImageFilter

    m = mask.convert("L")
    if feather > 0:
        m = m.filter(ImageFilter.MinFilter(2 * feather + 1))   # 向内收缩 feather px
        m = m.filter(ImageFilter.GaussianBlur(feather))        # 边缘软化，仍不越界
    m = m.point(lambda v: 255 if v >= 128 else 0)
    out = src.convert("RGBA").copy()
    out.paste(gen.convert("RGBA"), (0, 0), m)
    return out.convert("RGB")


# ---------------- W4 姿态驱动 -> 精灵帧 ----------------

def poseswap(eng: Engine, *, pose_images: Sequence[str | Path], prompt: str,
             negative: Optional[str] = None, ref_image: Optional[str | Path] = None,
             lora: Optional[str] = None, lora_scale: float = 0.8,
             ipa_scale: float = 0.8, cn_scale: float = 1.0, steps: int = 25,
             cfg: float = 7.0, seed: int = 20261027, width: int = 768, height: int = 768,
             cols: int = 8, pixel: int = 0, palette: int = 0,
             matte: bool = False, matte_backend: str = "auto",
             export: Optional[str] = None,
             preset: Optional[str] = None,
             anim_name: str = "walk", fps: int = 10, tag: str = "w4") -> Dict[str, Any]:
    """逐帧生成 -> 锚点对齐 -> 打包 sprite sheet。固定 seed 是跨帧一致的前提。

    preset 为 None 表示不套契约（保持向后兼容）；w4 + LoRA 时建议显式传，
    否则武侠 LoRA 会自己画水墨背景与题字，抠图抠不净。
    """
    negative = negative or NEG_DEFAULT
    if preset is not None:
        prompt, negative = presets.build(prompt, negative, preset)
    pipe = eng.pose_pipe(with_ipadapter=ref_image is not None)
    eng.apply_lora(pipe, lora, lora_scale)
    if ref_image is not None:
        pipe.set_ip_adapter_scale(ipa_scale)
        ref = Image.open(ref_image).convert("RGB").resize((width, height))
    else:
        ref = None

    frames: List[Image.Image] = []
    per_frame: List[float] = []
    torch.cuda.reset_peak_memory_stats()
    for pose_path in pose_images:
        pose = Image.open(pose_path).convert("RGB").resize((width, height))
        dt = _timer()
        kwargs: Dict[str, Any] = dict(
            prompt=prompt, negative_prompt=negative, image=pose,
            num_inference_steps=steps, guidance_scale=cfg,
            controlnet_conditioning_scale=cn_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
            width=width, height=height,
        )
        if ref is not None:
            kwargs["ip_adapter_image"] = ref
        out = pipe(**kwargs).images[0]
        per_frame.append(dt())
        if matte:
            out = matte_mod.despeckle(
                matte_mod.matte_auto(out.convert("RGBA"), matte_backend, eng.matte_dir))
        if pixel > 1:
            out = pp.pixelate(out, pixel)
        if palette:
            out = pp.quantize_palette(out, palette)
        frames.append(out.convert("RGBA"))

    aligned = pp.align_sequence(frames)
    meta = {
        "workflow": "w4", "prompt": prompt, "negative": negative or NEG_DEFAULT,
        "ref_image": str(ref_image) if ref_image else None, "lora": lora,
        "ipa_scale": ipa_scale if ref else None, "cn_scale": cn_scale,
        "steps": steps, "cfg": cfg, "seed": seed, "frames": len(frames), "matte": matte,
        "per_frame_seconds": per_frame,
        "seconds_total": round(sum(per_frame), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    name = _stamp(tag)
    fdir = ASSETS / f"{name}_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(aligned):
        f.save(fdir / f"frame_{i:04d}.png")
    info = pp.pack_sheet(aligned, ASSETS / f"{name}_sheet.png", cols=cols, sidecar=meta)
    info.update(meta)
    info["frames_dir"] = str(fdir)
    Path(ASSETS / f"{name}_sheet.png.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), "utf-8")
    if export:
        info["export"] = anim_mod.export_godot(
            aligned, export, anim_name=anim_name, fps=fps, loop=True, cols=cols)
    with (ASSETS / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sheet": info["sheet"], **meta}, ensure_ascii=False) + "\n")
    return info
