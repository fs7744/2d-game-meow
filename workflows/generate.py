"""工作流实现层。统一接口 generate(params) -> dict(asset)。

对应文档 §5：
W1 txt2img / W2 img2img / W3 inpaint / W4 poseswap(+sprite sheet)。
所有函数都是同步阻塞的，由 app.py 的任务队列串行调度（单卡不并发）。
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from . import anim as anim_mod
from . import clip as clip_mod
from . import matte as matte_mod
from . import postprocess as pp
from . import presets
from . import quality
from .engine import Engine

ASSETS = Path(__file__).resolve().parent.parent / "assets"
PROJECTS_DIR = ASSETS / "projects"

NEG_DEFAULT = (
    "photo, photorealistic, 3d render, blurry, jpeg artifacts, gradient background, "
    "multiple characters, text, watermark, signature, cropped, out of frame, "
    "anti-aliasing, soft edges"
)


_seq = itertools.count()


def scope_dir(scope: str = "") -> Path:
    """项目产物目录：assets/projects/<scope>/。

    scope 是前端的项目 id（p…），只保留 ASCII 字母数字与 -_，空则用 default。
    每个项目的 PNG / sidecar / raw / 帧序列 / 精灵图都在自己的目录里，
    项目之间互不可见（后端还有一层校验，见 img2img 的源图检查）。
    """
    safe = "".join(
        ch for ch in (scope or "") if ch.isascii() and (ch.isalnum() or ch in "-_")
    )[:64] or "default"
    d = PROJECTS_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp(tag: str) -> str:
    """产物文件名：`<tag>_<时间>_<序号>`。

    秒级时间戳不够：同一秒内连跑两次（队列很快、测试同 seed 对照）会得到同名文件，
    后一次直接把前一次的 PNG 覆盖掉——两路结果指向同一个文件，对照实验全废。
    序号不追求跨进程唯一，只要同进程内单调即可。
    """
    return f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{next(_seq) % 10000:04d}"


def _save(img: Image.Image, name: str, tags: str, meta: dict,
          raw: Image.Image | None = None, scope: str = "") -> Dict[str, Any]:
    """落盘 PNG + sidecar + 索引。

    raw 非空时额外存一份 `<name>_raw.png`（后处理**之前**的原图）。图生图的源图
    必须用这张：已经像素化到 128 有效像素的成品再喂回 img2img，模型只会把色块
    重新糊成插画，重绘幅度稍大就丢掉像素感。
    """
    out_dir = scope_dir(scope)
    out = out_dir / f"{name}.png"
    img.save(out)
    side = {"file": str(out), "tags": tags, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scope": scope or "default", **meta}
    if raw is not None:
        raw_path = out_dir / f"{name}_raw.png"
        raw.save(raw_path)
        side["raw"] = str(raw_path)
    Path(str(out) + ".json").write_text(json.dumps(side, ensure_ascii=False, indent=2), "utf-8")
    with (ASSETS / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(side, ensure_ascii=False) + "\n")
    # url 是相对 assets 的路径：产物现在在 projects/<id>/ 子目录里，只取 basename 会 404
    rel = out.relative_to(ASSETS).as_posix()
    return {"file": str(out), "url": f"/files/{rel}", "sidecar": str(out) + ".json",
            "meta": side}


def _timer():
    t0 = time.perf_counter()
    return lambda: round(time.perf_counter() - t0, 2)


def _postprocess(img: Image.Image, eng: Engine, *, matte: bool, matte_backend: str,
                 denoise: int, pixel: int, palette: int, clean_iso: int,
                 info: Dict[str, Any] | None = None):
    """抠图 → 像素化 → 量化 → 去孤立点。返回 (图, 贴边边, 主体数)。

    顺序固定，不能改：先抠图（按原始颜色判连通域更准），再像素化与量化。
    反过来会把背景噪点量化成格纹，抠图也会因为色阶被压而失效。
    三张卡（文生图 / 图生图 / 姿态帧）共用这一份，避免各写一遍走样。
    """
    edge: list[str] = []
    subjects = None
    if matte:
        # 保留 alpha：转回 RGB 会把透明区域压成黑色，等于白抠
        img = matte_mod.despeckle(
            matte_mod.matte_auto(img.convert("RGBA"), matte_backend, eng.matte_dir,
                                 info=info if info is not None else {}),
            denoise_iters=denoise)
        edge = matte_mod.edge_touch(img)
        # 多主体是 SD1.5 画小怪物的通病（史莱姆尤甚），负向词压不住，只能量出来
        subjects = pp.foreground_blobs(img)
    if pixel > 1:
        img = pp.pixelate(img, pixel)
    if palette:
        img = pp.quantize_palette(img, palette)
    if clean_iso and (pixel > 1 or palette):
        # 量化之后才去孤立点：量化本身就会把相邻色合并，此时再判「孤立」才准
        img = pp.clean_isolated(img, passes=clean_iso)
    return img, edge, subjects


def _skip_embeds(pipe, prompt: str, negative: str, clip_skip: int, model: str):
    """clip_skip>1 时返回 (正向 embeds, 负向 embeds)，否则 None。

    只有 SD1.5 走这条路：SDXL 的两个文本编码器 + pooled embeds 要一起改，
    收益远小于 SD1.5（而且 SDXL 模型页基本不写 clip skip）。
    """
    if not clip_skip or clip_skip <= 1:
        return None
    if model != "sd15":
        print(f"[warn] clip_skip={clip_skip} 只对 SD1.5 生效，{model} 已忽略")
        return None
    return clip_mod.embeds(pipe, prompt, negative, clip_skip)


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
            matte: bool = False, matte_backend: str = "auto", denoise: int = 1,
            clean_iso: int = 1, model: str = "sd15", clip_skip: int = 1,
            scope: str = "", tag: str = "w1") -> Dict[str, Any]:
    prompt, negative = presets.build(prompt, negative, preset)
    ntok = presets.rough_tokens(prompt)
    if ntok > 77:
        print(f"[warn] 正向提示词约 {ntok} token，超过 SD1.5 的 77 上限，尾部会被截断")
    pipe = eng.txt2img_pipe(model)
    eng.apply_lora(pipe, lora, lora_scale)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    neg = negative or NEG_DEFAULT
    gen = torch.Generator("cuda").manual_seed(seed)
    kw: Dict[str, Any] = dict(prompt=prompt, negative_prompt=neg)
    emb = _skip_embeds(pipe, prompt, neg, clip_skip, model)
    if emb is not None:
        kw = dict(prompt_embeds=emb[0], negative_prompt_embeds=emb[1])
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    img = pipe(width=width, height=height, num_inference_steps=steps,
               guidance_scale=cfg, generator=gen, **kw).images[0]
    seconds = dt()
    # 后处理之前的原图另存一份：图生图要拿它当源图（见 _save 的说明）
    raw = img.copy() if (matte or pixel > 1 or palette) else None
    matte_info: Dict[str, Any] = {}
    img, edge, subjects = _postprocess(
        img, eng, matte=matte, matte_backend=matte_backend, denoise=denoise,
        pixel=pixel, palette=palette, clean_iso=clean_iso, info=matte_info)
    return _save(img, _stamp(tag), "txt2img", {
        "workflow": "w1", "prompt": prompt, "negative": negative or NEG_DEFAULT,
        "width": width, "height": height, "steps": steps, "cfg": cfg, "seed": seed,
        "lora": lora, "lora_scale": lora_scale if lora else None,
        "pixel_block": pixel, "palette": palette, "matte": matte,
        "matte_backend": matte_backend, "denoise": denoise, "clean_iso": clean_iso,
        "matte_used": matte_info.get("backend"), "matte_fallback": matte_info.get("fallback"),
        "edge_touch": edge, "model": model,
        "clip_skip": clip_skip if emb is not None else 1,
        "subjects": subjects,
        "seconds": seconds, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


# ---------------- W2 图生图 ----------------

def img2img(eng: Engine, *, init_image: str | Path, prompt: str,
            negative: Optional[str] = None, strength: float = 0.55, steps: int = 25,
            cfg: float = 7.0, seed: int = -1, lora: Optional[str] = None,
            lora_scale: float = 0.8, model: str = "sd15", clip_skip: int = 1,
            preset: Optional[str] = None, pixel: int = 0, palette: int = 0,
            matte: bool = False, matte_backend: str = "auto", denoise: int = 1,
            clean_iso: int = 1, scope: str = "", tag: str = "w2") -> Dict[str, Any]:
    """在图生图里重画原图的**内容**（改姿势细节、换配色、加装饰），构图与轮廓保持。

    与 W1 的区别在于「底」是已有图，所以：
    - 输出尺寸由源图决定，不接受 width/height（diffusers 也这么做）。
    - `strength` 是重绘幅度：0.3 只动细节，0.6 明显改画风，0.85 基本重画。
    - 后处理（抠图/像素化/量化）与 W1 完全同一套。源图**应当**是未经像素化的
      原图（txt2img 会额外存 `<name>_raw.png`），拿成品精灵回炉会把色块糊掉。
    """
    if preset is not None:
        prompt, negative = presets.build(prompt, negative, preset)
    # 项目隔离：源图必须在本项目目录里。别的项目的产物（assets/projects/<别的>/…）
    # 一律拒绝——项目之间互不可见。历史公共区（assets 根目录下的老文件）仍然可用。
    if scope:
        src_path = Path(init_image).resolve()
        own = scope_dir(scope).resolve()
        root = PROJECTS_DIR.resolve()
        if str(src_path).startswith(str(root)) and not str(src_path).startswith(str(own)):
            raise ValueError(
                f"源图不属于当前项目：{src_path.name} 在别的项目目录里（项目之间相互隔离）")
    pipe = eng.img2img_pipe(model)
    eng.apply_lora(pipe, lora, lora_scale)
    ref = Image.open(init_image).convert("RGB")
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    neg = negative or NEG_DEFAULT
    kw: Dict[str, Any] = dict(prompt=prompt, negative_prompt=neg)
    emb = _skip_embeds(pipe, prompt, neg, clip_skip, model)
    if emb is not None:
        kw = dict(prompt_embeds=emb[0], negative_prompt_embeds=emb[1])
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    img = pipe(image=ref, strength=strength, num_inference_steps=steps,
               guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(seed),
               **kw).images[0]
    seconds = dt()
    raw = img.copy() if (matte or pixel > 1 or palette) else None
    matte_info: Dict[str, Any] = {}
    img, edge, subjects = _postprocess(
        img, eng, matte=matte, matte_backend=matte_backend, denoise=denoise,
        pixel=pixel, palette=palette, clean_iso=clean_iso, info=matte_info)
    return _save(img, _stamp(tag), "img2img", {
        "workflow": "w2", "prompt": prompt, "negative": neg,
        "init_image": str(init_image), "strength": strength, "steps": steps,
        "cfg": cfg, "seed": seed, "lora": lora,
        "lora_scale": lora_scale if lora else None,
        "pixel_block": pixel, "palette": palette, "matte": matte,
        "matte_backend": matte_backend, "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "edge_touch": edge, "subjects": subjects, "model": model,
        "clip_skip": clip_skip if emb is not None else 1,
        "seconds": seconds, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


# ---------------- W3 局部重绘 ----------------

def inpaint(eng: Engine, *, image: str | Path, mask: str | Path, prompt: str,
            negative: Optional[str] = None, strength: float = 0.9, steps: int = 25,
            cfg: float = 7.5, seed: int = -1, model: str = "sd15", scope: str = "",
            tag: str = "w3") -> Dict[str, Any]:
    """只重画 mask 区域，并把结果按羽化蒙版贴回——选区外像素逐点不变。"""
    pipe = eng.inpaint_pipe(model)
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
        "strength": strength, "steps": steps, "cfg": cfg, "seed": seed, "model": model,
        "raw": None,
        "seconds": seconds, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, scope=scope)


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


# ---------------- WBG 抠图去背景 / 换背景 ----------------

def matte_bg(eng: Engine, *, init_image: str | Path,
             background: str | Path | None = None,
             bg_color: str | Sequence[int] | None = None,
             matte_backend: str = "auto", denoise: int = 1,
             scope: str = "", tag: str = "wbg") -> Dict[str, Any]:
    """抠图去背景 / 换背景。**不需要模型**，纯图像处理，所以不占显存、瞬间出。

    三种输出：
      background=None + bg_color=None → 透明底前景（纯去背景，PNG 带 alpha）
      bg_color="#rrggbb"                 → 纯色背景
      background=<另一张图>              → 换背景（前景叠到那张图上）

    指定了背景时，会**额外**存一份 `<name>_cut.png`（透明底前景），
    方便回头再换别的背景。
    """
    src = Image.open(init_image).convert("RGBA")
    # 抠图：与其它工作流共用同一套（含 u2net 抠空自动回退的保护）
    cut = matte_mod.despeckle(
        matte_mod.matte_auto(src, matte_backend, eng.matte_dir), denoise_iters=denoise)
    fg = float((np.asarray(cut)[..., 3] > 127).mean())

    out = cut
    mode = "alpha"
    bg_path = None
    if background is not None:
        bg = Image.open(background).convert("RGBA")
        # 背景铺满前景尺寸：cover 裁切，不拉伸变形
        w, h = cut.size
        scale = max(w / bg.width, h / bg.height)
        bg = bg.resize((max(1, int(bg.width * scale)), max(1, int(bg.height * scale))),
                       Image.LANCZOS)
        left = (bg.width - w) // 2
        top = (bg.height - h) // 2
        bg = bg.crop((left, top, left + w, top + h))
        canvas = bg.copy()
        canvas.alpha_composite(cut)
        out = canvas
        mode = "image"
        bg_path = str(background)
    elif bg_color:
        mode = "color"
        col = _parse_color(bg_color)
        out = Image.new("RGBA", cut.size, (col[0], col[1], col[2], 255))
        out.alpha_composite(cut)

    name = _stamp(tag)
    meta = {
        "workflow": "wbg", "scope": scope or "default",
        "init_image": str(init_image), "background": bg_path,
        "bg_mode": mode, "bg_color": str(bg_color) if mode == "color" else None,
        "matte_backend": matte_backend, "fg_ratio": round(fg, 4),
        "width": out.width, "height": out.height,
    }
    res = _save(out, name, "matte_bg", meta, scope=scope)
    if mode != "alpha":
        # 透明底那份也留着（换背景后还能回头换）
        cut_path = scope_dir(scope) / f"{name}_cut.png"
        cut.save(cut_path)
        res["meta"]["cut"] = str(cut_path)
        res["meta"]["cut_url"] = f"/files/{cut_path.relative_to(ASSETS).as_posix()}"
    return res


def _parse_color(c: str | Sequence[int]) -> tuple[int, int, int]:
    """把 "#rrggbb" / "rgb(r,g,b)" / (r,g,b) 统一成 RGB 三元组。"""
    if isinstance(c, (tuple, list)):
        return (int(c[0]), int(c[1]), int(c[2]))
    s = str(c).strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (255, 255, 255)


# ---------------- W4 姿态驱动 -> 精灵帧 ----------------

def poseswap(eng: Engine, *, pose_images: Sequence[str | Path], prompt: str,
             negative: Optional[str] = None, ref_image: Optional[str | Path] = None,
             lora: Optional[str] = None, lora_scale: float = 0.8,
             ipa_scale: float = 0.8, cn_scale: float = 1.0, steps: int = 25,
             cfg: float = 7.0, seed: int = 20261027, width: int = 768, height: int = 768,
             cols: int = 8, pixel: int = 0, palette: int = 0,
             matte: bool = False, matte_backend: str = "auto", denoise: int = 1,
             clean_iso: int = 1, export: Optional[str] = None,
             controlnet: str = "openpose", preset: Optional[str] = None,
             model: str = "sd15",
             candidates: int = 1, gate: bool = True, gate_min_points: int = 8,
             gate_retries: int = 2, gate_seed_step: int = 1000,
             clip_skip: int = 1, scope: str = "",
             anim_name: str = "walk", fps: int = 10, tag: str = "w4") -> Dict[str, Any]:
    """逐帧生成 -> 锚点对齐 -> 打包 sprite sheet。固定 seed 是跨帧一致的前提。

    preset 为 None 表示不套契约（保持向后兼容）；w4 + LoRA 时建议显式传，
    否则武侠 LoRA 会自己画水墨背景与题字，抠图抠不净。

    `candidates` / `gate`（批量挑帧 + 质量闸门）：
      每帧默认生成 `candidates` 个候选（种子 seed, seed+gate_seed_step, ...），
      用 quality.evaluate 打分后取最好的一个；候选里没有一个过闸门（检出关节
      数 < gate_min_points，或有边被画布切断）时，再补提最多 `gate_retries` 次。
      闸门的代价是每帧多花几十毫秒（姿态模型跑 CPU），换来的是不再往精灵图里
      塞没画出手脚、或者脚被裁掉的废帧——这类帧以前只能人眼一张张挑。
      姿态模型缺失时闸门自动关闭并告警，此时 candidates 仍然生效（只按分数挑，
      分数里只剩占比/贴边/漂移三项）。
    """
    negative = negative or NEG_DEFAULT
    if preset is not None:
        prompt, negative = presets.build(prompt, negative, preset)
    pipe = eng.pose_pipe(controlnet=controlnet, with_ipadapter=ref_image is not None,
                        model=model)
    eng.apply_lora(pipe, lora, lora_scale)
    if ref_image is not None:
        pipe.set_ip_adapter_scale(ipa_scale)
        ref = Image.open(ref_image).convert("RGB").resize((width, height))
    else:
        ref = None

    extractor = None
    if gate:
        extractor = eng.pose_extractor()
        if extractor is None:
            print("[warn] 质量闸门不可用（姿态模型或 onnxruntime 缺失），本次只按分数挑帧")
    candidates = max(1, int(candidates))
    # 闸门不可用时不留补提额度：没有判据的重提等于烧显卡
    retries = max(0, int(gate_retries)) if extractor is not None else 0
    attempts_max = candidates + retries
    # clip skip 的 embeds 每帧都一样，算一次就够
    emb = _skip_embeds(pipe, prompt, negative, clip_skip, model)

    def _one_frame(pose: Image.Image, frame_seed: int):
        dt = _timer()
        kwargs: Dict[str, Any] = dict(
            image=pose,
            num_inference_steps=steps, guidance_scale=cfg,
            controlnet_conditioning_scale=cn_scale,
            generator=torch.Generator("cuda").manual_seed(frame_seed),
            width=width, height=height,
        )
        if emb is None:
            kwargs.update(prompt=prompt, negative_prompt=negative)
        else:
            kwargs.update(prompt_embeds=emb[0], negative_prompt_embeds=emb[1])
        if ref is not None:
            kwargs["ip_adapter_image"] = ref
        out = pipe(**kwargs).images[0]
        secs = dt()
        out, _edge, _subj = _postprocess(
            out, eng, matte=matte, matte_backend=matte_backend, denoise=denoise,
            pixel=pixel, palette=palette, clean_iso=clean_iso, info=matte_info)
        return out.convert("RGBA"), secs

    matte_info: Dict[str, Any] = {}
    frames: List[Image.Image] = []
    per_frame: List[float] = []
    frame_scores: List[Dict[str, Any]] = []
    attempts_used = 0
    torch.cuda.reset_peak_memory_stats()
    for pose_path in pose_images:
        pose = Image.open(pose_path).convert("RGB").resize((width, height))
        prev = frames[-1] if frames else None      # 只在抠图后有意义，未抠图 drift 恒 0
        best: Optional[tuple[Dict[str, Any], Image.Image]] = None
        spent = 0.0
        used = 0
        for i in range(attempts_max):
            cand, secs = _one_frame(pose, seed + i * gate_seed_step)
            spent += secs
            used += 1
            ev = quality.evaluate(cand, extractor, prev=prev, min_points=gate_min_points)
            if best is None or ev["score"] > best[0]["score"]:
                best = (ev, cand)
            # 候选没凑齐就继续；凑齐后挑中的那个过了闸门就收工，不过才动用补提额度
            if used >= candidates and (extractor is None or best[0]["passed"]):
                break
        ev, frame = best
        ev = {**ev, "attempts": used}
        frame_scores.append(ev)
        attempts_used += used
        per_frame.append(round(spent, 2))
        frames.append(frame)

    aligned = pp.align_sequence(frames)
    passed = sum(1 for s in frame_scores if s["passed"])
    meta = {
        "workflow": "w4", "scope": scope or "default", "prompt": prompt, "negative": negative or NEG_DEFAULT,
        "ref_image": str(ref_image) if ref_image else None, "lora": lora,
        "ipa_scale": ipa_scale if ref else None, "cn_scale": cn_scale,
        "steps": steps, "cfg": cfg, "seed": seed, "frames": len(frames), "matte": matte,
        "matte_used": matte_info.get("backend"), "matte_fallback": matte_info.get("fallback"),
        "model": model,
        "clip_skip": clip_skip if emb is not None else 1,
        "candidates": candidates, "gate": bool(gate and extractor is not None),
        "gate_min_points": gate_min_points,
        "gate_retries": retries,
        "attempts": attempts_used,
        "gate_passed": passed,
        "frame_scores": frame_scores,
        "per_frame_seconds": per_frame,
        "seconds_total": round(sum(per_frame), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    name = _stamp(tag)
    out_dir = scope_dir(scope)
    fdir = out_dir / f"{name}_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(aligned):
        f.save(fdir / f"frame_{i:04d}.png")
    sheet_path = out_dir / f"{name}_sheet.png"
    info = pp.pack_sheet(aligned, sheet_path, cols=cols, sidecar=meta)
    info.update(meta)
    info["url"] = f"/files/{sheet_path.relative_to(ASSETS).as_posix()}"
    info["frames_dir"] = str(fdir)
    Path(out_dir / f"{name}_sheet.png.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), "utf-8")
    if export:
        info["export"] = anim_mod.export_godot(
            aligned, export, anim_name=anim_name, fps=fps, loop=True, cols=cols)
    with (ASSETS / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sheet": info["sheet"], **meta}, ensure_ascii=False) + "\n")
    return info
