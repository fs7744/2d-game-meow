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
from . import assetdb
from . import clip as clip_mod
from . import matte as matte_mod
from . import pixelfy as pxf
from . import postprocess as pp
from . import refcond
from . import repaint as rp
from . import presets
from . import quality
from . import framescore
from . import refprep
from . import sheet as sheet_mod
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
    assetdb.append_line(ASSETS / "index.jsonl", side)
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
    if info is not None:
        # 像素化**之前**的图（已抠图）。质量闸门要用它数关节：
        # 128 有效像素、32 色的像素画上，姿态模型本来就检不出几个点，
        # 拿产物评分会把好帧判成废帧，白白触发补提。
        info["pre_pixel"] = img.copy() if (pixel > 1 or palette) else img
    if pixel > 1:
        # 中位数取色：BOX 平均会把背景色按比例混进边缘块，轮廓外挂一圈灰边
        #（看着像抗锯齿）。实测并排对比，中位数的边缘更硬、色块更纯。
        img = pp.pixelate(img, pixel, grid_align=True, sample='median')
    if palette:
        img = pp.quantize_palette(img, palette)
    if clean_iso and (pixel > 1 or palette):
        # 量化之后才去孤立点：量化本身就会把相邻色合并，此时再判「孤立」才准
        img = pp.clean_isolated(img, passes=clean_iso)
    return img, edge, subjects


def _infer(eng: Engine, pipe, kw: Dict[str, Any], *, shrink: bool = True,
           label: str = "") -> Any:
    """带 OOM 恢复的推理调用：清缓存 → 降分辨率 → 重试（最多两次）。

    **为什么需要**：老实现里 OOM 只可能发生在 `pipe.to()`（`_fit` 里有降级），
    真正生成时撞顶就直接 `job.status = failed`。对"一次跑 20 个角色"的批量产线，
    这意味着跑到一半全废。这里把"降档继续跑"补上——`vramopt.plan_generation`
    本来就有降分辨率的逻辑，只是没接到这条路上。

    降档顺序：先清掉**其它**缓存管线（当前这条由局部变量持有，不会被回收），
    再按 3/4 缩边长（对齐到 8 的倍数）。缩过两次还不行就如实抛出。
    """
    for attempt in range(3):
        try:
            return pipe(**kw)
        except torch.OutOfMemoryError:
            if attempt >= 2:
                raise
            torch.cuda.empty_cache()
            eng.drop_others(())            # 清空缓存；正在用的这条仍被引用，不受影响
            w, h = kw.get("width"), kw.get("height")
            if not shrink or not w or not h:
                print(f"[warn] {label} OOM：已清空缓存后重试（不降分辨率）")
                continue
            nw = max(256, int(w * 0.75) // 8 * 8)
            nh = max(256, int(h * 0.75) // 8 * 8)
            if (nw, nh) == (w, h):
                raise
            print(f"[warn] {label} OOM：分辨率 {w}×{h} → {nw}×{nh} 后重试")
            kw = {**kw, "width": nw, "height": nh}
    return None                            # pragma: no cover  （循环内必 return/raise）


def _hires_fix(eng: Engine, img: Image.Image, *, scale: float, strength: float,
               steps: int, cfg: float, kw: Dict[str, Any], model: str,
               seed: int, lora: Optional[str], lora_scale: float,
               resample=Image.LANCZOS) -> Image.Image:
    """高清修复：把生成结果放大后**低幅度重画**一遍。

    **为什么需要**（评审 Q6）：SD1.5 的 ControlNet（openpose/canny v1.1）是 512²
    训练的，直接在 768² 上推理等于跨分辨率使用控制网，控制力下降、肢体容易糊。
    业界标准做法是**两段式**：512² 出结构（控制网在训练分辨率上最准）→ img2img
    低 strength 放大到 768/1024 补细节。

    **不是为了省显存/省时间——实测恰恰相反**（RTX 4090，8 步，strength 0.35）：

        输出 768²   sd15 512+hires  3.36GB / 2.7s   单段 768²  3.36GB / 2.0s
        输出 1152²  sdxl 768+hires 11.49GB / 5.4s   单段 1152² 11.48GB / 5.1s

    峰值由"第二段在目标尺寸上跑"决定，与单段同尺寸几乎一样；而多跑的第一段
    纯属额外开销。早先这里写过"显存峰值比单段更低"，是没量过就下的结论，已更正。

    真正站得住的理由只有一条：**控制网/底模在训练分辨率上更准**。所以它只对
    W4（ControlNet 定姿态）有意义；纯文生图用两段式只是换个构图来源，成本更高。

    另外它补的是细节，不是构图：第一段脚被裁，放大后仍然裁（实测可交付率没变）。
    精灵帧最终会被像素化抹平细节，所以这条链路基本用不上；立绘/宣传图才值得。

    `strength` 建议 0.3~0.4：再高会把结构重画掉，再低只是插值放大。
    """
    w = max(64, int(round(img.width * scale)) // 8 * 8)
    h = max(64, int(round(img.height * scale)) // 8 * 8)
    up = img.resize((w, h), resample)
    pipe = eng.img2img_pipe(model)
    eng.apply_lora(pipe, lora, lora_scale)      # 共享 UNet 时不会重复加载
    torch.cuda.reset_peak_memory_stats()
    out = _infer(eng, pipe, dict(
        image=up, strength=strength, num_inference_steps=steps,
        guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(seed),
        **kw), shrink=False, label="hires").images[0]
    return out


def _text_kwargs(pipe, prompt: str, negative: str, clip_skip: int, model: str):
    """返回 (传给 pipe 的文本参数, 实际生效的 clip_skip)。

    三条路径，顺序不能换：

    1. **短文本 + clip_skip<=1** → 原样传字符串。这条必须与历史行为逐像素一致
       （同一个 seed 的结果有测试钉着），所以不做任何"顺手优化"。
    2. **clip_skip>1** → 自算 embeds（transformers 5.x 下管道参数会崩，见 clip.py）。
    3. **超过 77 token** → 分段编码。CLIP 超长是**静默丢弃**，本项目的武侠契约
       自己就写到 113 token，等于尾部几条质量兜底词从来没生效。分段后不再丢。

    SDXL 不走 embeds：它有两个文本编码器 + pooled embeds，动一处要动一片，
    收益远小于 SD1.5，保持字符串路径。
    """
    if "XL" in type(pipe).__name__:
        if clip_skip and clip_skip > 1:
            print(f"[warn] clip_skip={clip_skip} 只对 SD1.5 生效，{model} 已忽略")
        return dict(prompt=prompt, negative_prompt=negative), 1
    if not clip_mod.available(pipe):
        return dict(prompt=prompt, negative_prompt=negative), 1
    if clip_skip and clip_skip > 1:
        pos_emb, neg_emb = clip_mod.embeds(pipe, prompt, negative, clip_skip)
        return (dict(prompt_embeds=pos_emb, negative_prompt_embeds=neg_emb),
                int(clip_skip))
    if clip_mod.needs_chunk(pipe, prompt, negative):
        pos_emb, neg_emb = clip_mod.embeds(pipe, prompt, negative, 1)
        return (dict(prompt_embeds=pos_emb, negative_prompt_embeds=neg_emb), 1)
    return dict(prompt=prompt, negative_prompt=negative), 1


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
            hires: float = 0.0, hires_strength: float = 0.35,
            hires_steps: int = 12, distill: str = "",
            refiner: bool = False, refiner_cfg: float = 6.0,
            refiner_steps: int = 20, refiner_end: float = 0.8,
            scope: str = "", tag: str = "w1") -> Dict[str, Any]:
    prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)
    # 只有真的超过分段上限（77×2）才会丢词，这时才值得告警
    ntok = presets.count(prompt)
    if ntok > presets.CHUNK_BUDGET:
        print(f"[warn] 正向提示词 {ntok} token，超过分段上限 "
              f"{presets.CHUNK_BUDGET}，尾部会被截断")
    pipe = eng.txt2img_pipe(model)
    if distill:
        eng.apply_distill(pipe, distill)
        d = eng.DISTILL_DEFAULTS.get(distill, {})
        if steps == 30:                   # 用默认步数说明用户没指定 → 换成蒸馏档位
            steps = d.get("steps", steps)
        if abs(cfg - 7.0) < 1e-6:
            cfg = d.get("cfg", cfg)
    eng.apply_lora(pipe, lora, lora_scale)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    neg = negative or NEG_DEFAULT
    gen = torch.Generator("cuda").manual_seed(seed)
    kw, clip_used = _text_kwargs(pipe, prompt, neg, clip_skip, model)
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    # **SDXL Refiner**：底模出 latent → refiner 接着精修最后一段去噪。
    # 它不是"换个更大的模型"，而是把收尾的去噪交给专攻细节的权重，
    # 所以对发丝/布料纹理的提升远大于对构图的提升。
    use_refiner = bool(refiner) and model == "sdxl" and eng.has_refiner()
    if refiner and model != "sdxl":
        print(f"[warn] Refiner 只支持 SDXL（当前 {model}），已忽略")
    if refiner and model == "sdxl" and not eng.has_refiner():
        print("[warn] 没找到 SDXL refiner 权重（models/checkpoints/sdxl-refiner），已忽略")

    call_kw: Dict[str, Any] = dict(width=width, height=height,
                                   num_inference_steps=steps,
                                   guidance_scale=cfg, generator=gen, **kw)
    # 注意：不能传 output_type="image"——新版 diffusers 已废弃它，会把输出换成
    # numpy 数组（img.size 变成整数，调用方按 PIL 用就炸）。
    out = _infer(eng, pipe, call_kw, label="w1")
    img = out.images[0]
    if use_refiner:
        # refiner 是 img2img 管线：**只接受图片**（传 latent 会被当成 PIL 图
        # 走一遍预处理，出来是糊掉的鬼影）。所以底模先解码成图再喂进去。
        rp = eng.refiner_pipe()
        # refiner 的 embeds 要自己组装（见 Engine.refiner_embeds 的说明）：
        # 直接传 prompt 会用底模那套 2816 维拼接，喂不进 refiner 的 UNet。
        pos_e, neg_e, pos_pool, neg_pool = eng.refiner_embeds(prompt, neg)
        img = _infer(eng, rp, dict(prompt_embeds=pos_e,
                                   negative_prompt_embeds=neg_e,
                                   pooled_prompt_embeds=pos_pool,
                                   negative_pooled_prompt_embeds=neg_pool,
                                   image=img,
                                   denoising_start=refiner_end,
                                   num_inference_steps=refiner_steps,
                                   guidance_scale=refiner_cfg,
                                   generator=gen),
                     label="refiner").images[0]
    seconds = dt()
    width, height = img.size          # OOM 降档后以实际尺寸为准，别在 meta 里撒谎
    if hires and hires > 1.0:
        # 两段式：先在底座分辨率出结构，再低幅度放大补细节（见 _hires_fix）
        img = _hires_fix(eng, img, scale=hires, strength=hires_strength,
                         steps=hires_steps, cfg=cfg, kw=kw, model=model,
                         seed=seed, lora=lora, lora_scale=lora_scale)
        width, height = img.size
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
        "clip_skip": clip_used,
        "subjects": subjects,
        "hires": hires if hires > 1.0 else None,
        "hires_strength": hires_strength if hires > 1.0 else None,
        "hires_steps": hires_steps if hires > 1.0 else None,
        "distill": distill or None,
        "refiner": bool(use_refiner), "refiner_cfg": refiner_cfg if use_refiner else None,
        "refiner_steps": refiner_steps if use_refiner else None,
        "refiner_end": refiner_end if use_refiner else None,
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
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)
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
    kw, clip_used = _text_kwargs(pipe, prompt, neg, clip_skip, model)
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    img = _infer(eng, pipe, dict(
        image=ref, strength=strength, num_inference_steps=steps,
        guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(seed),
        **kw), shrink=False, label="w2").images[0]
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
        "clip_skip": clip_used,
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
    raw = _infer(eng, pipe, dict(
        prompt=prompt, negative_prompt=negative or NEG_DEFAULT, image=src,
        mask_image=msk, strength=strength, num_inference_steps=steps,
        guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(seed),
        width=w8, height=h8), label="w3").images[0]
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
             alpha_mode: str = "binary", feather: int = 0,
             scope: str = "", tag: str = "wbg") -> Dict[str, Any]:
    """抠图去背景 / 换背景。**不需要模型**，纯图像处理，所以不占显存、瞬间出。

    三种输出：
      background=None + bg_color=None → 透明底前景（纯去背景，PNG 带 alpha）
      bg_color="#rrggbb"                 → 纯色背景
      background=<另一张图>              → 换背景（前景叠到那张图上）

    指定了背景时，会**额外**存一份 `<name>_cut.png`（透明底前景），
    方便回头再换别的背景。

    `alpha_mode`：像素素材用默认的 `binary`（硬边）；**高清插画资产改成 `soft`**，
    否则发丝/纱/光效会被压成一圈硬齿。`feather` 只对 soft 有意义。
    """
    src = Image.open(init_image).convert("RGBA")
    # 抠图：与其它工作流共用同一套（含 u2net 抠空自动回退的保护）
    cut = matte_mod.despeckle(
        matte_mod.matte_auto(src, matte_backend, eng.matte_dir,
                             alpha_mode=alpha_mode, feather=feather),
        denoise_iters=denoise)
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


# ---------------- W5 局部 AI 重绘 ----------------

# 约束条款：局部重绘的立命之本——没有它，模型倾向于把整张图重画一遍，
# 结果就是选区外也跟着变。每条都是踩坑换来的，但**必须在界面上可见可改**：
# 提示词不允许有用户看不见的部分。
#
# **必须写英文**：SD1.5 / SDXL 的 CLIP 只认英文。中文会被拆成乱 token，
# 而且整句约束有 100+ token，必然在 77 的上限处被截断——实测约束条款的后半段
# （"边界处不要出现接缝…"）直接被砍掉，等于约束根本没生效。
#
# **而且必须够短**：77 是「用户提示词 + 契约后缀 + 约束」三者共享的上限。
# 契约后缀（pixel 契约 26 token）已经吃掉三分之一，用户提示词通常也有 8~15，
# 约束只剩 30 token 左右。这句 23 token，留了余量。约束的措辞经过取舍：
# 保留三件最要紧的——只改选区、风格要延续、不要接缝；
# 砍掉的是那些「模型本来也不会做的」（加物体、改构图——在 img2img 里
# 选区外本来就不动，约束是给纯文生图用的）。
AI_REPAINT_CONSTRAINT = (
    "local inpaint, repaint only the masked area, keep the rest identical, "
    "match style and lighting, no seams"
)

# 负面词只放「重绘最容易出的四种错」，不放画风词（画风由原图自己决定）
AI_REPAINT_NEGATIVE = (
    "seam, edge highlight, color banding, visible seam, blurry, ghosting, "
    "duplicate limbs, extra characters, text, watermark, signature"
)


def ai_repaint(eng: Engine, *, image: str | Path, mask: str | Path, prompt: str,
               negative: Optional[str] = None, strength: float = 0.9,
               steps: int = 25, cfg: float = 7.0, seed: int = -1,
               model: str = "sd15", lora: Optional[str] = None,
               lora_scale: float = 0.8, clip_skip: int = 1,
               preset: Optional[str] = None,
               context: float = 0.35, expand: int = 8, feather: int = 4,
               match_color: bool = True, match_strength: float = 0.45,
               palette_snap: bool = False, palette_colors: int = 32,
               scope: str = "", tag: str = "w5") -> Dict[str, Any]:
    """局部 AI 重绘：裁剪「选区+上下文」→ 交给通用模型重画 → 贴回原图。

    **strength 的实测含义**（像素素材，768² crop，25 步）：
      0.80 → 选区几乎没变（模型保留原样，重画等于没做）
      0.90 → 改动开始明显
      0.95 → 改动最明显，但硬边会留矩形拼接痕（要配合羽化）
      1.00 → 整块重画，风格漂移最大
    所以默认 0.90：够改得动，又不至于把选区整块重画。

    **和 W3（专用 inpaint 权重）的区别**：W3 走 SD1.5 的 inpaint 微调权重，
    只有 512 分辨率、也不能挂 LoRA；本工作流把选区裁出来放大，交给**任意底模**
    （含像素 LoRA、SDXL）重画，再贴回去。所以：
      - 能用 LoRA（像素素材必须用像素 LoRA 才像像素画）
      - 小区域也能拿到足够的像素（放大到 512 再画）
      - 选区外的像素**逐点不变**（羽化只向内 + 二值蒙版相乘）

    代价是接缝需要收：羽化过渡 + 只在边界带做色彩校正 + 可选色板吸附。
    """
    if preset is not None:
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)
    src = Image.open(image).convert("RGBA")
    msk = rp.load_mask(mask)
    if msk.size != src.size:
        # 尺寸不一致一定是换了源图但没重涂蒙版。缩放蒙版在语义上是错的
        # （涂的地方和要改的地方对不上），所以明确报错而不是硬缩。
        raise ValueError(
            f"蒙版尺寸 {msk.size[0]}×{msk.size[1]} 与源图 "
            f"{src.width}×{src.height} 不一致：换图后要重新涂抹")

    # 提示词：用户正文 + 程序加的约束条款（都记进 meta，界面上可看）
    body = (prompt or "").strip()
    if not body:
        raise ValueError("提示词是空的：先写一句要重绘成什么")
    full_prompt = f"{body}, {AI_REPAINT_CONSTRAINT}"
    neg = (negative or "").strip() or AI_REPAINT_NEGATIVE

    # 1) 裁出「选区 + 上下文」
    crop, crop_mask, box = rp.crop_region(src, msk, expand=expand, context=context)
    target = rp.fit_size(crop.size)

    # 2) 交给通用模型重画。用图生图而不是文生图：上下文已经在裁剪图里了，
    #    模型要做的只是「在既有内容上改」，不是凭空重画。
    pipe = eng.img2img_pipe(model)
    eng.apply_lora(pipe, lora, lora_scale)
    # **必须转 RGB**：RGBA 进 VaeImageProcessor 会变成 4 通道，UNet 首层按 4 通道
    # 卷积 → 激活尺寸爆炸（实测 512² 直接 OOM，报 Tried to allocate 36.00 GiB）。
    # alpha 的信息在贴回阶段用（composite 会混合 alpha），送模型不需要它。
    #
    # 但不能直接 convert("RGB")：那样透明区会变黑，模型会把背景也当内容重画。
    # 用 flatten_rgb 把透明区填白（精灵素材大多是透明底）。
    payload = rp.flatten_rgb(crop)
    if payload.size != target:
        payload = payload.resize(target, Image.LANCZOS)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    kw, clip_used = _text_kwargs(pipe, full_prompt, neg, clip_skip, model)
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    gen = _infer(eng, pipe, dict(
        image=payload, strength=strength, num_inference_steps=steps,
        guidance_scale=cfg,
        generator=torch.Generator("cuda").manual_seed(seed), **kw),
        shrink=False, label="w5").images[0]
    seconds = dt()
    if gen.size != crop.size:
        gen = gen.resize(crop.size, Image.LANCZOS)

    # 3) 贴回：羽化只向内 -> 选区外逐点不变
    merged = rp.composite(crop, gen, crop_mask, feather_px=feather,
                           match_color=match_color,
                           match_strength=match_strength)

    # 4) 色板吸附（像素素材专用）：把重绘区里「接近源色」的杂色吸回源图色板。
    #    差得远的新颜色保留 —— 否则「要把这块改成红色」会被吸成灰。
    palette_snapped = 0.0
    if palette_snap:
        merged, pal, palette_snapped = rp.snap_palette(
            merged, src, crop_mask, colors=palette_colors)

    out = src.copy()
    out.paste(merged, box)

    warnings: List[str] = []
    min_side = min(box[2] - box[0], box[3] - box[1])
    if feather > max(2, min_side // 4):
        warnings.append(
            f"羽化 {feather}px 相对选区偏大：改动会溢出选区边缘一圈，"
            "想要干净边界就调小")
    if palette_snap and palette_snapped < 0.02:
        warnings.append("色板对齐几乎没有生效：新颜色离源图色板都太远")

    return _save(out, _stamp(tag), "ai_repaint", {
        "workflow": "w5", "scope": scope or "default",
        "prompt": full_prompt, "prompt_body": body,
        "constraint": AI_REPAINT_CONSTRAINT,
        "negative": neg,
        "image": str(image), "mask": str(mask),
        "strength": strength, "steps": steps, "cfg": cfg, "seed": seed,
        "model": model, "lora": lora,
        "lora_scale": lora_scale if lora else None,
        "clip_skip": clip_used,
        "box": list(box), "crop_size": list(crop.size), "target_size": list(target),
        "coverage": rp.mask_coverage(msk),
        "context": context, "expand": expand, "feather": feather,
        "match_color": bool(match_color), "match_strength": match_strength,
        "palette_snap": bool(palette_snap), "palette_snapped": palette_snapped,
        "warnings": warnings,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, scope=scope)


# ---------------- WPX 完美像素转换 ----------------

def pixelify(eng: Engine, *, init_image: str | Path,
             target: Optional[Sequence[int]] = None,
             grid_size: Optional[Sequence[int]] = None,
             grid_snap: bool = True,
             sample_method: str = "median",
             colors: int = 32, transparent_bg: bool = False,
             snap_to: Optional[int] = 16,
             scope: str = "", tag: str = "wpx") -> Dict[str, Any]:
    """完美像素转换：任意图 → **像素完美**的像素画。**不跑模型**，纯确定性处理。

    和 W1 里那个「像素化」的区别：那个是「BOX 降采样 + 最近邻放大 + 中位切分
    量化」三板斧，网格不对齐（AI 图的像素块常在 37.3px 这种非整格上），
    边界会被切在色块中间混成脏斑。这个走完整的**网格检测 → 吸附 → 每格单色**
    流程，输出的是「每格一色、边界对齐」的真正像素画。

    参数：
      target        目标原生像素尺寸（如 128×128）；不给就用检测到的网格
      grid_size     手动指定网格（跳过检测）；动画序列后续帧用它锁定首帧的网格
      grid_snap     关掉就退化成普通降采样
      sample_method 每格取色：median（默认，最稳）/ majority / center
      colors        共享色板色数；0 = 不量化（只做网格对齐）
      transparent_bg 把纯色背景转成透明
      snap_to       网格边长吸附到这个数的整数倍（16 = 精灵常用尺寸）；
                    None = 保留检测到的原始值

    输出是**原生像素尺寸**（如 128×128），不预放大——放大交给渲染端
    （前端用 image-rendering: pixelated），这样以后想换 4 倍/16 倍不用重算。
    """
    src = Image.open(init_image).convert("RGBA")
    res = pxf.refine_batch([src], grid_snap=grid_snap,
                           grid_size=grid_size, target_size=target,
                           sample_method=sample_method, colors=colors,
                           transparent_bg=transparent_bg, snap_to=snap_to)
    item = res["items"][0]
    out = item["image"]
    meta = item["meta"]
    if colors:
        meta["palette"] = [list(c) for c in res["palette"]]
    meta.update({
        "workflow": "wpx", "scope": scope or "default",
        "init_image": str(init_image),
        "target": list(target) if target else None,
        "grid_size": list(grid_size) if grid_size else None,
        "grid_snap": bool(grid_snap), "sample_method": sample_method,
        "colors_requested": int(colors),
        "transparent_bg": bool(transparent_bg),
        "snap_to": snap_to,
    })
    return _save(out, _stamp(tag), "pixelify", meta, scope=scope)


# ---------------- WPC 角色迁移 ----------------

def pose_clone(eng: Engine, *, pose_image: str | Path, char_image: str | Path,
                prompt: str, negative: Optional[str] = None,
                width: int = 768, height: int = 768,
                steps: int = 25, cfg: float = 7.0, seed: int = -1,
                cn_scale: float = 1.0, ipa_scale: float = 0.8,
                fill: float = 0.8, thr: float = 0.25,
                ref_mode: str = "auto",
                lora: Optional[str] = None, lora_scale: float = 0.8,
                clip_skip: int = 1, preset: Optional[str] = None,
                model: str = "sd15",
                pixel: int = 0, palette: int = 0, matte: bool = False,
                matte_backend: str = "auto", denoise: int = 1, clean_iso: int = 1,
                scope: str = "", tag: str = "wpc") -> Dict[str, Any]:
    """角色迁移：把**动作参考图**的姿态套到**角色图**上，生成角色做出该动作的图。

    三件套各管一件事：
      - **ControlNet 条件图** ← 动作参考图：定姿态（硬约束）
      - **IP-Adapter**        ← 角色图：定身份/外观（软约束）
      - **LoRA**              ← 定画风（可选）

    **参考模式**（ref_mode）决定「动作参考图」怎么变成条件图：
      - ``pose``    真人照片 → 人体检测 → RTMPose → OpenPose 骨架。
                    姿态语义最强：只保留关节位置，参考角色的衣服体型全丢掉。
      - ``outline`` 任意素材 → 抠主体 → 剪影边缘 → canny 条件图。
                    **游戏素材走这条路**：像素角色、怪物、机甲、非人形生物
                    都能当动作参考（真人检测器认不出它们，实测 0 个框）。
      - ``auto``    先试 pose，检测不到人体就自动降级到 outline（默认）。

    **为什么骨架要归一化**：ControlNet OpenPose 是空间敏感的——条件图里人物站在
    哪、占多大，生成的人就站在哪、占多大。参考图里的人物可能缩在角落，直接用
    原始骨架会画出「角落小人」。所以先按检测到的人物框把关键点归一化（人物居中、
    占画面 fill 比例），再渲染骨架图。

    **为什么不直接用 W4**：W4 是「骨架序列 → 精灵帧」，没有角色参考图；这里
    多了 IP-Adapter 这一路，是单帧的身份迁移。多帧批量请用 WSS（sprite_swap）。
    """
    if preset is not None:
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)

    ref = Image.open(pose_image).convert("RGB")
    # 1) 动作参考图 → 条件图（骨架或轮廓）
    extractor = eng.pose_extractor()
    cond, used_mode, cond_info = refcond.build_condition(
        ref, mode=ref_mode, size=max(width, height), extractor=extractor,
        thr=thr, fill=fill, matte_dir=eng.matte_dir)
    if used_mode == "pose":
        pipe_cn = "openpose"
    else:
        pipe_cn = "canny"

    # 2) 角色图 → IP-Adapter 参考
    # **必须先做参考图预处理**（refprep.prepare_reference）：
    #   - 透明区填白而不是黑（直接 convert("RGB") 会让透明区变黑，
    #     实测 66.5% 的角色图是纯黑背景，IP-Adapter 会把黑色当成角色的
    #     暗部，身份特征被污染）
    #   - 裁到主体包围盒 + 居中留白（CLIP Vision 在 224² 上切 14×14 patch，
    #     主体顶满画面时 patch 里只有局部纹理，编码不出整体身份）
    #   - 像素画用最近邻缩放（LANCZOS 会把像素块糊成渐变）
    char = refprep.prepare_reference(Image.open(char_image).convert("RGBA"), size=width)
    eng.ensure_headroom(need_gb=2.0, keep=("pose",))
    pipe = eng.pose_pipe(controlnet=pipe_cn, with_ipadapter=True, model=model)
    eng.apply_lora(pipe, lora, lora_scale)
    pipe.set_ip_adapter_scale(ipa_scale)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    neg = negative or NEG_DEFAULT
    kw, clip_used = _text_kwargs(pipe, prompt, neg, clip_skip, model)
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    out = _infer(eng, pipe, dict(
        image=cond.resize((width, height)), num_inference_steps=steps,
        guidance_scale=cfg, controlnet_conditioning_scale=cn_scale,
        ip_adapter_image=char,
        generator=torch.Generator("cuda").manual_seed(seed),
        width=width, height=height, **kw), label="wpc").images[0]
    seconds = dt()
    width, height = out.size
    raw = out.copy() if (matte or pixel > 1 or palette) else None
    matte_info: Dict[str, Any] = {}
    out, edge, subjects = _postprocess(
        out, eng, matte=matte, matte_backend=matte_backend, denoise=denoise,
        pixel=pixel, palette=palette, clean_iso=clean_iso, info=matte_info)
    return _save(out, _stamp(tag), "pose_clone", {
        "workflow": "wpc", "scope": scope or "default",
        "prompt": prompt, "negative": neg,
        "pose_image": str(pose_image), "char_image": str(char_image),
        "ref_mode": ref_mode, "ref_mode_used": used_mode,
        "ref_fallback_from": cond_info.get("fallback_from"),
        "ref_fallback_reason": cond_info.get("fallback_reason"),
        "joints": cond_info.get("joints"), "fill": fill,
        "width": width, "height": height,
        "steps": steps, "cfg": cfg, "seed": seed,
        "model": model,
        "cn_scale": cn_scale, "ipa_scale": ipa_scale,
        "lora": lora, "lora_scale": lora_scale if lora else None,
        "clip_skip": clip_used,
        "pixel_block": pixel, "palette": palette, "matte": matte,
        "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "edge_touch": edge, "subjects": subjects,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


# ---------------- WSS 序列帧角色替换 ----------------

def sprite_swap(eng: Engine, *, sheet_image: str | Path, char_image: str | Path,
                prompt: str = "", negative: Optional[str] = None,
                cols: int = 0, rows: int = 0,
                width: int = 768, height: int = 768,
                steps: int = 25, cfg: float = 7.0, seed: int = -1,
                cn_scale: float = 1.0, ipa_scale: float = 0.8,
                ref_mode: str = "outline", mode: str = "generate",
                engine_mode: str = "auto",
                lora: Optional[str] = None, lora_scale: float = 0.8,
                clip_skip: int = 1, preset: Optional[str] = None,
                model: str = "sd15",
                matte: bool = True, matte_backend: str = "auto",
                denoise: int = 1, clean_iso: int = 1,
                palette: int = 32, keep_alpha: bool = False,
                candidates: int = 2, gate_retries: int = 1,
                gate_seed_step: int = 1000,
                vram_profile: str = "auto",
                scope: str = "", tag: str = "wss") -> Dict[str, Any]:
    """序列帧角色替换：把一段游戏动画里的角色换成目标角色，逐帧对齐后重新打包。

    这是用户最常见的产线场景：手上有一张**游戏动画序列帧**（精灵图），
    想把里面的角色换成自己的角色，但动作必须一模一样。

    流程：
      1. 切帧：自动检测精灵图网格（或用 cols/rows 指定），逐格切出
      2. 每帧生成：轮廓条件（动作）+ IP-Adapter（身份）→ 新角色帧
      3. 对齐：按脚底锚点把每帧拉回同一落脚点，抑制模型自构图带来的漂移
      4. 缩回原格子尺寸 + 共享色板量化（整段动画共用一套颜色）
      5. 按原网格重新打包成精灵图

    **为什么输出要缩回原格子尺寸**：序列帧是要塞回游戏里的，尺寸必须
    和原帧一致（通常 32~128px）。在 768² 生成后缩回去，比在小尺寸上
    生成靠谱得多——SD1.5 在 64² 上出图是糊的。

    **共享色板**：逐帧各自量化会导致相邻帧颜色打架（第一帧的主色占满色位）。
    这里整段动画共用一块色板，见 pixelfy.refine_batch。

    **keep_alpha**：直接复用参考帧的 alpha（形状逐像素精确、帧间零漂移），
    只换 RGB。适合「同体型换配色」的换皮；代价是参考角色的体型特征
    （帽子、翅膀、武器）也会被保留。

    **mode**：
      - ``generate``（默认）跑模型：轮廓条件（动作）+ IP-Adapter（身份），
        换成**体型不同**的角色。
      - ``recolor``  不跑模型：纯确定性的明度重映射换皮，**帧间零漂移**、
        毫秒级，保留原角色的一切体型特征只换配色。同体型换色首选。

    **engine_mode** 决定用什么引擎生成：
      - ``animatediff``（默认，若权重可用）：一次生成整段序列，靠 motion module
        的**跨帧时序注意力**保证一致性。实测比逐帧又好又快。
      - ``frame``：逐帧生成 + 多候选择优。不依赖 motion adapter 权重，
        显存更省（4.5GB vs 13.1GB），但帧间一致性略差。
    """
    if preset is not None:
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)

    sheet = Image.open(sheet_image).convert("RGBA")
    if cols <= 0 or rows <= 0:
        detected = sheet_mod.detect_grid(sheet)
        if detected is None:
            raise ValueError(
                "认不出精灵图的网格：请手动指定列数和行数"
                "（列数 = 横向排了几帧，行数 = 纵向排了几帧）")
        cols, rows = detected
    frames = sheet_mod.split_sheet(sheet, cols, rows)
    if not frames:
        raise ValueError("精灵图里没有有效帧（整张是空的？）")

    if mode == "recolor":
        return _sprite_recolor(frames, char_image, cols, rows, palette=palette,
                               scope=scope, tag=tag, prompt=prompt)

    # 引擎选择：AnimateDiff（时序注意力）还是逐帧（多候选）
    if engine_mode == "auto":
        engine_mode = "animatediff" if eng.animatediff_available() else "frame"
    if engine_mode == "animatediff":
        if not eng.animatediff_available():
            raise ValueError(
                "AnimateDiff 的 motion adapter 权重缺失（约 1.7GB）。\n"
                "下载：python tools/fetch_animatediff.py\n"
                "或改用逐帧模式（engine_mode='frame'，显存更省但帧间一致性略差）")

    extractor = eng.pose_extractor()
    pipe_cn = "openpose" if ref_mode == "pose" else "canny"
    if engine_mode == "animatediff":
        return _sprite_animatediff(
            eng, frames, char_image, cols, rows,
            # **不要在这里再传 preset**：prompt/negative 上面已经 build 过一次，
            # 再 build 一遍会把契约后缀拼成两遍，白吃 15~20 个 token 的预算。
            prompt=prompt, negative=negative, preset=None,
            pipe_cn=pipe_cn, ref_mode=ref_mode, width=width, height=height,
            steps=steps, cfg=cfg, seed=seed, cn_scale=cn_scale, ipa_scale=ipa_scale,
            lora=lora, lora_scale=lora_scale, clip_skip=clip_skip,
            matte=matte, matte_backend=matte_backend, denoise=denoise,
            palette=palette, extractor=extractor, vram_profile=vram_profile,
            model=model, scope=scope, tag=tag)

    eng.ensure_headroom(need_gb=2.0, keep=("pose",))
    pipe = eng.pose_pipe(controlnet=pipe_cn, with_ipadapter=True, model=model)
    eng.apply_lora(pipe, lora, lora_scale)
    pipe.set_ip_adapter_scale(ipa_scale)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    neg = negative or NEG_DEFAULT
    kw, clip_used = _text_kwargs(pipe, prompt, neg, clip_skip, model)

    char = refprep.prepare_reference(Image.open(char_image).convert("RGBA"), size=width)
    char_ref_orig = Image.open(char_image).convert("RGBA")
    torch.cuda.reset_peak_memory_stats()
    total = _timer()

    generated: List[Image.Image] = []
    per_frame: List[float] = []
    used_modes: List[str] = []
    frame_scores: List[Dict[str, Any]] = []
    attempts_used = 0
    matte_info: Dict[str, Any] = {}

    def _cut(img: Image.Image) -> Image.Image:
        """抠图（候选评分前就要有 alpha 才能量剪影）。"""
        if not matte:
            return img.convert("RGBA")
        cut_img = matte_mod.matte_auto(img.convert("RGB"), matte_backend,
                                       eng.matte_dir, info=matte_info)
        return matte_mod.despeckle(cut_img, denoise_iters=denoise)

    # 显存守卫：整段序列共用一个候选数，所以在循环外算一次
    cand_note = ""
    if candidates > 1:
        from . import vramopt
        k, cand_note = vramopt.plan_candidates(eng.arch(model), max(width, height),
                                               candidates, eng.free_gb())
        if k == 0:
            raise ValueError(cand_note)
        if k < candidates:
            print(f"[warn] {cand_note}")
            candidates = k

    for _c, _r, frame in frames:
        cond, used_mode, _info = refcond.build_condition(
            frame, mode=ref_mode, size=max(width, height), extractor=extractor,
            matte_dir=eng.matte_dir)
        used_modes.append(used_mode)
        cond = cond.resize((width, height))

        # **多候选择优**（业界标准做法）：同 seed 也会漂 20~30%，
        # 所以每帧生成几个候选，按「动作贴合度 + 身份贴合度」挑最好的。
        # 挑不出好的就用补提额度再试几轮。
        dt = _timer()
        cands: List[Tuple[Image.Image, Dict[str, Any]]] = []
        prev = generated[-1] if generated else None
        ref_area = int(framescore.subject_mask(frame.convert("RGBA")).sum())
        want = max(1, candidates)
        for k in range(want):
            img = _infer(eng, pipe, dict(
                image=cond, num_inference_steps=steps, guidance_scale=cfg,
                controlnet_conditioning_scale=cn_scale,
                ip_adapter_image=char,
                generator=torch.Generator("cuda").manual_seed(seed + k * gate_seed_step),
                width=width, height=height, **kw), label="wss").images[0]
            cut_img = _cut(img)
            ev = framescore.evaluate(cut_img, frame, char_ref_orig, prev=prev,
                                    area_ref=ref_area)
            cands.append((cut_img, ev))
            if ev["passed"]:
                break                      # 已经够好，不必再提
        # 都没过闸门 → 用补提额度再试几轮
        if not any(ev["passed"] for _i, ev in cands) and gate_retries > 0:
            for k in range(gate_retries):
                img = _infer(eng, pipe, dict(
                    image=cond, num_inference_steps=steps, guidance_scale=cfg,
                    controlnet_conditioning_scale=cn_scale,
                    ip_adapter_image=char,
                    generator=torch.Generator("cuda").manual_seed(
                        seed + (len(cands) + k) * gate_seed_step),
                    width=width, height=height, **kw), label="wss").images[0]
                cut_img = _cut(img)
                ev = framescore.evaluate(cut_img, frame, char_ref_orig, prev=prev,
                                        area_ref=ref_area)
                cands.append((cut_img, ev))
                if ev["passed"]:
                    break
        best_img, best_ev, best_i = framescore.pick_best(cands)
        frame_scores.append({**best_ev, "attempts": len(cands), "picked": best_i})
        attempts_used += len(cands)
        per_frame.append(dt())
        generated.append(best_img)

    cut: List[Image.Image] = list(generated)

    if keep_alpha:
        # 用参考帧的 alpha 覆盖：形状逐像素精确，帧间零漂移
        for i, (_c, _r, frame) in enumerate(frames):
            cut[i].putalpha(frame.getchannel("A"))

    # 缩回原格子尺寸：序列帧要塞回游戏，尺寸必须和原帧一致。
    #
    # **必须从生成尺寸直接一步缩到位**：BOX 降采样会把每个像素格里的
    # 多种颜色**混**在一起，边界就散了——像素画最忌讳这个。
    # 正确做法是「按格采样，每格取一个代表色」。
    # 参考帧本身是像素画（每格一色），所以格数是已知的：
    #   生成图 768² → 原格 64²，比例 12:1，每 12×12 像素对应原图 1 格。
    cell_w = max(f.width for _c, _r, f in frames)
    cell_h = max(f.height for _c, _r, f in frames)
    # 整段动画共享一套色板（逐帧各自量化会让相邻帧颜色打架）
    refined = pxf.refine_batch(
        [c.convert("RGB") for c in cut],
        grid_snap=False,
        target_size=(cell_w, cell_h),
        colors=palette,
    )
    outs: List[Image.Image] = []
    for item, src in zip(refined["items"], cut):
        im = item["image"]
        if src.mode == "RGBA":
            im = im.convert("RGBA")
            # alpha 跟着缩到格子尺寸（src 还是生成时的 768²）
            a = src.getchannel("A").resize((cell_w, cell_h), Image.BILINEAR)
            im.putalpha(a)
            im = pxf.binarize_alpha(im)
        outs.append(im)

    # 帧间对齐：按脚底锚点拉回同一落脚点
    aligned = pp.align_sequence(outs)

    # 按原网格重新打包
    meta = {
        "workflow": "wss", "scope": scope or "default",
        "mode": "generate", "engine_mode": "frame",
        "prompt": prompt, "negative": neg,
        "sheet_image": str(sheet_image), "char_image": str(char_image),
        "grid": [cols, rows], "frames": len(aligned),
        "cell": [cell_w, cell_h],
        "ref_mode": ref_mode,
        "ref_modes_used": sorted(set(used_modes)),
        "model": model,
        "width": width, "height": height,
        "steps": steps, "cfg": cfg, "seed": seed,
        "cn_scale": cn_scale, "ipa_scale": ipa_scale,
        "lora": lora, "lora_scale": lora_scale if lora else None,
        "clip_skip": clip_used,
        "matte": matte, "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "keep_alpha": keep_alpha,
        "palette": palette,
        "palette_size": len(refined["palette"]) if palette else 0,
        # 逐帧评分（多候选择优的依据）
        "candidates": max(1, candidates),
        "candidates_note": cand_note or None,
        "gate_retries": gate_retries,
        "attempts": attempts_used,
        "frame_scores": frame_scores,
        "gate_passed": sum(1 for s in frame_scores if s["passed"]),
        # 整段序列的一致性（产线据此判断要不要重跑）
        **framescore.sequence_consistency(outs),
        "per_frame_seconds": [round(s, 2) for s in per_frame],
        "seconds_total": round(sum(per_frame), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    return _pack_sprite_frames(aligned, frames, cols, rows, cell_w, cell_h,
                              meta=meta, scope=scope, tag=tag)


def _pack_sprite_frames(aligned: Sequence[Image.Image], frames: Sequence[tuple],
                         cols: int, rows: int, cell_w: int, cell_h: int,
                         meta: Dict[str, Any], scope: str = "", tag: str = "wss"
                         ) -> Dict[str, Any]:
    """把帧落盘 + 按原网格打包成精灵图 + 写 sidecar 与索引。

    AnimateDiff 与逐帧两条路径共用——产出结构必须完全一致，
    否则下游（引擎导出、帧序列目录）会分叉。
    """
    name = _stamp(tag)
    out_dir = scope_dir(scope)
    fdir = out_dir / f"{name}_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(aligned):
        f.save(fdir / f"frame_{i:04d}.png")

    sheet_path = out_dir / f"{name}_sheet.png"
    grid = Image.new("RGBA", (cols * cell_w, rows * cell_h), (0, 0, 0, 0))
    for i, (_c, _r, _f) in enumerate(frames):
        if i >= len(aligned):
            break
        grid.paste(aligned[i], (_c * cell_w, _r * cell_h))
    grid.save(sheet_path)

    Path(str(sheet_path) + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    assetdb.append_line(ASSETS / "index.jsonl", {"sheet": str(sheet_path), **meta})
    return {
        "file": str(sheet_path),
        "url": f"/files/{sheet_path.relative_to(ASSETS).as_posix()}",
        "frames_dir": str(fdir),
        "sheet": str(sheet_path),
        "meta": meta,
    }


def _sprite_recolor(frames: Sequence[tuple], char_image: str | Path,
                    cols: int, rows: int, palette: int = 32,
                    scope: str = "", tag: str = "wss",
                    prompt: str = "") -> Dict[str, Any]:
    """确定性换皮：不跑模型，把序列帧的配色换成角色图的配色。

    逐帧按明度重映射到角色色板，alpha 原样保留 —— 于是
    **形状逐像素精确、帧间零漂移**，这是 AI 路线做不到的。

    适用：同体型换配色（把敌人换成我方配色、把骑士换成法师配色）。
    不适用：换体型（那要用 generate 模式）。
    """
    char = Image.open(char_image).convert("RGBA")
    pal = pxf.extract_palette([char], palette or 16)

    outs: List[Image.Image] = []
    for _c, _r, frame in frames:
        outs.append(pxf.recolor_by_luma(frame, pal, keep_alpha=True))

    cell_w = max(f.width for _c, _r, f in frames)
    cell_h = max(f.height for _c, _r, f in frames)
    aligned = pp.align_sequence(outs)

    meta = {
        "workflow": "wss", "scope": scope or "default",
        "mode": "recolor", "prompt": prompt,
        "char_image": str(char_image),
        "grid": [cols, rows], "frames": len(aligned), "cell": [cell_w, cell_h],
        "palette": palette, "palette_size": len(pal),
        "palette_colors": [list(c) for c in pal],
        "seconds_total": 0.0,
        "deterministic": True,
    }
    return _pack_sprite_frames(aligned, frames, cols, rows, cell_w, cell_h,
                              meta=meta, scope=scope, tag=tag)


def _sprite_animatediff(eng: Engine, frames: Sequence[tuple], char_image: str | Path,
                        cols: int, rows: int, *, prompt: str,
                        negative: Optional[str] = None, preset: Optional[str] = None,
                        pipe_cn: str = "canny", ref_mode: str = "outline",
                        width: int = 768, height: int = 768, steps: int = 25,
                        cfg: float = 7.0, seed: int = -1, cn_scale: float = 1.0,
                        ipa_scale: float = 0.8, lora: Optional[str] = None,
                        lora_scale: float = 0.8, clip_skip: int = 1,
                        matte: bool = True, matte_backend: str = "auto",
                        denoise: int = 1, palette: int = 32, extractor: Any = None,
                        repair: bool = True, repair_candidates: int = 2,
                        vram_profile: str = "auto", model: str = "sd15",
                        scope: str = "", tag: str = "wss") -> Dict[str, Any]:
    """AnimateDiff 路线：一次生成整段序列，靠时序注意力保证帧间一致。

    **为什么这条更好**（本项目同素材实测，8 帧 512²）：

        逐帧多候选    19.2s / 4.5GB / silhouette_drift 0.152 / 逐帧形状会漂
        AnimateDiff    7.4s / 13.1GB / silhouette_drift **0.097** / 8 帧完全一致

    原因：AnimateDiff 的 motion module 插在 UNet 的 ResNet/Attention 之后，
    提供**跨帧时序注意力**——生成第 N 帧时会参考相邻帧的潜在表示。
    这是架构级的约束，而「固定 seed」只是概率层面的。

    **显存**：motion module 要额外一份 UNet 激活，基线 8 帧 512² 是 9.86GB。
    引擎层开了两项优化（VAE 逐帧解码 + IP-Adapter encoder 用完即退）后
    降到 **5.78GB**，16 帧是 **7.10GB**（实测，详见 workflows/vramopt.py）。

    **自适应降级**：显存不够时按 `vram_profile` 降分辨率（帧数是用户给的
    动画内容，不能砍）。降分辨率是最后手段——SD1.5 在低分辨率出图偏糊，
    所以 meta 里会记录实际用的分辨率和降级原因。
    """
    if preset is not None:
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)
    neg = negative or NEG_DEFAULT
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    n = len(frames)

    # **显存规划**：装不下就降分辨率（帧数不能砍——那是用户的动画内容）。
    from . import vramopt
    n, size, vram_note = vramopt.plan_generation(n, max(width, height),
                                                  profile=vram_profile)
    width = height = size

    # AnimateDiff 是显存大户（8 帧 5.8GB、32 帧 14.2GB）：先清场再加载，
    # 别让闲置的文生图管线占着 2.6GB。
    eng.drop_others(("animatediff",))
    eng.ensure_headroom(need_gb=6.0, keep=("animatediff",))
    pipe = eng.animatediff_pipe(controlnet=pipe_cn, with_ipadapter=True, model=model)
    eng.apply_lora(pipe, lora, lora_scale)
    pipe.set_ip_adapter_scale(ipa_scale)

    # 条件帧序列：每一帧的轮廓条件
    conds: List[Image.Image] = []
    used_modes: List[str] = []
    for _c, _r, frame in frames:
        cond, used_mode, _info = refcond.build_condition(
            frame, mode=ref_mode, size=max(width, height), extractor=extractor,
            matte_dir=eng.matte_dir)
        used_modes.append(used_mode)
        conds.append(cond.resize((width, height)))

    char = refprep.prepare_reference(Image.open(char_image).convert("RGBA"), size=width)
    char_ref_orig = Image.open(char_image).convert("RGBA")

    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    text_kw, clip_used = _text_kwargs(pipe, prompt, neg, clip_skip, model)
    out = _infer(eng, pipe, dict(
        conditioning_frames=conds,
        controlnet_conditioning_scale=cn_scale,
        ip_adapter_image=char,
        num_frames=n,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=torch.Generator("cuda").manual_seed(seed),
        height=height, width=width,
        **text_kw), shrink=False, label="wss-animatediff")
    seconds = dt()
    raw_frames = [f.convert("RGB") for f in out.frames[0]]

    # 后处理与逐帧路线完全一致：抠图 → 按格采样 → 共享色板
    matte_info: Dict[str, Any] = {}
    cut: List[Image.Image] = []
    for g in raw_frames:
        if matte:
            c = matte_mod.matte_auto(g, matte_backend, eng.matte_dir, info=matte_info)
            cut.append(matte_mod.despeckle(c, denoise_iters=denoise))
        else:
            cut.append(g.convert("RGBA"))

    cell_w = max(f.width for _c, _r, f in frames)
    cell_h = max(f.height for _c, _r, f in frames)

    # **残缺帧补提**：AnimateDiff 的时序注意力是双刃剑——它让整段一致，
    # 但某一帧的轮廓条件弱时，模型会画残缺，而时序注意力还会把错误
    # 传播到邻帧。实测 8 帧里第 2 帧主体只剩 128px（正常 ~675px）。
    #
    # 补救：逐帧量出不合格的帧，用**逐帧方案**单独重生成那一帧顶上。
    # 混合策略——好的帧保留 AnimateDiff 的时序一致性，坏的帧单独修。
    # 参考面积按**最终格子尺寸**算（64²），不是生成尺寸（512²）。
    # 判定的量必须和最终产物同尺度——512² 下算出的比值与 64² 下的不一样，
    # 会漏判残缺帧（实测帧 6 在 64² 下 area_ratio=0.57 明显残缺，
    # 在 512² 下却算成 0.97 看着正常）。
    ref_areas = [int(framescore.subject_mask(frames[i][2].convert("RGBA")).sum())
                 for i in range(n)]
    # 缩到格子尺寸后再判定，量纲与最终产物一致
    _cw = max(f.width for _c, _r, f in frames)
    _ch = max(f.height for _c, _r, f in frames)
    _small = [c.convert("RGB").resize((_cw, _ch), Image.BOX) for c in cut]
    _small_rgba = []
    for c, sm in zip(cut, _small):
        im = sm.convert("RGBA")
        im.putalpha(c.getchannel("A").resize((_cw, _ch), Image.BILINEAR))
        _small_rgba.append(pxf.binarize_alpha(im))
    scores = [framescore.evaluate(_small_rgba[i], frames[i][2], char_ref_orig,
                                  area_ref=ref_areas[i])
              for i in range(n)]
    bad = [i for i in range(n) if not scores[i]["passed"]]
    # 补提产出要同时有「原尺寸版」（供后处理）与「格子尺寸版」（供评分）
    repaired_full: Dict[int, Image.Image] = {}
    repaired: List[int] = []
    if bad and repair:
        # **直接用 AnimateDiff 管线跑单帧**（num_frames=1）：零额外显存。
        # 单独 pose_pipe() 会再加载一整套 UNet/VAE/ControlNet/IP-Adapter
        # （实测多占 3.23GB），而两条管线的权重完全重复。
        # AnimateDiff 管线 num_frames=1 时 motion module 的时序注意力
        # 退化成恒等映射，效果就是普通单帧生成。
        fpipe = pipe
        for i in bad:
            cand: List[Tuple[Image.Image, Dict[str, Any], Image.Image]] = []
            for k in range(max(1, repair_candidates)):
                # **必须跑 2 帧，不能跑 1 帧**：AnimateDiff 的 motion module
                # 在 num_frames=1 时输出畸形（实测 area_ratio 5.5——
                # 主体膨胀了 5 倍，剪影 IoU 只有 0.17）；2 帧时
                # motion module 的时序注意力正常工作（IoU 0.81~0.85）。
                # 取 2 帧里的第一帧。
                out = fpipe(
                    prompt=prompt, negative_prompt=neg,
                    conditioning_frames=[conds[i], conds[i]],
                    controlnet_conditioning_scale=cn_scale,
                    ip_adapter_image=char,
                    num_frames=2,
                    num_inference_steps=steps, guidance_scale=cfg,
                    generator=torch.Generator("cuda").manual_seed(
                        seed + 7919 * (i + 1) + k),
                    height=height, width=width,
                )
                img = out.frames[0][0]
                c = (matte_mod.matte_auto(img, matte_backend, eng.matte_dir, info=matte_info)
                     if matte else img.convert("RGBA"))
                c = matte_mod.despeckle(c, denoise_iters=denoise)
                # 候选同样缩到格子尺寸再评分（与上面的判定同尺度）
                _cs = c.convert("RGB").resize((_cw, _ch), Image.BOX).convert("RGBA")
                _cs.putalpha(c.getchannel("A").resize((_cw, _ch), Image.BILINEAR))
                _cs = pxf.binarize_alpha(_cs)
                ev = framescore.evaluate(_cs, frames[i][2], char_ref_orig,
                                        area_ref=ref_areas[i])
                cand.append((c, ev, _cs))
            # 挑最好的（评分用的格子尺寸版）
            best_full, ev, _idx = max(
                cand, key=lambda t: (t[1]["passed"], t[1]["score"]))
            best_small = next(t[2] for t in cand if t[0] is best_full)
            # 补提后仍不合格就别换（换成一个更差的更糟）
            if ev["passed"] or ev["score"] > scores[i]["score"]:
                cut[i] = best_full          # 原尺寸版，供后处理
                _small_rgba[i] = best_small  # 格子尺寸版，供评分
                scores[i] = ev
                repaired.append(i)
            # 补提帧的身份可能没跟上（IP-Adapter 在不同 seed 下会漂，
            # 实测补提出来的帧会变成白色）。修法：**用整段动画的共享色板
            # 重新上色**——从所有帧（含好的帧）汇总出色板，再把补提帧按明度
            # 映射回去。这样补提帧的配色就与其它帧统一了。
            if repaired and repaired[-1] == i and palette:
                pal = pxf.extract_palette(
                    [c.convert("RGB") for c in cut if c is not cut[i]] or
                    [c.convert("RGB") for c in cut], palette)
                cut[i] = pxf.recolor_by_luma(cut[i], pal, keep_alpha=True)

    refined = pxf.refine_batch([c.convert("RGB") for c in cut], grid_snap=False,
                               target_size=(cell_w, cell_h), colors=palette)
    outs: List[Image.Image] = []
    for item, src in zip(refined["items"], cut):
        im = item["image"].convert("RGBA")
        im.putalpha(src.getchannel("A").resize((cell_w, cell_h), Image.BILINEAR))
        outs.append(pxf.binarize_alpha(im))
    aligned = pp.align_sequence(outs)

    frame_scores = [framescore.evaluate(aligned[i], frames[i][2], char_ref_orig,
                                       area_ref=ref_areas[i])
                    for i in range(len(aligned))]
    meta = {
        "workflow": "wss", "scope": scope or "default",
        "mode": "generate", "engine_mode": "animatediff",
        "prompt": prompt, "negative": neg,
        "char_image": str(char_image),
        "grid": [cols, rows], "frames": len(aligned), "cell": [cell_w, cell_h],
        "ref_mode": ref_mode, "ref_modes_used": sorted(set(used_modes)),
        "model": model,
        "width": width, "height": height,
        "vram_profile": vram_profile, "vram_note": vram_note,
        "steps": steps, "cfg": cfg, "seed": seed,
        "cn_scale": cn_scale, "ipa_scale": ipa_scale,
        "lora": lora, "lora_scale": lora_scale if lora else None,
        "clip_skip": clip_used,
        "matte": matte, "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "palette": palette,
        "palette_size": len(refined["palette"]) if palette else 0,
        "frame_scores": frame_scores,
        "gate_passed": sum(1 for s in frame_scores if s["passed"]),
        "repaired_frames": repaired,
        **framescore.sequence_consistency(aligned),
        "seconds_total": round(seconds, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    return _pack_sprite_frames(aligned, frames, cols, rows, cell_w, cell_h,
                              meta=meta, scope=scope, tag=tag)


# ---------------- W4 姿态驱动 -> 精灵帧 ----------------

POSE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def norm_pose_inputs(src: str | Path | Sequence[str | Path]) -> List[Path]:
    """把「目录 / 通配 / 文件列表」统一成**排序后**的帧路径列表。

    为什么要接受目录：W4 的入口是 `tools/pose_from_image.py`，它一次产出一个目录
    的骨架帧。如果前端必须把列表一个个展开，那界面上就得做多选——而用户手里本来
    只有一个目录路径。这里让"给个目录"就够，CLI 和 API 用同一套规则。

    **排序是必须的**：帧顺序决定动画顺序。按文件名自然序（frame_2 < frame_10），
    而不是字典序（frame_10 < frame_2）——否则 10 帧以上必然乱序。
    """
    import re

    def natural(p: Path):
        return [int(x) if x.isdigit() else x.lower()
                for x in re.split(r"(\d+)", p.name)]

    if isinstance(src, (str, Path)):
        s = str(src)
        # 通配：直接交给 glob（支持 ** 由调用方决定深度）
        if any(ch in s for ch in "*?["):
            base = Path(s)
            found = sorted(base.parent.glob(base.name), key=natural)
            return [p for p in found if p.is_file() and p.suffix.lower() in POSE_EXTS]
        p = Path(s)
        if p.is_dir():
            return sorted((x for x in p.iterdir()
                           if x.is_file() and x.suffix.lower() in POSE_EXTS), key=natural)
        if not p.exists():
            raise FileNotFoundError(f"骨架帧路径不存在：{s}")
        return [p]
    out = [Path(x) for x in src]
    missing = [str(x) for x in out if not Path(x).exists()]
    if missing:
        raise FileNotFoundError(f"骨架帧缺失：{', '.join(missing[:3])}")
    return out


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
             clip_skip: int = 1, scope: str = "", distill: str = "",
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
    # 输入可以是目录 / 通配 / 列表：前端只传目录，CLI 传列表，同一套规则
    pose_images = norm_pose_inputs(pose_images)
    if not pose_images:
        raise ValueError("没有可用的骨架帧（目录里没有 png/jpg，或通配没匹配到）")
    # cols<=0 表示"按帧数自动排一行"。以前直接传给 pack_sheet 会在
    # `(len+cols-1)//cols` 上 ZeroDivisionError——前端把 0 当"自动"传，必须兜住。
    if cols <= 0:
        cols = min(len(pose_images), 8)
    negative = negative or NEG_DEFAULT
    if preset is not None:
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)
    eng.ensure_headroom(need_gb=2.0, keep=("pose",))
    pipe = eng.pose_pipe(controlnet=controlnet, with_ipadapter=ref_image is not None,
                        model=model)
    if distill:
        eng.apply_distill(pipe, distill)
        d = eng.DISTILL_DEFAULTS.get(distill, {})
        if steps == 25:
            steps = d.get("steps", steps)
        if abs(cfg - 7.0) < 1e-6:
            cfg = d.get("cfg", cfg)
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
    # **显存守卫**：显存打满时不会报 OOM，但耗时能翻 3 倍（实测 sdxl 1024² 批量 2：
    # 23.98GB，5.0s → 17.47s）。所以宁可提前把候选数降下来——候选只影响成品率，
    # 不影响能不能跑，是显存不足时第一个该让步的东西。
    cand_note = ""
    if candidates > 1:
        from . import vramopt
        k, cand_note = vramopt.plan_candidates(eng.arch(model), max(width, height),
                                               candidates, eng.free_gb())
        if k == 0:
            raise ValueError(cand_note)
        if k < candidates:
            print(f"[warn] {cand_note}")
            candidates = k
        else:
            cand_note = ""
    # 闸门不可用时不留补提额度：没有判据的重提等于烧显卡
    retries = max(0, int(gate_retries)) if extractor is not None else 0
    attempts_max = candidates + retries
    # 文本参数每帧都一样，算一次就够（长文本会被分段编码，别再逐帧重算）
    text_kw, clip_used = _text_kwargs(pipe, prompt, negative, clip_skip, model)

    def _run(pose: Image.Image, seeds: Sequence[int]) -> List[Image.Image]:
        """一次前向出 N 个候选（N=len(seeds)）。

        **为什么批处理**：24GB 卡上单图只占 2.6~3.4GB，全程 batch=1 等于把
        20GB 显存闲着。候选本来是串行的多次前向，合成一次 batch 后实测
        512² 下 n=2 快 **1.75×**、n=4 快 **1.40×**（RTX 4090 / 8 步）。

        **代价（如实记录）**：不同 batch 尺寸会走到不同的 conv 内核，
        结果与"逐个生成"相差 1~4 级灰度（实测 mean abs diff 2.3~4.1/255），
        **不是逐像素一致**。同配置重跑仍然完全一致（有测试钉着），
        所以可复现性没有被破坏——变的是"跨配置对比"。
        """
        kwargs: Dict[str, Any] = dict(
            image=pose,
            num_inference_steps=steps, guidance_scale=cfg,
            controlnet_conditioning_scale=cn_scale,
            width=width, height=height,
        )
        if len(seeds) > 1:
            kwargs["num_images_per_prompt"] = len(seeds)
            kwargs["generator"] = [torch.Generator("cuda").manual_seed(s) for s in seeds]
        else:
            kwargs["generator"] = torch.Generator("cuda").manual_seed(seeds[0])
        kwargs.update(text_kw)          # 短文本=字符串，长文本=分段 embeds
        if ref is not None:
            kwargs["ip_adapter_image"] = ref
        return _infer(eng, pipe, kwargs, label="w4").images

    def _score(outs: Sequence[Image.Image]):
        """后处理 + 取出评分用的图。**闸门用像素化之前的图**（见 _postprocess）。"""
        got = []
        for raw in outs:
            info: Dict[str, Any] = {}
            img, _edge, _subj = _postprocess(
                raw, eng, matte=matte, matte_backend=matte_backend, denoise=denoise,
                pixel=pixel, palette=palette, clean_iso=clean_iso, info=info)
            # 抠图后端信息要带回 meta（每次 _postprocess 都会重置 pre_pixel）
            for k in ("backend", "fallback"):
                if k in info:
                    matte_info[k] = info[k]
            gate_img = info.pop("pre_pixel", None) or img
            got.append((img.convert("RGBA"), gate_img.convert("RGBA")))
        return got

    matte_info: Dict[str, Any] = {}
    frames: List[Image.Image] = []
    per_frame: List[float] = []
    frame_scores: List[Dict[str, Any]] = []
    attempts_used = 0
    prev_gate: Optional[Image.Image] = None
    torch.cuda.reset_peak_memory_stats()
    for pose_path in pose_images:
        pose = Image.open(pose_path).convert("RGB").resize((width, height))
        prev = prev_gate                # 只在抠图后有意义，未抠图 drift 恒 0
        best: Optional[tuple[Dict[str, Any], Image.Image, Image.Image]] = None
        spent = 0.0
        used = 0
        # 两轮：先凑候选，都没过闸门再用补提额度。每轮一次批调用。
        for round_seeds in (list(range(candidates)),
                            list(range(candidates, attempts_max))):
            round_seeds = [seed + i * gate_seed_step for i in round_seeds]
            if not round_seeds:
                continue
            dt = _timer()
            try:
                outs = _run(pose, round_seeds)
            except torch.OutOfMemoryError:
                # 批量撑爆显存就退回逐个跑：慢一点，但别让任务失败
                print("[warn] 批量候选 OOM，退回逐个生成")
                torch.cuda.empty_cache()
                outs = [img for s in round_seeds for img in _run(pose, [s])]
            spent += dt()
            used += len(round_seeds)
            for cand, gate_img in _score(outs):
                ev = quality.evaluate(gate_img, extractor, prev=prev,
                                      min_points=gate_min_points)
                if best is None or ev["score"] > best[0]["score"]:
                    best = (ev, cand, gate_img)
            # 候选已凑齐且挑中的过闸门 → 不动用补提额度
            if used >= candidates and (extractor is None or best[0]["passed"]):
                break
        ev, frame, prev_gate = best
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
        "model": model, "distill": distill or None,
        "clip_skip": clip_used,
        "candidates": candidates, "candidates_note": cand_note or None,
        "gate": bool(gate and extractor is not None),
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
    # 与 _save / _pack_sprite_frames 保持一致：结果里必须有 meta 嵌套。
    # 前端 fetchJob 只读 `result.meta`，w4 以前把字段平铺在顶层，于是
    # 前端拿不到 seed / seconds / candidates——接 w4 节点时就是这样踩空的。
    info["meta"] = meta
    info["url"] = f"/files/{sheet_path.relative_to(ASSETS).as_posix()}"
    info["frames_dir"] = str(fdir)
    Path(out_dir / f"{name}_sheet.png.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), "utf-8")
    if export:
        info["export"] = anim_mod.export_godot(
            aligned, export, anim_name=anim_name, fps=fps, loop=True, cols=cols)
    assetdb.append_line(ASSETS / "index.jsonl", {"sheet": info["sheet"], **meta})
    return info
