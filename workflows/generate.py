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
from . import lora as lora_reg
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


def _lora_trigger(eng: Engine, prompt: str, lora: Any) -> str:
    """把 LoRA 的触发词前置到提示词。

    **为什么必须自动做**：Pixelizer 的触发词是单 token 的
    `PixelArt_IL_One_Token_style_M_01`——不写完全不生效，写了但漏一个字符也
    完全不生效，而且**都不报错**。`pixel_sprites_32` 的 `pixelart` 同理。
    让用户在界面上手抄三十个字符不是"注意看文档"能解决的。

    只在配置里 trigger 非空时才加：空值是项目刻意留的实测结论
    （px_illustrious 加触发词会让多主体率上升），不能替用户补。
    提示词里已经出现过就不重复加。
    """
    items = lora if isinstance(lora, (list, tuple)) else [lora]
    heads: List[str] = []
    for it in items:
        t = lora_reg.trigger_for(eng.root, it)
        if t and t.lower() not in prompt.lower():
            heads.append(t)
    return f"{', '.join(heads)}, {prompt}" if heads else prompt


# 会传给 diffusers、且可能被它用 isinstance(x, float) 严格校验的参数。
# 目前只有 controlnet_conditioning_scale 会被卡，但把同类的都归一化，
# 免得下次升级 diffusers 又多卡一个再来查一遍。
# **clip_skip 不能进这个表**：它是整数语义（层数），转成 float 会进到
# encode_prompt 的切片里（`-(clip_skip + 2)`），语义不对。
_FLOAT_KEYS = (
    "controlnet_conditioning_scale", "guidance_scale", "strength",
    "control_guidance_start", "control_guidance_end",
)


def _floatify(kw: Dict[str, Any]) -> Dict[str, Any]:
    """把 kwargs 里的数值参数统一转成 Python float。

    **为什么必须做**：JSON 没有 int / float 之分——前端 slider 的 `1.0`
    序列化出去就是 `1`，Python 收到的是 `int`。而 diffusers 的 check_inputs 用
    `isinstance(controlnet_conditioning_scale, float)` 严格校验，`int` 直接
    TypeError（"must be type `float`"）。这个错误在管线都建好、进入采样前才抛，
    现场看起来像"模型坏了/权重不对"，实际只是 `1` 与 `1.0` 的差别。
    """
    out = dict(kw)
    for k in _FLOAT_KEYS:
        v = out.get(k)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _infer(eng: Engine, pipe, kw: Dict[str, Any], *, shrink: bool = True,
           label: str = "") -> Any:
    """带 OOM 恢复的推理调用：清缓存 → 降分辨率 → 重试（最多两次）。

    **为什么需要**：老实现里 OOM 只可能发生在 `pipe.to()`（`_fit` 里有降级），
    真正生成时撞顶就直接 `job.status = failed`。对"一次跑 20 个角色"的批量产线，
    这意味着跑到一半全废。这里把"降档继续跑"补上——`vramopt.plan_generation`
    本来就有降分辨率的逻辑，只是没接到这条路上。

    降档顺序：先清掉**其它**缓存管线（当前这条由局部变量持有，不会被回收），
    再按 3/4 缩边长（对齐到 8 的倍数）。缩过两次还不行就如实抛出。

    进来先做一次**浮点归一化**（见 `_as_float`）：JSON 没有 int/float 之分，
    前端 slider 的 `1.0` 到这里是 `int 1`，而 diffusers 对
    `controlnet_conditioning_scale` 用 `isinstance(x, float)` 严格校验，
    直接抛 TypeError。放在这里是单点收口——所有 ControlNet 调用都过它。
    """
    kw = _floatify(kw)
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


def _sdxl_embeds(pipe, prompt: str, negative: str, clip_skip: int):
    """SDXL 的 clip_skip embeds，走 diffusers 原生 encode_prompt。

    返回 (pos, neg, pos_pooled, neg_pooled) —— diffusers 的返回顺序就是
    `(prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
    negative_pooled_prompt_embeds)`，原样解包。

    **`do_classifier_free_guidance=True` 必须给**——踩过的坑：传 False 时
    diffusers 只算正向，`negative_prompt_embeds` 返回 **None**，于是负面词
    全部失效、CFG 退化成单向引导，出图一片灰雾（实测 std 从 38 掉到 19，
    画面糊成米色噪点）。这里要的是「两侧都算好、交给 pipe 直接用」，
    不是「让 pipe 自己拼 CFG」——pipe 拿到 embeds 就不会再重新编码了。
    """
    return pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,                  # 两个编码器用同一段文本
        negative_prompt=negative,
        negative_prompt_2=None,
        device=pipe.text_encoder.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        clip_skip=int(clip_skip),
    )


def _text_kwargs(pipe, prompt: str, negative: str, clip_skip: int, model: str):
    """返回 (传给 pipe 的文本参数, 实际生效的 clip_skip)。

    三条路径，顺序不能换：

    1. **短文本 + clip_skip<=1** → 原样传字符串。这条必须与历史行为逐像素一致
       （同一个 seed 的结果有测试钉着），所以不做任何"顺手优化"。
    2. **clip_skip>1** → 自算 embeds（transformers 5.x 下管道参数会崩，见 clip.py）。
    3. **超过 77 token** → 分段编码。CLIP 超长是**静默丢弃**，本项目的武侠契约
       自己就写到 113 token，等于尾部几条质量兜底词从来没生效。分段后不再丢。

    SDXL 走**原生的 encode_prompt(clip_skip=...)**：它有两个文本编码器 +
    pooled embeds，自己拼容易错（两侧形状、pooled、dtype 都得对齐），
    而 diffusers 的 StableDiffusionXLPipeline.encode_prompt 原生支持
    clip_skip，直接用它最稳。

    ⚠️ **SDXL 的 clip_skip 语义与 SD1.5 不同，这里必须换算**（踩过坑）：

    diffusers 的 SDXL encode_prompt 里写着：

        if clip_skip is None:
            embeds = hidden_states[-2]          # 默认 = 倒数第二层
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            embeds = hidden_states[-(clip_skip + 2)]

    也就是说 **SDXL 默认取的就是倒数第二层——那已经等价于 A1111/ComfyUI 里
    说的"clip skip 2"**（也正是动漫底模 Illustrious/Animagine 推荐的那档）。

    所以直接把用户的 clip_skip=2 传进去会跳到 **-4 层**，过度跳过：实测出图
    一片灰雾（std 从 38 掉到 19，画面糊成米色噪点、五官全无）。

    换算规则：**n ≤ 2 一律用默认（penultimate）**；n > 2 才真的多跳，
    传给 diffusers 的是 `n - 2`，让它落在 `-(n)` 层。
    """
    if "XL" in type(pipe).__name__:
        # 2 及以下 = SDXL 默认（倒数第二层）；3+ 才真的再往回跳
        if not (clip_skip and clip_skip > 2):
            return dict(prompt=prompt, negative_prompt=negative), 2
        pos, neg, pos_pooled, neg_pooled = _sdxl_embeds(
            pipe, prompt, negative, clip_skip - 2)
        return (dict(prompt_embeds=pos, negative_prompt_embeds=neg,
                     pooled_prompt_embeds=pos_pooled,
                     negative_pooled_prompt_embeds=neg_pooled), int(clip_skip))
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


# ---------------- Qwen-Image-2.1 ----------------

# 透明背景的「双段式声明」。
#
# Qwen-Image-2.1 的 VAE 有第 4 通道（alpha），但它**不会自己想起来用**——
# 必须在提示词首尾各加一句显式声明，否则输出恒为不透明（实测：不写这段，
# 提示词里写 "transparent background" 也只是画一块白底）。
# 这两句是从官方卡片抄来的原话，别改写：换说法会掉回不透明。
QWEN_ALPHA_HEAD = "This is an RGBA image with transparency. "
QWEN_ALPHA_TAIL = " The image has alpha channel and the background is transparent."

# Qwen 的负向词：**不能为空**（空了 true_cfg 静默失效，见 _qwen_call）。
# 内容也别照搬 SD 那套——它不理解 "16-bit pixel art" 这类风格词，
# 但认得出质量词与常见的构图失败。
QWEN_NEG = ("blurry, low quality, worst quality, jpeg artifacts, "
            "watermark, signature, text, deformed, extra limbs, bad anatomy, "
            "cropped, out of frame")


def _qwen_prompt(prompt: str, transparent: bool) -> str:
    """按是否要透明底拼最终提示词。"""
    body = (prompt or "").strip().strip(",")
    if not transparent:
        return body
    return f"{QWEN_ALPHA_HEAD}{body}{QWEN_ALPHA_TAIL}"


# 与「透明底」直接冲突的背景措辞。**必须清掉**：
# 提示词里写 "plain white background" 时，Qwen 会把它当正面要求执行——
# 实测同一个 seed，带这句时 alpha 全是中间值（81.5% 前景、77.5% 停在
# 16~240 的半透明雾），不带这句时是干净的（31% 前景、67.8% 全透明、
# 中间值只有 2.9%）。用户看到的「透明没生效 / 抠出来一圈白」多半就是这句。
_QWEN_BG_WORDS = (
    "plain white background", "pure white background", "white background",
    "plain background", "simple background", "flat background",
    "solid background", "clean background", "white backdrop", "white bg",
)


def _strip_bg_words(prompt: str) -> tuple[str, bool]:
    """去掉与透明底冲突的背景描述词。返回 (新提示词, 是否改动过)。"""
    import re

    out = prompt or ""
    hit = False
    for w in _QWEN_BG_WORDS:
        if re.search(re.escape(w), out, flags=re.I):
            out = re.sub(re.escape(w), "", out, flags=re.I)
            hit = True
    if hit:
        # 清掉因删除留下的 ", ,"、首尾逗号，以及句尾悬空的介词/冠词
        # （"a cat on a white background" → "a cat on a" → "a cat"）
        parts = []
        dangling = {"on", "in", "at", "with", "over", "against", "of", "and",
                    "a", "an", "the", "to", "for"}
        for seg in out.split(","):
            words = seg.split()
            while words and words[-1].lower().strip(".") in dangling:
                words.pop()
            seg = " ".join(words).strip()
            if seg:
                parts.append(seg)
        out = ", ".join(parts)
    return out, hit


def sanitize_alpha(img: Image.Image) -> Image.Image:
    """清理 Qwen 透明底的脏 alpha：半透明毛边全部二值化。

    实测透明区常出现 alpha=1~8 的「几乎全透明但不为零」像素，RGB 还是 VAE
    解出来的噪声色。不清掉的话，任何丢掉 alpha 的下游（转 RGB、别的看图工具）
    都会看到一片品红/黑底。这里 alpha<16 连同 RGB 一起清零，>240 拉满。
    """
    if img.mode != "RGBA":
        return img
    arr = np.asarray(img).copy()
    a = arr[:, :, 3]
    low = a < 16
    high = a > 240
    if not low.any() and not high.any():
        return img
    arr[low, 0:3] = 0
    arr[low, 3] = 0
    arr[high, 3] = 255
    return Image.fromarray(arr, "RGBA")


def _qwen_call(eng: Engine, *, prompt: str, negative: str, width: int, height: int,
               steps: int, seed: int, ref_image: Optional[Image.Image] = None,
               transparent: bool = False, model: str = "qwen-image-2.1",
               label: str = "w1", cfg: float = 4.0,
               output_resolution: Optional[int] = None) -> Image.Image:
    """Qwen 单次推理。带 ref_image 就走「指令改图」，不带就是文生图。

    三个**必须**记住的点（都是实测踩出来的）：
      1. `true_cfg_scale` 只有配 `negative_prompt` 才生效——管线里的判据是
         `do_true_cfg = true_cfg_scale > 1 and has_neg_prompt`，不传负向时
         CFG 静默失效（cfg=1 与 cfg=4 输出逐字节相同）。
      2. 尺寸必须是 16 的倍数（VAE 下采样 + patchify）。不是的话 diffusers
         只 warn 一句就静默改尺寸，最后 meta 里记的尺寸是假的。
      3. **`output_resolution` 必须显式传成目标边长**。管线的默认值是 1024，
         它决定「条件图被缩放到多大再编码」——于是 768² 的出图配 1024² 的
         参考图token：参考图的尺度比目标大一档（模型看到的细节密度对不上），
         图 token 也多 78%。实测（768²/8 步）：改图 16.28s / 20.14GB →
         对齐后 14.75s / 18.75GB，且输出不是同一个采样点（平均差 6.15/255），
         对齐后的那张构图更贴原图。
    """
    pipe = eng.txt2img_pipe(model)
    w = max(256, int(width) // 16 * 16)
    h = max(256, int(height) // 16 * 16)
    full = _qwen_prompt(prompt, transparent)
    gen = torch.Generator("cuda").manual_seed(int(seed))
    kw: Dict[str, Any] = dict(
        prompt=full,
        negative_prompt=negative or QWEN_NEG,
        width=w, height=h,
        num_inference_steps=max(8, min(int(steps), 60)),
        true_cfg_scale=float(cfg),
        generator=gen,
        # 条件图分辨率与目标对齐（见上第 3 点）
        output_resolution=int(output_resolution or max(w, h)),
    )
    if ref_image is not None:
        # 参考图统一缩放到目标尺寸：管线对多图原生支持，但单图这条路径
        # 不走 resize 的话会把原图尺寸带进 latent token 计算
        #
        # **多图**（list）用于「角色迁移」这类要同时给两张参考的场合：
        # 角色图 + 骨架图。Qwen 的条件图会一起编码成 vision context，
        # 所以顺序有意义：第一张是身份，第二张是姿态骨架。
        if isinstance(ref_image, (list, tuple)):
            kw["image"] = [
                im.convert("RGB").resize((w, h), Image.LANCZOS) for im in ref_image
            ]
        else:
            kw["image"] = ref_image.convert("RGB").resize((w, h), Image.LANCZOS)
    out = _infer(eng, pipe, kw, label=label)
    return sanitize_alpha(out.images[0].convert("RGBA"))


# ---- Qwen 的「局部重绘」公共件（w3 / w5 共用） ----

# 局部编辑的负向词：重点是「别画蛇添足」与「别出新脸」。
QWEN_EDIT_NEG = (
    "seam, visible seam, color banding, ghosting, duplicate limbs, extra characters, "
    "changed face, different character, text, watermark, signature, blurry"
)

# 约束条款。Qwen 看不懂 SD 的契约后缀，但**认得出整句自然语言指令**，
# 所以这里写成人话，而不是逗号堆叠的关键词。
QWEN_EDIT_RULES = (
    "Keep the character's face, hairstyle, body proportions, pose, art style and "
    "everything outside that area exactly the same. Do not add new objects, do not "
    "redraw the background, no visible seam."
)


def _qwen_work_size(size: tuple[int, int], *, cap: int = 1024,
                    lo: int = 512) -> tuple[int, int]:
    """把源图尺寸夹到 Qwen 的划算区间（16 的倍数，长边 ≤ cap）。

    为什么要有上限：Qwen 的开销随面积涨，1024² 的图 token 是 768² 的 1.78 倍，
    而像素素材原生只有 32~128px——放大到 1024 并不会多出细节，只会把
    「按 1024 画的笔触」再缩回去，白烧一倍显存与时间。
    """
    w, h = size
    f = min(1.0, cap / max(1, max(w, h)))
    def snap(v: float) -> int:
        # **向下取整到 16 的倍数**：向上取整会让「不缩放」的图反而被放大 1~15px，
        # 白白多算 token。宁可少 15px 也不放大（像素素材放大不会多出信息）。
        return max(16, int(v * f) // 16 * 16)
    return snap(w), snap(h)


def _region_words(mask: Image.Image) -> str:
    """把蒙版位置翻译成一句方位词（top-left / middle-center …）。

    **为什么需要**：Qwen 没有 mask 通道，模型只能从提示词知道「改哪里」。
    只给「把盔甲改成绿色」时它会挑最像盔甲的地方——常常整件重画。
    按蒙版包围盒的中心补一句方位，等于把「这里」说清楚，实测能明显减少
    选区外的改动（配合蒙版贴回后，选区外本来就保证逐点不变）。
    """
    bb = mask.getbbox()
    if bb is None:
        raise ValueError("蒙版是空的：先涂出要重绘的区域")
    cx = (bb[0] + bb[2]) / 2.0 / mask.width
    cy = (bb[1] + bb[3]) / 2.0 / mask.height
    vert = "top" if cy < 0.35 else ("bottom" if cy > 0.65 else "middle")
    horz = "left" if cx < 0.35 else ("right" if cx > 0.65 else "center")
    return f"{vert}-{horz}"


def _qwen_edit_prompt(body: str, region: str) -> str:
    body = (body or "").strip().rstrip(".")
    return (f"Edit this image: {body}. Apply the change only to the {region} area; "
            f"{QWEN_EDIT_RULES}")


def _qwen_local_edit(eng: Engine, *, src: Image.Image, mask: Image.Image,
                     body: str, negative: Optional[str], steps: int, cfg: float,
                     seed: int, model: str,
                     label: str) -> tuple[Image.Image, tuple[int, int], str, str]:
    """整图做条件 + 指令改图，返回 (生成图[源图尺寸], 工作尺寸, 指令, 方位)。

    **为什么整图而不是裁剪块**：裁剪块只有 400px 左右，放大到 768² 后模型会把
    它当新图重画——实测脸被抹掉、且「盔甲改翠绿」这条指令根本没执行
    （选区外像素差 0，但选区内是另一个人）。整图做条件时模型看到完整角色，
    身份与画风都在 vision context 里，只需要执行局部指令。
    代价是整图编码比裁剪块贵（768² vs 448²），换来的是「能用」。
    """
    region = _region_words(mask)
    prompt = _qwen_edit_prompt(body, region)
    work = _qwen_work_size(src.size)
    # 透明区填白：直接 convert("RGB") 会变黑，模型会把黑块当成内容重画
    payload = rp.flatten_rgb(src)
    if work != src.size:
        payload = payload.resize(work, Image.LANCZOS)
    gen = _qwen_call(
        eng, prompt=prompt, negative=negative or QWEN_EDIT_NEG,
        width=work[0], height=work[1], steps=steps, seed=seed,
        ref_image=payload, model=model, label=label, cfg=cfg)
    gen = gen.convert("RGB")
    if gen.size != src.size:
        gen = gen.resize(src.size, Image.LANCZOS)
    return gen, work, prompt, region


def _restore_alpha(out: Image.Image, src: Image.Image) -> Image.Image:
    """源图有透明底、而生成图整张不透明时，把源图的 alpha 还回去。

    Qwen 只有在提示词里写了「RGBA / transparent background」才会出透明通道；
    局部重绘不写（写了反而会把选区画成透明洞）。于是透明素材重绘后，
    选区里会多出一圈白底。判据用「生成图完全没有透明像素」——它说明模型
    这次根本没打算画透明，alpha 不可信，沿用源图的最稳。
    """
    if src.mode != "RGBA" or out.mode != "RGBA":
        return out
    a_out = np.asarray(out.getchannel("A"))
    a_src = np.asarray(src.getchannel("A"))
    if a_out.min() > 250 and a_src.min() < 250:
        # 复制一份再改：in-place 修改会让调用方手里的生成图被悄悄换掉
        out = out.copy()
        out.putalpha(src.getchannel("A"))
    return out




def _qwen_txt2img(eng: Engine, *, prompt: str, negative: Optional[str],
                  width: int, height: int, steps: int, seed: int,
                  transparent: bool, pixel: int, palette: int, preset: str,
                  matte: bool, matte_backend: str, denoise: int, clean_iso: int,
                  model: str, scope: str, tag: str) -> Dict[str, Any]:
    """Qwen 文生图。

    **透明底时不能用 SD 那套契约后缀**：SPRITE_SUFFIX 里写着 "flat plain
    background"，跟"背景要透明"是直接矛盾的指令——实测会画出白底/灰底，
    alpha 通道不激活。所以透明场景换成只保留构图约束的短后缀。
    """
    if preset and preset != "none":
        if transparent:
            prompt, negative = presets.build(prompt, negative, "none",
                                             max_tokens=presets.CHUNK_BUDGET)
            prompt = f"{prompt}, single character, full body, centered" if prompt \
                else "single character, full body, centered"
        else:
            prompt, negative = presets.build(prompt, negative, preset,
                                             max_tokens=presets.CHUNK_BUDGET)
    stripped = False
    if transparent:
        # 透明底时把「白底/纯色底」这类措辞清掉：它们是和透明声明直接打架的
        # 正面要求，留着会让 alpha 变成一层雾（见 _QWEN_BG_WORDS 的实测）
        prompt, stripped = _strip_bg_words(prompt)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    img = _qwen_call(eng, prompt=prompt, negative=(negative or ""),
                     width=width, height=height, steps=steps, seed=seed,
                     transparent=transparent, model=model, label="w1")
    seconds = dt()
    width, height = img.size
    # 透明底的原图自带 alpha，再跑抠图会把 alpha 当背景重算一遍，纯属浪费
    raw = img.copy() if (pixel > 1 or palette) else None
    info: Dict[str, Any] = {}
    img2, edge, subjects = _postprocess(
        img, eng, matte=(matte and not transparent), matte_backend=matte_backend,
        denoise=denoise, pixel=pixel, palette=palette, clean_iso=clean_iso, info=info)
    return _save(img2, _stamp(tag), "txt2img", {
        "workflow": "w1", "prompt": _qwen_prompt(prompt, transparent),
        "negative": negative or QWEN_NEG,
        "width": width, "height": height, "steps": steps, "cfg": 4.0,
        "true_cfg": 4.0, "seed": seed,
        "lora": None, "lora_scale": None,
        "pixel_block": pixel, "palette": palette, "matte": matte,
        "matte_backend": matte_backend, "denoise": denoise, "clean_iso": clean_iso,
        "matte_used": info.get("backend"), "matte_fallback": info.get("fallback"),
        "alpha_reused": info.get("alpha_reused"),
        "prompt_bg_stripped": stripped or None,
        "edge_touch": edge, "model": model, "clip_skip": None,
        "subjects": subjects, "transparent": transparent,
        "hires": None, "distill": None, "refiner": False,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


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
            transparent: bool = False,
            scope: str = "", tag: str = "w1") -> Dict[str, Any]:
    if eng.arch(model) == "qwen21":
        return _qwen_txt2img(
            eng, prompt=prompt, negative=negative, width=width, height=height,
            steps=steps, seed=seed, transparent=transparent, pixel=pixel,
            palette=palette, preset=preset, matte=matte,
            matte_backend=matte_backend, denoise=denoise, clean_iso=clean_iso,
            model=model, scope=scope, tag=tag)
    prompt = _lora_trigger(eng, prompt, lora)
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
    # 精修幅度 = 1 − 精修起点（与 UI 上那个滑块直接对应）。
    # img2img 内部按 `int(steps × strength)` 取实际步数，所以 20 步 × 0.2 = 4 步。
    refiner_strength = max(0.03, min(0.60, 1.0 - float(refiner_end)))
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
        # refiner 是 img2img 管线：只接受图片（latent 是给「底模提前停」的
        # 两段式用的，这里底模已经跑满，所以先解码成图再喂进去）。
        rp = eng.refiner_pipe()
        # refiner 的 embeds 要自己组装（见 Engine.refiner_embeds 的说明）：
        # 直接传 prompt 会用底模那套 2048 维拼接，喂不进 refiner 的 1280 维 UNet。
        pos_e, neg_e, pos_pool, neg_pool = eng.refiner_embeds(prompt, neg)
        # **用 strength 表达精修幅度，不要用 denoising_start。**
        #
        # denoising_start 是「两段式」参数：底模 denoising_end 提前停、refiner
        # 从那个时间点接着跑。这里底模已经跑满全部步数，再传 denoising_start=0.8
        # 的语义就变成「把成品重新加噪到 80% 再只走 4 步」——实测出图糊成一团
        # 灰雾（拉普拉斯方差 1826 → 727，颜色也从白底线稿变灰底）。
        # 精修是「低幅度 refinement」，对应 img2img 的 strength：0.05~0.3。
        img = _infer(eng, rp, dict(prompt_embeds=pos_e,
                                   negative_prompt_embeds=neg_e,
                                   pooled_prompt_embeds=pos_pool,
                                   negative_pooled_prompt_embeds=neg_pool,
                                   image=img,
                                   strength=refiner_strength,
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
        # 实际生效的幅度与步数：界面上的「精修步数」是**总步数**，
        # img2img 只执行其中 strength 那一段（20 步 × 0.2 = 4 步）。
        # 不写清楚的话，用户会以为跑了 20 步却还是没细节。
        "refiner_strength": round(refiner_strength, 3) if use_refiner else None,
        "refiner_steps_run": (max(1, round(refiner_steps * refiner_strength))
                              if use_refiner else None),
        "seconds": seconds, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


# ---------------- W2 图生图 ----------------

def _qwen_img2img(eng: Engine, *, init_image: str | Path, prompt: str,
                  negative: Optional[str], steps: int, seed: int,
                  transparent: bool, preset: Optional[str], pixel: int,
                  palette: int, matte: bool, matte_backend: str, denoise: int,
                  clean_iso: int, model: str, scope: str,
                  tag: str) -> Dict[str, Any]:
    """Qwen 指令改图：源图当参考，提示词当指令。

    输出尺寸跟源图走（16 的倍数、必要时等比缩放），不额外接受 width/height——
    改图场景下"我要跟原图一样大"是压倒性预期。
    """
    if preset and preset != "none":
        prompt, negative = presets.build(prompt, negative, preset,
                                         max_tokens=presets.CHUNK_BUDGET)
    if transparent:
        # 同文生图：透明底时清掉「白底」这类与透明声明打架的措辞
        prompt, _stripped = _strip_bg_words(prompt)
    if scope:
        src = Path(init_image).resolve()
        own = scope_dir(scope).resolve()
        root = PROJECTS_DIR.resolve()
        if str(src).startswith(str(root)) and not str(src).startswith(str(own)):
            raise ValueError(
                f"源图不属于当前项目：{src.name} 在别的项目目录里（项目之间相互隔离）")
    ref_img = Image.open(init_image).convert("RGBA")
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    img = _qwen_call(eng, prompt=prompt, negative=(negative or ""),
                     width=ref_img.width, height=ref_img.height,
                     steps=steps, seed=seed, ref_image=ref_img,
                     transparent=transparent, model=model, label="w2")
    seconds = dt()
    raw = img.copy() if (pixel > 1 or palette) else None
    info: Dict[str, Any] = {}
    img, edge, subjects = _postprocess(
        img, eng, matte=(matte and not transparent), matte_backend=matte_backend,
        denoise=denoise, pixel=pixel, palette=palette, clean_iso=clean_iso, info=info)
    return _save(img, _stamp(tag), "img2img", {
        "workflow": "w2", "prompt": _qwen_prompt(prompt, transparent),
        "negative": negative or QWEN_NEG,
        "init_image": str(init_image), "strength": None, "steps": steps,
        "cfg": 4.0, "true_cfg": 4.0, "seed": seed, "lora": None,
        "lora_scale": None,
        "pixel_block": pixel, "palette": palette, "matte": matte,
        "matte_backend": matte_backend, "matte_used": info.get("backend"),
        "matte_fallback": info.get("fallback"),
        "edge_touch": edge, "subjects": subjects, "model": model,
        "clip_skip": None, "transparent": transparent,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


def img2img(eng: Engine, *, init_image: str | Path, prompt: str,
            negative: Optional[str] = None, strength: float = 0.55, steps: int = 25,
            cfg: float = 7.0, seed: int = -1, lora: Optional[str] = None,
            lora_scale: float = 0.8, model: str = "sd15", clip_skip: int = 1,
            preset: Optional[str] = None, pixel: int = 0, palette: int = 0,
            matte: bool = False, matte_backend: str = "auto", denoise: int = 1,
            clean_iso: int = 1, transparent: bool = False,
            scope: str = "", tag: str = "w2") -> Dict[str, Any]:
    """在图生图里重画原图的**内容**（改姿势细节、换配色、加装饰），构图与轮廓保持。

    与 W1 的区别在于「底」是已有图，所以：
    - 输出尺寸由源图决定，不接受 width/height（diffusers 也这么做）。
    - `strength` 是重绘幅度：0.3 只动细节，0.6 明显改画风，0.85 基本重画。
    - 后处理（抠图/像素化/量化）与 W1 完全同一套。源图**应当**是未经像素化的
      原图（txt2img 会额外存 `<name>_raw.png`），拿成品精灵回炉会把色块糊掉。

    **Qwen-Image-2.1 走的是「指令改图」**，不是 SD 那种加噪强度：它没有
    strength 参数，改多少完全由提示词决定。所以同一张源图、同一个提示词，
    SD 下调 strength 是渐变，Qwen 下是"照指令改"。别拿 strength 去预期它。
    """
    if eng.arch(model) == "qwen21":
        return _qwen_img2img(
            eng, init_image=init_image, prompt=prompt, negative=negative,
            steps=steps, seed=seed, transparent=transparent, preset=preset,
            pixel=pixel, palette=palette, matte=matte,
            matte_backend=matte_backend, denoise=denoise,
            clean_iso=clean_iso, model=model, scope=scope, tag=tag)
    if preset is not None:
        prompt = _lora_trigger(eng, prompt, lora)
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

def _qwen_inpaint(eng: Engine, *, image: str | Path, mask: str | Path, prompt: str,
                  negative: Optional[str], steps: int, cfg: float, seed: int,
                  model: str, feather: int, match_color: bool,
                  match_strength: float, transparent: bool,
                  scope: str, tag: str) -> Dict[str, Any]:
    """Qwen 版局部重绘：整图指令改图 + 蒙版贴回。

    **它不是真 inpaint**（Qwen 三条编辑管线都没有 mask / mask_image / strength
    参数，见 PLAN 的 C 段实测），所以承诺要分开写：

      · 选区**外**：逐点不变。这是硬承诺——贴回用「只向内羽化 + 二值蒙版相乘」，
        和 SD 那条路是同一个 `repaint.composite`。
      · 选区**内**：模型按指令重画，但**不保证只改那一点**。它看不到蒙版，
        只能从提示词里的方位词猜到要改哪儿；猜偏了的部分会被蒙版裁掉，
        表现是「改得不够」而不是「改到外面」。
    """
    src = Image.open(image).convert("RGBA")
    msk = rp.load_mask(mask)
    if msk.size != src.size:
        raise ValueError(
            f"蒙版尺寸 {msk.size[0]}×{msk.size[1]} 与源图 "
            f"{src.width}×{src.height} 不一致：换图后要重新涂抹")
    body = (prompt or "").strip()
    if not body:
        raise ValueError("提示词是空的：先写一句这块区域要变成什么")

    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    gen, work, full_prompt, region = _qwen_local_edit(
        eng, src=src, mask=msk, body=body, negative=negative, steps=steps,
        cfg=cfg, seed=seed, model=model, label="w3")
    out = rp.composite(src, gen.convert("RGBA"), msk, feather_px=feather,
                       match_color=match_color, match_strength=match_strength)
    if not transparent:
        out = _restore_alpha(out, src)
    seconds = dt()
    # 自检：承诺写进 meta，别让用户去猜「到底有没有溢出」
    diff = _outside_diff(src, out, msk)
    return _save(out, _stamp(tag), "inpaint", {
        "workflow": "w3", "prompt": full_prompt, "prompt_body": body,
        "negative": negative or QWEN_EDIT_NEG,
        "image": str(image), "mask": str(mask),
        "strength": None, "steps": steps, "cfg": cfg, "seed": seed,
        "model": model, "backend": "qwen-edit", "approx": True,
        "region": region, "work_size": list(work),
        "coverage": rp.mask_coverage(msk), "feather": feather,
        "match_color": bool(match_color), "match_strength": match_strength,
        "outside_max_diff": diff,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, scope=scope)


def _outside_diff(src: Image.Image, out: Image.Image, mask: Image.Image) -> int:
    """蒙版外最大像素差。正常恒为 0；非 0 说明贴回漏了（值得报警而不是沉默）。"""
    a = np.asarray(src.convert("RGB"), dtype=np.int16)
    b = np.asarray(out.convert("RGB"), dtype=np.int16)
    m = np.asarray(mask.convert("L")) > 127
    if not (~m).any():
        return 0
    return int(np.abs(a - b)[~m].max())


def inpaint(eng: Engine, *, image: str | Path, mask: str | Path, prompt: str,
            negative: Optional[str] = None, strength: float = 0.9, steps: int = 25,
            cfg: float = 7.5, seed: int = -1, model: str = "sd15", scope: str = "",
            feather: int = 4, match_color: bool = True,
            match_strength: float = 0.45, transparent: bool = False,
            tag: str = "w3") -> Dict[str, Any]:
    """只重画 mask 区域，并把结果按羽化蒙版贴回——选区外像素逐点不变。

    **Qwen 底模下是近似实现**：没有 mask 接口，走「整图指令改图 + 蒙版贴回」。
    `strength` 对它没有意义（Qwen 的改动力度由提示词决定），会被忽略。
    """
    eng.require(model, "inpaint")
    if eng.arch(model) == "qwen21":
        return _qwen_inpaint(
            eng, image=image, mask=mask, prompt=prompt, negative=negative,
            steps=steps, cfg=cfg, seed=seed, model=model, feather=feather,
            match_color=match_color, match_strength=match_strength,
            transparent=transparent, scope=scope, tag=tag)
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
    # 抠图：与其它工作流共用同一套（含 u2net 抠空自动回退的保护）。
    # info 要带出来：Qwen 出图自带 alpha 时这里会是 backend="alpha"（原样复用），
    # 用户看到「没重抠」才知道不是漏了。
    minfo: Dict[str, Any] = {}
    cut = matte_mod.despeckle(
        matte_mod.matte_auto(src, matte_backend, eng.matte_dir,
                             info=minfo, alpha_mode=alpha_mode, feather=feather),
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
        # 实际用了哪个后端 / 是否复用了输入自带的 alpha（Qwen 出图常见）
        "matte_used": minfo.get("backend"),
        "matte_fallback": minfo.get("fallback"),
        "alpha_reused": minfo.get("alpha_reused"),
        "alpha_rejected": minfo.get("alpha_rejected"),
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


def _qwen_ai_repaint(eng: Engine, *, image: str | Path, mask: str | Path,
                     prompt: str, negative: Optional[str], steps: int, cfg: float,
                     seed: int, model: str, preset: Optional[str],
                     context: float, expand: int, feather: int,
                     match_color: bool, match_strength: float,
                     palette_snap: bool, palette_colors: int, strategy: str,
                     scope: str, tag: str) -> Dict[str, Any]:
    """Qwen 版局部 AI 重绘。两种策略：

      · ``whole``（默认）：**整图做条件**，模型看得到完整角色 → 身份与画风
        都在 context 里。这是本项目实测唯一「指令被执行且脸不丢」的方案。
      · ``crop``：裁「选区 + 上下文」放大后改图（SD 那条路的老方案）。
        对 Qwen 不推荐：裁剪块放大到 768² 后会被当成**新图**重画，
        实测脸被抹掉、指令不执行（见 docs/PLAN-QWEN-ONLY.md 的 A 段）。
        保留它是为了对照实验，默认不启用。
    """
    src = Image.open(image).convert("RGBA")
    msk = rp.load_mask(mask)
    if msk.size != src.size:
        raise ValueError(
            f"蒙版尺寸 {msk.size[0]}×{msk.size[1]} 与源图 "
            f"{src.width}×{src.height} 不一致：换图后要重新涂抹")
    body = (prompt or "").strip()
    if not body:
        raise ValueError("提示词是空的：先写一句要重绘成什么")
    if preset and preset != "none":
        body, negative = presets.build(body, negative, preset,
                                       max_tokens=presets.CHUNK_BUDGET)
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    if strategy == "crop":
        # 老方案：裁剪块 → 改图 → 贴回裁剪块 → 贴回原图
        crop, crop_mask, box = rp.crop_region(src, msk, expand=expand,
                                              context=context)
        target = rp.fit_size(crop.size)
        payload = rp.flatten_rgb(crop).resize(target, Image.LANCZOS)
        region = _region_words(msk)
        full_prompt = _qwen_edit_prompt(body, region)
        gen = _qwen_call(
            eng, prompt=full_prompt, negative=negative or QWEN_EDIT_NEG,
            width=target[0], height=target[1], steps=steps, seed=seed,
            ref_image=payload, model=model, label="w5", cfg=cfg)
        gen = gen.convert("RGB").resize(crop.size, Image.LANCZOS)
        merged = rp.composite(crop, gen.convert("RGBA"), crop_mask,
                              feather_px=feather, match_color=match_color,
                              match_strength=match_strength)
        out = src.copy()
        out.paste(merged, box)
        work = target
    else:
        box = list(msk.getbbox() or (0, 0, src.width, src.height))
        gen, work, full_prompt, region = _qwen_local_edit(
            eng, src=src, mask=msk, body=body, negative=negative, steps=steps,
            cfg=cfg, seed=seed, model=model, label="w5")
        out = rp.composite(src, gen.convert("RGBA"), msk, feather_px=feather,
                           match_color=match_color,
                           match_strength=match_strength)
        out = _restore_alpha(out, src)

    # 色板吸附（像素素材专用）：把重绘区里「接近源色」的杂色吸回源图色板，
    # 差得远的新颜色保留——否则「要把这块改成红色」会被吸成灰。
    palette_snapped = 0.0
    if palette_snap:
        out, _pal, palette_snapped = rp.snap_palette(
            out, src, msk, colors=palette_colors)

    warnings: List[str] = []
    min_side = min(box[2] - box[0], box[3] - box[1]) if strategy == "crop" \
        else min(src.width, src.height)
    if strategy == "crop" and feather > max(2, min_side // 4):
        warnings.append(
            f"羽化 {feather}px 相对选区偏大：改动会溢出选区边缘一圈，"
            "想要干净边界就调小")
    if palette_snap and palette_snapped < 0.02:
        warnings.append("色板对齐几乎没有生效：新颜色离源图色板都太远")
    seconds = dt()
    return _save(out, _stamp(tag), "ai_repaint", {
        "workflow": "w5", "scope": scope or "default",
        "prompt": full_prompt, "prompt_body": body, "constraint": QWEN_EDIT_RULES,
        "negative": negative or QWEN_EDIT_NEG,
        "image": str(image), "mask": str(mask),
        "strength": None, "steps": steps, "cfg": cfg, "seed": seed,
        "model": model, "lora": None, "lora_scale": None, "clip_skip": None,
        "backend": "qwen-edit", "approx": True, "strategy": strategy,
        "region": region, "work_size": list(work),
        "box": [int(x) for x in box],
        "coverage": rp.mask_coverage(msk),
        "context": context if strategy == "crop" else None,
        "expand": expand if strategy == "crop" else None,
        "feather": feather,
        "match_color": bool(match_color), "match_strength": match_strength,
        "palette_snap": bool(palette_snap), "palette_snapped": palette_snapped,
        "outside_max_diff": _outside_diff(src, out, msk),
        "warnings": warnings,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, scope=scope)


def ai_repaint(eng: Engine, *, image: str | Path, mask: str | Path, prompt: str,
               negative: Optional[str] = None, strength: float = 0.9,
               steps: int = 25, cfg: float = 7.0, seed: int = -1,
               model: str = "sd15", lora: Optional[str] = None,
               lora_scale: float = 0.8, clip_skip: int = 1,
               preset: Optional[str] = None,
               context: float = 0.35, expand: int = 8, feather: int = 4,
               match_color: bool = True, match_strength: float = 0.45,
               palette_snap: bool = False, palette_colors: int = 32,
               strategy: str = "whole",
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
    eng.require(model, "ai_repaint")
    if eng.arch(model) == "qwen21":
        if strategy not in ("whole", "crop"):
            raise ValueError(f"未知策略 {strategy}（可选 whole / crop）")
        return _qwen_ai_repaint(
            eng, image=image, mask=mask, prompt=prompt, negative=negative,
            steps=steps, cfg=cfg, seed=seed, model=model, preset=preset,
            context=context, expand=expand, feather=feather,
            match_color=match_color, match_strength=match_strength,
            palette_snap=palette_snap, palette_colors=palette_colors,
            strategy=strategy, scope=scope, tag=tag)
    if preset is not None:
        prompt = _lora_trigger(eng, prompt, lora)
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


# ---------------- WAP 动漫人物 → 像素 ----------------

def image_to_pixel(eng: Engine, *,
                   init_image: Optional[str | Path] = None,
                   prompt: str = "", negative: Optional[str] = None,
                   target: Optional[Sequence[int]] = None,
                   colors: int = 32, snap_to: Optional[int] = 16,
                   grid_snap: bool = True,
                   sample_method: str = "median",
                   pre_enhance: float = 1.0,
                   edge_rebuild: bool = True,
                   source_width: int = 1024, source_height: int = 1024,
                   steps: int = 28, cfg: float = 8.0, seed: int = -1,
                   model: str = "sdxl-anime", clip_skip: int = 2,
                   matte: bool = True, matte_backend: str = "auto",
                   transparent: bool = False,
                   scope: str = "", tag: str = "wap") -> Dict[str, Any]:
    """任意图 → 像素画（不限于动漫人物）。**两步走**：先出高质量大图，再做像素化。

    **不只动漫**：照片、插画、3D 渲染、手绘图都能转——像素化那一半是通用的
    （网格对齐 + 色板量化 + 轮廓描边），与前一步出什么图无关。只在"没有源图、
    需要先生成"时才会用到动漫底模。

    为什么这样比「让 SD 直接画像素」好（本项目实测）：

        SD 直接画（像素 LoRA + pixel=6）  构图常裁到腿、颜色是连续渐变、
                                          输出 768² 但有效像素只有 128
        动漫大图 → 像素化                全身完整、1px 描边清晰、
                                          输出就是原生 128²/192²

    根因是**两种能力互相打架**：像素风要求极简色块和硬边，而人物结构
    （脸、手、持握武器）需要足够的分辨率理解。让模型一次做两件事，它会在
    两者之间妥协。分开做则各取所长：动漫底模负责「画对」，像素化负责
    「压成像素」。

    `init_image` 给了就直接转（任何图都行）；没给就用 `prompt` 先出图再转。
    后者默认走动漫底模，用户只写角色描述就能一步拿到像素精灵。

    **像素化三步增强**（都是实测出来的，缺一个质量就掉）：
      1. `pre_enhance`  量化前拉对比度/饱和度/锐度。不开时红披风变暗红、
                        金发变土黄、整体发灰；开 1.0 后鲜红 + 亮黄。
      2. 共享色板      整批共用一块色板，逐张各自量化会让相邻帧颜色打架。
                        且**必须排除透明像素**——抠图后透明区 RGB 是纯黑，
                        不排除的话 84% 的色位被背景吃掉（实测只剩 12 色）。
      3. `edge_rebuild` 轮廓描边（色板内暗色，色数不变）。量化后相邻色块
                        只有"颜色不同"没有"结构线"，缩小看是一坨色斑
                        （业界叫 melt）。只描外轮廓——对内部结构边全描
                        会把脸糊成一块灰。

    参数：
      target   目标原生像素尺寸（默认 192×192；游戏精灵常用 64/96/128/192）
      colors   共享色板色数（默认 32；8-bit 风格可给 16，16-bit 给 32~48）
      snap_to  网格边长吸附到该数的整数倍（16 = 精灵常用）
      matte    先把源图的背景抠掉（像素精灵必须有透明底）
    """
    t_all = _timer()
    meta: Dict[str, Any] = {"workflow": "topixel", "scope": scope or "default"}
    src_path: Optional[Path] = None
    gen_meta: Dict[str, Any] = {}

    # 1) 没有源图就先出一张动漫全身图
    if init_image is None:
        if not (prompt or "").strip():
            raise ValueError(
                "需要 prompt 或 init_image 之一："
                "要么描述角色（先出动漫图再转像素），要么给一张现成的图")
        if model != "sdxl-anime" and "anime" not in model and eng.arch(model) != "qwen21":
            # 用户可能选了别的底模；提示一次但不阻止（有人就想用自己的底模）
            print(f"[warn] 底模 {model} 不是动漫专模，动漫人物质量可能不如 sdxl-anime")
        r = txt2img(eng, prompt=prompt, negative=negative, preset="anime",
                    width=source_width, height=source_height,
                    steps=steps, cfg=cfg, seed=seed, model=model,
                    clip_skip=clip_skip, matte=matte,
                    matte_backend=matte_backend, transparent=transparent,
                    tag=f"{tag}_src", scope=scope)
        src_path = Path(r["file"])
        gen_meta = r["meta"]
        if seed < 0:
            seed = int(gen_meta.get("seed", 0))
    else:
        src_path = Path(init_image)

    # 2) 像素化
    src = Image.open(src_path).convert("RGBA")
    if matte and init_image is not None:
        # 外部图没有 alpha 时才需要抠；生成图已经抠过
        arr = np.asarray(src)
        if arr.shape[-1] == 4 and float((arr[..., 3] < 128).mean()) < 0.01:
            info: Dict[str, Any] = {}
            src = matte_mod.matte_auto(src.convert("RGB"), matte_backend,
                                       eng.matte_dir, info=info)
            meta["matte_used"] = info.get("backend")

    size = tuple(target) if target else (192, 192)
    res = pxf.refine_batch([src], grid_snap=grid_snap, target_size=size,
                           sample_method=sample_method, colors=colors,
                           snap_to=snap_to, pre_enhance=pre_enhance,
                           edge_rebuild=edge_rebuild)
    item = res["items"][0]
    out = item["image"]
    pix_meta = item["meta"]

    meta.update({
        "source": str(src_path),
        "generated": init_image is None,
        "prompt": prompt, "negative": negative,
        "model": model if init_image is None else None,
        "clip_skip": clip_skip if init_image is None else None,
        "seed": seed,
        "target": list(size), "colors_requested": int(colors),
        "palette": [list(c) for c in res["palette"]] if res["palette"] else [],
        "palette_size": len(res["palette"]) if res["palette"] else 0,
        "pre_enhance": pre_enhance, "edge_rebuild": edge_rebuild,
        "snap_to": snap_to, "grid_snap": grid_snap,
        "sample_method": sample_method,
        "pixel": pix_meta,
        "seconds_total": round(t_all(), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })
    if gen_meta:
        meta["source_seconds"] = gen_meta.get("seconds")
    return _save(out, _stamp(tag), "image_pixel", meta, scope=scope)


# 旧名保留：早期只用于动漫人物，现在通用了
anime_to_pixel = image_to_pixel


# ---------------- WPC 角色迁移 ----------------

def _qwen_pose_clone(eng: Engine, *, pose_image: str | Path, char_image: str | Path,
                     prompt: str, negative: Optional[str],
                     width: int, height: int, steps: int, cfg: float, seed: int,
                     ref_mode: str, thr: float, fill: float,
                     pixel: int, palette: int, matte: bool, matte_backend: str,
                     denoise: int, clean_iso: int, model: str, bg: str,
                     scope: str, tag: str) -> Dict[str, Any]:
    """Qwen-Image-2.1 版角色迁移：**双图条件**（角色图 + 姿态条件图）。

    为什么不用 SD 那套 ControlNet + IP-Adapter：
      - SD 那条路是**两个互相拉扯的约束**——IP-Adapter 抢身份、ControlNet 抢
        姿态，实测同一角色同一动作，出来的图经常是「身份丢了」或者「姿势变形」。
      - Qwen 的多图条件是**同一个 vision context**：两张图一起进模型，由它自己
        分配「谁是身份、谁是姿势」。实测（Qwen-2.1，768²/28 步）：姿势与骨架
        几乎逐点一致，连角色图的主色都被继承（蓝角色 → 蓝衣服）。

    姿态条件图仍然用本机的 DWPose（OpenPose 格式）：真人照片走骨架，
    像素/非人形素材走剪影（auto 模式自动降级），跟 SD 那条路一致。
    """
    ref = Image.open(pose_image).convert("RGB")
    cond, used_mode, cond_info = refcond.build_condition(
        ref, mode=ref_mode, size=max(width, height),
        # skeleton 模式输入就是画好的骨架，别再初始化 DWPose / FasterRCNN
        extractor=None if ref_mode == 'skeleton' else eng.pose_extractor(),
        thr=thr, fill=fill, matte_dir=eng.matte_dir)
    # 灰度化：Qwen 会把条件图当视觉参考，彩色骨架的关节色会被抄进成品
    cond = _qwen_cond_image(cond)
    char = refprep.prepare_reference(Image.open(char_image).convert("RGBA"), size=width)

    sem = (prompt or "").strip()
    instr = (
        "The first image is the character reference: keep the same face, hairstyle, "
        "outfit and colors. The second image is the pose condition (a body skeleton "
        "or silhouette). Redraw the same character in exactly that pose, "
        "full body, centered, "
        + ("clean simple black background" if (bg or "").lower() == "dark"
           else "clean simple background") + ".")
    if sem:
        instr += " " + sem

    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    out = _qwen_call(
        eng, prompt=instr, negative=negative or QWEN_NEG,
        ref_image=[char, cond], width=width, height=height, steps=steps,
        cfg=cfg, seed=seed, model=model, label="wpc")
    seconds = dt()
    width, height = out.size
    raw = out.copy() if (matte or pixel > 1 or palette) else None
    matte_info: Dict[str, Any] = {}
    out, edge, subjects = _postprocess(
        out, eng, matte=matte, matte_backend=matte_backend, denoise=denoise,
        pixel=pixel, palette=palette, clean_iso=clean_iso, info=matte_info)
    return _save(out, _stamp(tag), "pose_clone", {
        "workflow": "wpc", "scope": scope or "default",
        "prompt": instr, "negative": negative or QWEN_NEG,
        "pose_image": str(pose_image), "char_image": str(char_image),
        "backend": "qwen-multiref", "bg": bg,
        "ref_mode": ref_mode, "ref_mode_used": used_mode,
        "ref_fallback_from": cond_info.get("fallback_from"),
        "ref_fallback_reason": cond_info.get("fallback_reason"),
        "joints": cond_info.get("joints"), "fill": fill,
        "width": width, "height": height,
        "steps": steps, "cfg": cfg, "seed": seed,
        "model": model,
        "pixel_block": pixel, "palette": palette, "matte": matte,
        "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "edge_touch": edge, "subjects": subjects,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


def pose_clone(eng: Engine, *, pose_image: str | Path, char_image: str | Path,
                prompt: str, negative: Optional[str] = None,
                width: int = 768, height: int = 768,
                steps: int = 25, cfg: float = 7.0, seed: int = -1,
                cn_scale: float = 1.0, ipa_scale: float = 0.8,
                fill: float = 0.8, thr: float = 0.25,
                ref_mode: str = "auto", bg: str = "white",
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
    eng.require(model, "pose_clone")
    if eng.arch(model) == "qwen21":
        # Qwen 走双图条件（角色图 + 骨架图），不需要 ControlNet / IP-Adapter，
        # 也不吃 SD 的 LoRA 与提示词契约后缀——那条路在 Qwen 上是反效果。
        return _qwen_pose_clone(
            eng, pose_image=pose_image, char_image=char_image,
            prompt=prompt, negative=negative,
            width=width, height=height, steps=steps, cfg=cfg, seed=seed,
            ref_mode=ref_mode, thr=thr, fill=fill,
            pixel=pixel, palette=palette, matte=matte, matte_backend=matte_backend,
            denoise=denoise, clean_iso=clean_iso, model=model, bg=bg,
            scope=scope, tag=tag)
    if preset is not None:
        prompt = _lora_trigger(eng, prompt, lora)
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)
    # 角色图已经给了身份，提示词只是补语义。空串会让 CLIP 的文本条件近乎无效，
    # 画面容易退化成「一个模糊的人形」——所以给一句中性的兜底，而不是硬要用户写字。
    prompt = (prompt or "").strip() or "a single character, full body, facing viewer"

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
    # 预估值必须覆盖**实际要加载的量**，不能只留 2GB 余量：
    # SD15 的 UNet+ControlNet+IP-Adapter+VAE 约 6~7GB、SDXL 约 13GB。
    # 给 2GB 时，若 qwen 管线仍占着 16GB，检查会判"够用"而不驱逐，
    # 硬挤进去后 _fit 降级到 cpu offload —— 每一步都在 CPU/GPU 间搬权重，
    # 表现是「迁移中」转十几分钟、显存全满。
    eng.ensure_headroom(need_gb=13.0 if eng.is_xl(model) else 7.0,
                        keep=("pose",))
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

def _qwen_sprite_swap(eng: Engine, *, frames: Sequence[tuple],
                      sheet_image: str | Path, char_image: str | Path,
                      cols: int, rows: int, prompt: str, negative: Optional[str],
                      width: int, height: int, steps: int, cfg: float, seed: int,
                      ref_mode: str, model: str, matte: bool, matte_backend: str,
                      denoise: int, palette: int, keep_alpha: bool,
                      candidates: int, bg: str, scope: str,
                      tag: str) -> Dict[str, Any]:
    """Qwen 版序列帧替换（WSS）：逐帧「角色图 + 轮廓/骨架」双图条件。

    与 SD 那条路的差别只有一处但很关键：**没有 ControlNet 硬约束**。
    SD 是「轮廓条件钉死形状 + IP-Adapter 定身份」，两个约束各管一半；
    Qwen 是两张图一起进 vision context，由模型自己分配谁管身份、谁管动作。
    实测 3 帧骨架条件 DWPose 18/18、零贴边，动作贴合度够精灵产线用。

    **成本**：单帧 35s 级别，所以 candidates 默认 1（SD 那边是 2）。
    4 帧动画 = 约 2.5 分钟；8 帧 = 5 分钟。这是「Qwen 换 SD」绕不开的代价。
    """
    char = refprep.prepare_reference(Image.open(char_image).convert("RGBA"),
                                     size=max(width, height))
    char_orig = Image.open(char_image).convert("RGBA")
    extractor = eng.pose_extractor()
    sem = (prompt or "").strip()
    instr = ("The first image is the character reference: keep the same face, "
             "hairstyle, outfit and colors. The second image is the target pose "
             "(a skeleton or a silhouette). Redraw the same character in exactly "
             "that pose, full body, centered, " + _qwen_backdrop(bg) + "."
             + QWEN_FRAME_RULES)
    if sem:
        instr = f"{instr} {sem}"
    neg = (negative or "").strip() or QWEN_FRAME_NEG

    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    candidates = max(1, int(candidates))

    generated: List[Image.Image] = []
    per_frame: List[float] = []
    used_modes: List[str] = []
    frame_scores: List[Dict[str, Any]] = []
    attempts_used = 0
    matte_info: Dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()

    for _c, _r, frame in frames:
        cond, used_mode, _info = refcond.build_condition(
            frame, mode=ref_mode, size=max(width, height), extractor=extractor,
            matte_dir=eng.matte_dir)
        used_modes.append(used_mode)
        # 灰度化：彩色骨架会把颜色抄进成品（见 _qwen_cond_image）
        cond = _qwen_cond_image(cond.resize((width, height)))
        ref_area = int(framescore.subject_mask(frame.convert("RGBA")).sum())
        prev = generated[-1] if generated else None
        dt = _timer()
        cands: List[Tuple[Image.Image, Dict[str, Any]]] = []
        for k in range(candidates):
            raw = _qwen_call(
                eng, prompt=instr, negative=neg, ref_image=[char, cond],
                width=width, height=height, steps=steps, cfg=cfg,
                seed=seed + k * 1000, model=model, label="wss")
            info: Dict[str, Any] = {}
            cut, _edge, _subj = _postprocess(
                raw, eng, matte=matte, matte_backend=matte_backend,
                denoise=denoise, pixel=0, palette=0, clean_iso=0, info=info)
            for kk in ("backend", "fallback"):
                if kk in info:
                    matte_info[kk] = info[kk]
            ev = framescore.evaluate(cut, frame, char_orig, prev=prev,
                                    area_ref=ref_area)
            cands.append((cut, ev))
            if ev["passed"]:
                break
        best_img, best_ev, best_i = framescore.pick_best(cands)
        frame_scores.append({**best_ev, "attempts": len(cands), "picked": best_i})
        attempts_used += len(cands)
        per_frame.append(dt())
        generated.append(best_img)

    cut: List[Image.Image] = list(generated)
    if keep_alpha:
        # 用参考帧的 alpha 覆盖：形状逐像素精确，帧间零漂移。
        # **alpha 要缩到生成尺寸**——参考帧是格子尺寸（64²），生成图是 768²，
        # 直接 putalpha 会抛 "images do not match"。
        for i, (_c, _r, frame) in enumerate(frames):
            cut[i].putalpha(
                frame.getchannel("A").resize(cut[i].size, Image.NEAREST))
    else:
        # 尺度归一：逐帧的人物大小由模型自己决定，帧间能差几倍
        cut = pp.normalize_sequence(cut)

    # 缩回原格子尺寸：序列帧要塞回游戏，尺寸必须和原帧一致。
    # 用 pixelfy 的「按格采样」而不是 BOX 降采样——后者会把一格里的多种颜色
    # 混在一起，边界变脏（同 SD 路径的结论）。
    cell_w = max(f.width for _c, _r, f in frames)
    cell_h = max(f.height for _c, _r, f in frames)
    refined = pxf.refine_batch(
        [c.convert("RGB") for c in cut],
        grid_snap=False, target_size=(cell_w, cell_h), colors=palette)
    outs: List[Image.Image] = []
    for item, src in zip(refined["items"], cut):
        im = item["image"]
        if src.mode == "RGBA":
            im = im.convert("RGBA")
            a = src.getchannel("A").resize((cell_w, cell_h), Image.BILINEAR)
            im.putalpha(a)
            im = pxf.binarize_alpha(im)
        outs.append(im)

    aligned = pp.align_sequence(outs)
    meta = {
        "workflow": "wss", "scope": scope or "default",
        "mode": "generate", "engine_mode": "frame",
        "backend": "qwen-multiref", "approx": True, "bg": bg,
        "prompt": instr, "prompt_body": sem or None, "negative": neg,
        "sheet_image": str(sheet_image), "char_image": str(char_image),
        "grid": [cols, rows], "frames": len(aligned),
        "cell": [cell_w, cell_h],
        "ref_mode": ref_mode, "ref_modes_used": sorted(set(used_modes)),
        "model": model, "width": width, "height": height,
        "steps": steps, "cfg": cfg, "seed": seed,
        "cn_scale": None, "ipa_scale": None, "lora": None, "clip_skip": None,
        "matte": matte, "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "keep_alpha": keep_alpha,
        "palette": palette,
        "palette_size": len(refined["palette"]) if palette else 0,
        "candidates": candidates, "candidates_note": None,
        "gate_retries": 0, "attempts": attempts_used,
        "frame_scores": frame_scores,
        "gate_passed": sum(1 for s in frame_scores if s["passed"]),
        **framescore.sequence_consistency(outs),
        "per_frame_seconds": [round(s, 2) for s in per_frame],
        "seconds_total": round(sum(per_frame), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    return _pack_sprite_frames(aligned, frames, cols, rows, cell_w, cell_h,
                               meta=meta, scope=scope, tag=tag)


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
                bg: str = "white",
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
    # recolor 是纯确定性换色、**不跑模型**，任何底模都能用，所以不拦。
    # 只有 generate 模式要 ControlNet + IP-Adapter 这套 SD 组件。
    if mode != "recolor":
        eng.require(model, "sprite_swap")
    if preset is not None:
        prompt = _lora_trigger(eng, prompt, lora)
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)
    # 同 WPC：身份来自角色图（IP-Adapter），描述只是补语义；空串会让文本条件失效
    prompt = (prompt or "").strip() or "a single character, full body, facing viewer"

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

    if eng.arch(model) == "qwen21":
        # Qwen 没有 AnimateDiff（motion module 是 SD 系组件），只能逐帧。
        # 候选数强制 1：一帧 35s，默认 2 个候选会把 8 帧动画拖到 9 分钟。
        return _qwen_sprite_swap(
            eng, frames=frames, sheet_image=sheet_image, char_image=char_image,
            cols=cols, rows=rows,
            prompt=prompt, negative=negative, width=width, height=height,
            steps=steps, cfg=cfg, seed=seed, ref_mode=ref_mode, model=model,
            matte=matte, matte_backend=matte_backend, denoise=denoise,
            palette=palette, keep_alpha=keep_alpha, candidates=1,
            bg=bg, scope=scope, tag=tag)

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

    # 预估值必须覆盖**实际要加载的量**，不能只留 2GB 余量：
    # SD15 的 UNet+ControlNet+IP-Adapter+VAE 约 6~7GB、SDXL 约 13GB。
    # 给 2GB 时，若 qwen 管线仍占着 16GB，检查会判"够用"而不驱逐，
    # 硬挤进去后 _fit 降级到 cpu offload —— 每一步都在 CPU/GPU 间搬权重，
    # 表现是「迁移中」转十几分钟、显存全满。
    eng.ensure_headroom(need_gb=13.0 if eng.is_xl(model) else 7.0,
                        keep=("pose",))
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
        prompt = _lora_trigger(eng, prompt, lora)
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

    # **别把 RGBA 转成 RGB 再送进去**：convert("RGB") 会把透明区变成纯黑，
    # 而序列帧的透明区常常占 70%+ ——色板会被背景黑吃掉大半色位（实测主体
    # 只分到 2 色）。refine_batch 内部会按 alpha 排除透明像素再提色板。
    refined = pxf.refine_batch([c.convert("RGBA") for c in cut], grid_snap=False,
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


# Qwen 上的姿态条件只能靠「图 + 指令」软引导（没有 ControlNet），
# 所以指令必须把「第二张图是什么、要照着做」说清楚，否则模型会把它当画风参考。
QWEN_POSE_INSTR = {
    "skeleton": ("The images are: first the character reference, then a body skeleton "
                 "showing the target pose. Redraw the same character in exactly that "
                 "pose, full body, centered, {bg}."),
    "outline": ("The images are: first the character reference, then the silhouette "
                "of the target pose. Redraw the same character in exactly that pose, "
                "full body, centered, {bg}."),
    "solo": ("This image is a body skeleton showing a pose. Draw a single character "
             "in exactly that pose, full body, centered, {bg}."),
}

# 逐帧的底色。**白毛/白甲的角色不要用白底**：分割模型分不出主体与背景，
# 会把额头、脸颊、甚至整条腿判成背景（实测白猫骑士：丢头顶+胸口，或丢两条腿）。
# 换成深色底后，色键与语义分割都有可靠信号。
QWEN_BACKDROPS = {
    "white": "plain white background",
    "dark": "plain solid black background",
}


def _qwen_backdrop(bg: str) -> str:
    """把 bg 参数翻成提示词里的背景短语（未知值按原文用，方便自定义）。"""
    return QWEN_BACKDROPS.get((bg or "white").strip().lower(),
                              (bg or "").strip() or QWEN_BACKDROPS["white"])

# 构图约束：**必须有**。Qwen 的软引导下，只写「full body, centered」实测会画出
# 拇指大的小人（前景占比 2%，低于精灵产线 5% 的下限），而且帧与帧的尺度互相
# 差 3 倍。把「占画面高度」写成具体数字后，尺度才稳定（配合
# postprocess.normalize_sequence 做确定性兜底）。
QWEN_FRAME_RULES = (" The figure must fill about 75% of the image height, standing "
                    "with the feet near the bottom of the frame.")

# 逐帧路线的负向：除了通用质量词，重点是「缺件/散架」——实测 Qwen 在骨架上
# 偶尔会把人头画丢，或者把帽子/手画成浮空的一块。
QWEN_FRAME_NEG = (QWEN_NEG + ", floating parts, detached head, disconnected limbs, "
                  "extra head, missing head, two characters")


def _qwen_cond_image(img: Image.Image) -> Image.Image:
    """把条件图统一成「白底 + 深色线条」。

    **为什么必须做**（实测踩出来的）：OpenPose 骨架是**黑底 + 彩色**的
    （`tools/make_pose.render` 用一组彩色画肢体和关节点）。Qwen 把它当**视觉
    参考**而不是控制信号，于是会顺手把颜色抄进成品——实测有一帧真的画了个
    「红色小帽子」浮在头顶，那其实是骨架图上红色的鼻子关节点。

    转灰度 + 反相到白底之后，模型只能读到「线在哪」，抄不到颜色。
    亮度均值判方向：黑底骨架会被反相，用户自己画的浅底骨架保持原样。
    """
    g = img.convert("L")
    if float(np.asarray(g, dtype=np.float32).mean()) < 110:
        from PIL import ImageOps
        g = ImageOps.invert(g)
    return g.convert("RGB")


def _qwen_poseswap(eng: Engine, *, pose_images: Sequence[Path], prompt: str,
                   negative: Optional[str], ref_image: Optional[str | Path],
                   steps: int, cfg: float, seed: int, width: int, height: int,
                   cols: int, pixel: int, palette: int, matte: bool,
                   matte_backend: str, denoise: int, clean_iso: int, bg: str,
                   export: Optional[str], controlnet: str, model: str,
                   candidates: int, gate: bool, gate_min_points: int,
                   gate_retries: int, gate_seed_step: int, scope: str,
                   anim_name: str, fps: int, tag: str) -> Dict[str, Any]:
    """Qwen 版姿态序列（W4）：逐帧「骨架图当参考 + 指令」软引导。

    **它是近似实现**：SD 那条路是 ControlNet 硬约束（每个关节都钉住），
    这里只能是「第二张图是骨架 + 指令说照着这个姿势画」。实测 3 帧
    DWPose 18/18 全中、零贴边，姿势一致性够用；但没有硬约束意味着
    复杂动作（跳跃、交叉腿）仍可能跑偏，闸门不过就补提。

    **成本**：35s 级别的单帧成本，所以默认 candidates=1、补提上限压到 1。
    SD 时代那套「一帧出 4 个候选再挑」在这里等于把 8 帧动画变成 20 分钟。
    """
    kind = "skeleton" if controlnet == "openpose" else "outline"
    head = QWEN_POSE_INSTR[kind].format(bg=_qwen_backdrop(bg)) + QWEN_FRAME_RULES
    sem = (prompt or "").strip()
    neg = (negative or "").strip() or QWEN_FRAME_NEG

    char = None
    if ref_image is not None:
        char = refprep.prepare_reference(Image.open(ref_image).convert("RGBA"),
                                         size=max(width, height))
        head = head.replace(
            "The images are: first the character reference, then ",
            "The first image is the character reference: keep the same face, "
            "hairstyle, outfit and colors. The second image is ")
    # 用户正文追加在最后：身份/姿态/构图是必须的，风格词是补充。
    instr = f"{head} {sem}".strip() if sem else head

    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    candidates = max(1, int(candidates))
    extractor = eng.pose_extractor() if gate else None
    if gate and extractor is None:
        print("[warn] 质量闸门不可用（姿态模型缺失），本次只按分数挑帧")
    # 35s/帧的前提下，补提额度必须收紧：没有判据的重提就是烧掉一分钟
    retries = max(0, min(1, int(gate_retries))) if extractor is not None else 0
    attempts_max = candidates + retries

    matte_info: Dict[str, Any] = {}
    frames: List[Image.Image] = []
    per_frame: List[float] = []
    frame_scores: List[Dict[str, Any]] = []
    attempts_used = 0
    prev_gate: Optional[Image.Image] = None
    torch.cuda.reset_peak_memory_stats()

    for pose_path in pose_images:
        cond = _qwen_cond_image(
            Image.open(pose_path).convert("RGB").resize((width, height),
                                                        Image.LANCZOS))
        refs: Any = [char, cond] if char is not None else cond
        best: Optional[tuple[Dict[str, Any], Image.Image, Image.Image]] = None
        spent = 0.0
        used = 0
        for round_seeds in (list(range(candidates)),
                            list(range(candidates, attempts_max))):
            for i in round_seeds:
                if used >= attempts_max:
                    break
                dt = _timer()
                raw = _qwen_call(
                    eng, prompt=instr, negative=neg, ref_image=refs,
                    width=width, height=height, steps=steps, cfg=cfg,
                    seed=seed + i * gate_seed_step, model=model, label="w4")
                info: Dict[str, Any] = {}
                img, _edge, _subj = _postprocess(
                    raw, eng, matte=matte, matte_backend=matte_backend,
                    denoise=denoise, pixel=pixel, palette=palette,
                    clean_iso=clean_iso, info=info)
                for k in ("backend", "fallback"):
                    if k in info:
                        matte_info[k] = info[k]
                gate_img = info.pop("pre_pixel", None) or img
                ev = quality.evaluate(gate_img.convert("RGBA"), extractor,
                                      prev=prev_gate,
                                      min_points=gate_min_points)
                # **单主体判据**：Qwen 偶尔会把头画丢、把帽子画成浮空的一块。
                # 这种帧 DWPose 照样能检出 16 个关节（躯干还在），闸门放它过去；
                # 但前景连通域数会变成 2+，这才是「缺件」的直接证据。
                if _subj is not None:
                    ev["blobs"] = _subj
                    if _subj > 1 and ev["passed"]:
                        ev = {**ev, "passed": False,
                              "score": ev["score"] - 20.0,
                              "reject": f"前景有 {_subj} 块（头/手与身体分离）"}
                spent += dt()
                used += 1
                if best is None or ev["score"] > best[0]["score"]:
                    best = (ev, img.convert("RGBA"), gate_img.convert("RGBA"))
                if extractor is not None and ev["passed"] and used >= candidates:
                    break
            if used >= candidates and (extractor is None or best[0]["passed"]):
                break
        ev, frame, prev_gate = best
        frame_scores.append({**ev, "attempts": used})
        attempts_used += used
        per_frame.append(round(spent, 2))
        frames.append(frame)

    # 尺度归一：Qwen 没有 ControlNet，逐帧「人物占多大」是模型自己决定的
    # （实测同一段 3 帧能差 3 倍，最小的那帧前景只占 2%，直接被产线判废）。
    frames = pp.normalize_sequence(frames)
    aligned = pp.align_sequence(frames)
    passed = sum(1 for s in frame_scores if s["passed"])
    meta = {
        "workflow": "w4", "scope": scope or "default",
        "prompt": instr, "prompt_body": sem or None, "negative": neg,
        "ref_image": str(ref_image) if ref_image else None,
        "backend": "qwen-pose-ref", "approx": True,
        "condition": kind, "bg": bg, "cn_scale": None, "ipa_scale": None,
        "lora": None, "lora_scale": None, "clip_skip": None,
        "steps": steps, "cfg": cfg, "seed": seed, "frames": len(frames),
        "width": width, "height": height,
        "matte": matte, "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "model": model, "distill": None,
        "candidates": candidates, "candidates_note": None,
        "gate": bool(gate and extractor is not None),
        "gate_min_points": gate_min_points, "gate_retries": retries,
        "attempts": attempts_used, "gate_passed": passed,
        "frame_scores": frame_scores,
        "pixel_block": pixel, "palette": palette,
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
             bg: str = "white",
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
    eng.require(model, "poseswap")
    # 输入可以是目录 / 通配 / 列表：前端只传目录，CLI 传列表，同一套规则
    pose_images = norm_pose_inputs(pose_images)
    if not pose_images:
        raise ValueError("没有可用的骨架帧（目录里没有 png/jpg，或通配没匹配到）")
    # cols<=0 表示"按帧数自动排一行"。以前直接传给 pack_sheet 会在
    # `(len+cols-1)//cols` 上 ZeroDivisionError——前端把 0 当"自动"传，必须兜住。
    if cols <= 0:
        cols = min(len(pose_images), 8)
    if eng.arch(model) == "qwen21":
        # Qwen 不吃 SD 的 preset 契约后缀（那是 CLIP 的词表游戏），也不吃 LoRA
        return _qwen_poseswap(
            eng, pose_images=pose_images, prompt=prompt, negative=negative,
            ref_image=ref_image, steps=steps, cfg=cfg, seed=seed,
            width=width, height=height, cols=cols, pixel=pixel, palette=palette,
            matte=matte, matte_backend=matte_backend, denoise=denoise,
            clean_iso=clean_iso, bg=bg, export=export, controlnet=controlnet,
            model=model, candidates=candidates, gate=gate,
            gate_min_points=gate_min_points, gate_retries=gate_retries,
            gate_seed_step=gate_seed_step, scope=scope,
            anim_name=anim_name, fps=fps, tag=tag)
    negative = negative or NEG_DEFAULT
    if preset is not None:
        prompt = _lora_trigger(eng, prompt, lora)
        prompt, negative = presets.build(prompt, negative, preset,
                                    max_tokens=presets.CHUNK_BUDGET)
    # 预估值必须覆盖**实际要加载的量**，不能只留 2GB 余量：
    # SD15 的 UNet+ControlNet+IP-Adapter+VAE 约 6~7GB、SDXL 约 13GB。
    # 给 2GB 时，若 qwen 管线仍占着 16GB，检查会判"够用"而不驱逐，
    # 硬挤进去后 _fit 降级到 cpu offload —— 每一步都在 CPU/GPU 间搬权重，
    # 表现是「迁移中」转十几分钟、显存全满。
    eng.ensure_headroom(need_gb=13.0 if eng.is_xl(model) else 7.0,
                        keep=("pose",))
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
