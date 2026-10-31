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
from . import matte as matte_mod
from . import pixelfy as pxf
from . import postprocess as pp
from . import refcond
from . import repaint as rp
from . import presets
from . import quality
from . import framescore
from . import gate as gate_mod
from . import refprep
from . import qwen_mask
from . import outpaint as op_mod
from . import swap as swap_mod
from . import sheet as sheet_mod
from . import paths
from . import projectstore
from .engine import DEFAULT_MODEL, Engine

# 资产根目录**只在 paths 里定义一次**。以前这里和 app.py 各写一份
# `ROOT/"assets"`，导致「把测试产物写到别处」得改两个地方，漏一个就污染真目录。
ASSETS = paths.ASSETS
PROJECTS_DIR = paths.PROJECTS_DIR



_seq = itertools.count()

#: 快速模式用的少步蒸馏 LoRA（lightx2v/Qwen-Image-2.1-Lightning，64MB）。
#: 它是 **diffusers 原生格式**，可直接 `load_lora_weights`。
FAST_LORA = "qwen21_lightning"
#: 快速模式的步数上限（Lightning 在 8 步就收敛，实测画质几乎不掉）
FAST_STEPS = 8


def _fast_meta(eng: Engine, applied: bool = True) -> Dict[str, Any]:
    """快速模式的状态记进 meta（产线判据要看这个）。

    `applied`：**本次调用是否真的走了少步分支**。有几条路不走——
    真蒙版 inpaint（`qwen_mask.masked_inpaint` 自己跑管线，不挂 Lightning）、
    VNCCS（LoRA 槽位归它自己）、扩图（挂的是 Outpaint LoRA）。
    那些路即使全局开关开着也是满步数，meta 不写清楚就会把成本算错。
    """
    on = bool(getattr(eng, "fast_mode", False))
    used = bool(on and applied)
    return {"fast_mode": on, "fast_applied": used,
            "fast_lora": FAST_LORA if used else None}


def _clamp_steps(steps: int) -> int:
    """管线侧的步数钳制。

    `_qwen_call` 与 `qwen_mask.masked_inpaint` 都是 `max(8, min(steps, 60))`：
    低于 8 步 Qwen 出糊块，高于 60 步收益趋零。meta 必须记**钳制后**的值。
    """
    return max(8, min(int(steps), 60))


def _eff_steps(eng: Engine, steps: int, fast: bool = True) -> int:
    """**实际执行**步数（非请求值）。

    meta 里必须记这个而不是用户请求的步数——产线判据（每可用帧成本）看的是
    真正跑了多少步。两个地方会改它：管线的钳制（[8, 60]）与快速模式（压到 8）。
    `fast=False` 用于**不走**少步分支的路（见 `_fast_meta`）。
    """
    n = _clamp_steps(steps)
    if fast and getattr(eng, "fast_mode", False):
        n = min(n, FAST_STEPS)
    return n


def scope_dir(scope: str = "") -> Path:
    """项目产物目录：assets/projects/<scope>/。

    scope 是前端的项目 id（p…），只保留 ASCII 字母数字与 -_，空则用 default。
    每个项目的 PNG / sidecar / raw / 帧序列 / 精灵图都在自己的目录里，
    项目之间互不可见（后端还有一层校验，见 img2img 的源图检查）。

    **清洗规则不在本文件实现**：调 `projectstore.safe_scope`（全项目唯一一份），
    否则产物和 `project.json` 会因两套规则算出不同目录名而分家。
    """
    d = PROJECTS_DIR / projectstore.safe_scope(scope)
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

    进来先做一次**浮点归一化**（见 `_as_float`）：JSON 没有 int/float 之分，
    前端 slider 的 `1.0` 到这里是 `int 1`，而 diffusers 对
    `controlnet_conditioning_scale` 用 `isinstance(x, float)` 严格校验，
    直接抛 TypeError。放在这里是单点收口——所有 ControlNet 调用都过它。
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


# 质量档位（实测，RTX 4090，fp16，DPM++ 2M Karras）：
#   草稿  512² /  8 步  —— 0.30s，只用来验证构图
#   可用  768² / 30 步  —— 1.8s，SD1.5 的甜点：再大就开始重复肢体（1024² 实测出 3~4 个猫头）
#   精修  768² / 45 步 + cfg 6.5 —— 2.6s，细节更稳，代价是慢
# 像素化统一用 pixel=6 / palette=32（768 ÷ 6 = 128 有效像素，正好是精灵尺寸）；
# 512² 配 pixel=4 会把有效像素压到 128 但源信息本来就少，实测明显更糊。


# ---------------- Qwen-Image-2.1 ----------------

# 透明背景的「双段式声明」。
#
# Qwen-Image-2.1 的 VAE 有第 4 通道（alpha），但它**不会自己想起来用**——
# 必须在提示词首尾各加一句显式声明，否则输出恒为不透明（实测：不写这段，
# 提示词里写 "transparent background" 也只是画一块白底）。
# 这两句是从官方卡片抄来的原话，别改写：换说法会掉回不透明。
QWEN_ALPHA_HEAD = "This is an RGBA image with transparency. "
QWEN_ALPHA_TAIL = " The image has alpha channel and the background is transparent."

# Qwen 的负向词：**只在真 CFG 打开时才传**（cfg > 1，见 _qwen_call）。
# 原因：真 CFG 会让 transformer 前向翻倍（时间 +40%），而实测单次前向
# 的画质并不差——所以默认 cfg=1.0，这个负向词默认不参与计算。
# 内容别照搬 SD 那套——它不理解 "16-bit pixel art" 这类风格词，
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
               label: str = "w1", cfg: float = 1.0,
               output_resolution: Optional[int] = None,
               fast: bool = False, fast_lora: bool = True) -> Image.Image:
    """Qwen 单次推理。带 ref_image 就走「指令改图」，不带就是文生图。

    三个**必须**记住的点（都是实测踩出来的）：
      1. **真 CFG 的代价是时间翻倍**：管线在 `true_cfg_scale > 1` 且给了负向词时
         每个 step 跑两遍 transformer（有条件 + 无条件）。实测 768²/24 步：
         21.5s（真 CFG）vs 13.0s（单次前向），DiT 前向占 86% 的总时间。
         画质对照（6 个提示词）没掉，所以默认走单次前向（cfg=1.0）。
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
    # **快速模式**：挂 Lightning 少步蒸馏 LoRA + 把步数压到 8。
    # 实测（768²/RTX4090）：24 步 17.3s → 8 步 9.3s（-46%），画质几乎不掉。
    # 逐帧路线（w4/wss）一直卡在 37s/帧，8 步后单帧 9.3s，8 帧 ≈ 75s。
    #
    # **状态式**（与 LoRA 同一套）：`fast` 由调用方写进 `eng.fast_mode`，
    # 这样不必把它穿过每条工作流的参数链。
    #
    # `fast_lora=False` 让调用方**独占 LoRA 槽位**：VNCCS 姿态迁移要挂自己的
    # LoRA，被这里再挂一个 Lightning 会把它顶掉（两个 adapter 同时生效也会
    # 让姿态遵循度崩掉）。
    if fast_lora:
        if fast or getattr(eng, "fast_mode", False):
            # Lightning 走**临时槽**：用户挂的 LoRA 不会被顶掉
            eng.apply_loras(pipe, [{"name": FAST_LORA, "scale": 1.0}],
                            scope="temp")
            steps = min(int(steps), FAST_STEPS)
        else:
            # 快速模式关了 → 顺手清掉临时槽里残留的 Lightning
            # （清空后如果用户还挂着别的 LoRA，_sync_loras 会重新挂回它们）
            eng.unload_loras(pipe, scope="temp")
    gen = torch.Generator("cuda").manual_seed(int(seed))
    kw: Dict[str, Any] = dict(
        prompt=full,
        width=w, height=h,
        num_inference_steps=max(8, min(int(steps), 60)),
        true_cfg_scale=float(cfg),
        generator=gen,
        # 条件图分辨率与目标对齐（见上第 3 点）
        output_resolution=int(output_resolution or max(w, h)),
    )
    # **负向词只在真 CFG 打开时才传**。
    # 真 CFG（true_cfg_scale > 1）会让管线每个 step 跑两遍 transformer
    # （有条件 + 无条件）——24 步就是 48 次前向，实测占 86% 的时间。
    # cfg = 1.0 时只有一次前向（21.5s → 13.0s，快 40%），实测画质不掉
    # （6 个提示词对照见 assets/_cfg_ab/cmp.png）。
    # 不传负向词，就不会有「双次前向」这件事——不依赖 diffusers 内部的
    # do_true_cfg 判定，行为自己说得清。
    if float(cfg) > 1.0:
        kw["negative_prompt"] = negative or QWEN_NEG

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


def _qwen_work_size(size: tuple[int, int], *, cap: int = 1024) -> tuple[int, int]:
    """把源图尺寸夹到 Qwen 的划算区间（**32 的倍数**，长边 ≤ cap）。

    为什么要有上限：Qwen 的开销随面积涨，1024² 的图 token 是 768² 的 1.78 倍，
    而像素素材原生只有 32~128px——放大到 1024 并不会多出细节，只会把
    「按 1024 画的笔触」再缩回去，白烧一倍显存与时间。

    **为什么是 32 而不是 16**：管线的 `multiple_of = vae_scale_factor * 2 = 32`
    （diffusers `pipeline_qwenimage21.py`）。给 16 的倍数时管线会**静默 resize**
    到 32 的倍数，于是 meta 里记的尺寸与实际出的图对不上。
    """
    w, h = size
    f = min(1.0, cap / max(1, max(w, h)))

    def snap(v: float) -> int:
        # **向下取整到 32 的倍数**：向上取整会让「不缩放」的图反而被放大几 px，
        # 白白多算 token。宁可少一点也不放大（像素素材放大不会多出信息）。
        return max(32, int(v * f) // 32 * 32)

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
                  cfg: float = 1.0, model: str, scope: str,
                  tag: str) -> Dict[str, Any]:
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
                     transparent=transparent, model=model, label="w1",
                     cfg=cfg)
    seconds = dt()
    width, height = img.size
    # 透明底的原图自带 alpha，再跑抠图会把 alpha 当背景重算一遍，纯属浪费
    raw = img.copy() if (pixel > 1 or palette) else None
    info: Dict[str, Any] = {}
    img2, edge, subjects = _postprocess(
        img, eng, matte=(matte and not transparent), matte_backend=matte_backend,
        denoise=denoise, pixel=pixel, palette=palette, clean_iso=clean_iso, info=info)
    # 真 CFG（cfg>1）时管线才会收到负向词；cfg=1 时它一次都没传过，
    # meta 里再写一份负向词会让人以为它生效了。
    neg_used = (negative or QWEN_NEG) if float(cfg) > 1.0 else None
    return _save(img2, _stamp(tag), "txt2img", {
        "workflow": "w1", "prompt": _qwen_prompt(prompt, transparent),
        "negative": neg_used,
        "negative_applied": bool(float(cfg) > 1.0),
        "width": width, "height": height, "steps": _eff_steps(eng, steps), **_fast_meta(eng),
        "cfg": float(cfg),
        # 真 CFG 开着时每个 step 跑两遍 transformer（时间翻倍）
        "true_cfg": bool(float(cfg) > 1.0),
        "seed": seed,
        "lora": None, "lora_scale": None,
        "pixel_block": pixel, "palette": palette,
        # 透明底时抠图会被跳过（原始 alpha 直接复用），meta 要记**实际**值
        "matte": bool(matte and not transparent),
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
            width: int = 768, height: int = 768, steps: int = 16, cfg: float = 1.0,
            seed: int = -1, pixel: int = 0, palette: int = 0, preset: str = "sprite",
            matte: bool = False, matte_backend: str = "auto", denoise: int = 1,
            clean_iso: int = 1, transparent: bool = False,
            model: str = DEFAULT_MODEL, scope: str = "", tag: str = "w1"
            ) -> Dict[str, Any]:
    """文生图（Qwen-Image-2.1 原生 DiT）。

    参数里没有 `lora` / `clip_skip` / `hires` / `distill` / `refiner`：那是 SD
    时代的开关，Qwen 一个都不吃（不同主干、不同文本编码器、没有精修器）。
    前端如果还传这些键，`app._run` 会按签名过滤掉并记进 `meta.params_ignored`。
    """
    eng.require(model, "txt2img")
    return _qwen_txt2img(
        eng, prompt=prompt, negative=negative, width=width, height=height,
        steps=steps, seed=seed, transparent=transparent, pixel=pixel,
        palette=palette, preset=preset, matte=matte,
        matte_backend=matte_backend, denoise=denoise, clean_iso=clean_iso,
        cfg=cfg, model=model, scope=scope, tag=tag)


# ---------------- W2 图生图 ----------------

def _qwen_img2img(eng: Engine, *, init_image: str | Path, prompt: str,
                  negative: Optional[str], steps: int, seed: int,
                  transparent: bool, preset: Optional[str], pixel: int,
                  palette: int, matte: bool, matte_backend: str, denoise: int,
                  clean_iso: int, cfg: float = 1.0, model: str, scope: str,
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
                     transparent=transparent, model=model, label="w2",
                     cfg=cfg)
    seconds = dt()
    # **raw 一律存**：它是「后处理之前的原图」，前端图生图节点的
    # 「用原图（未像素化），保留细节」指的就是它。以前只在 pixel>1 或 palette>0
    # 时才存，而 Qwen 默认 pixel=0（不像素化）→ 默认配置下 raw 根本不落盘，
    # 下游那条路直接断掉。多存一张 PNG 的代价可以忽略。
    raw = img.copy()
    info: Dict[str, Any] = {}
    img, edge, subjects = _postprocess(
        img, eng, matte=(matte and not transparent), matte_backend=matte_backend,
        denoise=denoise, pixel=pixel, palette=palette, clean_iso=clean_iso, info=info)
    neg_used = (negative or QWEN_NEG) if float(cfg) > 1.0 else None
    return _save(img, _stamp(tag), "img2img", {
        "workflow": "w2", "prompt": _qwen_prompt(prompt, transparent),
        "negative": neg_used,
        "negative_applied": bool(float(cfg) > 1.0),
        "init_image": str(init_image), "strength": None,
        "steps": _eff_steps(eng, steps), **_fast_meta(eng),
        "cfg": float(cfg), "true_cfg": bool(float(cfg) > 1.0),
        "seed": seed, "lora": None,
        "lora_scale": None,
        # 实际输出尺寸（w2 之前根本没记，产线无从得知）
        "width": img.width, "height": img.height,
        "pixel_block": pixel, "palette": palette,
        "matte": bool(matte and not transparent),
        "matte_backend": matte_backend, "matte_used": info.get("backend"),
        "matte_fallback": info.get("fallback"),
        "edge_touch": edge, "subjects": subjects, "model": model,
        "clip_skip": None, "transparent": transparent,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


def img2img(eng: Engine, *, init_image: str | Path, prompt: str,
            negative: Optional[str] = None, steps: int = 16, cfg: float = 1.0,
            seed: int = -1, model: str = DEFAULT_MODEL,
            preset: Optional[str] = None, pixel: int = 0, palette: int = 0,
            matte: bool = False, matte_backend: str = "auto", denoise: int = 1,
            clean_iso: int = 1, transparent: bool = False,
            scope: str = "", tag: str = "w2") -> Dict[str, Any]:
    """图生图 = **指令改图**：源图当参考，提示词当指令。

    Qwen 没有 `strength`（加噪幅度）这个概念——改多少完全由提示词决定，
    所以这里不接 strength。输出尺寸跟源图走（16 的倍数）。
    """
    eng.require(model, "img2img")
    return _qwen_img2img(
        eng, init_image=init_image, prompt=prompt, negative=negative,
        steps=steps, seed=seed, transparent=transparent, preset=preset,
        pixel=pixel, palette=palette, matte=matte,
        matte_backend=matte_backend, denoise=denoise,
        clean_iso=clean_iso, cfg=cfg, model=model, scope=scope, tag=tag)


# ---------------- W3 局部重绘 ----------------

def _qwen_inpaint(eng: Engine, *, image: str | Path, mask: str | Path, prompt: str,
                  negative: Optional[str], steps: int, cfg: float, seed: int,
                  model: str, feather: int, match_color: bool,
                  match_strength: float, transparent: bool,
                  scope: str, tag: str,
                  strategy: str = "mask") -> Dict[str, Any]:
    """Qwen 版局部重绘。

    两种策略：

      · ``mask``（默认，**真 inpaint**）：走 `qwen_mask.masked_inpaint`——
        蒙版内换成噪声、每步把蒙版外拉回原 latent（ComfyUI 的 no-offset 做法）。
        **模型自己知道哪儿是选区**，所以指令能被准确执行，身份也不丢。
        实测 768²/20 步 ~15s，蒙版外最大像素差 19（视觉上完全一致）。
      · ``whole``（旧方案，保留做对照）：整图指令改图 + 蒙版贴回。
        选区外靠 `repaint.composite` 保证逐点不变，但**模型看不到蒙版**，
        改哪儿靠提示词里的方位词猜 → 指令可能不执行。

    两条路径的承诺不同，meta 里分别记录（`backend` / `approx`）。
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
    region = _region_words(msk)
    if strategy == "mask":
        # **真蒙版路径的工作尺寸下限是 768**（实测 512 蒙版边缘表达不出来，
        # 选区不会被改）。见 workflows/qwen_mask.py 的说明。
        work = (max(768, _qwen_work_size(src.size)[0]),
                max(768, _qwen_work_size(src.size)[1]))
        full_prompt = _qwen_edit_prompt(body, region)
        pipe = eng.txt2img_pipe(model)
        gen = qwen_mask.masked_inpaint(
            pipe, src=src, mask=msk, prompt=full_prompt,
            negative=negative or QWEN_EDIT_NEG, steps=steps, cfg=cfg, seed=seed,
            work=work, feather=2)
        out = gen.convert("RGBA")
        if not transparent:
            out = _restore_alpha(out, src)
        approx = False
        backend = "qwen-mask-inpaint"
        # masked_inpaint 自己跑管线，**不挂 Lightning** → 快速模式对它无效
        used_fast = False
    else:
        gen, work, full_prompt, region = _qwen_local_edit(
            eng, src=src, mask=msk, body=body, negative=negative, steps=steps,
            cfg=cfg, seed=seed, model=model, label="w3")
        out = rp.composite(src, gen.convert("RGBA"), msk, feather_px=feather,
                           match_color=match_color, match_strength=match_strength)
        if not transparent:
            out = _restore_alpha(out, src)
        approx = True
        backend = "qwen-edit"
        used_fast = True          # 走 _qwen_call，会挂 Lightning
    seconds = dt()
    # 自检：承诺写进 meta，别让用户去猜「到底有没有溢出」
    diff = _outside_diff(src, out, msk)
    return _save(out, _stamp(tag), "inpaint", {
        "workflow": "w3", "prompt": full_prompt, "prompt_body": body,
        "negative": negative or QWEN_EDIT_NEG,
        "image": str(image), "mask": str(mask),
        # 实际输出尺寸（w3/w5 以前完全不记，产线无从得知）
        "width": out.width, "height": out.height,
        "strength": None,
        "steps": _eff_steps(eng, steps, fast=used_fast),
        **_fast_meta(eng, applied=used_fast),
        "cfg": cfg, "seed": seed,
        "model": model, "backend": backend, "approx": approx,
        "strategy": strategy,
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
            negative: Optional[str] = None, steps: int = 16, cfg: float = 1.0,
            seed: int = -1, model: str = DEFAULT_MODEL, scope: str = "",
            feather: int = 4, match_color: bool = True,
            match_strength: float = 0.45, transparent: bool = False,
            tag: str = "w3", strategy: str = "mask") -> Dict[str, Any]:
    """局部重绘（默认 **真 inpaint**：蒙版内换噪声 + 每步蒙版外拉回原 latent）。

    选区外与原图几乎逐点一致（`meta.outside_max_diff`，实测 19，视觉无差）；
    选区内模型**看得见蒙版**，指令能被准确执行。
    `strategy="whole"` 退回旧的「整图改图 + 蒙版贴回」（近似，留作对照）。
    """
    eng.require(model, "inpaint")
    return _qwen_inpaint(
        eng, image=image, mask=mask, prompt=prompt, negative=negative,
        steps=steps, cfg=cfg, seed=seed, model=model, feather=feather,
        match_color=match_color, match_strength=match_strength,
        transparent=transparent, scope=scope, tag=tag, strategy=strategy)


# ---------------- WBG 抠图去背景 / 换背景 ----------------

def matte_bg(eng: Engine, *, init_image: str | Path,
             background: str | Path | None = None,
             bg_color: str | Sequence[int] | None = None,
             matte_backend: str = "auto", denoise: int = 1,
             alpha_mode: str = "binary", feather: int = 0,
             model: str = DEFAULT_MODEL,
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
    eng.require(model, "matte")
    dt = _timer()
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
        "seconds": round(dt(), 2),
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
    """把 #rrggbb / #rgb / rgb(r,g,b) / (r,g,b) 统一成 RGB 三元组。

    **非法输入直接报错**，不静默回退成白色：以前解析失败就返回白底，
    而 meta 里记的是原始字符串，看起来像是生效了——实际画出来是白的。
    """
    if isinstance(c, (tuple, list)):
        if len(c) < 3:
            raise ValueError(f"颜色元组至少要 3 个分量：{c}")
        vals = [int(c[0]), int(c[1]), int(c[2])]
        for v in vals:
            if not 0 <= v <= 255:
                raise ValueError(f"颜色分量超出 0~255：{c}")
        return (vals[0], vals[1], vals[2])
    s = str(c).strip()
    if s.lower().startswith("rgb("):
        body = s[4:-1] if s.endswith(")") else s[4:]
        parts = [x.strip() for x in body.split(",")]
        if len(parts) != 3:
            raise ValueError(f"颜色格式应为 rgb(r,g,b)：{c}")
        try:
            vals = [int(x) for x in parts]
        except ValueError as exc:
            raise ValueError(f"颜色分量不是整数：{c}") from exc
        for v in vals:
            if not 0 <= v <= 255:
                raise ValueError(f"颜色分量超出 0~255：{c}")
        return (vals[0], vals[1], vals[2])
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(
            f"颜色格式应为 #rrggbb / #rgb / rgb(r,g,b)：{c}")
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError as exc:
        raise ValueError(f"颜色里有非十六进制字符：{c}") from exc


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
    if strategy == "mask":
        # **真蒙版路径**：模型看得见蒙版 → 指令能被准确执行，身份不丢。
        # 与 w3 同一个机制（workflows/qwen_mask.py）。
        work = (max(768, _qwen_work_size(src.size)[0]),
                max(768, _qwen_work_size(src.size)[1]))
        region = _region_words(msk)
        full_prompt = _qwen_edit_prompt(body, region)
        pipe = eng.txt2img_pipe(model)
        gen = qwen_mask.masked_inpaint(
            pipe, src=src, mask=msk, prompt=full_prompt,
            negative=negative or QWEN_EDIT_NEG, steps=steps, cfg=cfg, seed=seed,
            work=work, feather=2)
        out = gen.convert("RGBA")
        out = _restore_alpha(out, src)
        box = list(msk.getbbox() or (0, 0, src.width, src.height))
        approx = False
        backend = "qwen-mask-inpaint"
        used_fast = False      # masked_inpaint 自己跑管线，不挂 Lightning
    elif strategy == "crop":
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
        approx = True
        backend = "qwen-edit"
        used_fast = True       # 走 _qwen_call
    else:
        box = list(msk.getbbox() or (0, 0, src.width, src.height))
        gen, work, full_prompt, region = _qwen_local_edit(
            eng, src=src, mask=msk, body=body, negative=negative, steps=steps,
            cfg=cfg, seed=seed, model=model, label="w5")
        out = rp.composite(src, gen.convert("RGBA"), msk, feather_px=feather,
                           match_color=match_color,
                           match_strength=match_strength)
        out = _restore_alpha(out, src)
        approx = True
        backend = "qwen-edit"
        used_fast = True       # 走 _qwen_local_edit → _qwen_call

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
        # 实际输出尺寸
        "width": out.width, "height": out.height,
        "strength": None,
        "steps": _eff_steps(eng, steps, fast=used_fast),
        **_fast_meta(eng, applied=used_fast),
        "cfg": cfg, "seed": seed,
        "model": model, "lora": None, "lora_scale": None, "clip_skip": None,
        "backend": backend, "approx": approx, "strategy": strategy,
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
               negative: Optional[str] = None, steps: int = 16, cfg: float = 1.0,
               seed: int = -1, model: str = DEFAULT_MODEL,
               preset: Optional[str] = None,
               context: float = 0.35, expand: int = 8, feather: int = 4,
               match_color: bool = True, match_strength: float = 0.45,
               palette_snap: bool = False, palette_colors: int = 32,
               strategy: str = "mask",
               scope: str = "", tag: str = "w5") -> Dict[str, Any]:
    """局部 AI 重绘。与 w3 同一个机制，多了色板吸附与裁剪策略。

    `strategy`：
      · `mask`（默认，**真 inpaint**）：模型看得见蒙版 → 指令准确执行、身份不丢。
      · `whole`（旧默认，整图作条件 + 蒙版贴回，近似）。
      · `crop`（裁块重画，质量不合格，仅留作对照实验）。
    """
    eng.require(model, "ai_repaint")
    if strategy not in ("mask", "whole", "crop"):
        raise ValueError(f"未知策略 {strategy}（可选 mask / whole / crop）")
    return _qwen_ai_repaint(
        eng, image=image, mask=mask, prompt=prompt, negative=negative,
        steps=steps, cfg=cfg, seed=seed, model=model, preset=preset,
        context=context, expand=expand, feather=feather,
        match_color=match_color, match_strength=match_strength,
        palette_snap=palette_snap, palette_colors=palette_colors,
        strategy=strategy, scope=scope, tag=tag)


# ---------------- WOUTFIT / WFACE 换装与换脸 ----------------

def _swap_impl(eng: Engine, *, kind: str, base_image: str | Path,
               ref_image: Optional[str | Path] = None,
               ref_images: Optional[Sequence[str]] = None,
               prompt: str = "", negative: Optional[str] = None,
               width: int = 0, height: int = 0, steps: int = 16,
               cfg: float = 1.0, seed: int = -1,
               model: str = DEFAULT_MODEL,
               scope: str = "", tag: str = "") -> Dict[str, Any]:
    """换装 / 换脸共用实现：多张图进同一个 vision context + 结构化指令。

    **不需要任何专用权重**——Qwen 的多图条件会自己分配"谁是身份、谁是服装"，
    只要指令里写清楚每张图负责什么（`swap.OUTFIT_PROMPT` / `FACE_PROMPT`）。

    参考图可以给多张（`ref_images`）：同一件服装的正面/背面/细节，或同一个人
    的不同角度，都会让最终结果更稳。张数变化时指令模板也跟着换（`n_refs`），
    否则模型会以为要换好几套衣服。
    """
    ref_paths = _ref_image_list(ref_image, ref_images)
    base = Image.open(base_image).convert("RGBA")
    refs = [Image.open(p).convert("RGBA") for p in ref_paths]
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    # 输出尺寸跟随基底图（换装/换脸的"画布"就是基底图）
    w = max(256, (int(width) if width else base.width) // 16 * 16)
    h = max(256, (int(height) if height else base.height) // 16 * 16)

    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    full = swap_mod.swap_prompt(kind, prompt, n_refs=len(refs))
    img = _qwen_call(
        eng, prompt=full, negative=(negative or QWEN_EDIT_NEG),
        width=w, height=h, steps=steps, seed=seed,
        # 图 1 = 基底，图 2..N = 参考（服装 / 人脸）
        ref_image=[base, *refs], model=model, label=tag, cfg=cfg,
        # 条件图的编码尺度要跟输出一致，否则默认 1024²：参考图的细节密度
        # 比目标大一档，图 token 多 78%（见 _qwen_call 的第 3 条）
        output_resolution=max(w, h))
    img = img.convert("RGBA")
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    seconds = dt()
    wf = "woutfit" if kind == "outfit" else "wface"
    return _save(img, _stamp(tag), wf, {
        "workflow": wf, "kind": kind,
        "prompt": full, "prompt_body": prompt, "negative": negative or QWEN_EDIT_NEG,
        "base_image": str(base_image), "ref_image": str(ref_paths[0]),
        "ref_images": ref_paths, "n_refs": len(ref_paths),
        "width": w, "height": h, "steps": _eff_steps(eng, steps),
        "cfg": cfg, "seed": seed, "model": model,
        "backend": "qwen-multiref", "approx": False,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        **_fast_meta(eng),
    }, scope=scope)


def outfit_swap(eng: Engine, *, image: str | Path,
                outfit: Optional[str | Path] = None,
                outfits: Optional[Sequence[str]] = None,
                prompt: str = "", negative: Optional[str] = None,
                width: int = 0, height: int = 0, steps: int = 16,
                cfg: float = 1.0, seed: int = -1, model: str = DEFAULT_MODEL,
                scope: str = "", tag: str = "woutfit") -> Dict[str, Any]:
    """**换装**：把 `outfit`（图 2）的服装穿到 `image`（图 1）的人物身上。

    保留图 1 的身份、姿势、背景；服装的剪裁/颜色/材质来自图 2。
    游戏素材用途：同一角色出不同装备/皮肤（商店套装、等级外观）。

    `outfits` 可以给多张（同一套服装的正面/背面/细节），服装细节会更准。
    """
    eng.require(model, "outfit_swap")
    return _swap_impl(eng, kind="outfit", base_image=image, ref_image=outfit,
                      ref_images=outfits,
                      prompt=prompt, negative=negative, width=width,
                      height=height, steps=steps, cfg=cfg, seed=seed,
                      model=model, scope=scope, tag=tag)


def face_swap(eng: Engine, *, image: str | Path,
              face: Optional[str | Path] = None,
              faces: Optional[Sequence[str]] = None,
              prompt: str = "", negative: Optional[str] = None,
              width: int = 0, height: int = 0, steps: int = 16,
              cfg: float = 1.0, seed: int = -1, model: str = DEFAULT_MODEL,
              scope: str = "", tag: str = "wface") -> Dict[str, Any]:
    """**换脸**：把 `face`（图 2）的脸/发型换到 `image`（图 1）的人物上。

    保留图 1 的姿势、服装、场景；面部特征/发型/胡须来自图 2。
    游戏素材用途：同一套动作换主角脸（换主角、捏脸结果落图）。

    `faces` 可以给多张（同一人的正脸/侧脸），五官会更稳定。
    """
    eng.require(model, "face_swap")
    return _swap_impl(eng, kind="face", base_image=image, ref_image=face,
                      ref_images=faces,
                      prompt=prompt, negative=negative, width=width,
                      height=height, steps=steps, cfg=cfg, seed=seed,
                      model=model, scope=scope, tag=tag)


# ---------------- WOUT 扩图 ----------------

def outpaint(eng: Engine, *, image: str | Path, direction: str = "right",
             expand: float = 0.5, prompt: str = "", negative: Optional[str] = None,
             steps: int = 16, cfg: float = 1.0, seed: int = -1,
             model: str = DEFAULT_MODEL, lora_scale: float = 0.9,
             blend: bool = True, scope: str = "", tag: str = "wout"
             ) -> Dict[str, Any]:
    """**扩图**：把图往某个方向扩展，新区域按场景连续性补全。

    靠 Outpaint LoRA（`ausboss/Qwen-Image-2.1-Outpaint-LoRA`）+ 中性灰 padding：
    灰色是"这里要补内容"的信号，LoRA 让模型偏向"接缝连贯、保持原图"。
    **提示词必须以 LoRA 的触发句开头**（见 `outpaint.OUTPAINT_TRIGGER`）。

    `direction`：right / left / up / down / all
    `expand`：往该方向扩多少（相对对应边长，0.5 = 扩一半）
    `blend=True`：原图区域软边贴回（保证原图像素基本不变）

    游戏素材用途：方形背景 → 16:9 横版 / 竖版，地图块延展，立绘左右留白。
    """
    eng.require(model, "outpaint")
    if direction not in ("right", "left", "up", "down", "all"):
        raise ValueError(f"未知方向 {direction}（可选 right/left/up/down/all）")
    if not (0.1 <= float(expand) <= 2.0):
        raise ValueError(f"扩展比例 {expand} 超出范围（0.1~2.0）")

    src = Image.open(image).convert("RGBA")
    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    pipe = eng.txt2img_pipe(model)
    # 扩图**必须**挂 Outpaint LoRA（标准权重不认识灰色 padding 的语义）。
    # 走**临时槽**并在 finally 卸载：以前它永久留在共享管线里，之后所有
    # 生成都带着"灰色 padding = 待补全"的语义出图（而 meta 还写着 lora: None）。
    eng.apply_loras(pipe, [{"name": eng.OUTPAINT_LORA, "scale": float(lora_scale)}],
                    scope="temp")
    try:
        res = op_mod.outpaint(
            pipe, src=src, direction=direction, expand=float(expand),
            prompt=prompt, negative=negative, steps=steps, cfg=cfg, seed=seed,
            blend=blend)
    finally:
        eng.unload_loras(pipe, scope="temp")
    out = res["image"].convert("RGBA")
    seconds = dt()

    meta = op_mod.outpaint_meta(
        res, workflow="wout", direction=direction, expand=float(expand),
        lora=eng.OUTPAINT_LORA, lora_scale=float(lora_scale),
        prompt_body=prompt, image=str(image),
        # 扩图挂的是 Outpaint LoRA，**不走少步分支**
        steps=_eff_steps(eng, steps, fast=False),
        cfg=cfg, seed=seed, model=model, backend="qwen-outpaint",
        blend=bool(blend), approx=False,
        seconds=seconds,
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        **_fast_meta(eng, applied=False))
    return _save(out, _stamp(tag), "outpaint", meta, scope=scope)


# ---------------- WPX 完美像素转换 ----------------

def pixelify(eng: Engine, *, init_image: str | Path,
             target: Optional[Sequence[int]] = None,
             grid_size: Optional[Sequence[int]] = None,
             grid_snap: bool = True,
             sample_method: str = "median",
             colors: int = 32, transparent_bg: bool = False,
             snap_to: Optional[int] = 16,
             model: str = DEFAULT_MODEL,
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
    dt = _timer()
    eng.require(model, "pixelify")
    dt = _timer()
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
        # 统一的尺寸字段（其它工作流都有这两个键）
        "width": out.width, "height": out.height,
        "seconds": round(dt(), 2),
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
                   steps: int = 16, cfg: float = 1.0, seed: int = -1,
                   model: str = DEFAULT_MODEL,
                   matte: bool = True, matte_backend: str = "auto",
                   transparent: bool = False,
                   scope: str = "", tag: str = "wap") -> Dict[str, Any]:
    """任意图 → 像素画（不限于动漫人物）。**两步走**：先出高质量大图，再做像素化。

    **不只动漫**：照片、插画、3D 渲染、手绘图都能转——像素化那一半是通用的
    （网格对齐 + 色板量化 + 轮廓描边），与前一步出什么图无关。只在"没有源图、
    需要先生成"时才会用到动漫底模。

    为什么分两步（先出大图、再像素化）：

        让模型直接画像素   构图常裁到腿、颜色是连续渐变，
                           输出 768² 但有效像素只有 128
        先出大图 → 像素化  全身完整、1px 描边清晰，
                           输出就是原生 128²/192²

    根因是**两种能力互相打架**：像素风要求极简色块和硬边，而人物结构
    （脸、手、持握武器）需要足够的分辨率理解。让模型一次做两件事，它会在
    两者之间妥协。分开做则各取所长：模型负责「画对」，像素化负责「压成像素」。

    **Qwen 这边不用像素 LoRA**：它直接就能画出像样的像素风（实测 768²/24 步
    出 16-bit 猫骑士），所以「先出图」那一步默认不像素化，像素化只在这一步做。

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
    eng.require(model, "topixel")
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
        if model != DEFAULT_MODEL:
            # 用户可能选了别的底模；提示一次但不阻止（有人就想用自己的底模）
            print(f"[warn] 底模 {model} 不是本工作台的默认底模，"
                  f"人物质量可能不如 {DEFAULT_MODEL}")
        r = txt2img(eng, prompt=prompt, negative=negative, preset="anime",
                    width=source_width, height=source_height,
                    steps=steps, cfg=cfg, seed=seed, model=model,
                    matte=matte,
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


# ---------------- 多张参考图 ----------------

#: 一条工作流最多吃几张参考图。Qwen 的多图上限实测 10 张仍可用（512²/8 步
#: 12.3s / 14.1GB），但再多没有收益、显存还会涨，这里收在 6 张。
MAX_REF_IMAGES = 6


def _ref_image_list(single, many) -> list[str]:
    """把「单张 + 多张」两种传法统一成列表。

    前端老版本只传 `char_image`，新版本传 `char_images`；两者都给时以
    `char_images` 为准（前端在数组里已经把第一张写进去了）。
    """
    out = [str(p) for p in (many or []) if p]
    if not out and single:
        out = [str(single)]
    if not out:
        raise ValueError("缺少参考图：char_image / char_images 至少给一张")
    if len(out) > MAX_REF_IMAGES:
        raise ValueError(
            f"参考图最多 {MAX_REF_IMAGES} 张，收到 {len(out)} 张")
    return out


# ---------------- WPC 角色迁移 ----------------

def _qwen_pose_clone(eng: Engine, *, pose_image: str | Path,
                     char_image: Optional[str | Path] = None,
                     prompt: str = "", negative: Optional[str] = None,
                     width: int = 768, height: int = 768,
                     steps: int = 16, cfg: float = 1.0, seed: int = -1,
                     ref_mode: str = "auto", thr: float = 0.25, fill: float = 0.8,
                     pixel: int = 0, palette: int = 0, matte: bool = False,
                     matte_backend: str = "auto",
                     denoise: int = 1, clean_iso: int = 1,
                     model: str = DEFAULT_MODEL, bg: str = "white",
                     scope: str = "", tag: str = "wpc",
                     char_images: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Qwen-Image-2.1 版角色迁移：**双图条件**（角色图 + 姿态条件图）。

    为什么不用 SD 那套 ControlNet + IP-Adapter：
      - SD 那条路是**两个互相拉扯的约束**——IP-Adapter 抢身份、ControlNet 抢
        姿态，实测同一角色同一动作，出来的图经常是「身份丢了」或者「姿势变形」。
      - Qwen 的多图条件是**同一个 vision context**：两张图一起进模型，由它自己
        分配「谁是身份、谁是姿势」，角色图的主色会被继承（蓝角色 → 蓝衣服）。

    ⚠️ 早前这里写着「姿势与骨架几乎逐点一致」——**那是误读**。当时的依据是
    「DWPose 在成品上检出 18 个关节」，只说明画出了像人形的角色，不说明姿势对得上。
    2026-10-30 按「成品关节 vs 骨架关节」的坐标距离重测：平均偏差 67~118px/768²，
    腿部经常被角色图的站姿覆盖（详见 `_qwen_poseswap` 的实测表与 PLAN §2.8）。

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
    # 多张角色图：正面 + 侧面 + 细节图一起喂，能显著提升身份一致度
    char_paths = _ref_image_list(char_image, char_images)
    chars = [refprep.prepare_reference(Image.open(p).convert("RGBA"), size=width)
             for p in char_paths]

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
        ref_image=[*chars, cond], width=width, height=height, steps=steps,
        cfg=cfg, seed=seed, model=model, label="wpc",
        output_resolution=max(width, height))
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
        "pose_image": str(pose_image), "char_image": str(char_paths[0]),
        "char_images": char_paths, "n_refs": len(char_paths),
        "backend": "qwen-multiref", "bg": bg,
        "ref_mode": ref_mode, "ref_mode_used": used_mode,
        "ref_fallback_from": cond_info.get("fallback_from"),
        "ref_fallback_reason": cond_info.get("fallback_reason"),
        "joints": cond_info.get("joints"), "fill": fill,
        "width": width, "height": height,
        "steps": _eff_steps(eng, steps), **_fast_meta(eng), "cfg": cfg, "seed": seed,
        "model": model,
        "pixel_block": pixel, "palette": palette, "matte": matte,
        "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "edge_touch": edge, "subjects": subjects,
        "seconds": seconds,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, raw=raw, scope=scope)


def pose_clone(eng: Engine, *, pose_image: str | Path,
               char_image: Optional[str | Path] = None,
               prompt: str = "", negative: Optional[str] = None,
               width: int = 768, height: int = 768,
               steps: int = 16, cfg: float = 1.0, seed: int = -1,
               fill: float = 0.8, thr: float = 0.25,
               ref_mode: str = "auto", bg: str = "white",
               model: str = DEFAULT_MODEL,
               pixel: int = 0, palette: int = 0, matte: bool = False,
               matte_backend: str = "auto", denoise: int = 1, clean_iso: int = 1,
               scope: str = "", tag: str = "wpc",
               char_images: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """角色迁移：把**动作参考图**的姿势套到**角色图**上。

    Qwen 走**双图条件**（角色图 + 姿态条件图进同一个 vision context），
    所以没有 ControlNet / IP-Adapter 的强度旋钮（`cn_scale` / `ipa_scale` 已删）。

    ⚠️ 动作跟随的真实水平：上身跟得好，**腿部常被参考图的站姿覆盖**
    （2026-10-30 实测，见 `_qwen_poseswap` 的对照表与 PLAN §2.8）。
    """
    eng.require(model, "pose_clone")
    return _qwen_pose_clone(
        eng, pose_image=pose_image, char_image=char_image,
        char_images=char_images,
        prompt=prompt, negative=negative,
        width=width, height=height, steps=steps, cfg=cfg, seed=seed,
        ref_mode=ref_mode, thr=thr, fill=fill,
        pixel=pixel, palette=palette, matte=matte, matte_backend=matte_backend,
        denoise=denoise, clean_iso=clean_iso, model=model, bg=bg,
        scope=scope, tag=tag)


# ---------------- WVNC 姿态迁移（VNCCS PoseStudio LoRA 路线） ----------------

#: VNCCS 官方工作流里的固定指令（逐字）。
#:
#: `image 1` = 骨架、`image 2` = 角色图。这个顺序不能反——社区那个
#: "完美保持角色一致性"的效果，一半来自这句指令 + 专用 LoRA。
_VNCCS_TEMPLATE = ("replace the pose of <image 2> with the pose of <image 1>. "
                   "keep the character of <image 2>")


def _vnccs_fit(im: Image.Image, w: int, h: int) -> Image.Image:
    """等比缩到框内、居中贴白底。

    **不能硬拉**：512×768 的骨架拉成方形会把腿压短，实测姿态只跟一半
    （双腿交叉退化成双腿收窄）。
    """
    r = min(w / im.width, h / im.height)
    nw, nh = max(1, round(im.width * r)), max(1, round(im.height * r))
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(im.resize((nw, nh), Image.LANCZOS),
                 ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _vnccs_skeleton(im: Image.Image) -> Image.Image:
    """把黑底骨架图的背景换成白底。

    **必须做**：OpenPose 骨架（DWPose / tools/make_pose.render）都是黑底，
    整张贴到白底画布上之后，中间仍是一块**黑底矩形**——模型读到的是
    "一个黑色剪影"，于是把角色画成背面（实测：黑底骨架 → 输出背影；
    同一张骨架换白底 → 正面）。

    判据用**饱和度**而不是亮度：OpenPose 的线是纯色（红黄绿青蓝），
    背景是纯黑（无彩色分量）。深色的蓝线饱和度仍然很高，不会被误伤。
    """
    import numpy as np
    arr = np.asarray(im.convert("RGB"), dtype=np.int16)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = mx - mn
    dark = mx < 70
    bg = (sat < 24) & dark                 # 无彩色分量且很暗 → 背景
    out = arr.copy()
    out[bg] = 255
    return Image.fromarray(out.astype("uint8"), "RGB")


def _vnccs_canvas_size(pose: Image.Image, width: int, height: int) -> tuple[int, int]:
    """画布尺寸。默认**取骨架的比例**，不走官方"取角色图"。

    官方示例取角色图尺寸，前提是"角色图与骨架同为竖版"。我们的角色图常是
    方形立绘、骨架是竖版，取方形时骨架只能缩在中间，模型会理解成"小幅度
    动作"——实测交叉腿会退化成收窄。按骨架比例铺满才跟得住（PLAN §2.16b）。
    """
    if width and height:
        return max(256, width // 32 * 32), max(256, height // 32 * 32)
    scale = max(1.0, 768 / max(pose.size))
    return (max(256, int(pose.size[0] * scale) // 32 * 32),
            max(256, int(pose.size[1] * scale) // 32 * 32))


def _qwen_vnccs(eng: Engine, *, pose_image: str | Path,
                char_image: Optional[str | Path] = None,
                prompt: str = "", negative: Optional[str] = None,
                width: int = 0, height: int = 0, steps: int = 25,
                cfg: float = 1.0, seed: int = -1,
                lora_scale: float = 1.0, transparent: bool = False,
                model: str = DEFAULT_MODEL,
                pixel: int = 0, palette: int = 0, matte: bool = False,
                matte_backend: str = "auto", denoise: int = 1, clean_iso: int = 1,
                scope: str = "", tag: str = "wvnc",
                char_images: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """VNCCS 路线的姿态迁移：**专用 LoRA + 双图指令**，不用 ControlNet。

    这条路线是照 MIUProject 官方 ComfyUI 工作流复刻的（`VNCCS_PoseStudio_QI2.1.json`，
    14 个节点里没有任何 ControlNet）。相比 `_qwen_pose_clone`：
      - 图序相反：[骨架, 角色]（那边是 [角色, 条件]）
      - prompt 用官方固定模板，而不是自己写的"The first image is..."
      - 挂 VNCCS LoRA（1.0）
      - 条件图**不灰度化**（LoRA 是按 PoseStudio 的彩色骨架训的）
      - 25 步、cfg=1（官方值），不挂 Lightning

    实测（768×1152）：画风与参考图一致、服装发型全保住、姿态能跟住。
    代价是姿态精度不如 ControlNet 的硬约束。
    """
    pose = Image.open(pose_image).convert("RGB")
    char_paths = _ref_image_list(char_image, char_images)
    chars = [Image.open(p).convert("RGB") for p in char_paths]
    w, h = _vnccs_canvas_size(pose, width, height)

    instr = _VNCCS_TEMPLATE
    sem = (prompt or "").strip()
    if sem:
        # **身份描述必须与参考图一致**：模型会让 prompt 压过参考图。
        # 实测照着别的角色写描述，出来就真成了那个角色（连画风都换）。
        instr = f"{instr}\n{sem}"

    pipe = eng.txt2img_pipe(model)
    lora_file = eng.lora_dir() / f"{eng.VNCCS_LORA}.safetensors"
    if not lora_file.is_file():
        raise FileNotFoundError(
            f"缺 VNCCS LoRA：{lora_file}\n"
            f"下载 MIUProject/VNCCS_PoseStudio_QI2.1 的 V1，用 "
            f"tools/convert_lora.py 转成 diffusers 格式后放到 models/lora/")

    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    torch.cuda.reset_peak_memory_stats()
    dt = _timer()
    eng.apply_loras(pipe, [{"name": eng.VNCCS_LORA, "scale": float(lora_scale)}],
                    scope="temp")
    try:
        out = _qwen_call(
            eng, prompt=instr, negative=negative or "",
            # **顺序：骨架在前、角色在后**（官方 image 1 / image 2）
            ref_image=[_vnccs_fit(pose, w, h),
                       *[_vnccs_fit(c, w, h) for c in chars]],
            width=w, height=h, steps=steps, cfg=cfg, seed=seed,
            model=model, label="wvnc", transparent=transparent,
            output_resolution=max(w, h),
            fast_lora=False,          # LoRA 槽位归 VNCCS
        )
    finally:
        eng.unload_loras(pipe, scope="temp")
    seconds = dt()

    raw = out.copy() if (matte or pixel > 1 or palette) else None
    matte_info: Dict[str, Any] = {}
    out, edge, subjects = _postprocess(
        out, eng, matte=matte, matte_backend=matte_backend, denoise=denoise,
        pixel=pixel, palette=palette, clean_iso=clean_iso, info=matte_info)
    # `_postprocess` 的 info 里有 `pre_pixel`（PIL Image），整个展开进 meta
    # 会在写 sidecar 时 `json.dumps` 报 "Object of type Image is not
    # serializable"。只挑可序列化的几个键（其它工作流也这么做）。
    safe_info = {k: matte_info[k] for k in ("backend", "fallback")
                 if k in matte_info}
    return _save(out, _stamp(tag), "vnccs_pose", {
        "workflow": "wvnccs", "scope": scope or "default",
        "prompt": instr, "negative": negative or "",
        "pose_image": str(pose_image), "char_image": str(char_paths[0]),
        "char_images": char_paths, "n_refs": len(char_paths),
        "backend": "qwen-vnccs-lora", "lora": eng.VNCCS_LORA,
        "lora_scale": float(lora_scale),
        "canvas": [w, h], "steps": _eff_steps(eng, steps, fast=False),
        "cfg": cfg, "seed": seed,
        "transparent": bool(transparent),
        **_fast_meta(eng, applied=False),
        "seconds": round(seconds, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "edge_px": edge, "subjects": subjects,
        **({"raw": str(_save(raw, _stamp(tag), "raw", {"role": "pre_post"},
                             scope=scope)["path"])} if raw is not None else {}),
        **safe_info,
    }, scope=scope)


def vnccs_pose(eng: Engine, *, pose_image: str | Path,
               char_image: Optional[str | Path] = None,
               char_images: Optional[Sequence[str]] = None,
               prompt: str = "", negative: Optional[str] = None,
               width: int = 0, height: int = 0,
               steps: int = 25, cfg: float = 1.0, seed: int = -1,
               lora_scale: float = 1.0, transparent: bool = False,
               model: str = DEFAULT_MODEL,
               pixel: int = 0, palette: int = 0, matte: bool = False,
               matte_backend: str = "auto", denoise: int = 1, clean_iso: int = 1,
               scope: str = "", tag: str = "wvnc") -> Dict[str, Any]:
    """姿态迁移（VNCCS 路线）：把骨架图的动作套到角色图上，**保住角色一致性**。

    `width` / `height` 留 0 表示按**骨架的比例**自动定画布（推荐）。手动指定时
    记得用竖版（如 768×1152），方形画布会让模型把人物撑满、头顶出画面。

    `prompt` 里要写**与角色图一致**的外观描述（发色、发饰、服装、配色），
    模型会让 prompt 压过参考图；写错了等于换了个角色。
    """
    eng.require(model, "vnccs_pose")
    return _qwen_vnccs(
        eng, pose_image=pose_image, char_image=char_image,
        char_images=char_images,
        prompt=prompt, negative=negative, width=width, height=height,
        steps=steps, cfg=cfg, seed=seed, lora_scale=lora_scale,
        transparent=transparent, model=model,
        pixel=pixel, palette=palette, matte=matte, matte_backend=matte_backend,
        denoise=denoise, clean_iso=clean_iso, scope=scope, tag=tag)


# ---------------- WSS 序列帧角色替换 ----------------

def _qwen_sprite_swap(eng: Engine, *, frames: Sequence[tuple],
                      sheet_image: str | Path,
                      char_image: Optional[str | Path] = None,
                      char_images: Optional[Sequence[str]] = None,
                      cols: int, rows: int, prompt: str, negative: Optional[str],
                      width: int, height: int, steps: int, cfg: float, seed: int,
                      ref_mode: str, model: str, matte: bool, matte_backend: str,
                      denoise: int, palette: int, keep_alpha: bool,
                      candidates: int, bg: str, scope: str,
                      tag: str, engine: str = "wpc",
                      lora_scale: float = 1.0,
                      gate_retries: int = 0,
                      gate_seed_step: int = 1000) -> Dict[str, Any]:
    """Qwen 版序列帧替换（WSS）：逐帧「角色图 + 轮廓/骨架」双图条件。

    与 SD 那条路的差别只有一处但很关键：**没有 ControlNet 硬约束**。
    SD 是「轮廓条件钉死形状 + IP-Adapter 定身份」，两个约束各管一半；
    Qwen 是两张图一起进 vision context，由模型自己分配谁管身份、谁管动作。
    实测 3 帧骨架条件 DWPose 18/18、零贴边，动作贴合度够精灵产线用。

    **成本**：单帧 35s 级别，所以 candidates 默认 1（SD 那边是 2）。
    4 帧动画 = 约 2.5 分钟；8 帧 = 5 分钟。这是「Qwen 换 SD」绕不开的代价。
    """
    use_vnccs = (engine or "wpc").lower() in ("vnccs", "wvnccs")
    char_paths = _ref_image_list(char_image, char_images)
    char_orig = Image.open(char_paths[0]).convert("RGBA")
    if use_vnccs:
        # VNCCS 走"原图等比贴入"而不是 `prepare_reference` 的留白版本：
        # 它的 LoRA 是按「骨架 / 角色图直接拼」训的，主体在画面里的占比
        # 与训练分布一致才跟得准（留白 12% 会让角色显得比骨架小一号）。
        chars = [_vnccs_fit(Image.open(p).convert("RGBA"), width, height)
                 for p in char_paths]
        if steps < 20:
            # VNCCS LoRA 是按完整采样训的，不是少步蒸馏；步数不够时姿态和
            # 细节会一起崩（实测参考值 25）。
            steps = 20
    else:
        chars = [refprep.prepare_reference(Image.open(p).convert("RGBA"),
                                           size=max(width, height))
                 for p in char_paths]
    extractor = eng.pose_extractor()
    sem = (prompt or "").strip()
    if use_vnccs:
        # 官方固定指令（图 1 骨架 / 图 2 角色）直接可用——wss 的语义与它一致。
        # 只追加背景与帧规则的尾部约束，不重写主体指令。
        instr = (f"{_VNCCS_TEMPLATE} {_qwen_backdrop(bg)}." + QWEN_FRAME_RULES)
    else:
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

    # VNCCS 的 LoRA **在整段循环外挂一次**：逐帧反复 load/unload 每帧都要
    # 重读 152MB 权重，8 帧就会多花一分多钟。
    vnccs_pipe = None
    if use_vnccs:
        vnccs_pipe = eng.txt2img_pipe(model)
        lora_file = eng.lora_dir() / f"{eng.VNCCS_LORA}.safetensors"
        if not lora_file.is_file():
            raise FileNotFoundError(
                f"缺 VNCCS LoRA：{lora_file}\n"
                f"下载 MIUProject/VNCCS_PoseStudio_QI2.1 的 V1，用 "
                f"tools/convert_lora.py 转成 diffusers 格式后放到 models/lora/")
        eng.apply_loras(vnccs_pipe, [{"name": eng.VNCCS_LORA,
                                      "scale": float(lora_scale)}], scope="temp")
    try:
        for _c, _r, frame in frames:
            cond, used_mode, _info = refcond.build_condition(
                frame, mode=ref_mode, size=max(width, height),
                extractor=extractor, matte_dir=eng.matte_dir)
            used_modes.append(used_mode)
            if use_vnccs:
                # **不灰度化**：VNCCS LoRA 是按 PoseStudio 的彩色骨架训的，
                # 灰度化会掉遵循度。等比贴入、不拉伸（拉伸会把腿压短）。
                cond = _vnccs_fit(_vnccs_skeleton(cond.convert("RGB")),
                                  width, height)
            else:
                # 灰度化：彩色骨架会把颜色抄进成品（见 _qwen_cond_image）
                cond = _qwen_cond_image(cond.resize((width, height)))
            ref_area = int(framescore.subject_mask(frame.convert("RGBA")).sum())
            prev = generated[-1] if generated else None
            dt = _timer()

            def _gen(k: int, _cond=cond) -> tuple[Image.Image, Dict[str, Any]]:
                raw = _qwen_call(
                    eng, prompt=instr, negative=neg,
                    # 图序按引擎反转：VNCCS 是 [骨架, 角色]，旧路线是 [角色, 骨架]
                    ref_image=([_cond, *chars] if use_vnccs
                               else [*chars, _cond]),
                    width=width, height=height, steps=steps, cfg=cfg,
                    seed=seed + k * 1000, model=model, label="wss",
                    output_resolution=max(width, height),
                    fast_lora=not use_vnccs)   # LoRA 槽位归 VNCCS
                info: Dict[str, Any] = {}
                cut, _edge, _subj = _postprocess(
                    raw, eng, matte=matte, matte_backend=matte_backend,
                    denoise=denoise, pixel=0, palette=0, clean_iso=0, info=info)
                for kk in ("backend", "fallback"):
                    if kk in info:
                        matte_info[kk] = info[kk]
                return cut, {}

            def _score(cut: Image.Image, _info: Dict[str, Any], *,
                       _frame=frame, _prev=prev,
                       _area=ref_area) -> Dict[str, Any]:
                return framescore.evaluate(cut, _frame, char_orig, prev=_prev,
                                           area_ref=_area)

            best_img, best_ev, _finfo, used, _nh = gate_mod.run_candidates(
                generate=_gen, score=_score,
                policy=gate_mod.GatePolicy(
                    candidates=candidates, retries=gate_retries,
                    seed_step=gate_seed_step,
                    stop_on_first_pass=True))
            frame_scores.append({**best_ev, "attempts": used})
            attempts_used += used
            per_frame.append(dt())
            generated.append(best_img)
    finally:
        if vnccs_pipe is not None:
            eng.unload_loras(vnccs_pipe, scope="temp")

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
    for item, src in zip(refined["items"], cut, strict=True):
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
        "sheet_image": str(sheet_image), "char_image": str(char_paths[0]),
        "char_images": char_paths, "n_chars": len(char_paths),
        "grid": [cols, rows], "frames": len(aligned),
        "cell": [cell_w, cell_h],
        "ref_mode": ref_mode, "ref_modes_used": sorted(set(used_modes)),
        "model": model, "width": width, "height": height,
        "steps": _eff_steps(eng, steps, fast=not use_vnccs),
        **_fast_meta(eng, applied=not use_vnccs),
        "cfg": cfg, "seed": seed,
        "engine": "vnccs" if use_vnccs else "wpc",
        "lora": (eng.VNCCS_LORA if use_vnccs else None),
        "lora_scale": (float(lora_scale) if use_vnccs else None),
        "cn_scale": None, "ipa_scale": None, "clip_skip": None,
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


def sprite_swap(eng: Engine, *, sheet_image: str | Path,
                char_image: Optional[str | Path] = None,
                char_images: Optional[Sequence[str]] = None,
                prompt: str = "", negative: Optional[str] = None,
                cols: int = 0, rows: int = 0,
                width: int = 768, height: int = 768,
                steps: int = 16, cfg: float = 1.0, seed: int = -1,
                ref_mode: str = "outline", mode: str = "generate",
                engine_mode: str = "auto",
                model: str = DEFAULT_MODEL,
                matte: bool = True, matte_backend: str = "auto",
                denoise: int = 1, clean_iso: int = 1,
                palette: int = 32, keep_alpha: bool = False,
                candidates: int = 1, gate_retries: int = 1,
                gate_seed_step: int = 1000,
                vram_profile: str = "auto", bg: str = "white",
                engine: str = "wpc", lora_scale: float = 1.0,
                scope: str = "", tag: str = "wss") -> Dict[str, Any]:
    """序列帧角色替换（**近似实现**）：逐帧双图条件 → 缩放回格子 → 共享色板 → 打包。

    `mode`：`generate`（跑模型，逐帧） / `recolor`（确定性换色，毫秒级、帧间零漂移）。
    `engine_mode` 是 SD 时代的 AnimateDiff 开关，Qwen 下恒为逐帧（参数保留兼容）。
    `bg`：白底 / 深色底——白毛角色用白底会被抠图吃掉脸和腿，换 dark 更稳。
    `engine`：
      - `wpc`（默认，旧路线）—— 角色图 + 轮廓/骨架双图 + 自写指令
      - `wvnccs` —— 逐帧挂 VNCCS LoRA，图序 [骨架, 角色]、官方固定指令、
        条件图不灰度化、步数下限 20。**帧间角色一致性明显更好**（同一角色
        跨 8 帧不会换脸换衣服），代价是单帧更慢（完整 25 步 + LoRA）。
    """
    if mode != "recolor":
        eng.require(model, "sprite_swap")
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
    prompt = (prompt or "").strip() or "a single character, full body, facing viewer"

    if mode == "recolor":
        return _sprite_recolor(frames, char_image, cols, rows, palette=palette,
                               scope=scope, tag=tag, prompt=prompt)
    return _qwen_sprite_swap(
        eng, frames=frames, sheet_image=sheet_image, char_image=char_image,
        char_images=char_images,
        cols=cols, rows=rows, prompt=prompt, negative=negative,
        width=width, height=height, steps=steps, cfg=cfg, seed=seed,
        ref_mode=ref_mode, model=model, matte=matte,
        matte_backend=matte_backend, denoise=denoise, palette=palette,
        keep_alpha=keep_alpha, candidates=candidates, bg=bg, scope=scope, tag=tag,
        engine=engine, lora_scale=lora_scale,
        gate_retries=gate_retries, gate_seed_step=gate_seed_step)


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
        # 落盘的是**精灵图**，尺寸 = 列数×格宽；生成画布是另一回事
        "sheet_size": [cols * cell_w, rows * cell_h],
        "palette": palette, "palette_size": len(pal),
        "palette_colors": [list(c) for c in pal],
        "seconds_total": 0.0,
        "deterministic": True,
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
                   anim_name: str, fps: int, two_stage: bool = False,
                   engine: str = "wpc", lora_scale: float = 1.0,
                   char_images: Optional[Sequence[str]] = None,
                   tag: str = "w4") -> Dict[str, Any]:
    """Qwen 版姿态序列（W4）：逐帧「骨架图当参考 + 指令」软引导。

    **它是近似实现**：SD 那条路是 ControlNet 硬约束（每个关节都钉住），
    这里只能是「第二张图是骨架 + 指令说照着这个姿势画」。

    ⚠️ **姿态贴合度的真实水平**（2026-10-30 实测，别被「18 个关节」骗了）：
    DWPose 能在成品上检出 16~18 个关节点，那只说明「画出了一个像人形的角色」，
    **不说明姿势和骨架一致**。按「成品关节 vs 骨架关节」的归一化坐标平均距离量：

        角色图 + 骨架（现状）      108px / 768²   双腿基本并拢，迈步没体现
        加强指令                   97px          仍并拢
        骨架加粗 2.6×              118px         仍并拢
        交换图序（骨架在前）       104px         仍并拢
        去掉角色图（只给骨架+文字） 67px         腿真的迈开（代价：身份漂移）

    原因是**参考图在抢姿态**：角色图里是站姿，模型优先照抄它。所以这里的产出
    动作偏保守——上身会跟，腿部常被参考图的站姿覆盖。

    **两阶段模式**（`two_stage=True`）把「定姿态」与「定身份」拆开：
    先只给骨架出一张草图（姿态准、身份只是个「差不多的角色」），
    再拿「角色图 + 草图」做一次指令改图把身份换回来。
    实测（3 帧 × 768²/24 步）姿态偏差 **102px → 77px（-25%）**，
    代价是每帧多一次生成（33s → 50s）。角色图放**前面**那版配色一致度最高
    （0.49 vs 现状 0.46），采纳。
    代价里的另一项：白袍会被画成金甲（白袍像素占比 0.11 → 0.04），
    也就是**服装细节会漂**。要精确动作就开，要保服装就别开。

    **成本**：28s 级别的单帧成本，所以默认 candidates=1、补提上限压到 1。
    """
    kind = "skeleton" if controlnet == "openpose" else "outline"
    head = QWEN_POSE_INSTR[kind].format(bg=_qwen_backdrop(bg)) + QWEN_FRAME_RULES
    sem = (prompt or "").strip()
    neg = (negative or "").strip() or QWEN_FRAME_NEG

    use_vnccs = (engine or "wpc").lower() in ("vnccs", "wvnccs")
    char = None
    chars: List[Image.Image] = []
    if use_vnccs:
        # VNCCS 路线：官方模板 + 专用 LoRA，图序 [骨架, 角色]。
        # 模板本身就说清了"把 <image 2> 的姿势换成 <image 1> 的、保留 <image 2> 的角色"，
        # 所以这里只要把骨架放前面、角色放后面。
        char_paths = _ref_image_list(ref_image, char_images)
        chars = [_vnccs_fit(Image.open(p).convert("RGBA"), width, height)
                 for p in char_paths]
        head = (f"{_VNCCS_TEMPLATE} {_qwen_backdrop(bg)}." + QWEN_FRAME_RULES)
        if steps < 20:
            # LoRA 是按完整采样训的，不是少步蒸馏：步数不够姿态和细节一起崩
            steps = 20
    elif ref_image is not None:
        # 旧路线也支持多张角色图（同一角色的不同角度）——前端一直在传，
        # 以前这一支只读第一张，多图被静默丢掉。
        char_paths = _ref_image_list(ref_image, char_images)
        chars = [refprep.prepare_reference(Image.open(p).convert("RGBA"),
                                           size=max(width, height))
                 for p in char_paths]
        char = chars[0]
        if len(chars) > 1:
            # 多张角色图时必须说清这是同一个人的不同角度，
            # 否则模型会把其中一张当成骨架。
            head = head.replace(
                "The images are: first the character reference, then ",
                f"The first {len(chars)} images are character references of the "
                f"SAME character from different angles: keep the same face, "
                f"hairstyle, outfit and colors. The last image is ")
        head = head.replace(
            "The images are: first the character reference, then ",
            "The first image is the character reference: keep the same face, "
            "hairstyle, outfit and colors. The second image is ")
    # 用户正文追加在最后：身份/姿态/构图是必须的，风格词是补充。
    instr = f"{head} {sem}".strip() if sem else head

    # ---- 两阶段模式的两条指令 ----
    # 阶段 1：只给骨架，靠文字描述角色（不喂角色图 → 没人抢姿态）
    solo_instr = ("This image is a body skeleton showing a pose. Draw "
                  + (sem or "a single character")
                  + " in exactly that pose, full body, centered, "
                  + _qwen_backdrop(bg) + "." + QWEN_FRAME_RULES)
    # 阶段 2：角色图在前（它是「画谁」）、草图在后（它是「怎么站」）
    reskin_instr = (
        "The first image is the character reference: keep exactly the same "
        "face, hairstyle, outfit, armor and colors. The second image is a "
        "rough draft showing the target body pose. Redraw the character from "
        "the first image in exactly the pose of the second image, full body, "
        "centered, " + _qwen_backdrop(bg) + "." + QWEN_FRAME_RULES)
    # VNCCS 本身就是「骨架 + 角色」双图路线，两阶段（先草图再换皮）是多余的，
    # 而且两阶段的第二段用 [角色图, 草图]，与 VNCCS 的图序相反。
    use_two_stage = bool(two_stage) and char is not None and not use_vnccs
    # 两阶段模式下**不用少步蒸馏**：Lightning 是按无控制训练的，
    # 8 步配双图条件（草图 + 换皮）质量会掉；宁可多花一倍时间。
    # 做法是把临时槽清空（Lightning 由 _qwen_call 按需挂），并让两次调用
    # 都 fast_lora=False。
    if use_two_stage:
        try:
            eng.unload_loras(eng.txt2img_pipe(model), scope="temp")
        except Exception:                 # noqa: BLE001
            pass

    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    candidates = max(1, int(candidates))
    extractor = eng.pose_extractor() if gate else None
    if gate and extractor is None:
        print("[warn] 质量闸门不可用（姿态模型缺失），本次只按分数挑帧")
    # 35s/帧的前提下，补提额度必须收紧：没有判据的重提就是烧掉一分钟
    retries = max(0, min(1, int(gate_retries))) if extractor is not None else 0

    matte_info: Dict[str, Any] = {}
    frames: List[Image.Image] = []
    per_frame: List[float] = []
    frame_scores: List[Dict[str, Any]] = []
    attempts_used = 0
    prev_gate: Optional[Image.Image] = None
    # 本次是否真的走少步分支：VNCCS 要独占 LoRA 槽位、两阶段不用蒸馏
    fast_ok = (not use_vnccs) and (not use_two_stage)
    # **非人形主体**（史莱姆、机甲、道具）：DWPose 一个关节都检不出来，
    # 闸门永远过不了 —— 这时补提就是纯烧钱（每帧 25~30s 白花）。
    # 一旦某帧「画面里有前景、但关节数为 0」，就判定这个角色不是人形：
    # 关掉后续所有补提，并在 meta 里写清楚原因，不让它看起来像「生成失败」。
    non_humanoid = False
    torch.cuda.reset_peak_memory_stats()

    # VNCCS 的 LoRA **整段循环外挂一次**（逐帧 load/unload 每帧多读 152MB）
    vnccs_pipe = None
    if use_vnccs:
        vnccs_pipe = eng.txt2img_pipe(model)
        lora_file = eng.lora_dir() / f"{eng.VNCCS_LORA}.safetensors"
        if not lora_file.is_file():
            raise FileNotFoundError(
                f"缺 VNCCS LoRA：{lora_file}\n"
                f"下载 MIUProject/VNCCS_PoseStudio_QI2.1 的 V1，用 "
                f"tools/convert_lora.py 转成 diffusers 格式后放到 models/lora/")
        eng.apply_loras(vnccs_pipe, [{"name": eng.VNCCS_LORA,
                                      "scale": float(lora_scale)}], scope="temp")
    try:
        for pose_path in pose_images:
            if use_vnccs:
                # **不灰度化**：VNCCS LoRA 按彩色骨架训；**等比贴入**：拉伸会把腿压短
                cond = _vnccs_fit(
                _vnccs_skeleton(Image.open(pose_path).convert("RGB")),
                width, height)
                refs: Any = [cond, *chars]
            else:
                cond = _qwen_cond_image(
                    Image.open(pose_path).convert("RGB").resize((width, height),
                                                                Image.LANCZOS))
                refs: Any = ([*chars, cond] if chars else cond)
            spent = 0.0

            def _gen(i: int, *, _cond=cond, _refs=refs
                     ) -> tuple[Image.Image, Dict[str, Any]]:
                """跑一个候选：出图 → 后处理 → 交给打分函数。

                `cond` / `refs` 用**默认参数绑定当下这一帧的值**：它们是循环变量，
                闭包直接引用会指向"最后一次循环"的值（B023）。现在
                `run_candidates` 是同步调用，不会串帧，但这种写法只要以后把
                生成改成并行/延迟，就会静默拿错帧的条件图——绑定一次消除隐患。
                """
                nonlocal spent
                dt = _timer()
                sd = seed + i * gate_seed_step
                if use_two_stage:
                    # 阶段 1：只有骨架 + 文字（没人跟它抢姿态）
                    draft = _qwen_call(
                        eng, prompt=solo_instr, negative=neg, ref_image=_cond,
                        width=width, height=height, steps=steps, cfg=cfg,
                        seed=sd, model=model, label="w4-draft",
                        # 两次调用共用同一组 adapter：不要每步都挂/卸一次 Lightning
                        fast_lora=False)
                    # 阶段 2：角色图在前、草图在后 → 换回身份，姿态保住
                    raw = _qwen_call(
                        eng, prompt=reskin_instr, negative=neg,
                        ref_image=[char, draft],
                        width=width, height=height, steps=steps, cfg=cfg,
                        seed=sd, model=model, label="w4-reskin",
                        fast_lora=False)
                else:
                    raw = _qwen_call(
                        eng, prompt=instr, negative=neg, ref_image=_refs,
                        width=width, height=height, steps=steps, cfg=cfg,
                        seed=sd, model=model, label="w4",
                        output_resolution=max(width, height),
                        fast_lora=not use_vnccs)   # LoRA 槽位归 VNCCS
                info: Dict[str, Any] = {}
                img, _edge, _subj = _postprocess(
                    raw, eng, matte=matte, matte_backend=matte_backend,
                    denoise=denoise, pixel=pixel, palette=palette,
                    clean_iso=clean_iso, info=info)
                for k in ("backend", "fallback"):
                    if k in info:
                        matte_info[k] = info[k]
                gate_img = info.pop("pre_pixel", None) or img
                spent += dt()
                return img.convert("RGBA"), {"gate": gate_img.convert("RGBA"),
                                             "subjects": _subj}

            def _score(img: Image.Image, info: Dict[str, Any], *,
                       _prev=prev_gate) -> Dict[str, Any]:
                ev = quality.evaluate(info["gate"].convert("RGBA"), extractor,
                                      prev=_prev,
                                      min_points=gate_min_points)
                # **单主体判据**：Qwen 偶尔会把头画丢、把帽子画成浮空的一块。
                # 这种帧 DWPose 照样能检出 16 个关节（躯干还在），闸门放它过去；
                # 但前景连通域数会变成 2+，这才是「缺件」的直接证据。
                _subj = info.get("subjects")
                if _subj is not None:
                    ev["blobs"] = _subj
                    if _subj > 1 and ev["passed"]:
                        ev = {**ev, "passed": False,
                              "score": ev["score"] - 20.0,
                              "reject": f"前景有 {_subj} 块（头/手与身体分离）"}
                return ev

            frame, ev, finfo, used, nh = gate_mod.run_candidates(
                generate=_gen, score=_score,
                policy=gate_mod.GatePolicy(
                    candidates=candidates, retries=retries,
                    seed_step=gate_seed_step, stop_on_first_pass=False))
            if nh and not non_humanoid:
                non_humanoid = True
                print("[warn] 主体不是人形（DWPose 检不出关节）："
                      "关闭补提，后续帧只出 1 次")
            if frame is None:
                raise RuntimeError("这一帧没有产出任何候选")
            prev_gate = finfo.get("gate")
            frame_scores.append({**ev, "attempts": used})
            attempts_used += used
            per_frame.append(round(spent, 2))
            frames.append(frame)
    finally:
        if vnccs_pipe is not None:
            eng.unload_loras(vnccs_pipe, scope="temp")

    # 尺度归一：Qwen 没有 ControlNet，逐帧「人物占多大」是模型自己决定的
    # （实测同一段 3 帧能差 3 倍，最小的那帧前景只占 2%，直接被产线判废）。
    frames = pp.normalize_sequence(frames)
    aligned = pp.align_sequence(frames)
    passed = sum(1 for s in frame_scores if s["passed"])
    meta = {
        "workflow": "w4", "scope": scope or "default",
        "prompt": instr, "prompt_body": sem or None, "negative": neg,
        "ref_image": str(ref_image) if ref_image else None,
        "backend": "qwen-vnccs-lora" if use_vnccs else "qwen-pose-ref",
        "approx": True,
        "engine": "vnccs" if use_vnccs else "wpc",
        "lora": (eng.VNCCS_LORA if use_vnccs else None),
        "lora_scale": (float(lora_scale) if use_vnccs else None),
        "n_chars": len(chars),
        # 注意：这里原来还有一行 `"lora": None, "lora_scale": None`——
        # 重复键会让上面刚写的 engine/lora **被静默覆盖成 None**。
        "condition": kind, "bg": bg, "cn_scale": None, "ipa_scale": None,
        "clip_skip": None,
        # 两阶段模式每个候选跑**两次**（草图 + 换皮），总步数要 ×2，
        # 否则产线算「每可用帧成本」会少算一半。
        "steps": _eff_steps(eng, steps, fast=fast_ok) * (2 if use_two_stage else 1),
        "steps_per_pass": _eff_steps(eng, steps, fast=fast_ok),
        "passes": 2 if use_two_stage else 1,
        **_fast_meta(eng, applied=fast_ok),
        "cfg": cfg, "seed": seed, "frames": len(frames),
        "width": width, "height": height,
        # 落盘的是精灵图：列数×格宽。w4 的格子就是生成画布尺寸，
        # 行数由帧数除以列数**向上取整**（w4 没有 rows 参数）。
        "sheet_size": ([cols * aligned[0].width,
                        (-(-len(aligned) // cols)) * aligned[0].height]
                       if aligned else None),
        "matte": matte, "matte_used": matte_info.get("backend"),
        "matte_fallback": matte_info.get("fallback"),
        "model": model, "distill": None,
        "two_stage": use_two_stage,
        "candidates": candidates, "candidates_note": None,
        "gate": bool(gate and extractor is not None),
        "gate_min_points": gate_min_points, "gate_retries": retries,
        "attempts": attempts_used, "gate_passed": passed,
        # 非人形时 gate_passed 必然是 0，这不是失败，得说清楚
        "gate_non_humanoid": non_humanoid,
        "gate_note": ("主体不是人形：DWPose 检不出关节，闸门不适用，"
                      "已跳过补提" if non_humanoid else None),
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
             char_images: Optional[Sequence[str]] = None,
             engine: str = "wpc", lora_scale: float = 1.0,
             steps: int = 16, cfg: float = 1.0, seed: int = -1,
             width: int = 768, height: int = 768,
             cols: int = 8, pixel: int = 0, palette: int = 0,
             matte: bool = False, matte_backend: str = "auto", denoise: int = 1,
             clean_iso: int = 1, export: Optional[str] = None,
             controlnet: str = "openpose", model: str = DEFAULT_MODEL,
             candidates: int = 1, gate: bool = True, gate_min_points: int = 8,
             gate_retries: int = 1, gate_seed_step: int = 1000,
             bg: str = "white", scope: str = "",
             two_stage: bool = False,
             anim_name: str = "walk", fps: int = 10, tag: str = "w4"
             ) -> Dict[str, Any]:
    """姿态序列（**近似实现**）：一批骨架图 → 逐帧生成 → 归一/对齐 → 打包精灵图。

    `controlnet` 这个参数名沿用了 SD 时代，Qwen 只用来区分条件图类型：
    `openpose` → 骨架图（灰度化后当参考图），其它 → 轮廓/剪影。
    也没有 `cn_scale` / `ipa_scale`——Qwen 是双图条件，没有强度旋钮。

    ⚠️ 动作幅度偏保守：腿部常被角色参考图的站姿覆盖（PLAN §2.8 有实测表）。
    `two_stage=True` 走两段式（先出姿态草图再换皮），实测姿态偏差 102px → 77px，
    代价是每帧多一次生成（33s → 50s）。
    """
    eng.require(model, "poseswap")
    pose_images = norm_pose_inputs(pose_images)
    if not pose_images:
        raise ValueError("没有可用的骨架帧（目录里没有 png/jpg，或通配没匹配到）")
    # cols<=0 表示"按帧数自动排一行"（前端把 0 当"自动"传，必须兜住）
    if cols <= 0:
        cols = min(len(pose_images), 8)
    return _qwen_poseswap(
        eng, pose_images=pose_images, prompt=prompt, negative=negative,
        ref_image=ref_image, char_images=char_images,
        engine=engine, lora_scale=lora_scale,
        steps=steps, cfg=cfg, seed=seed,
        width=width, height=height, cols=cols, pixel=pixel, palette=palette,
        matte=matte, matte_backend=matte_backend, denoise=denoise,
        clean_iso=clean_iso, bg=bg, export=export, controlnet=controlnet,
        model=model, candidates=candidates, gate=gate, two_stage=two_stage,
        gate_min_points=gate_min_points, gate_retries=gate_retries,
        gate_seed_step=gate_seed_step, scope=scope,
        anim_name=anim_name, fps=fps, tag=tag)
