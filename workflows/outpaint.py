"""扩图（Outpainting）：把图往某个方向扩展，新区域由模型按场景连续性补全。

## 原理（ComfyUI 的 Outpaint 做法）

Qwen-Image-2.1 没有专门的 outpainting 权重，社区做法是：

1. 把源图贴到一块更大的画布上，**空白处填中性灰**（#808080）
   —— 灰色本身就是"这里要补内容"的信号，模型认识它
2. 挂 **Outpaint LoRA**（`ausboss/Qwen-Image-2.1-Outpaint-LoRA`），
   它让模型偏向"接缝连贯、保持原图"
3. 提示词**必须以 LoRA 的触发句开头**：
   `Outpaint the image: replace the solid gray areas with a seamless
   continuation of the scene, keeping the existing picture unchanged.`
   后面可以接一段 `Scene: ...` 描述（只描述原图里看得见的东西）
4. 采样 → 裁剪回目标画布

## 灰底为什么必须中性

填成别的颜色会**污染新区域的色调**（模型会把它当成场景的一部分）。
ComfyUI 的文档明确写"Keep the gray padding uniform and neutral"。

## 实测（见 docs/PLAN-QWEN-ONLY.md §2.15）

- 768² 源图往右扩一倍（1536×768）→ 8 步约 20s
- 原图区域几乎不变，新区域能延续地面/背景/光照
- 一次扩太大容易跑偏，**建议分多次小幅扩**（ComfyUI 也是这么建议的）
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

#: Outpaint LoRA 的触发句。**必须放在提示词最前面**——LoRA 是照着这句训的，
#: 换个说法它的"接缝连贯"偏置就不生效了。
OUTPAINT_TRIGGER = (
    "Outpaint the image: replace the solid gray areas with a seamless "
    "continuation of the scene, keeping the existing picture unchanged."
)

#: 中性灰。不能用别的颜色：模型会把它当成场景的一部分，污染新区域的色调。
PAD_GRAY = (128, 128, 128)


def pad_canvas(src: Image.Image, *, direction: str = "right",
               expand: float = 0.5) -> Tuple[Image.Image, Tuple[int, int]]:
    """把源图贴到更大的灰底画布上。

    `direction`：`right` / `left` / `up` / `down` / `all`
    `expand`：往该方向扩多少（相对源图对应边长的比例，0.5 = 扩一半）

    返回 `(画布, 源图在画布上的左上角坐标)`。
    """
    w, h = src.size
    d = direction or "right"
    ex = max(0.0, float(expand))
    add_w = int(round(w * ex)) if d in ("left", "right", "all") else 0
    add_h = int(round(h * ex)) if d in ("up", "down", "all") else 0

    if d == "all":
        canvas_w, canvas_h = w + add_w * 2, h + add_h * 2
        ox, oy = add_w, add_h
    elif d == "right":
        canvas_w, canvas_h = w + add_w, h
        ox, oy = 0, 0
    elif d == "left":
        canvas_w, canvas_h = w + add_w, h
        ox, oy = add_w, 0
    elif d == "down":
        canvas_w, canvas_h = w, h + add_h
        ox, oy = 0, 0
    elif d == "up":
        canvas_w, canvas_h = w, h + add_h
        ox, oy = 0, add_h
    else:
        raise ValueError(f"未知方向 {direction}（可选 right/left/up/down/all）")

    # 画布尺寸必须对齐到 **32**，不是 16。
    # QwenImage21Pipeline 里有 `multiple_of = vae_scale_factor * 2 = 32`，
    # 不是 32 的倍数它只 warn 一句就**静默 resize**——于是 meta 里记的画布
    # 尺寸是假的，下游按它算的偏移也全错。踩过一次。
    def _snap32(v: int) -> int:
        return max(256, (int(v) + 31) // 32 * 32)

    canvas_w = _snap32(canvas_w)
    canvas_h = _snap32(canvas_h)
    # 画布必须装得下 (offset + 源图)：不够就再补一格
    if canvas_w < ox + w:
        canvas_w = _snap32(ox + w)
    if canvas_h < oy + h:
        canvas_h = _snap32(oy + h)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), PAD_GRAY + (255,))
    canvas.paste(src.convert("RGBA"), (ox, oy))
    return canvas, (ox, oy)


def build_prompt(scene: str = "", extra: str = "") -> str:
    """组提示词：触发句打头，可选 Scene 描述，再接用户的额外要求。

    `scene` 只该描述**原图里看得见的**东西（模型拿它当上下文），
    不要去描述要扩出来的内容——那部分交给模型自己按连贯性推。
    """
    parts = [OUTPAINT_TRIGGER]
    if scene.strip():
        parts.append(f"Scene: {scene.strip()}")
    if extra.strip():
        parts.append(extra.strip())
    return " ".join(parts)


def outpaint(pipe, *, src: Image.Image, direction: str = "right",
             expand: float = 0.5, prompt: str = "", negative: Optional[str] = None,
             steps: int = 20, cfg: float = 1.0, seed: int = 0,
             lora: str = "qwen21_outpaint", lora_scale: float = 0.9,
             blend: bool = True) -> Dict[str, object]:
    """扩图主流程。返回 `{"image", "canvas", "offset", "work"}`。

    `blend=True` 时把**原图区域**按软边贴回（保证原图像素基本不变）；
    扩出来的区域保留模型结果。`blend=False` 纯看模型输出（对照用）。
    """
    canvas, (ox, oy) = pad_canvas(src, direction=direction, expand=expand)
    w, h = canvas.size

    # 长边超过 1536 时先降一档：Qwen 的开销随面积涨，一次扩太大也容易跑偏
    work = (w, h)
    if max(w, h) > 1536:
        f = 1536 / max(w, h)
        work = (max(256, int(w * f) // 16 * 16), max(256, int(h * f) // 16 * 16))

    full = build_prompt(prompt if prompt else "")
    gen = torch.Generator(pipe._execution_device).manual_seed(int(seed))

    kw: Dict[str, object] = dict(
        prompt=full,
        image=canvas.resize(work, Image.LANCZOS),
        width=work[0], height=work[1],
        num_inference_steps=max(8, min(int(steps), 60)),
        true_cfg_scale=float(cfg),
        generator=gen,
        output_resolution=max(work),
    )
    if float(cfg) > 1.0 and negative:
        kw["negative_prompt"] = negative

    res = pipe(**kw)
    out = res.images[0].convert("RGBA")
    if out.size != canvas.size:
        out = out.resize(canvas.size, Image.LANCZOS)

    if blend:
        # 原图区域贴回：软边过渡，避免接缝处出现一条色差线。
        # 只贴原图那一块（含 8px 羽化），扩出来的区域完全保留模型结果。
        out = _blend_source(out, canvas, src, (ox, oy))

    # 扩展区域的 alpha 强制为**不透明**。
    # Qwen 原生输出 RGBA，没有透明指令时它给的 alpha 是一层软噪声
    # （实测扩图产物里 30% 的像素半透明）——扩图补的是场景，那层 alpha
    # 没有意义，留着会让游戏里背景透出来。原图区域的 alpha 保持源图的。
    out = _opaque_outside(out, src.size, (ox, oy))

    return {"image": out, "canvas": canvas, "offset": (ox, oy), "work": work,
            "prompt": full}


def _blend_source(out: Image.Image, canvas: Image.Image, src: Image.Image,
                  offset: Tuple[int, int], feather: int = 8) -> Image.Image:
    """把原图像素按软边贴回扩图结果。

    **为什么要贴回**：模型会重绘整张画布，原图区域也有 VAE 重建误差
    （哪怕只有几个像素级）。扩图的语义是"扩展"，原图部分不该变。
    """
    ox, oy = offset
    w, h = src.size
    # 羽化蒙版只作用在**原图那一块**上：边缘几个像素与模型结果混合，
    # 内部保持源图像素。不能对整张画布做 composite（尺寸也不一样）。
    patch_mask = Image.new("L", (w, h), 255)
    if feather > 0:
        from PIL import ImageFilter
        patch_mask = patch_mask.filter(ImageFilter.GaussianBlur(feather))
    result = out.copy()
    result.paste(src.convert("RGBA"), (ox, oy), patch_mask)
    return result


def _opaque_outside(out: Image.Image, src_size: Tuple[int, int],
                    offset: Tuple[int, int]) -> Image.Image:
    """把原图矩形**外面**的 alpha 强制设为不透明。

    为什么要做：模型给的 alpha 在「扩图补场景」这个语义下是噪声。
    实测（768² 扩右 50%）：不处理时产物有 30.6% 的半透明像素，
    游戏里会看到背景从新扩展的区域透出来。
    """
    ox, oy = offset
    w, h = src_size
    a = out.getchannel("A")
    keep = Image.new("L", out.size, 0)          # 只有原图那块保留原 alpha
    keep.paste(255, (ox, oy, ox + w, oy + h))
    opaque = Image.new("L", out.size, 255)
    out = out.copy()
    out.putalpha(Image.composite(a, opaque, keep))
    return out


def outpaint_meta(result: Dict[str, object], **extra) -> Dict[str, object]:
    """把扩图的关键信息整理进 meta（前端与产线判据要看）。"""
    canvas = result["canvas"]
    ox, oy = result["offset"]
    meta = {
        "direction": extra.get("direction"),
        "expand": extra.get("expand"),
        "canvas_size": list(canvas.size),
        "offset": [ox, oy],
        "work_size": list(result["work"]),
        "lora": extra.get("lora"),
        "trigger": OUTPAINT_TRIGGER,
    }
    meta.update({k: v for k, v in extra.items() if k not in meta})
    return meta
