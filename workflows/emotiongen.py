"""表情批量生成（只改脸）—— 逐行移植 VNCCS 的 QI2 emotion crop 分支。

**为什么"只改脸"而不是整图重画**：一套角色有 N 个姿势帧，要换成 M 个表情。
整图重画的话动作、衣服、构图每次都会漂；官方只把**脸框那一块**裁出来放大重画，
再按羽化蒙版贴回原图，所以除了脸以外逐像素不变。

官方链路（两处文件，按行抄）：

| 这里 | 官方 |
|---|---|
| `pad_to_square_multiple` | `character_generator.py:1571` `_pad_image_to_square_multiple` |
| `crop_feather_mask` | `:1516` `_crop_feather_mask` |
| `paste_crop_direct` | `:1483` `_paste_crop_direct` |
| `crop_target_size` | `:1636-1643` + ComfyUI `ImageScaleToTotalPixels` |
| `detect_face` | `:1533` `_vnccs_bbox_extract_face` |
| `emotion_text` / `emotion_prompt` | `emotion_generator_v2.py:310` + `:1694` + `:4142-4145` |
| `negative_prompt` | `emotion_generator_v2.py:669-672` |
| `run_batch` | `emotion_generator_v2.py:675-761` 编排 + `:1591-1708` 单张 |

**没搬的**（官方有、但 QI2 分支要么不生效要么要别的模型，见
`docs/角色中心-VNCCS移植评估.md`）：`guide_size`、`guide_size_for`、`max_size`、
`bbox_crop_factor`（QI2 的 detect 里硬编码 1.0）、`noise_mask`、`force_inpaint`、
`use_sam` / `sam_*`、`cycle`、`inpaint_model`、`tiled_*`。

**两处跟我们自己的管线有关的取舍**（都不是猜的，是显存/架构约束）：

1. `crop_max_side` 默认 1024。官方 `target_size` 默认 2048（= 2.0 MP，约 1440²），
   而我们的 Qwen 单卡 24GB 在 1440² 上单张要 60s 以上且显存压线；脸框本身通常
   只有 200~400px，1024² 已经远超源信息量。要按官方原值跑就传 `crop_max_side=2048`。
2. 人脸检测走 **CPU**。Qwen 的 transformer 已经占了卡，YOLO 再挤进去会让出图
   阶段的显存碎片化；一张 1024 的图在 CPU 上检测约 100ms，可以忽略。
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

from workflows import characters
from workflows import paths

# ---------------------------------------------------------------- 官方常量（逐字）

#: 官方 `character_generator.py:740-745`（QI2 版本，**唯一**用在 QI2 上的表情模板）
EMOTION_PROMPT_TEMPLATE = (
    "Upscale face image.\n"
    "Make character's face emotion {emotion}\n"
    "Change only face. Keep original neck colour, clothes and hairs\n"
    "keep character's clothes"
)

#: 官方 `character_generator.py:988`（`bg_remove.preset == native` 时追加）
NATIVE_BACKGROUND_PROMPT = "Transparent background with alpha channel."

#: 官方 `emotion_generator_v2.py:669-672`：表情生成额外加的负向词。
#: **水/汗滴单独列出来**是因为「哭」「难过」这类表情模型会顺手画眼泪，
#: 而精灵帧不要那种东西（眼泪是另一个图层的事）。
EMOTION_NEGATIVE_EXTRA = (
    ", (facial droplet), (water drop), (water), (water droplets), (water drops)"
)

#: 官方 Step3（表情）的七个真正生效的参数，见 `character_generator.py:1591-1708`
#: 与 `:4120-4145`。其余参数的默认值是给 SDXL/FaceDetailer 的，QI2 下无效。
EMOTION_DEFAULTS: Dict[str, Any] = {
    "target_size": 2048,       # → 2.0 MP（不是边长！见 `_resolution_scale_value:994`）
    "bbox_threshold": 0.5,     # Step3 专用（Step1/2 是 0.1）
    "bbox_dilation": 50,       # Step3 专用（Step1/2 是 10）
    "drop_size": 10,
    "feather": 50,             # Step3 专用（Step1/2 是 5）
    "face_denoise": 1.0,       # QI2 强制，workflow 里配的 0.55 无效（`:4136`）
}

#: 裁切推理边长上限（见模块注释第 1 条）
MAX_CROP_SIDE = 1024

#: 官方 `character_generator.py:3370-3371`：贴边重抠区域的**膨胀**与**羽化**半径。
#:
#: 这两个数不是随手定的：膨胀 8px 是为了把"生成图里改了但差异蒙版没涵盖到"的那
#: 一圈（生成模型在边界处会轻微抖动）包进来；羽化 4 是让重抠和原 alpha 之间过渡。
#: 调大会把不该动的头发/肩膀也重抠一遍（那里的幕布色估计会偏）。
DETAILER_MATTE_EXPAND_RADIUS = 8
DETAILER_MATTE_FEATHER_RADIUS = 4

#: 官方 `:3837`：贴边 patch 外面要垫一圈幕布色，抠像才估得准背景色。
DETAILER_CHROMA_CONTEXT = 16

#: 官方 `:2265` 的 latent 对齐步长；`_pad_image_to_square_multiple` 也用它
CROP_MULTIPLE = 32


class EmotionError(RuntimeError):
    """表情生成的可读错误（API 直接把它转成 400 / 503）。"""


class FaceModelMissing(EmotionError):
    """人脸检测权重缺失（官方这里是 `MODEL_MISSING`）。"""


# ---------------------------------------------------------------- 纯几何（零模型）
#
# 这一组是官方那六个函数的 1:1 翻译，**PT 与 torch 两种实现必须等价**：
# 官方吃 HWC float tensor，我们吃 PIL。所有 `:行号` 都指 `character_generator.py`。


def pad_to_square_multiple(
    image: Image.Image, multiple: int = CROP_MULTIPLE,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """正方形补边到 `multiple` 的倍数（官方 `:1571` `_pad_image_to_square_multiple`）。

    **pad 用 replicate（边缘复制）**，不是补 0 也不是补 127（灰）：
    脸框紧贴头发/脖子时，补黑边会让模型顺着黑边画出一条硬线，
    解码后又正好落在羽化带里，贴回去就是一圈脏边。
    """
    w, h = image.size
    side = max(int(multiple), math.ceil(max(w, h) / int(multiple)) * int(multiple))
    pad_w, pad_h = side - w, side - h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    if pad_w == 0 and pad_h == 0:
        return image, (left, top, right, bottom)
    arr = np.asarray(image)
    mode = "edge"                                  # = torch 的 mode="replicate"
    if arr.ndim == 3:
        padded = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode=mode)
    else:
        padded = np.pad(arr, ((top, bottom), (left, right)), mode=mode)
    return Image.fromarray(padded, mode=image.mode), (left, top, right, bottom)


def crop_feather_mask(height: int, width: int, feather: int = 5) -> np.ndarray:
    """裁切块的边缘斜坡蒙版（官方 `:1516` `_crop_feather_mask`）。

    值 = `min(到上下边距离, 到左右边距离) / feather`，clamp 到 [0,1]：
    中间是 1（全用新图），四周 feather 像素内线性降到 0（全用原图），
    所以贴回去没有硬边。
    """
    height = max(1, int(height))
    width = max(1, int(width))
    feather = max(0, int(feather))
    if feather <= 0:
        return np.ones((height, width), dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    y_distance = np.minimum(y, (height - 1) - y).reshape(height, 1)
    x_distance = np.minimum(x, (width - 1) - x).reshape(1, width)
    edge_distance = np.minimum(y_distance, x_distance)
    return np.clip(edge_distance / float(feather), 0.0, 1.0).astype(np.float32)


def paste_mask(height: int, width: int, feather: int = 5) -> np.ndarray:
    """贴回用的蒙版 = 斜坡蒙版再做一次高斯模糊（官方 `_blur_detail_mask:1449`）。

    官方在这里调 Impact Pack 的 `tensor_gaussian_blur_mask(mask, feather)`，
    核半径就是 feather。所以我们先用 PIL 的 `GaussianBlur(radius=feather)`
    复现同一条曲线，再归一化回 0~1。
    """
    ramp = (crop_feather_mask(height, width, feather) * 255.0).round().astype(np.uint8)
    if int(feather) <= 0:
        return ramp.astype(np.float32) / 255.0
    blurred = Image.fromarray(ramp, mode="L").filter(
        ImageFilter.GaussianBlur(radius=float(feather)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def paste_crop_direct(
    base: Image.Image, patch: Image.Image, region: Sequence[int],
    mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """把 patch 按蒙版贴回 base 的 `region`（官方 `:1483` `_paste_crop_direct`）。

    **通道数只到 patch 的通道数**（官方 `patch[:, :, :, :result.shape[-1]]`）：
    patch 是 RGB 时不动 base 的 alpha——这正是"复用原 alpha"快路径成立的前提，
    否则贴一块 alpha=255 的方形补丁回去，透明背景上就多一个方块。

    ⚠️ **调用方必须保证 patch 是 3 通道**。传 RGBA 的话 alpha 会被一起混，
    透明背景就被盖掉了（官方那条路径的 patch 恒为 3 通道，所以没这个问题；
    我们的 `qwen_edit_image` 返回 RGBA，`crop_generate` 里已显式 `convert("RGB")`）。
    """
    x1, y1, x2, y2 = [int(v) for v in region]
    if x2 <= x1 or y2 <= y1:
        return base
    target = (x2 - x1, y2 - y1)
    if patch.size != target:
        patch = patch.resize(target, Image.LANCZOS)
    base_arr = np.asarray(base).astype(np.float32)
    patch_arr = np.asarray(patch).astype(np.float32)
    channels = min(base_arr.shape[-1], patch_arr.shape[-1])
    sub = base_arr[y1:y2, x1:x2, :channels]
    if mask is None:
        base_arr[y1:y2, x1:x2, :channels] = patch_arr[..., :channels]
    else:
        m = mask
        if m.shape != (y2 - y1, x2 - x1):
            m = np.asarray(
                Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), mode="L")
                .resize((x2 - x1, y2 - y1), Image.BILINEAR),
                dtype=np.float32) / 255.0
        m = m[:, :, None]
        base_arr[y1:y2, x1:x2, :channels] = (1.0 - m) * sub + m * patch_arr[..., :channels]
    out = np.clip(base_arr, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(out, mode=base.mode)


#: 官方 `:3643-3647` 的三种幕布色
CHROMA_SCREENS: Dict[str, Tuple[int, int, int]] = {
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
}


def chroma_background_name(background: str) -> str:
    """背景名 → 抠像用的幕布色名（官方 `_emotion_chroma_background:3634-3640`）。

    **白/黑/透明一律当 green** —— 官方注释写得很直白：`A saturated screen is
    substantially more reliable than white/black`。白底抠像会把白毛白甲一起判成背景，
    而表情重抠只处理脸框那一小片，猜错的代价是脸上出现一个洞。
    """
    name = str(background or "").strip().lower()
    return name if name in CHROMA_SCREENS else "green"


def solid_rgb(background: str) -> Tuple[int, int, int]:
    """背景名 → RGB 三元组（官方 `_background_rgb:3642-3649`）。"""
    return CHROMA_SCREENS[chroma_background_name(background)]


def detailer_region_mask(
    delta: np.ndarray,
    expand_radius: int = DETAILER_MATTE_EXPAND_RADIUS,
    feather_radius: int = DETAILER_MATTE_FEATHER_RADIUS,
) -> np.ndarray:
    """差异图 → 要重做的区域蒙版（官方 `_expanded_detailer_region:3683-3706`）。

    官方拿 FaceDetailer 返回的细化蒙版当输入；**我们没有 FaceDetailer**，所以用
    同一文件里 `_detailer_mask_from_result:3667-3681` 的 fallback 分支：
    「生成结果与原图的逐像素差异 > 1e-5」。那本来就是官方为"拿不到蒙版"准备的
    等价物，语义一样——都是"这块被改过了"。

    然后两步（数值照官方）：
      · `max_pool2d(kernel=2r+1, padding=r)` → **膨胀** r 像素；
      · `avg_pool2d(kernel=2r+1, padding=r)` → 平滑，再与原蒙版取 **max**。

    取 max 而不是直接替换成平滑结果，是官方刻意的：单纯平均会让蒙版边缘
    （值 1）被拉低到 0.5 以下，重抠不到边，接缝还在。

    > 膨胀用 PIL 的 `MaxFilter`（C 实现）。它与 `max_pool2d` 的唯一差别是边界：
    > torch 补 -inf、PIL 复制边缘。对蒙版（边缘必为 0 或整片为 1）两者结果相同。
    """
    arr = np.clip(np.asarray(delta, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    mask_img = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L")
    expand = max(0, int(expand_radius))
    if expand:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(expand * 2 + 1))
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    feather = max(0, int(feather_radius))
    if feather:
        softened = np.asarray(
            mask_img.filter(ImageFilter.BoxBlur(feather)), dtype=np.float32) / 255.0
        mask = np.maximum(mask, softened)
    return np.clip(mask, 0.0, 1.0)


def merge_emotion_rgba(
    original: Image.Image, keyed: Image.Image, region: np.ndarray,
) -> Image.Image:
    """按权重把新图混进原图，**在预乘空间里混**（官方 `_merge_emotion_rgba:3707-3748`）。

    直接对 straight-alpha 的 RGB 做 lerp 是错的：那块原来的 alpha 可能是 0.3
    （半透明发梢），新图是全不透明，混完颜色会偏亮。官方先乘回 alpha 再插值，
    之后 un-premultiply 还原，所以半透明区域的过渡是物理正确的。

    三个细节照抄：
      · `alpha = lerp(old_a, new_a, w)`；
      · `premult = lerp(rgb*old_a, rgb*new_a, w)`；
      · `alpha <= 1e-6` 时退回 straight 空间的 lerp（否则除零）；
      · `w <= 1e-6` 的像素**直接保留原图**（不是"混 0 权重"，避免舍入误差啃边）。
    """
    base = np.asarray(original.convert("RGBA"), dtype=np.float32) / 255.0
    h, w = base.shape[:2]
    patch = keyed.convert("RGBA")
    if patch.size != (w, h):
        patch = patch.resize((w, h), Image.LANCZOS)
    new = np.asarray(patch, dtype=np.float32) / 255.0

    weight = np.asarray(region, dtype=np.float32)
    if weight.shape != (h, w):
        weight = np.asarray(
            Image.fromarray((np.clip(weight, 0, 1) * 255).astype(np.uint8), mode="L")
            .resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    weight = np.clip(weight, 0.0, 1.0)[:, :, None]

    old_a = np.clip(base[..., 3:4], 0.0, 1.0)
    new_a = np.clip(new[..., 3:4], 0.0, 1.0)
    alpha = old_a + (new_a - old_a) * weight
    premult = (base[..., :3] * old_a) + ((new[..., :3] * new_a) - (base[..., :3] * old_a)) * weight
    straight = np.where(
        alpha > 1e-6,
        premult / np.maximum(alpha, 1e-6),
        (base[..., :3] + (new[..., :3] - base[..., :3]) * weight),
    )
    merged = np.concatenate([np.clip(straight, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)], axis=-1)
    out = np.where(weight <= 1e-6, base, merged)
    return Image.fromarray(
        (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGBA")


def chroma_key_patch(
    patch: Image.Image, chroma_key: Callable[[Image.Image], Image.Image],
    background: str = "Green", context: int = DETAILER_CHROMA_CONTEXT,
) -> Image.Image:
    """把一块 patch 贴到幕布色画布上再抠像，最后切回来（官方 `:3846-3896`）。

    **两件事都必须做**：

    1. **先把透明区合成到幕布色上**（官方 `_prepare_emotion_detailer_input:3651-3666`：
       `rgb * alpha + screen * (1 - alpha)`）。不合成的话透明处 RGB 是 **0（黑）**，
       抠像只认幕布色，黑块会被整块当"前景"留下来 —— 真机实测：52.5% 的透明背景
       原样变成不透明黑，整张图失去透明。
    2. **四周垫一圈幕布色**：`matte_screen` 按"图边框颜色"估幕布色，patch 四周
       全是角色像素时估出来是肤色/发色，抠像会把整块吃掉。官方垫
       `border = max(4, context)` 宽，边框就一定是幕布色。

    `chroma_key` 失败（抛异常）时返回原 patch —— 抠像失败不该丢图。
    """
    screen = solid_rgb(background)
    if patch.mode == "RGBA":
        flat = Image.alpha_composite(Image.new("RGBA", patch.size, screen + (255,)),
                                     patch).convert("RGB")
    else:
        flat = patch.convert("RGB")
    w, h = flat.size
    border = max(4, int(context))
    canvas = Image.new("RGB", (w + border * 2, h + border * 2), screen)
    canvas.paste(flat, (border, border))
    try:
        keyed = chroma_key(canvas)
    except Exception:                                  # noqa: BLE001
        # 抠像失败就把**原 patch** 还回去（保留原来的 alpha），不是压平后的
        return patch.convert("RGBA")
    if keyed is None:
        return patch.convert("RGBA")
    keyed = keyed.convert("RGBA")
    return keyed.crop((border, border, border + w, border + h))


def has_transparency(image: Image.Image, threshold: int = 250) -> bool:
    """这张图有没有真正的透明像素（决定走哪条抠像路）。

    阈值 250 而不是 255：JPEG 往返、PNG 重采样之后 alpha 常留下 252~254 的
    \"几乎不透明\"值，按 255 判会把这些图当成有透明，然后对着一条几乎看不见的边
    做重抠 —— 判据是官方 `source_alpha < 0.999`（≈ 254.7），这里取整到 250。
    """
    if image.mode not in ("RGBA", "LA", "PA", "P"):
        return False
    try:
        alpha = image.convert("RGBA").getchannel("A")
    except (ValueError, OSError):
        return False
    return alpha.getextrema()[0] < threshold


def refine_alpha_edges(
    *,
    full: Image.Image,
    source: Image.Image,
    region: Tuple[int, int, int, int],
    chroma_key: Optional[Callable[[Image.Image], Image.Image]] = None,
    background: str = "Green",
    expand_radius: int = DETAILER_MATTE_EXPAND_RADIUS,
    feather_radius: int = DETAILER_MATTE_FEATHER_RADIUS,
    context: int = DETAILER_CHROMA_CONTEXT,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """贴回之后的 **alpha 精修**（官方 `_run_emotion_bg_remove:3750-3901` 的单帧版）。

    ## 这一步在修什么

    贴回只混 RGB、不动 alpha（`paste_crop_direct` 只取 patch 的通道数）。所以
    "生成图里那片应该是透明的"（比如一缕头发飘出轮廓、耳朵后面露出背景）在整图上
    还是**不透明**的 —— 结果是脸上贴着一块幕布色的脏斑。这一步就是把那片重抠。

    ## 官方的判据：贴边才重抠

    脸框**碰到原图的透明边**才需要重抠；完全落在不透明区域里的（`interior_only`）
    直接复用原 alpha —— 省掉一次抠像，而且避免在纯肤色区域误判幕布色。

    ## 为什么 `interior_only` 在这可以直接跳过

    官方那两个分支的净效果不同：

      · `interior_only`：`keyed` 就是**贴回后的整图**本身，`weight=1` 处取它、
        `weight=0` 处取原图 —— 而贴回时框外一个像素都没动，所以**结果 == 输入**。
        官方走一遍 `_merge_emotion_rgba` 只是为了让代码不分支。
      · 贴边：`keyed` 是**抠过像**的，RGB 去了溢色、alpha 变了，才是真正的改动。

    所以这里只在贴边时动手，等价于官方，少一次无意义的整块插值。

    ## 坐标

    官方在**整图尺寸**上算 region（`detailer_mask` 就是框矩形，见 `:1692-1698`，
    QI2 分支没有 FaceDetailer）。这里同样在整图上算，然后只把「真正需要重抠的
    那一小块 + context」裁出来处理 —— 抠像按边框估幕布色，块越小越准。
    """
    src = source.convert("RGBA")
    w, h = src.size
    out = full.convert("RGBA")
    if out.size != (w, h):
        out = out.resize((w, h), Image.LANCZOS)

    # ---- detailer mask = 裁切框（官方 :1692-1698 那个 zeros + 框内填 1） ----
    raw_mask = np.zeros((h, w), dtype=np.float32)
    x1, y1, x2, y2 = [int(v) for v in region]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return out, {"touched": False, "reason": "空框"}
    raw_mask[y1:y2, x1:x2] = 1.0
    weight = detailer_region_mask(raw_mask, expand_radius, feather_radius)

    src_alpha = np.asarray(src, dtype=np.float32)[..., 3] / 255.0
    # 官方 `:3812`：region 里有像素落在半透明/透明处 → 贴边
    touches = bool(np.any((weight > 1e-4) & (src_alpha < 0.999)))
    if not touches:
        return out, {"touched": False, "reason": "整块都在不透明区域内"}

    if chroma_key is None:
        # 没有抠像后端：只能保留原 alpha（= 现状），但要说清楚，别让调用方以为修过了
        return out, {"touched": True, "keyed": False,
                     "reason": "没有抠像后端，贴边区域未重抠"}

    # ---- 只处理需要重抠的那一块（官方 `:3823-3826` 的 bbox + context） ----
    ys, xs = np.nonzero(weight > 1e-4)
    cy0 = max(0, int(ys.min()) - context)
    cy1 = min(h, int(ys.max()) + 1 + context)
    cx0 = max(0, int(xs.min()) - context)
    cx1 = min(w, int(xs.max()) + 1 + context)

    # **保留 RGBA**：透明区在送抠像前要合成成幕布色（见 chroma_key_patch）。
    # 先 convert("RGB") 会把透明压成黑，黑块抠不掉，整张图就失去透明了。
    patch = out.crop((cx0, cy0, cx1, cy1))
    keyed = chroma_key_patch(patch, chroma_key, background, context=context)
    merged = merge_emotion_rgba(
        src.crop((cx0, cy0, cx1, cy1)), keyed,
        weight[cy0:cy1, cx0:cx1])
    result = out.copy()
    result.paste(merged, (cx0, cy0))
    return result, {
        "touched": True, "keyed": True,
        "region": [cx0, cy0, cx1, cy1],
        "screen": chroma_background_name(background),
    }


def cut_padding(
    patch: Image.Image, padding: Tuple[int, int, int, int],
) -> Image.Image:
    """把 pad 反向切掉（官方 `:1671-1675`）。

    官方是 `paste_crop[:, top:aligned_h-bottom, left:aligned_w-right]` ——
    **右下用的是 aligned - bottom 而不是 aligned - bottom + 1**，因为 slice 的
    终点是开区间。这里用 `crop(left, top, w-right, h-bottom)`，语义一致。
    """
    left, top, right, bottom = padding
    w, h = patch.size
    return patch.crop((left, top, w - right, h - bottom))


def crop_target_size(
    width: int, height: int, target_size: int = EMOTION_DEFAULTS["target_size"],
) -> Tuple[int, int]:
    """裁切块的推理尺寸（官方 `:1636-1643` + ComfyUI `ImageScaleToTotalPixels`）。

    `target_size` **不是边长**，是"总像素档位"：`megapixels = target_size / 1024`
    （`_resolution_scale_value:994` 先 clamp 到 1024~4096）。然后按总像素等比放大，
    边长**向下取整到 32 的倍数**（`resolution_steps=32`）——向上取整会让图比
    目标大一档，白算 token。
    """
    w = max(1, int(width))
    h = max(1, int(height))
    try:
        scale_value = int(round(float(target_size)))
    except (TypeError, ValueError):
        scale_value = int(EMOTION_DEFAULTS["target_size"])
    megapixels = max(1024, min(4096, scale_value)) / 1024.0
    total = megapixels * 1024.0 * 1024.0
    factor = math.sqrt(total / float(w * h))
    tw = max(CROP_MULTIPLE, int(round(w * factor)) // CROP_MULTIPLE * CROP_MULTIPLE)
    th = max(CROP_MULTIPLE, int(round(h * factor)) // CROP_MULTIPLE * CROP_MULTIPLE)
    return tw, th


def clamp_size(
    width: int, height: int, max_side: int = MAX_CROP_SIDE,
) -> Tuple[int, int]:
    """把推理尺寸等比压到长边 ≤ `max_side`，边长仍取 32 的倍数（向下）。"""
    limit = max(CROP_MULTIPLE, int(max_side))
    long_side = max(width, height)
    if long_side <= limit:
        return int(width), int(height)
    f = limit / float(long_side)

    def snap(v: float) -> int:
        return max(CROP_MULTIPLE, int(v * f) // CROP_MULTIPLE * CROP_MULTIPLE)

    return snap(width), snap(height)


# ---------------------------------------------------------------- 人脸检测（YOLO）


_FACE_MODEL: Any = None


def face_model_path() -> str:
    return os.path.join(str(paths.BBOX_DIR), "face_yolov8m.pt")


def load_face_model() -> Any:
    """懒加载 YOLO 人脸检测（官方 = Impact Pack 的 `UltralyticsDetectorProvider`）。

    权重路径照官方 `bbox/face_yolov8m.pt`（`character_generator.py:845`），
    是 ComfyUI 生态的惯例位置，我们放 `models/bbox/`。
    """
    global _FACE_MODEL
    if _FACE_MODEL is not None:
        return _FACE_MODEL
    path = face_model_path()
    if not os.path.isfile(path):
        raise FaceModelMissing(
            f"缺人脸检测权重：{path}（官方 bbox/face_yolov8m.pt）")
    try:
        from ultralytics import YOLO
    except ImportError as exc:                       # pragma: no cover - 环境问题
        raise EmotionError(f"没装 ultralytics：{exc}") from exc
    _FACE_MODEL = YOLO(path)
    return _FACE_MODEL


def detect_face(
    image: Image.Image, *, threshold: float = EMOTION_DEFAULTS["bbox_threshold"],
    dilation: int = EMOTION_DEFAULTS["bbox_dilation"],
    drop_size: int = EMOTION_DEFAULTS["drop_size"],
    model: Any = None,
) -> Optional[Tuple[int, int, int, int]]:
    """找出**第一个**人脸框，按官方规则外扩后返回 `(x1, y1, x2, y2)`。

    官方 `:1533` `_vnccs_bbox_extract_face` 有两层外扩：把 `dilation` 传给
    Impact 的 `detect()`（它在内部已经按 dilation 扩过一次），拿到 `crop_region`
    后又**手动再扩一次** `x1-dilation …`。所以净效果是 `2 × dilation`（默认 100px）
    ——这是照抄官方行为，不是笔误。脸框大一圈是有意的：要带上头发、脖子和
    一点肩膀，模型才知道这张脸长在什么身上。

    返回 `None` 表示没检出脸——官方此时**整图重画**（`:1626-1632`）。
    """
    w, h = image.size
    try:
        detector = model if model is not None else load_face_model()
        # device="cpu"：见模块注释第 2 条（不给 Qwen 抢显存）
        results = detector.predict(np.asarray(image.convert("RGB")),
                                   conf=float(threshold), verbose=False, device="cpu")
    except EmotionError:
        raise
    except Exception as exc:                          # noqa: BLE001
        raise EmotionError(f"人脸检测失败：{exc}") from exc
    if not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
    # 官方取 `regions[0]`；立绘只有一张脸，取置信度最高的那个等价且更稳
    best = int(np.argmax(conf)) if conf is not None and len(conf) else 0
    x1, y1, x2, y2 = [float(v) for v in xyxy[best][:4]]
    if (x2 - x1) < int(drop_size) or (y2 - y1) < int(drop_size):
        return None
    grow = int(dilation) * 2
    x1 = max(0, int(round(x1)) - grow)
    y1 = max(0, int(round(y1)) - grow)
    x2 = min(w, int(round(x2)) + grow)
    y2 = min(h, int(round(y2)) + grow)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------- prompt


def emotion_text(
    natural_prompt: str, description: str, fallback_emotion: str,
) -> str:
    """官方 `emotion_generator_v2.py:310-318` `build_anima_emotion_prompt`。

    表情库每条都有 `natural_prompt`（一句自然语言）与 `description`
    （danbooru 风格标签）。**两句一起给**：自然语言决定"要改成什么"，
    标签再钉住具体特征（`furrowed_brow` 这种模型认得，光说 "angry" 会飘）。

    `natural_prompt` 为空时退回 `The character expresses <key>.`（官方同款兜底）。
    """
    text = str(natural_prompt or "").strip()
    if not text:
        text = f"The character expresses {str(fallback_emotion or '').replace('-', ' ')}."
    tags = str(description or "").strip()
    if tags:
        text = f"{text}\n\nEmotion Tags: {tags}"
    return text


def emotion_prompt(
    natural_prompt: str, description: str, fallback_emotion: str,
    *, template: Optional[str] = None, transparent: bool = False,
) -> str:
    """官方 `:1694` `_qi2_emotion_prompt` + `:4142-4145`。

    QI2 分支**只走模板**：不拼 `face_details`、不拼种族/发型/服装
    （那是 SDXL/FaceDetailer 分支的事）。模板里没有 `{emotion}` 占位时
    追加一行（官方 else 分支）。

    `transparent=True` 时在末尾补 `NATIVE_BACKGROUND_PROMPT`（官方 `:4145`）。
    """
    selected = emotion_text(natural_prompt, description, fallback_emotion)
    tpl = str(template if template is not None else EMOTION_PROMPT_TEMPLATE).strip()
    if not tpl:
        out = selected
    elif "{emotion}" in tpl:
        out = tpl.replace("{emotion}", selected).strip()
    else:
        out = f"{tpl}\n{selected}".strip()
    if transparent:
        out = f"{out}\n{NATIVE_BACKGROUND_PROMPT}"
    return out


def negative_prompt(base_negative: str) -> str:
    """官方 `emotion_generator_v2.py:669-672`：角色负向词 + 表情专用尾巴。"""
    return f"{str(base_negative or '')}{EMOTION_NEGATIVE_EXTRA}"


# ---------------------------------------------------------------- 表情库


_EMOTION_CACHE: Optional[Dict[str, Any]] = None


def emotion_catalog() -> Dict[str, Any]:
    """官方表情库（`web/public/vnccs/emotions/emotions.json`，前端取同一份）。"""
    global _EMOTION_CACHE
    if _EMOTION_CACHE is None:
        path = os.path.join(str(paths.VNCCS_EMOTION_DIR), "emotions.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _EMOTION_CACHE = json.load(f)
        except (OSError, ValueError):
            _EMOTION_CACHE = {}
    return _EMOTION_CACHE


def flat_emotions() -> List[Dict[str, Any]]:
    """拍平成一条条表情（带 `category`），顺序固定（官方 json 的顺序）。"""
    out: List[Dict[str, Any]] = []
    for category, items in (emotion_catalog() or {}).items():
        for item in items or []:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["category"] = category
            out.append(entry)
    return out


def find_emotion(key: str) -> Optional[Dict[str, Any]]:
    """按 `safe_name` / `key` 查一条表情（大小写、空格、下划线都不敏感）。"""
    target = " ".join(str(key or "").replace("_", " ").lower().split())
    if not target:
        return None
    for entry in flat_emotions():
        for field in ("safe_name", "key"):
            value = " ".join(str(entry.get(field) or "").replace("_", " ").lower().split())
            if value and value == target:
                return entry
    return None


def resolve_emotions(keys: Sequence[str]) -> List[Dict[str, Any]]:
    """把前端发来的表情标识收敛成表情库条目。

    **认不出的标识不丢弃也不报错**：它就是一个自定义表情名（`ensure_safe_name`
    约束过路径安全），没有 `description` / `natural_prompt` 也能跑——官方
    此时退回 `The character expresses <key>.`。真正一条都不剩才报错。
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for key in keys or []:
        raw = str(key or "").strip()
        if not raw:
            continue
        entry = find_emotion(raw)
        if entry is None:
            entry = {"key": raw, "safe_name": raw,
                     "description": "", "natural_prompt": ""}
        safe_name = characters.ensure_safe_name(
            str(entry.get("safe_name") or entry.get("key")), "表情名")
        if safe_name.casefold() in seen:
            continue
        seen.add(safe_name.casefold())
        entry = dict(entry)
        entry["safe_name"] = safe_name
        out.append(entry)
    if not out:
        raise EmotionError("没有选中任何表情")
    return out


# ---------------------------------------------------------------- 单张处理


def crop_generate(
    source: Image.Image,
    *,
    box: Optional[Tuple[int, int, int, int]],
    prompt: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    target_size: int = EMOTION_DEFAULTS["target_size"],
    max_side: int = MAX_CROP_SIDE,
    feather: int = EMOTION_DEFAULTS["feather"],
    edit: Callable[..., Image.Image],
    # ---- 第二段：官方 `_run_emotion_bg_remove` 的 alpha 精修 ----
    refine_alpha: bool = True,
    chroma_key: Optional[Callable[[Image.Image], Image.Image]] = None,
    background: str = "Green",
    matte_expand_radius: int = DETAILER_MATTE_EXPAND_RADIUS,
    matte_feather_radius: int = DETAILER_MATTE_FEATHER_RADIUS,
    chroma_context: int = DETAILER_CHROMA_CONTEXT,
) -> Dict[str, Any]:
    """一张源图 → 换好表情的整图（**纯几何 + 一次 `edit` 调用**）。

    `edit(image, prompt, negative, seed, steps, cfg) -> PIL.Image`。
    单拎出来是为了能在没有 GPU 的测试里用一个假的 `edit` 跑通整条几何链路
    （pad 对齐、贴回坐标、羽化无接缝都能验）。

    `refine_alpha`：贴回之后再走官方的第二段（`_run_emotion_bg_remove`），
    **只在脸框碰到原图透明边时**才重抠那一小片。源图不透明时这一步什么都不做
    （见 `refine_alpha_edges` 的说明）。
    """
    src = source.convert("RGBA")
    w, h = src.size
    # 官方 `:1626-1632`：没检出脸就整图重画（crop_region 就是整张图）
    region = (0, 0, w, h) if box is None else (
        max(0, int(box[0])), max(0, int(box[1])),
        min(w, int(box[2])), min(h, int(box[3])))
    crop = src.crop(region)
    aligned, padding = pad_to_square_multiple(crop, CROP_MULTIPLE)
    tw, th = clamp_size(*crop_target_size(*aligned.size, target_size), max_side=max_side)
    scaled = aligned if aligned.size == (tw, th) else aligned.resize((tw, th), Image.LANCZOS)
    generated = edit(scaled, prompt, negative, seed, steps, cfg)
    if generated is None:
        raise EmotionError("出图没有返回图像")
    if generated.size != (tw, th):
        generated = generated.resize((tw, th), Image.LANCZOS)
    # 官方 `:1668-1675`：生成图缩回 aligned 尺寸，再把 pad 切掉
    back = generated.resize(aligned.size, Image.LANCZOS)
    patch = cut_padding(back, padding)
    # **贴回用的块必须转 3 通道**。`paste_crop_direct` 按 patch 的通道数决定混几个
    # 通道（官方 `patch[:, :, :, :result.shape[-1]]`），而我们的 `qwen_edit_image`
    # 返回的是 **RGBA**（`generate.py:455` 的 `convert("RGBA")`）——
    # 不转的话 alpha 会跟着一起混进去：源图的透明背景被 patch 的 255 按羽化权重
    # 盖掉。真机实测：900² 源图 52.5% 透明，贴回后背景 alpha 变成 61，
    # 再走 refine 就彻底不透明了（看起来像"透明功能失效"）。
    # 官方 patch 恒为 3 通道，alpha 由原图复用 —— 这才是"只混 RGB"的前提。
    patch = patch.convert("RGB")
    crop_h = region[3] - region[1]
    crop_w = region[2] - region[0]
    mask = paste_mask(crop_h, crop_w, feather)
    full = paste_crop_direct(src, patch, region, mask)
    refine_info: Dict[str, Any] = {"touched": False, "reason": "未启用"}
    if refine_alpha:
        full, refine_info = refine_alpha_edges(
            full=full, source=src, region=region,
            chroma_key=chroma_key, background=background,
            expand_radius=matte_expand_radius,
            feather_radius=matte_feather_radius,
            context=chroma_context,
        )
    return {
        "image": full,            # 贴回后的整图（存 Sprites/）
        "face": generated,        # 生成的裁切块原图（存 Faces/，官方 `:4638`）
        "box": region,
        "crop_size": [crop_w, crop_h],
        "aligned_size": list(aligned.size),
        "work_size": [tw, th],
        "padding": list(padding),
        "detected": box is not None,
        "alpha_refine": refine_info,
    }


# ---------------------------------------------------------------- 批量


def _write_temp_png(image: Image.Image) -> str:
    """落盘交给 `characters.publish_image`，所以先在产物目录里写个临时文件。

    写在系统临时目录（不是 assets）—— 它只是转手用的，publish 完就删。
    """
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".png", prefix="meow_emo_")
    os.close(fd)
    image.save(path, format="PNG")
    return path


def run_batch(
    engine: Any,
    params: Dict[str, Any],
    *,
    job: Optional[Dict[str, Any]] = None,
    edit: Optional[Callable[..., Image.Image]] = None,
    detect: Optional[Callable[..., Optional[Tuple[int, int, int, int]]]] = None,
    matte_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
) -> Dict[str, Any]:
    """表情批量生成（官方 `emotion_generator_v2.py:675-761` 的编排）。

    两层循环：事件序 = 表情外层、源帧内层（官方一致）。**同一个种子跑整批**
    （官方 `pipe_seed` 是整批共用的），这样同一批里只有表情不同，别的都一致。

    单张失败**不中断整批**：50 张里第 3 张崩了，不该把前两张的成果一起丢掉。
    失败的记进 `failed`，前端能看见是哪一张、什么原因。
    """
    name = characters.ensure_safe_name(str(params.get("character") or ""), "角色名")
    costume = characters.ensure_safe_name(str(params.get("costume") or "Naked"), "服装名")
    if not characters.character_exists(name):
        raise EmotionError(f"没有这个角色：{name}")

    cfg_data = characters.load_config(name) or {}
    info = cfg_data.get("character_info") or {}
    emos = resolve_emotions(params.get("emotions") or [])
    negative = negative_prompt(str(info.get("negative_prompt") or ""))

    steps = int(params.get("steps") or 25)
    cfg_scale = float(params.get("cfg") or 3.0)
    target_size = int(params.get("target_size") or EMOTION_DEFAULTS["target_size"])
    max_side = int(params.get("crop_max_side") or MAX_CROP_SIDE)
    feather = int(params.get("feather") if params.get("feather") is not None
                  else EMOTION_DEFAULTS["feather"])
    do_matte = bool(params.get("matte"))
    # 背景默认**留空**（= 走原生透明）。以前默认 "Green" 会把整条链路拖回
    # "绿幕 + 抠图"，那正是用户抱怨的"我没选抠图你为啥要抠"。
    background = str(params.get("background_color")
                     or info.get("background_color") or "")
    # 透明判定同 posegen：显式 `transparent` 优先；否则实色幕布出实色图、
    # 其余走原生 alpha。以前默认 False = 强制"绿幕 + 抠图"。
    _tr = params.get("transparent")
    if _tr is None:
        transparent = background.strip().lower() not in {
            "green", "blue", "white", "grey", "magenta"}
    else:
        transparent = bool(_tr)

    sources = characters.source_sprite_paths(name, costume)
    if not sources:
        raise EmotionError(
            f"「{name}」的「{costume}」没有可用源帧：表情是从 Neutral 姿势帧上改脸的，"
            "先在「出图」页生成一张底图")
    wants = params.get("poses")
    wanted = set(int(v) for v in wants) if wants else None
    targets = [(i, p) for i, p in enumerate(sources, start=1)
               if wanted is None or i in wanted]
    if not targets:
        raise EmotionError(f"选中的源帧序号不在范围里（共 {len(sources)} 张）")

    seed = params.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(
        os.urandom(3), "big")

    if edit is None:
        from workflows import generate as gen
        edit = _default_edit(engine, transparent=transparent)
    if matte_fn is None:
        matte_fn = _default_matte
    detector = detect or detect_face

    total = len(targets) * len(emos)
    done = 0
    started = time.time()
    items: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for emo in emos:
        prompt = emotion_prompt(
            emo.get("natural_prompt", ""), emo.get("description", ""),
            emo.get("key") or emo.get("safe_name") or "",
            transparent=transparent)
        for index, src_path in targets:
            done += 1
            if job is not None:
                job["progress"] = {
                    "done": done - 1, "total": total,
                    "current": f"{emo.get('safe_name')} / 第 {index} 帧",
                }
            t0 = time.time()
            try:
                with Image.open(src_path) as opened:
                    source = opened.convert("RGBA")
                box = detector(source)
                # **源图有没有 alpha 决定走哪条抠像路**：
                #   · 已经是透明底 → 官方这条路：贴回后只在「脸框碰到透明边」处
                #     重抠那一小片（`refine_alpha_edges`），整图不再动；
                #   · 还是纯色幕布（全不透明）→ 没有 alpha 可复用，只能整图抠。
                # 以前不分情况一律整图抠，透明源图会被再抠一遍，容易啃掉半透明发梢。
                source_alpha = has_transparency(source)
                out = crop_generate(
                    source, box=box, prompt=prompt, negative=negative,
                    seed=seed, steps=steps, cfg=cfg_scale, target_size=target_size,
                    max_side=max_side, feather=feather, edit=edit,
                    refine_alpha=source_alpha,
                    chroma_key=matte_fn if source_alpha else None,
                    background=background)
                image = out["image"]
                if do_matte and not source_alpha and matte_fn is not None:
                    try:
                        image = matte_fn(image)
                    except Exception as exc:          # noqa: BLE001
                        # 抠像失败不丢图：原图还在，把原因记进这一条就行
                        out["matte_error"] = str(exc)
                sprite_path = _publish(name, image, costume, emo["safe_name"], index, "sprite")
                face_path = _publish(name, out["face"], costume, emo["safe_name"], index, "face")
                items.append({
                    "index": index, "emotion": emo["safe_name"],
                    "source": src_path, "sprite": sprite_path, "face": face_path,
                    "box": list(out["box"]), "detected": out["detected"],
                    "work_size": out["work_size"],
                    "alpha_refine": out.get("alpha_refine"),
                    "seconds": round(time.time() - t0, 2),
                })
                if job is not None:
                    job["progress"]["done"] = done
            except Exception as exc:                  # noqa: BLE001
                failed.append({"index": index, "emotion": emo["safe_name"],
                               "error": f"{type(exc).__name__}: {exc}"})
                if job is not None:
                    job["progress"]["done"] = done
    if job is not None:
        job["progress"] = {"done": total, "total": total, "current": ""}
    return {
        "character": name, "costume": costume,
        "emotions": [e["safe_name"] for e in emos],
        "seed": seed, "count": len(items), "total": total,
        "seconds": round(time.time() - started, 2),
        "items": items, "failed": failed,
    }


def _publish(name: str, image: Image.Image, costume: str, emotion: str,
             index: int, kind: str) -> str:
    """按官方命名落盘（Sprites / Faces），返回最终路径。"""
    tmp = _write_temp_png(image)
    try:
        if kind == "face":
            return characters.publish_face_image(name, tmp, costume, emotion, index)
        return characters.publish_image(
            name, tmp, costume=costume, emotion=emotion,
            set_cover=False, index=index)["path"]
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _default_edit(engine: Any, *, transparent: bool) -> Callable[..., Image.Image]:
    from workflows import generate as gen

    def _edit(image: Image.Image, prompt: str, negative: str,
              seed: int, steps: int, cfg: float) -> Image.Image:
        return gen.qwen_edit_image(
            engine, image=image, prompt=prompt, negative=negative,
            seed=seed, steps=steps, cfg=cfg, transparent=transparent,
            label="wemo")

    return _edit


def _default_matte(image: Image.Image) -> Image.Image:
    from workflows import matte_screen

    out = matte_screen.matte_pil(image)
    return out if out is not None else image
