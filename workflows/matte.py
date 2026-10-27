"""抠图 / 去白边：纯色背景 → 硬边 alpha，并清掉混色边。

为什么不用大模型（u2net / BiRefNet）：
生成的精灵背景本来就是**平坦纯色**（提示词契约里写死了 plain flat background），
这种图用连通域色键就能拿到硬边，几毫秒出结果、零权重、完全确定。
语义分割留给真照片或复杂背景——那时再接 BiRefNet。

三个关键点：
1. 只把**与画面边界连通**的同色区域当背景。否则角色身上与背景同色的部分
   （白猫的肚子对白底）会被一起抠掉。
2. 去白边 = 清掉「紧贴背景、且颜色明显被背景混过」的那一圈像素。判据用
   混色比例：|边缘色−背景| / |角色内侧色−背景| < 阈值 时才清。
3. alpha 一律二值（0/255）。像素素材要硬边，半透明会引入新的脏边。
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def estimate_bg_color(img: Image.Image, border: int = 2) -> tuple[int, int, int]:
    """取四边各 border 像素的众数作为背景色。"""
    arr = np.asarray(img.convert("RGB"))
    strips = [arr[:border].reshape(-1, 3), arr[-border:].reshape(-1, 3),
              arr[:, :border].reshape(-1, 3), arr[:, -border:].reshape(-1, 3)]
    px = np.concatenate(strips, axis=0)
    colors, counts = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
    return tuple(int(v) for v in colors[counts.argmax()])


def matte(img: Image.Image, bg: Iterable[int] | None = None, tol: int = 32,
          fringe: bool = True, fringe_ratio: float = 3.0,
          fringe_passes: int = 3) -> Image.Image:
    """纯色背景抠图，返回 RGBA（alpha 只有 0/255）。

    bg: 背景色；None 时从画面四边自动估计。
    tol: 判为同色的最大通道差。
    fringe: 是否去掉混色边（白边/绿边）。
    fringe_ratio: 混色边判据的倍数（越大越保守）。
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    bg_arr = np.array(list(bg) if bg is not None else estimate_bg_color(img), dtype=np.int16)
    dist = np.abs(rgb - bg_arr).max(axis=2)

    near = (dist <= tol).astype(np.uint8)
    n, lab = cv2.connectedComponents(near, connectivity=4)
    border_labels = set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) \
        | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    border_labels.discard(0)                        # 0 是「非背景色」那一类
    bg_mask = np.isin(lab, list(border_labels)) if border_labels else np.zeros_like(near, bool)

    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)

    if fringe:
        # 混色边通常 1~3 px 厚，单次膨胀只能剥掉最外一圈，所以要迭代。
        k = np.ones((3, 3), np.uint8)
        for _ in range(max(1, fringe_passes)):
            cur_bg = alpha == 0
            touching = cv2.dilate(cur_bg.astype(np.uint8), k).astype(bool)
            blended = touching & (~cur_bg) & (dist <= tol * fringe_ratio)
            if not blended.any():
                break
            alpha[blended] = 0

    out = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA")


def has_fringe(rgba: Image.Image, bg=(255, 255, 255), ratio: float = 1.6) -> int:
    """诊断：统计仍像混色边的像素数（越大说明白边越明显）。"""
    arr = np.asarray(rgba.convert("RGBA")).astype(np.int16)
    a = arr[..., 3]
    if a.min() == 255:
        return 0
    rgb = arr[..., :3]
    dist = np.abs(rgb - np.array(bg, dtype=np.int16)).max(axis=2)
    near_transparent = cv2.dilate((a == 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge = near_transparent & (a > 0)
    return int((edge & (dist < 255 / ratio * 0.5)).sum())


def trim(rgba: Image.Image, pad: int = 0) -> Image.Image:
    """裁到 alpha 包围盒（抠图后通常顺手裁掉多余留白）。"""
    bbox = rgba.getchannel("A").getbbox()
    if not bbox:
        return rgba
    l, t, r, b = bbox
    l, t = max(0, l - pad), max(0, t - pad)
    r, b = min(rgba.width, r + pad), min(rgba.height, b + pad)
    return rgba.crop((l, t, r, b))
