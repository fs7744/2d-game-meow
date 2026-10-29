"""参考图 → ControlNet 条件图。三种模式，覆盖不同的动作参考素材。

背景：原来的「角色迁移」只认**真人照片**——走人体检测（FasterRCNN）→
RTMPose 骨架 → ControlNet OpenPose。这条路对游戏素材彻底失效：
实测像素画上人体检测器一个框都检不出（放大、模糊、加描边等预处理全都试过，
仍然是 0 框），因为 FasterRCNN 是在 COCO 自然照片上训练的。

所以这里给出三条不依赖人体检测的路：

  **pose（骨架）**
      真人照片 → 人体检测 → RTMPose → OpenPose-18 骨架图。
      姿态语义最强：只保留关节位置，参考角色的衣服/体型全丢掉，
      所以生成的图是「角色穿着自己的衣服摆参考的姿势」。
      只对真人有效。

  **outline（轮廓）**
      任意素材 → 抠出主体 → 剪影边缘 → ControlNet canny。
      保留的是**轮廓形状**（含手持道具），所以能把像素角色、怪物、机甲
      甚至非人形生物当动作参考。代价是轮廓里的道具也会被抄走
      （实测参考图里的剑会出现在结果里）。

  **auto（自动）**
      先试 pose：检测得到人体就用骨架；检测不到就退回 outline。
      这样真人素材走更精确的骨架路，非真人素材自动降级，不至于直接报错。

  **skeleton（已画好的骨架图）**
      输入就是 OpenPose 格式的骨架图（骨骼编辑器节点导出的），原样 resize 使用。
      用来做「手动摆关节 → 出姿势」的可控编辑，不需要人体检测。

**为什么轮廓要在原分辨率上取边再放大**：反过来（先放大再取边）会把像素画的
锯齿边抹平，条件图就丢掉了像素感，生成结果会变成普通插画。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# canny 阈值：剪影是二值图，用低阈值抓全轮廓，高阈值只留强边
OUTLINE_CANNY_LO = 100
OUTLINE_CANNY_HI = 200

MODES = ("pose", "outline", "auto", "skeleton")


def _alpha_mask(img: Image.Image) -> Optional[np.ndarray]:
    """有 alpha 且不全不透明时，返回 alpha 二值掩膜。"""
    rgba = np.asarray(img.convert("RGBA"))
    a = rgba[..., 3]
    if a.min() >= 250:
        return None
    return (a > 127).astype(np.uint8) * 255


def _subject_mask(img: Image.Image, matte_dir=None) -> np.ndarray:
    """主体掩膜（uint8, 0/255）。

    优先用 alpha（像素游戏素材几乎都是透明底，最准也最快）；
    没有 alpha 就退回「四角背景色差分」，最后才用 u2net 语义分割
    （要是有的话）——语义分割对像素画的小怪物经常认不出主体。
    """
    m = _alpha_mask(img)
    if m is not None:
        return m
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    corner = np.concatenate([
        rgb[:4, :4].reshape(-1, 3), rgb[:4, -4:].reshape(-1, 3),
        rgb[-4:, :4].reshape(-1, 3), rgb[-4:, -4:].reshape(-1, 3),
    ])
    colors, counts = np.unique(corner, axis=0, return_counts=True)
    bg = colors[int(counts.argmax())]
    diff = np.abs(rgb - bg).max(axis=2)
    mask = (diff > 12).astype(np.uint8) * 255
    if mask.mean() > 0.98:
        # 整张图都被判成前景（背景色估计失败）→ 退回语义分割
        try:
            from . import matte as matte_mod

            if matte_dir is not None and matte_mod.ai_available(matte_dir):
                cut = matte_mod.matte_ai(img.convert("RGB"), matte_dir)
                return np.asarray(cut.getchannel("A"))
        except Exception:                   # noqa: BLE001
            pass
    return mask


def outline_condition(img: Image.Image, size: int = 768,
                      matte_dir=None,
                      canny_lo: int = OUTLINE_CANNY_LO,
                      canny_hi: int = OUTLINE_CANNY_HI) -> Tuple[Image.Image, Dict[str, Any]]:
    """剪影轮廓 → ControlNet canny 条件图。

    **不是单纯的边缘图**：实测「只给边缘」会让模型把整个画面填成纹理——
    成品里主体只剩 3 个像素（参考帧是 677 个），完全不可用。原因是
    canny 条件图天然是「白线画在黑底」，模型看到满屏黑底加一个闭合轮廓，
    最省力的解释就是「整屏都是画布」。

    所以这里给的是**自制软边缘**：保留参考图自身的灰度结构（让模型知道
    「这个闭合形状内部是有内容的实体」），再把边缘加重成硬边。
    实测对比（同一张像素参考图，最终成品的主体像素数 / 参考值）：

        只给边缘          3 / 677     ← 不可用
        白底 + 中灰主体  628 / 677
        白底 + 灰主体 + 内部结构边  590 / 677
        **自制软边缘**   **709 / 677** ← 轮廓 IoU 0.899

    放大用最近邻：像素画的边缘是硬的，LANCZOS 会把它糊成灰带。
    """
    mask = _subject_mask(img, matte_dir)
    if mask.max() == 0:
        raise ValueError("参考图里找不到主体：图是全透明的，或者主体和背景同色")
    # 闭运算：像素画的描边常有 1px 缺口，不补的话轮廓会断成几段
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(mask, canny_lo, canny_hi)

    # 参考图的灰度结构（透明底合成到白底上）
    rgba = img.convert("RGBA")
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, (0, 0), rgba)
    gray = np.asarray(flat.convert("L"), dtype=np.float32)
    # 主体内部压暗、背景提亮：强化「哪里是主体」
    inside = mask > 0
    soft = np.where(inside, gray * 0.55, 255.0 - (255.0 - gray) * 0.25)
    cond_arr = np.minimum(soft, 255.0 - edges).astype(np.uint8)

    cond = Image.fromarray(cond_arr).convert("RGB").resize((size, size), Image.NEAREST)
    info = {
        "edge_pixels": int((edges > 0).sum()),
        "subject_ratio": round(float(inside.mean()), 4),
    }
    return cond, info


def pose_condition(img: Image.Image, extractor, thr: float = 0.25,
                   fill: float = 0.8, size: int = 768) -> Tuple[Image.Image, Dict[str, Any]]:
    """真人照片 → OpenPose-18 骨架条件图。

    骨架的归一化规则见 pose.normalized_points：人物按检测框居中、按 fill
    占满画面高度——ControlNet OpenPose 是空间敏感的，参考图里缩在角落的
    人会变成结果里的「角落小人」。
    """
    if extractor is None:
        raise ValueError("姿态模型不可用")
    kps = extractor.normalized_points(img.convert("RGB"), thr=thr, fill=fill)
    if len(kps) < 8:
        raise ValueError(
            f"动作参考图里只识别出 {len(kps)} 个关节（至少要 8 个）："
            "参考图里的人物可能太小、被遮挡或不是人体")
    import sys
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.make_pose import render

    cond = render(kps, size=(size, size))
    return cond, {"joints": len(kps)}


def build_condition(img: Image.Image, mode: str = "auto", size: int = 768,
                    extractor=None, thr: float = 0.25, fill: float = 0.8,
                    matte_dir=None) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """按模式构建条件图。返回 (条件图, 实际使用的模式, 信息)。

    `auto` 的降级链：pose → outline。
    降级发生时信息里带 `fallback_from`，让调用方能如实告诉用户
    「你的参考图不是真人，已经改用轮廓方式」。
    """
    if mode not in MODES:
        raise ValueError(f"未知参考模式 {mode}（可选 {', '.join(MODES)}）")

    if mode == "skeleton":
        # 输入**本身就是画好的 OpenPose 骨架图**（骨骼编辑器节点导出的）。
        # 绝不能再走 pose/auto：骨架图上没有真人，FasterRCNN 一个框都检不出，
        # auto 会静默降级成 outline（剪影边缘），把精心画的骨架毁成一张黑图。
        cond = img.convert("RGB").resize((size, size), Image.LANCZOS)
        return cond, "skeleton", {"mode": "skeleton", "joints": None}

    if mode in ("pose", "auto"):
        try:
            cond, info = pose_condition(img, extractor, thr=thr, fill=fill, size=size)
            info["mode"] = "pose"
            if mode == "auto":
                info["fallback_from"] = None
            return cond, "pose", info
        except Exception as exc:            # noqa: BLE001
            if mode == "pose":
                raise
            reason = f"{type(exc).__name__}: {exc}"

    cond, info = outline_condition(img, size=size, matte_dir=matte_dir)
    info["mode"] = "outline"
    if mode == "auto":
        info["fallback_from"] = "pose"
        info["fallback_reason"] = reason
    return cond, "outline", info
