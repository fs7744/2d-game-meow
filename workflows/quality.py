"""帧质量评分：给 W4 的「一帧出多个候选、自动挑最好的」和「关节不够就补提」提供判据。

四个信号，全部从产物图上直接量，不依赖生成过程：
  points   姿态模型在**生成图**上重新检出的人形关节点数（0~18）。最强的一个：
           手断了、腿糊了、画成两个半身，点数立刻掉。要求生成图本身能被检出，
           等于要求「画出来的确实是一个能被姿态模型认出来的人」。
  coverage 抠图后前景占比。偏大说明没出全身（半身/糊成一坨），偏小说明人物太小。
           没开抠图时 alpha 全 255，这个信号无意义，按 1.0 放过。
  edges    贴边（被画布切断）。精灵图要对齐脚底，脚被裁掉后对齐必然跳。
  drift    与上一帧的 alpha 质心偏移（按图像对角线归一化）。帧与帧跳太远，播起来抖。

score 只是加权和，用来**排序**，绝对值没有意义；passed 才是闸门判据。
没有姿态模型时（权重缺失）points 恒为 0、passed 恒为 False —— 调用方据此关闭闸门，
不要拿 False 当「质量差」去无限重提。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from . import matte as matte_mod

W_POINTS = 10.0     # 每检出一个关节点
W_COVERAGE = 5.0    # 占比落在合理区间
W_EDGE = 8.0        # 每有一条被切断的边
W_DRIFT = 20.0      # 质心偏移（0~1）

# 前景占比的舒适区间。低于 LO 人物太小（像素化后几乎没细节），
# 高于 HI 通常是半身/特写，精灵格子里放不下。
COVERAGE_LO, COVERAGE_HI = 0.06, 0.55


def _alpha(rgba: Image.Image) -> np.ndarray:
    return np.asarray(rgba.convert("RGBA").getchannel("A"))


def has_alpha(rgba: Image.Image) -> bool:
    """有没有真透明区域。全不透明说明没抠图，coverage / edges 都不可用。"""
    return bool(_alpha(rgba).min() < 255)


def coverage(rgba: Image.Image) -> float:
    """前景（alpha>127）占全图比例；没抠图时返回 1.0 表示「不可用」。"""
    a = _alpha(rgba)
    if a.min() == 255:
        return 1.0
    return float((a > 127).mean())


def centroid(rgba: Image.Image) -> Optional[Tuple[float, float]]:
    a = _alpha(rgba) > 127
    if not a.any():
        return None
    ys, xs = np.nonzero(a)
    return float(xs.mean()), float(ys.mean())


def drift(prev: Image.Image, cur: Image.Image) -> float:
    """两帧质心距离 / (对角线/2)，截到 0~1。任一帧没有前景时返回 1.0（最差）。"""
    p, c = centroid(prev), centroid(cur)
    if p is None or c is None:
        return 1.0
    d = float(np.hypot(p[0] - c[0], p[1] - c[1]))
    return min(1.0, d / (float(np.hypot(cur.width, cur.height)) / 2.0))


def _coverage_fit(cov: float) -> float:
    if cov >= 1.0:                  # 没抠图，不评价
        return 1.0
    if cov < COVERAGE_LO:
        return cov / COVERAGE_LO
    if cov > COVERAGE_HI:
        return max(0.0, (1.0 - cov) / (1.0 - COVERAGE_HI))
    return 1.0


def evaluate(img: Image.Image, extractor: Any = None, prev: Optional[Image.Image] = None,
             min_points: int = 8, thr: float = 0.25,
             use_edges: bool = True) -> Dict[str, Any]:
    """给一帧打分。extractor 为 None 时 points=0、passed=False（表示「无法判定」）。"""
    rgba = img.convert("RGBA")
    pts = len(extractor.visible_points(img, thr=thr)) if extractor is not None else 0
    cov = coverage(rgba)
    edges = matte_mod.edge_touch(rgba) if (use_edges and cov < 1.0) else []
    dr = drift(prev, rgba) if prev is not None else 0.0
    score = (W_POINTS * pts + W_COVERAGE * _coverage_fit(cov)
             - W_EDGE * len(edges) - W_DRIFT * dr)
    passed = extractor is not None and pts >= min_points and not edges
    return {
        "points": pts,
        "coverage": round(cov, 4),
        "edge": edges,
        "drift": round(dr, 4),
        "score": round(score, 3),
        "passed": passed,
    }
