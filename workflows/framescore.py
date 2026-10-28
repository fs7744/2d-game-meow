"""序列帧的帧级质量评分与多候选择优（wss 用）。

**为什么需要它**（业界实测结论）：
  - IP-Adapter + ControlNet 的组合，帧间一致性只有 **70~80%**
  - **同 seed 也不保证一致**——CLIP Vision 编码会引入方差
  - 逐帧生成时，角色的配色、比例、细节会逐帧漂移（实测 20~30%）
  - 业界通用的解法不是「调参」，而是**生成多个候选择优**：
    每帧生成 N 张，按分数挑最好的那张

**和 W4 的质量闸门有什么不同**：
  W4 评估的是「人形骨骼」——像素角色、怪物、机甲全都检不出关节，
  分数恒为 0。所以这里另起一套**与主体类型无关**的信号：

  silhouette  与参考帧轮廓的贴合度（IoU）。这一条最关键：
      动作必须对 —— 生成的剪影要跟参考帧的剪影重合。
  identity   与角色参考图的身份贴合度（色彩分布 + 主体占比）。
      换的是角色，不能换掉 —— 配色要落在角色色板附近。
  coverage   前景占比。太大 = 没出全身/糊成一坨；太小 = 主体缩成一小点。
  edges      贴边。被切断的部位在动画里会「闪」。
  drift      与上一帧的质心偏移。帧间跳太远播起来抖。

score 只用于**排序**，passed 才是闸门判据。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import matte as matte_mod

# 各信号的权重（用于排序，绝对值无意义）
W_SILHOUETTE = 40.0     # 与参考帧剪影的 IoU
W_IDENTITY = 25.0       # 与角色参考的身份贴合度
W_COVERAGE = 15.0       # 前景占比落在舒适区
W_EDGE = 12.0           # 每条被切断的边
W_DRIFT = 18.0          # 与上一帧的质心偏移

# 前景占比舒适区间
COVERAGE_LO, COVERAGE_HI = 0.05, 0.60


def _alpha(rgba: Image.Image) -> np.ndarray:
    return np.asarray(rgba.convert("RGBA").getchannel("A"))


def subject_mask(rgba: Image.Image) -> np.ndarray:
    """主体掩膜（bool）。优先 alpha；全不透明时退回「非背景色」。"""
    a = _alpha(rgba)
    if a.min() < 250:
        return a > 127
    arr = np.asarray(rgba.convert("RGB"), dtype=np.int16)
    bg = np.array([255, 255, 255], dtype=np.int16)
    return np.abs(arr - bg).max(axis=2) > 16


def coverage(rgba: Image.Image) -> float:
    """前景占比；没抠图时返回 1.0 表示「不可用」。"""
    a = _alpha(rgba)
    if a.min() == 255:
        return 1.0
    return float((a > 127).mean())


def centroid(rgba: Image.Image) -> Optional[Tuple[float, float]]:
    m = subject_mask(rgba)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def drift(prev: Image.Image, cur: Image.Image) -> float:
    """两帧质心距离 / (对角线/2)，截到 0~1。"""
    p, c = centroid(prev), centroid(cur)
    if p is None or c is None:
        return 1.0
    d = float(np.hypot(p[0] - c[0], p[1] - c[1]))
    return min(1.0, d / (float(np.hypot(cur.width, cur.height)) / 2.0))


def silhouette_iou(a: Image.Image, b: Image.Image) -> float:
    """两图主体剪影的 IoU（按各自尺寸归一化后比较）。

    这是「动作对不对」的判据：轮廓条件让生成图贴合参考帧的剪影，
    IoU 高说明动作真的跟上了。
    """
    ma = subject_mask(a.convert("RGBA"))
    mb = subject_mask(b.convert("RGBA"))
    if not ma.any() or not mb.any():
        return 0.0
    # 缩放到同一尺寸再比
    size = (256, 256)
    from PIL import Image as _I
    ma_s = np.asarray(_I.fromarray(ma.astype(np.uint8) * 255).resize(size, _I.NEAREST)) > 127
    mb_s = np.asarray(_I.fromarray(mb.astype(np.uint8) * 255).resize(size, _I.NEAREST)) > 127
    inter = np.logical_and(ma_s, mb_s).sum()
    union = np.logical_or(ma_s, mb_s).sum()
    return float(inter / union) if union else 0.0


def _palette_profile(rgba: Image.Image, bins: int = 12) -> np.ndarray:
    """主体的色彩分布特征（归一化直方图）。

    用来判断「生成结果的配色是否接近角色参考图」。
    不直接比 RGB（角色在动，明暗本来就会变），比的是**色相/明度的
    分布形状**——同一个角色换动作时，色彩分布应当是相近的。
    """
    m = subject_mask(rgba)
    if not m.any():
        return np.zeros(bins * 3, dtype=np.float32)
    arr = np.asarray(rgba.convert("RGB"), dtype=np.float32)
    px = arr[m]
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    # 饱和度与明度
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    val = mx / 255.0
    # 简易色相（0~1）
    hue = np.zeros_like(r)
    d = mx - mn
    nz = d > 0
    idx = nz & (mx == r)
    hue[idx] = ((g[idx] - b[idx]) / d[idx]) % 6
    idx = nz & (mx == g)
    hue[idx] = ((b[idx] - r[idx]) / d[idx]) + 2
    idx = nz & (mx == b)
    hue[idx] = ((r[idx] - g[idx]) / d[idx]) + 4
    hue = (hue / 6.0) % 1.0

    prof = np.concatenate([
        np.histogram(hue, bins=bins, range=(0, 1))[0],
        np.histogram(sat, bins=bins, range=(0, 1))[0],
        np.histogram(val, bins=bins, range=(0, 1))[0],
    ]).astype(np.float32)
    total = prof.sum()
    return prof / total if total > 0 else prof


def identity_similarity(gen: Image.Image, ref: Image.Image) -> float:
    """生成图与角色参考图的色彩分布相似度（0~1）。

    用余弦相似度衡量「色相/饱和度/明度的分布形状像不像」。
    这不是人脸识别，但对「换角色保动作」这个任务是合适的判据：
    同一个角色在不同帧里的色彩分布应当稳定。
    """
    a = _palette_profile(gen)
    b = _palette_profile(ref)
    if a.sum() <= 0 or b.sum() <= 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _coverage_fit(cov: float) -> float:
    if cov >= 1.0:
        return 1.0
    if cov < COVERAGE_LO:
        return cov / COVERAGE_LO
    if cov > COVERAGE_HI:
        return max(0.0, (1.0 - cov) / (1.0 - COVERAGE_HI))
    return 1.0


def evaluate(gen: Image.Image, ref_frame: Image.Image, char_ref: Image.Image,
             prev: Optional[Image.Image] = None,
             area_ref: Optional[float] = None) -> Dict[str, Any]:
    """给一帧打分。

    gen       生成结果（RGBA）
    ref_frame 这一帧对应的参考帧（提供动作基准：剪影）
    char_ref  角色参考图（提供身份基准：配色）
    prev      上一帧（已接受的），用于算漂移
    area_ref  参考的主体面积（像素数）。给了就多一道「面积塌陷」检查：
              残缺帧的剪影 IoU 可能擦边达标，但主体面积会明显偏小
              （实测残缺帧 128px vs 正常 675px，IoU 0.384 刚好擦过 0.40 的闸门）
    """
    rgba = gen.convert("RGBA")
    cov = coverage(rgba)
    edges = matte_mod.edge_touch(rgba) if cov < 1.0 else []
    sil = silhouette_iou(rgba, ref_frame.convert("RGBA"))
    ident = identity_similarity(rgba, char_ref.convert("RGBA"))
    dr = drift(prev, rgba) if prev is not None else 0.0
    area = int(subject_mask(rgba).sum())
    # 面积塌陷率：相对参考帧的主体面积。残缺帧会显著小于 1。
    area_ratio = (area / area_ref) if (area_ref and area_ref > 0) else 1.0
    score = (W_SILHOUETTE * sil + W_IDENTITY * ident
             + W_COVERAGE * _coverage_fit(cov)
             - W_EDGE * len(edges) - W_DRIFT * dr)
    # 闸门：动作要跟上（剪影至少 40% 重合）、没被切断、且主体没塌掉
    #
    # area_ratio 阈值 0.65 的由来：实测残缺帧的 area_ratio 是 0.57，
    # 正常帧在 0.87~1.06。取 0.65 能把残缺帧挑出来补提，又不会误伤
    # 「体型本来就比参考小」的正常帧。
    passed = sil >= 0.40 and not edges and area_ratio >= 0.65
    return {
        "silhouette": round(sil, 4),
        "identity": round(ident, 4),
        "coverage": round(cov, 4),
        "area": area,
        "area_ratio": round(area_ratio, 4),
        "edge": edges,
        "drift": round(dr, 4),
        "score": round(score, 3),
        "passed": passed,
    }


def pick_best(candidates: Sequence[Tuple[Image.Image, Dict[str, Any]]]
              ) -> Tuple[Image.Image, Dict[str, Any], int]:
    """从候选里挑分数最高的一张。

    返回 (图, 评分, 候选下标)。同分时取第一个（候选按 seed 递增生成，
    第一个就是「标准 seed」的结果，保证可复现）。
    """
    if not candidates:
        raise ValueError("没有候选")
    best_i = 0
    best_s = candidates[0][1]["score"]
    for i, (_img, ev) in enumerate(candidates):
        if ev["score"] > best_s + 1e-6:
            best_s = ev["score"]
            best_i = i
    img, ev = candidates[best_i]
    return img, ev, best_i


def sequence_consistency(frames: Sequence[Image.Image]) -> Dict[str, Any]:
    """整段序列的一致性指标（写进 meta，供产线判断要不要重跑）。

    identity_drift  相邻帧的身份相似度下降幅度（越小越稳）
    silhouette_drift 相邻帧剪影的 IoU（动作连贯度）
    """
    if len(frames) < 2:
        return {"identity_drift": 0.0, "silhouette_drift": 1.0, "worst_pair": -1}
    idents: List[float] = []
    sils: List[float] = []
    for i in range(len(frames) - 1):
        idents.append(identity_similarity(frames[i], frames[i + 1]))
        sils.append(silhouette_iou(frames[i], frames[i + 1]))
    worst = int(np.argmin(idents))
    return {
        "identity_drift": round(float(1.0 - np.mean(idents)), 4),
        "silhouette_drift": round(float(1.0 - np.mean(sils)), 4),
        "worst_pair": worst,
    }
