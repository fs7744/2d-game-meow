"""精灵图网格检测：二维联合评分。

这个模块解决的是「游戏动画序列帧怎么切」的问题。用户手上的序列帧通常
已经是一张排好的精灵图（sprite sheet），需要按格子切开逐帧处理。

为什么不能用简单办法（都实测过，见 tools/probe_grid*.py）：
  - 「找完全空行空列」：帧贴边时根本没有空行，4×1 被判成 10×3
  - 「投影谷值 + 间距一致性」：单帧内容不重复时横向怎么切都不跨界，
    纵向 1×4 被误判成 2×1
  - 「单轴评分」：剑这类**独立配件**是单独连通域，切宽一点会把
    「身体」和「剑」分到不同格子，单轴分数反而变高（4×1 被判成 6×1）

最终判据是二维的：**把图切成 cols×rows 格后，每一格里应当恰好有一个角色**。
  1. 每个连通域不得跨格（错切一定会把一个身体切成两半）
  2. 每格非空
  3. 各格「最大连通域占该格前景的比例」要高（一个角色 + 少量配件）
  4. 各格最大连通域的**面积要一致**（同样大小的角色，帧间不会差很多）
     —— 这条专门杀掉「身体/剑被拆到不同格子」的过度切分

判据 4 是关键：切错时各格的主导物大小差异巨大（身体 vs 一把剑），
切对时各帧的角色大小基本一致。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

# 各格主导连通域面积的最大允许差异（max/min）。3.0 允许同一角色的不同
# 帧有较明显的体型变化（比如起跳时伸展），但挡得住「身体 vs 一把剑」那种
# 量级差异（实测 614 : 63 ≈ 9.7）。
DOMAINANT_SPREAD = 3.0

# 主导连通域至少要占该格前景的这么多，否则该格算「多个主体」
DOMINANT_MIN_RATIO = 0.35


def foreground_mask(img: Image.Image) -> np.ndarray:
    """前景掩膜（bool 数组）。

    优先用 alpha；不透明时用「与背景色不同」近似——游戏精灵图几乎都是
    透明底，这条分支主要服务于截图 / 白底素材。
    """
    rgba = np.asarray(img.convert("RGBA"))
    if rgba[..., 3].min() < 250:
        return rgba[..., 3] > 127
    rgb = rgba[..., :3].astype(np.int16)
    corner = np.concatenate([
        rgb[:4, :4].reshape(-1, 3), rgb[:4, -4:].reshape(-1, 3),
        rgb[-4:, :4].reshape(-1, 3), rgb[-4:, -4:].reshape(-1, 3),
    ])
    colors, counts = np.unique(corner, axis=0, return_counts=True)
    bg = colors[int(counts.argmax())]
    return np.abs(rgb - bg).max(axis=2) > 12


def _label_components(mask: np.ndarray):
    """全局连通域标记（8 邻域）。返回 (labels, areas, bboxes)。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA] if n > 0 else np.zeros(0, dtype=int)
    boxes = [(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
              int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
             for i in range(n)]
    return labels, areas, boxes


def _crosses(labels: np.ndarray, areas: np.ndarray, boxes: Sequence[tuple],
             cols: int, rows: int, significant_min: int) -> bool:
    """是否有**显著**连通域跨越了格子边界。

    只有足够大的连通域才算数：小的配件（剑穗、飘带）跨一点点格子边界
    不该否决整个切法。
    """
    if significant_min <= 0:
        return False
    h, w = labels.shape
    cw, ch = w / cols, h / rows
    for i in range(1, len(areas)):
        if areas[i] < significant_min:
            continue
        x, y, bw, bh = boxes[i]
        x2, y2 = x + bw - 1, y + bh - 1
        if int(x // cw) != int(x2 // cw) or int(y // ch) != int(y2 // ch):
            return True
    return False


def _grid_score(labels: np.ndarray, areas: np.ndarray,
                cols: int, rows: int) -> Optional[float]:
    """给一个 cols×rows 候选网格打分（越大越好）；不可行时返回 None。

    打分 = 各格「最大连通域占该格前景的比例」的平均值。
    切对时每格一个角色，比值接近 1；切粗了每格两个角色，比值掉到 0.5 上下。
    """
    h, w = labels.shape
    cw, ch = w / cols, h / rows
    ratios: List[float] = []
    dominants: List[int] = []
    for r in range(rows):
        y0, y1 = int(round(r * ch)), int(round((r + 1) * ch))
        y1 = max(y1, y0 + 1)
        for c in range(cols):
            x0, x1 = int(round(c * cw)), int(round((c + 1) * cw))
            x1 = max(x1, x0 + 1)
            cell = labels[y0:y1, x0:x1]
            ids = cell[cell > 0]
            total = int(ids.size)
            if total == 0:
                return None                 # 空格子 → 切多了
            counts = np.bincount(ids)
            biggest = int(counts[counts > 0].max())
            ratios.append(biggest / total)
            dominants.append(biggest)

    if not dominants:
        return None
    # 各格主导物大小必须一致——这一条挡住「身体/剑被拆到不同格子」
    lo, hi = min(dominants), max(dominants)
    if lo <= 0 or hi / lo > DOMAINANT_SPREAD:
        return None
    if float(np.mean(ratios)) < DOMINANT_MIN_RATIO:
        return None
    return float(np.mean(ratios))


def detect_grid(img: Image.Image, max_cells: int = 16) -> Optional[Tuple[int, int]]:
    """自动检测精灵图网格。返回 (cols, rows)；检测不可靠时返回 None。

    返回 None 时调用方应当让用户手动指定行列数——宁可问一句，也不要
    默默切错（切错的精灵图会静默丢掉一半帧，比报错更难查）。
    """
    mask = foreground_mask(img)
    if not mask.any():
        return None
    labels, areas, boxes = _label_components(mask)
    if len(areas) <= 1:
        return None
    h, w = mask.shape
    # 显著连通域的面积门槛：最大连通域的 8%，用来忽略小配件
    significant_min = int(areas[1:].max() * 0.08) if len(areas) > 1 else 0

    max_c = max(1, min(max_cells, w // 8))
    max_r = max(1, min(max_cells, h // 8))

    best: Optional[Tuple[float, int, int]] = None
    for cols in range(1, max_c + 1):
        for rows in range(1, max_r + 1):
            if cols == 1 and rows == 1:
                continue
            if _crosses(labels, areas, boxes, cols, rows, significant_min):
                continue
            s = _grid_score(labels, areas, cols, rows)
            if s is None:
                continue
            # 同分时取更大的格子数（更细的切分优先——切粗了会漏帧）
            if best is None or s > best[0] + 1e-6 or (
                    abs(s - best[0]) <= 1e-6 and cols * rows > best[1] * best[2]):
                best = (s, cols, rows)

    if best is None:
        return None
    _, cols, rows = best
    if cols == 1 and rows == 1:
        return None
    return cols, rows


def split_sheet(img: Image.Image, cols: int, rows: int,
                trim: bool = False) -> List[Tuple[int, int, Image.Image]]:
    """按网格切分精灵图，返回 [(col, row, RGBA 图), ...]，跳过空格。

    `trim=True` 时把每帧裁到内容包围盒——像素游戏里帧内常有大片留白，
    不裁的话每帧尺寸不一，打包出来的图集单元格会被撑大。
    """
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    sheet = img.convert("RGBA")
    cw = sheet.width // cols
    ch = sheet.height // rows
    if cw < 1 or ch < 1:
        raise ValueError(f"网格 {cols}×{rows} 超出图片尺寸 {sheet.size}")

    out: List[Tuple[int, int, Image.Image]] = []
    for r in range(rows):
        for c in range(cols):
            cell = sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
            bbox = cell.getchannel("A").getbbox()
            if bbox is None:
                continue                     # 空格子
            if trim:
                cell = cell.crop(bbox)
            out.append((c, r, cell))
    return out
