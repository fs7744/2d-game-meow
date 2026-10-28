"""完美像素网格对齐：FFT 检测网格 → Sobel 精修网格线 → 每格单色采样。

移植自参考项目 perfectPixel（theamusing/perfectPixel, MIT）的纯 numpy 实现
（perfect_pixel_noCV2.py），去掉了 matplotlib 调试依赖与 print 噪音。

解决什么问题：AI 生成的「像素风」图片网格尺寸未知、扭曲、非正方，直接最近邻
降采样会错位混色。本模块先检测真实网格、把网格线吸附到最强梯度峰，再按格
重采样为单色，输出**像素完美**（pixel-perfect）的图——这是「像素画」与
「低分辨率图」的区别。

典型用法（生成侧强约束提示词 + 本模块收尾）：
    生成端："精确 32×32 像素网格，每格单色，无抗锯齿无渐变，限定调色板"
    收尾端：本模块自动检测真实网格并精修，输出 32×32 的干净像素图。

**为什么要自己实现而不直接用 cv2**：
  - 网格检测用的是 FFT + 一维投影，cv2 没有现成等价物；
  - 采样与吸附只需要 Sobel，而本仓库的 cv2 是可选 GPU 依赖，这个模块必须
    在没装 cv2 的机器上也能跑，所以手写卷积。
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


# ----------------------------
# 基础工具（无 cv2 依赖）
# ----------------------------


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8/float -> gray float32"""
    img = image_rgb.astype(np.float32)
    if img.ndim == 2:
        return img
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def normalize_minmax(x: np.ndarray, a=0.0, b=1.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32) + a
    y = (x - mn) / (mx - mn)
    return (a + (b - a) * y).astype(np.float32)


def conv2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution (same) for grayscale float32（位移窗口求和，向量化）。"""
    img = image.astype(np.float32, copy=False)
    k = kernel.astype(np.float32, copy=False)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for dy in range(kh):
        for dx in range(kw):
            w = k[dy, dx]
            if w == 0:
                continue
            out += w * pad[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out


def sobel_xy(gray: np.ndarray, ksize: int = 3):
    """Return (gx, gy) similar to cv2.Sobel for ksize 3 or 5."""
    if ksize == 3:
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)
    elif ksize == 5:
        kx = np.array([[-5, -4, 0, 4, 5],
                       [-8, -10, 0, 10, 8],
                       [-10, -20, 0, 20, 10],
                       [-8, -10, 0, 10, 8],
                       [-5, -4, 0, 4, 5]], dtype=np.float32)
        ky = kx.T
    else:
        raise ValueError("ksize must be 3 or 5")
    return conv2d_same(gray, kx), conv2d_same(gray, ky)


def magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


# ----------------------------
# FFT 网格尺寸检测
# ----------------------------


def compute_fft_magnitude(gray_image):
    f = np.fft.fft2(gray_image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = 1 - np.log1p(mag)
    return normalize_minmax(mag, 0.0, 1.0)


def smooth_1d(v, k=17):
    k = int(k)
    if k < 3:
        return v
    if k % 2 == 0:
        k += 1
    sigma = k / 6.0
    x = np.arange(k) - k // 2
    ker = np.exp(-(x * x) / (2 * sigma * sigma))
    ker = ker / (ker.sum() + 1e-8)
    return np.convolve(v, ker, mode="same")


def detect_peak(proj, peak_width=6, rel_thr=0.35, min_dist=6):
    """在频谱投影里找中心对称双峰，峰距/2 = 像素网格尺寸。"""
    center = len(proj) // 2
    mx = float(proj.max())
    if mx < 1e-6:
        return None
    thr = mx * float(rel_thr)

    candidates = []
    for i in range(1, len(proj) - 1):
        is_peak = True
        for j in range(1, peak_width):
            if i - j < 0 or i + j >= len(proj):
                continue
            if proj[i - j + 1] < proj[i - j] or proj[i + j - 1] < proj[i + j]:
                is_peak = False
                break
        if is_peak and proj[i] >= thr:
            left_climb = 0
            for k in range(i, 0, -1):
                if proj[k] > proj[k - 1]:
                    left_climb = abs(proj[i] - proj[k - 1])
                else:
                    break
            right_fall = 0
            for k in range(i, len(proj) - 1):
                if proj[k] > proj[k + 1]:
                    right_fall = abs(proj[i] - proj[k + 1])
                else:
                    break
            candidates.append({
                "index": i,
                "climb": left_climb,
                "fall": right_fall,
                "score": max(left_climb, right_fall),
            })

    if not candidates:
        return None

    left = [c for c in candidates if c["index"] < center - min_dist and c["index"] > center * 0.25]
    right = [c for c in candidates if c["index"] > center + min_dist and c["index"] < center * 1.75]

    left.sort(key=lambda x: x["score"], reverse=True)
    right.sort(key=lambda x: x["score"], reverse=True)

    if not left or not right:
        return None
    return abs(right[0]["index"] - left[0]["index"]) / 2


def estimate_grid_fft(gray, peak_width=6):
    """Return (grid_w, grid_h) or (None, None)。"""
    H, W = gray.shape

    mag = compute_fft_magnitude(gray)
    band_row = W // 2
    band_col = H // 2
    row_sum = np.sum(mag[:, W // 2 - band_row: W // 2 + band_row], axis=1)
    col_sum = np.sum(mag[H // 2 - band_col: H // 2 + band_col, :], axis=0)

    row_sum = normalize_minmax(row_sum, 0.0, 1.0).flatten()
    col_sum = normalize_minmax(col_sum, 0.0, 1.0).flatten()

    row_sum = smooth_1d(row_sum, k=17)
    col_sum = smooth_1d(col_sum, k=17)

    scale_row = detect_peak(row_sum, peak_width=peak_width)
    scale_col = detect_peak(col_sum, peak_width=peak_width)

    if scale_row is None or scale_col is None or scale_col <= 0:
        return None, None
    return scale_col, scale_row


def _plateau_peaks(sig: np.ndarray, rel_thr: float, min_interval: int) -> List[int]:
    """在梯度投影里找峰，**允许平台**（相邻等高的多个像素算一个峰）。

    为什么必须允许平台：真实像素画的边界是**锐边**（无抗锯齿），
    Sobel 在边界两侧会给出两个**等高**的相邻值（实测 2832, 2832），
    严格的 `s[i] > s[i-1] and s[i] > s[i+1]` 一个峰都找不到——
    于是网格检测对「本来就是像素画」的图反而失效（这是移植代码的
    一个真实局限，本项目实测时才发现）。

    **阈值不能直接用 max**：锐边会让两个相邻平台**叠加**，max 是单条边界的
    两倍（实测 max=5664 而单边界只有 2832），按 max 的 20% 算阈值
    （3352）会把所有边界全挡掉。所以这里改成**按平台自身的高度过滤**：
    先切出所有平台，用平台高度的中位数当参考（鲁棒，不受叠加影响）。

    平台取**中点**作为峰位置：中点正好落在边界上，比取左端更准。
    """
    n = len(sig)
    # 1) 切出所有等高平台 [start, end, height)
    plats: List[Tuple[int, int, float]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and sig[j] == sig[i]:
            j += 1
        plats.append((i, j, float(sig[i])))
        i = j
    if not plats:
        return []
    # 2) 只取「非零且是局部高点」的平台
    cand: List[Tuple[int, int, float]] = []
    for idx, (a, b, h) in enumerate(plats):
        if h <= 0:
            continue
        left = plats[idx - 1][2] if idx > 0 else -1.0
        right = plats[idx + 1][2] if idx + 1 < len(plats) else -1.0
        if h > left and h > right:
            cand.append((a, b, h))
    if not cand:
        return []
    # 3) 阈值用平台高度的中位数（不被叠加上限污染）
    heights = np.array([c[2] for c in cand], dtype=np.float32)
    thr = float(np.median(heights)) * float(rel_thr)

    peaks: List[int] = []
    for a, b, h in cand:
        if h < thr:
            continue
        mid = (a + b - 1) // 2
        if not peaks or mid - peaks[-1] >= min_interval:
            peaks.append(mid)
    return peaks


def estimate_grid_gradient(gray, rel_thr=0.2):
    """FFT 失败时的回退：Sobel 梯度投影峰 + 间隔中位数。"""
    H, W = gray.shape
    grad_x, grad_y = sobel_xy(gray, ksize=3)
    grad_x_sum = np.sum(np.abs(grad_x), axis=0).reshape(-1)
    grad_y_sum = np.sum(np.abs(grad_y), axis=1).reshape(-1)

    peak_x, peak_y = [], []
    min_interval = 4
    peak_x = _plateau_peaks(grad_x_sum, rel_thr, min_interval)
    peak_y = _plateau_peaks(grad_y_sum, rel_thr, min_interval)

    # 下限放宽到 3：格子数少的时候内部边界线不够 4 条（4 行 8 列的图只有
    # 3 条内部横线），原来的 `< 4` 会让这类图整个检测失败。
    # 间隔中位数需要至少 2 个间隔（即 3 条线）才稳。
    if len(peak_x) < 3 or len(peak_y) < 3:
        return None, None

    intervals_x = [peak_x[i] - peak_x[i - 1] for i in range(1, len(peak_x))]
    intervals_y = [peak_y[i] - peak_y[i - 1] for i in range(1, len(peak_y))]
    if not intervals_x or not intervals_y:
        return None, None

    scale_x = W / np.median(intervals_x)
    scale_y = H / np.median(intervals_y)
    return int(round(scale_x)), int(round(scale_y))


def detect_grid_scale(image, peak_width=6, max_ratio=1.5, min_size=4.0):
    """检测网格尺寸 (grid_w, grid_h)；失败返回 (None, None)。

    合理性校验（min_size=4、max_pixel=20、宽高比≤1.5）不通过时回退梯度法——
    低对比/纯色图会让 FFT 给出离谱结果，静默采用比报错更危险。
    """
    gray = rgb_to_gray(image)
    H, W = gray.shape

    grid_w, grid_h = estimate_grid_fft(gray, peak_width=peak_width)
    if grid_w is None or grid_h is None:
        grid_w, grid_h = estimate_grid_gradient(gray)
    else:
        pixel_size_x = W / grid_w
        pixel_size_y = H / grid_h
        if (min(pixel_size_x, pixel_size_y) < min_size
                or max(pixel_size_x, pixel_size_y) > 20.0
                or pixel_size_x / pixel_size_y > max_ratio
                or pixel_size_y / pixel_size_x > max_ratio):
            grid_w, grid_h = estimate_grid_gradient(gray)

    if grid_w is None or grid_h is None:
        return None, None

    pixel_size_x = W / grid_w
    pixel_size_y = H / grid_h
    if pixel_size_x / pixel_size_y > max_ratio or pixel_size_y / pixel_size_x > max_ratio:
        pixel_size = min(pixel_size_x, pixel_size_y)
    else:
        pixel_size = (pixel_size_x + pixel_size_y) / 2.0

    return int(round(W / pixel_size)), int(round(H / pixel_size))


# ----------------------------
# 网格线精修（Sobel 吸附）
# ----------------------------


def find_best_grid(origin, range_val_min, range_val_max, grad_mag, thr=0):
    """从估计网格线出发，在 ±range 内吸附到最强梯度峰。"""
    best = round(origin)
    peaks = []
    mx = np.max(grad_mag)
    if mx < 1e-6:
        return best
    rel_thr = mx * thr
    for i in range(-round(range_val_min), round(range_val_max) + 1):
        candidate = round(origin + i)
        if candidate <= 0 or candidate >= len(grad_mag) - 1:
            continue
        if (grad_mag[candidate] > grad_mag[candidate - 1]
                and grad_mag[candidate] > grad_mag[candidate + 1]
                and grad_mag[candidate] >= rel_thr):
            peaks.append((grad_mag[candidate], candidate))
    if not peaks:
        return best
    peaks.sort(key=lambda x: x[0], reverse=True)
    return peaks[0][1]


def refine_grids(image, grid_x, grid_y, refine_intensity=0.25):
    """从图像中心向两侧逐格步行，每条网格线在 ±cell×intensity 内吸附梯度峰。

    两个必须防的输入（都是用户可经 API 直达的网格尺寸）：
    - 网格 <= 0：cell_w = W / grid_x 除零，或负网格让 x 单向递减、
      循环条件永真 → **死循环 + x_coords 无限增长直到 OOM**；
    - 网格远大于边长：cell_w < 0.5 时 find_best_grid 找不到峰会
      round(origin) 把小数进度抹掉，x 每轮被打回同一个整数，永远推不动
      → 同样是死循环（实测 64×64 图 grid=2000 卡死 5600+ 迭代）。
    所以在入口校验，并给四个 while 都加「本轮没推进就 break」的保险。
    """
    H, W = image.shape[:2]
    # 网格必须在 [1, 边长] 内：再密就没有「每格一个像素」的意义，精修直接
    # 退化成按像素取样（调用方此时应走降采样路径）
    if not (1 <= grid_x <= W and 1 <= grid_y <= H):
        return [], []
    cell_w = W / grid_x
    cell_h = H / grid_y

    gray = rgb_to_gray(image)
    gx, gy = sobel_xy(gray, ksize=3)

    grad_x_sum = np.sum(np.abs(gx), axis=0).reshape(-1)
    grad_y_sum = np.sum(np.abs(gy), axis=1).reshape(-1)

    x_coords: List[int] = []
    y_coords: List[int] = []

    def _walk(start: float, step: float, limit: float, forward: bool,
              grad: Any, out: List[int]) -> None:
        """向一侧逐格步行。连续两轮落到同一个整数线即停——吸附失败也不许空转。

        判据必须比「取整后的线号」而不是浮点位置：find_best_grid 无峰时返回
        round(origin)，浮点位置每轮都在变（32.032 → 32 → 32.032…），比浮点
        永远判不出停滞；比线号才能识别「小数点后在原地打转」的死循环。
        """
        pos = float(start)
        last_line: Optional[int] = None
        stalled = 0
        while (pos < limit if forward else pos > limit):
            line = int(find_best_grid(pos, step * refine_intensity,
                                      step * refine_intensity, grad))
            if last_line is not None and line == last_line:
                stalled += 1
                if stalled >= 2:
                    break
            else:
                stalled = 0
            last_line = line
            out.append(line)
            pos = line + step if forward else line - step

    mid_x = find_best_grid(W / 2, cell_w, cell_w, grad_x_sum)
    mid_y = find_best_grid(H / 2, cell_h, cell_h, grad_y_sum)
    _walk(mid_x, cell_w, W + cell_w / 2, True, grad_x_sum, x_coords)
    _walk(mid_x - cell_w, cell_w, -cell_w / 2, False, grad_x_sum, x_coords)
    _walk(mid_y, cell_h, H + cell_h / 2, True, grad_y_sum, y_coords)
    _walk(mid_y - cell_h, cell_h, -cell_h / 2, False, grad_y_sum, y_coords)

    return sorted(x_coords), sorted(y_coords)


# ----------------------------
# 按格采样
# ----------------------------


def sample_center(image, x_coords, y_coords):
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)
    centers_x = ((x[1:] + x[:-1]) * 0.5).astype(np.int32)
    centers_y = ((y[1:] + y[:-1]) * 0.5).astype(np.int32)
    return image[centers_y[:, None], centers_x[None, :]]


def sample_majority(image, x_coords, y_coords, max_samples=128, iters=6, seed=0):
    """每格 k=2 的 k-means 多数表决取主色——最契合「每格一色」。"""
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]

    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)
    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)

    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H)
        y1 = np.clip(y1, 0, H)
        if y1 <= y0:
            y1 = min(y0 + 1, H)
        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W)
            x1 = np.clip(x1, 0, W)
            if x1 <= x0:
                x1 = min(x0 + 1, W)

            cell = img[y0:y1, x0:x1].reshape(-1, C)
            n = cell.shape[0]
            if n == 0:
                out[j, i] = 0
                continue
            if n > max_samples:
                cell = cell[rng.integers(0, n, size=max_samples)]

            c0 = cell[0]
            c1 = cell[np.argmax(((cell - c0) ** 2).sum(1))]
            m1 = np.zeros(n, dtype=bool)
            for _ in range(max(1, iters)):
                d0 = ((cell - c0) ** 2).sum(1)
                d1 = ((cell - c1) ** 2).sum(1)
                m1 = d1 < d0
                if np.any(~m1):
                    c0 = cell[~m1].mean(0)
                if np.any(m1):
                    c1 = cell[m1].mean(0)
            out[j, i] = c1 if m1.sum() >= (~m1).sum() else c0

    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


def sample_median(image, x_coords, y_coords):
    """每格取中位色——对格内噪声最稳，且网格偏移 2px 时仍是最低误差。"""
    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]

    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)
    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)

    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H)
        y1 = np.clip(y1, 0, H)
        if y1 <= y0:
            y1 = min(y0 + 1, H)
        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W)
            x1 = np.clip(x1, 0, W)
            if x1 <= x0:
                x1 = min(x0 + 1, W)
            cell = img[y0:y1, x0:x1].reshape(-1, C)
            if cell.shape[0] == 0:
                out[j, i] = 0
            else:
                out[j, i] = np.median(cell, axis=0)

    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


def perfect_pixel(
    image: np.ndarray,
    sample_method: str = "median",
    grid_size: Optional[Tuple[int, int]] = None,
    min_size: float = 4.0,
    peak_width: int = 6,
    refine_intensity: float = 0.25,
    return_coords: bool = False,
):
    """主入口：返回 (refined_w, refined_h, scaled_image[, x_coords, y_coords])。

    grid_size 手动指定时跳过检测（动画帧序列首帧检测、后续帧锁定即用此参数，
    避免逐帧检测抖动）。检测失败返回 (None, None, 原图) —— 调用方必须判空，
    不能把没处理过的图当精修结果用。

    return_coords=True 时额外返回网格线坐标：alpha 通道必须用与 RGB **同一批**
    坐标采样，否则两条通路各自 refine 出来的格数可能差 1，拼回 RGBA 时形状对不上。
    """
    H, W = image.shape[:2]
    if grid_size is not None:
        scale_col, scale_row = grid_size
    else:
        scale_col, scale_row = detect_grid_scale(
            image, peak_width=peak_width, max_ratio=1.5, min_size=min_size)
        if scale_col is None or scale_row is None:
            return (None, None, image) if not return_coords else (None, None, image, [], [])

    size_x = int(round(scale_col))
    size_y = int(round(scale_row))
    x_coords, y_coords = refine_grids(image, size_x, size_y, refine_intensity)

    refined_size_x = len(x_coords) - 1
    refined_size_y = len(y_coords) - 1
    if refined_size_x < 1 or refined_size_y < 1:
        return (None, None, image) if not return_coords else (None, None, image, [], [])

    if sample_method == "majority":
        scaled_image = sample_majority(image, x_coords, y_coords)
    elif sample_method == "median":
        scaled_image = sample_median(image, x_coords, y_coords)
    else:
        scaled_image = sample_center(image, x_coords, y_coords)

    refined_size_y, refined_size_x = scaled_image.shape[:2]

    # 手动指定网格是「契约」：坐标行走在边界处可能多/少一格，这里用最近邻
    # 索引对齐到用户要求的尺寸。自动检测路径不这么做（检测结果就是真值）。
    if grid_size is not None and (refined_size_x, refined_size_y) != (size_x, size_y):
        sy = (np.arange(size_y) * refined_size_y / size_y).astype(np.int32)
        sx = (np.arange(size_x) * refined_size_x / size_x).astype(np.int32)
        scaled_image = scaled_image[sy][:, sx]
        refined_size_y, refined_size_x = scaled_image.shape[:2]

    if return_coords:
        return refined_size_x, refined_size_y, scaled_image, x_coords, y_coords
    return refined_size_x, refined_size_y, scaled_image
