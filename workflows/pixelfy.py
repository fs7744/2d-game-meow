"""完美像素转换：把任意图转成**像素完美**的像素画（纯确定性，不碰模型）。

这是 pixelforge「像素精修」节点的移植版，思路照搬它已验证的结论：

流程（全部确定性）：
    网格检测（FFT + Sobel 吸附） → 每格中位色采样 → 共享调色板量化
    → alpha 二值化 → 孤立噪点清理 → 输出**原生像素尺寸**图

**为什么不能简单降采样**：AI 出的「像素风」图网格尺寸未知、扭曲、非正方，直接
resize 会把色块边界切在中间，混色成脏斑。「每格一色、边界对齐」才是像素画与
「低分辨率图」的区别。

**为什么采样用中位色而不是多数表决**（pixelforge 实测，8px 格 + 抗锯齿 + 噪声）：
    median  重建误差 2.4、色数 550
    majority(k-means k=2) 误差 5.0、色数 1378
    center  误差 11.9（最差）
median 对格内噪声更稳，且在网格偏移 2px 时仍保持最低误差。多数表决另有代价：
每个格跑 k-means，64×64 图要跑几千次，慢一个数量级。

**输出尺寸的取舍**：只存原生像素尺寸（如 64×64），不预放大 8 倍。前端用
`image-rendering: pixelated` 最近邻显示即可——放大交给渲染端。存放大版等于把
「8 倍」这个显示参数焊死在素材里，之后想要 4 倍或 16 倍还得重算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import perfect_pixel as pp

RGB = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def _to_array(img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """RGBA 图 → (RGB array, alpha array)。"""
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    return arr[..., :3], arr[..., 3]


def _from_array(rgb: np.ndarray, alpha: Optional[np.ndarray]) -> Image.Image:
    if alpha is None:
        return Image.fromarray(rgb.astype(np.uint8), "RGB")
    out = np.dstack([rgb.astype(np.uint8), alpha.astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


def _box_downsample(rgb: np.ndarray, alpha: Optional[np.ndarray],
                    target_w: int, target_h: int):
    """无网格对齐时的退路：区域平均降采样（避免最近邻随机取点的脏斑）。

    RGB 用 BOX（区域平均后再量化），alpha 用最近邻（保持硬边）。
    """
    h, w = rgb.shape[:2]
    target_w = max(1, min(int(target_w), w))
    target_h = max(1, min(int(target_h), h))
    rgb_img = Image.fromarray(rgb, "RGB").resize((target_w, target_h), Image.BOX)
    arr = np.asarray(rgb_img)
    a_small = None
    if alpha is not None:
        a_img = Image.fromarray(alpha, "L").resize((target_w, target_h), Image.NEAREST)
        a_small = np.asarray(a_img)
    return arr, a_small


def estimate_bg_color(rgb_img: Image.Image) -> Optional[np.ndarray]:
    """四角采样估计纯色背景；四角不一致（场景/透明底）返回 None。

    像素主体多数是居中构图，四角是背景。只有四块区域的中位色足够接近时
    才认定「这是纯色背景」，避免把场景图/渐变底的像素也当成背景剔除。
    """
    arr = np.asarray(rgb_img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    s = max(1, min(h, w) // 16)
    patches = [arr[:s, :s], arr[:s, w - s:], arr[h - s:, :s], arr[h - s:, w - s:]]
    meds = np.stack([np.median(p.reshape(-1, 3), axis=0) for p in patches])
    spread = float(np.abs(meds - meds.mean(axis=0)).max())
    if spread <= 18.0:
        return meds.mean(axis=0)
    return None


def apply_palette(rgb: Image.Image, palette: Sequence[RGB]) -> Image.Image:
    """把每个像素映射到最近色板色（感知加权距离 R2G4B3）。

    注意 dtype：差值平方最大 255²×4 ≈ 26 万，int16（上限 32767）会溢出成
    负数，导致「最近色」算错——实测白色被映射成红色就是这条。必须 int32。
    """
    if not palette:
        return rgb.convert("RGB")
    arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    pal = np.asarray(palette, dtype=np.int32)          # (K, 3)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3).astype(np.int32)
    out = np.empty_like(flat)
    block = 65536
    for s in range(0, len(flat), block):
        chunk = flat[s:s + block]
        d = ((chunk[:, None, 0] - pal[None, :, 0]) ** 2 * 2
             + (chunk[:, None, 1] - pal[None, :, 1]) ** 2 * 4
             + (chunk[:, None, 2] - pal[None, :, 2]) ** 2 * 3)
        idx = np.argmin(d, axis=1)
        out[s:s + block] = pal[idx]
    return Image.fromarray(out.astype(np.uint8).reshape(h, w, 3), "RGB")


def count_out_of_palette(rgb: Image.Image, palette: Sequence[RGB]) -> int:
    """统计色板外颜色数（验收指标，量化后必须为 0）。"""
    if not palette:
        return 0
    colors = rgb.convert("RGB").getcolors(maxcolors=1 << 24) or []
    pal_set = {(int(r), int(g), int(b)) for r, g, b in palette}
    return sum(1 for _cnt, c in colors if c not in pal_set)


def binarize_alpha(img: Image.Image) -> Image.Image:
    """alpha 二值化：像素画只有「有/无」两态，半透明边是插画不是像素画。"""
    rgba = img.convert("RGBA")
    rgba.putalpha(rgba.getchannel("A").point(lambda a: 255 if a >= 128 else 0))
    return rgba


def alpha_binary_ratio(img: Image.Image) -> float:
    """alpha 是二值的比例（1.0 = 干净）。"""
    if img.mode != "RGBA":
        return 1.0
    a = np.asarray(img.getchannel("A"))
    if a.size == 0:
        return 1.0
    mid = ((a > 0) & (a < 255)).sum()
    return round(1.0 - mid / a.size, 4)


# --------------------------------------------------------------------------- #
# 调色板提取
# --------------------------------------------------------------------------- #

def _collect_pixels(images: Sequence[Image.Image], max_side: int = 384,
                    max_total: int = 300_000) -> np.ndarray:
    """把若干张图缩到小尺寸后拼成一维 RGB 像素数组（可复现，不随机采样）。"""
    chunks = []
    for im in images:
        a = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w = a.shape[:2]
        scale = max(1.0, max(h, w) / max_side)
        if scale > 1:
            a = np.asarray(Image.fromarray(a).resize(
                (max(1, int(w / scale)), max(1, int(h / scale))), Image.BOX))
        chunks.append(a.reshape(-1, 3))
    if not chunks:
        return np.zeros((0, 3), dtype=np.uint8)
    px = np.concatenate(chunks, axis=0)
    if len(px) > max_total:
        step = len(px) // max_total + 1
        px = px[::step]
    return px


def extract_palette(images: Sequence[Image.Image], colors: int = 32) -> List[RGB]:
    """从一批图提取共享色板（频次加权的最远点采样，确定性、无随机）。

    为什么不用 PIL 的 MEDIANCUT：它对**每张图独立**量化，多张图合起来色板会
    打架（第一张的主色占满色位，后面图的主体只剩几档）。这里的做法是
    「先汇总所有像素 → 频次加权最远点采样」，一张图的主体色不会把另一张挤掉。
    """
    n = max(2, min(int(colors), 256))
    px = _collect_pixels(images)
    if len(px) == 0:
        return [(0, 0, 0), (255, 255, 255)]

    # 1) 粗量化合并噪声（AI 出的「纯色」常带 ±3 抖动，不合并会占满色位）。
    #    桶宽随目标色数走：48 色这类高色数如果还用 8 级桶（32 档），半透明主体
    #    的中间色阶会被并掉，通透感直接断层；4 级桶（64 档）能保住细腻色阶。
    step = 4 if n >= 32 else 8
    q = ((px.astype(np.int32) // step) * step + step // 2).clip(0, 255).astype(np.uint8)
    uniq, counts = np.unique(q, axis=0, return_counts=True)
    uniq = uniq.astype(np.int32)
    counts = counts.astype(np.float32)

    # 2) 先取最高频色，再逐次挑「远且不算少」的颜色
    order = np.argsort(-counts)
    chosen: List[int] = [int(order[0])]
    min_dist = _weighted_dist(uniq, uniq[chosen[0]])
    freq_w = np.sqrt(counts / counts.max())

    while len(chosen) < n:
        score = min_dist * freq_w
        score[chosen] = -1.0
        i = int(np.argmax(score))
        if score[i] <= 1.0:
            break
        chosen.append(i)
        min_dist = np.minimum(min_dist, _weighted_dist(uniq, uniq[i]))

    pal = [tuple(int(c) for c in uniq[i]) for i in chosen]
    return pal or [(0, 0, 0), (255, 255, 255)]


def _weighted_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R2G4B3 加权的欧氏距离（绿色权重最高，符合人眼敏感度）。"""
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    return np.sqrt(d[:, 0] * 2 + d[:, 1] * 4 + d[:, 2] * 3)


def _snap_dim(value: int, base: int = 16, tol: float = 1.5) -> int:
    """把检测到的网格边长吸附到「base 的整数倍」，并夹在 8~256。

    为什么：AI 出的「像素风」没有严格网格（实测检测出 157×153），
    而精灵素材的尺寸几乎都是 16/32/64/128 的倍数——吸附后既能对齐到
    整齐的像素格，输出尺寸也规整。

    **只在「够接近」时才吸附**（容差 tol，默认 1.5 倍）：检测值 8 在
    base=16 下不该被抬成 16（那等于把像素密度砍半，细节全丢）；
    而 29 → 32、157 → 160 这种只差几个像素的就该吸附。
    超过容差说明检测值本身就是另一档规格，保留原值更忠实。
    """
    raw = int(value)
    base = max(1, int(base))
    # 已经是 base 的整数倍就原样保留
    if raw % base == 0:
        return max(8, min(256, raw))
    snapped = int(round(raw / base)) * base
    if snapped <= 0:
        # 夹到最近的合法下限（比 base 还小的检测值）
        return max(8, min(256, raw))
    # 吸附后与原值差太多（密度变化过大）就放弃吸附
    if max(snapped, raw) > min(snapped, raw) * tol:
        return max(8, min(256, raw))
    return max(8, min(256, snapped))


# --------------------------------------------------------------------------- #
# 噪点清理
# --------------------------------------------------------------------------- #

def clean_specks(rgb: np.ndarray, max_diff: int = 40) -> np.ndarray:
    """像素画收尾清理：去掉孤立噪点、填 1 像素孔洞。

    只在「8 邻域没有任何同色像素」且「与邻域多数色的差异不超过 max_diff」
    时替换。后一个条件很关键：单像素眼睛/高光与周围差异大，必须保留；
    量化残留的近似色噪点差异小，才清掉。1 像素描边通常有 2 个以上同色
    邻居，不会被误删。
    """
    h, w = rgb.shape[:2]
    if h < 3 or w < 3:
        return rgb
    padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    nbrs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nbrs.append(padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
    stack = np.stack(nbrs, axis=0)                  # (8, h, w, 3)
    same = (stack == rgb[None]).all(axis=-1)        # (8, h, w)
    isolated = same.sum(axis=0) == 0

    best_color = rgb.copy()
    best_count = np.zeros((h, w), dtype=np.int32)
    for i in range(8):
        cnt = (stack == stack[i][None]).all(axis=-1).sum(axis=0)
        better = cnt > best_count
        best_count = np.where(better, cnt, best_count)
        best_color = np.where(better[..., None], stack[i], best_color)

    diff = np.abs(rgb.astype(np.int32) - best_color.astype(np.int32)).max(axis=-1)
    fill = isolated & (best_count >= 6) & (diff <= max_diff)
    return np.where(fill[..., None], best_color, rgb)


def recolor_by_luma(img: Image.Image, palette: Sequence[RGB],
                    keep_alpha: bool = True) -> Image.Image:
    """把图按**明度**重映射到给定色板（换皮）。

    像素游戏里的「换皮」：同一套动作、同一套形状，只把角色换成另一个配色。
    这条路**不跑模型**——纯确定性、毫秒级、帧间零漂移，是序列帧换角色
    最稳的一条。

    **为什么按明度映射而不是按最近色**：最近色会把不同材质（金属 vs 布料）
    混成同一色，明度映射能保住光照结构（高光/阴影/描边），这正是换皮要
    保留的信息。

    参数:
        palette     目标色板（RGB 元组序列）
        keep_alpha  保留原图的 alpha（True=只换色不换形）
    """
    if not palette:
        return img.convert("RGBA" if keep_alpha else "RGB")
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    pal = np.asarray(palette, dtype=np.float32)
    # 明度排序，让「明度最接近的颜色」可以用 searchsorted 直接找
    luma_pal = pal @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    order = np.argsort(luma_pal)
    pal_sorted = pal[order]
    luma_sorted = luma_pal[order]

    rgb = arr[..., :3]
    alpha = arr[..., 3]
    fg = alpha > 127
    out = arr.copy()
    if fg.any():
        luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        # |luma - luma_sorted| 的最小值 → searchsorted 在有序数组上二分
        pos = np.searchsorted(luma_sorted, luma[fg])
        pos = np.clip(pos, 1, len(luma_sorted) - 1)
        left = np.abs(luma_sorted[pos - 1] - luma[fg])
        right = np.abs(luma_sorted[pos] - luma[fg])
        pick = np.where(left <= right, pos - 1, pos)
        out[..., :3][fg] = pal_sorted[pick]
    if not keep_alpha:
        out[..., 3] = 255
    return Image.fromarray(out.astype(np.uint8), "RGBA")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def refine_one(
    img: Image.Image,
    *,
    grid_snap: bool = True,
    grid_size: Optional[Sequence[int]] = None,
    target_size: Optional[Sequence[int]] = None,
    sample_method: str = "median",
    palette: Optional[Sequence[Sequence[int]]] = None,
    transparent_bg: bool = False,
    snap_to: Optional[int] = None,
) -> Dict[str, Any]:
    """单张精修。palette 为空时不做调色板量化（只做网格对齐）。

    返回 {image, meta}：image 是原生像素尺寸的 RGBA 图。
    """
    rgb, alpha = _to_array(img)
    h, w = rgb.shape[:2]
    meta: Dict[str, Any] = {"sourceSize": [w, h]}

    detected = False
    grid: Optional[Tuple[int, int]] = None
    if grid_snap:
        gs = None
        if grid_size and len(grid_size) == 2:
            gs = (int(grid_size[0]), int(grid_size[1]))
        elif snap_to:
            # **吸附必须在检测之前**：AI 出的「像素风」本来就没有严格网格
            # （实测检测出 157×153 这种），直接用会让输出尺寸很怪——
            # 精灵素材没法用 157×153。所以先检测、再把网格吸附到标准尺寸，
            # 然后用吸附后的网格重新走一遍采样（perfect_pixel 支持手动指定
            # grid_size，会跳过检测直接按给定网格吸附网格线）。
            gw0, gh0 = pp.detect_grid_scale(rgb)
            if gw0 is not None and gh0 is not None:
                gs = (_snap_dim(gw0, snap_to), _snap_dim(gh0, snap_to))
                meta["gridDetected"] = [int(gw0), int(gh0)]
                meta["gridSnapped"] = list(gs)
        gw, gh, sampled, xs, ys = pp.perfect_pixel(
            rgb, sample_method=sample_method, grid_size=gs, return_coords=True)
        if gw is None or gh is None:
            meta["gridFailed"] = True
        else:
            rgb = sampled
            detected = gs is None
            grid = (gw, gh)
            if "gridDetected" in meta:
                detected = True   # 检测到了再吸附的，仍算自动检测
            if alpha is not None:
                # alpha 用**同一批**网格坐标采样（中位数 + 二值化阈值）：
                # 单独再 refine 一次会因舍入差出 1 格，拼回 RGBA 时形状对不上
                a_cells = pp.sample_median(alpha[..., None], xs, ys)[..., 0]
                alpha = np.where(a_cells >= 128, 255, 0).astype(np.uint8)
                # 手动指定网格时 RGB 可能被硬对齐过，alpha 跟着同尺寸走
                if alpha.shape != rgb.shape[:2]:
                    ah, aw = alpha.shape
                    th, tw = rgb.shape[:2]
                    sy = (np.arange(th) * ah / th).astype(np.int32)
                    sx = (np.arange(tw) * aw / tw).astype(np.int32)
                    alpha = alpha[sy][:, sx]

    if grid is None:
        # 没检测到网格 / 关闭网格对齐：按目标尺寸或原尺寸的 1/8 降采样
        if target_size and len(target_size) == 2:
            tw, th = int(target_size[0]), int(target_size[1])
        else:
            tw, th = max(1, w // 8), max(1, h // 8)
        rgb, alpha = _box_downsample(rgb, alpha, tw, th)
        grid = (rgb.shape[1], rgb.shape[0])

    # 目标尺寸：检测结果与用户要求不一致时缩到目标
    if target_size and len(target_size) == 2:
        # 必须夹在 [1, 当前尺寸]：target_size 来自用户输入且无取值校验，
        # 传个 99999 就让 PIL 去分配约 30GB（实测 120s 无输出，线程挂死）。
        # 精修是**缩小**语义，不允许放大——放大只会让像素块发虚。
        tw = max(1, min(int(target_size[0]), w))
        th = max(1, min(int(target_size[1]), h))
        if (tw, th) != (rgb.shape[1], rgb.shape[0]):
            # 缩小用 BOX（块平均）而不是 NEAREST：128→64 的 NEAREST 只取
            # 每 2×2 块的左上角，等于丢掉四分之三的像素；块平均保住整体
            # 色量，随后调色板量化会把中间色收回去。alpha 仍用 NEAREST 保硬边。
            rgb_img = Image.fromarray(rgb, "RGB").resize((tw, th), Image.BOX)
            rgb = np.asarray(rgb_img)
            if alpha is not None:
                alpha = np.asarray(Image.fromarray(alpha, "L").resize(
                    (tw, th), Image.NEAREST))
            grid = (tw, th)

    # 调色板量化
    if palette:
        pal_list = [(int(c[0]), int(c[1]), int(c[2])) for c in palette]
        rgb_img = apply_palette(Image.fromarray(rgb, "RGB"), pal_list)
        rgb = np.asarray(rgb_img)

    # 收尾清理：量化后容易出现「孤立噪点」和「1 像素孔洞」，清一遍再输出
    rgb = clean_specks(rgb)

    # 背景透明：按四角估计纯色背景，把接近背景的像素置为透明。
    # 阈值 14 很保守，只清「几乎就是背景色」的像素，避免啃掉主体阴影。
    if transparent_bg:
        bg = estimate_bg_color(Image.fromarray(rgb, "RGB"))
        if bg is not None:
            diff = np.abs(rgb.astype(np.int32) - bg.astype(np.int32)).max(axis=2)
            mask = diff <= 14
            if alpha is None:
                alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            alpha = np.where(mask, 0, alpha).astype(np.uint8)

    out = _from_array(rgb, alpha)
    if out.mode == "RGBA":
        out = binarize_alpha(out)

    colors_used = len(out.convert("RGB").getcolors(maxcolors=1 << 24) or [])
    meta.update({
        "grid": [int(grid[0]), int(grid[1])],
        "detected": detected,
        "colors": colors_used,
        "offPalette": count_out_of_palette(out, palette) if palette else 0,
        "alphaBinary": alpha_binary_ratio(out) if out.mode == "RGBA" else 0.0,
        "size": [out.width, out.height],
    })
    return {"image": out, "meta": meta}


def refine_batch(
    images: Sequence[Image.Image],
    *,
    grid_snap: bool = True,
    grid_size: Optional[Sequence[int]] = None,
    target_size: Optional[Sequence[int]] = None,
    sample_method: str = "median",
    colors: int = 32,
    transparent_bg: bool = False,
    snap_to: Optional[int] = None,
) -> Dict[str, Any]:
    """一批图统一精修：共享色板 + 逐张网格对齐。

    共享色板在 batch 层解析：先各自网格对齐，再汇总所有结果提取一块色板，
    最后统一映射。逐张各自取色板会显著偏色。
    """
    # 第一遍：网格对齐（不带色板），拿到像素级中间结果
    interim = [
        refine_one(im, grid_snap=grid_snap, grid_size=grid_size,
                   target_size=target_size, sample_method=sample_method,
                   palette=None, transparent_bg=transparent_bg, snap_to=snap_to)
        for im in images
    ]
    pal = extract_palette([r["image"] for r in interim], colors) if colors else []

    out: List[Dict[str, Any]] = []
    for r in interim:
        src = r["image"]
        im = src
        if pal:
            rgb_img = apply_palette(src.convert("RGB"), pal)
            # 量化映射后可能又冒出新孤立色，再清一次
            rgb_img = Image.fromarray(clean_specks(np.asarray(rgb_img)), "RGB")
            if src.mode == "RGBA":
                im = rgb_img.convert("RGBA")
                im.putalpha(src.getchannel("A"))
                im = binarize_alpha(im)
            else:
                im = rgb_img
            r["meta"].update({
                "paletteSize": len(pal),
                "colors": len(im.convert("RGB").getcolors(maxcolors=1 << 24) or []),
                "offPalette": count_out_of_palette(im, pal),
            })
        out.append({"image": im, "meta": r["meta"]})
    return {"items": out, "palette": pal}
