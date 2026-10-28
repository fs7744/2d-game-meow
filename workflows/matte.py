"""抠图 / 去白边：两种后端。

**color（默认）**：纯色背景 → 连通域色键 → 硬边 alpha。零权重、毫秒级、完全确定。
**u2net**：语义分割（isnet-general-use.onnx，170MB）。给彩色/复杂背景用。

什么时候必须上 u2net：实测武侠 LoRA 出的白袍角色，浅色衣襟与白底同色且贴边，
色键会把衣服甚至脸一起抠掉（"去白边"把贴边的白色当混色边剥走）。这类图只能
靠语义分割——它知道那是人，不知道那是不是白色。

原始说明：
纯色背景 → 硬边 alpha，并清掉混色边。

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

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
from PIL import Image


def estimate_bg_color(img: Image.Image, border: int = 2) -> tuple[int, int, int]:
    """背景色估计：**四角 patch 的众数**优先，退化时才用四条边的众数。

    为什么不用整条边：实测武侠类 LoRA 会在画面上边缘写字、左右边缘画黑条，
    四边取样会把「题字的黑」当成背景色，于是整张图抠反（背景留下、人物被抠掉）。
    角落是整幅图里最不容易被主体和题字占据的地方。
    """
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    p = max(4, min(h, w) // 12)                       # 角落 patch 边长

    corners = np.concatenate([
        arr[:p, :p].reshape(-1, 3), arr[:p, -p:].reshape(-1, 3),
        arr[-p:, :p].reshape(-1, 3), arr[-p:, -p:].reshape(-1, 3),
    ], axis=0)

    def mode(px: np.ndarray) -> tuple[int, int, int, int]:
        colors, counts = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
        i = int(counts.argmax())
        return (*[int(v) for v in colors[i]], int(counts[i]))

    cr, cg, cb, ccount = mode(corners)
    if ccount >= 0.35 * len(corners):                 # 四角足够一致才采信
        return (cr, cg, cb)

    strips = [arr[:border].reshape(-1, 3), arr[-border:].reshape(-1, 3),
              arr[:, :border].reshape(-1, 3), arr[:, -border:].reshape(-1, 3)]
    r, g, b, _ = mode(np.concatenate(strips, axis=0))
    return (r, g, b)


def strip_fringe(alpha: np.ndarray, rgb: np.ndarray, bg: Iterable[int],
                 tol: int = 32, ratio: float = 3.0, passes: int = 3) -> np.ndarray:
    """剥掉紧贴背景、且颜色明显被背景混过的那一圈（白边 / 蓝边）。

    单独抽出来是因为**两个后端都需要**：色键要，语义分割同样要。
    实测 u2net 只给 alpha、不做这步，结果角色轮廓外残留一圈背景色的 8px 块
    （像素化后就是一圈白点/蓝点），看起来像散落噪点 —— 那不是颜色问题，
    是这圈没剥干净。
    """
    bg_arr = np.array(list(bg), dtype=np.int16)
    dist = np.abs(rgb.astype(np.int16) - bg_arr).max(axis=2)
    k = np.ones((3, 3), np.uint8)
    for _ in range(max(1, passes)):
        cur_bg = alpha == 0
        touching = cv2.dilate(cur_bg.astype(np.uint8), k).astype(bool)
        blended = touching & (~cur_bg) & (dist <= tol * ratio)
        if not blended.any():
            break
        alpha[blended] = 0
    return alpha


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
        alpha = strip_fringe(alpha, rgb, bg_arr, tol, fringe_ratio, fringe_passes)

    out = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA")


def despeckle(rgba: Image.Image, min_ratio: float = 0.05,
              denoise_iters: int = 0) -> Image.Image:
    """清掉前景里的孤立小块（碎屑、印章、签名）。

    只保留面积不小于「最大连通块 × min_ratio」的前景块。实测武侠 LoRA 会在
    人物右侧画出红色印章与几粒色屑，抠图后仍留在 alpha 里，打包进图集就是脏点。

    阈值定 5% 而不是 2%：实测那枚印章占前景约 3%，2% 清不掉。代价是「角色 +
    完全分离且很小的手持物」会被误删 —— 但那种构图本来就该重画。

    denoise_iters > 0 时再叠一层**形态学开运算**（腐蚀→膨胀）：这一步对付的是
    「贴着主体的散落色点」——它们和主体是同一个连通域，面积过滤拿它们没办法。
    实测 SDXL 的像素风会在轮廓外留下这种点（看图就是一圈脏边），开运算能把宽度
    小于 kernel 的突起削掉，主体形状几乎不变。默认 0（不改变既有行为）。
    """
    arr = np.asarray(rgba.convert("RGBA")).copy()
    fg = (arr[..., 3] > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 2 and denoise_iters <= 0:
        return Image.fromarray(arr, "RGBA")

    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        thresh = float(areas.max()) * max(0.0, min_ratio)
        keep = [i + 1 for i, a in enumerate(areas) if a >= thresh]
        arr[~np.isin(lab, keep), 3] = 0

    if denoise_iters > 0:
        fg = (arr[..., 3] > 0).astype(np.uint8)
        opened = cv2.morphologyEx(
            fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=denoise_iters)
        arr[opened == 0, 3] = 0
    return Image.fromarray(arr, "RGBA")


def edge_stats(rgba: Image.Image) -> dict:
    """各边「前景贴边宽度」占整边的比例。

    为什么要占比而不是布尔：全身立绘本来就贴近画幅，边缘碰上一两像素是正常构图，
    只有贴边宽度明显（这里用 2%）才说明该部位真的被切断。
    """
    a = np.asarray(rgba.convert("RGBA").getchannel("A"))
    fg = a > 0
    h, w = fg.shape
    if h == 0 or w == 0:
        return {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    return {
        "top": round(float(fg[0].sum()) / w, 4),
        "bottom": round(float(fg[-1].sum()) / w, 4),
        "left": round(float(fg[:, 0].sum()) / h, 4),
        "right": round(float(fg[:, -1].sum()) / h, 4),
    }


def edge_touch(rgba: Image.Image, thresh: float = 0.02) -> list[str]:
    """被切断的边（贴边宽度 > thresh）。

    SD1.5 生成全身立绘时裁脚是常态，而提示词层面压不住（加 "feet visible" 只会
    让人物缩小、细节变糊，实测）。与其反复调 prompt，不如把「裁了」变成可观测
    信号：产物 meta 里带上 edge_touch，产线按它筛图或加宽重跑。
    """
    st = edge_stats(rgba)
    return [k for k, v in st.items() if v > thresh]


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


# ---------------- 语义分割后端（u2net / isnet） ----------------

_AI_SESSIONS: dict[str, Any] = {}


def load_ai_session(model_dir: str | Path, name: str = "isnet-general-use"):
    """加载（并缓存）分割模型。rembg 通过 U2NET_HOME 找本地权重，不走网络。"""
    key = f"{model_dir}:{name}"
    if key in _AI_SESSIONS:
        return _AI_SESSIONS[key]

    os.environ.setdefault("U2NET_HOME", str(Path(model_dir).resolve()))
    from rembg import new_session          # 延迟导入：不用 u2net 时不该付这个代价

    _AI_SESSIONS[key] = new_session(name)
    return _AI_SESSIONS[key]


def matte_ai(img: Image.Image, model_dir: str | Path, name: str = "isnet-general-use",
             thresh: int = 128, session: Any = None, feather: int = 0,
             fringe: bool = True) -> Image.Image:
    """语义分割抠图，alpha 仍二值化（像素素材要硬边）。

    分割只给「这是不是主体」，不会顺手清掉被背景混过的那一圈边 —— 所以这里
    也要走一遍 strip_fringe，否则轮廓外残留一圈背景色块。
    """
    sess = session or load_ai_session(model_dir, name)
    from rembg import remove

    rgb = np.asarray(img.convert("RGB"))
    cut = remove(img.convert("RGB"), session=sess, post_process_mask=True)
    arr = np.asarray(cut.convert("RGBA")).copy()
    arr[..., 3] = np.where(arr[..., 3] >= thresh, 255, 0).astype(np.uint8)

    if fringe:
        arr[..., 3] = strip_fringe(arr[..., 3], rgb, estimate_bg_color(img))
    out = Image.fromarray(arr, "RGBA")

    if feather > 0:                        # 分割边一般已很干净，默认不羽化
        m = np.asarray(out.getchannel("A")).astype(np.float32) / 255.0
        m = cv2.GaussianBlur(m, (0, 0), feather)
        arr = np.asarray(out).copy()
        arr[..., 3] = (m * 255).astype(np.uint8)
        out = Image.fromarray(arr, "RGBA")
    return out


AI_MODEL = "isnet-general-use.onnx"


def ai_available(model_dir: Optional[str | Path]) -> bool:
    return bool(model_dir) and (Path(str(model_dir)) / AI_MODEL).exists()


def fg_ratio(rgba: Image.Image) -> float:
    """前景（alpha>127）占比。用来判断一次抠图是不是「抠出个寂寞」。"""
    a = np.asarray(rgba.convert("RGBA").getchannel("A"))
    return float((a > 127).mean())


# 一次抠图的合理占比区间。低于 LO 基本等于什么都没剩下（主体被吃光 / 模型没认出主体），
# 高于 HI 说明背景大面积留着（色键把背景当成了主体）。
FG_LO, FG_HI = 0.01, 0.95


def _sane(rgba: Image.Image) -> bool:
    return FG_LO <= fg_ratio(rgba) <= FG_HI


def matte_auto(img: Image.Image, backend: str = "auto",
               model_dir: Optional[str | Path] = None,
               info: Optional[Dict[str, Any]] = None, **kw: Any) -> Image.Image:
    """统一入口。

    backend:
      auto  — 有分割模型就用 u2net，**抠空了自动回退色键**（推荐）
      color — 强制纯色键控（快，但白袍角色会误伤）
      u2net — 强制语义分割（不判断、不回退，出了问题自己认）
    未知值退回 color：不该因为拼错就整单失败。

    为什么要回退：实测草稿档（256²/8 步）出的图，u2net 只留下 **0.44%** 的前景
    ——模型没认出那是主体，于是「抠图成功」地交出一张几乎全透明的图，下游的
    贴边检测、主体计数、像素化全部拿到空数据，最后产出一张空精灵。这种失败是
    静默的，比报错难查得多。占比不合理时换成色键再比一次，取占比更合理的那个。
    """
    info = info if info is not None else {}
    want_ai = backend in ("u2net", "ai", "isnet")
    if backend == "auto":
        if not ai_available(model_dir):
            info.update(backend="color", fallback=False)
            return matte(img, **kw)
        cut = matte_ai(img, model_dir, **kw)
        if _sane(cut):
            info.update(backend="u2net", fallback=False)
            return cut
        alt = matte(img, **kw)
        pick_alt = _sane(alt) or fg_ratio(alt) > fg_ratio(cut)
        info.update(backend="color" if pick_alt else "u2net", fallback=True)
        return alt if pick_alt else cut

    if want_ai and ai_available(model_dir):
        info.update(backend="u2net", fallback=False)
        return matte_ai(img, model_dir, **kw)
    info.update(backend="color", fallback=False)
    return matte(img, **kw)
