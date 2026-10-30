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


def flatten_white(img: Image.Image) -> Image.Image:
    """把透明区合成到**白底**上，返回 RGB。

    **别用 `convert("RGB")`**：那只是丢掉 alpha 通道，透明区的 RGB 是什么就留
    什么——Qwen 的透明输出在透明区常常是 VAE 解出来的噪声色（偏黑/偏品红）。
    分割模型（isnet / birefnet）和色键都会把这片噪声当成主体或背景的一部分，
    实测会直接把主体的白色部分咬掉一圈。合成到白底后，模型看到的和用户看到的
    是同一张图。
    """
    rgba = img.convert("RGBA")
    if rgba.getchannel("A").getextrema()[0] >= 250:
        return rgba.convert("RGB")          # 本来就不透明，不必多一次拷贝
    out = Image.new("RGB", rgba.size, (255, 255, 255))
    out.paste(rgba, (0, 0), rgba)
    return out


def alpha_trust(rgba: Image.Image) -> tuple[bool, Dict[str, Any]]:
    """判断「输入自带的 alpha 能不能直接用」。

    场景：Qwen-Image-2.1 原生出透明底（VAE 第 4 通道），本来就不需要再抠。
    但它的 alpha 也可能是一层雾（提示词和解耦声明打架时，实测 77% 的像素停在
    半透明中间值）。所以要看三个数：

      · `a0`   完全透明像素占比 —— 没有真透明区就谈不上「自带 alpha」
      · `mid`  16~240 的中间值占比 —— 雾状 alpha 的指纹，正常 < 10%
      · `fg`   前景占比 —— 低于 1% 是抠空了，高于 95% 等于没抠

    返回 (是否可信, 观测值)。调用方把观测值写进 meta，出问题能查。
    """
    a = np.asarray(rgba.convert("RGBA").getchannel("A"))
    a0 = float((a == 0).mean())
    mid = float(((a >= 16) & (a <= 240)).mean())
    fg = float((a > 127).mean())
    ok = a0 >= 0.10 and mid <= 0.12 and 0.01 <= fg <= 0.95
    return ok, {"a0": round(a0, 4), "mid": round(mid, 4), "fg": round(fg, 4)}


def fill_holes(rgba: Image.Image, max_ratio: float = 0.08) -> Image.Image:
    """把主体**内部**的空洞填回实心（只填不与画布边界连通的洞）。

    为什么需要：白色/浅色主体配浅色背景时，分割模型会把脸上、胸口的高光一起
    判成背景——实测白猫骑士的额头与脸颊被咬掉一块、胸前出现一个灰洞。
    这类「洞」有两个特征：不与外界连通、面积远小于主体。补回来几乎不会误伤
    （两腿之间的空隙是**通到边界**的，不会被填）。

    `max_ratio` 是洞面积上限（相对全图），防的是「主体本来是环形的」被填死。
    """
    arr = np.asarray(rgba.convert("RGBA")).copy()
    a = arr[..., 3]
    fg = (a > 127).astype(np.uint8)
    if fg.sum() == 0:
        return rgba
    h, w = fg.shape
    # 补 1px 边框再从 (0,0) 泛洪：否则贴着画布边缘的背景填不到，整张图会被
    # 当成一个洞填满（这是 flood fill 填洞的经典边界坑）。
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = fg
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = (ff[1:-1, 1:-1] == 0) & (fg == 0)
    if not holes.any():
        return rgba
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), connectivity=4)
    lim = max_ratio * h * w
    filled = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= lim:
            arr[..., 3][lab == i] = 255
            filled += area
    if filled == 0:
        return rgba
    return Image.fromarray(arr, "RGBA")


def matte(img: Image.Image, bg: Iterable[int] | None = None, tol: int = 32,
          fringe: bool = True, fringe_ratio: float = 3.0,
          fringe_passes: int = 3, holes: bool = True) -> Image.Image:
    """纯色背景抠图，返回 RGBA（alpha 只有 0/255）。

    bg: 背景色；None 时从画面四边自动估计。
    tol: 判为同色的最大通道差。
    fringe: 是否去掉混色边（白边/绿边）。
    fringe_ratio: 混色边判据的倍数（越大越保守）。
    holes: 是否把主体内部的空洞补回实心（见 fill_holes）。
    """
    flat = flatten_white(img) if img.mode == "RGBA" else img
    rgb = np.asarray(flat.convert("RGB")).astype(np.int16)
    bg_arr = np.array(list(bg) if bg is not None else estimate_bg_color(flat),
                      dtype=np.int16)
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
    rgba = Image.fromarray(out, "RGBA")
    return fill_holes(rgba) if holes else rgba


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
             fringe: bool = True, alpha_mode: str = "binary",
             holes: bool = True) -> Image.Image:
    """语义分割抠图，alpha 默认二值化（像素素材要硬边）。

    `alpha_mode="soft"`：保留分割模型给出的半透明边。**高清插画资产要用这个**——
    发丝、纱、光效压成 0/255 之后会变成一圈硬齿。像素素材相反：半透明边在
    放大时就是脏边，所以默认仍是 binary（行为不变）。

    分割只给「这是不是主体」，不会顺手清掉被背景混过的那一圈边 —— 所以这里
    也要走一遍 strip_fringe，否则轮廓外残留一圈背景色块。

    `holes`：把主体内部被误判成背景的洞补回实心（见 `fill_holes`）。
    白色主体配浅色背景时（白猫＋白底），分割模型会把额头、胸口一起啃掉，
    这不是"边界切多了"而是"内部挖洞"，只能靠填洞救。
    """
    sess = session or load_ai_session(model_dir, name)
    from rembg import remove

    # 透明区要先合成到白底：直接 convert("RGB") 会让透明区保留 VAE 噪声色，
    # 分割模型看到一片脏底，结果连主体的白色部分一起判成背景
    flat = flatten_white(img)
    rgb = np.asarray(flat)
    cut = remove(flat, session=sess, post_process_mask=True)
    arr = np.asarray(cut.convert("RGBA")).copy()
    if alpha_mode != "soft":
        arr[..., 3] = np.where(arr[..., 3] >= thresh, 255, 0).astype(np.uint8)

    if fringe:
        arr[..., 3] = strip_fringe(arr[..., 3], rgb, estimate_bg_color(flat))
    out = Image.fromarray(arr, "RGBA")
    if holes:
        out = fill_holes(out)

    if feather > 0:                        # 分割边一般已很干净，默认不羽化
        m = np.asarray(out.getchannel("A")).astype(np.float32) / 255.0
        m = cv2.GaussianBlur(m, (0, 0), feather)
        arr = np.asarray(out).copy()
        arr[..., 3] = (m * 255).astype(np.uint8)
        out = Image.fromarray(arr, "RGBA")
    return out


AI_MODEL = "isnet-general-use.onnx"
# rembg 的模型名。**按文件存在性自动选**：放了 birefnet 就用它，
# 没有就还是 isnet（不改变现有机器的行为）。
# BiRefNet 是 2024 的 SOTA 分割模型，细长结构（剑、法杖、飘带）与半透明边缘
# 明显强于 2022 的 isnet——而"剑被吃掉半截"正是精灵素材最常见的废图原因。
AI_MODELS = (
    ("birefnet-general.onnx", "birefnet-general"),
    ("isnet-general-use.onnx", "isnet-general-use"),
)

# 高清插画资产不该把 alpha 压成二值：发丝、纱、光效都需要半透明。
# 像素素材相反——半透明边在放大的时候会变成脏边。所以由调用方给 alpha_mode。
ALPHA_MODES = ("auto", "binary", "soft")


def pick_model(model_dir: Optional[str | Path]) -> tuple[str, str] | None:
    """本机可用的分割权重（文件名, rembg 模型名），优先级从上到下。"""
    if not model_dir:
        return None
    for fname, name in AI_MODELS:
        if (Path(str(model_dir)) / fname).exists():
            return fname, name
    return None


def ai_available(model_dir: Optional[str | Path]) -> bool:
    return pick_model(model_dir) is not None


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
      auto  — 输入自带可信 alpha 就直接用；否则有分割模型就用 u2net，
              **抠空了自动回退色键**（推荐）
      color — 强制纯色键控（快，但白袍角色会误伤）
      u2net — 强制语义分割（不判断、不回退，出了问题自己认）
    未知值退回 color：不该因为拼错就整单失败。

    **`auto` 会先看「这张图是不是已经有 alpha」**：Qwen-Image-2.1 的 VAE 有第
    4 通道，出图时勾了透明底就已经是抠好的 RGBA。再拿分割模型重猜一遍是纯损失
    ——实测白猫骑士（白毛＋白底）会被 isnet 把额头、脸颊一起判成背景咬掉，
    用户看到的就是「抠图把脸抠没了」。判据见 `alpha_trust`，
    只有 alpha 是雾状（提示词和透明声明打架时会出现）才会真的重抠。

    为什么要回退：实测草稿档（256²/8 步）出的图，u2net 只留下 **0.44%** 的前景
    ——模型没认出那是主体，于是「抠图成功」地交出一张几乎全透明的图，下游的
    贴边检测、主体计数、像素化全部拿到空数据，最后产出一张空精灵。这种失败是
    静默的，比报错难查得多。占比不合理时换成色键再比一次，取占比更合理的那个。
    """
    info = info if info is not None else {}
    # 两种后端的参数集不同（alpha_mode 只有分割后端认），混着传会 TypeError。
    # 这里按后端分流，避免以后加参数时又踩一次。
    ai_kw = {k: v for k, v in kw.items()
             if k in ("thresh", "feather", "fringe", "alpha_mode", "session",
                      "holes")}
    color_kw = {k: v for k, v in kw.items()
                if k in ("bg", "tol", "fringe", "fringe_ratio", "fringe_passes",
                         "holes")}

    if backend == "auto":
        rgba = img if img.mode == "RGBA" else img.convert("RGBA")
        ok, obs = alpha_trust(rgba)
        if ok:
            # 自带 alpha 已经是答案：只做连通域清理（碎屑/孤立点），不再重猜
            info.update(backend="alpha", fallback=False, alpha_reused=True, **obs)
            return rgba.copy()
        if obs["a0"] > 0.01:               # 有透明区但不可信（雾状）→ 留痕
            info.update(alpha_rejected=True, **obs)

        if not ai_available(model_dir):
            info.update(backend="color", fallback=False)
            return matte(img, **color_kw)
        name = _model_name(model_dir)
        cut = matte_ai(img, model_dir, name=name, **ai_kw)
        if _sane(cut):
            info.update(backend=_short(name), fallback=False)
            return cut
        alt = matte(img, **color_kw)
        pick_alt = _sane(alt) or fg_ratio(alt) > fg_ratio(cut)
        info.update(backend="color" if pick_alt else _short(name), fallback=True)
        return alt if pick_alt else cut

    want_ai = backend in ("u2net", "ai", "isnet", "birefnet")
    if want_ai and ai_available(model_dir):
        name = _model_name(model_dir)
        if backend == "birefnet" and not name.startswith("birefnet"):
            # 显式要 birefnet 但只有 isnet：降级要留痕，不能静默
            info["downgraded"] = f"birefnet 权重缺失，退回 {name}"
        info.update(backend=_short(name), fallback=False)
        return matte_ai(img, model_dir, name=name, **ai_kw)
    info.update(backend="color", fallback=False)
    return matte(img, **color_kw)


def _model_name(model_dir: Optional[str | Path]) -> str:
    pick = pick_model(model_dir)
    return pick[1] if pick else "isnet-general-use"


def _short(name: str) -> str:
    """meta 里的后端名保持向后兼容：isnet 仍记作 u2net（老产物/前端都在用）。"""
    return "birefnet" if name.startswith("birefnet") else "u2net"
