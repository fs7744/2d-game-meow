"""角色出图后的纯色幕抠像（PIL 进出，内部走移植来的 `chroma_screen_matte`）。

**为什么单独一层**：`chroma_screen_matte.py` 是官方的 tensor 实现（整文件搬运，
只改了设备选择），它吃 HWC float tensor。产线上我们手里是 PNG 文件，所以这里
只负责「读图 → 转 tensor → 抠 → 存 RGBA PNG」这段胶水，算法一行都不写。

**零模型依赖**：不下载任何权重，不需要 GPU（CPU 能跑，官方也是这么用的）。

⚠️ **白底要小心**：幕布色是从**边框取样**估出来的，白底 + 白毛/白甲的角色会把
主体一起判成背景（我们项目里这条踩过：见 README「抠图」）。所以调用方在
`background_color` 是 white / grey 时应当先提示用户，而不是闷头抠。
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

from workflows import chroma_screen_matte
from workflows import paths

#: 官方默认参数（chroma_screen_matte.screen_matte 的签名默认值）
DEFAULT_PARAMS: Dict[str, Any] = {
    "tolerance": 0.15,
    "softness": 0.12,
    "despill_strength": 0.65,
    "edge_width": 3,
    "matte_cleanup": 0.1,
    "foreground_recover": 0.35,
    "edge_decontaminate": 0.75,
    "edge_choke": 0.08,
}

#: 这几档幕布色适合自动抠；白/灰因为容易吃掉浅色主体，交给用户决定
SAFE_KEY_COLORS = {"green", "blue", "magenta"}


def _to_tensor(im: Image.Image) -> torch.Tensor:
    """PIL → HWC float tensor（0~1）。透明图先压到白底上，免得 alpha 干扰取样。"""
    rgb = im.convert("RGBA")
    flat = Image.new("RGBA", rgb.size, (255, 255, 255, 255))
    flat.alpha_composite(rgb)
    # bytearray 而不是 bytes：torch 会警告"缓冲不可写"（见调用处的实测）
    arr = torch.frombuffer(bytearray(flat.convert("RGB").tobytes()), dtype=torch.uint8)
    h, w = rgb.size[1], rgb.size[0]
    t = arr.view(h, w, 3).to(torch.float32) / 255.0
    return t


#: 全局「抠图方式」的小缓存。角色中心一次批量可能抠上百帧，每帧都开一次
#: SQLite 连接不值当；设置改了最多晚 TTL 生效，用户感知不到。
_BACKEND_CACHE: Dict[str, Any] = {"at": 0.0, "v": "auto"}
_BACKEND_TTL = 2.0
_ALLOWED_BACKENDS = ("auto", "color", "u2net", "ai", "isnet", "birefnet")


def reset_backend_cache() -> None:
    """让下一次调用重新读设置（设置刚改完 / 测试里用）。"""
    _BACKEND_CACHE["at"] = 0.0


def global_backend() -> str:
    """读全局「抠图方式」（顶栏 ⚙ 设置，存在 SQLite）。读不到就当 auto。

    **auto 保持原来的官方幕布键控**：角色出图的背景是已知纯色幕，
    `chroma_screen_matte` 就是为它调的，换成语义分割反而是降级。
    只有用户明确选了 u2net / color，才切到 `matte.matte_auto` 那套算法。
    """
    now = time.time()
    if now - float(_BACKEND_CACHE["at"]) < _BACKEND_TTL:
        return str(_BACKEND_CACHE["v"])
    v = "auto"
    try:
        from workflows import metadb
        v = str(metadb.get_settings().get("matte_backend") or "auto")
    except Exception:                      # noqa: BLE001
        v = "auto"
    if v not in _ALLOWED_BACKENDS:
        v = "auto"
    _BACKEND_CACHE.update(at=now, v=v)
    return v


def matte_pil(im: Image.Image, **params: Any) -> Optional[Image.Image]:
    """抠一张**已经在内存里**的图，返回 RGBA。

    表情批量生成要边走边抠（贴回 → 抠 → 落盘），先写盘再读回来纯属白折腾，
    而且中间那个临时文件还得管。算法与 `matte_rgba` 完全一样，只是不吃路径。

    **算法跟全局设置走**：`auto` 用官方幕布键控（默认，背景是纯色幕时最准）；
    用户选了 u2net / color 就交给 `matte.matte_auto`，与画布节点同一条路。
    以前这个入口硬编码一种算法，所以「抠图方式」对角色中心完全不生效。
    """
    if im is None:
        return None
    backend = global_backend()
    if backend != "auto":
        from workflows import matte as matte_mod
        return matte_mod.matte_auto(im, backend, model_dir=paths.MODELS_DIR / "matte")
    tensor = _to_tensor(im)
    kw = dict(DEFAULT_PARAMS)
    kw.update({k: v for k, v in params.items() if v is not None and k in DEFAULT_PARAMS})
    with torch.inference_mode():
        rgba_t, _matte, _debug = chroma_screen_matte.screen_matte(
            tensor, output_mode="straight_rgba", **kw)
    arr = (rgba_t.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr, mode="RGBA")


def matte_rgba(src: str, **params: Any) -> Optional[Image.Image]:
    """抠一张图，返回 RGBA 的 PIL（失败返回 None，让调用方保留原图）。"""
    if not os.path.isfile(src):
        return None
    try:
        with Image.open(src) as im:
            im.load()
            return matte_pil(im, **params)
    except (OSError, ValueError) as exc:
        raise ValueError(f"读不了这张图：{src}（{exc}）") from exc


def matte_image_file(
    src: str, out: Optional[str] = None, **params: Any
) -> Tuple[Optional[str], Optional[str]]:
    """抠图并存成 PNG。返回 `(输出路径, 失败原因)`。

    **失败不抛**：抠像是增强——原图还在，不能因为抠图失败就把整次出图判失败。
    """
    try:
        rgba = matte_rgba(src, **params)
    except Exception as exc:                       # noqa: BLE001
        return None, str(exc)
    if rgba is None:
        return None, f"图不存在：{src}"
    target = out or (os.path.splitext(src)[0] + "_matte.png")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    rgba.save(target, format="PNG")
    return target, None


def should_auto_matte(background_color: str, requested: Optional[bool] = None) -> bool:
    """要不要自动抠。`requested` 明确给了就以它为准（用户开关优先）。"""
    if requested is not None:
        return bool(requested)
    return str(background_color or "").strip().lower() in SAFE_KEY_COLORS
