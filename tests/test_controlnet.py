"""另外两条 ControlNet 通路：Canny（形状迁移）、Tile（高清补细节）。"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from workflows import postprocess as pp


def _src(assets, size=256) -> Image.Image:
    p = assets / "_test_cn_src.png"
    im = Image.new("RGB", (size, size), (245, 245, 245))
    # 一个有明确轮廓的形体：方块 + 圆
    from PIL import ImageDraw

    d = ImageDraw.Draw(im)
    d.rectangle([size * 0.25, size * 0.35, size * 0.7, size * 0.85], fill=(40, 60, 120))
    d.ellipse([size * 0.3, size * 0.12, size * 0.65, size * 0.45], fill=(200, 120, 60))
    im.save(p)
    return im


def _canny(im: Image.Image) -> Image.Image:
    g = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, 80, 160)
    return Image.fromarray(np.stack([e] * 3, axis=-1))


def test_canny_controlnet_conditioning_has_effect(eng, assets):
    src = _src(assets)
    cond = _canny(src)
    pipe = eng.pose_pipe(controlnet="canny", with_ipadapter=False)
    kw = dict(prompt="pixel art game sprite, blue robot, flat colors",
              negative_prompt="photo, blurry", image=cond, num_inference_steps=12,
              guidance_scale=7.5, width=256, height=256)

    a = pipe(**kw, controlnet_conditioning_scale=0.0,
             generator=torch.Generator("cuda").manual_seed(1)).images[0]
    b = pipe(**kw, controlnet_conditioning_scale=1.0,
             generator=torch.Generator("cuda").manual_seed(1)).images[0]
    assert np.abs(np.asarray(a, dtype=int) - np.asarray(b, dtype=int)).max() > 0, \
        "conditioning_scale 0 与 1 结果相同，说明 ControlNet 没接上"


def test_tile_controlnet_upscales(eng, assets):
    src = _src(assets, 256)
    pipe = eng.pose_pipe(controlnet="tile", with_ipadapter=False)
    out = pipe(prompt="pixel art game sprite, clean flat colors",
               negative_prompt="photo, blurry", image=src, num_inference_steps=12,
               guidance_scale=7.0, width=512, height=512,
               controlnet_conditioning_scale=1.0,
               generator=torch.Generator("cuda").manual_seed(2)).images[0]
    assert out.size == (512, 512)
    # 结构应当保留：缩回 256 后与原图的相关性明显高于随机
    back = np.asarray(out.resize((256, 256), Image.LANCZOS), dtype=float)
    ref = np.asarray(src, dtype=float)
    corr = np.corrcoef(back.mean(axis=2).ravel(), ref.mean(axis=2).ravel())[0, 1]
    assert corr > 0.5, f"tile 输出与原图结构相关性过低: {corr:.2f}"


def test_pixel_postprocess_on_controlnet_output(eng, assets):
    src = _src(assets)
    cond = _canny(src)
    pipe = eng.pose_pipe(controlnet="canny", with_ipadapter=False)
    out = pipe(prompt="pixel art game sprite, blue robot, flat colors",
               negative_prompt="photo, blurry", image=cond, num_inference_steps=8,
               guidance_scale=7.5, width=256, height=256,
               controlnet_conditioning_scale=0.9,
               generator=torch.Generator("cuda").manual_seed(3)).images[0]
    px = pp.quantize_palette(pp.pixelate(out, 2), 12)
    colors = {p[:3] for p in px.convert("RGBA").getdata() if p[3] > 127}
    assert len(colors) <= 12
    # 像素块均匀性：2x2 块内必须同色
    arr = np.asarray(px.convert("RGB"))
    assert (arr[0::2, 0::2] == arr[1::2, 0::2]).all()
    assert (arr[0::2, 0::2] == arr[0::2, 1::2]).all()
