"""实测（v2）：用官方管线的 `latents=` + `callback_on_step_end` 做 masked inpaint。

v1 失败的原因：手写采样循环时，条件图 latent 我用 `_encode_vae_image` 算了一份，
但 `prepare_latents` 内部也会算一份，两者走的归一化/打包路径不完全一致，
结果模型看到的前缀是错的，输出全糊。

v2 的做法：**不手写循环**，把初始 latents 交给管线的 `latents=` 入参，
用 `callback_on_step_end` 在每步结束后把蒙版外拉回原 latent。
管线负责条件图编码、img_shapes、img_mask、scheduler 一切细节。

ComfyUI 对应机制（comfy/samplers.py:641-642）：
    out = out * denoise_mask + self.latent_image * latent_mask
即「蒙版外每步被拉回原 latent」——采样器级，与权重无关。

跑法：python tools/verify_qwen_masked_inpaint.py
"""

import os
import sys
import time

sys.path.insert(0, '.')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from workflows.engine import Engine

OUT = 'assets/_mask_inpaint'
INSTR = ("change the armor to bright emerald green, keep the character identity, "
         "face, hairstyle and background unchanged")


def make_mask(src: Image.Image) -> Image.Image:
    m = Image.new('L', src.size, 0)
    d = ImageDraw.Draw(m)
    w, h = src.size
    d.ellipse([int(w * 0.34), int(h * 0.22), int(w * 0.66), int(h * 0.48)], fill=255)
    return m


def outside_diff(a: Image.Image, b: Image.Image, m: Image.Image) -> int:
    x = np.asarray(a.convert('RGB'), dtype=np.int16)
    y = np.asarray(b.convert('RGB'), dtype=np.int16)
    mm = np.asarray(m.convert('L')) > 127
    if not (~mm).any():
        return 0
    return int(np.abs(x - y)[~mm].max())


def masked_inpaint(pipe, src: Image.Image, msk: Image.Image, prompt: str,
                   steps: int = 20, cfg: float = 1.0, seed: int = 5,
                   work: int = 768, feather: int = 2):
    """ComfyUI 式 no-offset masked inpaint（用官方管线 + 回调钩子）。"""
    dev = pipe._execution_device
    dtype = torch.bfloat16
    w = h = work

    cond = src.convert('RGBA').resize((w, h), Image.LANCZOS)

    # --- 原图的 VAE latent ---
    # **必须与管线内部算的那一份逐字节一致**，否则回调里的空间网格错位，
    # 输出就是彩色条纹。原因：管线会用 output_resolution 经 calculate_dimensions
    # 重算条件图尺寸（不一定是 w×h），自己按 w×h 编码会对不上。
    # 所以这里复刻管线的那段逻辑，拿到与 input_images_latents 完全相同的 latent。
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    output_resolution = max(w, h)
    iw, ih, _ = calculate_dimensions(output_resolution * output_resolution, w / h)
    vae_img = pipe.image_processor.preprocess(cond, width=iw, height=ih).unsqueeze(2)
    with torch.no_grad():
        lat0 = pipe._encode_vae_image(
            vae_img.to(dev, dtype), torch.Generator("cpu").manual_seed(seed))
    lat0 = lat0.squeeze(2)                                     # [1, C, lh, lw]
    lh, lw = lat0.shape[2], lat0.shape[3]

    # --- 蒙版缩到 VAE latent 空间 ---
    m = msk.convert('L').resize((lw, lh), Image.BILINEAR)
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    m_arr = np.asarray(m).astype(np.float32) / 255.0
    m_t = torch.from_numpy(m_arr).to(dev, dtype)[None, None]   # [1,1,lh,lw]

    # --- 初始 latents：flow matching 下 σ_0=1，纯噪声 ---
    gen = torch.Generator(dev).manual_seed(seed)
    eps = torch.randn((1, lat0.shape[1], lh, lw), generator=gen,
                      device=dev, dtype=dtype)
    # 管线的 latents 是 pack 过的 [B, seq, C]
    packed = pipe._pack_latents(eps, 1, lat0.shape[1], lh, lw)

    def _cb(pipe_, i, tstep, kwargs):
        """每步结束后把蒙版外拉回原 latent（ComfyUI 的 denoise_mask 语义）。"""
        lt = kwargs['latents']
        b, seq, c = lt.shape
        # **Qwen 2.1 的 latents 是 2x2 patchify 过的**（管线里 `latents.shape[1] // 4`
        # 就是这个意思），所以 seq = (lh/2)*(lw/2)*4 之外的排布不能想当然。
        # 用管线的 _pack_latents / _unpack_latents 来做往返，保证与官方一致。
        spatial = pipe_._unpack_latents(lt, h, w, pipe_.vae_scale_factor)
        spatial = spatial.reshape(b, c, lh, lw)
        # flow matching：x_t = (1-σ)·x0 + σ·ε
        sig = pipe_.scheduler.sigmas
        # 步后 latent 已降到 σ_{i+1}
        sigma = float(sig[i + 1]) if i + 1 < len(sig) else 0.0
        kept = (1 - sigma) * lat0 + sigma * eps
        spatial = spatial * m_t + kept * (1 - m_t)
        kwargs['latents'] = pipe_._pack_latents(spatial, b, c, lh, lw)
        return kwargs

    t0 = time.time()
    res = pipe(
        prompt=prompt,
        image=cond,
        width=w, height=h,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        generator=torch.Generator(dev).manual_seed(seed),
        output_resolution=max(w, h),
        latents=packed,
        callback_on_step_end=_cb,
        callback_on_step_end_tensor_inputs=['latents'],
    )
    dt = time.time() - t0
    return res.images[0].convert('RGBA'), dt


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = Engine('models')
    src = Image.open('assets/_pc_char.png').convert('RGBA')
    msk = make_mask(src)
    msk.save(f'{OUT}/mask.png')
    src.convert('RGB').save(f'{OUT}/src.png')

    pipe = eng.txt2img_pipe('qwen-image-2.1')

    out, dt = masked_inpaint(pipe, src, msk, INSTR, steps=20, cfg=1.0, seed=5)
    out.save(f'{OUT}/masked_inpaint.png')
    # 比较要在同一尺寸上做
    src_c = src.convert('RGB').resize(out.size, Image.LANCZOS)
    msk_c = msk.resize(out.size, Image.LANCZOS)
    od = outside_diff(src_c, out.convert('RGB'), msk_c)
    print(f'[masked inpaint] {dt:.1f}s  蒙版外最大像素差={od}', flush=True)

    a = src.convert('RGB').resize((384, 384))
    b = out.convert('RGB').resize((384, 384))
    c = Image.new('RGB', (384, 384), (20, 20, 24))
    dd = ImageDraw.Draw(c)
    dd.rectangle([int(384 * 0.34), int(384 * 0.22), int(384 * 0.66), int(384 * 0.48)],
                 outline=(255, 80, 80), width=3)
    s = Image.new('RGB', (384 * 3, 384), (20, 20, 24))
    s.paste(a, (0, 0)); s.paste(b, (384, 0)); s.paste(c, (768, 0))
    s.save(f'{OUT}/cmp.png')
    print('对比图:', f'{OUT}/cmp.png', flush=True)


if __name__ == '__main__':
    main()
