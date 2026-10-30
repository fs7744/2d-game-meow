"""ControlNet Union 端到端验证：结构控制是否真的生效。

三件事一起测：
  1. 权重能不能 strict 加载（180 key 全对）
  2. 显存够不够（base fp8 + control bf16 ≈ 14GB，24G 卡上要实测）
  3. **控制有没有用**——同一提示词，带骨架 vs 不带骨架，出图是否听骨架

第 3 点是唯一有说服力的判据。只看"没报错"等于没测。

用法: python tools/verify_controlnet_union.py [--pose 骨架图] [--size 768] [--steps 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image

from workflows.engine import Engine
from workflows import controlnet_union as cnu

W = "models/controlnet/qwen21_union.safetensors"


def mem(tag: str) -> None:
    if torch.cuda.is_available():
        print(f"  [{tag}] reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB "
              f"peak={torch.cuda.max_memory_reserved()/1024**3:.2f}GB", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="assets/_pose_walk_02.png")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=24,
                    help=">8 = 不挂 Lightning 跑这么多步（官方示例用 40）；<=8 = 挂 Lightning")
    ap.add_argument("--fast", action="store_true", help="强制挂 Lightning（步数压到 8）")
    ap.add_argument("--letterbox", action="store_true",
                    help="条件图保持比例居中贴到方形画布（官方训练是 bucket+crop，"
                         "推理示例却是裸拉伸；竖图拉伸会把腿压扁）")
    ap.add_argument("--scale", type=float, default=0.45, help="单值；24 步甜点 0.45，8 步甜点 0.6")
    ap.add_argument("--scales", default="", help="逗号分隔，一次加载扫多个强度")
    ap.add_argument("--prompt",
                    default="a female warrior in blue leather armor, standing with "
                            "legs wide apart, arms hanging at her sides, facing viewer, "
                            "full body, 2D game character sprite, flat cel shading, "
                            "clean lineart, detailed")
    ap.add_argument("--transparent", action="store_true",
                    help="让模型直接出 RGBA 透明底（走 generate 的 alpha 指令；"
                         "会顺带清掉与透明底冲突的背景词）")
    ap.add_argument("--control", default="pose", help="pose / canny / depth / none")
    ap.add_argument("--mask", default="zero", choices=["zero", "full"],
                    help="mask 通道：zero=全重画（纯结构控制的正确填法）；full=全保留（只在 inpaint 且 masked-image 有内容时才对）")
    ap.add_argument("--masked", default="zero", choices=["zero", "self"],
                    help="masked-image 那 64 个通道填 0 还是条件图")
    ap.add_argument("--end", type=float, default=1.0,
                    help="只在采样的前这个比例内注入（1.0 = 全程）")
    ap.add_argument("--decay", default="none", choices=["none", "sigma", "linear"],
                    help="注入强度随时间衰减的方式")
    ap.add_argument("--head-cut", type=float, default=0.0,
                    help="顶部这个比例的图像行不注入控制（0.25 左右能把头部交给模型自由画）")
    args = ap.parse_args()
    args.scales = ([float(x) for x in args.scales.split(",") if x.strip()]
                   if args.scales.strip() else [args.scale])

    sz = args.size // 32 * 32          # 对齐 32（管线 multiple_of = vae_scale*2）
    if not os.path.isfile(W):
        print(f"权重不在：{W}")
        return 1

    print("加载管线…", flush=True)
    t0 = time.time()
    eng = Engine("models")
    pipe = eng.txt2img_pipe("qwen-image-2.1")
    print(f"管线 {time.time()-t0:.1f}s", flush=True)
    mem("管线加载完")

    tr = pipe.transformer
    n_blocks = len(tr.transformer_blocks)
    print(f"transformer blocks={n_blocks}，注入点={[i for i in cnu.CONTROL_LAYERS if i < n_blocks]}",
          flush=True)

    print("加载 control 分支（strict 校验）…", flush=True)
    t0 = time.time()
    branch = cnu.load_branch(W, dtype=torch.bfloat16, device="cuda")
    print(f"分支 {time.time()-t0:.1f}s", flush=True)
    mem("分支上卡")

    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype

    # --- 条件 ---
    cond = None
    ctrl_lat = None
    if args.control != "none":
        assert os.path.isfile(args.pose), f"条件图不存在：{args.pose}"
        cond = Image.open(args.pose).convert("RGB")
        orig_size = cond.size
        if args.letterbox:
            # 保持比例居中贴到方形画布（黑底）。官方**训练**是 bucket-resize +
            # CenterCrop，**推理示例**却是裸拉伸——两者本来就不一致。竖版骨架
            # 被拉伸会把腿压扁，模型看到的是"短腿宽身"，姿态自然走样。
            canvas = Image.new("RGB", (sz, sz), (0, 0, 0))
            r = min(sz / cond.width, sz / cond.height)
            nw, nh = max(1, round(cond.width * r)), max(1, round(cond.height * r))
            canvas.paste(cond.resize((nw, nh), Image.LANCZOS),
                         ((sz - nw) // 2, (sz - nh) // 2))
            cond = canvas
        print(f"条件图 {args.pose} {orig_size} → {cond.size}"
              f"{'（letterbox）' if args.letterbox else '（拉伸）'}", flush=True)
        # 129 通道里 mask 那 1 个通道的语义不明确：官方说纯结构控制时
        # "zero-pads the mask"，但如果 0 表示"不要控制"，那等于自废武功。
        # 两个填法都试一遍，用出图效果定论。
        m = None
        if args.mask == "full":
            # 白 = 重画；encoder_condition 内部会转成 1=保留，所以白图 → 0
            m = Image.new("L", (sz, sz), 255)
            print("  mask 通道传全白（=全重画）", flush=True)
        mi = cond if args.masked == "self" else None
        if mi is not None:
            print("  masked-image 通道填条件图本身", flush=True)
        t0 = time.time()
        ctrl_lat = cnu.encoder_condition(
            control=cond, mask=m, masked_image=mi, vae=vae,
            size=(sz, sz), device="cuda", dtype=vae_dtype)
        print(f"条件编码 {time.time()-t0:.1f}s shape={tuple(ctrl_lat.shape)}", flush=True)
        print(f"  latent  min={ctrl_lat.min():.2f} max={ctrl_lat.max():.2f} "
              f"absmean={ctrl_lat.abs().mean():.2f} nan={torch.isnan(ctrl_lat).any().item()}",
              flush=True)

    steps = max(4, int(args.steps))
    use_fast = bool(args.fast) or steps <= 8
    if use_fast:
        eng.apply_loras(pipe, [{"name": eng.FAST_LORA, "scale": 1.0}])
        steps = min(steps, eng.FAST_STEPS)
    print(f"步数 {steps}{'（挂 Lightning）' if use_fast else '（未蒸馏）'}", flush=True)

    # 透明底要走 generate 里的 alpha 指令，并清掉与透明底冲突的背景词
    # （"plain background" 之类会让 alpha 变成一片半透明雾）
    from workflows.generate import _qwen_prompt, _strip_bg_words
    _clean, _changed = _strip_bg_words(args.prompt)
    full_prompt = _qwen_prompt(_clean, args.transparent)
    if args.transparent:
        print(f"透明底：清掉背景词={_changed}", flush=True)

    def flatten(im: Image.Image) -> Image.Image:
        """把透明底铺到白底上，否则拼对比图时透明区会变黑。"""
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            return bg
        return im.convert("RGB")

    def run(label: str, inject: bool, scale: float = 1.0) -> Image.Image:
        inj = None
        if inject:
            inj = cnu.ControlInjection(tr, branch, scale=scale,
                                       head_cut=args.head_cut)
            inj.set_condition(ctrl_lat)
            print(f"  注入已挂：{len(inj.layers)} 个点 {inj.layers}", flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            kw = {}
            if inj is not None:
                # **有 control 时必须关 KV cache**。pipeline 默认 `use_kv_cache=True`，
                # 它会把文本 token 当 prefix 缓存起来，之后每个 block 只处理**图像**
                # token（2304），而 control 链算的是**完整 joint**（2331）——
                # 两者形状对不上，hint 只能硬塞到图像段，信号全错。
                # 官方实现里就是 `cache_enabled = use_kv_cache and ... and control_context is None`。
                kw["use_kv_cache"] = False
            if inj is not None and (args.end < 1.0 or args.decay != "none"):
                # 两个手段都作用在「注入的时间分布」上。不加限制时：
                # 8 步注入 8 次还算干净，24 步注入 24 次画面直接变成骨架的
                # 彩色线本身（模型在复制条件图，不是按姿态生成）——说明常量
                # skip 在低噪声后期占比过大，必须按时间衰减。
                def _cb(pp, step, timestep, cbkw, _n=steps, _i=inj):
                    frac = step / max(_n, 1)
                    if frac >= args.end:
                        _i.enabled = False
                        return cbkw
                    _i.enabled = True
                    if args.decay == "sigma":
                        # flow matching 的 timestep ≈ sigma*1000：早期 σ≈1 强，
                        # 晚期 σ→0 弱，结构在早期定、细节不被污染
                        sig = float(timestep) / 1000.0
                        _i.scale = args.scale * max(0.0, min(1.0, sig))
                    elif args.decay == "linear":
                        _i.scale = args.scale * (1.0 - frac)
                    return cbkw
                kw["callback_on_step_end"] = _cb
            out = pipe(prompt=full_prompt, width=sz, height=sz,
                       num_inference_steps=steps, true_cfg_scale=1.0,
                       generator=torch.Generator("cuda").manual_seed(7),
                       output_resolution=sz, **kw).images[0]
            dt = time.time() - t
            print(f"  {label}: {dt:.1f}s", flush=True)
            mem(label)
            p = f"assets/_cn_{label}.png"
            img = out if args.transparent else out.convert("RGB")
            img.save(p)
            if args.transparent and img.mode == "RGBA":
                al = img.split()[3]
                fg = sum(1 for v in al.getdata() if v > 200) / (img.width * img.height)
                print(f"  → {p}  RGBA 不透明占比 {fg:.1%}", flush=True)
            else:
                print(f"  → {p}", flush=True)
            return out
        finally:
            if inj is not None:
                if inj.trace:
                    for li, ho, sk in inj.trace:
                        print(f"    层{li:>2}: hidden_absmean={ho:.3f} skip_absmean={sk:.3f} "
                              f"比值={sk/max(ho,1e-9):.2f}", flush=True)
                # 双流的关键证据：skip 应当**随采样步变化**（旧实现是常量）
                if inj.step_trace:
                    print(f"    各步首层 skip absmean: "
                          f"{[round(v, 3) for v in inj.step_trace]}", flush=True)
                print(f"    设备错配 = {inj._device_mismatch}", flush=True)
                mk = inj._image_pad_mask
                if mk is not None:
                    n_img = int(mk.sum())
                    # 前缀方案是否恰好等价：图像位置是否连续地落在末尾
                    idx = mk.nonzero().flatten()
                    contiguous = bool((idx.diff() == 1).all().item()) if len(idx) > 1 else True
                    print(f"    图像位置 {n_img}/{len(mk)}，是否连续后缀 = {contiguous}"
                          f"（False 就说明前缀切片是错的）", flush=True)
                else:
                    print("    拿不到 img_mask（走了前缀兜底）", flush=True)
                inj.remove()

    print(f"\n[1] 无控制（对照组） steps={steps}", flush=True)
    base = run("none", False)

    if cond is not None:
        import numpy as np
        a = np.asarray(flatten(base), dtype=np.float32)
        tiles = [flatten(base), cond.resize((sz, sz))]
        for sc in args.scales:
            tag = (f"{args.control}_m{args.mask[0]}{args.masked[0]}"
                   f"_s{sc:g}{'' if args.decay == 'none' else '_' + args.decay}")
            print(f"\n[2] 带 {args.control} 控制 scale={sc} "
                  f"mask={args.mask} masked={args.masked} "
                  f"end={args.end} decay={args.decay}", flush=True)
            ctl = run(tag, True, sc)
            b = np.asarray(flatten(ctl), dtype=np.float32)
            print(f"  与对照组平均像素差 {abs(b-a).mean():.2f}/255 "
                  f"（≈0 说明控制没生效）", flush=True)
            tiles.append(flatten(ctl))
        # 并排存一张，方便直接看：[无控制, 条件图, scale1, scale2, ...]
        side = Image.new("RGB", (sz * len(tiles), sz), "white")
        for i, t in enumerate(tiles):
            side.paste(t, (i * sz, 0))
        side.save("assets/_cn_compare.png")
        print(f"\n  → assets/_cn_compare.png （从左到右：无控制 / 条件图 / "
              f"scale={','.join(f'{s:g}' for s in args.scales)}）", flush=True)

    mem("结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
