"""SD1.5 能力实测：逐阶段测量耗时与峰值显存。

覆盖文档 W1（文生图）/ W3（局部重绘）/ W4（姿态驱动 + IP-Adapter 身份锁定）。
每个阶段单独跑，互不复用显存占用，结果写 assets/_bench.json。

用法：
    python tools/smoke.py w1
    python tools/smoke.py w3
    python tools/smoke.py w4 --pose-from assets/_bench_w1.png --ref assets/_bench_ref.png
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from diffusers import (  # noqa: E402
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
BENCH = ASSETS / "_bench.json"

DTYPE = torch.float16
PROMPT = (
    "pixel art game sprite, cute cat knight, red cape, katana, front view, "
    "full body, centered, plain white background, 16-bit, clean hard outline"
)
NEG = "photo, blurry, jpeg artifacts, gradient background, multiple characters, text, watermark"


def detect_variant(root: Path) -> str | None:
    fp16 = Path(root) / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    return "fp16" if fp16.exists() else None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak() -> float:
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


def reset() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def record(stage: str, **kw) -> None:
    data = json.loads(BENCH.read_text("utf-8")) if BENCH.exists() else {}
    data[stage] = kw
    BENCH.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    log(f"{stage}: {json.dumps(kw, ensure_ascii=False)}")


def load_base() -> StableDiffusionPipeline:
    log("loading sd15 ...")
    t0 = time.perf_counter()
    base = MODELS / "checkpoints" / "sd15"
    pipe = StableDiffusionPipeline.from_pretrained(
        base,
        torch_dtype=DTYPE,
        variant=detect_variant(base),
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    log(f"sd15 loaded in {time.perf_counter() - t0:.1f}s")
    return pipe


def stage_w1(steps: int) -> None:
    pipe = load_base()
    gen = torch.Generator("cuda").manual_seed(20261027)
    reset()
    t0 = time.perf_counter()
    img = pipe(
        PROMPT, negative_prompt=NEG, num_inference_steps=steps,
        guidance_scale=7.5, width=512, height=768, generator=gen,
    ).images[0]
    dt = time.perf_counter() - t0
    out = ASSETS / "_bench_w1.png"
    img.save(out)
    record("w1_txt2img_512x768", seconds=round(dt, 2), steps=steps, peak_vram_gb=peak(), out=str(out))

    # 同模型改走 img2img（W2），复用内存中的底座
    i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
    reset()
    t0 = time.perf_counter()
    img2 = i2i(
        PROMPT, image=img.resize((512, 768)), strength=0.55, num_inference_steps=steps,
        guidance_scale=7.0, generator=torch.Generator("cuda").manual_seed(7),
    ).images[0]
    dt = time.perf_counter() - t0
    out2 = ASSETS / "_bench_w2.png"
    img2.save(out2)
    record("w2_img2img_512x768", seconds=round(dt, 2), steps=steps, strength=0.55,
           peak_vram_gb=peak(), out=str(out2))
    del i2i
    reset()
    return img


def stage_w3(steps: int) -> None:
    log("loading sd15-inpainting ...")
    t0 = time.perf_counter()
    ip = MODELS / "checkpoints" / "sd15-inpaint"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        ip,
        torch_dtype=DTYPE, variant=detect_variant(ip),
        safety_checker=None, requires_safety_checker=False, local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    log(f"inpaint loaded in {time.perf_counter() - t0:.1f}s")

    src = Image.open(ASSETS / "_bench_w1.png").convert("RGB")
    w, h = src.size
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(mask)
    d.rectangle([int(w * 0.3), int(h * 0.45), int(w * 0.7), int(h * 0.8)], fill=255)

    reset()
    t0 = time.perf_counter()
    img = pipe(
        "pixel art, silver plate armor with blue trim",
        negative_prompt=NEG, image=src, mask_image=mask, strength=0.9,
        num_inference_steps=steps, guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(3),
    ).images[0]
    dt = time.perf_counter() - t0
    out = ASSETS / "_bench_w3.png"
    img.save(out)
    record("w3_inpaint_512x768", seconds=round(dt, 2), steps=steps, peak_vram_gb=peak(), out=str(out))


def extract_pose(src: Path) -> Image.Image:
    """DWPose 提骨骼（走 controlnet_aux，权重取本地 models/pose_keypoint/dwpose）。"""
    from controlnet_aux import DwposeDetector

    det = DwposeDetector.from_pretrained(str(MODELS / "pose_keypoint" / "dwpose"))
    return det(Image.open(src).convert("RGB"), include_hand=True, include_face=False,
               detect_resolution=512, image_resolution=768)


def stage_w4(steps: int, pose_src: Path, ref: Path | None, ipa_scale: float,
             pose_file: Path | None = None) -> None:
    if pose_file is not None:
        pose = Image.open(pose_file).convert("RGB")
        log(f"use prebuilt pose {pose_file}")
    else:
        pose = extract_pose(pose_src)
    out_pose = ASSETS / "_bench_pose.png"
    pose.save(out_pose)
    log(f"pose extracted -> {out_pose}")

    log("loading controlnet openpose ...")
    cn_pth = MODELS / "controlnet" / "control_v11p_sd15_openpose.pth"
    cn_dir = MODELS / "controlnet" / "openpose"
    if (cn_dir / "config.json").exists():
        cn = ControlNetModel.from_pretrained(
            cn_dir, torch_dtype=DTYPE, variant=detect_variant(cn_dir), local_files_only=True)
    else:
        cn = ControlNetModel.from_single_file(
            str(cn_pth), config=str(MODELS / "controlnet" / "_cfg_openpose"), torch_dtype=DTYPE)
    base = MODELS / "checkpoints" / "sd15"
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base, controlnet=cn, torch_dtype=DTYPE, variant=detect_variant(base),
        safety_checker=None, requires_safety_checker=False, local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    pipe.set_progress_bar_config(disable=True)
    # ref 为 None 时不启用 IP-Adapter（权重未下全时先验 ControlNet 这一段）
    if ref is not None:
        log("loading ip-adapter ...")
        pipe.load_ip_adapter(
            str(MODELS / "ipadapter"), subfolder="models", weight_name="ip-adapter_sd15.bin"
        )
        pipe.set_ip_adapter_scale(ipa_scale)
    pipe = pipe.to("cuda")

    ref_img = Image.open(ref).convert("RGB").resize((512, 768)) if ref else None
    reset()
    t0 = time.perf_counter()
    kwargs = dict(
        prompt="pixel art game sprite, red cape, katana, front view, full body, centered, "
               "plain white background, 16-bit, clean hard outline",
        negative_prompt=NEG,
        image=pose,
        num_inference_steps=steps,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(20261027),
        width=512,
        height=768,
    )
    if ref_img is not None:
        kwargs["ip_adapter_image"] = ref_img
    img = pipe(**kwargs).images[0]
    dt = time.perf_counter() - t0
    out = ASSETS / "_bench_w4.png"
    img.save(out)
    record("w4_pose_512x768", seconds=round(dt, 2), steps=steps, ipadapter=ref_img is not None,
           ipa_scale=ipa_scale if ref_img else None, peak_vram_gb=peak(),
           out=str(out), pose=str(out_pose))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["w1", "w3", "w4"])
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--pose-from", type=Path, default=ASSETS / "_bench_w2.png")
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--pose-file", type=Path, default=None)
    ap.add_argument("--ipa-scale", type=float, default=0.8)
    ap.add_argument("--no-ipa", action="store_true", help="不启用 IP-Adapter（权重未就绪时用）")
    a = ap.parse_args()

    log(f"torch {torch.__version__} cuda {torch.cuda.is_available()} "
        f"free/total {[round(x / 2**30, 1) for x in torch.cuda.mem_get_info()]}")
    if a.stage == "w1":
        stage_w1(a.steps)
    elif a.stage == "w3":
        stage_w3(a.steps)
    else:
        ref = None if a.no_ipa else (a.ref or (ASSETS / "_bench_w1.png"))
        stage_w4(a.steps, a.pose_from, ref, a.ipa_scale, a.pose_file)
    log("done, peak process vram = %s GB" % peak())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
