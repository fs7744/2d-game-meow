"""LoRA 训练：给自己的角色 / 风格训一个 SD1.5 LoRA。

为什么自己写循环而不用 diffusers 的 examples 脚本：
examples 依赖 `datasets` + `accelerate` + 一堆 CLI 参数，Windows 上装起来麻烦，
而且它假设 512×512 的 hub 风格数据集。这里只需要「一个文件夹 + 每张图一个 caption」，
手写循环反而更短、更好排错，也方便把进度回调接到 API。

产出的格式与 `Engine.apply_lora` 一致（diffusers 的 unet_lora_layers 格式），
训完直接在下拉里就能选，不需要转换。

两类用法：
  风格 LoRA —— 20~40 张同风格图 + 同一句 caption（dataset.caption 兜底）
  角色 LoRA —— 15~30 张同角色不同姿势 + 每张一个 .txt

显存（RTX 4090 24GB，fp32 UNet + gradient checkpointing，512 分辨率 batch 1）：
实测峰值约 6~8GB，够跑；耗时约 0.3~0.5s/step。
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

SD15 = "checkpoints/sd15"

# SD1.5 UNet 里值得挂 LoRA 的线性层（attention 的 q/k/v/out）
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Dataset:
    """文件夹数据集：每张图配一个同名 .txt；没有就退回统一 caption。"""

    def __init__(self, root: str | Path, resolution: int = 512,
                 caption: str = "", center_crop: bool = False) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据集目录不存在: {self.root}")

        self.items: List[tuple[Path, str]] = []
        for f in sorted(self.root.rglob("*")):
            if f.suffix.lower() not in IMG_EXT:
                continue
            txt = f.with_suffix(".txt")
            cap = txt.read_text(encoding="utf-8").strip() if txt.exists() else caption.strip()
            if not cap:
                continue                       # 没 caption 的图训出来是噪声，直接跳过
            self.items.append((f, cap))

        if not self.items:
            raise ValueError(
                f"{root} 里没有可用样本：需要图片 + 同名 .txt，或给一个统一 caption")
        self.resolution = resolution
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.items)

    def tensor(self, i: int) -> tuple[torch.Tensor, str]:
        path, cap = self.items[i]
        im = Image.open(path).convert("RGB")

        # 等比缩放 + 居中裁切（比直接 resize 更能保住角色比例）
        r = self.resolution
        w, h = im.size
        scale = r / min(w, h)
        im = im.resize((max(r, int(round(w * scale))), max(r, int(round(h * scale)))),
                       Image.LANCZOS)
        w, h = im.size
        if self.center_crop:
            left, top = (w - r) // 2, (h - r) // 2
            im = im.crop((left, top, left + r, top + r))
        else:
            im = im.resize((r, r), Image.LANCZOS)

        arr = torch.from_numpy(
            __import__("numpy").asarray(im, dtype="float32") / 127.5 - 1.0
        ).permute(2, 0, 1)
        return arr, cap


def _encode_prompt(tokenizer, text_encoder, prompts: Sequence[str], device, dtype):
    tok = tokenizer(
        list(prompts), padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        out = text_encoder(tok.input_ids.to(device))[0]
    return out.to(dtype=dtype)


def train(
    models_root: str | Path,
    data_dir: str | Path,
    output: str,
    *,
    caption: str = "",
    rank: int = 8,
    alpha: Optional[int] = None,
    lr: float = 1e-4,
    steps: int = 400,
    batch_size: int = 1,
    resolution: int = 512,
    seed: int = 42,
    save_every: int = 0,
    center_crop: bool = False,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """训练一个 LoRA 并保存到 models/lora/<output>.safetensors。"""
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, \
        UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    root = Path(models_root)
    base = root / SD15
    if not (base / "unet").exists():
        raise FileNotFoundError(f"底模缺失: {base / 'unet'}")

    alpha = alpha or rank
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Dataset(data_dir, resolution=resolution, caption=caption, center_crop=center_crop)

    def emit(**kw: Any) -> None:
        if progress:
            progress(kw)

    emit(stage="loading", samples=len(ds))

    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    # 训练用 fp32：fp16 的 UNet 在无 loss scaling 时梯度会溢出，LoRA 学不到东西
    vae.to(device, dtype=torch.float32).requires_grad_(False).eval()
    text_encoder.to(device, dtype=torch.float32).requires_grad_(False).eval()
    unet.to(device, dtype=torch.float32)
    unet.requires_grad_(False)

    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
        target_modules=TARGET_MODULES,
    ))
    unet.enable_gradient_checkpointing()

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in lora_params)
    opt = torch.optim.AdamW(lora_params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)

    emit(stage="train", total=steps, trainable=n_train, rank=rank, alpha=alpha)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    t0 = time.perf_counter()
    losses: List[float] = []
    order: List[int] = []

    unet.train()
    for step in range(1, steps + 1):
        if not order:
            order = list(range(len(ds)))
            random.shuffle(order)

        batch = [ds.tensor(order.pop()) for _ in range(min(batch_size, len(ds)))]
        imgs = torch.stack([b[0] for b in batch]).to(device)
        caps = [b[1] for b in batch]

        with torch.no_grad():
            latents = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        ts = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                           device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, ts)

        ctx = _encode_prompt(tokenizer, text_encoder, caps, device, torch.float32)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pred = unet(noisy, ts, encoder_hidden_states=ctx).sample
            # SD1.5 用 epsilon 预测（v-prediction 的模型另算，这里不处理）
            loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        scaler.step(opt)
        scaler.update()

        losses.append(float(loss.detach()))
        if step % 10 == 0 or step == steps:
            recent = losses[-20:]
            emit(stage="train", step=step, total=steps,
                 loss=round(sum(recent) / len(recent), 5),
                 elapsed=round(time.perf_counter() - t0, 1),
                 peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2)
                 if device == "cuda" else None)

        if save_every and step % save_every == 0:
            _save(models_root, output, tokenizer, text_encoder, vae, unet, noise_scheduler,
                  suffix=f"-step{step}")
            emit(stage="checkpoint", step=step)

    path = _save(models_root, output, tokenizer, text_encoder, vae, unet, noise_scheduler)
    info = {
        "stage": "done", "path": str(path), "samples": len(ds), "steps": steps,
        "rank": rank, "alpha": alpha, "lr": lr, "resolution": resolution,
        "trainable_params": n_train,
        "final_loss": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 5),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    emit(**info)
    return info


def _save(models_root, output, tokenizer, text_encoder, vae, unet, noise_scheduler,
          suffix: str = "") -> Path:
    """导出为 models/lora/<output><suffix>.safetensors。

    注意 save_lora_weights 的入参是**目录**，且文件名写死为
    pytorch_lora_weights.safetensors。直接传文件路径会得到一个同名目录，
    产物就"消失"了（实测踩过）。这里先落临时目录再改成扁平文件名，
    这样 Engine.apply_lora(按文件名热切换) 与前端下拉都能直接认出它。
    """
    import shutil

    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out_dir = Path(models_root) / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=noise_scheduler, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False,
    )
    tmp = out_dir / f".tmp_{output}{suffix}"
    if tmp.exists():
        shutil.rmtree(tmp)
    # 必须只取 LoRA 那一层：直接传 unet 会把整个 UNet（fp32，3.3GB）存进去，
    # 而且格式不对、加载回来并不生效。get_peft_model_state_dict 只拿 lora_A/B，
    # 再用 convert_state_dict_to_diffusers 把 peft 前缀去掉。
    lora_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    pipe.save_lora_weights(str(tmp), unet_lora_layers=lora_sd, safe_serialization=True)

    src = tmp / "pytorch_lora_weights.safetensors"
    dst = out_dir / f"{output}{suffix}.safetensors"
    shutil.move(str(src), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    return dst


def plan_steps(samples: int, epochs: int = 8, batch_size: int = 1) -> int:
    """按样本数估算步数：小数据集多跑几轮，但设上限避免过拟合。"""
    per_epoch = max(1, math.ceil(samples / max(1, batch_size)))
    return max(100, min(2000, per_epoch * epochs))
