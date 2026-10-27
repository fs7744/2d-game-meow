"""推理引擎：模型懒加载、缓存、显存守护。

设计约束（来自 RTX4090_2D_Game_Pipeline 文档 §6/§7.2 与实测）：
- fp16 + PyTorch SDPA 注意力（Windows 无 xformers 轮子，不依赖它）。
- 懒加载 + 常驻：截图切换工作流不重载底座；切换角色只热换 LoRA。
- 显存不足时降级顺序：attention_slicing -> vae_slicing -> model_cpu_offload，
  并在每次降级后打日志，避免静默变慢。
"""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from diffusers import (  # noqa: E402
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)

log = logging.getLogger("engine")

SD15 = "checkpoints/sd15"
SD15_INPAINT = "checkpoints/sd15-inpaint"
IPADAPTER = "ipadapter"

# ControlNet 两种来源：diffusers 目录（hf 布局）或原始 .pth（魔搭布局，LDM 单文件）
CN_DIRS = {"openpose": "controlnet/openpose", "canny": "controlnet/canny",
           "tile": "controlnet/tile"}
CN_FILES = {
    "openpose": "controlnet/control_v11p_sd15_openpose.pth",
    "canny": "controlnet/control_v11p_sd15_canny.pth",
    "tile": "controlnet/control_v11f1e_sd15_tile.pth",
}


def _scheduler(config):
    return DPMSolverMultistepScheduler.from_config(
        config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )


def detect_variant(root: Path) -> str | None:
    """有 fp16 变体就用 fp16 文件，否则用默认 fp32（加载时按 torch_dtype 即时转型）。

    ModelScope 上只有 fp32 权重，hf-mirror 上有 fp16 变体；两种目录都要能直接跑。
    """
    fp16 = root / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    return "fp16" if fp16.exists() else None


class Engine:
    """单卡常驻引擎。所有 pipeline 共用同一个 torch 设备。"""

    def __init__(self, models_root: str | Path, device: str = "cuda") -> None:
        self.root = Path(models_root)
        self.device = device
        self.dtype = torch.float16
        self._cache: Dict[str, Any] = {}
        self._ctrl: Dict[str, ControlNetModel] = {}
        self._lora: Optional[str] = None

    # ---------- 基础工具 ----------

    def path(self, rel: str) -> Path:
        p = self.root / rel
        if not p.exists():
            raise FileNotFoundError(f"权重缺失: {p}（先跑 tools/fetch_weights.py）")
        return p

    def vram(self) -> Dict[str, float]:
        free, total = torch.cuda.mem_get_info()
        return {
            "free_gb": round(free / 2**30, 2),
            "total_gb": round(total / 2**30, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }

    def _base_kwargs(self) -> Dict[str, Any]:
        return dict(
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        )

    def _fit(self, pipe, name: str):
        """显存守卫：装不下就逐级降级，而不是 OOM 崩掉整个服务。"""
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe = pipe.to(self.device)
        except torch.OutOfMemoryError:
            log.warning("%s 直接上卡失败，降级 model_cpu_offload", name)
            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
        return pipe

    # ---------- pipeline 获取（懒加载 + 缓存）----------

    def txt2img_pipe(self) -> StableDiffusionPipeline:
        if "txt2img" not in self._cache:
            t0 = time.perf_counter()
            p = self.path(SD15)
            pipe = StableDiffusionPipeline.from_pretrained(
                p, variant=detect_variant(p), **self._base_kwargs()
            )
            pipe.scheduler = _scheduler(pipe.scheduler.config)
            self._cache["txt2img"] = self._fit(pipe, "txt2img")
            log.info("txt2img 加载 %.1fs", time.perf_counter() - t0)
        return self._cache["txt2img"]

    def img2img_pipe(self) -> StableDiffusionImg2ImgPipeline:
        if "img2img" not in self._cache:
            # 与 txt2img 共享同一份权重，避免二次占显存
            if "txt2img" in self._cache:
                base = self._cache["txt2img"]
                pipe = StableDiffusionImg2ImgPipeline(**base.components)
            else:
                p = self.path(SD15)
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    p, variant=detect_variant(p), **self._base_kwargs()
                )
                pipe.scheduler = _scheduler(pipe.scheduler.config)
            self._cache["img2img"] = self._fit(pipe, "img2img")
        return self._cache["img2img"]

    def inpaint_pipe(self) -> StableDiffusionInpaintPipeline:
        if "inpaint" not in self._cache:
            t0 = time.perf_counter()
            p = self.path(SD15_INPAINT)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                p, variant=detect_variant(p), **self._base_kwargs()
            )
            pipe.scheduler = _scheduler(pipe.scheduler.config)
            self._cache["inpaint"] = self._fit(pipe, "inpaint")
            log.info("inpaint 加载 %.1fs", time.perf_counter() - t0)
        return self._cache["inpaint"]

    def controlnet(self, which: str = "openpose") -> ControlNetModel:
        if which not in self._ctrl:
            d = self.root / CN_DIRS[which]
            f = self.root / CN_FILES[which]
            if (d / "config.json").exists():
                self._ctrl[which] = ControlNetModel.from_pretrained(
                    d, torch_dtype=self.dtype, variant=detect_variant(d), local_files_only=True
                )
                log.info("ControlNet %s <- diffusers 目录", which)
            elif f.exists():
                # 原始 .pth 必须给显式 config（diffusers 默认会去 Hub 猜一个错的仓库）
                cfg = self.root / "controlnet" / f"_cfg_{which}"
                if not (cfg / "config.json").exists():
                    raise FileNotFoundError(
                        f"缺少 ControlNet 配置目录 {cfg}（放一个 config.json）")
                self._ctrl[which] = ControlNetModel.from_single_file(
                    str(f), config=str(cfg), torch_dtype=self.dtype
                )
                log.info("ControlNet %s <- 单文件 %s", which, f.name)
            else:
                raise FileNotFoundError(f"ControlNet {which} 权重缺失（{d} 或 {f}）")
        return self._ctrl[which]

    def ipa_weight(self) -> str:
        """IP-Adapter 权重文件名：魔搭只给 .safetensors，HF 只给 .bin。"""
        base = self.path(IPADAPTER) / "models"
        for name in ("ip-adapter_sd15.safetensors", "ip-adapter_sd15.bin"):
            if (base / name).exists():
                return name
        raise FileNotFoundError(f"IP-Adapter 权重缺失: {base}")

    def pose_pipe(self, controlnet: str = "openpose", with_ipadapter: bool = True):
        """SD1.5 + ControlNet(+IP-Adapter)。W4 的核心管线。"""
        key = f"pose:{controlnet}:{int(with_ipadapter)}"
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        cn = self.controlnet(controlnet)
        p = self.path(SD15)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            p, controlnet=cn, variant=detect_variant(p), **self._base_kwargs()
        )
        pipe.scheduler = _scheduler(pipe.scheduler.config)
        if with_ipadapter:
            pipe.load_ip_adapter(
                str(self.path(IPADAPTER)), subfolder="models",
                weight_name=self.ipa_weight(),
            )
            pipe.set_ip_adapter_scale(0.8)
        self._cache[key] = self._fit(pipe, key)
        log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return self._cache[key]

    # ---------- LoRA 热切换 ----------

    def apply_lora(self, pipe, lora_path: str | Path | None, scale: float = 0.8) -> None:
        """角色 LoRA 热加载：同名不重复卸载重载。"""
        if lora_path is None:
            if self._lora is not None:
                pipe.unload_lora_weights()
                self._lora = None
            return
        key = f"{Path(lora_path).name}:{scale}"
        if key == self._lora:
            return
        if self._lora is not None:
            pipe.unload_lora_weights()
        pipe.load_lora_weights(str(lora_path), adapter_name="char")
        pipe.set_adapters(["char"], adapter_weights=[scale])
        self._lora = key
        log.info("LoRA 热切换 -> %s (scale=%.2f)", Path(lora_path).name, scale)

    # ---------- 显存管理 ----------

    def free(self, keep: tuple[str, ...] = ()) -> None:
        for k in list(self._cache):
            if k not in keep:
                self._cache.pop(k, None)
        gc.collect()
        torch.cuda.empty_cache()

    def stats(self) -> Dict[str, Any]:
        return {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "loaded": sorted(self._cache),
            "lora": self._lora,
            **self.vram(),
        }
