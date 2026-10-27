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
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

log = logging.getLogger("engine")

SD15 = "checkpoints/sd15"
SDXL = "checkpoints/sdxl"
SD15_INPAINT = "checkpoints/sd15-inpaint"
IPADAPTER = "ipadapter"

# 可自选的底模。SDXL 画质更好但：慢约 3×、显存约 2×、且 SD1.5 的 LoRA 不能通用。
BASE_MODELS = {"sd15": SD15, "sdxl": SDXL}
DEFAULT_MODEL = "sd15"

# ControlNet 两种来源：diffusers 目录（hf 布局）或原始 .pth（魔搭布局，LDM 单文件）
CN_DIRS = {"openpose": "controlnet/openpose", "canny": "controlnet/canny",
           "tile": "controlnet/tile"}
CN_FILES = {
    "openpose": "controlnet/control_v11p_sd15_openpose.pth",
    "canny": "controlnet/control_v11p_sd15_canny.pth",
    "tile": "controlnet/control_v11f1e_sd15_tile.pth",
}

# SDXL 的 ControlNet 是另一套权重（shape 不同，混用会直接报错）
CN_SDXL_DIRS = {"openpose": "controlnet/sdxl_openpose"}


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
        self.DEFAULT_MODEL = DEFAULT_MODEL        # 给 API/前端当默认值用
        self.dtype = torch.float16
        self._cache: Dict[str, Any] = {}
        self._ctrl: Dict[str, ControlNetModel] = {}
        self._lora: Optional[str] = None
        # 语义分割抠图模型目录（isnet-general-use.onnx）。存在即启用 u2net 后端。
        self.matte_dir = self.root / "matte"

    # ---------- 底模 ----------

    def available_models(self) -> list[str]:
        """本机已经下好的底模。"""
        return [k for k, rel in BASE_MODELS.items()
                if (self.root / rel / "model_index.json").exists()]

    def _check_model(self, model: str) -> Path:
        rel = BASE_MODELS.get(model)
        if rel is None:
            raise ValueError(f"未知底模 {model}（可选 {', '.join(BASE_MODELS)}）")
        p = self.root / rel
        if not (p / "model_index.json").exists():
            raise FileNotFoundError(f"权重缺失: {p}（见 README「模型」）")
        return p

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

    def txt2img_pipe(self, model: str = DEFAULT_MODEL):
        key = f"txt2img:{model}"
        if key not in self._cache:
            t0 = time.perf_counter()
            p = self._check_model(model)
            cls = StableDiffusionXLPipeline if model == "sdxl" else StableDiffusionPipeline
            pipe = cls.from_pretrained(p, variant=detect_variant(p), **self._base_kwargs())
            pipe.scheduler = _scheduler(pipe.scheduler.config)
            self._cache[key] = self._fit(pipe, key)
            log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return self._cache[key]

    def img2img_pipe(self, model: str = DEFAULT_MODEL):
        key = f"img2img:{model}"
        if key not in self._cache:
            tkey = f"txt2img:{model}"
            if tkey in self._cache:            # 共享权重，避免二次占显存
                cls = StableDiffusionXLImg2ImgPipeline if model == "sdxl" \
                    else StableDiffusionImg2ImgPipeline
                pipe = cls(**self._cache[tkey].components)
            else:
                p = self._check_model(model)
                cls = StableDiffusionXLImg2ImgPipeline if model == "sdxl" \
                    else StableDiffusionImg2ImgPipeline
                pipe = cls.from_pretrained(p, variant=detect_variant(p), **self._base_kwargs())
                pipe.scheduler = _scheduler(pipe.scheduler.config)
            self._cache[key] = self._fit(pipe, key)
        return self._cache[key]

    def inpaint_pipe(self, model: str = DEFAULT_MODEL):
        """W3 局部重绘。只有 SD1.5 有专用 inpaint 权重（SDXL 的 inpaint 质量更差）。"""
        if model != "sd15":
            raise ValueError(f"W3 局部重绘目前只支持 SD1.5，收到 {model}")
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

    def controlnet(self, which: str = "openpose", model: str = DEFAULT_MODEL) -> ControlNetModel:
        key = f"{which}:{model}"
        if key in self._ctrl:
            return self._ctrl[key]
        if model == "sdxl":
            d = self.root / CN_SDXL_DIRS.get(which, "")
            if not (d / "config.json").exists():
                raise FileNotFoundError(
                    f"SDXL 版 ControlNet {which} 权重缺失: {d}（SD1.5 与 SDXL 的 ControlNet 不能混用）")
            self._ctrl[key] = ControlNetModel.from_pretrained(
                d, torch_dtype=self.dtype, variant=detect_variant(d), local_files_only=True)
            log.info("ControlNet %s (sdxl) <- %s", which, d.name)
            return self._ctrl[key]
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

    def pose_pipe(self, controlnet: str = "openpose", with_ipadapter: bool = True,
                  model: str = DEFAULT_MODEL):
        """SD1.5 + ControlNet(+IP-Adapter)。W4 的核心管线。

        只支持 SD1.5：我们的 ControlNet OpenPose 权重就是 SD1.5 版，SDXL 需要另一套
        （xinsir/controlnet-openpose-sdxl）。与其悄悄跑错，不如显式拒绝。
        """
        key = f"pose:{controlnet}:{int(with_ipadapter)}:{model}"
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        cn = self.controlnet(controlnet, model)
        p = self._check_model(model)
        if model == "sdxl":
            if with_ipadapter:
                raise ValueError("SDXL 的 IP-Adapter 权重未装，请用 with_ipadapter=False")
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                p, controlnet=cn, variant=detect_variant(p), **self._base_kwargs()
            )
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                p, controlnet=cn, variant=detect_variant(p), **self._base_kwargs()
            )
        pipe.scheduler = _scheduler(pipe.scheduler.config)
        if with_ipadapter and model == "sd15":
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
        try:
            pipe.load_lora_weights(str(lora_path), adapter_name="char")
            pipe.set_adapters(["char"], adapter_weights=[scale])
        except Exception as exc:            # noqa: BLE001
            # 常见原因：这不是标准 LoRA，而是 LoKr / LoHa / LyCORIS 变体
            # （实测「水墨风格 ink wash」就是 LoKr，key 叫 lokr_w1）——
            # diffusers 的 load_lora_weights 认不出来，报的是一堆 key 重命名失败，
            # 用户根本看不懂。这里翻成人话并把权重卸干净，别让脏状态留在管道里。
            self._lora = None
            raise ValueError(
                f"LoRA 加载失败: {Path(lora_path).name} —— {exc}\n"
                "大概率不是标准 LoRA（LoKr/LoHa/LyCORIS 变体暂不支持），"
                "请换一个标准 LoRA（权重 key 形如 lora_unet_*_lora_A.weight）。"
            ) from exc
        self._lora = key
        log.info("LoRA 热切换 -> %s (scale=%.2f)", Path(lora_path).name, scale)

    # ---------- 显存管理 ----------

    def free(self, keep: tuple[str, ...] = ()) -> None:
        """释放显存。

        ControlNet 是独立缓存（self._ctrl），容易漏——实测只清 _cache 时
        仍残留 4.16GB。keep 里出现某条 ControlNet 名（openpose/canny/tile）才保留。
        """
        for k in list(self._cache):
            if k not in keep:
                self._cache.pop(k, None)
        for name in list(self._ctrl):
            if not any(name in item for item in keep):
                self._ctrl.pop(name, None)
        if not keep:
            self._lora = None
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
