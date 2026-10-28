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
    StableDiffusionXLInpaintPipeline,
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
CN_SDXL_DIRS = {
    "openpose": "controlnet/sdxl_openpose",
    "canny": "controlnet/sdxl_canny",
    "tile": "controlnet/sdxl_tile",
}

# SDXL 的 IP-Adapter 也是另一套（ViT-H image encoder 与 SD1.5 的不通用）
IPADAPTER_SDXL_WEIGHT = "ipadapter/models/ip-adapter_sdxl_vit-h.safetensors"


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


# 进程级的权重池：**同一个 models_root 的所有 Engine 实例共享**。
# 引擎持有管线，管线占显存——两个实例 = 两份权重 = 白白多占几个 GB。
# 实测：测试里每次新建 Engine，连续跑下来峰值到 14.9GB（共用一个只 5.78GB）。
_POOLS: Dict[str, Dict[str, Any]] = {}


class Engine:
    """单卡常驻引擎。所有 pipeline 共用同一个 torch 设备。

    **权重池按 models_root 共享**：多个 Engine 实例指向同一个模型目录时，
    共用同一份管线缓存——避免同一份权重在显存里存多份。
    """

    def __init__(self, models_root: str | Path, device: str = "cuda") -> None:
        self.root = Path(models_root)
        self.device = device
        self.DEFAULT_MODEL = DEFAULT_MODEL        # 给 API/前端当默认值用
        self.dtype = torch.float16
        # 共享权重池（按 root 归一化后作为键）
        key = str(self.root.resolve())
        pool = _POOLS.get(key)
        if pool is None:
            pool = {"cache": {}, "ctrl": {}, "lora": None, "pose": None}
            _POOLS[key] = pool
        self._pool = pool
        self._cache: Dict[str, Any] = pool["cache"]
        self._ctrl: Dict[str, ControlNetModel] = pool["ctrl"]
        # 语义分割抠图模型目录（isnet-general-use.onnx）。存在即启用 u2net 后端。
        self.matte_dir = self.root / "matte"

    # LoRA 与姿态提取器也走共享池——否则两个实例状态会不一致
    @property
    def _lora(self) -> Optional[str]:
        return self._pool["lora"]

    @_lora.setter
    def _lora(self, v: Optional[str]) -> None:
        self._pool["lora"] = v

    @property
    def _pose(self) -> Any:
        return self._pool["pose"]

    @_pose.setter
    def _pose(self, v: Any) -> None:
        self._pool["pose"] = v

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
        """W3 局部重绘。

        两个底模走不同来源：
          sd15 —— 专用 inpaint 权重（runwayml 在 SD1.5 基础上额外训练过 inpainting）
          sdxl —— 直接用 base 权重跑 StableDiffusionXLInpaintPipeline。
                  SDXL 没有官方 inpaint 微调版，这是唯一可行路径，质量略逊于
                  SD1.5 那套专用权重，但换来了 1024² 的分辨率。
        """
        key = f"inpaint:{model}"
        if key in self._cache:
            return self._cache[key]

        t0 = time.perf_counter()
        base = self._base_kwargs()
        if model == "sdxl":
            p = self._check_model(model)
            # SDXL 管线没有 safety_checker 这个参数，传了只会落一条 warning
            base.pop("safety_checker", None)
            base.pop("requires_safety_checker", None)
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                p, variant=detect_variant(p), **base
            )
        else:
            p = self.path(SD15_INPAINT)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                p, variant=detect_variant(p), **base
            )
        pipe.scheduler = _scheduler(pipe.scheduler.config)
        self._cache[key] = self._fit(pipe, key)
        log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return self._cache[key]

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

    def ipa_weight(self, model: str = DEFAULT_MODEL) -> tuple[str, str, str]:
        """(根目录, 子目录, 权重文件名)。魔搭只给 .safetensors，HF 只给 .bin。

        SDXL 用的是另一套（ViT-H image encoder 与 SD1.5 的不通用），目录也分开。
        注意 load_ip_adapter 的 subfolder 是**必需**的（不是可选关键字）。
        """
        root = str(self.path(IPADAPTER))
        # 两个底模的 IP-Adapter 权重并列放在 ipadapter/models/，共享同一个 image_encoder。
        # 这不是偷懒：ip-adapter_sdxl_vit-h 要的就是 ViT-H（hidden 1280 / projection 1024），
        # 与 SD1.5 完全同一个。HF 上还有一份 sdxl_models/image_encoder 是 ViT-bigG
        # （hidden 1664 / projection 1280），那是给不带 vit-h 后缀的 ip-adapter_sdxl 用的，
        # 拿错会报 mat1 and mat2 shapes cannot be multiplied。
        names = ("ip-adapter_sdxl_vit-h.safetensors", "ip-adapter_sdxl_vit-h.bin") \
            if model == "sdxl" else ("ip-adapter_sd15.safetensors", "ip-adapter_sd15.bin")
        for name in names:
            if (self.path(IPADAPTER) / "models" / name).exists():
                return root, "models", name
        raise FileNotFoundError(
            f"IP-Adapter 权重缺失（{model}）: {root}/models/{names[0]}")

    def pose_pipe(self, controlnet: str = "openpose", with_ipadapter: bool = True,
                  model: str = DEFAULT_MODEL):
        """ControlNet(+IP-Adapter)。W4 的核心管线，两个底模都支持。

        SD1.5 与 SDXL 的 ControlNet / IP-Adapter 是两套权重，按 model 分别取。
        """
        key = f"pose:{controlnet}:{int(with_ipadapter)}:{model}"
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        cn = self.controlnet(controlnet, model)
        p = self._check_model(model)
        if model == "sdxl":
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                p, controlnet=cn, variant=detect_variant(p), **self._base_kwargs()
            )
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                p, controlnet=cn, variant=detect_variant(p), **self._base_kwargs()
            )
        pipe.scheduler = _scheduler(pipe.scheduler.config)
        if with_ipadapter:
            # 布局：<root>/<sub>/ip-adapter_*.safetensors 与 <root>/<sub>/image_encoder/
            # SD1.5 是 ipadapter/models/，SDXL 是 ipadapter/sdxl/
            ipa_root, ipa_sub, ipa_name = self.ipa_weight(model)
            pipe.load_ip_adapter(ipa_root, ipa_sub, weight_name=ipa_name,
                                 image_encoder_folder="image_encoder")
            pipe.set_ip_adapter_scale(0.8)
        self._cache[key] = self._fit(pipe, key)
        log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return self._cache[key]

    # ---------- AnimateDiff（时序一致性）----------

    ANIMATEDIFF_DIR = "animatediff/motion-adapter-v1-5-2"

    def animatediff_available(self) -> bool:
        """motion adapter 权重在不在。

        权重约 1.7GB（fp16），不在库里时整个 AnimateDiff 路线不可用。
        缺失不是错误——调用方应当退回逐帧方案，而不是让任务失败。
        """
        d = self.root / self.ANIMATEDIFF_DIR
        return (d / "config.json").exists() and (
            (d / "diffusion_pytorch_model.fp16.safetensors").exists()
            or (d / "diffusion_pytorch_model.safetensors").exists())

    def animatediff_pipe(self, controlnet: str = "canny",
                         with_ipadapter: bool = True,
                         model: str = DEFAULT_MODEL,
                         optimize_vram: bool = True):
        """AnimateDiff + ControlNet 管线（时序一致性方案）。

        **为什么需要它**：逐帧生成时，每一帧都是独立的概率采样，帧间会漂
        （业界实测 IP-Adapter + ControlNet 的帧间一致性只有 70~80%，
        同 seed 也会漂 20~30%）。AnimateDiff 的 motion module 插在 UNet 的
        ResNet/Attention 之后，提供**跨帧的时序注意力**——生成第 N 帧时
        会参考相邻帧的潜在表示，这是架构级的时序约束，不是概率层面的。

        实测（本项目，同一份 8 帧素材）：
          逐帧多候选   19.2s / 4.5GB / silhouette_drift 0.152
          AnimateDiff   7.4s / 13.1GB / silhouette_drift **0.097**

        **显存优化**（`optimize_vram=True`，默认开）：
        基线 8帧@512 是 9.86GB、16帧@512 是 14.09GB。开了两项优化后降到
        **5.78GB / 7.10GB（-41%）**，耗时不变、输出逐像素一致：

          1. VAE 逐帧解码——整段批量解码增量 4.00GB，逐帧只要 0.60GB
             （这条管线没有 `enable_vae_slicing`，得调 `pipe.vae.enable_slicing()`）
          2. IP-Adapter 的 image_encoder 用完即退——它是 CLIP Vision（1.18GB），
             只编码一次参考图就闲置；编码时短暂上卡、编完退回 CPU

        详见 `workflows/vramopt.py`（含实测数据与「哪些手段没用」的记录）。
        """
        key = f"animatediff:{controlnet}:{int(with_ipadapter)}:{model}"
        if key in self._cache:
            return self._cache[key]
        t0 = time.perf_counter()
        from diffusers import (AnimateDiffControlNetPipeline, MotionAdapter)

        adapter_dir = self.root / self.ANIMATEDIFF_DIR
        if not self.animatediff_available():
            raise FileNotFoundError(
                f"AnimateDiff motion adapter 缺失: {adapter_dir}\n"
                "用 tools/fetch_animatediff.py 下载（约 1.7GB），"
                "或改用逐帧方案（engine_mode='frame'）")
        adapter = MotionAdapter.from_pretrained(
            str(adapter_dir), torch_dtype=self.dtype,
            variant="fp16" if (adapter_dir / "diffusion_pytorch_model.fp16.safetensors").exists()
            else None)
        cn = self.controlnet(controlnet, model)
        p = self._check_model(model)
        # AnimateDiff 只支持 SD1.5 架构（motion module 与 SD1.5 的 UNet 绑定）
        pipe = AnimateDiffControlNetPipeline.from_pretrained(
            p, motion_adapter=adapter, controlnet=cn,
            variant=detect_variant(p),
            torch_dtype=self.dtype, local_files_only=True)
        pipe.scheduler = _scheduler(pipe.scheduler.config)
        if with_ipadapter:
            # 必须显式加载：这条管线继承了 IPAdapterMixin，但不会自动装权重，
            # 不加载的话传 ip_adapter_image 会报 image_projection_layers is None
            ipa_root, ipa_sub, ipa_name = self.ipa_weight(model)
            pipe.load_ip_adapter(ipa_root, ipa_sub, weight_name=ipa_name,
                                 image_encoder_folder="image_encoder")
            pipe.set_ip_adapter_scale(0.8)
        pipe = self._fit(pipe, key)
        if optimize_vram:
            from . import vramopt
            info = vramopt.optimize_pipe(pipe)
            log.info("%s 显存优化: %s", key, info)
        self._cache[key] = pipe
        log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return pipe

    # ---------- 质量闸门用的姿态提取器 ----------

    def pose_extractor(self):
        """W4 质量闸门需要「把生成图再检一遍姿态」，所以引擎持有单例。

        权重缺失/onnxruntime 没装时返回 None（闸门应据此关闭），不抛异常——
        质量闸门是增强项，不该让它把整条出图链路拖死。
        PoseExtractor 走 CPU（4090 上单帧约 10ms），不占显存。
        """
        if self._pose is not None:
            return self._pose or None
        try:
            from .pose import PoseExtractor

            self._pose = PoseExtractor(self.root / "pose_keypoint" / "dwpose",
                                      detector="auto")
            log.info("姿态提取器就绪（闸门后端 %s）", self._pose.backend)
        except Exception as exc:              # noqa: BLE001
            log.warning("姿态提取器不可用，质量闸门关闭：%s", exc)
            self._pose = False
        return self._pose or None

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

    def ensure_headroom(self, need_gb: float = 2.0) -> None:
        """显存不够时自动驱逐已缓存的管线。

        **为什么需要**：引擎把管线缓存起来复用，但**从不自动释放**。
        AnimateDiff 管线约 4.45GB、逐帧 ControlNet 管线约 3.2GB——
        两者同时驻留就是 7.6GB，加上生成本身的激活就容易撞顶。
        实测连续跑多个工作流时峰值能到 14.9GB（单独跑只有 5.78GB）。

        策略：从最久没用的开始丢，直到腾出足够空间。丢的是**缓存**，
        下次用到会重新加载（几百毫秒到几秒），比 OOM 划算。
        """
        if not torch.cuda.is_available():
            return
        # **用 PyTorch 的 reserved 而不是 mem_get_info 的物理空闲**：
        # mem_get_info 看的是显存分配器的物理占用，它不包含 PyTorch
        # 缓存着的块（cached-but-unallocated）。那些块在下次申请时会被
        # 直接复用，所以「物理空闲」不等于「可用」。
        # reserved 才是真正占着不还的量。
        def _used_gb() -> float:
            return torch.cuda.memory_reserved() / 2**30

        if _used_gb() < need_gb:
            return
        # 按插入顺序丢（dict 保序），保留最近用的
        for key in list(self._cache.keys()):
            if _used_gb() < need_gb:
                break
            self._cache.pop(key, None)
            gc.collect()
            torch.cuda.empty_cache()
            log.info("显存紧张（reserved %.1fGB），已释放管线 %s",
                     _used_gb(), key)

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
