"""推理引擎：模型懒加载、缓存、显存守护。

设计约束（来自 RTX4090_2D_Game_Pipeline 文档 §6/§7.2 与实测）：
- fp16 + PyTorch SDPA 注意力（Windows 无 xformers 轮子，不依赖它）。
- 懒加载 + 常驻：截图切换工作流不重载底座；切换角色只热换 LoRA。
- 显存不足时降级顺序：attention_slicing -> vae_slicing -> model_cpu_offload，
  并在每次降级后打日志，避免静默变慢。
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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
# **"sd15" / "sdxl" 只是两个约定的名字**，不是能力上限：放到
# models/checkpoints/<任意名字>/ 且带 model_index.json 的 UNet 管线会在
# Engine 初始化时自动登记进来（架构按 model_index 的 _class_name 判定），
# 所以"换一个社区 SD1.5 微调"不需要改代码，只要换目录。
BASE_MODELS = {"sd15": SD15, "sdxl": SDXL}
DEFAULT_MODEL = "sd15"

# LoRA 应用状态挂在 UNet 上的属性名。**必须挂在 unet 而不是 Engine**：
# pose 管线复用同一份 UNet 后，两条管线共享的是同一个 unet 对象，
# 状态跟着权重走才不会出现「A 管线加载过、B 管线以为自己也加载了」。
LORA_ATTR = "_meow_lora_key"

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


def _norm_lora_items(lora_path, scale) -> list[tuple[Path, float]]:
    """把「单个 / 列表」统一成 [(路径, 强度)]。

    强度给单个数值表示所有 LoRA 共用；给列表但不够长时按 0.8 补齐
    （LoRA 是增强项，长度不齐不该让整个任务失败）。
    """
    if lora_path is None:
        return []
    if isinstance(lora_path, (str, Path)):
        paths = [lora_path] if str(lora_path) else []
    else:
        paths = [x for x in lora_path if x]
    if not paths:
        return []
    if scale is None or isinstance(scale, (int, float)):
        scales = [float(scale) if scale else 0.8] * len(paths)
    else:
        scales = [float(x) if x else 0.8 for x in scale]
        while len(scales) < len(paths):
            scales.append(0.8)
    return [(Path(p), float(s)) for p, s in zip(paths, scales)]


def _lora_load_error(name: str, msg: str) -> str:
    """把 diffusers/PEFT 的报错翻成人话。**先分辨三种成因**，处置完全不同：

    - 模块结构对不上（DiT / Flux / anima）→ 换多少个 LoRA 都没用；
    - size mismatch → 拿错基座（SD1.5 的 LoRA 装进 SDXL）；
    - 其它 → 大概率是 LoKr/LoHa/LyCORIS 变体。

    一律归因为"变体不支持"是最糟的：用户会去换 LoRA，但换多少个都一样。
    """
    if ("NoMatchingPeftModule" in msg or "not found in the base model" in msg) \
            and "size mismatch" not in msg:
        return (
            f"LoRA 与当前底模的模块结构对不上: {name}\n"
            "它引用的层（如 q.proj / adaln_modulation）在这套 UNet 里不存在，"
            "说明不是 SD1.5/SDXL 的 LoRA（可能是 Flux/anima 这类 DiT 架构）。\n"
            "用 python -m workflows.lora 看它的实际架构。"
        )
    if "size mismatch" in msg or "shape" in msg:
        return (
            f"LoRA 与当前底模不匹配: {name}\n"
            "这个 LoRA 是用另一个架构训的（SD1.5 的 LoRA 不能用于 SDXL，反之亦然）。"
            "多个 LoRA 叠加时**必须同架构**。\n"
            f"详情: {msg[:300]}"
        )
    return (
        f"LoRA 加载失败: {name} —— {msg}\n"
        "大概率不是标准 LoRA（LoKr/LoHa/LyCORIS 变体暂不支持），"
        "请换一个标准 LoRA（权重 key 形如 lora_unet_*_lora_A.weight）。"
    )


def _count_lora_params(modules) -> int:
    """数一数这些模块里有多少个 LoRA 参数（名字带 lora 的）。

    用来判断 `load_lora_weights` 到底是"真注入了"还是"读完发现没有能对上的键"。
    遍历 named_parameters 是毫秒级，相对加载本身可以忽略。
    """
    total = 0
    for m in modules:
        if m is None:
            continue
        try:
            total += sum(1 for name, _ in m.named_parameters() if "lora" in name.lower())
        except Exception:                     # noqa: BLE001  非 nn.Module 就跳过
            continue
    return total


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
        self._discover_bases()
        # 共享权重池（按 root 归一化后作为键）
        key = str(self.root.resolve())
        pool = _POOLS.get(key)
        if pool is None:
            # OrderedDict 让 ensure_headroom 能按「最久没用」的顺序驱逐
            pool = {"cache": OrderedDict(), "ctrl": {}, "lora": None, "pose": None}
            _POOLS[key] = pool
        self._pool = pool
        self._cache: "OrderedDict[str, Any]" = pool["cache"]
        self._ctrl: Dict[str, ControlNetModel] = pool["ctrl"]
        # 语义分割抠图模型目录（isnet-general-use.onnx）。存在即启用 u2net 后端。
        self.matte_dir = self.root / "matte"

    # ---------- 底模发现与架构判定 ----------

    def _discover_bases(self) -> None:
        """把 models/checkpoints/*/ 里所有**能做文生图**的底模登记进 BASE_MODELS。

        为什么要自动发现：**画质的主因是基座权重，不是架构**。原始 SD1.5 在
        "角色 + 武器 + 铠甲"这类复合概念上本来就弱，换一个社区微调（同架构、
        同显存、同步数）往往就能解决。但把基座写死在代码里，每次换模型都要改
        两个文件——于是没人愿意换。这里让"多放一个目录"就等于"多一个可选底模"。

        **必须排除 inpaint 专用权重**：`sd15-inpaint` 的 UNet 是 9 通道
        （多出来的 4 通道喂加噪原图与蒙版），拿它跑文生图会在采样时报
        `size of tensor a (9) must match b (4)`。这类目录要能被认出并跳过，
        否则它会出现在前端下拉里，用户一选就崩。
        """
        ck = self.root / "checkpoints"
        if not ck.is_dir():
            return
        for d in sorted(ck.iterdir()):
            if not (d / "model_index.json").is_file():
                continue
            name = d.name
            if name in BASE_MODELS:
                continue
            if not self._txt2img_ok(d):
                log.info("跳过 %s：不是文生图权重（inpaint/专用管线）", name)
                continue
            BASE_MODELS[name] = f"checkpoints/{name}"
            log.info("发现底模 %s（架构 %s）：%s", name, self._arch_of(d), d.name)

    def _txt2img_ok(self, path: Path) -> bool:
        """底模能不能跑文生图：看管线类名与 UNet 输入通道。

        过一道就够了：inpaint 类名不同、UNet 是 9 通道；SDXL base 是 4 通道。
        """
        try:
            idx = json.loads((path / "model_index.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if "Inpaint" in str(idx.get("_class_name") or ""):
            return False
        try:
            unet = json.loads((path / "unet" / "config.json").read_text("utf-8"))
            return int(unet.get("in_channels", 4)) == 4
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return True                   # 读不到就按可用处理，别误伤正常权重

    def _arch_of(self, path: Path) -> str:
        """按 model_index.json 的 _class_name 判定架构（sd15 / sdxl）。

        不能靠目录名：用户可能把 SDXL 微调放进 `checkpoints/my_style/`。
        """
        cache = getattr(self, "_arch_cache", None)
        if cache is None:
            cache = self._arch_cache = {}
        key = str(path)
        if key in cache:
            return cache[key]
        try:
            idx = json.loads((path / "model_index.json").read_text("utf-8"))
            cls = str(idx.get("_class_name") or "")
        except (OSError, json.JSONDecodeError):
            cls = ""
        arch = "sdxl" if ("XL" in cls or "xl" in cls) else "sd15"
        cache[key] = arch
        return arch

    def arch(self, model: str = DEFAULT_MODEL) -> str:
        """'sd15' | 'sdxl'。所有"要不要走 XL 管线"的判断都该用它，而不是比字符串。"""
        return self._arch_of(self._check_model(model))

    def is_xl(self, model: str = DEFAULT_MODEL) -> bool:
        return self.arch(model) == "sdxl"


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

    def _fit(self, pipe, name: str, *, vae_slicing: Optional[bool] = None):
        """显存守卫：装不下就逐级降级，而不是 OOM 崩掉整个服务。

        降级顺序与模块文档一致：**先省（VAE/attention 切片）→ 再挪（cpu offload）**。
        老实现只做了最后一步，等于"一上来就搬到 CPU 来回倒腾"，实测反而更慢更占。

        `vae_slicing`：SDXL 1024² 的 VAE 解码峰值约 1.5~2GB、AnimateDiff 整段解码
        实测占 4GB（见 vramopt.py），这些场景必须切；SD1.5 单图 512~768² 切片
        只有 0.05s 开销、省不到 100MB，默认不开。
        """
        pipe.set_progress_bar_config(disable=True)
        if vae_slicing is None:
            vae_slicing = False

        def _slice_vae():
            nonlocal vae_slicing
            if vae_slicing:
                return
            try:
                pipe.vae.enable_slicing()
                vae_slicing = True
            except Exception:                     # noqa: BLE001
                pass

        if vae_slicing:
            _slice_vae()
        try:
            return pipe.to(self.device)
        except torch.OutOfMemoryError:
            log.warning("%s 直接上卡失败，降级 attention/vae 切片后重试", name)
        torch.cuda.empty_cache()
        try:
            pipe.enable_attention_slicing()
            _slice_vae()
            return pipe.to(self.device)
        except torch.OutOfMemoryError:
            log.warning("%s 切片后仍装不下，降级 model_cpu_offload", name)
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        return pipe


    # ---------- pipeline 获取（懒加载 + 缓存）----------

    def _touch(self, key: str) -> None:
        """把刚用过的管线移到 LRU 队尾（OrderedDict 的插入顺序即"最近使用"）。"""
        try:
            self._cache.move_to_end(key)
        except KeyError:                          # pragma: no cover
            pass

    def _release_ipa_if_needed(self, pipe) -> None:
        """共享 UNet 上不能同时存在 IP-Adapter 投影层与"没有参考图"的调用。

        `load_ip_adapter` 是往 **UNet** 里插 `encoder_hid_proj` 的。pose 管线复用
        txt2img 的 UNet 之后，这个补丁会跟着 UNet 一起被文生图看到——而文生图没有
        `ip_adapter_image`，UNet 就会抛
        `argument of type 'NoneType' is not iterable`。

        所以切回"没有参考图"的管线前，先把 IP-Adapter 摘掉；下次进 pose 管线时
        再挂回去。摘/挂是几百毫秒级，换回来的是**少一份 1.7GB UNet**。
        """
        unet = getattr(pipe, "unet", None)
        if unet is None or getattr(unet, "encoder_hid_proj", None) is None:
            return
        owner = self._pool.get("ipa_owner")
        if owner is None:
            return
        try:
            owner.unload_ip_adapter()
            log.info("暂时卸载 IP-Adapter（共享 UNet 要跑无参考图的管线）")
        except Exception as exc:              # noqa: BLE001
            log.warning("卸载 IP-Adapter 失败，退回独占 UNet：%s", exc)
        self._pool["ipa_owner"] = None

    def _get(self, key: str):
        pipe = self._cache.get(key)
        if pipe is not None:
            self._touch(key)
        return pipe

    def txt2img_pipe(self, model: str = DEFAULT_MODEL):
        key = f"txt2img:{model}"
        if key not in self._cache:
            t0 = time.perf_counter()
            p = self._check_model(model)
            xl = self.is_xl(model)
            cls = StableDiffusionXLPipeline if xl else StableDiffusionPipeline
            pipe = cls.from_pretrained(p, variant=detect_variant(p), **self._base_kwargs())
            pipe.scheduler = _scheduler(pipe.scheduler.config)
            self._cache[key] = self._fit(pipe, key, vae_slicing=xl)
            log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        pipe = self._get(key)
        self._release_ipa_if_needed(pipe)
        return pipe

    def img2img_pipe(self, model: str = DEFAULT_MODEL):
        key = f"img2img:{model}"
        if key not in self._cache:
            tkey = f"txt2img:{model}"
            xl = self.is_xl(model)
            cls = StableDiffusionXLImg2ImgPipeline if xl else StableDiffusionImg2ImgPipeline
            if tkey in self._cache:            # 共享权重，避免二次占显存
                pipe = cls(**self._cache[tkey].components)
            else:
                p = self._check_model(model)
                pipe = cls.from_pretrained(p, variant=detect_variant(p), **self._base_kwargs())
                pipe.scheduler = _scheduler(pipe.scheduler.config)
            self._cache[key] = self._fit(pipe, key, vae_slicing=xl)
        pipe = self._get(key)
        self._release_ipa_if_needed(pipe)
        return pipe


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
            return self._get(key)

        t0 = time.perf_counter()
        base = self._base_kwargs()
        xl = self.is_xl(model)
        if xl:
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
        self._cache[key] = self._fit(pipe, key, vae_slicing=xl)
        log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return self._get(key)

    def controlnet(self, which: str = "openpose", model: str = DEFAULT_MODEL) -> ControlNetModel:
        key = f"{which}:{model}"
        if key in self._ctrl:
            return self._ctrl[key]
        if self.is_xl(model):
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
            if self.is_xl(model) else ("ip-adapter_sd15.safetensors", "ip-adapter_sd15.bin")
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
        cached = self._get(key)
        if cached is not None:
            # 缓存命中也可能需要补挂：中间的文生图调用会把共享 UNet 上的
            # IP-Adapter 摘掉（见 _release_ipa_if_needed）。
            if with_ipadapter:
                if getattr(cached.unet, "encoder_hid_proj", None) is None:
                    self._attach_ipa(cached, model)
                else:
                    self._pool["ipa_owner"] = cached
            return cached
        t0 = time.perf_counter()
        cn = self.controlnet(controlnet, model)
        xl = self.is_xl(model)
        cls = StableDiffusionXLControlNetPipeline if xl else StableDiffusionControlNetPipeline
        # **共享底座的 UNet/VAE/TextEncoder**：老实现走 from_pretrained，等于把
        # 同一份 1.7GB UNet 在显存里又存了一份（README 实测 7.36GB 的来源）。
        # txt2img 管线已经在缓存里就直接复用它的组件，省 2~3GB 且 LoRA 只需加载一次。
        base = self._get(f"txt2img:{model}")
        if base is not None:
            pipe = cls(controlnet=cn, **dict(base.components))
            pipe.scheduler = _scheduler(pipe.scheduler.config)
            log.info("%s 复用已加载的底座权重", key)
        else:
            p = self._check_model(model)
            pipe = cls.from_pretrained(
                p, controlnet=cn, variant=detect_variant(p), **self._base_kwargs()
            )
            pipe.scheduler = _scheduler(pipe.scheduler.config)
        if with_ipadapter:
            self._attach_ipa(pipe, model)
        self._cache[key] = self._fit(pipe, key, vae_slicing=xl)
        log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return self._get(key)

    def _attach_ipa(self, pipe, model: str) -> None:
        """把 IP-Adapter 挂到管线（共享 UNet 上重复挂会形状不匹配，先摘再挂）。"""
        owner = self._pool.get("ipa_owner")
        if owner is not None and owner is not pipe:
            try:
                owner.unload_ip_adapter()
            except Exception:                  # noqa: BLE001
                pass
            self._pool["ipa_owner"] = None
        ipa_root, ipa_sub, ipa_name = self.ipa_weight(model)
        pipe.load_ip_adapter(ipa_root, ipa_sub, weight_name=ipa_name,
                             image_encoder_folder="image_encoder")
        pipe.set_ip_adapter_scale(0.8)
        self._pool["ipa_owner"] = pipe

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
            return self._get(key)
        t0 = time.perf_counter()
        from diffusers import (AnimateDiffControlNetPipeline, MotionAdapter)

        # motion module 与 SD1.5 的 UNet 结构绑定，SDXL 上装不进去。
        # 这里必须显式报错：静默回退到 SD1.5 会让"切了底模却没效果"再次发生。
        if self.is_xl(model):
            raise ValueError(
                f"AnimateDiff 只支持 SD1.5 架构，{model} 是 SDXL。\n"
                "改用逐帧引擎（engine_mode='frame'），它支持两个底模。")
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

    # ---------- 少步蒸馏（LCM / Hyper-SD）----------

    DISTILL_DIR = "lora_distill"
    DISTILL_FILES = {
        "lcm": "lcm-lora-sdv1-5.safetensors",
        "hyper": "hyper-sd15-8steps-lora.safetensors",
    }
    # 蒸馏模型都是「低步数 + 低 CFG」训练的，步数给多了反而变差
    DISTILL_DEFAULTS = {"lcm": {"steps": 8, "cfg": 1.5},
                        "hyper": {"steps": 8, "cfg": 1.0}}

    def distill_available(self) -> list[str]:
        d = self.root / self.DISTILL_DIR
        return [k for k, f in self.DISTILL_FILES.items() if (d / f).exists()]

    def apply_distill(self, pipe, name: str = "lcm") -> str:
        """加载少步蒸馏 LoRA 并把调度器换成 LCM，返回用到的名字。

        为什么要它（评审 Q3）：30 步 × 2 候选的预算，换成 8 步 × 6 候选更划算——
        画质靠**挑**而不是靠步数，而"挑"正是本项目最擅长的环节。

        **风险要说清**：蒸馏 LoRA 会削弱风格保真（公开评测里 Hyper-SD 的 FID
        明显高于教师模型），而本项目重度依赖像素 LoRA。所以这里是**可选**能力，
        必须在自己的素材上 A/B（判据用每格单色率 + framescore 身份相似度），
        不能默认打开。
        """
        name = (name or "").lower()
        if name not in self.DISTILL_FILES:
            raise ValueError(f"未知蒸馏方案 {name}（可选 {', '.join(self.DISTILL_FILES)}）")
        f = self.root / self.DISTILL_DIR / self.DISTILL_FILES[name]
        if not f.exists():
            raise FileNotFoundError(
                f"蒸馏 LoRA 缺失: {f}\n"
                f"放到 {self.DISTILL_DIR}/ 下即可启用（LCM-LoRA 约 130MB），"
                "或把 distill 留空用原始步数。")
        pipe.load_lora_weights(str(f), adapter_name="distill")
        from diffusers import LCMScheduler

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        self._pool["distill"] = name
        log.info("少步蒸馏已启用: %s（建议 %s）", name,
                 self.DISTILL_DEFAULTS[name])
        return name

    def _active_adapters(self, pipe) -> list[str]:
        try:
            return list(pipe.get_active_adapters() or [])
        except Exception:                     # noqa: BLE001
            return []

    # ---------- LoRA 热切换 ----------

    def apply_lora(self, pipe, lora_path: str | Path | None | Sequence[str | Path],
                   scale: float | Sequence[float] = 0.8) -> None:
        """LoRA 热加载，**支持多个叠加**（例如 像素风格 + 角色）。

        单个时照旧传一个路径；多个时 `lora_path` 传列表、`scale` 传列表
        （或给一个数值表示所有 LoRA 共用该强度）。

        每个 LoRA 挂成独立 adapter（lora0 / lora1 / …），再用 `set_adapters`
        分别给权重——diffusers 会把它们的效果**线性叠加**，这就是 A1111 里
        `<lora:a:0.8> <lora:b:0.5>` 的等价物。

        **状态记在 UNet 上，不是记在 Engine 上**。记在 Engine 上的话，
        "W1 加载了像素 LoRA，接着跑 W4"会命中同一个 key 直接 return，
        而 W4 的管线根本没加载过——静默丢风格，且没有任何报错。
        记在 unet 上还顺带解决了共享权重（pose 管线复用 txt2img 的 UNet）的问题：
        两条管线看的是同一份权重，状态天然一致。
        """
        items = _norm_lora_items(lora_path, scale)
        unet = getattr(pipe, "unet", None)
        target = unet if unet is not None else pipe
        cur = getattr(target, LORA_ATTR, None)

        def _unload() -> None:
            try:
                pipe.unload_lora_weights()
            except Exception:                 # noqa: BLE001  没装过就别让它炸
                pass

        def _reset() -> None:
            _unload()
            setattr(target, LORA_ATTR, None)
            self._pool["lora"] = None

        if not items:
            if cur is not None:
                _reset()
            return

        key = ";".join(f"{Path(p).name}:{sc}" for p, sc in items)
        if key == cur:
            self._pool["lora"] = key            # stats() 用，只作展示
            return
        if cur is not None:
            _unload()
            setattr(target, LORA_ATTR, None)

        holders = [getattr(pipe, n, None) for n in
                   ("unet", "text_encoder", "text_encoder_2", "transformer")]
        names: list[str] = []
        weights: list[float] = []
        for i, (path, sc) in enumerate(items):
            adapter = f"lora{i}"
            before = _count_lora_params(holders)
            try:
                pipe.load_lora_weights(str(path), adapter_name=adapter)
            except Exception as exc:            # noqa: BLE001
                _reset()
                raise ValueError(_lora_load_error(Path(path).name, str(exc))) from exc
            # **逐个校验**：多个 LoRA 时只看总数会漏——一个空挂、另一个正常，
            # 总数照样上涨，但空挂那个永远不会生效。
            if _count_lora_params(holders) <= before:
                _reset()
                raise ValueError(
                    f"这个 LoRA 没起作用: {Path(path).name}\n"
                    "文件能读、但**没有一个权重键能对上当前底模**（diffusers 只给了 "
                    "warning，不报错）。常见于把视频模型（Wan）或 DiT（Flux/anima）的 "
                    "LoRA 挂到 SD1.5/SDXL 上，或反过来。\n"
                    "用 python -m workflows.lora 看一眼它的实际架构。"
                )
            names.append(adapter)
            weights.append(sc)

        # 蒸馏 LoRA 是另一个 adapter，set_adapters 只列 lora* 会把它关掉
        if "distill" in self._active_adapters(pipe):
            names.append("distill")
            weights.append(1.0)
        pipe.set_adapters(names, adapter_weights=weights)
        setattr(target, LORA_ATTR, key)
        self._pool["lora"] = key
        log.info("LoRA 热切换 -> %s",
                 ", ".join(f"{Path(p_).name}@{sc}" for p_, sc in items))

    # ---------- 显存管理 ----------

    # 精灵管线的推荐参数。**SDXL 推荐 768 而不是 1024**：
    # 实测 1024² 下角色占比更大、更容易顶边，可交付率从 92% 掉到 75%，
    # 加上生成慢一倍，每可用帧贵 2.3 倍（见 tools/bench_sprite.py）。
    RECOMMEND = {
        "sd15": {"width": 768, "height": 768, "steps": 30, "cfg": 7.0,
                 "pixel": 6, "palette": 32},
        "sdxl": {"width": 768, "height": 768, "steps": 24, "cfg": 6.5,
                 "pixel": 6, "palette": 32},
    }

    def recommend(self, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """这个底模在**精灵管线**上的推荐参数（尺寸/步数/像素格/色数）。

        额外带 `max_candidates`：这个底模在推荐尺寸下，W4（ControlNet +
        IP-Adapter）最多能批量几个候选。前端拿它当输入框上限——
        实测 sdxl@1024 批量 2 会占满 24GB，**不报 OOM 但耗时从 5.0s 崩到 17.5s**，
        靠用户自己猜是猜不出来的。
        """
        rec = dict(self.RECOMMEND.get(self.arch(model), self.RECOMMEND["sd15"]))
        rec["max_candidates"] = self.max_candidates(model, rec["width"])
        return rec

    def max_candidates(self, model: str = DEFAULT_MODEL, size: int | None = None,
                       total_gb: float = 0.0) -> int:
        """W4 在 `size` 边长下最多能批量几个候选（按**本机总显存**算）。

        与 `vramopt.plan_candidates` 的区别：那个看**当前空闲**显存（运行时决策），
        这个看**总显存**（给前端设输入框上限、提前拦住明显跑不动的配置）。
        两者都要有：前者防止跑着跑着劣化，后者防止用户一开始就设成跑不动的值。
        """
        from . import vramopt

        if size is None:
            size = self.RECOMMEND.get(self.arch(model), self.RECOMMEND["sd15"])["width"]
        if total_gb <= 0:
            if not torch.cuda.is_available():
                return 1                    # 没卡就别鼓励批量
            total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        budget = total_gb * 0.9             # 与 plan_candidates 的 safety 保持一致
        k = 1
        while k < 6 and vramopt.estimate_pose_gb(self.arch(model), int(size), k + 1) <= budget:
            k += 1
        return k

    def free_gb(self) -> float:
        """当前还能给新工作负载用的显存（GB），扣掉 PyTorch 已保留的块。"""
        return self._free_gb()

    def _free_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        _, total = torch.cuda.mem_get_info()
        # **用 reserved 反推占用**：mem_get_info 的 free 不含 PyTorch 缓存着的块，
        # 那些块下次申请会被直接复用，所以「物理空闲」小于「可用」。
        # 这里按 reserved 算"已经占掉多少"，剩下的才是真能给新管线用的。
        return max(0.0, (total - torch.cuda.memory_reserved()) / 2**30)

    def _ctrl_gc(self) -> None:
        """丢掉没有被任何缓存管线引用的 ControlNet。

        ControlNet 存在独立字典里，**不随管线一起释放**——老实现只清 _cache，
        实测残留 4.16GB。这里按"还有没有管线名字里带它"来判断。
        """
        # IP-Adapter 的归属管线同理：它若已被驱逐，pool 里的引用会把它吊在显存里
        owner = self._pool.get("ipa_owner")
        if owner is not None and not any(owner is p for p in self._cache.values()):
            self._pool["ipa_owner"] = None
        names = set(self._ctrl)
        if not names:
            return
        live = " ".join(self._cache.keys())
        for name in list(names):
            kind = name.split(":")[0]
            if kind not in live:
                self._ctrl.pop(name, None)

    def drop_others(self, keep: Sequence[str] = ()) -> None:
        """丢掉所有不匹配 `keep` 前缀的缓存管线。

        与 `ensure_headroom` 的分工：那个是"不够了才腾"，这个是"这条工作流
        明确要吃一大块显存，先清场"。AnimateDiff 32 帧要 14GB，留一条闲置的
        2.6GB 文生图管线在旁边没有任何好处。
        """
        for key in list(self._cache):
            if not any(k in key for k in keep):
                self._cache.pop(key, None)
        self._ctrl_gc()
        gc.collect()
        torch.cuda.empty_cache()

    def ensure_headroom(self, need_gb: float = 2.0,
                        keep: Sequence[str] = ()) -> None:
        """显存不够时按 LRU 驱逐缓存管线，**直到腾出 need_gb 的空闲**。

        `need_gb` 的语义是"**还需要多少空闲显存**"，不是"已用量上限"。
        老实现比较的是 `memory_reserved() < need_gb`，任何管线加载后都恒不成立，
        于是每次调用都把整个缓存清空——cache 形同虚设，每次 wss/wpc 都要
        重新加载 UNet（数秒），而这笔成本没有出现在计时里，所以一直没被发现。

        `keep`：子串匹配的管线前缀，绝不驱逐。调用方应把"马上要用的那条"传进来，
        否则会把自己刚加载好的管线扔掉再重载。
        """
        if not torch.cuda.is_available():
            return
        if self._free_gb() >= need_gb:
            return
        # OrderedDict 的迭代顺序就是「最久没用 → 最近用」
        for key in list(self._cache.keys()):
            if self._free_gb() >= need_gb:
                break
            if any(k in key for k in keep):
                continue
            self._cache.pop(key, None)
            self._ctrl_gc()
            gc.collect()
            torch.cuda.empty_cache()
            log.info("显存紧张（空闲 %.1fGB < %.1fGB），已释放管线 %s",
                     self._free_gb(), need_gb, key)

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
