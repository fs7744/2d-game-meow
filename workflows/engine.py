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
# 运行态只跑 Qwen：SD 权重已移到 models/_archive/sd/（不删，可恢复）。
# 回退 SD：把归档里的权重移回原位并改回 "sd15"，或直接切到 git tag `sd`。
DEFAULT_MODEL = "qwen-image-2.1"

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


def _fp8_linear_cls():
    """fp8(E4M3) 权重量化 Linear：权重存 fp8 buffer，前向时反量化成 bf16 再算。

    **为什么权重不直接放 `weight`**：diffusers 到处用「取第一个 Parameter 的
    dtype」推断计算精度，时间步张量（0~1000）会被转成 fp8，而 e4m3 上限只有
    448 —— 一转就溢出成 inf/NaN，采样直接崩。所以布局是「真数据放 buffer
    `w_fp8`，`weight` 置成 0 字节 bf16 Parameter 专供 dtype 探针」。

    类只建一次（闭包里带 torch，模块顶层 import 会拖慢 CLI 启动）。
    """
    cached = getattr(_fp8_linear_cls, "_cls", None)
    if cached is not None:
        return cached
    import torch.nn.functional as F

    class _FP8Linear(torch.nn.Linear):
        def forward(self, x):  # type: ignore[override]
            w_fp8 = getattr(self, "w_fp8", None)
            if w_fp8 is None:                 # 理论上不会走到
                return F.linear(x, self.weight.to(x.dtype), self.bias)
            dt = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
            if x.dtype != dt:
                x = x.to(dt)
            w = w_fp8.to(device=x.device, dtype=dt)
            sc = getattr(self, "w_scale", None)
            if sc is not None:
                w = w * sc.to(device=w.device, dtype=dt)[:, None]
            b = self.bias
            if b is not None:
                b = b.to(device=w.device, dtype=dt)
            return F.linear(x, w, b)

    _fp8_linear_cls._cls = _FP8Linear
    return _FP8Linear


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
        cls = str(idx.get("_class_name") or "")
        if "Inpaint" in cls:
            return False
        # **Refiner 也要排除**：SDXL refiner 的管线类是 Img2Img（它吃底模的
        # latent 再精修），不是文生图底模。留在下拉里会让用户选中后跑出废图。
        if "Img2Img" in cls:
            return False
        # Qwen-Image 没有 unet/，主干是 transformer/。不特判的话下面读 unet
        # 会走 except 分支返回 True —— 碰巧"能用"，但 `in_channels` 那层保护
        # 就失效了（将来放个 9 通道的 qwen inpaint 进来会被误放行）。
        if "Qwen" in cls and "Transformer" in cls:
            try:
                tr = json.loads((path / "transformer" / "config.json").read_text("utf-8"))
                return int(tr.get("in_channels", 64)) == 64
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return False
        try:
            unet = json.loads((path / "unet" / "config.json").read_text("utf-8"))
            return int(unet.get("in_channels", 4)) == 4
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return True                   # 读不到就按可用处理，别误伤正常权重

    def _arch_of(self, path: Path) -> str:
        """按 model_index.json 的 _class_name 判定架构（sd15 / sdxl / qwen21）。

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
        # 顺序重要：QwenImage21Pipeline 不含 "XL"，先判它，否则会被当 sd15
        if "QwenImage" in cls:
            arch = "qwen21"
        elif "XL" in cls or "xl" in cls:
            arch = "sdxl"
        else:
            arch = "sd15"
        cache[key] = arch
        return arch

    def arch(self, model: str = DEFAULT_MODEL) -> str:
        """'sd15' | 'sdxl' | 'qwen21'。所有"走哪条管线"的判断都该用它。"""
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
            self.evict_other_models(model)
            p = self._check_model(model)
            if self.arch(model) == "qwen21":
                pipe = self._load_qwen21(p)
                self._cache[key] = pipe
                log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
                return pipe
            xl = self.is_xl(model)
            cls = StableDiffusionXLPipeline if xl else StableDiffusionPipeline
            pipe = cls.from_pretrained(p, variant=detect_variant(p), **self._base_kwargs())
            pipe.scheduler = _scheduler(pipe.scheduler.config)
            self._cache[key] = self._fit(pipe, key, vae_slicing=xl)
            log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        pipe = self._get(key)
        self._release_ipa_if_needed(pipe)
        return pipe

    # ---------- Qwen-Image-2.1 ----------

    def _load_qwen21(self, root: Path):
        """Qwen-Image-2.1：单流 DiT（7B）+ Qwen3-VL 编码器，一个权重干三件事。

        文生图 / 指令改图（给 image 即走编辑路径）/ 原生 RGBA 透明输出。
        与 SD 系列不共用任何组件，所以是独立管线，不进共享 UNet 池。

        **必须分组件加载**（实测本机 32G 内存）：
        transformer 13.3G bf16 + text_encoder 16.3G bf16 一次性 from_pretrained
        会把内存打满（连 43G 目录一起读会直接 swap）。这里 transformer 走
        预量化 fp8 缓存（6.7G，项目里已带），text_encoder 保持 bf16 靠
        cpu offload 在编码后搬出显存。

        torchao 的 fp8 量化在本机不可用（要求 torch≥2.11，本机 2.6），
        所以 text_encoder 不做量化 —— 换来的是峰值 16.4G 仍能进 24G 卡。
        """
        from diffusers import QwenImage21Pipeline
        from transformers import Qwen3VLForConditionalGeneration

        # **加载前先清场**：qwen 峰值 16.4GB，SDXL 的 ControlNet 管线约 13GB，
        # 两条并存必然挤爆。不清的话 _fit 会降级到 cpu offload（每步在 CPU/GPU
        # 间搬权重），用户看到的是「一直转、显存满」而不是报错。
        # 反方向也一样——generate.py 里 pose 系加载前会按需驱逐 qwen。
        self.ensure_headroom(need_gb=17.0)
        tor = self._load_qwen_transformer(root)
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            str(root / "text_encoder"), torch_dtype=torch.bfloat16,
            local_files_only=True)
        pipe = QwenImage21Pipeline.from_pretrained(
            str(root), transformer=tor, text_encoder=te,
            torch_dtype=torch.bfloat16, local_files_only=True)
        # 峰值 16.4G 是靠 offload 拿到的：不 offload 时 6.6G(transformer fp8)
        # + 16.3G(text_encoder) 同时驻留会 OOM。代价是每张多几秒搬运，
        # 但 24G 卡上没有别的选择。
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _load_qwen_transformer(self, root: Path):
        """加载 transformer：优先 fp8 缓存（6.7G），没有就 bf16 完整加载（13.3G）。

        fp8 缓存的键布局是 `w_fp8` + `w_scale` 两个 buffer 顶替 bf16 的 `weight`
        （weight 留一个 0 长度 bf16 Parameter 专供 diffusers 的 dtype 探针）。
        不能走 load_state_dict —— 它对 assign=True 仍做形状校验，0 长度占位会
        报一屏 size mismatch，所以逐键直接挂到模块上。
        """
        from safetensors.torch import load_file

        fp8_dir = root / "transformer-fp8"
        use_fp8 = (fp8_dir / "model.safetensors").is_file() and \
            os.environ.get("MEOW_QWEN_FP8", "1") != "0"
        if not use_fp8:
            from diffusers import QwenImage21Transformer2DModel
            t = QwenImage21Transformer2DModel.from_pretrained(
                str(root / "transformer"), torch_dtype=torch.bfloat16,
                local_files_only=True)
            return t.eval()

        from accelerate import init_empty_weights
        from diffusers import QwenImage21Transformer2DModel

        cfg = QwenImage21Transformer2DModel.load_config(str(fp8_dir / "config.json"))
        with init_empty_weights():
            model = QwenImage21Transformer2DModel.from_config(cfg)

        sd = load_file(str(fp8_dir / "model.safetensors"), device="cpu")
        linear_keys = sorted({k[: -len(".w_fp8")] for k in sd if k.endswith(".w_fp8")})
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for t in linear_keys:
            model.get_submodule(t).weight = torch.nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=dev), requires_grad=False)
        for k, v in sd.items():
            if k.endswith(".w_fp8") or k.endswith(".w_scale"):
                mod = model.get_submodule(k.rsplit(".", 1)[0])
                mod.register_buffer(k.rsplit(".", 1)[1], v.to(dev), persistent=False)
                continue
            mod = model.get_submodule(k.rsplit(".", 1)[0])
            name = k.rsplit(".", 1)[1]
            if name in mod._parameters and mod._parameters[name] is not None:
                mod._parameters[name] = torch.nn.Parameter(v.to(dev), requires_grad=False)
            else:
                mod._buffers[name] = v.to(dev)
        del sd
        cls_fp8 = _fp8_linear_cls()
        for t in linear_keys:
            model.get_submodule(t).__class__ = cls_fp8
        stuck = [k for k, v in list(model.named_parameters()) + list(model.named_buffers())
                 if getattr(v, "is_meta", False)]
        if stuck:
            raise RuntimeError(f"qwen transformer 有 {len(stuck)} 个参数停在 meta：{stuck[:5]}")
        log.info("qwen transformer fp8 就绪：%d 个 Linear，%.1fGB",
                 len(linear_keys), torch.cuda.memory_allocated() / 2**30)
        return model.eval()

    def img2img_pipe(self, model: str = DEFAULT_MODEL):
        key = f"img2img:{model}"
        if key not in self._cache:
            self.evict_other_models(model)
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
        # keep 留空：当前这条还没进缓存，别的底模的 inpaint 管线也没必要留着
        self.evict_other_models(model)
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
        # pose 管线要 UNet + ControlNet + IP-Adapter，跟别的底模绝不共存
        self.evict_other_models(model, keep=(f"pose:{controlnet}",))
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

    # ---------- SDXL Refiner（精修器） ----------

    REFINE_DIR = "checkpoints/sdxl-refiner"

    def refiner_path(self) -> Optional[Path]:
        """SDXL refiner 权重目录；没下载就返回 None。"""
        p = self.root / self.REFINE_DIR
        return p if (p / "model_index.json").is_file() else None

    def has_refiner(self) -> bool:
        return self.refiner_path() is not None

    def refiner_pipe(self):
        """加载 SDXL refiner（img2img 管线）。没有权重就返回 None。

        refiner 的作用：吃底模输出的 latent，再走一遍去噪补细节。它不是
        "换个更大的模型"，而是把最后 20% 的去噪交给一个专攻细节的权重，
        所以对发丝/布料纹理的提升比对构图的提升大得多。
        """
        key = "refiner:sdxl"
        if key in self._cache:
            return self._get(key)
        p = self.refiner_path()
        if p is None:
            return None
        from diffusers import StableDiffusionXLImg2ImgPipeline

        t0 = time.perf_counter()
        # refiner 仓库里通常**只有 text_encoder_2**（OpenCLIP），ViT-L 的
        # text_encoder 要复用底模的——本来就是同一个 CLIP-L，直接共享权重，
        # 省 1.3GB 磁盘也省一份显存。
        extra: Dict[str, Any] = {}
        if not (p / "text_encoder" / "config.json").is_file():
            base = self.txt2img_pipe("sdxl")
            extra = dict(text_encoder=base.text_encoder, tokenizer=base.tokenizer)
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            str(p), variant=detect_variant(p),
            **extra, **self._base_kwargs())      # torch_dtype 在 _base_kwargs 里
        self._cache[key] = self._fit(pipe, key, vae_slicing=True)
        log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return self._get(key)

    def refiner_embeds(self, prompt: str, negative: str, clip_skip: int = 2):
        """给 refiner 准备文本 embeds（SDXL refiner 的输入跟底模不一样）。

        **cross-attn 只吃 text_encoder_2（OpenCLIP bigG）的完整 1280 维隐层。**
        refiner UNet 的 `cross_attention_dim=1280`，而 OpenCLIP bigG 的 hidden
        size 正好是 1280——两者是同一个数，不是巧合。

        ⚠️ 曾经写成 `CLIP-L(768) + OpenCLIP 前 512 = 1280`：维度凑对了，
        但**语义完全错位**。refiner 拿到的是"半段 CLIP-L + 半段 OpenCLIP"的
        拼接向量，它按错误的语义去噪，出图直接糊成一团灰雾（实测拉普拉斯
        方差 1826 → 480，颜色也从白底线稿变成灰底，用户报的就是这个）。
        换成完整 1280 维后同一组参数下 1582，视觉效果与不精修基本一致。

        pooled 用 `text_embeds`（text_projection 之后的 1280 维），这也是
        官方 encode_prompt 的做法；**不要**再拼 CLIP-L 的 768。

        返回 (正 embeds, 负 embeds, 正 pooled, 负 pooled)。
        """
        base = self.txt2img_pipe("sdxl")
        tok2, enc2 = base.tokenizer_2, base.text_encoder_2

        def _pack(text: str):
            ids = tok2(text, padding="max_length",
                       max_length=tok2.model_max_length, truncation=True,
                       return_tensors="pt").input_ids.to(enc2.device)
            out = enc2(ids, output_hidden_states=True)
            # SDXL 默认取倒数第二层（= 业界说的「clip skip 2」）；n > 2 才真多跳，
            # 语义与 generate._text_kwargs 里对底模的处理保持一致。
            idx = -2 if clip_skip <= 2 else -clip_skip
            return out.hidden_states[idx], out[0]

        pos, pos_pool = _pack(prompt)
        neg, neg_pool = _pack(negative)
        return pos, neg, pos_pool, neg_pool

    # ---------- 显存管理 ----------

    # 精灵管线的推荐参数。**SDXL 推荐 768 而不是 1024**：
    # 实测 1024² 下角色占比更大、更容易顶边，可交付率从 92% 掉到 75%，
    # 加上生成慢一倍，每可用帧贵 2.3 倍（见 tools/bench_sprite.py）。
    # 架构 → 支持的工作流集合。
    #
    # **这张表是"某个底模能不能跑这个节点"的唯一判据**：
    #   · generate.py 各函数入口调 `require()`，提前报人话错误；
    #   · /models 把它发给前端，节点上的底模下拉据此过滤。
    #
    # 为什么要这张表：Qwen-Image 的管线**没有 unet / tokenizer**
    # （主干是 transformer，文本编码器是 Qwen3-VL），任何依赖 SD 组件的
    # 工作流都会在 `cls(controlnet=cn, **base.components)` 那里崩成
    # "Pipeline expected ['unet', 'tokenizer', ...] but only {...} were passed"——
    # 这行报错完全看不出是"底模选错了"，用户只会以为程序坏了。
    _ARCH_WORKFLOWS = {
        "sd15": frozenset({
            "txt2img", "img2img", "inpaint", "poseswap", "matte",
            "ai_repaint", "pixelify", "pose_clone", "sprite_swap", "topixel",
        }),
        "sdxl": frozenset({
            "txt2img", "img2img", "inpaint", "poseswap", "matte",
            "ai_repaint", "pixelify", "pose_clone", "sprite_swap", "topixel",
        }),
        # Qwen 保留不依赖 SD 组件的：纯生成 + 纯后处理 + **角色迁移**。
        # 角色迁移在 Qwen 上走的是「双图条件」（角色图 + 骨架图进同一个
        # vision context），不是 SD 的 ControlNet + IP-Adapter，所以能用。
        # 它的 img2img 是「指令改图」，不是加噪强度。
        "qwen21": frozenset({"txt2img", "img2img", "matte", "pixelify", "topixel",
                             "pose_clone"}),
    }

    # 工作流 → 中文名（报错时用，别让用户看到 wpc/wss 这种内部代号）
    _WF_LABEL = {
        "txt2img": "文生图", "img2img": "图生图", "inpaint": "局部重绘",
        "poseswap": "姿态序列", "matte": "抠图", "ai_repaint": "局部 AI 重绘",
        "pixelify": "像素转换", "pose_clone": "角色迁移",
        "sprite_swap": "序列帧替换", "topixel": "转像素",
    }

    def supports(self, model: str, workflow: str) -> bool:
        """这个底模能不能跑这个工作流。未知架构一律返回 False（保守）。"""
        return workflow in self._ARCH_WORKFLOWS.get(self.arch(model), frozenset())

    def workflows_of(self, model: str) -> list[str]:
        """这个底模支持的全部工作流（给前端的下拉过滤用）。"""
        return sorted(self._ARCH_WORKFLOWS.get(self.arch(model), frozenset()))

    def require(self, model: str, workflow: str) -> None:
        """不支持的组合直接抛人话错误，别让它崩在 diffusers 内部的组件校验里。"""
        if self.supports(model, workflow):
            return
        label = self._WF_LABEL.get(workflow, workflow)
        arch = self.arch(model)
        if arch == "qwen21":
            raise ValueError(
                f"「{label}」用不了 Qwen-Image 底模：它的管线没有 SD 的 UNet / "
                f"CLIP 文本编码器，而{label}依赖 SD 的 ControlNet / IP-Adapter 组件。"
                f"到这个节点上把底模换成 SD1.5 或 SDXL（节点可以各自选底模，"
                f"不影响画布上其它节点）。")
        raise ValueError(
            f"「{label}」不支持底模 {model}（架构 {arch}）。换回 SD1.5 / SDXL 试试。")

    RECOMMEND = {
        "sd15": {"width": 768, "height": 768, "steps": 30, "cfg": 7.0,
                 "pixel": 6, "palette": 32},
        # 色数 64 是实测选出来的：同一张 SDXL 产物按 16/32/64 量化对比，
        # 64 色最干净（披风与甲片的亮/中/暗层次分明、杂色最少），
        # 16 色会糊成一块、32 色局部有杂色拼接。SD1.5 画面本身更脏，
        # 仍用 32 免得把噪点也放大成色阶。
        "sdxl": {"width": 768, "height": 768, "steps": 24, "cfg": 6.5,
                 "pixel": 6, "palette": 64},
        # Qwen-Image-2.1：**默认不像素化**（pixel/palette 归零）。
        # 它直接就能画出像样的像素风（实测 768² / 20 步直接出 16-bit 猫骑士），
        # 再过一遍后处理反而把它的色阶优势抹平。步骤 20~40 是官方区间，
        # 40 是默认，但精灵场景 20 已经够（实测 38s vs 更久，构图没差别）。
        # cfg 这个字段对它没有意义（真正的引导是 true_cfg_scale=4.0），
        # 填 4.0 只是让界面显示的值与实际生效值一致。
        "qwen21": {"width": 1024, "height": 1024, "steps": 20, "cfg": 4.0,
                   "pixel": 0, "palette": 0},
    }

    # **按底模名覆盖**（优先于按架构的 RECOMMEND）。
    # 动漫底模（Illustrious / Animagine）的正确用法与精灵管线不同：
    #   · 1024²——它们是在 Danbooru 高分辨率插画上训的，768² 会掉细节
    #   · pixel=0 / palette=0——动漫立绘要保画质，**不要**像素化
    #     （像素化是给精灵的，把插画压成 6px 格子等于毁掉它）
    #   · clip_skip=2——SDXL 默认就是倒数第二层（见 generate._text_kwargs
    #     的反直觉换算），这里写出来只是让前端能显示出来给用户看
    #   · preset=anime——质量词 + 简单背景 + 短负向
    MODEL_RECOMMEND = {
        # cfg 8.0 是**实测**选的，不是照抄网上的 6.5：
        # 同一 group 提示词、同 seed，6.5 下兽人五官松散、法师法杖顶端结构
        # 混乱；8.0 下五官清晰立体、武器造型完整、手部姿势更自然。
        # 网上流传的 "Illustrious 用 CFG 5~7" 是通用建议，本项目实测 8.0 更好。
        "sdxl-anime": {"width": 1024, "height": 1024, "steps": 28, "cfg": 8.0,
                       "pixel": 0, "palette": 0, "clip_skip": 2,
                       "preset": "anime"},
    }

    def recommend(self, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """这个底模在**精灵管线**上的推荐参数（尺寸/步数/像素格/色数）。

        额外带 `max_candidates`：这个底模在推荐尺寸下，W4（ControlNet +
        IP-Adapter）最多能批量几个候选。前端拿它当输入框上限——
        实测 sdxl@1024 批量 2 会占满 24GB，**不报 OOM 但耗时从 5.0s 崩到 17.5s**，
        靠用户自己猜是猜不出来的。

        **按底模名的覆盖优先于按架构**：动漫底模要 1024² 且不像素化，
        跟着 sdxl 的 768²+pixel=6 走会把插画毁掉。
        """
        rec = self.MODEL_RECOMMEND.get(model)
        if rec is None:
            rec = self.RECOMMEND.get(self.arch(model), self.RECOMMEND["sd15"])
        rec = dict(rec)
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

    def evict_other_models(self, model: str, keep: Sequence[str] = ()) -> None:
        """切到某个底模时，把**其它底模**的管线全部释放。

        为什么要主动驱逐，而不是等显存不够再腾：
          qwen 峰值 16.4GB、SDXL 的 ControlNet 管线约 13GB，24G 卡上两条并存
          必然挤爆。而"挤爆"的表现**不是报错**——`_fit` 会静默降级到
          `enable_model_cpu_offload()`，每一步都在 CPU/GPU 之间搬权重，
          慢几十倍。用户看到的是「一直转、显存占满、跑不完」，
          完全看不出是底模并存导致的。

        判据用 `key.split(":")` 里的**完全相等**，不能用子串包含——
        `sd15` 是 `sd15-inpaint` 的前缀，子串匹配会把正在用的那条也误杀。

        `keep`：调用方马上要用、绝不驱逐的前缀（如 ("pose",)）。
        """
        dropped: list[str] = []
        for key in list(self._cache):
            if model in key.split(":") or any(k in key for k in keep):
                continue
            self._cache.pop(key, None)
            dropped.append(key)
        # ControlNet 存在独立字典里，不跟着管线一起走。SDXL 那份的键带
        # `:model` 后缀，SD1.5 那份只有 which——两个方向都得清，
        # 否则切底模后旧 CN 还占着 1.4~2.4GB。
        for name in list(self._ctrl):
            if ":" in name:
                if name.rsplit(":", 1)[-1] != model:
                    self._ctrl.pop(name, None)
            elif self.is_xl(model):
                self._ctrl.pop(name, None)
        if dropped:
            log.info("底模切到 %s：驱逐其它底模的管线 %s", model, ", ".join(dropped))
        self._ctrl_gc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
