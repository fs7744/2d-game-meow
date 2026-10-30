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
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402

log = logging.getLogger("engine")

# 运行态只有 Qwen：SD 权重已移到 models/_archive/sd/（git tag `sd` 之前的状态）。
DEFAULT_MODEL = "qwen-image-2.1"


def _lora_keys(path: Path) -> List[str]:
    """只读 safetensors 的头部，拿到 key 列表（不加载张量数据）。

    safetensors 的布局是：8 字节小端长度 + JSON 头 + 数据。
    只读头就能判断这是不是 diffusers 原生格式。
    """
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 8 * 1024 * 1024:          # 头不该这么大，防御一下
            return []
        head = json.loads(f.read(n))
    return [k for k in head if k != "__metadata__"]


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
                sc = sc.to(device=w.device, dtype=dt)
                # **必须压成 1 维再 [:, None]**：缓存里 scale 可能是 [out]
                # （transformer-fp8）也可能是 [out,1]（text_encoder-fp8）。
                # 对 [out,1] 直接 `sc[:, None]` 会广播成 [out,out,in] 的三维权重，
                # F.linear 于是抛 "t() expects a tensor with <= 2 dimensions"——
                # 报错点在 vision tower 里，完全看不出是 scale 的形状问题。
                if sc.dim() > 1:
                    sc = sc.reshape(-1)
                w = w * sc[:, None]
            b = self.bias
            if b is not None:
                b = b.to(device=w.device, dtype=dt)
            return F.linear(x, w, b)

    _fp8_linear_cls._cls = _FP8Linear
    return _FP8Linear


def _qwen_vl_cls(cfg_dir: Path):
    """从 text_encoder 的 config.json 里认出模型类（Qwen3VLForConditionalGeneration）。

    **不能写死 import 路径**：`transformers.models.qwen3_vl` 这种模块名随版本
    改过（qwen3_vl / qwen3_vl_moe），而 config 里的 `architectures` 一直是稳的。
    所以按 `model_type` 拼模块名、按 `architectures[0]` 取名——两个都来自权重目录，
    与安装的 transformers 版本无关。
    """
    import importlib

    cfg = json.loads((cfg_dir / "config.json").read_text("utf-8"))
    arch = (cfg.get("architectures") or ["Qwen3VLForConditionalGeneration"])[0]
    mtype = cfg.get("model_type", "qwen3_vl")
    mod = importlib.import_module(f"transformers.models.{mtype}")
    return getattr(mod, arch)
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
            # OrderedDict 让 ensure_headroom 能按「最久没用」的顺序驱逐
            pool = {"cache": OrderedDict(), "pose": None}
            _POOLS[key] = pool
        self._pool = pool
        self._cache: "OrderedDict[str, Any]" = pool["cache"]
        # 语义分割抠图模型目录（isnet-general-use.onnx）。存在即启用 u2net 后端。
        self.matte_dir = self.root / "matte"

    # ---------- 底模 ----------

    def arch(self, model: str = DEFAULT_MODEL) -> str:
        """恒为 `'qwen21'`。

        **保留这个方法而不是删掉**：`generate.py` 的每个工作流入口都要拿它做
        分派与 meta 记录，`/models` 的 `arch` 字段前端也在用（按架构过滤 LoRA）。
        SD 那条路已经删干净了，所以它现在只有一个返回值——但留着它，
        调用点的语义（"这个底模走哪条管线"）才不会散成硬编码。
        """
        return "qwen21"

    def is_xl(self, model: str = DEFAULT_MODEL) -> bool:
        return False

    @property
    def _pose(self) -> Any:
        return self._pool["pose"]

    @_pose.setter
    def _pose(self, v: Any) -> None:
        self._pool["pose"] = v

    def available_models(self) -> list[str]:
        """本机已经下好的底模（Qwen-only 下就是它一个）。"""
        if (self.root / "checkpoints" / DEFAULT_MODEL / "model_index.json").exists():
            return [DEFAULT_MODEL]
        return []

    def _check_model(self, model: str) -> Path:
        if model != DEFAULT_MODEL:
            raise ValueError(
                f"未知底模 {model}：这个工作台现在只有 {DEFAULT_MODEL}"
                f"（SD 权重已归档，见 README「底模」）")
        p = self.root / "checkpoints" / model
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
    def _touch(self, key: str) -> None:
        """把刚用过的管线移到 LRU 队尾（OrderedDict 的插入顺序即"最近使用"）。"""
        try:
            self._cache.move_to_end(key)
        except KeyError:                          # pragma: no cover
            pass
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
            pipe = self._load_qwen21(p)
            self._cache[key] = pipe
            log.info("%s 加载 %.1fs", key, time.perf_counter() - t0)
        return self._get(key)

    # ---------- Qwen-Image-2.1 ----------

    # ---------------- Qwen 的 LoRA 支持 ----------------

    def lora_dir(self) -> Path:
        """Qwen LoRA 目录（只放 **diffusers 原生格式** 的权重）。

        ComfyUI 格式的要先转：`python tools/convert_lora.py in.safetensors out.safetensors`
        —— 键名规则不同（`diffusion_model.` + `lora_A/B` vs
        `transformer.` + `lora.down/up`），不能直接挂。
        """
        return self.root / "lora"

    def available_loras(self) -> List[Dict[str, Any]]:
        """列出可用的 Qwen LoRA。

        **只认 diffusers 原生格式**（key 前缀 `transformer.`）——ComfyUI 格式
        挂上去会静默不生效（`load_lora_weights` 匹配不到任何层），比报错更坏。
        """
        out: List[Dict[str, Any]] = []
        d = self.lora_dir()
        if not d.is_dir():
            return out
        for f in sorted(d.glob("*.safetensors")):
            try:
                keys = _lora_keys(f)
            except Exception:                 # noqa: BLE001
                continue
            if not any(k.startswith("transformer.") for k in keys):
                continue
            out.append({
                "name": f.stem,
                "file": str(f),
                "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
                "tensors": len(keys),
            })
        return out

    def apply_loras(self, pipe, specs: Sequence[Dict[str, Any]]) -> List[str]:
        """把 LoRA 挂到 Qwen 管线上，返回实际生效的 adapter 名列表。

        `specs`：`[{"name": ..., "scale": 0.8}, ...]`

        **状态式设计**：挂上的 LoRA 会一直生效，直到下一次 `apply_loras`
        （传空列表）或 `unload_loras`。这样 LoRA 不必穿过每个工作流的
        参数链——设置一次，后续所有生成自动带上它。

        **实测可行**（2026-10-30）：fp8 量化后的 transformer **照样能挂 LoRA**——
        diffusers 的 LoRA 层是包在 `nn.Linear` 外面的，fp8 的 `w_fp8` 在
        `forward` 里反量化，LoRA 分支是并联加上去的，不冲突。
        实测 AnyAngle（rank 24，384 张量）挂上后 1440 个 LoRA 模块生效，
        出图只慢 1.1s（12.6s → 13.7s）。
        """
        active: List[str] = []
        # 先全卸，避免上一次调用残留
        for a in list(getattr(pipe, "_loaded_lora_adapters", []) or []):
            try:
                pipe.delete_adapters(a)
            except Exception:                 # noqa: BLE001
                pass
        try:
            pipe._loaded_lora_adapters = []
        except Exception:                     # noqa: BLE001
            pass

        for spec in specs:
            name = str(spec.get("name") or "").strip()
            scale = float(spec.get("scale", 1.0))
            if not name:
                continue
            f = self.lora_dir() / f"{name}.safetensors"
            if not f.is_file():
                f = self.lora_dir() / name
            if not f.is_file():
                raise FileNotFoundError(f"LoRA 文件不存在：{name}（见 models/lora/）")
            adapter = f"qwen_{name}"
            pipe.load_lora_weights(str(f), adapter_name=adapter)
            active.append((adapter, scale))
        if active:
            pipe.set_adapters([a for a, _ in active], [s for _, s in active])
            try:
                pipe._loaded_lora_adapters = [a for a, _ in active]
            except Exception:                 # noqa: BLE001
                pass
        # 记下当前状态：供 /loras 接口与前端显示。
        # 记**原始名**（不是 adapter 名）——前端拿这个名字去 /loras 列表里找。
        self._active_loras = [
            {"name": str(spec.get("name") or "").strip(), "scale": sc,
             "adapter": a}
            for (a, sc), spec in zip(active, [x for x in specs if x.get("name")])
        ]
        return [a for a, _ in active]

    def unload_loras(self, pipe) -> None:
        """卸掉所有 LoRA（回到基座模型）。"""
        for a in list(getattr(pipe, "_loaded_lora_adapters", []) or []):
            try:
                pipe.delete_adapters(a)
            except Exception:                 # noqa: BLE001
                pass
        try:
            pipe._loaded_lora_adapters = []
        except Exception:                     # noqa: BLE001
            pass
        self._active_loras = []

    def active_loras(self) -> List[Dict[str, Any]]:
        """当前生效的 LoRA（供 /loras 与前端显示）。"""
        return list(getattr(self, "_active_loras", []) or [])

    # ---------------- 快速模式（少步蒸馏） ----------------

    #: 快速模式用的少步蒸馏 LoRA（lightx2v/Qwen-Image-2.1-Lightning，64MB）。
    #: **diffusers 原生格式**，可直接 `load_lora_weights`。
    FAST_LORA = "qwen21_lightning"
    #: 快速模式的步数上限（Lightning 在 8 步就收敛，实测画质几乎不掉）
    FAST_STEPS = 8

    #: 扩图用的 Outpaint LoRA（ausboss/Qwen-Image-2.1-Outpaint-LoRA，152MB）。
    #: 标准权重不认识中性灰 padding 的语义，**扩图必须挂它**。
    OUTPAINT_LORA = "qwen21_outpaint"

    #: 姿态迁移用的 VNCCS PoseStudio LoRA（MIUProject，152MB，默认格式已转 diffusers）。
    #:
    #: 这是"完美保角色一致性"那条路的**全部秘密**：官方 ComfyUI 工作流里
    #: 一个 ControlNet 节点都没有，就是「这张 LoRA + 双图 + 一句固定指令」。
    #: 指令见 `generate._VNCCS_TEMPLATE`。
    #:
    #: 与 `FAST_LORA` 不能同时挂（`_qwen_call` 的 `fast_lora=False` 会跳过
    #: Lightning），也不要和 Outpaint 混用。
    VNCCS_LORA = "vnccs_pose"

    def set_fast_mode(self, on: bool) -> Dict[str, Any]:
        """开关快速模式（挂/卸 Lightning + 后续生成步数压到 8）。

        实测（768²/RTX4090）：24 步 17.3s → 8 步 9.3s（-46%），画质几乎不掉。
        """
        self.fast_mode = bool(on)
        if not self.fast_mode:
            try:
                pipe = self.txt2img_pipe(self.DEFAULT_MODEL)
                self.unload_loras(pipe)
            except Exception:                 # noqa: BLE001
                pass
        return {"fast_mode": self.fast_mode, "lora": self.FAST_LORA if on else None,
                "max_steps": self.FAST_STEPS if on else None}


    def _load_qwen21(self, root: Path):
        """Qwen-Image-2.1：单流 DiT（7B）+ Qwen3-VL 编码器，一个权重干三件事。

        文生图 / 指令改图（给 image 即走编辑路径）/ 原生 RGBA 透明输出。
        与 SD 系列不共用任何组件，所以是独立管线，不进共享 UNet 池。

        **必须分组件加载**（实测本机 32G 内存）：
        transformer 13.3G bf16 + text_encoder 16.3G bf16 一次性 from_pretrained
        会把内存打满（连 43G 目录一起读会直接 swap）。所以两块都走**预量化
        fp8 缓存**：transformer 6.7G、text_encoder 8.8G（见 tools/make_qwen_fp8.py）。

        **峰值从哪来**：改造前峰值 16.59GB（文生图 768²），几乎就是 bf16
        text_encoder 的 16.3GB 本身——它是整个管线的显存与内存双料峰值。
        量化到 fp8 后同一测法降到 ~10GB（见 docs/PLAN-QWEN-ONLY.md 的实测表）。

        torchao 的在线 fp8 在本机不可用（要求 torch≥2.11，本机 2.6），
        所以量化是一次性离线做的：`tools/make_qwen_fp8.py` 逐输出通道 scale，
        布局与 `_fp8_linear_cls` 一致。
        """
        from diffusers import QwenImage21Pipeline

        # **加载前先清场**：qwen 峰值 16.4GB，SDXL 的 ControlNet 管线约 13GB，
        # 两条并存必然挤爆。不清的话 _fit 会降级到 cpu offload（每步在 CPU/GPU
        # 间搬权重），用户看到的是「一直转、显存满」而不是报错。
        # 反方向也一样——generate.py 里 pose 系加载前会按需驱逐 qwen。
        self.ensure_headroom(need_gb=17.0)
        tor = self._load_qwen_transformer(root)
        te = self._load_qwen_text(root)
        pipe = QwenImage21Pipeline.from_pretrained(
            str(root), transformer=tor, text_encoder=te,
            torch_dtype=torch.bfloat16, local_files_only=True)
        # 编码完就把 text_encoder 挪走：它只在开头用一次，之后全程闲置。
        # 两块 fp8 都常驻时是 6.7+8.8=15.5GB，再加 VAE 与激活就压线了；
        # 需要「全程常驻」的机器可以设 MEOW_QWEN_OFFLOAD=0（每张省 1~2s 搬运）。
        if os.environ.get("MEOW_QWEN_OFFLOAD", "1") != "0":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        # 每次前向都是 768²~1024² 的 latent，VAE 解码是第二块峰值；
        # 切片只多 0.05s，换来解码峰值 -1.5GB（与 SDXL 同一结论）。
        try:
            pipe.vae.enable_slicing()
        except Exception:                     # noqa: BLE001
            pass
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _load_qwen_text(self, root: Path):
        """加载 Qwen3-VL 文本编码器：优先 fp8 缓存（8.8G），否则 bf16（16.3G）。

        fp8 缓存由 `tools/make_qwen_fp8.py` 产出，键布局与 transformer-fp8 相同
        （`w_fp8` + `w_scale` 顶替 `weight`）。加载路径也照搬：
        `init_empty_weights` 建骨架 → 逐键挂 buffer/Parameter → 换 Linear 的类。

        不能走 `from_pretrained`：它对 0 长度占位的 `weight` 做形状校验，
        报一屏 size mismatch（这条坑在 transformer 上已经踩过一次）。
        """
        from safetensors.torch import load_file

        fp8_dir = root / "text_encoder-fp8"
        use_fp8 = (fp8_dir / "model.safetensors").is_file() and \
            os.environ.get("MEOW_QWEN_TEXT_FP8", "1") != "0"
        if not use_fp8:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration.from_pretrained(
                str(root / "text_encoder"), torch_dtype=torch.bfloat16,
                local_files_only=True,
                attn_implementation="sdpa")

        from accelerate import init_empty_weights

        cls = _qwen_vl_cls(fp8_dir)
        config = cls.config_class.from_pretrained(str(fp8_dir))
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = cls(config)

        sd = load_file(str(fp8_dir / "model.safetensors"), device="cpu")
        linear_keys = sorted({k[: -len(".w_fp8")] for k in sd if k.endswith(".w_fp8")})
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for t in linear_keys:
            model.get_submodule(t).weight = torch.nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=dev), requires_grad=False)
        for k, v in sd.items():
            prefix, name = k.rsplit(".", 1)
            mod = model.get_submodule(prefix)
            if name in ("w_fp8", "w_scale"):
                mod.register_buffer(name, v.to(dev), persistent=False)
            elif name in mod._parameters and mod._parameters[name] is not None:
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
            raise RuntimeError(
                f"qwen text_encoder 有 {len(stuck)} 个参数停在 meta：{stuck[:5]}")
        log.info("qwen text_encoder fp8 就绪：%d 个 Linear", len(linear_keys))
        return model.eval()

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

    # ---------- 能力表 ----------

    # 架构 → 支持的工作流集合。**这张表是「某个底模能不能跑这个节点」的唯一判据**：
    #   · generate.py 各函数入口调 `require()`，提前报人话错误；
    #   · /models 把它发给前端，节点上的底模下拉据此过滤。
    _ARCH_WORKFLOWS = {
        # Qwen 十个工作流全支持，但分两类（近似的那四个见 _APPROX）。
        "qwen21": frozenset({
            "txt2img", "img2img", "inpaint", "poseswap", "matte",
            "ai_repaint", "pixelify", "pose_clone", "sprite_swap", "topixel",
            "outpaint", "outfit_swap", "face_swap", "vnccs_pose",
        }),
    }

    # 架构 → **近似实现**的工作流集合。
    #
    # 「近似」的定义很具体：**原工作流的硬约束在这个底模上不存在，只能用软引导
    # 或后处理凑出同样的产物**。用户必须一眼看出差别，否则会拿「Qwen 的局部重绘」
    # 去期待 SD inpaint 那种「涂哪儿改哪儿」。
    #
    #   · inpaint      —— 没有 mask 接口，靠「整图指令改图 + 蒙版贴回」，
    #                     选区外保证不变，**选区内不保证只改那一点**。
    #   · ai_repaint   —— 同上（同一个机制）。
    #   · poseswap     —— 没有 ControlNet，姿态靠「骨架图当参考图 + 指令」软引导，
    #                     实测腿部常被参考图的站姿覆盖（PLAN §2.8）。
    #   · sprite_swap  —— 同上，逐帧软引导（也逐帧 28s 级别的成本）。
    #
    # 其余六个是**原生**：txt2img / img2img（指令改图是 Qwen 的正牌用法）/
    # matte / pixelify / topixel 不跑模型，pose_clone 走双图条件（本来就是多图原生）。
    # 「近似」= 该工作流原本依赖的硬约束在 Qwen 上不存在，只能用软引导凑。
    # inpaint / ai_repaint 已移出这张表：它们现在走**真蒙版路径**
    # （workflows/qwen_mask.py），模型看得见蒙版、指令能被准确执行。
    _APPROX = {
        "qwen21": frozenset({"poseswap", "sprite_swap"}),
    }

    # 工作流 → 中文名（报错时用，别让用户看到 wpc/wss 这种内部代号）
    _WF_LABEL = {
        "txt2img": "文生图", "img2img": "图生图", "inpaint": "局部重绘",
        "poseswap": "姿态序列", "matte": "抠图", "ai_repaint": "局部 AI 重绘",
        "pixelify": "像素转换", "pose_clone": "角色迁移",
        "sprite_swap": "序列帧替换", "topixel": "转像素",
        "outpaint": "扩图", "outfit_swap": "换装", "face_swap": "换脸",
        "vnccs_pose": "姿态迁移(VNCCS)",
    }

    def supports(self, model: str, workflow: str) -> bool:
        """这个底模能不能跑这个工作流。未知架构一律返回 False（保守）。"""
        return workflow in self._ARCH_WORKFLOWS.get(self.arch(model), frozenset())

    def workflows_of(self, model: str) -> list[str]:
        """这个底模支持的全部工作流（给前端的下拉过滤用）。"""
        return sorted(self._ARCH_WORKFLOWS.get(self.arch(model), frozenset()))

    def approx_of(self, model: str) -> list[str]:
        """这个底模上**近似实现**的工作流（前端要在节点上打标）。"""
        return sorted(self._APPROX.get(self.arch(model), frozenset()))

    def is_approx(self, model: str, workflow: str) -> bool:
        """这个组合是不是「用软引导/后处理凑出硬约束」的近似实现。"""
        return workflow in self._APPROX.get(self.arch(model), frozenset())

    def require(self, model: str, workflow: str) -> None:
        """不支持的组合直接抛人话错误，别让它崩在 diffusers 内部的组件校验里。"""
        if self.supports(model, workflow):
            return
        label = self._WF_LABEL.get(workflow, workflow)
        arch = self.arch(model)
        raise ValueError(
            f"「{label}」不支持底模 {model}（架构 {arch}）。"
            f"这个底模能跑：{'、'.join(self._WF_LABEL.get(w, w) 
                                    for w in self.workflows_of(model)) or '无'}。")

    RECOMMEND = {
        # Qwen-Image-2.1：**默认不像素化**（pixel/palette 归零）。
        # 它直接就能画出像样的像素风（实测 768² / 24 步直接出 16-bit 猫骑士），
        # 再过一遍后处理反而把它的色阶优势抹平。
        #
        # **默认 768² 而不是 1024²**：Qwen 是逐帧路线（没有 ControlNet /
        # AnimateDiff），1024² 的图 token 是 768² 的 1.78 倍，实测 768²/20 步
        # 24~27s 一帧、1024² 要 45s 上下；而精灵成品最后要缩到 32~128px，
        # 1024² 那点细节在缩图上根本看不见。要画立绘再手动调大。
        #
        # **步数 16**：2026-10-30 重测的下限（关掉真 CFG 之后）。
        # 旧结论「低于 16 步出糊块」是在**真 CFG 开着**时测的——那时每个 step
        # 跑两遍 transformer，采样轨迹被 guidance 改写，低步数会散架。
        # 现在单次前向，12 个提示词 × 5 档步数重测：
        #   8 步 7.3s / 12 步 8.7s / 16 步 10.1s / 20 步 11.5s / 24 步 12.7s
        # 「缺件」在所有档位都是 0，但 128px 缩图上 8~12 步的**浅色主体**
        # （白猫、白甲、幽灵）对比度不够、细节发白；16 步起才稳定。
        # 姿态贴合度：16 步 95px vs 24 步 106px（不差）。
        # 例外：**白/浅色主体建议 20~24 步**，那是唯一需要往上调的场景。
        # **cfg = 1.0**：真 CFG 关掉 → 每个 step 只跑一次 transformer。
        # 实测（2026-10-30）：768²/24 步的 21.5s 里，DiT 前向占 86%，
        # 而 true_cfg_scale > 1 会让前向翻倍（48 次 vs 24 次）。
        # 关掉后 13.0s（-40%），6 个提示词的对照里画质没掉（甚至更精细）。
        # 用户要更强的负向约束时把 cfg 调到 > 1，代价是时间翻倍。
        "qwen21": {"width": 768, "height": 768, "steps": 16, "cfg": 1.0,
                   "pixel": 0, "palette": 0},
    }

    def recommend(self, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """这个底模在**精灵管线**上的推荐参数（尺寸/步数/像素格/色数）。

        额外带 `max_candidates`：这个底模在推荐尺寸下，W4（逐帧候选）最多能
        批量几个候选。前端拿它当输入框上限——实测 Qwen 在 768² 上多一个候选
        就是多 5GB 峰值与 28s，靠用户自己猜是猜不出来的。

        """
        rec = dict(self.RECOMMEND.get(self.arch(model), self.RECOMMEND["qwen21"]))
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
            size = self.RECOMMEND.get(self.arch(model), self.RECOMMEND["qwen21"])["width"]
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
    def evict_other_models(self, model: str, keep: Sequence[str] = ()) -> None:
        """切到某个底模时，把**其它底模**的管线全部释放。

        判据用 `key.split(":")` 里的**完全相等**，不能用子串包含——
        `sd15` 是 `sd15-inpaint` 的前缀，子串匹配会把正在用的那条也误杀。
        （SD 已删，这条判据保留：Qwen 的键是 `txt2img:qwen-image-2.1`，
        冒号前缀的写法沿用，避免以后加第二个底模时又踩一次。）

        `keep`：调用方马上要用、绝不驱逐的前缀。
        """
        dropped: list[str] = []
        for key in list(self._cache):
            if model in key.split(":") or any(k in key for k in keep):
                continue
            self._cache.pop(key, None)
            dropped.append(key)
        if dropped:
            log.info("底模切到 %s：驱逐其它底模的管线 %s", model, ", ".join(dropped))
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
            gc.collect()
            torch.cuda.empty_cache()
            log.info("显存紧张（空闲 %.1fGB < %.1fGB），已释放管线 %s",
                     self._free_gb(), need_gb, key)

    def free(self, keep: tuple[str, ...] = ()) -> None:
        """释放显存（清掉不在 keep 里的缓存管线）。"""
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
            **self.vram(),
        }
