"""Prompt Enhancement：拿**图像模型自己的 Qwen3-VL 文本编码器**当 LLM 用。

**这不是我提的方案，是官方的做法**：VNCCS 的 PE 走 ComfyUI 原生 `TextGenerate`，
喂的就是 QI2 的 CLIP（`character_creator_v2.py:1293`）—— 那个 CLIP 是
Qwen3-VL 8B，跟我们 `engine.py:478` 加载的 `Qwen3VLForConditionalGeneration`
是同一个东西。所以 PE **不需要任何新权重**。

为什么官方这么选（两条都成立）：
- **同分布**：改写出来的措辞，跟图像模型训练时见到的文本一致，风格 LoRA 吃得更准；
- **不重复加载**：编码器本来每次出图都要加载一次来编码 prompt，PE 只是让它
  多干一会儿，不是额外再拉一个模型进来。

采样参数照抄官方 `character_creator_v2.py:159-174`（`QI2_TEXT_GENERATION_DEFAULTS`）。
system prompt 读随项目发行的官方文件 `character_template/qwen_image_21_t2i_system_prompt.txt`。

> ⚠️ **代价**：PE 要跑自回归解码，文本编码器得**常驻 GPU**；而我们的管线为了
> 省显存是「编码完就挪走」（`engine.py:439`，两块 fp8 常驻 15.5GB 已经压线）。
> 24GB 上跟 transformer 抢是真的，8B 解码 2048 token 也不快。
> 所以每个入口都做了**失败降级**：PE 挂了就退回未扩写的 prompt，不阻断出图。
>
> **2026-11-08 修过一次真故障**：报 `Expected all tensors to be on the same
> device ... wrapper_CUDA__index_select`。原因是 `enable_model_cpu_offload()`
> 的 `CpuOffload` 钩子在 forward 前后搬模块、但不搬输入，与「fp8 编码器直接建在
> cuda 上」叠加后 input_ids 落在 cpu、权重在 cuda。现在 `_OnDevice` 会在 generate
> 期间**摘掉钩子**、自检每个张量都在同一张卡上，结束后把钩子挂回去。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

from workflows import paths

log = logging.getLogger(__name__)

#: 官方 `character_creator_v2.py:159-174`
QI2_TEXT_DEFAULTS: Dict[str, Any] = {
    "max_length": 2048,
    "temperature": 0.3,
    "top_k": 64,
    "top_p": 0.95,
    "min_p": 0.05,
    "repetition_penalty": 1.05,
    "seed": 0,
    "presence_penalty": 0.0,
}

#: 官方那两份提示词模板（随项目发行，已搬进 web/public/vnccs/character_template/）
SYSTEM_PROMPT_FILES = {
    # 造角色（T2I）：把请求改写成"你看到的画面"的一段英文描述
    "t2i": "qwen_image_21_t2i_system_prompt.txt",
    # 字段扩写：把 character_fields 的 JSON 扩成精确英文描述
    "fields": "qi2_prompt_rewriter.txt",
    # 换装（编辑式）
    "edit": "qi2_edit_prompt_rewriter.txt",
}


#: 官方那段「一律英文」换成「一律中文」用的替换段落。
#:
#: 只换这一段，八步推理与 JSON 输出格式原样保留 —— 那是模型真正需要的结构约束，
#: 重写一遍只会引入偏差。Qwen-Image 的文本编码器本来就是 Qwen3-VL，中文是原生
#: 能力，没有任何理由让中文用户看英文提示词。
_ZH_LANGUAGE_SECTION = (
    "## 输出语言\n\n"
    "描述一律用简体中文书写，无论请求用的是什么语言。唯一的例外是画面里要出现的文字：\n"
    "那部分保持它原本的文字与字形。\n"
)


def _with_chinese_output(text: str) -> str:
    """把官方 system prompt 的「一律英文」段落换掉（找不到就补在末尾）。

    模板里那一段以 `## Language` 开头、下一个 `## ` 结束；我们只做这一处替换，
    其余一字不动（官方文件也不会被改写，替换只发生在内存里）。
    """
    if not text:
        return text
    head, sep, tail = text.partition("## Language")
    if not sep:
        return text.rstrip() + "\n\n" + _ZH_LANGUAGE_SECTION
    nxt = tail.find("\n## ")
    rest = "" if nxt < 0 else tail[nxt:]
    return head + _ZH_LANGUAGE_SECTION + rest


def system_prompt_of(kind: str = "t2i", lang: str = "en") -> str:
    """读官方的 system prompt 文件。文件缺失返回空串（**不自己编一套**）。

    `lang="zh"` 时只把「输出语言」那段换成中文，其余照官方。
    """
    name = SYSTEM_PROMPT_FILES.get(kind)
    if not name:
        return ""
    path = os.path.join(str(paths.VNCCS_TEMPLATE_DIR), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        log.warning("官方提示词模板读不到：%s", path)
        return ""
    return _with_chinese_output(text) if lang == "zh" else text


def _tokenizer_of(pipe: Any):
    """Qwen 管线的 tokenizer 在不同版本里挂的位置不一样，两处都试。"""
    tok = getattr(pipe, "tokenizer", None)
    if tok is not None:
        return tok
    proc = getattr(pipe, "processor", None)
    return getattr(proc, "tokenizer", None) if proc is not None else None


class _OnDevice:
    """把文本编码器临时搬到 GPU 上跑生成，用完放回原处。

    **为什么要放回**：管线开了 cpu offload 时，编码器平时住在 CPU（这就是
    我们省显存的办法）。PE 完不还原，后面每次出图都多占 8.8G，
    用户会以为"跑了几次之后怎么变慢了"。

    **为什么要摘掉 accelerate 钩子**：`enable_model_cpu_offload()` 会在组件上
    挂 `CpuOffload`，它在 forward 前后把模块整体搬 CPU/GPU，但**不保证输入
    跟着走**。而我们的 fp8 文本编码器是直接建在 cuda 上的（`engine._load_qwen_text`），
    两者一叠加就出现「权重被钩子搬到 cuda、input_ids 还在 cpu」——
    报 `Expected all tensors to be on the same device ... wrapper_CUDA__index_select`。
    所以这里把钩子摘下来，整个 generate 期间自己独占显卡，结束再原样挂回去。
    """

    def __init__(self, module: Any, device: Optional[str] = None,
                 pipe: Any = None):
        self.mod = module
        self.pipe = pipe
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.origin: Optional[torch.device] = None
        self.hook: Any = None

    def __enter__(self):
        from accelerate.hooks import remove_hook_from_module

        self.hook = getattr(self.mod, "_hf_hook", None)
        if self.hook is not None:
            try:
                remove_hook_from_module(self.mod, recurse=True)
            except Exception:                     # noqa: BLE001
                self.hook = None                  # 摘不掉就按原样走（下面会校验设备）
        try:
            self.origin = next(self.mod.parameters()).device
        except StopIteration:                     # 空权重（fp8 骨架）
            self.origin = None
        self.mod.to(self.device)
        stray = _off_device(self.mod, self.device)
        if stray:
            # fp8 的 w_fp8 / w_scale 是 buffer，漏一个就又是混设备。
            # 与其让 generate 抛一句看不懂的 index_select，不如在这里说清楚；
            # 但要先把钩子和设备还原，否则管线后面的出图会一直用错设备。
            self.__exit__()
            raise RuntimeError(
                f"文本编码器没能整体搬到 {self.device}，仍有 {len(stray)} 个张量在 "
                f"{sorted(set(stray))[:4]}（示例：{stray[:3]}）")
        return self.mod

    def __exit__(self, *exc):
        from accelerate.hooks import add_hook_to_module

        try:
            if self.origin is not None:
                self.mod.to(self.origin)
        except Exception:                         # noqa: BLE001
            pass
        if self.hook is not None:
            try:
                add_hook_to_module(self.mod, self.hook)
            except Exception:                     # noqa: BLE001
                pass
        # **钩子必须真的挂回去**：文本编码器是 `components` 里很靠前的一个，
        # 它一旦没有 `_hf_hook`，`DiffusionPipeline._execution_device` 会退回
        # `self.device`（cpu），下一轮出图就报
        # `Cannot generate a cpu tensor from a generator of type cuda`。
        # 挂不回去就整条链重建一次，宁可多花 1 秒也不能留个坏状态。
        if not hasattr(self.mod, "_hf_hook") and self.pipe is not None:
            try:
                self.pipe.enable_model_cpu_offload(
                    device=getattr(self.pipe, "_offload_device", None)
                    or ("cuda" if torch.cuda.is_available() else "cpu"))
            except Exception as exc:              # noqa: BLE001
                log.warning("PE 结束后没能恢复 cpu offload 链：%s", exc)
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return False


def _embed_device(model: Any) -> Optional[torch.device]:
    """输入张量该去的设备：优先取 token 嵌入层的权重（`model.device` 只认第一个参数）。"""
    try:
        emb = model.get_input_embeddings()
        w = getattr(emb, "weight", None)
        if w is not None and not getattr(w, "is_meta", False):
            return w.device
    except Exception:                             # noqa: BLE001
        pass
    return None


def _weight_gb(module: Any) -> float:
    """模块占多少 GB（参数 + buffer，跳过 meta）。用来判断这块卡此刻装不装得下。"""
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        if getattr(t, "is_meta", False):
            continue
        total += t.numel() * t.element_size()
    return total / 2 ** 30


def _off_device(module: Any, device: Any) -> List[str]:
    """找出没在同一张设备上的参数/缓冲区（名字 + 设备），用于自检与报错。"""
    want = str(device).split(":")[0]
    out: List[str] = []
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        d = getattr(t, "device", None)
        if d is None or getattr(t, "is_meta", False):
            continue
        if str(d).split(":")[0] != want:
            out.append(f"{name}@{d}")
    return out


def _free_vram_for_pe(pipe: Any) -> bool:
    """PE 前只**清缓存**，不碰 offload 链。

    ⚠️ 踩过的坑：这里以前调 `pipe.maybe_free_model_hooks()`，它的实现是
    `remove_all_hooks()` + `enable_model_cpu_offload()`。只要中间那步失败，
    管线就**彻底没有钩子**、组件全留在 cpu —— 下一次出图报
    `Cannot generate a cpu tensor from a generator of type cuda`，
    而真正的凶手（PE）已经在日志里一闪而过。
    管线每轮 `__call__` 结束本来就会自己 offload，这里不需要再动它。
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def _template(tok: Any, messages: List[Dict[str, str]]) -> str:
    """套 chat 模板。**必须关掉 thinking**：Qwen3 默认会输出一大段思考，
    官方也是显式 `thinking=False`（`qwen_vl.py:53-64` 的 `enable_thinking=false`）。

    不同 transformers 版本对 `enable_thinking` 的支持不一样，不支持就退回去。
    """
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate_text(
    pipe: Any,
    request: str,
    system_prompt: str = "",
    *,
    max_new_tokens: Optional[int] = None,
    **overrides: Any,
) -> str:
    """用管线的文本编码器跑一次文本生成。

    **任何失败都返回空串**，让调用方退回未扩写的 prompt —— PE 是增强，
    不该成为出图的必要条件。
    """
    te = getattr(pipe, "text_encoder", None)
    tok = _tokenizer_of(pipe)
    if te is None or tok is None:
        log.warning("PE 跳过：拿不到 text_encoder / tokenizer")
        return ""
    req = str(request or "").strip()
    if not req:
        return ""

    params = dict(QI2_TEXT_DEFAULTS)
    params.update({k: v for k, v in overrides.items() if v is not None})
    n_new = int(max_new_tokens or params.pop("max_length", 2048))
    # 官方那几个采样参数里的 seed / presence_penalty 我们不用（前者要整套
    # generator，后者 transformers 不一定支持），只留真正会传进 generate 的。
    gen: Dict[str, Any] = {
        "max_new_tokens": n_new,
        "do_sample": True,
        "temperature": float(params.get("temperature", 0.3)),
        "top_k": int(params.get("top_k", 64)),
        "top_p": float(params.get("top_p", 0.95)),
        "min_p": float(params.get("min_p", 0.05)),
        "repetition_penalty": float(params.get("repetition_penalty", 1.05)),
    }

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": req})

    try:
        text = _template(tok, messages)
        _free_vram_for_pe(pipe)
        need = _weight_gb(te) + 1.5                 # 权重 + 2048 token 解码的 KV/激活
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            if free < need:
                # 不是"坏了"，是这块卡此刻装不下：明确跳过，别让用户以为出 bug
                log.info("PE 跳过：可用显存 %.1fGB < 需要 %.1fGB", free, need)
                return ""
        with torch.no_grad():
            with _OnDevice(te, pipe=pipe) as model:
                # 输入必须落在**权重真正所在的设备**上：fp8 骨架里有空 Parameter，
                # `model.device` 取第一个参数不一定准，嵌入层最可靠。
                dev = _embed_device(model) or model.device
                inputs = {k: (v.to(dev) if torch.is_tensor(v) else v)
                          for k, v in tok(text, return_tensors="pt").items()}
                # transformers 版本差异：min_p 是后加的，老版本没有就退掉
                for drop in ((), ("min_p",)):
                    kwargs = {k: v for k, v in gen.items() if k not in drop}
                    try:
                        out = model.generate(**inputs, **kwargs)
                        break
                    except TypeError as exc:
                        if "min_p" not in str(exc) or drop:
                            raise
                        continue
                cut = inputs["input_ids"].shape[-1]
                seq = out[0][cut:] if out.ndim > 1 else out[cut:]
        return tok.decode(seq, skip_special_tokens=True).strip()
    except Exception as exc:                      # noqa: BLE001
        log.warning("Prompt Enhancement 失败（退回原 prompt）：%s", exc)
        log.debug("PE 失败详情", exc_info=True)
        return ""


def enhance(
    pipe: Any, prompt: str, kind: str = "t2i", system_prompt: Optional[str] = None,
    lang: str = "en", **overrides: Any,
) -> str:
    """PE：改写一段 prompt。失败返回空串（**调用方要自己兜底**）。

    `lang="zh"` → 用「输出中文」版的官方模板，改出来的描述就是中文。
    """
    if system_prompt is None:
        system_prompt = system_prompt_of(kind, lang)
    return generate_text(pipe, prompt, system_prompt, **overrides)
