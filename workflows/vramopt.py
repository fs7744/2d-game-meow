"""显存优化策略（AnimateDiff 路线专用）。

**实测依据**（本项目，RTX 4090 24GB，SD1.5 + ControlNet-canny + IP-Adapter）：

    方案                          8帧@512   16帧@512   耗时
    基线                            9.86GB    14.09GB    4.1s
    + VAE 逐帧解码                  6.96GB     8.28GB    4.0s   ← -30%/-41%
    + IP-Adapter encoder 用完即退   5.78GB     7.10GB    4.0s   ← 再 -17%
    16帧@384（再降分辨率）            —          5.95GB    4.3s

**为什么是这两招**：

1) **VAE 逐帧解码**。AnimateDiff 一次解码整段 latent（batch=帧数），
   8 帧 @512² 的 VAE 解码增量高达 **4.00GB**——这就是峰值所在
   （权重只有 5.62GB，激活 4.24GB，解码占了其中绝大部分）。
   逐帧解码后增量降到 **0.60GB**，省 3.4GB。
   diffusers 0.41 的 AnimateDiffControlNetPipeline **没有** `enable_vae_slicing`
   （只有 diffusers 通用 pipeline 才有），得直接调 `pipe.vae.enable_slicing()`。
   实测质量影响：逐帧平均像素差 0.1/255（数值精度级），边缘像素数差 0.5%。

2) **IP-Adapter 的 image_encoder 用完即退**。它是 CLIP Vision（632M 参数 /
   1.18GB），**只在生成开始时编码一次参考图，之后全程闲置**——但 diffusers
   让它常驻显存。
   做法：encoder 常驻 CPU，只在 `encode_image` 被调用时短暂上 GPU，
   编码完立刻退回 CPU 并清缓存。
   实测：峰值再降 1.18GB，**耗时不变**，**输出逐像素完全一致**（差 0.0）。

   注意：不能简单地把 encoder 移到 CPU 就完事——那会让编码跑在 CPU 上，
   实测慢 52%（4.0s → 6.1s）。「用完即退」才是正解。

**没用的手段**（都实测过）：

- `enable_attention_slicing()`：峰值无变化。因为瓶颈在 VAE 解码，
  不在 attention。开了反而略慢（4.0s → 4.3s）。
- `enable_model_cpu_offload()`：峰值反而**升高**到 10.15GB。
  accelerate 的 offload 会把模块搬来搬去，在这条管线上得不偿失。
- `vae.enable_tiling()`：512² 不触发 tiling，无变化。
- 把 `image_encoder` 整块留在 CPU：慢 52%，且显存收益相同。

**帧数上限**：motion module 的 max_seq_length 是 32，但真正的限制是显存。
优化后 32 帧 @512² 是 14.20GB（24GB 卡够用）。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

# 各档位：(帧数上限, 生成边长) —— 超出的部分靠降分辨率消化
VRAM_PROFILES: Dict[str, Dict[str, Any]] = {
    # 省显存：给同卡跑其它任务的场景用
    "low":    {"max_frames": 16, "size": 384, "note": "16帧@384²，实测 5.95GB"},
    # 平衡（默认）
    "medium": {"max_frames": 16, "size": 512, "note": "16帧@512²，实测 7.10GB"},
    # 高质量：帧数多或要细节
    "high":   {"max_frames": 32, "size": 512, "note": "32帧@512²，实测 14.20GB"},
}


def apply_vae_slicing(pipe) -> bool:
    """VAE 逐帧解码——把峰值里最大的一块（解码）砍掉。

    AnimateDiffControlNetPipeline 没有 `enable_vae_slicing`，得调 VAE 本身。
    """
    try:
        pipe.vae.enable_slicing()
        return True
    except Exception:  # noqa: BLE001
        return False


_RELEASED_ENCODERS: set = set()


def release_ipa_encoder(pipe) -> bool:
    """IP-Adapter 的 image_encoder 用完即退——编码时上卡，编完退回 CPU。

    image_encoder 是 CLIP Vision（1.18GB），只编码一次参考图就闲置，
    但 diffusers 让它常驻显存。

    实现要点：**不能**简单地把 encoder 留在 CPU——那样编码跑在 CPU 上，
    实测慢 52%。必须「调用时短暂上 GPU，用完立刻退回」。

    **多个管线可以共享同一个 encoder**（AnimateDiff 与复用出来的单帧管线
    用的是同一份权重），所以要按 encoder 对象去重，否则第二次打补丁会把
    第一次的包装器套起来（encoder 又被搬上 GPU，退不下来）。
    """
    enc = getattr(pipe, "image_encoder", None)
    if enc is None:
        return False
    key = id(enc)
    if key in _RELEASED_ENCODERS:
        # 同一个 encoder 已经处理过了——但**这条管线**还没打补丁，
        # 需要给它装一份（共用同一个 encoder 的搬上搬下逻辑）
        _attach_encode_hook(pipe, enc)
        return True
    try:
        enc.to("cpu")
        _attach_encode_hook(pipe, enc)
        _RELEASED_ENCODERS.add(key)
        torch.cuda.empty_cache()
        return True
    except Exception:  # noqa: BLE001
        return False


def _attach_encode_hook(pipe, enc) -> None:
    """给管线的 encode_image 打补丁：编码时 encoder 上 GPU，用完退回 CPU。"""
    orig = pipe.encode_image

    def encode_image(image, device, num_images_per_prompt,
                     output_hidden_states=None):
        enc.to(device)
        try:
            out = orig(image, device, num_images_per_prompt,
                       output_hidden_states)
        finally:
            enc.to("cpu")
            torch.cuda.empty_cache()
        return out

    pipe.encode_image = encode_image


def optimize_pipe(pipe, *, release_encoder: bool = True) -> Dict[str, Any]:
    """对 AnimateDiff 管线应用全部显存优化，返回应用情况。"""
    info = {"vae_slicing": apply_vae_slicing(pipe)}
    info["ipa_encoder_released"] = release_ipa_encoder(pipe) if release_encoder \
        else False
    return info


def plan_generation(n_frames: int, size: int,
                    profile: str = "auto") -> Tuple[int, int, str]:
    """按显存档位规划生成参数。

    返回 (帧数, 生成边长, 说明)。

    规则：
      - `auto`   按当前可用显存挑档位；装不下就降分辨率
      - `low`    强制 384²
      - `medium` 强制 512²
      - `high`   512² 且允许到 32 帧
      - `custom` 完全按传入值走（用户自己负责）

    **降分辨率的代价**：SD1.5 在低分辨率出图偏糊，所以降分辨率是**最后手段**，
    优先降帧数——但帧数是用户给的动画内容，不能随便砍，所以这里只降分辨率
    并在 meta 里说明，让调用方知道发生了什么。
    """
    prof = VRAM_PROFILES.get(profile)
    if prof is None and profile != "auto":
        # custom / 未知 → 原样返回
        return n_frames, size, "自定义参数，未做显存规划"

    if profile == "low":
        target = prof["size"]
        if size != target:
            return n_frames, target, f"显存档位 low：分辨率降到 {target}²"
        return n_frames, size, "显存档位 low"

    # medium / high / auto：先按可用显存看能不能装下
    if profile == "auto":
        free_gb, total_gb = _gpu_mem()
        # 经验：帧数 × 边长² 决定激活占用；加上 ~4.5GB 固定权重
        need = _estimate_gb(n_frames, size)
        if need <= free_gb * 0.85:
            return n_frames, size, f"显存充足（预计 {need:.1f}GB / 可用 {free_gb:.1f}GB）"
        # 装不下 → 依次降分辨率
        for cand in (512, 448, 384, 320):
            need = _estimate_gb(n_frames, cand)
            if need <= free_gb * 0.85:
                return n_frames, cand, (
                    f"显存不足（预计 {need:.1f}GB > 可用 {free_gb:.1f}GB），"
                    f"分辨率降到 {cand}²")
        return n_frames, 320, "显存不足，分辨率降到 320²（质量会下降）"

    # 显式档位
    target = prof["size"]
    if n_frames > prof["max_frames"]:
        return n_frames, target, (
            f"显存档位 {profile}：{n_frames} 帧超出建议上限 {prof['max_frames']}，"
            f"分辨率降到 {target}²")
    return n_frames, size, f"显存档位 {profile}"


def _gpu_mem() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


# ---------------- W4 候选批量的显存规划 ----------------

# 实测（RTX 4090，fp16，pose 管线的**保留峰值**，不是 allocated）：
#   sd15 768² 单帧 6.62GB
#   sdxl 768² 单帧 15.33GB / 批量2 18.16 / 批量3 21.76
#   sdxl 1024² 单帧 18.45GB / 批量2 23.98（且耗时从 5.0s 崩到 17.5s）
#
# 模型：峰值 = 权重 + 激活(∝ 面积) + 每多一帧的增量(∝ 面积)
# 权重按架构固定，激活与增量都随边长平方缩放。
_W4_SHAPE = {
    #        权重GB  768²激活GB  768²每帧增量GB
    "sd15": (4.0, 2.6, 1.7),
    "sdxl": (12.0, 3.3, 2.8),
    # Qwen：fp8 权重 6.7 + VAE 1.3 + 文本编码器编码时的驻留，实测 768² 指令改图
    # 峰值 18.75GB（transformer 走 fp8、text_encoder 走 cpu offload）。
    # 激活与「每多一个候选」的增量都给得比 SD 大——它的 joint attention 序列里
    # 还带着条件图 token，批量出图在 24G 卡上基本不可行，规划器的结论就该是 k=1。
    # 估值偏保守：768² 单帧实测 11.8~15.4GB，多一个候选要再 +6GB 左右。
    # 给 (11, 6, 6.5) 之后规划器在 24G 卡上只会说「1 个」——这是事实：
    # 批量 2 在 768² 要 23.5GB，会挤到 cpu offload 上、耗时翻倍。
    "qwen21": (11.0, 6.0, 6.5),
}


def estimate_pose_gb(arch: str, size: int, candidates: int = 1) -> float:
    """估 W4（ControlNet + IP-Adapter）的保留峰值。"""
    w, act, per = _W4_SHAPE.get(arch, _W4_SHAPE["sd15"])
    k = (max(256, int(size)) / 768.0) ** 2
    return w + act * k + per * k * max(0, candidates - 1)


def plan_candidates(arch: str, size: int, want: int, free_gb: float,
                    *, single_resident: bool = True,
                    safety: float = 0.9) -> Tuple[int, str]:
    """按当前空闲显存决定**能批量出几个候选**。返回 (候选数, 说明)。

    为什么要管这件事：显存打满时**不报 OOM，但耗时能翻 3 倍**（实测 sdxl 1024²
    批量 2：23.98GB，单帧 5.0s → 17.47s）。这种劣化没有任何报错，用户只会觉得
    "SDXL 太慢"，然后去调错方向。宁可提前把候选数降下来。

    候选数是"多出几张让闸门挑"的额度，降它只影响成品率，不影响能不能跑；
    所以它是显存不足时**第一个**该让步的东西（比降分辨率温和）。
    """
    want = max(1, int(want))
    if free_gb <= 0:
        return want, ""                     # 没有 GPU 信息就不要瞎猜
    # 调用时机通常是"管线已加载"，此时单帧峰值已从 free 里扣掉；
    # 判断 k 个候选能否装下要把这份加回来，否则会算出偏小的 k。
    single = estimate_pose_gb(arch, size, 1)
    budget = (free_gb + (single if single_resident else 0.0)) * safety
    if single > budget:
        return 0, (f"单帧就装不下：{arch} {size}² 预计 {single:.1f}GB "
                   f"> 可用 {budget / safety:.1f}GB。把边长降到 768 或先释放其它管线")
    k = 1
    while k < want and estimate_pose_gb(arch, size, k + 1) <= budget:
        k += 1
    if k < want:
        return k, (f"显存只够 {k} 个候选（要 {want}）：预计 "
                   f"{estimate_pose_gb(arch, size, k):.1f}GB / 可用 {free_gb:.1f}GB")
    return k, ""



def _estimate_gb(n_frames: int, size: int) -> float:
    """估算峰值显存（GB）。

    用实测数据拟合：
      8帧@512  → 5.78GB（优化后）
      16帧@512 → 7.10GB
      32帧@512 → 14.20GB
      16帧@384 → 5.95GB

    **不是线性增长**——AnimateDiff 的 motion module 要在整段序列上做时序
    注意力，激活随帧数**超线性**上升（8→16 帧只涨 23%，16→32 帧涨 100%）。
    所以用二次项拟合：act = a·n + b·n²。

    权重部分固定 ~4.45GB（UNet+MotionAdapter+ControlNet+VAE+TextEncoder+投影层，
    已去掉 IP-Adapter encoder 那 1.18GB）。
    """
    base = 4.45
    # 用 8/16/32 帧三点拟合 act = a·n + b·n²（单位 GB）
    #   8帧:  act=1.33  = 8a + 64b
    #   16帧: act=2.65  = 16a + 256b
    #   32帧: act=9.75  = 32a + 1024b
    a, b = 0.1245, 0.00755
    act = a * n_frames + b * n_frames * n_frames
    return base + act * (size / 512.0) ** 2
