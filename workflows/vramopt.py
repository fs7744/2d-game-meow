"""显存估算：W4（逐帧）批量候选装不装得下。

**只剩一件事**：`estimate_pose_gb` —— 估「逐帧生成 N 个候选」的峰值显存。
调用方是 `engine.max_candidates`：按**本机总显存**算出上限，发给前端当输入框
的上限（用户把候选数填成 3，在 24G 卡上就是 3 倍的耗时 + 撞顶劣化，而且不报错）。

**为什么这个模块变薄了**：SD 时代那套 AnimateDiff 显存优化（VAE 逐帧解码、
IP-Adapter encoder 用完即退、`plan_generation` 按档位降分辨率）随 SD 管线一起
删了——Qwen 路线没有 motion module，也没有 IP-Adapter，那些函数一个调用点都没有。
**别照着旧版把它们加回来**：它们改的是 `AnimateDiffControlNetPipeline` 的部件，
Qwen 的管线里没有这些东西。

**降档的责任分散在别处**（各管一段，别在这里重复实现）：

| 场景 | 实现在哪 | 做什么 |
|---|---|---|
| 已经 OOM 了 | `generate._infer` | 清缓存 → 缩 3/4 边长重试（最多两次） |
| 加载前腾地方 | `engine.ensure_headroom` | 按 LRU 驱逐缓存管线，直到空够 need_gb |
| 别让用户设跑不动的值 | `engine.max_candidates` → 本模块 | 用总显存算候选数上限 |
"""
from __future__ import annotations

#: 实测（RTX 4090，fp8 transformer + cpu offload 的 text_encoder）：
#:     Qwen 768² 单帧 11.8~15.4GB（保留峰值，不是 allocated）
#:     再多一个候选约 +6GB
#:
#: 数值故意**偏保守**：估低了会放行跑不动的配置，代价是耗时翻倍且不报错
#: （与 SD 时代 sdxl 1024² 批量 2 的教训相同）；估高只是少给一个候选。
#:
#: 键是**架构**（`engine.arch()` 的返回值），不是底模目录名——LoRA / 分辨率
#: 可以按名字不同，峰值的量级只跟架构有关。未知架构按 qwen21 算（保守）。
_W4_SHAPE = {
    #        权重GB  768²激活GB  768²每帧增量GB
    "qwen21": (11.0, 6.0, 6.5),
}


def estimate_pose_gb(arch: str, size: int, candidates: int = 1) -> float:
    """估逐帧生成（W4 / 姿态序列）的保留峰值显存（GB）。

    峰值 ≈ 权重 + 激活(∝面积) + 每多一帧的增量(∝面积) × (候选数 - 1)。

    `candidates` 是**同时批量出几张**，不是"总共跑几张"——候选是按批出的，
    多一个候选就是多一份激活 + 多一份解码，所以按 `candidates - 1` 累加。
    """
    w, act, per = _W4_SHAPE.get(arch, _W4_SHAPE["qwen21"])
    k = (max(256, int(size)) / 768.0) ** 2
    return w + act * k + per * k * max(0, candidates - 1)
