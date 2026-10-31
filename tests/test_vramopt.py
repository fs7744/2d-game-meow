"""显存估算的测试（`workflows/vramopt.py`）。

现在只有一件事要测：`estimate_pose_gb` 算得对不对。它是
`engine.max_candidates` 的唯一依据——估错了要么放行跑不动的配置，
要么白白掐掉一个候选。

SD 时代的 AnimateDiff 优化（VAE 逐帧解码 / IP-Adapter encoder 用完即退 /
`plan_generation` 按档位降分辨率）已随管线一起删除，对应测试也一并删了：
那些函数在 Qwen 路线里连调用点都没有，测它们只是维护幻觉。
"""
from __future__ import annotations

from workflows import vramopt as V


def test_estimate_grows_with_size():
    """面积翻倍 → 激活跟着涨（峰值不是常数，不能拿固定值当上限）。"""
    a = V.estimate_pose_gb("qwen21", 768, 1)
    b = V.estimate_pose_gb("qwen21", 1024, 1)
    # 768² 实测 11.8~15.4GB；估算**故意偏高**（估低会放行跑不动的配置）
    assert 16.0 < a < 18.5, a
    assert b > a * 1.1, (a, b)


def test_estimate_grows_with_candidates():
    """多一个候选约 +6GB（实测），且 ≤1 一律按单帧算。"""
    one = V.estimate_pose_gb("qwen21", 768, 1)
    two = V.estimate_pose_gb("qwen21", 768, 2)
    assert two - one >= 5.0, (one, two)
    assert V.estimate_pose_gb("qwen21", 768, 0) == one
    assert V.estimate_pose_gb("qwen21", 768, 1) == one


def test_24g_card_only_fits_one_candidate_at_768():
    """24GB 卡在 768² 上只够 1 个候选 —— 「前端候选数上限是 1」的看守。

    估算一旦改小，用户就会被放行去跑 2 个候选：768² 批量 2 要 23.5GB，
    会挤到 cpu offload 上，耗时翻倍且**不报错**。
    """
    budget = 24.0 * 0.9                 # 与 engine.max_candidates 的 safety 一致
    assert V.estimate_pose_gb("qwen21", 768, 1) <= budget
    assert V.estimate_pose_gb("qwen21", 768, 2) > budget


def test_unknown_arch_falls_back_to_qwen():
    """未知架构按 Qwen 算（保守），不能退回某个已删除的 SD 档位。"""
    assert V.estimate_pose_gb("nope", 768, 1) == V.estimate_pose_gb("qwen21", 768, 1)
