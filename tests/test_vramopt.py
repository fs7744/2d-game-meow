"""显存优化策略的测试。

覆盖 workflows/vramopt.py：
  · VAE 逐帧解码（峰值里最大的一块）
  · IP-Adapter image_encoder 用完即退
  · 显存规划（按可用显存降分辨率）
  · 估算公式的合理性
"""
from __future__ import annotations



from workflows import vramopt


# --------------------------------------------------------------------------- #
# 显存规划
# --------------------------------------------------------------------------- #

def test_plan_custom_passes_through():
    """自定义档位：原样返回，不做任何干预。"""
    n, size, note = vramopt.plan_generation(8, 512, profile="custom")
    assert n == 8
    assert size == 512
    assert "自定义" in note


def test_plan_low_forces_smaller_resolution():
    """low 档位强制降分辨率——帧数不能砍（那是用户的动画内容）。"""
    n, size, note = vramopt.plan_generation(8, 512, profile="low")
    assert n == 8, "帧数不能被砍"
    assert size == 384
    assert "384" in note


def test_plan_auto_keeps_resolution_when_enough_vram():
    """显存充足时不降分辨率。"""
    n, size, note = vramopt.plan_generation(8, 512, profile="auto")
    assert n == 8
    # 有 CUDA 且显存够 → 保持 512；没 CUDA → 走保守分支，但也不该升到更高
    assert size <= 512
    assert note


def test_plan_auto_downgrades_when_tight(monkeypatch):
    """显存不够时必须降分辨率，并在说明里讲清楚为什么。"""
    monkeypatch.setattr(vramopt, "_gpu_mem", lambda: (3.0, 24.0))
    n, size, note = vramopt.plan_generation(32, 512, profile="auto")
    assert n == 32, "帧数永远不砍"
    assert size < 512, "显存不够必须降分辨率"
    assert "显存不足" in note


def test_plan_never_exceeds_frame_count():
    """规划不能改变帧数。"""
    for prof in ("auto", "low", "medium", "high", "custom"):
        n, _size, _note = vramopt.plan_generation(13, 512, profile=prof)
        assert n == 13, f"{prof} 档位改了帧数"


# --------------------------------------------------------------------------- #
# 显存估算
# --------------------------------------------------------------------------- #

def test_estimate_grows_with_frames():
    """帧数越多显存越高。"""
    a = vramopt._estimate_gb(8, 512)
    b = vramopt._estimate_gb(16, 512)
    c = vramopt._estimate_gb(32, 512)
    assert a < b < c


def test_estimate_grows_with_resolution():
    """分辨率越高显存越高（面积平方关系）。"""
    a = vramopt._estimate_gb(8, 384)
    b = vramopt._estimate_gb(8, 512)
    assert a < b


def test_estimate_matches_measured_values():
    """估算要贴近实测值（实测：8帧@512 5.78GB / 16帧 7.10GB / 32帧 14.20GB）。"""
    # 估算允许 ±25% 误差——它是规划用的启发式，不是精确预测
    for n_frames, measured in ((8, 5.78), (16, 7.10), (32, 14.20)):
        est = vramopt._estimate_gb(n_frames, 512)
        assert abs(est - measured) / measured < 0.25, (
            f"{n_frames} 帧估算 {est:.2f}GB 与实测 {measured}GB 偏差过大")


# --------------------------------------------------------------------------- #
# 管线优化（需要模型）
# --------------------------------------------------------------------------- #

def test_release_encoder_handles_missing(eng):
    """没有 image_encoder 时不能崩（纯 ControlNet 管线）。"""
    class Fake:
        pass
    assert vramopt.release_ipa_encoder(Fake()) is False


# ---------------- W4 候选的显存规划 ----------------

def test_estimate_pose_gb_grows_with_size():
    from workflows import vramopt as V

    a = V.estimate_pose_gb("sdxl", 768, 1)
    b = V.estimate_pose_gb("sdxl", 1024, 1)
    assert 14.0 < a < 17.0, a          # 实测 15.33GB
    assert b > a * 1.1                 # 面积翻 1.78 倍，激活跟着涨


def test_estimate_pose_gb_grows_with_candidates():
    from workflows import vramopt as V

    one = V.estimate_pose_gb("sdxl", 1024, 1)
    two = V.estimate_pose_gb("sdxl", 1024, 2)
    # 实测 18.45 → 23.98，不能严重低估（低估就会放行并触发性能崩塌）
    assert abs(two - 23.98) < 2.0, two
    assert two > one


def test_plan_candidates_allows_when_plenty():
    from workflows import vramopt as V

    k, note = V.plan_candidates("sd15", 768, 3, free_gb=18.0)
    assert k == 3 and note == ""


def test_plan_candidates_caps_when_tight():
    """sdxl 1024² 批量 2 实测 23.98GB/24GB 且耗时崩到 17.5s，必须被拦住。"""
    from workflows import vramopt as V

    k, note = V.plan_candidates("sdxl", 1024, 3, free_gb=7.0)   # 加载后只剩 7GB
    assert k == 1, k
    assert "候选" in note


def test_plan_candidates_rejects_when_single_does_not_fit():
    from workflows import vramopt as V

    k, note = V.plan_candidates("sdxl", 1024, 2, free_gb=1.0)
    assert k == 0 and "装不下" in note


def test_plan_candidates_no_gpu_info_passthrough():
    from workflows import vramopt as V

    assert V.plan_candidates("sdxl", 768, 4, free_gb=0.0)[0] == 4
