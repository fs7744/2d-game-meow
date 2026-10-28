"""推理链路：确定性、局部重绘的「选区外不变」、姿态序列、显存回收。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from workflows import generate as g
from workflows.engine import Engine, detect_variant


def _arr(p: str) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def test_detect_variant_matches_layout(root):
    assert detect_variant(root / "models" / "checkpoints" / "sd15") is None       # sf 只有 fp32
    assert detect_variant(root / "models" / "checkpoints" / "sd15-inpaint") == "fp16"


def test_txt2img_same_seed_is_identical(eng, assets):
    kw = dict(prompt="blue slime monster", width=256, height=256, steps=12, seed=1234,
              pixel=4, palette=16)
    a = g.txt2img(eng, tag="det_a", **kw)
    b = g.txt2img(eng, tag="det_b", **kw)
    assert _arr(a["file"]).tolist() == _arr(b["file"]).tolist(), "同 seed 结果必须逐像素一致"


def test_txt2img_different_seed_differs(eng):
    a = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=12,
                  seed=1, tag="seed_a")
    b = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=12,
                  seed=2, tag="seed_b")
    assert _arr(a["file"]).tolist() != _arr(b["file"]).tolist()


def test_txt2img_palette_respected(eng):
    r = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=12,
                  seed=9, palette=8, tag="pal")
    colors = {p[:3] for p in Image.open(r["file"]).convert("RGBA").getdata() if p[3] > 127}
    assert len(colors) <= 8
    assert r["meta"]["palette"] == 8 and r["meta"]["seed"] == 9


def test_inpaint_leaves_outside_mask_untouched(eng, assets):
    src = assets / "_test_inpaint_src.png"
    mask = assets / "_test_inpaint_mask.png"
    im = Image.new("RGB", (256, 256), (30, 60, 90))
    ImageDraw.Draw(im).rectangle([60, 60, 120, 120], fill=(200, 30, 30))
    im.save(src)
    m = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(m).rectangle([100, 100, 180, 180], fill=255)   # 只盖住右下角一块
    m.save(mask)

    r = g.inpaint(eng, image=src, mask=mask, prompt="green grass block", steps=12,
                  seed=5, tag="inp")
    before = np.asarray(Image.open(src).convert("RGB")).astype(int)
    after = np.asarray(Image.open(r["file"]).convert("RGB")).astype(int)
    maskarr = np.asarray(Image.open(mask).convert("L"))

    outside = maskarr == 0
    diff = np.abs(after - before).max(axis=2)
    # 羽化带内允许变化，选区外 3px 以外必须逐点不变
    assert diff[outside].max() <= 0, "选区外像素被改动了"
    # 选区内必须真的变了
    assert diff[maskarr == 255].max() > 0


def test_poseswap_produces_frames_and_sheet(eng, pose_frames):
    r = g.poseswap(eng, pose_images=pose_frames[:3], prompt="white cat knight, red cape",
                   steps=10, seed=77, width=256, height=256, cols=3,
                   pixel=4, palette=16, tag="ps", ipa_scale=0.8)
    assert r["frames"] == 3
    assert len(r["per_frame_seconds"]) == 3
    assert r["cols"] == 3 and r["rows"] == 1
    sheet = Image.open(r["sheet"])
    assert sheet.size == (r["cell_w"] * 3, r["cell_h"])
    from pathlib import Path
    assert r["frames_dir"] and Path(r["frames_dir"]).is_dir()
    assert len(list(Path(r["frames_dir"]).glob("frame_*.png"))) == 3


def test_clip_skip_changes_result_and_is_recorded(eng):
    """clip_skip=2 必须真的改变画面，并把生效值记进 meta（否则没法复现）。"""
    kw = dict(prompt="blue slime monster", width=256, height=256, steps=8, seed=4321,
              preset="none", tag="cs")
    a = g.txt2img(eng, **kw)
    b = g.txt2img(eng, clip_skip=2, **kw)
    assert a["meta"]["clip_skip"] == 1
    assert b["meta"]["clip_skip"] == 2
    assert _arr(a["file"]).tolist() != _arr(b["file"]).tolist()


def test_subjects_reported_when_matted(eng):
    """抠图后才谈得上数主体：不抠图必须是 None，抠图则要给出整数。

    这里不苛求 >=1——草稿档（8 步/256²）出的图连主体都没有，逼着它「数出 1 个」
    等于测生成质量而不是测这段逻辑。真正「两块前景数成 2」的可控用例在
    test_postprocess.test_foreground_blobs_counts_separated_subjects。
    """
    off = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=8,
                    seed=99, preset="none", tag="subj_off")
    assert off["meta"]["subjects"] is None

    on = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=8,
                   seed=99, preset="none", matte=True, tag="subj_on")
    assert isinstance(on["meta"]["subjects"], int) and on["meta"]["subjects"] >= 0


def test_poseswap_gate_off_keeps_single_attempt(eng, pose_frames, monkeypatch):
    """关掉闸门就只生成 candidates 个候选，不再补提。"""
    monkeypatch.setattr(eng, "_pose", _FakePose(1), raising=False)
    r = g.poseswap(eng, pose_images=list(pose_frames)[:1], prompt="white cat knight",
                   steps=6, seed=11, width=256, height=256, cols=1, tag="gate_off",
                   candidates=1, gate=False, gate_retries=2)
    assert r["gate"] is False
    assert r["attempts"] == 1
    assert r["gate_retries"] == 0, "闸门关闭时不该预留补提额度"


def test_poseswap_gate_retries_until_budget_spent(eng, pose_frames, monkeypatch):
    """关节数一直不达标 -> 用满 1+retries 次，meta 里能看出重提过。"""
    monkeypatch.setattr(eng, "_pose", _FakePose(2), raising=False)
    r = g.poseswap(eng, pose_images=list(pose_frames)[:1], prompt="white cat knight",
                   steps=6, seed=12, width=256, height=256, cols=1, tag="gate_retry",
                   candidates=1, gate=True, gate_min_points=8, gate_retries=2)
    assert r["attempts"] == 3
    assert r["gate_passed"] == 0
    assert r["frame_scores"][0]["attempts"] == 3
    assert len(r["per_frame_seconds"]) == 1, "补提的耗时必须并进该帧，不能多出一帧"


def test_poseswap_batch_picks_best_candidate(eng, pose_frames, monkeypatch):
    """批量候选：第一个候选关节少，第二个够 -> 挑第二个，且不再补提。"""
    seq = iter([3, 14])

    class _Seq(_FakePose):
        def visible_points(self, img, thr=0.25):   # noqa: ARG002
            return {i: (0.5, 0.5) for i in range(next(seq, 14))}

    monkeypatch.setattr(eng, "_pose", _Seq(0), raising=False)
    r = g.poseswap(eng, pose_images=list(pose_frames)[:1], prompt="white cat knight",
                   steps=6, seed=13, width=256, height=256, cols=1, tag="batch_pick",
                   candidates=2, gate=True, gate_min_points=8, gate_retries=3)
    assert r["attempts"] == 2, "第二个候选已过闸门，不该再动用补提额度"
    assert r["gate_passed"] == 1
    assert r["frame_scores"][0]["points"] == 14


class _FakePose:
    """替身姿态模型：固定返回 n 个可见关节（真实模型在 test_pose.py 里测）。"""

    def __init__(self, n: int) -> None:
        self.n = n

    def visible_points(self, img, thr: float = 0.25):   # noqa: ARG002
        return {i: (0.5, 0.5) for i in range(self.n)}


def test_engine_free_clears_cache(eng):
    eng.txt2img_pipe()
    assert eng.stats()["loaded"]
    eng.free()
    assert eng.stats()["loaded"] == []


def test_missing_weights_raise_clear_error(root):
    e = Engine(root / "models_nonexistent")
    try:
        e.txt2img_pipe()
    except FileNotFoundError as exc:
        assert "权重缺失" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("缺权重时必须抛 FileNotFoundError")
