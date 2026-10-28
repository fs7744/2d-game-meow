"""推理链路：确定性、局部重绘的「选区外不变」、姿态序列、显存回收。"""
from __future__ import annotations

import numpy as np
from pathlib import Path
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


def test_img2img_postprocess_and_raw(eng, assets):
    """W2 与 W1 共用同一套后处理，并把「未后处理的原图」留下来给下游当源图。

    图生图拿成品精灵回炉（已经像素化到 128 有效像素）会把色块重新糊成插画，
    所以 raw 是这条链路的关键产物，不能省。
    """
    from pathlib import Path as _P

    src = assets / "_test_w2_src.png"
    im = Image.new("RGB", (256, 256), (40, 80, 160))
    ImageDraw.Draw(im).rectangle([60, 60, 190, 200], fill=(230, 60, 60))
    im.save(src)

    r = g.img2img(eng, init_image=src, prompt="blue slime monster", strength=0.5,
                  steps=8, seed=3, preset="none", pixel=4, palette=16, matte=True,
                  tag="w2_test")
    m = r["meta"]
    assert m["workflow"] == "w2"
    assert _P(m["init_image"]) == src
    assert (m["pixel_block"], m["palette"], m["matte"]) == (4, 16, True)
    assert m["strength"] == 0.5 and m["steps"] == 8
    assert m["raw"] and _P(m["raw"]).exists(), "必须留下未后处理的原图"

    out = Image.open(r["file"]).convert("RGBA")
    colors = {p[:3] for p in out.getdata() if p[3] > 127}
    assert len(colors) <= 16, len(colors)
    transparent = float((np.asarray(out)[..., 3] == 0).mean())
    assert 0 < transparent < 0.99, transparent


def test_img2img_without_postprocess_skips_raw(eng, assets):
    """没开后处理时不该多存一份 raw（白占一倍磁盘，而且 raw 与成品完全一样）。"""
    src = assets / "_test_w2_plain.png"
    Image.new("RGB", (128, 128), (90, 90, 90)).save(src)
    r = g.img2img(eng, init_image=src, prompt="grey square", steps=4, seed=1,
                  preset="none", tag="w2_plain")
    assert r["meta"].get("raw") is None


def test_txt2img_raw_saved_for_downstream(eng):
    """W1 开了后处理就要留 raw —— 图生图节点默认拿它当源图。"""
    r = g.txt2img(eng, prompt="blue slime monster", width=256, height=256, steps=8,
                  seed=4, preset="none", pixel=4, tag="w1_raw")
    raw = r["meta"].get("raw")
    assert raw and np.asarray(Image.open(raw)).shape == (256, 256, 3)
    assert raw != r["file"]


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


def test_scope_isolates_output_dirs(eng):
    """项目隔离：产物写进 assets/projects/<scope>/，两个项目各写各的。"""
    a = g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=6, seed=1,
                  preset="none", scope="projA", tag="iso_a")
    b = g.txt2img(eng, prompt="red blob", width=256, height=256, steps=6, seed=2,
                  preset="none", scope="projB", tag="iso_b")
    na = a["file"].replace("\\", "/")
    nb = b["file"].replace("\\", "/")
    assert "/projects/projA/" in na, na
    assert "/projects/projB/" in nb, nb
    assert a["url"].startswith("/files/projects/projA/")


def test_cross_project_source_is_blocked(eng):
    """拿别的项目的图当图生图的源，必须被拒（项目之间相互隔离）。"""
    a = g.txt2img(eng, prompt="blue slime", width=256, height=256, steps=6, seed=1,
                  preset="none", scope="projA", tag="iso_c")
    import pytest

    with pytest.raises(ValueError, match="不属于当前项目"):
        g.img2img(eng, init_image=a["file"], prompt="red blob", steps=4,
                  scope="projB", tag="iso_x")
    # 同项目内引用不受影响
    ok = g.img2img(eng, init_image=a["file"], prompt="blue slime", steps=4,
                   scope="projA", tag="iso_ok")
    assert "/projects/projA/" in ok["file"].replace("\\", "/")


def _bg_src(assets) -> str:
    """一张白底红方块的图，当抠图源。"""
    p = assets / "_test_bg_src.png"
    im = Image.new("RGB", (128, 128), (255, 255, 255))
    ImageDraw.Draw(im).rectangle([30, 20, 98, 108], fill=(200, 40, 40))
    im.save(p)
    return str(p)


def test_matte_bg_alpha(eng, assets):
    """去背景：产物是透明底 PNG，背景被抠掉。"""
    src = _bg_src(assets)
    r = g.matte_bg(eng, init_image=src, tag="bg_alpha")
    im = Image.open(r["file"]).convert("RGBA")
    a = np.asarray(im)[..., 3]
    assert r["meta"]["bg_mode"] == "alpha"
    assert (a == 0).mean() > 0.2, "白底应该被抠掉"
    assert (a == 255).mean() > 0.1, "主体应该留下"
    assert r["meta"]["fg_ratio"] > 0.1


def test_matte_bg_color(eng, assets):
    """纯色背景：角落是那个颜色，并且额外存一份透明底。"""
    src = _bg_src(assets)
    r = g.matte_bg(eng, init_image=src, bg_color="#3355ff", tag="bg_color")
    im = Image.open(r["file"]).convert("RGBA")
    assert r["meta"]["bg_mode"] == "color"
    assert im.getpixel((1, 1))[:3] == (51, 85, 255), im.getpixel((1, 1))
    assert r["meta"]["cut"] and Path(r["meta"]["cut"]).exists()


def test_matte_bg_with_background_image(eng, assets):
    """换背景：前景叠到另一张图上，尺寸与前景一致。"""
    src = _bg_src(assets)
    bgp = assets / "_test_bg_bg.png"
    Image.new("RGB", (256, 256), (20, 160, 60)).save(bgp)
    r = g.matte_bg(eng, init_image=src, background=str(bgp), tag="bg_image")
    im = Image.open(r["file"]).convert("RGBA")
    assert r["meta"]["bg_mode"] == "image"
    assert im.size == (128, 128)
    assert im.getpixel((1, 1))[:3] == (20, 160, 60)


def _rp_pair(assets):
    """造一对「源图 + 蒙版」：白底上一个红方块，蒙版盖住方块中央。"""
    src = assets / "_rp_src.png"
    im = Image.new("RGB", (256, 256), (250, 250, 250))
    ImageDraw.Draw(im).rectangle([78, 78, 178, 178], fill=(200, 40, 40))
    im.save(src)
    mk = assets / "_rp_mask.png"
    m = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(m).rectangle([98, 98, 158, 158], fill=255)
    m.save(mk)
    return src, mk


def test_ai_repaint_keeps_outside_pixels(eng, assets):
    """局部 AI 重绘的立命之本：选区外的像素一个都不变。"""
    src, mk = _rp_pair(assets)
    r = g.ai_repaint(eng, image=src, mask=mk, prompt="blue gem",
                     steps=6, seed=11, preset="none", tag="rp_out")
    m = r["meta"]
    assert m["workflow"] == "w5"
    assert m["coverage"] > 0
    assert m["box"][2] > m["box"][0] and m["box"][3] > m["box"][1]

    a = np.asarray(Image.open(src).convert("RGBA")).astype(int)
    b = np.asarray(Image.open(r["file"]).convert("RGBA")).astype(int)
    outside = np.asarray(Image.open(mk)) <= 127
    # 源图与产物同尺寸，否则比较无意义
    assert a.shape == b.shape
    assert (a[outside] == b[outside]).all(), "选区外有像素变了"


def test_ai_repaint_records_prompt_and_warnings(eng, assets):
    """程序加的约束条款必须记进 meta（界面上要能看见、可改）。"""
    src, mk = _rp_pair(assets)
    r = g.ai_repaint(eng, image=src, mask=mk, prompt="blue gem",
                     steps=6, seed=12, preset="none", feather=0,
                     tag="rp_meta")
    m = r["meta"]
    assert m["prompt_body"] == "blue gem"
    # 约束条款必须是英文：SD1.5 的 CLIP 只认英文，中文会被拆成乱 token，
    # 而且整句必然在 77 token 处被截断（实测后半段直接被砍掉）
    assert "masked" in m["constraint"]
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in m["constraint"])
    assert m["prompt"].startswith("blue gem")
    assert m["negative"]
    assert isinstance(m["warnings"], list)


def test_ai_repaint_rejects_mismatched_mask(eng, assets):
    """蒙版尺寸和源图对不上（换了图没重涂）必须报错，而不是硬缩蒙版。"""
    src, mk = _rp_pair(assets)
    small = assets / "_rp_small.png"
    Image.new("L", (64, 64), 255).save(small)
    import pytest

    with pytest.raises(ValueError, match="不一致"):
        g.ai_repaint(eng, image=src, mask=small, prompt="x", steps=4,
                     preset="none", tag="rp_bad")


def test_ai_repaint_rejects_empty_prompt(eng, assets):
    src, mk = _rp_pair(assets)
    import pytest

    with pytest.raises(ValueError, match="提示词"):
        g.ai_repaint(eng, image=src, mask=mk, prompt="  ", steps=4,
                     preset="none", tag="rp_nop")


def test_ai_repaint_empty_mask_errors(eng, assets):
    src, _ = _rp_pair(assets)
    empty = assets / "_rp_empty.png"
    Image.new("L", (256, 256), 0).save(empty)
    import pytest

    with pytest.raises(ValueError, match="蒙版"):
        g.ai_repaint(eng, image=src, mask=empty, prompt="x", steps=4,
                     preset="none", tag="rp_empty")


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
