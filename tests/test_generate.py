"""推理链路：确定性、局部重绘的「选区外不变」、姿态序列、显存回收。"""
from __future__ import annotations

import json

import numpy as np
import pytest
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


def test_pixelify_makes_one_color_per_cell(eng, assets):
    """完美像素转换：输出必须每格一色（像素画与低分辨率图的区别）。"""
    src = assets / "_px_src.png"
    im = Image.new("RGB", (128, 128), (250, 250, 250))
    d = ImageDraw.Draw(im)
    for y in range(8):
        for x in range(8):
            d.rectangle([x * 16, y * 16, (x + 1) * 16 - 1, (y + 1) * 16 - 1],
                        fill=((x * 37) % 256, (y * 53) % 256, 128))
    im.save(src)

    r = g.pixelify(eng, init_image=src, colors=16, tag="px_test")
    m = r["meta"]
    assert m["workflow"] == "wpx"
    assert m["detected"] is True
    assert m["grid"] == [8, 8], m["grid"]
    assert m["offPalette"] == 0, m["offPalette"]
    assert m["alphaBinary"] == 1.0

    out = Image.open(r["file"]).convert("RGB")
    assert out.size == (8, 8), out.size
    # 每格一色：8×8 的图每个像素就是一个格
    arr = np.asarray(out).reshape(-1, 3)
    assert len(arr) == 64


def test_pixelify_target_size_forces_dimensions(eng, assets):
    """指定 target 时输出必须是那个尺寸（且只能缩小不能放大）。"""
    src = assets / "_px_tgt.png"
    Image.new("RGB", (256, 256), (30, 60, 120)).save(src)
    r = g.pixelify(eng, init_image=src, target=[64, 64], colors=16, tag="px_tgt")
    out = Image.open(r["file"])
    assert out.size == (64, 64), out.size
    assert r["meta"]["target"] == [64, 64]


def test_pixelify_without_colors_keeps_grid_alignment(eng, assets):
    """colors=0 = 只做网格对齐，不量化（色数应该远多于色板数）。"""
    src = assets / "_px_nq.png"
    im = Image.new("RGB", (128, 128), (250, 250, 250))
    d = ImageDraw.Draw(im)
    for y in range(8):
        for x in range(8):
            d.rectangle([x * 16, y * 16, (x + 1) * 16 - 1, (y + 1) * 16 - 1],
                        fill=((x * 37) % 256, (y * 53) % 256, 128))
    im.save(src)
    r = g.pixelify(eng, init_image=src, colors=0, tag="px_nq")
    assert r["meta"]["colors_requested"] == 0
    assert "palette" not in r["meta"]


def _pose_clone_inputs(assets):
    """造「动作参考图 + 角色图」。

    动作参考图**必须是真人照片**——姿态模型只认得出真人，合成简笔人形
    连人体检测都过不了（FasterRCNN 检不出，检测器直接 fallback 成整图框，
    姿态回归在整图上乱跑，输出一堆坐标离谱的"关节"）。
    所以这里用仓库里的固定素材 tests/fixtures/pose_ref_person.png。
    """
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pose_ref_person.png"
    if not fixture.exists():
        import pytest
        pytest.skip("缺少真人姿态参考素材 tests/fixtures/pose_ref_person.png")
    pose = assets / "_pc_pose.png"
    Image.open(fixture).convert("RGB").save(pose)

    char = assets / "_pc_char.png"
    cim = Image.new("RGB", (256, 256), (250, 250, 250))
    ImageDraw.Draw(cim).ellipse([80, 60, 176, 190], fill=(40, 170, 200))
    cim.save(char)
    return pose, char


def test_pose_clone_records_both_sources(eng, assets):
    """角色迁移：动作参考图定姿态、角色图定身份，两个源都要记进 meta。"""
    pose, char = _pose_clone_inputs(assets)
    r = g.pose_clone(eng, pose_image=pose, char_image=char,
                      prompt="a blue slime monster", steps=6, seed=3,
                      preset="none", width=512, height=512, scope="",
                      tag="pc_test")
    m = r["meta"]
    assert m["workflow"] == "wpc"
    assert Path(m["pose_image"]).name == "_pc_pose.png"
    assert Path(m["char_image"]).name == "_pc_char.png"
    assert m["joints"] >= 12, m["joints"]
    assert m["cn_scale"] == 1.0 and m["ipa_scale"] == 0.8
    assert Path(r["file"]).exists()


def test_pose_clone_rejects_image_without_person(eng, assets):
    """显式要求骨架模式时，检测不到人体必须报错，而不是静默出一张没姿态的图。

    注意：以前这里会「成功」——检测器 fallback 成整图框，姿态回归在整张图上
    乱跑，输出一堆坐标离谱的关节，看起来像识别成功了，其实完全没用。

    （默认的 auto 模式是另一回事：它会降级到轮廓路线，见下面那条用例。）
    """
    _, char = _pose_clone_inputs(assets)
    blank = assets / "_pc_blank.png"
    Image.new("RGB", (256, 256), (250, 250, 250)).save(blank)

    with pytest.raises(ValueError, match="没检测到人物"):
        g.pose_clone(eng, pose_image=blank, char_image=char,
                     prompt="x", steps=4, preset="none", width=256, height=256,
                     ref_mode="pose", tag="pc_bad")


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


# ---------------- WSS 序列帧角色替换 ----------------

def _sprite_sheet(assets, cols=4, rows=1, cell=64) -> Path:
    """造一张游戏序列帧（像素骑士走路，透明底）。

    刻意做成**人体检测器认不出来**的像素画：这是这条链路的真正场景。
    """
    def frame(i):
        im = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([26, 48, 30, 58], fill=(40, 40, 60, 255))
        d.rectangle([33, 48, 37, 58], fill=(40, 40, 60, 255))
        d.rectangle([25, 30, 38, 48], fill=(120, 30, 40, 255))
        d.rectangle([27, 18, 36, 30], fill=(235, 200, 160, 255))
        d.rectangle([26, 16, 37, 21], fill=(160, 165, 175, 255))
        d.rectangle([21, 31, 25, 42], fill=(120, 30, 40, 255))
        d.rectangle([38, 31, 42, 42], fill=(120, 30, 40, 255))
        d.rectangle([44, 20, 46, 40], fill=(200, 200, 210, 255))
        return im

    sheet = Image.new("RGBA", (cols * cell, rows * cell), (0, 0, 0, 0))
    n = 0
    for r in range(rows):
        for c in range(cols):
            sheet.paste(frame(n), (c * cell, r * cell))
            n += 1
    p = assets / "_wss_sheet.png"
    sheet.save(p)
    return p


def _slime_char(assets) -> Path:
    p = assets / "_wss_char.png"
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([10, 20, 54, 58], fill=(70, 190, 90, 255))
    d.ellipse([14, 24, 50, 54], fill=(110, 225, 120, 255))
    im.save(p)
    return p


def test_sprite_swap_recolor_is_deterministic_and_keeps_shape(eng, assets):
    """换皮模式：不跑模型、结果确定、形状与参考帧逐像素一致。

    这是「同体型换配色」的最稳路线：alpha 原样保留，所以帧间零漂移。
    """
    from workflows import sheet as sheet_mod

    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    frames = sheet_mod.split_sheet(Image.open(sheet).convert("RGBA"), 4, 1)
    ref_alpha = [np.asarray(f.getchannel("A")) for _c, _r, f in frames]

    a = g.sprite_swap(eng, sheet_image=sheet, char_image=char, prompt="",
                      cols=4, rows=1, mode="recolor", palette=8, tag="wss_rc_a")
    b = g.sprite_swap(eng, sheet_image=sheet, char_image=char, prompt="",
                      cols=4, rows=1, mode="recolor", palette=8, tag="wss_rc_b")

    assert a["meta"]["mode"] == "recolor"
    assert a["meta"]["deterministic"] is True
    assert a["meta"]["frames"] == 4
    assert a["meta"]["grid"] == [4, 1]
    assert a["meta"]["seconds_total"] == 0.0, "换皮不该花显卡时间"

    # 两次跑出来的图必须逐像素一致
    assert (_arr(a["file"]).tolist() == _arr(b["file"]).tolist()), "换皮必须确定性"

    # 形状（alpha）必须与参考帧一致
    out_frames = sorted(Path(a["frames_dir"]).glob("frame_*.png"))
    assert len(out_frames) == 4
    for i, p in enumerate(out_frames):
        alpha = np.asarray(Image.open(p).convert("RGBA").getchannel("A"))
        assert alpha.shape == ref_alpha[i].shape
        assert (alpha == ref_alpha[i]).all(), f"第 {i} 帧形状变了"


def test_sprite_swap_generate_keeps_frames_and_palette(eng, assets):
    """AI 重绘模式：帧数不减、格子尺寸回到原帧大小、整段共享色板。

    这条链路跑真模型（SD1.5 + canny ControlNet + IP-Adapter），
    步数压到最低，只验结构不验画质。
    """
    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    r = g.sprite_swap(eng, sheet_image=sheet, char_image=char,
                      prompt="green slime monster", negative=None, preset="sprite",
                      cols=4, rows=1, width=512, height=512, steps=6, seed=7,
                      cn_scale=1.0, ipa_scale=0.8, palette=8, matte=True,
                      tag="wss_gen")
    m = r["meta"]
    assert m["workflow"] == "wss"
    assert m["mode"] == "generate"
    assert m["frames"] == 4
    assert m["cell"] == [64, 64], "格子尺寸必须回到原帧大小（序列帧要塞回游戏）"
    assert m["grid"] == [4, 1]
    assert m["ref_modes_used"] == ["outline"], "像素画应当走轮廓模式"
    assert len(r["frames_dir"]) > 0
    frames = sorted(Path(r["frames_dir"]).glob("frame_*.png"))
    assert len(frames) == 4
    for p in frames:
        assert Image.open(p).size == (64, 64)
    # 精灵图尺寸 = 列数 × 格子宽
    assert Image.open(r["sheet"]).size == (4 * 64, 64)


def test_sprite_swap_rejects_unrecognizable_grid(eng, assets):
    """单帧（认不出网格）时必须报错要求手填，不能默默当成 1×1 悄悄丢掉内容。"""
    single = assets / "_wss_single.png"
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(single)
    char = _slime_char(assets)
    with pytest.raises(ValueError, match="认不出精灵图的网格"):
        g.sprite_swap(eng, sheet_image=single, char_image=char, prompt="x",
                      cols=0, rows=0, tag="wss_bad")


def test_pose_clone_auto_falls_back_on_pixel_art(eng, assets):
    """角色迁移的 auto 模式：像素画参考要走轮廓，不能报错。

    这是「不止真人」的核心承诺——真人走骨架（更准），
    游戏素材走轮廓（能用）。
    """
    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    r = g.pose_clone(eng, pose_image=sheet, char_image=char,
                     prompt="green slime monster", negative=None, preset="sprite",
                     width=512, height=512, steps=6, seed=7, ref_mode="auto",
                     matte=True, tag="pc_auto")
    m = r["meta"]
    assert m["ref_mode"] == "auto"
    assert m["ref_mode_used"] == "outline"
    assert m["ref_fallback_from"] == "pose"
    assert m["ref_fallback_reason"]
    assert Path(r["file"]).exists()


def test_sprite_swap_multi_candidate_picks_best(eng, assets):
    """多候选择优：每帧生成多张候选，按分数挑最好的一张。

    业界实测：IP-Adapter + ControlNet 的帧间一致性只有 70~80%，
    **同 seed 也会漂 20~30%**（CLIP Vision 编码引入方差）。
    所以业界通用解法是「生成多个候选择优」，不是调参。
    """
    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    r = g.sprite_swap(eng, sheet_image=sheet, char_image=char,
                      prompt="green slime monster", negative=None, preset="sprite",
                      cols=4, rows=1, width=512, height=512, steps=6, seed=11,
                      cn_scale=1.0, ipa_scale=0.8, palette=8, matte=True,
                      candidates=3, gate_retries=1, engine_mode="frame",
                      tag="wss_cand")
    m = r["meta"]
    assert m["engine_mode"] == "frame"
    assert m["candidates"] == 3
    assert len(m["frame_scores"]) == 4, "每帧都要有评分"
    for ev in m["frame_scores"]:
        assert "silhouette" in ev and "identity" in ev
        assert "attempts" in ev and "picked" in ev
    assert m["attempts"] >= 4, "至少每帧提过 1 张"
    # 序列一致性指标要写进 meta（产线据此判断要不要重跑）
    assert "identity_drift" in m and "silhouette_drift" in m


def test_sprite_swap_output_is_pixel_perfect(eng, assets):
    """输出必须**每格一色**——像素画的定义。

    之前的实现是「BOX 降采样后再量化」，BOX 会把每个像素格里的多种颜色
    混在一起，边界散开、色块不干净。改成「从生成尺寸直接按格采样」后，
    每格单色率应当是 100%。
    """
    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    r = g.sprite_swap(eng, sheet_image=sheet, char_image=char,
                      prompt="green slime monster", negative=None, preset="sprite",
                      cols=4, rows=1, width=512, height=512, steps=6, seed=13,
                      cn_scale=1.0, ipa_scale=0.8, palette=8, matte=True,
                      candidates=1, tag="wss_px")
    frames = sorted(Path(r["frames_dir"]).glob("frame_*.png"))
    assert frames
    for p in frames:
        arr = np.asarray(Image.open(p).convert("RGBA"))
        h, w = arr.shape[:2]
        # 统计「整帧里每 1×1 像素本来就是一色」——像素画缩到原尺寸后
        # 每个输出像素就是一个格子，必须单色（量化保证了这点）
        opaque = arr[..., 3] > 127
        if opaque.sum() < 10:
            continue
        colors = {tuple(c) for c in arr[..., :3][opaque].tolist()}
        assert len(colors) <= 8, f"{p.name} 前景色数 {len(colors)} 超过色板上限"


def test_pose_clone_uses_clean_reference(eng, assets):
    """角色迁移也要用预处理过的参考图。

    回归点：以前直接 `convert("RGB")`，透明区变黑——实测 66.5% 的角色图
    是纯黑背景，IP-Adapter 会把黑色当成角色的暗部，身份特征被污染。
    """
    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    r = g.pose_clone(eng, pose_image=sheet, char_image=char,
                     prompt="green slime monster", negative=None, preset="sprite",
                     width=512, height=512, steps=6, seed=17, ref_mode="auto",
                     matte=True, tag="pc_clean")
    assert Path(r["file"]).exists()


def test_engine_animatediff_available(eng):
    """motion adapter 在不在。缺失时整个 AnimateDiff 路线不可用（但不是错误）。"""
    ok = eng.animatediff_available()
    assert isinstance(ok, bool)
    if ok:
        d = eng.root / eng.ANIMATEDIFF_DIR
        assert (d / "config.json").exists()


def test_sprite_swap_animatediff_mode(eng, assets):
    """AnimateDiff 路线：一次生成整段，时序注意力保证帧间一致。

    这条链路的价值：业界实测 IP-Adapter + ControlNet 的帧间一致性只有
    70~80%，逐帧生成必然漂；AnimateDiff 的 motion module 提供**跨帧
    时序注意力**，是架构级的约束。
    """
    if not eng.animatediff_available():
        import pytest
        pytest.skip("缺少 AnimateDiff motion adapter 权重")

    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    r = g.sprite_swap(eng, sheet_image=sheet, char_image=char,
                      prompt="green slime monster", negative=None, preset="sprite",
                      cols=4, rows=1, width=256, height=256, steps=6, seed=21,
                      cn_scale=1.0, ipa_scale=0.8, palette=8, matte=True,
                      engine_mode="animatediff", tag="wss_ad")
    m = r["meta"]
    assert m["engine_mode"] == "animatediff"
    assert m["frames"] == 4
    assert m["cell"] == [64, 64]
    assert len(m["frame_scores"]) == 4
    # 序列一致性指标必须写进 meta
    assert "identity_drift" in m and "silhouette_drift" in m
    # 残缺帧补提：记录哪些帧被单独重生成过
    assert "repaired_frames" in m
    frames = sorted(Path(r["frames_dir"]).glob("frame_*.png"))
    assert len(frames) == 4
    for p in frames:
        assert Image.open(p).size == (64, 64)


def test_sprite_swap_animatediff_missing_weight_errors(eng, assets, monkeypatch):
    """权重缺失时必须明确报错并给出下载指引，而不是默默退回逐帧。"""
    import pytest
    monkeypatch.setattr(eng, "animatediff_available", lambda: False)
    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    with pytest.raises(ValueError, match="motion adapter"):
        g.sprite_swap(eng, sheet_image=sheet, char_image=char,
                      prompt="x", cols=4, rows=1, engine_mode="animatediff",
                      tag="wss_ad_missing")


def test_sprite_swap_animatediff_vram_is_bounded(eng, assets):
    """AnimateDiff 路线的显存必须有界，且 meta 里记录了规划结果。

    回归点：补提曾经用独立的 pose_pipe()，那会**再加载一整套**
    UNet/VAE/ControlNet/IP-Adapter（实测多占 3.23GB）——两条管线权重重复。
    现在补提直接用 AnimateDiff 管线跑 2 帧，零额外权重。
    """
    if not eng.animatediff_available():
        import pytest
        pytest.skip("缺少 AnimateDiff motion adapter 权重")

    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    r = g.sprite_swap(eng, sheet_image=sheet, char_image=char,
                      prompt="green slime monster", negative=None, preset="sprite",
                      cols=4, rows=1, width=256, height=256, steps=6, seed=31,
                      cn_scale=1.0, ipa_scale=0.8, palette=8, matte=True,
                      engine_mode="animatediff", vram_profile="custom",
                      tag="wss_vram")
    m = r["meta"]
    assert "vram_profile" in m and "vram_note" in m
    assert m["vram_profile"] == "custom"
    assert m["peak_vram_gb"] > 0
    # 显存必须有界——不能因为重复加载权重而膨胀
    assert m["peak_vram_gb"] < 8.0, f"峰值显存 {m['peak_vram_gb']}GB 过高"


def test_sprite_swap_vram_profile_low_downgrades(eng, assets):
    """low 档位要真的降分辨率（帧数不砍）。"""
    if not eng.animatediff_available():
        import pytest
        pytest.skip("缺少 AnimateDiff motion adapter 权重")

    sheet = _sprite_sheet(assets)
    char = _slime_char(assets)
    r = g.sprite_swap(eng, sheet_image=sheet, char_image=char,
                      prompt="green slime monster", negative=None, preset="sprite",
                      cols=4, rows=1, width=512, height=512, steps=6, seed=37,
                      cn_scale=1.0, ipa_scale=0.8, palette=8, matte=True,
                      engine_mode="animatediff", vram_profile="low",
                      tag="wss_lowvram")
    m = r["meta"]
    assert m["width"] == 384 and m["height"] == 384
    assert m["frames"] == 4, "帧数不能被砍"
    assert "low" in m["vram_note"]


# ---------------- W4 的输入归一化（纯逻辑，不跑模型） ----------------

def test_norm_pose_inputs_sorts_naturally(tmp_path):
    """帧顺序决定动画顺序，必须按**自然序**排。

    字典序会把 frame_10 排在 frame_2 前面（'1' < '2'），10 帧以上的走路序列
    就会乱序——而且不会报错，只是播出来是跳的。
    """
    for i in (1, 2, 10, 11):
        (tmp_path / f"frame_{i}.png").write_bytes(b"x")
    got = [p.name for p in g.norm_pose_inputs(tmp_path)]
    assert got == ["frame_1.png", "frame_2.png", "frame_10.png", "frame_11.png"], got


def test_norm_pose_inputs_accepts_dir_glob_and_list(tmp_path):
    """目录 / 通配 / 显式列表三种写法都要能用（前端传目录，CLI 传列表）。"""
    for i in (0, 1, 2):
        (tmp_path / f"p{i}.png").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("不是图", "utf-8")

    assert len(g.norm_pose_inputs(tmp_path)) == 3, "非图片文件要被过滤"
    assert len(g.norm_pose_inputs(str(tmp_path / "p*.png"))) == 3
    listed = [tmp_path / "p2.png", tmp_path / "p0.png"]
    assert [p.name for p in g.norm_pose_inputs(listed)] == ["p2.png", "p0.png"], \
        "显式列表要保持给定顺序，不重排"


def test_norm_pose_inputs_reports_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        g.norm_pose_inputs(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        g.norm_pose_inputs([tmp_path / "gone.png"])


def test_poseswap_accepts_dir_and_auto_cols(eng, assets):
    """传目录 + cols=0：自动按帧数排一行。

    以前 cols 直接进 pack_sheet 的 `(len+cols-1)//cols`，0 会 ZeroDivisionError。
    前端把 0 当"自动"传，所以必须在入口兜住。
    """
    d = assets / "_test_pose_dir"
    d.mkdir(exist_ok=True)
    poses = sorted(assets.glob("_pose_walk_*.png"))[:2]
    if len(poses) < 2:                      # pragma: no cover
        pytest.skip("缺骨架夹具")
    for i, p in enumerate(poses):
        Image.open(p).save(d / f"frame_{i}.png")

    r = g.poseswap(eng, pose_images=d, prompt="white cat knight", steps=4, seed=11,
                   width=256, height=256, cols=0, candidates=1, gate=False,
                   gate_retries=0, tag="_test_pose_dir")
    m = r["meta"]
    assert m["frames"] == 2
    # cols 在顶层（pack_sheet 的 info），meta 里只有与采样相关的字段
    assert r["cols"] == 2, "cols=0 应自动按帧数排成一行"
    assert Image.open(r["sheet"]).size == (256 * 2, 256)


def test_sdxl_clip_skip_semantics_is_offset_by_two(eng):
    """SDXL 的 clip_skip 语义与 SD1.5 不同——必须换算，不能照抄。

    diffusers 的 SDXL encode_prompt 里：

        if clip_skip is None:
            embeds = hidden_states[-2]        # 默认 = 倒数第二层
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            embeds = hidden_states[-(clip_skip + 2)]

    也就是 **SDXL 默认取的就是倒数第二层**，那已经等价于业界说的
    "clip skip 2"（动漫底模 Illustrious/Animagine 推荐的那档）。

    踩过的坑：直接把 clip_skip=2 传进去会跳到 **-4 层**，过度跳过——
    实测出图一片灰雾（std 从 38 掉到 19，五官全无）。

    所以 n ≤ 2 一律用默认；n > 2 才真的多跳（传给 diffusers 的是 n-2）。
    """
    if "sdxl" not in eng.available_models():
        import pytest
        pytest.skip("本机没有 SDXL 权重")

    def run(cs: int):
        return g.txt2img(eng, prompt="1girl, solo, blue eyes",
                         negative="worst quality", preset="none",
                         width=512, height=512, steps=6, seed=5,
                         model="sdxl", clip_skip=cs, matte=False, tag="cs")

    # 1 和 2 都应当落在默认档（penultimate），不应报错也不应过度跳过
    a = run(1)
    b = run(2)
    assert a["meta"]["clip_skip"] == 2
    assert b["meta"]["clip_skip"] == 2
    # 3 才是真的多跳一层
    c = run(3)
    assert c["meta"]["clip_skip"] == 3


def test_anime_preset_has_quality_tags_and_short_negative():
    """anime 契约：质量词短、负向词短、要求简单背景（游戏素材要能抠图）。

    业界对 Illustrious/Animagine 的共识：
      · 质量词只要 `masterpiece, best quality`，堆更多是负优化
      · 负向词要短——这些模型已经排除了低质量样本，堆词反而压制正常特征
    背景要简单是因为下游要抠图。
    """
    from workflows import presets

    pos, neg = presets.build("1girl, solo, cat ears", None, preset="anime")
    assert "masterpiece" in pos and "best quality" in pos
    assert "simple background" in pos
    # 负向只留必要的几条，不能把"通用负面词表"整段抄进来
    assert "worst quality" in neg and "bad hands" in neg
    assert "watermark" in neg


def test_anime_preset_specifies_style_explicitly():
    """风格必须显式给——Illustrious v0.1 官方就注明未做美学预调。

    实测三档对比（同一提示词、同 seed）：
      不给风格词            → 偏写实厚涂，不是日系动漫
      flat colors+cel shading → 赛璐璐平涂，全身也最完整
      anime style+clean lineart → 线条最锐利
    两个都给能同时拿到「色块分明 + 线条干净」。
    """
    from workflows import presets

    suffix, _ = presets.PRESETS["anime"]
    assert "flat colors" in suffix and "cel shading" in suffix
    assert "clean lineart" in suffix


def test_anime_preset_allows_user_framing_tags():
    """构图 tag（full body / upper body）由用户给，契约不写死。

    实测：不给构图 tag 时前景占 83.5% 且四边贴边；用户加 "full body"
    后前景 24.5%、无贴边。所以契约只约束背景与居中，不抢构图——
    否则想要头像的用户会被强制成全身。
    """
    from workflows import presets

    pos, _ = presets.build("1girl, full body", None, preset="anime")
    assert "full body" in pos                 # 用户给的必须保留
    suffix, _ = presets.PRESETS["anime"]
    assert "full body" not in suffix          # 契约自己不写死
    assert "centered" in suffix


def test_anime_preset_registered():
    """anime 要能在 preset 列表里被选到（前端下拉靠这个）。"""
    from workflows import presets

    assert "anime" in presets.PRESETS
    with pytest.raises(ValueError, match="未知 preset"):
        presets.build("x", None, preset="nope")


# ---------------- topixel 任意图 → 像素 ----------------

def test_image_to_pixel_from_image(eng, assets):
    """有源图时直接转像素：不跑模型，纯确定性，输出是原生像素尺寸。

    这条链路的价值：把「画对人物」和「压成像素」拆开做。让 SD 一次做两件事
    会在两者之间妥协——实测直接画像素常裁到腿、颜色是连续渐变。
    """
    # 造一张「动漫图」：有 alpha、有颜色块
    src = assets / "_wap_src.png"
    im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([90, 60, 166, 150], fill=(200, 40, 50, 255))     # 红甲
    d.ellipse([104, 32, 152, 80], fill=(235, 200, 160, 255))      # 脸
    d.rectangle([96, 150, 116, 220], fill=(40, 60, 200, 255))     # 蓝腿
    d.rectangle([140, 150, 160, 220], fill=(40, 60, 200, 255))
    im.save(src)

    r = g.image_to_pixel(eng, init_image=src, target=(64, 64), colors=16,
                         matte=False, tag="wap_img")
    m = r["meta"]
    assert m["workflow"] == "topixel"
    assert m["generated"] is False, "给了源图就不该再跑模型"
    assert m["pixel"]["size"] == [64, 64], "输出是原生像素尺寸"
    assert m["pixel"]["colors"] <= 16
    assert m["pixel"]["offPalette"] == 0, "不能有板外色"
    assert Path(r["file"]).exists()


def test_image_to_pixel_requires_prompt_or_image(eng):
    """既没源图也没提示词时明确报错，而不是跑一次空生成。"""
    with pytest.raises(ValueError, match="prompt 或 init_image"):
        g.image_to_pixel(eng, prompt="", init_image=None, tag="wap_bad")


def test_image_to_pixel_generates_source_when_prompt_only(eng, assets):
    """没源图就用提示词先生成动漫全身图再转——这是产线的默认用法。"""
    if not eng.available_models() or "sdxl-anime" not in eng.available_models():
        import pytest as _p
        _p.skip("未下载动漫底模")

    r = g.image_to_pixel(
        eng, prompt="1girl, solo, red cape, blue armor, full body",
        target=(64, 64), colors=16, model="sdxl-anime",
        source_width=512, source_height=512, steps=6, seed=7,
        tag="wap_gen")
    m = r["meta"]
    assert m["generated"] is True
    assert m["source_seconds"] is not None, "要记录生成源图花了多久"
    assert m["pixel"]["size"] == [64, 64]
    assert Image.open(r["file"]).size == (64, 64)


def test_image_to_pixel_uses_shared_palette_and_edge_rebuild(eng, assets):
    """色板与增强开关要真的写进 meta（产线据此复现/判断）。"""
    src = assets / "_wap_src2.png"
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 20, 88, 100], fill=(180, 60, 60, 255))
    im.save(src)

    r = g.image_to_pixel(eng, init_image=src, target=(32, 32), colors=8,
                         pre_enhance=1.0, edge_rebuild=True,
                         matte=False, tag="wap_meta")
    m = r["meta"]
    assert m["pre_enhance"] == 1.0
    assert m["edge_rebuild"] is True
    assert m["palette_size"] > 0 and len(m["palette"]) == m["palette_size"]
    assert m["pixel"]["offPalette"] == 0


def test_lora_trigger_is_prepended_before_prompt(tmp_path):
    """后端自动把 LoRA 的触发词前置——单 token 触发词漏一个字符就静默失效。

    Pixelizer 的触发词是 `PixelArt_IL_One_Token_style_M_01`（30 个字符），
    不写完全不生效、写错也完全不生效、**两种情况都不报错**。让用户手抄不是
    "注意看文档"能解决的，所以由后端按 loras.json 自动补。
    """
    from workflows.engine import Engine

    d = tmp_path / "lora"
    d.mkdir(parents=True)
    (d / "a.safetensors").write_bytes(b"")           # 只需存在，不加载
    (d / "loras.json").write_text(json.dumps({
        "a": {"trigger": "pixelart"}, "b": {"trigger": ""}},
        ensure_ascii=False), encoding="utf-8")
    eng = Engine(tmp_path)

    assert g._lora_trigger(eng, "1girl, solo", "a").startswith("pixelart, ")
    # 配置里 trigger 为空（刻意不用）→ 不加
    assert g._lora_trigger(eng, "1girl", "b") == "1girl"
    # 没挂 LoRA → 原样返回
    assert g._lora_trigger(eng, "1girl", "") == "1girl"
    assert g._lora_trigger(eng, "1girl", None) == "1girl"
    # 已经写过就不重复加（大小写不敏感）
    assert g._lora_trigger(eng, "PixelArt, 1girl", "a") == "PixelArt, 1girl"
    # 多个 LoRA 各补各的
    out = g._lora_trigger(eng, "1girl", ["a", "b"])
    assert out.startswith("pixelart, ") and out.count("pixelart") == 1
