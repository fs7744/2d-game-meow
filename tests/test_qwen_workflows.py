"""Qwen-only 的四条「近似工作流」逻辑测试（**不加载模型**）。

覆盖的是「换底模之后语义还对不对」，不是画质：
  · 能力表 / 近似标记（前端靠它过滤与打标）
  · 局部重绘的硬承诺：**选区外像素逐点不变**
  · 条件图分辨率对齐（output_resolution）有没有真的传下去
  · 逐帧尺度的确定性归一（normalize_sequence）
  · 透明底素材重绘后 alpha 不被白底吃掉

模型调用一律用 monkeypatch 换成一张纯色图：这里要验的是**贴回/缩放/落盘**
这条链路，跑真模型只会把单测拖到几分钟且不可复现。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from workflows import generate as g
from workflows import postprocess as pp


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """把产物目录挪到 tmp，别在仓库里堆测试图。"""
    monkeypatch.setattr(g, "ASSETS", tmp_path)
    monkeypatch.setattr(g, "PROJECTS_DIR", tmp_path / "projects")

    calls = []

    def fake_qwen(eng, *, prompt, negative, width, height, steps, seed,
                  ref_image=None, transparent=False, model="qwen-image-2.1",
                  label="w1", cfg=4.0, output_resolution=None,
                  fast=False, fast_lora=True):
        calls.append({"prompt": prompt, "width": width, "height": height,
                      "ref": ref_image, "output_resolution": output_resolution,
                      "steps": steps, "cfg": cfg})
        # 整块重画成绿色：这样「选区内变了、选区外没变」两个断言都能验
        return Image.new("RGBA", (width, height), (0, 200, 0, 255))

    monkeypatch.setattr(g, "_qwen_call", fake_qwen)
    return {"root": tmp_path, "calls": calls}


def _src(tmp_path, size=(256, 256)):
    im = Image.new("RGB", size, (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.rectangle([80, 80, 176, 176], fill=(200, 30, 30))
    p = tmp_path / "src.png"
    im.save(p)
    return p


def _mask(tmp_path, size=(256, 256)):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle([90, 90, 166, 140], fill=255)
    p = tmp_path / "mask.png"
    m.save(p)
    return p


# ---------------- 能力表 ----------------

def test_qwen_supports_every_workflow(eng):
    """Qwen 现在支持 14 个工作流。

    比最初的 10 个多出四个，都是 2026-10-30/31 加的（RunComfy 调研的产物）：
    `outpaint`（扩图，靠 Outpaint LoRA）、`outfit_swap`（换装）、
    `face_swap`（换脸）后两个是「两图 + 结构化指令」，不需要新权重；
    `vnccs_pose`（姿态迁移，挂 VNCCS LoRA + 官方双图指令，不用 ControlNet）。
    """
    got = set(eng.workflows_of("qwen-image-2.1"))
    assert got == {"txt2img", "img2img", "inpaint", "poseswap", "matte",
                   "ai_repaint", "pixelify", "pose_clone", "sprite_swap",
                   "topixel", "outpaint", "outfit_swap", "face_swap",
                   "vnccs_pose"}
    # require() 不该再对 Qwen 抛错——它曾经是「Qwen 不支持 ControlNet 类节点」的闸门
    for wf in got:
        eng.require("qwen-image-2.1", wf)


def test_qwen_approx_flags(eng):
    """近似集合必须**恰好**是那两个：标错会让用户以为 Qwen 有真 inpaint。

    inpaint / ai_repaint 已移出这张表——它们现在走**真蒙版路径**
    （workflows/qwen_mask.py）：蒙版内换噪声、每步把蒙版外拉回原 latent，
    模型**看得见蒙版**，所以不再是"靠提示词方位词猜哪儿要改"的近似实现。
    """
    assert set(eng.approx_of("qwen-image-2.1")) == {"poseswap", "sprite_swap"}
    assert not eng.is_approx("qwen-image-2.1", "txt2img")
    assert not eng.is_approx("qwen-image-2.1", "inpaint")
    assert not eng.is_approx("qwen-image-2.1", "ai_repaint")
    # SD 没有近似实现（它有 ControlNet / mask 通道）。
    # 这里查表而不是 `approx_of("sd15")`：SD 权重已归档，arch() 会因缺权重抛错
    assert "sd15" not in eng._APPROX and "sdxl" not in eng._APPROX


def test_require_reports_available_workflows(eng, monkeypatch):
    monkeypatch.setitem(eng._ARCH_WORKFLOWS, "qwen21", frozenset({"txt2img"}))
    with pytest.raises(ValueError) as ei:
        eng.require("qwen-image-2.1", "inpaint")
    assert "文生图" in str(ei.value)


# ---------------- 尺寸 / 方位 / 提示词 ----------------

@pytest.mark.parametrize("size,expect_max", [
    ((768, 768), 768), ((1536, 1024), 1024), ((64, 64), 64), ((1000, 700), 1024),
])
def test_qwen_work_size(size, expect_max):
    w, h = g._qwen_work_size(size)
    assert w % 16 == 0 and h % 16 == 0
    assert max(w, h) <= max(expect_max, 16)
    # 只缩不放：小图放大不会多出信息
    assert w <= size[0] and h <= size[1]


def test_region_words():
    m = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(m).rectangle([5, 5, 25, 25], fill=255)
    assert g._region_words(m) == "top-left"
    m2 = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(m2).rectangle([60, 60, 90, 90], fill=255)
    assert g._region_words(m2) == "bottom-right"
    with pytest.raises(ValueError):
        g._region_words(Image.new("L", (10, 10), 0))


def test_edit_prompt_carries_region_and_rules():
    p = g._qwen_edit_prompt("change the armor to green.", "middle-center")
    assert "middle-center" in p
    assert "face" in p and "seam" in p
    assert not p.endswith("..")


# ---------------- 局部重绘的硬承诺 ----------------

def test_qwen_inpaint_keeps_outside_pixels(eng, sandbox, tmp_path):
    """真蒙版路径：蒙版外几乎不变（VAE 重建本身的误差量级），选区内被重画。

    注意「几乎」不是「逐点」：真蒙版路径让**模型**去保持选区外的原样，
    剩余差异来自 VAE 编码-解码本身（实测 19~46/255，视觉上无差）。
    旧的整图改图+贴回路径才是逐点为 0，但那时模型看不见蒙版、指令常不执行。
    """
    src, msk = _src(tmp_path), _mask(tmp_path)
    r = g.inpaint(eng, image=src, mask=msk, prompt="emerald green armor",
                  steps=8, model="qwen-image-2.1", scope="t")
    meta = r["meta"]
    assert meta["backend"] == "qwen-mask-inpaint" and meta["approx"] is False
    assert meta["strategy"] == "mask"

    out = Image.open(r["file"]).convert("RGB")
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.int16)
    b = np.asarray(out, dtype=np.int16)
    m = np.asarray(Image.open(msk).convert("L")) > 127
    # 蒙版外：误差在 VAE 重建量级内（不是逐点 0，但也不该有结构性变化）
    assert np.abs(a - b)[~m].max() <= 60
    assert np.abs(a - b)[m].mean() > 10        # 选区内确实被重画


def test_qwen_inpaint_whole_strategy_still_available(eng, sandbox, tmp_path):
    """旧的「整图改图 + 蒙版贴回」路径保留：逐点为 0，但属近似实现。"""
    src, msk = _src(tmp_path), _mask(tmp_path)
    r = g.inpaint(eng, image=src, mask=msk, prompt="emerald green armor",
                  steps=8, model="qwen-image-2.1", scope="t", strategy="whole")
    meta = r["meta"]
    assert meta["backend"] == "qwen-edit" and meta["approx"] is True
    assert meta["outside_max_diff"] == 0


def test_qwen_call_aligns_output_resolution(eng, monkeypatch):
    """条件图分辨率必须等于目标边长，不是管线默认的 1024。

    这个参数在 `_qwen_call` 里补齐（而不是每个调用点各写一遍），所以直接测
    它：管线与推理都换成假的，只看真正传进 diffusers 的 kwargs。
    """
    seen = {}

    class _Out:
        def __init__(self, kw):
            self.images = [Image.new("RGBA", (kw["width"], kw["height"]))]

    monkeypatch.setattr(eng, "txt2img_pipe", lambda model: object())
    monkeypatch.setattr(g, "_infer", lambda e, pipe, kw, **k: (seen.update(kw)
                                                               or _Out(kw)))
    g._qwen_call(eng, prompt="x", negative="n", width=768, height=512,
                 steps=8, seed=1)
    assert seen["output_resolution"] == 768
    # 显式给的优先（局部编辑会按工作尺寸传）
    g._qwen_call(eng, prompt="x", negative="n", width=768, height=512,
                 steps=8, seed=1, output_resolution=512)
    assert seen["output_resolution"] == 512


def test_qwen_inpaint_rejects_empty_prompt(eng, sandbox, tmp_path):
    with pytest.raises(ValueError):
        g.inpaint(eng, image=_src(tmp_path), mask=_mask(tmp_path), prompt="  ",
                  steps=8, model="qwen-image-2.1", scope="t")


def test_qwen_ai_repaint_both_strategies(eng, sandbox, tmp_path):
    src, msk = _src(tmp_path, (192, 192)), _mask(tmp_path, (192, 192))
    for strategy in ("whole", "crop"):
        r = g.ai_repaint(eng, image=src, mask=msk, prompt="green armor",
                         steps=8, model="qwen-image-2.1", strategy=strategy,
                         scope="t")
        assert r["meta"]["outside_max_diff"] == 0, strategy
        assert r["meta"]["strategy"] == strategy
        assert r["meta"]["backend"] == "qwen-edit"


def test_qwen_ai_repaint_rejects_bad_strategy(eng, sandbox, tmp_path):
    with pytest.raises(ValueError):
        g.ai_repaint(eng, image=_src(tmp_path), mask=_mask(tmp_path), prompt="x",
                     steps=8, model="qwen-image-2.1", strategy="magic", scope="t")


# ---------------- alpha 与逐帧尺度 ----------------

def test_restore_alpha_keeps_transparency(tmp_path):
    src = Image.new("RGBA", (32, 32), (10, 10, 10, 0))
    ImageDraw.Draw(src).rectangle([8, 8, 24, 24], fill=(200, 0, 0, 255))
    gen = Image.new("RGBA", (32, 32), (0, 200, 0, 255))     # 模型出了不透明的图
    out = g._restore_alpha(gen, src)
    assert np.asarray(out.getchannel("A")).min() == 0
    # 没有透明底时不该动
    opaque = Image.new("RGBA", (32, 32), (1, 2, 3, 255))
    assert g._restore_alpha(gen, opaque).getchannel("A").getextrema() == (255, 255)


def test_normalize_sequence_equalizes_subject_scale():
    """三帧人物高度 30/60/120px → 归一后高度一致，且都在画布里。"""
    frames = []
    for h in (30, 60, 120):
        im = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        # 竖条（宽 30 / 高 30~120）：避免触发「宽度不超画布 96%」的限幅，
        # 那个限幅是给宽体角色用的，会把高度也一起压下来
        ImageDraw.Draw(im).rectangle([113, 200 - h, 143, 200], fill=(255, 0, 0, 255))
        frames.append(im)
    out = pp.normalize_sequence(frames)
    hs = [f.getchannel("A").getbbox()[3] - f.getchannel("A").getbbox()[1]
          for f in out]
    assert max(hs) - min(hs) <= 2
    for f in out:
        bb = f.getchannel("A").getbbox()
        assert bb[1] >= 0 and bb[3] <= 256


def test_normalize_sequence_respects_fill_and_clamps():
    frames = []
    for h in (40, 80):
        im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(im).rectangle([85, 190 - h, 115, 190], fill=(0, 0, 255, 255))
        frames.append(im)
    out = pp.normalize_sequence(frames, fill=0.7)
    for f in out:
        bb = f.getchannel("A").getbbox()
        assert abs((bb[3] - bb[1]) - 140) <= 2
    # 显式给一个荒唐的小值也不能通过：下限 lo=0.55 会把「拇指大的小人」拉回来
    small = pp.normalize_sequence(frames, fill=0.1)
    bb = small[0].getchannel("A").getbbox()
    assert (bb[3] - bb[1]) >= int(0.55 * 200) - 2


# ---------------- 逐帧工作流能跑通（不加载模型） ----------------

def _skeleton(tmp_path, n=3):
    d = tmp_path / "skel"
    d.mkdir()
    for i in range(n):
        im = Image.new("RGB", (256, 256), (255, 255, 255))
        ImageDraw.Draw(im).line([128, 40, 128, 200], fill=(0, 0, 0), width=6)
        im.save(d / f"frame_{i:02d}.png")
    return d


def test_qwen_poseswap_packs_sheet(eng, sandbox, tmp_path):
    r = g.poseswap(eng, pose_images=_skeleton(tmp_path), prompt="a cat knight",
                   steps=8, model="qwen-image-2.1", width=256, height=256,
                   matte=False, candidates=1, gate=False, scope="t")
    assert r["meta"]["workflow"] == "w4"
    assert r["meta"]["backend"] == "qwen-pose-ref"
    assert r["meta"]["approx"] is True
    assert r["meta"]["frames"] == 3
    assert len(sandbox["calls"]) == 3


def test_qwen_sprite_swap_keeps_cell_size(eng, sandbox, tmp_path):
    sheet = Image.new("RGBA", (3 * 64, 64), (0, 0, 0, 0))
    for i in range(3):
        im = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse([6, 6, 42, 42], fill=(30, 60, 200, 255))
        sheet.paste(im, (i * 64 + 8, 8), im)
    sp = tmp_path / "sheet.png"
    sheet.save(sp)
    char = tmp_path / "char.png"
    Image.new("RGBA", (128, 128), (200, 40, 40, 255)).save(char)

    r = g.sprite_swap(eng, sheet_image=sp, char_image=char, cols=3, rows=1,
                      steps=8, model="qwen-image-2.1", width=256, height=256,
                      matte=False, palette=16, scope="t")
    m = r["meta"]
    assert m["workflow"] == "wss" and m["approx"] is True
    assert m["cell"] == [64, 64]
    assert m["frames"] == 3
    assert Image.open(r["file"]).size == (64 * 3, 64)


# ---------------- 透明底：提示词不能和 alpha 声明打架 ----------------

def test_strip_bg_words_removes_conflicts():
    """透明底时「white background」这类措辞必须被清掉。

    实测同一个 seed：带这句时 Qwen 的 alpha 是一层雾（81.5% 前景、77.5% 停在
    16~240 的中间值），清掉后是干净的（31% 前景、67.8% 全透明、中间值 2.9%）。
    """
    p, hit = g._strip_bg_words("a pixel art cat, plain white background, full body")
    assert hit and "white background" not in p.lower()
    assert p == "a pixel art cat, full body"
    # 删完不能留悬空介词
    assert g._strip_bg_words("a cat on a white background")[0] == "a cat"
    # 没有冲突词就不该动
    same, hit2 = g._strip_bg_words("a cat, golden armor")
    assert not hit2 and same == "a cat, golden armor"


# ---------------- 边界：Qwen 路径不许碰 SD 管线 ----------------
#
# engine.py 里还留着 SD 时代的管线方法（inpaint_pipe / pose_pipe / controlnet /
# apply_lora / refiner_pipe / animatediff_pipe），它们是回退路径的一部分。
# 这条测试把「Qwen 运行态下走不到」变成可执行的断言：全换成 raise，
# 四个 Qwen 工作流照跑——一旦哪天有人把 SD 分支接回主路径，这里立刻红。

def test_qwen_workflows_never_touch_sd_pipelines(eng, sandbox, tmp_path,
                                                 monkeypatch):
    def boom(*a, **k):
        raise AssertionError("Qwen 路径调用了 SD 管线方法")

    for name in ("inpaint_pipe", "pose_pipe", "controlnet", "apply_lora",
                 "refiner_pipe", "animatediff_pipe", "apply_distill",
                 "ipa_weight"):
        monkeypatch.setattr(eng, name, boom, raising=False)

    src, msk = _src(tmp_path), _mask(tmp_path)
    g.inpaint(eng, image=src, mask=msk, prompt="green armor", steps=8,
              model="qwen-image-2.1", scope="t")
    g.ai_repaint(eng, image=src, mask=msk, prompt="green armor", steps=8,
                 model="qwen-image-2.1", scope="t")
    g.poseswap(eng, pose_images=_skeleton(tmp_path), prompt="cat knight",
               steps=8, model="qwen-image-2.1", width=256, height=256,
               matte=False, candidates=1, gate=False, scope="t")


# ---------------- 非人形主体：闸门不该空转 ----------------
#
# 史莱姆、机甲、道具这类主体，DWPose 一个关节都检不出来。原来的逻辑是
# 「闸门没过就补提一次」——对人形有意义（没画好就重来），对非人形则是
# 纯烧钱：每帧 25~30s 白花，而且最后 meta 里 gate_passed=0 看起来像生成失败。

def test_poseswap_skips_retries_for_non_humanoid(eng, sandbox, tmp_path,
                                                  monkeypatch):
    from workflows import quality

    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (256, 256), (255, 255, 255)).save(pose_dir / f"p{i}.png")

    # 姿态提取器：假装能跑，但一个关节也检不出（史莱姆就是这样）
    class _NoPoints:
        def visible_points(self, img, thr=0.25):
            return {}

    monkeypatch.setattr(eng, "pose_extractor", lambda: _NoPoints())

    # 评估结果：画面里有前景（coverage 0.2），但关节数为 0
    monkeypatch.setattr(quality, "evaluate",
                        lambda img, extractor=None, prev=None, min_points=8:
                        {"points": 0, "coverage": 0.2, "edge": [], "drift": 0.0,
                         "score": 1.0, "passed": False})

    r = g.poseswap(eng, pose_images=list(pose_dir.glob("*.png")),
                   prompt="a blue slime monster", steps=4, width=256, height=256,
                   matte=False, candidates=1, gate=True, gate_retries=1,
                   scope="t")
    meta = r["meta"]

    assert meta["gate_non_humanoid"] is True
    assert "不是人形" in (meta["gate_note"] or "")
    # 2 帧 × 只出 1 次 = 2 次模型调用；补提被关掉了
    assert len(sandbox["calls"]) == 2, f"仍在补提：{len(sandbox['calls'])} 次调用"
    assert meta["attempts"] == 2


def test_poseswap_keeps_retries_for_humanoid(eng, sandbox, tmp_path,
                                             monkeypatch):
    """人形主体仍然保留补提（别把这个优化做过头）。"""
    from workflows import quality

    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (256, 256), (255, 255, 255)).save(pose_dir / f"p{i}.png")

    class _Points:
        def visible_points(self, img, thr=0.25):
            return {k: (0.5, 0.5) for k in range(14)}

    monkeypatch.setattr(eng, "pose_extractor", lambda: _Points())
    monkeypatch.setattr(quality, "evaluate",
                        lambda img, extractor=None, prev=None, min_points=8:
                        {"points": 14, "coverage": 0.2, "edge": [], "drift": 0.0,
                         "score": 1.0, "passed": True})

    r = g.poseswap(eng, pose_images=list(pose_dir.glob("*.png")),
                   prompt="a cat knight", steps=4, width=256, height=256,
                   matte=False, candidates=1, gate=True, gate_retries=1,
                   scope="t")
    meta = r["meta"]
    assert meta["gate_non_humanoid"] is False
    assert meta["gate_note"] is None
    # 闸门一次就过 → 每帧只出 1 次
    assert len(sandbox["calls"]) == 2


# ---------------- 两阶段：先出姿态草图，再换皮 ----------------

def test_poseswap_two_stage_calls_model_twice(eng, sandbox, tmp_path):
    """两阶段模式：每帧两次模型调用（草图 + 换皮），且角色图排在前面。"""
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    Image.new("RGB", (256, 256), (255, 255, 255)).save(pose_dir / "p0.png")
    char = tmp_path / "char.png"
    Image.new("RGBA", (256, 256), (40, 170, 200, 255)).save(char)

    r = g.poseswap(eng, pose_images=[pose_dir / "p0.png"], ref_image=char,
                   prompt="a cat knight", steps=4, width=256, height=256,
                   matte=False, candidates=1, gate=False, two_stage=True,
                   scope="t")
    assert r["meta"]["two_stage"] is True
    assert len(sandbox["calls"]) == 2, "两阶段应该每帧两次模型调用"
    # 第二次调用：角色图在前、草图在后（顺序是有讲究的，见 PLAN §2.8）
    reskin = sandbox["calls"][1]
    assert isinstance(reskin["ref"], list) and len(reskin["ref"]) == 2


def test_poseswap_two_stage_falls_back_without_ref(eng, sandbox, tmp_path):
    """没有角色参考图时，两阶段无从谈起（没有身份可换），自动退回单阶段。"""
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    Image.new("RGB", (256, 256), (255, 255, 255)).save(pose_dir / "p0.png")

    r = g.poseswap(eng, pose_images=[pose_dir / "p0.png"],
                   prompt="a cat knight", steps=4, width=256, height=256,
                   matte=False, candidates=1, gate=False, two_stage=True,
                   scope="t")
    assert r["meta"]["two_stage"] is False
    assert len(sandbox["calls"]) == 1


# ---------------- 真 CFG：时间翻倍的那一半 ----------------
#
# 实测（2026-10-30）：768²/24 步的总耗时里 DiT 前向占 86%，而
# `true_cfg_scale > 1 且给了负向词` 时管线每个 step 跑**两遍** transformer
# （有条件 + 无条件）→ 24 步 = 48 次前向。关掉后 21.1s → 13.6s（-36%），
# 6 个提示词的画质对照没有下降（assets/_cfg_ab/cmp.png）。

def test_negative_prompt_only_sent_with_true_cfg(eng, monkeypatch):
    """真 CFG 关着时**不能**传负向词——传了就会触发双次前向。

    这里直接调 `_qwen_call`（而不是走 monkeypatch 掉的假实现），
    用一个假管线把 kwargs 截下来看。
    """
    seen = {}

    class FakePipe:
        def __call__(self, **kw):
            seen.clear()
            seen.update(kw)

            class _Out:
                images = [Image.new("RGBA", (16, 16))]
            return _Out()

    monkeypatch.setattr(eng, "txt2img_pipe",
                        lambda model="qwen-image-2.1": FakePipe())

    g._qwen_call(eng, prompt="x", negative="no blur", width=256, height=256,
                 steps=8, seed=1, cfg=1.0)
    assert "negative_prompt" not in seen
    assert seen["true_cfg_scale"] == 1.0

    g._qwen_call(eng, prompt="x", negative="no blur", width=256, height=256,
                 steps=8, seed=1, cfg=4.0)
    assert seen["negative_prompt"] == "no blur"
    assert seen["true_cfg_scale"] == 4.0


def test_cfg_reaches_the_model_call(eng, sandbox):
    """cfg 必须真的传到推理层——以前 txt2img 的 cfg 参数根本没往下传
    （内部实现 `_qwen_txt2img` 的签名里压根没有 cfg，meta 里写死 4.0）。"""
    r = g.txt2img(eng, prompt="a cat", width=256, height=256, steps=8,
                  seed=1, preset="none", pixel=0, palette=0, scope="t")
    assert r["meta"]["cfg"] == 1.0
    assert r["meta"]["true_cfg"] is False


# ---------------- 多张参考图 ----------------

def test_ref_image_list_merges_single_and_many():
    """`char_image`（旧字段）与 `char_images`（新字段）要能合并。

    前端老版本只传单张，新版本传数组；两边都给时以数组为准——数组里
    已经包含第一张了，重复合并会让同一张图在条件里出现两次。
    """
    # 只给多张
    assert g._ref_image_list(None, ["a.png", "b.png"]) == ["a.png", "b.png"]
    # 只给单张（旧客户端）
    assert g._ref_image_list("a.png", None) == ["a.png"]
    assert g._ref_image_list("a.png", []) == ["a.png"]
    # 都给：以数组为准
    assert g._ref_image_list("a.png", ["a.png", "b.png"]) == ["a.png", "b.png"]
    # 空的要报错，不能静默出一张无条件的图
    with pytest.raises(ValueError):
        g._ref_image_list(None, [])
    # 超过上限要报错（Qwen 实测 10 张还能跑，这里收在 6 张）
    with pytest.raises(ValueError, match=r"最多"):
        g._ref_image_list(None, [f"{i}.png" for i in range(g.MAX_REF_IMAGES + 1)])


def test_multi_swap_prompt_declares_the_same_garment():
    """多张服装图时必须显式说「这是同一件」。

    只把 `<image3> <image4>` 拼进去、不说清是同一件，模型会当成
    好几套衣服，实测出现上下身拼装。
    """
    one = g.swap_mod.swap_prompt("outfit", n_refs=1)
    assert "<image3>" not in one and "<image2>" in one

    three = g.swap_mod.swap_prompt("outfit", n_refs=3)
    assert "<image4>" in three
    assert "SAME garment" in three
    assert "do not blend several different costumes" in three

    # 换脸同理：多张是同一个人
    two = g.swap_mod.swap_prompt("face", n_refs=2)
    assert "<image3>" in two
    assert "SAME person" in two

    # 用户的补充要求要接在后面
    extra = g.swap_mod.swap_prompt("outfit", "外套改成皮夹克", n_refs=2)
    assert extra.endswith("外套改成皮夹克")


# ---------------- 逐帧引擎（wss / VNCCS） ----------------

def test_sprite_swap_exposes_engine_switch():
    """`sprite_swap` 要把 `engine` 接住，否则前端传了会被 app.py 当成多余键丢掉。"""
    import inspect
    params = inspect.signature(g.sprite_swap).parameters
    assert "engine" in params
    assert params["engine"].default == "wpc"      # 默认仍走旧路线
    assert "lora_scale" in params
    # w4（姿态序列）也要能切引擎
    assert "engine" in inspect.signature(g.poseswap).parameters
    assert "char_images" in inspect.signature(g.poseswap).parameters
    assert "lora_scale" in inspect.signature(g.poseswap).parameters
    # vnccs_pose 本身就是 VNCCS 路线，不需要 engine 开关，但要能调 LoRA 权重
    assert "lora_scale" in inspect.signature(g.vnccs_pose).parameters
    # 多图：姿态迁移两条路 + 换装 + 换脸都要接得住
    assert "char_images" in inspect.signature(g.vnccs_pose).parameters
    assert "char_images" in inspect.signature(g.pose_clone).parameters
    assert "outfits" in inspect.signature(g.outfit_swap).parameters
    assert "faces" in inspect.signature(g.face_swap).parameters


def test_vnccs_lora_must_exist(eng):
    """VNCCS 引擎依赖 models/lora 下的权重，缺了要给明确报错而不是静默降级。

    这条不加载模型——只确认 `Engine.VNCCS_LORA` 指的文件在磁盘上，
    以及缺文件时抛的是 FileNotFoundError（信息里带下载地址）。
    """
    assert eng.VNCCS_LORA == "vnccs_pose"
    lora = eng.lora_dir() / f"{eng.VNCCS_LORA}.safetensors"
    assert lora.is_file(), f"缺 VNCCS LoRA：{lora}"
    # 权重里不该再有 gate_up（ComfyUI 的合并版 SwiGLU，diffusers 认不得）
    from safetensors import safe_open
    with safe_open(str(lora), framework="pt") as f:
        keys = list(f.keys())
    assert not [k for k in keys if "gate_up" in k], \
        "gate_up 没拆干净：这部分权重会被 load_lora_weights 静默丢掉"
    # 拆分后每块应该有 gate_layer 与 proj 两组
    assert [k for k in keys if "img_mlp.gate_layer" in k]
    assert [k for k in keys if "img_mlp.proj." in k]


def test_skeleton_background_is_whitened():
    """黑底骨架必须转白底，否则模型把角色画成背面（实测三帧全是背影）。

    判据用饱和度：OpenPose 的线是纯色，背景是纯黑（无彩色分量）。
    """
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    import numpy as np
    black = _Image.new("RGB", (64, 64), (0, 0, 0))
    _ImageDraw.Draw(black).line([(10, 10), (50, 50)], fill=(255, 0, 0), width=6)
    out = g._vnccs_skeleton(black)
    arr = np.asarray(out)
    assert arr[2, 2].tolist() == [255, 255, 255], "背景没被换成白色"
    # 红线要保留
    assert (arr[..., 0] > 200).any() and (arr[..., 1] < 60).any(), "彩色线被误伤了"
    # 深蓝线（饱和度高）也要保留
    blue = _Image.new("RGB", (64, 64), (0, 0, 0))
    _ImageDraw.Draw(blue).line([(10, 10), (50, 50)], fill=(0, 0, 120), width=6)
    barr = np.asarray(g._vnccs_skeleton(blue))
    assert (barr[..., 2] > 80).any(), "深蓝线被当成背景抹掉了"


def test_poseswap_meta_has_no_duplicate_lora_keys():
    """meta 里不能有第二个 `lora` 键。

    重复键会让前面写的值**静默被覆盖**——w4 之前就是这样，
    engine 记成了 vnccs、lora 却显示 None，看不出用的哪个 LoRA。
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(g._qwen_poseswap))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        dup = {k for k in keys if keys.count(k) > 1}
        assert not dup, f"meta 里有重复键：{dup}"


# --------------------------------------------------------------------------- #
# 透明底：模型的 alpha 不可信时要重抠（「透明底怎么这么多白的」的看守）
# --------------------------------------------------------------------------- #

def _foggy(alpha_bg: int) -> Image.Image:
    """一张 256² 图：白背景 + 中间一个深色方块。

    `alpha_bg=180` 模拟 Qwen 画坏的雾状 alpha（既不全透也不全实）；
    `alpha_bg=0` 是模型正常输出的干净透明底。
    """
    im = Image.new("RGBA", (256, 256), (250, 250, 250, alpha_bg))
    ImageDraw.Draw(im).rectangle([88, 88, 168, 200], fill=(30, 30, 40, 255))
    return im


def test_foggy_alpha_is_recut_not_quantized_into_white(eng):
    """**雾状 alpha 必须在量化之前被识破并重抠。**

    用户报的原话是「为何透明背景这么多白色的」。链路是这样坏的：
    模型输出的背景 alpha 是 128~240 的雾 → `quantize_palette` 的
    `>127 → 255` 把它实化 → 背景变成两大块实心白。
    一条 1024² 的实测图里，雾占 68%，实化后不透明率从 14% 跳到 82%。

    判据用「不透明占比」：重抠之后只剩中间那个方块（约 10%），
    如果没有重抠，整张图（≈100%）都会被判成不透明。
    """
    info: dict = {}
    out, _edge, _subj = g._postprocess(
        _foggy(180), eng, matte=False, matte_backend="color", denoise=0,
        pixel=0, palette=16, clean_iso=0, info=info, transparent=True)
    a = np.asarray(out)[:, :, 3]
    assert info.get("alpha_recut") is True, "雾状 alpha 没被识破"
    assert (a > 240).mean() < 0.35, f"背景被实化了：{(a > 240).mean():.0%} 不透明"
    assert ((a >= 16) & (a <= 240)).mean() == 0, "量化后不应留下半透明像素"


def test_clean_alpha_is_reused_not_recut(eng):
    """模型画干净时要**直接用**它的 alpha——重抠会把白毛/白鞋咬掉。"""
    info: dict = {}
    out, _edge, _subj = g._postprocess(
        _foggy(0), eng, matte=False, matte_backend="color", denoise=0,
        pixel=0, palette=16, clean_iso=0, info=info, transparent=True)
    assert info.get("alpha_reused") is True
    assert info.get("alpha_recut") is None
    a = np.asarray(out)[:, :, 3]
    assert 0.05 < (a > 240).mean() < 0.35, (a > 240).mean()


# --------------------------------------------------------------------------- #
# 透明底：模型的 alpha 不可信时要重抠（「透明底怎么这么多白的」的看守）
# --------------------------------------------------------------------------- #

def _foggy(alpha_bg: int) -> Image.Image:
    """一张 256² 图：白背景 + 中间一个深色方块。

    `alpha_bg=180` 模拟 Qwen 画坏的雾状 alpha（既不全透也不全实）；
    `alpha_bg=0` 是模型正常输出的干净透明底。
    """
    im = Image.new("RGBA", (256, 256), (250, 250, 250, alpha_bg))
    ImageDraw.Draw(im).rectangle([88, 88, 168, 200], fill=(30, 30, 40, 255))
    return im


def test_foggy_alpha_is_recut_not_quantized_into_white(eng):
    """**雾状 alpha 必须在量化之前被识破并重抠。**

    用户报的原话是「为何透明背景这么多白色的」。链路是这样坏的：
    模型输出的背景 alpha 是 128~240 的雾 → `quantize_palette` 的
    `>127 → 255` 把它实化 → 背景变成两大块实心白。
    一条 1024² 的实测图里，雾占 68%，实化后不透明率从 14% 跳到 82%。

    判据用「不透明占比」：重抠之后只剩中间那个方块（约 10%），
    如果没有重抠，整张图（≈100%）都会被判成不透明。
    """
    info: dict = {}
    out, _edge, _subj = g._postprocess(
        _foggy(180), eng, matte=False, matte_backend="color", denoise=0,
        pixel=0, palette=16, clean_iso=0, info=info, transparent=True)
    a = np.asarray(out)[:, :, 3]
    assert info.get("alpha_recut") is True, "雾状 alpha 没被识破"
    assert (a > 240).mean() < 0.35, f"背景被实化了：{(a > 240).mean():.0%} 不透明"
    assert ((a >= 16) & (a <= 240)).mean() == 0, "量化后不应留下半透明像素"


def test_clean_alpha_is_reused_not_recut(eng):
    """模型画干净时要**直接用**它的 alpha——重抠会把白毛/白鞋咬掉。"""
    info: dict = {}
    out, _edge, _subj = g._postprocess(
        _foggy(0), eng, matte=False, matte_backend="color", denoise=0,
        pixel=0, palette=16, clean_iso=0, info=info, transparent=True)
    assert info.get("alpha_reused") is True
    assert info.get("alpha_recut") is None
    a = np.asarray(out)[:, :, 3]
    assert 0.05 < (a > 240).mean() < 0.35, (a > 240).mean()
