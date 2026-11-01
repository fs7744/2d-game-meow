"""表情批量生成（workflows/emotiongen.py，移植自 VNCCS 的 QI2 emotion crop 分支）。

**这一组测的是"几何有没有抄对"**：pad 到 32 倍数、羽化蒙版、贴回坐标、
推理尺寸公式、prompt 逐字。这些地方错一点点，表现都只是"贴回去有一圈脏边"
或"只有脸变了一半"——肉眼很难判断，必须靠断言钉死。

出图本身用**假的 `edit`**（返回纯色），不碰 GPU。真模型的验证在
`test_characters.py` 那条真实出图链路上，这里不重复烧卡。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workflows import characters as ch
from workflows import emotiongen as eg


# ---------------------------------------------------------------- 几何


def test_pad_to_square_multiple_uses_replicate():
    """官方 `:1571`：补到 32 的倍数，**用边缘复制**补边（不是黑/灰）。"""
    img = Image.new("RGB", (100, 60), (10, 20, 30))
    # 右边缘涂一条红，用来验"补出来的边是复制的，不是填充色"
    arr = np.asarray(img).copy()
    arr[:, 99] = (255, 0, 0)
    arr[59, :] = (0, 255, 0)
    out, padding = eg.pad_to_square_multiple(Image.fromarray(arr), 32)
    assert out.size == (128, 128), out.size
    assert padding == (14, 34, 14, 34), padding       # (left, top, right, bottom)
    a = np.asarray(out)
    # 右侧补出来的列 = 原图最右那一列（红）；左侧补的是原图最左列
    assert tuple(a[40, 127]) == (255, 0, 0)
    assert tuple(a[40, 0]) == (10, 20, 30)
    # 下方补出来的行 = 原图最下那一行（绿）
    assert tuple(a[127, 40]) == (0, 255, 0)
    assert tuple(a[0, 40]) == (10, 20, 30)


def test_pad_already_aligned_is_noop():
    img = Image.new("RGB", (64, 64), (1, 2, 3))
    out, padding = eg.pad_to_square_multiple(img, 32)
    assert out.size == (64, 64)
    assert padding == (0, 0, 0, 0)


def test_crop_feather_mask_ramps_from_zero():
    """官方 `:1516`：中间 1、四边 0，过渡宽度 = feather。"""
    m = eg.crop_feather_mask(100, 100, feather=10)
    assert m.shape == (100, 100)
    assert m[50, 50] == pytest.approx(1.0)
    assert m[0, 50] == pytest.approx(0.0)
    assert m[50, 0] == pytest.approx(0.0)
    # 第 5 行距离上边 5px → 5/10 = 0.5
    assert m[5, 50] == pytest.approx(0.5, abs=1e-6)


def test_paste_mask_is_softened_ramp():
    """贴回蒙版 = 斜坡蒙版再高斯模糊（官方 `_blur_detail_mask:1449`）。"""
    m = eg.paste_mask(100, 100, feather=5)
    assert m.shape == (100, 100)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    assert m[50, 50] > 0.9
    # 模糊之后边缘不再是 0/1 的二值跳变，而是一条连续的带
    edge = m[0, 50]
    assert 0.0 < edge < 0.9


def test_paste_crop_direct_only_touches_region():
    """官方 `:1483`：区域外逐点不变；区域中心变成补丁色。"""
    base = Image.new("RGB", (200, 200), (5, 6, 7))
    patch = Image.new("RGB", (50, 50), (200, 10, 10))
    out = eg.paste_crop_direct(base, patch, (60, 80, 110, 130))
    a = np.asarray(out)
    b = np.asarray(base)
    outside = np.ones((200, 200), dtype=bool)
    outside[80:130, 60:110] = False
    assert np.array_equal(a[outside], b[outside]), "区域外被改动了"
    assert tuple(a[100, 85]) == (200, 10, 10)


def test_paste_zero_mask_keeps_original():
    base = Image.new("RGB", (100, 100), (5, 6, 7))
    patch = Image.new("RGB", (40, 40), (200, 10, 10))
    out = eg.paste_crop_direct(base, patch, (10, 10, 50, 50),
                               np.zeros((40, 40), dtype=np.float32))
    assert np.array_equal(np.asarray(out), np.asarray(base))


def test_paste_does_not_touch_alpha_when_patch_is_rgb():
    """**复用原 alpha 的前提**：patch 是 RGB 时不动 base 的 alpha（官方 `:1490`）。"""
    base = Image.new("RGBA", (60, 60), (10, 10, 10, 0))
    patch = Image.new("RGB", (20, 20), (255, 0, 0))
    out = eg.paste_crop_direct(base, patch, (20, 20, 40, 40))
    assert np.asarray(out)[:, :, 3].max() == 0, "透明区域内被贴出了不透明方块"


def test_crop_target_size_matches_official_formula():
    """官方 `:1636` + `ImageScaleToTotalPixels`：按总像素等比，再向下取 32 倍数。"""
    # 2048 → 2.0 MP：1024² 放大 sqrt(2) ≈ 1.414 → 1448 → 向下取 32 → 1440
    assert eg.crop_target_size(1024, 1024, 2048) == (1440, 1440)
    # 1024 → 1.0 MP：尺寸不变（已经是 1024）
    assert eg.crop_target_size(1024, 1024, 1024) == (1024, 1024)
    # 长宽比要保住
    w, h = eg.crop_target_size(512, 1024, 1024)
    assert w % 32 == 0 and h % 32 == 0
    assert abs((h / w) - 2.0) < 0.05, (w, h)


def test_clamp_size_caps_long_side():
    assert eg.clamp_size(1440, 1440, 1024) == (1024, 1024)
    assert eg.clamp_size(512, 256, 1024) == (512, 256)
    w, h = eg.clamp_size(1440, 720, 1024)
    assert w == 1024 and h % 32 == 0 and h < 720


def test_cut_padding_removes_exactly_the_pad():
    img = Image.new("RGB", (128, 128), (9, 9, 9))
    out = eg.cut_padding(img, (14, 34, 14, 34))
    assert out.size == (100, 60), out.size


# ---------------------------------------------------------------- prompt（逐字）


def test_emotion_template_is_official_text():
    """官方 `character_generator.py:740-745`。改一个字就红。"""
    assert eg.EMOTION_PROMPT_TEMPLATE == (
        "Upscale face image.\n"
        "Make character's face emotion {emotion}\n"
        "Change only face. Keep original neck colour, clothes and hairs\n"
        "keep character's clothes"
    )


def test_emotion_text_joins_natural_prompt_and_tags():
    """官方 `emotion_generator_v2.py:310`：自然语言 + `Emotion Tags:` 两段。"""
    out = eg.emotion_text("The character smiles.", "smile, closed_mouth", "smile")
    assert out == "The character smiles.\n\nEmotion Tags: smile, closed_mouth"


def test_emotion_text_falls_back_to_key():
    """没有 natural_prompt 时退回 `The character expresses <key>.`（官方同款）。"""
    assert eg.emotion_text("", "", "smug") == "The character expresses smug."


def test_emotion_prompt_fills_template():
    """QI2 分支**只走模板**，不拼 face_details / 种族 / 服装（官方 `:4142`）。"""
    out = eg.emotion_prompt("The character looks angry.", "angry, scowl", "angry")
    assert out.startswith("Upscale face image.\nMake character's face emotion The character looks angry.")
    assert "Emotion Tags: angry, scowl" in out
    assert out.endswith("keep character's clothes")


def test_emotion_prompt_appends_native_background():
    """`bg_remove.preset == native` 时追加透明底句（官方 `:4145`）。"""
    out = eg.emotion_prompt("x", "", "smile", transparent=True)
    assert out.endswith(eg.NATIVE_BACKGROUND_PROMPT)


def test_negative_prompt_has_water_drops():
    """官方 `emotion_generator_v2.py:669-672`：表情额外负向词逐字。"""
    out = eg.negative_prompt("bad quality")
    assert out.startswith("bad quality")
    assert "(facial droplet)" in out and "(water drops)" in out


# ---------------------------------------------------------------- 人脸检测


class _Tensor:
    """只实现 `detect_face` 用到的那两个方法（`cpu()` / `numpy()`）。"""

    def __init__(self, values):
        self._v = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._v

    def __len__(self):
        return len(self._v)


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYolo:
    def __init__(self, xyxy, conf):
        self._boxes = _Boxes(xyxy, conf) if len(xyxy) else None

    def predict(self, arr, conf=0.5, verbose=False, device="cpu"):
        return [_Result(self._boxes)]


def test_detect_face_grows_by_twice_dilation():
    """官方 `:1533` 里 dilation 被用了**两次**（detect 内部一次 + 手动一次）。"""
    fake = _FakeYolo([[400, 400, 500, 500]], [0.9])
    box = eg.detect_face(Image.new("RGB", (1000, 1000)), dilation=50, model=fake)
    assert box == (300, 300, 600, 600), box


def test_detect_face_clips_to_image():
    """外扩后超出边界的部分要被 clamp 回图内（官方同样 `max(0, …)` / `min(w, …)`）。"""
    fake = _FakeYolo([[10, 10, 60, 60]], [0.9])
    box = eg.detect_face(Image.new("RGB", (120, 120)), dilation=50, model=fake)
    assert box == (0, 0, 120, 120), box


def test_detect_face_returns_none_without_box():
    fake = _FakeYolo([], [])
    assert eg.detect_face(Image.new("RGB", (200, 200)), model=fake) is None


def test_detect_face_on_real_mannequin():
    """真机：人偶图的脸上确实能检出框（官方权重 face_yolov8m）。"""
    src = Path(eg.paths.TEST_OUT) / "pose_editor" / "mannequin_280.png"
    if not src.is_file():
        pytest.skip("缺素材 tests/_out/pose_editor/mannequin_280.png")
    if not os.path.isfile(eg.face_model_path()):
        pytest.skip("缺人脸权重 models/bbox/face_yolov8m.pt")
    with Image.open(src) as opened:
        img = opened.convert("RGBA")
    box = eg.detect_face(img)
    assert box is not None, "人偶脸没检出来"
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= img.width and 0 <= y1 < y2 <= img.height


# ---------------------------------------------------------------- 单张：贴回对齐


def _source(size=(300, 260)) -> Image.Image:
    """造一张「背景 + 中间一块脸」的源图，用来验贴回位置。"""
    img = Image.new("RGBA", size, (40, 160, 60, 255))
    arr = np.asarray(img).copy()
    arr[80:150, 110:190] = (230, 200, 180, 255)     # 脸
    return Image.fromarray(arr)


def test_crop_generate_keeps_everything_outside_the_box():
    src = _source()
    calls = {}

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        calls["size"] = image.size
        calls["prompt"] = prompt
        return Image.new("RGB", image.size, (0, 0, 255))

    # feather=0 → 蒙版全是 1，框内应当**逐点等于**生成色（验坐标没串）
    out = eg.crop_generate(
        src, box=(100, 70, 200, 160), prompt="P", negative="N", seed=7,
        steps=25, cfg=3.0, feather=0, edit=fake_edit)
    image = out["image"]
    assert image.size == src.size, "整图尺寸被改了"
    a = np.asarray(image)
    b = np.asarray(src)
    outside = np.ones((src.height, src.width), dtype=bool)
    outside[70:160, 100:200] = False
    assert np.array_equal(a[outside], b[outside]), "脸框外被改动了"
    # 框中心是生成色（蓝色），说明确实贴上去了
    assert tuple(a[115, 150][:3]) == (0, 0, 255)
    # 推理尺寸对齐 32 且不超过上限
    assert calls["size"][0] % 32 == 0 and calls["size"][1] % 32 == 0
    assert max(calls["size"]) <= eg.MAX_CROP_SIDE
    assert out["detected"] is True
    assert out["face"].size == calls["size"]


def test_crop_generate_without_face_redraws_whole_image():
    """没检出脸就整图重画（官方 `:1626-1632`）。"""
    src = _source()

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        return Image.new("RGB", image.size, (0, 0, 255))

    out = eg.crop_generate(src, box=None, prompt="P", negative="N", seed=1,
                           steps=8, cfg=1.0, feather=0, edit=fake_edit)
    assert out["detected"] is False
    assert out["box"] == (0, 0, src.width, src.height)
    # 整图都换了（四角也是生成色）
    a = np.asarray(out["image"])
    assert tuple(a[5, 5][:3]) == (0, 0, 255)


def test_crop_generate_feather_has_no_hard_seam():
    """羽化带里应该是**渐变**，不是一条 0/255 的硬边（否则贴回去一圈脏线）。

    用接近真实的几何：600² 的图上裁 200² 的框、feather=20。官方默认 feather=50
    是给「脸框外扩 100px 后 300~500px 的块」用的，套在 100px 的小块上整块
    都会被糊成中间值——那是参数不匹配，不是算法错。
    """
    src = Image.new("RGBA", (600, 600), (40, 160, 60, 255))

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        return Image.new("RGB", image.size, (0, 0, 255))

    out = eg.crop_generate(src, box=(200, 200, 400, 400), prompt="P", negative="N",
                           seed=1, steps=8, cfg=1.0, feather=20, edit=fake_edit)
    a = np.asarray(out["image"]).astype(int)
    # 沿框的左边一行横穿过去，蓝色通道应该有一串中间值
    strip = a[300, 190:420, 2]
    mid = int(((strip > 60) & (strip < 200)).sum())
    assert mid >= 8, f"羽化带里几乎没有中间值（{strip.tolist()[:20]}…）"
    # 框正中必须基本等于生成色（羽化不能把中间也吃掉）
    assert a[300, 300, 2] > 240, a[300, 300]


# ---------------------------------------------------------------- 批量：落盘命名


def _make_character(name: str, sprites: int = 2) -> None:
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    target = Path(ch.sprites_dir(name, "Naked", "neutral"))
    target.mkdir(parents=True, exist_ok=True)
    src = _source((320, 300))
    for i in range(sprites):
        src.save(target / f"sprite_pose_{i + 1:04d}.png")


def test_run_batch_writes_official_names_and_index():
    """产物序号跟**源帧**对齐：`sprite_pose_0002` → `sprite_angry_0002`（官方 `:4541`）。"""
    name = "EmoBatch"
    _make_character(name, sprites=3)

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        return Image.new("RGB", image.size, (10, 20, 200))

    res = eg.run_batch(
        None,
        {"character": name, "emotions": ["angry"], "poses": [2],
         "matte": False, "steps": 4},
        edit=fake_edit,
        detect=lambda image: (80, 60, 180, 150),
    )
    assert res["count"] == 1 and not res["failed"], res
    item = res["items"][0]
    sprite = Path(item["sprite"])
    face = Path(item["face"])
    assert sprite.name == "sprite_angry_0002.png", sprite.name
    assert face.name == "face_angry_0002.png", face.name
    assert sprite.parent.name == "angry" and sprite.parent.parent.name == "Naked"
    assert face.parent.parent.parent.name == "Faces", face
    with Image.open(sprite) as im:
        assert im.size == (320, 300), "整图尺寸被改了"
    ch.delete_character(name)


def test_run_batch_reports_progress():
    name = "EmoProgress"
    _make_character(name, sprites=2)

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        return Image.new("RGB", image.size, (1, 2, 3))

    job: dict = {}
    eg.run_batch(None, {"character": name, "emotions": ["smile", "angry"],
                        "matte": False, "steps": 4},
                 job=job, edit=fake_edit, detect=lambda image: None)
    assert job["progress"]["done"] == 4 and job["progress"]["total"] == 4
    ch.delete_character(name)


def test_run_batch_requires_source_sprites():
    """没有源帧要能一句话说清原因（而不是空跑一次）。"""
    name = "EmoEmpty"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    with pytest.raises(eg.EmotionError) as exc:
        eg.run_batch(None, {"character": name, "emotions": ["angry"]}, edit=lambda *a: None)
    assert "源帧" in str(exc.value)
    ch.delete_character(name)


def test_run_batch_survives_single_failure():
    """单张失败不能把整批成果一起丢掉。"""
    name = "EmoPartial"
    _make_character(name, sprites=2)
    seen = {"n": 0}

    def flaky_edit(image, prompt, negative, seed, steps, cfg):
        seen["n"] += 1
        if seen["n"] == 1:
            raise RuntimeError("模拟显存不足")
        return Image.new("RGB", image.size, (9, 9, 9))

    res = eg.run_batch(None, {"character": name, "emotions": ["smile"],
                              "matte": False, "steps": 4},
                       edit=flaky_edit, detect=lambda image: None)
    assert res["count"] == 1, res
    assert len(res["failed"]) == 1
    assert "显存不足" in res["failed"][0]["error"]
    ch.delete_character(name)


def test_resolve_emotions_normalizes_and_dedupes():
    out = eg.resolve_emotions(["Angry", "angry 😠", "my custom face"])
    names = [e["safe_name"] for e in out]
    assert names[0] == "angry"
    assert len(names) == 2, names                 # 前两个是同一条
    assert " " not in names[1] or names[1] == "my custom face"
    with pytest.raises(eg.EmotionError):
        eg.resolve_emotions([])


def test_emotion_catalog_is_the_shipped_official_one():
    """表情库只有一份：`web/public/vnccs/emotions/emotions.json`（前端取同一份）。"""
    items = eg.flat_emotions()
    assert len(items) >= 150, len(items)
    assert all("natural_prompt" in e for e in items)
    assert len({e["category"] for e in items}) >= 9


# ---------------------------------------------------------------- 服务层


@pytest.fixture(scope="module")
def client():
    """真实的 FastAPI app（worker 线程也在跑，所以**别在这里提交真任务**）。"""
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def test_api_detect_lists_frames(client):
    name = "EmoApi"
    _make_character(name, sprites=2)
    r = client.post(f"/characters/{name}/emotions/detect", json={"costume": "Naked"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["items"]) == 2
    # 官方默认值要露出来（前端据此显示 2048 / feather 50）
    assert body["defaults"]["target_size"] == 2048
    assert body["max_crop_side"] == eg.MAX_CROP_SIDE
    ch.delete_character(name)


def test_api_detect_on_empty_costume_returns_zero(client):
    """没有源帧时返回 0 条（不 500）——前端据此提示"先去出图页"。"""
    name = "EmoApiEmpty"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    r = client.post(f"/characters/{name}/emotions/detect", json={"costume": "Naked"})
    assert r.status_code == 200
    assert r.json()["count"] == 0
    ch.delete_character(name)


def test_api_generate_rejects_empty_emotions(client):
    name = "EmoApiNoEmo"
    _make_character(name, sprites=1)
    r = client.post(f"/characters/{name}/emotions/generate", json={"emotions": []})
    assert r.status_code == 400
    assert "表情" in r.json()["detail"]
    ch.delete_character(name)


def test_api_generate_without_sources_is_400(client):
    """没有源帧要给出**能照做的**提示，而不是一句 not found。"""
    name = "EmoApiNoSrc"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    r = client.post(f"/characters/{name}/emotions/generate",
                    json={"emotions": ["angry"]})
    assert r.status_code == 400
    assert "源帧" in r.json()["detail"]
    ch.delete_character(name)


def test_api_generate_enqueues_job(client, monkeypatch):
    """提交后进的是同一个单卡队列，而且 /jobs 能查到 progress。

    `run_batch` 在这里被换掉——这条测的是**接口与队列契约**，不是模型；
    真出图由 `crop_generate` 那几条（假 edit）和人工真机验证覆盖。
    """
    name = "EmoApiJob"
    _make_character(name, sprites=1)

    def fake_run(engine, params, job=None):
        if job is not None:
            job["progress"] = {"done": 1, "total": 1, "current": ""}
        return {"count": 1, "failed": [], "items": [], "total": 1}

    monkeypatch.setattr(eg, "run_batch", fake_run)
    r = client.post(f"/characters/{name}/emotions/generate",
                    json={"emotions": ["angry", "smile"], "matte": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2 and body["emotions"] == ["angry", "smile"]
    job_id = body["job_id"]
    for _ in range(60):
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("status") in ("done", "failed"):
            break
        import time as _t
        _t.sleep(0.05)
    assert job["status"] == "done", job
    assert job["workflow"] == "wemo"
    ch.delete_character(name)
