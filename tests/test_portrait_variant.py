"""立绘工具箱：换装 / 换表情（跟待处理页同源的专用链路）。

钉住的契约：
  · 换装走 `wpo`：官方 Clothes Core LoRA + 六字段拼出的 `Dress the character:`；
  · 换表情走 `wpe`：官方脸部裁剪局部重绘（只改脸），带 target_size / feather；
  · `publish` 收编成 `kind=portrait` 并带上当前生命阶段；
  · 路径限制在角色 `Portraits/` 下；参数白名单夹取；走同一个单卡队列。

测试把两个生成函数换成假的，只验「入队参数」，绝不让 worker 真加载 Qwen；
**队列本身保持真实**（换成假队列会把后台 worker 线程抛死）。
"""
from __future__ import annotations

import os
from urllib.parse import quote

from PIL import Image, ImageDraw

import app as appmod
from workflows import characters as ch
from workflows import charactergen


def _fresh(name: str) -> str:
    if ch.character_exists(name):
        ch.delete_character(name)
    return ch.create_character(name)["character_info"]["name"]


def _seed(name: str, filename: str = "portrait_seed.png") -> str:
    folder = ch.portraits_dir(name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    im = Image.new("RGB", (64, 64), (0, 255, 0))
    ImageDraw.Draw(im).rectangle((20, 20, 44, 44), fill=(220, 30, 30))
    im.save(path, format="PNG")
    return path


def _fake(*_a, **_kw):
    return {"file": "", "meta": {"fake": True}}


def _stub(monkeypatch):
    """挡住真模型：worker 会立刻跑完，publish 找不到文件只记 publish_error。"""
    monkeypatch.setattr(appmod.gen, "portrait_outfit", _fake)
    monkeypatch.setattr(appmod.gen, "portrait_expression", _fake)


def _post(client, name: str, path: str, kind: str,
          params=None, stage: str = ""):
    return client.post(f"/characters/{quote(name)}/portraits/variant",
                       json={"path": path, "kind": kind,
                             "params": params or {}, "stage": stage})


def test_outfit_uses_clothes_fields_and_lora(client, monkeypatch):
    name = _fresh("VariantOutfitProbe")
    src = _seed(name)
    _stub(monkeypatch)

    r = _post(client, name, src, "outfit", {
        "fields": {"top": "白大褂", "bottom": "黑长裤", "shoes": "短靴",
                   "head": "护目镜", "face": "", "negative_prompt": "no hat"},
        "costume": "Nurse", "steps": 28, "cfg": 3.5, "lora_scale": 0.8,
    }, stage="stage_abc")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "outfit" and body["job_id"]
    assert body["before"].startswith("/files/"), body["before"]
    # 换装提示词在端点就拼好了（前端也能预览）
    assert "Dress the character:" in body["prompt"]
    assert "白大褂" in body["prompt"] and "黑长裤" in body["prompt"]

    job = appmod._jobs[body["job_id"]]
    assert job["workflow"] == "wpo", "换装要走挂 Clothes LoRA 的专用链路"
    params = job["params"]
    assert params["image"] == os.path.realpath(src)
    assert "Dress the character:" in params["prompt"]
    assert "Keep the same face" in params["prompt"]
    # 官方负向词 + 用户「不要出现」
    assert "bad quality" in params["negative"]
    assert "no hat" in params["negative"]
    assert params["lora_scale"] == 0.8
    assert params["steps"] == 28 and params["cfg"] == 3.5
    pub = params["publish"]
    assert pub["kind"] == "portrait" and pub["character"] == name
    assert pub["costume"] == "Nurse" and pub["stage"] == "stage_abc"
    assert os.path.isfile(src), "原图不能被改"


def test_outfit_requires_at_least_one_clothes_field(client, monkeypatch):
    name = _fresh("VariantOutfitEmptyProbe")
    src = _seed(name)
    _stub(monkeypatch)
    r = _post(client, name, src, "outfit", {"fields": {"top": "  ", "bottom": ""}})
    assert r.status_code == 400, r.text


def test_outfit_accepts_legacy_description(client, monkeypatch):
    """旧调用只给一句描述也要能跑（不强制六字段）。"""
    name = _fresh("VariantLegacyProbe")
    src = _seed(name)
    _stub(monkeypatch)
    r = _post(client, name, src, "outfit", {"description": "护士服，白大褂"})
    assert r.status_code == 200, r.text
    job = appmod._jobs[r.json()["job_id"]]
    assert "护士服" in job["params"]["prompt"]


def test_emotion_uses_partial_redraw_pipeline(client, monkeypatch):
    name = _fresh("VariantEmotionProbe")
    src = _seed(name)
    _stub(monkeypatch)

    r = _post(client, name, src, "emotion",
              {"emotion_name": "smile", "costume": "Naked",
               "target_size": 1024}, stage="default")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "emotion"

    job = appmod._jobs[body["job_id"]]
    assert job["workflow"] == "wpe", "换表情要走脸部局部重绘链路"
    params = job["params"]
    assert params["emotion"] == "smile"
    # prompt 交给 portrait_expression 用官方 emotiongen 拼，端点不自己造
    assert "prompt" not in params
    assert params["target_size"] == 1024
    assert params["crop_max_side"] >= 512
    assert "feather" in params
    assert params["publish"]["emotion"] == "smile"
    assert params["publish"]["stage"] == "default"


def test_rejects_missing_params_bad_kind_and_bad_path(client, monkeypatch):
    name = _fresh("VariantRejectProbe")
    src = _seed(name)
    _stub(monkeypatch)

    assert _post(client, name, src, "emotion").status_code == 400
    assert _post(client, name, src, "wat",
                 {"fields": {"top": "x"}}).status_code == 400
    cfg = ch.config_path(name)
    assert _post(client, name, cfg, "outfit",
                 {"fields": {"top": "x"}}).status_code == 400
    assert _post(client, name, "/tmp/not-mine.png", "emotion",
                 {"emotion_name": "smile"}).status_code == 400


def test_clamps_untrusted_params(client, monkeypatch):
    name = _fresh("VariantClampProbe")
    src = _seed(name)
    _stub(monkeypatch)

    r = _post(client, name, src, "outfit",
              {"fields": {"top": "x" * 900}, "steps": 99999, "seed": -999,
               "cfg": "not-a-number", "lora_scale": "bad"})
    assert r.status_code == 200, r.text
    params = appmod._jobs[r.json()["job_id"]]["params"]
    assert params["steps"] == 60, "步数要夹住，别让客户端拖死单卡"
    assert params["seed"] == -1
    assert params["cfg"] == 3.0
    assert params["lora_scale"] == 1.0
    # 单个字段最多 200 字
    assert len(params["prompt"]) < 600


def test_official_field_order_is_used():
    """六字段按官方顺序拼（top / bottom / head / shoes / face），空值丢掉。"""
    desc = charactergen.clothes_description(
        {"top": "A", "bottom": "B", "head": "C", "shoes": "D", "face": ""})
    assert desc.splitlines() == ["A", "B", "C", "D"]


# ---------------- 表情多选 / 批量 ----------------

def test_emotion_batch_is_one_job_with_own_publish(client, monkeypatch):
    """多选表情 → 一个 `wpebatch` 任务，进度总量就是张数。

    批量**不带 publish**：worker 里每张查完立刻自己 `publish_portrait`，
    走 `_run` 那段只会收编最后一张。
    """
    name = _fresh("VariantBatchProbe")
    src = _seed(name)
    _stub(monkeypatch)

    r = _post(client, name, src, "emotion",
              {"emotions": ["smile", "angry", "sad"], "costume": "Naked",
               "target_size": 1024}, stage="stage_x")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 3
    job = appmod._jobs[body["job_id"]]
    assert job["workflow"] == "wpebatch", "多选要走进一个任务里的批量分支"
    assert job["progress"]["total"] == 3, "进度总量必须是张数"
    params = job["params"]
    assert params["emotions"] == ["smile", "angry", "sad"]
    assert "publish" not in params, "批量自己落盘，不能留给 _run 的 publish 段"
    assert params["character"] == name and params["costume"] == "Naked"
    assert params["stage"] == "stage_x"
    assert params["target_size"] == 1024


def test_emotion_single_still_uses_wpe(client, monkeypatch):
    """只选一个表情仍走 `wpe`（不因为新分支把老路径改掉）。"""
    name = _fresh("VariantSingleProbe")
    src = _seed(name)
    _stub(monkeypatch)
    r = _post(client, name, src, "emotion", {"emotions": ["smile"]})
    assert r.status_code == 200, r.text
    job = appmod._jobs[r.json()["job_id"]]
    assert job["workflow"] == "wpe"
    assert job["params"]["emotion"] == "smile"


def test_emotion_batch_dedupes_and_caps(client, monkeypatch):
    name = _fresh("VariantBatchCapProbe")
    src = _seed(name)
    _stub(monkeypatch)
    lots = ["smile", "smile"] + [f"e{i}" for i in range(20)]
    r = _post(client, name, src, "emotion", {"emotions": lots})
    assert r.status_code == 200, r.text
    params = appmod._jobs[r.json()["job_id"]]["params"]
    assert params["emotions"].count("smile") == 1, "重复表情要去重"
    assert len(params["emotions"]) == 12, "单任务上限 12 个，别让一张立绘跑一晚上"


def test_batch_helper_publishes_each_and_survives_one_failure(monkeypatch):
    """批量循环：每张各自收编；**一张失败不连累整批**。"""
    calls = []
    made = []

    def fake_gen(_eng, *, image, emotion, **_kw):
        if emotion == "boom":
            raise RuntimeError("脸检测失败")
        made.append(emotion)
        return {"file": image, "meta": {"emotion": emotion, "seconds": 1.2}}

    def fake_pub(**kw):
        calls.append(kw)
        return {"path": kw["src"], "file": f"portrait_{kw['emotion']}.png",
                "emotion": kw["emotion"]}

    monkeypatch.setattr(appmod.gen, "portrait_expression", fake_gen)
    monkeypatch.setattr(appmod.characters, "publish_portrait", fake_pub)

    job = {}
    out = appmod._portrait_expression_batch(None, {
        "image": "/tmp/x.png",
        "emotions": ["smile", "boom", "angry"],
        "character": "X", "costume": "Naked", "stage": "default",
        "steps": 25, "cfg": 3.0, "seed": -1,
        "target_size": 2048, "crop_max_side": 1024, "feather": 50,
    }, job)

    assert out["total"] == 3
    assert [i["emotion"] for i in out["items"]] == ["smile", "angry"]
    assert out["count"] == 2
    assert [f["emotion"] for f in out["failed"]] == ["boom"]
    assert "脸检测失败" in out["failed"][0]["error"]
    assert [c["emotion"] for c in calls] == ["smile", "angry"], \
        "每张都要自己收编（不用 _run 的 publish 段）"
    assert all(c["stage_id"] == "default" for c in calls)
    # 进度一路报到最后，前端才看得出「第 2/3 张」
    assert job["progress"]["total"] == 3
    assert job["progress"]["done"] == 3
