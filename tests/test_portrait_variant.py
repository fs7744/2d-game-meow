"""立绘工具箱：换装 / 换表情（真换图，产出同阶段的新立绘）。

钉住的契约：
  · 复用已有 `w2` 指令改图链路，不新增模型工作流；
  · 换表情用官方 `EMOTION_PROMPT_TEMPLATE` / `EMOTION_NEGATIVE_EXTRA`；
  · **画风中立**：强制 `preset="none"`，默认不抠图；
  · `publish` 收编成 `kind=portrait` 并带上当前生命阶段；
  · 路径限制在角色 `Portraits/` 下；参数白名单夹取；走同一个单卡队列。

测试把 `gen.img2img` 换成假的，只验「入队的参数」，绝不让 worker 真去加载 Qwen；队列本身保持真实，否则后台 worker 线程会被假队列抛死。
"""
from __future__ import annotations

import os
from urllib.parse import quote

from PIL import Image, ImageDraw

import app as appmod
from workflows import characters as ch
from workflows import emotiongen


def _fake_img2img(eng, **_kw):
    """挡住真模型：worker 会立刻跑完，但 publish 找不到文件只会记一条
    publish_error，不影响任务状态。返回的 file 故意留空。"""
    return {"file": "", "meta": {"fake": True}}


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


def _post(client, name: str, path: str, kind: str,
          params=None, stage: str = ""):
    return client.post(f"/characters/{quote(name)}/portraits/variant",
                       json={"path": path, "kind": kind,
                             "params": params or {}, "stage": stage})


def _queued(monkeypatch):
    """**不要**替换 `_q`：后台 worker 线程里 `_q.get()` 是模块全局查找，
    换成假队列（没有阻塞 get）会把 worker 直接抛死，后面所有依赖真实队列
    的测试都会挂。这里只把生成函数换成假的，队列保持原样。"""
    monkeypatch.setattr(appmod.gen, "img2img", _fake_img2img)
    return None


def test_outfit_enqueues_w2_and_publishes_new_portrait(client, monkeypatch):
    name = _fresh("VariantOutfitProbe")
    src = _seed(name)
    _queued(monkeypatch)

    r = _post(client, name, src, "outfit",
              {"description": "赛博朋克护士服，白大褂 + 护目镜",
               "costume": "Nurse", "steps": 28}, stage="stage_abc")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "outfit" and body["job_id"]
    assert body["before"].startswith("/files/"), body["before"]

    job = appmod._jobs[body["job_id"]]
    assert job["workflow"] == "w2", "复用已有指令改图链路，别新造工作流"
    params = job["params"]
    assert params["init_image"] == os.path.realpath(src)
    assert "赛博朋克护士服" in params["prompt"]
    assert "Keep the same face" in params["prompt"]
    assert params["preset"] == "none", "画风中立：不许挂 sprite 公共契约"
    assert params["matte"] is False, "默认不偷偷抠图"
    assert params["steps"] == 28
    pub = params["publish"]
    assert pub["kind"] == "portrait" and pub["character"] == name
    assert pub["costume"] == "Nurse" and pub["stage"] == "stage_abc"
    assert pub["matte"] is False
    assert os.path.isfile(src), "原图不能被改"


def test_emotion_uses_official_template_and_negative(client, monkeypatch):
    name = _fresh("VariantEmotionProbe")
    src = _seed(name)
    _queued(monkeypatch)

    r = _post(client, name, src, "emotion",
              {"emotion_name": "smile", "costume": "Naked"}, stage="default")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "Make character's face emotion smile" in body["prompt"]
    assert "Change only face" in body["prompt"], "官方模板原文要在"
    assert "Change ONLY the facial expression" in body["prompt"]

    params = appmod._jobs[r.json()["job_id"]]["params"]
    assert "facial droplet" in params["negative"], \
        "官方 EMOTION_NEGATIVE_EXTRA 要接上（哭类表情别顺手画眼泪）"
    assert params["publish"]["emotion"] == "smile"
    assert params["publish"]["stage"] == "default"
    assert params["preset"] == "none"
    # 官方常量确实被引用，而不是复制走样
    assert "Make character's face emotion" in emotiongen.EMOTION_PROMPT_TEMPLATE


def test_rejects_missing_params_bad_kind_and_bad_path(client, monkeypatch):
    name = _fresh("VariantRejectProbe")
    src = _seed(name)
    _queued(monkeypatch)

    assert _post(client, name, src, "outfit").status_code == 400
    assert _post(client, name, src, "emotion").status_code == 400
    assert _post(client, name, src, "wat", {"description": "x"}).status_code == 400
    cfg = ch.config_path(name)
    assert _post(client, name, cfg, "outfit",
                 {"description": "x"}).status_code == 400
    assert _post(client, name, "/tmp/not-mine.png", "emotion",
                 {"emotion_name": "smile"}).status_code == 400


def test_clamps_untrusted_params(client, monkeypatch):
    name = _fresh("VariantClampProbe")
    src = _seed(name)
    _queued(monkeypatch)

    r = _post(client, name, src, "outfit",
              {"description": "x" * 900, "steps": 99999, "seed": -999,
               "cfg": "not-a-number"})
    assert r.status_code == 200, r.text
    params = appmod._jobs[r.json()["job_id"]]["params"]
    assert params["steps"] == 60, "步数要夹住，别让客户端拖死单卡"
    assert params["seed"] == -1
    assert params["cfg"] == 1.0
    assert len(params["prompt"]) < 600, "服装描述要截断"
