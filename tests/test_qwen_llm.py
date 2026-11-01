"""本地 LLM 旁路（workflows/qwen_llm.py，移植自 VNCCS `qwen_vl.py` + Cloner）。

**这一层唯一的硬要求是"别把界面拖死"**：模型没装、fork 装错、推理卡住，
都必须变成一条能读的错误码，而不是 500 或永远转圈。所以这里测的多是
**失败路径**与**解析降级**，真推理只在最后一条里跑（要 GPU，会跳过）。

真模型的输出质量不归单测管——那要用真图看，见 docs 里的验证步骤。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from workflows import qwen_llm as ql


# ---------------------------------------------------------------- 解析降级


def test_parse_plain_json():
    assert ql.parse_wizard_json('{"sex": "female"}') == {"sex": "female"}


def test_parse_json_code_block():
    """模型最爱包一层 ```json（官方 `qwen_vl.py:100-110`）。"""
    out = ql.parse_wizard_json('```json\n{"sex": "male", "age": 20}\n```')
    assert out == {"sex": "male", "age": 20}


def test_parse_bare_code_block():
    out = ql.parse_wizard_json('```\n{"hair": "red hair"}\n```')
    assert out == {"hair": "red hair"}


def test_parse_list_wrapped_object():
    """模型会把单个对象包成数组（官方 `:96-97` 取 `[0]`）。"""
    out = ql.parse_wizard_json('[{"sex": "female"}]')
    assert out == {"sex": "female"}


def test_parse_with_prose_around():
    """前后带解释文字也要能捞出来。"""
    out = ql.parse_wizard_json('Sure! Here is the JSON:\n{"a": 1}\nHope it helps.')
    assert out == {"a": 1}


def test_parse_repairs_trailing_comma():
    """json_repair 兜的是"模型写坏的 JSON"（尾逗号、中文引号、少括号）。"""
    out = ql.parse_wizard_json('{"sex": "female", "hair": "long hair",}')
    assert out == {"sex": "female", "hair": "long hair"}


def test_parse_returns_none_for_garbage():
    assert ql.parse_wizard_json("I cannot read this image.") is None
    assert ql.parse_wizard_json("") is None


# ---------------------------------------------------------------- 收敛


def test_normalize_joins_lists_and_clamps():
    out = ql.normalize_character_wizard_traits({
        "race": ["elf", "human"], "hair": "  red hair ",
        "sex": "Male", "age": 300, "eyes": None,
    })
    assert out["race"] == "elf, human"
    assert out["hair"] == "red hair"
    assert out["sex"] == "male"
    assert out["age"] == 100          # clamp 到 [1,100]
    assert out["eyes"] == ""


def test_normalize_defaults_sex_and_age():
    out = ql.normalize_character_wizard_traits({})
    assert out["sex"] == "female" and out["age"] == 18


def test_filter_traits_whitelist_blocks_injection():
    """白名单外的键必须丢掉——模型多吐个 `config` 就能改到别处去（官方 `:374`）。"""
    out = ql.filter_traits({
        "sex": "female", "hair": "blue hair",
        "config": {"x": 1}, "positive_prompt": "hack", "__class__": "x",
    })
    assert set(out) == {"sex", "hair"}


def test_filter_traits_bools_and_lists():
    out = ql.filter_traits({"aesthetics": ["anime", "flat color"], "nsfw": "TRUE"})
    assert out["aesthetics"] == "anime, flat color"
    assert out["nsfw"] is True


# ---------------------------------------------------------------- 图像预处理


def test_background_rgb_variants():
    assert ql.background_rgb("green") == (0, 255, 0)
    assert ql.background_rgb("blue") == (0, 0, 255)
    assert ql.background_rgb("#102030") == (16, 32, 48)
    assert ql.background_rgb("") == (255, 255, 255)


def test_composite_alpha_flattens_transparent():
    """透明图必须压到背景色上：JPEG 没 alpha，直接转 RGB 会变一坨黑，
    模型就从"黑边"里读出"黑色轮廓"这种根本没画的东西。"""
    img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    img.putpixel((4, 4), (200, 30, 30, 255))
    flat = ql.composite_alpha(img, "white")
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)
    assert flat.getpixel((4, 4)) == (200, 30, 30)


def test_image_data_url_is_jpeg_and_downscaled(tmp_path):
    src = tmp_path / "big.png"
    Image.new("RGB", (1400, 900), (10, 120, 60)).save(src)
    url = ql.image_data_url(str(src))
    assert url.startswith("data:image/jpeg;base64,")
    import base64
    import io

    raw = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as im:
        assert max(im.size) == ql.IMAGE_MAX_SIDE, im.size


def test_image_data_url_missing_file():
    with pytest.raises(ql.LlmError) as exc:
        ql.image_data_url("D:/definitely/not/here.png")
    assert exc.value.code == "IMAGE_MISSING"


# ---------------------------------------------------------------- 失败路径


def test_validate_gguf_rejects_missing(tmp_path):
    with pytest.raises(ql.LlmError) as exc:
        ql.validate_gguf(str(tmp_path / "nope.gguf"))
    assert exc.value.code == "MODEL_MISSING"


def test_validate_gguf_rejects_tiny_file(tmp_path):
    p = tmp_path / "tiny.gguf"
    p.write_bytes(b"GGUF" + b"\0" * 100)
    with pytest.raises(ql.LlmError) as exc:
        ql.validate_gguf(str(p))
    assert exc.value.code == "MODEL_INVALID"


def test_validate_gguf_rejects_wrong_magic(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\0" * (2 * 1024 * 1024))
    with pytest.raises(ql.LlmError) as exc:
        ql.validate_gguf(str(p))
    assert exc.value.code == "MODEL_INVALID"


def test_missing_handler_is_dependency_missing():
    """装错 llama-cpp fork 的症状（官方拒绝降级到 Llava15，`:27-49`）。"""

    class _FakeLlamaChat:
        pass

    class _FakeLlama:
        llama_chat_format = _FakeLlamaChat

    with pytest.raises(ql.LlmError) as exc:
        ql.qwen_vl_chat_handler(_FakeLlama)
    assert exc.value.code == "DEPENDENCY_MISSING"
    assert "Qwen35ChatHandler" in exc.value.message


def test_infer_timeout_returns_timeout_code():
    """卡住的推理必须能超时退出——官方这条路**没有超时**（只有串行锁）。"""
    import threading

    gate = threading.Event()
    try:
        with pytest.raises(ql.LlmError) as exc:
            ql._infer(lambda: gate.wait(30), timeout=0.2)
        assert exc.value.code == "TIMEOUT"
    finally:
        gate.set()
        # 等后台线程把锁放掉，别污染后面的用例
        for _ in range(100):
            if not ql._INFER_LOCK.locked():
                break
            import time as _t
            _t.sleep(0.02)


def test_infer_wraps_exceptions():
    def boom():
        raise ValueError("模型炸了")

    with pytest.raises(ql.LlmError) as exc:
        ql._infer(boom, timeout=5)
    assert exc.value.code == "INFERENCE_ERROR"
    assert "模型炸了" in exc.value.message


# ---------------------------------------------------------------- 真推理


class _FakeLlama:
    """假的 create_chat_completion：返回一段包着代码块、还带思考痕迹的输出。"""

    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.content}}]}


def test_analyze_parses_and_returns_fields(monkeypatch, tmp_path):
    src = tmp_path / "char.png"
    Image.new("RGB", (64, 64), (200, 180, 160)).save(src)
    payload = ('{"sex": "Female", "age": 22, "race": ["elf"], '
               '"hair": "silver hair", "config": {"hack": 1}}')
    fake = _FakeLlama("```json\n" + payload + "\n```")
    monkeypatch.setattr(ql, "load_llm", lambda vision=True: fake)
    out = ql.analyze_character_image(str(src), background_color="white")
    assert out["fields"]["sex"] == "female"
    assert out["fields"]["age"] == 22
    assert out["fields"]["race"] == "elf"
    assert "config" not in out["fields"], "白名单没生效"
    # 采样参数照官方：max_tokens 1024 / temperature 0.2
    call = fake.calls[0]
    assert call["max_tokens"] == 1024 and call["temperature"] == 0.2
    # 送进去的消息：system + user(text + image_url)
    parts = call["messages"][1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_analyze_empty_response_is_invalid(monkeypatch, tmp_path):
    src = tmp_path / "c.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(src)
    monkeypatch.setattr(ql, "load_llm", lambda vision=True: _FakeLlama("   "))
    with pytest.raises(ql.LlmError) as exc:
        ql.analyze_character_image(str(src))
    assert exc.value.code == "INVALID_RESPONSE"


def test_analyze_non_json_keeps_raw(monkeypatch, tmp_path):
    src = tmp_path / "c.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(src)
    monkeypatch.setattr(ql, "load_llm", lambda vision=True: _FakeLlama("I see a cat."))
    with pytest.raises(ql.LlmError) as exc:
        ql.analyze_character_image(str(src))
    assert exc.value.code == "PARSE_ERROR"
    assert "cat" in exc.value.raw, "解析失败要把模型原话带出来，否则没法查"


def test_llm_status_reports_presence():
    st = ql.llm_status()
    assert st["model"].endswith(".gguf")
    assert "present" in st and "dependency" in st


@pytest.mark.skipif(
    not os.path.isfile(ql.model_path()) or os.environ.get("MEOW_SLOW") != "1",
    reason="要真跑 4B 模型（约 25s）：MEOW_SLOW=1 pytest tests/test_qwen_llm.py -m slow")
def test_real_model_reads_traits():
    """真推理一次：人偶图上也要能给出结构合法的属性（不校验内容质量）。"""
    src = Path(ql.paths.TEST_OUT) / "pose_editor" / "mannequin_280.png"
    if not src.is_file():
        pytest.skip("缺素材 mannequin_280.png")
    out = ql.analyze_character_image(str(src), timeout=300)
    assert out["fields"]["sex"] in ("male", "female")
    assert 1 <= out["fields"]["age"] <= 100
    assert isinstance(out["fields"], dict) and out["seconds"] > 0


# ---------------------------------------------------------------- 服务层


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def _character_with_sprite(name: str) -> None:
    from workflows import characters as ch

    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    target = Path(ch.sprites_dir(name, "Naked", "neutral"))
    target.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 64), (210, 190, 170)).save(target / "sprite_pose_0001.png")


def test_api_analyze_unknown_character(client):
    r = client.post("/characters/NoSuchChar/analyze", json={})
    assert r.status_code == 404


def test_api_analyze_without_image_or_sprites(client):
    from workflows import characters as ch

    name = "LlmNoImg"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    r = client.post(f"/characters/{name}/analyze", json={})
    assert r.status_code == 400
    assert "没有图" in r.json()["detail"]
    ch.delete_character(name)


def test_api_analyze_returns_fields(client, monkeypatch):
    """接口只**返回字段**，不写 config——写不写由用户在界面上决定。"""
    from workflows import characters as ch

    name = "LlmOk"
    _character_with_sprite(name)
    monkeypatch.setattr(ql, "analyze_character_image", lambda *a, **k: {
        "fields": {"sex": "female", "age": 21, "hair": "silver hair"},
        "seconds": 1.5, "raw": "{}", "image": "x"})
    r = client.post(f"/characters/{name}/analyze", json={"image": ""})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fields"]["hair"] == "silver hair"
    assert body["image"].startswith("/files/"), body["image"]
    # 没被写进 config
    cfg = ch.load_config(name) or {}
    assert (cfg.get("character_info") or {}).get("hair") != "silver hair"
    ch.delete_character(name)


def test_api_analyze_maps_error_codes(client, monkeypatch):
    """错误码要原样透出来，前端才能按码给不同话术。"""
    from workflows import characters as ch

    name = "LlmErr"
    _character_with_sprite(name)

    def boom(*a, **k):
        raise ql.LlmError("MMPROJ_MISSING", "缺视觉投影器")

    monkeypatch.setattr(ql, "analyze_character_image", boom)
    r = client.post(f"/characters/{name}/analyze", json={})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["error"] == "MMPROJ_MISSING" and "投影器" in detail["message"]

    def slow(*a, **k):
        raise ql.LlmError("TIMEOUT", "推理超时")

    monkeypatch.setattr(ql, "analyze_character_image", slow)
    r = client.post(f"/characters/{name}/analyze", json={})
    assert r.status_code == 504
    ch.delete_character(name)


def test_api_analyze_rejects_path_outside_assets(client, monkeypatch):
    from workflows import characters as ch

    name = "LlmPath"
    _character_with_sprite(name)
    r = client.post(f"/characters/{name}/analyze",
                    json={"image": "C:/Windows/System32/drivers/etc/hosts"})
    assert r.status_code == 400
    ch.delete_character(name)
