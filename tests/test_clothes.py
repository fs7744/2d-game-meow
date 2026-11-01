"""换装（workflows/clothesgen.py + charactergen 的换装 prompt）。

**要钉住的是"换装不是只存描述"**：它是把五个字段拼成图生图指令、拿裸装底图
重画一张，产物落进 `Sprites/<服装>/neutral/sprite_pose_0001.png`。
prompt 的指令句、字段顺序、背景段三者都对，风格 LoRA 才吃得准。

出图用**假的 `edit`**（返回纯色），不碰 GPU。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PIL import Image

from workflows import charactergen as cg
from workflows import characters as ch
from workflows import clothesgen


# ---------------------------------------------------------------- prompt（逐字）


def test_background_prompt_green_is_verbatim():
    """官方 `clothes_designer.py:132-135`：RGB / hex 都要写进 prompt。

    只说 "green" 时模型会画渐变、墨绿甚至带紫的绿，抠像 `tolerance`
    一放宽就把衣服一起吃掉了。
    """
    assert cg.clothes_background_prompt("Green") == (
        "flat uniform pure green background, exact RGB (0, 255, 0), "
        "hex #00FF00, no gradient")


def test_background_prompt_blue_is_verbatim():
    assert cg.clothes_background_prompt("Blue") == (
        "flat uniform pure blue background, exact RGB (0, 0, 255), "
        "hex #0000FF, no purple, no violet, no gradient")


def test_background_prompt_transparent():
    assert cg.clothes_background_prompt("Transparent") == (
        "Transparent background with alpha channel.")


def test_normalize_background_color_falls_back_to_green():
    """官方 `:354-356`：只认 Green/Blue/Transparent，其余一律 Green。"""
    assert cg.normalize_background_color("green") == "Green"
    assert cg.normalize_background_color("BLUE") == "Blue"
    assert cg.normalize_background_color("transparent") == "Transparent"
    assert cg.normalize_background_color("white") == "Green"
    assert cg.normalize_background_color("") == "Green"


def test_effective_background_transparent_only_for_qi2():
    assert cg.effective_background_color("transparent", "qi2") == "Transparent"
    assert cg.effective_background_color("transparent", "sdxl") == "Green"


def test_construct_prompt_generate_mode():
    """官方 `:427-439`：`Dress the character:` + 五行 + 背景段。"""
    costume = {"top": "red coat", "bottom": "black pants",
               "head": "", "shoes": "leather boots", "face": ""}
    positive, negative = cg.construct_clothes_prompt(costume, "green")
    assert positive == (
        "Dress the character:\nred coat\nblack pants\nleather boots\n"
        "flat uniform pure green background, exact RGB (0, 255, 0), "
        "hex #00FF00, no gradient")
    assert negative == (
        "bad quality, worst quality, (naked, nude, nipple, penis, vagina:2.0)")


def test_construct_prompt_field_order_is_official():
    """顺序是 `top/bottom/head/shoes/face`（官方 `:434` 写死），不是表单顺序。"""
    costume = {"top": "T", "bottom": "B", "head": "H", "shoes": "S", "face": "F"}
    positive, _ = cg.construct_clothes_prompt(costume, "green")
    assert positive.splitlines()[1:6] == ["T", "B", "H", "S", "F"]


def test_construct_prompt_clone_mode():
    """官方 `:424`：clone 模式不读服装字段，负向词要多一串"别画背景纹样"。"""
    positive, negative = cg.construct_clothes_prompt({}, "green", clone=True)
    assert positive == (
        "Dress character to clothes from image 2\n"
        "flat uniform pure green background, exact RGB (0, 255, 0), "
        "hex #00FF00, no gradient")
    assert "background scenery" in negative and "patterned background" in negative


def test_construct_prompt_blue_adds_purple_negative():
    """蓝底要额外压住紫色/渐变（官方 `:447-448`）。"""
    _, negative = cg.construct_clothes_prompt({"top": "x"}, "blue")
    assert "purple background" in negative and "gradient background" in negative


# ---------------------------------------------------------------- 底图选择


def _make_character(name: str, *, naked: int = 0, original: int = 0) -> None:
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    for costume, count in (("Naked", naked), ("Original", original)):
        if not count:
            continue
        d = Path(ch.sprites_dir(name, costume, "neutral"))
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = Image.new("RGB", (64, 96), (200, 180, 160))
            img.save(d / f"sprite_pose_{i + 1:04d}.png")
            time.sleep(0.01)          # 让 mtime 有区别（判定"最新"靠它）


def test_latest_sprite_prefers_naked():
    name = "ClothBase"
    _make_character(name, naked=1, original=1)
    got = clothesgen.base_sprite(name)
    assert got and "Naked" in got.replace("\\", "/")
    ch.delete_character(name)


def test_latest_sprite_falls_back_to_original():
    name = "ClothBaseO"
    _make_character(name, naked=0, original=1)
    got = clothesgen.base_sprite(name)
    assert got and "Original" in got.replace("\\", "/")
    ch.delete_character(name)


def test_latest_sprite_skips_archive_dirs():
    """**V1/V2 归档目录不算"最新"**（官方 `:203-205` 专门跳过）。

    不跳过的话重生成一次，"最新底图"就指到上一版去了，换装会照旧图改。
    """
    name = "ClothArchive"
    _make_character(name, naked=2)
    base = Path(ch.sprites_dir(name, "Naked", "neutral"))
    archive = base / "V1"
    archive.mkdir(parents=True, exist_ok=True)
    newer = archive / "sprite_pose_0099.png"
    Image.new("RGB", (64, 96), (1, 2, 3)).save(newer)
    # 把它设成"最新"的文件
    os.utime(newer, (time.time() + 100, time.time() + 100))
    got = clothesgen.base_sprite(name)
    assert "V1" not in got.replace("\\", "/"), got
    ch.delete_character(name)


def test_base_sprite_none_when_empty():
    name = "ClothEmpty"
    _make_character(name)
    assert clothesgen.base_sprite(name) is None
    ch.delete_character(name)


# ---------------------------------------------------------------- 出图与落盘


def test_run_design_writes_sprite_and_preview():
    name = "ClothRun"
    _make_character(name, naked=1)
    calls = {}

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        calls.update(prompt=prompt, negative=negative, seed=seed, steps=steps, cfg=cfg)
        return Image.new("RGB", image.size, (10, 200, 30))

    res = clothesgen.run_design(
        None,
        {"character": name, "costume": "Nurse", "background_color": "green",
         "matte": False, "steps": 25, "cfg": 3.0},
        edit=fake_edit, matte_fn=None)

    sprite = Path(res["sprite"])
    assert sprite.name == "sprite_pose_0001.png", sprite.name
    assert sprite.parent.name == "neutral" and sprite.parent.parent.name == "Nurse"
    assert Path(res["preview"]).name == "preview_Nurse.png"
    assert Path(res["preview"]).is_file()
    # prompt 是官方换装那条
    assert calls["prompt"].startswith("Dress the character:")
    assert "flat uniform pure green background" in calls["prompt"]
    assert "(naked, nude" in calls["negative"]
    # 采样照官方
    assert calls["steps"] == 25 and calls["cfg"] == 3.0
    # 新服装目录被建出来了
    assert "Nurse" in ch.list_costumes(name)
    ch.delete_character(name)


def test_run_design_requires_base_sprite():
    name = "ClothNoBase"
    _make_character(name)
    with pytest.raises(clothesgen.ClothesError) as exc:
        clothesgen.run_design(None, {"character": name, "costume": "Nurse"},
                              edit=lambda *a: None)
    assert "底图" in str(exc.value)
    ch.delete_character(name)


def test_run_design_uses_costume_fields():
    """五个字段来自 config 里那套服装（官方 `:433-437`）。"""
    name = "ClothFields"
    _make_character(name, naked=1)
    ch.save_costume_info(name, "Knight", {"top": "steel plate", "shoes": "greaves"})
    seen = {}

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        seen["prompt"] = prompt
        return Image.new("RGB", image.size, (0, 0, 0))

    clothesgen.run_design(None, {"character": name, "costume": "Knight",
                                 "background_color": "green", "matte": False},
                          edit=fake_edit)
    assert "steel plate" in seen["prompt"] and "greaves" in seen["prompt"]
    ch.delete_character(name)


def test_run_design_creates_missing_costume():
    """前端「新建服装」只把名字填进输入框，目录还没建——这里要替它建。"""
    name = "ClothAuto"
    _make_character(name, naked=1)
    clothesgen.run_design(None, {"character": name, "costume": "Brand New",
                                 "matte": False},
                          edit=lambda img, *a: Image.new("RGB", img.size, (0, 0, 0)))
    assert "Brand New" in ch.list_costumes(name)
    ch.delete_character(name)


def test_run_design_reports_progress():
    name = "ClothProg"
    _make_character(name, naked=1)
    job: dict = {}
    clothesgen.run_design(None, {"character": name, "costume": "X", "matte": False},
                          job=job,
                          edit=lambda img, *a: Image.new("RGB", img.size, (0, 0, 0)))
    assert job["progress"]["done"] == 1 and job["progress"]["total"] == 1
    ch.delete_character(name)


# ---------------------------------------------------------------- 服务层


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def test_api_design_requires_base(client):
    name = "ClothApiNoBase"
    _make_character(name)
    r = client.post(f"/characters/{name}/clothes/design", json={"costume": "X"})
    assert r.status_code == 400
    assert "底图" in r.json()["detail"]
    ch.delete_character(name)


def test_api_design_enqueues(client, monkeypatch):
    name = "ClothApiJob"
    _make_character(name, naked=1)

    def fake_run(engine, params, job=None):
        if job is not None:
            job["progress"] = {"done": 1, "total": 1, "current": ""}
        return {"costume": params["costume"], "sprite": "x", "seconds": 1}

    monkeypatch.setattr(clothesgen, "run_design", fake_run)
    r = client.post(f"/characters/{name}/clothes/design",
                    json={"costume": "Nurse", "matte": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["costume"] == "Nurse"
    assert body["defaults"]["steps"] == 25
    for _ in range(60):
        job = client.get(f"/jobs/{body['job_id']}").json()
        if job.get("status") in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["workflow"] == "wcloth"
    ch.delete_character(name)


def test_api_clothes_wizard(client, monkeypatch):
    from workflows import qwen_llm as ql

    name = "ClothApiWiz"
    _make_character(name)

    def fake(*a, **k):
        return {"fields": {"top": "white coat", "bottom": "", "shoes": "",
                           "head": "", "face": "goggles"},
                "seconds": 20.0, "raw": "{}", "description": "护士"}

    monkeypatch.setattr(ql, "clothes_wizard", fake)
    r = client.post(f"/characters/{name}/clothes/wizard", json={"description": "护士"})
    assert r.status_code == 200, r.text
    assert r.json()["fields"]["face"] == "goggles"
    # 不落盘
    cfg = ch.load_config(name) or {}
    assert not (cfg.get("costumes") or {}).get("Nurse")
    ch.delete_character(name)
