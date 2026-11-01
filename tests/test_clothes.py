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


# ---------------------------------------------------------------- 内衣底图（remove_clothes）
#
# 官方 `character_generator.py:878-900` + `_run_remove_clothes:2282-2340`。
#
# **这是一套具体服装（白色内衣），不是"去掉衣服"**：实现就是一条换装指令加
# Clothes LoRA，跟换护士服走同一条代码路径。区别只有三处：prompt 一句话、
# 采样 20 步 / cfg 1.0（换装是 25 / 3.0）、输入要先压到白底（encode 不吃 alpha）。


def test_underwear_prompt_is_official_verbatim():
    """官方 `:879` 就这一句，**不带背景段**（换装那条才有背景段）。"""
    assert clothesgen.UNDERWEAR_PROMPT == "Dress character: White underwear"
    assert "background" not in clothesgen.UNDERWEAR_PROMPT.lower()


def test_underwear_defaults_are_official():
    """官方 `:880-900`：target_size 1024、背景 White、20 步 / cfg 1.0。"""
    d = clothesgen.UNDERWEAR_DEFAULTS
    assert d["target_size"] == 1024
    assert d["background_color"] == "White"
    assert d["steps"] == 20 and d["cfg"] == 1.0
    assert d["costume"] == "Underwear"


def test_flatten_alpha_puts_color_behind_transparency():
    """透明区域要压到纯色上：不压的话 encode 收到的是预乘黑，
    模型看到"角色周围一圈黑边"，经常照着画一圈黑轮廓出来。"""
    img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    img.putpixel((2, 2), (200, 10, 10, 255))
    flat = clothesgen.flatten_alpha(img, "White")
    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == (255, 255, 255)      # 透明处 → 白底
    assert flat.getpixel((2, 2)) == (200, 10, 10)        # 角色像素不动
    assert clothesgen.flatten_alpha(img, "green").getpixel((0, 0)) == (0, 255, 0)


def test_remove_clothes_writes_underwear_sprite():
    name = "ClothUnderwear"
    _make_character(name, naked=1)
    calls = {}

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        calls.update(prompt=prompt, negative=negative, steps=steps, cfg=cfg,
                     mode=image.mode, corner=image.getpixel((0, 0)))
        return Image.new("RGB", image.size, (250, 250, 250))

    res = clothesgen.remove_clothes(
        None, {"character": name, "matte": False}, edit=fake_edit, matte_fn=None)

    assert res["costume"] == "Underwear"
    sprite = Path(res["sprite"])
    assert sprite.name == "sprite_pose_0001.png", sprite.name
    assert sprite.parent.parent.name == "Underwear"
    # prompt / 采样照官方
    assert calls["prompt"] == "Dress character: White underwear"
    assert calls["negative"] == ""
    assert calls["steps"] == 20 and calls["cfg"] == 1.0
    # 送到 encode 的必须是 RGB（QI2 encode 不吃 alpha）。
    # 这张底图本来就不透明，所以压平后角落仍是原色；透明底会变白底，
    # 那条逻辑在 test_flatten_alpha_puts_color_behind_transparency 里单独验。
    assert calls["mode"] == "RGB"
    assert calls["corner"] == (200, 180, 160)
    ch.delete_character(name)


def test_remove_clothes_custom_costume_name():
    """服装名可以换（比如想保留多套内衣），产物跟着换目录。"""
    name = "ClothUnderwearName"
    _make_character(name, naked=1)

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        return Image.new("RGB", image.size, (1, 1, 1))

    res = clothesgen.remove_clothes(
        None, {"character": name, "costume": "Inner", "matte": False},
        edit=fake_edit, matte_fn=None)
    assert res["costume"] == "Inner"
    assert Path(res["sprite"]).parent.parent.name == "Inner"
    ch.delete_character(name)


def test_remove_clothes_requires_base_sprite():
    name = "ClothUnderwearNoBase"
    _make_character(name)
    with pytest.raises(clothesgen.ClothesError) as exc:
        clothesgen.remove_clothes(None, {"character": name}, edit=lambda *a: None)
    assert "底图" in str(exc.value)
    ch.delete_character(name)


def test_remove_clothes_reports_progress():
    name = "ClothUnderwearProg"
    _make_character(name, naked=1)
    job = {}
    seen = []

    def fake_edit(image, prompt, negative, seed, steps, cfg):
        seen.append(dict(job.get("progress") or {}))
        return Image.new("RGB", image.size, (9, 9, 9))

    clothesgen.remove_clothes(None, {"character": name, "matte": False},
                              job=job, edit=fake_edit, matte_fn=None)
    assert seen and seen[0]["total"] == 1
    assert job["progress"]["done"] == 1
    ch.delete_character(name)


def test_api_underwear_enqueues(client, monkeypatch):
    """接口只负责入队 + 回显官方 prompt（真出图在 worker 里）。

    **必须把 `remove_clothes` 换掉**：worker 线程是真跑的，不换的话这条测试会
    去加载 Qwen 管线、占住单卡队列好几分钟，后面的用例全被拖垮（表现为
    `test_api` 里几条互不相干的断言莫名失败 —— 实测踩过）。
    """
    name = "ClothUnderwearApi"
    _make_character(name, naked=1)

    def fake_remove(engine, params, job=None, **kwargs):
        if job is not None:
            job["progress"] = {"done": 1, "total": 1, "current": ""}
        return {"character": params.get("character"), "costume": "Underwear",
                "sprite": "x.png", "preview": "p.png"}

    monkeypatch.setattr(clothesgen, "remove_clothes", fake_remove)
    r = client.post(f"/characters/{name}/clothes/underwear", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["costume"] == "Underwear"
    assert body["prompt"] == "Dress character: White underwear"
    assert body["defaults"]["steps"] == 20

    # 等任务收尾，别把它挂给下一个用例
    job_id = body["job_id"]
    for _ in range(80):
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("status") in ("done", "failed"):
            break
        import time as _t
        _t.sleep(0.05)
    assert job["status"] == "done", job
    ch.delete_character(name)


def test_api_underwear_without_base_is_400(client):
    name = "ClothUnderwearApiNoBase"
    _make_character(name)
    r = client.post(f"/characters/{name}/clothes/underwear", json={})
    assert r.status_code == 400
    assert "底图" in r.json()["detail"]
    ch.delete_character(name)
