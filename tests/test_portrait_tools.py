"""立绘悬浮工具箱后端：扣背景 / 完美像素化 / 删除。

钉住的契约：
  · 两种处理都**生成新的一份**，原图一个字节都不改；
  · 新文件落在原图同目录（自动继承生命阶段 / 服装 / 表情），且不覆盖；
  · 返回 before / after 两个 url，供弹窗做滑动比对；
  · 删除需要显式 confirm，且路径必须在这个角色 Portraits/ 目录内。
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw

from workflows import characters as ch


def _fresh(name: str) -> str:
    if ch.character_exists(name):
        ch.delete_character(name)
    return ch.create_character(name)["character_info"]["name"]


def _seed(name: str, filename: str = "portrait_seed.png",
          size=(64, 64)) -> str:
    """造一张绿幕 + 红色方块的立绘，放在角色 Portraits/ 下。"""
    folder = ch.portraits_dir(name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    w, h = size
    im = Image.new("RGB", (w, h), (0, 255, 0))
    ImageDraw.Draw(im).rectangle((w // 3, h // 3, w * 2 // 3, h * 2 // 3),
                                 fill=(220, 30, 30))
    im.save(path, format="PNG")
    return path


def _post(client, name: str, op: str, path: str, params=None):
    return client.post(f"/characters/{quote(name)}/portraits/process",
                       json={"path": path, "op": op, "params": params or {}})


def test_matte_creates_new_file_and_keeps_original(client):
    name = _fresh("ToolMatteProbe")
    src = _seed(name)
    before_bytes = Path(src).read_bytes()

    r = _post(client, name, "matte", src,
              {"backend": "color", "tol": 32, "fringe": True, "holes": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["before"] != body["after"]
    assert body["name"].endswith("_bg.png")
    assert os.path.isfile(src), "原图不能被删"
    assert Path(src).read_bytes() == before_bytes, "原图一个字节都不能改"
    assert os.path.isfile(body["path"]), "处理结果要真的落盘"
    assert body["meta"]["op"] == "matte" and body["meta"]["backend"] == "color"

    with Image.open(body["path"]) as out:
        assert out.mode == "RGBA"
        assert out.getpixel((0, 0))[3] == 0, "绿幕角应该被抠成透明"
        assert out.getpixel((30, 30))[3] == 255, "主体要保留"


def test_pixel_creates_new_file_and_never_overwrites(client):
    name = _fresh("ToolPixelProbe")
    src = _seed(name)
    before_bytes = Path(src).read_bytes()

    r1 = _post(client, name, "pixel", src,
               {"target": [16, 16], "colors": 8, "grid_snap": True,
                "sample_method": "median"})
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert b1["name"].endswith("_px.png")
    # 格子图是 16×16，但交付的立绘要放大回原图 64×64（否则画廊里小得没法看）
    assert b1["meta"]["native_size"] == [16, 16]
    assert b1["meta"]["output_size"] == [64, 64]
    assert b1["meta"]["output"] == "restore"
    with Image.open(b1["path"]) as out:
        assert out.size == (64, 64)

    r2 = _post(client, name, "pixel", src,
               {"target": [16, 16], "colors": 8, "grid_snap": True})
    assert r2.status_code == 200
    assert r2.json()["path"] != b1["path"], "第二次处理也必须是新文件，不能覆盖"
    assert Path(src).read_bytes() == before_bytes
    assert os.path.isfile(b1["path"]) and os.path.isfile(r2.json()["path"])


def test_two_ops_produce_distinct_files(client):
    name = _fresh("ToolBothProbe")
    src = _seed(name)
    a = _post(client, name, "matte", src, {"backend": "color"}).json()
    b = _post(client, name, "pixel", src, {"target": [16, 16], "colors": 8}).json()
    assert a["path"] != b["path"]
    assert a["name"].endswith("_bg.png") and b["name"].endswith("_px.png")


def test_rejects_unknown_op_and_bad_path(client):
    name = _fresh("ToolRejectProbe")
    src = _seed(name)
    assert _post(client, name, "wat", src).status_code == 400
    # 角色 config.json 在角色目录里，但不在 Portraits/ 下
    cfg = ch.config_path(name)
    assert _post(client, name, "matte", cfg, {"backend": "color"}).status_code == 400
    assert _post(client, name, "matte", "/tmp/not-mine.png",
                 {"backend": "color"}).status_code == 400


def test_delete_requires_confirm_then_removes(client):
    name = _fresh("ToolDeleteProbe")
    src = _seed(name)
    url = f"/characters/{quote(name)}/portraits/delete"

    r = client.post(url, json={"path": src})
    assert r.status_code == 400, "没确认不能直接删"
    assert os.path.isfile(src), "未确认时文件必须还在"

    r = client.post(url, json={"path": src, "confirm": True})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert not os.path.isfile(src)


def test_delete_rejects_outside_path_even_with_confirm(client):
    name = _fresh("ToolDeleteSafeProbe")
    cfg = ch.config_path(name)
    r = client.post(f"/characters/{quote(name)}/portraits/delete",
                    json={"path": cfg, "confirm": True})
    assert r.status_code == 400
    assert os.path.isfile(cfg), "角色配置不能被删"


def test_variant_path_is_unique_and_in_same_folder():
    name = _fresh("ToolPathProbe")
    src = _seed(name)
    a = ch.portrait_variant_path(src, "bg")
    b = ch.portrait_variant_path(src, "bg")
    assert a != b
    assert os.path.dirname(a) == os.path.dirname(src)
    assert not a.endswith("_matte.png"), "否则 list_portraits 会把它当副本跳过"


# ---------------- 像素化尺寸：长边推导 + 放大回原尺寸 ----------------

def test_pixel_long_edge_keeps_aspect_ratio(client):
    """3:4 立绘不能被压成正方形（以前前端写死 target=[N, N]）。"""
    name = _fresh("ToolPixelAspectProbe")
    src = _seed(name, size=(96, 128))
    r = _post(client, name, "pixel", src,
              {"long_edge": 32, "colors": 8, "grid_snap": False})
    assert r.status_code == 200, r.text
    b = r.json()
    m = b["meta"]
    assert m["target"] == [24, 32], m["target"]       # 96/128 = 3/4
    assert m["native_size"] == [24, 32]
    assert m["output_size"] == [96, 128], "交付尺寸要跟原立绘一致"
    with Image.open(b["path"]) as out:
        assert out.size == (96, 128)
        assert abs(out.width / out.height - 96 / 128) < 1e-6


def test_pixel_auto_restores_full_size(client):
    """自动检测出的格子图很小（768×1024 → 48×64），但交付必须是原尺寸。"""
    name = _fresh("ToolPixelAutoProbe")
    src = _seed(name, size=(768, 1024))
    b = _post(client, name, "pixel", src, {"colors": 16}).json()
    m = b["meta"]
    assert m["long_edge"] is None
    assert m["native_size"][0] < 768 and m["native_size"][1] < 1024
    assert m["output_size"] == [768, 1024]
    with Image.open(b["path"]) as out:
        assert out.size == (768, 1024)


def test_pixel_native_option_and_no_new_colors(client):
    """output=native 保留小图；restore 的 NEAREST 放大不能造出新颜色 / 半透明。"""
    name = _fresh("ToolPixelColorProbe")
    src = _seed(name, size=(96, 128))
    params = {"long_edge": 32, "colors": 8, "grid_snap": False}
    small = _post(client, name, "pixel", src,
                  {**params, "output": "native"}).json()
    big = _post(client, name, "pixel", src,
                {**params, "output": "restore"}).json()
    assert small["meta"]["output_size"] == [24, 32]
    assert big["meta"]["output_size"] == [96, 128]

    with Image.open(small["path"]) as a, Image.open(big["path"]) as b:
        ca = set(a.convert("RGBA").getdata())
        cb = set(b.convert("RGBA").getdata())
        assert cb == ca, "NEAREST 放大只复制像素，不该引入新颜色或新 alpha"
        assert all(x[3] in (0, 255) for x in cb), "alpha 硬边不能变半透明"
