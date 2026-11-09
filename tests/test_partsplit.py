"""拆部件：See-Through / 爆炸图 → 标准分层 PSD + 分层清单。

钉住的契约：
  · 爆炸图是纯 PIL（不需要模型），6 层坐标必须落在原图画布内；
  · 有 psd_tools（see-through 的 venv）时合成**标准分层 PSD**，层名 ASCII；
  · 清单 `layers` 按**从底到顶**排，`order` = 索引（0 = 最底）；
  · 端点只接受角色 `Portraits/` 下的图，方法白名单，走 wparts 队列；
  · 删除产物要挡 `..`。
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw

import app as appmod
from workflows import characters as ch
from workflows import partsplit
from workflows import paths


def _fresh(name: str) -> str:
    if ch.character_exists(name):
        ch.delete_character(name)
    return ch.create_character(name)["character_info"]["name"]


def _disk(url: str) -> str:
    """`/files/...` → 磁盘路径。**别硬编码 assets/**：测试里 ASSETS 可能是
    临时目录（conftest 会改），硬编码会指到仓库根。"""
    return str(Path(paths.ASSETS) / url[len("/files/"):])


def _seed(name: str, size=(256, 384)) -> str:
    folder = ch.portraits_dir(name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "seed.png")
    w, h = size
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 3 // 8, h // 16, w * 5 // 8, h // 4), fill=(246, 214, 178, 255))
    d.rectangle((w * 11 // 32, h // 4, w * 21 // 32, h * 65 // 100),
                fill=(190, 40, 50, 255))
    d.rectangle((w * 9 // 32, h // 4 + 4, w * 11 // 32, h * 57 // 100),
                fill=(190, 40, 50, 255))
    d.rectangle((w * 21 // 32, h // 4 + 4, w * 23 // 32, h * 57 // 100),
                fill=(190, 40, 50, 255))
    d.rectangle((w * 3 // 8, h * 65 // 100, w * 15 // 32, h * 96 // 100),
                fill=(40, 60, 160, 255))
    d.rectangle((w * 17 // 32, h * 65 // 100, w * 5 // 8, h * 96 // 100),
                fill=(40, 60, 160, 255))
    im.save(path)
    return path


def test_available_shape():
    av = partsplit.available()
    assert av["explode"]["ok"] is True
    assert "ok" in av["seethrough"] and "reason" in av["seethrough"]
    assert "psd" in av


def test_explode_six_layers_and_psd():
    name = _fresh("PartsExplodeProbe")
    src = _seed(name)
    mf = partsplit.split(name, src, "explode", {"overlap": 4})

    assert mf["method"] == "explode"
    assert mf["canvas"] == [256, 384]
    roles = [l["role"] for l in mf["layers"]]
    assert roles == ["leg_l", "leg_r", "arm_l", "arm_r", "torso", "head"], roles
    # 从底到顶：order 就是索引，头在最上
    assert [l["order"] for l in mf["layers"]] == list(range(6))
    for l in mf["layers"]:
        assert l["url"].startswith("/files/"), l["url"]
        assert 0 <= l["x"] <= 256 and 0 <= l["y"] <= 384, l
        assert l["x"] + l["width"] <= 256 and l["y"] + l["height"] <= 384, l
        assert os.path.isfile(_disk(l["url"]))

    # PSD：see-through 的 venv 在就要出标准分层 PSD
    if partsplit.psd_available():
        assert mf["psd"], f"PSD 没生成：{mf['notice']}"
        p = _disk(mf["psd"])
        assert os.path.getsize(p) > 1000
        import subprocess
        r = subprocess.run(
            [str(partsplit.ST_PY), "-c",
             "import sys;from psd_tools import PSDImage;"
             "p=PSDImage.open(sys.argv[1]);"
             "print(len(list(p)), [l.name for l in p])", p],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-400:]
        assert "6" in r.stdout and "head" in r.stdout, r.stdout


def test_list_results_reads_manifest():
    name = _fresh("PartsListProbe")
    src = _seed(name)
    mf = partsplit.split(name, src, "explode", {"overlap": 2})
    items = partsplit.list_results(name)
    assert items, "刚拆完必须能在列表里看到"
    hit = [x for x in items if x["id"] == mf["id"]]
    assert hit and hit[0]["layers"] == 6
    assert hit[0]["methodLabel"].startswith("爆炸图")


def test_delete_guards_outside_path():
    name = _fresh("PartsDeleteProbe")
    src = _seed(name)
    mf = partsplit.split(name, src, "explode", {})
    assert partsplit.delete_result(name, mf["id"]) is True
    try:
        partsplit.delete_result(name, "../Portraits")
    except partsplit.PartsError:
        pass
    else:
        raise AssertionError("越界 rid 必须被挡")
    assert os.path.isdir(os.path.join(ch.character_dir(name), "Portraits"))


def test_endpoint_rejects_bad_method_and_path(client):
    name = _fresh("PartsApiProbe")
    src = _seed(name)
    url = f"/characters/{quote(name)}/parts/split"

    r = client.post(url, json={"path": src, "method": "wat"})
    assert r.status_code == 400
    cfg = ch.config_path(name)
    assert client.post(url, json={"path": cfg, "method": "explode"}).status_code == 400
    assert client.post(url, json={"path": "/tmp/x.png",
                                  "method": "explode"}).status_code == 400

    lst = client.get(f"/characters/{quote(name)}/parts")
    assert lst.status_code == 200
    assert "available" in lst.json() and "items" in lst.json()


def test_endpoint_queues_wparts(client, monkeypatch):
    name = _fresh("PartsQueueProbe")
    src = _seed(name)

    fake = {"id": "fake_1", "method": "explode", "layers": [],
            "psd": None, "canvas": [256, 384]}
    monkeypatch.setattr(appmod.partsplit, "split", lambda *a, **k: dict(fake))

    r = client.post(f"/characters/{quote(name)}/parts/split",
                    json={"path": src, "method": "explode",
                          "params": {"overlap": 5}})
    assert r.status_code == 200, r.text
    body = r.json()
    job = appmod._jobs[body["job_id"]]
    assert job["workflow"] == "wparts"
    assert job["params"]["method"] == "explode"
    assert job["params"]["params"]["overlap"] == 5
    assert job["params"]["src"] == os.path.realpath(src)
