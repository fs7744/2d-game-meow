"""项目导入 / 导出 / 模板。

覆盖三件事：
  1. 归档能往返（打包 → 解包 → 节点结构一致）
  2. 素材跟着走（图片、**目录**、通配三种引用形态）
  3. 模板能存、能列、能实例化、能删
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from workflows import paths, projectio


def _png(color=(10, 20, 30), size=(16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def unique_scope():
    """每次测试一个**全新的**项目 scope。

    **为什么要唯一**：导入到已存在的 id 现在会报错（防止两个项目的素材
    混在一起）。用固定 scope 的话，第二次跑测试时上次的残留目录还在 →
    导入直接被拒，测试失败。
    """
    import uuid
    return "p_io_" + uuid.uuid4().hex[:10]


@pytest.fixture()
def sample_asset(tmp_path):
    """一张放在 assets 里的图片（走 paths.ASSETS，测试期间指向 tests/_out）。"""
    d = paths.ASSETS / "_projio_src"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "hero.png"
    p.write_bytes(_png())
    yield p
    try:
        p.unlink()
        d.rmdir()
    except OSError:
        pass


# ---------------- 往返 ----------------


def test_pack_unpack_roundtrip(sample_asset, unique_scope):
    """打包再解开：节点结构保持不变，素材路径指向新位置。"""
    payload = {
        "nodes": [
            {"id": "n1", "kind": "gen", "params": {"prompt": "白猫骑士"},
             "image": {"path": str(sample_asset), "url": "/files/xxx.png"}},
        ],
        "viewport": {"x": 1, "y": 2, "z": 1.5},
    }
    meta = {"name": "我的项目", "created": "2026-01-01T00:00:00"}

    blob = projectio.pack(meta, payload)
    assert blob[:2] == b"PK"                      # 是个 zip

    out = projectio.unpack(blob, unique_scope)
    nodes = out["payload"]["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["params"]["prompt"] == "白猫骑士"
    assert nodes[0]["kind"] == "gen"
    assert out["payload"]["viewport"]["z"] == 1.5
    assert out["meta"]["name"] == "我的项目"
    # 原路径没被改动（导出是只读的）
    assert sample_asset.exists()
    assert nodes[0]["image"]["path"] != str(sample_asset)


def test_unpack_lands_in_new_project_dir(sample_asset, unique_scope):
    """解包后的素材必须落在**新项目目录**下，不能落回原处。"""
    payload = {"nodes": [{"id": "n", "image": {"path": str(sample_asset)}}]}
    out = projectio.unpack(projectio.pack({}, payload), unique_scope)
    new_path = out["payload"]["nodes"][0]["image"]["path"]
    assert unique_scope in new_path.replace("\\", "/")
    assert paths.ASSETS in __import__("pathlib").Path(new_path).parents


def test_unpack_restores_url(sample_asset, unique_scope):
    """图片的 url 要跟着新 path 重算，否则前端拿到的是旧链接（404）。"""
    payload = {"nodes": [{"image": {"path": str(sample_asset),
                                    "url": "/files/old/hero.png"}}]}
    out = projectio.unpack(projectio.pack({}, payload), unique_scope)
    img = out["payload"]["nodes"][0]["image"]
    assert img["url"].startswith("/files/")
    assert unique_scope in img["url"]


# ---------------- 三种引用形态 ----------------


def test_directory_ref_is_preserved(sample_asset, unique_scope):
    """**目录**引用（psDir 指向骨架帧目录）要还原成同构的目录。"""
    src_dir = paths.ASSETS / "_projio_frames"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "frame_000.png").write_bytes(_png((1, 2, 3)))
    (src_dir / "frame_001.png").write_bytes(_png((4, 5, 6)))
    try:
        payload = {"nodes": [{"params": {"psDir": str(src_dir)}}]}
        out = projectio.unpack(projectio.pack({}, payload), unique_scope)
        new_dir = out["payload"]["nodes"][0]["params"]["psDir"]
        from pathlib import Path
        p = Path(new_dir)
        assert p.is_dir(), new_dir
        assert len(list(p.glob("*.png"))) == 2
    finally:
        for f in src_dir.glob("*"):
            f.unlink()
        src_dir.rmdir()


def test_glob_ref_becomes_directory(sample_asset, unique_scope):
    """通配引用（assets/_x_*.png）导入后折成目录——后端接受目录，更稳。"""
    from pathlib import Path
    d = paths.ASSETS / "_projio_glob"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a_01.png").write_bytes(_png())
    (d / "a_02.png").write_bytes(_png())
    try:
        payload = {"nodes": [{"params": {"psDir": str(d / "a_*.png")}}]}
        out = projectio.unpack(projectio.pack({}, payload), unique_scope)
        new_val = out["payload"]["nodes"][0]["params"]["psDir"]
        p = Path(new_val)
        assert p.is_dir(), f"通配应折成目录，实际 {new_val}"
        assert len(list(p.glob("*.png"))) == 2
    finally:
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()


def test_missing_asset_is_skipped_not_fatal(unique_scope):
    """引用的文件不存在时不能整个导出失败——跳过即可。"""
    payload = {"nodes": [{"params": {"maskPath": "D:/绝对不存在的路径/x.png"}}]}
    blob = projectio.pack({}, payload)
    out = projectio.unpack(blob, unique_scope)
    # 没被打包进来 → 占位符还原成空串
    assert out["payload"]["nodes"][0]["params"]["maskPath"] == ""


def test_windows_backslash_relative_path_is_packed(sample_asset, unique_scope):
    """**Windows 相对路径用反斜杠**：`assets\\projects\\x\\a.png`。

    以前只认 `startswith("assets/")`（正斜杠），于是这类路径既不被打包、
    导入后也丢图——而且**不报错**，用户只看到「图没了」。
    """
    rel = str(sample_asset).replace("/", "\\")
    # 造一个「相对 assets 根」的反斜杠路径（前端在 Windows 上就是这种形式）
    rel = rel.split("\\assets\\", 1)[-1] if "\\assets\\" in rel else rel
    assert "\\" in rel, rel
    payload = {"nodes": [{"source": {"path": rel}}]}
    out = projectio.unpack(projectio.pack({}, payload), unique_scope)
    new_path = out["payload"]["nodes"][0]["source"]["path"]
    assert new_path, "反斜杠相对路径没被打包，导入后路径被清空了"
    assert Path(new_path).is_file(), new_path


def test_relative_path_under_assets_is_packed(sample_asset, unique_scope):
    """「相对 assets 根」的路径（`projects/xxx/hero.png`）也要能解析。

    前端存的产物路径就是这种形式（相对 assets 根），不是绝对路径。
    """
    from workflows import paths
    rel = str(sample_asset).replace("\\", "/")
    marker = "/assets/"
    if marker in rel:
        rel = rel.split(marker, 1)[1]
    else:
        rel = str(sample_asset.relative_to(paths.ASSETS)).replace("\\", "/")
    payload = {"nodes": [{"source": {"path": rel}}]}
    out = projectio.unpack(projectio.pack({}, payload), unique_scope)
    new_path = out["payload"]["nodes"][0]["source"]["path"]
    assert new_path and Path(new_path).is_file(), new_path


# ---------------- 模板 ----------------


def test_template_save_list_instantiate_delete(sample_asset, unique_scope):
    """模板全流程：存 → 列 → 实例化 → 删。"""
    payload = {"nodes": [{"id": "n1", "params": {"prompt": "模板里的提示词"},
                          "image": {"path": str(sample_asset)}}]}
    meta = {"name": "像素精灵产线", "created": "2026-01-02T00:00:00"}

    info = projectio.save_template("t_io_" + unique_scope[-8:], meta, payload)
    assert info["bytes"] > 0

    items = projectio.list_templates()
    got = [t for t in items if t["id"] == "t_io_" + unique_scope[-8:]]
    assert got, items
    assert got[0]["name"] == "像素精灵产线"
    assert got[0]["nodes"] == 1

    # 实例化 → 素材复制到新项目目录
    inst = projectio.instantiate_template("t_io_" + unique_scope[-8:], unique_scope)
    assert inst["payload"]["nodes"][0]["params"]["prompt"] == "模板里的提示词"
    new_path = inst["payload"]["nodes"][0]["image"]["path"]
    assert unique_scope in new_path.replace("\\", "/")

    assert projectio.delete_template("t_io_" + unique_scope[-8:]) is True
    assert projectio.delete_template("t_io_" + unique_scope[-8:]) is False


def test_instantiate_missing_template_raises():
    with pytest.raises(projectio.ArchiveError):
        projectio.instantiate_template("t_not_exist", "p_io_not_exist")


# ---------------- API 契约 ----------------


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import app as app_module
    with TestClient(app_module.app) as c:
        yield c


def test_api_export_returns_downloadable_zip(client, sample_asset, unique_scope):
    r = client.post("/projects/export", json={
        "meta": {"name": "导出项目"},
        "payload": {"nodes": [{"id": "n", "image": {"path": str(sample_asset)}}]},
    })
    assert r.status_code == 200, r.text
    assert "attachment" in r.headers.get("content-disposition", "")
    assert r.content[:2] == b"PK"


def test_api_import_creates_new_project(client, sample_asset, unique_scope):
    r = client.post("/projects/export", json={
        "meta": {"name": "待导入"},
        "payload": {"nodes": [{"id": "n", "params": {"prompt": "hi"}}]},
    })
    blob = base64.b64encode(r.content).decode("ascii")

    r2 = client.post("/projects/import", json={
        "scope": unique_scope, "name": "导入后的名字",
        "data": blob,
    })
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["meta"]["name"] == "导入后的名字"
    assert body["payload"]["nodes"][0]["params"]["prompt"] == "hi"


def test_api_import_rejects_garbage(client, unique_scope):
    r = client.post("/projects/import", json={
        "scope": unique_scope, "data": base64.b64encode(b"not a zip at all").decode(),
    })
    assert r.status_code == 400


def test_api_template_flow(client, unique_scope):
    r = client.post("/templates", json={
        "meta": {"name": "API模板"},
        "payload": {"nodes": [{"id": "n", "params": {"prompt": "模板提示词"}}]},
    })
    assert r.status_code == 200, r.text
    tid = r.json()["id"]

    lst = client.get("/templates").json()
    assert any(t["id"] == tid for t in lst["items"])

    inst = client.post(f"/templates/{tid}/instantiate",
                        json={"scope": unique_scope, "data": ""}).json()
    assert inst["payload"]["nodes"][0]["params"]["prompt"] == "模板提示词"

    assert client.delete(f"/templates/{tid}").status_code == 200
    assert client.delete(f"/templates/{tid}").status_code == 404
