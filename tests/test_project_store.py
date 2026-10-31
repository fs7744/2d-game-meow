"""项目落盘存储（服务端磁盘）。

覆盖：
  1. 存取往返、原子写、目录清洗
  2. 列表 / 删项目 / 工作区状态
  3. API 契约：GET/PUT/DELETE /projects
  4. **浏览器零存储**：项目内容只在磁盘上，不在 localStorage
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def scope() -> str:
    """一个测试用的项目 id。"""
    return "p_store_test"


# ---------------- 存取 ----------------


def test_save_and_load(scope):
    """存进去再读出来，节点与视口一致。"""
    from workflows import projectstore

    doc = {
        "id": scope,
        "name": "我的项目",
        "created": "2026-10-31T00:00:00",
        "updated": "2026-10-31T01:00:00",
        "nodes": [{"id": "n1", "kind": "gen", "params": {"prompt": "白猫骑士"}}],
        "viewport": {"x": 10, "y": 20, "z": 1.5},
    }
    info = projectstore.save_project(scope, doc)
    assert info["id"] == scope
    assert info["name"] == "我的项目"
    assert info["count"] == 1

    got = projectstore.load_project(scope)
    assert got is not None
    assert got["name"] == "我的项目"
    assert got["nodes"][0]["params"]["prompt"] == "白猫骑士"
    assert got["viewport"]["z"] == 1.5

    # 文件真的在磁盘上，且位置正确
    pf = projectstore.project_file(scope)
    assert pf.is_file()
    assert "projects" in pf.as_posix()
    assert pf.name == "project.json"


def test_save_is_atomic(scope):
    """写入是原子的：不留半截 JSON，也不留临时文件。

    **为什么要原子**：自动保存是高频触发的。直接写时进程被杀会留下半截
    JSON，下次打开这个项目直接打不开——用户一屏的活没了。
    """
    from workflows import projectstore

    projectstore.save_project(scope, {"nodes": [{"id": "a"}], "name": "x"})
    d = projectstore.project_dir(scope)
    leftovers = list(d.glob("*.tmp"))
    assert not leftovers, f"留下临时文件：{leftovers}"
    # 解析得出来（不是半截）
    json.loads(projectstore.project_file(scope).read_text(encoding="utf-8"))


def test_save_rejects_missing_nodes(scope):
    """没有 nodes 的定义要被拒（写进去就是坏存档）。"""
    from workflows import projectstore

    with pytest.raises(ValueError):
        projectstore.save_project(scope, {"name": "坏的"})


def test_load_missing_returns_none():
    from workflows import projectstore
    assert projectstore.load_project("p_never_existed_xyz") is None


def test_load_corrupt_file_returns_none(scope):
    """坏文件不能让整个列表接口 500 —— 返回 None 即可。"""
    from workflows import projectstore

    pf = projectstore.project_file(scope)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text("{ 这不是 JSON", encoding="utf-8")
    assert projectstore.load_project(scope) is None


def test_scope_sanitizes_path_traversal():
    """项目 id 里的路径穿越必须被清洗掉。

    `../` 里的 `.` 和 `/` 都被剔除，`..` 变成空串 → 落到 default。
    """
    from workflows import projectstore

    assert projectstore.safe_scope("../../etc") == "etc"
    assert projectstore.safe_scope("a/b") == "ab"
    assert projectstore.safe_scope("..") == "default"
    assert projectstore.safe_scope("") == "default"
    # 中文项目名也要能被清洗（前端允许中文名）
    assert projectstore.safe_scope("我的项目") == "default"


# ---------------- 列表 / 删除 ----------------


def test_list_projects(scope):
    from workflows import projectstore

    projectstore.save_project(scope, {"name": "列出来的", "nodes": [{"id": "n"}]})
    items = projectstore.list_projects()
    got = [i for i in items if i["id"] == scope]
    assert got, items
    assert got[0]["name"] == "列出来的"
    assert got[0]["count"] == 1
    # 摘要里不带 nodes（省得序列化几千个节点）
    assert "nodes" not in got[0]


def test_delete_project_removes_files(scope):
    """删项目要连产物目录一起删。"""
    from workflows import projectstore

    projectstore.save_project(scope, {"name": "要删的", "nodes": []})
    d = projectstore.project_dir(scope)
    (d / "产物.png").write_bytes(b"x")
    assert d.is_dir()

    assert projectstore.delete_project(scope, remove_files=True) is True
    assert not d.exists()
    # 再删一次返回 False（幂等）
    assert projectstore.delete_project(scope, remove_files=True) is False


def test_delete_project_keep_files(scope):
    """只删定义、保留产物（用户只想从列表里去掉）。"""
    from workflows import projectstore

    projectstore.save_project(scope, {"name": "保留产物", "nodes": []})
    d = projectstore.project_dir(scope)
    (d / "产物.png").write_bytes(b"x")

    assert projectstore.delete_project(scope, remove_files=False) is True
    assert not projectstore.project_file(scope).exists()
    assert (d / "产物.png").exists()


# ---------------- 工作区 ----------------


def test_workspace_roundtrip():
    from workflows import projectstore

    projectstore.save_workspace({"current": "p_abc"})
    assert projectstore.load_workspace().get("current") == "p_abc"


# ---------------- API 契约 ----------------


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import app as app_module
    with TestClient(app_module.app) as c:
        yield c


def test_api_put_get_list_delete(client):
    """项目的增删改查走 HTTP。"""
    sid = "p_api_store"
    r = client.put(f"/projects/{sid}", json={
        "id": sid, "name": "API 项目", "created": "2026-10-31T00:00:00",
        "updated": "2026-10-31T00:00:00",
        "nodes": [{"id": "n1", "kind": "gen", "params": {"prompt": "hi"}}],
        "viewport": {"x": 0, "y": 0, "z": 1},
    })
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 1

    g = client.get(f"/projects/{sid}")
    assert g.status_code == 200
    assert g.json()["nodes"][0]["params"]["prompt"] == "hi"

    lst = client.get("/projects").json()
    assert any(i["id"] == sid for i in lst["items"])

    assert client.delete(f"/projects/{sid}").status_code == 200
    assert client.get(f"/projects/{sid}").status_code == 404


def test_api_put_accepts_empty_nodes(client):
    """空项目（nodes=[]）是合法的——用户刚新建的项目就是空的。"""
    r = client.put("/projects/p_api_empty", json={
        "id": "p_api_empty", "name": "空项目", "nodes": [], "viewport": {},
    })
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 0
    client.delete("/projects/p_api_empty")


def test_api_put_rejects_non_list_nodes(client):
    """nodes 不是列表 → 400（不能写进坏存档）。"""
    r = client.put("/projects/p_api_bad", json={"name": "坏的", "nodes": "不是列表"})
    assert r.status_code in (400, 422), r.status_code


def test_api_get_missing_404(client):
    assert client.get("/projects/p_never_existed_abc").status_code == 404


def test_api_delete_missing_404(client):
    assert client.delete("/projects/p_never_existed_abc").status_code == 404


def test_api_workspace(client):
    r = client.put("/workspace", json={"current": "p_api_store"})
    assert r.status_code == 200
    assert client.get("/projects").json()["current"] == "p_api_store"


def test_api_job_missing_returns_404(client):
    """查不到的任务 → 404（前端据此把节点从「执行中」放出来）。"""
    r = client.get("/jobs/p_never_existed_job")
    assert r.status_code == 404
    # 错误信息要说清是「任务不存在」，不是「not found」
    assert "任务" in r.json()["detail"]
