"""姿势序列端点（`GET /characters/{name}/poses[/meta]`）。

搬官方 `character_creator_v2.py:1666-1717` + `utils.py:936-981`。
枚举规则本身（6 个候选目录有优先级、第一个非空就停）在 `test_characters.py`
里测；这里测的是**端点契约**：先问数量、按 index 取模、索引谁来 Cities。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_mod
from workflows import characters as ch
from workflows import paths


@pytest.fixture(name="client")
def _client():
    return TestClient(app_mod.app)


@pytest.fixture(name="populated")
def _populated():
    """一个有 3 张 neutral 帧 + 1 张 angry 帧的角色。"""
    name = "PoseSrvr"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    neutral = ch.sprites_dir(name, "Naked", "neutral")
    angry = ch.sprites_dir(name, "Naked", "angry")
    os.makedirs(neutral, exist_ok=True)
    os.makedirs(angry, exist_ok=True)
    for i in (1, 2, 3):
        Path(neutral, f"sprite_pose_000{i}.png").write_bytes(bytes([i]))
    Path(angry, "sprite_angry_0001.png").write_bytes(b"a")
    yield name
    ch.delete_character(name)


def test_meta_returns_count_before_any_image(client, populated):
    """先把数量给出去，让前端**不用一次性拉所有图**。"""
    r = client.get(f"/characters/{populated}/poses/meta")
    assert r.status_code == 200
    assert r.json()["count"] == 3
    assert len(r.json()["files"]) == 3


def test_meta_is_scoped_by_emotion(client, populated):
    """同一个角色的不同表情是**不同序列**，不能混着翻。"""
    assert client.get(f"/characters/{populated}/poses/meta").json()["count"] == 3
    r = client.get(f"/characters/{populated}/poses/meta", params={"emotion": "angry"})
    assert r.json()["count"] == 1


def test_meta_empty_character(client):
    name = "PoseEmpty"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    try:
        r = client.get(f"/characters/{name}/poses/meta")
        assert r.status_code == 200
        assert r.json() == {"count": 0, "files": []}
    finally:
        ch.delete_character(name)


def test_index_wraps_around(client, populated):
    """翻到最后再翻一下回到第一张（`index % count`，官方 :1687）。"""
    last = client.get(f"/characters/{populated}/poses", params={"index": 2})
    assert last.headers["x-vnccs-preview-index"] == "2"
    # 第 3 张内容是 b"\x03"
    assert last.content == b"\x03"

    wrapped = client.get(f"/characters/{populated}/poses", params={"index": 3})
    assert wrapped.headers["x-vnccs-preview-index"] == "0"
    assert wrapped.content == b"\x01"

    negative = client.get(f"/characters/{populated}/poses", params={"index": -1})
    assert negative.headers["x-vnccs-preview-index"] == "2"


def test_count_header_and_no_store(client, populated):
    r = client.get(f"/characters/{populated}/poses", params={"index": 0})
    assert r.headers["x-vnccs-preview-count"] == "3"
    # 重新生成会替换同名文件，缓存旧图是用户"图没变"投诉的常见来源
    assert "no-store" in r.headers["cache-control"]


def test_missing_index_is_random_within_range(client, populated):
    """不给 index = 官方的随机展示（封面式），必须落在合法范围内。"""
    seen = set()
    for _ in range(12):
        r = client.get(f"/characters/{populated}/poses")
        i = int(r.headers["x-vnccs-preview-index"])
        assert 0 <= i < 3
        seen.add(i)
    assert len(seen) > 1, "连续取了十几次都给同一张的话，随机没生效"


def test_no_frames_returns_404(client):
    name = "PoseVoid"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    try:
        assert client.get(f"/characters/{name}/poses").status_code == 404
    finally:
        ch.delete_character(name)


def test_bad_names_are_rejected(client, populated):
    """路径片段是用户输入的，`../../` 必须被挡在 endoskeleton之外。"""
    for bad in ("../etc/passwd", "a/b", ".."):
        assert client.get(f"/characters/{populated}/poses/meta",
                          params={"costume": bad}).status_code == 400


def test_unknown_character_is_empty_not_error(client):
    """角色不在盘上 = 0 帧，不是错误（列表本来就是扫目录扫出来的）。"""
    r = client.get("/characters/NoSuchHero/poses/meta")
    assert r.status_code == 200
    assert r.json() == {"count": 0, "files": []}


def test_official_capitalised_dirs_are_readable(client):
    """官方工作流出的是 `Sprites/Original/Neutral/`（首字母大写）。

    我们要能直接读它的素材，否则"把别人的角色库拷进来"就翻不出任何图。
    """
    name = "PoseCap"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    d = Path(ch.character_dir(name), "Sprites", "Original", "Neutral")
    d.mkdir(parents=True, exist_ok=True)
    Path(d, "sprite_pose_0001.png").write_bytes(b"cap")
    try:
        r = client.get(f"/characters/{name}/poses/meta")
        assert r.json()["count"] == 1
        assert client.get(f"/characters/{name}/poses",
                          params={"index": 0}).content == b"cap"
    finally:
        ch.delete_character(name)


def test_frames_live_under_assets_so_files_endpoint_serves_them(client, populated):
    """角色目录必须在 assets 下——预览图靠 `/files/`，那个端点只服务 assets。

    目录位置一动，这里的 200 立刻变 404，比"UI 全是裂图"好排查。
    """
    meta = client.get(f"/characters/{populated}/poses/meta").json()
    assert meta["files"], "meta 里应该给出 /files/ 路径"
    for url in meta["files"]:
        assert url.startswith("/files/")
    r = client.get(f"/files/characters/{populated}/Sprites/Naked/neutral/sprite_pose_0002.png")
    assert r.status_code == 200, "角色素材走不通 /files/ 说明目录不在 assets 下"
    assert r.content == b"\x02"
