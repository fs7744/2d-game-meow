"""角色试玩验收台：只读资产扫描（workflows/playtest.py）。

钉死三件事：不写盘、帧 URL 走 /files/、方向缺口要如实报告而不是假装齐全。
当前角色目录只有「服装 × 表情」两维，所以三视角模式默认都该报缺方向 ——
这正是验收台要暴露给用户的信息。
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import app
from workflows import playtest


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "characters" / "Hero"
    root.mkdir(parents=True)
    return root


def _frames(root: Path, costume: str = "Naked", emotion: str = "neutral", n: int = 4) -> Path:
    d = root / "Sprites" / costume / emotion
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (d / f"sprite_pose_{i:04d}.png").write_bytes(b"png")
    # 整批替换时旧帧进 V1/，枚举必须跳过
    v1 = d / "V1"
    v1.mkdir(exist_ok=True)
    (v1 / "sprite_pose_0001.png").write_bytes(b"png")
    return d


def test_empty_character_reports_all_directions_missing(tmp_path: Path):
    out = playtest.collect_from(_root(tmp_path), "Hero", tmp_path)
    assert out["clipCount"] == 0
    assert out["frameCount"] == 0
    assert out["directions"]["present"] == []
    assert out["viewModes"]["topdown-4"]["ready"] is False
    assert out["viewModes"]["topdown-4"]["missing"] == ["down", "up", "left", "right"]
    assert len(out["viewModes"]["topdown-8"]["missing"]) == 8
    assert out["viewModes"]["side-2"]["missing"] == ["left", "right"]


def test_clips_use_files_urls_and_skip_archives(tmp_path: Path):
    root = _root(tmp_path)
    _frames(root)
    _frames(root, costume="Armor", emotion="angry", n=2)
    out = playtest.collect_from(root, "Hero", tmp_path)
    assert out["clipCount"] == 2
    assert out["frameCount"] == 6
    first = out["clips"][0]
    assert first["id"] == "Naked/neutral"
    assert first["frames"][0].startswith("/files/characters/Hero/Sprites/Naked/neutral/")
    assert all("V1" not in u for u in first["frames"])
    assert [u.rsplit("/", 1)[-1] for u in first["frames"]] == [
        "sprite_pose_0001.png", "sprite_pose_0002.png",
        "sprite_pose_0003.png", "sprite_pose_0004.png",
    ]


def test_defaults_are_marked_as_not_verified(tmp_path: Path):
    root = _root(tmp_path)
    _frames(root)
    out = playtest.collect_from(root, "Hero", tmp_path)
    assert out["atlas"]["source"] == "default"
    assert out["atlas"]["frameRate"] == playtest.DEFAULT_FRAME_RATE
    assert out["atlas"]["loop"] is True
    assert out["atlas"]["pivot"] == {"x": 0.5, "y": 1.0}
    assert out["clips"][0]["loop"] is True


def test_atlas_values_are_read_not_guessed(tmp_path: Path):
    root = _root(tmp_path)
    _frames(root)
    atlas = root / "Atlas"
    atlas.mkdir()
    (atlas / "atlas.json").write_text(json.dumps({
        "frames": {},
        "animations": {"Naked/neutral": {"frames": [], "frameRate": 12, "loop": False}},
        "meta": {"canvas": {"w": 128, "h": 192}, "pivot": {"x": 0.5, "y": 0.9}},
    }), encoding="utf-8")
    out = playtest.collect_from(root, "Hero", tmp_path)
    assert out["atlas"]["source"] == "atlas"
    assert out["atlas"]["frameRate"] == 12.0
    assert out["atlas"]["loop"] is False
    assert out["atlas"]["canvas"] == [128, 192]
    assert out["atlas"]["pivot"] == {"x": 0.5, "y": 0.9}
    assert out["clips"][0]["frameRate"] == 12.0


def test_side_view_mirrors_right_from_left():
    out = playtest._readiness("side-2", ["left"], 1)
    assert out["missing"] == []
    assert out["mirrored"] == ["right"]
    assert out["ready"] is True


def test_eight_directions_ready_only_when_all_present():
    keys = playtest.DIRECTIONS["topdown-8"]
    partial = playtest._readiness("topdown-8", keys[:-1], 1)
    assert partial["missing"] == [keys[-1]]
    assert partial["ready"] is False
    full = playtest._readiness("topdown-8", keys, 1)
    assert full["ready"] is True
    assert full["yawMap"]["down-left"] == 315


def test_route_is_read_only_and_checks_character(monkeypatch):
    client = TestClient(app.app)
    monkeypatch.setattr(app.characters, "character_exists", lambda _n: False)
    assert client.get("/characters/Ghost/playtest").status_code == 404
    monkeypatch.setattr(app.characters, "character_exists", lambda _n: True)
    called = []
    monkeypatch.setattr(app.playtest, "collect",
                        lambda n: called.append(n) or {"character": n, "clipCount": 0})
    r = client.get("/characters/Hero/playtest")
    assert r.status_code == 200
    assert r.json()["clipCount"] == 0
    assert called == ["Hero"]


def test_project_pose_sets_supply_directions(tmp_path: Path):
    """人偶节点按 YAWS 抽出的方向帧，要能让角色中心验收台算成「已补出」。"""
    root = _root(tmp_path)
    _frames(root)
    projects = tmp_path / "projects"
    set_dir = projects / "p1" / "poses" / "八向走路"
    set_dir.mkdir(parents=True)
    (set_dir / "set.json").write_text(json.dumps({
        "name": "八向走路",
        "frames": [
            {"index": 1, "t": 0.1, "yaw": 0, "direction": "down", "file": "f_001.png"},
            {"index": 2, "t": 0.1, "yaw": 90, "direction": "right", "file": "f_002.png"},
            {"index": 3, "t": 0.1, "yaw": 180, "direction": "up", "file": "f_003.png"},
            {"index": 4, "t": 0.1, "yaw": 270, "direction": "left", "file": "f_004.png"},
        ],
    }), encoding="utf-8")
    out = playtest.collect_from(root, "Hero", tmp_path, projects_root=projects)

    assert out["directions"]["present"] == ["down", "left", "right", "up"]
    assert out["directions"]["characterDirs"] == []
    assert out["directions"]["poseSets"][0]["name"] == "八向走路"
    assert out["directions"]["poseSets"][0]["directions"] == ["down", "left", "right", "up"]
    # 四方向齐、且有角色帧 → 就绪；八方向还差 4 个对角
    assert out["viewModes"]["topdown-4"]["ready"] is True
    assert out["viewModes"]["topdown-4"]["missing"] == []
    assert len(out["viewModes"]["topdown-8"]["missing"]) == 4
    # 横板 right 可由 left 镜像，只差 left？这里 left 已有，所以全齐
    assert out["viewModes"]["side-2"]["ready"] is True
