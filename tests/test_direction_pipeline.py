"""方向帧闭环：抽帧 → 生成 → 图集 → Godot 导出 → 验收台。

这一组钉住的是**同一份方向元数据在各环节的传递**，而不是单个模块：

  ``set.json.direction`` → posegen 按方向独立命名 → atlas.json 的
  ``<组>@<方向>`` 动画 → Godot 多动画 ``.tres`` / ``animations.json``
  → playtest 按方向分组给试玩台选帧。

老资产（没有 direction）在每一环都必须退回原行为，别让新功能把已有
角色目录 / 单动画 Godot 包弄坏。全部用小 PNG + 假出图函数，不碰 GPU。
"""
from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path

from PIL import Image

from workflows import characters as ch
from workflows import directions
from workflows import godotexport
from workflows import paths
from workflows import playtest
from workflows import posegen
from workflows import poseset
from workflows import spriteatlas as sa


def _png(color=(10, 20, 30, 255), size=(16, 24)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, "PNG")
    return buf.getvalue()


def _data_url(color=(10, 20, 30, 255)) -> str:
    return "data:image/png;base64," + base64.b64encode(_png(color)).decode()


def _character(name: str) -> Path:
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    base = Path(ch.sprites_dir(name, "Naked", "neutral"))
    base.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 48), (200, 100, 50)).save(base / "sprite_pose_0001.png")
    return base


def _pose_set(folder: Path, specs: list[tuple[str, int]]) -> list[str]:
    """建一个带 set.json 的假姿势集，返回帧路径列表。"""
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    out = []
    for i, (direction, yaw) in enumerate(specs, 1):
        fn = f"f_{i:03d}.png"
        (folder / fn).write_bytes(_png())
        frames.append({"index": i, "t": i * 0.1, "yaw": yaw,
                       "direction": direction, "file": fn})
        out.append(str(folder / fn))
    (folder / "set.json").write_text(
        json.dumps({"name": folder.name, "frames": frames}, ensure_ascii=False),
        encoding="utf-8")
    return out


# ---------------------------------------------------------------- 共享方向表


def test_direction_table_is_single_source():
    assert directions.DIRECTIONS["topdown-8"] == [
        "down", "down-right", "right", "up-right",
        "up", "up-left", "left", "down-left",
    ]
    assert directions.yaw_of("down-left") == 315
    assert directions.yaw_of("") == 0
    assert directions.validate_direction("SIDEWAYS") == ""
    assert directions.direction_from_name("sprite_down_0001.png") == "down"
    assert directions.direction_from_name("sprite_pose_0001.png") == ""
    assert directions.direction_from_name("Naked/neutral/sprite_up-left_0003.png") == "up-left"
    assert directions.frame_filename("right", 2) == "sprite_right_0002.png"
    assert directions.frame_filename("", 2) == "sprite_pose_0002.png"
    # 验收台与生成用的是同一张表，不能各写一份
    assert playtest.DIRECTIONS is directions.DIRECTIONS
    assert playtest.YAW_MAP is directions.YAW_MAP


# ---------------------------------------------------------------- 生成


def test_posegen_groups_frames_by_direction(tmp_path: Path):
    name = "DirPoseGen"
    _character(name)
    folder = tmp_path / "poses" / "八向"
    poses = _pose_set(folder, [("down", 0), ("down", 0),
                               ("right", 90), ("right", 90)])

    res = posegen.run_batch(
        None, {"character": name, "costume": "Naked", "poses": poses,
               "matte": False},
        pose_fn=lambda *a: Image.new("RGBA", (32, 48), (1, 2, 3, 255)),
        matte_fn=None)

    assert res["directions"] == ["down", "right"]
    assert [Path(p).name for p in res["sprites"]] == [
        "sprite_down_0001.png", "sprite_down_0002.png",
        "sprite_right_0001.png", "sprite_right_0002.png",
    ]
    assert [f["direction"] for f in res["items"]] == [
        "down", "down", "right", "right"]
    assert [f["yaw"] for f in res["items"]] == [0, 0, 90, 90]
    assert len(res["directionFrames"]["down"]) == 2
    assert len(res["directionFrames"]["right"]) == 2
    # set.json 是只读输入，出图不能改它
    assert json.loads((folder / "set.json").read_text(encoding="utf-8"))["frames"][0]["direction"] == "down"
    ch.delete_character(name)


def test_posegen_without_set_json_keeps_legacy_names(tmp_path: Path):
    name = "DirPoseLegacy"
    _character(name)
    folder = tmp_path / "plain"
    folder.mkdir()
    poses = []
    for i in range(1, 3):
        p = folder / f"f_{i:03d}.png"
        p.write_bytes(_png())
        poses.append(str(p))

    assert [f["direction"] for f in posegen.resolve_pose_frames(poses)] == ["", ""]
    res = posegen.run_batch(
        None, {"character": name, "poses": poses, "matte": False},
        pose_fn=lambda *a: Image.new("RGB", (32, 48)), matte_fn=None)
    assert [Path(p).name for p in res["sprites"]] == [
        "sprite_pose_0001.png", "sprite_pose_0002.png"]
    assert res["directions"] == [] and res["directionFrames"] == {}
    ch.delete_character(name)


# ---------------------------------------------------------------- 图集


def test_atlas_exports_one_animation_per_direction():
    name = "DirAtlas"
    base = _character(name)
    for fn in ("sprite_down_0001.png", "sprite_down_0002.png",
               "sprite_right_0001.png", "sprite_right_0002.png"):
        (base / fn).write_bytes(_png())

    result = sa.export_character_atlas(name, frame_rate=10)
    meta = json.loads(Path(result.meta_path).read_text(encoding="utf-8"))
    anims = meta["animations"]

    # 老消费者拿到的主组还在，而且包含全部帧
    assert "Naked/neutral" in anims
    assert len(anims["Naked/neutral"]["frames"]) == 5
    # 每个方向额外一条，带机位与轴心
    assert set(anims) >= {"Naked/neutral@down", "Naked/neutral@right"}
    down = anims["Naked/neutral@down"]
    assert down["direction"] == "down" and down["yaw"] == 0
    assert down["frameRate"] == 10 and down["pivot"] == {"x": 0.5, "y": 1.0}
    assert all("sprite_down_" in k for k in down["frames"])
    assert anims["Naked/neutral@right"]["yaw"] == 90
    assert result.direction_animations == 2
    assert meta["meta"]["directionAnimations"] == 2
    ch.delete_character(name)


def test_atlas_direction_subdirs_do_not_collide():
    """按「表情/<方向>/」分目录时，同名 0001.png 不能互相覆盖。"""
    name = "DirAtlasDirs"
    base = _character(name)
    for key in ("down", "right"):
        d = base / key
        d.mkdir()
        (d / "f_001.png").write_bytes(_png())
        (d / "f_002.png").write_bytes(_png())

    frames = sa.collect_frames(name)
    assert [f.direction for f in frames] == ["", "down", "down", "right", "right"]
    result = sa.export_character_atlas(name)
    meta = json.loads(Path(result.meta_path).read_text(encoding="utf-8"))
    assert len(meta["frames"]) == 5, "方向目录里的同名帧必须在 entries 里各不相同"
    assert any(k.endswith("down/f_001.png") for k in meta["frames"])
    assert result.direction_animations == 2
    ch.delete_character(name)


# ---------------------------------------------------------------- Godot


def test_godot_export_emits_multi_animation_bundle():
    project = "ptest_dirpipe"
    root = paths.PROJECTS_DIR / project
    shutil.rmtree(root, ignore_errors=True)
    try:
        frames = [
            {"index": i, "t": i * 0.1, "yaw": yaw, "direction": key,
             "pose": {}, "png": _data_url((20 * i, 30, 40, 255))}
            for i, (key, yaw) in enumerate(
                [("down", 0), ("down", 0), ("right", 90), ("right", 90)], 1)
        ]
        poseset.save_pose_set(project, "四向", frames)
        poseset.create_render(project, "四向成片", "四向")
        for i in range(1, 5):
            src = root / f"cand_{i}.png"
            src.write_bytes(_png((40 * i, 20, 20, 255)))
            poseset.add_candidate(project, "四向成片", i, src)

        b = godotexport.export_bundle(project, "四向成片", stem="dirwalk",
                                      anim_name="行走", fps=9, loop=True)
        assert b["multiAnimation"] is True
        assert b["animationCount"] == 2
        assert b["frames"] == 4 and b["skipped"] == 0
        assert [a["direction"] for a in b["animations"]] == ["down", "right"]
        assert [a["yaw"] for a in b["animations"]] == [0, 90]
        assert b["checks"]["pathsMatch"] is True, b["checks"]

        folder = root / godotexport.EXPORT_DIR / "dirwalk"
        tres = (folder / "dirwalk.tres").read_text(encoding="utf-8")
        assert "load_steps=5" in tres, tres.splitlines()[0]
        assert tres.count('[ext_resource type="Texture2D"') == 4
        assert '"name": &"行走@down"' in tres
        assert '"name": &"行走@right"' in tres
        assert tres.count('"frames": [') == 2, "两个方向各一条动画"

        manifest = json.loads((folder / "animations.json").read_text(encoding="utf-8"))
        assert manifest["default"] == "行走@down"
        assert [a["direction"] for a in manifest["animations"]] == ["down", "right"]
        assert all(a["frameRate"] == 9 for a in manifest["animations"])
        assert b["manifest"]["name"] == "animations.json"

        scene = (folder / "dirwalk.tscn").read_text(encoding="utf-8")
        assert 'autoplay = "行走@down"' in scene
        readme = (folder / "README.txt").read_text(encoding="utf-8")
        assert "多方向动画" in readme and "行走@right" in readme
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_godot_export_without_directions_stays_single_animation():
    """老成片集没有 direction：必须仍是一字不变的单动画 .tres。"""
    project = "ptest_dirpipe_legacy"
    root = paths.PROJECTS_DIR / project
    shutil.rmtree(root, ignore_errors=True)
    try:
        poseset.save_pose_set(project, "走路", [
            {"t": i * 0.2, "yaw": 0, "pose": {}, "png": _data_url()}
            for i in range(1, 3)
        ])
        poseset.create_render(project, "成片", "走路")
        for i in range(1, 3):
            src = root / f"cand_{i}.png"
            src.write_bytes(_png())
            poseset.add_candidate(project, "成片", i, src)

        b = godotexport.export_bundle(project, "成片", stem="walk", fps=8)
        assert b["multiAnimation"] is False and b["animationCount"] == 1
        assert b["manifest"] is None
        folder = root / godotexport.EXPORT_DIR / "walk"
        assert not (folder / "animations.json").exists()
        assert "load_steps=3" in (folder / "walk.tres").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- 验收台


def test_playtest_groups_direction_frames_and_reads_atlas(tmp_path: Path):
    root = tmp_path / "characters" / "Hero"
    d = root / "Sprites" / "Naked" / "neutral"
    d.mkdir(parents=True)
    for fn in ("sprite_down_0001.png", "sprite_down_0002.png",
               "sprite_right_0001.png", "sprite_right_0002.png"):
        (d / fn).write_bytes(_png())
    atlas = root / "Atlas"
    atlas.mkdir()
    (atlas / "atlas.json").write_text(json.dumps({
        "frames": {},
        "animations": {
            "Naked/neutral": {"frames": [], "frameRate": 11, "loop": True},
            "Naked/neutral@down": {"frames": [], "frameRate": 11,
                                   "loop": True, "direction": "down"},
        },
        "meta": {"canvas": {"w": 32, "h": 48}, "pivot": {"x": 0.5, "y": 1.0}},
    }), encoding="utf-8")

    out = playtest.collect_from(root, "Hero", tmp_path)
    clip = out["clips"][0]
    assert clip["directions"] == ["down", "right"]
    assert clip["defaultDirection"] == "down"
    assert len(clip["frames"]) == 2, "默认只播一个方向，不能把 4 帧串成转圈"
    assert all("sprite_down_" in u for u in clip["frames"])
    assert len(clip["directionFrames"]["right"]) == 2
    assert clip["totalFrames"] == 4 and out["frameCount"] == 4
    assert clip["frameRate"] == 11
    assert out["directions"]["characterDirs"] == ["down", "right"]
    assert out["directions"]["atlasDirs"] == ["down"]
    assert "down" in out["directions"]["present"]
    # 只有两个方向，四方向模式仍要如实报缺
    assert set(out["viewModes"]["topdown-4"]["missing"]) == {"up", "left"}


def test_pose_dirs_finds_project_pose_sets(monkeypatch, tmp_path: Path):
    """/pose_dirs 原来只扫两层，扫不到 assets/projects/<id>/poses/<集>/。"""
    import app

    fake = tmp_path / "assets"
    set_dir = fake / "projects" / "p1" / "poses" / "八向走路"
    set_dir.mkdir(parents=True)
    (set_dir / "f_001.png").write_bytes(_png())
    (set_dir / "f_002.png").write_bytes(_png())
    (set_dir / "set.json").write_text(json.dumps({"frames": [
        {"index": 1, "direction": "down", "yaw": 0, "file": "f_001.png"},
        {"index": 2, "direction": "right", "yaw": 90, "file": "f_002.png"},
    ]}), encoding="utf-8")
    monkeypatch.setattr(app, "ASSETS", fake)

    items = app.pose_dirs()["items"]
    hit = [i for i in items if Path(i["path"]) == set_dir]
    assert len(hit) == 1, items
    assert hit[0]["count"] == 2
    assert hit[0]["directions"] == ["down", "right"]
    assert hit[0]["name"] == "p1/poses/八向走路"
