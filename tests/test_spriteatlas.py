"""精灵图集导出（workflows/spriteatlas.py）。

这一层**不是官方功能**——官方只管把帧出到 `Sprites/<服装>/<表情>/`。
打成图集是"接进引擎"的那一步，也是最容易被写错的一步：

  · 对齐：帧的包围盒不一样，按左上角拼会让角色在动画里上下抖；
  · 留缝：帧贴边时引擎线性过滤会把邻居像素吸进来，边缘一圈杂色；
  · 不拉伸：把帧拉伸到统一尺寸会毁掉像素网格（这一套管线出来的图是像素化的）。

所以这一组测的几乎全是"几何/结构"，出图部分用几十像素的小图，不碰 GPU。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from workflows import characters as ch
from workflows import spriteatlas as sa


def _character(name: str, layouts=((96, 128),)) -> None:
    """建一个角色，按 `layouts` 依次往 Naked/neutral 里塞帧。"""
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    target = Path(ch.sprites_dir(name, "Naked", "neutral"))
    target.mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(layouts, start=1):
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        # 底部涂一块不透明，方便验"底边对齐"
        for y in range(size[1] - 4, size[1]):
            for x in range(size[0]):
                img.putpixel((x, y), (200, 50, 50, 255))
        img.save(target / f"sprite_pose_{i:04d}.png")


def _add_emotion(name: str, costume: str, emotion: str,
                 layouts=((96, 128),)) -> None:
    target = Path(ch.sprites_dir(name, costume, emotion))
    target.mkdir(parents=True, exist_ok=True)
    for i, size in enumerate(layouts, start=1):
        Image.new("RGBA", size, (10, 20, 30, 255)).save(
            target / f"sprite_{emotion}_{i:04d}.png")


# ---------------------------------------------------------------- 收集


def test_collect_frames_groups_by_costume_and_emotion():
    name = "AtlasCollect"
    _character(name, ((96, 128), (96, 128)))
    ch.save_costume_info(name, "Nurse", {})
    _add_emotion(name, "Nurse", "neutral", ((96, 128),))
    _add_emotion(name, "Naked", "smile", ((96, 128), (96, 128)))

    frames = sa.collect_frames(name)
    groups = {(f.costume, f.emotion) for f in frames}
    assert ("Naked", "neutral") in groups
    assert ("Nurse", "neutral") in groups
    assert ("Naked", "smile") in groups
    # 帧号要解析出来（动画排序靠它）
    smile = [f for f in frames if f.emotion == "smile"]
    assert [f.index for f in smile] == [1, 2]
    ch.delete_character(name)


def test_collect_frames_filters():
    name = "AtlasFilter"
    _character(name, ((96, 128),))
    ch.save_costume_info(name, "Nurse", {})
    _add_emotion(name, "Nurse", "neutral", ((96, 128),))

    only = sa.collect_frames(name, costumes=["Nurse"])
    assert {f.costume for f in only} == {"Nurse"}

    by_emo = sa.collect_frames(name, emotions=["neutral"])
    assert {f.emotion for f in by_emo} == {"neutral"}
    ch.delete_character(name)


def test_collect_frames_on_empty_character_is_readable():
    """没有任何帧时要给出**能照做**的提示，而不是一句 not found。"""
    name = "AtlasEmpty"
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    with pytest.raises(sa.AtlasError) as exc:
        sa.collect_frames(name)
    assert "没有可导出" in str(exc.value)
    ch.delete_character(name)


# ---------------------------------------------------------------- 对齐


def test_normalize_layout_aligns_bottom_and_centers():
    """底边对齐 + 水平居中 —— 精灵动画的硬要求。

    脚底是唯一在所有姿势里都该固定住的位置。按左上角拼的话，
    高一点的帧会让角色在动画里"跳"一下。
    """
    frames = [
        sa.Frame("Naked", "neutral", 1, "a.png", (96, 128)),
        sa.Frame("Naked", "neutral", 2, "b.png", (64, 120)),
    ]
    canvas = sa.normalize_layout(frames)
    assert canvas == (96, 128)
    # 第一帧正好铺满
    assert frames[0].offset == (0, 0)
    # 第二帧：水平居中 (96-64)//2 = 16，底边对齐 128-120 = 8
    assert frames[1].offset == (16, 8)


def test_normalize_layout_keeps_pixel_grid_when_sizes_differ():
    """尺寸不同的帧放进统一样布时**不拉伸**（拉伸会毁掉像素网格）。"""
    frames = [sa.Frame("Naked", "neutral", i, f"{i}.png", s)
              for i, s in enumerate([(40, 60), (48, 72)], start=1)]
    sa.normalize_layout(frames)
    assert all(f.size == (40, 60) or f.size == (48, 72) for f in frames)
    assert frames[1].offset == (0, 0)             # 最大的那帧铺满


# ---------------------------------------------------------------- 打包


def test_shelf_pack_wraps_and_keeps_padding():
    sizes = [(10, 10)] * 5
    positions, size = sa._shelf_pack(sizes, padding=2, max_width=24)
    # 10 + 2 + 10 = 22 ≤ 24，第三个放不下 → 换行
    assert positions[0] == (0, 0)
    assert positions[1] == (12, 0)
    assert positions[2] == (0, 12)
    assert size == (22, 34)                       # 三行：0, 12, 24 + 10


def test_shelf_pack_zero_padding_still_works():
    positions, size = sa._shelf_pack([(10, 10), (10, 10)], padding=0, max_width=10)
    assert positions == [(0, 0), (0, 10)]
    assert size == (10, 20)


def test_build_atlas_does_not_resize_frames(tmp_path):
    """拼图集时只**偏移放入**，绝不 resize —— 像素帧被重采样就糊了。

    用 2×3 的小图 + 一个独特色像素：被 resize 过的话那个像素就对不上了。
    """
    paths = []
    for i, color in enumerate([(255, 0, 0, 255), (0, 0, 255, 255)], start=1):
        img = Image.new("RGBA", (8, 12), (0, 0, 0, 0))
        img.putpixel((1, 1), color)
        path = tmp_path / f"f{i}.png"
        img.save(path)
        paths.append(str(path))

    frames = [
        sa.Frame("Naked", "neutral", 1, paths[0], (8, 12)),
        sa.Frame("Naked", "neutral", 2, paths[1], (8, 12)),
    ]
    atlas, positions, canvas = sa.build_atlas(frames, padding=1, max_width=64)
    assert canvas == (8, 12)
    assert atlas.size == (8 + 1 + 8, 12)
    assert positions == [(0, 0), (9, 0)]
    # 原尺寸像素原样落在预期位置（没被重采样）
    assert atlas.getpixel((1, 1)) == (255, 0, 0, 255)
    assert atlas.getpixel((10, 1)) == (0, 0, 255, 255)


# ---------------------------------------------------------------- 落盘


def test_export_writes_texturepacker_json_with_animations():
    name = "AtlasExport"
    _character(name, ((48, 64), (48, 64)))
    result = sa.export_character_atlas(name, padding=2, frame_rate=12)

    assert Path(result.image_path).is_file()
    assert Path(result.meta_path).is_file()
    assert result.frames == 2
    assert result.animations == 1                 # Naked/neutral 一组

    meta = json.loads(Path(result.meta_path).read_text(encoding="utf-8"))
    # TexturePacker Hash 的三个必备块
    assert set(meta) >= {"frames", "meta", "animations"}
    entry = meta["frames"]["Naked/neutral/sprite_pose_0001.png"]
    assert entry["frame"]["w"] == 48 and entry["frame"]["h"] == 64
    assert entry["trimmed"] is False and entry["rotated"] is False
    assert entry["sourceSize"] == {"w": 48, "h": 64}

    # 轴心在底边中点：地面站位就是帧的底边
    assert meta["meta"]["pivot"] == {"x": 0.5, "y": 1.0}
    assert meta["meta"]["character"] == name

    anim = meta["animations"]["Naked/neutral"]
    assert anim["frameRate"] == 12 and anim["loop"] is True
    # 帧序列按帧号排，不是按文件名乱序
    assert anim["frames"][0].endswith("sprite_pose_0001.png")
    assert anim["frames"][1].endswith("sprite_pose_0002.png")
    # 不留临时文件
    assert not list(Path(result.meta_path).parent.glob("*.tmp"))
    ch.delete_character(name)


def test_export_separates_costumes_and_emotions():
    name = "AtlasMulti"
    _character(name, ((32, 48),))
    ch.save_costume_info(name, "Nurse", {})
    _add_emotion(name, "Nurse", "neutral", ((32, 48),))
    _add_emotion(name, "Naked", "smile", ((32, 48),))

    result = sa.export_character_atlas(name)
    meta = json.loads(Path(result.meta_path).read_text(encoding="utf-8"))
    assert set(meta["animations"]) == {
        "Naked/neutral", "Nurse/neutral", "Naked/smile"}
    assert result.animations == 3
    ch.delete_character(name)


def test_export_single_frame_animation_is_noted():
    """单帧动画要留一条说明 —— 引擎侧照 loop 播会"卡住不动"，得知道原因。"""
    name = "AtlasSingle"
    _character(name, ((16, 24),))
    result = sa.export_character_atlas(name)
    assert result.frames == 1
    assert any("只有 1 帧" in n for n in result.notes)
    ch.delete_character(name)


def test_export_to_custom_dir():
    name = "AtlasDir"
    _character(name, ((16, 24),))
    out = Path(ch.character_dir(name)) / "Export"
    result = sa.export_character_atlas(name, out_dir=str(out))
    assert Path(result.image_path).parent == out
    assert Path(result.image_path).is_file()
    ch.delete_character(name)


# ---------------------------------------------------------------- API


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def test_api_atlas_export_and_info(client):
    name = "AtlasApi"
    _character(name, ((24, 32), (24, 32)))
    r = client.post(f"/characters/{name}/atlas", json={"padding": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frames"] == 2 and body["animations"] == 1
    assert body["image"].startswith("/files/"), body["image"]
    assert Path(body["image_path"]).is_file()

    info = client.get(f"/characters/{name}/atlas").json()
    assert info["exists"] is True
    assert info["frames"] == 2
    assert info["size"] == body["size"]
    ch.delete_character(name)


def test_api_atlas_info_before_export(client):
    name = "AtlasApiNoExport"
    _character(name, ((16, 16),))
    info = client.get(f"/characters/{name}/atlas").json()
    assert info["exists"] is False
    ch.delete_character(name)


def test_api_atlas_unknown_character(client):
    r = client.post("/characters/NoSuchAtlasChar/atlas", json={})
    assert r.status_code == 400
    assert "没有可导出" in r.json()["detail"] or "角色" in r.json()["detail"]
