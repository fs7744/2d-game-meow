"""动作模板 registry（workflows/motiontemplates.py）+ 导出消费。

钉住四件事：

1. 内置预设经过同一套 ``normalize`` 校验，方向表与 ``directions.py`` 一致；
2. 成本估算按 ``t/(1-(1-p)^k)``，横板镜像要真的省掉一半机位；
3. 用户模板可增删改，内置模板不可覆盖 / 删除；
4. ``set.json.source.templateId`` 能被 Godot 导出反查到，缺省 fps/loop 从模板来，
   **显式传参仍然优先**（老调用方行为不变）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app
from workflows import godotexport
from workflows import motiontemplates as mt
from workflows import paths
from workflows import poseset


@pytest.fixture
def isolated(monkeypatch, tmp_path: Path):
    """把用户模板文件挪到 tmp_path，避免测试写进真实 library/。"""
    monkeypatch.setattr(mt, "_path", lambda: tmp_path / "motion_templates.json")
    return tmp_path


def _png_bytes() -> bytes:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (16, 24), (10, 20, 30, 255)).save(buf, "PNG")
    return buf.getvalue()


def _data_url() -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()


# ---------------------------------------------------------------- 预设与校验


def test_builtins_are_valid_and_share_the_direction_table():
    from workflows import directions

    items = mt.builtins()
    assert [t["id"] for t in items] == [
        "topdown-walk-4", "topdown-full-8", "side-platformer-2"]
    for t in items:
        assert t["builtin"] is True
        assert t["viewMode"] in directions.DIRECTIONS
        assert t["directions"] == directions.direction_keys(t["viewMode"])
        assert t["yawMap"] == {d: directions.yaw_of(d) for d in t["directions"]}
        assert t["actions"] and t["pivot"] == {"x": 0.5, "y": 1.0}
        # 单次动作绝不循环
        assert all(not a["loop"] for a in t["actions"] if a["oneShot"])
    side = next(t for t in items if t["viewMode"] == "side-2")
    assert side["mirrorFrom"] == {"right": "left"}
    assert side["camera"]["ortho"] is True and side["camera"]["pitch"] == 0


def test_normalize_is_idempotent_and_clamps_dirty_values():
    raw = {"id": "My_Template", "name": "我的模板", "viewMode": "topdown-4",
           "directions": ["down", "SIDEWAYS", "right"],
           "actions": [{"name": "walk", "frames": 99999, "fps": -5, "loop": False}]}
    once = mt.normalize(raw)
    assert once["id"] == "my_template"
    assert once["directions"] == ["down", "right"], "非法方向要丢掉而不是留着"
    assert once["actions"][0]["frames"] == mt.MAX_FRAMES_PER_ACTION
    assert once["actions"][0]["fps"] == 0.1
    # loop=False 推导出 oneShot，且永远不循环
    assert once["actions"][0]["oneShot"] is True
    assert once["actions"][0]["loop"] is False
    assert mt.normalize(once) == once, "归一化必须幂等（estimate/playback 会反复调）"


def test_normalize_rejects_bad_input():
    with pytest.raises(mt.TemplateError):
        mt.normalize({"name": "x", "viewMode": "nope", "actions": [{"name": "a"}]})
    with pytest.raises(mt.TemplateError):
        mt.normalize({"name": "", "viewMode": "side-2", "actions": [{"name": "a"}]})
    with pytest.raises(mt.TemplateError):
        mt.normalize({"name": "x", "viewMode": "side-2", "actions": []})
    with pytest.raises(mt.TemplateError):
        mt.normalize({"id": "中文 id", "name": "x", "viewMode": "side-2",
                      "actions": [{"name": "a"}]})
    with pytest.raises(mt.TemplateError):
        mt.normalize({"name": "x", "viewMode": "side-2",
                      "actions": [{"name": "a"}, {"name": "a"}]})


# ---------------------------------------------------------------- 成本


def test_estimate_uses_batch_formula_and_mirror_saves_half():
    four = mt.get_template("topdown-walk-4")
    eight = mt.get_template("topdown-full-8")
    side = mt.get_template("side-platformer-2")
    assert four and eight and side

    e4 = mt.estimate(four)
    assert e4["directions"] == 4 and e4["renderDirections"] == 4
    frames = sum(a["frames"] for a in four["actions"])
    assert e4["framesPerDirection"] == frames
    assert e4["images"] == frames * 4
    # t/(1-(1-p)^k)：k=1、p=0.75 时是 t/p（要摊上失败重跑）
    assert e4["perFrameSeconds"] == pytest.approx(round(26 / 0.75, 2))
    assert mt.estimate(eight)["images"] == e4["images"] * 2

    es = mt.estimate(side)
    assert es["directions"] == 2 and es["renderDirections"] == 1
    assert es["mirroredDirections"] == ["right"]
    assert es["images"] == sum(a["frames"] for a in side["actions"])

    # 两个候选：每帧期望耗时按 t/(1-(1-p)^2) 摊，不是简单翻倍
    tuned = dict(four, budget={"candidates": 2, "singleFrameSeconds": 26,
                               "passRate": 0.75})
    e2 = mt.estimate(tuned)
    assert e2["candidates"] == 2
    assert e2["perFrameSeconds"] == pytest.approx(round(26 / (1 - 0.25 ** 2), 2))


# ---------------------------------------------------------------- CRUD


def test_user_templates_crud_and_builtins_are_immutable(isolated):
    assert len(mt.list_templates()) == 3
    with pytest.raises(mt.TemplateError):
        mt.save_template({"id": "topdown-walk-4", "name": "改内置",
                          "viewMode": "topdown-4", "actions": [{"name": "a"}]})
    with pytest.raises(mt.TemplateError):
        mt.delete_template("topdown-walk-4")

    doc = {"name": "自定义横板", "viewMode": "side-2",
           "actions": [{"name": "dash", "frames": 5, "fps": 18, "oneShot": True}]}
    created = mt.save_template(doc)
    assert len(created) == 4
    mine = next(t for t in created if t["name"] == "自定义横板")
    assert mine["builtin"] is False and mine["id"].startswith("mt")
    assert mt.playback(mine, "dash")["fps"] == 18
    assert mt.playback(mine, "dash")["loop"] is False

    mine["name"] = "改名了"
    updated = mt.save_template(mine)
    assert len(updated) == 4
    assert any(t["name"] == "改名了" for t in updated)
    assert len(mt.delete_template(mine["id"])) == 3
    # 手改坏一条不该让整个 registry 打不开
    (isolated / "motion_templates.json").write_text('{"items": [{"oops": 1}]}',
                                                    encoding="utf-8")
    assert len(mt.list_templates()) == 3


def test_rest_round_trip(isolated):
    client = TestClient(app.app)
    r = client.get("/motion-templates")
    assert r.status_code == 200 and r.json()["count"] == 3
    assert "topdown-walk-4" in r.json()["estimates"]

    assert client.get("/motion-templates/nope/estimate").status_code == 404
    e = client.get("/motion-templates/side-platformer-2/estimate").json()
    assert e["renderDirections"] == 1

    made = client.post("/motion-templates", json={
        "name": "REST 模板", "viewMode": "topdown-4",
        "actions": [{"name": "idle", "frames": 2, "fps": 6}],
    })
    assert made.status_code == 200 and made.json()["count"] == 4
    tid = next(t["id"] for t in made.json()["items"] if t["name"] == "REST 模板")
    bad = client.put(f"/motion-templates/{tid}", json={"name": "", "viewMode": "x"})
    assert bad.status_code == 400
    assert client.put("/motion-templates/topdown-walk-4",
                      json={"name": "x", "viewMode": "topdown-4",
                            "actions": [{"name": "a"}]}).status_code == 400
    assert client.delete(f"/motion-templates/{tid}").json()["count"] == 3


# ---------------------------------------------------------------- 落到导出


def _project_with_template(template_id: str):
    import shutil

    project = "ptest_motion_tpl"
    root = paths.PROJECTS_DIR / project
    shutil.rmtree(root, ignore_errors=True)
    frames = [
        {"index": i, "t": i * 0.1, "yaw": yaw, "direction": key,
         "pose": {}, "png": _data_url()}
        for i, (key, yaw) in enumerate(
            [("left", 270), ("left", 270), ("right", 90), ("right", 90)], 1)
    ]
    poseset.save_pose_set(project, "横板动作", frames,
                          source={"templateId": template_id, "clip": "walk"})
    poseset.create_render(project, "横板成片", "横板动作")
    for i in range(1, 5):
        src = root / f"cand_{i}.png"
        src.write_bytes(_png_bytes())
        poseset.add_candidate(project, "横板成片", i, src)
    return project, root


def test_template_for_pose_set_and_godot_playback():
    from workflows import directions

    project, root = _project_with_template("side-platformer-2")
    try:
        tpl = mt.template_for_pose_set(project, "横板动作")
        assert tpl and tpl["id"] == "side-platformer-2"
        assert tpl["directions"] == directions.direction_keys("side-2")
        assert mt.template_for_pose_set(project, "不存在的集") is None

        # 不传 fps/loop → 从模板第一个动作（idle, 8fps / 循环）取
        b = godotexport.export_bundle(project, "横板成片", stem="auto")
        assert b["fps"] == 8 and b["loop"] is True
        assert b["playbackSource"] == "template"
        assert b["templateId"] == "side-platformer-2"
        manifest = json.loads(
            (root / godotexport.EXPORT_DIR / "auto" / "animations.json")
            .read_text(encoding="utf-8"))
        assert manifest["templateId"] == "side-platformer-2"
        assert manifest["playbackSource"] == "template"
        assert all(a["frameRate"] == 8 for a in manifest["animations"])

        # 指定动作：walk 是 12 fps
        b2 = godotexport.export_bundle(project, "横板成片", stem="walk",
                                       template_action="walk")
        assert b2["fps"] == 12 and b2["loop"] is True

        # 显式传参永远优先（老调用方行为不变）
        b3 = godotexport.export_bundle(project, "横板成片", stem="explicit",
                                       fps=24, loop=False)
        assert b3["fps"] == 24 and b3["loop"] is False
        assert b3["playbackSource"] == "explicit"
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def test_playtest_reports_linked_template(tmp_path: Path):
    """抽帧留痕的 templateId 要能让角色中心验收台说清来源与成本。"""
    from workflows import playtest

    root = tmp_path / "characters" / "Hero"
    (root / "Sprites" / "Naked" / "neutral").mkdir(parents=True)
    projects = tmp_path / "projects"
    set_dir = projects / "p1" / "poses" / "四向走路"
    set_dir.mkdir(parents=True)
    (set_dir / "set.json").write_text(json.dumps({
        "source": {"templateId": "topdown-walk-4", "clip": "walk"},
        "frames": [{"index": 1, "direction": "down", "yaw": 0,
                    "file": "f_001.png"}],
    }, ensure_ascii=False), encoding="utf-8")

    out = playtest.collect_from(root, "Hero", tmp_path, projects_root=projects)
    tpls = out["directions"]["templates"]
    assert len(tpls) == 1
    t = tpls[0]
    assert t["id"] == "topdown-walk-4" and t["fps"] == 8 and t["loop"] is True
    assert t["directions"] == ["down", "up", "left", "right"]
    assert t["actionCount"] == 4 and t["project"] == "p1"
    assert t["estimate"]["images"] > 0 and t["estimate"]["totalMinutes"] > 0
    assert out["directions"]["poseSets"][0]["templateId"] == "topdown-walk-4"


def test_godot_without_template_falls_back_to_default():
    project, root = _project_with_template("")
    try:
        assert mt.template_for_pose_set(project, "横板动作") is None
        b = godotexport.export_bundle(project, "横板成片", stem="fallback")
        assert b["fps"] == 12 and b["loop"] is True
        assert b["playbackSource"] == "default" and b["templateId"] == ""
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
