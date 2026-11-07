"""多角色场景 / 相机构图参数的**数据契约**测试。

这一轮给内嵌 PoseStudio 加了多角色同框与整角色变换，最容易悄悄坏掉的不是
3D 渲染（那得真机验），而是「姿势库清单里的构图参数有没有传到前端」「姿势集
帧表里每 tab 的独立位置有没有落盘」「Pydantic 有没有把新字段吞掉」。
这三处一旦丢字段，界面看起来正常、载入姿势却会忽大忽小 —— 静默失败最难查。
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from workflows import paths, poselib, poseset

PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)
PNG_DATA_URL = "data:image/png;base64," + PNG_B64
PROJECT = "ptest_pose_scene"


@pytest.fixture(autouse=True)
def _clean():
    d = paths.PROJECTS_DIR / PROJECT
    if d.exists():
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    yield


# ---------------- 姿势库清单：构图参数与多角色别再丢 ----------------

def test_pose_library_exposes_camera_params():
    items = poselib.list_presets(force=True)
    assert items, "姿势库清单为空"

    with_cp = [i for i in items if i.get("cameraParams")]
    assert len(with_cp) > 100, "绝大多数官方姿势都带构图参数，不该被丢掉"

    sample = next(i for i in items if i["cameraParams"].get("zoom"))
    cp = sample["cameraParams"]
    assert set(cp) <= {"zoom", "offset_x", "offset_y", "yaw_deg", "pitch_deg"}
    assert all(isinstance(v, (int, float)) for v in cp.values()), cp
    assert cp["zoom"] > 0
    # prompt / pose_prompts 在官方库里全是空串，别把它们塞进清单
    assert all("prompt" not in i for i in items)


def test_pose_library_exposes_characters():
    items = poselib.list_presets(force=True)
    with_chars = [i for i in items if i.get("characters")]
    assert with_chars, "官方库里有多角色姿势，清单却一个都没给"

    c = with_chars[0]["characters"][0]
    assert {"id", "name", "transform", "rotation"} <= set(c)
    assert set(c["transform"]) == {"x", "y", "z", "zoom"}
    assert len(c["rotation"]) == 3
    # 最多 8 个角色：清单是给 191 个姿势用的，别把一个超大 JSON 塞进来
    assert all(len(i.get("characters") or []) <= 8 for i in items)


def test_yaw_still_guessed_from_tags():
    """官方 cameraParams.yaw_deg 几乎全是 0，机位还得靠 tags 猜。"""
    items = poselib.list_presets(force=True)
    rear = [i for i in items if "rear view" in " ".join(t.lower() for t in i["tags"])]
    if rear:
        assert all(i["yaw"] == 180 for i in rear)
    assert all(i["yaw"] in (0, 45, 90, 180) for i in items)


# ---------------- 姿势集：每 tab 独立位置要落盘、要读回 ----------------

def test_pose_set_round_trips_scene_extras():
    scene = {
        "characters": [
            {"id": "c1", "name": "角色 1", "color": "#fff", "pose": None,
             "placement": {"x": 0, "y": 0, "z": 0, "zoom": 1, "rotation": [0, 0, 0]}},
            {"id": "c2", "name": "角色 2", "color": "#6cc4ef", "pose": None,
             "placement": {"x": 3, "y": 0, "z": 0, "zoom": 1, "rotation": [0, 0, 0]}},
        ],
        "activeId": "c1", "tabs": ["t1", "t2"], "activeTab": "t1",
        "placements": {"t1": {"c1": {"x": 0, "y": 0, "z": 0},
                              "c2": {"x": 3, "y": 0, "z": 0}}},
        "mode": "move", "space": "world", "linkAll": True,
        "cameraParams": {"zoom": 1.8, "offset_x": 0.5},
    }
    frames = [
        {"t": 0.0, "yaw": 0, "pose": {"bones": {}}, "png": PNG_DATA_URL,
         "scene": scene, "camera": {"posZ": 20}, "characters": scene["characters"]},
        {"t": 0.25, "yaw": 90, "pose": {"bones": {}}, "png": PNG_DATA_URL},
    ]
    poseset.save_pose_set(PROJECT, "双人互动", frames)
    loaded = poseset.load_pose_set(PROJECT, "双人互动")

    assert loaded["frames"][0]["scene"] == scene, "多角色场景没写进 set.json"
    assert loaded["frames"][0]["camera"] == {"posZ": 20}
    assert loaded["frames"][0]["characters"] == scene["characters"]
    assert "scene" not in loaded["frames"][1], "没带的帧不该凭空多出字段"
    # 位置按 tab 独立存，别被序列化压平
    assert loaded["frames"][0]["scene"]["placements"]["t1"]["c2"]["x"] == 3


# ---------------- Pydantic：别把新字段吞掉 ----------------

def test_pose_set_request_keeps_scene_fields():
    from app import PoseSetFrameReq

    req = PoseSetFrameReq(
        t=0.5, yaw=90, pose={"bones": {}}, png=PNG_DATA_URL,
        scene={"characters": [{"id": "c1"}]},
        camera={"zoom": 2}, characters=[{"id": "c1", "name": "A"}],
    )
    dumped = req.model_dump()
    assert dumped["scene"] == {"characters": [{"id": "c1"}]}
    assert dumped["camera"] == {"zoom": 2}
    assert dumped["characters"][0]["name"] == "A"
