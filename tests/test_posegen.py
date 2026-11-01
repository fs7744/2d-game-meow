"""多姿势批量生成（workflows/posegen.py + generate.qwen_pose_image）。

**要钉住三件事**（都是官方 `character_generator.py:2108-2223` 里最容易做错的）：

1. 指令的图序（`image 1` = 姿势、`image 2` = 角色）与拼接方式（同行接 prompt）；
2. 整批**同一个种子**（每张随机会让衣服/发型飘）；
3. 发布是**整批替换**（旧图进 V1），不是逐张追加。

出图用假的 `pose_fn`，不碰 GPU。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PIL import Image

from workflows import characters as ch
from workflows import generate as gen
from workflows import posegen


# ---------------------------------------------------------------- 指令（逐字）


def test_pose_instruction_is_verbatim():
    """官方 `:2208-2213`：`Replace the pose of <image 2> ... {prompt}`。"""
    assert gen.POSE_INSTRUCTION == (
        "Replace the pose of <image 2> with the pose of <image 1>. "
        "Keep the character of <image 2>. ")


def test_pose_defaults_are_official():
    """官方 `pose_sampler`（`:825-833`）：20 步 / cfg 1.0。"""
    assert posegen.POSE_DEFAULTS["steps"] == 20
    assert posegen.POSE_DEFAULTS["cfg"] == 1.0


def test_prompt_for_repeats_last_when_short():
    """官方 `_pose_prompts:1240-1243`：列表短了就重复最后一条，不报错。"""
    assert posegen._prompt_for(0, ["a", "b"], 4) == "a"
    assert posegen._prompt_for(1, ["a", "b"], 4) == "b"
    assert posegen._prompt_for(3, ["a", "b"], 4) == "b"
    assert posegen._prompt_for(2, "same", 4) == "same"
    assert posegen._prompt_for(0, "", 1) == ""


# ---------------------------------------------------------------- 姿势来源


def test_resolve_pose_paths_accepts_files_url(tmp_path, monkeypatch):
    from workflows import paths
    monkeypatch.setattr(paths, "ASSETS", tmp_path)
    img = tmp_path / "p1.png"
    Image.new("RGB", (32, 32)).save(img)
    got = posegen._resolve_pose_paths(["/files/p1.png"])
    assert got == [str(img)]


def test_resolve_pose_paths_expands_dir(tmp_path, monkeypatch):
    from workflows import paths
    monkeypatch.setattr(paths, "ASSETS", tmp_path)
    d = tmp_path / "poses"
    d.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16)).save(d / f"f{i}.png")
    (d / "notes.txt").write_text("x")
    got = posegen._resolve_pose_paths([], str(d))
    assert len(got) == 3 and all(p.endswith(".png") for p in got)


def test_resolve_pose_paths_dedupes(tmp_path, monkeypatch):
    from workflows import paths
    monkeypatch.setattr(paths, "ASSETS", tmp_path)
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    got = posegen._resolve_pose_paths([str(img), str(img)])
    assert len(got) == 1


# ---------------------------------------------------------------- 批量


def _character(name: str, *, base: int = 1) -> None:
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    if base:
        d = Path(ch.sprites_dir(name, "Naked", "neutral"))
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 96), (200, 180, 160)).save(d / "sprite_pose_0001.png")


def _poses(tmp_path: Path, count: int) -> list:
    out = []
    for i in range(count):
        p = tmp_path / f"pose_{i}.png"
        Image.new("RGB", (48, 96), (255, 255, 255)).save(p)
        out.append(str(p))
    return out


def test_run_batch_publishes_all_and_replaces(tmp_path):
    name = "PoseBatch"
    _character(name)
    poses = _poses(tmp_path, 3)

    def fake_pose(pose_img, char_img, prompt):
        return Image.new("RGBA", (64, 96), (10, 20, 30, 255))

    res = posegen.run_batch(None, {"character": name, "costume": "Naked",
                                   "poses": poses, "matte": False},
                            pose_fn=fake_pose, matte_fn=None)
    assert res["count"] == 3
    names = [Path(p).name for p in res["sprites"]]
    assert names == ["sprite_pose_0001.png", "sprite_pose_0002.png",
                     "sprite_pose_0003.png"]
    target = Path(ch.sprites_dir(name, "Naked", "neutral"))
    # 原来那张底图被推进 V1（整批替换，不是追加）
    assert (target / "V1" / "sprite_pose_0001.png").is_file()
    assert (target / "sprite_pose_0003.png").is_file()
    ch.delete_character(name)


def test_run_batch_uses_one_seed(tmp_path):
    """整批共用一个种子（官方 `inherit_pipe=True`）。"""
    name = "PoseSeed"
    _character(name)
    seen = []

    def fake_pose(pose_img, char_img, prompt):
        seen.append(prompt)
        return Image.new("RGB", (32, 48), (0, 0, 0))

    res = posegen.run_batch(None, {"character": name, "poses": _poses(tmp_path, 2),
                                   "prompts": "站立", "matte": False},
                            pose_fn=fake_pose)
    assert len(set(seen)) == 1 and seen[0] == "站立"
    assert isinstance(res["seed"], int) and res["seed"] >= 0
    ch.delete_character(name)


def test_run_batch_reports_progress(tmp_path):
    name = "PoseProgress"
    _character(name)
    job: dict = {}
    posegen.run_batch(None, {"character": name, "poses": _poses(tmp_path, 2),
                             "matte": False}, job=job,
                      pose_fn=lambda *a: Image.new("RGB", (32, 48)))
    assert job["progress"]["done"] == 2 and job["progress"]["total"] == 2
    ch.delete_character(name)


def test_run_batch_survives_single_failure(tmp_path):
    """一张失败不能把整批丢掉（官方的循环里也是逐张 try）。"""
    name = "PosePartial"
    _character(name)
    calls = {"n": 0}

    def flaky(pose_img, char_img, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("显存不足")
        return Image.new("RGB", (32, 48), (1, 2, 3))

    res = posegen.run_batch(None, {"character": name, "poses": _poses(tmp_path, 2),
                                   "matte": False}, pose_fn=flaky)
    assert res["count"] == 1 and len(res["failed"]) == 1
    assert "显存不足" in res["failed"][0]["error"]
    ch.delete_character(name)


def test_run_batch_requires_poses():
    name = "PoseNoInput"
    _character(name)
    with pytest.raises(posegen.PoseGenError) as exc:
        posegen.run_batch(None, {"character": name, "poses": []}, pose_fn=lambda *a: None)
    assert "姿势图" in str(exc.value)
    ch.delete_character(name)


def test_run_batch_requires_character_base(tmp_path):
    name = "PoseNoBase"
    _character(name, base=0)
    poses = _poses(tmp_path, 1)          # 姿势图要给真的，否则先报"没有姿势图"
    with pytest.raises(posegen.PoseGenError) as exc:
        posegen.run_batch(None, {"character": name, "poses": poses},
                          pose_fn=lambda *a: None)
    assert "底图" in str(exc.value)
    ch.delete_character(name)


# ---------------------------------------------------------------- 服务层


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def test_api_pose_generate_requires_poses(client):
    name = "PoseApiNoInput"
    _character(name)
    r = client.post(f"/characters/{name}/poses/generate", json={"poses": []})
    assert r.status_code == 400
    assert "姿势图" in r.json()["detail"]
    ch.delete_character(name)


def test_api_pose_generate_enqueues(client, monkeypatch):
    from workflows import paths

    name = "PoseApiJob"
    _character(name)
    pose = Path(paths.ASSETS) / "pose_dir_x" / "f0.png"
    pose.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 48)).save(pose)

    def fake_run(engine, params, job=None):
        if job is not None:
            job["progress"] = {"done": 1, "total": 1, "current": ""}
        return {"count": 1, "sprites": ["x"], "failed": [], "seconds": 1}

    monkeypatch.setattr(posegen, "run_batch", fake_run)
    r = client.post(f"/characters/{name}/poses/generate",
                    json={"poses": [str(pose)], "matte": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1 and body["defaults"]["steps"] == 20
    for _ in range(60):
        job = client.get(f"/jobs/{body['job_id']}").json()
        if job.get("status") in ("done", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["workflow"] == "wposegen"
    ch.delete_character(name)
