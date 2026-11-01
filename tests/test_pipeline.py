"""一键出全套（workflows/pipeline.py）。

这一层只做**编排**，不重新实现任何一步。所以测试的重点不是"图对不对"，
而是三件事：

  · **顺序**：内衣 → 姿势 → 表情，顺序错了出来的图不能用
    （先出表情再出姿势，表情全白做）；
  · **依赖传递**：勾了内衣，姿势就得从内衣那张底图出发（`source_costume`），
    不能从上一套遗留的服装底图出发；
  · **进度换算**：子步骤只知道自己的 1/N，得换算成全局进度条；
    换算错了表现是"进度条来回跳"或"永远停在 3%"。

三步的实现在别处已经单独测过（`test_clothes` / `test_posegen` /
`test_emotiongen`），这里全部用注入的假函数，不碰 GPU。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from workflows import characters as ch
from workflows import pipeline


def _character(name: str, sprites: int = 1) -> None:
    if ch.character_exists(name):
        ch.delete_character(name)
    ch.create_character(name)
    d = Path(ch.sprites_dir(name, "Naked", "neutral"))
    d.mkdir(parents=True, exist_ok=True)
    for i in range(sprites):
        Image.new("RGB", (32, 48), (200, 180, 160)).save(
            d / f"sprite_pose_{i + 1:04d}.png")


def _poses(tmp_path, count: int) -> list:
    paths = []
    for i in range(count):
        p = tmp_path / f"pose_{i + 1}.png"
        Image.new("RGB", (32, 48), (10, 10, 10)).save(p)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------- 计划


def test_plan_rejects_empty_steps():
    """什么都不勾时要**说清楚**，而不是静默成功（用户会以为跑了）。"""
    name = "PipeEmpty"
    _character(name)
    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.run_pipeline(None, {"character": name})
    assert "至少勾一个" in str(exc.value)
    ch.delete_character(name)


def test_plan_emotions_only_skips_poses():
    """只给表情 → 只跑表情（不该强迫用户先出一遍姿势）。"""
    name = "PipeEmoOnly"
    _character(name)
    calls = []

    def fake_emotion(engine, params, **kw):
        calls.append(("emotions", params))
        return {"count": 1, "total": 1, "failed": [], "seconds": 0.1}

    out = pipeline.run_pipeline(
        None, {"character": name, "emotions": ["smile"]},
        emotion_fn=fake_emotion)
    assert out["steps"] == ["emotions"]
    assert [c[0] for c in calls] == ["emotions"]
    ch.delete_character(name)


def test_plan_order_is_underwear_poses_emotions(tmp_path):
    """顺序照官方：**先底图、再姿势、最后表情**。

    反了的话：先出表情时还没有姿势帧，表情无源；或者表情出完又被姿势整批替换掉。
    """
    name = "PipeOrder"
    _character(name)
    poses = _poses(tmp_path, 2)
    seen = []

    def fake_uw(engine, params, **kw):
        seen.append("underwear")
        return {"costume": "Underwear", "sprite": "x.png", "seconds": 0.1}

    def fake_pose(engine, params, **kw):
        seen.append("poses")
        return {"count": 2, "sprites": ["a", "b"], "seconds": 0.1}

    def fake_emo(engine, params, **kw):
        seen.append("emotions")
        return {"count": 4, "total": 4, "failed": [], "seconds": 0.1}

    out = pipeline.run_pipeline(
        None,
        {"character": name, "underwear": True, "poses": poses,
         "emotions": ["smile", "angry"]},
        underwear_fn=fake_uw, pose_fn=fake_pose, emotion_fn=fake_emo)
    assert seen == ["underwear", "poses", "emotions"], seen
    assert out["steps"] == ["underwear", "poses", "emotions"]
    ch.delete_character(name)


# ---------------------------------------------------------------- 依赖传递


def test_underwear_feeds_pose_source(tmp_path):
    """勾了内衣 → 姿势必须从**内衣那张底图**出发。

    这是官方的依赖（`naked_pose_generation` 吃 `remove_clothes` 的产物）。
    不传 `source_costume` 的话，姿势会从上一次遗留的服装底图出发，
    出来的帧穿着上一套衣服 —— 而且**不报错**，肉眼才发现。
    """
    name = "PipeSource"
    _character(name)
    poses = _poses(tmp_path, 1)
    captured = {}

    def fake_uw(engine, params, **kw):
        return {"costume": "Underwear", "sprite": "x.png", "seconds": 0.1}

    def fake_pose(engine, params, **kw):
        captured.update(params)
        return {"count": 1, "sprites": ["a"], "seconds": 0.1}

    out = pipeline.run_pipeline(
        None, {"character": name, "underwear": True, "poses": poses,
               "costume": "Nurse"},
        underwear_fn=fake_uw, pose_fn=fake_pose)
    assert captured["source_costume"] == "Underwear", captured.get("source_costume")
    # 产物仍要落在目标服装（Nurse）下，不是 Underwear
    assert captured["costume"] == "Nurse"
    assert out["underwear_costume"] == "Underwear"
    ch.delete_character(name)


def test_without_underwear_pose_source_is_empty(tmp_path):
    """没勾内衣时**不能**瞎指定 source_costume（否则会报"底图不存在"）。"""
    name = "PipeNoSource"
    _character(name)
    poses = _poses(tmp_path, 1)
    captured = {}

    def fake_pose(engine, params, **kw):
        captured.update(params)
        return {"count": 1, "sprites": ["a"], "seconds": 0.1}

    pipeline.run_pipeline(None, {"character": name, "poses": poses},
                          pose_fn=fake_pose)
    assert captured["source_costume"] == ""
    ch.delete_character(name)


# ---------------------------------------------------------------- 进度


def test_progress_is_rescaled_into_one_bar(tmp_path):
    """子步骤的 1/N 要换算成全局进度，不能各报各的。

    三个步骤权重按张数分（1 + 2 + 2×2 = 7）。内衣跑完 = 1/7，
    姿势跑到一半 ≈ 1/7 + 1/7，表情跑完 = 7/7。
    """
    name = "PipeProgress"
    _character(name)
    poses = _poses(tmp_path, 2)
    job = {}
    samples = []

    def fake_uw(engine, params, **kw):
        kw["job"]["progress"] = {"done": 1, "total": 1, "current": ""}
        samples.append(dict(job["progress"]))
        return {"costume": "Underwear", "sprite": "x", "seconds": 0.1}

    def fake_pose(engine, params, **kw):
        for i in (0, 1, 2):
            kw["job"]["progress"] = {"done": i, "total": 2, "current": f"{i}/2"}
            samples.append(dict(job["progress"]))
        return {"count": 2, "sprites": ["a", "b"], "seconds": 0.1}

    def fake_emo(engine, params, **kw):
        kw["job"]["progress"] = {"done": 1, "total": 1, "current": ""}
        samples.append(dict(job["progress"]))
        return {"count": 4, "total": 4, "failed": [], "seconds": 0.1}

    pipeline.run_pipeline(
        None,
        {"character": name, "underwear": True, "poses": poses,
         "emotions": ["smile", "angry"]},
        job=job, underwear_fn=fake_uw, pose_fn=fake_pose, emotion_fn=fake_emo)

    total = samples[0]["total"]
    assert total == 1 + 2 + 2 * 2, total          # 权重 = 张数
    # 单调不减（进度条不能往回走）
    dones = [s["done"] for s in samples]
    assert dones == sorted(dones), dones
    # 最后一条必须等于总量
    assert dones[-1] == total, (dones[-1], total)
    # 带步骤名，用户知道现在在做哪一步
    assert any(s.get("step") == "poses" for s in samples)
    ch.delete_character(name)


def test_progress_works_without_a_job(tmp_path):
    """不传 job 也要能跑（CLI / 测试 / 内部调用）。"""
    name = "PipeNoJob"
    _character(name)
    poses = _poses(tmp_path, 1)

    def fake_pose(engine, params, **kw):
        kw["job"]["progress"] = {"done": 1, "total": 1}   # job 是 None，写进临时对象
        return {"count": 1, "sprites": ["a"], "seconds": 0.1}

    out = pipeline.run_pipeline(None, {"character": name, "poses": poses},
                                pose_fn=fake_pose)
    assert out["poses"]["count"] == 1
    ch.delete_character(name)


# ---------------------------------------------------------------- 失败


def test_step_failure_does_not_stop_the_rest(tmp_path):
    """一步失败不该把整条流水线拖死 —— 用户还等着看后面的结果。

    以前这种"中间某步出错"最容易表现为整批没产出、也没有任何提示。
    """
    name = "PipeFail"
    _character(name)
    poses = _poses(tmp_path, 1)
    ran = []

    def fake_uw(engine, params, **kw):
        ran.append("underwear")
        raise RuntimeError("底图没生成出来")

    def fake_pose(engine, params, **kw):
        ran.append("poses")
        return {"count": 1, "sprites": ["a"], "seconds": 0.1}

    out = pipeline.run_pipeline(
        None, {"character": name, "underwear": True, "poses": poses},
        underwear_fn=fake_uw, pose_fn=fake_pose)
    assert ran == ["underwear", "poses"]
    assert out["poses"]["count"] == 1
    assert out["failed_steps"] and out["failed_steps"][0]["step"] == "underwear"
    assert "底图没生成出来" in out["failed_steps"][0]["error"]
    ch.delete_character(name)


def test_continue_on_error_false_raises(tmp_path):
    name = "PipeStrict"
    _character(name)
    poses = _poses(tmp_path, 1)

    def boom(engine, params, **kw):
        raise RuntimeError("炸了")

    with pytest.raises(pipeline.PipelineError) as exc:
        pipeline.run_pipeline(
            None, {"character": name, "poses": poses, "continue_on_error": False},
            pose_fn=boom)
    assert "出多姿势失败" in str(exc.value)
    ch.delete_character(name)


# ---------------------------------------------------------------- 服务层


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def test_api_pipeline_enqueues_and_reports_progress(client, monkeypatch, tmp_path):
    """接口只入队；worker 里的三步全部替掉（真跑要几分钟 + 一张卡）。"""
    name = "PipeApi"
    _character(name)
    poses = _poses(tmp_path, 2)

    import app as app_module

    def fake_uw(engine, params, job=None, **kw):
        if job is not None:
            job["progress"] = {"done": 1, "total": 1, "current": ""}
        return {"costume": "Underwear", "sprite": "x", "seconds": 0.1}

    def fake_pose(engine, params, job=None, **kw):
        if job is not None:
            job["progress"] = {"done": 2, "total": 2, "current": ""}
        return {"count": 2, "sprites": ["a", "b"], "seconds": 0.1}

    def fake_emo(engine, params, job=None, **kw):
        if job is not None:
            job["progress"] = {"done": 1, "total": 1, "current": ""}
        return {"count": 1, "total": 1, "failed": [], "seconds": 0.1}

    monkeypatch.setattr(app_module, "_pipeline_fns",
                        lambda: (fake_uw, fake_pose, fake_emo), raising=False)
    r = client.post(f"/characters/{name}/pipeline", json={
        "underwear": True, "poses": poses, "emotions": ["smile"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["steps"] == ["underwear", "poses", "emotions"]
    # 权重 = 张数：内衣 1 + 姿势 2 + 表情（2 帧 × 1 个）= 5
    assert body["total"] == 1 + 2 + 2

    job_id = body["job_id"]
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("status") in ("done", "failed"):
            break
        import time as _t
        _t.sleep(0.05)
    assert job["status"] == "done", job
    assert job["workflow"] == "wpipe"
    assert job["result"]["poses"]["count"] == 2
    ch.delete_character(name)


def test_api_pipeline_rejects_empty_steps(client):
    name = "PipeApiEmpty"
    _character(name)
    r = client.post(f"/characters/{name}/pipeline", json={})
    assert r.status_code == 400
    assert "至少勾一个" in r.json()["detail"]
    ch.delete_character(name)


def test_api_pipeline_unknown_character(client):
    r = client.post("/characters/NoSuchPipeChar/pipeline", json={"poses": ["x.png"]})
    assert r.status_code in (400, 404)
