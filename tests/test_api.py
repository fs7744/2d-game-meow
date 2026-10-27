"""服务层：端点契约、队列、路径穿越防护。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and "engine" in body
    assert body["engine"]["total_gb"] > 0


def test_engine_stats(client):
    r = client.get("/engine")
    assert r.status_code == 200
    assert "NVIDIA" in r.json()["device"]


def test_generate_roundtrip(client):
    payload = {"workflow": "w1", "params": {
        "prompt": "blue slime monster", "width": 256, "height": 256,
        "steps": 8, "seed": 42, "pixel": 4, "palette": 16, "tag": "apitest"}}
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = None
    for _ in range(120):                       # 最多等 60s
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job.get("error")
    out = Path(job["result"]["file"])
    assert out.exists() and out.stat().st_size > 0
    assert job["result"]["meta"]["seed"] == 42


def test_assets_lists_generated(client):
    r = client.get("/assets?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0
    assert any("apitest" in str(i) for i in body["items"])


def test_files_serves_asset(client):
    name = client.get("/assets?limit=1").json()["items"][0].get("file")
    assert name
    r = client.get(f"/files/{Path(name).name}")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_files_blocks_path_traversal(client):
    r = client.get("/files/..%2F..%2Fapp.py")
    assert r.status_code == 404
    r2 = client.get("/files/nope_not_here.png")
    assert r2.status_code == 404


def test_bad_workflow_rejected(client):
    r = client.post("/generate", json={"workflow": "w9", "params": {}})
    assert r.status_code == 400


def test_unknown_job_404(client):
    assert client.get("/jobs/deadbeef").status_code == 404


def test_queue_serializes_two_jobs(client):
    ids = []
    for s in (11, 12):
        r = client.post("/generate", json={"workflow": "w1", "params": {
            "prompt": "green mushroom monster", "width": 256, "height": 256,
            "steps": 6, "seed": s, "tag": "queue"}})
        ids.append(r.json()["job_id"])
    states = []
    for _ in range(120):
        states = [client.get(f"/jobs/{i}").json()["status"] for i in ids]
        if all(s in ("done", "failed") for s in states):
            break
        time.sleep(0.5)
    assert states == ["done", "done"]
    # 串行语义：后提交的开始时间不早于先提交的结束时间
    a = client.get(f"/jobs/{ids[0]}").json()
    b = client.get(f"/jobs/{ids[1]}").json()
    assert b["started"] >= a["finished"] - 0.05


def test_loras_endpoint(client):
    """LoRA 清单：真实目录里应当能扫到 wuxia_v2（含触发词）。"""
    r = client.get("/loras")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["items"], list)
    names = {x["name"] for x in body["items"]}
    if not names:
        pytest.skip("本机未放 LoRA 权重")
    item = next(x for x in body["items"] if x["name"] == "wuxia_v2")
    assert item["trigger"] == "wuxia"
    assert item["path"].startswith("lora/")
    assert item["size_mb"] > 0


def test_generate_resolves_lora_name(client):
    """params.lora 传名字要能解析成路径；名字不存在则降级而不是整单失败。"""
    import app as app_module

    resolved = app_module._resolve_lora({"lora": "wuxia_v2"})
    assert resolved["lora"].endswith("wuxia_v2.safetensors")

    missing = app_module._resolve_lora({"lora": "nope"})
    assert missing["lora"] is None and missing["lora_missing"] == "nope"

    assert app_module._resolve_lora({"prompt": "x"}) == {"prompt": "x"}


def test_generate_with_lora_name_roundtrip(client):
    """端到端：只给名字，服务端解析后能真跑出一张图。"""
    import app as app_module

    if not (app_module.ROOT / "models" / "lora" / "wuxia_v2.safetensors").exists():
        pytest.skip("本机无 wuxia_v2 权重")

    payload = {"workflow": "w1", "params": {
        "prompt": "Wuxia, swordsman, front view", "width": 256, "height": 256,
        "steps": 8, "seed": 7, "preset": "wuxia", "pixel": 4, "palette": 16,
        "matte": True, "lora": "wuxia_v2", "lora_scale": 0.8, "tag": "apilora"}}
    job_id = client.post("/generate", json=payload).json()["job_id"]

    for _ in range(120):
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert j["status"] == "done", j.get("error")
    meta = j["result"]["meta"]
    assert meta["lora"].endswith("wuxia_v2.safetensors")
    assert meta["lora_scale"] == 0.8
    assert isinstance(meta["edge_touch"], list)
