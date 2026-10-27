"""工作台服务入口（文档 §7.1 编排调度层）。

单卡串行：一个 worker 线程 + FIFO 队列，避免并发抢显存。
端点：
  GET  /health            服务与显存状态
  GET  /engine            加载中的 pipeline
  POST /generate          提交任务 -> {"job_id": ...}
  GET  /jobs/{job_id}     查询任务
  GET  /jobs              最近任务
  GET  /assets            资产清单（读 assets/index.jsonl）
  GET  /files/{path}      取产物
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from workflows import generate as gen
from workflows.engine import Engine

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

app = FastAPI(title="2D Game Asset Pipeline", version="0.1.0")
engine = Engine(ROOT / "models")

_jobs: Dict[str, Dict[str, Any]] = {}
_q: "queue.Queue[tuple[str, str, Dict[str, Any]]]" = queue.Queue()


class Job(BaseModel):
    workflow: str = Field(..., description="w1|w2|w3|w4")
    params: Dict[str, Any] = Field(default_factory=dict)


def _run(job_id: str, workflow: str, params: Dict[str, Any]) -> None:
    job = _jobs[job_id]
    job.update(status="running", started=time.time())
    try:
        fn = {"w1": gen.txt2img, "w2": gen.img2img, "w3": gen.inpaint, "w4": gen.poseswap}[workflow]
        job["result"] = fn(engine, **params)
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        job["trace"] = traceback.format_exc(limit=6)
    finally:
        job["finished"] = time.time()
        job["elapsed"] = round(job["finished"] - job["started"], 2)


def _worker() -> None:
    while True:
        job_id, workflow, params = _q.get()
        _run(job_id, workflow, params)
        _q.task_done()


threading.Thread(target=_worker, daemon=True, name="gen-worker").start()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "queued": _q.qsize(), "engine": engine.stats()}


@app.get("/engine")
def engine_stats() -> Dict[str, Any]:
    return engine.stats()


@app.post("/generate")
def submit(job: Job) -> Dict[str, Any]:
    if job.workflow not in ("w1", "w2", "w3", "w4"):
        raise HTTPException(400, "workflow 必须是 w1/w2/w3/w4")
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": job.workflow, "params": job.params,
                     "status": "queued", "created": time.time()}
    _q.put((job_id, job.workflow, job.params))
    return {"job_id": job_id, "status": "queued", "position": _q.qsize()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(404, "not found")
    return _jobs[job_id]


@app.get("/jobs")
def list_jobs(limit: int = 20) -> Dict[str, Any]:
    items = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)[:limit]
    return {"count": len(items), "items": items}


@app.get("/assets")
def assets(limit: int = 50) -> Dict[str, Any]:
    idx = ASSETS / "index.jsonl"
    if not idx.exists():
        return {"count": 0, "items": []}
    lines = idx.read_text("utf-8").strip().splitlines()[-limit:]
    return {"count": len(lines), "items": [json.loads(x) for x in reversed(lines)]}


@app.get("/files/{path:path}")
def files(path: str) -> FileResponse:
    target = (ASSETS / path).resolve()
    if not str(target).startswith(str(ASSETS.resolve())) or not target.exists():
        raise HTTPException(404, "not found")
    return FileResponse(target)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
