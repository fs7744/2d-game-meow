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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from workflows import generate as gen
from workflows import lora as lora_reg
from workflows import train as trainer
from workflows.engine import Engine

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

app = FastAPI(title="2D Game Asset Pipeline", version="0.1.0")
engine = Engine(ROOT / "models")

_jobs: Dict[str, Dict[str, Any]] = {}
_q: "queue.Queue[tuple[str, str, Dict[str, Any]]]" = queue.Queue()

# 训练任务单独跑：一次几分钟，绝不能塞进生成队列把出图全卡住
_trains: Dict[str, Dict[str, Any]] = {}


class Job(BaseModel):
    workflow: str = Field(..., description="w1|w2|w3|w4")
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainReq(BaseModel):
    name: str = Field(..., description="输出名，写到 models/lora/<name>.safetensors")
    data_dir: str = Field(..., description="数据集目录（相对项目根或绝对路径）")
    caption: str = Field("", description="没有 .txt 时的统一 caption")
    trigger: str = Field("", description="写进 LoRA 清单的触发词")
    label: str = Field("", description="清单里显示的名字，默认用 name")
    rank: int = 8
    alpha: int = 0
    lr: float = 1e-4
    steps: int = 0
    epochs: int = 8
    batch: int = 1
    resolution: int = 512
    seed: int = 42
    center_crop: bool = False


def _resolve_lora(params: Dict[str, Any]) -> Dict[str, Any]:
    """params.lora 允许填「名字」或「相对/绝对路径」，统一解析成绝对路径。

    名字不存在时清空而不是报错：LoRA 是增强项，不该因为它让整个任务失败，
    但要留痕到 job 里以便排查。
    """
    name = params.get("lora")
    if not name:
        return params
    path = lora_reg.resolve(ROOT / "models", str(name))
    params = dict(params)
    if path is None:
        params["lora"] = None
        params["lora_missing"] = str(name)
    else:
        params["lora"] = path
    return params


def _run(job_id: str, workflow: str, params: Dict[str, Any]) -> None:
    job = _jobs[job_id]
    job.update(status="running", started=time.time())
    try:
        fn = {"w1": gen.txt2img, "w2": gen.img2img, "w3": gen.inpaint, "w4": gen.poseswap}[workflow]
        job["result"] = fn(engine, **_resolve_lora(params))
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


def _register_lora(name: str, label: str, trigger: str) -> None:
    """训练完把元数据写进 loras.json —— 否则前端不知道触发词，等于白训。"""
    mf = ROOT / "models" / "lora" / "loras.json"
    try:
        data = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    except json.JSONDecodeError:
        data = {}
    prev = data.get(name) or {}
    data[name] = {
        **prev,
        "label": label or prev.get("label") or name,
        "trigger": trigger or prev.get("trigger") or "",
        "scale": prev.get("scale", 1.0),
        "preset": prev.get("preset", "pixel"),
        "note": prev.get("note") or "本地训练产出",
    }
    mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_train(train_id: str, req: "TrainReq") -> None:
    st = _trains[train_id]
    st.update(status="running", started=time.time())

    def on_progress(ev: Dict[str, Any]) -> None:
        st.update(ev)
        st["updated"] = time.time()

    try:
        data_dir = Path(req.data_dir)
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        steps = req.steps or trainer.plan_steps(
            len(trainer.Dataset(data_dir, resolution=req.resolution,
                                caption=req.caption, center_crop=req.center_crop)),
            epochs=req.epochs, batch_size=req.batch)

        info = trainer.train(
            ROOT / "models", data_dir, req.name,
            caption=req.caption, rank=req.rank, alpha=req.alpha or None,
            lr=req.lr, steps=steps, batch_size=req.batch,
            resolution=req.resolution, seed=req.seed,
            center_crop=req.center_crop, progress=on_progress,
        )
        _register_lora(req.name, req.label, req.trigger)
        st.update(info, status="done")
    except Exception as exc:  # noqa: BLE001
        st.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                  trace=traceback.format_exc(limit=6))
    finally:
        st["finished"] = time.time()


@app.post("/train")
def train_submit(req: TrainReq) -> Dict[str, Any]:
    """提交 LoRA 训练（后台线程，不占生成队列）。"""
    tid = uuid.uuid4().hex[:12]
    _trains[tid] = {"train_id": tid, "status": "queued", "name": req.name,
                    "params": req.model_dump(), "step": 0, "total": 0}
    threading.Thread(target=_run_train, args=(tid, req), daemon=True,
                     name=f"train-{tid}").start()
    return {"train_id": tid}


@app.get("/train/{train_id}")
def train_status(train_id: str) -> Dict[str, Any]:
    if train_id not in _trains:
        raise HTTPException(404, "no such train job")
    return _trains[train_id]


@app.get("/train")
def train_list() -> Dict[str, Any]:
    items = sorted(_trains.values(), key=lambda x: -(x.get("started") or 0))
    return {"count": len(items), "items": items[:20]}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "queued": _q.qsize(), "engine": engine.stats()}


@app.get("/engine")
def engine_stats() -> Dict[str, Any]:
    return engine.stats()


@app.get("/loras")
def loras() -> Dict[str, Any]:
    """可用 LoRA 清单（含触发词与推荐强度）。前端下拉直接用。"""
    items = lora_reg.scan(ROOT / "models")
    return {"count": len(items), "items": items}


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


# 前端构建产物（web/dist）。存在就挂在根路径，API 路由在上面已注册，不会被抢。
WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
else:  # pragma: no cover
    @app.get("/")
    def _no_web() -> Dict[str, str]:
        return {"hint": "前端未构建：cd web && npm install && npm run build"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
