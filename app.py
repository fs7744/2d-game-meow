"""工作台服务入口（文档 §7.1 编排调度层）。

单卡串行：一个 worker 线程 + FIFO 队列，避免并发抢显存。
端点：
  GET  /health            服务与显存状态
  GET  /engine            加载中的 pipeline
  POST /generate          提交任务 -> {"job_id": ...}（w1/w2/w3/w4/wbg/w5/wpx/wpc/wss）
  GET  /jobs/{job_id}     查询任务
  GET  /jobs              最近任务
  GET  /assets            资产清单（读 assets/index.jsonl）
  GET  /files/{path}      取产物（?download=1 带 Content-Disposition 直接下载）
  POST /upload            上传本机图片（原始字节）→ 磁盘路径 + 展示 URL
  POST /mask              保存局部重绘的蒙版（PNG，项目目录 masks/ 下同名覆盖）
  DELETE /assets          真删：删磁盘产物 + 清索引
"""
from __future__ import annotations

import io
import json
import os
import shutil

import numpy as np
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from PIL import Image

from workflows import generate as gen
from workflows import lora as lora_reg
from workflows import train as trainer
from workflows.engine import Engine

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
UPLOAD_DIR = ASSETS / "uploads"
PROJECTS_DIR = ASSETS / "projects"
MAX_UPLOAD = 20 * 1024 * 1024
UPLOAD_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp",
                  "BMP": ".bmp", "GIF": ".gif"}

app = FastAPI(title="2D Game Asset Pipeline", version="0.1.0")
engine = Engine(ROOT / "models")

_jobs: Dict[str, Dict[str, Any]] = {}
_q: "queue.Queue[tuple[str, str, Dict[str, Any]]]" = queue.Queue()

# 训练任务单独跑：一次几分钟，绝不能塞进生成队列把出图全卡住
_trains: Dict[str, Dict[str, Any]] = {}


class Job(BaseModel):
    workflow: str = Field(..., description="w1|w2|w3|w4")
    params: Dict[str, Any] = Field(default_factory=dict)


class DeleteReq(BaseModel):
    """真删请求。paths 里可以是绝对路径，也可以是相对 assets 的路径。"""

    paths: list[str] = Field(default_factory=list)


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
        fn = {"w1": gen.txt2img, "w2": gen.img2img, "w3": gen.inpaint,
              "w4": gen.poseswap, "wbg": gen.matte_bg,
              "w5": gen.ai_repaint, "wpx": gen.pixelify,
              "wpc": gen.pose_clone, "wss": gen.sprite_swap}[workflow]
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


@app.get("/models")
def models() -> Dict[str, Any]:
    """可用底模。前端下拉用它；没有对应权重就不显示，避免选了却跑不动。"""
    return {"current": engine.DEFAULT_MODEL, "available": engine.available_models()}


@app.get("/loras")
def loras() -> Dict[str, Any]:
    """可用 LoRA 清单（含触发词与推荐强度）。前端下拉直接用。"""
    items = lora_reg.scan(ROOT / "models")
    return {"count": len(items), "items": items}


@app.post("/generate")
def submit(job: Job) -> Dict[str, Any]:
    if job.workflow not in ("w1", "w2", "w3", "w4", "wbg", "w5", "wpx", "wpc", "wss"):
        raise HTTPException(400, "workflow 必须是 w1/w2/w3/w4/wbg/w5/wpx/wpc/wss")
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
def assets(limit: int = 50, project: str = "") -> Dict[str, Any]:
    """资产清单。带 project 时只返回该项目的产物（项目之间相互隔离）。"""
    idx = ASSETS / "index.jsonl"
    if not idx.exists():
        return {"count": 0, "items": []}
    lines = idx.read_text("utf-8").strip().splitlines()
    items = []
    for x in reversed(lines):
        if not x.strip():
            continue
        try:
            item = json.loads(x)
        except json.JSONDecodeError:
            continue
        if project and item.get("scope", "default") != project:
            continue
        # 陈旧条目（文件已被手工删掉）不再列出来，否则前端会拿到 404 的链接
        target = item.get("file")
        if target and not Path(str(target)).exists():
            continue
        items.append(item)
    items = items[:limit]
    return {"count": len(items), "items": items}


def _safe_asset_path(p: str) -> Optional[Path]:
    """把用户给的路径解析成 assets 内的真实文件；不在 assets 内一律拒绝。

    这是删除接口唯一的防线：路径穿越（../）能把 assets 外的文件删掉。
    """
    try:
        t = Path(p)
        if not t.is_absolute():
            # 相对路径有两种习惯：相对 assets、相对项目根。先试存在的那个
            cand_a = ASSETS / t
            cand_r = ROOT / t
            t = cand_a if cand_a.exists() else (cand_r if cand_r.exists() else cand_a)
        t = t.resolve()
    except (OSError, ValueError):
        return None
    root = ASSETS.resolve()
    if not str(t).startswith(str(root)):
        return None
    if t.name in ("index.jsonl", "uploads"):     # 索引与旧上传目录不能删
        return None
    if t.is_dir():
        # 只允许删项目目录（assets/projects/<某个项目>/），不能删 assets 或 projects 根
        if t == PROJECTS_DIR.resolve() or t == root:
            return None
    return t


@app.delete("/assets")
def delete_assets(req: DeleteReq) -> Dict[str, Any]:
    """真删：删磁盘上的产物（含 sidecar 与 raw），并把索引里的条目摘掉。

    删一个 PNG 会连带删：
      - `<name>.png`        成品
      - `<name>.png.json`   sidecar
      - `<name>_raw.png`    后处理之前的原图（如果存在）
    上传的图片（assets/uploads/）同样能删。

    返回逐条结果：删掉的、被拒绝的、没找到的——前端要能告诉用户「哪几条没删掉」。
    """
    deleted: list[str] = []
    skipped: list[str] = []
    gone: list[str] = []
    targets: set[Path] = set()
    for raw in req.paths:
        t = _safe_asset_path(raw)
        if t is None:
            skipped.append(raw)
            continue
        targets.add(t)
        if t.suffix == ".png":                  # 连带 sidecar 与 raw
            targets.add(t.with_suffix(t.suffix + ".json"))
            targets.add(t.with_name(t.stem + "_raw" + t.suffix))

    for t in sorted(targets):
        try:
            if t.is_dir():
                # 项目目录：整个删掉（真删项目）
                shutil.rmtree(t, ignore_errors=True)
                deleted.append(str(t))
            elif t.exists():
                t.unlink()
                deleted.append(str(t))
            else:
                gone.append(str(t))
        except OSError as exc:
            skipped.append(f"{t}: {exc}")

    # 索引里对应的条目也要摘掉，否则 /assets 还会列出已经不存在的文件
    idx = ASSETS / "index.jsonl"
    if idx.exists() and deleted:
        try:
            kept = []
            for line in idx.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                paths = {str(item.get("file") or ""), str(item.get("raw") or ""),
                         str(item.get("sheet") or "")}
                if paths & set(deleted):
                    continue
                # 目录被整体删掉时，目录下的条目也要摘
                if any(path and any(path.startswith(d + os.sep) for d in deleted)
                       for path in paths):
                    continue
                kept.append(line)
            idx.write_text("\n".join(kept) + ("\n" if kept else ""), "utf-8")
        except OSError:
            pass

    return {"deleted": deleted, "skipped": skipped, "not_found": gone,
            "count": len(deleted)}


@app.post("/mask")
async def save_mask(request: Request, name: str = "mask",
                    project: str = "") -> Dict[str, Any]:
    """保存局部重绘用的蒙版（PNG 原始字节），放在项目目录的 masks/ 下，同名覆盖。

    为什么单独一个端点而不是走 /upload：蒙版是**每个节点一份**的草稿，
    反复涂抹会覆盖同一个文件；走 /upload 每次都生成新文件，很快就会攒一堆垃圾。
    """
    data = await request.body()
    if not data:
        raise HTTPException(400, "空蒙版")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"蒙版太大（上限 {MAX_UPLOAD // 2**20}MB）")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        fmt = (im.format or "").upper()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"不是有效的图片：{exc}") from exc
    if fmt != "PNG":
        raise HTTPException(400, f"蒙版必须是 PNG（收到 {fmt}）")
    safe = "".join(
        ch for ch in (name or "mask") if ch.isascii() and (ch.isalnum() or ch in "-_")
    )[:64] or "mask"
    scope = "".join(
        ch for ch in (project or "") if ch.isascii() and (ch.isalnum() or ch in "-_")
    )[:64] or "default"
    d = PROJECTS_DIR / scope / "masks"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{safe}.png"
    out.write_bytes(data)
    arr = np.asarray(im.convert("L"))
    coverage = float((arr > 127).mean())
    return {"path": str(out), "url": f"/files/{out.relative_to(ASSETS).as_posix()}",
            "width": im.width, "height": im.height,
            "coverage": round(coverage, 4)}


@app.post("/upload")
async def upload_image(request: Request, name: str = "",
                       project: str = "") -> Dict[str, Any]:
    """接收一张本地图（请求体是**原始字节**，不是 multipart）。

    为什么不用 multipart：多一个依赖（python-multipart），而浏览器端
    `fetch(url, {body: file})` 直接就能发原始字节，少一层转换少一处出错。

    校验三件事：体积上限、真能解码（防止把任意文件塞进来）、扩展名白名单。
    文件名会被清洗（只留字母数字与 -_），再拼一个 uuid 前缀，避免覆盖与路径穿越。
    """
    data = await request.body()
    if not data:
        raise HTTPException(400, "空文件")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"文件太大（上限 {MAX_UPLOAD // 2**20}MB）")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()                                  # 真的解码一遍，坏文件到这里就现原形
        fmt = (im.format or "PNG").upper()
        size = im.size
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(400, f"不是有效的图片：{exc}") from exc
    if fmt not in UPLOAD_FORMATS:
        raise HTTPException(400, f"不支持的图片格式：{fmt}")
    ext = UPLOAD_FORMATS[fmt]
    # 磁盘名只用 ASCII：中文名会让 URL 必须百分号编码，jsdom / 某些代理会出问题。
    # 原名另外回传，前端可以显示。
    stem = "".join(
        ch for ch in Path(name).stem if ch.isascii() and (ch.isalnum() or ch in "-_")
    )[:40] or "image"
    # 项目隔离：上传的文件放进**本项目**的目录，别的项目拿不到
    if project:
        out_dir = PROJECTS_DIR / ("".join(
            ch for ch in project if ch.isascii() and (ch.isalnum() or ch in "-_"))[:64]
        ) / "uploads"
    else:
        out_dir = UPLOAD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{uuid.uuid4().hex[:10]}_{stem}{ext}"
    out.write_bytes(data)
    return {"path": str(out), "url": f"/files/{out.relative_to(ASSETS).as_posix()}",
            "name": Path(name).name or out.name,
            "w": size[0], "h": size[1], "format": fmt, "bytes": len(data)}


@app.get("/files/{path:path}")
def files(path: str, download: int = 0) -> FileResponse:
    """取产物。download=1 时带 Content-Disposition，浏览器直接存盘。"""
    target = (ASSETS / path).resolve()
    if not str(target).startswith(str(ASSETS.resolve())) or not target.exists():
        raise HTTPException(404, "not found")
    if download:
        # FileResponse(filename=…) 会设成 attachment；产物名已经带后缀
        return FileResponse(target, filename=target.name)
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
