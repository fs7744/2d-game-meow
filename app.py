"""工作台服务入口（文档 §7.1 编排调度层）。

单卡串行：一个 worker 线程 + FIFO 队列，避免并发抢显存。
端点：
  GET  /health            服务与显存状态
  GET  /engine            加载中的 pipeline
  POST /generate          提交任务 -> {"job_id": ...}（w1/w2/w3/w4/wbg/w5/wpx/wpc/wss/topixel）
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
import inspect
import json
import os
import shutil

import numpy as np
import queue
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from PIL import Image

from workflows import generate as gen
from workflows import assetdb
from workflows.assetdb import AssetIndex
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
index = AssetIndex(ASSETS)

# 任务表要有上限：老实现是无界 dict，长跑服务会把每个任务的完整 result/meta
# 一直留在内存里（SPA 一开就是几 MB）。只保留最近 N 条，且优先淘汰已结束的。
JOB_KEEP = 200
TRAIN_KEEP = 50
_jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_q: "queue.Queue[tuple[str, str, Dict[str, Any]]]" = queue.Queue()

# 训练任务单独跑：一次几分钟，绝不能塞进生成队列把出图全卡住。
# 但"不进队列"不等于"可以并发占卡"——训练 4.7GB + 生成 5.8GB 很容易撞顶，
# 所以两者共用一把 GPU 锁，串行使用显卡。
_gpu_lock = threading.Lock()


def _prune(table: "OrderedDict[str, Dict[str, Any]]", keep: int) -> None:
    """淘汰最老的条目，**跳过还在跑的**（否则前端轮询会突然 404）。"""
    for k in list(table.keys()):
        if len(table) <= keep:
            break
        if table[k].get("status") in ("queued", "running"):
            continue
        table.pop(k, None)


class Job(BaseModel):
    workflow: str = Field(..., description="w1|w2|w3|w4")
    params: Dict[str, Any] = Field(default_factory=dict)


class DeleteReq(BaseModel):
    """真删请求。paths 里可以是绝对路径，也可以是相对 assets 的路径。"""

    paths: list[str] = Field(default_factory=list)


def _run(job_id: str, workflow: str, params: Dict[str, Any]) -> None:
    job = _jobs[job_id]
    job.update(status="running", started=time.time())
    try:
        fn = {"w1": gen.txt2img, "w2": gen.img2img, "w3": gen.inpaint,
              "w4": gen.poseswap, "wbg": gen.matte_bg,
              "w5": gen.ai_repaint, "wpx": gen.pixelify,
              "wpc": gen.pose_clone, "wss": gen.sprite_swap,
              "topixel": gen.image_to_pixel,
              "wout": gen.outpaint,
              "woutfit": gen.outfit_swap, "wface": gen.face_swap,
              "wvnccs": gen.vnccs_pose,
              # 旧名保留：早期只用于动漫人物
              "wap": gen.image_to_pixel}[workflow]
        # **按目标函数的签名过滤参数**：Qwen-only 之后 lora / clip_skip /
        # hires / distill / refiner / strength / cn_scale / ipa_scale 这些键
        # 已经没有对应的形参了。旧前端（或手写的 curl）还会传，直接 TypeError
        # 太粗暴；这里丢掉并记进 meta，用户翻产物时能看到「这条被忽略了」。
        sig = inspect.signature(fn)
        # **排除第一个位置参数**（eng）：不然客户端传 {"eng": 123} 会被收下，
        # 然后 fn(engine, eng=123) 抛 "got multiple values for argument"。
        kw_names = set(list(sig.parameters)[1:])
        accepted = {k: v for k, v in params.items() if k in kw_names}
        # 多余的键**全部记账**：以前把 lora / lora_scale 排除在外（那时还有
        # 专门的解析逻辑），现在 LoRA 整套已删，它们和 clip_skip / strength
        # 一样是纯粹的多余键，不报出来就等于静默丢弃。
        ignored = sorted(set(params) - set(accepted))
        with _gpu_lock:                    # 单卡不并发
            job["result"] = fn(engine, **accepted)
        if ignored:
            job["result"].setdefault("meta", {})["params_ignored"] = ignored
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


@app.get("/models")
def models() -> Dict[str, Any]:
    """可用底模 + 每个底模在精灵管线上的**推荐参数**。

    前端选中模型时应当把 `recommend` 带出来的值填进表单——实测 SDXL 在 1024²
    反而更差（角色占比大、更易顶边，可交付率 92% → 75%），如果前端默认按
    "SDXL 就该 1024"来填，用户会稳定拿到更差的结果。
    """
    have = engine.available_models()
    return {"current": engine.DEFAULT_MODEL, "available": have,
            "recommend": {m: engine.recommend(m) for m in have},
            # 底模名 → 架构映射。**前端要靠它过滤 LoRA**：LoRA 是按架构兼容的
            # （sd15 / sdxl），而底模名可以是任意目录名（`sdxl-anime` 的架构是
            # sdxl）。按名字比会把 `sdxl-anime` 上的 SDXL LoRA 全过滤掉。
            "arch": {m: engine.arch(m) for m in have},
            # 能力表：前端据此决定哪些开关该显示。
            #   transparent —— 原生 RGBA（只有 Qwen-Image-2.1 的 VAE 有第 4 通道）；
            #                  SD 系列要做透明底只能靠抠图，那是另一条路。
            #   instruction —— 图生图是否走「指令改图」（无 strength 语义）。
            #   workflows   —— 这个底模能跑哪些工作流。前端按节点的 workflow
            #                  过滤它的底模下拉：Qwen 没有 SD 的 UNet/CLIP，
            #                  角色迁移/序列帧替换这类依赖 ControlNet 的节点
            #                  选它就是必崩，不如直接不让选。
            #   approx      —— workflows 里哪些是**近似实现**（硬约束不存在，
            #                  靠软引导/后处理凑）。前端要在节点上打标，
            #                  用户据此决定「能不能拿它替代 SD 的那条路」。
            #   lora —— 是否支持 LoRA。**Qwen 现在也支持了**（2026-10-30 实测：
            #   fp8 量化后的 transformer 照样能挂 diffusers 原生格式的 LoRA，
            #   实测 AnyAngle rank24 / 384 张量挂上后 1440 个 LoRA 模块生效，
            #   出图只慢 1.1s）。但 Qwen 的 LoRA 生态要**先转格式**
            #   （tools/convert_lora.py），ComfyUI 格式挂上去会静默不生效。
            "caps": {m: {"transparent": engine.arch(m) == "qwen21",
                         "instruction": engine.arch(m) == "qwen21",
                         "lora": True,
                         "workflows": engine.workflows_of(m),
                         "approx": engine.approx_of(m)}
                     for m in have},
            # refiner 已随 SD 一起删掉（Qwen 没有精修器）。字段留着恒 False，
            # 是为了老前端的 `if (refiner) 显示开关` 不至于拿到 undefined 就崩。
            "refiner": False}


class LoraReq(BaseModel):
    """挂载 LoRA 的请求。`items` 为空表示卸载全部。"""

    items: list[dict] = Field(default_factory=list)


@app.get("/loras")
def list_loras() -> Dict[str, Any]:
    """可用的 **Qwen** LoRA（只列 diffusers 原生格式的）。

    ComfyUI 格式（`diffusion_model.` + `lora_A/B`）要先转：
    `python tools/convert_lora.py in.safetensors out.safetensors`
    """
    return {"items": engine.available_loras(), "active": engine.active_loras()}


@app.post("/loras/apply")
def apply_loras(req: LoraReq) -> Dict[str, Any]:
    """挂载 LoRA（状态式：一直生效，直到下次 apply 或 unload）。

    `items`：`[{"name": "qwen21_anyangle", "scale": 0.8}]`
    """
    try:
        pipe = engine.txt2img_pipe(engine.DEFAULT_MODEL)
        active = engine.apply_loras(pipe, req.items)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"挂载 LoRA 失败：{exc}") from exc
    return {"active": active, "items": engine.active_loras()}


class FastModeReq(BaseModel):
    """快速模式开关（少步蒸馏）。"""

    on: bool = False


@app.post("/fast_mode")
def set_fast_mode(req: FastModeReq) -> Dict[str, Any]:
    """开关快速模式：挂 Lightning 少步蒸馏 LoRA + 步数压到 8。

    实测（768²/RTX4090）：24 步 17.3s → 8 步 9.3s（-46%），画质几乎不掉。
    """
    try:
        return engine.set_fast_mode(req.on)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"设置快速模式失败：{exc}") from exc


@app.post("/loras/unload")
def unload_loras() -> Dict[str, Any]:
    """卸掉全部 LoRA，回到基座模型。"""
    try:
        pipe = engine.txt2img_pipe(engine.DEFAULT_MODEL)
        engine.unload_loras(pipe)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"卸载 LoRA 失败：{exc}") from exc
    return {"active": [], "items": []}


#: 能当 W4 输入的图片扩展名。与 generate.POSE_EXTS 保持一致。
_POSE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class PoseExtractReq(BaseModel):
    path: str


def _inside(root: Path, target: Path) -> bool:
    """`target` 是否真的在 `root` 目录内。

    **不能用 `str(target).startswith(str(root))`**：那是**字符串前缀**比较，
    不是路径比较——`assets2/x.png`、`assets_BACKUP/x.png` 都会通过。
    实测这会让 `DELETE /assets` 删到 assets 目录之外的文件、
    `GET /files/../assets2/x.png` 读到外部文件。

    用 `is_relative_to`（Python 3.9+），它按路径段比较，`assets2` 不算 `assets`。
    """
    try:
        return target.is_relative_to(root)
    except AttributeError:            # Python < 3.9 兜底
        try:
            return os.path.commonpath([str(root), str(target)]) == str(root)
        except ValueError:
            return False


def _readable_asset_path(p: str) -> Optional[Path]:
    """只读用途的路径校验：必须在 assets 内、是文件、是图片。

    `_safe_asset_path` 是给删除接口写的（还带"不能删 uploads/index"这类规则），
    读图用它会误伤；单独写一个，只做最小必要的安全约束。
    """
    try:
        t = Path(p)
        if not t.is_absolute():
            t = (ASSETS / t) if (ASSETS / t).exists() else (ROOT / t)
        t = t.resolve()
    except (OSError, ValueError):
        return None
    if not _inside(ASSETS.resolve(), t):
        return None
    if not t.is_file() or t.suffix.lower() not in (
            ".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        return None
    return t


@app.post("/pose/extract")
def pose_extract(req: PoseExtractReq) -> Dict[str, Any]:
    """骨骼编辑器「从图片提取」：一张图 → OpenPose-18 归一化关节点。

    复用项目里已有的 DWPose（FasterRCNN 检人 + RTMPose 关键点），坐标经过
    `normalized_points` 归一化——人物居中、按 fill 占满画面高度，正好和
    编辑器里的 0~1 坐标一致，不用再换算。

    返回的 `azimuth_guess` 只是**提示**：2D 图没有深度，正面/背面在骨架上
    是对称的，无法可靠区分；只有"肩膀宽度很窄"能较有把握地判成侧面。
    前端一律按正面导入，方向交给用户点按钮切换。
    """
    target = _readable_asset_path(req.path)
    if target is None:
        raise HTTPException(400, "图片路径不在 assets 内（或不是图片文件）")
    try:
        img = Image.open(target).convert("RGB")
        img.load()
    except Exception as exc:                       # noqa: BLE001
        raise HTTPException(400, f"打不开图片：{exc}") from exc
    ex = engine.pose_extractor()
    if ex is None:
        raise HTTPException(503, "姿态模型未安装（缺 dwpose onnx 或 onnxruntime）")
    try:
        pts = ex.normalized_points(img, thr=0.25, fill=0.8, min_points=8)
    except ValueError as exc:
        # 真人检测器认不出像素画 / 插画 / 版画——和 WPC 的 pose 模式同一个限制。
        # 给能照做的下一步，别让用户对着 "0 boxes" 猜。
        raise HTTPException(
            422, f"没识别出人体：{exc}。骨架提取目前只对真人 / 写实人像有效；"
                 "游戏像素素材、插画请用上面的姿势预设手动摆，或用角色迁移的"
                 "「轮廓」参考模式。") from exc
    except Exception as exc:                       # noqa: BLE001
        raise HTTPException(500, f"姿态提取失败：{exc}") from exc

    points: List[Any] = []
    for i in range(18):
        if i in pts:
            x, y = pts[i]
            points.append([round(float(x), 4), round(float(y), 4)])
        else:
            points.append(None)
    guess = 0
    if 2 in pts and 5 in pts and abs(pts[2][0] - pts[5][0]) < 0.07:
        # 肩膀在画面上几乎重叠 → 原图接近侧面。鼻尖相对颈的方向决定朝左/朝右。
        guess = 90 if (0 in pts and 1 in pts and pts[0][0] >= pts[1][0]) else 270
    return {"points": points, "joints": len(pts), "azimuth_guess": guess}


@app.get("/pose_dirs")
def pose_dirs(limit: int = 40) -> Dict[str, Any]:
    """可以当 W4 输入的**骨架帧目录**（含 ≥2 张图的目录，按帧数降序）。

    为什么要给这个接口：W4 的入口 `tools/pose_from_image.py` 一次产出一个目录，
    用户在界面上手里只有一个路径。让他手抄绝对路径既不友好也容易错。
    这里只扫 assets 下两层（深度与产物结构对齐），不递归全盘——
    assets 里已有几千个文件，深扫会把接口拖慢。
    """
    found: List[Dict[str, Any]] = []

    def scan(d: Path, depth: int) -> None:
        if depth > 2:
            return
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        imgs = sorted(e.name for e in entries
                      if e.is_file() and Path(e.name).suffix.lower() in _POSE_EXTS)
        if len(imgs) >= 2:
            found.append({"path": str(d), "count": len(imgs),
                          "sample": imgs[:3], "name": d.name})
        for e in entries:
            if e.is_dir() and not e.name.startswith("."):
                scan(Path(e.path), depth + 1)

    scan(ASSETS, 0)
    found.sort(key=lambda x: (-x["count"], x["path"]))
    return {"count": len(found[:limit]), "items": found[:limit]}


@app.post("/generate")
def submit(job: Job) -> Dict[str, Any]:
    # 白名单**必须与 `_run` 的分发表同步**：漏一个就是 400，而错误信息里
    # 又列不出来（用户会以为功能没做）。加新工作流时两处一起改。
    allowed = ("w1", "w2", "w3", "w4", "wbg", "w5", "wpx", "wpc",
               "wss", "topixel", "wap", "wvnccs",
               # 2026-10-30 新增
               "wout", "woutfit", "wface")
    if job.workflow not in allowed:
        raise HTTPException(
            400, f"workflow 必须是 {'/'.join(allowed)}（wap 是 topixel 的旧名）")
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": job.workflow, "params": job.params,
                     "status": "queued", "created": time.time()}
    _q.put((job_id, job.workflow, job.params))
    _prune(_jobs, JOB_KEEP)
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
    """资产清单。带 project 时只返回该项目的产物（项目之间相互隔离）。

    走 SQLite 读缓存（workflows/assetdb.py）：只解析 jsonl 里**新增**的字节，
    查询按 id 倒序取一页。老实现每次都全量读取 + 逐行 json.loads + 逐条 stat，
    3 万条时要解析 26MB 并阻塞主线程。
    """
    items = index.recent(limit=limit, project=project)
    if items:
        return {"count": len(items), "items": items}
    # 索引库不可用（只读盘/权限）时退回全量解析：功能不能因为优化而消失。
    # 用 errors="replace" 读：索引里混进坏字节时，坏行被跳过，而不是让整个
    # 清单接口 500（实测出现过一次 NUL 块）。
    idx = ASSETS / "index.jsonl"
    if not idx.exists():
        return {"count": 0, "items": []}
    lines = idx.read_bytes().decode("utf-8", "replace").strip().splitlines()
    items = []
    for x in reversed(lines):
        if not x.strip():
            continue
        try:
            item = json.loads(x)
        except ValueError:
            continue
        if project and item.get("scope", "default") != project:
            continue
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
    if not _inside(root, t):
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
    if deleted:
        index.forget(deleted)
    idx = ASSETS / "index.jsonl"
    if idx.exists() and deleted:
        try:
            kept = []
            for line in idx.read_bytes().decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
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
            # 原子重写：中途崩掉不能留下半个索引文件
            assetdb.rewrite_atomic(idx, "\n".join(kept) + ("\n" if kept else ""))
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
    if not _inside(ASSETS.resolve(), target) or not target.exists():
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
