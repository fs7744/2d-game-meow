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
  GET  /library            整个用户资料库（提示词收藏 + 姿势库 + 界面偏好，均在磁盘）
  POST /library/seed       把浏览器 localStorage 里的旧存档搬进磁盘（只补不覆盖）
  POST /library/prompts    收藏一条提示词（正向+负向）
  DELETE /library/prompts/{pid}
  POST /library/poses      存一条姿势（官方 pose JSON + 预览图）
  PUT  /library/poses/{pid}  改名
  DELETE /library/poses/{pid}
  PUT  /library/prefs      写界面偏好
  GET  /pose_library       随项目发行的 VNCCS 官方姿势库清单（191 个）
  POST /pose/extract       一张图 → OpenPose-18 归一化关节点
  GET  /pose_dirs          可以当 W4 输入的骨架帧目录
  DELETE /assets          真删：删磁盘产物 + 清索引
"""
from __future__ import annotations

import io
import inspect
import random
import logging
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import unicodedata

import numpy as np
import queue
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from PIL import Image

from workflows import generate as gen
from workflows import assetdb
from workflows import characters
from workflows import charactergen
from workflows import clothesgen
from workflows import posegen
from workflows import spriteatlas
from workflows import pipeline
from workflows import emotiongen
from workflows import qwen_llm
from workflows import library
from workflows import matte_screen
from workflows import paths
from workflows import poselib
from workflows import projectio
from workflows import projectstore
from workflows.assetdb import AssetIndex
from workflows.engine import Engine

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("meow.app")
ASSETS = paths.ASSETS
UPLOAD_DIR = paths.UPLOAD_DIR
PROJECTS_DIR = paths.PROJECTS_DIR
MAX_UPLOAD = 20 * 1024 * 1024

#: 项目归档（.mproj）上限。**必须显式检查**：`/projects/import` 的 body 是
#: base64 字符串，FastAPI 默认不限长度，一个几百 MB 的归档会先整个进内存再解码。
#: 按 base64 的长度上限估算（4/3 膨胀），在**解码前**挡掉，别等解完才发现太大。
MAX_ARCHIVE = 200 * 1024 * 1024
MAX_ARCHIVE_B64 = MAX_ARCHIVE * 4 // 3 + 16
UPLOAD_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp",
                  "BMP": ".bmp", "GIF": ".gif"}

app = FastAPI(title="2D Game Asset Pipeline", version="0.1.0")
engine = Engine(paths.MODELS_DIR)
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


def _pipeline_fns():
    """pipeline 三步的默认实现。

    **抽成函数是为了测试能整体替换**：编排层要验的是顺序/进度/依赖传递，
    真跑一遍三步要几分钟还占一张卡，测试里换成假的就行。
    """
    from workflows import clothesgen as _cg
    from workflows import emotiongen as _eg
    from workflows import posegen as _pg

    return _cg.remove_clothes, _pg.run_batch, _eg.run_batch


def _run(job_id: str, workflow: str, params: Dict[str, Any]) -> None:
    job = _jobs[job_id]
    job.update(status="running", started=time.time())
    try:
        if workflow == "wposegen":
            # **多姿势批量**：N 张姿势串行出 N 张帧，最后整批替换（旧图进 V1）。
            from workflows import posegen as _posegen
            with _gpu_lock:                # 单卡不并发
                job["result"] = _posegen.run_batch(engine, params, job=job)
            job["status"] = "done"
            return
        if workflow == "watlas":
            # **图集导出**。它不碰 GPU（纯 PIL 拼接），所以**不抢 `_gpu_lock`**：
            # 不然用户在生图排着队的时候连图集都导不出来。但帧多的时候要读几百张
            # 图 + 拼一张大图，会占住几秒到几十秒，仍然走队列 —— 别让 HTTP
            # 请求线程干这个（以前是同步的，帧一多请求就超时）。
            from workflows import spriteatlas as _atlas
            job["progress"] = {"done": 0, "total": 2, "current": "收集帧清单"}
            result = _atlas.export_character_atlas(
                str(params.get("character") or ""),
                costumes=params.get("costumes") or None,
                emotions=params.get("emotions") or None,
                padding=int(params.get("padding") or 2),
                max_width=int(params.get("max_width") or 4096),
                frame_rate=float(params.get("frame_rate") or 8.0),
                job=job)
            job["result"] = {
                "character": result.character,
                "image": _asset_url(result.image_path),
                "image_path": result.image_path,
                "meta_path": result.meta_path,
                "frames": result.frames,
                "animations": result.animations,
                "size": list(result.size),
                "canvas": list(result.canvas),
                "bytes": result.bytes,
                "notes": result.notes,
            }
            job["status"] = "done"
            return
        if workflow == "wpipe":
            # **一键出全套**：编排层按官方 order 串起「内衣 → 姿势 → 表情」，
            # 每一步仍走各自的实现，这里只负责顺序与进度换算。
            from workflows import pipeline as _pipeline
            uw_fn, pose_fn, emo_fn = _pipeline_fns()
            with _gpu_lock:                # 整条流水线独占单卡
                job["result"] = _pipeline.run_pipeline(
                    engine, params, job=job,
                    underwear_fn=uw_fn, pose_fn=pose_fn, emotion_fn=emo_fn)
            job["status"] = "done"
            return
        if workflow == "wunderwear":
            # **内衣底图**：官方 `_run_remove_clothes` 那一步 —— 把自己的角色图
            # 换成白色内衣，得到一张干净底图（后续每套服装都从它出发）。
            # 20 步 / cfg 1.0，比换装短（官方 `:894-900`）。
            from workflows import clothesgen as _clothes
            with _gpu_lock:
                job["result"] = _clothes.remove_clothes(engine, params, job=job)
            job["status"] = "done"
            return
        if workflow == "wcloth":
            # **换装**：一次一张（25 步、cfg 3.0、挂 Clothes Core LoRA），
            # 出完落两处：官方的 cache 预览 + 这套服装的第一张底图。
            from workflows import clothesgen as _clothes
            with _gpu_lock:                # 单卡不并发
                job["result"] = _clothes.run_design(engine, params, job=job)
            job["status"] = "done"
            return
        if workflow == "wemo":
            # **表情批量生成**：一层循环出 N×M 张，必须自己报进度，否则前端
            # 只能干看着「执行中」。它也不走 `publish` 那段——落盘在
            # `emotiongen.run_batch` 里按源帧序号做（Sprites + Faces 两处）。
            from workflows import emotiongen
            with _gpu_lock:                # 单卡不并发
                job["result"] = emotiongen.run_batch(engine, params, job=job)
            job["status"] = "done"
            return
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
        if ignored and isinstance(job.get("result"), dict):
            job["result"].setdefault("meta", {})["params_ignored"] = ignored
        # **发布到角色库**：任务参数里带 `publish` 时，出图后顺手按官方命名
        # 规则收进 `assets/characters/<角色>/Sprites/<服装>/<表情>/`。
        # 放在这里（而不是让前端再发一个请求）是因为：出了图却没进角色库，
        # 用户看到的就是"明明跑了但素材页空着"。
        pub = params.get("publish")
        if isinstance(pub, dict) and isinstance(job.get("result"), dict):
            src = job["result"].get("raw") or job["result"].get("file") or ""
            try:
                job["result"]["published"] = characters.publish_image(
                    name=str(pub.get("character") or ""), src=str(src),
                    costume=str(pub.get("costume") or "Naked"),
                    emotion=str(pub.get("emotion") or "neutral"),
                    matte=bool(pub.get("matte")),
                )
            except Exception as pexc:                     # noqa: BLE001
                # 落盘失败**不能**把整个任务判失败：图已经生成出来了，
                # 让用户能拿到它，另外把原因写进 result。
                job["result"]["publish_error"] = str(pexc)
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        job["trace"] = traceback.format_exc(limit=6)
    finally:
        job["finished"] = time.time()
        job["elapsed"] = round(job["finished"] - job["started"], 2)


def _worker() -> None:
    """取任务并执行。

    **整个执行过程包在 try 里**：`_run` 自己已经处理了业务异常，但万一它
    自己在收尾时炸了（比如 `_jobs` 被并发改动），任务就会永远停在
    `queued` —— 前端那边节点就一直显示「排队中…」，按钮禁用，用户干等。
    这里兜住最后一道：任何漏网的异常都把任务标成 failed。
    """
    while True:
        job_id, workflow, params = _q.get()
        try:
            _run(job_id, workflow, params)
        except BaseException as exc:       # noqa: BLE001 —— worker 不能死
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="failed",
                           error=f"任务执行异常：{type(exc).__name__}: {exc}")
                job["finished"] = time.time()
        finally:
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


@app.get("/library")
def get_library() -> Dict[str, Any]:
    """整个用户资料库一次给全（提示词收藏 + 姿势库 + 界面偏好）。

    冷启动只发一次请求，省得界面为了画三个面板连发三次。
    见 workflows/library.py 的模块注释：这些都**只落在服务端磁盘**。
    """
    return library.snapshot()


class PromptReq(BaseModel):
    text: str
    negative: str = ""


class PoseReq(BaseModel):
    name: str
    pose: Any
    yaw: int = 0
    #: 预览图（dataURL，可选）。没有也能存，只是库里不显示缩略图。
    preview: str = ""


class PoseRenameReq(BaseModel):
    name: str


class LibrarySeedReq(BaseModel):
    """从 localStorage 迁移过来的旧数据。缺哪个字段就不动哪个库。"""
    prompts: Optional[List[Dict[str, Any]]] = None
    poses: Optional[List[Dict[str, Any]]] = None
    prefs: Optional[Dict[str, Any]] = None


@app.post("/library/prompts")
def library_add_prompt(req: PromptReq) -> Dict[str, Any]:
    """收藏一条提示词（正向 + 负向一起去重）。返回新的完整列表。"""
    try:
        return {"prompts": library.add_prompt(req.text, req.negative)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/library/prompts/{pid}")
def library_remove_prompt(pid: str) -> Dict[str, Any]:
    return {"prompts": library.remove_prompt(pid)}


@app.post("/library/poses")
def library_add_pose(req: PoseReq) -> Dict[str, Any]:
    """存一条姿势。**同名覆盖**（用户改完胳膊想原地更新）。"""
    try:
        return {"poses": library.add_pose(req.name, req.pose, req.yaw,
                                          req.preview)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/library/poses/{pid}")
def library_rename_pose(pid: str, req: PoseRenameReq) -> Dict[str, Any]:
    return {"poses": library.rename_pose(pid, req.name)}


@app.delete("/library/poses/{pid}")
def library_remove_pose(pid: str) -> Dict[str, Any]:
    return {"poses": library.remove_pose(pid)}


@app.put("/library/prefs")
def library_put_prefs(req: Dict[str, Any]) -> Dict[str, Any]:
    """写界面偏好（蒙版颜色/透明度等）。局部合并，不传的键不动。"""
    if not isinstance(req, dict):
        raise HTTPException(400, "偏好必须是对象")
    return {"prefs": library.save_prefs(req)}


@app.post("/library/seed")
def library_seed(req: LibrarySeedReq) -> Dict[str, Any]:
    """把浏览器里剩下的旧存档搬进磁盘，**只补不覆盖**。

    用户可能在两个浏览器里都攒过东西；谁也不该被谁覆盖。搬完由前端清掉
    localStorage —— 那里从此一个字都不留。
    """
    return library.seed_library(req.prompts, req.poses, req.prefs)


@app.get("/pose_library")
def pose_library_list(refresh: int = 0) -> Dict[str, Any]:
    """随项目发行的 VNCCS 官方姿势库清单（191 个）。

    返回的是**指路信息**（静态 URL），不给姿势本体：本体 7MB，一次性塞进
    JSON 里首屏就废了；用户点哪个才取哪个 json（每个几 KB）。

    `refresh=1` 强制重扫（往目录里丢了新姿势时用）。
    """
    items = poselib.list_presets(force=refresh == 1)
    return {"count": len(items), "items": items,
            "categories": poselib.categories()}


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
def get_job(job_id: str) -> Any:
    """查任务状态。

    **查不到时返回 404，响应体里带 `status: "failed"`**：前端轮询同时看状态码
    和状态字段，两种情况都能把节点从「执行中」放出来（以前只 404、body 里没有
    status，前端得靠"连续失败 3 次"才敢判定，节点会白转 2 秒）。
    服务重启、任务被淘汰（只留最近 200 条）都会走到这里。
    """
    job = _jobs.get(job_id)
    if job is None:
        msg = "任务不存在或已被淘汰（服务重启 / 只保留最近 200 条）"
        # `detail` 保留（= 原来 `HTTPException(404, ...)` 的 body 形状，
        # 旧前端与测试都读它），另外补 `status` 让前端能直接判失败。
        return JSONResponse(
            status_code=404,
            content={"detail": msg, "job_id": job_id, "status": "failed",
                     "error": msg})
    return job


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
    safe = projectstore.clean_scope(name) or "mask"
    scope = projectstore.safe_scope(project)
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
    stem = projectstore.clean_scope(Path(name).stem)[:40] or "image"
    # 项目隔离：上传的文件放进**本项目**的目录，别的项目拿不到。
    # **清洗后为空就别拼项目目录**：`PROJECTS_DIR / "" / "uploads"` 等于
    # `assets/projects/uploads/`，会和「所有项目」这一层混在一起。
    scope = projectstore.clean_scope(project)
    out_dir = (PROJECTS_DIR / scope / "uploads") if scope else UPLOAD_DIR
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
        raise HTTPException(404, "文件不存在")
    if download:
        # FileResponse(filename=…) 会设成 attachment；产物名已经带后缀
        return FileResponse(target, filename=target.name)
    return FileResponse(target)


# ---------------- 角色中心 ----------------
#
# 角色的磁盘存储见 workflows/characters.py（移植自 VNCCS 的 utils.py，
# 目录与 config 结构与官方一致）。这里只做 HTTP 编排。
#
# 封面 / sprite 的 URL 走 `/files/`（和产物同一条路），所以角色目录必须落在
# assets 下 —— 见 paths.CHARACTERS_DIR 的注释。


def _asset_url(path: str) -> str:
    """磁盘绝对路径 → `/files/<相对 assets 的路径>`。"""
    if not path:
        return ""
    try:
        rel = Path(path).resolve().relative_to(ASSETS.resolve())
    except (ValueError, OSError):
        return ""
    return "/files/" + "/".join(rel.parts)


class CharacterCreateReq(BaseModel):
    name: str
    info: Dict[str, Any] = Field(default_factory=dict)


class CharacterRenameReq(BaseModel):
    name: str


class CostumeReq(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)


@app.get("/characters")
def list_characters_api() -> Dict[str, Any]:
    """角色库列表（扫目录，无索引文件）。"""
    items = characters.list_character_summaries()
    for it in items:
        it["cover_url"] = _asset_url(it.pop("cover", "") or "")
    return {"count": len(items), "items": items}


@app.post("/characters")
def create_character_api(req: CharacterCreateReq) -> Dict[str, Any]:
    """建角色。已存在 → 409（不静默覆盖）。"""
    try:
        cfg = characters.create_character(req.name, req.info or None)
    except characters.CharacterConflict as exc:
        # **必须先判冲突**：CharacterConflict 是 ValueError 的子类，
        # 顺序反了的话 409 会被 400 吃掉，前端就只能拿到一句"名字不合法"
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": req.name, "config": cfg}


@app.get("/characters/{name}")
def get_character_api(name: str) -> Dict[str, Any]:
    """角色详情：config + 服装 + 已有素材。"""
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    cfg = characters.load_config(safe) or {}
    costumes = characters.list_costumes(safe)
    assets: Dict[str, Any] = {}
    for c in costumes:
        base = characters.safe_join_under(characters.character_dir(safe), "Sprites", c)
        if not os.path.isdir(base):
            continue
        for sub in sorted(os.listdir(base)):
            if not os.path.isdir(os.path.join(base, sub)):
                continue
            if characters.ARCHIVE_DIR_RE.match(sub):
                continue          # 旧版本目录不混进当前素材
            assets.setdefault(c, {})[sub] = [
                _asset_url(p) for p in characters.list_sprite_images(safe, c, sub)]
    return {
        "name": safe,
        "config": cfg,
        "costumes": costumes,
        "assets": assets,
        "cover_url": _asset_url(characters.cover_image(safe) or ""),
    }


@app.put("/characters/{name}/info")
def update_character_info(name: str, req: Dict[str, Any]) -> Dict[str, Any]:
    """改角色属性（局部合并）。"""
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    with characters.storage_lock(characters.character_dir(safe)):
        cfg = characters.load_config(safe, strict=True)
        if cfg is None:
            raise HTTPException(404, f"没有这个角色：{safe}")
        info = cfg.setdefault("character_info", {})
        info.update({k: v for k, v in (req or {}).items() if k != "name"})
        characters.save_config(safe, cfg)
    return {"name": safe, "character_info": info}


@app.put("/characters/{name}/rename")
def rename_character_api(name: str, req: CharacterRenameReq) -> Dict[str, Any]:
    try:
        cfg = characters.rename_character(name, req.name)
    except characters.CharacterConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": req.name, "config": cfg}


@app.delete("/characters/{name}")
def delete_character_api(name: str, files: int = 1) -> Dict[str, Any]:
    """删角色。`files=0` 只删 config，素材留在盘上（用户只想从列表里去掉）。"""
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    ok = characters.delete_character(safe, remove_files=files == 1)
    if not ok:
        raise HTTPException(404, f"没有这个角色：{safe}")
    return {"deleted": safe, "files": files == 1}


@app.put("/characters/{name}/costumes/{costume}")
def save_costume_api(name: str, costume: str, req: CostumeReq) -> Dict[str, Any]:
    try:
        cfg = characters.save_costume_info(name, costume, req.data or {})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": name, "costumes": cfg.get("costumes", {})}


class CharacterPromptReq(BaseModel):
    """拼装 prompt 的请求。style 省略就用角色自己存的那个。

    `enhance`：跑 Prompt Enhancement（用图像模型自己的 Qwen3-VL 改写）。
    """
    style: str = ""
    info: Dict[str, Any] = Field(default_factory=dict)
    enhance: bool = True


class CharacterGenerateReq(BaseModel):
    """出图请求。

    `workflow`：w1（文生图，造角色底图）/ w2（图生图，换装）。
    `publish`：{costume, emotion} —— 出图后按官方命名收进角色目录。
    `params`：直接透传给生成函数（steps / cfg / width / height / seed …）。
    """
    workflow: str = "w1"
    costume: str = "Naked"
    emotion: str = "neutral"
    prompt: str = ""
    negative: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    init_image: str = ""
    #: 跑 Prompt Enhancement（用图像模型自己的 Qwen3-VL）。失败自动退回未扩写。
    enhance: bool = True
    #: 收进角色目录前先抠掉纯色幕布（官方 bg_remove 阶段）。抠失败保留原图。
    matte: Optional[bool] = None


#: 本地 LLM 的错误码 → HTTP 状态（码值照官方 `character_cloner.py:224-402`）。
#: 前端按码决定话术：装环境 / 重试 / 手填。
_LLM_STATUS = {
    "DEPENDENCY_MISSING": 503, "MODEL_MISSING": 503,
    "MODEL_INVALID": 503, "MMPROJ_MISSING": 503,
    "IMAGE_MISSING": 404, "TIMEOUT": 504,
    "INVALID_RESPONSE": 502, "PARSE_ERROR": 502,
    "BAD_REQUEST": 400,
}


def _raise_llm_error(exc: "qwen_llm.LlmError") -> None:
    """把 LLM 的带码异常转成 HTTP（两个 LLM 端点共用）。"""
    raise HTTPException(
        _LLM_STATUS.get(exc.code, 500),
        {"error": exc.code, "message": exc.message, "raw": exc.raw},
    )


class ClothesDesignReq(BaseModel):
    """换装出图请求（官方 Clothes Designer 的 QI2 分支）。

    `background_color` 留空就用角色自己的 `background_color`；
    官方只认 Green/Blue/Transparent，别的值一律当 Green。
    """
    costume: str = ""
    background_color: str = ""
    #: 官方的另一种模式：照着第 2 张图的衣服穿（服装字段不参与）
    clone: bool = False
    steps: int = int(charactergen.CLOTHES_DEFAULTS["steps"])
    cfg: float = float(charactergen.CLOTHES_DEFAULTS["cfg"])
    seed: int = -1
    lora_scale: float = 1.0
    #: 出完顺带抠掉纯色幕布（绿/蓝底才有意义）
    # 默认**不抠**：QI2 自带透明底，抠图是用户的选择（见 generate._postprocess）
    matte: bool = False


class PoseGenReq(BaseModel):
    """多姿势批量生成请求（官方 `character_generator.py:2108-2223`）。

    `poses` 给几张姿势图的路径；或者给 `pose_dir`（我们 `/pose_dirs` 扫出来的
    骨架帧目录）。两条都空就报错——**不猜用户想用哪张**。
    """
    costume: str = "Naked"
    poses: List[str] = Field(default_factory=list)
    pose_dir: str = ""
    #: 背景（官方会往 prompt 追一句"改成纯 X 色"）；空 = 用角色自己的
    background_color: str = ""
    #: 背景（官方会往 prompt 追一句"改成纯 X 色"）；空 = 用角色自己的
    background_color: str = ""
    #: 一句话；也可以给"每张一句"的列表（官方 `_pose_prompts` 支持）
    prompts: Any = ""
    steps: int = int(posegen.POSE_DEFAULTS["steps"])
    cfg: float = float(posegen.POSE_DEFAULTS["cfg"])
    #: **整批共用一个种子**（官方 `inherit_pipe=True`），-1 = 现场随机一个
    seed: int = -1
    transparent: bool = False
    # 默认**不抠**：QI2 自带透明底，抠图是用户的选择（见 generate._postprocess）
    matte: bool = False
    lora: str = ""


class PipelineReq(BaseModel):
    """一键出全套的请求（编排层：内衣 → 姿势 → 表情，顺序照官方）。

    **出不出姿势看有没有给姿势来源**（`poses` 或 `pose_dir`），没有单独的开关：
    只出表情的用户不该被迫先跑一遍姿势。
    """
    costume: str = "Naked"
    #: 先换白内衣底图（官方 `remove_clothes`），得到干净底图给姿势当起点
    underwear: bool = False
    poses: List[str] = Field(default_factory=list)
    pose_dir: str = ""
    #: 姿势的追加描述（一句话，或"每张一句"的列表）
    poses_prompt: Any = ""
    emotions: List[str] = Field(default_factory=list)
    steps: int = int(pipeline.PIPELINE_DEFAULTS["steps"])
    cfg: float = float(pipeline.PIPELINE_DEFAULTS["cfg"])
    emotion_steps: int = int(pipeline.PIPELINE_DEFAULTS["emotion_steps"])
    emotion_cfg: float = float(pipeline.PIPELINE_DEFAULTS["emotion_cfg"])
    emotion_target_size: int = int(pipeline.PIPELINE_DEFAULTS["emotion_target_size"])
    background_color: str = ""
    seed: int = -1
    # 默认**不抠**：QI2 自带透明底，抠图是用户的选择（见 generate._postprocess）
    matte: bool = False
    transparent: bool = False
    #: 某一步失败后是否继续后面的（默认继续，别把整条流水线拖死）
    continue_on_error: bool = True


@app.post("/characters/{name}/pipeline")
def character_pipeline_api(name: str, req: PipelineReq) -> Dict[str, Any]:
    """一键出全套：换内衣 → 出姿势 → 出表情（顺序照官方 `:3115-3123`）。

    它**不重新实现任何一步**，只是按顺序调用三个已有产线，并把进度换算成一条。
    所以参数也是那三条产线的合集。

    两步依赖关系（容易搞错）：
      · 勾了内衣 → 姿势**从内衣那张底图出发**（`source_costume=Underwear`），
        不是从上一套遗留的服装底图；
      · 表情永远在姿势帧之上（所以姿势必须在表情之前跑完）。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        costume = characters.ensure_safe_name(req.costume or "Naked", "服装名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")

    params: Dict[str, Any] = req.model_dump()
    params["character"] = safe
    params["costume"] = costume
    try:
        steps = pipeline.plan_steps(safe, costume, params)
    except pipeline.PipelineError as exc:
        raise HTTPException(400, str(exc)) from exc

    total = float(sum(s["weight"] for s in steps))
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "job_id": job_id, "workflow": "wpipe", "params": params,
        "status": "queued", "created": time.time(),
        "progress": {"done": 0.0, "total": total,
                     "step": steps[0]["key"], "current": steps[0]["label"]},
    }
    _q.put((job_id, "wpipe", params))
    _prune(_jobs, JOB_KEEP)
    return {
        "job_id": job_id, "status": "queued", "costume": costume,
        "steps": [s["key"] for s in steps],
        "total": total,
        "labels": {s["key"]: s["label"] for s in steps},
        "position": _q.qsize(),
    }


class UnderwearReq(BaseModel):
    """内衣底图请求（官方 `remove_clothes`，`character_generator.py:878-900`）。

    **这是一套具体服装（白色内衣），不是"脱光"**：官方实现就是一条换装指令
    `Dress character: White underwear` 加 Clothes LoRA，产物是穿内衣的底图，
    用来给后续每套服装提供统一的干净起点。

    采样比换装短：**20 步 / cfg 1.0**（换装是 25 / 3.0）。
    """
    #: 这套内衣存成哪个服装名（默认 Underwear，可以覆盖）
    costume: str = "Underwear"
    steps: int = int(clothesgen.UNDERWEAR_DEFAULTS["steps"])
    cfg: float = float(clothesgen.UNDERWEAR_DEFAULTS["cfg"])
    seed: int = -1
    lora_scale: float = 1.0
    #: 出完顺带抠掉背景
    # 默认**不抠**：QI2 自带透明底，抠图是用户的选择（见 generate._postprocess）
    matte: bool = False


class ClothesWizardReq(BaseModel):
    """服装 AI 向导：一句话 → top/bottom/shoes/head/face 五个字段。"""
    description: str = ""
    timeout: float = 300.0


class CharacterWizardReq(BaseModel):
    """AI 向导：一句话 → 9 个属性字段（官方 `character_creator_v2.py:1740`）。

    **不落盘**：生成结果给用户在界面上改，确认后才写进角色。
    """
    description: str = ""
    timeout: float = 300.0


class CharacterAnalyzeReq(BaseModel):
    """「看图反推属性」的请求（官方 Cloner 旁路，要本地 LLM）。

    `image` 可以是 assets 内的路径，也可以是 `/files/...` 的展示 URL
    （前端手里两个都有，让它传最方便的那个）。
    `background_color`：透明图送模型前要压到某个背景色上（green/blue/#RRGGBB）。
    """
    image: str = ""
    background_color: str = ""
    timeout: float = 300.0


class EmotionDetectReq(BaseModel):
    """表情生成的「先看脸框」请求（**不跑模型**，纯 YOLO，CPU 上百毫秒）。

    前端用它先告出「这套服装有几张源帧、每张的脸框在哪」，用户确认后再提交批量。
    """
    costume: str = "Naked"
    #: 只看这几帧（1 基序号）；空 = 全部源帧
    poses: List[int] = Field(default_factory=list)
    threshold: float = float(emotiongen.EMOTION_DEFAULTS["bbox_threshold"])
    dilation: int = int(emotiongen.EMOTION_DEFAULTS["bbox_dilation"])
    drop_size: int = int(emotiongen.EMOTION_DEFAULTS["drop_size"])


class EmotionGenerateReq(BaseModel):
    """表情批量生成请求。

    `emotions` 传官方表情库的 `safe_name`（认不出就当自定义表情名处理）。
    `poses` 是**源帧的 1 基序号**——产物序号跟源帧对齐（`sprite_pose_0003` →
    `sprite_angry_0003`），两张对着翻才不会错位（官方 `:4541-4543`）。
    """
    costume: str = "Naked"
    emotions: List[str] = Field(default_factory=list)
    poses: List[int] = Field(default_factory=list)
    seed: int = -1
    steps: int = 25
    cfg: float = 3.0
    #: 官方默认 2048（= 2.0 MP）；实际裁切尺寸还会被 `crop_max_side` 压一档
    target_size: int = int(emotiongen.EMOTION_DEFAULTS["target_size"])
    crop_max_side: int = int(emotiongen.MAX_CROP_SIDE)
    feather: int = int(emotiongen.EMOTION_DEFAULTS["feather"])
    #: 要不要把 `Transparent background with alpha channel.` 追加进 prompt
    transparent: bool = False
    #: 贴回后再走一遍官方的纯色幕抠像（bg_remove）
    # 默认**不抠**：QI2 自带透明底，抠图是用户的选择（见 generate._postprocess）
    matte: bool = False


@app.post("/characters/wizard")
def character_wizard_api(req: CharacterWizardReq) -> Dict[str, Any]:
    """一句话造角色（官方 `character_creator_v2.py:1740-1869`）。

    纯文本，不需要视觉投影器。生成完还有一层**后端后处理**（`:588-604`）：
    race 只留物种、肤色按用户原话反推、女性没提胸围就补 `small_breasts`。
    那层不是可选项——模型的默认审美会把这些字段填错。

    **不落盘**：跟看图反推一样，先给用户看。
    """
    try:
        result = qwen_llm.wizard_character(
            req.description, timeout=float(req.timeout or 300.0))
    except qwen_llm.LlmError as exc:
        _raise_llm_error(exc)
    return {
        "fields": result.get("fields") or {},
        "seconds": result.get("seconds", 0),
        "raw": result.get("raw", ""),
        "description": result.get("description", ""),
    }


@app.post("/characters/{name}/analyze")
def character_analyze_api(name: str, req: CharacterAnalyzeReq) -> Dict[str, Any]:
    """一张角色图 → 11 个属性字段（官方 `character_cloner.py:219-412`）。

    **只返回字段，不写 config**：属性错了会一路影响后面所有出图，必须让用户
    在界面上看一眼、确认后再存（官方的 Cloner UI 也是这个顺序）。

    这个端点是**旁路**：模型没装 / 依赖缺 / 超时，一律回带 error code 的 5xx，
    前端提示"手填就行"，不能因此让角色中心不可用。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")

    raw_path = str(req.image or "").strip()
    if not raw_path:
        # 没指定就取这个角色最新的一张精灵帧（常见用法：看自己刚出的图）
        for costume in characters.list_costumes(safe):
            images = characters.list_pose_preview_files(safe, costume, None)
            if images:
                raw_path = images[-1]
                break
        if not raw_path:
            raise HTTPException(400, "这个角色还没有图可分析：先在「出图」页生成一张")
    if raw_path.startswith("/files/"):
        raw_path = raw_path[len("/files/"):]
    target = _readable_asset_path(raw_path)
    if target is None:
        raise HTTPException(400, "图片路径不在 assets 内（或不是图片文件）")

    try:
        result = qwen_llm.analyze_character_image(
            str(target), background_color=req.background_color,
            timeout=float(req.timeout or 300.0))
    except qwen_llm.LlmError as exc:
        _raise_llm_error(exc)
    except OSError as exc:
        raise HTTPException(500, f"读图失败：{exc}") from exc
    # **显式挑字段**：`result["image"]` 是磁盘路径，直接展开会把展示 URL 覆盖掉
    # （前端拿它当 <img src>，塞绝对路径就是一张裂图）
    return {
        "character": safe,
        "image": _asset_url(str(target)),
        "fields": result.get("fields") or {},
        "seconds": result.get("seconds", 0),
        "raw": result.get("raw", ""),
    }


class CharacterFromImageReq(BaseModel):
    """「从参考图一键建角色」的入参。"""
    name: str
    #: 参考图：assets 里的相对路径，或 `/files/...` 的展示 URL
    image: str = ""
    #: 收编到哪套服装（默认裸装）
    costume: str = "Naked"
    #: 收编成哪个表情的帧（默认平静）
    emotion: str = "neutral"
    #: 幕布色（抠图按它取样）
    background_color: str = ""
    #: 收编时先抠掉纯色背景（官方 bg_remove）
    # 默认**不抠**：QI2 自带透明底，抠图是用户的选择（见 generate._postprocess）
    matte: bool = False
    timeout: float = 300.0


@app.post("/characters/from-image")
def character_from_image_api(req: CharacterFromImageReq) -> Dict[str, Any]:
    """一张参考图 → 一个角色（官方 `character_cloner.py` 的「一键建角色」）。

    **为什么要有这个**：反推属性（`/characters/{name}/analyze`）要求角色**已经存在**
    （不存在就 404）。所以原来的流程是「手动新建一个空角色 → 反推 → 逐条粘进去 →
    再去出图页生成」——四步，中间任何一步关掉浏览器，前面的功夫就白费。

    这里把三步合成一步：

    1. **先反推**（本地 Qwen3.5-4B）—— 失败就**什么都不留**，
       不会在列表里留下一个空壳角色
    2. **再建角色**，把反推出来的属性直接写进 config
    3. **收编参考图**成第一张精灵帧（`publish_image`，顺带抠背景）

    属性反推错了也不要紧：建完照样能改（`PUT /characters/{name}/info`），改完再出图。
    重跑反推的代价是重新建一个角色，所以名字先想好。

    **没有官方那么全**：官方的 `VNCCS_CharacterCloneGenerator` 还会跑一整条
    支线（`original_pose_generation → remove_clothes → naked_pose_generation`），
    一次出原服装 + 裸装两套精灵帧。那需要生成模型跑三次，
    我们把「出图」留给「一键出全套」那一步（`/characters/{name}/pipeline`）。
    """
    try:
        safe = characters.ensure_safe_name(req.name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if characters.character_exists(safe):
        raise HTTPException(409, f"角色已存在：{safe}")

    raw_path = str(req.image or "").strip()
    if not raw_path:
        raise HTTPException(400, "没给参考图")
    if raw_path.startswith("/files/"):
        raw_path = raw_path[len("/files/"):]
    target = _readable_asset_path(raw_path)
    if target is None:
        raise HTTPException(400, "图片路径不在 assets 内（或不是图片文件）")

    # 1. 先反推 —— 失败就什么都不留
    try:
        result = qwen_llm.analyze_character_image(
            str(target), background_color=req.background_color,
            timeout=float(req.timeout or 300.0))
    except qwen_llm.LlmError as exc:
        _raise_llm_error(exc)
    except OSError as exc:
        raise HTTPException(500, f"读图失败：{exc}") from exc

    fields = result.get("fields") or {}
    # `filter_traits` 已经有白名单（防注入），这里再去掉 `name`：
    # 它会盖掉角色名（`_default_character_info` 里 name 是角色名）。
    info = {k: v for k, v in fields.items() if k != "name"}

    # 2. 建角色，属性直接写进 config
    try:
        characters.create_character(safe, info=info)
    except characters.CharacterConflict as exc:
        raise HTTPException(409, str(exc)) from exc

    # 3. 收编参考图成第一张精灵帧（publish_image 自带抠像，抠完另存 _matte.png）
    try:
        pub = characters.publish_image(
            safe, str(target),
            costume=str(req.costume or "Naked"),
            emotion=str(req.emotion or "neutral"),
            matte=bool(req.matte))
    except (ValueError, OSError) as exc:
        # 角色已经建好了（属性在里头），只是图没收编成功 ——
        # 别把整个角色删掉，用户还能手动出图。
        raise HTTPException(500, f"角色已建好，但参考图收编失败：{exc}") from exc

    return {
        "character": safe,
        "fields": fields,
        "image": _asset_url(str(target)),
        "sprite": _asset_url(str(pub.get("path") or "")),
        "matte": _asset_url(str(pub.get("matte") or "")) if pub.get("matte") else "",
        "seconds": result.get("seconds", 0),
        "raw": result.get("raw", ""),
    }


@app.post("/characters/{name}/prompt")
def character_prompt_api(name: str, req: CharacterPromptReq) -> Dict[str, Any]:
    """把 character_info + 风格拼成官方结构的 prompt。

    前端拿它预览（也能改完再发回去出图）——**这一步不跑模型**，所以改属性时
    能立刻看到会喂给模型什么。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    cfg = characters.load_config(safe) or {}
    info = dict(cfg.get("character_info") or {})
    info.update(req.info or {})
    style = req.style or str(info.get("style") or "")
    # PE 用图像模型自己的 Qwen3-VL 文本编码器，**不新增权重**；
    # 但它要常驻 GPU 跑自回归解码——显存紧或没装好时静默退回未扩写版。
    pipe = None
    if req.enhance:
        try:
            pipe = engine.txt2img_pipe()
        except Exception as exc:                   # noqa: BLE001
            log.warning("PE 拿不到管线，退回未扩写：%s", exc)
            pipe = None
    built = charactergen.enhance_character_prompt(
        pipe, info, style, enhance=req.enhance)
    return {
        "name": safe,
        "style": style,
        "prompt": built["prompt"],
        "raw_prompt": built["raw"],
        "enhanced": built["applied"] == "1",
        "negative": str(info.get("negative_prompt") or ""),
        "resolution": charactergen.generation_resolution(
            int(charactergen.QI2_DEFAULTS["target_size"])),
        "defaults": dict(charactergen.QI2_DEFAULTS),
    }


@app.post("/characters/{name}/generate")
def character_generate_api(name: str, req: CharacterGenerateReq) -> Dict[str, Any]:
    """给角色出一张图，出完**自动收进角色目录**。

    复用画布那条 `/generate` 队列（单卡串行、可轮询），只是多带一个
    `publish` 参数让 worker 收编产物。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        costume = characters.ensure_safe_name(req.costume or "Naked", "服装名")
        emotion = characters.ensure_safe_name(req.emotion or "neutral", "表情名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    allowed = ("w1", "w2")
    if req.workflow not in allowed:
        raise HTTPException(400, f"角色出图只支持 {'/'.join(allowed)}")

    cfg = characters.load_config(safe) or {}
    info = cfg.get("character_info") or {}
    if req.prompt.strip():
        prompt = req.prompt.strip()
    else:
        pipe = None
        if req.enhance:
            try:
                pipe = engine.txt2img_pipe()
            except Exception as exc:               # noqa: BLE001
                log.warning("PE 拿不到管线，退回未扩写：%s", exc)
                pipe = None
        prompt = charactergen.enhance_character_prompt(
            pipe, info, str(info.get("style") or ""),
            enhance=req.enhance)["prompt"]
    negative = req.negative.strip() or str(info.get("negative_prompt") or "")
    w, h = charactergen.generation_resolution(
        int((req.params or {}).get("target_size") or 0)
        or charactergen.QI2_DEFAULTS["target_size"])

    params: Dict[str, Any] = {
        "prompt": prompt,
        "negative": negative,
        "width": w, "height": h,
        "steps": charactergen.QI2_DEFAULTS["steps"],
        "cfg": charactergen.QI2_DEFAULTS["cfg"],
        "transparent": str(info.get("background_color") or "").lower()
        == "transparent",
        "publish": {"character": safe, "costume": costume, "emotion": emotion,
                    # 没给就按幕布色决定：绿/蓝/品红自动抠，白/灰不动
                    # （白底会把白毛白甲一起判成背景，这条我们踩过）
                    "matte": matte_screen.should_auto_matte(
                        str(info.get("background_color") or ""), req.matte)},
    }
    if req.workflow == "w2":
        # 换装：底图取当前这套服装里已有的最新一张（没有就退回 Naked）
        refs = characters.list_sprite_images(safe, costume, "neutral") \
            or characters.list_sprite_images(safe, "Naked", "neutral")
        if not refs:
            raise HTTPException(400, "换装需要一张底图：先在「属性」页生成一张")
        params["image_path"] = req.init_image.strip() or refs[-1]
    params.update({k: v for k, v in (req.params or {}).items()
                   if k not in ("prompt", "negative", "publish")})

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": req.workflow, "params": params,
                     "status": "queued", "created": time.time()}
    _q.put((job_id, req.workflow, params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "position": _q.qsize(),
            "prompt": prompt, "resolution": [w, h]}


@app.post("/characters/{name}/poses/generate")
def pose_generate_api(name: str, req: PoseGenReq) -> Dict[str, Any]:
    """多姿势批量生成（官方 `character_generator.py:2108-2223`）。

    一张姿势出一张帧，**整批共用一个种子**（官方 `inherit_pipe=True`），
    出完走 `staged_image_batch` **整批替换**这套服装的 Neutral 帧（旧图进 V1）。

    ⚠️ **这一步会替换掉现有的 Neutral 帧**（旧的在 `V1/` 里能找回）。
    跟「表情」那条（逐张追加）不是一回事：那个是往同一组帧上叠表情。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        costume = characters.ensure_safe_name(req.costume or "Naked", "服装名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")

    poses = posegen._resolve_pose_paths(req.poses or [], req.pose_dir or "")
    if not poses:
        raise HTTPException(
            400, "没有可用的姿势图：给几张姿势图，或指定一个骨架帧目录")

    params: Dict[str, Any] = {
        "character": safe, "costume": costume,
        "poses": poses, "prompts": req.prompts,
        "background_color": req.background_color,
        "background_color": req.background_color,
        "steps": int(req.steps), "cfg": float(req.cfg), "seed": int(req.seed),
        "transparent": bool(req.transparent), "matte": bool(req.matte),
        "lora": req.lora,
    }
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": "wposegen", "params": params,
                     "status": "queued", "created": time.time(),
                     "progress": {"done": 0, "total": len(poses), "current": ""}}
    _q.put((job_id, "wposegen", params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "costume": costume,
            "total": len(poses), "position": _q.qsize(),
            "defaults": dict(posegen.POSE_DEFAULTS)}


@app.post("/characters/{name}/clothes/wizard")
def clothes_wizard_api(name: str, req: ClothesWizardReq) -> Dict[str, Any]:
    """一句话服装描述 → 五个字段（官方 `clothes_designer.py:885-994`）。

    **不落盘**：结果填进服装表单，用户改完点「保存服装」才写进 config。
    """
    try:
        characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(name):
        raise HTTPException(404, f"没有这个角色：{name}")
    try:
        result = qwen_llm.clothes_wizard(
            req.description, timeout=float(req.timeout or 300.0))
    except qwen_llm.LlmError as exc:
        _raise_llm_error(exc)
    return {
        "fields": result.get("fields") or {},
        "seconds": result.get("seconds", 0),
        "raw": result.get("raw", ""),
        "description": result.get("description", ""),
    }


class AtlasReq(BaseModel):
    """精灵图集导出请求（**不是官方功能**，是我们补的工程环节）。

    官方只负责把帧出到 `Sprites/<服装>/<表情>/`；打成图集是接引擎的那一步。
    """
    costumes: List[str] = Field(default_factory=list, description="空 = 全部")
    emotions: List[str] = Field(default_factory=list, description="空 = 全部")
    #: 帧间距。**必须留缝**：贴边的话引擎线性过滤会把邻居像素吸进来
    padding: int = 2
    max_width: int = 4096
    frame_rate: float = 8.0


@app.post("/characters/{name}/atlas")
def character_atlas_export(name: str, req: AtlasReq) -> Dict[str, Any]:
    """把这个角色的精灵帧打成 `atlas.png` + `atlas.json`（**走队列**）。

    输出在 `<角色目录>/Atlas/`。JSON 是 TexturePacker Hash（Aseprite 能直接导入），
    另外多一层 `animations`：把「服装/表情」映射成帧序列，引擎侧不用再自己分组。

    **对齐规则**：所有帧统一到「最大宽高」的画布，**水平居中 + 底边对齐**。
    不同姿势的包围盒不一样，按左上角拼会让角色在动画里上下抖。

    **为什么改成异步**：以前是同步的，几十帧要 1~2 秒还行，几百帧就几秒到几十秒，
    HTTP 请求会超时、界面也没有任何进度。现在提交后拿 `job_id` 去轮询。

    **没帧时立刻报错**（不入队）：让用户等一个必然失败的任务是最糟的体验。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # 先确认有帧，别入队一个注定失败的任务
    try:
        frames = spriteatlas.collect_frames(
            safe, req.costumes or None, req.emotions or None)
    except spriteatlas.AtlasError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"读角色目录失败：{exc}") from exc

    params: Dict[str, Any] = {
        "character": safe,
        "costumes": list(req.costumes or []),
        "emotions": list(req.emotions or []),
        "padding": max(0, int(req.padding)),
        "max_width": max(64, int(req.max_width)),
        "frame_rate": float(req.frame_rate or 8.0),
    }
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "job_id": job_id, "workflow": "watlas", "params": params,
        "status": "queued", "created": time.time(),
        "progress": {"done": 0, "total": 2, "current": "排队中"},
    }
    _q.put((job_id, "watlas", params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "frames": len(frames),
            "position": _q.qsize()}


@app.get("/characters/{name}/atlas")
def character_atlas_info(name: str) -> Dict[str, Any]:
    """图集导出了没有 / 多大 / 多少帧。

    **不返回整份 JSON**：`frames` 可能有上千条，前端只用得上统计。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    folder = Path(characters.character_dir(safe)) / "Atlas"
    image = folder / "atlas.png"
    meta_file = folder / "atlas.json"
    if not image.is_file() or not meta_file.is_file():
        return {"character": safe, "exists": False}
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"character": safe, "exists": False, "broken": True}
    frames = meta.get("frames") or {}
    animations = meta.get("animations") or {}
    info = meta.get("meta") or {}
    return {
        "character": safe,
        "exists": True,
        "image": _asset_url(str(image)),
        "frames": len(frames),
        "animations": len(animations),
        "size": [info.get("size", {}).get("w", 0), info.get("size", {}).get("h", 0)],
        "canvas": [info.get("canvas", {}).get("w", 0), info.get("canvas", {}).get("h", 0)],
        "bytes": image.stat().st_size,
        "modified": image.stat().st_mtime,
    }


@app.post("/characters/{name}/clothes/underwear")
def clothes_underwear_api(name: str, req: UnderwearReq) -> Dict[str, Any]:
    """角色 → 白色内衣底图（官方 `_run_remove_clothes:2282-2340`）。

    这是官方的**预处理一步**：给自己的角色图换成白色内衣，得到一张干净底图，
    后面每套服装都从它出发 —— 这样新衣服不会带着上一套的袖子/配饰残留。

    产物落到 `Sprites/<服装>/neutral/sprite_pose_0001.png`（默认服装名
    `Underwear`），之后出姿势、出表情都从它走。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        costume = characters.ensure_safe_name(
            req.costume or clothesgen.UNDERWEAR_DEFAULTS["costume"], "服装名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    if not clothesgen.base_sprite(safe):
        raise HTTPException(
            400,
            f"「{safe}」还没有裸装/原始底图：先选一张角色图当起点，"
            "在「出图」页生成一张（Naked 或 Original 服装）")

    params: Dict[str, Any] = {
        "character": safe, "costume": costume,
        "steps": int(req.steps), "cfg": float(req.cfg), "seed": int(req.seed),
        "lora_scale": float(req.lora_scale), "matte": bool(req.matte),
    }
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": "wunderwear", "params": params,
                     "status": "queued", "created": time.time(),
                     "progress": {"done": 0, "total": 1, "current": ""}}
    _q.put((job_id, "wunderwear", params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "costume": costume,
            "position": _q.qsize(),
            "prompt": clothesgen.UNDERWEAR_PROMPT,
            "defaults": dict(clothesgen.UNDERWEAR_DEFAULTS)}


@app.post("/characters/{name}/clothes/design")
def clothes_design_api(name: str, req: ClothesDesignReq) -> Dict[str, Any]:
    """按服装描述生成一张换装底图（官方 Clothes Designer 的 QI2 分支）。

    **它跟「出图」是两回事**：这里挂 Clothes Core LoRA、走换装专用 prompt
    （`Dress the character:` + 五个字段 + 背景段），产物落进
    `Sprites/<服装>/neutral/sprite_pose_0001.png`——那是这套服装的第一张图，
    后面出姿势/表情都从它出发。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        costume = characters.ensure_safe_name(req.costume or "", "服装名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    if not clothesgen.base_sprite(safe):
        raise HTTPException(
            400,
            f"「{safe}」还没有裸装/原始底图：换装是从底图改的，"
            "先在「出图」页生成一张（Naked 或 Original 服装）")

    params: Dict[str, Any] = {
        "character": safe, "costume": costume,
        "background_color": req.background_color,
        "clone": bool(req.clone),
        "steps": int(req.steps), "cfg": float(req.cfg), "seed": int(req.seed),
        "lora_scale": float(req.lora_scale),
        "matte": bool(req.matte),
    }
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": "wcloth", "params": params,
                     "status": "queued", "created": time.time(),
                     "progress": {"done": 0, "total": 1, "current": ""}}
    _q.put((job_id, "wcloth", params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "costume": costume,
            "position": _q.qsize(),
            "defaults": dict(charactergen.CLOTHES_DEFAULTS)}


@app.post("/characters/{name}/emotions/detect")
def character_emotions_detect(name: str, req: EmotionDetectReq) -> Dict[str, Any]:
    """列出这套服装的**源帧**，以及每帧检出的脸框（官方 `:1533`）。

    不带模型出图，只跑 YOLO。前端拿它先告诉用户「会在这几张上改脸、脸框多大」——
    没有源帧时要能立刻说清「先去出图页生成一张底图」，而不是提交后才报错。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        sources = characters.source_sprite_paths(safe, req.costume or "Naked")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    wanted = set(int(v) for v in req.poses) if req.poses else None
    items: List[Dict[str, Any]] = []
    for index, path in enumerate(sources, start=1):
        if wanted is not None and index not in wanted:
            continue
        box = None
        size = [0, 0]
        error = ""
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGBA")
            size = [image.width, image.height]
            box = emotiongen.detect_face(
                image, threshold=req.threshold, dilation=req.dilation,
                drop_size=req.drop_size)
        except emotiongen.FaceModelMissing as exc:
            raise HTTPException(503, str(exc)) from exc
        except emotiongen.EmotionError as exc:
            error = str(exc)
        items.append({
            "index": index, "path": path, "url": _asset_url(path),
            "size": size, "box": list(box) if box else None,
            "detected": box is not None, "error": error,
        })
    return {
        "character": safe, "costume": req.costume or "Naked",
        "count": len(items), "items": items,
        "defaults": dict(emotiongen.EMOTION_DEFAULTS),
        "max_crop_side": emotiongen.MAX_CROP_SIDE,
    }


@app.post("/characters/{name}/emotions/generate")
def character_emotions_generate(name: str, req: EmotionGenerateReq) -> Dict[str, Any]:
    """表情批量生成（官方 Emotion Studio）：N 张源帧 × M 个表情，只改脸。

    走同一个单卡队列（一次出很多张，不能跟画布上的任务抢显存），
    所以立刻返回 `job_id`，前端轮询 `/jobs/<id>` 看 `progress`。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        costume = characters.ensure_safe_name(req.costume or "Naked", "服装名")
        emotions = emotiongen.resolve_emotions(req.emotions)
    except emotiongen.EmotionError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    sources = characters.source_sprite_paths(safe, costume)
    if not sources:
        raise HTTPException(
            400,
            f"「{safe}」的「{costume}」没有源帧：表情是从 Neutral 姿势帧上改脸的，"
            "先在「出图」页生成一张底图")

    params: Dict[str, Any] = {
        "character": safe, "costume": costume,
        "emotions": [e["safe_name"] for e in emotions],
        "poses": [int(v) for v in req.poses],
        "seed": int(req.seed), "steps": int(req.steps), "cfg": float(req.cfg),
        "target_size": int(req.target_size),
        "crop_max_side": int(req.crop_max_side),
        "feather": int(req.feather),
        "transparent": bool(req.transparent),
        "matte": bool(req.matte),
    }
    total = len(params["poses"] or sources) * len(params["emotions"])
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": "wemo", "params": params,
                     "status": "queued", "created": time.time(),
                     "progress": {"done": 0, "total": total, "current": ""}}
    _q.put((job_id, "wemo", params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "total": total,
            "sources": len(sources), "emotions": params["emotions"],
            "position": _q.qsize()}


@app.get("/characters/{name}/poses/meta")
def character_poses_meta(name: str, costume: str = "", emotion: str = "") -> Dict[str, Any]:
    """一个可翻看的姿势序列有多少张（官方 `character_creator_v2.py:1701`）。

    **先问数量再取图**：一次性把 N 张全塞进 JSON 的话，精灵帧一多（几十上百张）
    首屏就卡住。官方也是这个顺序，前端拿着 count 再按 index 单张取。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        files = characters.list_pose_preview_files(safe, costume or None, emotion or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"count": len(files), "files": [_asset_url(p) for p in files]}


@app.get("/characters/{name}/poses")
def character_poses(name: str, index: Optional[int] = None, costume: str = "",
                    emotion: str = "") -> Any:
    """取序列里的某一帧（官方 `character_creator_v2.py:1666`）。

    `index` 省略就是**随机一张**——官方在没人指定时用它做封面式的随机展示。
    越界按 `index % count` 取模，所以前端翻到头会绕回第一张。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
        files = characters.list_pose_preview_files(safe, costume or None, emotion or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not files:
        raise HTTPException(404, "这个角色还没有精灵帧素材（Sprites/ 是空的）")
    count = len(files)
    pick = random.randrange(count) if index is None else index % count
    return FileResponse(files[pick], headers={
        "X-VNCCS-Preview-Index": str(pick),
        "X-VNCCS-Preview-Count": str(count),
        "Cache-Control": "no-store",
    })


@app.delete("/characters/{name}/costumes/{costume}")
def delete_costume_api(name: str, costume: str) -> Dict[str, Any]:
    try:
        characters.delete_costume(name, costume)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"deleted": costume, "costumes": characters.list_costumes(name)}


# ---------------- 项目导入 / 导出 / 模板 ----------------
#
# 项目（画布 + 节点 + 参数）由**前端**持有（localStorage），素材在服务端 assets/ 下。
# 这两个端点把两边拼成一个单文件归档 `.mproj`，让项目能拷走、能换机、能当模板复用。


class ProjectArchiveReq(BaseModel):
    """导出/存模板的请求体。

    `meta`  —— 项目元信息（name / created / viewport 等）
    `payload` —— 主体数据（nodes 等）
    """

    meta: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)


class ProjectImportReq(BaseModel):
    """导入请求：归档的原始字节 + 新项目 id。"""

    scope: str = Field(..., description="新项目的 id（前端生成，assets/projects/<id>/）")
    name: str = Field("", description="导入后的项目名；空则用归档里的名字")
    data: str = Field("", description=".mproj 文件内容（base64）")


class ProjectDocReq(BaseModel):
    """项目定义的保存请求体（前端自动保存用）。"""

    id: str = Field("", description="项目 id")
    name: str = Field("", description="项目名")
    created: str = Field("", description="创建时间（ISO）")
    updated: str = Field("", description="更新时间（ISO）")
    nodes: list = Field(default_factory=list, description="节点列表")
    viewport: Dict[str, Any] = Field(default_factory=dict)
    # **乐观锁**：客户端读到的那份项目的 updated 时间戳。
    # 与磁盘上的不一致 → 409（被别人改过了），而不是覆盖。
    expect_updated: Optional[str] = Field(
        None, description="客户端读到的 updated；用于检出并发修改")
    # **新建标记**：为 true 时若 id 已被占用 → 409，不覆盖。
    create: bool = Field(False, description="true=新建（id 已存在则报错），false=更新")


def _persist_imported(scope: str, meta: Dict[str, Any],
                       payload: Dict[str, Any]) -> None:
    """导入 / 模板实例化后，把项目定义**落盘**。

    不落盘的话，导入出来的项目只在浏览器内存里——刷新一下就没了，
    用户会以为「导入没成功」。
    """
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return
    try:
        projectstore.save_project(scope, {
            "id": scope,
            "name": str(meta.get("name") or scope),
            "created": str(meta.get("created") or ""),
            "updated": str(meta.get("updated") or time.strftime("%Y-%m-%dT%H:%M:%S")),
            "nodes": nodes,
            "viewport": payload.get("viewport") or {"x": 120, "y": 90, "z": 1},
        })
    except (OSError, ValueError):
        # 落盘失败不该让导入整体失败：素材已经拷好了，
        # 用户至少能在这次会话里用（前端仍会再保存一次）。
        pass


# ---------------- 项目落盘存储 ----------------
#
# 项目定义落在 `assets/projects/<id>/project.json`（服务端本地磁盘），
# 不是浏览器 localStorage。localStorage 降级为「即时加载的缓存」：
# 前端先用缓存把画布画出来，再拉服务端版本覆盖。


@app.get("/projects")
def list_projects() -> Dict[str, Any]:
    """磁盘上所有项目（摘要，不带节点）。

    前端启动时拉这个列表来决定侧栏显示什么。
    """
    items = projectstore.list_projects()
    ws = projectstore.load_workspace()
    return {"count": len(items), "items": items, "current": ws.get("current", "")}


@app.get("/projects/{scope}")
def get_project(scope: str) -> Dict[str, Any]:
    """读一个项目的完整定义（节点 + 视口）。"""
    doc = projectstore.load_project(scope)
    if doc is None:
        raise HTTPException(404, "项目不存在")
    return doc


@app.put("/projects/{scope}")
def put_project(scope: str, req: ProjectDocReq) -> Dict[str, Any]:
    """保存一个项目的定义（原子写）。

    前端在改动画布后防抖调用。**这是幂等的**——同一个项目反复保存不会
    产生垃圾文件（同一个 path 覆盖）。

    **两种冲突在这里被挡住**（返回 409，不覆盖别人的数据）：
      · `create=true` 但 id 已被占用 → 409（两个标签页撞了 id）；
      · `expect_updated` 与磁盘上的不符 → 409（别人改过这个项目了）。
    """
    try:
        info = projectstore.save_project(
            scope, req.model_dump(),
            expect_updated=req.expect_updated, create=req.create)
    except projectstore.ProjectConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"保存项目失败：{exc}") from exc
    return info


@app.post("/projects/allocate")
def allocate_project() -> Dict[str, Any]:
    """向服务端要一个**保证没用过**的项目 id。

    **为什么 id 要服务端分配**：原先由浏览器用「时间戳 + 序号」生成，
    两个标签页在同一毫秒建项目会拿到**完全相同**的 id（实测 `pmdr5q6ns0`），
    后保存的直接覆盖先保存的——用户的画布无声消失，还以为保存成功了。

    这里用「随机后缀 + 存在性检查」分配，撞了就换，绝不返回已存在的 id。
    """
    return {"id": projectstore.allocate_project_id()}


@app.delete("/projects/{scope}")
def delete_project(scope: str, files: int = 1) -> Dict[str, Any]:
    """删项目。

    `files=1`（默认）连整个项目目录一起删（产物、上传、蒙版）；
    `files=0` 只删 project.json，产物留在盘上。
    """
    ok = projectstore.delete_project(scope, remove_files=bool(files))
    if not ok:
        raise HTTPException(404, "项目不存在")
    return {"deleted": projectstore.safe_scope(scope), "files": bool(files)}


def _reveal_in_file_manager(path: Path) -> None:
    """在系统文件管理器里打开一个目录。

    **只适用于本地单机**：服务默认只监听 127.0.0.1（见 `__main__`），
    这个入口是给「打开项目文件夹」用的便捷操作，不是远程文件管理。

    三个细节：
    - **列表传参、不经过 shell**：路径本身来自 `safe_scope`（只留 ASCII
      字母数字与 `-_`），没有注入面，但仍然不该拼字符串给 shell。
    - **Windows 用 explorer 且不检查返回码**：explorer.exe 即使成功打开也
      返回 1，检查返回值会把成功当失败。
    - 用 Popen 不等待：文件管理器是常驻进程，等它退出会把请求挂住。
    """
    target = str(path)
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", target])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        # 无桌面环境（纯命令行服务器）时 xdg-open 不存在，会抛 FileNotFoundError
        subprocess.Popen(["xdg-open", target])


@app.post("/projects/{scope}/reveal")
def reveal_project(scope: str) -> Dict[str, Any]:
    """在文件管理器里打开这个项目的目录（产物、上传、蒙版都在里面）。

    **为什么需要**：用户想知道图存哪了、想直接把整个项目拷走、想拿 raw 图去
    别的工具里二次处理——都得先能找到那个目录。项目 id 是服务端分配的
    （`p` + 日期 + 随机串），让人自己去 `assets/projects/` 下翻不现实。
    """
    d = projectstore.project_dir(scope)
    if not d.is_dir():
        raise HTTPException(404, "项目目录不存在（这个项目还没生成过产物）")
    try:
        _reveal_in_file_manager(d)
    except OSError as exc:
        raise HTTPException(500, f"打不开文件管理器：{exc}") from exc
    return {"opened": str(d)}


@app.put("/workspace")
def put_workspace(req: Dict[str, Any]) -> Dict[str, Any]:
    """记「当前打开的是哪个项目」——重开浏览器时回到同一个画布。"""
    projectstore.save_workspace(req)
    return {"ok": True}


@app.post("/projects/export")
def export_project(req: ProjectArchiveReq) -> Response:
    """把项目打包成 `.mproj` 下载。

    归档里含 `project.json`（节点，路径已替换成占位符）+ 它引用到的所有素材。
    路径占位符在**导入**时才换成新位置，所以导出不会改动磁盘上的任何东西。
    """
    try:
        data = projectio.pack(req.meta, req.payload)
    except Exception as exc:                       # noqa: BLE001
        raise HTTPException(400, f"导出失败：{exc}") from exc
    name = str(req.meta.get("name") or "project").strip() or "project"
    # **HTTP 头只允许 latin-1**：直接塞中文会让 starlette 编码头时抛
    # UnicodeEncodeError（整个导出 500，踩过）。
    #
    # 但**不该因此把中文名丢掉** —— 老浏览器的 `filename=` 给一个 ASCII 兜底，
    # 同时用 RFC 5987 的 `filename*=UTF-8''<百分号编码>` 给真名。
    # 现代浏览器优先用后者，下载下来就是「白猫骑士.mproj」。
    ascii_name = "".join(ch for ch in name if ch.isascii()
                         and (ch.isalnum() or ch in "-_"))[:48] or "project"
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition":
                f'attachment; filename="{ascii_name}.mproj"; '
                f"filename*=UTF-8''{quote(name[:64] + '.mproj')}",
        },
    )


@app.post("/projects/import")
def import_project(req: ProjectImportReq) -> Dict[str, Any]:
    """导入一个 `.mproj` → 一个**新项目**（素材落在新项目目录，路径重写）。

    绝不覆盖已有项目：新项目的 id 由前端传进来（前端才是项目的持有者），
    素材落到 `assets/projects/<scope>/imported/`。
    """
    import base64

    if len(req.data) > MAX_ARCHIVE_B64:
        raise HTTPException(
            413, f"归档太大（上限 {MAX_ARCHIVE // 2**20}MB，"
                 f"收到约 {len(req.data) * 3 // 4 // 2**20}MB）")
    try:
        raw = base64.b64decode(req.data, validate=True)
    except Exception as exc:                       # noqa: BLE001
        raise HTTPException(400, f"归档内容不是合法的 base64：{exc}") from exc
    if not raw:
        raise HTTPException(400, "归档内容为空")
    try:
        result = projectio.unpack(raw, req.scope)
    except projectio.ArchiveError as exc:
        raise HTTPException(400, str(exc)) from exc
    meta = dict(result["meta"])
    payload = dict(result["payload"])
    if req.name.strip():
        meta["name"] = req.name.strip()
    _persist_imported(req.scope, meta, payload)
    return {"meta": meta, "payload": payload,
            "imported_files": result["imported_files"]}


@app.get("/templates")
def list_templates() -> Dict[str, Any]:
    """可用模板列表。"""
    items = projectio.list_templates()
    return {"count": len(items), "items": items}


@app.post("/templates")
def save_template(req: ProjectArchiveReq) -> Dict[str, Any]:
    """把当前项目存成模板（`.mproj` 落在 `templates/`）。"""
    meta = dict(req.meta)
    name = str(meta.get("name") or "模板").strip() or "模板"
    # **模板 id 保留中文**：以前按 ASCII 过滤，中文项目名的模板全变成
    # `template` / `template_12345`，文件列表里根本看不出哪个是哪个。
    # 只去掉文件系统不接受的那几个字符，其余照原名（中文文件名没问题）。
    tid = "".join(ch for ch in name
                  if ch not in characters._FORBIDDEN_CHARS
                  and not unicodedata.category(ch).startswith("C"))[:48]
    tid = tid.strip().strip(".").strip()
    if not tid:
        tid = "template"
    # 同名模板不覆盖：加时间后缀
    if projectio._template_file(tid).exists():
        tid = f"{tid}_{int(time.time()) % 100000}"
    try:
        info = projectio.save_template(tid, meta, req.payload)
    except Exception as exc:                       # noqa: BLE001
        raise HTTPException(400, f"保存模板失败：{exc}") from exc
    return {"id": info["id"], "name": name, "bytes": info["bytes"]}


@app.post("/templates/{tid}/instantiate")
def instantiate_template(tid: str, req: ProjectImportReq) -> Dict[str, Any]:
    """从模板新建一个项目（素材复制到新项目目录，路径重写）。"""
    try:
        result = projectio.instantiate_template(tid, req.scope)
    except projectio.ArchiveError as exc:
        raise HTTPException(404, str(exc)) from exc
    meta = dict(result["meta"])
    payload = dict(result["payload"])
    if req.name.strip():
        meta["name"] = req.name.strip()
    _persist_imported(req.scope, meta, payload)
    return {"meta": meta, "payload": payload,
            "imported_files": result["imported_files"]}


@app.delete("/templates/{tid}")
def delete_template(tid: str) -> Dict[str, Any]:
    """删模板。"""
    ok = projectio.delete_template(tid)
    if not ok:
        raise HTTPException(404, "模板不存在")
    return {"deleted": tid}


# **.mjs 必须是 text/javascript**：浏览器对 module script 强制 MIME 校验，
# 认错类型就直接拒绝加载（本机 Python 的 mimetypes 不认识 .mjs，给的是
# text/plain → 页面里的模块一个都不跑，只能空等到超时）。
# 官方 VNCCS PoseStudio 里有一批 .mjs，这里一次性登记好。
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")

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
