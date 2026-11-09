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
  GET/PUT /settings        全局设置（底模 / 抠图方式 / 生效 LoRA / 快速模式，SQLite）
  GET  /loras              可用 LoRA；POST /loras/apply、/loras/unload
  POST /loras/inspect      导入第一步：暂存 + 只读格式检测（是否需要转换）
  POST /loras/commit       导入第二步：用户确认后落盘（必要时先转格式）
  POST /loras/convert      把已在 models/lora/ 里的格式不对的文件就地转换
  DELETE /loras/{name}     删除已导入的 LoRA
  GET  /projects/{id}/pose_sets            项目里的姿势集（动作抽帧存下来的）
  POST /projects/{id}/pose_sets            新建一个姿势集文件夹（同名拒绝）
  GET  /projects/{id}/pose_sets/{name}     姿势集的帧表
  GET  /projects/{id}/pose_renders         项目里的成片集
  POST /projects/{id}/pose_renders         新建成片集文件夹（同名拒绝）
  GET  /projects/{id}/pose_renders/{name}  每帧的候选 + 当前选中
  POST /projects/{id}/pose_renders/{name}/select     选某一帧的第几个候选
  POST /projects/{id}/pose_renders/{name}/run        生成/补生成（候选追加）
  POST /projects/{id}/pose_sets/{name}/gif           原始帧 → GIF（审核用）
  POST /projects/{id}/pose_renders/{name}/gif        成片 → GIF（取当前选中的候选）
  POST /projects/{id}/pose_renders/{name}/godot      成片 → Godot 4 工程包（zip）
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
from workflows import stages
from workflows import charactergen
from workflows import clothesgen
from workflows import directions
from workflows import posegen
from workflows import spriteatlas
from workflows import pipeline
from workflows import emotiongen
from workflows import qwen_llm
from workflows import library
from workflows import metadb
from workflows import motiontemplates
from workflows import matte
from workflows import matte_screen
from workflows import pixelfy
from workflows import paths
from workflows import poselib
from workflows import playtest
from workflows import posesubjects
from workflows import partsplit
from workflows import poseset
from workflows import projectio
from workflows import projectstore
from workflows import lorastore
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


def _restore_global_settings() -> None:
    """启动时把全局设置灌回运行态。

    **不能为了恢复 LoRA 去加载管线**：那会在用户还没点「生成」时就占掉十几 GB
    显存。`Engine.restore_loras()` 只记规格，真正挂载推迟到第一次 `txt2img_pipe()`。
    """
    try:
        s = metadb.get_settings()
        engine.restore_loras(s.get("loras") or [])
        engine.set_fast_mode(bool(s.get("fast_mode")))
    except Exception as exc:                  # noqa: BLE001
        log.warning("恢复全局设置失败（不影响使用）：%s", exc)


_restore_global_settings()

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


def _portrait_expression_batch(engine, params: Dict[str, Any],
                               job: Dict[str, Any]) -> Dict[str, Any]:
    """一张立绘 → N 个表情（官方脸部局部重绘，逐个出图、逐个收编）。

    为什么放在**一个任务里循环**而不是让前端连发 N 次：单卡串行，N 次请求会
    各自排一次队、各自冷启一次人脸检测；更要紧的是进度只有一份——用户要看
    「第 2/6 张」，不是六个互不相干的进度条。

    每张都复用 `portrait_expression`（detect_face → crop → 重绘 → 贴回），
    所以衣服 / 姿势 / 背景逐像素不变；出完立刻 `publish_portrait` 进当前阶段。
    一张失败只记进 `failed`，不连累整批。
    """
    emotions = [str(x) for x in (params.get("emotions") or []) if str(x).strip()]
    total = max(1, len(emotions))
    items: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, emo in enumerate(emotions, 1):
        job["progress"] = {"done": i - 1, "total": total,
                           "current": f"表情 {i}/{total}：{emo}"}
        try:
            out = gen.portrait_expression(
                engine, image=str(params.get("image") or ""), emotion=emo,
                negative=str(params.get("negative") or ""),
                steps=int(params.get("steps") or 25),
                cfg=float(params.get("cfg") or 3.0),
                seed=int(params.get("seed", -1)),
                target_size=int(params.get("target_size") or 2048),
                crop_max_side=int(params.get("crop_max_side") or 1024),
                feather=int(params.get("feather", 50)),
                scope=str(params.get("scope") or ""), tag="wpe",
            )
            meta = out.get("meta") or {}
            # 落盘目录用**官方解析后**的 safe_name，别用前端原样传的字符串
            real = str(meta.get("emotion") or emo)
            pub = characters.publish_portrait(
                name=str(params.get("character") or ""),
                src=str(out.get("raw") or out.get("file") or ""),
                costume=str(params.get("costume") or "Naked"),
                emotion=real, matte=False,
                stage_id=str(params.get("stage") or ""),
            )
            items.append({
                "emotion": real,
                "url": _asset_url(str(pub.get("path") or "")),
                "path": pub.get("path"), "file": pub.get("file"),
                "seconds": meta.get("seconds"),
            })
        except Exception as exc:                     # noqa: BLE001
            failed.append({"emotion": emo,
                           "error": f"{type(exc).__name__}: {exc}"})
        job["progress"] = {"done": i, "total": total,
                           "current": f"表情 {i}/{total}：{emo}"}
    job["progress"] = {"done": total, "total": total,
                       "current": f"完成 {len(items)}/{total}"}
    return {"items": items, "failed": failed, "total": total,
            "count": len(items), "kind": "portrait_expression_batch",
            "seconds": round(time.time() - t0, 1)}


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
        if workflow == "wparts":
            # **拆部件**：See-Through 在独立 venv 的子进程里跑 2~3 分钟，
            # 但占的是同一块 GPU，照样要拿 `_gpu_lock`，别跟 Qwen 抢显存。
            # 进度由 partsplit 直接写 `job["progress"]`（阶段上报）。
            with _gpu_lock:
                job["result"] = partsplit.split_job(engine, params, job=job)
            job["status"] = "done"
            return
        if workflow == "wpebatch":
            # **立绘批量换表情**：一个任务里按表情循环，每张走官方脸部局部重绘
            # 并立刻收编进当前阶段。落盘在辅助函数里做，所以不走下面那段
            # `publish`（否则只会收编最后一张）。
            with _gpu_lock:                # 单卡不并发
                job["result"] = _portrait_expression_batch(engine, params, job)
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
        if workflow == "wpcbatch":
            # **角色迁移批量**：一个姿势集 → 一套成片，每帧若干候选。
            # 候选是**追加**进 render.json 的（旧的不删，用户可能更喜欢上一版），
            # 所以同一个任务可以对同一帧反复跑而不丢东西。
            from workflows import posebatch
            with _gpu_lock:                # 单卡不并发
                job["result"] = posebatch.run_batch(engine, params, job=job)
            job["status"] = "done"
            return
        fn = {"w1": gen.txt2img, "w2": gen.img2img, "w3": gen.inpaint,
              "w4": gen.poseswap, "wbg": gen.matte_bg,
              "w5": gen.ai_repaint, "wpx": gen.pixelify,
              "wpc": gen.pose_clone, "wss": gen.sprite_swap,
              "topixel": gen.image_to_pixel,
              "wout": gen.outpaint,
              "woutfit": gen.outfit_swap, "wface": gen.face_swap,
              # 立绘工具箱：换装（挂 Clothes LoRA）/ 换表情（脸部局部重绘）
              "wpo": gen.portrait_outfit, "wpe": gen.portrait_expression,
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
        # 全局「抠图方式」以服务端设置为准：请求没显式给就补上。**只在目标函数
        # 真的收这个形参时补**——否则会被下面的签名过滤丢进 params_ignored，
        # 在产物里留一条「matte_backend 被忽略」的假警告。
        if "matte_backend" in kw_names and "matte_backend" not in params:
            try:
                params["matte_backend"] = metadb.get_settings()["matte_backend"]
            except Exception:                 # noqa: BLE001
                params["matte_backend"] = "auto"
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
                # 立绘走 Portraits/（跟动作帧分开），别的一律进 Sprites/
                publish = (characters.publish_portrait
                           if str(pub.get("kind") or "") == "portrait"
                           else characters.publish_image)
                is_portrait = str(pub.get("kind") or "") == "portrait"
                extra = ({"stage_id": str(pub.get("stage") or "")}
                         if is_portrait else {})
                job["result"]["published"] = publish(
                    name=str(pub.get("character") or ""), src=str(src),
                    costume=str(pub.get("costume") or "Naked"),
                    emotion=str(pub.get("emotion") or "neutral"),
                    matte=bool(pub.get("matte")), **extra,
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
        # 语义分割会话按「一个任务」复用、任务结束释放（学 qwen_llm 的
        # 默认用完卸载）。只在收尾做一次，批量抠上百帧不会每帧重建 ONNX，
        # 也不会让 rembg 永久占着显存跟 Qwen 抢 17GB headroom。
        # 释放失败绝不能把已经成功的生成任务判成失败。
        try:
            matte.release_ai_sessions()
        except Exception:                     # noqa: BLE001
            pass
        # 任务历史只记摘要（状态 / 耗时），绝不记图片、精灵帧或模型路径。
        # 索引写失败不能反过来影响生成任务本身 —— 主流程已经在上面收尾了。
        try:
            metadb.record_task(
                job_id,
                character=str(params.get("character") or ""),
                workflow=workflow,
                status=str(job.get("status") or "failed"),
                seconds=float(job.get("elapsed") or 0),
                error=str(job.get("error") or ""),
            )
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning("记录任务历史失败", exc_info=True)


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
    """扫一遍 `models/lora/`，返回**全部** .safetensors + 当前挂载状态。

    - `items`：可直接挂载的（diffusers 原生格式）
    - `skipped`：目录里存在但格式不对的（ComfyUI / 其它架构），前端显示为
      「需要转换」，可以调 `POST /loras/convert` 就地转换
    - `dir` / `count`：扫的是哪个目录、一共几个文件（用户加完文件点刷新即可）
    """
    scan = engine.scan_loras()
    items = [x for x in scan if x["format"] == "diffusers"]
    skipped = [x for x in scan if x["format"] != "diffusers"]
    return {"items": items, "active": engine.active_loras(),
            "skipped": skipped, "count": len(scan),
            "dir": str(engine.lora_dir())}


@app.post("/loras/apply")
def apply_loras(req: LoraReq) -> Dict[str, Any]:
    """勾选 / 取消 LoRA（**只改设置，不加载模型**）。

    `items`：`[{"name": "qwen21_anyangle", "scale": 0.8}]`，空数组 = 全卸。

    这里以前先 `txt2img_pipe()` 再挂：用户只是想勾一个复选框，却要等 15GB
    管线进显存（几十秒；权重缺失时直接 400，界面表现为「勾了又弹回去」）。
    现在只把规格记进引擎（`Engine.set_user_loras`），真正的挂载推迟到下一次
    生成的 `txt2img_pipe()` —— 与 `PUT /settings` 的懒恢复同一条路。
    """
    try:
        engine.set_user_loras(req.items)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"挂载 LoRA 失败：{exc}") from exc
    active = engine.active_loras()
    return {"active": [a.get("adapter") or a["name"] for a in active],
            "items": active}


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
    """卸掉全部 LoRA，回到基座模型（同样只改设置，不加载模型）。"""
    try:
        engine.set_user_loras([])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"卸载 LoRA 失败：{exc}") from exc
    return {"active": [], "items": []}


# ---------------------------------------------------------------------------
# 全局设置 + LoRA 导入
#
# 以前底模 / 抠图方式 / LoRA / 快速模式只在前端 useState 里，重开一次全丢；
# 现在 SQLite 是权威，服务端启动自动恢复（见 _restore_global_settings）。
# ---------------------------------------------------------------------------

class SettingsReq(BaseModel):
    """`PUT /settings` 的局部补丁；None 的键不动。"""

    base_model: Optional[str] = None
    matte_backend: Optional[str] = None
    loras: Optional[List[Dict[str, Any]]] = None
    fast_mode: Optional[bool] = None


@app.get("/settings")
def get_global_settings() -> Dict[str, Any]:
    """读全局设置（底模 / 抠图方式 / 生效的 LoRA / 快速模式）。"""
    return metadb.get_settings()


@app.put("/settings")
def put_global_settings(req: SettingsReq) -> Dict[str, Any]:
    """写全局设置并让运行态立刻跟上（都不加载模型、不占显存）。"""
    patch = req.model_dump(exclude_none=True)
    s = metadb.set_settings(patch)
    if "loras" in patch:
        engine.restore_loras(s["loras"])
    if "fast_mode" in patch:
        engine.set_fast_mode(s["fast_mode"])
    return s


@app.post("/loras/inspect")
async def lora_inspect(request: Request, name: str = "") -> Dict[str, Any]:
    """导入第一步：收原始字节 → 暂存 → **只读**格式检测，不写 models/lora/。

    返回 `need_convert` / `convertible` / `reason`，前端据此决定要不要弹
    「需要转换，是否继续」的确认框。用户没确认之前，磁盘上只有一份暂存文件。
    """
    if name and not name.lower().endswith(".safetensors"):
        raise HTTPException(400, "只支持 .safetensors 格式的 LoRA 文件")
    lorastore.STAGE_DIR.mkdir(parents=True, exist_ok=True)
    token = lorastore.new_token()
    tmp = lorastore.stage_path(token)
    total = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                total += len(chunk)
                if total > lorastore.MAX_LORA_BYTES:
                    raise HTTPException(
                        413, f"文件太大（上限 {lorastore.MAX_LORA_BYTES // 2**30}GB）")
                f.write(chunk)
        if total == 0:
            raise HTTPException(400, "空文件")
        info = lorastore.inspect(tmp, name)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:                  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"无法读取这个 LoRA：{exc}") from exc
    return {"token": token, **info}


class LoraCommitReq(BaseModel):
    """导入第二步：`convert=True` 才是用户确认过转换。"""

    token: str
    convert: bool = False
    name: str = ""


@app.post("/loras/commit")
def lora_commit(req: LoraCommitReq) -> Dict[str, Any]:
    """把暂存文件（必要时先转格式）落进 `models/lora/`。"""
    try:
        out = lorastore.commit(req.token, req.convert, req.name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:                  # noqa: BLE001
        raise HTTPException(500, f"导入 LoRA 失败：{exc}") from exc
    # 同名重导入：清掉内存里旧的挂载规格，再按磁盘上的新文件重新恢复。
    engine.forget_lora(out["name"])
    engine.restore_loras(metadb.get_settings().get("loras") or [])
    return out


class LoraConvertReq(BaseModel):
    """就地转换一个已经在 models/lora/ 里的 LoRA。"""

    name: str


@app.post("/loras/convert")
def lora_convert(req: LoraConvertReq) -> Dict[str, Any]:
    """把目录里格式不对的 LoRA 就地转成 diffusers（原名不变，先备份）。"""
    try:
        out = lorastore.convert_in_place(req.name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:                  # noqa: BLE001
        raise HTTPException(500, f"转换 LoRA 失败：{exc}") from exc
    # 转换后重新评估挂载规格（原来因为格式不对被 restore 丢掉的条目会回来）
    engine.forget_lora(out["item"]["name"])
    engine.restore_loras(metadb.get_settings().get("loras") or [])
    return out


@app.delete("/loras/{name}")
def lora_delete(name: str) -> Dict[str, Any]:
    """删掉一个已导入的 LoRA（文件 + 全局设置里的挂载记录）。"""
    try:
        removed = lorastore.delete(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    engine.forget_lora(removed)
    s = metadb.get_settings()
    rest = [x for x in (s.get("loras") or []) if x.get("name") != removed]
    s = metadb.set_settings({"loras": rest})
    engine.restore_loras(s["loras"])
    return {"ok": True, "name": removed, "active": engine.active_loras()}


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
    # **优先 3D**：MediaPipe 给米制 3D 点，能算出真实躯干朝向，把侧身照片
    # 「转正」后重投影成 2D 再交给编辑器（见 workflows/pose3d.py 头注释）。
    # 没有深度时，侧身照片的两髋投影只隔身高的 1.x%，官方转换器会判出 80°+
    # 骨盆倾斜 → 人偶躺倒。没有 mediapipe 就退回 DWPose 的 2D 点（功能不变差）。
    pts: Dict[int, Any] = {}
    extra: Dict[str, Any] = {}
    ex3 = engine.pose3d()
    if ex3 is not None:
        try:
            pts, extra = ex3.points2d(img)
        except ValueError:
            pts = {}          # 3D 没检出人，交给 DWPose 再试一次
        except Exception as exc:                       # noqa: BLE001
            log.warning("3D 姿态提取失败，回退 DWPose：%s", exc)
            pts = {}
    if not pts:
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

    # 朝向提示：3D 那条路能直接量出躯干相对相机的 yaw，比"肩膀宽度"猜得准
    guess = 0
    if extra.get("body_yaw") is not None:
        guess = int(round((float(extra["body_yaw"]) % 360) / 45) % 8) * 45
    elif 2 in pts and 5 in pts and abs(pts[2][0] - pts[5][0]) < 0.07:
        # 肩膀在画面上几乎重叠 → 原图接近侧面。鼻尖相对颈的方向决定朝左/朝右。
        guess = 90 if (0 in pts and 1 in pts and pts[0][0] >= pts[1][0]) else 270
    return {
        "points": points,
        "joints": len(pts),
        "azimuth_guess": guess,
        # `reprojected=True` 表示这组点是"3D 转正后投影"的，前端据此把官方
        # 转换器从 2D 反推的 modelRotation 归零（否则会用投影压缩过的髋线
        # 再判一次骨盆倾斜，人偶还是躺）。
        "reprojected": bool(extra.get("backend") == "mediapipe"),
        "body_yaw": extra.get("body_yaw"),
        "backend": extra.get("backend", "dwpose"),
    }


@app.get("/library")
def get_library() -> Dict[str, Any]:
    """整个用户资料库一次给全（提示词收藏 + 姿势库 + 界面偏好）。

    冷启动只发一次请求，省得界面为了画三个面板连发三次。
    见 workflows/library.py 的模块注释：这些都**只落在服务端磁盘**。
    """
    return library.snapshot()


# ---------------------------------------------------------------- 动作模板


@app.get("/motion-templates")
def motion_templates_list() -> Dict[str, Any]:
    """内置预设 + 用户模板 + 每套的逐帧生成成本估算（一次给全，避免 N+1）。

    模板只存元数据（不存图 / 权重），落 ``library/motion_templates.json``；
    内置三套是代码常量，不落盘，升级不会跟用户改过的文件打架。
    """
    return motiontemplates.summary()


@app.get("/motion-templates/{tid}/estimate")
def motion_template_estimate(tid: str) -> Dict[str, Any]:
    t = motiontemplates.get_template(tid)
    if not t:
        raise HTTPException(404, f"没有这套动作模板：{tid}")
    return motiontemplates.estimate(t)


@app.post("/motion-templates")
def motion_template_create(req: Dict[str, Any]) -> Dict[str, Any]:
    """新建一套模板（id 由服务端生成，返回完整列表）。"""
    if not isinstance(req, dict):
        raise HTTPException(400, "模板必须是对象")
    try:
        return motiontemplates.summary(motiontemplates.save_template(req))
    except motiontemplates.TemplateError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/motion-templates/{tid}")
def motion_template_update(tid: str, req: Dict[str, Any]) -> Dict[str, Any]:
    """更新一套用户模板。**内置 id 不可覆盖**（要改就另存）。"""
    if not isinstance(req, dict):
        raise HTTPException(400, "模板必须是对象")
    doc = dict(req)
    doc["id"] = tid
    try:
        return motiontemplates.summary(motiontemplates.save_template(doc))
    except motiontemplates.TemplateError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/motion-templates/{tid}")
def motion_template_delete(tid: str) -> Dict[str, Any]:
    try:
        return motiontemplates.summary(motiontemplates.delete_template(tid))
    except motiontemplates.TemplateError as exc:
        raise HTTPException(400, str(exc)) from exc


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

    **例外**：画布项目里的多机位姿势集在
    ``assets/projects/<项目>/poses/<集>/``（深度 3），两层扫描够不到。
    上一版因此让角色中心「先在画布存一组姿势集」那句话点不通。这里定向
    补扫 ``poses/`` 与 ``pose_frames/`` 两类目录，不把全局深度加深；
    带 ``directions`` 的是按 YAWS 抽出来的多机位姿势集。
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
    seen = {item["path"] for item in found}

    def scan_set(d: Path, kind: str) -> None:
        if not d.is_dir() or d.name.startswith(".") or str(d) in seen:
            return
        try:
            imgs = sorted(p.name for p in d.iterdir()
                          if p.is_file() and p.suffix.lower() in _POSE_EXTS)
        except OSError:
            return
        if len(imgs) < 2:
            return
        item: Dict[str, Any] = {
            "path": str(d), "count": len(imgs), "sample": imgs[:3],
            "name": f"{d.parents[1].name}/{kind}/{d.name}",
        }
        set_file = d / "set.json"
        if set_file.is_file():
            try:
                doc = json.loads(set_file.read_text(encoding="utf-8"))
                dirs = {
                    directions.validate_direction(f.get("direction"))
                    for f in (doc.get("frames") or []) if isinstance(f, dict)
                }
                dirs.discard("")
                if dirs:
                    item["directions"] = sorted(dirs)
            except (OSError, ValueError):
                pass
        seen.add(str(d))
        found.append(item)

    projects = ASSETS / "projects"
    for kind in ("poses", "pose_frames"):
        for set_dir in sorted(projects.glob(f"*/{kind}/*")):
            scan_set(set_dir, kind)
    found.sort(key=lambda x: (-x["count"], x["path"]))
    return {"count": len(found[:limit]), "items": found[:limit]}


@app.post("/generate")
def submit(job: Job) -> Dict[str, Any]:
    # 白名单**必须与 `_run` 的分发表同步**：漏一个就是 400，而错误信息里
    # 又列不出来（用户会以为功能没做）。加新工作流时两处一起改。
    allowed = ("w1", "w2", "w3", "w4", "wbg", "w5", "wpx", "wpc",
               "wss", "topixel", "wap", "wvnccs",
               # 2026-10-30 新增
               "wout", "woutfit", "wface", "wpo", "wpe", "wpebatch")
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
                       project: str = "", subjects: int = 0) -> Dict[str, Any]:
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
    # 多角色人偶参考图：主体数随图片走（磁盘 sidecar，不进数据库）。
    # 生成时 generate.py 读它来决定提示词用单主体还是复数契约。
    n_subjects = posesubjects.normalize_count(subjects)
    posesubjects.write_subjects(out, n_subjects)
    return {"path": str(out), "url": f"/files/{out.relative_to(ASSETS).as_posix()}",
            "pose_subjects": n_subjects,
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
    #: 角色级描述（创建时可选）。用户明确不要参考图了。
    description: str = ""


class CharacterRenameReq(BaseModel):
    name: str


class CostumeReq(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)


@app.get("/characters")
def list_characters_api() -> Dict[str, Any]:
    """角色库列表：磁盘文件 + SQLite 元数据。

    磁盘负责封面 / 服装 / 帧数与生成所需的属性；标签、时间、自定义标签、
    备注以 metadb 为准（磁盘 config 里没有、也不该塞进 prompt 的信息）。
    每次列目录都同步一次索引，新建 / 删除的角色会自动进 / 出元数据表。
    """
    metadb.sync_characters()
    items = characters.list_character_summaries()
    meta = metadb.list_meta()
    for it in items:
        m = meta.get(it["name"]) or {}
        it["cover_url"] = _asset_url(it.pop("cover", "") or "")
        it["tags"] = m.get("tags", it.get("tags") or [])
        it["custom_tags"] = m.get("custom_tags", [])
        it["note"] = m.get("note", "")
        it["created_at"] = m.get("created_at", it.get("created_at") or 0)
        it["updated_at"] = m.get("updated_at", it.get("updated_at") or 0)
    return {"count": len(items), "items": items}


@app.patch("/characters/{name}/meta")
def update_character_meta_api(name: str, req: Dict[str, Any]) -> Dict[str, Any]:
    """写角色元数据（自定义标签 / 备注）。只存元数据，不碰任何文件。"""
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    custom = req.get("custom_tags")
    note = req.get("note")
    try:
        return metadb.update_meta(
            safe,
            [str(t) for t in custom] if isinstance(custom, list) else None,
            str(note) if note is not None else None,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/character-tasks")
def recent_character_tasks_api(limit: int = 50) -> Dict[str, Any]:
    """最近生成任务摘要（只记状态与耗时，不记文件 / 模型路径）。"""
    return {"items": metadb.recent_tasks(max(1, min(limit, 200)))}


@app.post("/characters")
def create_character_api(req: CharacterCreateReq) -> Dict[str, Any]:
    """建角色。已存在 → 409（不静默覆盖）。"""
    try:
        cfg = characters.create_character(req.name, req.info or None,
                                          req.description)
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
        stages.normalize(cfg)
        stages.update_info(cfg, {k: v for k, v in (req or {}).items() if k != "name"})
        characters.save_config(safe, cfg)
    return {"name": safe, "character_info": cfg["character_info"]}


class StagesSaveReq(BaseModel):
    #: 前端提交**完整数组**，数组顺序就是展示顺序（拖拽排序直接落这个）
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    active_stage: str = ""
    description: Optional[str] = None


def _stages_payload(safe: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """阶段列表 + 每阶段几张立绘缩略图，列表页一行一条直接铺。"""
    items = []
    for st in cfg.get("stages") or []:
        ps = characters.list_portraits(safe, stage_id=st["id"])
        items.append({
            "id": st["id"],
            "title": st.get("title") or "",
            "description": st.get("description") or "",
            "info": st.get("info") or {},
            "portrait_count": len(ps),
            "portraits": [{"url": _asset_url(x["file"]), "name": x["name"]}
                          for x in ps[:5]],
        })
    return {
        "name": safe, "stages": items,
        "active_stage": cfg.get("active_stage") or "",
        "description": cfg.get("description") or "",
    }


@app.get("/characters/{name}/stages")
def character_stages_api(name: str) -> Dict[str, Any]:
    """生命阶段列表。老角色第一次进来会补一个「初始设定」阶段并落盘。"""
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    with characters.storage_lock(characters.character_dir(safe)):
        cfg = stages.normalize(characters.load_config(safe) or {})
        characters.save_config(safe, cfg)
    return _stages_payload(safe, cfg)


@app.put("/characters/{name}/stages")
def save_character_stages_api(name: str, req: StagesSaveReq) -> Dict[str, Any]:
    """全量保存阶段（新建 / 改名 / 描述 / 拖拽排序 / 删除 / 切换激活）。

    **只删列表里的条目，不删磁盘立绘** —— 用户可能还想找回那段设定图。
    切换激活阶段会把它的 info 写回顶层 `character_info`，现有全部出图链路
    不需要带 stage 参数就自动跟着当前阶段走。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    with characters.storage_lock(characters.character_dir(safe)):
        cfg = characters.load_config(safe, strict=True)
        if cfg is None:
            raise HTTPException(404, f"没有这个角色：{safe}")
        stages.normalize(cfg)
        new_list = stages.sanitize(req.stages)
        if not new_list:
            raise HTTPException(400, "至少要保留一个生命阶段")
        cfg["stages"] = new_list
        stages.set_active(cfg, req.active_stage or new_list[0]["id"])
        if req.description is not None:
            cfg["description"] = str(req.description).strip()[:1000]
        characters.save_config(safe, cfg)
    return _stages_payload(safe, cfg)


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
        # 没指定就取这个角色最新的一张立绘（没有立绘的老角色再退回动作帧）
        portraits = characters.list_portraits(safe)
        if portraits:
            raw_path = portraits[0]["file"]
        else:
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


@app.get("/characters/{name}/portraits")
def character_portraits_api(name: str, stage: str = "") -> Dict[str, Any]:
    """这个角色的**全部立绘**（新的在前）。

    立绘以前混在 `Sprites/<服装>/neutral/` 里，被当成第 1 帧播进试玩台。
    现在单独放 `Portraits/`：这里只列立绘，试玩台/图集只认动作帧。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    items = []
    for p in characters.list_portraits(safe, stage_id=stage or None):
        items.append({
            "url": _asset_url(p["file"]),
            "path": p["file"],
            "name": p["name"],
            "costume": p["costume"],
            "emotion": p["emotion"],
            "mtime": p["mtime"],
        })
    return {"count": len(items), "items": items}


class PortraitProcessReq(BaseModel):
    path: str
    #: matte = 扣背景，pixel = 完美像素化
    op: str = "matte"
    params: Dict[str, Any] = Field(default_factory=dict)


class PortraitDeleteReq(BaseModel):
    path: str
    #: 删除必须显式确认：前端弹窗确认后才会传 true，避免脚本/误触直接删
    confirm: bool = False


def _checked_portrait(safe: str, raw: str) -> str:
    """把客户端给的立绘路径限制在这个角色 Portraits/ 目录内（挡 .. 与软链）。"""
    root = characters.portraits_dir(safe)
    target = os.path.realpath(str(raw or ""))
    if not target or not characters.is_path_under(root, target):
        raise HTTPException(400, "只能处理这个角色 Portraits/ 下的立绘")
    if not os.path.isfile(target):
        raise HTTPException(404, "这张立绘不存在")
    return target


@app.post("/characters/{name}/portraits/process")
def process_portrait_api(name: str, req: PortraitProcessReq) -> Dict[str, Any]:
    """对一张立绘做扣背景 / 完美像素化，**生成新的一份**，绝不改原图。

    返回处理前后的 url，供弹窗做滑动比对；新文件落在原图同目录，因此自动
    继承同一个生命阶段 / 服装 / 表情。两个算法都是纯确定性后处理，不跑模型。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    src_path = _checked_portrait(safe, req.path)
    op = str(req.op or "").strip().lower()
    if op not in ("matte", "pixel"):
        raise HTTPException(400, f"不支持的处理：{op}")
    p = req.params or {}
    try:
        with Image.open(src_path) as im:
            im.load()
            src = im.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"读不了这张立绘：{exc}") from exc

    meta: Dict[str, Any] = {"op": op}
    try:
        if op == "matte":
            backend = str(p.get("backend") or "auto").lower()
            if backend not in ("auto", "color", "u2net", "ai", "isnet", "birefnet"):
                backend = "auto"
            tol = max(8, min(120, int(p.get("tol", 32))))
            fringe = bool(p.get("fringe", True))
            holes = bool(p.get("holes", True))
            # 「自动」跟着全局抠图方式走（与角色中心其它抠图路径一致）
            eff = backend
            if eff == "auto":
                gb = matte_screen.global_backend()
                if gb != "auto":
                    eff = gb
            info: Dict[str, Any] = {}
            out = matte.matte_auto(
                src, backend=eff, model_dir=paths.MODELS_DIR / "matte",
                info=info, tol=tol, fringe=fringe, holes=holes)
            meta.update({"backend": eff, "tol": tol, "fringe": fringe,
                         "holes": holes, **info})
            suffix = "bg"
        else:
            # **目标尺寸按「长边」解释，短边按原图比例推导**。
            # 以前前端写死 [N, N]，768×1024（3:4）的立绘被压成 64×64（1:1），
            # 人物整个变形；滑动比对两边宽高比不同也对不齐。
            src_w, src_h = src.size
            long_edge = p.get("long_edge")
            if long_edge is None:
                raw_target = p.get("target")
                if isinstance(raw_target, (list, tuple)) and len(raw_target) == 2:
                    long_edge = max(int(raw_target[0]), int(raw_target[1]))
                elif isinstance(raw_target, (int, float)):
                    long_edge = int(raw_target)
            target = None
            if long_edge is not None:
                n = max(16, min(512, int(long_edge)))
                if src_w >= src_h:
                    tw, th = n, max(1, round(n * src_h / src_w))
                else:
                    th, tw = n, max(1, round(n * src_w / src_h))
                target = (tw, th)
            colors = max(0, min(128, int(p.get("colors", 32))))
            sample = str(p.get("sample_method") or "median")
            if sample not in ("median", "majority", "center"):
                sample = "median"
            grid_snap = bool(p.get("grid_snap", True))
            transparent_bg = bool(p.get("transparent_bg", False))
            pre_enhance = float(p.get("pre_enhance", 0.0))
            pre_enhance = max(0.0, min(2.0, pre_enhance))
            edge_rebuild = bool(p.get("edge_rebuild", False))
            res = pixelfy.refine_batch(
                [src], grid_snap=grid_snap, grid_size=None,
                target_size=target, sample_method=sample, colors=colors,
                transparent_bg=transparent_bg, snap_to=16,
                pre_enhance=pre_enhance, edge_rebuild=edge_rebuild)
            native = res["items"][0]["image"]
            native_size = [native.width, native.height]
            # pixelfy 输出的是**格子图**（768×1024 自动检测后只有 48×64），
            # 那是精灵帧要的原生像素尺寸，但直接存成一张「立绘」就小到没法用。
            # 项目里 postprocess.pixelate() 的标准做法是降采样后按 NEAREST
            # 放大回原尺寸：像素块保持硬边、不引入新颜色、alpha 不产生半透明。
            # `output=native` 保留原生小图，给要拿去做精灵帧素材的人。
            output_mode = str(p.get("output") or "restore").lower()
            if output_mode not in ("restore", "native"):
                output_mode = "restore"
            out = (native.resize((src_w, src_h), Image.NEAREST)
                   if output_mode == "restore" else native)
            meta.update({
                "long_edge": int(long_edge) if long_edge is not None else None,
                "target": list(target) if target else None,
                "colors": colors, "grid_snap": grid_snap,
                "sample_method": sample, "transparent_bg": transparent_bg,
                "pre_enhance": pre_enhance, "edge_rebuild": edge_rebuild,
                "native_size": native_size, "output": output_mode,
                "output_size": [out.width, out.height],
            })
            suffix = "px"
    except HTTPException:
        raise
    except Exception as exc:                    # noqa: BLE001
        raise HTTPException(400, f"处理失败：{exc}") from exc

    new_path = characters.portrait_variant_path(src_path, suffix)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    out.save(new_path, format="PNG")
    return {
        "before": _asset_url(src_path),
        "after": _asset_url(new_path),
        "path": new_path,
        "name": os.path.basename(new_path),
        "meta": meta,
    }


@app.post("/characters/{name}/portraits/delete")
def delete_portrait_api(name: str, req: PortraitDeleteReq) -> Dict[str, Any]:
    """删一张立绘。前端弹窗确认 + 这里再验一次 `confirm`，双保险防手抖。"""
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    if not req.confirm:
        raise HTTPException(400, "删除需要确认")
    target = _checked_portrait(safe, req.path)
    ok = characters.delete_portrait_file(target)
    return {"ok": ok, "deleted": os.path.basename(target)}


class PartsSplitReq(BaseModel):
    #: 要拆的立绘（必须在角色 Portraits/ 下）
    path: str
    #: 这份产物归属的生命阶段（空 = 默认阶段）；决定落 `Parts[/s_<id>]/`
    stage: str = ""
    #: seethrough = See-Through 语义分层；explode = 爆炸图 / 几何切件
    method: str = "seethrough"
    params: Dict[str, Any] = Field(default_factory=dict)


class PortraitVariantReq(BaseModel):
    path: str
    #: outfit = 换装，emotion = 换表情
    kind: str = "outfit"
    params: Dict[str, Any] = Field(default_factory=dict)
    #: 当前生命阶段：新立绘要落回同一个阶段
    stage: str = ""


#: 换装指令：只改衣服，其余全部保持（跟官方 swap 的「谁是身份」约束同思路）。
_PORTRAIT_OUTFIT_KEEP = (
    "Keep the same face, hairstyle, hair colour, body proportions, pose, "
    "background, lighting and art style. Do not change anything except the clothing."
)
#: 表情指令在官方模板之外补的约束：整图指令改图不像官方脸部 crop 那样
#: 天然锁定区域，得明说「只改脸」。
_PORTRAIT_EMOTION_EXTRA = (
    " Change ONLY the facial expression. Keep the same pose, clothes, hair, "
    "background and art style."
)


@app.get("/characters/{name}/parts")
def character_parts_api(name: str, stage: str = "") -> Dict[str, Any]:
    """该生命阶段已拆过的产物 + 两种拆法在本机能不能用。

    `stage` 缺省 = 默认阶段（只看 `Parts/` 根，不看 `s_*`）：
    拆部件页属于某个阶段，列的就是这个阶段的历史，别把别的阶段的产物混进来。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    return {"available": partsplit.available(),
            "stage": stage or "default",
            "items": partsplit.list_results(safe, stage or "default")}


@app.get("/characters/{name}/parts/{rid}")
def character_part_api(name: str, rid: str, stage: str = "") -> Dict[str, Any]:
    """单次拆件的完整清单（分层预览要层的坐标 / 文件 / 顺序）。"""
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    mf = partsplit.load_manifest(safe, rid, stage or "default")
    if not mf:
        raise HTTPException(404, f"没有这次拆件产物：{rid}")
    return mf


class PartsDeleteReq(BaseModel):
    path: str          # 兼容前端统一传 path；这里只认目录名（rid）
    stage: str = ""
    confirm: bool = False


@app.post("/characters/{name}/parts/delete")
def character_part_delete_api(name: str, req: PartsDeleteReq) -> Dict[str, Any]:
    """删一次拆件产物（整目录）。跟立绘删除一样必须 `confirm=true`。"""
    if not req.confirm:
        raise HTTPException(400, "删除必须带 confirm=true")
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    rid = os.path.basename(str(req.path or "").strip().rstrip("/\\"))
    if not rid or rid in (".", ".."):
        raise HTTPException(400, "要删哪一份？")
    try:
        ok = partsplit.delete_result(safe, rid, req.stage or "default")
    except partsplit.PartsError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not ok:
        raise HTTPException(404, f"没有这次拆件产物：{rid}")
    return {"deleted": rid}


@app.post("/characters/{name}/parts/split")
def character_parts_split_api(name: str, req: PartsSplitReq) -> Dict[str, Any]:
    """拆部件：产出**标准分层 PSD + 分层预览清单**，走单卡队列。

    See-Through 要在它自己的 venv 里跑 2~3 分钟（17G 权重、torch 2.11），
    爆炸图是纯 PIL、几秒；两者都返回 job_id，前端用统一任务轮询。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    src = _checked_portrait(safe, req.path)
    method = str(req.method or "").strip().lower()
    if method not in ("seethrough", "explode"):
        raise HTTPException(400, f"不支持的拆法：{method}")
    if method == "seethrough":
        av = partsplit.seethrough_available()
        if not av["ok"]:
            raise HTTPException(400, av["reason"])
    params: Dict[str, Any] = {
        "character": safe, "src": src, "method": method,
        "stage": str(req.stage or "default"),
        "params": dict(req.params or {}),
    }
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": "wparts", "params": params,
                     "status": "queued", "created": time.time(),
                     "progress": {"done": 0, "total": 100, "current": "排队中…"}}
    _q.put((job_id, "wparts", params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "position": _q.qsize(),
            "before": _asset_url(src), "method": method}


@app.post("/characters/{name}/portraits/variant")
def portrait_variant_api(name: str, req: PortraitVariantReq) -> Dict[str, Any]:
    """对一张立绘**真换装 / 真换表情**，产出同一个阶段下的新一份立绘。

    跟待处理页那两块**用同一套模型能力**，只是输入换成单张立绘：
      · 换装 → `wpo`：官方 Clothes Core LoRA（`VNCCS_QI2_ClothesCoreV2.6`）
        + 六字段拼出的 `Dress the character:` 提示词，25 步 / cfg 3.0；
      · 换表情 → `wpe`：官方脸部裁剪局部重绘（detect_face → crop →
        qwen_edit_image → 贴回），**衣服 / 姿势 / 背景逐像素不变**。
    出图后由 worker 的 `publish` 收编进 `Portraits/<服装>/<表情>/`，
    原图一个字节不改。走同一个单卡队列，立刻返回 `job_id`。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    src_path = _checked_portrait(safe, req.path)
    kind = str(req.kind or "").strip().lower()
    if kind not in ("outfit", "emotion"):
        raise HTTPException(400, f"不支持的变体：{kind}")
    p = req.params or {}
    emo = ""
    emos: List[str] = []          # 换装用不到，但返回值统一带它
    fields: Dict[str, str] = {}
    lora_scale = 1.0
    target_size = int(emotiongen.EMOTION_DEFAULTS["target_size"])
    crop_max_side = 1024
    feather = int(emotiongen.EMOTION_DEFAULTS["feather"])

    if kind == "outfit":
        # 六个结构化字段（上装 / 下装 / 鞋 / 头部 / 脸部 / 不要出现），
        # 跟待处理页的 CostumeForm 是同一组，按官方顺序拼描述。
        raw_fields = p.get("fields")
        if isinstance(raw_fields, dict):
            fields = {k: str(raw_fields.get(k) or "").strip()[:200]
                      for k in characters.COSTUME_TEXT_FIELDS}
        desc = charactergen.clothes_description(
            {k: fields.get(k, "") for k in charactergen.CLOTHES_FIELDS})
        if not desc:
            # 兼容旧调用：只给一句描述也能跑
            desc = str(p.get("description") or "").strip()[:400]
        if not desc:
            raise HTTPException(
                400, "换装至少要填一个字段（上装 / 下装 / 鞋 / 头部 / 脸部）")
        prompt = (f"Dress the character://n{desc}//n"
                  + _PORTRAIT_OUTFIT_KEEP)
        negative = charactergen.CLOTHES_NEGATIVE
        extra_neg = fields.get("negative_prompt", "")
        if extra_neg:
            negative = f"{negative}, {extra_neg}"
        try:
            lora_scale = max(0.0, min(2.0, float(p.get("lora_scale", 1.0) or 1.0)))
        except (TypeError, ValueError):
            lora_scale = 1.0
    else:
        # 表情库支持多选：`emotions` 给数组就走批量（一个任务里逐个出图）。
        # `emotion_name` 保留为单选兼容路径。
        raw_emos = p.get("emotions")
        if isinstance(raw_emos, (list, tuple)):
            for x in raw_emos:
                v = str(x or "").strip()[:80]
                if v and v not in emos:
                    emos.append(v)
        emo = str(p.get("emotion_name") or "").strip()[:80]
        if emo and emo not in emos:
            emos.insert(0, emo)
        emos = emos[:12]                     # 单任务上限：再多就跑一晚上了
        emo = emos[0] if emos else ""
        if not emos:
            raise HTTPException(400, "换表情至少要选一个表情")
        # prompt / 负向词都交给 portrait_expression 用官方 emotiongen 拼，
        # 这里只把角色自己的负面词带过去（官方会再追加 EMOTION_NEGATIVE_EXTRA）。
        cfg0 = characters.load_config(safe) or {}
        prompt = ""
        negative = str((cfg0.get("character_info") or {})
                       .get("negative_prompt") or "")
        try:
            target_size = max(512, min(4096, int(p.get("target_size", target_size))))
            crop_max_side = max(512, min(1536, int(p.get("crop_max_side",
                                                          crop_max_side))))
            feather = max(0, min(200, int(p.get("feather", feather))))
        except (TypeError, ValueError):
            pass

    try:
        costume = characters.ensure_safe_name(
            str(p.get("costume") or "Naked"), "服装名")
        emotion_tag = characters.ensure_safe_name(
            str(p.get("emotion_tag") or (emo if kind == "emotion" else "neutral")),
            "表情名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    def _clamp_int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(p.get(key, default))))
        except (TypeError, ValueError):
            return default

    try:
        cfg = max(0.5, min(8.0, float(p.get("cfg", 3.0) or 3.0)))
    except (TypeError, ValueError):
        cfg = 3.0

    # 一次选多个表情 → 专用批量分支（一个 job、一份进度、逐个收编）
    workflow = ("wpo" if kind == "outfit"
                else ("wpebatch" if len(emos) > 1 else "wpe"))
    params: Dict[str, Any] = {
        "image": src_path,
        "negative": negative,
        "steps": _clamp_int("steps", 25, 8, 60),
        "cfg": cfg,
        "seed": _clamp_int("seed", -1, -1, 2**31 - 1),
        "publish": {
            "character": safe,
            "costume": costume,
            "emotion": emotion_tag,
            "kind": "portrait",
            "stage": str(req.stage or ""),
            "matte": False,
        },
    }
    if kind == "outfit":
        params["prompt"] = prompt
        params["lora_scale"] = lora_scale
    elif workflow == "wpebatch":
        # 批量**自己落盘**（每张表情进各自的 `<表情>/` 目录），不走 `_run` 的
        # publish 段——那段只会收编最后一张。所以这里直接给角色/服装/阶段。
        params.pop("publish", None)
        params.update({
            "emotions": emos, "character": safe, "costume": costume,
            "stage": str(req.stage or ""),
            "target_size": target_size, "crop_max_side": crop_max_side,
            "feather": feather,
        })
    else:
        params.update({"emotion": emo, "target_size": target_size,
                       "crop_max_side": crop_max_side, "feather": feather})
    total = len(emos) if workflow == "wpebatch" else 1
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": workflow, "params": params,
                     "status": "queued", "created": time.time(),
                     # 批量时 total 就是张数：前端进度条才有意义
                     "progress": {"done": 0, "total": total,
                                  "current": (f"{total} 个表情排队中…"
                                              if total > 1 else "")}}
    _q.put((job_id, workflow, params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "position": _q.qsize(),
            "before": _asset_url(src_path), "kind": kind, "prompt": prompt,
            "count": total, "emotions": emos}


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
    # 中文提示词模式不走 PE（官方改写器只输出英文），也就没必要去加载管线。
    pipe = None
    if charactergen.pe_eligible(info, req.enhance):
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

    cfg = stages.normalize(characters.load_config(safe) or {})
    info = cfg.get("character_info") or {}
    active_stage = str(cfg.get("active_stage") or "")
    if req.prompt.strip():
        prompt = req.prompt.strip()
    else:
        pipe = None
        if charactergen.pe_eligible(info, req.enhance):
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
                    # 立绘归到当前生命阶段；阶段编辑页的右栏只显示这一阶段的图
                    "stage": active_stage,
                    # 立绘是「长什么样」，跟动作帧不是一类东西
                    "kind": "portrait",
                    # 没给就按幕布色决定：绿/蓝/品红自动抠，白/灰不动
                    # （白底会把白毛白甲一起判成背景，这条我们踩过）
                    "matte": matte_screen.should_auto_matte(
                        str(info.get("background_color") or ""), req.matte)},
    }
    if req.workflow == "w2":
        # 换装：底图优先用这套服装的**最新立绘**（立绘才是定妆图），
        # 没有立绘的老角色再退回动作帧。
        all_p = characters.list_portraits(safe, active_stage)   # 本阶段的立绘，新的在前
        same = [p["file"] for p in all_p if p["costume"] == costume]
        base = (same or [p["file"] for p in all_p] or [""])[0]
        if not base:
            # 老角色（立绘还躺在 Sprites 里）：取序号最大那张
            sprites = (characters.list_sprite_images(safe, costume, "neutral")
                       or characters.list_sprite_images(safe, "Naked", "neutral"))
            base = sprites[-1] if sprites else ""
        if not base and not req.init_image.strip():
            raise HTTPException(400, "换装需要一张底图：先在「属性」页生成立绘")
        params["image_path"] = req.init_image.strip() or base
    params.update({k: v for k, v in (req.params or {}).items()
                   if k not in ("prompt", "negative", "publish")})
    # **画风只由画风库自己那条定义决定**：立绘绝不能挂 presets 的公共契约。
    # gen.txt2img 默认 preset="sprite"，会追加 "flat plain background /
    # ample empty space around" 这类构图词（跟立绘「占满画面高度」直接打架），
    # 更糟的是 SPRITE_NEG_PAINT 里写着 "photo, 3d render" —— 用户选写实 /
    # 3D 渲染画风时，等于用负向词把自己选的画风压掉。Q 版等风格同理被拉平。
    # 放在 params.update 之后：客户端手传的 preset 也盖不掉这个约束。
    params["preset"] = "none"

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


@app.get("/characters/{name}/playtest")
def character_playtest_assets(name: str) -> Dict[str, Any]:
    """导出前的试玩验收：只读扫现有精灵帧 + atlas 元数据 + 方向缺口报告。

    **不写盘、不生成图**。前端据此把角色放进场景里跑一遍，并检查
    四方向 / 八方向 / 横板三种视角模式各缺哪些方向。
    """
    try:
        safe = characters.ensure_safe_name(name, "角色名")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not characters.character_exists(safe):
        raise HTTPException(404, f"没有这个角色：{safe}")
    return playtest.collect(safe)


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
        "directionAnimations": int(info.get("directionAnimations") or 0),
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
    """磁盘上所有画布（摘要，不带节点）+ SQLite 元数据。

    「项目」这个内部 id 在界面上一律叫「画布」；磁盘目录名 / API 路径保持
    不变（改路径会牵动几十处调用）。封面只给 URL，图片本体仍在磁盘。
    """
    items = projectstore.list_projects()
    try:
        metadb.sync_canvases()
        meta = metadb.list_canvas_meta()
    except Exception:  # 索引不可用时列表仍要能打开
        logging.getLogger(__name__).warning("同步画布元数据失败", exc_info=True)
        meta = {}
    for it in items:
        m = meta.get(str(it.get("id") or "")) or {}
        it["cover_url"] = _asset_url(m.get("cover_path") or "")
        it["tags"] = m.get("tags", [])
        it["custom_tags"] = m.get("custom_tags", [])
        it["note"] = m.get("note", "")
        it["created_ts"] = m.get("created_at", 0)
        it["updated_ts"] = m.get("updated_at", 0)
    ws = projectstore.load_workspace()
    return {"count": len(items), "items": items, "current": ws.get("current", "")}


@app.patch("/projects/{scope}/meta")
def update_canvas_meta_api(scope: str, req: Dict[str, Any]) -> Dict[str, Any]:
    """写画布元数据（自定义标签 / 备注）。只存元数据，不碰 project.json 的节点。"""
    custom = req.get("custom_tags")
    note = req.get("note")
    try:
        return metadb.update_canvas_meta(
            scope,
            [str(t) for t in custom] if isinstance(custom, list) else None,
            str(note) if note is not None else None,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


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


# ------------------------------------------------------------------ *
# 项目内的姿势集 / 成片集
#
# **为什么放在项目目录而不是用户资料库**：一整套动作（N 帧人偶图）属于
# 这一次的动作采样，跟项目和那条动画绑定；资料库存的是"跨项目复用的单个姿势"。
# 目录结构与取舍见 `workflows/poseset.py` 的模块注释。
# ------------------------------------------------------------------ *


class PoseSetFrameReq(BaseModel):
    t: float = 0.0
    yaw: int = 0
    pose: Any = None
    #: 人偶图，dataURL 或裸 base64（宿主手里就是 dataURL，不用再转一次）
    png: str
    #: 抽帧那一刻的多角色场景快照。**不解析、不校验**，原样交给 poseset
    #: 写进 set.json；姿势集回放时才能连每个角色的位置一起还原。
    scene: Optional[Dict[str, Any]] = None
    camera: Optional[Dict[str, Any]] = None
    characters: Optional[List[Any]] = None
    #: 多机位抽帧的方向 key（down / up-right / left …）。单方向抽帧不带。
    #: **必须显式声明**：Pydantic 默认吞掉多余字段，不声明就静默丢失。
    direction: Optional[str] = None


class PoseSetCreateReq(BaseModel):
    name: str
    source: Dict[str, Any] = Field(default_factory=dict)
    frames: list[PoseSetFrameReq]


class PoseRenderCreateReq(BaseModel):
    name: str
    poseSet: str
    prompt: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class PoseRenderSelectReq(BaseModel):
    frame: int
    candidate: int


class PoseRenderRunReq(BaseModel):
    """跑（或补跑）成片集。`frames` 为空 = 整个姿势集。"""
    frames: list[int] = Field(default_factory=list)
    candidates: int = 1
    #: 走哪条引擎（和「角色迁移」节点的 `pcEngine` 同一个字段）
    engine: str = "wvnccs"
    char_image: str = ""
    char_images: list[str] = Field(default_factory=list)
    prompt: str = ""
    negative: str = ""
    steps: int = 16
    cfg: float = 1.0
    width: int = 768
    height: int = 768
    seed: int = -1
    ref_mode: str = "auto"
    bg: str = "white"
    matte: bool = False
    matte_backend: str = "auto"
    pixel: int = 0
    palette: int = 0
    model: str = ""


@app.get("/projects/{scope}/pose_sets")
def project_pose_sets(scope: str) -> Dict[str, Any]:
    """项目里存过的姿势集（人偶编辑器抽帧存下来的那几套）。"""
    items = poseset.list_pose_sets(scope)
    return {"project": scope, "count": len(items), "items": items}


@app.post("/projects/{scope}/pose_sets")
def project_pose_set_create(scope: str, req: PoseSetCreateReq) -> Dict[str, Any]:
    """新建一个姿势集文件夹。**同名一律拒绝**（用户的心智是"再存一份"）。"""
    try:
        one = poseset.save_pose_set(
            scope, req.name,
            [f.model_dump() for f in req.frames], source=req.source)
    except poseset.PoseSetExists as exc:
        raise HTTPException(409, str(exc)) from exc
    except poseset.PoseSetError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"set": one, "items": poseset.list_pose_sets(scope)}


@app.get("/projects/{scope}/pose_sets/{name}")
def project_pose_set(scope: str, name: str) -> Dict[str, Any]:
    try:
        return {"set": poseset.load_pose_set(scope, name)}
    except poseset.PoseSetError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/projects/{scope}/pose_renders")
def project_pose_renders(scope: str) -> Dict[str, Any]:
    items = poseset.list_renders(scope)
    return {"project": scope, "count": len(items), "items": items}


@app.post("/projects/{scope}/pose_renders")
def project_pose_render_create(scope: str, req: PoseRenderCreateReq
                               ) -> Dict[str, Any]:
    """建一个成片集文件夹（同名拒绝），帧表按姿势集先铺好。"""
    try:
        one = poseset.create_render(scope, req.name, req.poseSet,
                                    prompt=req.prompt, params=req.params)
    except poseset.PoseSetExists as exc:
        raise HTTPException(409, str(exc)) from exc
    except poseset.PoseSetError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"render": one, "items": poseset.list_renders(scope)}


@app.get("/projects/{scope}/pose_renders/{name}")
def project_pose_render(scope: str, name: str) -> Dict[str, Any]:
    try:
        return {"render": poseset.load_render(scope, name),
                "items": poseset.list_renders(scope)}
    except poseset.PoseSetError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/projects/{scope}/pose_renders/{name}/select")
def project_pose_render_select(scope: str, name: str,
                               req: PoseRenderSelectReq) -> Dict[str, Any]:
    """选某一帧的第几个候选。**旧候选一直留着**，随时能选回去。"""
    try:
        return {"render": poseset.select_candidate(scope, name, req.frame,
                                                   req.candidate)}
    except poseset.PoseSetError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/projects/{scope}/pose_sets/{name}/gif")
def project_pose_set_gif(scope: str, name: str,
                         frame_ms: int = 125, max_side: int = 512) -> Dict[str, Any]:
    """把姿势集合成 GIF（给团队看"这套动作连起来顺不顺"）。"""
    try:
        return {"gif": poseset.export_gif(scope, poseset.POSES_DIR, name,
                                          frame_ms=frame_ms, max_side=max_side)}
    except poseset.PoseSetError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/projects/{scope}/pose_renders/{name}/gif")
def project_pose_render_gif(scope: str, name: str,
                            frame_ms: int = 125, max_side: int = 512
                            ) -> Dict[str, Any]:
    """把成片集合成 GIF：每帧取**当前选中的那张候选**。

    审核场景要的就是"我挑完之后的成片连起来什么样"，把全部候选都塞进去没意义。
    """
    try:
        return {"gif": poseset.export_gif(scope, poseset.RENDERS_DIR, name,
                                          frame_ms=frame_ms, max_side=max_side)}
    except poseset.PoseSetError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/projects/{scope}/pose_renders/{name}/godot")
def project_pose_render_godot(
    scope: str, name: str,
    stem: str = "", anim: str = "",
    fps: Optional[float] = None, loop: Optional[bool] = None,
    folder: str = "sprites", origin: str = "bottom",
    texture_filter: str = "nearest",
) -> Dict[str, Any]:
    """成片集 → Godot 4 工程包（逐帧 PNG + SpriteFrames + AnimatedSprite2D + zip）。

    每帧取**当前选中的候选**：用户挑过的就是他要交付的版本。

    ``fps`` / ``loop`` 不传时，顺着成片集 → 姿势集 → ``set.json.source.templateId``
    反查动作模板取值；模板也没有才退回 12 fps / 循环。显式传参永远优先。
    """
    from workflows import godotexport
    try:
        return {"bundle": godotexport.export_bundle(
            scope, name, stem=stem, anim_name=anim, fps=fps, loop=loop,
            folder=folder, origin=origin, texture_filter=texture_filter)}
    except godotexport.GodotExportError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/projects/{scope}/pose_renders/{name}/run")
def project_pose_render_run(scope: str, name: str, req: PoseRenderRunReq
                            ) -> Dict[str, Any]:
    """生成（或补生成）成片：走同一个单卡队列，立刻返回 `job_id`。

    前端把 `job_id` 挂到节点的 `job` 上，现有的轮询/看门狗就会接管进度。
    """
    params: Dict[str, Any] = {
        "project": scope, "render": name,
        "frames": [int(v) for v in req.frames],
        "candidates": int(req.candidates),
        "engine": req.engine,
        "char_image": req.char_image,
        "char_images": list(req.char_images),
        "prompt": req.prompt, "negative": req.negative,
        "steps": int(req.steps), "cfg": float(req.cfg),
        "width": int(req.width), "height": int(req.height),
        "seed": int(req.seed), "ref_mode": req.ref_mode, "bg": req.bg,
        "matte": bool(req.matte), "matte_backend": req.matte_backend,
        "pixel": int(req.pixel), "palette": int(req.palette),
        "model": req.model or None,
    }
    # 先把"要跑几帧"算准，前端才好显示总张数（也顺手挡掉不存在的帧）
    try:
        detail = poseset.load_render(scope, name)
    except poseset.PoseSetError as exc:
        raise HTTPException(404, str(exc)) from exc
    # **必须把姿势集名带下去**：`posebatch.run_batch` 靠它去 `poses/<名字>/set.json`
    # 找源帧。漏了这一行 → 每帧都抛「缺成片集名或姿势集名」，而 `render.json`
    # 已经建好了，界面上的表现就是"点全部生成，出来一堆空格子但没有图"。
    params["pose_set"] = str(detail.get("poseSet") or "")
    if not params["pose_set"]:
        raise HTTPException(400, "这个成片集没记着属于哪套姿势集（render.json 被改过？）")
    planned = params["frames"] or [f["index"] for f in detail["frames"]]
    total = max(1, len(planned) * max(1, params["candidates"]))

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"job_id": job_id, "workflow": "wpcbatch", "params": params,
                     "status": "queued", "created": time.time(),
                     "progress": {"done": 0, "total": total, "current": ""}}
    _q.put((job_id, "wpcbatch", params))
    _prune(_jobs, JOB_KEEP)
    return {"job_id": job_id, "status": "queued", "total": total,
            "render": name, "frames": planned, "position": _q.qsize()}


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
