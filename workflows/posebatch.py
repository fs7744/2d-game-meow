"""角色迁移「批量生成」：一个姿势集 → 一套成片（每帧若干候选）。

**为什么不复用 `posegen.run_batch`**：那条是官方 `pose_generation` 的整批语义
（整批替换 `Sprites/<服装>/neutral/`、整批一个种子、出完就发布到角色库）。
这里的需求完全不同：

- 产物落在**画布项目**的 `pose_frames/<名字>/` 下，不进角色库；
- 每帧要能**单独重生成**，且**旧候选不能删**（用户可能觉得上一版更好）；
- 生成用的是**角色迁移**那条管线（`generate.pose_clone`：DWPose 条件图 +
  双图 vision context），这样批量出来的和用户逐张点「生成」拿到的是同一套效果。

**逐帧逐候选串行**：QI2 一次只能跑一张（显存），并发只会把卡顶爆。
进度按"已出的张数"报，前端进度条才有意义。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflows import poseset
from workflows import projectstore

#: 单次批量最多出多少张（帧 × 候选）。一条 16 帧的动作 × 3 候选就是 48 张，
#: 每张 20~28 秒 —— 已经是一小时级别的任务，再大就该拆开跑。
MAX_IMAGES = 160


class PoseBatchError(RuntimeError):
    """可读错误（API 转 400）。"""


def _sanitize(project: str) -> str:
    if not projectstore.clean_scope(project):
        raise PoseBatchError("缺少项目 id")
    return project


def run_batch(engine: Any, params: Dict[str, Any],
              job: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """按成片集逐帧生成，候选**追加**进 `render.json`（不覆盖旧的）。

    `params`：
      project / render / pose_set / frames（要跑哪几帧，缺省=全部）
      candidates（每帧几个候选）/ char_image / char_images / prompt / negative
      steps / cfg / seed / width / height / ref_mode / bg / matte / pixel …
    """
    project = _sanitize(str(params.get("project") or ""))
    render = str(params.get("render") or "").strip()
    pose_set = str(params.get("pose_set") or "").strip()
    if not render or not pose_set:
        raise PoseBatchError("缺成片集名或姿势集名")

    from workflows import generate as gen        # 延迟 import：这个模块要 torch

    all_idx = poseset.frame_indexes(project, pose_set)
    wanted = params.get("frames")
    if isinstance(wanted, (list, tuple)) and wanted:
        want = {int(v) for v in wanted}
        indexes = [i for i in all_idx if i in want]
        missing = sorted(want - set(all_idx))
        if missing:
            raise PoseBatchError(f"姿势集里没有第 {missing[0]} 帧")
    else:
        indexes = list(all_idx)
    if not indexes:
        raise PoseBatchError("这个姿势集里没有帧")

    candidates = max(1, min(poseset.MAX_CANDIDATES, int(params.get("candidates") or 1)))
    total = len(indexes) * candidates
    if total > MAX_IMAGES:
        raise PoseBatchError(
            f"这次要出 {total} 张（{len(indexes)} 帧 × {candidates} 候选），"
            f"超过单次上限 {MAX_IMAGES} 张：减少候选数，或分批跑")

    char_image = str(params.get("char_image") or "")
    char_images = [str(p) for p in (params.get("char_images") or []) if str(p or "").strip()]
    if not char_image and char_images:
        char_image = char_images[0]
    if not char_image:
        raise PoseBatchError("缺角色图：角色迁移要先接一张角色图")

    base_seed = params.get("seed")
    base_seed = int(base_seed) if base_seed is not None and int(base_seed) >= 0 else -1
    common = {
        "prompt": str(params.get("prompt") or ""),
        "negative": params.get("negative") or None,
        "width": int(params.get("width") or 768),
        "height": int(params.get("height") or 768),
        "steps": int(params.get("steps") or 16),
        "cfg": float(params.get("cfg") or 1.0),
        "fill": float(params.get("fill") or 0.8),
        "thr": float(params.get("thr") or 0.25),
        "ref_mode": str(params.get("ref_mode") or "auto"),
        "bg": str(params.get("bg") or "white"),
        "model": params.get("model") or None,
        "matte": bool(params.get("matte")),
        "matte_backend": str(params.get("matte_backend") or "auto"),
        "pixel": int(params.get("pixel") or 0),
        "palette": int(params.get("palette") or 0),
        "denoise": int(params.get("denoise") or 1),
        "clean_iso": int(params.get("clean_iso") or 1),
        "char_images": char_images or None,
    }
    if common["model"] is None:
        common.pop("model")

    done = 0
    failed: List[Dict[str, Any]] = []
    started = time.time()
    for idx in indexes:
        pose_path = poseset.frame_source_path(project, pose_set, idx)
        for c in range(candidates):
            if job is not None:
                job["progress"] = {
                    "done": done, "total": total,
                    "current": f"第 {idx} 帧 · 候选 {c + 1}/{candidates}",
                }
            # 每个候选用不同种子（同种子出的是同一张）。base_seed=-1 时交给
            # `pose_clone` 自己随机（它内部 `torch.randint`）。
            seed = (base_seed + c) if base_seed >= 0 else -1
            try:
                res = gen.pose_clone(
                    engine, pose_image=pose_path, char_image=char_image,
                    seed=seed, scope=project, tag="wpcb", **common)
            except Exception as exc:                     # noqa: BLE001
                failed.append({"frame": idx, "candidate": c + 1,
                               "error": f"{type(exc).__name__}: {exc}"})
                done += 1
                continue

            src = Path(str(res.get("file") or ""))
            try:
                poseset.add_candidate(project, render, idx, src,
                                      seed=(res.get("meta") or {}).get("seed"))
            except poseset.PoseSetError as exc:
                failed.append({"frame": idx, "candidate": c + 1, "error": str(exc)})
            # `pose_clone` 会顺手写 sidecar / raw，候选已经搬进成片集，
            # 这两份留在项目根目录只会让目录越来越乱
            for extra in (res.get("sidecar"), res.get("raw")):
                try:
                    if extra:
                        Path(str(extra)).unlink(missing_ok=True)
                except OSError:
                    pass
            done += 1

    if job is not None:
        job["progress"] = {"done": done, "total": total, "current": ""}
    return {
        "render": render,
        "pose_set": pose_set,
        "frames": indexes,
        "candidates": candidates,
        "images": done - len(failed),
        "failed": failed,
        "seconds": round(time.time() - started, 1),
    }
