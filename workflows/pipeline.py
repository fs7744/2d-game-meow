"""一键出全套 —— 把「换内衣 → 出姿势 → 出表情」按官方顺序串成流水线。

## 为什么要有这一层

官方的 `character_generator.py` 本来就是**一条流水线**，`order` 写死在
`:3115-3123`：

```
original_pose_generation → original_upscaler → original_bg_remove
→ remove_clothes
→ naked_pose_generation → naked_upscaler → naked_bg_remove
```

（表情是另一条，`emotion_generator_v2.py`，在姿势帧之上跑。）

我们这边原本是一个个独立按钮：用户得自己记住"先换内衣、再出姿势、最后出表情"，
顺序错了出来的图就不能用（比如先出表情再出姿势，表情全白做）。而且每一步都要
回来点一次、等一次。

这一层只做**编排**：不重新实现任何一步，就是按顺序调用
`clothesgen.remove_clothes` / `posegen.run_batch` / `emotiongen.run_batch`，
把它们的进度换算成一条全局进度条。

## 没搬的两段

- **upscaler**（`SeedVR2`）：明确不移植（见 `docs/角色中心-VNCCS移植评估.md`）。
  我们的 Qwen 直出分辨率已经够，再放大没有收益还要吃显存。
- **original_*** 那一支（原图出姿势）：官方区分"原图分支"和"裸体分支"，
  各出一套。我们用 `costume` 参数表达同一件事，不需要两条。

## 失败策略

默认 `continue_on_error=True`：某一步失败不影响后面的步骤（比如表情里有 2 个
认不出来，不该让整条流水线崩掉，用户还等着看姿势帧）。失败的步骤记进
`failed_steps`，前端能看见是哪一步、什么原因。

关掉的话（`False`）遇到第一个错误就抛 —— 适合"要么全对要么别动"的场景。
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from workflows import characters
from workflows import clothesgen
from workflows import emotiongen
from workflows import posegen


class PipelineError(RuntimeError):
    """编排层的可读错误（API 转 400）。"""


#: 三步的默认开关。**顺序就是执行顺序**，不能乱：
#: 先得到干净底图（内衣），姿势从它出发，表情再叠在姿势帧上。
PIPELINE_DEFAULTS: Dict[str, Any] = {
    "underwear": False,        # 换白内衣底图（官方 remove_clothes）
    # 出不出姿势看**有没有给姿势来源**（`poses` 列表或 `pose_dir`），没有单独的开关：
    # 只出表情的用户不该被迫先跑一遍姿势。
    "emotions": [],            # 表情列表（空 = 跳过）
    "steps": 20,               # 姿势采样
    "cfg": 1.0,
    "emotion_steps": 25,       # 表情采样（官方 :4136 用 25 / 3.0）
    "emotion_cfg": 3.0,
    "emotion_target_size": 2048,
    # **抠图默认关**：QI2 自己能出透明底（原生 alpha），多跑一遍分割模型只会
    # 啃掉头发和挂饰的边缘。跑不跑是用户的选择 —— 前端有「抠背景」复选框。
    "matte": False,
    "transparent": False,
    "continue_on_error": True,
}

#: 内衣底图存成哪套服装（后续姿势从它出发）
UNDERWEAR_COSTUME = "Underwear"


class _StepProgress(dict):
    """子步骤的 job 代理：把它的 progress 换算成流水线的全局进度。

    子步骤（`run_batch` 之类）只会 `job["progress"] = {"done":..., "total":...}`，
    不知道自己在整条流水线的哪个位置。这里拦下这次赋值，按 (起点, 权重) 换算。

    继承 `dict` 而不是包一层：子步骤除了赋值还会 `job.get(...)` / 判断 `is not None`，
    包一层对象反而要转发一堆协议。
    """

    def __init__(self, parent: Optional[Dict[str, Any]], base: float, weight: float,
                 key: str, label: str, total: float) -> None:
        super().__init__()
        self._parent = parent
        self._base = float(base)
        self._weight = float(weight)
        self._key = key          # 程序用的稳定标识（underwear / poses / emotions）
        self._label = label      # 给人看的
        self._total = float(total)

    def __setitem__(self, key: str, value: Any) -> None:
        if key != "progress":
            super().__setitem__(key, value)
            return
        payload = value if isinstance(value, dict) else {}
        done = float(payload.get("done") or 0.0)
        total = float(payload.get("total") or 1.0)
        frac = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
        if self._parent is None:      # 没有 job 就只记账，不发布
            return
        current = str(payload.get("current") or "").strip()
        self._parent["progress"] = {
            "done": round(self._base + self._weight * frac, 2),
            "total": self._total,
            "step": self._key,
            "current": f"{self._label}：{current}" if current else self._label,
        }


def plan_steps(name: str, costume: str,
               params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """算出要跑哪几步、每步的权重（权重 = 预计张数，用来分进度）。

    **必须在开跑前算完**：姿势会替换这套服装的帧，跑完再算表情的源帧数就跟
    开跑时的预期不一样了。估算偏一点只会让进度条最后跳一下，比"分母随时变"好。
    """
    steps: List[Dict[str, Any]] = []
    if params.get("underwear"):
        steps.append({"key": "underwear", "label": "换白内衣底图", "weight": 1})

    poses = posegen.resolve_pose_paths(
        params.get("poses") or [], str(params.get("pose_dir") or ""))
    if params.get("poses") or params.get("pose_dir"):
        # 给了姿势来源就一定要跑这一步（哪怕解析出来是空，也要让它报错说出来）
        steps.append({"key": "poses", "label": "出多姿势", "weight": max(1, len(poses))})

    emotes = params.get("emotions") or []
    if emotes:
        # 源帧数：跑了姿势就用姿势数（那是即将产出的帧），否则看现有的 Neutral 帧
        if any(s["key"] == "poses" for s in steps):
            sources = max(1, len(poses))
        else:
            try:
                sources = max(1, len(characters.source_sprite_paths(name, costume)))
            except (OSError, ValueError):
                sources = 1
        steps.append({"key": "emotions", "label": "出表情",
                      "weight": sources * len(emotes)})

    if not steps:
        raise PipelineError(
            "没有要做的步骤：至少勾一个（换内衣 / 出姿势 / 出表情）")
    return steps


def run_pipeline(
    engine: Any,
    params: Dict[str, Any],
    *,
    job: Optional[Dict[str, Any]] = None,
    underwear_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    pose_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    emotion_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    matte_fn: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """按官方顺序跑完整条流水线，返回每一步的摘要。

    `*_fn` 是给测试用的注入点（不碰 GPU 就能验编排与进度换算）。
    默认分别走 `clothesgen.remove_clothes` / `posegen.run_batch` /
    `emotiongen.run_batch`。
    """
    name = characters.ensure_safe_name(str(params.get("character") or ""), "角色名")
    costume = characters.ensure_safe_name(
        str(params.get("costume") or "Naked"), "服装名")
    if not characters.character_exists(name):
        raise PipelineError(f"没有这个角色：{name}")

    steps = plan_steps(name, costume, params)
    total_weight = float(sum(s["weight"] for s in steps))

    underwear_fn = underwear_fn or clothesgen.remove_clothes
    pose_fn = pose_fn or posegen.run_batch
    emotion_fn = emotion_fn or emotiongen.run_batch
    continue_on_error = bool(params.get("continue_on_error", True))

    seed = params.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(
        os.urandom(3), "big")
    do_matte = bool(params.get("matte"))
    transparent = bool(params.get("transparent"))

    if job is not None:
        job["progress"] = {"done": 0.0, "total": total_weight,
                           "step": steps[0]["key"], "current": steps[0]["label"]}

    started = time.time()
    results: Dict[str, Any] = {}
    failed_steps: List[Dict[str, Any]] = []
    done_weight = 0.0
    # 姿势从哪套服装的底图出发：跑了内衣就用它，否则用目标服装
    source_costume = ""

    for step in steps:
        key, label, weight = step["key"], step["label"], float(step["weight"])
        proxy = _StepProgress(job, done_weight, weight, key, label, total_weight)
        try:
            if key == "underwear":
                out = underwear_fn(
                    engine,
                    {"character": name, "costume": UNDERWEAR_COSTUME,
                     "matte": do_matte, "seed": seed},
                    job=proxy)
                source_costume = UNDERWEAR_COSTUME
                results["underwear"] = {
                    "costume": out.get("costume") or UNDERWEAR_COSTUME,
                    "sprite": out.get("sprite"),
                    "seconds": out.get("seconds"),
                }
            elif key == "poses":
                out = pose_fn(
                    engine,
                    {"character": name, "costume": costume,
                     "poses": params.get("poses") or [],
                     "pose_dir": params.get("pose_dir") or "",
                     "prompts": params.get("poses_prompt") or "",
                     "steps": int(params.get("steps") or PIPELINE_DEFAULTS["steps"]),
                     "cfg": float(params.get("cfg") or PIPELINE_DEFAULTS["cfg"]),
                     "seed": seed, "matte": do_matte,
                     "transparent": transparent,
                     "background_color": params.get("background_color") or "",
                     "source_costume": source_costume},
                    job=proxy,
                    matte_fn=matte_fn)
                results["poses"] = {
                    "count": out.get("count", 0),
                    "sprites": out.get("sprites") or [],
                    "seconds": out.get("seconds"),
                }
            elif key == "emotions":
                out = emotion_fn(
                    engine,
                    {"character": name, "costume": costume,
                     "emotions": params.get("emotions") or [],
                     "steps": int(params.get("emotion_steps")
                                  or PIPELINE_DEFAULTS["emotion_steps"]),
                     "cfg": float(params.get("emotion_cfg")
                                  or PIPELINE_DEFAULTS["emotion_cfg"]),
                     "target_size": int(params.get("emotion_target_size")
                                        or PIPELINE_DEFAULTS["emotion_target_size"]),
                     "seed": seed, "matte": do_matte,
                     "transparent": transparent},
                    job=proxy, matte_fn=matte_fn)
                results["emotions"] = {
                    "count": out.get("count", 0),
                    "total": out.get("total", 0),
                    "failed": out.get("failed") or [],
                    "seconds": out.get("seconds"),
                }
        except Exception as exc:                          # noqa: BLE001
            failed_steps.append({"step": key, "label": label,
                                 "error": f"{type(exc).__name__}: {exc}"})
            if not continue_on_error:
                raise PipelineError(f"{label}失败：{exc}") from exc
        done_weight += weight
        if job is not None:
            job["progress"] = {"done": round(done_weight, 2), "total": total_weight,
                               "step": key, "current": f"{label}完成"}

    if job is not None:
        job["progress"] = {"done": total_weight, "total": total_weight,
                           "step": "", "current": ""}

    return {
        "character": name, "costume": costume,
        "steps": [s["key"] for s in steps],
        "underwear_costume": source_costume,
        "seed": seed,
        "underwear": results.get("underwear"),
        "poses": results.get("poses"),
        "emotions": results.get("emotions"),
        "failed_steps": failed_steps,
        "seconds": round(time.time() - started, 2),
    }
