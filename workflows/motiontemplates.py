"""动作模板 registry —— 把「白模 × 动作 × 方向 × 视角模式」变成可复用资产。

借鉴 ai_game_workbench 的 ``ACTION_DEFINITIONS``
（``routes/godotExport.ts:43-75``，``{action, sourceSegments, fps, loop}``），
但按我们的链路重做：那边是「一条动画一个硬编码常数」，这里是**角色级可复用
描述符**，并且不碰视频、不碰云端、不碰官方 VNCCS 的目录契约。

## 为什么需要这一层

方向帧的元数据链（``set.json.direction`` → posegen → atlas → Godot）已经通了，
但每一步的「这条动作叫什么 / 几帧 / 多少 fps / 循环还是单次 / 哪个机位」还散在
API 常数、前端输入框和用户脑子里。模板把这些收成一处：

  * 抽帧面板选模板 → 自动填帧数与机位模式，``templateId`` 写进 ``set.json.source``；
  * Godot 导出未显式给 fps/loop 时，顺着 ``poseSet`` 反查模板取值；
  * 试玩验收台按模板说明方向覆盖与动作时序。

## 存储

用户模板落 ``library/motion_templates.json``（``paths.LIBRARY_DIR``，测试由
``MEOW_LIBRARY_DIR`` 覆盖），与提示词收藏 / 界面偏好同一处，**只存元数据不存图**。
内置预设是下面的常量，**不落盘**：升级时不会跟用户改过的旧文件冲突。

## 成本

逐帧 Qwen 是这条链里唯一昂贵的环节。按项目既有口径
``t / (1 - (1-p)^k)``（t=单帧秒数，p=通过率，k=候选数）估时；横板靠镜像
省掉一半机位。估算只给提示，不替用户决定跑不跑。
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import directions as dirs
from . import paths

TEMPLATES_FILE = "motion_templates.json"
MAX_TEMPLATES = 100
MAX_ACTIONS = 24
MAX_FRAMES_PER_ACTION = 240

_LOCK = threading.RLock()
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_VIEW_MODES = frozenset(dirs.DIRECTIONS)


class TemplateError(ValueError):
    """模板校验 / 存储的可读错误（API 转 400）。"""


# ---------------------------------------------------------------- 内置预设


def _action(name: str, clip: str, frames: int, fps: float,
            loop: bool = True, one_shot: bool = False) -> Dict[str, Any]:
    return {"name": name, "clip": clip, "frames": frames, "fps": fps,
            "loop": loop, "oneShot": one_shot}


#: 随项目发行、**不落盘**的三套起步模板。动作名与方向 key 保持英文原值
#: （项目约定：标识值原样传，只在渲染处显示中文）。
BUILTIN_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "topdown-walk-4",
        "name": "俯视 RPG · 四方向基础",
        "viewMode": "topdown-4",
        "source": {"kind": "glb", "file": "", "clip": "walk"},
        "camera": {"pitch": 45, "ortho": False},
        "actions": [
            _action("idle", "idle", 4, 8),
            _action("walk", "walk", 8, 12),
            _action("run", "run", 8, 14),
            _action("attack", "attack", 6, 10, loop=False, one_shot=True),
        ],
        "pivot": {"x": 0.5, "y": 1.0},
        "qwenPromptContract": "top-down 45 degree view, full body, consistent outfit",
        "budget": {"candidates": 1, "singleFrameSeconds": 26, "passRate": 0.75},
        "subjects": 1,
        "note": "四机位 0°/90°/180°/270°，斜俯视；attack 是单次动作不循环。",
    },
    {
        "id": "topdown-full-8",
        "name": "俯视 RPG · 八方向全套",
        "viewMode": "topdown-8",
        "source": {"kind": "glb", "file": "", "clip": "walk"},
        "camera": {"pitch": 45, "ortho": False},
        "actions": [
            _action("idle", "idle", 4, 8),
            _action("walk", "walk", 8, 12),
            _action("run", "run", 8, 14),
            _action("attack", "attack", 6, 10, loop=False, one_shot=True),
        ],
        "pivot": {"x": 0.5, "y": 1.0},
        "qwenPromptContract": "top-down 45 degree view, eight facing directions, consistent outfit",
        "budget": {"candidates": 1, "singleFrameSeconds": 26, "passRate": 0.75},
        "subjects": 1,
        "note": "YAWS 全 8 机位，成本是四方向的两倍；生成前先看估算。",
    },
    {
        "id": "side-platformer-2",
        "name": "横板 · 左右 + 镜像",
        "viewMode": "side-2",
        "source": {"kind": "glb", "file": "", "clip": "walk"},
        "camera": {"pitch": 0, "ortho": True},
        "actions": [
            _action("idle", "idle", 4, 8),
            _action("walk", "walk", 8, 12),
            _action("run", "run", 8, 14),
            _action("jump", "jump", 6, 10, loop=False, one_shot=True),
            _action("attack", "attack", 6, 12, loop=False, one_shot=True),
            _action("hit", "hit", 3, 10, loop=False, one_shot=True),
        ],
        "pivot": {"x": 0.5, "y": 1.0},
        "qwenPromptContract": "side view, orthographic, full body, facing left; right is mirrored",
        "budget": {"candidates": 1, "singleFrameSeconds": 26, "passRate": 0.75},
        "subjects": 1,
        "note": "只出左向，右向水平镜像派生；脚底锚点必须在画面底边，跳跃才不飘。",
    },
]


# ---------------------------------------------------------------- 读写


def _path() -> Path:
    return paths.LIBRARY_DIR / TEMPLATES_FILE


def _read_user() -> List[Dict[str, Any]]:
    p = _path()
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return []
    items = doc.get("items") if isinstance(doc, dict) else doc
    if isinstance(items, dict):       # 兼容早期 {id: template} 形态
        items = list(items.values())
    return [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []


def _write_user(items: List[Dict[str, Any]]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "items": items},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    with _LOCK:
        os.replace(str(tmp), str(p))


def _new_id() -> str:
    return f"mt{int(time.time() * 1000):x}{secrets.token_hex(2)}"


# ---------------------------------------------------------------- 校验与归一


def _as_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def _as_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def normalize(doc: Dict[str, Any], *, builtin: bool = False) -> Dict[str, Any]:
    """把任意输入收敛成一个合法模板（缺省补齐、脏值钳制、绝不静默改语义）。"""
    if not isinstance(doc, dict):
        raise TemplateError("模板必须是对象")
    tid = str(doc.get("id") or "").strip().lower() or _new_id()
    if not _ID_RE.match(tid):
        raise TemplateError("模板 id 只能用小写字母、数字、- 和 _，且不超过 48 字符")
    name = str(doc.get("name") or "").strip()
    if not name:
        raise TemplateError("模板要有名字")
    mode = str(doc.get("viewMode") or "").strip()
    if mode not in _VIEW_MODES:
        raise TemplateError(f"viewMode 必须是 {' / '.join(sorted(_VIEW_MODES))}")

    required = dirs.direction_keys(mode)
    raw_dirs = doc.get("directions")
    keys = ([dirs.validate_direction(d) for d in raw_dirs]
            if isinstance(raw_dirs, list) and raw_dirs else list(required))
    keys = [k for k in keys if k]
    if not keys:
        raise TemplateError("directions 里没有合法的方向 key")

    raw_actions = doc.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise TemplateError("模板至少要有 1 个动作")
    actions: List[Dict[str, Any]] = []
    for i, a in enumerate(raw_actions[:MAX_ACTIONS], 1):
        if not isinstance(a, dict):
            raise TemplateError(f"第 {i} 个动作格式不对")
        aname = str(a.get("name") or "").strip()
        if not aname:
            raise TemplateError(f"第 {i} 个动作要有名字")
        one_shot = bool(a.get("oneShot") or (a.get("loop") is False))
        actions.append({
            "name": aname,
            "clip": str(a.get("clip") or aname).strip(),
            "frames": _as_int(a.get("frames"), 8, 1, MAX_FRAMES_PER_ACTION),
            "fps": _as_float(a.get("fps"), 12.0, 0.1, 60.0),
            # 单次动作永远不循环（对方项目 loop / oneShot 分裂过）
            "loop": False if one_shot else bool(a.get("loop", True)),
            "oneShot": one_shot,
        })
    names = [a["name"] for a in actions]
    if len(names) != len(set(names)):
        raise TemplateError("动作名不能重复")

    source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    camera = doc.get("camera") if isinstance(doc.get("camera"), dict) else {}
    pivot = doc.get("pivot") if isinstance(doc.get("pivot"), dict) else {}
    budget = doc.get("budget") if isinstance(doc.get("budget"), dict) else {}
    mirror = (doc.get("mirrorFrom") if isinstance(doc.get("mirrorFrom"), dict)
              else (dirs.MIRROR_FROM if mode == "side-2" else {}))
    mirror = {k: dirs.validate_direction(v) for k, v in mirror.items()}
    mirror = {k: v for k, v in mirror.items()
              if k in dirs.DIRECTION_KEYS and v} or {}

    return {
        "id": tid,
        "name": name,
        "viewMode": mode,
        "builtin": bool(builtin),
        "source": {
            "kind": str(source.get("kind") or "glb"),
            "file": str(source.get("file") or ""),
            "clip": str(source.get("clip") or actions[0]["clip"]),
        },
        "directions": keys,
        "yawMap": {k: dirs.yaw_of(k) for k in keys},
        "camera": {
            "pitch": _as_float(camera.get("pitch"), 45 if mode != "side-2" else 0,
                               -90, 90),
            "ortho": bool(camera.get("ortho", mode == "side-2")),
        },
        "mirrorFrom": mirror,
        "actions": actions,
        "pivot": {
            "x": _as_float(pivot.get("x"), 0.5, 0, 1),
            "y": _as_float(pivot.get("y"), 1.0, 0, 1),
        },
        "qwenPromptContract": str(doc.get("qwenPromptContract") or ""),
        "budget": {
            "candidates": _as_int(budget.get("candidates"), 1, 1, 8),
            "singleFrameSeconds": _as_float(budget.get("singleFrameSeconds"), 26, 1, 3600),
            "passRate": _as_float(budget.get("passRate"), 0.75, 0.01, 1.0),
        },
        "subjects": _as_int(doc.get("subjects"), 1, 1, 8),
        "note": str(doc.get("note") or ""),
    }


# ---------------------------------------------------------------- 查询


def builtins() -> List[Dict[str, Any]]:
    """内置预设每次现算（常量里是紧凑写法，经同一套 normalize 校验）。"""
    return [normalize(t, builtin=True) for t in BUILTIN_TEMPLATES]


def list_templates() -> List[Dict[str, Any]]:
    """内置在前，用户模板在后；同 id 时用户模板不能覆盖内置。"""
    out = builtins()
    ids = {t["id"] for t in out}
    for raw in _read_user():
        try:
            t = normalize(raw)
        except TemplateError:
            continue                       # 手改坏一条不该让整个 registry 打不开
        if t["id"] not in ids:
            t["builtin"] = False
            out.append(t)
            ids.add(t["id"])
    return out


def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    return next((t for t in list_templates() if t["id"] == template_id), None)


def save_template(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """新建或更新用户模板；**内置 id 不可覆盖**（要改就另存一份）。"""
    t = normalize(doc)
    if any(b["id"] == t["id"] for b in builtins()):
        raise TemplateError(f"「{t['id']}」是内置模板，不能覆盖；换个 id 另存")
    with _LOCK:
        items = _read_user()
        idx = next((i for i, x in enumerate(items)
                    if str(x.get("id") or "") == t["id"]), -1)
        if idx >= 0:
            items[idx] = t
        else:
            if len(items) >= MAX_TEMPLATES:
                raise TemplateError(f"最多存 {MAX_TEMPLATES} 套模板")
            items.append(t)
        _write_user(items)
    return list_templates()


def delete_template(template_id: str) -> List[Dict[str, Any]]:
    if any(b["id"] == template_id for b in builtins()):
        raise TemplateError("内置模板不能删")
    with _LOCK:
        items = [x for x in _read_user()
                 if str(x.get("id") or "") != template_id]
        _write_user(items)
    return list_templates()


# ---------------------------------------------------------------- 成本 / 播放参数


def estimate(template: Dict[str, Any]) -> Dict[str, Any]:
    """逐帧 Qwen 的成本估算：``t / (1 - (1-p)^k)``（项目既有口径）。

    横板的 ``mirrorFrom`` 能吃掉一半机位 —— 少生成一个方向就少一半图。
    """
    t = normalize(template)
    b = t["budget"]
    k, single, p = b["candidates"], b["singleFrameSeconds"], b["passRate"]
    per_frame = single / (1 - (1 - p) ** k)
    render_dirs = max(1, len(t["directions"]) - len(t["mirrorFrom"]))
    frames = sum(a["frames"] for a in t["actions"])
    images = frames * render_dirs * k
    return {
        "templateId": t["id"],
        "actions": len(t["actions"]),
        "framesPerDirection": frames,
        "directions": len(t["directions"]),
        "renderDirections": render_dirs,
        "mirroredDirections": sorted(t["mirrorFrom"]),
        "candidates": k,
        "images": images,
        "perFrameSeconds": round(per_frame, 2),
        "totalSeconds": round(images * per_frame, 1),
        "totalMinutes": round(images * per_frame / 60, 1),
        "subjects": t["subjects"],
    }


def playback(template: Dict[str, Any], action: str = "") -> Dict[str, Any]:
    """给导出链用：某个动作（默认第一个）的 fps / loop / oneShot。"""
    t = normalize(template)
    a = next((x for x in t["actions"] if x["name"] == action), t["actions"][0])
    return {"fps": a["fps"], "loop": a["loop"], "oneShot": a["oneShot"],
            "action": a["name"], "templateId": t["id"],
            "viewMode": t["viewMode"]}


def template_for_pose_set(project: str, pose_set: str, *,
                          character: str = "", stage: str = "") -> Optional[Dict[str, Any]]:
    """顺着画布项目里的 ``poses/<集>/set.json`` 找 ``source.templateId``。

    抽帧面板选了模板就把 id 写进 ``set.json.source``；成片集又记着 ``poseSet``，
    所以 Godot 导出不用用户再填一遍 fps/loop。找不到就返回 None，调用方退回默认。
    """
    from . import poseset

    if not pose_set or not (project or character):
        return None
    doc = poseset._read_json(                                  # noqa: SLF001
        poseset._sub(project, poseset.POSES_DIR, pose_set,    # noqa: SLF001
                     character=character, stage=stage)
        / poseset.SET_FILE)
    source = doc.get("source") if isinstance(doc, dict) else {}
    tid = str((source or {}).get("templateId") or "") if isinstance(source, dict) else ""
    return get_template(tid) if tid else None


def summary(templates: Iterable[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """列表页用：模板 + 每套的成本，一次请求给全，前端不用 N+1。"""
    items = list(templates) if templates is not None else list_templates()
    return {
        "version": 1,
        "count": len(items),
        "items": items,
        "estimates": {t["id"]: estimate(t) for t in items},
    }
