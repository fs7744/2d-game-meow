"""2D cutout 骨骼数据模型：骨架 / 绑定 / 动作（纯数据，无渲染）。

移植自 FrameBaker（MIT，`D:\\code\\gen\\others\\FrameBaker-main`）的
`packages/shared/src/animation.ts`，**按 2D cutout 的需要砍掉 3D 专属部分**：

| FrameBaker | 这里 | 为什么砍 |
|---|---|---|
| `Transform{translation:Vec3, rotation:Quaternion, scale:Vec3}` | `{x, y, rotation}` | cutout 只有平面旋转，四元数在 2D 里退化成一个角度 |
| 16 元素列优先仿射矩阵 | 不存 | 求值交给 Godot；预览用「平移 + 旋转 + 缩放」直接合成 |
| `warp` / `deform` 轨道 | 不做 | 网格形变要蒙皮权重（Godot 侧是 Polygon2D + Skeleton2D + 权重），
属于另一条线，先别混进来 |
| `Vec3` 关键帧 | `float` / `[x, y]` | 同上 |

**保留的语义**（这几条才是 FrameBaker 真正值钱的部分，别自己另发明一套）：

1. `schemaVersion` + `kind` 显式标在每份文件头上，**未知字段直接报错**——不做
   「静默忽略」，否则版本漂移会安静地导出错东西（FrameBaker 的 `json.ts` 就是
   这个态度）。
2. 骨骼用 `id` + `parentId` 表达层级，`rest` 是**相对父骨**的变换。
3. 轨道用 `targetId` + `property` 指向「哪根骨的哪个属性」，和 Godot
   AnimationPlayer 的 `NodePath("骨名:属性")` **一一对应**——导出是纯翻译，不做映射表。
4. 关键帧插值 `step` / `linear` / `cubic-bezier`（Godot 的 interp = 0 / 1 / 2）。

**坐标系**：和 Godot 2D 完全一致 —— x 向右、**y 向下**、旋转顺时针为正、单位弧度。
不要改成数学坐标（y 向上）：那会在导出、预览、切件三处各写一次取反，
这类转换本项目已经踩过（姿态图的 y 轴）。

**轨道里存的是绝对值，不是相对 rest 的增量**。原因：Godot 的 AnimationPlayer
写的就是 `position` / `rotation` 的绝对值，存增量的话导出时得把 rest 加回去，
而「rest 改了、动画忘了同步」这种错误没人能一眼看出来。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

#: 每骨可被动画驱动的属性 → (值的长度, 类型说明)。
#:
#: 只有这四个。加属性前先确认 Godot 侧有同名属性，否则导出的轨道是死轨
#: （Godot 会静默忽略无效属性路径，播放时什么都不动，很难查）。
PROPERTIES: Dict[str, Tuple[int, str]] = {
    "position": (2, "像素坐标 [x, y]，相对父骨"),
    "rotation": (1, "弧度，顺时针为正"),
    "scale": (2, "缩放 [sx, sy]，1 = 原尺寸"),
    "visible": (1, "0 / 1，切换部件显隐（换方向时用）"),
}

#: 插值方式 → Godot `Animation` 的 `interp` 值。导出时直接查这张表。
INTERP_TO_GODOT = {"step": 0, "linear": 1, "cubic-bezier": 2}

MAX_BONES = 256
MAX_KEYFRAMES_PER_TRACK = 4096
MAX_NAME = 64


class RigError(ValueError):
    """数据不合法（调用方转 400，文案要能直接给用户看）。"""


# ---------------------------------------------------------------- 基础校验


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RigError(f"{what} 必须是数字，收到 {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise RigError(f"{what} 不能是 NaN / inf")
    return f


def _int(v: Any, what: str, lo: Optional[int] = None,
         hi: Optional[int] = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise RigError(f"{what} 必须是整数，收到 {type(v).__name__}")
    if lo is not None and v < lo:
        raise RigError(f"{what} 不能小于 {lo}（收到 {v}）")
    if hi is not None and v > hi:
        raise RigError(f"{what} 不能大于 {hi}（收到 {v}）")
    return v


def _id(v: Any, what: str) -> str:
    """一律 ASCII 标识符：这些 id 会变成 Godot 里的**节点名**。"""
    s = str(v or "").strip()
    if not s:
        raise RigError(f"{what} 不能为空")
    if len(s) > 128:
        raise RigError(f"{what} 太长（最多 128 字符）")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", s):
        raise RigError(
            f"{what} 只能用字母、数字、下划线、点、连字符（收到 {s!r}）——"
            "它会变成 Godot 的节点名，中文和空格在 NodePath 里要转义，容易出错")
    return s


def _name(v: Any, what: str, default: str = "") -> str:
    """显示名可以是中文（Godot 的动画名允许）。"""
    s = str(v if v is not None else default).strip()
    return s[:MAX_NAME]


def _strict_keys(doc: Dict[str, Any], allowed: Iterable[str], what: str) -> None:
    """未知字段直接报错。**这是故意的**：静默忽略会让写错字段名的配置看起来
    “生效了”，实际用的是默认值——排查时能烧掉半天。"""
    extra = sorted(set(doc) - set(allowed))
    if extra:
        raise RigError(f"{what} 有不认识的字段：{'、'.join(extra)}"
                       f"（允许：{'、'.join(sorted(allowed))}）")


# ---------------------------------------------------------------- 骨架


def bone(bone_id: str, parent: Optional[str] = None, *, x: float = 0.0,
         y: float = 0.0, rotation: float = 0.0, name: str = "") -> Dict[str, Any]:
    """建一根骨。`x/y/rotation` 是**相对父骨**的静止姿势。"""
    return {
        "id": _id(bone_id, "骨 id"),
        "name": _name(name or bone_id, "骨名"),
        "parentId": parent,
        "rest": {"x": float(x), "y": float(y), "rotation": float(rotation)},
    }


def new_skeleton(skeleton_id: str, name: str,
                 bones: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "skeleton",
        "id": _id(skeleton_id, "骨架 id"),
        "name": _name(name or skeleton_id, "骨架名"),
        "coordinateSystem": {"unit": "pixel", "xAxis": "right",
                             "yAxis": "down", "rotation": "clockwise-radian"},
        "bones": list(bones),
    }


def validate_skeleton(doc: Any) -> Dict[str, Any]:
    """校验骨架：版本、字段、id 唯一、父骨存在、无环、rest 合法。"""
    if not isinstance(doc, dict):
        raise RigError("骨架必须是 JSON 对象")
    _strict_keys(doc, ("schemaVersion", "kind", "id", "name",
                       "coordinateSystem", "bones", "semanticProfile"), "骨架")
    if doc.get("kind") != "skeleton":
        raise RigError(f"kind 必须是 skeleton（收到 {doc.get('kind')!r}）")
    if _int(doc.get("schemaVersion"), "schemaVersion") != SCHEMA_VERSION:
        raise RigError(
            f"骨架版本是 {doc.get('schemaVersion')}，本代码只认 {SCHEMA_VERSION}："
            "升级了格式就同步改这里，别悄悄兼容")

    bones = doc.get("bones")
    if not isinstance(bones, list) or not bones:
        raise RigError("骨架里一根骨都没有（bones 不能为空）")
    if len(bones) > MAX_BONES:
        raise RigError(f"骨太多（{len(bones)} > {MAX_BONES}）：2D cutout 用不到这么细")

    seen: Dict[str, Dict[str, Any]] = {}
    for i, b in enumerate(bones):
        if not isinstance(b, dict):
            raise RigError(f"第 {i} 根骨不是对象")
        _strict_keys(b, ("id", "name", "parentId", "rest", "tipOffset", "semantic"),
                     f"第 {i} 根骨")
        bid = _id(b.get("id"), f"第 {i} 根骨的 id")
        if bid in seen:
            raise RigError(f"骨 id 重复：{bid}")
        rest = b.get("rest")
        if not isinstance(rest, dict):
            raise RigError(f"骨 {bid} 缺 rest")
        _strict_keys(rest, ("x", "y", "rotation"), f"骨 {bid} 的 rest")
        _num(rest.get("x", 0), f"骨 {bid} 的 rest.x")
        _num(rest.get("y", 0), f"骨 {bid} 的 rest.y")
        _num(rest.get("rotation", 0), f"骨 {bid} 的 rest.rotation")
        seen[bid] = b

    roots = 0
    for bid, b in seen.items():
        p = b.get("parentId")
        if p is None:
            roots += 1
            continue
        p = _id(p, f"骨 {bid} 的 parentId")
        if p not in seen:
            raise RigError(f"骨 {bid} 的父骨 {p} 不存在")
        if p == bid:
            raise RigError(f"骨 {bid} 把自己当父骨了")

    if roots != 1:
        raise RigError(
            f"骨架必须只有一个根骨（现在 {roots} 个）：多个根的话 Godot 里会变成"
            "平级节点，动一个不会带动另一个")

    # 环检测：每根骨沿父链往上，最多走 n 步
    for bid in seen:
        cur, hops = bid, 0
        while cur is not None and hops <= len(seen):
            cur = seen[cur].get("parentId")
            hops += 1
        if hops > len(seen):
            raise RigError(f"骨的父子关系成环了（从 {bid} 往上走不出去）")

    return doc


# ---------------------------------------------------------------- 绑定


def new_binding(skeleton_id: str, slots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "character-binding",
        "id": f"{skeleton_id}_binding",
        "name": f"{skeleton_id} 绑定",
        "skeletonId": _id(skeleton_id, "skeletonId"),
        "slots": list(slots),
    }


def validate_binding(doc: Any, skeleton: Dict[str, Any]) -> Dict[str, Any]:
    """校验绑定：每个 slot 挂的骨必须存在，pivot 合法，z 唯一性不强制但会排序。"""
    if not isinstance(doc, dict):
        raise RigError("绑定必须是 JSON 对象")
    _strict_keys(doc, ("schemaVersion", "kind", "id", "name", "skeletonId", "slots"),
                 "绑定")
    if doc.get("kind") != "character-binding":
        raise RigError(f"kind 必须是 character-binding（收到 {doc.get('kind')!r}）")
    if _int(doc.get("schemaVersion"), "schemaVersion") != SCHEMA_VERSION:
        raise RigError(f"绑定版本是 {doc.get('schemaVersion')}，本代码只认 {SCHEMA_VERSION}")
    if str(doc.get("skeletonId")) != str(skeleton.get("id")):
        raise RigError(f"绑定的 skeletonId（{doc.get('skeletonId')}）"
                       f"和骨架 id（{skeleton.get('id')}）对不上")

    bone_ids = {b["id"] for b in skeleton["bones"]}
    slots = doc.get("slots")
    if not isinstance(slots, list) or not slots:
        raise RigError("绑定里没有任何部件（slots 不能为空）")

    seen: set = set()
    for i, s in enumerate(slots):
        if not isinstance(s, dict):
            raise RigError(f"第 {i} 个 slot 不是对象")
        _strict_keys(s, ("id", "boneId", "texture", "pivot", "offset", "z", "name"),
                     f"第 {i} 个 slot")
        sid = _id(s.get("id"), f"第 {i} 个 slot 的 id")
        if sid in seen:
            raise RigError(f"slot id 重复：{sid}")
        seen.add(sid)
        bid = _id(s.get("boneId"), f"slot {sid} 的 boneId")
        if bid not in bone_ids:
            raise RigError(f"slot {sid} 挂到了不存在的骨：{bid}")
        tex = str(s.get("texture") or "").strip()
        if not tex:
            raise RigError(f"slot {sid} 没有贴图文件名")
        if ".." in tex.replace("\\", "/").split("/"):
            raise RigError(f"slot {sid} 的贴图路径不许带 ..（{tex}）")
        if s.get("pivot") not in ("top", "bottom", "center"):
            raise RigError(f"slot {sid} 的 pivot 必须是 top / bottom / center"
                           f"（收到 {s.get('pivot')!r}）")
        off = s.get("offset") or [0, 0]
        if not isinstance(off, (list, tuple)) or len(off) != 2:
            raise RigError(f"slot {sid} 的 offset 必须是 [x, y]")
        _num(off[0], f"slot {sid} 的 offset.x")
        _num(off[1], f"slot {sid} 的 offset.y")
        _int(s.get("z", 0), f"slot {sid} 的 z")
    return doc


# ---------------------------------------------------------------- 动作


def new_track(target: str, prop: str, keyframes: Sequence[Dict[str, Any]],
              interpolation: str = "linear") -> Dict[str, Any]:
    return {
        "targetId": _id(target, "轨道 targetId"),
        "property": prop,
        "interpolation": interpolation,
        "keyframes": list(keyframes),
    }


def new_motion(name: str, tracks: Sequence[Dict[str, Any]], *, fps: float = 24.0,
               loop: bool = True, duration: Optional[float] = None) -> Dict[str, Any]:
    """建一段动作。`duration` 不传就从关键帧最大时间推。"""
    lo = 0.0
    hi = 0.0
    for t in tracks:
        for k in t.get("keyframes") or []:
            v = float(k.get("time") or 0.0)
            lo = min(lo, v)
            hi = max(hi, v)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "motion-clip",
        "name": _name(name, "动作名"),
        "fps": float(fps),
        "loop": bool(loop),
        "duration": float(duration if duration is not None else max(hi, 1e-3)),
        "tracks": list(tracks),
    }


def _check_value(prop: str, value: Any, what: str) -> Any:
    """按属性检查关键帧的值形状。"""
    n, _ = PROPERTIES[prop]
    if prop == "rotation":
        return _num(value, what)
    if prop == "visible":
        v = _num(value, what)
        if v not in (0.0, 1.0):
            raise RigError(f"{what} 只能是 0 或 1（收到 {v}）")
        return v
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise RigError(f"{what} 必须是长度 {n} 的数组")
    return [_num(x, f"{what}[{i}]") for i, x in enumerate(value)]


def validate_motion(doc: Any, skeleton: Dict[str, Any]) -> Dict[str, Any]:
    """校验动作：目标骨存在、属性合法、关键帧时间单调、值形状对。"""
    if not isinstance(doc, dict):
        raise RigError("动作必须是 JSON 对象")
    _strict_keys(doc, ("schemaVersion", "kind", "name", "fps", "loop", "duration",
                       "tracks", "events", "rootMotionPolicy", "extensions"), "动作")
    if doc.get("kind") != "motion-clip":
        raise RigError(f"kind 必须是 motion-clip（收到 {doc.get('kind')!r}）")
    if _int(doc.get("schemaVersion"), "schemaVersion") != SCHEMA_VERSION:
        raise RigError(f"动作版本是 {doc.get('schemaVersion')}，本代码只认 {SCHEMA_VERSION}")
    fps = _num(doc.get("fps"), "fps")
    if not 0.1 <= fps <= 120:
        raise RigError(f"fps 要在 0.1~120 之间（收到 {fps}）")
    _num(doc.get("duration"), "duration")

    bone_ids = {b["id"] for b in skeleton["bones"]}
    tracks = doc.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise RigError("动作里一条轨道都没有（tracks 不能为空）")

    seen: set = set()
    for i, t in enumerate(tracks):
        if not isinstance(t, dict):
            raise RigError(f"第 {i} 条轨道不是对象")
        _strict_keys(t, ("targetId", "property", "interpolation", "keyframes"),
                     f"第 {i} 条轨道")
        tid = _id(t.get("targetId"), f"第 {i} 条轨道的 targetId")
        if tid not in bone_ids:
            raise RigError(f"第 {i} 条轨道指向不存在的骨：{tid}")
        prop = t.get("property")
        if prop not in PROPERTIES:
            raise RigError(f"骨 {tid} 的轨道属性 {prop!r} 不支持"
                           f"（可用：{'、'.join(PROPERTIES)}）")
        key = (tid, prop)
        if key in seen:
            raise RigError(f"骨 {tid} 的 {prop} 有两条轨道：同一属性只能一条，"
                           "要更多关键帧就往那条里加")
        seen.add(key)
        if t.get("interpolation") not in INTERP_TO_GODOT:
            raise RigError(f"骨 {tid} 的插值 {t.get('interpolation')!r} 不支持"
                           f"（可用：{'、'.join(INTERP_TO_GODOT)}）")
        kfs = t.get("keyframes")
        if not isinstance(kfs, list) or not kfs:
            raise RigError(f"骨 {tid} 的 {prop} 轨道没有关键帧")
        if len(kfs) > MAX_KEYFRAMES_PER_TRACK:
            raise RigError(f"骨 {tid} 的 {prop} 关键帧太多（{len(kfs)} > "
                           f"{MAX_KEYFRAMES_PER_TRACK}）")
        last = -1e18
        for j, k in enumerate(kfs):
            if not isinstance(k, dict):
                raise RigError(f"骨 {tid}/{prop} 的第 {j} 个关键帧不是对象")
            _strict_keys(k, ("time", "value"), f"骨 {tid}/{prop} 第 {j} 帧")
            tm = _num(k.get("time"), f"骨 {tid}/{prop} 第 {j} 帧的 time")
            if tm < last - 1e-9:
                raise RigError(f"骨 {tid}/{prop} 的关键帧时间不是递增的"
                               f"（第 {j} 帧 {tm} 小于上一帧 {last}）")
            last = tm
            _check_value(prop, k.get("value"), f"骨 {tid}/{prop} 第 {j} 帧的值")
    return doc


# ---------------------------------------------------------------- 求值


def _bezier(p1: float, p2: float, u: float) -> float:
    """一维 cubic-bezier(0, p1, 1, p2) 在参数 u 处的值（x 恒等特例）。"""
    return 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3


def _solve_bezier_x(x1: float, x2: float, x: float) -> float:
    """已知曲线 x 求参数 t（牛顿 + 二分兜底）。

    cubic-bezier 的 x(t) 未必单调（x1/x2 越界时会回折），FrameBaker 的做法是
    把控制点限制在 [0,1]——这里沿用：调用方已经校验过范围。
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if _bezier(x1, x2, mid) < x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _lerp_value(prop: str, a: Any, b: Any, u: float) -> Any:
    if prop == "rotation" or prop == "visible":
        if prop == "visible":
            return 1.0 if a >= 0.5 else 0.0      # 显隐不插值
        return float(a) + (float(b) - float(a)) * u
    return [float(a[i]) + (float(b[i]) - float(a[i])) * u for i in range(len(a))]


def sample_track(track: Dict[str, Any], t: float) -> Any:
    """在时刻 t 采样一条轨道（t 不 wrap，调用方负责循环）。"""
    kfs = track["keyframes"]
    if t <= kfs[0]["time"]:
        return kfs[0]["value"]
    if t >= kfs[-1]["time"]:
        return kfs[-1]["value"]
    # 二分找所在区间（关键帧可能上千，线性扫会拖慢预览）
    lo, hi = 0, len(kfs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if kfs[mid]["time"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = kfs[lo], kfs[hi]
    span = b["time"] - a["time"]
    if span <= 1e-12:
        return b["value"]
    u = (t - a["time"]) / span
    mode = track.get("interpolation") or "linear"
    if mode == "step":
        return a["value"]
    if mode == "cubic-bezier":
        c = track.get("bezier") or {"x1": 0.25, "y1": 0.1, "x2": 0.25, "y2": 1.0}
        u = _bezier(float(c["y1"]), float(c["y2"]),
                    _solve_bezier_x(float(c["x1"]), float(c["x2"]), u))
    return _lerp_value(track["property"], a["value"], b["value"], u)


def local_transforms(skeleton: Dict[str, Any], motion: Optional[Dict[str, Any]] = None,
                     t: float = 0.0) -> Dict[str, Dict[str, float]]:
    """每根骨的**局部**变换（rest + 轨道覆盖）。缺轨道的属性用 rest。"""
    out: Dict[str, Dict[str, float]] = {}
    for b in skeleton["bones"]:
        r = b["rest"]
        out[b["id"]] = {"x": float(r["x"]), "y": float(r["y"]),
                        "rotation": float(r["rotation"]),
                        "sx": 1.0, "sy": 1.0, "visible": 1.0}
    if motion:
        tt = float(t)
        if motion.get("loop"):
            d = float(motion.get("duration") or 0.0)
            if d > 1e-9:
                tt = tt % d
        for tr in motion["tracks"]:
            v = sample_track(tr, tt)
            node = out[tr["targetId"]]
            if tr["property"] == "position":
                node["x"], node["y"] = float(v[0]), float(v[1])
            elif tr["property"] == "rotation":
                node["rotation"] = float(v)
            elif tr["property"] == "scale":
                node["sx"], node["sy"] = float(v[0]), float(v[1])
            elif tr["property"] == "visible":
                node["visible"] = float(v)
    return out


def world_transforms(skeleton: Dict[str, Any],
                     local: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """把局部变换合成为世界变换（x/y/rotation/scale），父骨先算。

    2D 合成：世界角 = 父世界角 + 局部角；世界位置 = 父世界位置 +
    旋转(父世界角) · (局部位置 × 父世界缩放)。
    """
    order = _topo_order(skeleton)
    world: Dict[str, Dict[str, float]] = {}
    for bid in order:
        node = local[bid]
        parent = _parent_of(skeleton, bid)
        if parent is None:
            world[bid] = {"x": node["x"], "y": node["y"],
                          "rotation": node["rotation"],
                          "sx": node["sx"], "sy": node["sy"],
                          "visible": node["visible"]}
            continue
        pw = world[parent]
        ang = pw["rotation"]
        px, py = node["x"] * pw["sx"], node["y"] * pw["sy"]
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        world[bid] = {
            "x": pw["x"] + px * cos_a - py * sin_a,
            "y": pw["y"] + px * sin_a + py * cos_a,
            "rotation": ang + node["rotation"],
            "sx": pw["sx"] * node["sx"], "sy": pw["sy"] * node["sy"],
            "visible": min(pw["visible"], node["visible"]),
        }
    return world


def _parent_of(skeleton: Dict[str, Any], bid: str) -> Optional[str]:
    for b in skeleton["bones"]:
        if b["id"] == bid:
            return b.get("parentId")
    return None


def _topo_order(skeleton: Dict[str, Any]) -> List[str]:
    """根在前、子在后（父链深度排序，稳定）。"""
    depth: Dict[str, int] = {}
    bones = {b["id"]: b for b in skeleton["bones"]}

    def d(bid: str) -> int:
        if bid in depth:
            return depth[bid]
        p = bones[bid].get("parentId")
        depth[bid] = 0 if p is None else d(p) + 1
        return depth[bid]

    return sorted(bones, key=lambda b: (d(b), b))


# ---------------------------------------------------------------- 落盘


def dump(doc: Dict[str, Any], path: Path) -> Path:
    """写 JSON：UTF-8、缩进 2、结尾换行（diff 友好）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def load(path: Path, kind: str, skeleton: Optional[Dict[str, Any]] = None
         ) -> Dict[str, Any]:
    """读 + 校验。`kind` 是 skeleton / character-binding / motion-clip。"""
    path = Path(path)
    if not path.is_file():
        raise RigError(f"找不到文件：{path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RigError(f"{path.name} 不是合法 JSON：{exc}") from exc
    if kind == "skeleton":
        return validate_skeleton(doc)
    if kind == "character-binding":
        if skeleton is None:
            raise RigError("校验绑定前要先载入骨架")
        return validate_binding(doc, skeleton)
    if kind == "motion-clip":
        if skeleton is None:
            raise RigError("校验动作前要先载入骨架")
        return validate_motion(doc, skeleton)
    raise RigError(f"不认识的 kind：{kind}")
