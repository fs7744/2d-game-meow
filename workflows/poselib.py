"""VNCCS 官方姿势库（191 个预设）的清单扫描。

**这些是随项目发行的** imported 资源**，不在 `library/` 里**：那边的姿势库是用户自己攒的
（`library/poses.json`），这边的 Stem pose library 是随项目发行的现成素材，
本身就在静态目录下，浏览器直接当静态文件取。

    web/public/vnccs-pose/PoseLibrary/
    ├── local_user_poses/Uncategorized/<名>.json + .webp
    ├── Totemistyk/General_Poses_PoseStudio/poses/<分类>/<名>.json
    │                                    previews/<分类>/<名>.webp
    ├── Totemistyk/Ballet_PoseStudio/…
    ├── MIUProject/VNCCS_PoseLibrary_Main/…
    └── MIUProject/MySuperPoseRepo/…

JSON 是 PoseStudio 的**原生格式**（bones / ikEffectorPositions /
poleTargetPositions / hipBonePosition …），`core.setPose(json)` 能直接吃，
所以这份清单只需要指路，不用转换。

为什么连 `_library.tags` 也一起读出来：标签写着这张姿势是
`front view` / `side view` / `rear view` —— 挑姿势时能直接按拍摄角度筛，
也能顺手给导出机位一个合理的默认值（见 `_guess_yaw`）。只给路径的话，
用户就得挨个点开才知道是不是要的那个。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflows import paths

#: 静态服务下的 URL 前缀（app.py 把 web/dist 挂在根）
URL_PREFIX = "/vnccs-pose/PoseLibrary"

#: 清单阈值：**tag 要读 191 个 json**，每次请求都扫一遍没必要。
#: 目录内容平时不变，缓存 5 分钟足够；改完文件想立刻生效就重启或等一会。
_CACHE_TTL_SEC = 300
_CACHE_TTL_ERR_SEC = 5
_cache_items: Optional[List[Dict[str, Any]]] = None
_cache_at = 0.0


def pose_library_dir() -> Path:
    return paths.POSE_LIBRARY_DIR


def _rel(p: Path) -> str:
    """文件 → 静态 URL（逐段编码，空格/中文都要转义）。"""
    try:
        rel = p.relative_to(pose_library_dir())
    except ValueError:
        return ""
    return "/".join([URL_PREFIX, *[part.replace(' ', '%20')
                                   for part in rel.parts]])


def _guess_yaw(tags: List[str]) -> int:
    """从标签猜一个导出机位。

    官方库里绝大多数姿势的 cameraParams.yaw_deg 都是 0 —— 那个值是"当时编辑時
    的相机"，作者摆姿势时几乎都在正面看；真正的拍摄角度写在 tags 里。
    所以默认机位按标签给：侧面 90°、背面 180°。猜错也只是默认值，
    用户在节点上还能改。
    """
    s = " ".join(t.lower() for t in tags)
    if "rear view" in s or "back view" in s:
        return 180
    if "profile view" in s or "side view" in s:
        return 90
    if "three quarter" in s or "3/4" in s:
        return 45
    return 0


#: cameraParams 里真正带构图信息的数值键。官方 json 里还混着 prompt /
#: pose_prompts（实测全是空串）和一堆非数值字段，整对象塞进 191 条清单
#: 只会让响应白白变大，所以只挑这五个数值键。
_CAMERA_NUM_KEYS = ("zoom", "offset_x", "offset_y", "yaw_deg", "pitch_deg")
#: 一条清单最多带几个角色，避免多角色姿势把清单撑爆
_MAX_CHARS = 8


def _as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_num(v: Any, default: float = 0.0) -> float:
    """只收真正的数值；官方 json 偶尔混着字符串/None/布尔，别让它们进清单。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    return float(v)


def _camera_params(doc: Any) -> Dict[str, float]:
    """从该姿势 json 的 cameraParams 只摘数值构图键，缺失就给空 dict。"""
    cam = _as_dict(_as_dict(doc).get("cameraParams"))
    out: Dict[str, float] = {}
    for k in _CAMERA_NUM_KEYS:
        v = cam.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[k] = float(v)
    return out


def _pose_characters(doc: Any) -> List[Dict[str, Any]]:
    """从官方 characters[] 派生精简放置：transform 缺省全 0/zoom=1，
    rotation 缺省 [0,0,0]，最多 8 个。id/name 缺失时给个稳定兜底。"""
    raw = _as_dict(doc).get("characters")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(raw[:_MAX_CHARS], 1):
        c = _as_dict(c)
        t = _as_dict(c.get("transform"))
        rot_raw = c.get("rotation")
        rot = rot_raw if isinstance(rot_raw, (list, tuple)) else []
        out.append({
            "id": str(c.get("id") or f"ch_{i}"),
            "name": str(c.get("name") or f"角色 {i}"),
            "transform": {
                "x": _as_num(t.get("x")),
                "y": _as_num(t.get("y")),
                "z": _as_num(t.get("z")),
                "zoom": _as_num(t.get("zoom"), 1.0),
            },
            "rotation": [_as_num(rot[j]) if j < len(rot) else 0.0 for j in range(3)],
        })
    return out


def _walk() -> List[Dict[str, Any]]:
    root = pose_library_dir()
    if not root.is_dir():
        return []
    items: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            p = Path(dirpath) / fn
            # 每个仓库根目录那份 pose_library.json 是仓库自己的清单，
            # 不是一个姿势——把它列出来用户会点出一个空姿势。
            if fn == "pose_library.json":
                continue
            stem = p.stem
            # 预览图：4 个仓库用平行目录 previews/<分类>/<名>.webp，
            # local_user_poses 是跟 json 同目录放下
            prev = p.with_suffix(".webp")
            if not prev.is_file():
                parts = list(p.relative_to(root).parts)
                if "poses" in parts:
                    i = parts.index("poses")
                    parts = parts[:i] + ["previews"] + parts[i + 1:]
                    prev = root.joinpath(*parts).with_suffix(".webp")
            category = ""
            tags: List[str] = []
            doc: Any = None                # try 外先占位：读失败时下面的提取也要用到它
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(doc, dict):
                    lib = doc.get("_library") if isinstance(doc.get("_library"), dict) else {}
                    category = str(lib.get("category") or "")
                    tags = [str(t) for t in (lib.get("tags") or []) if t]
            except (OSError, ValueError, UnicodeDecodeError):
                pass                      # 单个文件坏了跳过，不影响其余 190 个
            if not category:
                category = p.parent.name if p.parent != root else ""
            items.append({
                "id": p.relative_to(root).as_posix(),
                "name": stem,
                "category": category,
                "tags": tags,
                # 官方 cameraParams.yaw_deg 几乎全是 0，拍摄角度写在 tags 里，
                # 所以机位仍按标签猜，不改成读 cameraParams。
                "yaw": _guess_yaw(tags),
                "json": _rel(p),
                "preview": _rel(prev) if prev.is_file() else "",
                "cameraParams": _camera_params(doc),
                "characters": _pose_characters(doc),
            })
    items.sort(key=lambda x: (x["category"], x["name"].lower()))
    return items


def list_presets(force: bool = False) -> List[Dict[str, Any]]:
    """扫一遍目录给清单（带 TTL 缓存）。"""
    global _cache_items, _cache_at
    now = time.time()
    ttl = _CACHE_TTL_SEC if _cache_items else _CACHE_TTL_ERR_SEC
    if force or _cache_items is None or now - _cache_at > ttl:
        try:
            _cache_items = _walk()
        except OSError:
            _cache_items = []
        _cache_at = now
    return _cache_items


def categories() -> List[str]:
    """分类清单，**大小写去重**。

    官方库里有 `Kneeling` 和 `kneeling` 两种写法（两个仓库各写各的），不去重的话
    筛选项会出现两条看着一样的分类，用户以为第二个是坏的。
    """
    seen: Dict[str, str] = {}
    for it in list_presets():
        c = (it["category"] or "").strip()
        if not c:
            continue
        seen.setdefault(c.lower(), c)
    return sorted(seen.values())
