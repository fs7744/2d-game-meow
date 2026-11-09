"""角色**生命阶段**（life stage）。

一个角色不是只有一套设定：幼年 / 成年 / 战损 / 觉醒…… 都是同一个角色的不同
阶段，各自有独立的 `character_info`（角色设定）与自己的立绘。用户 2026-11-09
明确的新信息架构：

    角色库 → 阶段列表（可排序）→ 阶段编辑（角色设定 + 立绘）

## 存储

阶段列表、顺序与角色描述放在角色自己的 `config.json` 里（**数组顺序即展示顺序**，
不额外建表）：

```json
{
  "description": "角色级描述",
  "active_stage": "default",
  "stages": [
    {"id": "default", "title": "初始设定", "description": "", "info": {...}},
    {"id": "s_xxx",   "title": "成年", "description": "...", "info": {...}}
  ],
  "character_info": {...}   // 当前激活阶段 info 的镜像
}
```

**为什么留一份镜像**：现有全部出图 API（`/prompt`、`/generate`、姿势/表情/图集
那些）都从 `load_config()["character_info"]` 读设定。让 `active_stage` 的 info
始终同步到顶层，这些链路一个字都不用改，切换阶段就等于切换"当前设定"。

## 立绘目录

`Portraits/<服装>/<表情>/` 是**默认阶段**（`default`）的老位置，保证已有立绘
原地不动；其它阶段落 `Portraits/<阶段目录名>/<服装>/<表情>/`。删阶段只从列表
移除，**不删磁盘文件**（用户可能还要找回）。

## 兼容性

老角色没有 `stages` 字段时，`normalize()` 自动建一个「初始设定」阶段，
把顶层 `character_info` 与老立绘都归到它名下 —— 老数据不会凭空消失。
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

#: 老角色 / 新角色的第一个阶段都用这个 id；它的立绘仍放 Portraits/ 老位置
DEFAULT_ID = "default"
DEFAULT_TITLE = "初始设定"
MAX_STAGES = 50
_MAX_TEXT = 400

_STAGE_DIR_PREFIX = "s_"


def stage_dir_name(stage_id: str) -> str:
    """阶段 id → 磁盘目录名。`default` 没有独立目录（用 Portraits/ 根）。"""
    if str(stage_id) == DEFAULT_ID:
        return ""
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", str(stage_id or ""))[:48]
    return f"{_STAGE_DIR_PREFIX}{clean}" if clean else ""


def new_id() -> str:
    return "stage_" + uuid.uuid4().hex[:10]


def _text(value: Any, limit: int = _MAX_TEXT) -> str:
    return str(value or "").strip()[:limit]


def sanitize(raw: Any) -> List[Dict[str, Any]]:
    """前端全量提交时的校验：条数上限、字段类型、id 唯一、info 必须是对象。

    前端负责生成 id 与排序，这里只保证写进磁盘的东西不会把后续读取搞崩。
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in raw[:MAX_STAGES]:
        if not isinstance(item, dict):
            continue
        sid = _text(item.get("id"), 64) or new_id()
        if sid in seen:                       # 重复 id 直接另发一个，别覆盖别人的设定
            sid = new_id()
        seen.add(sid)
        info = item.get("info")
        if not isinstance(info, dict):
            info = {}
        out.append({
            "id": sid,
            "title": _text(item.get("title"), 120) or "未命名阶段",
            "description": _text(item.get("description")),
            "info": dict(info),
        })
    return out


def normalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """补齐 `stages / active_stage / description`，并把当前阶段设定同步到顶层。

    老角色（没有 stages）会得到一个「初始设定」阶段，info 就是原来的
    `character_info`；这样阶段列表不会是空的，老立绘也仍挂在它下面。
    """
    if not isinstance(cfg, dict):
        cfg = {}
    cfg["description"] = _text(cfg.get("description"), 1000)
    stages = sanitize(cfg.get("stages"))
    if not stages:
        info = cfg.get("character_info")
        stages = [{
            "id": DEFAULT_ID,
            "title": DEFAULT_TITLE,
            "description": "",
            "info": dict(info) if isinstance(info, dict) else {},
        }]
    cfg["stages"] = stages
    ids = [s["id"] for s in stages]
    active = str(cfg.get("active_stage") or "")
    if active not in ids:
        active = ids[0]
    cfg["active_stage"] = active
    sync_active(cfg)
    return cfg


def active(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """当前激活的那条阶段（找不到就给第一条）。"""
    stages = cfg.get("stages") or []
    aid = cfg.get("active_stage")
    for s in stages:
        if s.get("id") == aid:
            return s
    return stages[0] if stages else {}


def sync_active(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把激活阶段的 info 写回顶层 `character_info`（现有出图链路的唯一入口）。"""
    cur = active(cfg)
    if cur:
        cfg["character_info"] = dict(cur.get("info") or {})
    return cfg


def set_active(cfg: Dict[str, Any], stage_id: str) -> Dict[str, Any]:
    ids = [s["id"] for s in (cfg.get("stages") or [])]
    if stage_id in ids:
        cfg["active_stage"] = stage_id
    return sync_active(cfg)


def update_info(cfg: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """/characters/{name}/info 改属性时，同步改到当前阶段里，别让阶段设定回退。"""
    info = cfg.get("character_info")
    merged = {**(info if isinstance(info, dict) else {}), **(patch or {})}
    cfg["character_info"] = merged
    cur = active(cfg)
    if cur:
        cur["info"] = dict(merged)
    return cfg
