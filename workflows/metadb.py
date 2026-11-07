"""本地元数据索引（SQLite，Python 标准库，零安装）。

**它不存文件**：图片、精灵帧、模型权重、生成用的属性 config 一律留在磁盘原处。

**分工**：磁盘负责文件与生成产线，SQLite 是**角色元数据的权威存储** ——
派生标签、稳定的创建/编辑时间、用户自定义标签、备注、任务历史摘要。
不必为兼容外部 VNCCS 目录格式牺牲数据模型：自定义标签 / 备注 / 时间这类
磁盘上没有、也不该塞进 prompt config 的信息，就该有自己的持久化位置。
删库只会丢元数据，不会丢任何文件（素材始终在 `assets/characters/` 下）。

为什么是 SQLite 而不是别的：
- `sqlite3` 在 Python 标准库里，**不需要 pip、不需要服务进程、不需要单独安装**；
- 单文件 + WAL + 事务，本机单用户 / 后台 worker 多线程读写都够稳；
- DuckDB 要额外安装且面向 OLAP 分析，TinyDB 只是 JSON 换皮（无事务/索引/并发），
  libSQL/Turso 需要额外依赖或同步服务 —— 对这个「几十到几百条元数据 + 点查排序」
  的场景都是过度工程。

表：
  characters  角色元数据（name 主键；tags/custom_tags/costumes 存 JSON 字符串）
  canvases    画布元数据（id 主键）
  tasks       任务历史摘要（job_id 主键；只记状态与耗时，不记图片路径）
  settings    全局设置（key 主键；value 是 JSON）
              —— 底模 / 抠图方式 / 生效的 LoRA / 快速模式。这些以前只在
              前端内存里，重开一次就没了；现在服务端是权威，重启自动恢复。
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from workflows import characters, paths

#: 放在 library/ 下（已被 .gitignore 忽略，测试用 MEOW_LIBRARY_DIR 覆盖）。
#: 这是**可重建的派生数据**，不是交付资产，所以不进 assets/。
DB_PATH = paths.LIBRARY_DIR / "meta.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    name        TEXT PRIMARY KEY,
    tags        TEXT NOT NULL DEFAULT '[]',
    custom_tags TEXT NOT NULL DEFAULT '[]',
    note        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    cover_path  TEXT NOT NULL DEFAULT '',
    sprites     INTEGER NOT NULL DEFAULT 0,
    costumes    TEXT NOT NULL DEFAULT '[]',
    synced_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    job_id     TEXT PRIMARY KEY,
    character  TEXT NOT NULL DEFAULT '',
    workflow   TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT '',
    seconds    REAL NOT NULL DEFAULT 0,
    error      TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS canvases (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    tags        TEXT NOT NULL DEFAULT '[]',
    custom_tags TEXT NOT NULL DEFAULT '[]',
    note        TEXT NOT NULL DEFAULT '',
    cover_path  TEXT NOT NULL DEFAULT '',
    node_count  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    synced_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_characters_updated ON characters(updated_at);
CREATE INDEX IF NOT EXISTS idx_canvases_updated ON canvases(updated_at);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
"""


def _connect() -> sqlite3.Connection:
    paths.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init_db() -> None:
    """建表（幂等）。每次启动 / 首次访问时调一次即可。"""
    with _connect() as con:
        con.executescript(_SCHEMA)


def _jload(raw: Any) -> List[str]:
    try:
        v = json.loads(raw or "[]")
        return [str(x) for x in v if str(x).strip()]
    except (TypeError, ValueError):
        return []


def _row_to_meta(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "name": row["name"],
        "tags": _jload(row["tags"]),
        "custom_tags": _jload(row["custom_tags"]),
        "note": row["note"] or "",
        "created_at": float(row["created_at"] or 0),
        "updated_at": float(row["updated_at"] or 0),
        "cover_path": row["cover_path"] or "",
        "sprites": int(row["sprites"] or 0),
        "costumes": _jload(row["costumes"]),
    }


def sync_characters() -> int:
    """扫一次角色目录，把派生元数据同步进索引库，返回角色数。

    - `created_at` 第一次落库后**固定不变**（文件系统 ctime 在跨平台/拷贝目录
      时不可靠，落库后以库为准）；
    - `updated_at` 取「磁盘 mtime vs 库里记录」的较大值：刷新页面不会把时间
      推到「现在」，但改了 config / 加了素材 / 编辑了标签都会推进；
    - 磁盘上已删掉的角色，索引里也一并清掉。
    """
    init_db()
    names = characters.list_characters()
    now = time.time()
    with _connect() as con:
        old = {r["name"]: r for r in con.execute("SELECT * FROM characters")}
        for name in names:
            try:
                s = characters.character_summary(name)
            except Exception:
                continue                       # 单个坏目录不该让整个列表挂掉
            prev = old.get(name)
            created = float(prev["created_at"]) if prev else float(s.get("created_at") or now)
            disk_updated = float(s.get("updated_at") or created)
            updated = max(disk_updated, float(prev["updated_at"]) if prev else 0.0)
            custom = _jload(prev["custom_tags"]) if prev else []
            note = (prev["note"] or "") if prev else ""
            con.execute(
                """INSERT INTO characters
                   (name, tags, custom_tags, note, created_at, updated_at,
                    cover_path, sprites, costumes, synced_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                     tags=excluded.tags, updated_at=excluded.updated_at,
                     cover_path=excluded.cover_path, sprites=excluded.sprites,
                     costumes=excluded.costumes, synced_at=excluded.synced_at""",
                (name, json.dumps(s.get("tags") or [], ensure_ascii=False),
                 json.dumps(custom, ensure_ascii=False), note,
                 created, updated, s.get("cover") or "",
                 int(s.get("sprites") or 0),
                 json.dumps(s.get("costumes") or [], ensure_ascii=False), now),
            )
        for stale in set(old) - set(names):
            con.execute("DELETE FROM characters WHERE name = ?", (stale,))
    return len(names)


def list_meta() -> Dict[str, Dict[str, Any]]:
    init_db()
    with _connect() as con:
        return {r["name"]: _row_to_meta(r) for r in con.execute("SELECT * FROM characters")}


def get_meta(name: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _connect() as con:
        row = con.execute("SELECT * FROM characters WHERE name = ?", (name,)).fetchone()
        return _row_to_meta(row) if row else None


def update_meta(name: str, custom_tags: Optional[List[str]] = None,
                note: Optional[str] = None) -> Dict[str, Any]:
    """写用户附加信息（自定义标签 / 备注）。这两项磁盘 config 里没有，
    只能存在索引库里 —— 这正是需要一个小数据库的原因。"""
    init_db()
    now = time.time()
    with _connect() as con:
        row = con.execute("SELECT * FROM characters WHERE name = ?", (name,)).fetchone()
        if not row:
            sync_characters()
            row = con.execute("SELECT * FROM characters WHERE name = ?", (name,)).fetchone()
        if not row:
            raise KeyError(f"角色不存在：{name}")
        tags = ([str(t).strip() for t in (custom_tags or []) if str(t).strip()]
                if custom_tags is not None else _jload(row["custom_tags"]))
        text = note if note is not None else (row["note"] or "")
        con.execute(
            "UPDATE characters SET custom_tags = ?, note = ?, updated_at = ? WHERE name = ?",
            (json.dumps(tags, ensure_ascii=False), text, now, name),
        )
        row = con.execute("SELECT * FROM characters WHERE name = ?", (name,)).fetchone()
        return _row_to_meta(row)


def record_task(job_id: str, character: str = "", workflow: str = "",
                status: str = "", seconds: float = 0.0, error: str = "") -> None:
    """任务历史摘要（只记状态/耗时，不记任何文件路径）。"""
    init_db()
    with _connect() as con:
        con.execute(
            """INSERT INTO tasks (job_id, character, workflow, status, seconds, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id) DO UPDATE SET
                 status=excluded.status, seconds=excluded.seconds, error=excluded.error""",
            (job_id, character, workflow, status, float(seconds or 0), error[:400], time.time()),
        )


def recent_tasks(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))]


# --------------------------------------------------------------------------
# 画布（原「项目」）：同一套元数据思路，标签从节点类型派生，封面只记路径字符串
# --------------------------------------------------------------------------

def _row_to_canvas(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"] or "",
        "tags": _jload(row["tags"]),
        "custom_tags": _jload(row["custom_tags"]),
        "note": row["note"] or "",
        "cover_path": row["cover_path"] or "",
        "node_count": int(row["node_count"] or 0),
        "created_at": float(row["created_at"] or 0),
        "updated_at": float(row["updated_at"] or 0),
    }


def _canvas_facts(item: Dict[str, Any]) -> Dict[str, Any]:
    """从 project.json 派生标签与封面路径（只读元数据，不搬文件）。"""
    from workflows import projectstore
    import os

    scope = str(item.get("id") or "")
    doc = projectstore.load_project(scope) or {}
    nodes = doc.get("nodes") or []
    kinds = []
    for n in nodes:
        k = str(n.get("kind") or "").strip()
        if k and k not in kinds:
            kinds.append(k)
    if nodes and "nodes" not in kinds:
        kinds.append(f"{len(nodes)} 节点")
    cover = ""
    try:
        d = projectstore.project_dir(scope)
        imgs = [os.path.join(d, f) for f in os.listdir(d)
                if f.lower().endswith((".png", ".webp", ".jpg", ".jpeg"))]
        if imgs:
            cover = max(imgs, key=lambda p: os.path.getmtime(p))
    except (OSError, ValueError):
        pass
    return {"tags": kinds[:16], "cover": cover, "node_count": len(nodes)}


def sync_canvases() -> int:
    """把磁盘上的画布列表同步进元数据索引，返回画布数。"""
    from workflows import projectstore
    import os

    init_db()
    items = projectstore.list_projects()
    now = time.time()
    with _connect() as con:
        old = {r["id"]: r for r in con.execute("SELECT * FROM canvases")}
        for it in items:
            cid = str(it.get("id") or "")
            if not cid:
                continue
            prev = old.get(cid)
            facts = _canvas_facts(it)
            try:
                from workflows import paths as _paths
                pj = os.path.join(_paths.PROJECTS_DIR, cid, "project.json")
                disk_updated = os.path.getmtime(pj) if os.path.isfile(pj) else now
            except OSError:
                disk_updated = now
            created = float(prev["created_at"]) if prev else disk_updated
            updated = max(disk_updated, float(prev["updated_at"]) if prev else 0.0)
            custom = _jload(prev["custom_tags"]) if prev else []
            note = (prev["note"] or "") if prev else ""
            con.execute(
                """INSERT INTO canvases
                   (id, name, tags, custom_tags, note, cover_path, node_count,
                    created_at, updated_at, synced_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     name=excluded.name, tags=excluded.tags,
                     updated_at=excluded.updated_at,
                     cover_path=excluded.cover_path,
                     node_count=excluded.node_count, synced_at=excluded.synced_at""",
                (cid, str(it.get("name") or ""),
                 json.dumps(facts["tags"], ensure_ascii=False),
                 json.dumps(custom, ensure_ascii=False), note,
                 facts["cover"], int(facts["node_count"]),
                 created, updated, now),
            )
        for stale in set(old) - {str(i.get("id") or "") for i in items}:
            con.execute("DELETE FROM canvases WHERE id = ?", (stale,))
    return len(items)


def list_canvas_meta() -> Dict[str, Dict[str, Any]]:
    init_db()
    with _connect() as con:
        return {r["id"]: _row_to_canvas(r) for r in con.execute("SELECT * FROM canvases")}


def update_canvas_meta(cid: str, custom_tags: Optional[List[str]] = None,
                       note: Optional[str] = None) -> Dict[str, Any]:
    """写画布的自定义标签 / 备注（磁盘 project.json 里不该塞这些）。"""
    init_db()
    with _connect() as con:
        row = con.execute("SELECT * FROM canvases WHERE id = ?", (cid,)).fetchone()
        if not row:
            sync_canvases()
            row = con.execute("SELECT * FROM canvases WHERE id = ?", (cid,)).fetchone()
        if not row:
            raise KeyError(f"画布不存在：{cid}")
        tags = ([str(t).strip() for t in (custom_tags or []) if str(t).strip()]
                if custom_tags is not None else _jload(row["custom_tags"]))
        text = note if note is not None else (row["note"] or "")
        con.execute("UPDATE canvases SET custom_tags = ?, note = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(tags, ensure_ascii=False), text, time.time(), cid))
        return _row_to_canvas(con.execute(
            "SELECT * FROM canvases WHERE id = ?", (cid,)).fetchone())


# --------------------------------------------------------------------------
# 全局设置：底模 / 抠图方式 / 生效的 LoRA / 快速模式
# --------------------------------------------------------------------------

#: 允许写入的 key 与默认值。**白名单**而不是任意键：前端 `PUT /settings`
#: 是局部合并，写错一个 key 不该悄悄留一条永远用不上的脏数据。
DEFAULT_SETTINGS: Dict[str, Any] = {
    "base_model": "qwen-image-2.1",
    "matte_backend": "auto",
    "loras": [],            # [{"name": str, "scale": float}]
    "fast_mode": False,
}

#: matte.matte_auto 接受的全部后端值（未知值会静默回落 color，别让 UI 存进去）。
MATTE_BACKENDS = ("auto", "color", "u2net", "ai", "isnet", "birefnet")


def get_settings() -> Dict[str, Any]:
    """读全局设置；缺的键用默认值补齐（老库首次升级时表里是空的）。"""
    init_db()
    out = dict(DEFAULT_SETTINGS)
    with _connect() as con:
        for row in con.execute("SELECT key, value FROM settings"):
            k = row["key"]
            if k not in DEFAULT_SETTINGS:
                continue
            try:
                out[k] = json.loads(row["value"])
            except (TypeError, ValueError):
                pass
    return out


def set_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    """局部合并写全局设置，返回合并后的完整设置。

    只收白名单里的 key；`loras` 逐项清洗成 `{name, scale}`，避免把整个
    LoRA 对象（含绝对路径）塞进元数据。
    """
    init_db()
    clean: Dict[str, Any] = {}
    for key, value in (patch or {}).items():
        if key not in DEFAULT_SETTINGS:
            continue
        if key == "matte_backend":
            v = str(value or "auto")
            clean[key] = v if v in MATTE_BACKENDS else "auto"
        elif key == "loras":
            items = []
            for it in (value if isinstance(value, list) else []):
                if not isinstance(it, dict):
                    continue
                name = str(it.get("name") or "").strip()
                if not name:
                    continue
                try:
                    scale = float(it.get("scale", 1.0))
                except (TypeError, ValueError):
                    scale = 1.0
                items.append({"name": name, "scale": round(scale, 4)})
            clean[key] = items
        elif key == "fast_mode":
            clean[key] = bool(value)
        else:
            clean[key] = str(value or DEFAULT_SETTINGS[key])
    now = time.time()
    with _connect() as con:
        for k, v in clean.items():
            con.execute(
                "INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (k, json.dumps(v, ensure_ascii=False), now),
            )
    return get_settings()
