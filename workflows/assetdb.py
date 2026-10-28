"""资产索引：SQLite 读缓存，jsonl 仍是唯一写入日志。

**为什么需要**：`/assets` 老实现每次请求都全量读取 `assets/index.jsonl`、
逐行 `json.loads`、再对每一条做 `Path.exists()`。文件随产线增长（当前 2.6MB /
2974 条，每条还带 frame_scores），3 万条时每次打开画布要解析 26MB 并阻塞主线程。

**为什么不直接把 jsonl 换掉**：jsonl 是 append-only 的审计日志，天然抗损坏
（一行坏了不影响其它行），而且已有产物目录里全是它。所以保留写入侧不动，
只加一层**增量**读缓存：记录已消费的字节偏移，每次只解析新增部分。

写入方仍是 `generate._save` / `poseswap`（直接追加 jsonl），这里只读。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_WRITE_LOCK = threading.Lock()


def append_line(path: str | Path, obj: Dict[str, Any]) -> None:
    """向 jsonl 追加一条，**一次 write 写完、进程内加锁**。

    为什么不用 `open("a") + write`：多个写入方（生成 worker、训练线程、子进程）
    同时追加时，缓冲式写入可能交错出一段 **NUL 字节**。实测在 3599 行的索引里
    出现过一次 2990 字节的全零块——单条索引损坏会让 `/assets` 整个 500，代价远高于
    这点加固成本。这里用 `O_APPEND` + 单次 `os.write`，让一条记录要么整条落地、
    要么完全不落地。
    """
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    with _WRITE_LOCK:
        fd = os.open(str(path), flags, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


def rewrite_atomic(path: str | Path, text: str) -> None:
    """原子重写（写临时文件再替换）。删条目时重写整表，中途崩了不能留半个文件。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with _WRITE_LOCK:
        os.replace(str(tmp), str(path))


SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    file    TEXT UNIQUE,
    raw     TEXT,
    sheet   TEXT,
    scope   TEXT,
    workflow TEXT,
    created TEXT,
    meta    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_scope ON assets(scope, id DESC);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


class AssetIndex:
    def __init__(self, assets_dir: str | Path) -> None:
        self.dir = Path(assets_dir)
        self.jsonl = self.dir / "index.jsonl"
        self.db_path = self.dir / "index.db"
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ready = False

    # ---------- 连接 ----------

    def _db(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        except sqlite3.Error:
            # 只读盘/权限问题不该让清单接口整个挂掉——调用方会退回全量解析
            self._conn = None
        return self._conn

    # ---------- 增量同步 ----------

    def _offset(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM state WHERE key='offset'").fetchone()
        return int(row[0]) if row else 0

    def _set_offset(self, conn: sqlite3.Connection, value: int) -> None:
        conn.execute(
            "INSERT INTO state(key, value) VALUES('offset', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(value),))
        conn.commit()

    def sync(self) -> int:
        """把 jsonl 中尚未入索引的行读进来。返回新增条数。

        **按字节偏移增量读**：日志只追加，所以 seek(offset) 之后读到的就是新增部分。
        文件被截断/重建（offset 大于文件大小）时整体重建索引——这种情况只会发生在
        手工删了 jsonl，重建比猜测安全。
        """
        conn = self._db()
        if conn is None or not self.jsonl.exists():
            return 0
        with self._lock:
            size = self.jsonl.stat().st_size
            off = self._offset(conn)
            if off > size:                       # 文件被重建 → 清表重来
                conn.execute("DELETE FROM assets")
                conn.commit()
                off = 0
            if off == size:
                return 0
            added = 0
            # **按字节读**：文本模式在迭代中调 tell() 会抛
            # "telling position disabled by next() call"，而且多字节字符的偏移
            # 也不是字节偏移。这里一次读入新增部分，只消费到最后一个完整换行。
            with self.jsonl.open("rb") as fh:
                fh.seek(off)
                blob = fh.read()
            cut = blob.rfind(b"\n")
            if cut < 0:                       # 最后一行还没写完，等下次
                return 0
            complete = blob[:cut + 1]
            # **先整体解码再按行解析**：`json.loads(bytes)` 会做编码探测，
            # 一旦某个片段撞上 UTF-32 的字节模式就会抛 UnicodeDecodeError
            # （实测被一段 NUL 字节触发）。这里用 utf-8 + replace 兜住坏字节，
            # 坏行随后被 JSONDecodeError 过滤掉，不影响其它行。
            text = complete.decode("utf-8", "replace")
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                file = str(item.get("file") or "")
                if not file:
                    continue
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO assets"
                        "(file, raw, sheet, scope, workflow, created, meta) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (file, item.get("raw"), item.get("sheet"),
                         item.get("scope") or "default",
                         item.get("workflow"), str(item.get("created") or ""),
                         json.dumps(item, ensure_ascii=False)))
                    added += 1
                except sqlite3.Error:
                    continue
            self._set_offset(conn, off + len(complete))
            return added

    # ---------- 查询 / 删除 ----------

    def recent(self, limit: int = 50, project: str = "") -> List[Dict[str, Any]]:
        """最近 N 条。**先查库再验存在**——只对返回的这一页做 stat。"""
        conn = self._db()
        if conn is None:
            return []
        self.sync()
        sql = "SELECT meta FROM assets"
        args: list[Any] = []
        if project:
            sql += " WHERE scope=?"
            args.append(project)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, int(limit)))
        try:
            rows = conn.execute(sql, args).fetchall()
        except sqlite3.Error:
            return []
        out: List[Dict[str, Any]] = []
        for (meta,) in rows:
            try:
                item = json.loads(meta)
            except json.JSONDecodeError:
                continue
            target = item.get("file")
            # 陈旧条目（文件被手工删了）不返回，否则前端拿到 404 链接
            if target and not Path(str(target)).exists():
                continue
            out.append(item)
        return out

    def forget(self, paths: Sequence[str]) -> int:
        """删除若干产物对应的索引行（连带按目录前缀匹配）。"""
        conn = self._db()
        if conn is None or not paths:
            return 0
        n = 0
        with self._lock:
            for p in paths:
                try:
                    cur = conn.execute(
                        "DELETE FROM assets WHERE file=? OR raw=? OR sheet=?", (p, p, p))
                    n += cur.rowcount or 0
                    cur = conn.execute(
                        "DELETE FROM assets WHERE file LIKE ? OR sheet LIKE ?",
                        (p + "%", p + "%"))
                    n += cur.rowcount or 0
                except sqlite3.Error:
                    continue
            conn.commit()
        return n
