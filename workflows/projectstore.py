"""项目定义的落盘存储（服务端本地磁盘）。

**为什么要有这个**：项目（画布 + 节点 + 参数）原先只存在浏览器的 localStorage 里。
那意味着：

- 换台机器 / 换个浏览器，项目就没了；
- 清一次浏览器数据，一周的画布全丢；
- 后端完全不知道「有哪些项目」——产物在 `assets/projects/<id>/`，
  但那份产物对应的是哪套节点参数，只有浏览器知道。

所以项目定义落到**服务端本地磁盘**：每个项目一个目录，定义文件是
`assets/projects/<id>/project.json`。这样一个项目的东西全在一个目录里：

    assets/projects/<id>/
    ├── project.json     项目定义（节点 + 视口 + 元信息）
    ├── w1_..._0000.png  产物
    ├── uploads/         拖进来的本地图
    └── masks/           局部重绘的蒙版

**localStorage 保留，但降级为缓存**：前端启动时先拿缓存把画布画出来（不等网络），
再从服务端拉权威版本覆盖。服务连不上时仍然能用缓存继续工作（离线兜底）。

**写入是原子的**：先写临时文件再 `os.replace`。半截的 JSON 会让项目打不开，
而自动保存是高频触发的，不能冒这个险。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflows import paths

#: 项目定义文件名（放在项目目录里，跟产物在一起）
PROJECT_FILE = "project.json"

#: 工作区状态文件：记「当前打开的是哪个项目」
WORKSPACE_FILE = "workspace.json"

#: 写锁。**必须用 RLock（可重入）**：`save_project` 持锁时调 `_atomic_write`，
# 而后者内部也要拿锁——普通 Lock 重复获取会直接死锁（测试表现为整个 pytest
# 卡住、不报错）。
_WRITE_LOCK = threading.RLock()


class ProjectConflict(Exception):
    """项目冲突：要么 id 已被占用，要么内容被别人改过。

    **必须让调用方能区分「冲突」与「出错」**：冲突是可恢复的
    （换个 id 重来 / 提示用户），出错不是。
    """


def safe_scope(scope: str) -> str:
    """项目 id → 安全的目录名。

    与 `generate.scope_dir` 的清洗规则**必须一致**：两边算出不同的目录名，
    产物和定义就散在两个地方了。

    清洗掉一切非 ASCII 字母数字与 `-_` 的字符，顺带挡住路径穿越
    （`../` 里的 `.` 和 `/` 会被去掉，`..` 变成空串 → 用 default）。
    """
    s = "".join(
        ch for ch in (scope or "") if ch.isascii() and (ch.isalnum() or ch in "-_")
    )[:64]
    return s or "default"


def project_dir(scope: str) -> Path:
    """项目目录 `assets/projects/<scope>/`（不创建）。"""
    return paths.PROJECTS_DIR / safe_scope(scope)


def project_file(scope: str) -> Path:
    """项目定义文件 `assets/projects/<scope>/project.json`。"""
    return project_dir(scope) / PROJECT_FILE


def workspace_file() -> Path:
    """工作区状态文件 `assets/workspace.json`。"""
    return paths.ASSETS / WORKSPACE_FILE


# ---------------------------------------------------------------- 写


def _atomic_write(path: Path, text: str) -> None:
    """原子写：先写临时文件再替换。

    **为什么必须原子**：自动保存是高频触发的。直接 `write_text` 时进程被杀 /
    断电会留下半截 JSON，下次打开这个项目就报解析错误——用户一屏的活没了。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with _WRITE_LOCK:
        os.replace(str(tmp), str(path))


def save_project(scope: str, doc: Dict[str, Any],
                 expect_updated: Optional[str] = None,
                 create: bool = False) -> Dict[str, Any]:
    """把一个项目定义写到磁盘。返回落盘的摘要。

    `doc` 要有 `nodes`（列表）。`id` 缺省用 scope，`name` 缺省用 scope。

    **`create=True`（新建）**：磁盘上已经有这个 id → 抛 `ProjectConflict`。
    以前是直接覆盖，于是两个标签页在同一毫秒建项目（id 撞了）时，
    后保存的把先保存的**静默覆盖**——用户的画布无声消失，还以为保存成功了。

    **乐观锁**（`expect_updated`）：调用方带上「我读到的那份项目的 updated 时间戳」。
    磁盘上当前的 updated 与之不符 → 抛 `ProjectConflict`。
    这解决的是**两个标签页同时编辑同一个项目**：A 保存后 B 再保存，
    B 的 `expect_updated` 已经是旧的了 → 冲突报错，而不是把 A 的改动覆盖掉。
    """
    if not isinstance(doc, dict):
        raise ValueError("项目定义必须是对象")
    nodes = doc.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("项目定义缺少 nodes 列表")
    sid = safe_scope(scope)
    now = doc.get("updated") or ""
    out = {
        "id": str(doc.get("id") or sid),
        "name": str(doc.get("name") or sid),
        "created": str(doc.get("created") or now),
        "updated": str(doc.get("updated") or now),
        "nodes": nodes,
        "viewport": doc.get("viewport") or {"x": 120, "y": 90, "z": 1},
    }
    with _WRITE_LOCK:
        # **新建模式**：id 已被占用 → 冲突，不覆盖
        if create and (project_file(sid).exists() or project_dir(sid).exists()):
            raise ProjectConflict(f"项目 id 已被占用：{sid}")
        # **乐观锁校验**：在写之前比对磁盘上的 updated
        if expect_updated is not None:
            cur = load_project(sid)
            cur_updated = str((cur or {}).get("updated") or "")
            if cur is not None and cur_updated != expect_updated:
                raise ProjectConflict(
                    f"项目已被其他窗口修改（磁盘上是 {cur_updated or '空'}，"
                    f"你手上是 {expect_updated or '空'}）")
        _atomic_write(project_file(sid),
                      json.dumps(out, ensure_ascii=False, indent=1))
    return summarize(out)


def allocate_project_id(prefix: str = "p") -> str:
    """分配一个**保证没用过**的项目 id。

    **为什么由服务端分配**：id 原先由浏览器用「时间戳 + 序号」生成，
    两个标签页在同一毫秒建项目会拿到**完全相同**的 id（实测 `pmdr5q6ns0`），
    后保存的直接覆盖先保存的——用户毫无察觉地丢了项目。

    服务端分配用「随机后缀 + 存在性检查」，并带重试：
    万一撞了（概率极低）就换一个，绝不返回已存在的 id。
    """
    import secrets

    paths.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        sid = f"{prefix}{time.strftime('%y%m%d')}{secrets.token_hex(3)}"
        # 检查两处：项目定义文件、以及目录本身（有目录但没定义 = 残留）
        if not project_file(sid).exists() and not project_dir(sid).exists():
            return sid
    # 极端情况（20 次全撞）：退回带序号的
    n = 1
    while True:
        sid = f"{prefix}{time.strftime('%y%m%d')}{secrets.token_hex(2)}{n}"
        if not project_file(sid).exists() and not project_dir(sid).exists():
            return sid
        n += 1


def summarize(doc: Dict[str, Any]) -> Dict[str, Any]:
    """项目摘要（列表用，不带 nodes —— 几千个节点序列化起来很慢）。"""
    nodes = doc.get("nodes")
    return {
        "id": str(doc.get("id") or ""),
        "name": str(doc.get("name") or ""),
        "created": str(doc.get("created") or ""),
        "updated": str(doc.get("updated") or ""),
        "count": len(nodes) if isinstance(nodes, list) else 0,
    }


# ---------------------------------------------------------------- 读


def load_project(scope: str) -> Optional[Dict[str, Any]]:
    """读一个项目定义。文件不在或坏了都返回 None（**不抛**）。

    坏文件不能让整个列表接口 500 —— 一个项目坏了，其余项目照样要能用。
    """
    path = project_file(scope)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    doc.setdefault("id", safe_scope(scope))
    doc.setdefault("nodes", [])
    return doc


def list_projects() -> List[Dict[str, Any]]:
    """列出磁盘上所有项目（按更新时间倒序）。"""
    out: List[Dict[str, Any]] = []
    root = paths.PROJECTS_DIR
    if not root.is_dir():
        return out
    try:
        entries = list(os.scandir(root))
    except OSError:
        return out
    for e in entries:
        if not e.is_dir() or e.name.startswith("."):
            continue
        doc = load_project(e.name)
        if doc is None:
            continue                      # 没有定义文件 = 不是项目（或坏了）
        out.append(summarize(doc))
    out.sort(key=lambda p: p.get("updated") or "", reverse=True)
    return out


# ---------------------------------------------------------------- 删


def delete_project(scope: str, remove_files: bool = True) -> bool:
    """删项目。

    `remove_files=True`  → 连整个项目目录一起删（产物、上传、蒙版）。
    `remove_files=False` → 只删 project.json，产物留在盘上（用户只想从列表里去掉）。
    """
    sid = safe_scope(scope)
    pf = project_file(sid)
    if not pf.exists():
        return False
    if remove_files:
        shutil.rmtree(project_dir(sid), ignore_errors=True)
    else:
        try:
            pf.unlink()
        except OSError:
            return False
    return True


# ---------------------------------------------------------------- 工作区


def load_workspace() -> Dict[str, Any]:
    """读工作区状态（当前打开哪个项目）。坏了就当没有。"""
    path = workspace_file()
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def save_workspace(doc: Dict[str, Any]) -> None:
    """写工作区状态。"""
    try:
        _atomic_write(workspace_file(),
                      json.dumps(doc, ensure_ascii=False, indent=1))
    except OSError:
        pass
