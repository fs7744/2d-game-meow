"""项目归档：把「一个项目的画布 + 它引用的素材」打包成单个文件，导入时还原。

**为什么需要**：项目（画布 + 节点 + 参数）目前只存在浏览器的 localStorage 里，
素材（图）存在服务端的 `assets/` 下。这带来两个真实问题：

1. 换台机器 / 换个浏览器，项目就没了——localStorage 是**单机**的。
2. 项目目录里混着一堆试出来的图，删项目时 `rm -rf` 掉的是一整份工作。

所以做一个归档格式 `.mproj`（zip）：一份 `project.json` + 打包进来的素材。
导出得到一个可以随便拷走、发给别人的单文件；导入落到一个**新的**项目目录里，
路径全部重写，**不会覆盖任何已有项目**。

**素材是怎么找出来的**：不去猜「节点里哪些字段是路径」，而是在整棵节点 JSON 上
递归找**看起来像路径且真实存在**的字符串。这样新增节点类型时不用改这里——
`source.path` / `charSource.path` / `image.path` / `params.maskPath` /
`params.psDir`（目录！）全都自动覆盖。

**三种引用形态**：

| 形态 | 例子 | 导入后 |
|---|---|---|
| 文件 | `params.maskPath` | 单个文件 |
| 目录 | `psDir` 指向骨架帧目录 | **同构的目录**（结构保留） |
| 通配 | `assets/_pose_walk_*.png` | 折成目录（后端接受目录，更稳） |
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from workflows import paths

#: 归档格式标识。改动节点结构语义时递增，老归档仍按 v1 读。
FORMAT = "meow-project"
VERSION = 1

#: 节点 JSON 里路径占位符的格式。打包时把真实路径换成 `⟦ref:3⟧`，
#: 导入时再换回新位置。用一个几乎不会出现在路径里的字符序列做标记，
#: 免得跟真实文件名撞车。
_REF_RE = re.compile(r"^\u27e6ref:(\d+)\u27e7$")

#: 素材在归档里的存放根目录
_FILES_DIR = "files"

#: 能当素材的后缀（目录引用只收这些）
_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class ArchiveError(Exception):
    """归档格式不对（不是 .mproj、缺 project.json、版本不认识）。"""


# ---------------------------------------------------------------- 路径工具


def _scope_name(scope: str) -> str:
    """项目 id → 安全的目录名。与 generate.scope_dir 的清洗规则保持一致。"""
    safe = "".join(
        ch for ch in (scope or "") if ch.isascii() and (ch.isalnum() or ch in "-_")
    )[:64]
    if not safe:
        raise ArchiveError("项目 id 非法")
    return safe


def project_dir(scope: str) -> Path:
    """项目目录：`assets/projects/<scope>/`（不创建）。"""
    return paths.PROJECTS_DIR / _scope_name(scope)


def asset_rel(path: Path) -> str:
    """绝对路径 → 相对 assets 根的 posix 路径（给 /files/ 用）。"""
    try:
        return path.resolve().relative_to(paths.ASSETS.resolve()).as_posix()
    except ValueError:
        return path.name


def _looks_like_path(s: str) -> bool:
    if not s or len(s) > 1024:
        return False
    if "\n" in s or "\r" in s:
        return False
    if any(c in s for c in "*?"):
        return False
    p = Path(s)
    if p.is_absolute():
        return True
    # 相对路径：认三种写法
    #   1. `assets/xxx`   —— 相对仓库根（Windows 上是 `assets\xxx`）
    #   2. `projects/xxx` —— 相对 assets 根（前端存的产物路径就是这种）
    #   3. `./xxx`        —— 相对当前工作目录
    # **分隔符两种都要认**：只判正斜杠会漏掉 Windows 的反斜杠形式，
    # 素材就不被打包、导入后丢图——而且不报错，用户只看到「图没了」。
    norm = s.replace("\\", "/")
    if norm.startswith("assets/") or norm.startswith("./"):
        return True
    # 相对 assets 根：首段是 assets 下真实存在的子目录
    head = norm.split("/", 1)[0]
    if head and head not in (".", ".."):
        try:
            return (paths.ASSETS / head).is_dir()
        except OSError:
            return False
    return False


def _resolve_ref(value: str) -> Optional[Path]:
    """把一个「像路径」的字符串解析成真实存在的文件/目录。

    相对路径按三种习惯试：`assets/xxx`（相对仓库根）、`xxx`（相对 assets 根）、
    `xxx`（相对当前工作目录）。解析不出来就返回 None——**不猜**。
    """
    if not _looks_like_path(value):
        return None
    p = Path(value)
    if p.is_absolute():
        cand = [p]
    else:
        cand = [paths.ROOT / p, paths.ASSETS / p, Path.cwd() / p]
    for c in cand:
        try:
            if c.exists() and (c.is_file() or c.is_dir()):
                return c.resolve()
        except OSError:
            continue
    return None


def _resolve_glob(value: str) -> Optional[List[Path]]:
    """带通配的路径 → 匹配到的文件列表。

    **绝对路径要拆成「父目录 + 文件名模式」**：`Path.glob` 不接受绝对模式
    （`Non-relative patterns are unsupported`），所以先按最后一个分隔符切开。
    """
    if not value or any(c in value for c in "\n\r"):
        return None
    if not any(c in value for c in "*?"):
        return None
    pat = Path(value)
    if pat.is_absolute():
        parent = pat.parent
        pattern = pat.name
        if not parent.exists():
            return None
        # (搜索根, 模式)
        targets: List[Tuple[Path, str]] = [(parent, pattern)]
    else:
        targets = [(paths.ROOT, value), (paths.ASSETS, value)]
    out: List[Path] = []
    for base, pattern in targets:
        try:
            hits = sorted(base.glob(pattern))
        except (ValueError, OSError, NotImplementedError):
            continue
        for h in hits:
            if h.is_file() and h.suffix.lower() in _SUFFIXES:
                out.append(h.resolve())
    seen = set()
    uniq: List[Path] = []
    for h in out:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq or None


# ---------------------------------------------------------------- 引用收集


class Ref:
    """一个素材引用。

    kind:
      - `file` —— 单个文件
      - `dir`  —— 目录（保留内部结构）
      - `glob` —— 通配展开的一批文件（导入后折成目录）
    """

    __slots__ = ("kind", "files", "rel_names", "root")

    def __init__(self, kind: str, files: List[Path], root: Optional[Path] = None) -> None:
        self.kind = kind
        self.files = files
        self.root = root
        # 文件在归档内相对 root 的名字（dir 用相对路径，file/glob 用文件名）
        if kind == "dir" and root is not None:
            self.rel_names = [str(f.relative_to(root)).replace("\\", "/") for f in files]
        else:
            self.rel_names = [f.name for f in files]


def _dir_images(root: Path) -> List[Path]:
    """目录下的图片（自然序：frame_2 排在 frame_10 前面）。"""
    def natural(p: Path):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]

    try:
        return sorted(
            (f for f in root.rglob("*")
             if f.is_file() and f.suffix.lower() in _SUFFIXES),
            key=natural,
        )
    except OSError:
        return []


def collect_refs(payload: Dict[str, Any]) -> Tuple[List[Ref], Dict[str, int]]:
    """在整棵 JSON 上找出所有「素材引用」。

    返回 `(refs, index)`：`refs[i]` 是第 i 个引用，`index` 是
    「原始字符串 → 序号」的映射（同一个路径在多个节点里出现只打包一次）。
    """
    refs: List[Ref] = []
    index: Dict[str, int] = {}

    def add(key: str, ref: Ref) -> None:
        if key in index:
            return
        index[key] = len(refs)
        refs.append(ref)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            if node in index:
                return
            globbed = _resolve_glob(node)
            if globbed:
                # 通配 → 折成目录：后端接受目录，比通配更稳（不会因为
                # 文件名排序把动画顺序排乱）
                add(node, Ref("glob", globbed))
                return
            real = _resolve_ref(node)
            if real is None:
                # **看起来是路径但文件不在** → 也要记成引用。
                # 否则导入到别的机器后，这个绝对路径仍然留在节点里，
                # 指向一个永远不存在的文件，用户点了生成才报错。
                # 记成 missing → 导入时清空（用户重新指定即可）。
                if _looks_like_path(node):
                    add(node, Ref("missing", []))
                return
            if real.is_dir():
                add(node, Ref("dir", _dir_images(real), root=real))
            else:
                add(node, Ref("file", [real]))

    walk(payload)
    return refs, index


# ---------------------------------------------------------------- 打包


def pack(meta: Dict[str, Any], payload: Dict[str, Any]) -> bytes:
    """把项目打成 zip 字节流。

    `meta` 是项目元信息（name/created/viewport 等），`payload` 是节点等主体数据。
    素材引用从 `payload` 里收集。
    """
    refs, index = collect_refs(payload)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1) 先写素材。归档内的路径形如 files/<ref序号>/<相对名>
        for i, ref in enumerate(refs):
            for f, rel in zip(ref.files, ref.rel_names):
                try:
                    zf.writestr(f"{_FILES_DIR}/{i}/{rel}", f.read_bytes())
                except OSError:
                    # 读不到就跳过：不能因为一个坏文件让整个导出失败
                    continue
        # 2) 再写 project.json（路径换成占位符）
        doc = {
            "format": FORMAT,
            "version": VERSION,
            "meta": _rewrite(meta, index),
            "payload": _rewrite(payload, index),
            "ref_kinds": [r.kind for r in refs],
        }
        zf.writestr("project.json", json.dumps(doc, ensure_ascii=False, indent=2))
    return out.getvalue()


def _rewrite(obj: Any, index: Dict[str, int]) -> Any:
    """把 JSON 里的路径字符串换成占位符。"""
    if isinstance(obj, dict):
        return {k: _rewrite(v, index) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite(v, index) for v in obj]
    if isinstance(obj, str):
        i = index.get(obj)
        return obj if i is None else f"\u27e6ref:{i}\u27e7"
    return obj


# ---------------------------------------------------------------- 解包


def unpack(data: bytes, scope: str) -> Dict[str, Any]:
    """解开归档 → 一个**新项目**的 JSON（节点 + 视口 + 元信息）。

    素材落到 `assets/projects/<scope>/imported/<n>/`，路径重写成新位置。
    原项目不受任何影响。
    """
    scope = _scope_name(scope)
    dest_root = project_dir(scope) / "imported"
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"不是有效的归档文件：{exc}") from exc
    names = set(zf.namelist())
    if "project.json" not in names:
        raise ArchiveError("归档里没有 project.json")
    try:
        doc = json.loads(zf.read("project.json").decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ArchiveError(f"project.json 解析失败：{exc}") from exc
    if doc.get("format") != FORMAT:
        raise ArchiveError(f"不是本项目的归档（format={doc.get('format')!r}）")
    version = int(doc.get("version") or 0)
    if version > VERSION:
        raise ArchiveError(f"归档版本 {version} 比当前程序（v{VERSION}）新")

    kinds = doc.get("ref_kinds") or []
    new_paths: List[Any] = []
    for i, kind in enumerate(kinds):
        prefix = f"{_FILES_DIR}/{i}/"
        files = sorted(n for n in names if n.startswith(prefix))
        if kind == "missing":
            # 导出时文件就不在 → 导入后清空该路径
            new_paths.append(None)
            continue
        target = dest_root / str(i)
        target.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            # 单文件：取归档里那一个
            if not files:
                new_paths.append(None)
                continue
            name = Path(files[0]).name
            out_path = target / name
            with zf.open(files[0]) as src:
                out_path.write_bytes(src.read())
            new_paths.append(out_path)
            continue
        # dir / glob：还原成同构目录
        for n in files:
            rel = n[len(prefix):]
            if not rel or ".." in rel.split("/"):
                continue                      # 目录穿越，丢弃
            out_path = target / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(n) as src:
                out_path.write_bytes(src.read())
        new_paths.append(target if files else None)

    doc["meta"] = _restore(doc.get("meta") or {}, new_paths)
    doc["payload"] = _restore(doc.get("payload") or {}, new_paths)
    return {
        "meta": doc["meta"],
        "payload": doc["payload"],
        "imported_files": sum(1 for p in new_paths if p is not None),
    }


def _restore(obj: Any, new_paths: List[Any]) -> Any:
    """把占位符换回真实路径，并把同一对象里的 `/files/...` URL 一起重算。"""
    if isinstance(obj, dict):
        out = {k: _restore(v, new_paths) for k, v in obj.items()}
        # 同一层里既有 path 又有 url → url 由新 path 重新算
        p = out.get("path")
        u = out.get("url")
        if isinstance(p, str) and isinstance(u, str) and u.startswith("/files/"):
            rel = _try_rel(p)
            if rel:
                out["url"] = "/files/" + rel
        return out
    if isinstance(obj, list):
        return [_restore(v, new_paths) for v in obj]
    if isinstance(obj, str):
        m = _REF_RE.match(obj)
        if m:
            i = int(m.group(1))
            p = new_paths[i] if i < len(new_paths) else None
            return "" if p is None else str(p)
        return obj
    return obj


def _try_rel(p: str) -> Optional[str]:
    """绝对路径 → 相对 assets 的 posix 路径；不在 assets 内返回 None。"""
    try:
        return Path(p).resolve().relative_to(paths.ASSETS.resolve()).as_posix()
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------- 模板


def _template_file(tid: str) -> Path:
    return paths.TEMPLATES_DIR / f"{_scope_name(tid)}.mproj"


def save_template(tid: str, meta: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """把项目存成模板：归档写进 `templates/<tid>.mproj`。"""
    paths.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    data = pack(meta, payload)
    path = _template_file(tid)
    path.write_bytes(data)
    return {"id": path.stem, "file": str(path), "bytes": len(data)}


def list_templates() -> List[Dict[str, Any]]:
    """列出所有模板（按创建时间倒序）。"""
    out: List[Dict[str, Any]] = []
    if not paths.TEMPLATES_DIR.exists():
        return out
    for p in sorted(paths.TEMPLATES_DIR.glob("*.mproj")):
        info: Dict[str, Any] = {
            "id": p.stem, "name": p.stem, "bytes": p.stat().st_size,
            "created": p.stat().st_mtime, "nodes": 0,
        }
        try:
            with zipfile.ZipFile(p) as zf:
                doc = json.loads(zf.read("project.json").decode("utf-8"))
            meta = doc.get("meta") or {}
            payload = doc.get("payload") or {}
            info["name"] = meta.get("name") or p.stem
            info["created"] = meta.get("created") or info["created"]
            info["nodes"] = len(payload.get("nodes") or [])
        except Exception:            # noqa: BLE001 —— 坏模板不该让列表整个挂掉
            pass
        out.append(info)
    out.sort(key=lambda t: t.get("created") or 0, reverse=True)
    return out


def read_template(tid: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """读模板的 meta 与 payload（不解素材，只看内容）。"""
    path = _template_file(tid)
    if not path.exists():
        raise ArchiveError(f"模板不存在：{tid}")
    with zipfile.ZipFile(path) as zf:
        doc = json.loads(zf.read("project.json").decode("utf-8"))
    return doc.get("meta") or {}, doc.get("payload") or {}


def instantiate_template(tid: str, scope: str) -> Dict[str, Any]:
    """从模板实例化一个新项目（素材复制到新项目目录，路径重写）。"""
    path = _template_file(tid)
    if not path.exists():
        raise ArchiveError(f"模板不存在：{tid}")
    return unpack(path.read_bytes(), scope)


def delete_template(tid: str) -> bool:
    path = _template_file(tid)
    if not path.exists():
        return False
    path.unlink()
    return True
