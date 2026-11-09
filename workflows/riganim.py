"""骨骼动画产物：内嵌骨骼编辑器（web/src/rig）导出的骨架 / 动画 / 纹理图集，按**生命阶段**归档。

目录规则照抄 `partsplit.py`（同一套 `stages.stage_dir_name`），保证「成年 / 幼年」
的骨骼不会混在一起，老数据也不动：

    assets/characters/<角色>/RigAnim[/s_<阶段>]/<时间戳_id>/
        manifest.json
        skeleton-3.8.json      无损骨骼 JSON 3.8
        skeleton-4.2.json      无损骨骼 JSON 4.2
        skeleton-4.3.json      无损骨骼 JSON 4.3
        skeleton.skel          二进制骨骼（4.3）
        skeleton.atlas         纹理图集描述
        skeleton.png           纹理图集页
        project.rigby          完整可编辑工程（动画 / 约束 / 皮肤 / 事件 / 音频）
        godot/<角色>_godot.zip 点「导出 Godot」后生成

**为什么同时存 project.rigby**：Spine 系的 .json/.skel 只装得下骨架与动画，
编辑器的约束、皮肤、事件、音频和场景数据会丢。留一份工作文件，用户以后能从
列表里重新打开继续改，而不是每次都从 PSD 重来。

Godot 导出是**工程包**，不是可执行文件：rigby2d 的 Godot 运行时是 Rust 编出来的
GDExtension（`rigby_godot.dll` / `.so` / `.dylib`），上游仓库里没有预编译二进制。
包里放好 addon 配置、示例场景与说明书，二进制需要用户按 README 获取。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflows import partsplit
from workflows import paths
from workflows import stages as stages_mod

DIR_NAME = "RigAnim"
UPLOAD_DIR_NAME = "_uploads"

#: 发布时允许的文件名（桥接层一次把这些全交上来，多一个都不收）
EXPECTED_FILES = (
    "skeleton-3.8.json", "skeleton-4.2.json", "skeleton-4.3.json",
    "skeleton.skel", "skeleton.atlas", "skeleton.png", "project.rigby",
)
ALLOWED_EXT = {".json", ".skel", ".atlas", ".png", ".rigby"}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_BYTES = 256 * 1024 * 1024

#: Godot 运行时 addon（一比一取自 rigby2d，只放配置与声明、不含二进制）
GODOT_ADDON = paths.ROOT / "tools" / "godot_addon" / "rigby2d"


class RigAnimError(RuntimeError):
    """产物/输入不合法（调用方转 400，文案要能直接给用户看）。"""


# ---------------------------------------------------------------- 路径


def root(name: str, stage_id: str = "") -> Path:
    """产物根目录。默认阶段用 `RigAnim/` 根（老位置），其它阶段用 `RigAnim/s_<id>/`。"""
    base = paths.CHARACTERS_DIR / name / DIR_NAME
    sub = stages_mod.stage_dir_name(stage_id) if stage_id else ""
    return base / sub if sub else base


def _url(p: Path | str) -> str:
    try:
        rel = Path(p).resolve().relative_to(paths.ASSETS.resolve())
    except (ValueError, OSError):
        return ""
    return "/files/" + "/".join(rel.parts)


def _inside(base: Path, target: Path) -> bool:
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_filename(raw: Any) -> str:
    name = Path(str(raw or "").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:96]
    if not name or name in (".", ".."):
        raise RigAnimError(f"非法文件名：{raw!r}")
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise RigAnimError(f"不支持的文件类型：{name}")
    return name


def _manifest_path(folder: Path) -> Path:
    return folder / "manifest.json"


def _read_manifest(folder: Path) -> Optional[Dict[str, Any]]:
    p = _manifest_path(folder)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def new_id() -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------- 输入 PSD


def save_upload(name: str, stage_id: str, filename: str, data: bytes) -> Dict[str, str]:
    """用户自己导入的分层 PSD。只认文件头 `8BPS`，不解码（解码交给编辑器的 ag-psd）。"""
    if not data:
        raise RigAnimError("空文件")
    if len(data) > MAX_UPLOAD_BYTES:
        raise RigAnimError(f"PSD 太大（上限 {MAX_UPLOAD_BYTES // 2**20}MB）")
    if data[:4] != b"8BPS":
        raise RigAnimError("不是 PSD 文件（缺 8BPS 文件头）")
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename or "layered").stem)[:40] or "layered"
    out_dir = root(name, stage_id) / UPLOAD_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{uuid.uuid4().hex[:10]}_{stem}.psd"
    out.write_bytes(data)
    return {"path": str(out), "url": _url(out), "name": out.name}


def resolve_part_psd(name: str, rid: str, stage_id: str) -> Dict[str, str]:
    """从「拆部件」产物里取那份标准分层 PSD（manifest 的 `psd` 是 /files/ URL）。"""
    mf = partsplit.load_manifest(name, rid, stage_id)
    if not mf:
        raise RigAnimError(f"没有这次拆件产物：{rid}")
    url = str(mf.get("psd") or "")
    if not url:
        raise RigAnimError("这份拆件没有生成 PSD（当时 psd_tools 不可用），请换一份或自己导入")
    rel = url.split("/files/", 1)[-1] if "/files/" in url else url.lstrip("/")
    p = (paths.ASSETS / rel).resolve()
    if not _inside(paths.ASSETS.resolve(), p) or not p.is_file():
        raise RigAnimError("PSD 文件不在资产目录里或已被删除")
    return {"path": str(p), "url": url, "name": str(mf.get("psdName") or p.name)}


def resolve_project(name: str, rid: str, stage_id: str) -> Dict[str, str]:
    """从历史产物里取回完整可编辑工程，供「继续编辑」打开。"""
    folder = root(name, stage_id) / os.path.basename(str(rid).strip().rstrip("/\\"))
    if not _manifest_path(folder).is_file():
        raise RigAnimError(f"没有这份骨骼动画产物：{rid}")
    p = folder / "project.rigby"
    if not p.is_file():
        raise RigAnimError("这份产物没有可编辑工程（旧版本发布时没存 project.rigby）")
    return {"path": str(p), "url": _url(p), "name": "project.rigby"}


def resolve_upload(name: str, file_id: str, stage_id: str) -> Dict[str, str]:
    """自己上传的 PSD：前端只回传文件名（不回传绝对路径），后端在 _uploads/ 里找。"""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(str(file_id or "")).name)[:96]
    if not safe.lower().endswith(".psd") or safe in (".", ".."):
        raise RigAnimError("要打开哪个 PSD？")
    up = root(name, stage_id) / UPLOAD_DIR_NAME
    p = (up / safe).resolve()
    if not _inside(up.resolve(), p) or not p.is_file():
        raise RigAnimError("这个 PSD 不存在或已被删除")
    return {"path": str(p), "url": _url(p), "name": safe}


# ---------------------------------------------------------------- 发布 / 列表


def save_result(name: str, stage_id: str, source: Dict[str, Any],
                files: Dict[str, bytes], meta: Dict[str, Any]) -> Dict[str, Any]:
    """把编辑器发来的全部导出文件落盘，写成一份 manifest。失败要回滚整个目录。"""
    if not files:
        raise RigAnimError("没有收到任何导出文件")
    # 3.8 / 4.2 / 4.3 JSON、.skel、图集与可编辑工程都是交付物的一部分：
    # 缺 4.2/4.3 会让列表少下载项、缺 project.rigby 会让「继续编辑」失效、缺 4.3
    # 会让 Godot 示例场景挑不到动画。一律在落盘前拦下，别静默产出不完整的一份。
    missing = [n for n in EXPECTED_FILES if n not in files]
    if missing:
        raise RigAnimError("导出不完整，缺：" + "、".join(missing))
    if len(files) > 24:
        raise RigAnimError("文件数量超出上限")
    checked: List[tuple] = []
    for fn, data in files.items():
        safe = _safe_filename(fn)
        if not isinstance(data, (bytes, bytearray)):
            raise RigAnimError(f"{safe} 不是字节数据")
        if len(data) > MAX_FILE_BYTES:
            raise RigAnimError(f"{safe} 超过单文件上限")
        checked.append((safe, bytes(data)))

    rid = new_id()
    folder = root(name, stage_id) / rid
    if folder.exists():
        raise RigAnimError("产物目录重名，请重试")
    folder.mkdir(parents=True)
    try:
        for safe, data in checked:
            (folder / safe).write_bytes(data)
        animation_names = _animation_names(folder / "skeleton-4.3.json")
        manifest: Dict[str, Any] = {
            "id": rid,
            "created": int(time.time()),
            "stage": stage_id or "default",
            "source": {
                "kind": str(source.get("kind") or ""),
                "label": str(source.get("label") or "")[:200],
                "name": str(source.get("name") or "")[:200],
            },
            "bones": int(meta.get("bones") or 0),
            "slots": int(meta.get("slots") or 0),
            "animations": int(meta.get("animations") or 0),
            "constraints": int(meta.get("constraints") or 0),
            "animationNames": animation_names,
            "warnings": [str(w)[:300] for w in (meta.get("warnings") or [])][:20],
            "files": [n for n, _ in checked],
            "formats": {n: _url(folder / n) for n, _ in checked},
        }
        _manifest_path(folder).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return {"id": rid, "stage": manifest["stage"], "formats": manifest["formats"]}


def _animation_names(path: Path) -> List[str]:
    """从 4.3 JSON 里读动画名，给 Godot 示例场景挑一个默认动画。读不到就空。"""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        anims = obj.get("animations")
        if isinstance(anims, dict):
            return [str(k) for k in anims.keys()][:32]
    except (OSError, ValueError, AttributeError):
        pass
    return []


def load_manifest(name: str, rid: str, stage_id: str = "") -> Optional[Dict[str, Any]]:
    folder = root(name, stage_id) / os.path.basename(str(rid).strip().rstrip("/\\"))
    if not _inside(root(name, stage_id), folder.resolve()):
        return None
    return _read_manifest(folder)


def _godot_zip(folder: Path, character: str) -> Optional[Path]:
    p = folder / "godot" / f"{character}_godot.zip"
    return p if p.is_file() else None


def list_results(name: str, stage_id: str = "") -> List[Dict[str, Any]]:
    """只列这个生命阶段的产物（跟 `/parts?stage=` 一个语义）。"""
    base = root(name, stage_id)
    if not base.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for folder in base.iterdir():
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        mf = _read_manifest(folder)
        if not mf or "formats" not in mf:
            continue
        z = _godot_zip(folder, name)
        out.append({
            "id": mf.get("id") or folder.name,
            "created": mf.get("created") or 0,
            "stage": mf.get("stage") or (stage_id or "default"),
            "source": mf.get("source") or {},
            "bones": mf.get("bones") or 0,
            "slots": mf.get("slots") or 0,
            "animations": mf.get("animations") or 0,
            "constraints": mf.get("constraints") or 0,
            "animationNames": mf.get("animationNames") or [],
            "warnings": mf.get("warnings") or [],
            "formats": mf.get("formats") or {},
            "godot": _url(z) if z else None,
        })
    out.sort(key=lambda x: (x["created"], x["id"]), reverse=True)
    return out


def delete_result(name: str, rid: str, stage_id: str = "") -> bool:
    base = root(name, stage_id)
    folder = (base / os.path.basename(str(rid).strip().rstrip("/\\"))).resolve()
    if not _inside(base.resolve(), folder) or not folder.is_dir():
        return False
    if not _manifest_path(folder).is_file():
        return False
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        # Windows 上刚下载过的 Godot 包 / 图集可能被别的进程短暂占着，
        # 别把裸 PermissionError 抛成 500，给用户一句能照做的提示。
        raise RigAnimError(f"删除失败，文件可能正被占用，请稍后重试：{exc}") from exc
    return True


# ---------------------------------------------------------------- Godot 工程包


def godot_bundle(name: str, rid: str, stage_id: str = "") -> Dict[str, Any]:
    """生成 Godot 4.3+ 工程包 zip：addon 配置 + 导出产物 + 可直接打开的示例场景。

    运行时本体是 Rust GDExtension，上游仓库不附预编译二进制；包里 README 写清
    从哪里取 / 怎么编。这一点必须在 UI 上说破，不能让用户以为解压就能跑。
    """
    folder = root(name, stage_id) / os.path.basename(str(rid).strip().rstrip("/\\"))
    mf = _read_manifest(folder)
    if not mf:
        raise RigAnimError(f"没有这次骨骼动画产物：{rid}")
    formats = dict(mf.get("formats") or {})
    need = ("skeleton.skel", "skeleton.atlas", "skeleton.png")
    for n in need:
        if not (folder / n).is_file():
            raise RigAnimError(f"产物缺文件：{n}")
    if not GODOT_ADDON.is_dir():
        raise RigAnimError(f"缺少 Godot addon：{GODOT_ADDON}")

    out_dir = folder / "godot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_godot.zip"

    anim = (mf.get("animationNames") or [""])[0]
    anim_line = f'animation = "{anim}"\n' if anim else ""
    scene = (
        "[gd_scene load_steps=2 format=3]\n\n"
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n\n'
        '[node name="Main" type="Node2D"]\n'
        'script = ExtResource("1")\n\n'
        '[node name="Rig" type="RigbySkeleton2D" parent="."]\n'
        'position = Vector2(480, 560)\n'
        'skeleton_file = "res://character/skeleton.skel"\n'
        'atlas_file = "res://character/skeleton.atlas"\n'
        + anim_line
    )
    script = (
        "extends Node2D\n\n"
        "@onready var rig: RigbySkeleton2D = $Rig\n\n\n"
        "func _ready() -> void:\n"
        "\t# 换动画：rig.set_animation_on(0, \"动画名\", true, 0.2)\n"
        "\tpass\n"
    )
    project = (
        "; 由 Meow 角色中心导出：rigby2d 骨骼动画 Godot 工程包\n\n"
        "config_version=5\n\n"
        "[application]\n\n"
        'config/name="Meow Rig"\n'
        'run/main_scene="res://main.tscn"\n'
        'config/features=PackedStringArray("4.3")\n\n'
        "[display]\n\n"
        "window/size/viewport_width=960\n"
        "window/size/viewport_height=600\n\n"
        "[rendering]\n\n"
        'renderer/rendering_method="gl_compatibility"\n'
        'renderer/rendering_method.mobile="gl_compatibility"\n'
    )
    readme = f"""# {name} · Godot 骨骼动画工程包

由 Meow 角色中心导出（rid：{rid}）。骨架格式是二进制 `.skel`（Spine 4.3）+ `.atlas` + 图集页，
另外附了 3.8 / 4.2 / 4.3 三份无损 JSON，可按需替换 `main.tscn` 里的 `skeleton_file`。

## 还差一步：运行时二进制

Rigby2D 的 Godot 运行时是 Rust 写的 GDExtension，上游源码仓库**不含预编译库**，
所以本包里 `addons/rigby2d/bin/` 是空的。二选一：

1. 从 Rigby2D Releases 下载对应平台的 `rigby2d-godot-addon-<版本>-<平台>.zip`，
   把里面的 `bin/rigby_godot.dll`（Windows x64）/ `librigby_godot.so`（Linux x64）/
   `librigby_godot.dylib`（macOS）放进本包的 `addons/rigby2d/bin/`。
2. 自己编译：`cd runtimes/rust && cargo build --release -p rigby-godot`，
   产物同样拷进 `addons/rigby2d/bin/`。

## 使用

- Godot **4.3 或更新版本**。
- 解压整个 zip，用 Godot 打开这个文件夹（里面有 `project.godot`），重启编辑器让扩展加载。
- 打开 `main.tscn`，选中 `Rig` 节点即可在属性里换动画、调 Playing。
- 需要 `RigbySkeleton2D` / `RigbyScene2D` 节点的完整能力，见 `addons/rigby2d/README.md`。

## 许可

addon 内 `THIRD_PARTY_NOTICES.md` 列出了 Godot 侧的第三方组件（含 godot-rust，MPL-2.0）。
编辑器源码为 MIT OR Apache-2.0；本包不含 Rigby2D 的专有演示角色与品牌素材。
"""

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.godot", project)
        z.writestr("main.tscn", scene)
        z.writestr("main.gd", script)
        z.writestr("README-GODOT.md", readme)
        z.writestr("addons/rigby2d/bin/README.txt",
                   "把对应平台的 rigby_godot.dll / librigby_godot.so / librigby_godot.dylib 放这里。\n"
                   "详见工程根目录 README-GODOT.md。\n")
        for f in GODOT_ADDON.iterdir():
            if f.is_file():
                z.write(f, f"addons/rigby2d/{f.name}")
        for n in EXPECTED_FILES:
            p = folder / n
            if p.is_file():
                z.write(p, f"character/{n}")

    return {"url": _url(out), "name": out.name, "formats": formats}
