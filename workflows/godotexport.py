"""成片集 → Godot 4 工程包（`.tres` + `.tscn` + 逐帧 PNG + zip）。

照着 `D:\\code\\pixelforge` 的 `backend/app/pipelines/godot2d.py` 做的，
那一套在产线上跑过、坑都踩平了，这里只保留我们需要的形态：

**为什么直接生成资源文件，而不是导出 JSON 让用户自己拼**：Godot 的资源引用是
`res://` 绝对路径，SpriteFrames 又必须是资源文件（`load_steps` + `ext_resource`
这套结构）。这些是确定性文本，让用户手写没有意义 —— 直接给，拖进工程双击就能播。

**只用 `files` 模式（逐帧 PNG），不做图集**：我们的帧本来就是一个一个的 PNG，
拍平成图集要多一层拼板 + 坐标，一旦列数/单元格和参数对不上就会导出一个
「能播但错位」的包（pixelforge 那边专门为此加了尺寸自检）。逐帧没有这个风险，
代价只是文件多几个。

**包内路径必须自洽**：`.tres` 里写的是 `res://<folder>/frames/frame_000.png`，
所以 zip 的条目名都带 `<folder>/` 前缀 —— 解到工程根目录后路径恰好成立。
写完会自检一遍（每个 `res://` 引用都能在包内找到），不通过就在结果里报出来。
"""
from __future__ import annotations

import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from workflows import characters
from workflows import directions
from workflows import paths

#: 文件名主干 / 目录段的最大长度
_STEM_MAX = 40
_FOLDER_SEGS = 4
#: CanvasItem.TextureFilter：像素素材必须最近邻，否则糊成一片
_FILTER_VALUE = {"nearest": 1, "linear": 2}
#: 导出根（项目目录下）
EXPORT_DIR = "exports"


class GodotExportError(RuntimeError):
    """可读错误（API 转 400）。"""


# ---------------------------------------------------------------- 命名与转义


def safe_stem(value: Any, default: str = "anim") -> str:
    """文件名主干：只留 `[a-z0-9_-]`。

    **不做音译**：中文动画名（「行走」）译出来用户也认不出，统一退回 default；
    动画名本身仍然可以是中文（写在 `.tres` 的 `name` 里，Godot 允许）。
    """
    raw = str(value or "").strip().lower()
    out = re.sub(r"[^a-z0-9_-]+", "", raw).strip("-_")[:_STEM_MAX]
    return out or default


def safe_folder(value: Any, default: str = "sprites") -> str:
    """`res://` 下的相对目录：接受 `res://` 前缀，逐段清洗，`..` 一律丢掉。"""
    raw = str(value or "").strip().replace("\\", "/")
    if raw.lower().startswith("res://"):
        raw = raw[6:]
    segs: List[str] = []
    for seg in raw.split("/"):
        if seg in ("", ".", ".."):
            continue
        clean = re.sub(r"[^a-z0-9_-]+", "", seg.lower()).strip("-_")
        if clean:
            segs.append(clean[:_STEM_MAX])
    return "/".join(segs[:_FOLDER_SEGS]) or default


def _q(value: Any) -> str:
    """Godot 文本资源里的字符串转义（双引号 / 反斜杠）。"""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _node_name(value: Any, default: str = "Anim") -> str:
    """场景节点名：Godot 不允许 `. : @ / " %` 这几个字符。"""
    raw = str(value or "").strip() or default
    return re.sub(r'[.:@/"%]', "_", raw)[:40] or default


# ---------------------------------------------------------------- 资源文本


def build_sprite_frames(*, folder: str, count: int, anim_name: str,
                        fps: float, loop: bool) -> str:
    """Godot 4 的 SpriteFrames 资源文本（format=3，逐帧 ext_resource）。"""
    n = max(1, int(count))
    speed = round(max(0.1, float(fps)), 3)
    lines = [f'[gd_resource type="SpriteFrames" load_steps={n + 1} format=3]', ""]
    body: List[str] = []
    for i in range(n):
        lines.append('[ext_resource type="Texture2D" '
                     'path="res://{}/frames/frame_{:03d}.png" id="1_{:03d}"]'
                     .format(folder, i, i))
        body.append('{{\n"duration": 1.0,\n"texture": ExtResource("1_{:03d}")\n}}'
                    .format(i))
    lines.append("[resource]")
    lines.append("animations = [{")
    lines.append('"frames": [')
    lines.append(", ".join(body))
    lines.append("],")
    lines.append(f'"loop": {str(bool(loop)).lower()},')
    lines.append(f'"name": &"{_q(anim_name or "default")}",')
    lines.append(f'"speed": {speed}')
    lines.append("}]")
    lines.append("")
    return "\n".join(lines)


def build_sprite_frames_multi(*, folder: str,
                              animations: Sequence[Dict[str, Any]]) -> str:
    """多动画 SpriteFrames：一个方向一条动画，共用一个 `.tres`。

    ``animations`` 每项 ``{name, files, fps, loop}``；``files`` 是**包内相对
    路径**（``frames/frame_000.png``）。纹理编号全局连续，所以
    ``verify_bundle`` 的路径自检照常生效。

    ⚠️ 单动画仍走 ``build_sprite_frames``：那段文本已有逐字断言，别改它。
    """
    specs = [a for a in animations if a.get("files")]
    total = sum(len(a["files"]) for a in specs)
    lines = [f'[gd_resource type="SpriteFrames" load_steps={total + 1} format=3]', ""]
    blocks: List[str] = []
    idx = 0
    for a in specs:
        body: List[str] = []
        for rel in a["files"]:
            rid = f"1_{idx:03d}"
            lines.append('[ext_resource type="Texture2D" '
                         'path="res://{}/{}" id="{}"]'.format(folder, rel, rid))
            body.append('{{\n"duration": 1.0,\n"texture": ExtResource("{}")\n}}'
                        .format(rid))
            idx += 1
        speed = round(max(0.1, float(a.get("fps") or 12.0)), 3)
        blocks.append("\n".join([
            "{", '"frames": [', ", ".join(body), "],",
            f'"loop": {str(bool(a.get("loop", True))).lower()},',
            f'"name": &"{_q(a.get("name") or "default")}",',
            f'"speed": {speed}', "}",
        ]))
    lines += ["[resource]", "animations = [", ", ".join(blocks), "]", ""]
    return "\n".join(lines)


def build_animation_manifest(*, folder: str, stem: str,
                             animations: Sequence[Dict[str, Any]],
                             extra: Optional[Dict[str, Any]] = None) -> str:
    """``animations.json``：方向 ↔ 动画名 ↔ 帧率 ↔ 机位的清单。

    引擎侧的导入脚本 / 验收工具按它把动画挂到方向状态机，不用去解析 `.tres`。
    方向 key 保持英文原值（项目约定：标识值原样传，只在渲染处中文化）。
    """
    import json

    items = [{
        "name": a["name"],
        "direction": a.get("direction") or "",
        "yaw": a.get("yaw", 0),
        "frames": len(a["files"]),
        "frameRate": round(max(0.1, float(a.get("fps") or 12.0)), 3),
        "loop": bool(a.get("loop", True)),
    } for a in animations if a.get("files")]
    return json.dumps({
        "resource": f"res://{folder}/{stem}.tres",
        "default": items[0]["name"] if items else "default",
        "animations": items,
        **(extra or {}),
    }, ensure_ascii=False, indent=1) + "\n"


def build_scene(*, folder: str, stem: str, anim_name: str, node_name: str,
                cell: int, origin: str = "bottom",
                texture_filter: str = "nearest") -> str:
    """AnimatedSprite2D 场景（format=3）。

    `origin=bottom` 把画面底边（脚底）对到节点原点 —— 这样 `set_position(地面)`
    角色是站在地上的；居中会让它埋进去半截（pixelforge 那边踩过）。
    """
    filt = _FILTER_VALUE.get(texture_filter, 1)
    h = max(1, int(cell))
    offset = (f"Vector2(0, {-h / 2:g})" if origin == "bottom"
              else f"Vector2(0, {h / 2:g})" if origin == "top" else "Vector2(0, 0)")
    name = _q(anim_name or "default")
    return "\n".join([
        "[gd_scene load_steps=2 format=3]",
        "",
        '[ext_resource type="SpriteFrames" path="res://{}/{}.tres" id="1_frames"]'
        .format(folder, stem),
        "",
        '[node name="{}" type="AnimatedSprite2D"]'.format(_q(node_name)),
        f"texture_filter = {filt}",
        'sprite_frames = ExtResource("1_frames")',
        'animation = &"{}"'.format(name),
        'autoplay = "{}"'.format(name),
        "centered = true",
        f"offset = {offset}",
        "",
    ])


def build_readme(*, folder: str, stem: str, anim_name: str, count: int,
                 fps: float, loop: bool, cell: Tuple[int, int],
                 origin: str, anim_names: Sequence[str] = ()) -> str:
    """三步导入说明。**必须中文 + 写清 res:// 路径** —— 十次"导不进去"九次是路径。"""
    origin_text = {
        "bottom": "脚底（画面底边）在节点原点",
        "top": "顶部在节点原点",
        "center": "中心在节点原点",
    }.get(origin, "中心在节点原点")
    sample = f"res://{folder}/{stem}.tscn"
    text = f"""2D 游戏素材工具 · Godot 4 · 2D 序列动画
生成时间：{time.strftime('%Y-%m-%d %H:%M')}

动画名：{anim_name or 'default'}    帧数：{count}    帧率：{fps:g} fps    循环：{'是' if loop else '否'}
单帧尺寸：{cell[0]}×{cell[1]}

一、放进 Godot 工程（路径必须一致）
  把本包里的「{folder.split('/')[-1]}」整个文件夹拷到 Godot 工程根目录
  （和 project.godot 同级），让路径恰好是：
      res://{folder}/{stem}.tres
      res://{folder}/{stem}.tscn
      res://{folder}/frames/frame_000.png …
  资源文件里写死的就是这些路径 —— 放到别处 Godot 会报「资源缺失」
  （不是文件坏了，是路径对不上）。拷完回 Godot 等导入转圈结束。

二、使用（任选）
  A. 代码实例化整个场景：
         var anim = preload("{sample}").instantiate()
         add_child(anim)
         anim.play("{anim_name or 'default'}")
     autoplay 已经设好，add_child 之后就会自己播。
  B. 把 {stem}.tscn 从文件系统面板拖进场景树，当子节点用。
  C. 只要动画资源：把 {stem}.tres 拖到任意 AnimatedSprite2D 的
     Sprite Frames 属性上。

三、已经替你设好的
  · AnimatedSprite2D，autoplay = "{anim_name or 'default'}"：挂上就播。
  · texture_filter = 1（最近邻）：像素素材不会被糊成一片，不用改项目设置。
  · 原点：{origin_text}（想改：节点检查器里调 Offset）。
  · 循环 {'loop = true' if loop else 'loop = false'}，{fps:g} fps。

四、常见问题
  · 播放空白/黑块 → 多半是路径不对（见第一节），看输出面板缺哪个文件。
  · 边缘毛刺 → 确认 texture_filter 仍是 1；自定义材质也要设最近邻。
  · 角色位置偏上/偏下 → offset 是按单帧尺寸 {cell[1]} 算的，换了尺寸要同步改。
"""
    if len(anim_names) > 1:
        text += (
            f"\n五、多方向动画（{len(anim_names)} 条，一个方向一条）\n"
            + "".join(f"  · {n}\n" for n in anim_names)
            + "  animations.json 是「方向 ↔ 动画名 ↔ 帧率 ↔ 机位」的清单，\n"
              "  方向状态机照它切换；autoplay 指向第一条。\n"
        )
    return text


def verify_bundle(root: Path, folder: str) -> Dict[str, Any]:
    """包内自检：`.tres` / `.tscn` 里每个 `res://` 引用都能在包内找到。

    针对的真实事故：`.tres` 里的目录名和实际落盘差一层（或大小写不同），
    Godot 打开一片空白，用户以为导出坏了。包内自洽 = 解到工程根目录就一定能用。
    """
    missing: List[str] = []
    refs = 0
    for f in sorted(root.glob("*.tres")) + sorted(root.glob("*.tscn")):
        try:
            body = f.read_text(encoding="utf-8")
        except OSError as exc:
            missing.append(f"{f.name}: 读不出来（{exc}）")
            continue
        for m in re.finditer(r'path="res://([^"]+)"', body):
            refs += 1
            rel = m.group(1)
            pre = folder + "/"
            target = root / (rel[len(pre):] if rel.startswith(pre) else rel)
            if not target.is_file():
                missing.append(f"{f.name} → res://{rel}")
    return {"pathsMatch": not missing, "missing": missing[:8], "refsChecked": refs}


# ---------------------------------------------------------------- 主流程


def _render_entries(project: str, render: str
                    ) -> Tuple[Dict[str, Any], List[Tuple[Path, Dict[str, Any]]], int]:
    """成片集里**每帧当前选中的那张**，连同帧表条目一起返回。

    保留帧表条目是为了拿到 ``index`` —— 方向信息不在成片集里，而在它关联的
    姿势集 ``set.json`` 上，靠 index 才能对上。
    """
    from workflows import poseset

    folder = poseset._sub(project, poseset.RENDERS_DIR, render)   # noqa: SLF001
    doc = poseset._read_json(folder / poseset.RENDER_FILE)       # noqa: SLF001
    if not doc:
        raise GodotExportError(f"没有这个成片集：{render}")
    out: List[Tuple[Path, Dict[str, Any]]] = []
    skipped = 0
    for f in doc.get("frames") or []:
        cands = f.get("candidates") or []
        sel = int(f.get("selected") if f.get("selected") is not None else -1)
        p = folder / str(cands[sel].get("file") or "") if 0 <= sel < len(cands) else None
        if p and p.is_file():
            out.append((p, f))
        else:
            skipped += 1
    return doc, out, skipped


def frame_files(project: str, render: str) -> Tuple[List[Path], int]:
    """老入口：只要路径（顺序 = 成片集帧序），行为保持一字不变。"""
    _, entries, skipped = _render_entries(project, render)
    return [p for p, _ in entries], skipped


def _direction_index(project: str, pose_set: str) -> Dict[int, str]:
    """读关联姿势集的 ``set.json``，建立 ``帧 index → 方向``。

    方向是多机位抽帧时宿主写上去的（``poseset.EXTRA_FRAME_KEYS`` 透传）。
    老姿势集没有这个字段 → 返回空表 → 整个成片集退回单动画，行为不变。
    """
    from workflows import poseset

    if not pose_set:
        return {}
    folder = poseset._sub(project, poseset.POSES_DIR, pose_set)   # noqa: SLF001
    doc = poseset._read_json(folder / poseset.SET_FILE)          # noqa: SLF001
    out: Dict[int, str] = {}
    for f in doc.get("frames") or []:
        if not isinstance(f, dict):
            continue
        key = directions.validate_direction(f.get("direction"))
        if key:
            try:
                out[int(f.get("index") or 0)] = key
            except (TypeError, ValueError):
                continue
    return out


def frame_groups(project: str, render: str
                 ) -> Tuple[List[Dict[str, Any]], int]:
    """按方向把成片分组；无方向信息时只有一组（老行为）。

    返回 ``[{"direction", "yaw", "files": [Path...]}]``。方向顺序按
    ``topdown-8`` 表，混合了老帧的无方向组排最后，保证 autoplay 落在
    真正的方向动画上。
    """
    doc, entries, skipped = _render_entries(project, render)
    idx_dir = _direction_index(project, str(doc.get("poseSet") or ""))
    order_rank = {d: i for i, d in enumerate(directions.DIRECTIONS["topdown-8"])}
    groups: Dict[str, Dict[str, Any]] = {}
    for path, frame in entries:
        key = idx_dir.get(int(frame.get("index") or 0), "")
        if key not in groups:
            groups[key] = {"direction": key,
                           "yaw": directions.yaw_of(key), "files": []}
        groups[key]["files"].append(path)
    keys = sorted(groups, key=lambda k: (k == "", order_rank.get(k, 99)))
    return [groups[k] for k in keys], skipped


def export_bundle(project: str, render: str, *,
                  stem: str = "", anim_name: str = "",
                  fps: Optional[float] = None, loop: Optional[bool] = None,
                  template_action: str = "", folder: str = "sprites",
                  origin: str = "bottom",
                  texture_filter: str = "nearest") -> Dict[str, Any]:
    """成片集 → `exports/<stem>/`（文件夹）+ `exports/<stem>.zip`。

    同名重导整目录重建（顺手清掉上一次多出来的帧文件），不产生一堆历史包。

    ``fps`` / ``loop`` 缺省时顺着 ``render.json.poseSet`` →
    ``set.json.source.templateId`` 从动作模板取；模板也没有才用 12 fps / 循环。
    显式传参永远优先（旧调用方传 12.0 / True 时行为完全不变）。
    """
    from PIL import Image

    groups, skipped = frame_groups(project, render)
    files = [p for g in groups for p in g["files"]]
    if not files:
        raise GodotExportError("这套成片还没有选中的帧：先生成并挑好图再导出")
    # 只有真的抽出 ≥2 个方向才走多动画路径；老成片集仍然生成逐字相同的
    # 单动画 .tres（test_godotexport.py 对那段文本有精确断言）。
    multi = len(groups) > 1

    stem_ = safe_stem(stem or anim_name or render, "anim")
    folder_ = safe_folder(folder, "sprites")
    anim = str(anim_name or render or "default").strip() or "default"

    # ---- 播放参数：显式 > 动作模板 > 默认 ----
    # 之前 fps/loop 全靠 API 常数，动作的时序信息在导出环节凭空消失。
    # 抽帧选了模板就把 templateId 写进 set.json.source，这里顺着成片集反查。
    template_id = ""
    pb: Dict[str, Any] = {}
    if fps is not None and loop is not None:
        playback_source = "explicit"
    else:
        from workflows import motiontemplates
        from workflows import poseset

        render_doc = poseset._read_json(                                 # noqa: SLF001
            poseset._sub(project, poseset.RENDERS_DIR, render)          # noqa: SLF001
            / poseset.RENDER_FILE) or {}
        tpl = motiontemplates.template_for_pose_set(
            project, str(render_doc.get("poseSet") or ""))
        if tpl:
            pb = motiontemplates.playback(tpl, template_action)
            template_id = tpl["id"]
            playback_source = "template"
        else:
            playback_source = "default"
        if fps is not None or loop is not None:
            playback_source += "+partial-explicit"
    fps_ = max(0.1, min(float(fps if fps is not None
                              else pb.get("fps", 12.0)), 60.0))
    loop_ = bool(loop if loop is not None else pb.get("loop", True))
    origin_ = origin if origin in ("bottom", "center", "top") else "bottom"
    filt = texture_filter if texture_filter in _FILTER_VALUE else "nearest"

    project_root = paths.PROJECTS_DIR / _scope(project)
    out_root = project_root / EXPORT_DIR
    out_dir = out_root / stem_
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 帧尺寸：Godot 不会因为尺寸不同而报错，但播起来会抖 —— 记下来提示
    size: Tuple[int, int] = (0, 0)
    size_mismatch = 0
    anim_specs: List[Dict[str, Any]] = []
    cursor = 0
    for group in groups:
        names: List[str] = []
        for src in group["files"]:
            dst = frames_dir / f"frame_{cursor:03d}.png"
            shutil.copyfile(src, dst)
            try:
                with Image.open(dst) as im:
                    if not size[0]:
                        size = im.size
                    elif im.size != size:
                        size_mismatch += 1
            except OSError:
                pass
            names.append(f"frames/frame_{cursor:03d}.png")
            cursor += 1
        label = (f"{anim}@{group['direction']}"
                 if multi and group["direction"] else anim)
        anim_specs.append({"name": label, "files": names, "fps": fps_,
                           "loop": loop_, "direction": group["direction"],
                           "yaw": group["yaw"]})
    count = len(files)

    tres_text = (build_sprite_frames_multi(folder=folder_, animations=anim_specs)
                 if multi else
                 build_sprite_frames(folder=folder_, count=count, anim_name=anim,
                                     fps=fps_, loop=loop_))
    (out_dir / f"{stem_}.tres").write_text(tres_text, encoding="utf-8")
    primary = anim_specs[0]["name"] if anim_specs else anim
    (out_dir / f"{stem_}.tscn").write_text(
        build_scene(folder=folder_, stem=stem_, anim_name=primary,
                    node_name=_node_name(stem_), cell=size[1] or 128,
                    origin=origin_, texture_filter=filt), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        build_readme(folder=folder_, stem=stem_, anim_name=primary, count=count,
                     fps=fps_, loop=loop_, cell=size, origin=origin_,
                     anim_names=[a["name"] for a in anim_specs] if multi else []),
        encoding="utf-8")
    manifest_path = out_dir / "animations.json"
    if multi:
        manifest_path.write_text(
            build_animation_manifest(
                folder=folder_, stem=stem_, animations=anim_specs,
                extra={"playbackSource": playback_source,
                       "templateId": template_id,
                       "templateAction": pb.get("action", "")}), encoding="utf-8")

    checks = verify_bundle(out_dir, folder_)

    zip_path = out_root / f"{stem_}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                z.writestr(f"{folder_}/{f.relative_to(out_dir).as_posix()}",
                           f.read_bytes())

    warnings: List[str] = []
    if not checks["pathsMatch"]:
        warnings.append("包内路径自检未通过：" + "；".join(checks["missing"]))
    if size_mismatch:
        warnings.append(f"{size_mismatch} 帧尺寸与第 1 帧不同，播起来会抖（建议重生成）")
    if skipped:
        warnings.append(f"{skipped} 帧还没选中的候选，没导进包里")
    if count > 120:
        warnings.append(f"{count} 帧的逐帧动画在 Godot 里首帧加载会偏慢")

    def url(p: Path) -> str:
        try:
            return "/files/" + p.resolve().relative_to(paths.ASSETS.resolve()).as_posix()
        except (ValueError, OSError):
            return ""

    return {
        "ok": True,
        "stem": stem_,
        "folder": folder_,
        "frames": count,
        "skipped": skipped,
        "fps": fps_,
        "loop": loop_,
        "playbackSource": playback_source,
        "templateId": template_id,
        "animName": primary,
        "animationCount": len(anim_specs),
        "multiAnimation": multi,
        "animations": [
            {"name": a["name"], "direction": a["direction"], "yaw": a["yaw"],
             "frames": len(a["files"]), "frameRate": a["fps"], "loop": a["loop"]}
            for a in anim_specs
        ],
        "size": list(size),
        "origin": origin_,
        "dir": str(out_dir),
        "files": [
            {"name": f.name, "bytes": f.stat().st_size, "url": url(f)}
            for f in (out_dir / f"{stem_}.tres", out_dir / f"{stem_}.tscn",
                      out_dir / "README.txt", manifest_path)
            if f.is_file()
        ],
        "manifest": ({"name": manifest_path.name, "bytes": manifest_path.stat().st_size,
                      "url": url(manifest_path)} if manifest_path.is_file() else None),
        "zip": {"name": zip_path.name, "bytes": zip_path.stat().st_size, "url": url(zip_path)},
        "resPaths": {
            "tres": f"res://{folder_}/{stem_}.tres",
            "scene": f"res://{folder_}/{stem_}.tscn",
        },
        "checks": checks,
        "warnings": warnings,
    }


def _scope(project: str) -> str:
    from workflows import projectstore

    return projectstore.safe_scope(project)
