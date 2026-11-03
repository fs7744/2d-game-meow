"""rig 数据 → Godot 4 cutout 骨骼工程（Node2D 层级 + AnimationPlayer + zip）。

和逐帧那条线（`godotexport.py`）的分工：

| | 逐帧（已有） | 骨骼（本文件） |
|---|---|---|
| 产物 | `AnimatedSprite2D` + 每帧一张 PNG | `Node2D` 树 + 每个部件一张 PNG |
| 加动作 | 重新渲染一整套帧 | 多一段关键帧（几 KB） |
| 包体 | 帧数 × 单帧大小 | 部件图 + 轨道数 × 关键帧数 |
| 画面 | **忠实**：模型画成什么样就是什么样 | 部件刚性旋转，膝肘不会弯 |

结论：**骨骼不是替代逐帧，是给「要换动作、要小包体」的那批角色用**。
最终画面质量取决于拆件质量，跟这里无关 —— 所以本文件不做任何"修补"，
只做翻译。

三个格式细节（写错不报错、只是播不对，所以都在文件里写清）：

1. **`offset` 表达 pivot**。Sprite2D 默认 `centered = true`（纹理中心在原点），
   我们要的是「顶端中点 / 底端中点」在原点，就是 `offset = (0, ±h/2)`。
2. **`rotation` 是顺时针为正**（Godot 2D 的 y 轴向下）。运动学生成那一侧
   （`rigbuild._rad`）已经把符号翻好了，这里**原样写入，不再取反**。
3. **`AnimationPlayer.root_node` 默认 `..`**：AnimationPlayer 挂在 `Rig` 下，
   所以轨道路径写 `torso:position` 就指向 `Rig/torso`。别写成 `/root/Rig/torso`。
"""
from __future__ import annotations

import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from workflows import rigmodel

#: 文件名主干 / 目录段最大长度（和 godotexport 保持一致）
_STEM_MAX = 40
_FOLDER_SEGS = 4
#: CanvasItem.TextureFilter：像素素材必须最近邻
_FILTER_VALUE = {"nearest": 1, "linear": 2}
#: Animation.loop_mode：0 = 不循环，1 = 线性循环，2 = 乒乓
_LOOP_MODE = {"none": 0, "linear": 1, "pingpong": 2}


class GodotRigError(RuntimeError):
    """可读错误（API 转 400）。"""


# ---------------------------------------------------------------- 命名与转义


def safe_stem(value: Any, default: str = "rig") -> str:
    raw = str(value or "").strip().lower()
    out = re.sub(r"[^a-z0-9_-]+", "", raw).strip("-_")[:_STEM_MAX]
    return out or default


def safe_folder(value: Any, default: str = "rig_out") -> str:
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
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _f(value: Any) -> str:
    """Godot 文本资源里的浮点数：定点、去尾零、**但一定带小数点**。

    两个坑：
    1. 不用 `:g`（会产出 `1e-05`）：Godot 的变体解析能读，但混在
       `PackedFloat32Array` 里一眼看不出对不对，review 时容易放过真错误。
    2. **`0` 和 `0.0` 在 Godot 里不是一回事**。`rotation` 这类 float 属性的
       关键帧若写成裸整数 `0`，解析成 int，AnimationMixer 应用时不会把它写进
       float 属性 —— 结果是「position 动了、rotation 一动不动」，而且**零警告**。
       真机验出来的：把 `[0, -0.160437, ...]` 改成 `[0.0, -0.160437, ...]`
       之后手臂才摆起来。Vector2 的分量是 int 反而没事（构造器会转），
       但统一带小数点不会错。
    """
    s = f"{float(value):.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    if s in ("", "-0.0"):
        return "0.0"
    return s


def _node_name(value: Any, default: str = "Rig") -> str:
    raw = str(value or "").strip() or default
    return re.sub(r'[.:@/"%]', "_", raw)[:40] or default


# ---------------------------------------------------------------- 载入 rig


def load_rig(rig_dir: Path) -> Dict[str, Any]:
    """读一套 rig 目录并校验。缺动作不算错（可以只导出骨架）。"""
    d = Path(rig_dir)
    sk_path = d / "skeleton.json"
    bd_path = d / "binding.json"
    if not sk_path.is_file():
        raise GodotRigError(f"缺 skeleton.json：{sk_path}（先建骨，再导出）")
    if not bd_path.is_file():
        raise GodotRigError(f"缺 binding.json：{bd_path}")
    skeleton = rigmodel.load(sk_path, "skeleton")
    binding = rigmodel.load(bd_path, "character-binding", skeleton)

    actions: Dict[str, Dict[str, Any]] = {}
    adir = d / "actions"
    if adir.is_dir():
        for p in sorted(adir.glob("*.json")):
            m = rigmodel.load(p, "motion-clip", skeleton)
            actions[safe_stem(m.get("name") or p.stem, p.stem)] = m
    return {"skeleton": skeleton, "binding": binding, "actions": actions, "dir": d}


# ---------------------------------------------------------------- 资源文本


def bone_paths(skeleton: Dict[str, Any]) -> Dict[str, str]:
    """骨 id → 相对场景根的节点路径（根骨是 "."，因为它由场景根自己承担）。

    **轨道路径必须是完整路径，不能只写骨名。** Godot 对解析不了的 NodePath
    只打一条 `couldn't resolve track` 警告，播放时那条轨道就是不动 ——
    在编辑器里看起来"动画能播"，只有手臂/腿不摆，极难查。这条是拿真机验出来的。
    """
    out: Dict[str, str] = {}
    for b in skeleton["bones"]:
        p = b.get("parentId")
        if p is None:
            out[b["id"]] = "."
        else:
            out[b["id"]] = b["id"] if out.get(p, ".") == "." else f"{out[p]}/{b['id']}"
    return out


def build_animation(skeleton: Dict[str, Any], motion: Dict[str, Any],
                    *, root_prefix: str = "") -> str:
    """一段动作 → Godot 4 `Animation` 资源文本。

    轨道顺序固定（先骨序、后属性序）：Godot 不关心，但固定了 diff 才稳定，
    多人改同一份导出物时不会天天冲突。
    """
    paths = bone_paths(skeleton)
    for tr in motion["tracks"]:
        if paths.get(tr["targetId"], ".") == ".":
            raise GodotRigError(
                f"根骨（{tr['targetId']}）不能有动画轨道：它对应场景根节点，"
                "整体位移请用 Rig 节点的 position，别烘进动作里")
    bone_order = {b["id"]: i for i, b in enumerate(skeleton["bones"])}
    prop_order = {"position": 0, "rotation": 1, "scale": 2, "visible": 3}
    tracks = sorted(motion["tracks"],
                    key=lambda t: (bone_order.get(t["targetId"], 1 << 20),
                                   prop_order.get(t["property"], 9)))
    loop_mode = _LOOP_MODE["linear"] if motion.get("loop") else _LOOP_MODE["none"]
    length = float(motion.get("duration") or 0.0)
    fps = float(motion.get("fps") or 24.0)
    step = 1.0 / fps if fps > 0 else 0.0

    lines = ['[gd_resource type="Animation" format=3]', "", "[resource]",
             f'resource_name = "{_q(motion.get("name") or "anim")}"',
             f"length = {_f(length)}"]
    if loop_mode:
        lines.append(f"loop_mode = {loop_mode}")
    if step:
        lines.append(f"step = {_f(step)}")
    lines.append("")

    for ti, tr in enumerate(tracks):
        prop = tr["property"]
        interp = rigmodel.INTERP_TO_GODOT[tr["interpolation"]]
        # 完整节点路径（torso/arm_l），不是骨名 —— 见 bone_paths 的注释
        path = f'{paths[tr["targetId"]]}:{prop}'
        kfs = tr["keyframes"]
        times = ", ".join(_f(k["time"]) for k in kfs)
        values = ", ".join(_fmt_value(prop, k["value"]) for k in kfs)
        lines += [
            f"tracks/{ti}/type = \"value\"",
            f"tracks/{ti}/imported = false",
            f"tracks/{ti}/enabled = true",
            f'tracks/{ti}/path = NodePath("{_q(path)}")',
            f"tracks/{ti}/interp = {interp}",
            f"tracks/{ti}/loop_wrap = true",
            f"tracks/{ti}/keys = {{",
            f'"times": PackedFloat32Array({times}),',
            f'"transitions": PackedFloat32Array({", ".join(["1"] * len(kfs))}),',
            '"update": 0,',
            f'"values": [{values}]',
            "}",
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _fmt_value(prop: str, value: Any) -> str:
    if prop == "rotation":
        return _f(value)
    if prop == "visible":
        return "true" if float(value) >= 0.5 else "false"
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise GodotRigError(f"{prop} 的值必须是 [x, y]，收到 {value!r}")
    return f"Vector2({_f(value[0])}, {_f(value[1])})"


def build_library(folder: str, motion_ids: Sequence[str]) -> str:
    """AnimationLibrary 资源文本。

    `_data` 的键是动画名（Godot 里 `play("walk")` 用的就是它）。
    """
    lines = [f'[gd_resource type="AnimationLibrary" load_steps={len(motion_ids) + 1} '
             "format=3]", ""]
    for i, mid in enumerate(motion_ids):
        lines.append('[ext_resource type="Animation" '
                     f'path="res://{folder}/actions/{mid}.tres" id="{i + 1}_{mid}"]')
    lines += ["", "[resource]", "_data = {"]
    for i, mid in enumerate(motion_ids):
        lines.append(f'"{_q(mid)}": ExtResource("{i + 1}_{mid}"),')
    lines += ["}", ""]
    return "\n".join(lines)


def build_scene(*, folder: str, stem: str, skeleton: Dict[str, Any],
                binding: Dict[str, Any], textures: Dict[str, Tuple[int, int]],
                library_name: Optional[str], motion_ids: Sequence[str],
                texture_filter: str = "nearest") -> str:
    """场景文本：`Rig`(Node2D) → 每根骨一个 Node2D → 每个 slot 一个 Sprite2D。

    骨的 `position` / `rotation` 写 **rest 值**；动画轨道会覆盖它们。
    `z_index` 按绑定的 z 写死（方向差异就是靠它，不靠节点顺序 —— 节点顺序
    由层级决定，改不动）。
    """
    filt = _FILTER_VALUE.get(texture_filter, 1)
    bones = {b["id"]: b for b in skeleton["bones"]}
    slots = binding["slots"]
    by_bone: Dict[str, List[Dict[str, Any]]] = {}
    for s in slots:
        by_bone.setdefault(s["boneId"], []).append(s)

    # res:// 引用：每张部件图 + 动画库
    ext: List[str] = []
    tex_id: Dict[str, str] = {}
    for i, s in enumerate(sorted(slots, key=lambda x: x["id"])):
        key = f"1_tex_{i}"
        tex_id[s["id"]] = key
        ext.append('[ext_resource type="Texture2D" path="res://{}/{}" id="{}"]'
                   .format(folder, s["texture"], key))
    if library_name and motion_ids:
        ext.append('[ext_resource type="AnimationLibrary" '
                   f'path="res://{folder}/{library_name}.tres" id="2_lib"]')
    load_steps = len(ext) + 1

    lines = [f"[gd_scene load_steps={load_steps} format=3]", ""]
    lines += ext
    lines += ["", '[node name="{}" type="Node2D"]'.format(_node_name(stem)),
              f"texture_filter = {filt}", ""]

    # 层级：按骨序建 Node2D，parent 用完整相对路径。
    # **根骨不建节点**：场景根就是它（位置由游戏代码 set_position 决定），
    # 多挂一个空 Node2D 会让 AnimationPlayer 的路径全部多一层。
    path_of = bone_paths(skeleton)
    for b in skeleton["bones"]:
        bid = b["id"]
        parent = b.get("parentId")
        if parent is None:
            for s_ in by_bone.get(bid, []):
                raise GodotRigError(
                    f"部件 {s_['id']} 挂到了根骨上：根骨没有独立节点，部件会丢")
            continue
        rest = b["rest"]
        lines.append('[node name="{}" type="Node2D" parent="{}"]'.format(
            _q(bid), _q(path_of[parent])))
        if rest["x"] or rest["y"]:
            lines.append(f"position = Vector2({_f(rest['x'])}, {_f(rest['y'])})")
        if rest["rotation"]:
            lines.append(f"rotation = {_f(rest['rotation'])}")
        lines.append("")

        for s in by_bone.get(bid, []):
            # offset 由 rigbuild 算好（= 图中心 - 关节在图内的位置）。
            # **这里不要再按 pivot 加 ±h/2**：那是「枢轴在切区边」时代的算法，
            # 两套一叠加，部件会整体偏半个身高。
            off = s.get("offset") or [0, 0]
            off_x, off_y = float(off[0]), float(off[1])
            lines.append('[node name="{}" type="Sprite2D" parent="{}"]'.format(
                _q(s["id"]), _q(path_of[bid])))
            lines.append(f"z_index = {int(s.get('z') or 0)}")
            lines.append(f"texture_filter = {filt}")
            lines.append(f'texture = ExtResource("{tex_id[s["id"]]}")')
            lines.append(f"offset = Vector2({_f(off_x)}, {_f(off_y)})")
            lines.append("")

    if library_name and motion_ids:
        lines.append('[node name="AnimationPlayer" type="AnimationPlayer" parent="."]')
        lines.append("libraries = {")
        lines.append(f'"": ExtResource("2_lib")')
        lines.append("}")
        lines.append(f'autoplay = "{_q(motion_ids[0])}"')
        lines.append("")
    return "\n".join(lines)


def build_readme(*, folder: str, stem: str, skeleton: Dict[str, Any],
                 binding: Dict[str, Any], actions: Dict[str, Dict[str, Any]],
                 parts_bytes: int, scene_bytes: int) -> str:
    anim_list = "、".join(actions) or "（只有骨架，没有动作）"
    play_sample = list(actions)[0] if actions else "walk"
    return f"""2D 游戏素材工具 · Godot 4 · 骨骼（cutout）角色
生成时间：{time.strftime('%Y-%m-%d %H:%M')}

骨架：{skeleton.get('name') or stem}    {len(skeleton['bones'])} 根骨
部件：{len(binding['slots'])} 个
动作：{anim_list}
包体：部件图 {parts_bytes / 1024:.0f} KB + 场景与动画 {scene_bytes / 1024:.0f} KB

一、放进 Godot 工程（路径必须一致）
  把本包里的「{folder.split('/')[-1]}」整个文件夹拷到工程根目录（和 project.godot 同级），
  让路径恰好是：
      res://{folder}/{stem}.tscn
      res://{folder}/anim.tres
      res://{folder}/actions/*.tres
      res://{folder}/parts/*.png
  路径写死在资源里 —— 放到别处会报「资源缺失」（不是文件坏了）。

二、使用
  A. 代码实例化：
         var rig = preload("res://{folder}/{stem}.tscn").instantiate()
         add_child(rig)
         rig.position = Vector2(400, 360)      # 脚底落点（根节点原点就是脚底）
         rig.get_node("AnimationPlayer").play("{play_sample}")
  B. 拖进场景树当子节点用；动作在 AnimationPlayer 面板里能直接看到。

  切换动作：play("walk") / play("idle")。**换方向不靠 play**，
  侧面看的话整个 rig 的 scale.x = -1 或换一套拆分（见第四节）。

三、已经替你设好的
  · 根节点 = 脚底中心：set_position(地面线) 就站得住，不用算半张图。
  · texture_filter = 1（最近邻）：像素素材不糊边，不用改项目设置。
  · 每个部件绕自己的关节转（rotation 的枢轴由 Sprite2D.offset 表达）。
  · 循环动作 loop_mode = 1，播完自动接上；末帧复制了首帧，接缝不跳。
  · z_index 按「正面」排好：腿在躯干后、臂在躯干前、头最前。

四、想改什么
  · 动作太多/太少：回工具里重新生成（walk / idle），或直接改
    rig/actions/*.json 后重新导出 —— 那是纯文本，格式见 README 末尾。
  · 加新动作：在 rig/actions/ 下放一份同格式 JSON（或用工具生成），
    重新导出即可，**不需要重新渲染任何图**。
  · 换朝向：改 binding 里各 slot 的 z 值（front/back/side 三套写在
    rigbuild.Z_ORDER 里），或直接在 Godot 里调 z_index。

五、常见问题
  · 播放空白：多是路径不对，看输出面板缺哪个文件。
  · 部件位置错乱：`parts/*.png` 被替换成了别的尺寸。枢轴是按图高算的
    offset，换图必须重新建骨。
  · 角色陷进地里：根节点在脚底，别再把 Sprite 的 offset 改回来。
  · 转动时关节露缝：几何切件的固有问题（相邻部件没有重叠圆弧）。
    要修得在部件图上补一圈关节色块，或改用真拆件产物。

六、数据格式（rig/ 下面，可手改）
  skeleton.json   {{schemaVersion, kind:"skeleton", bones:[{{id, parentId, rest:{{x,y,rotation}}}}]}}
  binding.json    {{kind:"character-binding", slots:[{{id, boneId, texture, pivot, z}}]}}
  actions/*.json  {{kind:"motion-clip", fps, loop, duration,
                    tracks:[{{targetId, property, interpolation, keyframes:[{{time, value}}]}}]}}
  property ∈ position(像素) / rotation(弧度, 顺时针为正) / scale / visible
"""


# ---------------------------------------------------------------- 自检


def verify_bundle(root: Path, folder: str, *, expect_actions: int,
                  expect_bones: int) -> Dict[str, Any]:
    """交付前自检：路径存在、动画数与骨架节点数对得上。

    这三条各自对应一类真实事故：
      · 路径差一层 → Godot 打开一片空白；
      · 动画数不符 → `play("walk")` 报「找不到动画」；
      · 轨道指向不存在的节点 → **静默失效**（Godot 不报错，播放时就是不动）。
    """
    missing: List[str] = []
    refs = 0
    texts = sorted(root.rglob("*.tres")) + sorted(root.rglob("*.tscn"))
    for f in texts:
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

    # 动画文件数 vs 期望
    anim_files = sorted((root / "actions").glob("*.tres")) if (root / "actions").is_dir() else []
    anim_ok = len(anim_files) == expect_actions

    # 轨道目标节点必须存在：从 .tscn 里还原**完整节点路径**
    # （只收集名字是不够的——轨道路径是 `torso/arm_l` 这种，拿名字比会全部对不上）
    scene = next(iter(sorted(root.glob("*.tscn"))), None)
    node_paths: set = set()
    bad_targets: List[str] = []
    if scene is not None:
        body = scene.read_text(encoding="utf-8")
        for m in re.finditer(r'\[node name="([^"]+)"(?: type="[^"]*")?'
                             r'(?: parent="([^"]+)")?', body):
            name, parent = m.group(1), m.group(2)
            if not parent:
                node_paths.add(name)          # 场景根
            elif parent == ".":
                node_paths.add(name)
            else:
                node_paths.add(f"{parent}/{name}")
        for anim in anim_files:
            ab = anim.read_text(encoding="utf-8")
            for m in re.finditer(r'path = NodePath\("([^"]+):([a-z_]+)"\)', ab):
                if m.group(1) not in node_paths:
                    bad_targets.append(f"{anim.name} → {m.group(1)}")
    bones_ok = len(node_paths) >= expect_bones

    return {
        "pathsMatch": not missing,
        "missing": missing[:8],
        "refsChecked": refs,
        "actionsMatch": anim_ok,
        "actionsFound": len(anim_files),
        "trackTargetsValid": not bad_targets,
        "badTargets": bad_targets[:8],
        "bonesPresent": bones_ok,
    }


# ---------------------------------------------------------------- 统计


def _dir_bytes(paths: Iterable[Path]) -> int:
    total = 0
    for p in paths:
        if p.is_file():
            total += p.stat().st_size
    return total


# ---------------------------------------------------------------- 主流程


def export_bundle(rig_dir: Path, out_root: Path, *, stem: str = "rig",
                  folder: str = "rig_out", texture_filter: str = "nearest",
                  actions: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """rig 目录 → Godot 工程文件夹 + zip。

    `actions` 不传就导出全部；传了按名字过滤（大小写不敏感）。
    """
    rig = load_rig(rig_dir)
    skeleton, binding = rig["skeleton"], rig["binding"]
    all_actions: Dict[str, Dict[str, Any]] = rig["actions"]

    want = [safe_stem(a) for a in actions] if actions else list(all_actions)
    chosen = {k: all_actions[k] for k in want if k in all_actions}
    if actions and len(chosen) != len(want):
        missing = [a for a in want if a not in all_actions]
        raise GodotRigError(
            f"没有这些动作：{'、'.join(missing)}（现有：{'、'.join(all_actions) or '无'}）")

    stem_ = safe_stem(stem, "rig")
    folder_ = safe_folder(folder, "rig_out")
    filt = texture_filter if texture_filter in _FILTER_VALUE else "nearest"

    out_dir = Path(out_root) / stem_
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    (out_dir / "actions").mkdir(parents=True, exist_ok=True)
    (out_dir / "rig").mkdir(parents=True, exist_ok=True)

    # 1) 部件图：拷过去，顺手量尺寸（offset 按图高算）
    textures: Dict[str, Tuple[int, int]] = {}
    src_dir = Path(rig_dir)
    for s in binding["slots"]:
        src = src_dir / str(s["texture"]).replace("\\", "/")
        if not src.is_file():
            raise GodotRigError(f"找不到部件图：{s['texture']}（在 {src_dir} 下）")
        dst = out_dir / "parts" / f"{s['id']}.png"
        shutil.copyfile(src, dst)
        with Image.open(dst) as im:
            textures[s["id"]] = im.size

    # 2) 动作 → Animation 资源
    for name, motion in chosen.items():
        (out_dir / "actions" / f"{name}.tres").write_text(
            build_animation(skeleton, motion), encoding="utf-8")

    # 3) AnimationLibrary
    lib_name = "anim" if chosen else None
    if chosen:
        (out_dir / f"{lib_name}.tres").write_text(
            build_library(folder_, list(chosen)), encoding="utf-8")

    # 4) 场景
    (out_dir / f"{stem_}.tscn").write_text(
        build_scene(folder=folder_, stem=stem_, skeleton=skeleton, binding=binding,
                    textures=textures, library_name=lib_name,
                    motion_ids=list(chosen), texture_filter=filt),
        encoding="utf-8")

    # 5) 原始 rig 数据：用户可以改了再导，或给别的工具用
    for f in ("skeleton.json", "binding.json"):
        shutil.copyfile(src_dir / f, out_dir / "rig" / f)
    copied_actions = 0
    for name in chosen:
        src = src_dir / "actions" / f"{name}.json"
        if src.is_file():
            (out_dir / "rig" / "actions").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out_dir / "rig" / "actions" / f"{name}.json")
            copied_actions += 1
    if not copied_actions:
        # 没有 JSON 就不留空目录（解压后空文件夹看着像出错）
        shutil.rmtree(out_dir / "rig" / "actions", ignore_errors=True)

    scene_bytes = _dir_bytes(list(out_dir.rglob("*.tres")) + list(out_dir.glob("*.tscn")))
    parts_bytes = _dir_bytes((out_dir / "parts").glob("*.png"))
    (out_dir / "README.txt").write_text(
        build_readme(folder=folder_, stem=stem_, skeleton=skeleton, binding=binding,
                     actions=chosen, parts_bytes=parts_bytes, scene_bytes=scene_bytes),
        encoding="utf-8")

    checks = verify_bundle(out_dir, folder_, expect_actions=len(chosen),
                          expect_bones=len(skeleton["bones"]))

    zip_path = Path(out_root) / f"{stem_}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                z.writestr(f"{folder_}/{f.relative_to(out_dir).as_posix()}",
                           f.read_bytes())

    warnings: List[str] = []
    if not checks["pathsMatch"]:
        warnings.append("包内路径自检未通过：" + "；".join(checks["missing"]))
    if not checks["actionsMatch"]:
        warnings.append(f"动画资源数不符（{checks['actionsFound']} / {len(chosen)}）")
    if not checks["trackTargetsValid"]:
        warnings.append("有轨道指向不存在的节点（Godot 会静默失效）："
                        + "；".join(checks["badTargets"]))
    if not chosen:
        warnings.append("没有动作：导出的只是静态姿势，play() 会失败")
    # 逐帧方案下 8 帧就得 8 张图；骨骼方案只是多几十个 float。这条提示让用户
    # 一眼看到「换动作几乎不要钱」。
    sampled = max((len(t["keyframes"]) for m in chosen.values()
                   for t in m["tracks"]), default=0)
    if chosen and sampled:
        warnings.append(
            f"骨骼动画按 {sampled} 个采样点写入，动画资源共 {scene_bytes / 1024:.1f} KB"
            f"；逐帧方案要 {sampled} 张 PNG")

    return {
        "ok": True,
        "stem": stem_,
        "folder": folder_,
        "bones": len(skeleton["bones"]),
        "slots": len(binding["slots"]),
        "actions": list(chosen),
        "dir": str(out_dir),
        "files": [
            {"name": f.name, "bytes": f.stat().st_size}
            for f in sorted(out_dir.glob("*.tres")) + sorted(out_dir.glob("*.tscn"))
            + [out_dir / "README.txt"]
        ],
        "zip": {"name": zip_path.name, "bytes": zip_path.stat().st_size},
        "resPaths": {
            "scene": f"res://{folder_}/{stem_}.tscn",
            "library": f"res://{folder_}/anim.tres" if chosen else "",
        },
        "bytes": {"parts": parts_bytes, "scene": scene_bytes,
                  "total": _dir_bytes([p for p in out_dir.rglob("*") if p.is_file()])},
        "checks": checks,
        "warnings": warnings,
    }
