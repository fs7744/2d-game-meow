"""Blender 无头后处理：AI 生成的 GLB → 游戏可用的 GLB（可选自动绑骨）。

    blender --background --python tools/blender_postprocess.py -- \
        --in model.glb --out model_game.glb --ratio 0.15 --height 1.8 --rig

为什么需要这一步（AI 生成 3D 的两个通病）：

1. **面数爆炸**。TRELLIS.2 默认输出几十万到上百万面，进引擎加载慢、手机上跑不动。
   Blender 的 Collapse Decimate 能在几乎不改变轮廓的前提下砍掉 85~95%。
2. **没有骨架**。生成的网格只有顶点，没有骨骼和权重。角色要动就得绑骨 ——
   本脚本用 Rigify 的人形元骨架 + 自动权重（`ARMATURE_AUTO`），全程无人工。

**顺序不能反**：先减面 → 再展 UV → 再绑骨。
  · 先绑骨后减面：Decimate 会重建拓扑，权重全丢；
  · 先展 UV 后减面：减面后 UV 会扭曲（Blender 会保留 UV 但接缝会破）；
  · `ARMATURE_AUTO` 用的是"热扩散"权重，网格越干净越准，减面后反而更好算。

**自动绑骨的前提**（不满足就只能出静态模型，脚本会明确报错而不是静默跳过）：
  · 人形（有头/躯干/四肢的清晰分界）；
  · **T-pose 或 A-pose**（四肢张开）。自然站姿的手臂贴着身体，热扩散会把手臂
    权重算到躯干上，一动就撕开；
  · 模型是"一个人"。多个物体/多个人会绑到同一套骨架上。

道具、场景、硬表面**不需要骨骼** —— 别加 `--rig`。

脚本所有 `bpy.ops` 都记录了前置条件（选中/激活/模式），因为 Blender 的算子
在上下文不对时会抛 `RuntimeError: context is incorrect`，而这些错误在无头模式
下没有 UI 提示，非常难查。
"""
from __future__ import annotations

import argparse
import os
import sys

import bpy


# ---------------------------------------------------------------- 工具


def log(msg: str) -> None:
    """无头模式下 print 就是唯一的反馈通道，统一加前缀方便过滤。"""
    print(f"[post] {msg}", flush=True)


def fail(msg: str) -> None:
    log(f"失败：{msg}")
    sys.exit(2)


def clear_scene() -> None:
    """清空默认场景。**必须 Objects 模式**，否则 `delete` 算子不生效。"""
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures,
                  bpy.data.images, bpy.data.actions):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def tri_count(objs) -> int:
    """三角面数（Blender 的 polygons 是 n-gon，要按三角形折算）。"""
    total = 0
    for o in objs:
        me = o.data
        me.calc_loop_triangles()
        total += len(me.loop_triangles)
    return total


def world_bounds(objs):
    """所有网格的世界空间包围盒（min / max），**逐个顶点算**。

    不用 `o.bound_box`：它是缓存的轴对齐盒，`transform_apply()` 之后不会立刻更新，
    读到的是旧值。踩过 —— 目标身高 1.8，实际出来 1.387，就是拿旧包围盒又缩了一次。
    顶点数上千时也不慢（7k 顶点约 10ms）。
    """
    import mathutils

    lo = mathutils.Vector((1e30, 1e30, 1e30))
    hi = mathutils.Vector((-1e30, -1e30, -1e30))
    for o in objs:
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co
            for i in range(3):
                lo[i] = min(lo[i], p[i])
                hi[i] = max(hi[i], p[i])
    return lo, hi


# ---------------------------------------------------------------- 各阶段


def import_glb(path: str):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    added = [o for o in bpy.context.scene.objects if o not in before]
    meshes = [o for o in added if o.type == "MESH"]
    if not meshes:
        fail(f"{os.path.basename(path)} 里没有网格（导入的物体类型：{[o.type for o in added]}）")
    log(f"导入 {len(meshes)} 个网格、{tri_count(meshes)} 面")
    return meshes


def cleanup(meshes, weld: bool = False) -> None:
    """删游离面 / 重算法线。**默认不焊点**（weld=False），原因见下。

    `remove_doubles`（merge by distance）在带 UV 的 AI 网格上是**有害的**：
    glTF 导出时顶点会按 UV 缝和法线拆分，一个位置本来就有多个顶点。把它们焊
    回去会制造大量非流形边 —— 实测 TRELLIS 产物焊完有 11.6 万条边被 3 个面共用、
    10.9 万条被 4 个面共用，随后 Decimate 只能干到 4% → 40%（减不动）。

    只有在确认输入是「同一顶点被拆成几十份的 vertex soup」时才值得开 `--weld`。
    """
    for o in meshes:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if weld:
            # 阈值 0.0001（相对 1.8m 的人约 0.1mm）：太大会把小细节焊掉
            bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.delete_loose()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
    log(f"清理完成{'（含焊点）' if weld else ''}，{tri_count(meshes)} 面")


def normalize(meshes, height: float) -> None:
    """缩放到目标身高、水平居中、底面落在 z=0。

    **必须走「空父级统一缩放」，不能逐顶点改坐标。** 这是踩出来的：

    · 逐物体设 `scale`：物体之间的**相对位置不会跟着缩**，包围盒不是等比变小
      （实测目标 1.8，出来 1.271）。
    · 逐顶点改 `v.co`：对**带骨骼**的模型是灾难 —— 顶点动了、骨架没动，
      蒙皮关系当场错位，Godot 里只剩一条腿在画面里。

    空父级方案把 mesh 和 armature 一起缩放，蒙皮关系天然保持。

    Blender 是 **Z 向上**，glTF 是 **Y 向上**；导入自动转 Z-up，导出 `export_yup`
    转回去。所以这里的"身高"是 z 方向。
    """
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    allobjs = list(meshes) + arms

    # **按「主体网格」算身高，不是所有网格的并集。**
    # AI 产物常带环境球 / 特效球 / 背景板，会撑大包围盒：实测一个 2m 直径的
    # Icosphere 把包围盒从 1.832 撑到 2.832，按并集归一化后角色只剩 1.27 米。
    # 主体取顶点数最多的 —— 角色本体总是最密。
    subject = max(meshes, key=lambda o: len(o.data.vertices))
    lo, hi = world_bounds([subject])
    size = hi - lo
    if max(size.x, size.y, size.z) <= 1e-9:
        fail("包围盒是 0：网格可能是空的")
    all_lo, all_hi = world_bounds(meshes)
    if (all_hi - all_lo).z > size.z * 1.15:
        log(f"  注意：场景里有比主体更大的物体（主体 z 范围 {size.z:.2f}，"
            f"全部 {(all_hi - all_lo).z:.2f}）—— 已按主体归一化，可能是特效/环境物")

    # 若 z 不是最大维度，多半是模型躺倒了（朝向没转过来）。报出来让人判断，
    # 不默默按 max 缩放 —— 按 max 会把"武器举高"也算成身高，人物比例就小了。
    if size.z < max(size.x, size.y) * 0.8:
        log(f"⚠ 垂直方向（z={size.z:.3f}）不是最大维度 "
            f"(x={size.x:.3f} y={size.y:.3f})：模型可能躺倒，检查输入朝向")

    scale = height / size.z if size.z > 1e-9 else height / max(size.x, size.y, size.z)
    cx = (lo.x + hi.x) / 2.0
    cy = (lo.y + hi.y) / 2.0
    dz = lo.z
    log(f"原始尺寸 {size.x:.3f} × {size.y:.3f} × {size.z:.3f} → 缩放 {scale:.4f}"
        f"{'（含 ' + str(len(arms)) + ' 个骨架）' if arms else ''}")

    root = bpy.data.objects.new("NormRoot", None)
    bpy.context.scene.collection.objects.link(root)
    for o in allobjs:
        mw = o.matrix_world.copy()
        o.parent = root
        o.matrix_world = mw          # 先保持视觉位置，缩放由父级统一施加
    root.scale = (scale, scale, scale)
    root.location = (-cx * scale, -cy * scale, -dz * scale)
    bpy.context.view_layer.update()

    # 解除父子（保留变换）后 apply，产物节点上不带缩放 —— 引擎里再叠动作才干净
    bpy.ops.object.select_all(action="DESELECT")
    for o in allobjs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.data.objects.remove(root, do_unlink=True)

    lo2, hi2 = world_bounds([subject])
    log(f"归一化后 主体高 {hi2.z - lo2.z:.3f}，底面 z={lo2.z:.4f}，"
        f"水平中心 ({(lo2.x + hi2.x) / 2:.3f}, {(lo2.y + hi2.y) / 2:.3f})")


def decimate(meshes, ratio: float, target_tris: int) -> None:
    """Collapse 减面。`ratio` 是保留比例（0.15 = 留 15%）。"""
    if not (0.01 <= ratio <= 1.0):
        fail(f"ratio 要在 0.01~1 之间（收到 {ratio}）")
    before_all = tri_count(meshes)
    for o in meshes:
        before = tri_count([o])
        mod = o.modifiers.new(name="Decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        # use_collapse_triangulate：让结果一定是三角面，引擎只吃三角
        mod.use_collapse_triangulate = True
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=mod.name)
        log(f"  {o.name}: {before} → {tri_count([o])} 面")
    total = tri_count(meshes)
    log(f"减面完成，共 {total} 面")
    expect = int(before_all * ratio)
    if before_all and total > expect * 1.5:
        # 减不动的典型原因：网格非流形（焊点造成的），Decimate 找不到可塌陷的边
        log(f"  ⚠ 只减到 {total}（按 ratio 应约 {expect}）：网格非流形、减不动。"
            "检查是否传了 --weld，或改用 --ratio 更小的多遍减面")
    if target_tris and total > target_tris:
        log(f"  仍高于目标 {target_tris}，可把 --ratio 再调小")


def unwrap(meshes, angle_limit: float = 66.0, island_margin: float = 0.002) -> None:
    """Smart UV Project。**必须在编辑模式且全选**，否则只展选中面。"""
    for o in meshes:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=angle_limit * 3.14159265 / 180.0,
                                 island_margin=island_margin)
        bpy.ops.object.mode_set(mode="OBJECT")
    log("UV 展开完成")


def weighted_vertices(meshes) -> int:
    """有多少顶点真的拿到了权重（>0）。

    **判"绑骨成功"只能看这个数**。`parent_set` 失败时不报错，而且照样会把
    159 个顶点组建出来 —— 只看 `len(vertex_groups)` 会得出"绑好了"的结论，
    实际每个顶点权重都是 0，glTF 导出器于是根本不写 skin，Godot 里就是个静态网格。
    """
    n = 0
    for o in meshes:
        for v in o.data.vertices:
            if any(g.weight > 0 for g in v.groups):
                n += 1
    return n


def add_human_rig(meshes, mode: str = "auto"):
    """→ (是否绑成功, 骨架对象或 None)。

    `mode`：auto（热扩散，失败自动回退 envelope）/ auto（固定）/ envelope。
    """
    """Rigify 人形元骨架 + 自动权重。

    **为什么用 Rigify 而不是 Mixamo**：Mixamo 是网页服务，要人工上传，
    无法进无人流水线。Rigify 是 Blender 内置插件，`armature_human_metarig_add`
    生成的是 1.8m 标准人形骨架（与 normalize 的目标身高一致）。

    返回是否成功；失败不影响静态导出（调用方决定要不要报错）。
    """
    try:
        bpy.ops.preferences.addon_enable(module="rigify")
    except Exception as exc:                                  # noqa: BLE001
        log(f"Rigify 插件启用失败：{exc}")
        return False, None

    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.armature_human_metarig_add()
    rig = bpy.context.object
    if rig is None or rig.type != "ARMATURE":
        log("元骨架没生成出来")
        return False, None
    rig.name = "Rig"
    log(f"元骨架 {rig.name}，{len(rig.data.bones)} 根骨")

    # 先给骨架生成实际的变形骨（Rigify 的 metarig 是"元"骨架，要 generate）
    bpy.context.view_layer.objects.active = rig
    try:
        bpy.ops.pose.rigify_generate()
        log(f"Rigify 生成后 {len(rig.data.bones)} 根骨")
    except Exception as exc:                                  # noqa: BLE001
        # 生成失败也能用 metarig 自己绑（骨数量少，但权重是能算的）
        log(f"Rigify generate 跳过（{exc}），改用元骨架直接绑")

    total = sum(len(o.data.vertices) for o in meshes)

    def bind(kind: str) -> int:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        try:
            bpy.ops.object.parent_set(type=kind)
        except Exception as exc:                              # noqa: BLE001
            log(f"{kind} 失败：{exc}")
            return 0
        return weighted_vertices(meshes)

    got = bind("ARMATURE_AUTO") if mode == "auto" else 0
    if got < total * 0.5:
        # **热扩散在 AI 生成的网格上基本必然失败**：TRELLIS 的产物是一堆
        # 互相穿插的独立壳（裙摆、纱袖、头发各是独立面片），热扩散要求闭合
        # 流形体积，于是它给每个顶点算出的权重都是 0 —— 日志只有一句
        # "Bone Heat Weighting: failed to find solution for one or more bones"，
        # 而 parent_set 依然把 159 个顶点组建好了，看起来像成功。
        #
        # envelope 是纯几何判定（顶点到骨的包络体距离），不要求流形，质量粗但可用。
        log(f"热扩散只覆盖 {got}/{total} 个顶点 → 回退 envelope 权重")
        got = bind("ARMATURE_ENVELOPE")

    if got < total * 0.5:
        log(f"绑骨失败：{got}/{total} 个顶点有权重（模型可能不是 T-pose 人形）")
        return False, None

    log(f"绑定完成：{got}/{total} 个顶点有权重，"
        f"{sum(len(o.vertex_groups) for o in meshes)} 个顶点组，"
        f"骨架 {len(rig.data.bones)} 根骨")
    return True, rig


def export_glb(path: str, meshes, *, keep_anim: bool = True,
               armatures=()) -> None:
    """导出。**只导出选中的物体**，避免把多余的空物体/相机带出去。

    `export_yup=True` 把 Blender 的 Z-up 转成 glTF 的 Y-up（Godot 用 Y-up）。
    `export_apply=False`：modifier 已经在前面 apply 过，再 apply 一次会重复。
    """
    # **只选明确要导的物体**。不能写「选中场景里所有 MESH/ARMATURE」——
    # Rigify 会生成一堆 WGT-* 控件网格，它们不在 View Layer 里，select_set 直接抛
    # "Object 'WGT-...' can't be selected because it is not in View Layer"，
    # 导出中断（绑骨本身已经成功，只是出不了文件）。
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(meshes) + list(armatures):
        try:
            o.select_set(True)
        except RuntimeError:
            pass
    bpy.context.view_layer.objects.active = meshes[0]
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=False,
        export_skins=True,
        # **默认保留动画**：AI 生成的模型通常没有动画（所以没影响），
        # 但如果输入本身带动画（比如先天的走路循环），关掉就是静默丢数据。
        export_animations=keep_anim,
        export_materials="EXPORT",
        export_image_format="AUTO",    # 保留原贴图格式（webp 就还是 webp）
    )
    log(f"已导出 {path}（{os.path.getsize(path) / 1024:.0f} KB）")


# ---------------------------------------------------------------- 主流程


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description="AI 3D 资产后处理")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--ratio", type=float, default=0.15, help="减面保留比例")
    ap.add_argument("--target-tris", type=int, default=0, help="只用于日志提示")
    ap.add_argument("--height", type=float, default=1.8, help="归一化身高（米）")
    ap.add_argument("--no-decimate", action="store_true")
    ap.add_argument("--no-uv", action="store_true")
    ap.add_argument("--rig", action="store_true", help="自动绑骨（仅人形 T/A-pose）")
    ap.add_argument("--rig-mode", choices=("auto", "envelope"), default="auto",
                    help="auto=热扩散+失败回退 envelope；envelope=直接用包络权重")
    ap.add_argument("--skip-cleanup", action="store_true")
    ap.add_argument("--weld", action="store_true",
                    help="清理时做 merge by distance（仅当输入是 vertex soup 时用；"
                         "带 UV 的网格焊点会挡住减面）")
    ap.add_argument("--no-anim", action="store_true", help="不导出动画（省体积）")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.src):
        fail(f"找不到输入：{args.src}")

    clear_scene()
    meshes = import_glb(args.src)

    normalize(meshes, args.height)
    # **减面必须在 cleanup 之前**。焊点会制造非流形边，Decimate 之后完全减不动
    # （实测 4% 的请求只能做到 40%）。先减面、再清理，两个目的都能达到。
    if not args.no_decimate:
        decimate(meshes, args.ratio, args.target_tris)
    if not args.skip_cleanup:
        cleanup(meshes, weld=args.weld)
    if not args.no_uv:
        unwrap(meshes)

    rigged = False
    rig = None
    if args.rig:
        rigged, rig = add_human_rig(meshes, mode=args.rig_mode)

    export_glb(args.dst, meshes, keep_anim=not args.no_anim,
               armatures=[rig] if rig else [])
    log(f"完成：{'已绑骨' if rigged else '未绑骨'}，{tri_count(meshes)} 面")
    # 明确告诉调用方结果，别让"以为绑了其实没绑"流到下游
    log(f"RESULT rigged={str(rigged).lower()} tris={tri_count(meshes)}")


if __name__ == "__main__":
    main()
