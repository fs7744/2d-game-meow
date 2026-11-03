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
    """所有网格的世界空间包围盒（min / max / size）。"""
    import mathutils

    lo = mathutils.Vector((1e30, 1e30, 1e30))
    hi = mathutils.Vector((-1e30, -1e30, -1e30))
    for o in objs:
        for corner in o.bound_box:
            p = o.matrix_world @ mathutils.Vector(corner)
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


def cleanup(meshes) -> None:
    """合并重复点 / 删游离面 / 重算法线。

    `merge by distance` 的阈值用 0.0001（相对 1.8m 的人约 0.1mm）：太大会把小
    细节焊掉，太小对 AI 网格没用（它们的重复点通常是完全重合的）。
    """
    for o in meshes:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.delete_loose()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
    log(f"清理完成，{tri_count(meshes)} 面")


def normalize(meshes, height: float) -> None:
    """缩放到目标身高、水平居中、底面落在 y=0。

    Blender 是 **Z 向上**，glTF 是 **Y 向上**；导出时 `+Y up` 会做一次旋转转换。
    这里按 Blender 的约定落地（z=0 是地面），导出后自然就是 y=0 是地面。
    """
    lo, hi = world_bounds(meshes)
    size = hi - lo
    tallest = max(size.z, size.x, size.y)
    if tallest <= 0:
        fail("包围盒是 0：网格可能是空的")
    scale = height / size.z if size.z > 0 else (height / tallest)
    log(f"原始尺寸 {size.x:.3f} × {size.y:.3f} × {size.z:.3f} → 缩放 {scale:.4f}")

    for o in meshes:
        o.scale = [s * scale for s in o.scale]

    # 缩放后重新量（bound_box 是本地空间，matrix_world 还没更新）
    bpy.context.view_layer.update()
    lo, hi = world_bounds(meshes)
    cx = (lo.x + hi.x) / 2.0
    cy = (lo.y + hi.y) / 2.0
    dz = -lo.z
    for o in meshes:
        o.location.x -= cx
        o.location.y -= cy
        o.location.z += dz
    bpy.context.view_layer.update()

    # 应用变换：不 Apply 的话导出后节点上带缩放，引擎里再叠动作容易出问题
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    lo, hi = world_bounds(meshes)
    log(f"归一化后 高 {hi.z - lo.z:.3f}，底面 z={lo.z:.4f}")


def decimate(meshes, ratio: float, target_tris: int) -> None:
    """Collapse 减面。`ratio` 是保留比例（0.15 = 留 15%）。"""
    if not (0.01 <= ratio <= 1.0):
        fail(f"ratio 要在 0.01~1 之间（收到 {ratio}）")
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


def add_human_rig(meshes) -> bool:
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
        return False

    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.armature_human_metarig_add()
    rig = bpy.context.object
    if rig is None or rig.type != "ARMATURE":
        log("元骨架没生成出来")
        return False
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

    # 自动权重：网格与骨架同时选中，**骨架必须是激活对象**
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except Exception as exc:                                  # noqa: BLE001
        log(f"自动权重失败：{exc}")
        return False

    # 检查是否真的产生了顶点组（失败时 parent_set 不报错但没有组）
    groups = sum(len(o.vertex_groups) for o in meshes)
    if groups == 0:
        log("自动权重没有产生任何顶点组 —— 模型多半不是 T-pose 人形")
        return False
    log(f"绑定完成，网格上共 {groups} 个顶点组")
    return True


def export_glb(path: str, meshes) -> None:
    """导出。**只导出选中的物体**，避免把多余的空物体/相机带出去。

    `export_yup=True` 把 Blender 的 Z-up 转成 glTF 的 Y-up（Godot 用 Y-up）。
    `export_apply=False`：modifier 已经在前面 apply 过，再 apply 一次会重复。
    """
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.context.scene.objects:
        if o.type in ("MESH", "ARMATURE"):
            o.select_set(True)
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
        export_animations=False,       # AI 生成的模型没有动画，别导空轨道
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
    ap.add_argument("--skip-cleanup", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.src):
        fail(f"找不到输入：{args.src}")

    clear_scene()
    meshes = import_glb(args.src)

    if not args.skip_cleanup:
        cleanup(meshes)
    normalize(meshes, args.height)
    if not args.no_decimate:
        decimate(meshes, args.ratio, args.target_tris)
    if not args.no_uv:
        unwrap(meshes)

    rigged = False
    if args.rig:
        rigged = add_human_rig(meshes)

    export_glb(args.dst, meshes)
    log(f"完成：{'已绑骨' if rigged else '未绑骨'}，{tri_count(meshes)} 面")
    # 明确告诉调用方结果，别让"以为绑了其实没绑"流到下游
    log(f"RESULT rigged={str(rigged).lower()} tris={tri_count(meshes)}")


if __name__ == "__main__":
    main()
