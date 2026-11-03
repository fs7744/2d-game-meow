"""造一个无骨骼的 T-pose 人形 GLB，用来验证「自动绑骨」。

    blender --background --python tools/make_test_human.py -- --out D:/tmp/tpose.glb

**为什么要有这个夹具**：自动绑骨（`ARMATURE_AUTO` 热扩散权重）在"手臂贴着身体"
或者"非人形"的网格上会**静默失败**——不报错，但顶点组是空的。要验这条路径，
必须有一个形状正确、姿势正确的输入。

尺寸**照着 Rigify 的 human metarig 比例**造（metarig 是 1.8m 标准人形）：
骨骼位置是固定的，模型比例不对的话权重会算歪，那验的就不是绑骨而是比例。

    脚 z=0.00   膝 z=0.46   髋 z=0.95   腰 z=1.05
    胸 z=1.25   颈 z=1.50   头 z=1.60~1.80
    肩 z≈1.45, x=±0.19      手 x=±0.75（T-pose 水平伸直）

用最简几何（球 + 圆柱）就够：热扩散只看"哪里是实体、哪里是空隙"，
不需要真实的人体细节——细节反而让验证结果难以解释。
"""
from __future__ import annotations

import argparse
import math
import sys

import bpy


def clear() -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def add_capsule(name: str, p0, p1, radius: float) -> None:
    """在两点之间放一个圆柱（当肢体用）。"""
    import mathutils

    a, b = mathutils.Vector(p0), mathutils.Vector(p1)
    mid = (a + b) / 2
    vec = b - a
    length = vec.length
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=length,
                                        location=mid)
    obj = bpy.context.object
    obj.name = name
    # 让圆柱的 +Z 轴对准 a→b
    quat = mathutils.Vector((0, 0, 1)).rotation_difference(vec)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = quat


def add_sphere(name: str, center, radius: float, scale=(1, 1, 1)) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12,
                                         radius=radius, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale


def add_box(name: str, center, half) -> None:
    bpy.ops.mesh.primitive_cube_add(size=2, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = half


def main() -> None:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    clear()

    # 头（椭球，稍扁）
    add_sphere("head", (0, 0, 1.69), 0.105, scale=(1.0, 0.88, 1.15))
    # 颈
    add_capsule("neck", (0, 0, 1.50), (0, 0, 1.60), 0.045)
    # 躯干：胸腔 + 腹
    add_box("chest", (0, 0, 1.33), (0.175, 0.10, 0.13))
    add_box("belly", (0, 0, 1.10), (0.15, 0.095, 0.11))
    # 骨盆
    add_box("pelvis", (0, 0, 0.95), (0.16, 0.10, 0.08))

    for side, sx in (("l", 1), ("r", -1)):
        # 上臂 / 前臂：T-pose 水平伸直
        add_capsule(f"upperarm_{side}", (sx * 0.19, 0, 1.45), (sx * 0.47, 0, 1.45), 0.055)
        add_capsule(f"forearm_{side}", (sx * 0.47, 0, 1.45), (sx * 0.72, 0, 1.45), 0.045)
        add_sphere(f"hand_{side}", (sx * 0.78, 0, 1.45), 0.055, scale=(1.2, 0.7, 1.0))
        # 大腿 / 小腿
        add_capsule(f"thigh_{side}", (sx * 0.10, 0, 0.92), (sx * 0.11, 0, 0.48), 0.075)
        add_capsule(f"shin_{side}", (sx * 0.11, 0, 0.48), (sx * 0.11, 0, 0.06), 0.06)
        # 脚
        add_box(f"foot_{side}", (sx * 0.11, 0.05, 0.035), (0.055, 0.11, 0.035))

    # 合并成一个网格（绑骨要单一实体，多个物体各自绑太脆）
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    body = bpy.context.object
    body.name = "TPoseBody"

    # apply 变换，让顶点坐标就是世界坐标（导出/绑骨都更干净）
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    import mathutils

    lo = mathutils.Vector((1e30, 1e30, 1e30))
    hi = mathutils.Vector((-1e30, -1e30, -1e30))
    for v in body.data.vertices:
        p = body.matrix_world @ v.co
        for i in range(3):
            lo[i] = min(lo[i], p[i])
            hi[i] = max(hi[i], p[i])
    print(f"[make] 顶点 {len(body.data.vertices)}，"
          f"尺寸 x={hi.x - lo.x:.3f} y={hi.y - lo.y:.3f} z={hi.z - lo.z:.3f}")

    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB",
                              use_selection=False, export_yup=True,
                              export_apply=True, export_skins=False,
                              export_animations=False)
    print(f"[make] 已写 {args.out}（无骨骼，供绑骨测试用）")


if __name__ == "__main__":
    main()
