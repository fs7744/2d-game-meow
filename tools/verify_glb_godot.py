"""在真 Godot 里验证一个 GLB：能不能导入、导入后是什么、有没有骨骼/动画。

    python tools/verify_glb_godot.py 模型.glb
    python tools/verify_glb_godot.py 模型.glb --shot        # 顺带渲染一张预览
    python tools/verify_glb_godot.py 模型.glb --godot <path>

**为什么必须真机验**：`.glb` 文本层面"合法"不代表 Godot 能正确用 —— 我们在骨骼
那条线已经吃过一次亏（`AnimationMixer` 对非法 NodePath 只打警告、静默不动）。
GLB 这边对应的静默失败是：

  · 材质里的 `KHR_texture_basisu`（basis 压缩）Godot 读不了，模型变纯白；
  · 贴图是 webp 时，Godot < 4.3 直接丢贴图；
  · 非三角图元（LINES / TRIANGLE_STRIP）被忽略，模型缺块；
  · 骨骼没有动画轨道时，Skeleton3D 存在但 AnimationPlayer 是空的。

这些都只能靠加载后读节点属性发现。

产物是临时工程，默认落在 `tests/_out/glbcheck/<stem>/`，可以留着用 Godot 打开看。
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workflows import glb as glbmod                     # noqa: E402
from workflows import paths                             # noqa: E402

GODOT_CANDIDATES = [
    r"D:\GAMES_tool\Godot_v4.7.2-stable_mono_win64\Godot_v4.7.2-stable_mono_win64_console.exe",
    r"D:\GAMES_tool\Godot_v4.8-beta1_mono_win64\Godot_v4.8-beta1_mono_win64_console.exe",
]

#: 探针：加载 glb 场景，把"Godot 眼里的样子"打出来
PROBE_GD = """extends SceneTree

func _init() -> void:
	var packed: PackedScene = load("res://model.glb")
	if packed == null:
		print("VERIFY_FAIL 场景加载失败（多半是导入失败，看上面 Godot 自己打的错误）")
		quit(1)
		return
	var root_node: Node = packed.instantiate()
	root.add_child(root_node)
	await process_frame

	var meshes: Array[Node] = root_node.find_children("*", "MeshInstance3D", true, false)
	var skels: Array[Node] = root_node.find_children("*", "Skeleton3D", true, false)
	var players: Array[Node] = root_node.find_children("*", "AnimationPlayer", true, false)

	print("VERIFY_MESHES ", meshes.size())
	var total_surf := 0
	var total_verts := 0
	var aabb := AABB()
	var first := true
	var missing_tex := 0
	for m in meshes:
		var mi: MeshInstance3D = m
		if mi.mesh == null:
			print("VERIFY_MESH_NULL ", mi.name)
			continue
		var n: int = mi.mesh.get_surface_count()
		total_surf += n
		var box: AABB = mi.mesh.get_aabb()
		aabb = box if first else aabb.merge(box)
		first = false
		var verts := 0
		var mats := 0
		for s in n:
			var arr: Array = mi.mesh.surface_get_arrays(s)
			var v: PackedVector3Array = arr[Mesh.ARRAY_VERTEX]
			verts += v.size()
			if mi.mesh.surface_get_material(s) != null:
				mats += 1
		total_verts += verts
		print("VERIFY_MESH %s surfaces=%d verts=%d mat=%d aabb=%s" % [
			mi.name, n, verts, mats, box.size])
	print("VERIFY_TOTAL surfaces=%d verts=%d aabb=%s" % [total_surf, total_verts, aabb.size])

	print("VERIFY_SKELETONS ", skels.size())
	for s in skels:
		var sk: Skeleton3D = s
		print("VERIFY_SKELETON %s bones=%d" % [sk.name, sk.get_bone_count()])

	print("VERIFY_PLAYERS ", players.size())
	for p in players:
		var ap: AnimationPlayer = p
		print("VERIFY_ANIMS ", ap.get_animation_list())
		for an in ap.get_animation_list():
			var a: Animation = ap.get_animation(an)
			print("VERIFY_ANIM %s len=%.3f tracks=%d" % [an, a.length, a.get_track_count()])
	quit(0)
"""

SHOT_GD = """extends SceneTree

func _init() -> void:
	RenderingServer.set_default_clear_color(Color(0.10, 0.11, 0.13))
	root.size = Vector2i({w}, {h})
	var scene: PackedScene = load("res://model.glb")
	var model: Node3D = scene.instantiate()
	root.add_child(model)
	# **必须等一帧再取 global_transform**：节点刚 add_child 但还没进树时
	# `get_global_transform()` 会报 "!is_inside_tree()" 并返回 Identity，
	# 由此算出的包围盒是错的 —— 相机会对着空气，截出来只剩背景色。
	await process_frame
	# 相机对准包围盒
	var aabb := AABB()
	var first := true
	for m in model.find_children("*", "MeshInstance3D", true, false):
		var mi: MeshInstance3D = m
		if mi.mesh == null:
			continue
		var box: AABB = mi.global_transform * mi.mesh.get_aabb()
		aabb = box if first else aabb.merge(box)
		first = false
	var center: Vector3 = aabb.get_center()
	var radius: float = max(aabb.size.length() * 0.5, 0.5)
	var cam := Camera3D.new()
	cam.far = radius * 20.0
	# **两个必须**（都踩过，症状都是"截图全空"）：
	#  1. 用 `look_at_from_position` 而不是 `look_at` —— SceneTree 脚本的 _init 阶段
	#     节点还没真正进树，`look_at` 会报 "Node not inside tree" 然后静默不转，
	#     相机朝 -Z，模型在视野外，截出来就是一张纯背景色。
	#  2. `make_current()` —— 手动挂的 Camera3D 不会自动成为当前相机。
	# 距离按 FOV 算，别拍脑袋乘系数：半径 r 的球要占满画面，相机距离 ≈ r / tan(fov/2)。
	# （早先写死 radius*1.4/1.6，角色只占画面 40%，看不清细节。）
	var dist: float = radius / tan(deg_to_rad(cam.fov * 0.5))
	var dir := Vector3(0.75, 0.45, 1.0).normalized()
	cam.look_at_from_position(center + dir * dist * 1.05, center, Vector3.UP)
	root.add_child(cam)
	cam.make_current()
	print("VERIFY_CAM pos=%s center=%s radius=%.2f" % [cam.position, center, radius])
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-40, 35, 0)
	light.light_energy = 1.4
	root.add_child(light)
	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.10, 0.11, 0.13)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.6, 0.62, 0.7)
	e.ambient_light_energy = 0.7
	env.environment = e
	root.add_child(env)
	for i in 6:
		await process_frame
	root.get_texture().get_image().save_png("{out}")
	print("VERIFY_SHOT {out}")
	quit(0)
"""


def find_godot(explicit: str = "") -> str:
    if explicit:
        return explicit
    for p in GODOT_CANDIDATES:
        if Path(p).is_file():
            return p
    which = shutil.which("godot") or shutil.which("godot4")
    if which:
        return which
    raise SystemExit("找不到 Godot，用 --godot 指定（要 console 版才有输出）")


def main() -> int:
    ap = argparse.ArgumentParser(description="在 Godot 里验证 GLB")
    ap.add_argument("glb")
    ap.add_argument("--out", default="", help="临时工程目录（默认 tests/_out/glbcheck/<stem>）")
    ap.add_argument("--godot", default="")
    ap.add_argument("--shot", action="store_true", help="渲染一张预览图")
    ap.add_argument("--normalize", type=float, default=0.0, metavar="HEIGHT",
                    help="先规范化到指定身高（米）再验，例如 1.8")
    args = ap.parse_args()

    src = Path(args.glb)
    if not src.is_file():
        raise SystemExit(f"找不到文件：{src}")

    # 1) 纯 Python 体检先跑一遍（不依赖引擎，秒出）
    info = glbmod.inspect(src)
    report = glbmod.grade(info)
    print(f"== {src.name}  {info['bytes'] / 1024:.0f} KB ==")
    print(f"  {glbmod.summary_line(info)}")
    print(f"  包围盒 {info['aabb']['size'] if info['aabb'] else '未知'}")
    print(f"  级别 {report['level']}")
    for n in report["notes"]:
        print("  ·", n)

    work = Path(args.out) if args.out else (paths.ROOT / "tests" / "_out" / "glbcheck" / src.stem)
    if work.is_dir():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    target = src
    if args.normalize:
        target = work / "model_norm.glb"
        r = glbmod.normalize(src, target, target_height=args.normalize)
        print(f"  规范化：缩放 {r['scale']:.4f}，原尺寸 {r['originalSize']}")

    shutil.copyfile(target, work / "model.glb")
    (work / "project.godot").write_text(
        'config_version=5\n\n[application]\nconfig/name="GlbCheck"\n'
        'config/features=PackedStringArray("4.7", "GL Compatibility")\n\n'
        '[rendering]\nrenderer/rendering_method="gl_compatibility"\n', encoding="utf-8")
    (work / "probe.gd").write_text(PROBE_GD, encoding="utf-8")

    godot = find_godot(args.godot)
    print("\n== Godot 导入 ==")
    imp = subprocess.run([godot, "--headless", "--path", str(work), "--import"],
                         capture_output=True, text=True, timeout=300)
    errs = [ln for ln in ((imp.stdout or "") + (imp.stderr or "")).splitlines()
            if "ERROR" in ln or "error" in ln.lower()]
    if errs:
        print("  导入期报错：")
        for ln in errs[:6]:
            print("   ", ln[:160])

    print("\n== Godot 加载 ==")
    r = subprocess.run([godot, "--headless", "--path", str(work), "--script", "res://probe.gd"],
                       capture_output=True, text=True, timeout=300)
    log = (r.stdout or "") + (r.stderr or "")
    keep = [ln for ln in log.splitlines() if ln.startswith("VERIFY")]
    print("\n".join("  " + ln for ln in keep) or "  （没有输出：脚本没跑起来）")
    problems = [ln for ln in log.splitlines()
                if "couldn't resolve" in ln or "Node not found" in ln]
    for ln in problems[:4]:
        print("  ⚠", ln[:160])

    ok = any("VERIFY_TOTAL" in ln for ln in keep)
    if args.shot and ok:
        shot = work / "preview.png"
        (work / "shot.gd").write_text(
            SHOT_GD.format(w=640, h=640, out=shot.resolve().as_posix()), encoding="utf-8")
        subprocess.run([godot, "--path", str(work), "--rendering-driver", "opengl3",
                        "--resolution", "640x640", "--script", "res://shot.gd"],
                       capture_output=True, text=True, timeout=300)
        print(f"  预览图 {shot}" if shot.is_file() else "  预览渲染失败（看 stderr）")

    print(f"\n结论：{'Godot 能加载' if ok else 'Godot 加载失败'}　工程留在 {work}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
