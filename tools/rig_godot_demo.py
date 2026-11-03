"""一张角色图 → 2D cutout 骨架 → Godot 4 工程（命令行一键跑）。

    python tools/rig_godot_demo.py 角色图.png
    python tools/rig_godot_demo.py 角色图.png --out assets/projects/demo/rig --stem hero
    python tools/rig_godot_demo.py 角色图.png --verify     # 顺带用 Godot 真机验一遍
    python tools/rig_godot_demo.py 角色图.png --shots      # 真渲染几帧导出 PNG

**这个脚本本身也是链路的一部分**：它证明「切件 → 建骨 → 动作 → Godot 能播」
这四段能在一条命令里跑通，而且 `--verify` 是拿真 Godot 验的（不是文本自检）。

关于 `--verify` 为什么必须存在：`godotrig.verify_bundle` 只能查「路径/数量」，
查不出两类致命错误 —— 轨道路径解析不了（Godot 只警告、播放时静默不动）和
关键帧写成裸整数（float 属性收不到）。这两个都是真机跑出来的，不是想出来的。

产物：
    <out>/rig/{skeleton.json, binding.json, actions/*.json, parts/*.png}
    <out>/exports/<stem>/{<stem>.tscn, anim.tres, actions/, parts/, README.txt}
    <out>/exports/<stem>.zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image                                    # noqa: E402

from workflows import godotrig, paths, rigbuild          # noqa: E402

#: 本机装的 Godot（`--verify` 找不到会自动搜索这几个位置）
GODOT_CANDIDATES = [
    r"D:\GAMES_tool\Godot_v4.7.2-stable_mono_win64\Godot_v4.7.2-stable_mono_win64_console.exe",
    r"D:\GAMES_tool\Godot_v4.8-beta1_mono_win64\Godot_v4.8-beta1_mono_win64_console.exe",
]


def find_godot(explicit: str = "") -> str:
    if explicit:
        return explicit
    for p in GODOT_CANDIDATES:
        if Path(p).is_file():
            return p
    which = shutil.which("godot") or shutil.which("godot4")
    if which:
        return which
    raise SystemExit("找不到 Godot。用 --godot 指定可执行文件路径（要 console 版）")


#: 真机验证脚本：加载场景 → 检查动画/节点 → 沿时间轴采样骨骼
PROBE_GD = """extends SceneTree

func _init() -> void:
	var packed: PackedScene = load("res://{folder}/{stem}.tscn")
	if packed == null:
		print("RIG_FAIL scene_null")
		quit(1)
		return
	var rig: Node2D = packed.instantiate()
	root.add_child(rig)
	await process_frame
	var player: AnimationPlayer = rig.get_node_or_null("AnimationPlayer")
	if player == null:
		print("RIG_FAIL no_animation_player")
		quit(1)
		return
	print("RIG_ANIMS ", player.get_animation_list())
	var rig_paths := {{}}
	for b in {bones!r}:
		rig_paths[b] = b
	var nodes: Array[Node] = rig.find_children("*", "Node2D", true, false)
	print("RIG_NODES ", nodes.size())
	# 逐条动画：播放并采样，确认 rotation / position 真的在变
	for an in player.get_animation_list():
		player.play(an)
		var anim: Animation = player.get_animation(an)
		var moved := 0
		var samples := []
		for i in 5:
			var t: float = anim.length * float(i) / 4.0
			player.seek(t, true)
			var snap := []
			for n in nodes:
				if n.name == rig.name:
					continue
				snap.append(snappedf(n.rotation, 0.0001))
				snap.append(snappedf(n.position.y, 0.01))
			samples.append(snap)
		for i in range(1, samples.size()):
			if samples[i] != samples[0]:
				moved += 1
		print("RIG_ANIM %s len=%.3f tracks=%d moved_steps=%d" % [an, anim.length, anim.get_track_count(), moved])
	quit(0)
"""

SHOT_GD = """extends SceneTree

func _init() -> void:
	RenderingServer.set_default_clear_color(Color(0.09, 0.10, 0.12))
	root.size = Vector2i({w}, {h})
	var rig: Node2D = load("res://{folder}/{stem}.tscn").instantiate()
	root.add_child(rig)
	rig.position = Vector2({cx}, {cy})
	var player: AnimationPlayer = rig.get_node("AnimationPlayer")
	var jobs := {jobs}
	var i := 0
	for j in jobs:
		player.play(j[0])
		player.seek(j[1], true)
		for k in 4:
			await process_frame
		var img: Image = root.get_texture().get_image()
		img.save_png("{out}/frame_{idx}_%s_%s.png" % [j[0], str(j[1]).replace(".", "_")])
		print("RIG_SHOT ", j[0], " ", j[1])
		i += 1
	quit(0)
"""


def make_project(proj_dir: Path, folder: str, stem: str, bones: list,
                 zip_path: Path, godot: str) -> None:
    """建一个最小 Godot 工程并把导出包解到根目录（res:// 路径才成立）。"""
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "project.godot").write_text(
        'config_version=5\n\n'
        '[application]\n'
        'config/name="RigCheck"\n'
        'config/features=PackedStringArray("4.7", "GL Compatibility")\n\n'
        '[rendering]\n'
        'renderer/rendering_method="gl_compatibility"\n'
        'textures/canvas_textures/default_texture_filter=0\n',
        encoding="utf-8")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(proj_dir)
    (proj_dir / "probe.gd").write_text(
        PROBE_GD.format(folder=folder, stem=stem, bones=list(bones)), encoding="utf-8")
    # PNG 要先导入才能 load()：headless --import 会生成 .import
    subprocess.run([godot, "--headless", "--path", str(proj_dir), "--import"],
                   capture_output=True, text=True, timeout=180)


def run_godot(godot: str, proj_dir: Path, script: str = "res://probe.gd") -> str:
    r = subprocess.run([godot, "--headless", "--path", str(proj_dir), "--script", script],
                       capture_output=True, text=True, timeout=180)
    return (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser(description="角色图 → 骨骼 → Godot 工程")
    ap.add_argument("image", help="角色图（透明底或纯色底，单主体全身）")
    ap.add_argument("--out", default=str(ROOT / "tests" / "_out" / "rigdemo"),
                    help="产物目录（默认 tests/_out/rigdemo）")
    ap.add_argument("--stem", default="hero", help="文件名主干（ASCII）")
    ap.add_argument("--folder", default="", help="res:// 下的目录名（默认 <stem>_rig）")
    ap.add_argument("--skeleton-id", default="rig")
    ap.add_argument("--actions", default="walk,idle", help="要生成的动作，逗号分隔")
    ap.add_argument("--frames", type=int, default=8, help="相位密度（一圈几个姿态）")
    ap.add_argument("--samples", type=int, default=24, help="关键帧采样数")
    ap.add_argument("--fps", type=float, default=8.0, help="相位帧率（决定时长）")
    ap.add_argument("--swing", type=float, default=13.0, help="腿摆幅（度）")
    ap.add_argument("--arm", type=float, default=12.0, help="臂摆幅（度）")
    ap.add_argument("--head-ratio", type=float, default=0.0,
                    help="头顶到颈线的比例（0 = 用默认 0.26；Q 版要调大）")
    ap.add_argument("--godot", default="", help="Godot console 可执行文件")
    ap.add_argument("--verify", action="store_true", help="用真 Godot 验一遍")
    ap.add_argument("--shots", action="store_true", help="真渲染几帧导出 PNG")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.is_file():
        raise SystemExit(f"找不到图片：{src}")
    out = Path(args.out)
    stem = godotrig.safe_stem(args.stem, "hero")
    folder = args.folder or f"{stem}_rig"

    proportions = {}
    if args.head_ratio:
        proportions["headRatio"] = args.head_ratio

    acts = [a.strip() for a in args.actions.split(",") if a.strip()]
    img = Image.open(src).convert("RGBA")
    print(f"源图 {src.name} {img.size}")

    info = rigbuild.build_all(
        img, out, skeleton_id=args.skeleton_id, proportions=proportions or None,
        actions=acts, frames=args.frames, samples=args.samples, fps=args.fps,
        swing=args.swing, arm=args.arm)
    print("部件:", {k: tuple(v) for k, v in info["parts"].items()})
    print("枢轴:", info["joints"])
    if info["emptyParts"]:
        print("⚠ 这些部件切出来是空的（侧面图常见）：", "、".join(info["emptyParts"]))

    res = godotrig.export_bundle(Path(info["dir"]), out / "exports",
                                 stem=stem, folder=folder)
    print(f"导出 {res['folder']}/{stem}.tscn  骨 {res['bones']}  部件 {res['slots']}")
    print("体积:", {k: f"{v / 1024:.0f} KB" for k, v in res["bytes"].items()})
    print("动作:", ", ".join(f"{k}({v['duration']:.2f}s/{v['keys']}键)"
                             for k, v in info["actions"].items()))
    for w in res["warnings"]:
        print("  ·", w)
    bad = [k for k, v in res["checks"].items() if v is False]
    if bad:
        print("✗ 自检未通过:", bad, res["checks"])

    if not (args.verify or args.shots):
        print(f"\n产物：{out}")
        print("加 --verify 用 Godot 真机验（推荐），加 --shots 出几帧截图")
        return 0

    godot = find_godot(args.godot)
    proj = out / "_godot"
    make_project(proj, folder, stem, [b["id"] for b in info["skeleton"]["bones"]],
                 out / "exports" / f"{stem}.zip", godot)
    log = run_godot(godot, proj)
    keep = [ln for ln in log.splitlines()
            if ln.startswith("RIG_") or "couldn't resolve" in ln or "Node not found" in ln]
    print("\n--- Godot 真机 ---")
    print("\n".join(keep) or "（没有任何输出，说明脚本没跑起来）")
    if "RIG_FAIL" in log:
        print("✗ 工程打不开，看上面的输出")
        return 1

    if args.shots:
        w, h = 480, 720
        first = acts[0] if acts else "walk"
        shot_dir = out / "shots"
        shot_dir.mkdir(parents=True, exist_ok=True)
        jobs = [[first, 0.0], [first, 0.25], [first, 0.5], [first, 0.75]]
        (proj / "shot.gd").write_text(SHOT_GD.format(
            w=w, h=h, cx=w // 2, cy=h - 40, folder=folder, stem=stem,
            jobs=json.dumps(jobs), out=shot_dir.resolve().as_posix(), idx=0), encoding="utf-8")
        r = subprocess.run([godot, "--path", str(proj), "--rendering-driver", "opengl3",
                            "--resolution", f"{w}x{h}", "--script", "res://shot.gd"],
                           capture_output=True, text=True, timeout=240)
        shots = sorted(shot_dir.glob("frame_*.png"))
        print(f"截图 {len(shots)} 张 → {shot_dir}")
        if not shots:
            print("  （渲染失败，看 stderr）", (r.stderr or "")[-300:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
