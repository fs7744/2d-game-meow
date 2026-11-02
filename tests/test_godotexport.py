"""成片集 → Godot 4 工程包（`workflows/godotexport.py`）。

**为什么值得单测**：产物是**给引擎吃的文本资源**，写错一个字节 Godot 就打不开，
而这里的错误在浏览器里完全看不出来（界面只显示"导出成功"）。所以断言落在
"文件内容对不对"上：

- `.tres` 的 `load_steps` 与帧数一致，每帧一个 `ext_resource`；
- `.tscn` 是 AnimatedSprite2D、`autoplay` 指向同一个动画名、最近邻过滤；
- zip 的**条目带 `<folder>/` 前缀** —— 这是"解到工程根目录，res:// 路径恰好成立"
  的唯一保证（pixelforge 那边踩过：路径差一层，Godot 一片空白）；
- 包内自检（`verify_bundle`）必须通过。
"""
from __future__ import annotations

import base64
import io
import zipfile

from PIL import Image

from workflows import godotexport
from workflows import paths
from workflows import poseset

PROJECT = "ptest_godot"


def _png(level: int, side: int = 16) -> str:
    buf = io.BytesIO()
    Image.new("RGBA", (side, side), (level, level, level, 255)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _setup(n: int = 3) -> str:
    """建一个姿势集 + 成片集，每帧出一张候选并选中。返回成片集名。"""
    import shutil

    root = paths.PROJECTS_DIR / PROJECT
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    poseset.save_pose_set(PROJECT, "走路", [
        {"t": i * 0.2, "yaw": 0, "pose": {}, "png": _png(30 * i + 20)}
        for i in range(1, n + 1)
    ])
    poseset.create_render(PROJECT, "走路成片", "走路")
    tmp = paths.PROJECTS_DIR / PROJECT
    for i in range(1, n + 1):
        src = tmp / f"wpcb_{i}.png"
        src.write_bytes(base64.b64decode(_png(40 * i).split(",")[1]))
        poseset.add_candidate(PROJECT, "走路成片", i, src)
    return "走路成片"


def test_bundle_contents_are_valid_godot_4():
    name = _setup(3)
    b = godotexport.export_bundle(PROJECT, name, stem="walk", anim_name="行走",
                                  fps=8, loop=True)

    assert b["frames"] == 3 and b["skipped"] == 0
    assert b["checks"]["pathsMatch"] is True, b["checks"]

    folder = paths.PROJECTS_DIR / PROJECT / godotexport.EXPORT_DIR / "walk"
    tres = (folder / "walk.tres").read_text(encoding="utf-8")
    # load_steps = 帧数 + 主资源
    assert "load_steps=4" in tres, tres.splitlines()[0]
    assert tres.count('[ext_resource type="Texture2D"') == 3
    assert 'path="res://sprites/frames/frame_000.png"' in tres
    assert '"name": &"行走"' in tres, "动画名可以是中文（Godot 允许）"
    assert '"speed": 8.0' in tres
    assert '"loop": true' in tres

    tscn = (folder / "walk.tscn").read_text(encoding="utf-8")
    assert 'type="AnimatedSprite2D"' in tscn
    assert 'path="res://sprites/walk.tres"' in tscn
    assert 'autoplay = "行走"' in tscn
    assert "texture_filter = 1" in tscn, "像素素材必须最近邻，否则糊"
    # 脚底对齐：offset.y = -高度/2
    assert "offset = Vector2(0, -8)" in tscn

    # 逐帧 PNG 都在
    frames = sorted((folder / "frames").glob("*.png"))
    assert [f.name for f in frames] == ["frame_000.png", "frame_001.png", "frame_002.png"]

    # README 要把 res:// 路径写清楚（十次导不进去九次是路径）
    readme = (folder / "README.txt").read_text(encoding="utf-8")
    assert "res://sprites/walk.tscn" in readme


def test_zip_entries_carry_folder_prefix():
    """zip 里的条目必须带 `<folder>/` 前缀 —— 解到工程根目录 res:// 才成立。"""
    name = _setup(2)
    b = godotexport.export_bundle(PROJECT, name, stem="walk", folder="my sprites")
    zpath = paths.PROJECTS_DIR / PROJECT / godotexport.EXPORT_DIR / "walk.zip"
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    assert all(n.startswith("mysprites/") for n in names), names
    assert "mysprites/walk.tres" in names
    assert "mysprites/frames/frame_000.png" in names
    # folder 被清洗过（空格去掉、小写），返回的 resPaths 也要用清洗后的
    assert b["folder"] == "mysprites"
    assert b["resPaths"]["scene"] == "res://mysprites/walk.tscn"


def test_bundle_skips_unselected_frames_and_warns():
    name = _setup(3)
    # 把第 3 帧的选中清掉：候选留着但没选 → 应该跳过并给警告
    folder = paths.PROJECTS_DIR / PROJECT / poseset.RENDERS_DIR / name
    doc = poseset._read_json(folder / poseset.RENDER_FILE)     # noqa: SLF001
    doc["frames"][2]["selected"] = -1
    poseset._write_json(folder / poseset.RENDER_FILE, doc)     # noqa: SLF001

    b = godotexport.export_bundle(PROJECT, name, stem="walk")
    assert b["frames"] == 2 and b["skipped"] == 1
    assert any("没选中" in w for w in b["warnings"]), b["warnings"]


def test_unselected_everything_is_an_error():
    name = _setup(2)
    folder = paths.PROJECTS_DIR / PROJECT / poseset.RENDERS_DIR / name
    doc = poseset._read_json(folder / poseset.RENDER_FILE)     # noqa: SLF001
    for f in doc["frames"]:
        f["selected"] = -1
    poseset._write_json(folder / poseset.RENDER_FILE, doc)     # noqa: SLF001

    try:
        godotexport.export_bundle(PROJECT, name)
    except godotexport.GodotExportError as exc:
        assert "还没有选中的帧" in str(exc)
    else:
        raise AssertionError("一帧都没选还导出成功，用户会拿到一个空包")


def test_name_sanitizers():
    """文件名必须 ASCII（跨平台安全），动画名可以中文。"""
    assert godotexport.safe_stem("走路 walk 02") == "walk02"
    assert godotexport.safe_stem("行走") == "anim"          # 全中文 → 退回默认
    assert godotexport.safe_stem("") == "anim"
    # res:// 前缀与 .. 都要洗掉（导出物是往磁盘上写的）
    assert godotexport.safe_folder("res://My Sprites/../x") == "mysprites/x"
    assert godotexport.safe_folder("") == "sprites"
    # 节点名不能带 Godot 的保留字符
    assert godotexport._node_name("a/b:c") == "a_b_c"      # noqa: SLF001
