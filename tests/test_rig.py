"""骨骼链路（切件 → 建骨 → 动作 → Godot 导出）的纯计算测试。

**不加载任何模型、不读真实资产**：图是用 PIL 现画的小人，全程 <1 秒。
这样做是刻意的——这条链路最容易错的地方是「比例/枢轴/坐标符号」，
不是模型效果；用真图反而会掩盖这些（真图的 bbox 恰好合理，写错的公式
也能算出差不多的数）。

三类断言各自对应一个已经踩过或极易踩的坑：
  1. 比例：颈 < 肩 < 髋 < 脚 —— 反过来切出来的部件是空的。
  2. 枢轴/坐标系：根骨必须在 (0,0)、Godot 的 rotation 顺时针为正 ——
     符号错了人偶会整个倒过来（姿态那条线为此查过半天）。
  3. 导出格式：轨道路径必须指向真实存在的节点 —— Godot 对无效 NodePath
     **不报错、静默失效**，播放时就是不动，最难查。
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from workflows import godotrig, rigbuild, rigmodel


def dummy_character(w: int = 120, h: int = 240) -> Image.Image:
    """画一个正面小人：头 + 躯干 + 四肢，透明底。

    比例按 1:2 的头身比（Q 版常见），这样 headRatio=0.26 的默认值会落在
    头下缘下方一点——和真实情况一样，不刻意对齐。
    """
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = w // 2
    d.ellipse([cx - 22, 8, cx + 22, 56], fill=(240, 200, 170, 255))       # 头
    d.rectangle([cx - 26, 56, cx + 26, 140], fill=(90, 120, 200, 255))     # 躯干
    d.rectangle([cx - 44, 60, cx - 26, 130], fill=(70, 100, 180, 255))     # 左臂
    d.rectangle([cx + 26, 60, cx + 44, 130], fill=(70, 100, 180, 255))     # 右臂
    d.rectangle([cx - 24, 140, cx - 2, 232], fill=(50, 60, 90, 255))       # 左腿
    d.rectangle([cx + 2, 140, cx + 24, 232], fill=(50, 60, 90, 255))       # 右腿
    return im


# ---------------------------------------------------------------- 切件比例


def test_plan_regions_orders_human_landmarks():
    """颈线必须高于肩线、肩线高于髋线、髋线高于脚。"""
    bbox = (0, 0, 100, 200)
    r = rigbuild.plan_regions(bbox)
    neck = r["torso"]["box"][1]
    shoulder = r["arm_l"]["box"][1]
    hip = r["leg_l"]["box"][1]
    assert neck < shoulder < hip < bbox[3], (neck, shoulder, hip)


def test_plan_regions_joints_use_anatomy_not_crop_edges():
    """枢轴取**解剖位置**（颈线/肩线/髋线），不取切区边。

    切区为了盖缝隙会上下各外扩 overlap，但关节没动。用切区边的话头和躯干的
    枢轴差 2×overlap，人偶会"缩脖子"——单帧看不出来，一播就明显。
    """
    r = rigbuild.plan_regions((0, 0, 100, 200), overlap=6)
    neck = r["torso"]["joint"][1]
    assert r["head"]["joint"][1] == pytest.approx(neck), "头必须绕颈转，和躯干同点"
    assert r["torso"]["box"][1] < neck < r["torso"]["box"][3]
    assert r["arm_l"]["joint"][1] == pytest.approx(r["arm_r"]["joint"][1])
    assert r["leg_l"]["joint"][1] == pytest.approx(r["leg_r"]["joint"][1])

    # 枢轴 x 用**解剖位置**：肩在躯干边缘、髋比肩窄。
    # 用切区中心是另一个坑——臂区从角色边缘算起，中心落在手臂中段，绕肘甩。
    x0, _y0, x1, _y1 = (0, 0, 100, 200)
    cx = 50.0
    sh = 100 * rigbuild.DEFAULT_PROPORTIONS["shoulderHalf"]
    hp = 100 * rigbuild.DEFAULT_PROPORTIONS["hipHalf"]
    assert r["arm_l"]["joint"][0] == pytest.approx(cx - sh)
    assert r["arm_r"]["joint"][0] == pytest.approx(cx + sh)
    assert r["leg_l"]["joint"][0] == pytest.approx(cx - hp)
    assert r["leg_r"]["joint"][0] == pytest.approx(cx + hp)
    assert r["head"]["joint"][0] == pytest.approx(cx)
    # 关节必须落在自己的切区里，否则贴图会偏出去
    for role, info in r.items():
        bx0, _by0, bx1, _by1 = info["box"]
        assert bx0 <= info["joint"][0] <= bx1, f"{role} 的关节跑到切区外了"


def test_overlap_does_not_move_joints():
    """改 overlap 只影响切区范围，绝不能移动关节。"""
    a = rigbuild.plan_regions((0, 0, 100, 200), overlap=0)
    b = rigbuild.plan_regions((0, 0, 100, 200), overlap=8)
    for role in a:
        assert a[role]["joint"][1] == pytest.approx(b[role]["joint"][1]), role
        assert b[role]["box"][1] <= a[role]["box"][1] + 1, role


def test_plan_regions_rejects_inverted_proportions():
    with pytest.raises(rigbuild.RigBuildError):
        rigbuild.plan_regions((0, 0, 100, 200),
                              proportions={"headRatio": 0.8, "hipRatio": 0.5})


def test_split_produces_all_six_parts():
    split = rigbuild.split_image(dummy_character())
    assert set(split["parts"]) == set(rigbuild.ROLES)
    for role, im in split["parts"].items():
        assert im.width > 0 and im.height > 0, role
    assert not split["empty"], split["empty"]


def test_split_rejects_blank_image():
    with pytest.raises(rigbuild.RigBuildError):
        rigbuild.split_image(Image.new("RGBA", (64, 64), (255, 255, 255, 255)))


# ---------------------------------------------------------------- 骨架


def _rig():
    split = rigbuild.split_image(dummy_character())
    joints = {r: split["regions"][r]["joint"] for r in rigbuild.ROLES}
    return rigbuild.build_rig(split["parts"], joints)


def test_root_bone_is_at_origin():
    """根骨必须在 (0,0)：它的位置由游戏代码 `set_position(地面线)` 决定。

    烘成 (cx, foot_y) 的话，实例化后角色会偏移半个身位，用户以为是导出错。
    """
    sk, _ = _rig()
    root = next(b for b in sk["bones"] if b["parentId"] is None)
    assert (root["rest"]["x"], root["rest"]["y"]) == (0.0, 0.0)


def test_bone_hierarchy_is_single_rooted():
    sk, binding = _rig()
    ids = {b["id"] for b in sk["bones"]}
    assert ids == {"root", "torso", "head", "arm_l", "arm_r", "leg_l", "leg_r"}
    parents = {b["id"]: b["parentId"] for b in sk["bones"]}
    assert parents["head"] == "torso" and parents["arm_l"] == "torso"
    assert parents["leg_l"] == "root" == parents["leg_r"]
    # 每个部件都挂上了骨
    assert {s["boneId"] for s in binding["slots"]} == ids - {"root"}


def test_head_and_arms_are_above_torso_origin():
    """头/臂在躯干上方 = 相对 y 为负（y 轴向下）。"""
    sk, _ = _rig()
    by = {b["id"]: b["rest"] for b in sk["bones"]}
    assert by["torso"]["y"] < 0, "躯干应该在根骨（脚底）上方"
    assert by["head"]["y"] <= 0, "头应该从颈线向上（相对躯干为负或零）"
    assert by["leg_l"]["y"] < 0, "髋在脚底上方"


# ---------------------------------------------------------------- 动作


def test_walk_tracks_and_loop_closure():
    sk, _ = _rig()
    rest = {b["id"]: {"x": b["rest"]["x"], "y": b["rest"]["y"],
                      "rotation": b["rest"]["rotation"]} for b in sk["bones"]}
    m = rigbuild.build_action("walk", frames=8, samples=24, rest=rest)
    rigmodel.validate_motion(m, sk)

    props = {(t["targetId"], t["property"]) for t in m["tracks"]}
    assert props == {
        ("torso", "position"),
        ("leg_l", "position"), ("leg_r", "position"),
        ("leg_l", "rotation"), ("leg_r", "rotation"),
        ("leg_l", "scale"), ("leg_r", "scale"),
        ("arm_l", "rotation"), ("arm_r", "rotation"),
    }, props
    # 24 个采样 + 1 个收尾帧
    for t in m["tracks"]:
        assert len(t["keyframes"]) == 25
        assert t["keyframes"][0]["value"] == t["keyframes"][-1]["value"], \
            f"{t['targetId']}/{t['property']} 首尾帧不一致，循环会跳"
        assert t["keyframes"][-1]["time"] == pytest.approx(m["duration"])


def test_arm_rotation_sign_is_godot_clockwise():
    """设计参数里 arm 是「逆时针为正」，写进轨道必须是负值（Godot 顺时针为正）。

    这条符号错了一整条胳膊会反着甩，而且看起来"能动"，很容易当成参数没调好。
    """
    sk, _ = _rig()
    rest = {b["id"]: {"x": b["rest"]["x"], "y": b["rest"]["y"],
                      "rotation": b["rest"]["rotation"]} for b in sk["bones"]}
    m = rigbuild.build_action("walk", frames=8, samples=8, arm=12.0, rest=rest)
    arm = next(t for t in m["tracks"]
               if t["targetId"] == "arm_l" and t["property"] == "rotation")
    values = [k["value"] for k in arm["keyframes"]]
    assert min(values) < 0 < max(values), values
    assert max(abs(v) for v in values) == pytest.approx(math.radians(12.0), abs=1e-6)


def test_idle_has_no_leg_tracks():
    sk, _ = _rig()
    rest = {b["id"]: {"x": b["rest"]["x"], "y": b["rest"]["y"],
                      "rotation": b["rest"]["rotation"]} for b in sk["bones"]}
    m = rigbuild.build_action("idle", frames=8, samples=16, rest=rest)
    assert not [t for t in m["tracks"] if t["targetId"].startswith("leg")]


def test_sample_track_linear_and_step():
    tr = rigmodel.new_track("torso", "rotation", [
        {"time": 0.0, "value": 0.0}, {"time": 1.0, "value": 1.0}])
    assert rigmodel.sample_track(tr, 0.5) == pytest.approx(0.5)
    assert rigmodel.sample_track(tr, -1) == 0.0        # 端点外夹住
    assert rigmodel.sample_track(tr, 2) == 1.0
    tr["interpolation"] = "step"
    assert rigmodel.sample_track(tr, 0.99) == 0.0


def test_world_transform_accumulates_parent_rotation():
    """父骨转了 90°，子骨的世界位置应该跟着转过去（不是简单相加坐标）。"""
    sk = rigmodel.validate_skeleton(rigmodel.new_skeleton("t", "t", [
        rigmodel.bone("root", None, x=0, y=0),
        rigmodel.bone("a", "root", x=10, y=0),
        rigmodel.bone("b", "a", x=10, y=0),
    ]))
    local = {b["id"]: {"x": b["rest"]["x"], "y": b["rest"]["y"],
                       "rotation": b["rest"]["rotation"],
                       "sx": 1.0, "sy": 1.0, "visible": 1.0} for b in sk["bones"]}
    world = rigmodel.world_transforms(sk, local)
    assert (world["b"]["x"], world["b"]["y"]) == pytest.approx((20.0, 0.0))

    # 让 a 转 90°（顺时针，y 向下 → x 轴转到 +y 方向）
    local["a"]["rotation"] = math.pi / 2
    world = rigmodel.world_transforms(sk, local)
    assert world["a"]["x"] == pytest.approx(10.0)
    assert world["b"]["x"] == pytest.approx(10.0)
    assert world["b"]["y"] == pytest.approx(10.0)


# ---------------------------------------------------------------- 导出


def test_export_bundle_paths_and_tracks(tmp_path):
    rigdir = tmp_path / "rig"
    info = rigbuild.build_all(dummy_character(), rigdir, actions=("walk", "idle"))
    assert set(info["actions"]) == {"walk", "idle"}

    out = tmp_path / "exports"
    # build_all 的 out_dir 是「项目目录」，rig 根在它下面一层
    res = godotrig.export_bundle(Path(info["dir"]), out, stem="hero", folder="hero_rig")
    assert res["ok"]
    assert res["checks"]["pathsMatch"], res["checks"]["missing"]
    assert res["checks"]["actionsMatch"]
    assert res["checks"]["trackTargetsValid"], res["checks"]["badTargets"]

    # 场景节点：Rig + 6 根骨（**根骨不建节点**，由 Rig 自己承担）+ 6 sprite
    # + AnimationPlayer
    scene = (out / "hero" / "hero.tscn").read_text(encoding="utf-8")
    assert scene.count('[node name=') == 1 + 6 + 6 + 1
    assert '[node name="root"' not in scene, "根骨多了一个空节点，轨道会全部多一层"
    assert "texture_filter = 1" in scene
    assert "libraries = {" in scene

    # 动画资源：NodePath 指向骨节点，值是 Vector2 / float
    anim = (out / "hero" / "actions" / "walk.tres").read_text(encoding="utf-8")
    # **轨道路径必须是完整节点路径**：只写骨名时 torso 恰好在根下所以能命中，
    # arm_l 在 torso 下就 `couldn't resolve track` —— Godot 只警告，播放时静默不动
    assert 'path = NodePath("torso:position")' in anim
    assert 'path = NodePath("torso/arm_l:rotation")' in anim, "手臂轨道没写完整路径"
    assert 'NodePath("leg_l:rotation")' in anim

    # **float 属性关键帧必须是 0.0 而不是 0**：写成裸整数时 Godot 解析成 int，
    # AnimationMixer 不会把它写进 float 属性 —— position 会动、rotation 不动，
    # 而且零警告（真机验出来的）。这里直接扫 values 数组里的裸数字。
    for line in anim.splitlines():
        if line.startswith('"values": ['):
            raw = line[len('"values": ['):]
            for token in raw.replace("Vector2(", "").replace(")", "").split(","):
                tok = token.strip().rstrip("]").strip()
                if tok and tok not in ("true", "false") and "." not in tok:
                    raise AssertionError(f"values 里有裸整数 {tok!r}，Godot 会当 int 丢掉")

    # **float 属性关键帧必须是 0.0 而不是 0**：写成裸整数时 Godot 解析成 int，
    # AnimationMixer 不会把它写进 float 属性 —— position 会动、rotation 不动，
    # 而且零警告（真机验出来的）。这里直接扫 values 数组里的裸数字。
    for line in anim.splitlines():
        if line.startswith('"values": ['):
            raw = line[len('"values": ['):]
            for token in raw.replace("Vector2(", "").replace(")", "").split(","):
                tok = token.strip().rstrip("]").strip()
                if tok and tok not in ("true", "false") and "." not in tok:
                    raise AssertionError(f"values 里有裸整数 {tok!r}，Godot 会当 int 丢掉")
    assert "PackedFloat32Array(" in anim
    assert "Vector2(" in anim
    assert "loop_mode = 1" in anim

    # zip 条目带 folder 前缀（解到工程根目录路径才成立）
    import zipfile
    with zipfile.ZipFile(out / "hero.zip") as z:
        names = z.namelist()
    assert all(n.startswith("hero_rig/") for n in names)
    assert any(n.endswith("hero.tscn") for n in names)


def test_export_rejects_unknown_action(tmp_path):
    info = rigbuild.build_all(dummy_character(), tmp_path / "proj", actions=("walk",))
    with pytest.raises(godotrig.GodotRigError):
        godotrig.export_bundle(Path(info["dir"]), tmp_path / "out", actions=["fly"])
