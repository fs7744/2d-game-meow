"""3D 姿态重投影的纯计算部分（不加载 9MB 模型，秒级）。

为什么要单独测：`convertOpenPoseToPose` 的行为完全由喂进去的 2D 点决定，
而"把 3D 点转正再投影"这一段的两个细节一旦写错，表现是**人偶躺倒/被拉长**，
在界面上很难看出是代码错还是原图姿势本来如此：

  · 归一化必须**等比**（x/y 同一个 scale）。官方那个 `_projectNamed3DTo2D`
    是 x/y 各自拉到画布 60% —— 人会被横向拉宽，腿变短；
  · 转正只能绕 Y（yaw），“把头往下压”之类的旋转会把姿势搞乱。
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from workflows.pose3d import (
    MP_TO_OP, Pose3D, normalize_points, to_op_points, yaw_from_shoulders,
)


def _world(scale: float = 1.0):
    """造一具"正对相机"的假骨架（单位米，髋为原点，y 向下）。

    坐标取自一张真实世界点的大致比例：肩宽 0.35、髋宽 0.30、身高 1.7。
    """
    pts = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(33)]
    s = scale
    pts[0] = SimpleNamespace(x=0.02, y=-0.55, z=-0.05)          # nose
    pts[11] = SimpleNamespace(x=+0.175 * s, y=-0.42, z=0.0)      # l_shoulder
    pts[12] = SimpleNamespace(x=-0.175 * s, y=-0.42, z=0.0)      # r_shoulder
    pts[13] = SimpleNamespace(x=+0.42 * s, y=-0.20, z=0.05)
    pts[14] = SimpleNamespace(x=-0.42 * s, y=-0.20, z=0.05)
    pts[15] = SimpleNamespace(x=+0.62 * s, y=0.0, z=0.08)
    pts[16] = SimpleNamespace(x=-0.62 * s, y=0.0, z=0.08)
    pts[23] = SimpleNamespace(x=+0.15, y=0.0, z=0.0)             # l_hip
    pts[24] = SimpleNamespace(x=-0.15, y=0.0, z=0.0)             # r_hip
    pts[25] = SimpleNamespace(x=+0.16, y=0.45, z=0.0)
    pts[26] = SimpleNamespace(x=-0.16, y=0.45, z=0.0)
    pts[27] = SimpleNamespace(x=+0.17, y=0.88, z=0.0)
    pts[28] = SimpleNamespace(x=-0.17, y=0.88, z=0.0)
    pts[5] = SimpleNamespace(x=+0.03, y=-0.60, z=-0.05)          # l_eye
    pts[2] = SimpleNamespace(x=-0.03, y=-0.60, z=-0.05)          # r_eye
    pts[7] = SimpleNamespace(x=+0.08, y=-0.58, z=0.0)
    pts[8] = SimpleNamespace(x=-0.08, y=-0.58, z=0.0)
    return pts


def test_mapping_is_complete_and_unique():
    """每个 OP 点各取一个 MediaPipe 点，不能有重复（左脚踝踩过的坑）。"""
    assert len(MP_TO_OP) == 17          # 0 鼻 + 16 个肢体/五官，脖子是算的
    assert len(set(MP_TO_OP.values())) == 17, "同一个 MediaPipe 点被用了两次"
    assert set(MP_TO_OP) == set(range(18)) - {1}, "OP 1（脖子）是算出来的"


def test_frontal_pose_keeps_proportions():
    """正对相机的人：归一化后肩宽/身高比 ≈ 真实比（不能被拉伸）。"""
    world = _world()
    pts = to_op_points(world, yaw_from_shoulders(world))
    norm, span_m = normalize_points(pts)

    shoulder = abs(norm[5][0] - norm[2][0])
    height = max(v[1] for v in norm.values()) - min(v[1] for v in norm.values())
    # 真实：肩宽 0.35 / 身高（≈1.7）≈ 0.206；等比投影后比值得保持
    assert abs(shoulder / height - 0.21) < 0.06, shoulder / height
    assert span_m == pytest.approx(1.48, abs=0.15)   # 眼→踝


def test_turned_body_gets_rotated_back():
    """身体侧转 60° 时，转正后肩宽（投影）应接近正对时的宽度。"""
    world = _world()
    yaw0 = yaw_from_shoulders(world)
    assert abs(yaw0) < 1e-6

    # 把整具骨架绕 Y 转 60°：肩宽的 x 分量变小、z 分量变大
    a = math.radians(60)
    for p in world:
        x, z = p.x, p.z
        p.x = x * math.cos(a) - z * math.sin(a)
        p.z = x * math.sin(a) + z * math.cos(a)
    yaw1 = yaw_from_shoulders(world)
    assert math.degrees(abs(yaw1)) == pytest.approx(60, abs=1)

    # 不转正 → 肩宽被投影压扁到一半；转正 → 恢复
    raw = to_op_points(world, 0.0)
    fixed = to_op_points(world, yaw1)
    assert abs(fixed[5][0] - fixed[2][0]) > abs(raw[5][0] - raw[2][0]) * 1.5


def test_points_within_canvas():
    world = _world()
    pts = to_op_points(world, yaw_from_shoulders(world))
    norm, _ = normalize_points(pts)
    for i, (x, y) in norm.items():
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f"点 {i} 出画布: {x},{y}"


def test_missing_model_is_not_available(tmp_path):
    """没放模型时要能问一句"可用吗"而不是抛异常（调用方据此回退 DWPose）。"""
    assert Pose3D.available(tmp_path / "nope.task") is False
    p = Pose3D(tmp_path / "nope.task")
    assert p.backend == "none"
    assert p.error and "模型缺失" in p.error
