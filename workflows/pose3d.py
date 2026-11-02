"""图片 → 3D 人体姿态 → 重新投影成「正对相机」的 2D 关节点。

**为什么需要它**：3D 人偶编辑器吃的是 2D 关节点，官方 `convertOpenPoseToPose`
拿这些点做两骨 IK，**没有深度信息**。一张侧身照片里，两侧髋部在画面上几乎
重合（投影被压缩），转换器会把这条"近乎竖直的短线段"当成"骨盆竖起来了"，
于是整个人偶绕 Z 轴转 80°+ —— 用户看到的是一坨麻花。

实测那张瑜伽照片：DWPose 的 2D 点完全正确（18 个点都落在身体上），但两髋
投影间距只有身高的 1.5%，转换器判出 -81° 骨盆倾斜；人偶躺倒。

MediaPipe Pose Landmarker 给的是 **`pose_world_landmarks`：米制 3D 点**（髋部
原点，y 向下，z 朝相机为负）。有了真实 3D 就能：

  1. 用双肩连线在 xz 平面的角度算出**躯干朝向**；
  2. 把整具骨架绕 Y 轴转到"正对相机"；
  3. 再把 (x, y) 投影成 2D —— 这时髋线宽度是真实的（实测同一张图：两髋
     投影差 0.015 → 0.156，髋线角度 -81° → -31°）。

代价：MediaPipe 的 2D 精度不如 DWPose（PDJ 91.8% vs DWPose 更高），所以
**能拿到 3D 就用 3D 重投影，拿不到就退回 DWPose**（见 `points2d` 的调用方）。

模型只有 9 MB（`pose_landmarker_full.task`，Apache-2.0），放
`models/pose_keypoint/mediapipe/`；没装 `mediapipe` 包或没放模型时
`Pose3D.available()` 为 False，调用方自动回退，不会把功能卡死。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

#: MediaPipe 33 点 → OpenPose-18 索引（值是 MediaPipe 下标）
#: 注意 MediaPipe 的 left/right 是**被摄者本人**的左右，和 OpenPose 一致。
MP_TO_OP: Dict[int, int] = {
    0: 0,      # nose
    2: 12, 3: 14, 4: 16,       # r_shoulder / r_elbow / r_wrist
    5: 11, 6: 13, 7: 15,       # l_shoulder / l_elbow / l_wrist
    8: 24, 9: 26, 10: 28,      # r_hip / r_knee / r_ankle
    11: 23, 12: 25, 13: 27,    # l_hip / l_knee / l_ankle
    14: 5, 15: 2,              # r_eye / l_eye
    16: 8, 17: 7,              # r_ear / l_ear
}
#: MediaPipe 下标（可读名字，别在代码里散落魔法数字）
MP_L_SHO, MP_R_SHO = 11, 12
MP_L_HIP, MP_R_HIP = 23, 24

#: 人物在画面里占的比例。和 `pose.normalized_points` 的 fill 保持一致，
#: 免得"从图片提姿势"和"ControlNet 骨架"两条链路出来的大小不一样。
FILL = 0.8


class Pose3D:
    """MediaPipe Pose Landmarker 的薄封装（懒加载，失败即不可用）。"""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._lm = None
        self._err: Optional[str] = None
        self._load()

    # ---------- 可用性 ----------
    @classmethod
    def available(cls, model_path: str | Path) -> bool:
        """不抛异常地问一句"能用吗"。调用方据此决定要不要回退。"""
        return Path(model_path).exists()

    def _load(self) -> None:
        if not self.model_path.exists():
            self._err = f"模型缺失：{self.model_path}"
            return
        try:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except Exception as exc:                     # noqa: BLE001  没装包
            self._err = f"mediapipe 未安装：{exc}"
            return
        try:
            opts = vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self._lm = vision.PoseLandmarker.create_from_options(opts)
        except Exception as exc:                     # noqa: BLE001
            self._err = f"加载失败：{exc}"

    @property
    def backend(self) -> str:
        return "mediapipe" if self._lm else "none"

    @property
    def error(self) -> Optional[str]:
        return self._err

    # ---------- 推理 ----------
    def world_points(self, img: Image.Image):
        """一张图 → (world 33 点, normalized 33 点)。检测不到人抛 ValueError。"""
        if self._lm is None:
            raise RuntimeError(self._err or "Pose3D 不可用")
        import mediapipe as mp

        arr = np.asarray(img.convert("RGB"))
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        if not res.pose_world_landmarks:
            raise ValueError("没检测到人体")
        return res.pose_world_landmarks[0], (res.pose_landmarks[0] if res.pose_landmarks else None)

    def points2d(self, img: Image.Image) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, float]]:
        """→ (OpenPose-18 的 0~1 点, 元信息)。

        步骤就是模块注释里那三步：算躯干朝向 → 绕 Y 转正 → 投影归一化。
        `body_yaw` 一并返回，调用方可以据此把编辑器的相机也对到同一朝向
        （不返回的话，用户看到的人偶朝向和原图对不上，会以为姿势错了）。
        """
        world, _ = self.world_points(img)
        yaw = yaw_from_shoulders(world)
        pts = to_op_points(world, yaw)
        out, span_m = normalize_points(pts, FILL)
        return out, {
            "body_yaw": round(math.degrees(yaw), 1),
            "span_m": round(span_m, 3),
            "backend": "mediapipe",
        }


# --------------------------------------------------------------------------
# 纯计算部分（不碰模型，可以直接单测）
# --------------------------------------------------------------------------

def yaw_from_shoulders(world) -> float:
    """躯干相对相机的朝向（弧度）。双肩连线在 xz 平面上的角度。

    人正对相机时接近 0；侧身 90°。**别用髋线**：这个动作里骨盆相对肩膀
    拧了 30° 以上，用髋会把人偶转歪。
    """
    dx = world[MP_L_SHO].x - world[MP_R_SHO].x
    dz = world[MP_L_SHO].z - world[MP_R_SHO].z
    return math.atan2(dz, dx)


def to_op_points(world, yaw: float) -> Dict[int, Tuple[float, float]]:
    """绕 Y 转正后取正面投影（只留 x/y，z 是把身体转正的依据，投影后不需要）。"""
    c, s = math.cos(-yaw), math.sin(-yaw)
    pts: Dict[int, Tuple[float, float]] = {}
    for op, mi in MP_TO_OP.items():
        p = world[mi]
        pts[op] = (p.x * c - p.z * s, p.y)
    # 脖子 = 双肩中点（OpenPose-18 的 1 号点，MediaPipe 没有）
    (lsx, lsy), (rsx, rsy) = pts[5], pts[2]
    pts[1] = ((lsx + rsx) / 2, (lsy + rsy) / 2)
    return pts


def normalize_points(pts: Dict[int, Tuple[float, float]],
                     fill: float = FILL) -> Tuple[Dict[int, Tuple[float, float]], float]:
    """**等比**映射到 0~1（x/y 用同一个 scale），人物占 `fill`。

    官方 `_projectNamed3DTo2D` 是 x/y 各自拉到 0.6 画布——那会把人物拉伸，
    腿长/头身比全错，姿势再准也摆不对。这里必须等比。

    返回 (归一化后的点, 人物跨度米数)。
    """
    xs = [v[0] for v in pts.values()]
    ys = [v[1] for v in pts.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) / max(0.05, float(fill))
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    out = {i: (round(min(1.0, max(0.0, 0.5 + (v[0] - cx) / span)), 4),
               round(min(1.0, max(0.0, 0.5 + (v[1] - cy) / span)), 4))
           for i, v in pts.items()}
    return out, span * float(fill)
