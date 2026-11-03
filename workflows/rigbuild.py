"""从「一张角色图」或「一组部件图」建出 2D cutout 骨架，并生成动作关键帧。

这条链路的三段（本文件负责前两段）：

    角色图 ──切件──▶ 6 个部件 ──建骨──▶ 骨架 + 绑定 ──生成──▶ 动作关键帧
                                        （rigmodel.py 的数据结构）

**为什么要自己切件**：pixelforge 那边的拆件走 See-Through（要 CUDA、要权重），
本项目的产线只有 Qwen。几何切件是**可用性兜底**：按人形比例把整图切成六个
区域（头 / 躯干 / 两臂 / 两腿），相邻区域重叠几像素，转动时不会露缝。

它不适用的场景要说清：**双手垂在身侧的自然站姿**，手臂区和躯干区会在 y 上
大量重叠，切出来的「臂」带一条身体边缘，转起来是纸片。想要好效果必须走真
拆件（See-Through / 分层重绘 / 人工 PSD），产物同样是「6 张 PNG + 枢轴坐标」，
`build_rig()` 对两种来源一视同仁 —— 这是刻意的：**切件质量和方法无关，
建骨只认图 + 枢轴**。

**枢轴即骨骼位置**：每个部件的旋转中心（肩 / 颈 / 髋）就是对应骨的世界位置，
贴图的 pivot 点就落在那里。这条约定省掉一整层偏移换算 —— 早期版本的
`paste_joint` 要把图垫到 2 倍画布再算中心，就是为了在没有骨骼数据时模拟这件事。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from workflows import rigmodel

#: 部件角色（与 rigwalk / pixelforge 的 parts 命名保持一致，方便直接复用）
ROLES = ("head", "torso", "arm_l", "arm_r", "leg_l", "leg_r")

#: 旋转枢轴：头绕颈（底端），其余绕近端关节（顶端）
PIVOT = {"head": "bottom", "torso": "top", "arm_l": "top",
         "arm_r": "top", "leg_l": "top", "leg_r": "top"}

#: 绘制层级（数值越大越靠前）。正面：臂在躯干前；背面：臂被躯干挡住。
Z_ORDER = {
    "front": {"leg_l": 1, "leg_r": 2, "torso": 3, "arm_l": 4, "arm_r": 5, "head": 6},
    "back": {"arm_l": 1, "arm_r": 2, "leg_l": 3, "leg_r": 4, "torso": 5, "head": 6},
    "side": {"leg_r": 1, "arm_r": 2, "torso": 3, "leg_l": 4, "arm_l": 5, "head": 6},
}

#: 人形比例默认值（相对身高）。**Q 版角色要调大 headRatio**：默认 0.26 是按
#: 6.5 头身估的，3 头身的角色头会切掉半个。
DEFAULT_PROPORTIONS = {
    "headRatio": 0.26,      # 头顶 → 颈线
    "shoulderDrop": 0.035,  # 颈线 → 肩线
    "hipRatio": 0.55,       # 头顶 → 髋线
    "torsoWidth": 0.46,     # 躯干宽 / 全身宽
    "armReach": 0.10,       # 手臂区向下延伸到髋线以下多少（占身高）
    # 肩/髋关节的横向位置（相对角色中线）。**这两个决定肢体绕哪里转**：
    # 用切区中心是错的——臂区从角色边缘算起，中心落在手臂中段，
    # 转起来整条臂"绕着肘部甩"。肩关节在躯干边缘，髋关节比肩窄。
    "shoulderHalf": 0.23,   # ≈ torsoWidth / 2
    "hipHalf": 0.10,
}


class RigBuildError(RuntimeError):
    """切件/建骨失败（API 转 400，文案要能直接给用户看）。"""


# ---------------------------------------------------------------- 基础度量


def alpha_bbox(img: Image.Image, thr: int = 8) -> Tuple[int, int, int, int]:
    """主体包围盒。有 alpha 用 alpha；没有就把「接近纯白」当背景。

    像素素材常常是「白底不透明」的（没走抠图），这时按亮度判定 —— 判错的
    后果是包围盒多出一圈白边，切出来的部件带着白框，转起来很丑。
    """
    img = img.convert("RGBA")
    a = img.split()[-1]
    if a.getextrema()[0] < 250:          # 真的有透明区
        box = a.point(lambda v: 255 if v > thr else 0).getbbox()
        if box:
            return box
    rgb = img.convert("RGB")
    gray = rgb.convert("L")
    # 白底：>240 视为背景。**不用 point().getbbox()** —— 它按非零像素算，
    # 得先反相才能得到「非白」区域。
    mask = gray.point(lambda v: 0 if v > 240 else 255)
    box = mask.getbbox()
    if not box:
        raise RigBuildError("这张图整片是白的：找不到角色轮廓（是不是传了空图/纯色图？）")
    return box


def _crop(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 <= x0 or y1 <= y0:
        raise RigBuildError(f"切区是空的（{box}）：比例参数把部件切没了，调大一些")
    return img.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------- 几何切件


def plan_regions(bbox: Tuple[int, int, int, int], *,
                 proportions: Optional[Dict[str, float]] = None,
                 overlap: int = 3) -> Dict[str, Dict[str, Any]]:
    """按人形比例算出六个部件的切区（像素坐标）与枢轴点。

    返回 `{role: {"box": (x0,y0,x1,y1), "joint": (x,y), "pivot": "top"}}`。
    纯计算、无 IO —— 单独拎出来是为了能在测试里钉住比例（`tests/test_rig.py`）。
    """
    p = dict(DEFAULT_PROPORTIONS)
    p.update(proportions or {})
    x0, y0, x1, y1 = bbox
    W, H = x1 - x0, y1 - y0
    if W < 8 or H < 8:
        raise RigBuildError(f"角色太小（{W}×{H}）：至少 8×8 才能切件")
    ov = max(0, int(overlap))
    cx = (x0 + x1) / 2.0

    neck_y = y0 + H * p["headRatio"]
    shoulder_y = neck_y + H * p["shoulderDrop"]
    hip_y = y0 + H * p["hipRatio"]
    tw = W * p["torsoWidth"]
    half_t = tw / 2.0
    arm_bottom = min(y1, hip_y + H * p["armReach"])

    if not (neck_y < shoulder_y < hip_y < y1):
        raise RigBuildError(
            f"比例参数不成立：颈线 {neck_y:.0f} / 肩线 {shoulder_y:.0f} / "
            f"髋线 {hip_y:.0f} 没有按「颈 < 肩 < 髋 < 脚」递增（图高 {H}）")

    torso_box = (cx - half_t, neck_y - ov, cx + half_t, hip_y + ov)
    head_box = (x0, y0, x1, neck_y + ov)
    arm_l_box = (x0, shoulder_y - ov, cx - half_t + ov, arm_bottom)
    arm_r_box = (cx + half_t - ov, shoulder_y - ov, x1, arm_bottom)
    leg_l_box = (x0, hip_y - ov, cx + ov, y1)
    leg_r_box = (cx - ov, hip_y - ov, x1, y1)

    # **枢轴用解剖位置**（颈线 / 肩线 / 髋线 + 各自的横向位置），不用切区边也不是
    # 切区中心。两个都踩过：
    #   · 用切区边 → 切区为了盖缝隙各外扩 overlap，头/躯干的枢轴差 2×overlap，
    #     一播就"缩脖子"（单帧看不出来）；
    #   · 用切区中心 → 臂区是从角色边缘算起的，中心落在手臂中段，整条臂绕着肘甩。
    sh_half = W * p["shoulderHalf"]
    hip_half = W * p["hipHalf"]
    regions: Dict[str, Dict[str, Any]] = {}
    for role, box, joint in (
            ("head", head_box, (cx, neck_y)),
            ("torso", torso_box, (cx, neck_y)),
            ("arm_l", arm_l_box, (cx - sh_half, shoulder_y)),
            ("arm_r", arm_r_box, (cx + sh_half, shoulder_y)),
            ("leg_l", leg_l_box, (cx - hip_half, hip_y)),
            ("leg_r", leg_r_box, (cx + hip_half, hip_y))):
        bx0, by0, bx1, by1 = (int(round(v)) for v in box)
        regions[role] = {"box": (bx0, by0, bx1, by1),
                         "joint": (float(joint[0]), float(joint[1])),
                         "pivot": PIVOT[role]}
    return regions


def split_image(img: Image.Image, *,
                proportions: Optional[Dict[str, float]] = None,
                overlap: int = 3) -> Dict[str, Any]:
    """一张角色图 → 六张部件图 + 枢轴坐标（几何切件）。"""
    src = img.convert("RGBA")
    bbox = alpha_bbox(src)
    regions = plan_regions(bbox, proportions=proportions, overlap=overlap)
    parts: Dict[str, Image.Image] = {}
    empty: List[str] = []
    for role in ROLES:
        box = regions[role]["box"]
        crop = _crop(src, box)
        tb = crop.split()[-1].getbbox()
        # 切区里没有不透明像素 = 这个部件根本不存在（比如侧面图里看不到另一只臂）
        if tb is None:
            empty.append(role)
            parts[role] = crop
            continue
        # 裁掉四周透明边：头这种「横跨全宽」的切区能省掉一大半像素，
        # 包体直接受益。裁完 box 要跟着改，否则贴图偏移算错。
        if tb != (0, 0, crop.width, crop.height):
            parts[role] = crop.crop(tb)
            regions[role]["box"] = (box[0] + tb[0], box[1] + tb[1],
                                    box[0] + tb[2], box[1] + tb[3])
        else:
            parts[role] = crop
    if len(empty) >= 3:
        raise RigBuildError(
            f"切出来的部件大部分是空的（{'、'.join(empty)}）：这张图多半不是"
            "单个正面/侧面的全身角色图。先用「拆件」拿到部件，再来建骨")
    return {"parts": parts, "regions": regions, "bbox": bbox, "empty": empty,
            "size": src.size,
            # 脚底用角色包围盒底边：腿的切区底边和它一致，但包围盒更直观
            "anchors": {"cx": (bbox[0] + bbox[2]) / 2.0, "footY": float(bbox[3])}}


# ---------------------------------------------------------------- 建骨


def build_rig(parts: Dict[str, Image.Image], joints: Dict[str, Tuple[float, float]],
              *, skeleton_id: str = "rig", direction: str = "front",
              texture_dir: str = "parts", foot_y: Optional[float] = None,
              cx: Optional[float] = None,
              boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """部件 + 枢轴 → (骨架, 绑定)。

    `joints` 是每个部件枢轴的**世界坐标**（同一张图的坐标系）。根骨放在
    「脚底中心」——这才是游戏里 `set_position(地面线)` 该落的位置。
    """
    missing = [r for r in ROLES if r not in parts or r not in joints]
    if missing:
        raise RigBuildError(f"缺少部件或枢轴：{'、'.join(missing)}")

    # 根骨原点 = 脚底中心。优先用调用方给的（切件时知道真实包围盒底边）；
    # 没给就按「腿枢轴 + 腿图高」估 —— 会多出 overlap 那几像素，能接受。
    if foot_y is None:
        foot_y = max(joints[r][1] + parts[r].height for r in ("leg_l", "leg_r")
                     if r in parts)
    if cx is None:
        cx = sum(j[0] for j in joints.values()) / len(joints)

    def rel(child: Tuple[float, float], parent: Tuple[float, float]) -> Tuple[float, float]:
        return (child[0] - parent[0], child[1] - parent[1])

    root_at = (cx, foot_y)
    torso_at = joints["torso"]
    # **根骨放在 (0,0)**，不是 (cx, foot_y)：根骨的世界位置就是「角色站在哪」，
    # 那是游戏运行时用 `set_position(地面线)` 决定的，不该烘进骨架里。
    # cx/foot_y 只在建骨时用来把其它骨换算到「脚底中心」这个原点。
    bones = [
        rigmodel.bone("root", None, x=0.0, y=0.0, name="根（脚底）"),
        rigmodel.bone("torso", "root", x=rel(torso_at, root_at)[0],
                      y=rel(torso_at, root_at)[1], name="躯干"),
        rigmodel.bone("head", "torso", x=rel(joints["head"], torso_at)[0],
                      y=rel(joints["head"], torso_at)[1], name="头"),
        rigmodel.bone("arm_l", "torso", x=rel(joints["arm_l"], torso_at)[0],
                      y=rel(joints["arm_l"], torso_at)[1], name="左臂"),
        rigmodel.bone("arm_r", "torso", x=rel(joints["arm_r"], torso_at)[0],
                      y=rel(joints["arm_r"], torso_at)[1], name="右臂"),
        rigmodel.bone("leg_l", "root", x=rel(joints["leg_l"], root_at)[0],
                      y=rel(joints["leg_l"], root_at)[1], name="左腿"),
        rigmodel.bone("leg_r", "root", x=rel(joints["leg_r"], root_at)[0],
                      y=rel(joints["leg_r"], root_at)[1], name="右腿"),
    ]
    skeleton = rigmodel.validate_skeleton(
        rigmodel.new_skeleton(skeleton_id, skeleton_id, bones))

    zmap = Z_ORDER.get(direction, Z_ORDER["front"])
    slots = []
    for role in ROLES:
        w, h = parts[role].size
        # **贴图偏移 = 让「解剖关节」落在贴图的哪一点上**。
        # Sprite2D 默认把贴图中心放在节点原点，所以要把关节在图内的位置
        # 挪到中心去：offset = 图中心 - 关节在图内的坐标。
        # 这一条同时吸收了 overlap 和裁透明边带来的位移 —— 不用各处分别补偿。
        box = (boxes or {}).get(role)
        if box:
            jx, jy = joints[role]
            off_x = w / 2.0 - (jx - box[0])
            off_y = h / 2.0 - (jy - box[1])
        else:
            off_x = 0.0
            off_y = h / 2.0 if PIVOT[role] == "top" else -h / 2.0
        slots.append({
            "id": role,
            "name": role,
            "boneId": role,                    # 部件与骨同名：导出时 NodePath 直接用
            "texture": f"{texture_dir}/{role}.png",
            "pivot": PIVOT[role],
            "offset": [round(off_x, 3), round(off_y, 3)],
            "z": zmap[role],
        })
    binding = rigmodel.validate_binding(
        rigmodel.new_binding(skeleton["id"], slots), skeleton)
    return skeleton, binding


# ---------------------------------------------------------------- 生成动作


def _rad(deg: float) -> float:
    """**度 → 弧度并取负**：设计参数用的是「逆时针为正」的直觉（和 PIL 一致），
    Godot 的 rotation 是顺时针为正。这个符号只在这里翻转一次，
    下游（预览 / 导出）一律用 Godot 语义。"""
    return -math.radians(float(deg))


def build_action(kind: str, *, frames: int = 8, samples: int = 24, fps: float = 8.0,
                 swing: float = 13.0, arm: float = 12.0, bob: float = 5.0,
                 lift: float = 7.0, squash: float = 0.06, hold: bool = False,
                 rest: Optional[Dict[str, Dict[str, float]]] = None,
                 skeleton: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """按相位算出一段动作的关键帧。

    `frames` 是**相位密度**（8 = 一圈 8 个姿态），`samples` 是**实际采样数**。
    两者分开是这条链路的关键：骨骼动画的包体只和「轨道数 × 关键帧数」有关，
    采样加密几乎不增体积，却能把 8 帧的阶梯感抹平 —— 逐帧方案做不到这点
    （加密 = 多 3 倍 PNG）。

    `rest` 不传就从骨架里取（通常是 `{id: {x, y, rotation}}`）。
    """
    if kind not in ("walk", "idle"):
        raise RigBuildError(f"不认识的动作用途：{kind}（现在支持 walk / idle）")
    n_phase = max(2, int(frames))
    n_s = max(n_phase, int(samples))
    fps = max(1.0, float(fps))
    duration = n_phase / fps

    if rest is None:
        if skeleton is None:
            raise RigBuildError("生成动作要么传 rest 变换，要么传骨架")
        rest = {b["id"]: {"x": b["rest"]["x"], "y": b["rest"]["y"],
                          "rotation": b["rest"]["rotation"]}
                for b in skeleton["bones"]}

    tracks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def key(target: str, prop: str, i: int, value: Any) -> None:
        t = round(i * duration / n_s, 6)
        tracks.setdefault((target, prop), []).append({"time": t, "value": value})

    for i in range(n_s):
        # **相位必须按采样数走一整圈**，不是按 frames。写成 `i / n_phase` 时，
        # 24 个采样会走 3 圈（i 到 24 时 ph=6π）—— 播放起来是"抽搐式快进"，
        # 而且 t=0 与 t=0.5 姿态重复。frames 只决定「一圈多长」（duration）。
        ph = 2 * math.pi * i / n_s
        if kind == "idle":
            b = bob * 0.35 * math.sin(2 * ph)
            a = arm * 0.18 * math.sin(ph + math.pi)
            key("torso", "position", i, [rest["torso"]["x"], rest["torso"]["y"] + b])
            key("arm_l", "rotation", i, _rad(a))
            key("arm_r", "rotation", i, _rad(-a))
            continue

        b = bob * math.sin(2 * ph)
        leg_a = swing * math.sin(ph)
        arm_a = arm * 0.25 * math.sin(ph) if hold else arm * math.sin(ph + math.pi)
        lift_l = lift * max(0.0, math.sin(ph))
        lift_r = lift * max(0.0, -math.sin(ph))
        sq = squash * abs(math.sin(ph))

        key("torso", "position", i, [rest["torso"]["x"], rest["torso"]["y"] + b])
        # 腿挂根骨：bob 必须自己加（躯干的 bob 带不到它）
        key("leg_l", "position", i, [rest["leg_l"]["x"],
                                     rest["leg_l"]["y"] + b - lift_l])
        key("leg_r", "position", i, [rest["leg_r"]["x"],
                                     rest["leg_r"]["y"] + b - lift_r])
        key("leg_l", "rotation", i, _rad(leg_a))
        key("leg_r", "rotation", i, _rad(-leg_a))
        key("leg_l", "scale", i, [1.0, 1.0 - sq])
        key("leg_r", "scale", i, [1.0, 1.0 - sq])
        key("arm_l", "rotation", i, _rad(arm_a))
        key("arm_r", "rotation", i, _rad(arm_a if hold else -arm_a))

    # 收尾：把第 0 帧复制到最后，循环才能严丝合缝地接上
    for (target, prop), keys in tracks.items():
        keys.append({"time": round(duration, 6), "value": keys[0]["value"]})

    order = {p: i for i, p in enumerate(("position", "rotation", "scale"))}
    motion = rigmodel.new_motion(
        kind,
        [rigmodel.new_track(t, p, k) for (t, p), k in
         sorted(tracks.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9)))],
        fps=fps, loop=True, duration=duration)
    return motion


# ---------------------------------------------------------------- 一套 rig 落盘


RIG_DIR = "rig"
SKELETON_FILE = "skeleton.json"
BINDING_FILE = "binding.json"
ACTIONS_DIR = "actions"


def build_all(img: Image.Image, out_dir: Path, *, skeleton_id: str = "rig",
              direction: str = "front",
              proportions: Optional[Dict[str, float]] = None,
              overlap: int = 3, actions: Sequence[str] = ("walk", "idle"),
              frames: int = 8, samples: int = 24, fps: float = 8.0,
              swing: float = 13.0, arm: float = 12.0, bob: float = 5.0,
              lift: float = 7.0, squash: float = 0.06, hold: bool = False
              ) -> Dict[str, Any]:
    """一张图 → 完整的 rig 目录（部件 PNG + skeleton.json + binding.json + 动作）。

    目录结构（Godot 导出会原样引用这些相对路径）：

        <out>/rig/parts/*.png
        <out>/rig/skeleton.json
        <out>/rig/binding.json
        <out>/rig/actions/walk.json
    """
    out = Path(out_dir)
    split = split_image(img, proportions=proportions, overlap=overlap)
    parts: Dict[str, Image.Image] = split["parts"]
    joints = {r: split["regions"][r]["joint"] for r in ROLES}

    skeleton, binding = build_rig(
        parts, joints, skeleton_id=skeleton_id, direction=direction,
        foot_y=split["anchors"]["footY"], cx=split["anchors"]["cx"],
        boxes={r: split["regions"][r]["box"] for r in ROLES})
    parts_dir = out / RIG_DIR / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for role, im in parts.items():
        im.save(parts_dir / f"{role}.png", "PNG", optimize=True)

    rest = {b["id"]: {"x": b["rest"]["x"], "y": b["rest"]["y"],
                      "rotation": b["rest"]["rotation"]} for b in skeleton["bones"]}

    made: Dict[str, Any] = {}
    for kind in actions:
        motion = build_action(kind, frames=frames, samples=samples, fps=fps,
                              swing=swing, arm=arm, bob=bob, lift=lift,
                              squash=squash, hold=hold, rest=rest, skeleton=skeleton)
        rigmodel.validate_motion(motion, skeleton)
        rigmodel.dump(motion, out / RIG_DIR / ACTIONS_DIR / f"{kind}.json")
        made[kind] = {"duration": motion["duration"], "tracks": len(motion["tracks"]),
                      "keys": sum(len(t["keyframes"]) for t in motion["tracks"])}

    rigmodel.dump(skeleton, out / RIG_DIR / SKELETON_FILE)
    rigmodel.dump(binding, out / RIG_DIR / BINDING_FILE)

    return {
        "skeleton": skeleton,
        "binding": binding,
        "actions": made,
        "parts": {r: list(im.size) for r, im in parts.items()},
        "bbox": split["bbox"],
        "emptyParts": split["empty"],
        "joints": {r: [round(v, 2) for v in joints[r]] for r in ROLES},
        "dir": str(out / RIG_DIR),
    }
