"""姿态提取：YOLOX 人体检测 → 裁剪 → RTMPose(SimCC) → COCO-18 骨架图。

为什么自己实现而不用 controlnet_aux.DwposeDetector：
该版本依赖 mmpose + mmdet + mmcv（Windows 上编译极麻烦），且它要的是 .pth 而非 onnx。
这里直接用 DWPose 官方导出的两个 onnx：
  - yolox_l.onnx        人体检测（YOLOX-L, COCO）
  - dw-ll_ucoco_384.onnx 姿态回归（RTMPose, SimCC 头）
链路与 DWPose 一致：检测框 → 扩成正方形（×1.25）→ 缩放到 288×384 → 回归 17 点 → 映射回原图。

输出与 tools/make_pose.py 的 render() 对齐（COCO-18 + OpenPose 配色），
可直接作为 ControlNet OpenPose 的条件图。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---- RTMPose ----
IN_H, IN_W = 384, 288
SPLIT = 2.0
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# ---- YOLOX ----
DET_SIZE = 640
DET_PAD = 114
OBJ_THR = 0.30
NMS_THR = 0.45
PERSON_CLS = 0
EXPAND = 1.25

# COCO-17
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

# COCO-17 -> OpenPose-18（OpenPose 的 R/L 与 COCO 同侧，只是编号不同）
COCO_TO_OP = {
    2: R_SHO, 3: R_ELB, 4: R_WRI,
    5: L_SHO, 6: L_ELB, 7: L_WRI,
    8: R_HIP, 9: R_KNE, 10: R_ANK,
    11: L_HIP, 12: L_KNE, 13: R_ANK,
    14: R_EYE, 15: L_EYE, 16: R_EAR, 17: L_EAR,
}


def _letterbox(rgb: np.ndarray, size: int = DET_SIZE) -> tuple[np.ndarray, float]:
    """等比缩放 + 边补 114 到 size×size，返回 (NCHW float32, ratio)。

    注意两件事：PIL 不吃 float 数组（先转 uint8）；YOLOX 训练用的是 BGR、
    0~255 不做归一化，所以这里翻通道且不除以 255。
    """
    h, w = rgb.shape[:2]
    ratio = min(size / w, size / h)
    nw, nh = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    canvas = np.full((size, size, 3), DET_PAD, dtype=np.float32)
    src = rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    resized = np.asarray(Image.fromarray(src).resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    canvas[:nh, :nw] = resized[:, :, ::-1]          # RGB -> BGR，对齐 YOLOX 训练约定
    return canvas.transpose(2, 0, 1)[None], ratio


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
        order = rest[iou <= thr]
    return keep


class PoseExtractor:
    """单例即可，两个 onnx 都是 CPU 推理（4090 上 RTMPose 单帧 ~10ms）。"""

    def __init__(self, model_dir: str | Path, device: str = "cpu",
                 detector: str = "auto", use_detector: bool | None = None) -> None:
        """detector: auto | yolox | torchvision | none。

        **默认走 torchvision 的 FasterRCNN**，不是 YOLOX。
        原因：随包的 yolox_l.onnx 输出的是**未解码**的原始 head 结果
        （(1, 8400, 85)，前 4 列是网格坐标 + log 尺寸），而它的 anchor 布局
        与官方 YOLOX 的 anchor-major（25200 条）不一致——8400 恰好等于格子数，
        说明每格只输出一个 anchor。按官方 decode 解出来的框全是垃圾
        （实测 (-1,1,1,4)），人体裁剪完全错位，姿态回归置信度全线偏低
        （0.03~0.16，18 个关节一个都不过 0.3 的阈值）。

        torchvision 的 FasterRCNN 直接输出像素坐标，实测同一张图
        **18/18 关节全部识别**，框也准（人物位置对得上）。权重来自
        download.pytorch.org（国内可达），首次使用会自动下载约 70MB。

        yolox 仍保留为可选后端（detector="yolox"），但**结果不可信**（见
        person_boxes 里的说明）——它没有可靠的解码方式。
        """
        import onnxruntime as ort

        if use_detector is False:
            detector = "none"
        d = Path(model_dir)
        pose_p = d / "dw-ll_ucoco_384.onnx"
        det_p = d / "yolox_l.onnx"
        if not pose_p.exists():
            raise FileNotFoundError(f"姿态模型缺失: {pose_p}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device == "cuda" else ["CPUExecutionProvider"]

        self.pose = ort.InferenceSession(str(pose_p), providers=providers)
        self.pose_in = self.pose.get_inputs()[0].name
        self.device = device
        self.det = None
        self._tv = None
        self.backend = "none"

        if detector in ("auto", "torchvision"):
            try:
                self._init_torchvision()
                self.backend = "torchvision"
            except Exception:                     # 没装 torchvision / 权重取不到
                if detector == "torchvision":
                    raise
                self.backend = "none"
        if self.backend == "none" and detector == "yolox" and det_p.exists():
            self.det = ort.InferenceSession(str(det_p), providers=providers)
            self.det_in = self.det.get_inputs()[0].name
            self.backend = "yolox"

    # ---------- 人体检测 ----------
    def _init_torchvision(self) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W,
            fasterrcnn_mobilenet_v3_large_320_fpn as build,
        )

        self._tv = build(weights=W.DEFAULT).eval()
        if self.device == "cuda":
            self._tv = self._tv.to("cuda")
        self._torch = torch

    def _torchvision_boxes(self, img: Image.Image):
        import torch
        from torchvision.transforms import functional as TF

        t = TF.pil_to_tensor(img.convert("RGB")).float().div(255)
        if self.device == "cuda":
            t = t.to("cuda")
        with torch.no_grad():
            out = self._tv([t])[0]
        keep = (out["labels"] == 1) & (out["scores"] > 0.4)     # COCO 1 = person
        boxes = out["boxes"][keep].detach().cpu().numpy()
        scores = out["scores"][keep].detach().cpu().numpy()
        if len(boxes) == 0:
            return [(1.0, 0.0, 0.0, float(img.width), float(img.height))]
        idx = _nms(boxes, scores, NMS_THR)
        return [(float(scores[i]), *map(float, boxes[i])) for i in idx]

    def person_boxes(self, img: Image.Image,
                     allow_fallback: bool = True
                     ) -> List[Tuple[float, float, float, float, float]]:
        """返回 [(score, x1, y1, x2, y2)]，坐标是原图像素。

        `allow_fallback=False` 时，**检测不到人就返回空列表**（而不是整图框）。
        这个区别很重要：fallback 的整图框会让姿态回归在整张图上乱跑，
        输出一堆坐标离谱的「关节」——看起来像识别成功了，其实完全没用。
        """
        if self.backend == "torchvision":
            boxes = self._torchvision_boxes(img)
            if not allow_fallback:
                # _torchvision_boxes 没检测到时返回整图框，这里识别出来
                if (len(boxes) == 1 and boxes[0][1] <= 1 and boxes[0][2] <= 1
                        and boxes[0][3] >= img.width - 1
                        and boxes[0][4] >= img.height - 1):
                    return []
            return boxes
        if self.det is None:
            return ([(1.0, 0.0, 0.0, float(img.width), float(img.height))]
                    if allow_fallback else [])
        # ⚠️ 这个 onnx **不可用**：它输出的是未解码的 head 结果
        # ((1, 8400, 85)，前 4 列是网格坐标 + log 尺寸)，而它的 anchor 布局
        # 与官方 YOLOX 的 anchor-major（3×8400 = 25200 条）不一致——8400 恰好
        # 等于格子数，说明每格只输出一个 anchor，按官方 decode 解出来的框全是
        # 垃圾（实测 (-1,1,1,4)），人体裁剪完全错位，姿态回归置信度全线偏低
        # （0.03~0.16，18 个关节一个都不过 0.3）。
        # 没有可靠的解码方式，所以这个后端默认不启用；需要时用 detector="yolox"
        # 显式指定，但结果不可信。
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        tensor, ratio = _letterbox(arr)
        out = self.det.run(None, {self.det_in: tensor})[0][0]      # (N, 85)
        boxes = out[:, :4]
        scores = out[:, 4] * out[:, 5 + PERSON_CLS]
        keep = scores > OBJ_THR
        if not keep.any():
            return ([(1.0, 0.0, 0.0, float(img.width), float(img.height))]
                    if allow_fallback else [])
        boxes, scores = boxes[keep], scores[keep]
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        xyxy /= ratio
        idx = _nms(xyxy, scores, NMS_THR)
        return [(float(scores[i]), *map(float, xyxy[i])) for i in idx]

    @staticmethod
    def _square_crop(box, w: int, h: int, expand: float = EXPAND):
        _, x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * expand
        half = side / 2
        return (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))

    # ---------- 姿态回归 ----------
    def _pose_in_crop(self, img: Image.Image, crop) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = [int(round(v)) for v in crop]
        x2, y2 = max(x2, x1 + 2), max(y2, y1 + 2)
        patch = img.crop((x1, y1, x2, y2)).convert("RGB").resize((IN_W, IN_H), Image.BILINEAR)
        arr = np.asarray(patch, dtype=np.float32)
        arr = (arr - MEAN) / STD
        tensor = arr.transpose(2, 0, 1)[None].astype(np.float32)
        simcc_x, simcc_y = self.pose.run(None, {self.pose_in: tensor})
        kx = np.argmax(simcc_x[0], axis=-1) / SPLIT
        ky = np.argmax(simcc_y[0], axis=-1) / SPLIT
        score = (simcc_x[0].max(axis=-1) + simcc_y[0].max(axis=-1)) / 2.0
        return np.stack([kx, ky], axis=-1), score

    def keypoints(self, img: Image.Image, pick: str = "largest") -> Dict[int, Tuple[float, float, float]]:
        """OpenPose-18 索引 -> (x, y, score)，坐标归一化到 0~1。多人时取最大框。"""
        boxes = self.person_boxes(img)
        if pick == "largest":
            box = max(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
        else:
            box = max(boxes, key=lambda b: b[0])
        crop = self._square_crop(box, img.width, img.height)
        kxy, score = self._pose_in_crop(img, crop)

        cx1, cy1, cx2, cy2 = crop
        sx, sy = (cx2 - cx1) / IN_W, (cy2 - cy1) / IN_H
        xs = (cx1 + kxy[:, 0] * sx) / img.width
        ys = (cy1 + kxy[:, 1] * sy) / img.height

        out: Dict[int, Tuple[float, float, float]] = {
            0: (float(xs[NOSE]), float(ys[NOSE]), float(score[NOSE])),
            1: (float((xs[L_SHO] + xs[R_SHO]) / 2), float((ys[L_SHO] + ys[R_SHO]) / 2),
                float(min(score[L_SHO], score[R_SHO]))),
        }
        for op_idx, coco_idx in COCO_TO_OP.items():
            out[op_idx] = (float(xs[coco_idx]), float(ys[coco_idx]), float(score[coco_idx]))
        return out

    def visible_points(self, img: Image.Image, thr: float = 0.25) -> Dict[int, tuple[float, float]]:
        return {i: (x, y) for i, (x, y, s) in self.keypoints(img).items() if s >= thr}

    def pose_map(self, img: Image.Image, size: Optional[tuple[int, int]] = None,
                 thr: float = 0.25) -> Image.Image:
        import sys
        from pathlib import Path as _P

        root = _P(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tools.make_pose import render

        return render(self.visible_points(img, thr), size=size or img.size)

    def normalized_points(self, img: Image.Image, thr: float = 0.25,
                          fill: float = 0.8, min_points: int = 10
                          ) -> Dict[int, Tuple[float, float]]:
        """关键点按**人物框**归一化到 0~1（人物居中、占满画面）。

        ControlNet OpenPose 是**空间敏感**的：条件图里人物站在哪、占多大，
        生成结果里的人就站在哪、占多大。所以直接拿参考图的原始骨架是不行的
        ——参考图里的人物可能缩在角落一角，画出来就是个角落小人。

        归一化规则：
          - 用检测到的最大人物框（比关键点包围盒更稳：手伸出去时关键点框
            会比人体框大一圈）
          - 人物框按 `fill` 比例缩放（0.8 = 人物高度占画面 80%），居中放置
          - 超出画面的关节点裁掉（clip），ControlNet 不接受画外坐标
        """
        # 严格模式：检测不到人就报错，不要拿整图框硬跑
        boxes = self.person_boxes(img, allow_fallback=False)
        if not boxes:
            raise ValueError(
                "动作参考图里没检测到人物：参考图要是能识别人体的真人照片"
                "（姿态模型认不出插画、版画、像素画）")
        box = max(boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
        _, x1, y1, x2, y2 = box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        # 人体框通常是竖直的（人比宽高），但检测框可能偏方——
        # 取长边当「身高」，短边按同样比例缩放，避免人被压扁
        side = max(bw, bh)
        side = side / max(0.05, min(1.0, float(fill)))
        # 人物脚底对齐到画面底部附近（比例 0.95），头顶留出 fill 高度
        cx = (x1 + x2) / 2
        top = y2 - side * float(fill)
        left = cx - side / 2

        # **自适应选点**：不是「置信度 > thr 就要」，而是「取置信度最高的
        # min_points 个点，且不低于 thr」。
        #
        # 为什么：参考图的质量参差不齐——老照片、版画、插画、背景里还有别的人，
        # 姿态模型的置信度普遍偏低（实测一张黑白老照片里最高只有 0.16）。
        # 死守 0.25 的阈值会让这类图一个点都取不到，直接报「识别不出关节」。
        # 取 top-K 相当于「宁可要一个大致对的骨架，也不要没有骨架」——
        # 骨架本身只是给 ControlNet 一个姿态提示，差一点也能用。
        kp = self.keypoints(img)
        keep = sorted(kp.items(), key=lambda kv: -kv[1][2])
        pts: Dict[int, Tuple[float, float]] = {}

        def _add(i: int, x: float, y: float) -> None:
            nx = (x * img.width - left) / side
            ny = (y * img.height - top) / side
            pts[i] = (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))

        # 第一轮：所有「过阈值」的点都要（这是骨架的主体）
        for i, (x, y, sc) in keep:
            if sc >= thr:
                _add(i, x, y)
        # 第二轮：不够 min_points 时，从剩下的里按置信度补齐
        # （min_points 是**下限**不是上限——写反了会导致只取 10 个点，
        #   肩膀/脖子全丢，骨架只剩头+一只手+两条腿）
        if len(pts) < min_points:
            for i, (x, y, _sc) in keep:
                if i in pts:
                    continue
                _add(i, x, y)
                if len(pts) >= min_points:
                    break
        return pts

    def normalized_pose_map(self, img: Image.Image, size: tuple[int, int],
                            thr: float = 0.25, fill: float = 0.8,
                            min_points: int = 10) -> Image.Image:
        """按人物框归一化的骨架图（OpenPose-18 格式，可直接喂 ControlNet）。"""
        import sys
        from pathlib import Path as _P

        root = _P(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tools.make_pose import render

        return render(self.normalized_points(img, thr=thr, fill=fill,
                                             min_points=min_points), size=size)
