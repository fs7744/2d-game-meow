"""渲染 VNCCS 风格的人偶条件图：用 MakeHuman v2 网格做 Phong 软光栅化。

**为什么是软光栅化而不是 three.js**：先验证「真网格人偶能不能让 LoRA 认朝向」时
写了这个 numpy 光栅器，实测一次就对了（正面/侧面/背面全跟住），而线框和手绘人偶
都失败。它已经能产出正确的条件图，所以在引入 three.js 之前，它是最快的正解。

网格来源（CC0，37.8MB）：
    D:/code/gen/others/ComfyUI_VNCCS_Utils-main/web/pose_studio/assets/pose_studio_makehuman.v2.bin.gz
格式：`VNMHJS02` 魔数 + 4 字节小端 header 长度 + JSON header + 对齐后的二进制段。
header 里 base_vertices / base_indices / skin_indices / skin_weights 是 offset+length
描述符；bones 是 {name, parent, head_joint, tail_joint}（53 根 MakeHuman 骨）。

用法：
    python tools/render_mannequin.py --mesh <bin.gz> --yaw 90 -o out.png

TODO（下一步）：pose 参数（18 关节 → 骨骼旋转 IK）+ 蒙皮（skin_indices/weights）。
现在只渲染 base_vertices 的 A-pose 静息姿态。
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image

MAGIC = b"VNMHJS02"


def load_mesh(path: str | Path) -> dict:
    raw = gzip.open(path, "rb").read()
    assert raw[:8] == MAGIC, "不是 VNMHJS02 网格"
    hl = struct.unpack("<I", raw[8:12])[0]
    hdr = json.loads(raw[12:12 + hl].decode())
    off = 12 + hl + ((4 - ((12 + hl) % 4)) % 4)

    def view(dtype, desc):
        return np.frombuffer(raw, dtype=dtype, count=desc["length"], offset=off + desc["offset"])

    verts = view(np.float32, hdr["base_vertices"]).reshape(-1, 3)
    idx = view(np.uint32, hdr["base_indices"]).reshape(-1, 3)
    return {"verts": verts, "idx": idx, "bones": hdr["bones"], "header": hdr}


def _normals(verts, idx):
    tri = verts[idx]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    vn = np.zeros_like(verts)
    np.add.at(vn, idx.reshape(-1), np.repeat(n, 3, axis=0))
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    return vn


def rasterize(verts, idx, vn, yaw_deg: float, size: int = 768):
    """正交投影 + 单方向光 Phong，z-buffer 光栅化。返回 RGB。"""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    x = verts[:, 0] * c - verts[:, 2] * s
    y = verts[:, 1]
    z = verts[:, 0] * s + verts[:, 2] * c

    pad = 0.07
    sc = (1 - 2 * pad) * size / max(x.max() - x.min(), y.max() - y.min())
    sx = (x - (x.max() + x.min()) / 2) * sc + size / 2
    sy = size / 2 - (y - (y.max() + y.min()) / 2) * sc
    depth = -z

    Np = vn.copy()
    Np[:, 0], Np[:, 2] = vn[:, 0] * c - vn[:, 2] * s, vn[:, 0] * s + vn[:, 2] * c
    L = np.array([0.4, 0.8, 0.45]); L /= np.linalg.norm(L)
    diff = np.clip(Np @ L, 0, 1)
    shade = (0.35 + 0.65 * diff) * 255  # ambient + directional（模仿 VNCCS 灯光）

    H = W = size
    img = np.full((H, W), 255.0, np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    for a0, b0, c0 in idx:
        p = np.array([[sx[a0], sy[a0], depth[a0]],
                      [sx[b0], sy[b0], depth[b0]],
                      [sx[c0], sy[c0], depth[c0]]])
        col = (shade[a0] + shade[b0] + shade[c0]) / 3
        mnx = int(math.floor(max(0, p[:, 0].min()))); mxx = int(math.ceil(min(W - 1, p[:, 0].max())))
        mny = int(math.floor(max(0, p[:, 1].min()))); mxy = int(math.ceil(min(H - 1, p[:, 1].max())))
        if mxx <= mnx or mxy <= mny:
            continue
        area = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0])
        if abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(mnx, mxx + 1), np.arange(mny, mxy + 1))
        px, py = gx + 0.5, gy + 0.5
        w0 = ((p[1, 1] - p[2, 1]) * (px - p[2, 0]) + (p[2, 0] - p[1, 0]) * (py - p[2, 1])) / area
        w1 = ((p[2, 1] - p[0, 1]) * (px - p[2, 0]) + (p[0, 0] - p[2, 0]) * (py - p[2, 1])) / area
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        dp = w0 * p[0, 2] + w1 * p[1, 2] + w2 * p[2, 2]
        sel = inside & (dp < zbuf[mny:mxy + 1, mnx:mxx + 1])
        if not sel.any():
            continue
        img[mny:mxy + 1, mnx:mxx + 1][sel] = col
        zbuf[mny:mxy + 1, mnx:mxx + 1][sel] = dp[sel]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8), "L").convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    m = load_mesh(args.mesh)
    vn = _normals(m["verts"], m["idx"])
    rasterize(m["verts"], m["idx"], vn, args.yaw, args.size).save(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
