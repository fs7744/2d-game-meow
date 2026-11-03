"""把「重画过的视角图」投影回网格 UV —— StableGen 那套贴图思路的最小实现。

为什么不用 TRELLIS 的体素烘焙
------------------------------
ComfyUI 核心节点的 `BakeTextureFromVoxel` 是把纹理 SLat 解码出的**稀疏体素**
做三线性采样再烘到 UV 上，等于给表面涂一层 512³ 的低频色块。实测结果：灰蓝斑块、
没有脸、没有布料褶皱 —— 观感像泥塑。

官方（`visualbruno/ComfyUI-Trellis2`）走的是 TRELLIS 原生纹理链：
texture SLat → **nvdiffrast 光栅化** → UV 图集。那个链在 Windows 上要
nvdiffrast/cumesh/o_voxel 的预编译轮子，而它们只有 cp311/cp312（我们是 py3.13）。

所以走 StableGen 那条：**绕模型渲染灰模 → 图像模型重画 → 投影回 UV**。
几何本来就是对的（四视图看过），缺的只是表面信息；重画能补上脸、发丝、布料纹理，
投影保证它贴回同一个几何。

投影为什么用「顶点泼溅」而不是逐三角形光栅化
--------------------------------------------
光栅化（每个三角形在 UV 空间里求覆盖）最准，但 20 万三角形在纯 numpy 里要逐包围盒
循环，几十秒到几分钟，而且对退化三角形要额外处理。这里改成：每个顶点带着自己的
UV 和 3D 位置，投影到相机后取色，**在 UV 图上泼一个半径 r 的圆盘**，再做权重归一、
空洞修补。顶点密度在 2048² 图集下约 20 texel/顶点，泼溅 + 修补足够填满；
角色贴图本身是平滑渐变，边缘略有软化可以接受。
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
         5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class TexProjectError(RuntimeError):
    """几何读不出来或投影参数不对。消息面向用户。"""


# ---------------------------------------------------------------- 读 GLB

def read_geometry(path: str | Path) -> Dict[str, np.ndarray]:
    """→ {positions (N,3), normals (N,3)|None, uvs (N,2), triangles (M,3)}。

    glTF 会按 UV 缝/法线拆顶点，所以 UV 是**逐顶点**的（不是逐角），
    这和 `TEXCOORD_0` 的访问器一一对应，直接用同一套索引即可。
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise TexProjectError(f"{path.name} 不是 GLB")
    off = 12
    gltf: Optional[dict] = None
    bin_chunk = b""
    while off + 8 <= len(raw):
        clen, ctype = struct.unpack_from("<II", raw, off)
        chunk = raw[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:          # JSON
            gltf = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:        # BIN
            bin_chunk = chunk
        off += 8 + clen
    if gltf is None:
        raise TexProjectError(f"{path.name} 里没有 JSON chunk")

    def accessor(idx: int) -> np.ndarray:
        a = gltf["accessors"][idx]
        bv = gltf["bufferViews"][a["bufferView"]]
        dtype = _COMP[a["componentType"]]
        n = _NCOMP[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        count = a["count"]
        stride = bv.get("byteStride") or (np.dtype(dtype).itemsize * n)
        if stride == np.dtype(dtype).itemsize * n:
            arr = np.frombuffer(bin_chunk, dtype=dtype, count=count * n, offset=start)
            return arr.reshape(count, n)
        # 交错缓冲：逐元素搬运
        out = np.empty((count, n), dtype=dtype)
        for i in range(count):
            out[i] = np.frombuffer(bin_chunk, dtype=dtype, count=n,
                                   offset=start + i * stride)
        return out

    prim = None
    for mesh in gltf.get("meshes", []):
        for p in mesh.get("primitives", []):
            if "TEXCOORD_0" in p.get("attributes", {}) and prim is None:
                prim = p
    if prim is None:
        raise TexProjectError("模型没有 UV（TEXCOORD_0）：先做 UV 展开再投影")

    attrs = prim["attributes"]
    pos = accessor(attrs["POSITION"]).astype(np.float32)
    uvs = accessor(attrs["TEXCOORD_0"]).astype(np.float32)
    # glTF 的 UV 原点在左上，图像坐标 y 向下；这里统一成 [0,1] 直接乘图宽高
    normals = None
    if "NORMAL" in attrs:
        normals = accessor(attrs["NORMAL"]).astype(np.float32)
    if "indices" in prim:
        idx = accessor(prim["indices"]).reshape(-1).astype(np.int64)
    else:
        idx = np.arange(len(pos), dtype=np.int64)
    tris = idx.reshape(-1, 3)
    if normals is None:
        normals = _face_normals(pos, tris)
    return {"positions": pos, "uvs": uvs, "normals": normals, "triangles": tris}


def normalize_uvs(geom: Dict[str, np.ndarray], *, margin: float = 0.01
                  ) -> Dict[str, np.ndarray]:
    """把 UV 等比缩放到占满 [0,1]（四周留 margin）。

    Blender 的 Smart UV Project 在 `scale_to_bounds=False` 时**按世界尺度保留相对
    大小**，图集常常只用掉几个百分点 —— 实测 200k 面只有 5.3% 被覆盖，每个三角形
    摊到 4 个像素，投影出来全是麻点。归一化后同样的三角形能拿到 ~40 像素。
    """
    uv = geom["uvs"].astype(np.float64)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    if span <= 1e-9:
        raise TexProjectError("UV 是退化的（所有顶点同一个 uv）")
    scale = (1.0 - 2 * margin) / span
    out = dict(geom)
    out["uvs"] = ((uv - lo) * scale + margin).astype(np.float32)
    return out


def _face_normals(pos: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """没有法线时按面积加权累加面法线到顶点。"""
    n = np.zeros_like(pos)
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    lens[lens < 1e-12] = 1.0
    return (n / lens).astype(np.float32)


# ---------------------------------------------------------------- 相机

def look_at_matrix(eye: Sequence[float], target: Sequence[float],
                   up: Sequence[float] = (0, 1, 0)) -> np.ndarray:
    """世界 → 相机（OpenGL 约定：相机看向 -Z）。"""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-9:                     # 正上方俯视时换一根参考轴
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = right, true_up, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def view_matrix(yaw_deg: float, *, target=(0, 0.95, 0), distance=3.0,
                height=1.0) -> np.ndarray:
    """绕 Y 轴一圈的机位。yaw=0 时相机在 +Z（正对角色正面）。"""
    r = math.radians(yaw_deg)
    eye = (math.sin(r) * distance, height, math.cos(r) * distance)
    return look_at_matrix(eye, target)


def project_points(world_to_cam: np.ndarray, pts: np.ndarray, *,
                   fov_deg: float, width: int, height: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """→ (像素坐标 (N,2), 深度 (N,))。深度为正表示在相机前方。"""
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = (world_to_cam @ ph.T).T[:, :3]
    depth = -cam[:, 2]
    f = (height / 2) / math.tan(math.radians(fov_deg) / 2)
    safe = np.where(depth <= 1e-6, 1e-6, depth)
    x = cam[:, 0] * f / safe + width / 2
    y = -cam[:, 1] * f / safe + height / 2
    return np.stack([x, y], axis=1), depth


# ---------------------------------------------------------------- 投影

def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    c00 = img[y0, x0].astype(np.float32)
    c10 = img[y0, x0 + 1].astype(np.float32)
    c01 = img[y0 + 1, x0].astype(np.float32)
    c11 = img[y0 + 1, x0 + 1].astype(np.float32)
    return (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
            + c01 * (1 - fx) * fy + c11 * fx * fy)


def _disk_offsets(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros((1, 2), dtype=np.int64)
    ax = np.arange(-radius, radius + 1)
    gx, gy = np.meshgrid(ax, ax, indexing="xy")
    keep = gx * gx + gy * gy <= radius * radius
    return np.stack([gx[keep], gy[keep]], axis=1).astype(np.int64)


def project_view(geom: Dict[str, np.ndarray], image: np.ndarray,
                 world_to_cam: np.ndarray, *, fov_deg: float,
                 atlas: int, splat: int = 3, power: float = 2.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """把一个视角投进图集。→ (颜色加权和 (atlas,atlas,3), 权重和 (atlas,atlas))。"""
    h, w = image.shape[:2]
    xy, depth = project_points(world_to_cam, geom["positions"],
                               fov_deg=fov_deg, width=w, height=h)
    # 面朝相机的权重：法线和视线夹角越大越不采信（掠射角会糊）
    cam_pos = -world_to_cam[:3, :3].T @ world_to_cam[:3, 3]
    view_dir = cam_pos[None, :] - geom["positions"]
    view_dir /= np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-9)
    facing = np.einsum("ij,ij->i", geom["normals"], view_dir)
    weight = np.clip(facing, 0.0, 1.0) ** power

    inside = (depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w - 1) \
        & (xy[:, 1] >= 0) & (xy[:, 1] < h - 1) & (weight > 1e-3)
    if not inside.any():
        return np.zeros((atlas, atlas, 3), np.float32), np.zeros((atlas, atlas), np.float32)

    colors = _bilinear(image, xy[inside])                # (K,3) 0~255
    uvs = geom["uvs"][inside]
    wts = weight[inside]

    px = np.round(uvs[:, 0] * (atlas - 1)).astype(np.int64)
    py = np.round(uvs[:, 1] * (atlas - 1)).astype(np.int64)
    np.clip(px, 0, atlas - 1, out=px)
    np.clip(py, 0, atlas - 1, out=py)

    offs = _disk_offsets(splat)
    acc = np.zeros((atlas, atlas, 3), np.float32)
    wacc = np.zeros((atlas, atlas), np.float32)
    for dx, dy in offs:
        tx = np.clip(px + dx, 0, atlas - 1)
        ty = np.clip(py + dy, 0, atlas - 1)
        np.add.at(acc, (ty, tx), colors * wts[:, None])
        np.add.at(wacc, (ty, tx), wts)
    return acc, wacc


def fill_holes(tex: np.ndarray, have: np.ndarray, *, rounds: int = 6
               ) -> np.ndarray:
    """把没被任何视角覆盖的 texel 填上。**必须填满**，否则引擎里是黑麻点。

    那是 UV 图集空洞，不是模型破洞 —— 但观感一模一样，所以不能留。

    优先用 `cv2.inpaint`（边缘扩散、看不出补丁）；没有 cv2 时退回迭代邻域均值
    （只在缺口很窄时够用，实测 4096 图集里的大块空洞要几百轮才收敛）。
    """
    if (~have).any():
        try:
            import cv2
            img = np.clip(tex, 0, 255).astype(np.uint8)
            mask = (~have).astype(np.uint8) * 255
            filled = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
            return filled.astype(np.float32)
        except Exception:                                 # noqa: BLE001
            pass
    out = tex.copy()
    mask = have.copy()
    for _ in range(rounds):
        if mask.all():
            break
        # 3×3 邻域里已知像素的均值
        acc = np.zeros_like(out)
        cnt = np.zeros(mask.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sh = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
                sm = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                acc += sh * sm[..., None]
                cnt += sm
        new = cnt > 0
        out[new] = acc[new] / cnt[new][..., None]
        mask |= new
    return out


def rasterize_uv(geom: Dict[str, np.ndarray], *, atlas: int,
                 chunk: int = 4000, max_px_per_tri: int = 20000
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把每个三角形铺进 UV 图集，返回每个被覆盖 texel 的 (像素索引, 3D 位置, 法线)。

    为什么必须光栅化而不是泼顶点：TRELLIS 的 `UnwrapMesh` 出的是**上千个小图块**
    （实测 8555 个 chart），顶点在图集里是稀疏的点，泼出来全是麻点、填不满任何一块。
    三角形覆盖才是图集的真实占用。

    做法是「按三角形展开候选像素」而不是逐像素求交点：每个三角形取 UV 包围盒里的
    像素，用重心坐标筛掉盒内三角形外的部分。20 万三角形 / 2048² 图集实测约 20 秒。
    """
    uv = geom["uvs"].astype(np.float64) * (atlas - 1)
    tris = geom["triangles"]
    pos = geom["positions"].astype(np.float64)
    nrm = geom["normals"].astype(np.float64)

    pix_chunks: List[np.ndarray] = []
    pos_chunks: List[np.ndarray] = []
    nrm_chunks: List[np.ndarray] = []
    for s0 in range(0, len(tris), chunk):
        t = tris[s0:s0 + chunk]
        a, b, c = uv[t[:, 0]], uv[t[:, 1]], uv[t[:, 2]]
        denom = ((b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
                 + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1]))
        ok = np.abs(denom) > 1e-12
        lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(np.int64)
        hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(np.int64)
        lo = np.clip(lo, 0, atlas - 1)
        hi = np.clip(hi, 0, atlas - 1)
        w = hi[:, 0] - lo[:, 0] + 1
        h = hi[:, 1] - lo[:, 1] + 1
        area = np.where(ok, w * h, 0)
        # 退化/超大的块（UV 出错或整块贴满）跳过，否则内存会被一个坏三角形吃光
        ok &= (area > 0) & (area <= max_px_per_tri)
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        cnt = area[idx]
        tri_ids = np.repeat(idx, cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        local = np.arange(cnt.sum()) - start
        gw = w[tri_ids]
        px = lo[tri_ids, 0] + local % gw
        py = lo[tri_ids, 1] + local // gw

        aa, bb, cc = a[tri_ids], b[tri_ids], c[tri_ids]
        d = denom[tri_ids]
        x, y = px.astype(np.float64), py.astype(np.float64)
        l0 = ((bb[:, 1] - cc[:, 1]) * (x - cc[:, 0])
              + (cc[:, 0] - bb[:, 0]) * (y - cc[:, 1])) / d
        l1 = ((cc[:, 1] - aa[:, 1]) * (x - cc[:, 0])
              + (aa[:, 0] - cc[:, 0]) * (y - cc[:, 1])) / d
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        tid = t[tri_ids[inside]]
        w0, w1, w2 = l0[inside][:, None], l1[inside][:, None], l2[inside][:, None]
        pix_chunks.append((py[inside] * atlas + px[inside]).astype(np.int64))
        pos_chunks.append(w0 * pos[tid[:, 0]] + w1 * pos[tid[:, 1]] + w2 * pos[tid[:, 2]])
        nv = w0 * nrm[tid[:, 0]] + w1 * nrm[tid[:, 1]] + w2 * nrm[tid[:, 2]]
        nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-9)
        nrm_chunks.append(nv)

    if not pix_chunks:
        raise TexProjectError("UV 光栅化什么都没有覆盖 —— UV 可能是空的")
    pix = np.concatenate(pix_chunks)
    return pix, np.concatenate(pos_chunks), np.concatenate(nrm_chunks)


def bake_atlas(geom: Dict[str, np.ndarray],
               views: Sequence[Tuple[np.ndarray, np.ndarray]],
               *, atlas: int = 2048, fov_deg: float = 34.0,
               fill_rounds: int = 8) -> Tuple[np.ndarray, float]:
    """多视角融合成一张图集。→ (uint8 (atlas,atlas,3), 覆盖率 0~1)。

    `views` = [(世界→相机矩阵, 该视角的图像 ndarray), ...]。

    可见性用「法线·视线」的幂次加权近似（掠射角几乎不计分），不做逐视角深度缓冲：
    背面朝相机的面天然得分 0，所以前后重叠基本不冲突；代价是凹处（比如裙摆内壁）
    可能从错误视角拿到颜色，靠多视角平均压下去。
    """
    pix, pos, nrm = rasterize_uv(geom, atlas=atlas)
    n_tex = atlas * atlas
    acc = np.zeros((n_tex, 3), np.float32)
    wacc = np.zeros(n_tex, np.float32)

    for m, img in views:
        h, w = img.shape[:2]
        flat = img.reshape(-1, 3).astype(np.float32)
        xy, depth = project_points(m, pos, fov_deg=fov_deg, width=w, height=h)
        cam_pos = -m[:3, :3].T @ m[:3, 3]
        vdir = cam_pos[None, :] - pos
        vdir /= np.maximum(np.linalg.norm(vdir, axis=1, keepdims=True), 1e-9)
        wt = np.clip(np.einsum("ij,ij->i", nrm, vdir), 0.0, 1.0) ** 2
        ok = (depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w - 1) \
            & (xy[:, 1] >= 0) & (xy[:, 1] < h - 1) & (wt > 1e-3)
        if not ok.any():
            continue
        colors = _bilinear(img, xy[ok])
        # 按像素索引累加：同一 texel 会有多个三角形覆盖（接缝），取最大权重那次的颜色
        # 更稳，但简单累加平均也能接受 —— 这里用平均，避免接缝出现硬边
        np.add.at(acc, pix[ok], colors * wt[ok][:, None])
        np.add.at(wacc, pix[ok], wt[ok])

    covered = wacc > 1e-6
    tex = np.zeros((n_tex, 3), np.float32)
    tex[covered] = acc[covered] / wacc[covered][:, None]
    tex = tex.reshape(atlas, atlas, 3)
    tex = fill_holes(tex, covered.reshape(atlas, atlas), rounds=fill_rounds)
    coverage = float(covered.mean())
    return np.clip(tex, 0, 255).astype(np.uint8), coverage


def write_texture(rgb: np.ndarray, path: str | Path) -> Path:
    from PIL import Image
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(p)
    return p
