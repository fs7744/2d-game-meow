"""GLB（glTF 2.0 二进制）读写与体检：判断一个 3D 产物"能不能进 Godot"。

**为什么自己写而不用 trimesh**：项目只依赖 PIL / numpy，引 trimesh 会顺带拖进
scipy、networkx 一串；而我们要做的事只有三件 —— 读 JSON chunk、统计面数/贴图/
骨骼、改根节点变换。GLB 的结构就是「12 字节头 + 两块 chunk」，用不着完整场景图库。

**GLB 文件格式（这一节看完就不需要查规范了）**：

    offset 0   magic   u32  = 0x46546C67（ASCII "glTF"）
    offset 4   version u32  = 2
    offset 8   length  u32  整个文件字节数
    offset 12  第一块 chunk：length u32 + type u32 + data
                type = 0x4E4F534A（"JSON"）→ glTF 的 JSON 树
                type = 0x004E4942（"BIN\\0"）→ 顶点/索引/贴图的二进制缓冲

JSON 里 `buffers[0].uri` 为空表示"数据就在 BIN chunk 里"；有 uri 才是外部文件
（那种不是自包含 GLB，Godot 也能读但容易丢文件，体检里会标出来）。

**体检判据是本项目产线的经验值，不是 glTF 规范**：面数、贴图尺寸的阈值见
`BUDGET`，超了不算错，但会在报告里给出"这个体量在 Godot 里会怎样"。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

#: 面数 / 贴图的建议上限（超过只是"建议"，不是错误）
BUDGET = {
    "mobile": {"triangles": 20_000, "texture": 1024},
    "desktop": {"triangles": 120_000, "texture": 2048},
}

#: glTF primitive.mode → 名字（只有 4 是三角形）
_MODES = {0: "POINTS", 1: "LINES", 2: "LINE_LOOP", 3: "LINE_STRIP",
          4: "TRIANGLES", 5: "TRIANGLE_STRIP", 6: "TRIANGLE_FAN"}


class GlbError(ValueError):
    """文件不合法 / 不是 GLB（调用方转 400）。"""


# ---------------------------------------------------------------- 读写


def read_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    """→ (gltf JSON 树, BIN chunk 字节)。BIN 可能为空。"""
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise GlbError("文件太短，不是 GLB")
    magic, version, length = struct.unpack("<III", data[:12])
    if magic != _GLB_MAGIC:
        raise GlbError("不是 GLB：文件头不是 glTF（普通 .gltf 是纯文本 JSON，"
                       "要么另存为 GLB，要么用 gltf 分支处理）")
    if version != 2:
        raise GlbError(f"只支持 glTF 2.0（文件是 {version}）")
    if length != len(data):
        # 长度对不上通常是被截断（下载中断），不是格式问题
        raise GlbError(f"文件长度不符：头里写 {length}，实际 {len(data)}（多半是下载不完整）")

    doc: Optional[Dict[str, Any]] = None
    binary = b""
    off = 12
    while off + 8 <= len(data):
        clen, ctype = struct.unpack("<II", data[off:off + 8])
        body = data[off + 8: off + 8 + clen]
        if len(body) < clen:
            raise GlbError("chunk 被截断")
        if ctype == _CHUNK_JSON:
            doc = json.loads(body.decode("utf-8").rstrip("\x00 "))
        elif ctype == _CHUNK_BIN:
            binary = body
        off += 8 + clen
    if doc is None:
        raise GlbError("没有 JSON chunk：不是合法 GLB")
    return doc, binary


def write_glb(path: Path, doc: Dict[str, Any], binary: bytes = b"") -> Path:
    """写回 GLB。JSON chunk 按规范用空格补齐到 4 字节（不是 \\0）。"""
    body = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    body += b" " * ((4 - len(body) % 4) % 4)
    bin_pad = binary + b"\x00" * ((4 - len(binary) % 4) % 4) if binary else b""

    chunks = struct.pack("<II", len(body), _CHUNK_JSON) + body
    if binary:
        chunks += struct.pack("<II", len(bin_pad), _CHUNK_BIN) + bin_pad
    total = 12 + len(chunks)
    out = struct.pack("<III", _GLB_MAGIC, 2, total) + chunks
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path


# ---------------------------------------------------------------- 贴图尺寸


def _png_size(b: bytes) -> Optional[Tuple[int, int]]:
    if len(b) < 24 or b[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", b[16:24])
    return int(w), int(h)


def _jpeg_size(b: bytes) -> Optional[Tuple[int, int]]:
    """扫 SOF0/SOF1/SOF2 段。JPEG 没有统一头，只能顺段走。"""
    if len(b) < 4 or b[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(b):
        if b[i] != 0xFF:
            i += 1
            continue
        marker = b[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return int(w), int(h)
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = struct.unpack(">H", b[i + 2:i + 4])[0]
        i += 2 + seg
    return None


def _image_size(gltf: Dict[str, Any], binary: bytes, image: Dict[str, Any]
                ) -> Tuple[Optional[Tuple[int, int]], int]:
    """→ ((w,h) 或 None, 字节数)。webkit/webp 不解析（TRELLIS 导出可能是 webp）。"""
    bv_index = image.get("bufferView")
    if bv_index is None:
        return None, 0
    views = gltf.get("bufferViews") or []
    if bv_index >= len(views):
        return None, 0
    v = views[bv_index]
    start = int(v.get("byteOffset") or 0)
    size = int(v.get("byteLength") or 0)
    blob = binary[start:start + size]
    mime = str(image.get("mimeType") or "")
    if mime == "image/png":
        return _png_size(blob), size
    if mime == "image/jpeg":
        return _jpeg_size(blob), size
    # webp：TRELLIS.2 的 GLB 默认用 webp（extension_webp=True），Godot 4.3+ 能读
    if mime == "image/webp" or blob[:4] == b"RIFF":
        return None, size
    return None, size


# ---------------------------------------------------------------- 统计


def _accessor_count(gltf: Dict[str, Any], idx: int) -> int:
    acc = (gltf.get("accessors") or [])
    if idx < 0 or idx >= len(acc):
        return 0
    return int(acc[idx].get("count") or 0)


def inspect(path: Path) -> Dict[str, Any]:
    """体检报告。字段名的取舍：**用 Godot 的概念**（mesh / skin / animation），
    而不是 glTF 的（mesh.primitive / skin.joints），因为报告是给"能不能进引擎"看的。
    """
    gltf, binary = read_glb(path)
    path = Path(path)

    meshes = gltf.get("meshes") or []
    prims = 0
    tris = 0
    mode_set = set()
    for m in meshes:
        for p in (m.get("primitives") or []):
            prims += 1
            mode = int(p.get("mode", 4))
            mode_set.add(mode)
            if "indices" in p:
                tris += _accessor_count(gltf, int(p["indices"])) // 3
            else:
                attr = (p.get("attributes") or {}).get("POSITION")
                if attr is not None:
                    tris += _accessor_count(gltf, int(attr)) // 3

    # AABB：把所有 POSITION accessor 的 min/max 并起来
    amin = [1e30, 1e30, 1e30]
    amax = [-1e30, -1e30, -1e30]
    for m in meshes:
        for p in (m.get("primitives") or []):
            idx = (p.get("attributes") or {}).get("POSITION")
            if idx is None:
                continue
            acc = (gltf.get("accessors") or [])[int(idx)]
            lo, hi = acc.get("min"), acc.get("max")
            if not lo or not hi:
                continue
            for i in range(3):
                amin[i] = min(amin[i], float(lo[i]))
                amax[i] = max(amax[i], float(hi[i]))
    has_aabb = amin[0] < 1e29
    size = [round(amax[i] - amin[i], 4) for i in range(3)] if has_aabb else [0, 0, 0]

    images = gltf.get("images") or []
    tex_sizes: List[Optional[Tuple[int, int]]] = []
    tex_bytes = 0
    for img in images:
        wh, n = _image_size(gltf, binary, img)
        tex_sizes.append(wh)
        tex_bytes += n

    skins = gltf.get("skins") or []
    joints = sum(len(s.get("joints") or []) for s in skins)
    anims = [str(a.get("name") or f"anim{i}") for i, a in enumerate(gltf.get("animations") or [])]

    nodes = gltf.get("nodes") or []
    # 自包含性：buffers 有外部 uri、或 images 有外部 uri，都说明"拷贝时容易丢文件"
    external = [str(b.get("uri")) for b in (gltf.get("buffers") or []) if b.get("uri")]
    external += [str(i.get("uri")) for i in images if i.get("uri")]

    return {
        "ok": True,
        "file": path.name,
        "bytes": path.stat().st_size,
        "generator": (gltf.get("asset") or {}).get("generator", ""),
        "version": (gltf.get("asset") or {}).get("version", ""),
        "meshes": len(meshes),
        "primitives": prims,
        "triangles": tris,
        "modes": sorted(mode_set),
        "nonTriangleModes": [_MODES.get(m, str(m)) for m in sorted(mode_set) if m != 4],
        "materials": len(gltf.get("materials") or []),
        "images": len(images),
        "textureBytes": tex_bytes,
        "textureSizes": [list(s) if s else None for s in tex_sizes],
        "maxTexture": max([max(s) for s in tex_sizes if s] or [0]),
        "nodes": len(nodes),
        "skins": len(skins),
        "joints": joints,
        "animations": anims,
        "aabb": {"min": [round(v, 4) for v in amin], "max": [round(v, 4) for v in amax],
                 "size": size} if has_aabb else None,
        "externalFiles": external,
        "extensionsUsed": gltf.get("extensionsUsed") or [],
    }


def grade(info: Dict[str, Any], target: str = "desktop") -> Dict[str, Any]:
    """按体量给出「能不能直接进 Godot」的判断与建议。

    这里给的是**建议**不是错误：100 万面 Godot 也能读，只是加载慢、手机上跑不动。
    """
    budget = BUDGET.get(target, BUDGET["desktop"])
    notes: List[str] = []
    level = "ok"

    if info["triangles"] > budget["triangles"] * 4:
        level = "heavy"
        notes.append(f"{info['triangles']:,} 面远超 {target} 预算 "
                     f"{budget['triangles']:,}：先用 Blender Decimate 减面")
    elif info["triangles"] > budget["triangles"]:
        level = "warn"
        notes.append(f"{info['triangles']:,} 面高于 {target} 预算 "
                     f"{budget['triangles']:,}，建议减面")

    if info["maxTexture"] > budget["texture"]:
        level = "warn" if level == "ok" else level
        notes.append(f"贴图 {info['maxTexture']}px 超过 {budget['texture']}px，"
                     "Godot 里会按原尺寸吃显存")

    if info["nonTriangleModes"]:
        notes.append(f"有非三角图元（{', '.join(info['nonTriangleModes'])}）："
                     "Godot 只渲染 TRIANGLES，其余会被忽略")
    if info["externalFiles"]:
        notes.append("引用了外部文件（" + "、".join(info["externalFiles"][:3]) +
                     "）：拷到工程里会缺资源，先另存为自包含 GLB")
    if info["skins"] and not info["animations"]:
        notes.append("有骨骼但没有动画：Godot 里是 Skeleton3D + 静态姿势，"
                     "要播放得另加 AnimationPlayer")
    if not info["skins"]:
        notes.append("无骨骼：Godot 里是静态 MeshInstance3D（道具/场景正常，"
                     "角色要动就得先绑骨）")
    if info["maxTexture"] and not any(info["textureSizes"]):
        notes.append("贴图是 webp/未知格式：Godot 4.3+ 支持 webp，更早版本读不出来")

    return {"level": level, "notes": notes, "budget": budget}


# ---------------------------------------------------------------- 规范化


def normalize(path: Path, out: Path, *, target_height: Optional[float] = 1.8,
              center_xz: bool = True, drop_to_ground: bool = True) -> Dict[str, Any]:
    """缩放 / 居中 / 落地，写成一个新的自包含 GLB。

    **为什么要做**：TRELLIS.2 输出的模型在 `[-0.5, 0.5]³` 的归一化空间里，
    高度是 1 而不是"角色身高"。Godot 里直接实例化会得到一个 1 米高、悬在原点
    的小模型 —— 用户会以为"生成坏了"。这里按包围盒换算成真实尺寸。

    不改顶点数据，只改**根节点的 scale / translation**：省内存，也保留原始精度。
    如果根节点带动画轨道，Godot 会以动画为准覆盖，所以这里额外提示调用方。
    """
    gltf, binary = read_glb(path)
    info = inspect(path)
    if not info["aabb"]:
        raise GlbError("模型没有 POSITION 的 min/max，算不出包围盒（文件可能不完整）")

    size = info["aabb"]["size"]
    lo = info["aabb"]["min"]
    hi = info["aabb"]["max"]
    tallest = max(size) or 1.0
    scale = (float(target_height) / tallest) if target_height else 1.0

    # 平移：把包围盒中心摆到原点（x/z），底面贴到 y=0
    cx = (lo[0] + hi[0]) / 2.0
    cz = (lo[2] + hi[2]) / 2.0
    ty = -lo[1] if drop_to_ground else 0.0
    tx = -cx if center_xz else 0.0
    tz = -cz if center_xz else 0.0

    nodes = gltf.setdefault("nodes", [])
    scenes = gltf.get("scenes") or []
    roots: List[int] = []
    if scenes:
        roots = list((scenes[0] or {}).get("nodes") or [])
    else:
        child_of = {c for n in nodes for c in (n.get("children") or [])}
        roots = [i for i in range(len(nodes)) if i not in child_of]

    if not roots:
        raise GlbError("场景里没有根节点，没法做规范化")

    for ri in roots:
        n = nodes[ri]
        base = n.get("scale") or [1, 1, 1]
        n["scale"] = [base[0] * scale, base[1] * scale, base[2] * scale]
        t = n.get("translation") or [0, 0, 0]
        # 平移要一起乘 scale（否则缩放后偏移量不对）
        n["translation"] = [t[0] + tx * scale, t[1] + ty * scale, t[2] + tz * scale]

    write_glb(out, gltf, binary)
    return {
        "in": str(path), "out": str(out),
        "scale": round(scale, 6),
        "targetHeight": target_height,
        "originalSize": size,
        "applied": {"scale": scale, "translate": [tx * scale, ty * scale, tz * scale]},
    }


def summary_line(info: Dict[str, Any]) -> str:
    """一行摘要（日志/前端提示用）。"""
    bits = [f"{info['triangles']:,} 面", f"{info['materials']} 材质"]
    if info["maxTexture"]:
        bits.append(f"贴图 {info['maxTexture']}px")
    bits.append(f"{info['joints']} 骨" if info["skins"] else "无骨骼")
    if info["animations"]:
        bits.append(f"{len(info['animations'])} 个动画")
    return " · ".join(bits)
