"""GLB 读写 / 体检 / 规范化的测试。

**全部用构造的最小 GLB，不依赖外部文件**：这批函数最容易错的是「字节偏移」
「chunk 对齐」「包围盒换算」，而不是"能不能读某个真实模型"。用手造的极小样本
反而更容易钉住这些（真实文件动辄 2MB，测试跑起来慢，而且改坏一处未必报错）。

真实模型的端到端验证在 `tools/verify_glb_godot.py`（要 Godot），不在 pytest 里。
"""
from __future__ import annotations

import json
import struct

import pytest

from workflows import glb as glbmod


def make_glb(path, *, positions_min=(-0.5, 0.0, -0.5),
             positions_max=(0.5, 1.0, 0.5), triangles=2,
             with_skin=False, with_anim=False, external_buffer=False):
    """造一个最小合法 GLB：一个 mesh + N 个三角面。

    顶点数固定 6（两个三角面），索引不写（走非索引路径）—— 两条统计分支
    都要能算对，另一条在 test_inspect_counts_indexed 里单独造。
    """
    doc = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "root", "mesh": 0}],
        "meshes": [{"name": "m", "primitives": [
            {"attributes": {"POSITION": 0}, "mode": 4}]}],
        "accessors": [{
            "bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3",
            "min": list(positions_min), "max": list(positions_max),
        }],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 6 * 12}],
    }
    if external_buffer:
        doc["buffers"] = [{"byteLength": 6 * 12, "uri": "model.bin"}]
    else:
        doc["buffers"] = [{"byteLength": 6 * 12}]
    if with_skin:
        doc["skins"] = [{"joints": [1, 2, 3]}]
        doc["nodes"].append({"name": "Bone"})
    if with_anim:
        doc["animations"] = [{"name": "Walk", "channels": [], "samplers": []}]

    bin_data = b"\x00" * (6 * 12)
    glbmod.write_glb(path, doc, b"" if external_buffer else bin_data)
    return path


def test_roundtrip_preserves_json(tmp_path):
    p = make_glb(tmp_path / "a.glb")
    doc, binary = glbmod.read_glb(p)
    assert doc["asset"]["version"] == "2.0"
    assert len(binary) == 6 * 12

    # 写回去再读，JSON 必须一致（chunk 对齐用空格补齐，不能补 \0）
    p2 = tmp_path / "b.glb"
    glbmod.write_glb(p2, doc, binary)
    doc2, bin2 = glbmod.read_glb(p2)
    assert doc2 == doc
    assert bin2 == binary


def test_chunk_length_is_multiple_of_four(tmp_path):
    """JSON chunk 必须 4 字节对齐，否则 Godot / 其他 loader 会读歪。

    规范要求 JSON chunk 用**空格**补齐（不是 \\0，\\0 会让 JSON 解析失败）。
    """
    p = make_glb(tmp_path / "a.glb")
    raw = p.read_bytes()
    clen, ctype = struct.unpack("<II", raw[12:20])
    assert ctype == 0x4E4F534A
    assert clen % 4 == 0
    body = raw[20:20 + clen]
    assert body.rstrip(b" ")  # 去掉尾随空格后仍是完整 JSON
    json.loads(body.decode("utf-8"))


def test_inspect_counts_triangles_and_aabb(tmp_path):
    p = make_glb(tmp_path / "a.glb")
    info = glbmod.inspect(p)
    assert info["triangles"] == 2          # 6 个顶点 / 3
    assert info["meshes"] == 1
    assert info["aabb"]["size"] == [1.0, 1.0, 1.0]
    assert info["skins"] == 0


def test_inspect_counts_indexed(tmp_path):
    """有 indices 时按索引数算面数（不是顶点数）。"""
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
             "min": [-1, -1, -1], "max": [1, 1, 1]},
            {"bufferView": 1, "componentType": 5123, "count": 12, "type": "SCALAR"},
        ],
        "bufferViews": [{"buffer": 0, "byteLength": 8 * 12},
                        {"buffer": 1, "byteLength": 12 * 2}],
        "buffers": [{"byteLength": 8 * 12 + 12 * 2}],
    }
    p = tmp_path / "i.glb"
    glbmod.write_glb(p, doc, b"\x00" * (8 * 12 + 12 * 2))
    assert glbmod.inspect(p)["triangles"] == 4      # 12 / 3


def test_inspect_reports_skin_and_anim(tmp_path):
    p = make_glb(tmp_path / "a.glb", with_skin=True, with_anim=True)
    info = glbmod.inspect(p)
    assert info["skins"] == 1 and info["joints"] == 3
    assert info["animations"] == ["Walk"]


def test_rejects_non_glb(tmp_path):
    p = tmp_path / "x.glb"
    p.write_bytes(b"not a glb at all, just text")
    with pytest.raises(glbmod.GlbError, match="不是 GLB"):
        glbmod.read_glb(p)


def test_rejects_truncated(tmp_path):
    """长度字段和实际字节数不符 = 下载被截断（最常见的"文件损坏"）。"""
    p = make_glb(tmp_path / "a.glb")
    raw = p.read_bytes()
    with open(p, "wb") as f:
        f.write(raw[:-8])
    with pytest.raises(glbmod.GlbError, match="长度不符"):
        glbmod.read_glb(p)


def test_normalize_scales_to_target_height_and_drops_to_ground(tmp_path):
    # 模型原始：x/z 在 ±0.5，y 在 0~1（已经站在地面）
    p = make_glb(tmp_path / "a.glb", positions_min=(-0.5, 0.0, -0.5),
                 positions_max=(0.5, 1.0, 0.5))
    out = tmp_path / "n.glb"
    r = glbmod.normalize(p, out, target_height=2.0)
    assert r["scale"] == pytest.approx(2.0)

    doc, _ = glbmod.read_glb(out)
    root = doc["nodes"][0]
    assert root["scale"] == pytest.approx([2.0, 2.0, 2.0])
    # 底面 y=0 → 不需要往上抬
    assert root["translation"][1] == pytest.approx(0.0)


def test_normalize_raises_below_ground_model(tmp_path):
    """模型本身浮在 y 上方时，要往下拉，让脚底贴地。"""
    p = make_glb(tmp_path / "a.glb", positions_min=(-0.5, 3.0, -0.5),
                 positions_max=(0.5, 4.0, 0.5))
    out = tmp_path / "n.glb"
    r = glbmod.normalize(p, out, target_height=1.0)
    # 缩放 1.0，平移应为 -3（把 y=3 的底面拉到 0）
    assert r["applied"]["translate"][1] == pytest.approx(-3.0)


def test_grade_flags_heavy_mesh():
    info = {"triangles": 900_000, "maxTexture": 4096, "nonTriangleModes": [],
            "externalFiles": [], "skins": 1, "animations": ["a"], "textureSizes": [[4096, 4096]]}
    g = glbmod.grade(info, "mobile")
    assert g["level"] == "heavy"
    assert any("Decimate" in n or "减面" in n for n in g["notes"])


def test_grade_flags_external_files_and_no_skin():
    info = {"triangles": 100, "maxTexture": 0, "nonTriangleModes": [],
            "externalFiles": ["model.bin"], "skins": 0, "animations": [],
            "textureSizes": []}
    notes = " ".join(glbmod.grade(info)["notes"])
    assert "外部文件" in notes
    assert "无骨骼" in notes
