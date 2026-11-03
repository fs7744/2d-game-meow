"""TRELLIS.2 编排的纯逻辑测试 —— 不加载权重、不跑 GPU，秒级。

推理本身（四个采样段 + 网格烘焙）只能真机验，见 docs/3D模型生成-Godot工作流.md。
这里钉的是**那几处错一次就白跑十分钟**的结构：图的连接关系、参数名、
预处理裁剪。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from workflows import trellisrun as tr


# ---------------------------------------------------------------- 节点图

def test_prompt_wires_four_samplers():
    """四个采样段缺一不可，而且必须各自接对上游。

    这是照官方模板 `3d_pixal3d_trellis2_image_to_model` 抄的连接。写错不会报错，
    只是出来一坨 —— 实测把预处理漏掉那次，出来是个 1×1×1 正方体。
    """
    p = tr.build_prompt("x.png")
    samplers = {k: v for k, v in p.items() if v["class_type"] == "KSampler"}
    assert len(samplers) == 4, f"应该有四个采样段，实际 {len(samplers)}"

    # ① 结构：空 latent + 条件
    assert samplers["s_struct"]["inputs"]["latent_image"] == ["empty", 0]
    assert samplers["s_struct"]["inputs"]["positive"] == ["cond", 0]
    # ② 形状 512：从 Trellis2ShapeStage 拿稀疏 latent
    assert samplers["s_shape"]["inputs"]["latent_image"] == ["stage_shape", 2]
    # ③ 上采样：从 UpsampleStage 拿级联后的 latent，调度器是 simple（模板如此）
    assert samplers["s_up"]["inputs"]["latent_image"] == ["up", 2]
    assert samplers["s_up"]["inputs"]["scheduler"] == "simple"
    # ④ 贴图：cfg 必须是 1.0（无分类器引导）
    assert samplers["s_tex"]["inputs"]["cfg"] == 1.0
    assert samplers["s_tex"]["inputs"]["positive"] == ["stage_tex", 0]


def test_prompt_stage_chain_is_linear():
    """结构 → 形状 → 上采样 → 贴图，中间的 voxel / shape_latent 必须接同一条链。"""
    p = tr.build_prompt("x.png")
    assert p["vox"]["inputs"]["samples"] == ["s_struct", 0]
    assert p["stage_shape"]["inputs"]["voxel"] == ["vox", 0]
    assert p["up"]["inputs"]["shape_latent"] == ["s_shape", 0]
    assert p["stage_tex"]["inputs"]["shape_latent"] == ["s_up", 0]
    # 贴图 VAE 解码要子分信息，来自形状网格的第二个输出
    assert p["vox_tex"]["inputs"]["shape_subdivides"] == ["mesh_shape", 1]
    # 两个 VAE 不能混：形状用 shape_vae，贴图用 texture_vae
    assert p["vox_tex"]["inputs"]["vae"] == ["vae_tex", 0]
    assert p["mesh_shape"]["inputs"]["vae"] == ["vae_shape", 0]


def test_prompt_model_patch_chain():
    """模型补丁链：UNET → CFGOverride → RescaleCFG → ModelSamplingSD3。

    这三层是 TRELLIS.2 训练时的采样设置，去掉能跑但明显糊。结构段用 A 链、
    后三段用 B 链（两套 cfg/rescale 值，模板里就是这么分的）。
    """
    p = tr.build_prompt("x.png")
    assert p["ms_a"]["inputs"]["model"] == ["rescale_a", 0]
    assert p["rescale_a"]["inputs"]["model"] == ["cfg_a", 0]
    assert p["cfg_a"]["inputs"]["model"] == ["unet", 0]
    assert p["s_struct"]["inputs"]["model"] == ["ms_a", 0]
    for k in ("s_shape", "s_up"):
        assert p[k]["inputs"]["model"] == ["rescale_b", 0], k
    assert p["rescale_b"]["inputs"]["multiplier"] == 0.5
    assert p["rescale_a"]["inputs"]["multiplier"] == 0.7
    assert p["cfg_a"]["inputs"]["start_percent"] == 0.667
    assert p["cfg_b"]["inputs"]["start_percent"] == 0.769


def test_texture_pipeline_uses_schema_names():
    """`BakeTextureFromVoxel` 的输入叫 texture_size，不是 resolution。

    踩过：凭直觉写 resolution，报 `execute() missing 1 required positional
    argument: 'texture_size'` —— 而且要等到 UV 展开跑完（四十多秒）才炸。
    输出顺序也照 schema：base_color / metallic / roughness。
    """
    p = tr.build_prompt("x.png")
    bake = p["bake"]["inputs"]
    assert "texture_size" in bake and "resolution" not in bake
    assert bake["reference_mesh"] == ["mesh_shape", 0]
    apply_ = p["apply"]["inputs"]
    assert apply_["base_color"] == ["bake", 0]
    assert apply_["metallic"] == ["bake", 1]
    assert apply_["roughness"] == ["bake", 2]
    # 贴图必须贴回 UV 展开后的那个网格，不能贴原始网格
    assert apply_["mesh"] == ["unwrap", 0]
    assert p["bake"]["inputs"]["mesh"] == ["unwrap", 0]


def test_mesh_postprocess_order_matches_official_template():
    """网格后处理顺序：Remesh → Decimate → SmoothNormals → Unwrap → Bake。

    这是官方模板 `3d_pixal3d_trellis2_image_to_model` 的顺序，每一步都不能省：
    漏 RemeshMesh → UV 图块互相穿插，烘出大块错色斑；把 Decimate 放到 UV 之后
    （在 Blender 里做）→ 表面全是碎洞。两种都实测过。
    """
    p = tr.build_prompt("x.png", decimate_faces=200000, remesh_resolution=768)
    assert p["remesh"]["inputs"]["mesh"] == ["mesh_shape", 0]
    assert p["decimate"]["inputs"]["mesh"] == ["remesh", 0]
    assert p["decimate"]["inputs"]["target_face_count"] == 200000
    assert p["smooth_pre"]["inputs"]["mesh"] == ["decimate", 0]
    assert p["unwrap"]["inputs"]["mesh"] == ["smooth_pre", 0]
    assert p["bake"]["inputs"]["mesh"] == ["unwrap", 0]
    assert p["apply"]["inputs"]["mesh"] == ["unwrap", 0]

    # 关掉 remesh 时链要自动接上，不能留悬空引用
    off = tr.build_prompt("x.png", remesh_resolution=0)
    assert "remesh" not in off
    assert off["decimate"]["inputs"]["mesh"] == ["mesh_shape", 0]

    # 关掉减面时同理
    off2 = tr.build_prompt("x.png", decimate_faces=0)
    assert "decimate" not in off2
    assert off2["smooth_pre"]["inputs"]["mesh"] == ["remesh", 0]


def test_remesh_dynamic_combo_keys_are_dotted():
    """`sign_mode` 的子控件要用点号键（`sign_mode.fill`）。

    踩过两次：塞成 dict → `execute() missing 1 required positional argument:
    'sign_mode'`；写成裸 `fill` → 子项拿到 None，`float(None)` 崩在节点内部。
    """
    p = tr.build_prompt("x.png")
    hi = p["remesh"]["inputs"]
    assert hi["sign_mode"] == "solid"
    assert hi["sign_mode.fill"] == 20.0
    assert hi["sign_mode.qef"] is False
    assert "fill" not in hi and "qef" not in hi


def test_texture_pipeline_uses_schema_names():
    """节点参数名照 schema 抄，不能凭直觉。

    踩过：`BakeTextureFromVoxel` 的输入叫 texture_size（不是 resolution），报错
    要等到 UV 展开跑完才出现；`BakeAmbientOcclusion` / `BakeNormalMapFromMesh`
    都收**两个网格**（low_poly + high_poly），只给一个会缺三个位置参数。
    """
    p = tr.build_prompt("x.png")
    bake = p["bake"]["inputs"]
    assert "texture_size" in bake and "resolution" not in bake

    ao = p["bake_ao"]["inputs"]
    assert ao["low_poly"] == ["unwrap", 0] and ao["high_poly"] == ["remesh", 0]
    for k in ("max_distance", "strength", "bias", "samples"):
        assert k in ao, k
    nm = p["bake_normal"]["inputs"]
    assert nm["low_poly"] == ["unwrap", 0] and nm["high_poly"] == ["remesh", 0]
    assert "ignore_backfaces" in nm

    # ApplyTextureToMesh 收五张图（base/metallic/roughness/occlusion/normal）
    ap = p["apply"]["inputs"]
    assert ap["base_color"] == ["bake", 0]
    assert ap["metallic"] == ["bake", 1]
    assert ap["roughness"] == ["bake", 2]
    assert ap["occlusion"] == ["bake_ao", 0]
    assert ap["normal_map"] == ["bake_normal", 0]


def test_prompt_without_texture_skips_uv():
    """只要几何时不该跑 UV/烘焙：那两步占一半时间。"""
    p = tr.build_prompt("x.png", with_texture=False)
    assert not any(v["class_type"] in ("UnwrapMesh", "BakeTextureFromVoxel",
                                       "ApplyTextureToMesh")
                   for v in p.values())
    assert p["save"]["inputs"]["mesh"] == ["smooth0", 0]
    assert p["save"]["inputs"]["filename_prefix"] == "meow3d/model"


def test_every_link_points_to_an_existing_node():
    """所有输入连线都要有落点 —— 手改图时最容易留下悬空引用。"""
    p = tr.build_prompt("x.png")
    for nid, node in p.items():
        for key, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[1], int):
                assert val[0] in p, f"{nid}.{key} 指向不存在的节点 {val[0]}"


# ---------------------------------------------------------------- 预处理

def _make_rgba(size, box, color=(200, 120, 60, 255)) -> Image.Image:
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    im.paste(Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), color), (box[0], box[1]))
    return im


def test_prepare_image_crops_to_subject(tmp_path):
    """有 alpha 时按主体裁剪、铺黑底、出正方形。

    这一步是整条链路最容易漏的：原图矩形会被当成物体本身，出来是盒子。
    """
    src = tmp_path / "in.png"
    _make_rgba((200, 300), (50, 80, 110, 280)).save(src)   # 主体 60×200
    dst = tmp_path / "out.png"
    info = tr.prepare_image(src, dst)
    assert info["matte"] == "alpha"
    assert info["subject"] == [60, 200]
    out = Image.open(dst)
    assert out.size == (1024, 1024)
    assert out.mode == "RGB"
    # 背景是黑的（官方 ImageCropToMask 的 background 默认就是黑）
    assert out.getpixel((5, 5)) == (0, 0, 0)
    # 主体在正中
    a = out.convert("L").point(lambda v: 255 if v > 30 else 0)
    bb = a.getbbox()
    assert bb is not None
    cx = (bb[0] + bb[2]) / 2
    assert abs(cx - 512) < 12, f"主体没居中：bbox={bb}"


def test_prepare_image_pads_uniformly(tmp_path):
    """正方形输出不能拉伸：宽高比要保持（用 pad 补边而不是 resize 到 1024²）。

    早期直接 `resize((1024,1024))` 会把瘦高的人压扁 —— 重建出来又矮又宽。
    """
    src = tmp_path / "in.png"
    _make_rgba((100, 400), (10, 10, 60, 390)).save(src)     # 50×380 的竖条
    dst = tmp_path / "out.png"
    tr.prepare_image(src, dst)
    im = Image.open(dst).convert("L")
    pill = im.point(lambda v: 255 if v > 30 else 0)
    bb = pill.getbbox()
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    # 原图 50:380 ≈ 1:7.6，缩放后比例要保持（允许 pad 带来的 1~2px 误差）
    assert h / w == pytest.approx(380 / 50, rel=0.06), f"比例失真 {w}×{h}"


def test_prepare_image_rejects_empty(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(src)
    with pytest.raises(tr.TrellisError, match="扣不出主体"):
        tr.prepare_image(src, tmp_path / "out.png")


# ---------------------------------------------------------------- 可用性

def test_available_reports_missing_weights(monkeypatch, tmp_path):
    """缺权重时要能一句话说清缺哪个，而不是等十分钟后抛 CUDA 错。"""
    monkeypatch.setattr(tr, "WEIGHTS_DIR", tmp_path)
    st = tr.available()
    assert st["ok"] is False
    assert any("缺权重" in r for r in st["reasons"])
    assert "trellis_2_int8_convrot.safetensors" in " ".join(st["reasons"])


def test_model_files_match_comfy_folders():
    """模型文件名必须和 ComfyUI 的目录约定对得上（硬链接按这个建）。"""
    assert tr.MODEL_FILES["diffusion_models"].endswith(".safetensors")
    assert len(tr.MODEL_FILES["vae"]) == 2      # shape + texture，不能少
    assert tr.MODEL_FILES["clip_vision"].startswith("dino")
