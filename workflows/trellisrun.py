"""TRELLIS.2 图生 3D —— **在项目进程内**跑 ComfyUI 的节点图，不起服务。

为什么要这么绕
--------------
微软官方 `microsoft/TRELLIS.2` 依赖 `nvdiffrast` / `cumesh` / `o-voxel` /
`flexgemm`，全部要编译 CUDA 扩展；本机没有 nvcc，PyPI 上也没有 Windows 轮子
（见 docs/3D模型生成-Godot工作流.md §2）。

但 ComfyUI 把 TRELLIS.2 **用纯 PyTorch 重写**了（`comfy/ldm/trellis2/`，
vae + model + flexgemm 三个文件，零编译依赖）。所以把 ComfyUI 当**库**用：
`import nodes` → 注册节点 → 用它的 `PromptExecutor` 跑一张 API 格式的图。
用户看不到 ComfyUI，也没有额外进程。

两处必须先做的前置
------------------
1. `tools/patch_comfy_kitchen.py`：comfy_kitchen 注册 torch custom op 时用了
   PEP585 的 `list[int]`，torch 2.6 的 `infer_schema` 只认 `typing.List`，
   import 阶段就抛错。
2. 模型目录要在 `import nodes` **之前**通过 `comfy.cli_args.args.models_directory`
   指过去 —— `folder_paths` 在 import 时就把路径算死了，之后再改没用。
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths

# 权重：四个 safetensors 放这里（魔搭 Comfy-Org/TRELLIS.2 那份，int8 约 8GB）
WEIGHTS_DIR = paths.ROOT / "models" / "trellis2"

# ComfyUI 源码。用环境变量覆盖，方便换机器 / 换版本
COMFY_ROOT = Path(os.environ.get(
    "MEOW_COMFYUI", str(paths.ROOT.parent / "gen" / "tools" / "ComfyUI-master")))

# ComfyUI 认的模型文件名（它按目录扫描，不看我们这边怎么命名）
MODEL_FILES = {
    "diffusion_models": "trellis_2_int8_convrot.safetensors",
    "vae": ["trellis_2_shape_vae_bf16.safetensors",
            "trellis_2_texture_vae_bf16.safetensors"],
    "clip_vision": "dino_v3_vit_l.safetensors",
}

_boot: Optional[Dict[str, Any]] = None


class TrellisError(RuntimeError):
    """生成链路不可用或推理失败。消息面向用户，别塞堆栈。"""


# ---------------------------------------------------------------- 环境准备

def available() -> Dict[str, Any]:
    """能跑吗？→ {ok, reasons, weights, comfy}。不加载模型，很快。"""
    reasons: List[str] = []
    if not COMFY_ROOT.is_dir():
        reasons.append(f"找不到 ComfyUI：{COMFY_ROOT}（设 MEOW_COMFYUI 环境变量）")
    missing = []
    for sub, names in MODEL_FILES.items():
        for n in ([names] if isinstance(names, str) else names):
            p = WEIGHTS_DIR / n
            if not p.is_file() or p.stat().st_size < 1024:
                missing.append(f"{sub}/{n}")
    if missing:
        reasons.append("缺权重：" + "、".join(missing))
    try:
        import torch
        if not torch.cuda.is_available():
            reasons.append("CUDA 不可用（TRELLIS.2 跑 CPU 不现实）")
    except Exception:                                     # noqa: BLE001
        reasons.append("torch 不可用")
    return {"ok": not reasons, "reasons": reasons,
            "comfy": str(COMFY_ROOT), "weights": str(WEIGHTS_DIR)}


def _link_models() -> Path:
    """把权重以**硬链接**放进 ComfyUI 的 models 子目录。

    同一个卷上做硬链接不占空间、瞬间完成 —— 8GB 的权重没必要复制一份。
    跨卷时退回复制（会很慢，但至少能用）。
    """
    root = paths.ROOT / "models" / "comfy"
    for sub, names in MODEL_FILES.items():
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for n in ([names] if isinstance(names, str) else names):
            src = WEIGHTS_DIR / n
            dst = d / n
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
    return root


def bootstrap() -> Dict[str, Any]:
    """import ComfyUI 并注册节点。幂等。→ {nodes, folder_paths, execution, ...}"""
    global _boot
    if _boot is not None:
        return _boot
    st = available()
    if not st["ok"]:
        raise TrellisError("；".join(st["reasons"]))

    models = _link_models()
    work = paths.ROOT / "tests" / "_out" / "trellis"
    in_dir, out_dir, tmp_dir = work / "input", work / "output", work / "temp"
    for d in (in_dir, out_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    prev_cwd = os.getcwd()
    sys.path.insert(0, str(COMFY_ROOT))
    os.chdir(COMFY_ROOT)          # ComfyUI 内部有相对路径假设（app 目录、custom_nodes）
    try:
        import comfy.cli_args as cli_args
        # **必须在 import nodes 之前**：folder_paths 在 import 时就算死了路径
        cli_args.args.models_directory = str(models)
        cli_args.args.output_directory = str(out_dir)
        cli_args.args.input_directory = str(in_dir)
        cli_args.args.temp_directory = str(tmp_dir)

        import asyncio
        import folder_paths
        import nodes
        import execution
        # **注册节点之前就要有 stub server**：`nodes_replacements.py` 在
        # `comfy_entrypoint` 里就访问 `PromptServer.instance`，那时还没建执行器，
        # 于是整份节点替换模块 import 失败（日志只有一行 WARNING，很容易漏）。
        _install_server_stub(_Server())
        asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False))

        # 再兜一层：显式设一遍（有些版本会从别处重算）
        folder_paths.set_output_directory(str(out_dir))
        folder_paths.set_input_directory(str(in_dir))
        folder_paths.set_temp_directory(str(tmp_dir))
    finally:
        os.chdir(prev_cwd)

    for need in ("UNETLoader", "VAELoader", "CLIPVisionLoader", "KSampler",
                 "Trellis2Conditioning", "EmptyTrellis2LatentStructure",
                 "Trellis2ShapeStage", "VaeDecodeStructureTrellis2",
                 "VaeDecodeShapeTrellis", "Trellis2UpsampleStage",
                 "Trellis2TextureStage", "VaeDecodeTextureTrellis",
                 "UnwrapMesh", "BakeTextureFromVoxel", "ApplyTextureToMesh",
                 "MeshSmoothNormals", "SaveGLB"):
        if need not in nodes.NODE_CLASS_MAPPINGS:
            raise TrellisError(
                f"ComfyUI 没注册节点 {need}（版本不匹配？先跑 tools/patch_comfy_kitchen.py）")

    _boot = {"nodes": nodes, "folder_paths": folder_paths,
             "execution": execution, "input": in_dir, "output": out_dir,
             "models": models}
    return _boot


# ---------------------------------------------------------------- 图构造

def build_prompt(image_name: str, *, seed: int = 42, with_texture: bool = True,
                 shape_steps: int = 12, refine_steps: int = 20,
                 texture_steps: int = 12, cfg: float = 7.5,
                 target_resolution: int = 1536, uv_resolution: int = 2048,
                 decimate_faces: int = 80000, remesh_resolution: int = 768,
                 bake_reference: bool = True, bake_ao_normal: bool = True,
                 uv_segmenter: str = "pec", uv_padding: int = 4,
                 filename_prefix: str = "meow3d/model") -> Dict[str, Any]:
    """拼 API 格式的节点图 —— 严格照官方模板 `3d_pixal3d_trellis2_image_to_model`。

    四段采样，缺一段结果就不对：

    | 段 | 做什么 | 关键参数 |
    |---|---|---|
    | 结构 | 图像条件 → 体素结构（32³） | steps 12 / cfg 7.5 / euler+normal |
    | 形状 512 | 稀疏 latent 细化 | steps 20 / cfg 7.5 |
    | 上采样 1024 | 1536 分辨率级联 | steps 12 / euler+**simple** |
    | 贴图 | 在同一条形状 latent 上画颜色 | steps 12 / **cfg 1**（无分类器引导）|

    模型侧三个补丁也别漏：`CFGOverride(1, 0.667)` + `RescaleCFG(0.7)` +
    `ModelSamplingSD3(shift=5)` 是 TRELLIS.2 训练时的采样设置，去掉会明显糊。
    """
    nodes_map: Dict[str, Any] = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": MODEL_FILES["diffusion_models"],
                            "weight_dtype": "default"}},
        "vae_shape": {"class_type": "VAELoader",
                      "inputs": {"vae_name": MODEL_FILES["vae"][0]}},
        "vae_tex": {"class_type": "VAELoader",
                    "inputs": {"vae_name": MODEL_FILES["vae"][1]}},
        "clip": {"class_type": "CLIPVisionLoader",
                 "inputs": {"clip_name": MODEL_FILES["clip_vision"]}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image_name}},

        # 模型补丁链：UNET → CFGOverride → RescaleCFG → ModelSamplingSD3
        "cfg_a": {"class_type": "CFGOverride",
                  "inputs": {"model": ["unet", 0], "cfg": 1.0,
                             "start_percent": 0.667, "end_percent": 1.0}},
        "rescale_a": {"class_type": "RescaleCFG",
                      "inputs": {"model": ["cfg_a", 0], "multiplier": 0.7}},
        "ms_a": {"class_type": "ModelSamplingSD3",
                 "inputs": {"model": ["rescale_a", 0], "shift": 5.0}},
        "cfg_b": {"class_type": "CFGOverride",
                  "inputs": {"model": ["unet", 0], "cfg": 1.0,
                             "start_percent": 0.769, "end_percent": 1.0}},
        "rescale_b": {"class_type": "RescaleCFG",
                      "inputs": {"model": ["cfg_b", 0], "multiplier": 0.5}},

        "cond": {"class_type": "Trellis2Conditioning",
                 "inputs": {"clip_vision_model": ["clip", 0],
                            "image": ["load", 0]}},
        "empty": {"class_type": "EmptyTrellis2LatentStructure",
                  "inputs": {"batch_size": 1}},

        # ① 结构：出 32³ 体素
        "s_struct": {"class_type": "KSampler",
                     "inputs": {"model": ["ms_a", 0], "seed": seed,
                                "steps": shape_steps, "cfg": cfg,
                                "sampler_name": "euler", "scheduler": "normal",
                                "positive": ["cond", 0], "negative": ["cond", 1],
                                "latent_image": ["empty", 0], "denoise": 1.0}},
        "vox": {"class_type": "VaeDecodeStructureTrellis2",
                "inputs": {"samples": ["s_struct", 0], "vae": ["vae_shape", 0],
                           "resolution": "32"}},

        # ② 形状 512
        "stage_shape": {"class_type": "Trellis2ShapeStage",
                        "inputs": {"positive": ["cond", 0],
                                   "negative": ["cond", 1],
                                   "voxel": ["vox", 0]}},
        "s_shape": {"class_type": "KSampler",
                    "inputs": {"model": ["rescale_b", 0], "seed": seed,
                               "steps": refine_steps, "cfg": cfg,
                               "sampler_name": "euler", "scheduler": "normal",
                               "positive": ["stage_shape", 0],
                               "negative": ["stage_shape", 1],
                               "latent_image": ["stage_shape", 2],
                               "denoise": 1.0}},
        "mesh_shape": {"class_type": "VaeDecodeShapeTrellis",
                       "inputs": {"samples": ["s_shape", 0],
                                  "vae": ["vae_shape", 0]}},

        # ③ 上采样到 1024 级
        "up": {"class_type": "Trellis2UpsampleStage",
               "inputs": {"positive": ["stage_shape", 0],
                          "negative": ["stage_shape", 1],
                          "shape_latent": ["s_shape", 0],
                          "vae": ["vae_shape", 0],
                          "target_resolution": target_resolution}},
        "s_up": {"class_type": "KSampler",
                 "inputs": {"model": ["rescale_b", 0], "seed": seed,
                            "steps": shape_steps, "cfg": cfg,
                            "sampler_name": "euler", "scheduler": "simple",
                            "positive": ["up", 0], "negative": ["up", 1],
                            "latent_image": ["up", 2], "denoise": 1.0}},

        # ④ 贴图
        "stage_tex": {"class_type": "Trellis2TextureStage",
                      "inputs": {"positive": ["up", 0], "negative": ["up", 1],
                                 "shape_latent": ["s_up", 0]}},
        "s_tex": {"class_type": "KSampler",
                  "inputs": {"model": ["ms_a", 0], "seed": seed,
                             "steps": texture_steps, "cfg": 1.0,
                             "sampler_name": "euler", "scheduler": "normal",
                             "positive": ["stage_tex", 0],
                             "negative": ["stage_tex", 1],
                             "latent_image": ["stage_tex", 2],
                             "denoise": 1.0}},
        "vox_tex": {"class_type": "VaeDecodeTextureTrellis",
                    "inputs": {"samples": ["s_tex", 0], "vae": ["vae_tex", 0],
                               "shape_subdivides": ["mesh_shape", 1]}},
    }

    if not with_texture:
        # 只要几何：直接把形状网格存出去（快，但没有颜色）
        nodes_map["smooth0"] = {"class_type": "MeshSmoothNormals",
                                "inputs": {"mesh": ["mesh_shape", 0],
                                           "crease_angle": 180.0}}
        nodes_map["save"] = {"class_type": "SaveGLB",
                             "inputs": {"mesh": ["smooth0", 0],
                                        "filename_prefix": filename_prefix}}
        return nodes_map

    # ---- 官方模板的网格后处理顺序（一个都不能省，也不能换位置）----
    #
    #   shape 网格
    #     → RemeshMesh(768, solid)   重建均匀、封闭、焊接好的拓扑
    #     → DecimateMesh             精确减到目标面数
    #     → MeshSmoothNormals        平滑法线（否则引擎里是逐面硬边）
    #     → UnwrapMesh               展 UV
    #     → BakeTextureFromVoxel     体素颜色 → 贴图
    #     → BakeAmbientOcclusion / BakeNormalMapFromMesh
    #     → ApplyTextureToMesh       五张图一起挂到网格上
    #     → MeshSmoothNormals → 保存
    #
    # **RemeshMesh 是最容易被跳过、也最不能跳过的一步**：TRELLIS 出的网格
    # 自相交、有双层壳、顶点按 UV 缝拆散。直接 Decimate 会把每条缝当边界塌陷
    # 出碎洞；直接 Unwrap 会让图块互相穿插，烘出来就是大块错色斑。
    # RemeshMesh 用窄带距离场 + Dual Contouring 重建，输出是"焊接好的均匀网格"，
    # 后面减面能精确落到目标面数，UV 也不再打架。
    src_mesh = ["mesh_shape", 0]
    if remesh_resolution > 0:
        nodes_map["remesh"] = {
            "class_type": "RemeshMesh",
            "inputs": {
                "mesh": src_mesh,
                "resolution": remesh_resolution,
                # solid = CelloCut 全局内外判定：去掉双层壳、补小洞，输出单层水密壳。
                # 模板用的就是这个（udf 对脏输入稳但不封闭；sdf 需要一致绕向）。
                #
                # **sub-widgets 用点号前缀当独立键**：DynamicCombo 的值是选项 key
                # （字符串 "solid"），子控件是 `sign_mode.fill` / `sign_mode.qef`。
                # 两种错法都踩过：塞成 dict → `missing 1 required positional
                # argument: 'sign_mode'`；写成裸 `fill` → 子项取到 None，
                # `float(None)` 崩在节点内部。
                "sign_mode": "solid",
                "sign_mode.fill": 20.0, "sign_mode.qef": False,
                "band": 1.0, "project_back": 0.0, "fix_poles": False,
                "smooth_iters": 20, "drop_small_components": 0.01,
                "precluster_max_verts": 20000000}}
        src_mesh = ["remesh", 0]

    if decimate_faces:
        nodes_map["decimate"] = {
            "class_type": "DecimateMesh",
            "inputs": {"mesh": src_mesh,
                       "target_face_count": decimate_faces,
                       "placement_mode": "midpoint"}}
        src_mesh = ["decimate", 0]

    nodes_map["smooth_pre"] = {
        "class_type": "MeshSmoothNormals",
        "inputs": {"mesh": src_mesh, "crease_angle": 180.0}}
    src_mesh = ["smooth_pre", 0]

    nodes_map.update({
        "unwrap": {"class_type": "UnwrapMesh",
                   "inputs": {"mesh": src_mesh, "segmenter": uv_segmenter,
                              "resolution": uv_resolution, "padding": uv_padding,
                              "weld_distance": 0.0002}},
    })

    # reference_mesh：把每个 texel 反投影到**重网格前**的原始密网格再采样。
    # 重网格/减面之后几何已经变粗，拿原始网格做参考能保住细节颜色。
    bake_inputs = {"mesh": ["unwrap", 0], "voxel_colors": ["vox_tex", 0],
                   "texture_size": uv_resolution}
    if bake_reference:
        bake_inputs["reference_mesh"] = ["mesh_shape", 0]
    nodes_map["bake"] = {"class_type": "BakeTextureFromVoxel",
                         "inputs": bake_inputs}

    apply_inputs = {"mesh": ["unwrap", 0], "base_color": ["bake", 0],
                    "metallic": ["bake", 1], "roughness": ["bake", 2]}
    if bake_ao_normal:
        # AO 和法线是**从几何烘的**（不是从体素），所以它们接的是同一个已展 UV 的
        # 网格；模板里还会单独 RemeshMesh 一份喂给它们，我们复用主链的网格即可。
        # 这两个节点都要**两个网格**：low_poly = 已展 UV 的减面网格，
        # high_poly = 它减面之前的密网格（有 remesh 就用 remesh 的输出）。
        # 只给一个网格会报缺参数，而且 AO/法线的意义就是"把高模细节烘进低模"。
        hi = ["remesh", 0] if remesh_resolution > 0 else ["mesh_shape", 0]
        nodes_map["bake_ao"] = {
            "class_type": "BakeAmbientOcclusion",
            "inputs": {"low_poly": ["unwrap", 0], "high_poly": hi,
                       "resolution": uv_resolution, "samples": 32,
                       "max_distance": 0.5, "strength": 1.0, "bias": 0.01}}
        nodes_map["bake_normal"] = {
            "class_type": "BakeNormalMapFromMesh",
            "inputs": {"low_poly": ["unwrap", 0], "high_poly": hi,
                       "resolution": uv_resolution, "cage_distance": 0.05,
                       "ignore_backfaces": True}}
        apply_inputs["occlusion"] = ["bake_ao", 0]
        apply_inputs["normal_map"] = ["bake_normal", 0]

    nodes_map.update({
        "apply": {"class_type": "ApplyTextureToMesh", "inputs": apply_inputs},
        "smooth": {"class_type": "MeshSmoothNormals",
                   "inputs": {"mesh": ["apply", 0], "crease_angle": 180.0}},
        "save": {"class_type": "SaveGLB",
                 "inputs": {"mesh": ["smooth", 0],
                            "filename_prefix": filename_prefix}},
    })
    return nodes_map


class _Noop:
    """什么都能调、什么都返回自己、什么都不做的占位对象。"""

    def __call__(self, *a, **kw):
        return None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self


class _Server:
    """`PromptExecutor` 要的最小 server 接口（见 comfy_execution/server_protocol.py）。

    真实实现是 aiohttp 的 PromptServer；我们只要它不炸，外加能把执行事件
    转成进度回调。

    **还要顶替 `PromptServer.instance` 这个类属性**：不少节点（`UnwrapMesh`、
    `RemeshMesh`…）直接写 `PromptServer.instance.send_progress_text(...)`，
    没有服务端时那里是 AttributeError，而且是在节点执行到一半才炸 ——
    UV 已经展开完了，结果整个 prompt 判失败。
    """

    class _Assets:
        enabled = False

    def __init__(self, on_event=None):
        self.client_id: Optional[str] = None
        self.last_node_id: Optional[str] = None
        self.sockets_metadata: Dict[str, Any] = {}
        self.asset_manager = self._Assets()
        self._on_event = on_event

    def send_sync(self, event, data, sid=None):            # noqa: ANN001
        if self._on_event:
            try:
                self._on_event(event, data)
            except Exception:                             # noqa: BLE001
                pass

    def send_progress_text(self, text, node_id=None, sid=None):   # noqa: ANN001
        self.send_sync("progress_text", {"node_id": node_id, "text": text})

    def queue_updated(self) -> None:
        pass

    def __getattr__(self, name):
        """未实现的成员一律给占位对象。

        ComfyUI 里有些模块（如 `nodes_replacements.py`）会去碰服务端的额外成员
        （`node_replace_manager` 等）。缺了它们只是少一层"节点替换"能力，不该
        让整个 import 失败 —— 否则日志里只剩一行 WARNING，排查时完全看不出
        「有些节点没注册」是这么来的。
        """
        if name.startswith("__"):
            raise AttributeError(name)
        return _Noop()


def _install_server_stub(server: "_Server") -> None:
    """把我们的 stub 挂成 `PromptServer.instance`（节点里到处这么用）。"""
    try:
        import server as comfy_server
        comfy_server.PromptServer.instance = server
    except Exception:                                     # noqa: BLE001
        # 拿不到 PromptServer 就算了：只有少数节点用它，报错信息会带节点名
        pass


# ---------------------------------------------------------------- 执行

def _stage_of(class_type: str) -> str:
    return {
        "Trellis2Conditioning": "图像条件", "VaeDecodeStructureTrellis2": "结构解码",
        "Trellis2ShapeStage": "形状·准备", "VaeDecodeShapeTrellis": "形状·解码",
        "Trellis2UpsampleStage": "上采样·准备", "Trellis2TextureStage": "贴图·准备",
        "VaeDecodeTextureTrellis": "贴图·解码",
        "RemeshMesh": "重建网格", "DecimateMesh": "减面",
        "UnwrapMesh": "UV 展开", "BakeTextureFromVoxel": "烘焙颜色",
        "BakeAmbientOcclusion": "烘焙 AO", "BakeNormalMapFromMesh": "烘焙法线",
        "ApplyTextureToMesh": "贴图应用", "SaveGLB": "保存 GLB",
    }.get(class_type, "")


def _birefnet_mask(im) -> Optional[Any]:
    """用 BiRefNet 出主体 mask（0~255 的 L 图）。没有权重就返回 None。

    为什么非它不可：色键（`matte.matte_auto`）只对**纯色背景**有效。实测一张
    浅灰渐变棚拍图，色键把整张图都当主体（alpha bbox = 整图，fg 48%），
    于是又变回"矩形当物体"。BiRefNet 是语义分割，浅色裙子叠浅色背景也能分开。
    """
    import numpy as np
    from PIL import Image

    path = WEIGHTS_DIR.parent / "comfy" / "background_removal" / "birefnet.safetensors"
    if not path.is_file():
        return None
    try:
        import torch
        from comfy.bg_removal_model import load as load_bg
        cache = _boot.setdefault("_bg", {}) if _boot is not None else {}
        if "model" not in cache:
            cache["model"] = load_bg(str(path))
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(rgb)[None]                    # [1,H,W,3]
        with torch.inference_mode():
            m = cache["model"].encode_image(t)[0]          # [H,W]
        # **必须 detach**：encode_image 没在 inference_mode 里跑时输出带 grad，
        # 直接 `.numpy()` 会 RuntimeError（"Can't call numpy() on Tensor that
        # requires grad"）—— 而这个异常被外层的兜底吞掉，表现为"悄悄退回色键"。
        arr = (m.detach().float().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr, "L")
    except Exception:                                      # noqa: BLE001
        return None


def prepare_image(src: Path, dst: Path, size: int = 1024,
                  pad: float = 1.1, bg_color=(0, 0, 0),
                  matte_backend: str = "birefnet") -> Dict[str, Any]:
    """扣主体 → 按主体包围盒裁剪（留 pad 倍余量）→ 补正方形 → 缩到 size²。

    **这是最容易跳过、后果最严重的一步。** 官方模板里 Conditioning 的输入是
    `ImageCropToMask`（1024×1024、pad_factor 1.1、背景黑色）的输出，不是原图。
    直接喂原图的后果实测过两次：

    · 像素角色（白底不透明，整张都是"物体"）→ 出来是一块**平板**；
    · 真人照片（背景有房间）→ 出来是一个 **1×1×1 的正方体**（4.17M 面）。

    机理：模型把"画面里所有非中性像素"当成物体，矩形画面本身就是个方盒子，
    于是结构体素被填实，后三阶段只是在给这个盒子雕花。

    抠图优先级：输入自带 alpha → BiRefNet（语义分割，能应付渐变背景）→ 色键。
    """
    from PIL import Image
    from . import matte as matte_mod

    im = Image.open(src).convert("RGBA")
    a = im.split()[-1]
    rgba, mode = im, "alpha"
    if a.getextrema()[0] >= 255:
        mask = _birefnet_mask(im) if matte_backend == "birefnet" else None
        if mask is not None:
            # mask 二值化：BiRefNet 输出是软边，0.35 以下算背景（软边留着会被
            # 当物体的一部分，边缘会长出薄壳）
            mask = mask.point(lambda v: 255 if v >= 90 else 0)
            rgba = im.copy()
            rgba.putalpha(mask)
            mode = "birefnet"
        else:
            rgba, mode = matte_mod.matte_auto(im, backend="auto"), "color"

    box = rgba.split()[-1].getbbox()
    if box is None:
        raise TrellisError("这张图扣不出主体（全透明，或与背景太接近）")
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w < 8 or h < 8:
        raise TrellisError(f"主体太小（{w}×{h}）：换一张主体更清楚的图")

    side = int(round(max(w, h) * pad))
    square = Image.new("RGBA", (side, side), (0, 0, 0, 255))
    crop = rgba.crop(box)
    square.alpha_composite(crop, ((side - w) // 2, (side - h) // 2))
    flat = Image.new("RGB", (side, side), tuple(bg_color))
    flat.paste(square, (0, 0), square.split()[-1])
    if side != size:
        flat = flat.resize((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    flat.save(dst, "PNG")
    return {"src": [im.width, im.height], "subject": [w, h],
            "matte": mode, "canvas": size, "bg": list(bg_color)}


def generate(image_path: Path, out_dir: Optional[Path] = None, *,
             seed: int = 42, with_texture: bool = True,
             progress=None, **kw: Any) -> Dict[str, Any]:
    """图片 → GLB。→ {glb, seconds, stages, prompt_id}。

    `progress(stage_text, fraction)` 可选，用来在界面上显示进度。
    """
    boot = bootstrap()
    image_path = Path(image_path)
    if not image_path.is_file():
        raise TrellisError(f"找不到输入图：{image_path}")

    # 输入图必须落在 ComfyUI 的 input 目录里，LoadImage 只认那里的相对路径
    name = f"meow3d_{int(time.time())}_{os.getpid()}.png"
    dst = Path(boot["input"]) / name
    prep = prepare_image(image_path, dst)

    out_dir = Path(out_dir) if out_dir else Path(paths.tool_out("trellis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"meow3d/{out_dir.name}/{image_path.stem}"

    prompt = build_prompt(name, seed=seed, with_texture=with_texture,
                          filename_prefix=prefix, **kw)

    stages: List[Dict[str, Any]] = []
    started = time.time()
    t0 = started
    total = sum(1 for v in prompt.values() if v["class_type"] == "KSampler")

    def on_event(event, data):                            # noqa: ANN001
        nonlocal t0
        if event == "executing" and isinstance(data, dict):
            nid = data.get("node")
            if not nid:
                return
            ct = (prompt.get(nid) or {}).get("class_type", "")
            label = _stage_of(ct)
            if label:
                now = time.time()
                stages.append({"node": ct, "label": label,
                               "seconds": round(now - t0, 1)})
                t0 = now
                if progress:
                    done = sum(1 for s in stages if s["node"] == "KSampler")
                    progress(label, min(0.95, done / max(1, total)))

    ex = boot["execution"]
    server = _Server(on_event)
    _install_server_stub(server)
    # `cache_args` 不能省：`execute_async` 无条件读 `cache_args["ram"]`
    # （只有 cache_type 是 RAM_PRESSURE 时才用，但键必须先存在），
    # 不传就 TypeError: 'NoneType' object is not subscriptable。
    # cache_type=False = 不做跨 prompt 缓存（我们一次一图，缓存没意义）。
    if _accepts_server(ex):
        executor = ex.PromptExecutor(server, cache_type=False,
                                     cache_args={"ram": 0, "ram_inactive": 0})
    else:                                                 # pragma: no cover
        executor = ex.PromptExecutor(cache_type=False,
                                     cache_args={"ram": 0, "ram_inactive": 0})

    prompt_id = f"meow3d-{int(started)}"
    if progress:
        progress("加载权重", 0.02)
    try:
        if _accepts_server(ex):
            executor.execute(prompt, prompt_id, {"client_id": "meow3d"},
                             ["save"])
        else:                                             # pragma: no cover
            executor.execute(prompt, prompt_id, {"client_id": "meow3d"})
    except Exception as exc:                              # noqa: BLE001
        raise TrellisError(f"TRELLIS.2 推理失败：{exc}") from exc

    glb = _find_newest(_output_glb_dir(boot), started)
    if glb is None:
        raise TrellisError(
            "推理跑完了但没找到 GLB 产物 —— 检查 ComfyUI 的输出目录 "
            f"（{boot['output']}）")
    if out_dir not in glb.parents:
        target = out_dir / glb.name
        shutil.move(str(glb), target)
        glb = target

    # 主动卸模型：不卸的话进程退出时 `ModelPatcher.__del__` 会去碰已经清空的
    # `CallbacksMP`（模块全局在 shutdown 阶段被置 None），尾巴上甩一段
    # `'NoneType' object has no attribute 'ON_DETACH'` —— 不影响产物，但很难看。
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
    except Exception:                                     # noqa: BLE001
        pass

    return {"glb": str(glb), "seconds": round(time.time() - started, 1),
            "stages": stages, "prompt_id": prompt_id,
            "prepare": prep, "bytes": glb.stat().st_size}


def _accepts_server(ex) -> bool:
    """新版 PromptExecutor(server) 需要 server；老版是 PromptExecutor()。"""
    import inspect
    try:
        sig = inspect.signature(ex.PromptExecutor.__init__)
        return "server" in sig.parameters
    except (TypeError, ValueError):
        return False


def _output_glb_dir(boot: Dict[str, Any]) -> Path:
    """SaveGLB 写到 output/<prefix 的目录部分>/。prefix 里带了子目录。"""
    return Path(boot["output"])


def _find_newest(root: Path, after: float) -> Optional[Path]:
    best: Optional[Path] = None
    best_t = after - 1.0
    for p in root.rglob("*.glb"):
        try:
            t = p.stat().st_mtime
        except OSError:
            continue
        if t >= best_t:
            best, best_t = p, t
    return best
