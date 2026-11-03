# 3D 模型生成 → Godot 工作流

> 状态：**验证层已完成并真机跑通；生成层在装环境**（2026-11-03）。
> 参考：ComfyUI 官方 TRELLIS.2 教程 + 资料库《AI 生成 3D 模型的 Blender 自动化后处理指南》。

## 1. 目标链路

```
角色图（T-pose 优先）
   │
   ├─① 生成  TRELLIS.2（微软，4B，MIT）→ GLB（高面数 + PBR 贴图）
   │       走外部 ComfyUI（本项目不内嵌，理由见 §2）
   │
   ├─② 后处理  Blender 无头脚本：清理 → 减面 → Smart UV → 归一化
   │       可选：Rigify 人形骨架 + 自动权重（绑骨）
   │
   ├─③ 体检   workflows/glb.py：面数/贴图/骨骼/包围盒/外部依赖
   │
   └─④ 验证   tools/verify_glb_godot.py：真 Godot 导入 + 节点检查 + 预览截图
            → Godot 里是 MeshInstance3D / Skeleton3D / AnimationPlayer
```

## 2. 怎么在项目里跑（不走 ComfyUI 服务）

### 2.1 官方仓库为什么不能直接用

TRELLIS.2 官方仓库（`microsoft/TRELLIS.2`，**MIT**）自己说了两件事，都卡人：

| 官方原话 | 影响 |
|---|---|
| "The code is currently tested only on **Linux**" | Windows 上要自己趟 |
| 依赖 `flash-attn` / `nvdiffrast` / `nvdiffrec` / `cumesh` / `o-voxel` / `flexgemm` | 多数要编译 CUDA 扩展 |

实测确认（2026-11-03）：

```
where nvcc                    → 无
CUDA Toolkit                  → 未安装（只有驱动，CUDA UMD 13.5）
pip index versions trellis2   → No matching distribution
pip index versions o-voxel    → No matching distribution
pip index versions flexgemm   → No matching distribution
pip index versions nvdiffrast → No matching distribution
```

**结论：pip 装不上，源码编译要先装 CUDA Toolkit 12.4 + MSVC，而且这些包作者只在 Linux 测过。**

### 2.2 可行路径：把 ComfyUI 的纯 torch 实现 import 进项目

ComfyUI 把 TRELLIS.2 做成了核心节点，但**它没有用微软那套 CUDA 扩展**——
自己用 PyTorch 重写了 VAE、DiT 和稀疏卷积：

```
comfy/ldm/trellis2/vae.py       1110 行   SparseTensor / SparseLinear
comfy/ldm/trellis2/model.py     1163 行   DiT + 三阶段投影
comfy/ldm/trellis2/flexgemm.py   212 行   纯 torch 的稀疏子流形卷积
comfy_extras/nodes_trellis2.py   893 行   各阶段的节点封装
```

依赖全是 `comfy.ldm.*`（ComfyUI 自己的模型基础设施）+ torch，**没有编译期依赖**。
所以可以：把 ComfyUI 源码 vendor 进项目 → `import comfy.ldm.trellis2.*` →
自己写推理编排。**不需要启动服务，不需要 ComfyUI 界面，用户无感知。**

实测（本机 Windows / Python 3.13 / torch 2.6.0+cu124）：

```
from comfy.ldm.trellis2.vae import SparseTensor          → OK
from comfy.ldm.trellis2.model import build_proj_transform_matrix → OK
from comfy.ldm.trellis2.flexgemm import sparse_submanifold_conv3d → OK
```

### 2.3 一个必须打的补丁（`tools/patch_comfy_kitchen.py`）

直接 import 会炸在 `comfy_kitchen`（ComfyUI 的量化算子包，int8 模型要用）：

```
ValueError: infer_schema(func): Parameter stride has unsupported type list[int]
```

原因：它注册 torch custom op 时用了 PEP 585 的 `list[int]`，而 **torch 2.6 的
`infer_schema` 只认 `typing.List[int]`**（torch 2.7+ 才支持内建泛型）。
报错发生在 import 阶段，整条链路起不来。

修法：把注解换成 `typing.List[...]`（语义等价，不动算子逻辑）。包内 16 个文件、
共 40 处，`tools/patch_comfy_kitchen.py` 一键完成，**幂等**。
升级 `comfy_kitchen` 后要重跑。

**为什么不直接升 torch**：升到 2.7+ 会牵动 Qwen 产线（本项目 2D 生成的主力），
先在注解层解决更安全。

### 2.4 显存必须串行

4090 24GB 上 TRELLIS.2 要 16~24GB，和 Qwen 不能共存。
3D 生成前必须 `engine.unload()`，跑完再装回来——**这段时间 2D 产线是停的**，
做节点时要按串行设计，不能假设两个能并行。

## 3. 已实测通过的部分

### 3.1 GLB 体检（纯 Python，0.5 秒，不依赖引擎）

`workflows/glb.py` 直接解析 GLB 的两块 chunk，给「能不能进 Godot」的判断：

```
== soldier.glb  2110 KB ==
  11,376 面 · 2 材质 · 贴图 1024px · 51 骨 · 4 个动画
  包围盒 [184.7946, 44.3863, 183.2203]
  级别 ok
```

判据（`BUDGET`）按手机 / PC 两档给**建议**而非错误：面数、贴图尺寸、非三角图元、
外部文件引用、有骨头无动画、无骨骼。

`normalize()` 只改根节点 scale/translation（不动顶点），把 TRELLIS 的
`[-0.5,0.5]³` 归一化空间换算成"身高 1.8 米、脚底落地、水平居中"。

### 3.2 Godot 真机验证

`tools/verify_glb_godot.py` 建临时工程 → `--import` → 加载 glb → 读节点属性：

```
VERIFY_MESHES 2
VERIFY_MESH vanguard_Mesh surfaces=1 verts=7325 mat=1
VERIFY_SKELETONS 1
VERIFY_SKELETON Skeleton3D bones=49
VERIFY_PLAYERS 1
VERIFY_ANIMS ["Idle", "Run", "TPose", "Walk"]
VERIFY_ANIM Walk len=1.033 tracks=88
结论：Godot 能加载
```

预览图：`tests/_out/glbcheck/soldier/preview.png`（T-pose 士兵，PBR 贴图完整）。

**这一步是必需的，不是锦上添花**：GLB 文本层面合法不代表 Godot 能用。
对应的静默失败包括 `KHR_texture_basisu`（模型变纯白）、webp 贴图在 4.3 以下被丢、
非三角图元被忽略、Skeleton3D 存在但没有动画轨道。

### 3.3 Godot 截图的两个坑（都踩了）

| 现象 | 原因 |
|---|---|
| 截图全白/纯背景色 | `cam.look_at()` 在 `_init()` 里报 `Node not inside tree`，静默不转 → 相机朝 -Z。改用 `look_at_from_position()` |
| 截图全白（第二次） | 算 AABB 时节点刚 `add_child` 还没进树，`global_transform` 返回 Identity → 相机对空气。先 `await process_frame` |

另外 `cam.make_current()` 必须有（手动挂的相机不会自动成为当前相机）。

## 4. 尚未完成（依赖环境安装）

| 项 | 状态 | 说明 |
|---|---|---|
| ComfyUI 源码 | ✅ 已下（13MB）+ 依赖已装进项目 venv | 只当**库**用，不起服务 |
| `comfy_kitchen` 注解补丁 | ✅ 已做（`tools/patch_comfy_kitchen.py`） | 幂等，升版本后要重跑 |
| `comfy.ldm.trellis2.*` import | ✅ 实测通过 | 纯 torch，无编译依赖 |
| **TRELLIS.2 权重** | ⏳ **下载中（8GB）** | **这是唯一硬阻塞**，见 4.1 |
| Blender | ⏳ 未装（350MB） | 后处理要用 |
| `tools/blender_postprocess.py` | ⚠️ 已写，**未实测** | 等 Blender |
| 推理编排（把三个阶段串起来） | 未写 | 等权重下完才能调 |
| 项目内节点 / API 端点 | 未做 | 等生成层跑通 |

### 4.1 权重下载是唯一的硬阻塞

`Comfy-Org/TRELLIS.2` 的文件与大小（实测 API）：

| 文件 | 大小 |
|---|---|
| `diffusion_models/trellis_2_int8_convrot.safetensors` | **5010 MB** |
| `clip_vision/dino_v3_vit_l.safetensors` | 1156 MB |
| `vae/trellis_2_shape_vae_bf16.safetensors` | 1045 MB |
| `vae/trellis_2_texture_vae_bf16.safetensors` | 905 MB |

（另有 `trellis_2_bf16.safetensors` 9859 MB 的未量化版，24GB 卡上用 int8 就够了。
背景移除 `birefnet` 不在这个仓库，需要另找。）

**测速实测（2026-11-03 21:25）**：

| 源 | 速度 | 8GB 需要 |
|---|---|---|
| hf-mirror.com 直连 | 45 KB/s | ~50 小时 |
| huggingface.co + 代理 7897 | 159 KB/s | ~14 小时 |
| 代理 + 6 并发 | **502 Bad Gateway**（代理不支持多隧道） | — |

所以：**模型必须由用户用更快的线路下**，或者挂着慢慢下。
纯代码部分（推理编排、后处理、验证）不依赖权重，可以先写完。

### 4.2 已确认的模型参数（从官方工作流模板读出）

| 节点 | 值 |
|---|---|
| UNETLoader | `trellis_2_int8_convrot.safetensors` |
| VAELoader ×2 | `trellis_2_shape_vae_bf16` / `trellis_2_texture_vae_bf16` |
| CLIPVisionLoader | `dino_v3_L_naf_fp32`（仓库里是 `dino_v3_vit_l`） |
| KSampler（shape） | steps 12 / cfg 7.5 / euler / normal |
| Trellis2UpsampleStage | 1536 |
| Save3DAdvanced | 1024 × 1024 |

## 5. Blender 后处理脚本（`tools/blender_postprocess.py`）

```bash
blender --background --python tools/blender_postprocess.py -- \
    --in model.glb --out model_game.glb --ratio 0.15 --height 1.8 --rig
```

阶段与顺序（**顺序不能反**）：

1. `cleanup`：merge by distance（阈值 0.0001）+ 删游离面 + 重算法线；
2. `normalize`：缩放到目标身高、水平居中、底面 z=0，然后 **apply transform**
   （不 apply 的话导出后节点带缩放，引擎里再叠动画容易出问题）；
3. `decimate`：Collapse 模式 + `use_collapse_triangulate`（引擎只吃三角面）；
4. `unwrap`：Smart UV Project（减面之后再展，否则 UV 会扭曲）；
5. `add_human_rig`（可选）：Rigify 人形元骨架 + `ARMATURE_AUTO` 自动权重。

**绑骨的前提，不满足就别加 `--rig`**：

- 人形、**T-pose 或 A-pose**（自然站姿的手臂贴身体，热扩散会把手臂权重算给躯干，一动就撕开）；
- 单个角色（多个物体会绑到同一套骨架）；
- 道具 / 场景 / 硬表面**不需要骨骼**，减面 + UV 就结束。

脚本结尾会打印 `RESULT rigged=true/false tris=N` ——
自动权重失败时 `parent_set` 不报错但不产生顶点组，脚本会检查顶点组数并如实返回 false，
不让"以为绑了其实没绑"流到下游。

## 6. 复现命令

```bash
# 体检 + Godot 验证 + 预览（需要 Godot，本机已有 4.7.2）
.venv/Scripts/python.exe tools/verify_glb_godot.py 模型.glb --shot

# 先规范化到 1.8 米再验
.venv/Scripts/python.exe tools/verify_glb_godot.py 模型.glb --normalize 1.8 --shot

# 单测（11 条，0.5 秒，纯构造样本）
.venv/Scripts/python.exe -m pytest tests/test_glb.py -q
```

Godot 位置：`D:\GAMES_tool\Godot_v4.7.2-stable_mono_win64\..._console.exe`（**console 版**才有 stdout）。

## 7. 装环境的顺序（下次接着做）

1. 解压 ComfyUI 源码 → `pip install -r requirements.txt`（**独立 venv**，别污染 Qwen 环境）；
2. 起服务 `python main.py --listen 127.0.0.1 --port 8188`；
3. 查 `/object_info`，确认 `Trellis2ShapeStage` 等节点存在（`threed.trellis_available()`）；
4. 按 `node_signature()` 的字段写 `generate()` 的 prompt；
5. 下模型（`models/diffusion_models` / `vae` / `clip_vision` / `background_removal`）；
6. 装 Blender → 跑 `blender_postprocess.py` 实测减面与绑骨；
7. 项目内加节点（输入角色图 → 生成 → 后处理 → 导出 Godot 工程）。

第 4 步之前不要写 prompt：字段靠猜会得到"能跑但结果不对"的工作流，比报错更难查。
