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

## 2. 为什么走外部 ComfyUI

TRELLIS.2 官方仓库（`microsoft/TRELLIS.2`，**MIT**）自己说了两件事，都卡人：

| 官方原话 | 影响 |
|---|---|
| "The code is currently tested only on **Linux**" | Windows 上要自己趟 |
| 依赖 `flash-attn` / `nvdiffrast` / `nvdiffrec` / `cumesh` / `o-voxel` / `flexgemm` | 多数要编译 CUDA 扩展 |

而 ComfyUI 把 TRELLIS.2 做成了**核心 3D 节点**（不是社区节点），Windows 轮子由它的生态解决。
所以：**ComfyUI 当生成后端，本项目当编排 + 后处理 + 验证**。

代价：多一个进程、一套 10~12GB 模型、**显存要串行**——
4090 24GB 上 TRELLIS.2 要 16~24GB，和 Qwen 不能共存，调用前必须先 `engine.unload()`。

配套：`workflows/threed.py` 是 HTTP 客户端（上传/提交/轮询/下载 + 能力探测）。

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

| 项 | 状态 | 阻塞 |
|---|---|---|
| ComfyUI 本体 | 下载中（GitHub codeload，约 30MB） | 网络慢 |
| TRELLIS.2 模型 | 未下载（10~12GB） | 走 hf-mirror |
| Blender | 未装（350MB，阿里云镜像可下） | 待下 |
| `tools/blender_postprocess.py` | **已写，未实测** | 等 Blender |
| TRELLIS.2 的 API prompt | **未写** | 必须对着 `/object_info` 写，不能猜 |
| 项目内节点 / API 端点 | 未做 | 等生成层跑通 |

### 4.1 为什么 prompt 不能凭猜写

ComfyUI 的 `/prompt` 要的是 `{node_id: {class_type, inputs}}`，字段名必须和节点定义
完全一致。写错的后果**不是报错**，而是节点用默认值——生成出莫名其妙的结果，
排查时根本不知道是自己填错了。

官方模板（`3d_pixal3d_trellis2_image_to_model`）是 66 个节点、45 种类型的**UI 格式**，
要靠 `/object_info` 的 `input_order` 才能准确转成 API 格式。
`workflows/threed.py::node_signature()` 就是干这个的探查工具。

已确认的关键参数（从模板读出）：

| 节点 | 值 |
|---|---|
| UNETLoader | `trellis_2_int8_convrot.safetensors` |
| VAELoader ×2 | `trellis_2_shape_vae_bf16` / `trellis_2_texture_vae_bf16` |
| CLIPVisionLoader | `dino_v3_L_naf_fp32.safetensors` |
| KSampler（shape） | steps 12 / cfg 7.5 / euler / normal |
| Trellis2UpsampleStage | 1536 |
| Save3DAdvanced | `3d/ComfyUI`，1024×1024 |

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
