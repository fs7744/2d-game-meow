# 3D 模型生成 → Godot 工作流

> 状态：**全链路已跑通并真机验证**（2026-11-03）。图 → GLB → 减面绑骨 → Godot 可加载，
> 单张约 2 分钟（4090）。
> 参考：ComfyUI 官方 TRELLIS.2 教程 + 资料库《AI 生成 3D 模型的 Blender 自动化后处理指南》。

## 1. 目标链路

```
角色图（T-pose 优先）
   │
   ├─① 生成  TRELLIS.2（微软，4B，MIT）→ GLB（高面数 + PBR 贴图）
   │       **在项目进程内跑**：import ComfyUI 的纯 torch 实现（workflows/trellisrun.py）
   │
   ├─② 后处理  Blender 无头脚本：清理 → 减面 → 归一化
   │       可选：Rigify 人形骨架 + 自动权重（绑骨）
   │
   ├─③ 体检   workflows/glb.py：面数/贴图/骨骼/包围盒/外部依赖
   │
   └─④ 验证   tools/verify_glb_godot.py：真 Godot 导入 + 节点检查 + 预览截图
            → Godot 里是 MeshInstance3D / Skeleton3D / AnimationPlayer
```

**一句话上手**：

```python
from workflows import trellisrun
r = trellisrun.generate("角色图.png", "输出目录")   # → {'glb': ..., 'seconds': 116.5}
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
| **TRELLIS.2 权重** | ✅ 8.0GB 已下（魔搭，12 分钟） | 见 4.1 |
| **BiRefNet 抠图权重** | ✅ 424MB（`Comfy-Org/birefnet`） | 预处理必需，见 4.3 |
| Blender | ✅ 4.5.9 已装 | 后处理 + 绑骨 |
| `tools/blender_postprocess.py` | ✅ 实测（含绑骨） | 见 §5 |
| **推理编排 `workflows/trellisrun.py`** | ✅ **全链路跑通** | 单张 ~2 分钟 |
| 项目内节点 / API 端点 | 未做 | 只剩接界面 |

### 4.1 权重下载是唯一的硬阻塞

`Comfy-Org/TRELLIS.2` 的文件与大小（实测 API）：

| 文件 | 大小 |
|---|---|
| `diffusion_models/trellis_2_int8_convrot.safetensors` | **5010 MB** |
| `clip_vision/dino_v3_vit_l.safetensors` | 1156 MB |
| `vae/trellis_2_shape_vae_bf16.safetensors` | 1045 MB |
| `vae/trellis_2_texture_vae_bf16.safetensors` | 905 MB |

（另有 `trellis_2_bf16.safetensors` 9859 MB 的未量化版，24GB 卡上用 int8 就够了。）

**测速实测（2026-11-03）**：

| 源 | 速度 | 8GB 需要 |
|---|---|---|
| hf-mirror.com 直连 | 45 KB/s | ~50 小时 |
| huggingface.co + 代理 7897 | 159 KB/s | ~14 小时 |
| 代理 + 6 并发 | **502 Bad Gateway**（不支持多隧道） | — |
| **ModelScope（魔搭）** | **6.5~8.5 MB/s** | **~20 分钟** |

> **下 HF 模型先查魔搭。** `Comfy-Org/TRELLIS.2` 和 `microsoft/TRELLIS.2-4B`
> 在魔搭都有镜像，直连不用代理，速度快 50 倍。命令：
>
> ```bash
> B="https://modelscope.cn/models/Comfy-Org/TRELLIS.2/resolve/master"
> curl -L -C - -o vae/xxx.safetensors "$B/vae/xxx.safetensors"
> ```
>
> 或者 `pip install modelscope` 后用 `snapshot_download('microsoft/TRELLIS.2-4B')`。
> 魔搭的 `microsoft/TRELLIS.2-4B` 是**官方 bf16 全量版**（16.24GB，9 个 safetensors），
> 和 `Comfy-Org` 那份的 int8 版本**格式不同**，别混用 —— ComfyUI 的实现读的是后者。

### 4.3 还要单独下一个抠图模型（424MB）

`Comfy-Org/birefnet` 的 `background_removal/birefnet.safetensors`，放到
`models/comfy/background_removal/`。**这不是可选项**，理由见 §3.4：
不抠图，图里那块矩形会被模型当成物体本身。

```bash
curl -L -o models/comfy/background_removal/birefnet.safetensors \
  "https://modelscope.cn/models/Comfy-Org/birefnet/resolve/master/background_removal/birefnet.safetensors"
```

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

**已实测通过**（`tools/make_test_human.py` 造 T-pose 人形 → `--rig`）：

```
导入 1 个网格、2184 面 → 减面(ratio 1.0 不减) → UV
元骨架 Rig，159 根骨 → Rigify 生成后 159 根骨
绑定完成，网格上共 159 个顶点组，骨架 159 根骨
RESULT rigged=true tris=2184
Godot：Skeleton3D bones=160，1 个 MeshInstance3D
转上臂骨 70° → 网格跟着弯曲（docs/img/rig-auto-pose.png）
```

**绑骨的前提，不满足就别加 `--rig`**：

- 人形、**T-pose 或 A-pose**（自然站姿的手臂贴身体，热扩散会把手臂权重算给躯干，一动就撕开）；
- 单个角色（多个物体会绑到同一套骨架）；
- 道具 / 场景 / 硬表面**不需要骨骼**，减面 + UV 就结束。

脚本结尾会打印 `RESULT rigged=true/false tris=N` ——
自动权重失败时 `parent_set` 不报错但不产生顶点组，脚本会检查顶点组数并如实返回 false，
不让"以为绑了其实没绑"流到下游。

### 5.1 绑骨这条路上踩的三个坑

| 现象 | 原因 | 修法 |
|---|---|---|
| 导出报 `Object 'WGT-RIG-Rig_eye.L' can't be selected because it is not in View Layer` | 写「选中场景里所有 MESH/ARMATURE」时把 Rigify 的 `WGT-*` 控件网格也选了，它们不在 View Layer | 只选明确要导的主体网格 + 骨架 |
| 转骨骼后 `get_aabb()` 尺寸不变，以为权重没生效 | **`MeshInstance3D.get_aabb()` 返回绑定姿势的包围盒，不随蒙皮变** | 判断权重必须比较**渲染图**（本文用像素差 + 目视） |
| Blender 里 `o.bound_box` 拿到旧值 | `transform_apply()` 后不刷新 | 逐顶点算（见 §5 的 normalize 注释） |

前者和后两个是同一类：**包围盒/尺寸这类"缓存值"在变形之后都不可信**。
验证蒙皮永远看渲染，不看数值。

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

## 7. 装环境的顺序

1. 下 ComfyUI 源码 → `pip install -r requirements.txt`（**独立 venv**，别污染 Qwen 环境）；
2. 跑 `tools/patch_comfy_kitchen.py`（torch 2.6 的 custom-op 注册不认 `list[int]`）；
3. 下模型到 `models/trellis2/`（4 个文件）+ `models/comfy/background_removal/`（BiRefNet）；
4. 装 Blender → `blender_postprocess.py --rig` 绑骨；
5. 接界面（节点 / API 端点）。

第 4 步之前不要凭猜写 prompt：字段错了不报错、节点只用默认值，结果"能跑但不对"，比报错难查。

---

## 8. 2026-11-03 全链路打通：实现要点与踩过的坑

这一节是**实测记录**，每条都有数字。§1~§7 是结论，这里讲结论是怎么来的。

### 8.1 进程内跑通的三个前提

```python
from workflows import trellisrun
r = trellisrun.generate("角色图.png", "输出目录")     # 单张 ~2 分钟
```

内部三步（`trellisrun.bootstrap()`，幂等）：

1. **模型目录必须在 `import nodes` 之前指过去**：`comfy.cli_args.args.models_directory`。
   `folder_paths` 在 import 那一刻就把路径算死了，之后再改没用。
2. **权重用硬链接**放进 `models/comfy/{diffusion_models,vae,clip_vision}`：
   同卷硬链接瞬间完成、不占空间，8GB 没必要复制一份。
3. **用 ComfyUI 的 `PromptExecutor` 跑 API 格式的图**，而不是自己逐个调节点：
   显存调度、缓存、依赖顺序它都管；自己调会在形状段就 OOM（实测 37GB > 24GB，
   因为中间张量全留在显存里）。

### 8.2 进程内执行的四个坑

| 现象 | 原因 | 修法 |
|---|---|---|
| `TypeError: 'NoneType' object is not subscriptable` | `execute_async` 无条件读 `cache_args["ram"]`，即使缓存类型是 NONE | 传 `cache_type=False, cache_args={"ram":0,"ram_inactive":0}` |
| `AttributeError: PromptServer has no attribute 'instance'` | 多个节点（`UnwrapMesh`、`RemeshMesh`…）直接 `PromptServer.instance.send_progress_text(...)` | 自己实现一个最小 server 并挂成 `PromptServer.instance` |
| 日志里一行 `IMPORT FAILED: nodes_replacements.py` | 节点注册**早于**我们挂 stub server | stub 要在 `init_extra_nodes()` **之前**装；并给它加 `__getattr__` 兜住未实现的成员 |
| 进程退出时 `'NoneType' object has no attribute 'ON_DETACH'` | `ModelPatcher.__del__` 在解释器 shutdown 阶段去碰已清空的模块全局 | 结束时 `comfy.model_management.unload_all_models()`；不影响产物 |

### 8.3 预处理决定成败（这一条最贵，别跳过）

官方模板里 `Trellis2Conditioning` 的输入是 `ImageCropToMask`（1024²、`pad_factor=1.1`、
背景黑色）的输出，**不是原图**。直接喂原图，实测两次：

| 输入 | 结果 | 面数 |
|---|---|---|
| 像素角色（白底、整张不透明，486×673） | 一块**平板**（包围盒 z=0.0005） | 78 万 |
| 真人照片（背景是房间） | 一个 **1×1×1 正方体** | 417 万 |

机理：模型把"画面里所有非中性像素"当成物体，矩形画面本身就是个方盒子 ——
结构体素被填实，后三阶段只是给这个盒子雕花。

正确预处理（`trellisrun.prepare_image`）：**抠主体 → 按主体包围盒裁 → 补正方形
（留 1.1 倍余量）→ 缩到 1024²**。抠图优先级：

1. 输入自带 alpha → 直接用（最准）；
2. **BiRefNet**（424MB，ComfyUI 的 `RemoveBackground` 用的就是它）→ 渐变背景也能分开；
3. 色键（`matte.matte_auto`）→ 只对纯色背景有效。

> 实测一张浅灰渐变棚拍图，色键把整张图都判成主体（alpha bbox = 整图、fg 48%），
> 于是又变回"矩形当物体"。所以 BiRefNet 不是可选项。

同一个坑还有一次变体：`_birefnet_mask` 里 `.numpy()` 报
"Can't call numpy() on Tensor that requires grad"，被外层 `except` 兜住 →
**悄悄退回色键**，表现就是"明明装了 BiRefNet 却还是平板"。少写一句 `detach()` 的代价。

### 8.4 网格后处理的顺序（官方模板，见 §9）

正确顺序是 **Remesh → Decimate → SmoothNormals → Unwrap → Bake**，我最初漏了第一步
`RemeshMesh`，只用 Decimate + Unwrap，于是贴图出现大块错色斑。三种做法的实测：

| 做法 | 结果 |
|---|---|
| 只 Decimate + Unwrap（漏了 Remesh） | 20 万面，表面连续但有**大块错色斑**（UV 图块互相穿插） |
| 生成后在 Blender 里减面（不 remesh） | **表面全是碎洞**：TRELLIS 顶点按 UV 缝拆过，Decimate 把每条缝当边界塌陷 |
| Blender 先焊点再减面 | 焊点制造非流形边（11.6 万条边被 3 个面共用），Decimate **锁死在 69 万面** |
| 官方顺序：RemeshMesh(solid) → Decimate → Unwrap | 见 §9 的实测 |

### 8.5 绑骨：热扩散在 AI 网格上必然失败

`parent_set(type='ARMATURE_AUTO')` 在 TRELLIS 产物上给出 **0 个有权重的顶点**
（81,517 个顶点全部为 0），日志只有一句 `Bone Heat Weighting: failed to find
solution for one or more bones`。原因：热扩散要求闭合流形体积，而 AI 网格是
一堆互相穿插的独立壳（裙摆、纱袖、头发各是独立面片）。连"焊好点的副本"也是 0。

| 方案 | 覆盖率 | 结果 |
|---|---|---|
| `ARMATURE_AUTO`（热扩散） | 0 / 81,517 | 顶点组建好了但全是 0 权重 |
| `ARMATURE_ENVELOPE`（包络） | 79,565 / 81,517 | 可用，质量粗 |

`add_human_rig(mode="auto")` 现在会先试热扩散，覆盖率 < 50% 自动回退 envelope，
并在日志里写清为什么。

**判"绑骨成功"只能看有权重的顶点数。** 早期只看 `len(vertex_groups)`：`parent_set`
失败时照样建出 159 个空组，于是"绑骨成功"，而 glTF 导出器发现没有有效权重 →
**根本不写 skin** → Godot 里是个静态 `MeshInstance3D`（`VERIFY_SKELETONS 0`）。
查了三轮才定位到权重全 0。

### 8.6 一次完整运行的实测量

输入：1000×1143 的 T-pose 参考图（浅灰渐变背景、白蓝汉服）

| 段 | 耗时 |
|---|---|
| 图像条件 | 1.1 s |
| 结构解码 | 10.6 s |
| 形状 512 | 5.4 s |
| 上采样 1536 准备 + 采样 | 0.6 s |
| **贴图段采样** | **46.2 s** |
| 贴图 VAE 解码 | 24.5 s |
| UV 展开（20 万面） | 1.2 s |
| 烘焙 4096² 贴图 | 1.5 s |
| 贴图应用 + 保存 | 10.2 s |
| **合计** | **~125 s** |

产物：188,182 面 / 4096 贴图 / 18MB GLB（减面到 20 万）；
不减面是 1.77M 面 / 73MB。

Blender 后处理（减面已在上游做过，这里只归一化 + 绑骨）：约 40 s，
输出 70k~20 万面 + 159 骨 + envelope 权重，9~19MB。

### 8.7 画质现状与调参

现状：**形状、姿态、配色都对**，贴图有块状错色（尤其裙摆下缘），棱面可见。

| 旋钮 | 影响 |
|---|---|
| `decimate_faces` | 8 万（快、棱面明显）→ 20 万（明显更好）→ 0（1.77M，最好但 73MB） |
| `uv_resolution` | 2048 / 4096；4096 对花纹更清楚，但错色斑也更明显 |
| `bake_reference` | 默认关。开着会用未减面网格反投影采样，理论上更准，**实测在 4096 图谱下投偏** |

要再上一档的话，正确的下一步是：**保留 1.77M 高模 → Blender 里重拓扑到目标面数
→ 用 Cycles 从高模烘 normal + base_color**。现在的做法是"减面后再烘"，亏在
UV 和几何都不够细。

---

## 9. 官方模板逐节点拆解

模板：`Comfy-Org/workflow_templates` 的 `3d_pixal3d_trellis2_image_to_model.json`
（66 节点 / 45 种类型）。它其实有**两条互斥路线**，由一个布尔开关选：
TRELLIS.2 原生（本文用的）和 Pixal3D（`UNETLoader pixal3d_int8` + MoGe 估 FOV）。
下面只讲 TRELLIS.2 这条。

### 9.1 输入预处理

```
LoadImage
  └─ RemoveBackground(LoadBackgroundRemovalModel = birefnet) → mask
       └─ ComfySwitchNode(默认 true：用 birefnet 的 mask；false：用图片自带 alpha)
            └─ ImageCropToMask(width 1024, height 1024, pad_factor 1.1,
                               grow_mask 0, background #000000) → 预处理图
```

**这一步不是可选的**：`pad_factor 1.1` + 黑底 + 扣主体，直接决定后面出来的是人还是盒子
（§8.3 的两次失败）。

### 9.2 条件编码

```
CLIPVisionLoader(dino_v3_L_naf_fp32)
预处理图 ──► Trellis2Conditioning ──► positive / negative
```

`positive` = DINOv3 特征（`global_512` 做 cross-attn 条件，(1,1029,1024)；
`global_1024` 放进 `embeds`，(1,4101,1024)）；`negative` = 同形状全零。

### 9.3 结构段 → 32³ 占据体素

```
EmptyTrellis2LatentStructure(batch 1)                     latent (1,32,16,16,16)
UNETLoader(trellis_2_int8_convrot)
  → CFGOverride(cfg 1.0, 0.667→1.0) → RescaleCFG(0.7) → ModelSamplingSD3(shift 5)
KSampler(steps 12, cfg 7.5, euler, normal, denoise 1)
VaeDecodeStructureTrellis2(vae=shape_vae, resolution "32") → VOXEL
```

### 9.4 形状段 → 稀疏 latent

```
Trellis2ShapeStage(positive, negative, voxel)
```

它做三件事：从体素里取占据坐标 `coords (N,4)`；按 `coord_resolution`
（这里 32 ≤ 32 → `shape_generation_512`）；把 `generation_mode / coords /
coord_counts` 塞进 conditioning —— **模型在采样时靠这几个键知道自己在哪一段**。

```
KSampler(steps 20, cfg 7.5, euler, normal)        ← 另一条模型链：
  UNET → CFGOverride(cfg 1.0, 0.769→1.0) → RescaleCFG(0.5)     （注意不是 0.7）
VaeDecodeShapeTrellis(vae=shape_vae) → MESH + SHAPE_SUBDIVIDES
```

`SHAPE_SUBDIVIDES` 要留给贴图段的 VAE 解码用（不是给人看的）。

### 9.5 上采样段 → 1024 级

```
Trellis2UpsampleStage(positive, negative, shape_latent, vae=shape_vae,
                      target_resolution 1536)
KSampler(steps 12, cfg 7.5, euler, **simple**)     ← 调度器换成 simple
```

### 9.6 贴图段

```
Trellis2TextureStage(positive, negative, shape_latent)
KSampler(steps 12, **cfg 1.0**, euler, normal)     ← 贴图段不做 CFG
VaeDecodeTextureTrellis(vae=texture_vae, shape_subdivides) → 带颜色的 VOXEL
```

### 9.7 网格后处理（我之前漏了第一步）

```
MESH(shape)
  → GetMeshInfo                                  只是打印面数
  → RemeshMesh(resolution 768, sign_mode "solid", band 1.0, project_back 0.0,
               fix_poles False, smooth_iters 20, drop_small_components 0.01,
               precluster_max_verts 20_000_000)
  → DecimateMesh(target_face_count 700_000, placement_mode "midpoint")
  → MeshSmoothNormals(180)
  → UnwrapMesh(segmenter "pec", resolution 2048, padding 1, weld_distance 0.0002)
  → BakeTextureFromVoxel(mesh=unwrap, voxel_colors=贴图体素,
                         resolution 2048, reference_mesh=MESH(shape))
  → BakeAmbientOcclusion(low_poly=unwrap, high_poly=remesh输出, 2048)
  → BakeNormalMapFromMesh(low_poly=unwrap, high_poly=remesh输出, 2048, cage 0.05)
  → ApplyTextureToMesh(mesh=unwrap, base_color, metallic, roughness,
                       occlusion, normal_map)
  → MeshSmoothNormals(180)
  → MeshToFile3D → Save3DAdvanced(1024×1024)
```

三个容易看漏的点：

1. **`RemeshMesh` 不能省**。它的官方描述就是"重建均匀三角化网格，规范化乱/非流形/
   自相交拓扑，用在 Decimate 之前以精确达到目标面数，输出保持焊接"。
   `sign_mode="solid"` 是 CelloCut 全局内外判定：**去掉双层壳、补小洞**，输出单层水密壳。
   跳过它的直接后果：UV 图块互相穿插 → 烘出大块错色斑（§8.4）。
2. **Decimate 的 70 万面不是"游戏面数"**，是模板给可编辑性留的上限；要上移动端
   自己往下调（我们用 8~20 万）。
3. **AO 和法线是"从几何烘"的**（low_poly=已展 UV 的减面网格，high_poly=remesh 后的密网格），
   不是从体素烘 —— 所以 `ApplyTextureToMesh` 收五张图：base_color / metallic /
   roughness / occlusion / normal_map。

**实测（2026-11-03，参考图 1000×1143 T-pose 汉服）**：

| 指标 | 漏 Remesh（Decimate+Unwrap） | 官方完整链 |
|---|---|---|
| 面数 | 188,182 | 199,992 |
| 贴图 | 4096，**大块错色斑** | 2048，干净 |
| 表面 | 连续但有斑 | 连续平滑、无碎洞 |
| Godot 预览 | — | `docs/img/trellis2-godot-preview.png` |
| 耗时 | 125 s | 150 s |

多花 25 秒（Remesh ~30 s + AO/法线烘焙），换来的是"能用"和"不能用"的差别。

### 9.8 和我们的实现对照

| 官方 | 我们的 `trellisrun.build_prompt` | 说明 |
|---|---|---|
| `ImageCropToMask`（节点） | `prepare_image()`（Python） | 用 alpha/BiRefNet 求包围盒，效果等价、少跑一个节点 |
| 条件 / 四段采样 | 完全一致 | 参数逐项照抄（含两套 CFGOverride/RescaleCFG） |
| `RemeshMesh` | `remesh_resolution=768` | 参数照抄 |
| `DecimateMesh(700k)` | `decimate_faces=80000` | 我们直接落到游戏面数 |
| `UnwrapMesh(padding 1)` | `uv_padding=4` | 4 更不容易串色 |
| AO / 法线 | `bake_ao_normal=True` | 照抄（samples 32，省点时间） |
| `Save3DAdvanced` | `SaveGLB` | 不需要视口截图 |
| Pixal3D 分支 | 未做 | 另一条路线，需要 `pixal3d_int8` 权重 + MoGe |
