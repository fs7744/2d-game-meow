# 2D Game Asset Pipeline（SD1.5 本地版）

单卡 RTX 4090 上的本地 2D 游戏资产生成工作台。**全栈 SD1.5**：文生图、图生图、局部重绘、
姿态驱动→精灵帧、精灵图打包。代码化推理（diffusers，不依赖 ComfyUI）+ FastAPI 服务。

设计基线来自资料库文档 `RTX4090_2D_Game_Pipeline` v1.1；本仓库把其中的 SDXL 部分收敛为 SD1.5，
原因与实测结论见文末「版本选型」。

---

## 现状

| 模块 | 状态 |
|---|---|
| 引擎层（懒加载 / 显存守卫 / LoRA 热切换） | 已完成，实测通过 |
| W1 文生图 / W2 图生图 | 已完成，实测通过 |
| W3 局部重绘 | 已完成，实测通过 |
| W4 姿态驱动 → 逐帧对齐 → 精灵图打包 | 已完成，实测通过（8 帧走路序列） |
| 后处理（像素化 / 16 色量化 / 锚点对齐 / 打包） | 已完成并自测（28px 漂移 → 0） |
| IP-Adapter 角色身份锁定 | 已接通，效果待调（见「实测」） |
| FastAPI 服务（单卡串行队列 + 资产索引） | 已完成 |
| 抠图 / 去白边（纯色连通域 + 迭代去白边） | 已完成，实测通过 |
| 游戏动效导出（GIF/APNG + Godot 4 + 通用 JSON） | 已完成，实测通过 |
| LoRA 选装（Civitai 精选 + 注册表 + 节点下拉） | 已完成，实测通过 |
| **LoRA 训练**（自包含训练循环 + CLI + 面板） | 已完成，实测通过 |
| 姿态提取（YOLOX 检测 + RTMPose 回归 → OpenPose 骨架） | 已完成，实测通过 |
| W5 视频驱动（Wan2.2 TI2V-5B，替代 MimicMotion） | 已完成，实测通过 |

## 实测数据

稳态耗时（`python tools/bench.py`，每次跑前预热，排除模型加载）。RTX 4090 24GB，fp16，DPM++ 2M Karras。

| 用例 | 耗时 | 峰值显存 |
|---|---|---|
| 冷启动加载 SD1.5 | 1.59s | — |
| W1 文生图 512² / 25 步 | **0.70s** | 2.61GB |
| W1 文生图 512×768 / 25 步 | **0.99s** | 2.91GB |
| W1 文生图 768² / 25 步 | **1.58s** | 3.36GB |
| W1 文生图 512² / 8 步（草稿档） | **0.30s** | 2.61GB |
| W2 图生图 512² / strength 0.45 | **0.44s** | 2.61GB |
| W2 图生图 512² / strength 0.65 | **0.50s** | 2.61GB |
| W3 局部重绘 512² / 25 步 | **0.75s** | 4.64GB |
| W4 姿态驱动 1 帧 512² | **1.17s** | 7.36GB |
| W4 姿态驱动 8 帧 512² | **8.07s（1.01s/帧）** | 7.36GB |
| W4 8 帧 + IP-Adapter 0.85 | **9.71s（1.21s/帧）** | 10.59GB |

两点说明：

1. **单工作流峰值只有 2.6–3.6GB**；表里 7.36GB 是同一个 Engine 里 txt2img 与 ControlNet 两套
   pipeline 同时常驻的结果（省掉反复加载，代价是多占 4GB）。
2. **产线估算**：一个角色的走/跑/待机/攻击四套动作（各 8 帧）≈ **32 秒**；
   一个 20 角色的关卡 ≈ 11 分钟。瓶颈不在推理，在挑帧与修瑕疵。

## 姿态提取（W4 的入口）

```
图片 / GIF / 视频 / 精灵图
        ↓  YOLOX-L（onnx, 640×640，person 类）
   人体检测框 → 扩成正方形 ×1.25 → 缩到 288×384
        ↓  RTMPose（dw-ll_ucoco_384.onnx, SimCC 头）
   17 个 COCO 关键点 → 补 neck → 重排为 OpenPose-18 → 按官方配色渲染
        ↓
   ControlNet OpenPose 条件图
```

```bash
# 单图 / 精灵图（按网格切）/ 视频（每 3 帧取一帧）
python tools/pose_from_image.py sprite.png          -o assets/pose
python tools/pose_from_image.py sheet.png --grid 8x1 -o assets/pose
python tools/pose_from_image.py walk.mp4  --step 3   -o assets/pose
```

两个实现细节值得记：

1. **不用 controlnet_aux 的 DwposeDetector**：它依赖 mmpose + mmdet + mmcv（Windows 编译极痛），
   而且要 .pth 不是 onnx。这里直接用 DWPose 官方导出的两个 onnx 自己串链路。
2. **检测器可换**：`--detector auto|yolox|torchvision|none`。`auto` 优先 YOLOX，
   权重不在时退到 torchvision FasterRCNN（权重来自 download.pytorch.org，国内可达），
   两者都没有则整图回归——实测整图模式对人形精灵也能出 17/18 个关节，只是比例偏差更大。

实测（CPU，512² 输入）：

| 模式 | 单帧耗时 | 可见关节 |
|---|---|---|
| 整图回归（无检测） | 26ms | 17/18 |
| YOLOX 检测 + 裁剪 | 515ms | 17/18 |

检测的价值在**构图不规整**的输入（人物偏在一角、多人、画面留白多）；精灵图本来就占满画面时，
整图模式又快又够用。

## LoRA：装什么、怎么用

`models/lora/*.safetensors` + 同目录 `loras.json`（标签 / 触发词 / 推荐强度）。
启动后 `GET /loras` 直接列出，节点里「更多设置 → LoRA」可选，选中会自动带出触发词
和匹配的提示词契约。

### 精选结论（全部实跑验证，不是看描述选的）

| LoRA | 体量 | 来源人气 | 判定 |
|---|---|---|---|
| **px_illustrious**（像素游戏素材） | 36MB | dl 55382 / ♥5357 | ✅ **像素风首选** |
| **wuxia_v2**（武侠） | 96MB | dl 17267 / ♥2793 | ✅ **武侠首选** |
| paseer_wuxia（武侠线稿） | 144MB | dl 2533 / ♥507 | ⚠️ 纯线稿，当形状底稿用 |
| px_toolkit（2D 像素工具包） | 36MB | dl 53451 / ♥5140 | ❌ 背景污染 + 多主体，隐藏 |
| guofeng（工笔重彩） | 72MB | dl 18595 / ♥2416 | ❌ 手部崩坏，隐藏 |
| xuer_wuxia | 144MB | dl 8871 / ♥1005 | ❌ 出的是真人摄影风，隐藏 |

> 人气高 ≠ 适合产线。上面三个 ❌ 的下载量都不低，实测才知道不行。
> 隐藏项保留文件（`"hidden": true`），前端不显示，便于复现判断。

### 两条必须记住的用法

**1. 像素 LoRA 要配 `preset=pixel`，绝不能配 `sprite`**

`sprite` 契约里写着 `16-bit pixel art`，而像素 LoRA 已经把风格学进去了，重复要求
会让画面在「插画」和「像素」之间打架——实测套 sprite 契约出的是彩色色块怪物。
正确组合：

```python
txt2img(prompt="pixelart, knight in armor holding sword", preset="pixel",
        lora="px_illustrious", lora_scale=0.75,
        width=512, height=576, pixel=4, palette=24, matte=True)
```

**2. 武侠 LoRA 会给画面盖印章、画水墨背景**

必须靠 `preset=wuxia` 的负向词压（`seal / signature / ink splatter / paper texture`），
而且这类图必须用 **u2net 语义分割**抠图——纯色键控会把白袍和脸一起抠掉（实测第一帧
脸直接消失）。另外 `wuxia_v2` 的产出**偏半身**（训练集如此），要全身走 W4 姿态驱动，
让骨架锁结构、LoRA 只给风格。

## LoRA 训练

自包含训练循环（`workflows/train.py`），不依赖 diffusers 的 examples 脚本，
产出直接是 `Engine.apply_lora` 能加载的格式，训完立刻出现在下拉里。

```bash
# 数据集：图片 + 同名 .txt（每张一句 caption）；没有 .txt 时用 --caption 兜底
dataset/my_knight/
  knight_01.png
  knight_01.txt   ->  "mk_knight, a knight in plate armor, pixel art, white background"
  ...

python tools/train_lora.py --data dataset/my_knight --name mk_knight \
    --caption "mk_knight, pixel art game sprite, white background" --steps 400
```

WebUI 顶栏「训练 LoRA」同一个功能，有进度条、loss、显存与可训练参数量。

**实测（RTX 4090 24GB，fp32 UNet + gradient checkpointing，512 分辨率 batch 1）**

| 项 | 数值 |
|---|---|
| 可训练参数 | rank8 → 1.59M；rank16 → 3.19M |
| 耗时 | 约 **0.34s/步**（60 步 25s / 420 步 143s） |
| 峰值显存 | **4.67GB** |
| 产出体积 | rank8 → 6.2MB；rank16 → 13MB |

**三个坑（都写进代码注释了）**

1. `save_lora_weights` 的第一个参数是**目录**不是文件，且文件名写死
   `pytorch_lora_weights.safetensors`。直接传文件路径会得到一个同名目录，产物「消失」。
2. 直接传 `unet_lora_layers=unet` 会把**整个 fp32 UNet（3.3GB）**存进去且格式不对。
   必须 `get_peft_model_state_dict(unet)` + `convert_state_dict_to_diffusers()` 只取 LoRA 层。
3. 触发词必须写进 `loras.json`，否则训完也没人知道该写什么词唤出它（等于白训）。

**关于样本量**：实测 6 张图 + 420 步只能学到「像素风的大致色块分布」，出图仍会跑形。
角色 LoRA 建议 15~30 张同一角色不同姿势，风格 LoRA 20~40 张，配合
`plan_steps()` 自动按样本数算步数（下限 100、上限 2000）。

## WebUI（画布 · 文生图）

React 18 + Vite + TS，**零 UI 库、零画布库**（与 pixelforge 同思路），构建产物 157KB / gzip 51KB。

**一条命令启动**（自动体检 → 按需构建前端 → 起服务 → 开浏览器）：

```bash
./start.sh                  # Git Bash / macOS / Linux
start.bat                   # Windows 双击或命令行
```

| 参数 | 作用 |
|---|---|
| `--port 8010` | 换端口 |
| `--rebuild` / `--no-build` | 强制重建前端 / 跳过构建 |
| `--check` | 只体检（Python、依赖、CUDA、权重、前端、端口），不启动 |
| `--dev` | 只起 API，前端另开 `cd web && npm run dev`（:5173 代理） |

体检项与实际输出：

```
✓ Python：.venv/Scripts/python.exe（Python 3.13.14）
✓ 依赖齐全（torch / diffusers / transformers / fastapi / uvicorn）
✓ CUDA 可用（NVIDIA GeForce RTX 4090）
✓ 权重就绪（14 个文件）
✓ 前端产物是最新的
```

手动方式：

```bash
cd web && npm ci && npm run build     # 构建
cd .. && python app.py                # FastAPI 同时托管 dist 与 API → http://127.0.0.1:8000
cd web && npm run dev                 # 或开发模式（:5173，代理到 8000）
npm run smoke                         # 无浏览器渲染冒烟（jsdom，25 项断言）
```

> `start.bat` 刻意保持**纯 ASCII**：cmd.exe 按 OEM 代码页解析 .bat，中文注释会切断命令行
> （实测把 `setlocal` 切成 `ld`）。中文提示只留在 `start.sh`。

**布局**（参考了 InvokeAI 的「队列可见/参数分层」、SD.Next 的「折叠压制参数面」、pixelforge 的视觉与画布机制）：

```
┌ 顶栏：品牌 · GPU · 缩放 [− 100% + 1:1 适应] ┐
├───────────────────────────────────────────┤
│ ┌节点列表┐                                  │
│ │图生成 +│      ┌─ 图生成卡 ─────┐          │
│ └────────┘      │ 提示词 / 尺寸 / 种子 │      │
│                 │ 更多设置 ▸         │      │
│                 │ [生成]  预览图      │      │
│                 └───────────────────┘      │
└───────────────────────────────────────────┘
```

**设计决策**

| 项 | 做法 | 理由 |
|---|---|---|
| 节点 | **只有一种「图生成」**，参数与结果同卡 | 生成与结果是一件事，拆两张卡会让每次生成多一次连线 |
| 双视图 | 出图后可切「只看图」，卡片只剩画面 + 种子徽标 | 画布上做对比时，参数是噪音 |
| 新建节点 | 节点列表**拖拽到落点**，或点 **+** 放到原点 (0,0) | 拖拽处落点用 `(client - viewport)/zoom` 换算成世界坐标 |
| 复制/删除 | 卡头按钮 + `Ctrl/Cmd+D` 复制、`Delete` 删除 | 输入框内不拦截按键 |
| 画布 | 自研 viewport(overflow hidden) + world(translate/scale) + 视差点阵网格 | 不上 react-flow：它会把节点模型绑死在库里；InvokeAI 的 README 已现身说法 dockview 之痛 |
| 持久化 | localStorage 防抖 400ms 保存 nodes + viewport | 刷新不丢布局 |
| 画布平移 | **拖空白处**（或空格 / 中键）；滚轮以光标为锚点缩放 | 节点多了全靠它 |
| 训练入口 | 顶栏「训练 LoRA」浮层面板，有进度条/loss/显存 | 不用切命令行 |
| **提示词语言** | 节点内固定提示「SD1.5 只支持英文」；输入含中日韩字符时实时转红警告 | 实测中文被拆成字节级乱 token（'背剑侠客' → `è ĥ Į å ī`），模型只会瞎猜甚至把乱码当文字画出来 |
| **提示词收藏** | 提示词框右上角 ☆ 收藏，左栏「提示词收藏」列出，点条目填入选中节点 | 常用提示词不必反复输入；按文本去重、可删除、localStorage 持久化 |

## 抠图 / 动效导出

```python
r = poseswap(eng, pose_images=poses, prompt="blue robot",
             pixel=4, palette=16, matte=True,          # 出透明底
             export="assets/export/robot", anim_name="walk", fps=10)  # 出 Godot 工程包
```

导出产物（`export_godot`）：图集 PNG、`SpriteFrames.tres`（每帧一个 AtlasTexture）、
`AnimatedSprite2D.tscn`（`texture_filter = 1` 最近邻、`autoplay` 已开、
`offset.y = -cell_h/2` 脚底对齐）、引擎无关 JSON、GIF/APNG 预览、zip 包。
全是确定性文本生成，同输入逐字节可复现。

抠图两个后端（`matte_backend="auto"` 默认有分割模型就走第二个）：

| 后端 | 耗时 | 适用 | 实测短板 |
|---|---|---|---|
| `color`（连通域色键） | **0.02s/帧** | 平坦纯色背景的精灵 | 白袍/浅色角色贴白底会被误抠（实测把脸抠没） |
| `u2net`（isnet-general-use, 170MB） | 1.6s/帧 | 复杂背景、水墨装饰、浅色主体 | 慢 80 倍 |

同一张武侠图对比：`color` 把背后的圆形水墨装饰整个留下、底部黑块也在；
`u2net` 只留人，脸、剑、衣摆全在。**武侠 / 立绘类一律走 u2net。**

抠图三个要点：

1. 只抠**与画面边界连通**的同色区域——否则白猫肚子贴着白底会被一起吃掉。
2. 去白边要**迭代**：混色边通常 1–3px 厚，单次膨胀只剥掉最外一圈（实测残留 184 像素 → 迭代 3 次后 0）。
3. alpha 一律二值。像素素材要硬边，半透明会引入新的脏边。

## W5 视频驱动（Wan2.2 而非 MimicMotion）

选 Wan2.2 TI2V-5B 而不是 MimicMotion，理由很实际：Wan 已在盘上（pixelforge 的 37GB），
MimicMotion 要另下 ~16GB（代理 480KB/s ≈ 9 小时）。MimicMotion 的优势是「真人动作驱动」，
等有明确需求再单独接。

```bash
python tools/video.py t2v "pixel art cat knight walking, flat background" -o assets/clip
python tools/video.py i2v assets/cat.png "the same character walking" -o assets/clip
```

实测（4090 24GB，bf16 + enable_model_cpu_offload）：

| 项 | 数值 |
|---|---|
| 加载 | 23–30s |
| 25 帧 384² / 10 步 | **26.8–30.6s（1.22s/帧）** |
| 峰值显存 | **10.7GB** |

比预期省：pixelforge 记的 19GB 是 fp8+offload；bf16 + 仅 offload 实测 10.7GB 就够。
注意视频底座要和 SD1.5 分开跑（先 `eng.free()`），否则两套权重抢显存。

**一个坦诚的限制**：人体姿态模型对**非人形角色**（四足/怪物）几乎无效——实测把生成的
四足生物视频喂进去，5 帧里只有 3 帧能提够关节。W5 用在人形角色上才顺。

## 测试

```bash
python -m pytest tests -q        # 112 个用例，约 105s（含 1 个需 RUN_WAN=1 的慢用例）
```

| 文件 | 覆盖 | 用例数 |
|---|---|---|
| `tests/test_postprocess.py` | 锚点、逐帧对齐、打包几何、色板上限、alpha 二值化、裁边 | 9 |
| `tests/test_generate.py` | 同 seed 逐像素一致、不同 seed 相异、色板生效、**重绘选区外不变**、序列帧数、显存回收、缺权重报错 | 8 |
| `tests/test_controlnet.py` | Canny conditioning 生效、Tile 放大结构相关性 > 0.5、像素块均匀性 | 3 |
| `tests/test_api.py` | 全部端点、队列串行性、路径穿越防护、400/404 | 9 |
| `tests/test_pose.py` | 检测器切换、18 点完整性、关节在画面内且头高于髋、骨架渲染、双后端一致性、精灵图切片 CLI | 6 |
| `tests/test_io_paths.py` | 视频抽帧、GIF、精灵图切片（含空格子）、CLI 错误码、非 8 倍数尺寸重绘、cols 夹取、环境自检 JSON、LoRA 空接口 | 11 |
| `tests/test_presets.py` | 契约拼接与优先级、77 token 预算、中文 token 估算、下载器过滤（safety_checker/bin/ckpt/fp16 优先） | 9 |
| `tests/test_fetch.py` | Range 分块拼接、断点续传、**拒绝非 206 响应**（防 429 错误体污染）、超长残留文件重置 | 5 |
| `tests/test_stability.py` | 连续 20 次推理显存不涨、`free()` 释放 pipeline 与 ControlNet、跨进程同 seed 一致、PNG 结构、sidecar 一致 | 5 |
| `tests/test_matte.py` | 背景估计、连通域不误抠内部同色区、去白边生效、alpha 二值、容差方向、裁边、真实生成图抠图 | 9 |
| `tests/test_anim.py` | GIF/APNG 帧数、索引几何、Godot tres 分区序列、tscn 像素设置（最近邻/autoplay/脚底偏移）、导出确定性 | 9 |
| `tests/test_video.py` | numpy/PIL 帧兼容、mp4 落地、模型路径校验、Wan 真实生成（`RUN_WAN=1`） | 6 |
| `tests/test_lora.py` | 注册表扫描/元数据/坏 JSON 容错、名称解析、去碎屑、贴边占比判定、武侠契约预算 | 13 |
| `tests/test_train.py` | 数据集读取与兜底、步数规划、**端到端 6 步真训练**（导出的体积上限、可被 scan/apply_lora 认出）、`/train` 全流程 | 9 |

已知限制：骨架质量决定成败——实测 8 帧序列里有 2 帧只提取到 8/18 个关节（手臂与身体重叠），
这两帧生成时会自由发挥，出现朝向漂移。产线上应当在提取后按「可见关节数」筛帧。

测试过程中修掉的两个真 bug：**inpaint 不传 width/height 时 diffusers 默认输出 512×512**（与源图不符，
贴回时 `images do not match`）；**composite 的羽化是双向扩散**（会把选区外像素一起吃掉），
改为先腐蚀再模糊。

## 目录

```
2d-game-meow/
├── app.py                  FastAPI 服务入口
├── workflows/
│   ├── engine.py           pipeline 懒加载、显存守卫、LoRA 热切换
│   ├── generate.py         w1/w2/w3/w4 四个工作流
│   └── postprocess.py      像素化、调色板量化、锚点对齐、精灵图打包
├── tools/
│   ├── fetch_weights.py    权重下载（走 hf-mirror）
│   ├── smoke.py            逐阶段测速/测显存
│   ├── make_pose.py        COCO-18 骨架图生成 / 渲染
│   ├── pose_from_image.py  从图片 / 视频 / 精灵图批量提骨骼
│   ├── video.py            文生视频 / 图生视频 CLI
│   ├── fetch_ranged.py     按 Range 分块下载（抗限流断流）
│   ├── bench.py            稳态基准
│   └── check_env.py        环境自检
├── models/                 权重（checkpoints / controlnet / ipadapter / loras / ...）
└── assets/                 产物 + sidecar(.json) + index.jsonl
```

## 环境（本机实测）

| 项 | 值 |
|---|---|
| GPU | RTX 4090 24GB（驱动 621.01 / CUDA UMD 13.5） |
| CPU / RAM | i9-13900KF 24C32T / **31.8GB** |
| OS | Windows 11 26300 |
| Python | 3.13.14，torch 2.6.0+cu124（`cuda.is_available()=True`） |
| 注意力 | torch SDPA（Windows + py3.13 无 xformers 轮子） |
| 模型源 | `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 直连不可达） |

安装：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速开始

```bash
# 1) 拉权重（魔搭，约 12.6GB：底座 4.0 + inpainting 2.2 + IP-Adapter 2.4 + ControlNet 4.0）
python tools/fetch_modelscope.py            # 全部
python tools/fetch_modelscope.py sd15       # 只拉某一段
python tools/fetch_modelscope.py --list     # 先看清单

# 2) 环境自检 + 分阶段实测（写 assets/_bench.json）
python tools/check_env.py
python tools/make_pose.py          # 生成 COCO-18 骨架帧（无 DWPose 时用）
python tools/smoke.py w1
python tools/smoke.py w3
python tools/smoke.py w4 --pose-file assets/_pose_walk_00.png --no-ipa

# 3) 起服务
python app.py            # http://127.0.0.1:8000/docs
```

提交任务：

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{
  "workflow": "w1",
  "params": {"prompt": "pixel art cat knight, red cape, katana, front view",
             "width": 512, "height": 768, "steps": 25, "pixel": 4, "palette": 16}
}'
```

## 设计约束（都是踩过的坑）

### 前端（这两条是用户实测报上来的）

1. **卡头按钮上不能启动拖拽**。`setPointerCapture` 会把后续 pointerup 重定向到卡头，
   浏览器按「pointerdown 与 pointerup 的最近公共祖先」派发 click —— 于是 click 落到卡头，
   「切回参数视图」永远点不动。修法：`e.target.closest('button, input, textarea, ...')`
   命中就直接 return，不抓手。
2. **画布平移不能靠 `data-bg` 判断**。`.canvas-grid` 是 `pointer-events:none`，点空白时
   事件目标是 `.canvas-viewport` 自己（没有 `data-bg`），判据永远不成立 → 画布完全拖不动。
   改成反向排除：不在节点/面板/控件上就算可拖。
3. **`.block` / `.fold-body select` 的样式漏了**，深色底上 LoRA 下拉几乎看不见。
   现在 `.fold-body` 里的 select 与 number 统一有边框、`color-scheme: dark`、深色 option。


1. **像素感由后处理保证，不靠提示词**。模型只会画出「看起来像像素」的糊图；
   必须走「提示词契约 → 网格吸附 → 共享调色板」三道关。
2. **逐帧生成必须做锚点回拉**。模型每帧会自己重新构图，角色在画面里上下漂移；
   按首帧脚底锚点做**整体整数平移**拉回，只平移不缩放（缩放会抹掉下蹲/抬腿的真实大小变化）。
3. **跨帧一致性靠固定 seed + 固定 LoRA + 固定文本**，不是靠提示词描述得详细。
4. **局部重绘只改选区**：羽化蒙版**向内**收缩后贴回，选区外像素逐点不变。
5. **单卡串行**：一个 worker 线程 + FIFO 队列，不做并发，避免两套 pipeline 抢显存。

## 版本选型：为什么全栈 SD1.5

| 能力 | SD1.5 | SDXL |
|---|---|---|
| 文生图 / 图生图 | 可行 | 画质与 prompt 遵循更好 |
| 局部重绘 | **更优**（专用 inpaint 权重，专为 512 训练） | inpaint 专模弱于自身基座 |
| 姿态 → 精灵帧 | **更优**（lllyasviel OpenPose v1.1，业界默认件） | 社区二次训练，边缘姿态更飘 |
| 视频驱动 | **唯一选项**（MimicMotion 底座即 SD1.5） | 无 |
| 权重体积 / 单步耗时 | 2GB / 基准 | 7GB / 慢 3–4 倍 |
| 授权 | CreativeML OpenRAIL-M | Community License（年营收 >$1M 需付费） |

关键判断：**游戏精灵的交付分辨率是 512–768，不是 1024**。SDXL 出 1024 再缩到 128 只会把细节噪声一起缩进去。
SD1.5 在 512 上「一次到位 + 16 色量化」更贴产线，且被视频模型绑死（MimicMotion 无 SDXL 版）。
