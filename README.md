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
| 抠图 / 去白边 | 未开始 |
| LoRA 训练脚本 | 未开始 |
| 姿态提取（YOLOX 检测 + RTMPose 回归 → OpenPose 骨架） | 已完成，实测通过 |
| W5 视频驱动（MimicMotion） | 未开始 |

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

## 测试

```bash
python -m pytest tests -q        # 65 个用例，约 80s
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
