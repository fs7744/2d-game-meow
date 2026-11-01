# 角色中心 · VNCCS 移植评估（2026-11-01）

> **这份文档回答两件事**：① `D:\code\gen\others\ComfyUI_VNCCS-main` 到底有什么；
> ② 按「只保留 Qwen-Image-2.1、其他模型不移植」这条线砍下去，哪些能搬、**哪些
> 必须别的模型（需要你拍板）**。
>
> 所有结论都带 `文件:行号` 证据（路径相对源仓库），不凭印象。

---

## 0. 结论先看

| 类别 | 数量 | 说明 |
|---|---|---|
| **可以直接搬**（只靠 Qwen-Image-2.1 或纯代码） | 约 70% | 角色数据模型、目录布局、风格库、表情库、抠像算法、造角色/换装/出图的**主链路** |
| **必须别的权重，等你决定** | **4 项** | 见 §4：AI 向导的 LLM、人脸框 YOLO、2 个 QI2 LoRA、SAM3/SeedVR2（可选） |
| **明确不移植**（你的指令） | 4 个模型家族 | Flux Klein 9B、MiniMax H3、Anima、Illustrious/SDXL 全家 |

**最重要的一个事实**：VNCCS 的 QI2 路径**没有用 diffusers**——全仓库 `grep diffusers`
零命中，它是用 ComfyUI 原生节点（`UNETLoader` / `CLIPLoader` / `TextEncodeQwenImage21` /
`KSampler` / `VAEDecode`）拼出来的（`nodes/vnccs_control_center.py:1304-1410`）。
所以我们**不能抄调用代码**，只能抄它的**数据模型、交互、参数与算法**，生成部分
接到我们自己已有的 Qwen 管线上。

---

## 1. 源仓库是什么

`ComfyUI_VNCCS-main`（VNCCS **3.3.0**），158MB，是一个 **ComfyUI 自定义节点包**：

| 部分 | 内容 |
|---|---|
| 工作流 | `workflows/VNCCS_3.2_Step1_CharacterCreator.json`、`Step1_CharacterCloner`、`Step2_CharacterClothes`、`Step3_CharacterEmotions` |
| 后端 | `nodes/` 26 个文件，其中 `character_generator.py` 22 万字符、`vnccs_control_center.py` 10 万、`character_creator_v2.py` 12 万 |
| 前端 | `web/` 13 个文件，**全是 ComfyUI 的 DOM widget**（不是独立页面） |
| 数据 | `character_template/`（风格库 + 提示词模板）、`emotions-config/`（表情库）、`control_center.json`（模型清单） |

**它没有独立页面**，一切都是画布节点里塞的一块巨大 DOM
（`web/vnccs_character_creator_v2.js:1567` `app.registerExtension`，
状态塞进节点的隐藏输入 `widget_data`，见 `nodes/character_creator_v2.py:2119`）。
所以「移植」不是搬页面，而是**把节点里的交互搬成我们自己的 React 页面**。

通信是**纯 HTTP REST + 轮询**（路由前缀 `/vnccs/`，挂在 ComfyUI 的
`PromptServer` 上），没有自建 WebSocket（`web/vnccs_transport.js:1` 用
`api.fetchApi` 包了一层，自动加 `X-VNCCS-CSRF: 1` 头）。

---

## 2. 功能全表（逐条：做什么 / 靠什么 / 能不能搬）

### 2.1 Character Creator V2（造角色，文生图）

`nodes/character_creator_v2.py:2109` / `web/vnccs_character_creator_v2.js`

| 功能 | 证据 | 依赖 | 能搬吗 |
|---|---|---|---|
| 建/删角色、写 config | `character_creator_v2.py:2130-2164` | 纯文件 | ✅ |
| 属性表单：性别/年龄/种族/肤色/发型发色/眼/脸/身材/胸/附加细节/NSFW/aesthetics/negative/LoRA/背景色/framing | `js:4184-4219`；字段清单 `py:2138-2146` | 纯表单 | ✅ |
| **画风选择器**：358 条风格 + 868 张预览 webp | `character_template/character_styles.json`（491KB）；`character_styles.mjs:4` `createStylePicker` | 纯数据 | ✅（见 §5 关于 83MB 预览图） |
| 标签构造器（种族/发色等可选项） | `js:3994` `openTagConstructor` → `/vnccs/get_tags?catalog=creator_v2`；数据源 `character_template/character_tags.json` | 纯数据 | ✅ |
| 生成参数：seed(+fixed/randomize)/steps/cfg/sampler/scheduler/分辨率 | `py:2130-2136`、`js:4405` | 接到我们的引擎 | ✅ |
| **QI2 提示词增强（Prompt Enhancement）** | `py:1293-1299` 走 ComfyUI 原生 `TextGenerate`，system prompt 读 `character_template/qwen_image_21_t2i_system_prompt.txt` | **QI2 自己的 CLIP = Qwen3-VL 8B** | ⚠️ 见 §4.1 |
| **Character Wizard（一句话 → 自动填字段）** | `py:1740-1869` `/vnccs/character_wizard`，`n_ctx=6144, max_tokens=900, temperature=0.3` | **Qwen3.5-4B-Q8_0.gguf + llama-cpp-python** | ❌ 见 §4.1 |
| 预览生成（出一张底图） | `py:1884-2035` → 存 `<角色>/cache/preview.png`，cache key 写进 PNG 元数据 | Qwen 文生图 | ✅ |

### 2.2 Character Cloner（拿现成图当角色）

`nodes/character_cloner.py:90` / `web/vnccs_character_cloner.js`

| 功能 | 证据 | 依赖 | 能搬吗 |
|---|---|---|---|
| 上传图 → 建角色（写 config、EXIF 矫正、alpha 合成到背景色） | `py:65-73, 185-201` | 纯 PIL | ✅ |
| 换姿态 / 生成裸体底图（`original` / `naked` 两条分支） | `character_generator.py:3048-3177` | Qwen 编辑 | ✅ |
| **Analyze Captions（看图反推角色属性）** | `py:304-388`：图缩到 512 → JPEG q85 → base64 → `llm.create_chat_completion(max_tokens=1024, temperature=0.2)` → `json_repair` 解析 | **Qwen3.5-4B gguf + mmproj（VLM）** | ❌ 见 §4.1 |

### 2.3 Clothes Designer（换装）

`nodes/clothes_designer.py:318` / `web/vnccs_clothes_designer.js`

| 功能 | 证据 | 依赖 | 能搬吗 |
|---|---|---|---|
| 服装 CRUD：top/bottom/head/shoes/face/negative | 白名单 `utils.py:30` `COSTUME_TEXT_FIELDS`；写入 `utils.py:832-845` | 纯 JSON | ✅ |
| 预览生成（带**全参数 sha256 指纹缓存**，命中直接返回上次） | `py:588-627, 787` → `cache/preview_<服装>.png` | Qwen 编辑 | ✅（缓存思路很值得照抄） |
| ETag 乐观锁（防两个窗口互相覆盖） | `py:819, 842-847` + `utils.py:841-843` `costume_ids` | 纯逻辑 | ✅ |
| CLONE CLOTHES / 衣服向导 | `py:890, 958` | **LLM（同上）** | ❌ 见 §4.1 |
| 批量出图 → `Sprites/<服装>/Neutral/sprite_pose_%04d.png` | `character_generator.py:3238, 3356` | Qwen | ✅ |

### 2.4 Emotion Studio（表情）

`nodes/emotion_generator_v2.py:517` / `character_generator.py:3367` / `web/vnccs_emotion_v2.js`

| 功能 | 证据 | 依赖 | 能搬吗 |
|---|---|---|---|
| **表情库：157 条 / 9 大类**（Anger18 Smile12 Smug12 Surprise11 Emotional13 Sexual24 Focus11 Pain5 Emotes51） | `emotions-config/emotions.json`；每条 4 字段 `key/description/safe_name/natural_prompt` | 纯数据 + 157 张 webp（6.3MB） | ✅ |
| 自定义表情（可上传、自动合并、legacy 迁移） | `emotion_library.py:28-53` | 纯逻辑 | ✅ |
| 批量生成表情，**QI2 路径是"只改脸"**：YOLO 找脸框 → 裁切 pad 到 32 倍数 → 放大 → 生图 → **羽化贴回原图**（其余像素不动） | `character_generator.py:1591-1698`；几何：`_pad_image_to_square_multiple:1571`、`_crop_feather_mask:1516`、`_paste_crop_direct:1483` | Qwen + **face_yolov8m.pt** | ⚠️ 见 §4.2 |
| 改完脸只对局部重抠再与原始 alpha 合成（避免整图重抠导致边缘漂移） | `character_generator.py:3683, 3707, 3750` | 纯 torch | ✅ 算法值得抄 |
| 非 QI2 走 `FaceDetailer` | `py:4188-4220` | Illustrious/Anima | ❌ 不移植 |

### 2.5 Character Generator（出图引擎）

`nodes/character_generator.py:1007`

| 功能 | 证据 | 依赖 | 能搬吗 |
|---|---|---|---|
| 主链路 stage 序：姿势 → 放大 → 抠图 → 落盘 | `py:2802` `['pose_generation','upscaler','bg_remove','_save_final_sprites']` | Qwen + 放大 + 抠图 | ✅（放大/抠图见 §4.3） |
| QI2 分支 prompt 模板："Replace the pose of `<image 2>` with the pose of `<image 1>`. Keep the character of `<image 2>`." | `py:2210-2212` | Qwen | ✅ **我们 wvnccs 已经用同一句**（`workflows/generate.py:1701`） |
| 产物版本化（重生成时旧图归档到 V1/V2） | `py:2732-2738`、`utils.py:264-270` | 纯逻辑 | ✅ |
| Klein 分支 | `py:2225-2280` | Flux Klein 9B | ❌ 不移植 |
| H3 分支（视频模型当静帧用，取首帧） | `py:1975-2106`，`H3_FRAME_COUNT=5`（`py:991`） | MiniMax H3 | ❌ 不移植 |

### 2.6 Control Center（模型下载器 + 管线组装）

`nodes/vnccs_control_center.py:1857` / `web/vnccs_control_center.js`

**不移植。** 我们不需要下载 5 个模型家族、也不需要 ComfyUI 的 loader 编排
（`py:1304-1410` 就是对 `comfy.sd.load_diffusion_model/load_clip/VAE` 的薄封装）。
但 `control_center.json` 里的 **QI2 条目清单**有参考价值（§3.2）。

---

## 3. 模型清单

### 3.1 砍掉的部分（按你的指令）

| 家族 | 涉及文件 | 砍掉的代价 |
|---|---|---|
| **Flux Klein 9B**（+ Qwen3-8B 编码器 + Flux2 VAE + Klein LoRA） | `control_center.json:4-13,96-106,141-151,194-213`；`vnccs_flux_klein_encoder.py` 整文件；`character_generator.py:2225-2280` | 失去"4 步出图"的快速选项。QI2 25 步照常 |
| **MiniMax H3**（视频模型当静帧，+ Qwen3-VL 32B + Video/Audio VAE + 3 个 LoRA） | `control_center.json:14-33,107-117,152-171,214-243`；`h3_still.py`；`character_generator.py:1975-2106` | 失去 H3 姿态分支。`h3_still.py:13-28` 还在 import 期探测 ComfyUI 版本，删掉正好 |
| **Anima**（Base v1.0 + Qwen3-0.6B 编码器 + VAE + Turbo LoRA） | `control_center.json:54-63,129-139,182-191,294-303`；`character_creator_v2.py:121-136` | 失去 `anima` profile 与 510 条画师 tag 风格 |
| **Illustrious / SDXL**（3 个 ckpt + DMD2 + Mimimeter 年龄 LoRA） | `control_center.json:64-93,284-293,304-313` | 失去 `illustrious` profile、FaceDetailer 表情模式、年龄 LoRA |

> ⚠️ 别误删：`control_center.json:118-128` 的 **QI2 Qwen3-VL 8B** 和 `:107-117` 的
> **H3 Qwen3-VL 32B** 是两个不同的编码器，8B 要留。

### 3.2 QI2 路径保留的权重，以及我们的现状

| VNCCS 里的权重 | 我们有没有 | 说明 |
|---|---|---|
| Qwen Image 2.1 INT8 ConvRot（UNet） | ✅ 有（diffusers 版 qwen-image-2.1） | 我们用 diffusers 管线，不用它的 int8 convrot 量化版 |
| Qwen3-VL 8B 文本编码器 | ✅ 有 | `workflows/engine.py:478` 就是 `Qwen3VLForConditionalGeneration` |
| QI2 VAE bf16 | ✅ 有 | 管线自带 |
| **VNCCS Pose Studio QI2** LoRA | ✅ **有** | `models/lora/vnccs_pose_v2.safetensors`（wvnccs 已在用，`generate.py:1798`） |
| **VNCCS Overhaul QI2** LoRA（二次元风格） | ❌ 缺 | 见 §4.4 |
| **VNCCS Clothes Core QI2** LoRA（服装一致性） | ❌ 缺 | 见 §4.4 |
| Viggle Turbo（6 步加速，r128） | ❌ 缺 | 可选，见 §4.4 |

**QI2 默认参数**（`vnccs_control_center.py:2073-2081`、`emotion_generator_v2.py:40-55`）：

```python
steps = 25            # qi2 默认（klein=4, h3=8）
cfg   = 3.0           # qi2 默认（其余 1.0）
sampler_name    = "euler"
pipe.scheduler  = "simple"     # DEFAULT_MODEL_SCHEDULER
# Turbo 分支：steps=8, cfg=1.0
```

**Turbo（Viggle）专用 sigma 表**（`nodes/qi2_viggle.py:8-19`，硬要求 steps==6 / cfg==1）：

```
nodes = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
shift = 0.5 + 0.4 * (tokens - 256) / (8192 - 256)
shifted = exp(shift) / (exp(shift) + (1/nodes - 1))
```

**参考图的 latent 加权方式**（`nodes/vnccs_qwen_encoder.py:250`）：权重取**平方**
`(w**2) * latent`，不是线性。这条值得跟我们 `generate.py` 里的实现核对一下。

---

## 4. 必须别的模型的功能 —— **已拍板（2026-11-01）**

> 下面四条原本是待定项，用户已逐条定下方案。**移植时按这个执行**。

| 项 | 决定 | 落地要点 |
|---|---|---|
| **4.1 AI 向导 / 看图反推** | ✅ **装 llama-cpp-python + 下 Qwen3.5-4B-Q8_0.gguf**（方案 A，与官方一致） | 必须是 **JamePeng fork** 的 llama-cpp-python（`install.py:8-18` 会校验 Project-URL）；权重 `unsloth/Qwen3.5-4B-GGUF` @ `e966cca`，落到 `models/llm/Qwen3.5-4B/`；看图还要 `mmproj-F16.gguf`；必须用 `Qwen35ChatHandler` + `enable_thinking=false`（`qwen_vl.py:27-64`）。依赖缺失时功能要返回 `DEPENDENCY_MISSING` 而不是崩（`character_cloner.py:225` 就是这么做的） |
| **4.2 表情的人脸框** | ✅ **下 face_yolov8m.pt**（约 50MB，ultralytics） | Emotion 的 QI2 路径照抄官方：找框 → 裁切 pad 到 32 倍数 → 放大 → 生图 → 羽化贴回（`character_generator.py:1532/1571/1516/1483`） |
| **4.3 两个 QI2 LoRA** | ✅ **下载 VNCCS Overhaul QI2 + Clothes Core QI2**（各约 300MB，`MIUProject/VNCCS_v3.0`） | 走我们已有的 `/loras` 挂载机制；**Viggle Turbo 暂不下**（可选加速） |
| **4.4 表情库内容** | ✅ **157 条全量原样入库并显示**（含 Sexual 🔥 24 条） | 与官方一致，不做内容过滤 |

### 还需要下载的权重 —— **已写成脚本 `tools/fetch_vnccs_weights.py`**

体积是**联网核对过的真实值**（走 hf-mirror 镜像查的 `files_metadata`）：

| 权重 | 来源 | 实测体积 | 落点 |
|---|---|---|---|
| `Qwen3.5-4B-Q8_0.gguf` | `unsloth/Qwen3.5-4B-GGUF` @ `e966cca` | **4.17 GB** | `models/llm/Qwen3.5-4B/` |
| `mmproj-F16.gguf` | 同上 | ~1 GB | 同上 |
| `VNCCS_QI2_AnimeOverhaulV2.safetensors` | `MIUProject/VNCCS_v3.0` | **152 MB**（不是 300！） | `models/lora/` |
| `VNCCS_QI2_ClothesCoreV2.6.safetensors` | 同上 | **152 MB** | `models/lora/` |
| `face_yolov8m.pt` | `Tenofas/ComfyUI`（用户提供，官方只给了路径） | **49.6 MB** | `models/bbox/` |

脚本用法：

```
python tools/fetch_vnccs_weights.py --dry-run                     # 只看要下什么
python tools/fetch_vnccs_weights.py --only lora                   # 只下两个 LoRA
python tools/fetch_vnccs_weights.py --endpoint https://hf-mirror.com   # 国内走镜像
```

三个坑（都是核对时抓出来的）：

1. **mmproj 的文件名跟官方代码对不上**。官方 `qwen_vl.py:68` 写的是
   `mmproj-Qwen3.5-4B-F16.gguf`，但仓库里**没这个文件**（实际是
   `mmproj-F16.gguf`），照官方那个名字下会 404。官方自己的查找顺序也是
   先找短名（`_find_qwen_vl_mmproj`，:171-187），带模型名的那两个只是兜底。
2. **LoRA 下完必须转格式**。官方发的是 ComfyUI 格式（`diffusion_model.` +
   `lora_A/B`），我们的引擎只认 diffusers 原生（`transformer.` +
   `lora.down/up`）—— 不转挂上去会**静默不生效**（一个层都匹配不到，
   界面上什么都不报，只是"出图跟没挂一样"）。脚本自动跑 `convert_lora.py`。
3. **两个 QI2 LoRA 的真实文件名带版本号**（AnimeOverhaulV2 /
   ClothesCoreV2.6），不是 `VNCCS_Overhaul_QI2` 这种写法。

运行时依赖：llama-cpp-python **必须装 JamePeng fork**（上游没有
`Qwen35ChatHandler`），官方是装按 Python/CUDA/平台拼好的预编译 wheel
（`install.py:21-47`），不是 `pip install llama-cpp-python`。脚本会打印
该装的那个 wheel URL，**不自动装**（装 wheel 有换掉 torch 的风险）。

### 还需要下载的权重（交给安装脚本/手动）

| 权重 | 来源 | 体积 | 落点 |
|---|---|---|---|
| `Qwen3.5-4B-Q8_0.gguf` + `mmproj-F16.gguf` | `unsloth/Qwen3.5-4B-GGUF` @ `e966cca` | 4.4GB + 1GB | `models/llm/Qwen3.5-4B/` |
| `face_yolov8m.pt` | ultralytics 官方 | ~50MB | `models/yolo/` |
| `VNCCS_Overhaul_QI2.safetensors` | `MIUProject/VNCCS_v3.0` | ~300MB | `models/lora/` |
| `VNCCS_Clothes_Core_QI2.safetensors` | `MIUProject/VNCCS_v3.0` | ~300MB | `models/lora/` |

> 上面这些**现在都还没下**。落地顺序建议：先把**数据结构 + 页面**做完（不需要
> 任何新权重，用我们已有的 Qwen 管线就能出图），权重按需再补。

---

## 4b. 原始选项（保留以备回看）

### 4.1a Prompt Enhancement：已按官方接上（用图像模型自己的 Qwen3-VL）

**PE 和「AI 向导」是两件事，官方用的也不是同一个模型**：

| 功能 | 官方用什么 |
|---|---|
| **Prompt Enhancement**（把结构化字段扩写成描述段落） | **QI2 自己的 CLIP = Qwen3-VL 8B**（`character_creator_v2.py:1293`，走原生 `TextGenerate`） |
| Character Wizard / Analyze Captions | Qwen3.5-4B gguf |

所以 PE **不需要 gguf**，用我们已有的 `Qwen3VLForConditionalGeneration`
（`engine.py:478`）就能做，零新权重。已按官方 `generate_creator_qi2_prompt`
（:1271-1317）整条流程实现（`workflows/qwen_text.py` + `charactergen.enhance_character_prompt`）：

1. 结构化 prompt → 2. **切掉 `LoRA triggers:` 那行再送进 PE**（:1285）
3. PE 改写 → 4. 透明底被改写没了的补回来（:1308）
5. 风格参考**在 PE 之后**追加（:1315）→ 6. 末尾补 LoRA 触发词

采样参数照抄 `:159-174`：max 2048 / temp 0.3 / top_k 64 / top_p 0.95 / min_p 0.05 /
rep_pen 1.05 / thinking=False。

> ⚠️ **已知代价（先记着，后面再解决）**
>
> 1. **显存**：PE 要跑自回归解码，文本编码器得**常驻 GPU**；而我们的管线为了
>    省显存是「编码完就挪走」（`engine.py:439`，注释写明两块 fp8 常驻 15.5GB
>    已经压线）。24GB 上跟 transformer 抢是真的。
> 2. **耗时**：8B 自回归最多 2048 token，PE 是每次出图都跑。
>
> 缓解：每个入口都做了**失败降级**（PE 挂了退回未扩写 prompt，不阻断出图），
> 前端有开关可关。待办：量一遍真实显存/耗时，再决定默认值与是否做
> 「PE 完立刻卸载」的策略。

### 4.1 AI 向导 / 看图反推属性（VNCCS 用 Qwen3.5-4B-GGUF）

**涉及**：Character Wizard（`character_creator_v2.py:1740-1869`）、
Cloner 的 Analyze Captions（`character_cloner.py:304-388`）、
Clothes Wizard（`clothes_designer.py:890,958`）。

**VNCCS 怎么做的**：本地 `llama-cpp-python` 直跑 GGUF，**不是 API**：

```python
# nodes/qwen_vl.py:66-76, 192-195
QWEN_VL_MODEL_FILENAME  = "Qwen3.5-4B-Q8_0.gguf"     # ~4.4GB
QWEN_VL_MMPROJ_FILENAME = "mmproj-F16.gguf"          # ~1GB（看图才要）
QWEN_VL_MODEL_REPO_ID   = "unsloth/Qwen3.5-4B-GGUF"
QWEN_VL_MODEL_REVISION  = "e966ccab6d3f3c91e94d858b4a01c921a5d7ef53"
# 且必须用 JamePeng fork 的 llama-cpp-python（install.py:8-18 校验 Project-URL）
```

**我的判断与三个选项（请选一个）**：

| 方案 | 代价 | 效果 |
|---|---|---|
| A. **装 llama-cpp-python + 下 4.4GB gguf** | 多一个依赖（还得是特定 fork）+ 4.4GB 磁盘；运行时与 Qwen 抢显存 | 与官方完全一致 |
| B. **用我们自己的 Qwen3-VL 文本编码器做生成**（推荐） | 要给它写一条"文本生成模式"的调用（现在只当编码器用一次就挪走，`engine.py:439`）；生成时要常驻显存（fp8 8.8G / bf16 16.3G） | 同族模型，效果应有可比性；**不新增权重** |
| C. **先不做向导，属性全部手填** | 0 | 功能少了"一句话造角色"，其余全在 |

> 顺带一提：**QI2 的 Prompt Enhancement 就是这个思路**——VNCCS 直接拿图像模型的
> CLIP（Qwen3-VL 8B）当 LLM 改写提示词（`character_creator_v2.py:1293-1299`
> 调原生 `TextGenerate`）。也就是说 **B 方案在 VNCCS 里是有先例的**，不是我瞎提。

### 4.2 表情的人脸框（VNCCS 用 face_yolov8m.pt）

`character_generator.py:1532` `_vnccs_bbox_extract_face`；Emotion 的 QI2 路径
"只改脸、其余像素不动"必须先有脸框。

**选项**：
- A. 下 `face_yolov8m.pt`（~50MB，ultralytics，非扩散模型）—— 与官方一致；
- B. 用我们已有的 DWPose 关键点**从 18 个关节里推脸框**（鼻/眼/耳就够定一个框）—— 不加权重，但精度差一点；
- C. 表情先不做"只改脸"，改成整图重绘——省事但会动到头发/背景。

### 4.3 放大与抠图增强（可选）

| 项 | VNCCS | 我们 |
|---|---|---|
| 放大 | SeedVR2（`seedvr2_3b_fp8_e4m3fn.safetensors`，`character_generator.py:754,2377-2534`） | 用我们自己的后处理/放大，**不移植** |
| 抠图增强 | SAM3（`vnccs_utils.py:215-278`，~1GB） | 我们已有 matte 后端（u2net/color），**不移植** |
| **绿幕/纯色幕抠像** | `chroma_screen_matte.py`（纯 torch，零模型依赖） | ✅ **照搬**（见 §5） |

### 4.4 两个 QI2 LoRA

`VNCCS Overhaul QI2`（二次元风格）与 `VNCCS Clothes Core QI2`（服装一致性），
来自 `MIUProject/VNCCS_v3.0`，各约 300MB；`Viggle Turbo` 约 500MB（可选加速）。

**没有它们会怎样**：造角色仍能出图（风格靠 prompt），但二次元味道与换装的
一致性会弱一截。**要不要下，请你定**（我们的 `/loras` 机制已经支持挂载）。

---

## 5a. 已搬完的：纯色幕抠像（`workflows/chroma_screen_matte.py`）

官方 `nodes/chroma_screen_matte.py` **整文件搬运，只改了一处**——`_processing_device()`
原来会去问 ComfyUI 的 `model_management.get_torch_device()`，改成按 torch 自己的可用性
挑设备（语义一致）。其余算法逐行未动。

- **纯 torch，零模型依赖**，CPU 能跑；官方自己也当独立模块用
  （源仓库 `scripts/benchmark_chroma.py` 直接 `sys.path.insert` 后 import）。
- `workflows/matte_screen.py` 是我们加的一层胶水：PNG → tensor → 抠 → RGBA PNG。
- 接在角色出图的收编步骤里（官方的 `bg_remove` 阶段），抠完**另存 `_matte.png`，
  原图保留**；抠失败只记原因，不判任务失败。
- **白底/灰底不自动抠**（`should_auto_matte`）：幕布色是按边框取样的，
  白毛白甲会被一起判成背景——这条我们项目里踩过。
- 验证：合成"绿幕 + 灰人偶"跑真值断言（幕布 alpha<8、主体 alpha>247、RGB 不偏色），
  蓝幕/品红幕同样过。`tests/test_matte_screen.py` 7 条。

## 5b. 已搬完的：精灵帧翻看器（`SpritePreview.tsx`）

移植自官方 `web/vnccs_common.js:739` `createSpritePreviewNavigator` + 后端
`character_creator_v2.py:1666-1717` / `utils.py:936-981`。

- 前端两步走：`poses/meta` 拿 count → `poses?index=N` 单张取（顺便给
  `X-VNCCS-Preview-Index` / `-Count` 两个响应头，与官方同名）。
- `index` 省略时**随机一张**（官方 :1686 的封面式展示），越界取模。
- 官方 `interactions.presentPreview` 那种「点图开大图」先留白，目前只做翻看。
- **目录大小写两种都认**：官方 `Neutral/` 与我们 `neutral/` 互通。
- 验证：`tests/test_char_poses.py` 11 条（含取模绕回、随机落在范围内、
  非法路径 400、官方大写目录可读、素材走 `/files/` 能真取到字节），
  前端冒烟一组 12 条。

## 5c. 已搬完的：属性标签选择器（`characterTags.ts` + `TagPicker.tsx`）

对应官方 Character Creator 里那排「Choose Presets」弹窗，**纯数据 + 两个纯函数**，
零模型依赖。移植自 `web/character_presets.mjs`（全文 60 行）与
`vnccs_character_creator_v2.js:4000-4060`。

**用的是哪份数据**：官方有两套标签库

| 文件 | 分类数 | 给谁用 |
|---|---|---|
| `character_presets_v2.json`（creator_v2） | **15 个**，覆盖全部属性字段 | **QI2**（`mode === 'qi2'` 时强制它，:4000） |
| `character_tags.json`（legacy） | 6 个，只有 hair/eyes/face | SDXL / Illustrious 老 UI |

我们只移植 QI2，所以用前者。legacy 那份也没白留着，注释里写清了为什么不用。

必须照搬的两个细节：

1. `toggle` 写进字段的是 `tag` 的**空格形式**（`blue_eyes` → `blue eyes`），
   多个 trait 用 `", "` 拼起来；
2. 已选项按 **tag / label / synonyms 三种别名**匹配，且**保留用户自定义文本**——
   否则手写的描述会被"选一次标签"整段清掉。

后端 `race_features` 早就支持这套格式（按逗号拆 + 处理 `(权重)`），所以
**多选做混血**（官方 UI 上写的 "Combine species for hybrids"）是通的。

`FIELD_CATEGORIES` 有 7 个字段，`breast_size` 是调用处开的**第 8 个入口**
（:4018：body 弹窗只给体型，胸围单独一个弹窗）——照搬了这个不对称。

## 5. 可以直接搬的纯代码 / 数据（无模型依赖）

### 5.1 `utils.py`（根目录，980 行）—— **零 ComfyUI 依赖，最值得整文件搬**

`import` 只有 stdlib + 自家 logger（`utils.py:1-20`），`folder_paths` 是当参数传进来的。

| 块 | 行号 | 作用 |
|---|---|---|
| 角色磁盘布局 `base_output_dir/character_dir/sprites_dir/faces_dir/sheets_dir` | 55,405,410,421,432 | `output/VNCCS/Characters/<角色>/` |
| 目录骨架 `ensure_character_structure` / `ensure_costume_structure` | 443,471 | `MAIN_DIRS=["Sprites","Faces","Sheets"]`（23-24） |
| **防路径穿越** `ensure_safe_name` / `is_path_under` / `safe_join_under` | 66,190,340 | Win/Linux 双栈（177） |
| 跨 OS 路径匹配 `get_full_path_agnostic` / `basename_agnostic` | 125,164 | |
| 分片文件锁 `character_storage_lock` | 236,29 | |
| **原子发布 + 旧版归档** `staged_image_batch` / `archive_current_image_batch` | 252,310 | 旧图进 V1/V2 |
| JSON 原子写 `atomic_output_path` / `save_config` / `load_config` | 674,690,650 | |
| 指纹/版本 `file_fingerprint` / `json_revision` | 220,816 | |
| 枚举 `list_characters` / `list_costumes` | 490,908 | 扫目录（无索引文件） |
| 种子语义 `generate_seed` / `inherit_seed` | 500,510 | 0 = 继承上游 |
| 提示词后处理 `apply_sex` / `append_age` / `build_face_details` / `dedupe_tokens` | 551,628,740,775 | |
| 服装 CRUD `save_costume_info` / `delete_costume` | 832,848 | |
| CSRF 装饰器 `privileged_route` | 392 | `X-VNCCS-CSRF: 1`；我们同源单用户，可省 |

### 5.2 抠像（纯 torch，零模型）

`nodes/chroma_screen_matte.py:201` `screen_matte`（它已被 `scripts/benchmark_chroma.py:27`
独立调用，最容易整块搬）；配套 `vnccs_utils.py` 的 `_build_soft_alpha:950`、
`_cleanup_alpha:989`、`_guided_edge_refine:1001`、`_edge_despill:1158`、
`_edge_decontaminate:1213`、`_suppress_connected_key_fringe:1038`——一整套去溢色流水线。

### 5.3 数据资产

| 资产 | 体积 | 说明 |
|---|---|---|
| `emotions-config/emotions.json` + `images/`（157 webp） | 48KB + **6.3MB** | ✅ 直接搬 |
| `character_template/character_styles.json`（358 条） | 491KB | ✅ 搬 |
| `character_template/character_tags.json`、`character_presets_v2.json` | 8.8KB + 36KB | ✅ 搬 |
| `character_template/qi2_prompt_rewriter.txt`、`qi2_edit_prompt_rewriter.txt`、`qwen_image_21_t2i_system_prompt.txt` | 几 KB | ✅ 搬（提示词模板，无版权问题的话） |
| `character_template/style_previews/`（868 webp） | **83MB** | ⚠️ 见下 |
| `character_template/anima_styles.json`（510 画师 tag） | 67KB | ❌ Anima 专用，不搬 |

**关于那 83MB 风格预览图**：868 张里有 510 张是 **Anima 画师 tag 的预览**（我们
不移植 Anima），所以只需要 358 张 ≈ **34MB**。三个选项：
- A. 全搬 358 张（+34MB 仓库，风格选择器一眼能挑）；
- B. **不搬预览图**，用我们自己的文生图按风格 prompt 现生成缩略图（省 34MB，
  但首次要跑 358 次出图）；
- C. 搬但压到更小尺寸/质量再压一轮（比如 128px webp q60，约 8~10MB）。

**我建议 C**，请你定。

### 5.4 前端可直接借鉴的交互（`web/vnccs_common.js` 1748 行）

| 导出 | 行号 | 搬到 React 时的对应 |
|---|---|---|
| `SAKURA_THEME_CSS` | 10-35 | 主题 CSS 变量（`--accent:#ff8fa3` 等） |
| `createTraitInput` | 554 | **标签/chips 输入**（种族/发色这类多选属性） |
| `createSpritePreviewNavigator` | 739 | **图片序列预览器**（先要 meta 拿 count，前后翻页 + 预取相邻图） |
| `showModal` / `showMessage` | 1395 / 1533 | 模态框（带焦点陷阱）/ toast |
| `createLoadingOverlay` | 1565 | 生成中的 spinner |
| `debounce` / `generateRandomSeed` | 667 / 1582 | |
| `sakura_interactions.mjs:361/388` `captureView/restoreView` | | **列表重渲染时保住滚动位置与焦点**——这个思路对 React 列表很有用 |

`sakura_interactions.mjs` 不是吉祥物，是共享 UI 控件 + 动效引擎（"Sakura" 是粉色
主题名，`vnccs_common.js:1587`），与业务无关。

---

## 6. 角色数据模型（移植的核心契约）

根目录 `output/VNCCS/Characters/<角色名>/`：

```
Alice/
├── Alice_config.json                 ← 唯一 manifest（utils.py:644-647）
├── cache/
│   ├── preview.png                   ← 角色封面
│   └── preview_<服装>.png + preview_info_<服装>.json
├── Sprites/
│   ├── Naked/                        ← 基础服装（utils.py:455）
│   │   ├── Neutral/sprite_pose_0001.png
│   │   └── V1/, V2/…                 ← 重生成时旧版归档
│   ├── Original/…                    ← Cloner 的原图分支
│   └── Dress/                        ← 一套服装 = 一个目录
│       ├── Neutral/sprite_pose_0001.png
│       └── angry/sprite_angry_0001.png
├── Faces/<服装>/<表情>/face_…        ← 脸部特写（把 sprite_ 换成 face_，character_generator.py:3982-3991）
└── Sheets/<服装>/<表情>/sheet_…      ← 打包精灵表
```

`Alice_config.json` 顶层字段（`character_creator_v2.py:2152-2156`）：

```jsonc
{
  "character_info": {
    "name": "Alice", "sex": "female", "age": 18, "race": "human",
    "skin_color": "", "hair": "black hair, waist-length hair", "eyes": "blue eyes",
    "face": "freckles", "body": "", "breast_size": "medium breasts",
    "additional_details": "", "nsfw": false, "aesthetics": "masterpiece",
    "negative_prompt": "bad quality,worst quality,...",
    "lora_prompt": "", "background_color": "green", "framing": "",
    "style": "ghibli_miyazaki", "custom_style": "", "seed": 0
  },
  "costumes": { "Dress": { "top": "", "bottom": "", "shoes": "", "head": "", "face": "", "negative_prompt": "" } },
  "costume_ids": { "Dress": "uuid4hex" },      // ETag 乐观锁用
  "character_path": ".../Characters/Alice",
  "folder_structure": { "main_directories": ["Sprites","Faces","Sheets"], "emotions": ["neutral"] },
  "config_version": "2.0"
}
```

**枚举方式**：扫目录，没有索引文件（`utils.py:490-497`）。

---

## 7. 移植计划（我打算这么推进）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **P0** | 本评估文档 + 你拍板 §4 的四个选项 | ✅ 文档已写，**等你定** |
| **P1** | 数据资产入库（表情库 6.3MB + 风格 json）+ 后端角色存储（`workflows/characters.py`，搬 `utils.py` 零依赖部分）+ API | 下一步 |
| **P2** | 前端「角色中心」页面（与画布同级 tab）：角色库网格 → 角色详情（属性表单 + 风格选择 + 预览） | |
| **P3** | 造角色（接我们的 w1 文生图）、换装（接图生图/编辑）、出图落盘（Sprites/Faces/Sheets） | |
| **P4** | 表情批量生成（取决于 §4.2）、克隆（取决于 §4.1） | |
| **P5** | 向导（取决于 §4.1）、抠像（搬 `chroma_screen_matte.py`） | |

**一个提醒**：表情库里 `Sexual 🔥` 24 条、`Emotional 😢` 等带 NSFW 倾向的条目
（`emotions-config/emotions.json` 原文如此，含 `ahegao` 等）。要不要在入库时
过滤掉这一类，请你定。
