# Qwen-only 迁移计划与落地结果

> 目标：把工作台从「SD1.5 / SDXL / Qwen 三底模」变成「**只有 Qwen-Image-2.1**」，
> 十个工作流全部用 Qwen 实现。
>
> 起点：`sd` tag（SD 时代完整基线，已推远端）。
> 本文记录**已经落地的东西 + 实测数据 + 明确做不到的东西**。
> 做不到的部分单独成章（§4），不要在后面"顺手再试一次"——每条都附了原因与证据。

---

## 0. 总览：十三个工作流现在都能在 Qwen 上跑

原计划的十个之外，2026-10-30 又从 RunComfy 的 Qwen 工作流里补了三个
（扩图 / 换装 / 换脸，见 §2.14）。

| # | 工作流 | Qwen 实现方式 | 性质 | 实测状态 |
|---|---|---|---|---|
| w1 | 文生图 | 原生 DiT | 原生 | ✅ 24 步 / 768² 可用 |
| w2 | 图生图 | **指令改图**（无 strength 语义） | 原生 | ✅ |
| w3 | 局部重绘 | **真蒙版**（蒙版内换噪声 + 每步蒙版外拉回原 latent） | 原生 | ✅ 指令准确执行 |
| w4 | 姿态序列 | 逐帧「骨架图当参考 + 指令」→ 尺度归一 → 脚底对齐 | **近似** | ✅ 3/3 可交付 |
| wss | 序列帧替换 | 逐帧「角色图 + 轮廓/骨架」双图条件 → 共享色板 | **近似** | ✅ 3 帧跑通 |
| w5 | 局部 AI 重绘 | 同 w3（共用一套）；旧路径与裁剪块策略保留做对照 | 原生 | ✅ 指令被执行、脸不丢 |
| wpc | 角色迁移 | 双图条件（角色 + 骨架） | 原生 | ✅（上一轮已落地） |
| wbg | 抠图 | 不跑模型 | 原生 | ✅ |
| wpx | 像素转换 | 不跑模型 | 原生 | ✅ |
| wap | 转像素 | 不跑模型 | 原生 | ✅ |
| **wout** | **扩图** | 中性灰 padding + Outpaint LoRA + 原图贴回 | 原生 | ✅ 无接缝、无灰块 |
| **woutfit** | **换装** | 两图 + `<image1>/<image2>` 结构化指令 | 原生 | ✅ 服装来自图 2、人还是图 1 |
| **wface** | **换脸** | 同上（保留/替换的角色反过来） | 原生 | ✅ 脸像图 2、姿势场景是图 1 |

「近似」的定义：**该工作流原本依赖的硬约束在 Qwen 上不存在**，改用软引导或
后处理凑出同类产物。后端 `engine._APPROX` 是唯一判据，`/models` 的
`caps.approx` 把它发给前端，节点卡头上显示橙色「近似」标记。
不要让用户在不知情的情况下拿它替代 SD 的那条路。

---

## 1. 这一轮做了什么（代码层）

| 改动 | 文件 | 要点 |
|---|---|---|
| 打开 Qwen 的四个工作流 | `engine._ARCH_WORKFLOWS["qwen21"]` | 十个全开；新增 `_APPROX` / `approx_of` / `is_approx` |
| Qwen 版局部重绘 | `generate._qwen_inpaint` / `_qwen_ai_repaint` | 整图条件 + 蒙版贴回；`outside_max_diff` 写进 meta 做自检 |
| Qwen 版姿态序列 | `generate._qwen_poseswap` | 逐帧软引导 + 尺度归一 + 脚底对齐 + 单主体闸门 |
| Qwen 版序列帧替换 | `generate._qwen_sprite_swap` | 双图条件 + 共享色板 + 灰阶条件图 |
| 条件图分辨率对齐 | `generate._qwen_call(output_resolution=…)` | 见 §2.2，**质量与显存同时受益** |
| 文本编码器 fp8 化 | `engine._load_qwen_text` + `tools/make_qwen_fp8.py` | 见 §2.1，显存峰值 -31% |
| 逐帧尺度归一 | `postprocess.normalize_sequence` | 见 §2.3，可交付率 1/3 → 3/3 |
| 条件图灰度化 | `generate._qwen_cond_image` | 见 §2.4，止住「抄骨架颜色」 |
| 透明底提示词防冲突 | `generate._strip_bg_words` | 见 §2.7，alpha 从「雾」变干净 |
| 抠图不再无视已有 alpha | `matte.alpha_trust` / `matte_auto` | 见 §2.7，IoU 1.000 原样复用 |
| 抠图输入修正 + 补洞 | `matte.flatten_white` / `fill_holes` | 见 §2.7，透明区不再变黑、内部洞补回 |
| 近似标记 | `app.py` `caps.approx` + 前端角标 | 用户一眼看到哪几个是近似 |
| 逐帧底色可选 | `generate.QWEN_BACKDROPS` + 前端「更多设置」 | 白毛角色换深底，抠图不再吃脸 |
| SD 代码划边界 | `engine.py` LEGACY 横幅 + 护栏测试 | 见 §6，删除待确认 |
| 默认参数 | `engine.RECOMMEND["qwen21"]` | 1024²→768²，20→24 步（都有实测依据） |

---

## 2. 实测数据（全部本机 RTX 4090 24GB / 32GB 内存）

> 度量约定：预热后多次取中位数；显存看 `memory_reserved`；改图类指标用
> 「蒙版外最大像素差」，它是硬承诺，必须恒为 0。

### 2.1 文本编码器 fp8 化：峰值 -31%，速度快 25%，内存少 8GB

`text_encoder` 是 Qwen3-VL（8.77B 参数，bf16 16.3GB）。它是**整条管线的显存与
内存双料峰值**——改造前文生图 768² 的保留峰值 16.59GB，基本就是它本身。

做法：`tools/make_qwen_fp8.py` 离线逐输出通道量化成 fp8(E4M3)，落到
`models/checkpoints/qwen-image-2.1/text_encoder-fp8/`（8.8GB），加载时按
`w_fp8` + `w_scale` 挂 buffer、把 Linear 换成 `_FP8Linear`（与 transformer-fp8
同一套机制）。本机 torch 2.6 用不了 torchao 的在线量化（要 ≥2.11），所以只能离线做。

| 指标（768² / 8 步，中位数） | bf16 编码器 | **fp8 编码器** | 变化 |
|---|---|---|---|
| 文生图 | 13.96s / 16.59GB | **10.45s / 11.41GB** | 快 25% / 峰值 -31% |
| 指令改图（条件图 1024²） | 16.12s / 20.14GB | **13.35s / 14.10GB** | 快 17% / 峰值 -30% |
| 指令改图（条件图对齐） | 14.68s / 18.75GB | **11.92s / 11.82GB** | 快 19% / 峰值 -37% |
| 管线加载 | 24.8s | **14.0s** | -44% |
| 磁盘 / 内存占用 | 16.3GB | **8.8GB** | -46% |

**保真度**（同 seed 42、同提示词、12 步，两版出图逐像素比）：
平均差 **1.10/255**，差 >128 的像素占 **0.24%**（笔触细节级差异，结构、五官、
配色完全一致）。不满意可 `MEOW_QWEN_TEXT_FP8=0` 退回 bf16。

### 2.2 条件图分辨率对齐：又一个「默认值陷阱」

`QwenImage21Pipeline` 的 `output_resolution` **默认 1024**，它决定「条件图被缩放到
多大再编码」。原来只传 `width/height`（768）而没传它，于是：参考图按 1024² 编码、
目标按 768² 生成——**模型看到的尺度比要画的大一档**，图 token 还多 78%。

| 768² / 8 步 | 条件图 1024² | 条件图对齐 768² | 变化 |
|---|---|---|---|
| 耗时 | 13.35s | **11.92s** | -11% |
| 峰值 | 14.10GB | **11.82GB** | -16% |
| 输出 | — | 与前者不是同一个采样点（平均差 6.19/255） | 对齐版更贴原图 |

所以 `_qwen_call` 里现在是 `output_resolution=int(output_resolution or max(w,h))`
——**所有走 Qwen 的路径都自动对齐**，包括角色迁移与逐帧工作流。

### 2.3 逐帧尺度归一：可交付率 1/3 → 3/3

Qwen 逐帧「人物画多大」由模型自己决定（当时还没有 ControlNet；现在有了 Union
也没解决这条——它的条件图不带尺度语义，见 §2.15）。同一段 3 帧里主体高度
实测能差 3 倍，最小的那帧前景只占 **2.1%**（精灵产线要求 5~60%），直接判废。

`postprocess.normalize_sequence`：量每帧 alpha 包围盒高度 → 取中位数当目标
（夹在 55%~85%）→ 裁主体、等比缩放、底边对齐 93% 处。**alpha 用 NEAREST 缩放**
（LANCZOS 会在四周长出半透明毛边，下一帧量 bbox 时把毛边算进去，帧高反而逐帧飘 5~7px）。

| 3 帧骨架序列（768² / 20 步） | 归一前 | **归一后** |
|---|---|---|
| DWPose 关节数 | 16 / 0 / 16 | **16 / 18 / 18** |
| 前景占比 | 9.1% / 5.2% / **2.1%** | **19.8% / 14.6% / 11.8%** |
| 可交付（过闸门 ∧ 5~60%） | 1/3 | **3/3** |

### 2.4 条件图灰度化：止住「抄骨架颜色」

`tools/make_pose.render` 画的是**黑底 + 彩色**骨架。Qwen 把它当**视觉参考**而不是
控制信号，于是顺手把颜色抄进成品——实测有一帧真的画了个「红色小帽子」浮在头顶，
那其实是骨架图上红色的鼻子关节点。

`_qwen_cond_image`：转灰度 + 亮度均值 <110 时反相（黑底骨架 → 白底深线）。
模型只能读到「线在哪」，抄不到颜色。改完之后 3 帧全部 blobs=1（前景连通域唯一）。

### 2.5 四个工作流的端到端数字（768²）

| 工作流 | 耗时 | 峰值 | 关键指标 |
|---|---|---|---|
| w3 局部重绘 | 36.5s | 15.4GB | 蒙版外最大差 **0** |
| w5 局部 AI 重绘 | 22.3s | 11.4GB | 蒙版外最大差 **0**，色板吸附 79% |
| w4 姿态序列（3 帧） | 24~26s/帧 | 15.4GB | 关节 16~18，占比 12~20%，可交付 **3/3** |
| wss 序列帧替换（3 帧） | 23.5~23.8s/帧 | 15.4GB | 格子 64²、共享色板 32 色、帧间无漂移 |

w5 的质量验收是人工看的：**盔甲被指令改成翠绿 + 金边，脸、披风、背景逐点未变**。
这正是 P0-2 要的「指令被执行且脸不丢」。

### 2.6 步数下限：**16 步**（旧结论作废，2026-10-30 重测）

**旧结论**（真 CFG 开着时测的）：「12 步没有头、16 步四肢断、24 步才完整」。

**它不成立了**。原因：那时的采样路径是「每个 step 跑两遍 transformer」
（有条件 + 无条件），guidance 把轨迹改写了，低步数会散架。
关掉真 CFG（见 §2.10）之后重测（`tools/exp_steps_wide.py`，12 个提示词 ×
5 档步数，768²，cfg=1，同 seed）：

| 步数 | 中位耗时 | 缺件（前景连通域 > 1） | 贴边 | 关节 < 8 |
|---|---|---|---|---|
| 8 | 7.25s | 0 | 1 | 0 |
| 12 | 8.69s | 0 | 0 | 1 |
| **16** | **10.05s** | 0 | 1 | 0 |
| 20 | 11.51s | 0 | 0 | 0 |
| 24 | 12.71s | 0 | 0 | 1 |

「缺件」在**所有档位都是 0**（8 步也没缺件）。真正的差别在目视：
128px 缩图上 8 步的**浅色主体**（白猫、白甲、幽灵、白机器人）对比度不够、
细节发白，12 步的白机器人也偏淡——**16 步起颜色与细节才稳定**。

姿态贴合度也不受影响（`tools/exp_steps_pose.py`，3 帧，768²，
成品关节 vs 骨架关节的平均距离）：

| 步数 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|
| 偏差 | 119px | **95px** | 110px | 106px |

1024² 立绘档：16 步 14.9s vs 24 步 19.8s，画质几乎看不出差别
（对照图 `assets/_steps_pose/portrait_cmp.png`）。

**结论**：`RECOMMEND["qwen21"].steps = 16`。唯一需要往上调的场景是
**白/浅色主体**（白猫、白甲、幽灵）——那类目标在低步数下对比度不够，
用 20~24 步。

### 2.7 「透明底」为什么没生效，以及抠图吃掉脸的三条链路

现场问题：**有些抠图把人物的脸抠掉了**，而 Qwen 明明支持透明背景。
查下来是三件事叠在一起，没有一件是模型不行。

#### (1) 提示词里写了白底 → alpha 变成一层雾

透明声明（`QWEN_ALPHA_HEAD/TAIL`）只说了「这是带 alpha 的图」，但正文里如果还
写着 `plain white background`，那是一个**和它直接冲突的正面要求**。同 seed 42、768²：

| 正文 | 前景占比 | 全透明像素 | 半透明中间值(16~240) |
|---|---|---|---|
| 带 `plain white background` | 81.5% | 3.4% | **77.5%** ← 雾 |
| 不带 | 31.0% | 67.8% | **2.9%** ← 干净 |

→ 现在 `transparent=True` 时自动清掉背景措辞（`_strip_bg_words`，连"a cat on a
white background" 删剩的悬空介词也一起收），改动记进 `meta.prompt_bg_stripped`。

#### (2) 已有 alpha 被无视，还要重抠一遍

`matte_auto` 以前**从不看输入的 alpha**，一律重新分割。Qwen 出图已经带 VAE 的
第 4 通道，重抠纯属自找麻烦。现在先过 `matte.alpha_trust`：

```
a0  = 完全透明像素占比      ≥ 10%
mid = 16~240 中间值占比     ≤ 12%
fg  = 前景占比              1% ~ 95%
```
三条都满足就**原样复用**（`meta.matte_used="alpha"`、`alpha_reused=True`，
与原 alpha 的 IoU = 1.000），雾状 alpha 才会真的重抠；显式选 `u2net`/`color`
后端则不走这套判断。

#### (3) 重抠时送进去的图是坏的

`matte_ai` 原来直接 `img.convert("RGB")` —— 那只是丢掉 alpha，透明区的 RGB
仍是 VAE 解出来的噪声色（偏黑/偏品红），分割模型把这片噪声当内容看。改成
`flatten_white`（合成到白底）后再分割，另外补 `fill_holes`：把主体**内部**被
误判成背景的洞补回实心（不与画布边界连通的才算洞，两腿之间的空隙不会被填）。

#### 但白毛 + 白底这件事本身不可靠（已知限制）

同一张白猫骑士（白毛白甲白底），两种喂法各丢不同部位：

| 喂给 isnet 的图 | 前景 | 丢哪 |
|---|---|---|
| 直接 `convert("RGB")`（透明区噪声色） | 20.9% | 头顶切角、胸口挖洞 |
| `flatten_white`（纯白底） | 15.6% | 头顶与胸口正常，**两条腿被吃**（腿也是白的） |

`fill_holes` 能把「内部洞」补回来，但**边界上被切掉的部分补不回来**。
所以正确处理不是调参，而是：**这类素材别走抠图**——出图时直接勾「透明底」，
alpha 让模型给。抠图是 SD 时代的补丁（那时模型不输出 alpha，只能事后猜）。

对照实验的另一半：给逐帧工作流（w4/wss）也开透明底**不行**——实测骨架图上
灰黑色的骨骼线会被画进成品（角色身上多出一根根棍子），因为模型把条件图当成了
要渲染的内容。逐帧继续用白底 + 抠图，白毛角色的限制在那里仍然存在。

### 2.8 帧间一致性实验：「首帧锚定」不可取，真正的短板是姿态没传过去

**假设**：Qwen 逐帧没有时序注意力，把**首帧成品**当第三张参考图（角色 / 骨架 /
上一帧）喂进去，模型会继承上一帧的光影与配色，帧间更稳。

**方法**：`tools/exp_frame_anchor.py`，3 帧骨架、同 seed、同参数，只改参考图数量。

| 方案 | identity_drift | silhouette_drift | 每帧 |
|---|---|---|---|
| 现状（角色 + 骨架） | 0.0001 | 0.0529 | 27s |
| 首帧锚定（角色 + 骨架 + 上一帧） | 0.0000 | **0.0051** | 28s |

剪影漂移降了 10 倍，但**目视观察是负收益**：三帧动作幅度被压平，几乎成了同一张图。
「更稳」的来源是模型倾向复制上一帧，而不是更准地跟随骨架。逐帧要的是**动作**，
所以锚定不进产线（脚本保留，随时可复跑）。

**更重要的发现（顺手测出来的）**：w4 的姿势贴合度没有之前认为的那么好。
早前「姿势几乎逐点一致」的判断依据是「DWPose 能检出 18 个关节」——那只证明
「画出了一个能被姿态模型认出的人形」，**不证明姿势和骨架一致**。

这次量的是「成品检测到的关节点」与「骨架关节点」的归一化坐标平均距离
（768² 画布，只比躯干与四肢 0~13 号点；踝距指标不可靠，检测器对像素猫的踝点
会退化到同一点，不能用来判断迈步）：

| 变体 | 关节平均偏差 | 目视 |
|---|---|---|
| 现状（角色图 + 骨架） | 108px | 双腿并拢站立，骨架的迈步没体现 |
| 加强指令（"pose must come from the second image"） | 97px | 仍并拢 |
| 骨架加粗 2.6 倍（stick 12→32） | 118px | 仍并拢 |
| 交换图序（骨架在前） | 104px | 仍并拢 |
| **去掉角色图**（只给骨架 + 文字描述） | **67px** | **腿真的迈开了** |

结论：**参考图在抢姿态**。角色图里是站立姿势，模型优先照抄它；骨架图只是弱信号。
「去掉角色图」能把姿态拿回来，代价是身份漂移（配色/披风长度变化）且要重新描述角色。

### 2.9 两阶段（草图 → 换皮）：姿态偏差 -25%，代价是慢一倍

上一节说的「没做的下一步」——两段式——2026-10-30 做了（`tools/exp_two_stage.py`）。

**做法**：把「定姿态」与「定身份」拆成两次生成。

```
阶段 1（草图）  只给骨架 + 一句角色描述        → 姿态准，身份只是个「差不多的角色」
阶段 2（换皮）  角色图 + 草图 → 指令改图        → 身份回来，姿态已被草图钉住
```

3 帧 × 768² / 24 步，同 seed。偏差 = 成品关节与骨架关节的平均距离（越小越贴）：

| 变体 | 姿态偏差 | 配色一致 | 白袍像素占比 | 每帧 |
|---|---|---|---|---|
| 现状（角色图 + 骨架一起给） | 102px | 0.46 | 0.11 | 33s |
| 两阶段（草图在前） | 77px | 0.45 | 0.04 | 51s |
| **两阶段（角色图在前）** | **77px** | **0.49** | 0.04 | 50s |

三条结论：

1. **两阶段确实把姿态拿回来了**：偏差 102px → 77px（-25%），目视腿真的迈开了。
2. **图序是个旋钮**：角色图在前（=「画谁」在前）时配色保持最好（0.49），
   草图在前时姿态一样准但身份漂。采纳「角色图在前」。
3. **代价有两笔**：每帧多一次生成（33s → 50s，+50%）；以及**服装细节会漂**——
   白袍被画成金甲（白袍像素占比 0.11 → 0.04）。所以它是「要精确动作就开」的开关，
   不是默认。

**顺带发现**：w4 的前端**从来不传角色参考图**——角色只靠提示词文字描述。
也就是说上面 §2.8 那批「参考图抢姿态」的实测（108px 等）是在**手工传了角色图**
的前提下做的，对默认用法并不成立（默认用法就是「去掉角色图」那档，偏差 67px）。
现在给 w4 节点补了「角色图（可选）」输入与「两阶段」开关，两种模式都能选。

**度量上的坑**：第一版身份度量用「12 色调色板是否相同」，结果两两都是 1.00——
区分度为零，看不出「整身金甲 vs 白袍」这种明显差别。换成 16 色直方图相似度
才有区分度。**度量本身要先验证它能分辨**，否则整轮实验的结论都是假的。

w4 / wss 的「近似」标记继续保留：两阶段只是把偏差从 102 降到 77，
离「逐点一致」仍然差得远。

---

### 2.10 速度：瓶颈在哪，两个真旋钮（单帧 21.1s → 10.1s，-52%）

**先拆账，再优化**（`tools/bench_qwen_profile.py`，768²/24 步，中位数）：

| 段 | 耗时 | 占比 |
|---|---|---|
| **DiT 前向** | **18.84s** | **86%** |
| 文本编码 | 1.79s | 8% |
| VAE 解码 | 1.12s | 5% |
| 权重搬运（cpu offload） | 0.12s | 0.5% |

于是两条「直觉上很贵」的路直接排除：**fp8 文本编码器只占 8%**，
**cpu offload 的搬运几乎不花钱**（0.12s）。要提速只能动 DiT 前向。

#### 旋钮一：真 CFG —— 每个 step 跑两遍 transformer

`true_cfg_scale > 1` 且给了负向词时，管线每个 step 跑**两遍** transformer
（有条件 + 无条件）→ 24 步 = 48 次前向。这就是「一帧 25~30s」的真正原因。

关掉（`true_cfg_scale = 1.0`，只跑一次前向）：

| 配置 | 耗时 | 前向次数 |
|---|---|---|
| 真 CFG（cfg=4） | 21.1s | 48 |
| **单次前向（cfg=1）** | **13.6s** | 24 |

画质对照（`tools/exp_cfg_ab.py`，6 个提示词 × 768²/24 步，同 seed）：
骑士 / 机甲 / 史莱姆 / 狼剑客 / 红袍法师 / 龙战士，
产线判据（单主体、贴边、前景占比、关节数）全过，目视也不降，
关掉真 CFG 的那列甚至更精细（对照图 `assets/_cfg_ab/cmp.png`）。

**所以默认 cfg = 1.0，负向词只在 cfg > 1 时才传**（不依赖 diffusers 内部
`do_true_cfg = scale > 1 and has_neg_prompt` 的判定，行为自己说得清）。
用户要更强的负向约束时把 CFG 调到 > 1，代价是时间翻倍。

顺带修了一个一直存在的断链：`txt2img(cfg=...)` / `img2img(cfg=...)`
**从来没把 cfg 传到内部实现**（`_qwen_txt2img` 的签名里压根没有 cfg，
meta 里写死 4.0）——也就是说界面上那个 CFG 输入框从来就没生效过。

#### 旋钮二：步数 24 → 16（见 §2.6）

#### 组合起来

| 配置 | 单帧耗时 |
|---|---|
| 起点（真 CFG + 24 步） | 21.1s |
| 关真 CFG | 13.6s（-36%） |
| **再把步数降到 16** | **10.1s（-52%）** |

#### 试过但没采用的

- **bf16 transformer 代替 fp8 缓存**：fp8 每层前向都要把权重反量化成 bf16，
  实测 DiT 前向 18.84s（fp8）vs 17.23s（bf16）——bf16 快 8.5%，
  但总耗时持平、显存 +2.3GB（11.41 → 13.74GB）。保持 fp8：内存更省，
  批量产线的余地更大。
- **512² 出图**：8.9s（比 768² 快 32%），但 **DWPose 在 512² 上经常检不出关节**
  （白猫骑士 0 个关节、法师 7 个）→ 质量闸门会误判。**不要低于 640²**。
- **torch.compile / flash-attn / triton**：本机 Windows 环境**没有 triton**
  （inductor 的快路径依赖它），也没有 flash-attn（Windows 无官方包）。
  注意力走的是 diffusers 默认的 SDPA，已经不慢。
  实测 `torch.compile(pipe.transformer, mode="reduce-overhead")` 直接报错：
  `Cannot find a working triton installation`。
- **蒸馏权重（Qwen-Image-Lightning，4/8 步）**：这是剩下唯一的大杠杆
  （4 步 + 无 CFG ≈ 现在的 1/2.5），权重只有 0.79GB（bf16）。
  **卡在下载速度**：HF 直连超时（`ConnectTimeout`），走 hf-mirror 只能到
  **81MB / 15 分钟（≈90KB/s）** → 0.79GB 要 2.5 小时。
  已把断点留在 `models/_lightning/.cache`（huggingface_hub 能续传）。
  另有一个未验证的风险：Lightning 的权重是按**官方 20B 版 Qwen-Image** 训的，
  本机这份 `qwen-image-2.1` 的 transformer 只有 13.3GB bf16（约 6.6B 参数），
  LoRA 的键名/形状能不能对上还没验证。

### 2.11 真·蒙版局部重绘（推翻 §4.1）

**一句话**：Qwen-Image-2.1 **能**做「涂哪改哪」，diffusers 只是没封装。
我们用管线现成的 `latents=` + `callback_on_step_end` 复刻了 ComfyUI 的做法，
w3 / w5 从此不再是「近似实现」。

#### 原理

ComfyUI 的 `Qwen Image 2.1 Inpainting | No-Offset Edits` 用**标准 2.1 权重**
（没有专用 inpaint 权重）实现，靠两件事：

1. `SetLatentNoiseMask`：初始 latent 里蒙版内换成噪声
2. 采样循环里的 `denoise_mask`：**每一步**把蒙版外拉回原 latent
   （`comfy/samplers.py:641-642`：`out = out * denoise_mask + latent_image * latent_mask`）

第 2 步是**采样器级**机制，与权重无关。核心公式（flow matching）：

```
x_t = (1 - σ) · x0 + σ · ε
```

`x0` 是原图的 VAE latent，`ε` 是**固定**的那份初始噪声，`σ` 是当前噪声级。
`σ=1` 时就是纯噪声（管线的正常起点），`σ→0` 时趋于 `x0`——模型会把蒙版外
还原成原图。

#### 实现

新模块 `workflows/qwen_mask.py`，核心 60 行。三个坑（都踩过）：

1. **pack / unpack 必须用管线自己的 `_pack_latents` / `_unpack_latents`**。
   Qwen-Image-2.1 的 latents 是 **2×2 patchify** 过的（管线里那句
   `latents.shape[1] // 4` 就是这个意思），手写 `transpose+reshape` 会让
   空间布局错位 → 输出满屏彩色条纹。
2. **条件图 latent 必须与管线内部算的那一份逐字节一致**。管线会用
   `output_resolution` 经 `calculate_dimensions` 重算条件图尺寸（不一定是
   你传的 w×h），自己按 w×h 编码会对不上 → 同样彩色条纹。
3. **VAE 只吃 RGBA（4 通道）**。给 RGB 报
   `expected input to have 4 channels, but got 3`——Qwen 的 VAE 原生透明底，
   不是 bug。

#### 实测（本机 RTX 4090 24GB）

| 项 | 真蒙版（`mask`） | 旧路径（`whole`） |
|---|---|---|
| 蒙版内指令执行 | ✅ 选区正确变绿 | ❌ 常不执行（靠方位词猜） |
| 蒙版外 | 最大像素差 **19~46**（视觉无差） | **0**（逐点不变） |
| 身份保持 | ✅ 脸/背景不动 | ✅ |
| 耗时（768²/20 步） | ~15s（探针）/ ~34s（含模型加载） | ~30s |
| 峰值显存 | 15.4GB | 15.4GB |

**工作尺寸下限是 768**：实测三档（同一素材、同一提示词、20 步）——
512²（latent 32×32）→ **选区没被改**（蒙版糊成一个点）；
768²（latent 48×48）→ 正确；1024² → 正确但慢 1.5 倍。
所以 `qwen_mask` 内部把工作尺寸夹到 768 起。

#### 承诺的变化（要诚实）

- 蒙版外**不再是逐点为 0**，而是「视觉上完全一致」（差异来自 VAE 编码-解码
  本身）。ComfyUI 官方也说明「不保证像素级完全相同」。
- 换来的是**蒙版内指令被准确执行**——这才是局部重绘的意义。
- 旧路径保留为 `strategy="whole"`，需要逐点为 0 时仍可用。

#### 影响

- `engine._APPROX["qwen21"]` 从 4 项减到 2 项（只剩 `poseswap` / `sprite_swap`）
- 前端不再给 w3 / w5 打「近似」角标
- 游戏素材价值：**改装备配色、改披风颜色、给角色加配件**这类操作
  从「靠提示词求模型」变成「模型看得见蒙版」，可靠度完全不同

### 2.12 调研：RunComfy 的 10 个 Qwen-Image-2.1 工作流

> 来源：https://www.runcomfy.com/comfyui-workflows/models/qwen-image-2.1
> （2026-10-30 逐页抓取，含 ComfyUI 源码核实）

| # | 工作流 | 核心 | 对我们项目的价值 | 结论 |
|---|---|---|---|---|
| 1 | Text-to-Image | INT8 权重、2048² 原生 | 我们跑 768²/24 步；官方推荐 **2048²/40 步**。2K 是原生分辨率不是拉伸 | **可采纳**：立绘/背景这类需要细节的走 2K；精灵帧仍用小图 |
| 2 | Turbo Edit | 8 步 INT8 编辑 | 速度。官方 Turbo 权重 `Qwen/Qwen-Image-2.1-Turbo` | **待验证**：见 §2.13 |
| 3 | Turbo Text-to-Image | 8 步 | 同上 | 同上 |
| 4 | Outpainting | 灰底 padding + **Outpaint LoRA** | **扩图**：场景/背景/图块扩展，游戏素材刚需（现在完全没有这个工作流） | **可采纳**：LoRA 152MB，hf-mirror 可下 |
| 5 | VNCCS PoseStudio | 3D 姿势编辑 | 我们已有更好的方案（双图条件 + 伪 3D 骨骼编辑器） | **不采纳** |
| 6 | **Inpainting（no-offset）** | **标准权重 + SetLatentNoiseMask** | **推翻 §4.1** | **已实现**（§2.11） |
| 7 | Multi-Image Editing | 最多 **10 张参考图** | 我们目前只用到 2 张（角色 + 骨架）。10 张能做「多角色同框 + 场景拼装」 | **可采纳**：管线原生支持，改前端即可 |
| 8 | Face & Head Swap | 两图 + 指令 | 换脸/换头。游戏里做「同一角色不同立绘」 | **可采纳**：本质是多图编辑，我们已有 |
| 9 | Outfit Change | 两图 + 指令 | **换装备**——游戏素材高频需求（同一角色不同服装） | **可采纳**：同上 |
| 10 | Image Editing | 基础图 + 可选参考图 + 指令 | 我们的 w2 已经是这个 | 已有 |

**关键发现**：这 10 个工作流里，**没有一个需要我们没有的底模**——
全部跑在标准 Qwen-Image-2.1 上。它们多出来的是**节点编排**，不是模型能力。
ComfyUI 生态的价值在于「把能力拼成工作流」，而我们做的是「把能力拼成产品」。

### 2.13 Lightning 少步蒸馏（已实测，速度 -46%）

`lightx2v/Qwen-Image-2.1-Lightning`（64MB，**diffusers 原生格式**，可直接挂）
是给 Qwen-Image-2.1 的少步蒸馏 LoRA——正好补上我们一直缺的「少步蒸馏」那一条
（§2.10 手动把步数从 24 降到 16 已经是一次提速，但画质也掉了）。

**实测（768² / RTX 4090 24GB）**：

| 配置 | 耗时 | 画质 |
|---|---|---|
| 24 步 无 LoRA（基线） | 17.3s | 基准 |
| **8 步 + Lightning** | **9.3s（-46%）** | **几乎不掉**（边缘锐利、细节完整） |
| 4 步 + Lightning | 7.4s（-57%） | 仍然干净 |
| 8 步 + Lightning（强度 0.7） | 8.9s | 与 1.0 无明显差别 |

**意义**：我们逐帧路线（w4 姿态序列 / wss 序列帧替换）的瓶颈一直是速度
（实测 37s/帧，8 帧 ≈ 3~9 分钟）。8 步 Lightning 直接把单帧压到 9.3s，
**8 帧 ≈ 75s**——逐帧从"不现实"变成"可用"。

**挂载**：侧栏「LoRA」分组有「**快速模式**」开关（推荐），勾选即挂 + 步数压到 8。
或 `POST /loras/apply {"items":[{"name":"qwen21_lightning","scale":1.0}]}`，
或 `POST /fast_mode {"on": true}`（一步到位）。

**meta 记的是实际步数**（`_eff_steps`）：快速模式把步数压到 8，meta 里就写 8，
不写用户请求的 24 —— 产线判据（每可用帧成本）看的是真正跑了多少步。

**端到端实测**（走 `generate.txt2img` 全链路，含加载）：
- 普通：14.1s（24 步）
- 快速：8.5s（8 步 + Lightning）→ **-40%**

### 2.14 三个新工作流：扩图 / 换装 / 换脸（Qwen 从 10 个变 13 个）

RunComfy 那 10 个工作流里还有三件值得做的，已全部落地。

#### 扩图（`wout`）

游戏素材刚需：方形背景 → 16:9 横版、地图块延展、立绘左右留白。

做法（ComfyUI 的 Outpaint 方案）：
1. 源图贴到更大画布，空白填**中性灰 #808080**
   —— 灰色本身就是"这里要补内容"的信号；填别的颜色会**污染新区域色调**
2. 挂 **Outpaint LoRA**（`ausboss/Qwen-Image-2.1-Outpaint-LoRA`，152MB，
   ComfyUI 格式，已用 `tools/convert_lora.py` 转好）
3. 提示词**必须**以 LoRA 触发句开头（`outpaint.OUTPAINT_TRIGGER`）：
   `Outpaint the image: replace the solid gray areas with a seamless
   continuation of the scene, keeping the existing picture unchanged.`
4. `blend=True` 时把**原图区域**软边贴回（扩图的语义是"扩展"，原图不该变）

实测（8 步）：森林场景 768² → 右扩 50% 成 1152×768，14s。
右侧延续了森林/地面/雾气，**无灰块残留、无可见接缝**。

**坑**：画布尺寸必须对齐 **32**（不是 16）。管线里
`multiple_of = vae_scale_factor * 2 = 32`，不是 32 的倍数它只 warn 一句就
**静默 resize**——于是 meta 里记的画布尺寸是假的、下游算的偏移全错。

#### 换装（`woutfit`）/ 换脸（`wface`）

两者本质相同：**两图 + 结构化指令**，不需要任何专用权重——
Qwen 的多图条件会自己分配"谁是身份、谁是服装"。

关键是两点：
- 指令里写清楚**每张图管什么**（`swap.OUTFIT_PROMPT` / `FACE_PROMPT`）
- 用**官方的多图引用语法** `<image1>` / `<image2>`（管线会自动插占位符），
  比"第一张图"这种自然语言描述更稳

| | 图1（基底） | 图2（参考） | 保留 | 替换 |
|---|---|---|---|---|
| 换装 | 人物 | 服装 | 身份/姿势/背景 | 剪裁/颜色/材质 |
| 换脸 | 场景人物 | 身份肖像 | 姿势/服装/背景 | 脸/发型/胡须 |

实测（16 步，各 ~8.8s）：
- 换装：人物穿上红皮夹克（含金扣），身份、姿势、场景全部保留
- 换脸：头变成银发绿眼，灰色 T 恤/短裤/姿势/场景全部保留

#### 多图上限（10 张）

| 张数 | 耗时(8 步) | 峰值 |
|---|---|---|
| 2 | 10.2s | 12.6GB |
| 4 | 13.5s | 15.2GB |
| 8 | 18.2s | 21.3GB（接近 24G 上限）|
| **10** | **12.3s** | **14.1GB** ✅ |

10 张能跑通的关键：**条件图编码分辨率可以与输出分辨率解耦**
（`output_resolution=448` 而输出仍是 768²）。默认 768 编码时 8 张就到
21.3GB，10 张会超。这条以后做「多角色同框 / 场景拼装」要用到。

### 2.15 ControlNet Union：已端到端跑通，控制生效（2026-10-30）

`alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union`（7.55GB）是**专为 2.1** 的
控制分支（不是 pixelforge 那个 20B 专用的 union）。它覆盖 8 种结构条件
（Canny / Depth / Grayscale / HED / Lineart / MLSD / Pose / Scribble）
**且支持蒙版 inpaint**。

#### 关键结论：不需要 VideoX-Fun 的代码

官方参考实现是 VideoX-Fun 的 `config/qwenimage21/qwenimage21_control.yaml`，
但**权重是纯 safetensors，结构能从 key 名和形状完全推出来**。我写了
`workflows/controlnet_union.py` 并做了**逐 key 验证**：

```
官方 key 数: 180        模块 key 数: 180
缺: （无）              多: （无）        形状不匹配: （无）
skip 数: 16             每个形状: (1, 2304, 4096)
✅ 100% 对齐官方权重
```

3 个必须踩准的对齐点：
1. `img_mlp` 的命名是 `proj / gate_layer / out`，与 diffusers 的
   `QwenImage21SwiGLUFeedForward` **逐字一致**（不是 `gate_up` 那种合并式）
2. `12288 = 4096 × 3` → `mlp_ratio = 3.0`
3. `norm_q/norm_k` 形状是 `[128]`（head_dim），不是 `[4096]`
4. `before_proj` **只有 block 0 有**（不是每块都有）——第一版放顶层就错了

#### 它是双流（VACE 风格）—— 第一版做错就崩在这

官方 ComfyUI 实现（PR #16519，`comfy/ldm/qwen_image21/model.py`）是**双流**：

```python
def init_stream(self, x, control, prefix_len):
    joint = torch.zeros_like(x)
    joint[:, prefix_len:] = self.control_img_in(control)   # 只填图像行
    return self.control_blocks[0].before_proj(joint) + x    # ★ 要加回 base hidden

def step(self, index, c, mod, pe, attn_fn, ...):
    c = block(c, mod, pe, attn_fn, ...)      # 完整 block，吃 modulation
    return c, block.after_proj(c)
```

```python
inj = ControlInjection(transformer, branch, scale=0.6)
inj.set_condition(control_latents)   # [1, 129, H/16, W/16]
... 跑 pipe(...) ...
inj.remove()                          # 必须卸，否则污染后续生成
```

两个要害：

1. **control block 是 base block 的完整副本**（官方 `QwenImage21FunControlBlock`
   直接继承 `QwenImage21TransformerBlock`），吃 timestep 调制（`modulation`，
   实测形状 `[2, 16384]`）和 rotary embedding（`[L, 64]` complex64）。
   参数上没有 norm 权重，因为 base 的 `img_norm1/2` 是
   `elementwise_affine=False`（无参），modulation 复用 base 的。
2. **每一步都重跑整条控制流**，skip 因此随 timestep 变化。

第一版按「算一次、整条采样链复用」做（`set_condition` 一次算好 16 个 skip），
skip 就成了常量偏置：每步叠加一次，低噪声后期相对占比越来越大。
症状非常典型——**8 步勉强能用、24 步必崩**，24 步 / scale 0.6 直接出骨架
本身的彩色线段（模型在照抄条件图）。改成双流后累积效应消失：

```
各步首层 skip absmean（24 步）:
[0.352, 0.285, 0.279, 0.277, 0.275, 0.273, 0.273, 0.275, 0.277,
 0.279, 0.283, 0.287, 0.293, 0.299, 0.305, 0.311, 0.316, 0.326,
 0.336, 0.346, 0.355, 0.367, 0.379, 0.393]     ← U 型，随步变化（旧实现是恒定值）
```

#### 六个必须踩准的坑（都是实测踩出来的）

**① 有 control 时必须关 KV cache** ← 最隐蔽的一个

pipeline 的 `use_kv_cache` 默认 `True`：它把文本 token 当 prefix 缓存，
之后每个 block 只处理**图像** token（实测 768² 下 2304 个），
而 control 链算的是**完整 joint**（2331 = 2304 图像 + 27 文本）—— 两者对不上。
文本 token 根本不经过 block，hint 只能硬塞进图像段，信号全错。

官方实现里就有这个判断：
`cache_enabled = use_kv_cache and ... and control_context is None`。
我们这边要显式传 `use_kv_cache=False`。

**② mask 通道填 0（全重画），不是 1**

官方 pipeline 是 `mask_latent = 1 - mask_condition`（白=重画），
所以这通道 **1 = 保留原图，0 = 重画**。填 1 等于告诉模型
"整张图保持 `masked_image`"，而纯结构控制时 masked_image 是空的
→ 模型老实输出一张空图。实测填 1 时 RGBA 不透明占比 **0.0%**（整张全透明）。

**这个坑最坑的地方是度量会骗人**：填 1 时"与对照组的像素差"反而更大（40→80），
看着像"控制力翻倍"，其实是画面在往空白塌。**像素差不能单独当判据**。

**③ prompt 必须描述完整画面 + 画布比例必须匹配内容** ← 决定成败的两环

官方示例（`examples/qwenimage21_fun/predict_t2i_control.py`）用的是
**`sample_size = [1728, 992]`（竖版 9:16）** + **几百字的中文详述 prompt**。
这两条我都没照做，于是花了整晚在一个错误配置上调参数。

**画布比例的影响**：768² 方形下，模型倾向把人物撑满画布，顶端的人头会被
挤出画面——表现是"身体、四肢、比例全对，就是没有头"，透明底时那块完全
透明。换 768×1024 竖版后，同一个骨架、同一个 prompt，头立刻回来了。
这不是 ControlNet 的锅：方形画布本来就没给"全身像"留出头部空间。

试过但**无效**的补救（都别再试）：
- 把骨架线加粗到 32px、加实心肤色圆当头部、白底/灰底互换
- 把骨架缩到占 72% 高度并居中（模型仍然把人物撑满）
- 顶部 25% 图像行不注入控制（`head_cut`，见下）
- 关掉透明底走 RGB

**prompt 的影响**：只写 "a 2D game character sprite, centered" 这种泛化描述时，
模型不知道要画什么，**直接把骨架原样画出来**——实测输出是一根粗黑线条人
或淡灰线条人。给足描述后同一个 seed 立刻出完整角色。官方那段 prompt 写到了
"亮紫色长发、两侧黑色蝴蝶结、纯白无袖连衣裙、单手叉腰、目光直视镜头、
湛蓝海水波光粼粼、右下角粉紫小花丛"这种颗粒度。

**④ 条件 latent 必须做 mean/std 归一化**

管线的 `_encode_vae_image` 是 `(x - latents_mean) / latents_std`
（`pipeline_qwenimage21.py:448`）。漏掉这一步，尺度差几倍，
16 层 skip 直接把 hidden 打成 NaN，出图**全黑**。

**⑤ control block 要继承而不是组合**

`QwenImage21TransformerBlock` 的 state_dict 只有 **9 个 key**
（attn 6 + img_mlp 3，norm 无参），对账：

```
16 × 9 (block) + 16 × 2 (after_proj) + 2 (before_proj，仅 0) + 2 (control_img_in) = 180
```

如果包一层再挂投影，key 会变成 `block.attn.to_q.weight`，多出 `block.` 前缀
就对不上了。必须 `class ControlBlock(QwenImage21TransformerBlock)`。

**⑥ `init_stream` 要加回 base 的 hidden**（`+ x`）

只把 control 投影进 joint 序列、不与 base hidden 相加，控制流就与 base 的表示
脱节——第一版正是漏了这个（还自己补了个无参 LayerNorm 去压残差，属于治标）。

> 附带一条：joint 序列里图像 token 是不是连续后缀？实测 **是**
> （2304/2350，`idx.diff()==1` 全 True），所以前缀切片也能用。
> 但 diffusers 源码注释写着 "the two are interleaved, so the mask cannot be
> sliced off as a prefix"，多图/参考图场景未必成立，所以 `init_stream` 现在
> 优先用 `image_pad_mask` 精确散射（从 transformer 入参 `img_mask` 复算），
> 拿不到才退回前缀。

#### 强度标定

**按官方配置（竖版 768×1024 + 几百字详述 prompt）全程跑 40 步、scale 1.0
完全正常**，出的是完整角色（有脸、有头发细节、姿态贴骨架）。

之前那张"强度表"是在**方形画布 + 短 prompt**下测的，结论已经作废——
那个配置下无论怎么调，要么糊要么空白，因为瓶颈根本不在强度。

| 配置 | 结果 |
|---|---|
| 竖版 768×1024 / 40 步 / 1.0 / 详述 prompt | ✅ 完整角色，姿态贴合 |
| 768² 方形 / 24 步 / 0.45 / 详述 prompt | 有身体没头 |
| 768² 方形 / 24 步 / 0.65+ | 过曝/空白 |

**本项目的 2D 角色是竖构图，就该用竖画布**（768×1024 或 576×1024），
不要图省事用方形。

#### 实测数据（RTX 4090 24GB / 768²）

| 项 | 数值 |
|---|---|
| 权重加载 | 26s（7.55GB，bf16） |
| 条件编码 | 1.2~2.0s（含 VAE encode + 归一化） |
| 控制流前向（16 个完整 block / 步） | 每步都跑，8 步共 128 次 block |
| 显存 reserved | 7.4GB（base 走 cpu offload） |
| 显存 peak | 18.7GB |
| 8 步 + 控制 | 9.4s（旧的一次性实现 7.5s；双流多跑 16 个 block ≈ +25%） |
| 24 步 + 控制 | 17.3s |

**显存结论：24G 卡够用**，不需要把 control 分支 fp8 化（峰值 18.7GB，
余量约 5GB）。

#### 控制效果：复刻官方配置后完全可用

按官方配置（**竖版 768×1024 / 40 步 / scale 1.0 / 几百字详述 prompt**）实测：

- 无控制：一张完整的紫发女孩，姿态由模型自己定
- 带控制：**同样是完整角色**——有脸、有蝴蝶结发饰、白裙、海边背景，
  姿势（双手向两侧下方张开、双腿分开）与骨架逐点对应

对比之前方形画布的结果（有身体没头），差别全在画布比例和 prompt 颗粒度上。

**一句总结**：这个 Union 分支天生对"配置"敏感——画布比例、prompt 详略、
步数/强度三者要一起对，单独调任何一个都调不出来。

#### 它对我们的价值

- **姿态硬约束**：w4 姿态序列 / wss 序列帧替换的短板正是"姿态只能软引导"，
  union 的 Pose 条件能把它变成硬约束。**接入后要重测 108px 的姿态偏差**
- **真 inpaint 的第二条路**：现在的 `qwen_mask` 是采样器级近似，
  union 的 mask 通道是**训练过的**那条路，质量可能更好（待对比）

#### 权重下载：并行分段（单连接被限速）

单连接走代理只有 475 KB/s，7.5GB 要 4 小时。**16 路并发能到 6.0 MB/s**
（12.6 倍），16 分钟下完。工具是 `tools/fetch_parallel.py`：

```bash
python tools/fetch_parallel.py <url> models/controlnet/qwen21_union.safetensors \
  --jobs 16 --chunk-mb 24
```

断点续传：每段独立 `.part` 文件，中断后重跑跳过已完成的段，全下完按序拼接。
它会把已有文件当作第 0 段的前缀（兼容之前 `curl -C -` 顺序续传的字节）。

### 2.16b VNCCS PoseStudio：**不用 ControlNet** 的姿势迁移路线（2026-10-31）

> 这一节推翻了我对"姿态控制必须接 ControlNet"的前提。

社区号称"完美保持角色一致性"的那个 RunComfy 工作流，实际做法是
**专用 LoRA + 双图指令**，全流程没有一个 ControlNet 节点。

#### 官方工作流的真实构成

从 `MIUProject/VNCCS_PoseStudio_QI2.1` 仓库里的
`VNCCS_PoseStudio_QI2.1.json` 读出来的节点清单（14 个）：

```
LoadImage              角色图（image_2）
VNCCS_PoseStudio       3D 摆姿势 → 渲染骨架（image_1）
TextEncodeQwenImage21  双图 + 指令 → 正/负条件
LoraLoaderModelOnly    VNCCS_QI2_PoseStudioV1.safetensors, strength=1.0
QwenImage21Cache       缓存加速（gpu/int8）
KSampler               steps=25, cfg=1, euler, simple, denoise=1
EmptyLatentImage ← GetImageSize(角色图)   ← **画布尺寸取自角色图**
VAELoader / VAEDecode / UNETLoader / CLIPLoader / SaveImage
```

**固定的指令模板**（逐字）：

```
replace the pose of <image 2> with the pose of <image 1>. keep the character of <image 2>
```

`image 1` = 骨架，`image 2` = 角色图；negative 是空字符串。

#### 我们的实现（`tools/verify_vnccs_pose.py`）

LoRA 152MB，ComfyUI 格式（`diffusion_model.` + `lora_A/B`，rank 32），
用 `tools/convert_lora.py` 转成 diffusers 格式（384 张量）后直接
`load_lora_weights`。

实测（24~25 步、cfg 1、LoRA 1.0）：

| 指标 | ControlNet Union 路线 | **VNCCS LoRA 路线** |
|---|---|---|
| 画风 | ❌ 会变（灰白/塑料感） | ✅ 与参考图一致（2D 平涂+黑轮廓） |
| 头部 | ❌ 方形画布下整个丢失 | ✅ 完整，脸清晰 |
| 肢体 | ⚠️ 偶发多一条腿 | ✅ 只有两条腿 |
| 服装/发型 | ⚠️ 部分丢失 | ✅ 皮甲缝线、腰带、护膝、靴扣、低马尾全保住 |
| 姿态精度 | ✅ 逐点贴合 | ⚠️ 能跟（交叉腿成立），但不如硬约束精确 |
| 耗时 | 17~51s | **18.6~30.4s** |

**结论：要"保角色一致性"，走 VNCCS LoRA 这条路；ControlNet Union 适合
"只要结构、不在乎外观"的场合。**

#### 两个必须踩准的细节

**① 画布要取骨架的比例，不是角色图的**

官方是"输出尺寸取角色图"，但那是建立在"角色图和骨架同为竖版"的前提上。
我们的角色图是 768² 方形、骨架是 512×768 竖版，取方形画布时骨架只能缩在
中间，模型把它理解成"小幅度动作"——**双腿交叉退化成双腿收窄**。
改成取骨架比例（→ 768×1152）后交叉腿立刻成立。

#### V1 vs V2（2026-10-31）

仓库 sibling 列表里有 `VNCCS_QI2_PoseStudioV2.safetensors`，但**实际下载 404**
（`Entry not found`，作者列了没传）。不过 `VNCCS_QI2_PoseStudioV1.1.safetensors`
的 metadata 里写着 `ss_output_name: VNCCS_QI2_PoseStudioV2_copy`，训练 step 1500 /
epoch 51（ai-toolkit 0.13.19）——**它就是 V2**。

权重对比：384 个张量 key 完全一致，但**没有一个张量数值相同**，
整体余弦相似度 0.959、单张量最大相对差 1.10 → 是重新训的，不是微调。

多 seed 实测（768×1152，25 步，同输入同 seed）：

| | seed 43 | seed 7 |
|---|---|---|
| V1 | 正面，双腿交叉，清晰度 270 | 正面，双腿交叉，**清晰度 328** |
| V2 | 正面，双腿交叉，清晰度 280 | **背面**（背对镜头），清晰度 297 |

**结论：V2 没有更好。** 画质指标两者相当（V2 在 seed43 略高、seed7 明显低），
但 V2 在 seed 7 上把姿态理解成了"转身背对"，稳定性不如 V1。
默认继续用 V1（`vnccs_pose.safetensors`），V2 留档备查。

**② 骨架要等比贴入，不能拉伸**

512×768 硬拉到 768×768 会把腿压短，姿态同样只跟一半。

#### ⚠️ ComfyUI LoRA 的 `img_mlp.gate_up` 会被**静默丢掉**（2026-10-31 修）

ComfyUI 把 SwiGLU 的 gate 与 up 合成**一个** Linear：

```python
gate, up = self.gate_up(x).chunk(2, dim=-1)        # 权重 [2*intermediate, hidden]
```

diffusers 是**两个独立的** Linear（`QwenImage21SwiGLUFeedForward`）：
`gate_layer` / `proj` / `out`，各 `[intermediate, hidden]`。键名对不上，
`load_lora_weights` 只打一句 `unexpected keys found in the model` 就继续了。

实测本机 **三个 LoRA 各丢 64/384 = 17% 的权重**：

| LoRA | gate_up 键数 |
|---|---|
| `vnccs_pose`（V1/V2） | 64 |
| `qwen21_anyangle` | 64 |
| `qwen21_outpaint` | 64 |
| `qwen21_lightning` | 0（它只 LoRA 了 attention） |

拆法是**精确**的，不是近似：`up`（B，`[2*intermediate, rank]`）按行对半分，
前半给 `gate_layer`、后半给 `proj`；`down`（A）两边**共用**，clone 成两份
（safetensors 不允许两个键指向同一块内存）。`tools/convert_lora.py` 已支持，
对已转换过的文件重跑一遍即可补回（它同时接受 `diffusion_model.` 与
`transformer.` 两种前缀）。修完 384 → 448 个张量，日志里 `unexpected keys` 归零。

#### 透明底：VNCCS 下**可用**（2026-10-31）

之前"VNCCC 透明底会被控制信号干扰"是错的——那是 ControlNet 路线的问题，
连坐到了 VNCCS。实测（`tools/verify_vnccs_alpha.py`，768²，25 步）：

| 指标 | 数值 |
|---|---|
| 不透明占比 | 15.8% |
| **主体占比** | **16.6%**（产线判据 5~60%） |
| 半透明占比 | 2.1%（干净，不是雾） |
| 全透明占比 | 83.3% |

判据要看这四个数，**不要看像素差**："空白 vs 白底" 的平均差可能很大也可能很小，
完全不反映内容是否存在（ControlNet 那次就被它骗过一次）。

### 2.16d 骨骼编辑器：加一层「3D 感人形」显示（2026-10-31）

用户反馈："纯骨架分不清"——线条骨架看不出这个姿势像不像人、朝向对不对。
于是给编辑器加了一层**有体积的人形体**，但**导出格式一个字没改**。

#### 显示层：三种视图（人形 / 线框 / 叠加）

人形不是贴图，是用同一份 18 个关节现画的：

| 部件 | 画法 |
|---|---|
| 躯干 | 肩-腰-髋六边形（侧面自然收窄） |
| 四肢 | 带锥度的四边形柱（近粗远细）+ 一条高光暗示圆柱受光 |
| 头 | 椭圆；横向半径随视角在 0.80·\|cos a\| + 0.66·\|sin a\| 之间变 |
| 五官 | 正面两眼一嘴；侧面一只眼；**背面无五官 + 发际线** |
| 遮挡 | 按正交旋转后的深度排序（远 → 近画） |

深度排序是体积感的关键：侧面时两条腿一前一后，近的必须盖住远的，
不排序就是纸片人。

**头部那 7 条边（鼻-颈、头骨四角、两耳到肩）在人形模式下不画**——
它们正好落在脸上，把五官糊成一团彩色。骨架格式本身不变，导出照样有。

#### 顺带修：预设姿势的比例是「大头短腿」

旧预设头占身高 22%、腿只占 34%，人形看着像吉祥物。已按真人 8 头身
做分段线性重映射（头顶 0.02 / 鼻 0.075 / 颈 0.155 / 肩 0.18 /
髋 0.52 / 膝 0.72 / 踝 0.93）。

这不只是观感问题：**导出的骨架比例失真，模型照着画出来的人也是 Q 版**。
实测同一角色、同一骨架动作，新比例下生成的人物腿更长、身形更自然
（`assets/_newpose_compare.png`），姿态贴合度没有明显变化。

#### 工具：不启动前端就能看效果

```bash
cd web && npx vite build --ssr src/components/SkeletonEditor.tsx --outDir ssr-tmp
node skeleton-preview.mjs 行走,跑步,张手 0,45,90,180     # → assets/_skel_preview.svg
```

转 PNG 用本机 Edge 无头模式（**不用装任何渲染库**）：

```bash
"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless \
  --disable-gpu --screenshot=assets/_skel_preview.png --window-size=880,870 \
  "file:///D:/code/2d-game-meow/assets/_skel_preview.svg"
```

### 2.16e 全项目 review（2026-10-31 上午）

对 9600 行后端 + 6400 行前端做了一轮全面审查（静态 + 实测 + 回归测试）。
找到并修掉的问题按严重程度：

#### 高危（会毁掉产物或越权）

1. **扩图的 Outpaint LoRA 永久留在共享管线里**。`outpaint()` 挂完 LoRA
   从不卸载，而 `pipe` 是被缓存复用的同一个对象 → 跑一次扩图之后，
   **所有**后续生成都带着「灰色 padding = 待补全」的语义出图，而那些工作流的
   meta 还写着 `lora: None`。已加 try/finally 卸载。
2. **工作流临时挂的 LoRA 会顶掉用户挂的**。`apply_loras` 是「先全卸再挂」的
   独占式，而 `/loras/apply` 设计成状态式（一直生效）。实测：用户挂 AnyAngle →
   跑一次快速模式 → AnyAngle 被静默卸载。
   **已改成双槽位**：`user`（用户挂的）+ `temp`（工作流临时用），合并加载。
3. **路径穿越**：`str(target).startswith(str(ASSETS))` 是**字符串前缀**比较，
   `assets2/x.png`、`assets_BACKUP/x.png` 都能通过 → `DELETE /assets` 能删到
   assets 目录之外的文件，`GET /files/../assets2/x.png` 能读外部文件。
   已换成 `Path.is_relative_to`（按路径段比较）。
4. **meta 里的 `true_cfg` 是重复键**：w1 里写了两次（先 float 后 bool），
   Python 静默取最后一个；w2 又只有 float → 同一字段两种类型。已统一成 bool。

#### 中危（meta 说的和实际做的不一样，产线判据会算错）

5. `steps` 记的是**请求值**，实际管线是 `max(8, min(steps, 60))`。请求 4 步
   实际跑 8 步，meta 记 4。已加 `_clamp_steps`。
6. **快速模式在三条路上根本不生效**（真蒙版 inpaint、VNCCS、扩图——它们要独占
   LoRA 槽位或不走 `_qwen_call`），meta 却写 `fast_lora: lightning`。
   已加 `fast_applied` 字段区分「全局开关开着」与「本次真的生效」。
7. **负向词**：`_qwen_call` 只在 cfg>1 时才传负向词，meta 却无条件记一份。
   已改为 cfg<=1 时记 None，并加 `negative_applied`。
8. `matte` 记的是请求值，实际执行的是 `matte and not transparent`。
9. w2/w3/w4/wss/wbg/wpx 的 meta **缺尺寸或记的是生成画布而非精灵图**。
   已统一补 `width/height` 与 `sheet_size`。
10. **wss 的候选数与补提次数被写死丢弃**（前端调了没用，但参数在签名里，
    所以连 `params_ignored` 都不记）。已透传。
11. **w4 的旧路线只读第一张角色图**，前端传的多图被静默丢掉。已支持多图。
12. **扩图的扩展区域沿用模型给的软 alpha**（实测 30.6% 半透明像素）。
    扩图补的是场景，那层 alpha 是噪声 → 游戏里背景会透出来。
    已强制扩展区不透明（2.9%）。

#### 低危

13. `WORKFLOW_OF` 漏了 outpaint/outfit/faceswap → 这三个节点的底模下拉
    整块不渲染。
14. `submitToPixel` 的默认底模名是**已删除的** `sdxl-anime`。
15. topixel 不透传画布尺寸 → 后端一直按 1024² 跑（45s 而非 25s）。
16. 换装/换脸前端传死 width=768 → 非方形基底图的人物被压扁
    （与函数语义「跟随基底图」相反）。已改为不传。
17. `_parse_color` 对非法输入静默返回白色，而 meta 记的是原始字符串。
18. `poseswap` 的默认 seed 是固定值（其它工作流都是随机）。
19. 14 个工作流里有 5 个没调 `eng.require()`（能力表一旦加第二个架构就会绕过）。
20. `app._run` 的签名过滤没排除第一个位置参数 `eng` → 客户端传 `eng=123`
    会报「got multiple values」。
21. `peak_vram_gb` 首轮多算约 10GB（模型加载在测量窗口内）→ 现在加载完就重置。

#### 新增的回归测试

`tests/test_review_round2.py`（12 条）钉住这轮的结构性修改：LoRA 双槽位、
路径穿越、meta 真实性、重复键、颜色校验、require 覆盖、扩图 alpha。

#### 新增的质量实测工具

`tools/verify_review_quality.py`：跑 w1/w2/w3/wbg/wpx/wout 六条工作流，
检查 meta 尺寸与落盘文件是否一致、主体占比是否在 5~60%、alpha 是否干净、
有无纯色/全黑，并**检查跑完扩图后 LoRA 有没有泄漏**。

实测结果（768²，RTX 4090）：

| 工作流 | 尺寸 | 主体占比 | 半透明 | 判定 |
|---|---|---|---|---|
| w1 文生图（透明底） | 768×768 | 29.3% | 0.8% | OK |
| w2 图生图 | 768×768 | 18.2% | 0.0% | OK |
| w3 局部重绘 | 768×768 | 16.9% | 4.4% | OK |
| wbg 抠图 | 768×768 | 16.9% | 4.4% | OK |
| wpx 像素化 | 64×64 | 16.8% | 0.0% | OK |
| wout 扩图 | 1152×768 | 44.6% | 2.9% | OK（修前 30.6%） |

### 2.16 生态侦察：HF 上 Qwen-Image-2.1 的可用资产

之前记的「HF 下载被代理挡住」**不成立**：本机有可用代理
**`http://127.0.0.1:7897`**，加上它之后 `huggingface.co` 直连也能通，
hf-mirror 走代理约 1.8MB/s。下载命令：

```bash
curl -x http://127.0.0.1:7897 -L -o out.safetensors \
  "https://hf-mirror.com/<repo>/resolve/main/<file>"
```

已确认的资产：

| 资产 | 仓库 | 大小 | 格式 | 用途 |
|---|---|---|---|---|
| **Lightning LoRA** | `lightx2v/Qwen-Image-2.1-Lightning` | 64MB | **diffusers 原生**（`transformer.…lora.down/up`） | 少步蒸馏（4/8 步）——**已实测，见 §2.13** |
| Turbo 权重 | `Qwen/Qwen-Image-2.1-Turbo` | 9.5GB×2 | diffusers | 8 步生成/编辑 |
| Turbo LoRA | `Comfy-Org/Qwen-Image-2.1` → `loras/qwen_image_2.1_turbo_lora_avg_rank_178` | — | ComfyUI | 给标准权重加 Turbo 能力 |
| 5/6/8 步 LoRA | 同上 `loras/p_q_image_2.1_{5,6,8}_steps` | 320MB | ComfyUI | 少步 |
| Outpaint LoRA | `ausboss/Qwen-Image-2.1-Outpaint-LoRA` | 152MB | ComfyUI | 扩图 |
| **ControlNet Union** | `alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union` | 7.2GB | ComfyUI | **8 种结构控制 + 蒙版 inpaint** |
| PE 提示词重写 | `Qwen/Qwen-Image-2.1-PE-T2I` / `PE-I2I` | 9B | transformers | 短提示词扩写成详细描述 |
| AnyAngle LoRA | `lilylilith/QI_2.1_AnyAngle` | 119MB | ComfyUI | 相机方位控制（**本机已下好**） |

**重要更正**：§4.3 说「ControlNet Union 是 20B 专用，2.1 用不了」——
那条结论**只对 pixelforge 里的 `qwen-image-controlnet-union` 成立**。
`alibaba-pai` 后来发布了**专为 Qwen-Image-2.1** 的 Fun-Controlnet-Union，
而且它的 control 分支 `control_in_dim = 129 = 64 控制latent | 1 mask | 64 遮罩图latent`
——**官方自己就把蒙版做进了控制分支**。
它需要 VideoX-Fun 的代码（不是 diffusers 原生），接进来是一条新工程线。

**LoRA 格式的坑**：ComfyUI 格式的 key 带 `diffusion_model.` 前缀、
用 `lora_A/lora_B` 命名；diffusers 用 `transformer.` 前缀 + `lora.down/up`。
两者**不能直接互换**，需要转换脚本（键名映射 + A/B 顺序）。
只有 `lightx2v` 那个是 diffusers 原生格式，可以直接挂。

除了前缀，**还有一个更隐蔽的**：ComfyUI 的 `img_mlp.gate_up` 是合并版
SwiGLU，diffusers 里没有这个模块名，这部分权重会被**静默丢弃**（本机三个
LoRA 各丢 17%）。详见 §2.16b 末尾那一节。

### 2.16c 序列帧替换（wss）也走 VNCCS（2026-10-31）

逐帧替换的老痛点是**帧间角色会漂**：8 帧跑下来可能换三次脸、换两套衣服，
因为每帧都是独立采样、没有任何跨帧的约束。VNCCS 的 LoRA + 固定指令把
"保留图 2 的角色"这件事变成了模型的显式任务，配上**同一个 seed 步进**，
实测 4 帧（512²，25 步）全都是同一个角色的同一套服装。

`gen.sprite_swap(..., engine="wvnccs")` 与旧路线的差别：

| | `wpc`（旧） | `wvnccs` |
|---|---|---|
| 图序 | [角色, 轮廓/骨架] | **[骨架, 角色]** |
| 指令 | 自写"The first image is..." | 官方 `replace the pose of <image 2>...` |
| 条件图 | 灰度化 | **不灰度化**（LoRA 按彩色骨架训） |
| 角色图 | `prepare_reference` 留白 12% | **`_vnccs_fit` 原图贴入** |
| 步数 | 用默认值 | **下限 20**（LoRA 非少步蒸馏） |
| LoRA 挂载 | 无 | **整段循环外挂一次**（逐帧 load/unload 每帧多读 152MB） |

实测（512²，4 帧，25 步）：总 49.6s、单帧 10.8~16.2s、闸门 3/4、峰值 15.4GB。
第一步慢是因为 LoRA 加载 + 首帧预热。

**前端**：序列帧节点原来的「引擎」下拉写的是 AnimateDiff（§4.4，早已随 SD
删除）和「显存档位」，都是死选项。已换成 VNCCS / 旧版，默认 VNCCS。

#### ⚠️ 骨架图必须先换成白底（w4 / wss 都一样）

OpenPose 骨架（DWPose、`tools/make_pose.render`）都是**黑底**。整张贴到白底
画布上之后，中间仍是一块**黑底矩形**——模型读到的是"一个黑色剪影"，
于是把角色画成**背面**。

实测（w4，3 帧，768²/25 步，同 seed）：

| 骨架底色 | 输出 |
|---|---|
| 黑底（原样） | 三帧全是**背影**，看不到脸 |
| **白底**（`_vnccs_skeleton`） | 三帧都是**正面**，蓝甲女战士，帧间身份一致 |

转换用**饱和度**判据而不是亮度：OpenPose 的线是纯色（红黄绿青蓝），
背景是纯黑（无彩色分量）；深蓝线饱和度仍然很高，不会被误伤。

#### w4（姿态序列）也接上了 VNCCS

`gen.poseswap(engine="wvnccs")`：图序 [骨架, 角色]、官方模板、条件图不灰度化、
角色图原图贴入、步数下限 20、LoRA 整段循环外挂一次。
**两阶段模式在 VNCCS 下自动关闭**——VNCCS 本身就是"骨架 + 角色"双图，
两阶段的第二段用 [角色图, 草图]，图序正好相反。

实测（3 帧，768²/25 步）：55.5s、单帧 17.8~19.8s、闸门 3/3、峰值 15.4GB。
帧间身份**完全一致**（同一张脸、同一套蓝甲），姿态是"双腿分开站立"，
没有完全复制骨架那种"腿一前一后"——仍是软约束的幅度不足。

另外修了一个**静默覆盖**：w4 的 meta 里原本有第二处 `"lora": None,
"lora_scale": None`，重复键把上面刚写的 engine/lora **覆盖成 None**，
meta 里看不到用了哪个 LoRA。

## 3. 之前验证过的结论（保留，仍然是事实）

### A. w5 的「裁剪块」方案：机制可行，质量不合格

裁剪「选区 + 上下文」→ 指令改图 → 羽化贴回，实测 54~59s、选区外差 0，
但**脸被抹掉、指令「盔甲改翠绿」没执行**。原因：裁剪块只有 448×400，放大到
768² 后模型把它当**新图重画**，context 里没有脸 → 身份丢失。

→ 已改为**整图做条件**（`strategy="whole"`，默认）。`strategy="crop"` 保留，
只用于对照实验。

### B. Qwen 逐帧出图（w4 / wss 的底座）：姿势准，代价是慢

| 指标 | 实测 | 对比 SD1.5 |
|---|---|---|
| 耗时 | 24~26s / 帧（768²/20 步） | 1.2s / 帧（**慢 20×**） |
| DWPose 关节数 | 16~18 / 18 | — |
| 贴边 | 0 | — |

### C. Qwen-Image-Edit 2511 做 mask inpaint：不可能

- `qwen-image-edit-2511` 的 `_class_name = QwenImageEditPlusPipeline`，目录里只有
  `transformer` + `processor`（缺 text_encoder / vae / tokenizer / scheduler）
- `QwenImage21Pipeline` / `QwenImageEditPipeline` / `QwenImageEditPlusPipeline` 的
  `__call__` **都没有 `mask` / `mask_image` / `strength`**（实测 grep 过签名）

---

## 4. **做不到的部分（硬限制，不要再试）**
这一节是这份文档的核心。每条都写明「为什么做不到」与「替代方案是什么」，
避免以后有人在同一个坑里重复调研。

### 4.1 ~~没有原生 mask inpaint~~ → **已解决（2026-10-30，见 §2.11）**

> **本节结论已被推翻。** 原判断只看了 diffusers 的 `__call__` 签名没有 `mask`
> 参数，就断定模型做不到。实际是：**diffusers 没封装，但模型做得到**——
> ComfyUI 用**标准 2.1 权重**（无专用 inpaint 权重）就能做真蒙版重绘，
> 靠的是采样器级的 `denoise_mask` 机制。
> 已实现并实测通过（蒙版内指令准确执行、蒙版外几乎不变）。
> 详见 **§2.11 真·蒙版局部重绘**。下面保留原文并标注推翻原因。

**原事实（对 diffusers 成立、对模型不成立）**：Qwen 三条编辑管线的 `__call__`
都没有 `mask` 参数；它的机制是「条件图编码成 vision context + VAE latent
tokens 前置到噪声」，改哪里由**指令**决定。SD 能做到「涂哪改哪」靠的是
9 通道 UNet（4 噪声 + 4 被遮原图 + 1 mask）在 latent 上做混合。

**推翻它的证据**：ComfyUI 的 `Qwen Image 2.1 Inpainting | No-Offset Edits`
工作流用的**就是标准 2.1 权重**（`UNETLoader` 加载 Qwen-Image-2.1，
没有专用 inpaint 权重），靠两件事实现：

1. `SetLatentNoiseMask`：初始 latent 里蒙版内换成噪声
2. 采样循环里的 `denoise_mask`：**每步**把蒙版外拉回原 latent
   （`comfy/samplers.py:641-642`）

第 2 步是**采样器级**机制，与权重无关 → diffusers 侧用
`latents=` 入参 + `callback_on_step_end` 钩子即可复刻。已落地。

**仍然成立的部分**：旧的「整图改图 + 蒙版贴回」路径（`strategy="whole"`）
仍在，保留做对照——它选区外逐点为 0，但模型看不见蒙版、指令常不执行。

### 4.2 Qwen 不吃 SD LoRA —— LoRA 生态整体失效

**事实**：LoRA 是挂到 UNet/DiT 的 attention 投影层上的增量权重，键名与结构必须
和主干严格对齐。Qwen 的主干是 `QwenImage21Transformer2DModel`（单流 DiT），
SD1.5/SDXL 的 27 个精选 LoRA 一个都挂不上；`/loras` 里 `caps.lora=false`。

**连带失效**：LoRA 训练面板、LoRA 扫描、触发词自动前置（`_lora_trigger`）。
这些代码仍在（SD 回退时要用），但 Qwen 路径完全不经过。

**替代方案**：风格靠提示词 + 参考图（img2img / 双图条件）。要真正的风格固化，
只能等 Qwen 生态出对应的 DiT LoRA，或者走「角色图 + 双图条件」这条软引导。

### 4.3 ~~没有 ControlNet 硬约束~~ → **已解决（2026-10-30，见 §2.15）**

> **已作废**：`alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union` 已端到端跑通，
> 不需要 VideoX-Fun 的代码（纯 safetensors，180 key 逐一对齐）。
> 改成**双流**（control block 用完整 Transformer block + 每步重算）之后，
> 8 步 / 0.6 能出与骨架逐点对应的姿态（双腿大开、双臂外张），
> 比第一版「一次性跳连」的软贴近强得多。细节与坑见 §2.15。

**原事实（对 pixelforge 的 union 成立）**：本机 pixelforge 有
`qwen-image-controlnet-union`，但它是 **20B 专用**
（`joint_attention_dim=3584`），Qwen-Image-2.1 的文本编码器是 **4096 维**，对不上。
要用得接 20B 全套（24G 卡装不下 bf16，需要 fp8 + offload），是另一条独立工程线。

**后果**：w4 / wss / wpc 的姿态一致性来自「把骨架图当参考图 + 指令」，不是逐关节
钉死。实测 3 帧骨架 DWPose 16~18/18、零贴边，够精灵产线用；但复杂动作（跳跃、
交叉腿、大幅透视）仍可能跑偏，只能靠质量闸门 + 单主体判据补提。

### 4.4 没有 AnimateDiff —— 帧间一致性没有时序注意力

**事实**：AnimateDiff 的 motion module 是挂 SD1.5 UNet 的时序注意力，架构完全不同。
Qwen 逐帧之间**没有任何跨帧耦合**。

**后果**：帧间一致性只能靠三个确定性手段兜底：同 seed、`normalize_sequence`
（尺度）、`align_sequence`（脚底）。抖动仍然比 AnimateDiff 大；wss 的
`engine_mode=animatediff` 分支在 Qwen 下不可用（已忽略，强制 `frame`）。

### 4.5 SDXL Refiner / hires-fix 不可用

这两个是 SD 组件（refiner 是 SDXL 的第二个 UNet，hires 是 img2img 管线）。
Qwen 上 `refiner` / `hires` 参数被忽略。要更大尺寸就直出更大尺寸
（1024² 比 768² 贵 1.78 倍），或者出完再走 `pixelify` 的网格流程。

### 4.6 速度差距缩不到 SD 的水平

24~26s/帧 vs SD1.5 的 1.2s/帧。已经做的优化（fp8 双模块、条件图对齐、
VAE 切片、候选择优压到 1）把 37s 压到 24s，但**量级差距仍在**：8 帧动画 3 分钟。

**替代方案**：`sprite_swap(mode="recolor")` 是确定性换色，毫秒级、帧间零漂移，
适合「同体型换配色」；只有「换体型」才必须走 generate 逐帧。

### 4.7 LoRA / ControlNet 类的外部权重下载仍然受限

`lilylilith/QI_2.1_AnyAngle`（多视角立绘 LoRA）在 `huggingface.co` / `hf-mirror.com`
持续 502，ModelScope 无镜像。**这一项没做成，不是代码问题**。

### 4.8 视频（Wan2.2）没有接进来

`tools/video.py` 实测 1.07s/帧可用，但没有 API 路由与前端节点——属于
「没做」，不是「做不到」。如果要做，建议作为 wss 的第三引擎。

### 4.9 白毛/浅色主体 + 浅色背景，抠图本身不可靠

**事实**（同 seed 对照，白猫骑士：白毛、白甲、白底）：直接送 RGB 给 isnet →
头顶切角、胸口挖洞、前景 20.9%；先合成白底再送 → 头顶与胸口正常，
**两条腿被吃**（腿也是白的）、前景 15.6%。两种喂法各丢不同部位，
`fill_holes` 只能补「内部洞」，补不了「边界被切掉」。

**根因**：分割模型靠图内对比区分前景/背景，主体与背景同色时没有可用的信号。

**替代方案**：这类素材出图时**直接勾透明底**，alpha 由 Qwen 给（干净时
前景 31%、全透明 67.8%、半透明中间值 2.9%）。逐帧工作流（w4/wss）不能走这条
路——实测透明模式下骨架线会被画进成品。

**另一条出口（已实现）**：逐帧工作流（w4 / wss / wpc）多了一个 `bg` 参数，
可选 `white`（默认）/ `dark`。角色本身偏白时选 `dark`，提示词里换成
`plain solid black background`，靠对比度抠——实测白猫骑士单帧：白底 18.9% 前景 /
18 关节，深底 19.3% / 18 关节，**两种都完整**，但深底那张没有丢掉头顶与腿。
深色系角色（黑袍/黑甲）反过来仍用白底，否则深色部件会被当背景挖掉。
前端三个节点的「更多设置」里都能选。

---

## 5. 显存 / 内存的完整策略（现状）

| 手段 | 效果 | 代价 |
|---|---|---|
| transformer 预量化 fp8（6.7GB） | 权重 -49% | 无（离线缓存，已带） |
| **text_encoder 预量化 fp8（8.8GB）** | 峰值 -31%、内存 -7.5GB | 同 seed 平均差 1.10/255 |
| 条件图分辨率对齐 | 峰值 -16%、速度 +11% | 无 |
| `vae.enable_slicing()` | 解码峰值约 -1.5GB（1024²） | +0.05s |
| `enable_model_cpu_offload()` | 24G 卡装得下的前提 | 每张多 1~2s 搬运 |
| 候选择优压到 1（Qwen） | 逐帧成本减半 | 成品率靠闸门补提 |

关掉 offload（`MEOW_QWEN_OFFLOAD=0`）可以让 transformer 常驻，实测每张省 1~2s；
两块 fp8 常驻合计 15.5GB，1024² 会压线，只在 768² 且不并跑别的管线时用。

`vramopt._W4_SHAPE["qwen21"] = (11.0, 6.0, 6.5)`：按实测峰值标定（权重 / 768² 激活 /
每多一个候选的增量，单位 GB），规划器在 24G 卡上对 Qwen 给出的候选数就是 1
——这是事实，不是保守。**别按旧值 (11.0, 5.0, 5.5) 改回去**：估低了会放行 768²
批量 2（实测 23.5GB），挤到 cpu offload 上耗时翻倍且不报错。

---

## 6. SD 遗留代码：**已全部删除**

`§6` 原来写的是「清理清单待确认」。2026-10-30 已执行，两个提交：

| 提交 | 内容 |
|---|---|
| `1831ade` | 后端：engine / generate / app / 模块 / 工具 / 测试 |
| `e92775e` | 前端：LoRA 下拉与叠加、训练面板、Refiner 控件、payload 字段 |

规模：`engine.py` 1493 → 606 行，`generate.py` 3028 → 1777 行，测试从
345 条降到 260 条（85 条全是 SD 用例）。

### 删了什么

- **engine**：14 个 SD 管线方法（`inpaint_pipe` / `controlnet` / `ipa_weight` /
  `pose_pipe` / `_attach_ipa` / `animatediff_*` / `distill_*` / `apply_lora` /
  `refiner_*`）+ `_fit` / `_base_kwargs` / `_discover_bases` / `_txt2img_ok` /
  `_arch_of` / `img2img_pipe` / `_ctrl_gc` / 三个 LoRA 助手 / 10 个
  `StableDiffusion*` 导入。`arch()` 现在恒返回 `qwen21`，
  `available_models()` 恒为 `[qwen-image-2.1]`。
- **generate**：六个工作流（w1/w2/w3/w5/wpc/wss/w4）的 SD 分支全删，
  改成 Qwen 实现的薄封装；连带删掉 `_text_kwargs` / `_sdxl_embeds` /
  `_hires_fix` / `_lora_trigger` / `_floatify` / `_sprite_animatediff` /
  `composite` / `NEG_DEFAULT` / `QUALITY_PRESETS`。
- **模块**：`workflows/clip.py`、`lora.py`、`train.py`（856 行）。
- **路由**：`/loras`、`/loras/scan`、`/train*` 五条 + `_resolve_lora` /
  `_register_lora` / `_run_train` / `TrainReq`。
- **工具**：`fetch_*`（6 个）、`smoke.py`、`bench.py`、`bench_sprite.py`、
  `ab_base` / `ab_quality` / `ab_matte` / `ab_repaint`、`train_lora.py`、
  `audit_loras.py`、`probe_ill_pixel.py`、`e2e_sprite_swap.py`。
- **前端**：节点「更多设置」里的 LoRA 下拉 / 叠加 LoRA / LoRA 强度 /
  Refiner 精修，侧栏 LoRA 计数与扫描按钮，顶栏「训练 LoRA」与 TrainPanel，
  `loraPayload` / `fetchLoras` / `scanLoras`，`GenParams` 的 lora 与 refiner 字段。

### 保留的两处「向后兼容」（不是漏删）

1. `app._run` 现在**按目标函数签名过滤参数**：旧请求里带 `lora` / `clip_skip` /
   `hires` / `distill` / `refiner` / `strength` / `cn_scale` / `ipa_scale` 不会
   TypeError，而是丢掉并写进 `meta.params_ignored`——用户翻产物能看到"这条被忽略了"，
   比静默消失诚实。
2. `/models` 的 `refiner` 字段仍在，恒 `False`：老前端的 `if (refiner)` 不会
   拿到 `undefined` 而崩。

### 第二轮清理（2026-10-31）

第一轮删的是**管线与路由**，`vramopt.py` 整套 AnimateDiff 显存优化当时漏了：
`VRAM_PROFILES` / `apply_vae_slicing` / `release_ipa_encoder` / `_attach_encode_hook` /
`optimize_pipe` / `plan_generation` / `_estimate_gb` / `plan_candidates`。
它们改的是 `AnimateDiffControlNetPipeline` 的部件（motion module、IP-Adapter
image_encoder），Qwen 的管线里没有这些对象，**一个调用点也没有**：
`plan_generation` 只在测试里被调用，`plan_candidates` 从未接线
（README 曾宣称「W4 出候选前按显存余量封顶」，实际代码里没有这一步）。

现在 `vramopt.py` 只剩 `estimate_pose_gb`（`engine.max_candidates` 用它算前端
输入框上限）。`_W4_SHAPE` 里的 sd15 / sdxl 两档也删了：未知架构按 qwen21
保守估，不再退回 SD 档位（用 SD 的权重/激活估 Qwen 的峰值会低估一半）。

同期删掉：`engine._scheduler`（引用未导入的 `DPMSolverMultistepScheduler`）、
`engine.detect_variant`（SD 的 fp16 UNet 探测）、`projectio.read_template`
（无调用），以及前端 `shortModel()` 里的 sd15 / sdxl 短名映射。

### 回退方式

```bash
git checkout pre-sd-purge        # 整段回退（推荐：删除后 Qwen 侧还有修复，
                                 #  cherry-pick 会把那些修复弄丢）
git revert e92775e && git revert 1831ade   # 只回退这两刀
```

## 7. 验证工具
| 脚本 | 用途 |
|---|---|
| `tools/verify_qwen_workflows.py` | 四个新工作流的端到端验证（可选 `--only e,f,g,h`） |
| `tools/exp_frame_anchor.py` | 帧间一致性实验（首帧锚定 vs 现状，见 §2.8） |
| `tools/exp_two_stage.py` | 两阶段姿态实验（草图 → 换皮，见 §2.9） |
| `tools/bench_qwen_profile.py` | 耗时拆解（DiT / 文本编码 / VAE / 搬运），见 §2.10 |
| `tools/exp_cfg_ab.py` | 真 CFG 的批量 A/B（时间与画质），见 §2.10 |
| `tools/exp_steps_wide.py` | 步数下限重测（12 提示词 × 5 档），见 §2.6 |
| `tools/exp_steps_pose.py` | 步数对姿态贴合度的影响，见 §2.6 |
| `tools/bench_qwen.py` | 单位成本 + 条件图分辨率对齐对照（`--quick` 冒烟） |
| `tools/make_qwen_fp8.py` | 文本编码器 fp8 缓存生成（一次即可，产物已带） |
| `tools/verify_qwen_only.py` | 早期 A/B/C 三项验证（历史数据来源） |
| `tools/bench_sprite.py` | 产线单位成本（SD 时代，Qwen 下仍可参考帧成本口径） |

测试：`tests/test_qwen_workflows.py`（19 条，不加载模型，7s 跑完）。
覆盖能力表、近似标记、选区外像素不变、`output_resolution` 真的传下去、
尺度归一、透明底 alpha、四个工作流能打包。

---

## 8. 回退

SD 代码已删除，**没有「挪回权重就跑起来」这条路了**，只能回退代码：

```bash
git checkout pre-sd-purge      # 删除前的最后状态（tag 已推远端）
# 或 git revert e92775e && git revert 1831ade
```

然后搬权重（`models/_archive/sd/` 里都还在）：

```bash
mv models/_archive/sd/checkpoints/sd15 models/checkpoints/
mv models/_archive/sd/controlnet models/ && mv models/_archive/sd/ipadapter models/
mv models/_archive/sd/animatediff models/
```

只关掉本次的显存优化（保留 Qwen）：
```bash
MEOW_QWEN_TEXT_FP8=0   # 文本编码器退回 bf16（峰值回到 16.6GB）
MEOW_QWEN_OFFLOAD=0    # 两块权重常驻，省搬运时间
```
