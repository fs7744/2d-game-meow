# Qwen-only 迁移计划

> 目标：把工作台从「SD1.5 / SDXL / Qwen 三底模」变成「**只有 Qwen-Image-2.1**」，
> 十个工作流全部用 Qwen 实现。
>
> 起点：`sd` tag（SD 时代完整基线，已推远端）→ 当前提交 `2f006c7`。
> 本文只列**还没做**的事和每件事的验收标准；已做的只留结论。

---

## 0. 已经做完的（不用再做）

| 项 | 状态 |
|---|---|
| `sd` tag 备份并推远端 | ✅ annotated tag `sd` → `fee24c9` → `d7accf6` |
| SD 权重归档到 `models/_archive/sd/`（移动，非删除，47G） | ✅ 底模/ControlNet/IP-Adapter/AnimateDiff/27 个 LoRA |
| 保留 Qwen 需要的：`matte/`（抠图）、`pose_keypoint/`（DWPose） | ✅ |
| 默认底模 = `qwen-image-2.1`（`engine.DEFAULT_MODEL` → `/models.current` → 前端跟随） | ✅ |
| 前端修掉「写死有 SDXL 就用 SDXL」的坑 | ✅ |
| 依赖 SD 权重的 12 个后端用例改 `skipif`（恢复权重自动生效） | ✅ |
| 清理 `test_api.py` 三组重复定义（其中一条断言一直被同名覆盖） | ✅ |
| 前端 smoke 330/330；后端 46 passed / 18 skipped | ✅ |

---

## 1. 已验证的结论（决定后面怎么做的实测数据）

### A. w5 局部重绘 → Qwen：**机制可行，但裁剪块方案质量不合格**

方案：裁剪「选区 + 上下文」→ Qwen 指令改图 → 羽化贴回（`repaint.composite`）。

| 指标 | 实测 |
|---|---|
| 耗时 | 54~59s |
| **选区外最大像素差** | **0**（逐点不变，硬承诺达成） |
| 选区内平均差 | 41.6（确实重画了） |
| 质量 | ❌ **不合格**：脸部被抹掉、指令「盔甲改翠绿」**没执行**（仍是金色） |

失败原因：裁剪块只有 448×400，放大到 768² 后模型把选区当成**新图重画**，
上下文里没有脸 → 身份丢失；`true_cfg_scale=4` 又让它更听指令、更不像原图。

→ 见待办 **P0-2**：两个改良方案对比后再定。

### B. Qwen 逐帧出图（w4 / wss 的底座）：姿势极准，代价是 30 倍慢

| 指标 | 实测 | 对比 SD1.5 |
|---|---|---|
| 耗时 | **37.0s / 帧** | 1.2s / 帧（**慢 30×**） |
| DWPose 关节数 | **18/18 全中**（三帧都是） | — |
| 贴边 | 0 | — |
| 可交付率 | 待补测（第一次跑 coverage=1.0 是因为 RGB 图无 alpha，判据要抠完再算） | — |

→ 姿势一致性满分，**瓶颈只有速度**。8 帧动画 = 5 分钟（SD 只要 10 秒）。
是否值得，取决于产线能不能接受这个成本 → 待办 **P0-1** 拿可交付率数据再定。

### C. Qwen-Image-Edit 2511 做 mask inpaint：**不可能**

- `qwen-image-edit-2511` 的 `model_index._class_name = QwenImageEditPlusPipeline`，
  目录里只有 `transformer` + `processor`（缺 text_encoder / vae / tokenizer / scheduler）
- `QwenImage21Pipeline` / `QwenImageEditPipeline` / `QwenImageEditPlusPipeline`
  的 `__call__` **都没有 `mask` / `mask_image` / `strength`**（实测 grep）
- SD inpaint 靠 **9 通道 UNet**（4 噪声 latent + 4 被遮原图 + 1 mask）做 masked latent 混合

→ **w3 局部重绘在当前生态无解**；只能用「整图指令改图 + 蒙版贴回」近似
（见 P0-2）。要真 mask，得等官方出接口，或换 ComfyUI 的 Qwen inpainting 节点。

---

## 2. 待办（按优先级）

### P0-1 拿 Qwen 逐帧的可交付率数据
- **做什么**：跑 `tools/verify_qwen_only.py` 的 B 段（3 帧 + 抠图 + `quality.evaluate`），
  补齐「前景占比 5~60%」这一项。
- **验收**：拿到 `coverage` / `passed` / 可交付率三个数，写进 README。
- **为什么**：产线判据是「单主体 ∧ 无贴边 ∧ 前景 5~60%」，缺一项就没法判断
  Qwen 能不能进精灵产线。

### P0-2 定案 Qwen 版局部重绘
- **做什么**：跑 `tools/verify_local_edit_variants.py`（已备好，未跑），对比三个方案：
  1. 原图（当前）：裁剪块 + cfg 4 → 已知失败（脸丢、指令不执行）
  2. **整图做条件 + 蒙版贴回**：模型看得到完整角色，只贴回选区
  3. 裁剪块 + **低 CFG（1.5）**：更贴近原图，指令权重下降
- **验收**：至少一个方案做到「指令被执行（盔甲变绿）且脸不丢」。
- **备注**：模型侧已证明 Qwen 能做「把红披风改成蓝的」这种整图指令编辑；
  难点只在**局部化**。

### P0-3 实现 w5 的 Qwen 分支（前提：P0-2 有可行方案）
- **做什么**：`generate.ai_repaint` 按 `arch(model)` 分流；Qwen 分支走
  「（整图或裁剪块）→ 指令改图 → `repaint.composite` 贴回」，去掉 `strength` 与 LoRA。
  前端：`caps.workflows` 放开 `ai_repaint`；隐藏「重绘幅度」滑块（Qwen 没有这个语义）。
- **验收**：w5 选 Qwen 底模能跑通，选区外像素差 = 0，meta 里记 `backend="qwen-edit"`。

### P0-4 w3 局部重绘的替代方案（明确取舍）
- **做法**：
  - 方案一（推荐）：把 w3 也做成「整图指令改图 + 蒙版贴回」，并在界面上写明
    **不能保证「涂哪改哪」**（Qwen 没有 mask 接口）。
  - 方案二：w3 只支持 SD，从能力表里移除并在节点上给说明。
- **验收**：用户能选到「Qwen 近似版 w3」，且界面说明它与真 inpaint 的差别。

### P0-5 w4 姿态序列 / wss 序列帧替换的 Qwen 化方案
- **做什么**：w4 = 骨架帧 → 逐帧 Qwen 出图 → 脚底对齐 → 打包；
  wss = 精灵图切帧 → 逐帧 Qwen（角色图 + 骨架图双图条件）→ 对齐 → 共享色板 → 打包。
  难点：37s/帧 × 帧数；且**没有 ControlNet 硬约束**，姿势靠骨架图 + 指令。
- **验收**：4 帧精灵图的端到端跑通，帧间脚底漂移 ≤ 2px，共享色板生效。
- **备选路线**：Qwen 出**关键帧** → SD（已归档）补中间帧 —— 但这违背「去 SD」，
  只有在「Qwen 逐帧质量/速度都不可接受」时才启用。

### P1-1 把 4 个「近似实现」标出来
- **做什么**：`/models` 的 `caps` 增加 `approx: boolean`（哪些工作流是近似而非原生），
  前端在节点上显示「Qwen 近似实现」标记。
- **验收**：用户一眼能看出 w3/w5/w4/wss 哪些是硬约束、哪些是软引导。

### P1-2 清理 SD 遗留代码
- **做什么**：Qwen-only 稳定后，删掉引擎里的 SD 管线分支（`inpaint_pipe` / `pose_pipe` /
  IP-Adapter / Refiner / AnimateDiff 路径）、LoRA 训练与扫描面板、`_ARCH_WORKFLOWS`
  的 SD 条目、`tools/fetch_*.py` 的 SD 下载器。
- **注意**：删之前先确认 P0-3~P0-5 全部落地；删完 `git checkout sd` 仍能回到旧版。
- **验收**：仓库里 grep 不到 `StableDiffusion*Pipeline` 的调用；测试全绿。

### P1-3 文档同步
- README「版本选型」章节改成两代对照；LoRA 章节标注「SD 生态，已归档」；
  补「Qwen-only 可行性验证」小节（贴 P0-1/P0-2 的实测数据）。

### P2-1 多视角立绘（AnyAngle LoRA）
- 需要 `lilylilith/QI_2.1_AnyAngle`（`huggingface.co` / `hf-mirror.com` 当前 502，
  ModelScope 直连可用但没有镜像）。骨骼编辑器的方向按钮已能覆盖大部分需求。

### P2-2 Qwen-Image-20B + ControlNet Union（pose 硬约束）
- 本机 pixelforge 有 `qwen-image-20b`（含 transformer-fp8）与 `qwen-image-controlnet-union`。
- **但 union 是 20B 专用**（`joint_attention_dim=3584`），Qwen-Image-2.1 的文本编码器
  是 4096，**对不上**。要走这条路得接 20B 全套（显存 24G 装不下 bf16，
  要 fp8 + offload），是独立的一条工程线。

### P2-3 W5 视频接进画布
- Wan2.2 TI2V-5B 已在 pixelforge，CLI（`tools/video.py`）实测 1.07s/帧，
  但**没有 API 路由也没有前端节点**。评审建议 A：作为 wss 的第三引擎。

### P2-4 架构解耦（评审建议 B/C）
- B：生成 / 后处理 / 落盘拆成可组合 stage（现在 9 个工作流各写一遍）。
- C：候选 + 闸门做成可插拔 scorer 层（pose / silhouette / identity / clip-sim），
  解决「W4 对非人形角色恒 0 分」。

---

## 3. 硬限制（再评估也改不掉的）

1. **Qwen 没有 mask inpaint**：三条编辑管线都没有 `mask`/`mask_image`/`strength`。
2. **Qwen 不吃 SD LoRA**：LoRA 训练、LoRA 扫描、27 个精选 LoRA 在 Qwen-only 下全部失效。
3. **速度差 30×**：37s/帧 vs SD1.5 的 1.2s/帧。批量序列帧是产线的硬伤。
4. **没有 ControlNet 硬约束**：姿态只能靠「骨架图当参考图 + 指令」软引导。
5. **显存**：Qwen 峰值 16~19GB（双图条件 19.3GB），24G 卡里 SD 管线放不下
   （现在靠 `evict_other_models` 驱逐）；32G 内存装不下全量 bf16，
   必须 fp8 缓存 + cpu offload（torchao 要 torch≥2.11，本机 2.6 用不了）。
6. **HF 下载受限**：代理访问 huggingface.co / hf-mirror.com 持续 502。

---

## 4. 回退

```bash
# 权重搬回（README 有完整清单）
mv models/_archive/sd/checkpoints/sd15 models/checkpoints/
mv models/_archive/sd/controlnet models/ && mv models/_archive/sd/ipadapter models/
mv models/_archive/sd/animatediff models/
mv models/_archive/sd/lora/*.safetensors models/lora/
cp models/_archive/sd/lora/loras.json.bak models/lora/loras.json
# 常量改回
# workflows/engine.py: DEFAULT_MODEL = "sd15"
# 或直接：git checkout sd
```

---

## 5. 验证工具

| 脚本 | 用途 |
|---|---|
| `tools/verify_qwen_only.py` | A/B/C 三项验证（w5 局部重绘 / Qwen 逐帧 / edit-2511 能否 mask） |
| `tools/verify_local_edit_variants.py` | P0-2 的三方案对比（整图做条件 vs 裁剪块低 CFG） |
| `tools/bench_sprite.py` | 产线单位成本（**必须用**，见下） |

> 度量约定：任何速度/显存结论都要**预热 + 多次取中位数**，单次调用波动可达 70%；
> 显存看 `memory_reserved`；测后处理要用**未像素化的 `__raw` 输入**。
