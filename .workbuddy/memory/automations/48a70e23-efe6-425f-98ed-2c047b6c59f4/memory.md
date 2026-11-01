# 自动化执行记录

## 2026-10-30（首次执行）

**任务**：review `docs/PLAN-QWEN-ONLY.md` → 自定计划 → 用 Qwen-Image-2.1 实现全部功能，
优化生成质量与显存/内存，做不到的写进文档。

**结果**：完成。提交 `88ea7d0`，已推 `origin/main`。

- 四个原判「做不了」的工作流（w3/w5/w4/wss）全部落地，均为近似实现并在前端打「近似」角标。
- 质量三招：条件图分辨率对齐（`output_resolution`）、逐帧尺度归一、条件图灰度化；步数默认 24。
- 显存：文本编码器 fp8 化（`tools/make_qwen_fp8.py`，产物 `models/checkpoints/qwen-image-2.1/text_encoder-fp8/`），
  峰值 16.59→11.41GB。开关 `MEOW_QWEN_TEXT_FP8=0`。
- 测试 338 passed / 86 skipped / 0 failed；前端 smoke 330/330；前端 dist 已重建。
- 文档：`docs/PLAN-QWEN-ONLY.md` 已重写（§4 为「明确做不到的八件事」），README 能力表改三档。

**下次可继续的（都在 PLAN 里）**：P2-1 多视角 LoRA（HF 502 阻塞）、P2-2 20B + ControlNet Union
（需 fp8+offload 全套）、P2-3 Wan2.2 视频接进画布、P2-4 架构解耦（stage / scorer 可插拔）。

**注意**：本机只有 RTX 4090 24GB / 32GB 内存，任何新结论必须预热 + 多次取中位数（见项目 MEMORY.md）；
重量级验证用 `tools/verify_qwen_workflows.py --only g,h` 分段跑，别一次全跑。
