/**
 * 节点的默认生成参数。
 *
 * 从 `App.tsx` 挪出来：角色中心的「序列帧动画」要造一个**不入画布的**姿态批量
 * 节点（喂给 `PoseBatchPanel`），需要同一份默认值。两边各写一份迟早会漂
 * （以前就踩过"画布默认 VNCCS、批量写死 wpc"那种双份默认值）。
 */
import type { GenParams } from '../types'

export const DEFAULT_PARAMS: GenParams = {
  prompt: '',
  negative: '',
  // 实测：512² 出全身角色必糊；768² 是 SD1.5 的甜点（1024² 会开始重复肢体）。
  // pixel=6 / palette=32 正好把 768 降到 128 有效像素 —— 标准精灵尺寸。
  width: 768,
  height: 768,
  // 16 步是实测下限（浅色主体可以调到 20~24）
  steps: 16,
  // Qwen 的真 CFG：1 = 单次前向（快 40%，默认）；>1 = 真 CFG（慢一倍）
  cfg: 1.0,
  seed: -1,
  preset: 'sprite',
  pixel: 6,
  palette: 32,
  matte: false,
  // 模型原样：勾上就一个像素都不动（默认关 = 清脏 alpha + 去白边）
  raw_output: false,
  // 抠图节点默认：透明底（纯去背景）
  bgMode: 'alpha',
  bgColor: '#ffffff',
  bgNodeId: '',
  // 局部重绘的蒙版（笔刷画完上传到项目目录，节点只记路径）
  maskPath: '',
  maskUrl: '',
  maskCoverage: 0,
  // 局部 AI 重绘（w5）：默认值照 pixelforge 的实测结论
  rpContext: 0.35,      // 选区外带 35% 上下文
  rpExpand: 8,          // 蒙版外扩 8px
  rpFeather: 4,         // 贴回羽化 4px
  rpMatchColor: true,   // 接缝带色彩校正
  rpPaletteSnap: false, // 色板吸附：像素素材才需要
  // 完美像素转换：0 = 自动检测网格；32 色是像素画的常见上限
  pxTarget: 0,
  pxColors: 32,
  pxSample: 'median',
  pxGridSnap: true,
  pxTransparentBg: false,
  pxSnapTo: 16,           // 网格吸附到 16 的整数倍（精灵常用尺寸）
  genStyle: 'sprite16',   // 产出风格：像素精灵 16-bit（最常用）
  fastMode: false,        // 快速模式（Lightning 少步蒸馏），默认关
  outpDirection: 'right', // 扩图方向（right/left/up/down/all）
  outpExpand: 0.5,        // 扩图比例（0.5 = 往该方向扩一半）
  // 转像素（topixel）：任意图 → 像素画（没有源图时先出图再转）
  apTarget: 192,          // 目标原生像素边长（192 装得下「全身 + 武器」）
  apColors: 32,           // 共享色板（32 色 = 16-bit 风格）
  apPreEnhance: 1.0,      // 量化前拉对比度——不开会发灰
  apEdgeRebuild: true,    // 轮廓描边——不描就「融化」
  apMatte: true,          // 先抠背景（转照片时可关掉保留背景）
  // 角色迁移：姿态强度 / 身份强度 / 人物占画面高度
  pcFill: 0.8,
  // 参考模式默认 auto：先试骨架，认不出人体自动降级到轮廓（游戏素材走这条路）
  pcRefMode: 'auto',
  // 角色迁移默认走 VNCCS 引擎：保角色一致性好得多（画风不变、头不丢、不多腿）。
  // 需要 models/lora/vnccs_pose.safetensors，缺了会在提交时给明确提示。
  pcEngine: 'wvnccs',
  // 两阶段（先出姿态草图再换皮）：动作更准，代价是每帧多一次生成
  twoStage: false,
  // 底色：默认白底；角色本身偏白（白毛/白甲）时改 dark，抠图才不会啃掉脸
  bgTone: 'white',
  // 序列帧替换：0 = 自动检测精灵图网格
  ssCols: 0,
  ssRows: 0,
  ssMode: 'generate',
  ssRefMode: 'outline',
  ssPalette: 32,
  // 多候选择优：业界实测同 seed 也会漂 20~30%，靠生成多张挑一张
  ssCandidates: 2,
  ssGateRetries: 1,
  // 引擎：AnimateDiff 走时序注意力，帧间一致性更好（实测 8 帧 7.4s / 13.1GB）
  ssEngineMode: 'auto',
  ssVramProfile: 'auto',
  // 姿态序列（w4）的输入：骨架帧目录（也接受通配，如 assets/_pose_walk_*.png）
  psDir: '',
  // 透明底默认关：只有 Qwen 底模支持，SD 系列开了会被后端忽略，
  // 默认打开会让用户误以为选 SD 时也能出透明图
  transparent: false,
  // 旧存档兼容字段：底模是**全局唯一**的 Qwen-Image-2.1，节点里不再提供选择。
  // migrate() 会把旧存档里的 sd15 / sdxl 一律清成空串，别让它们混进请求。
  model: '',
}

