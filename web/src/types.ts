export type Preset = 'sprite' | 'portrait' | 'wuxia' | 'pixel' | 'none'

/** 节点类型：图生成（w1）/ 图生图（w2）/ 图片 / 抠图换背景（wbg）/ 局部重绘（w3）/ 局部 AI 重绘（w5） */
export type NodeKind =
  | 'gen' | 'img2img' | 'image' | 'bg' | 'inpaint' | 'airepaint' | 'pixelify'
  | 'poseclone' | 'spriteswap'

/** 角色迁移 / 序列帧替换：动作参考图的解释方式 */
export type RefMode = 'auto' | 'pose' | 'outline'

/** 序列帧替换：怎么处理角色 */
export type SwapMode = 'generate' | 'recolor'

/** 序列帧替换：用哪个引擎生成 */
export type EngineMode = 'auto' | 'animatediff' | 'frame'

/** 显存档位：显存不够时降分辨率（帧数不能砍——那是用户的动画内容） */
export type VramProfile = 'auto' | 'low' | 'medium' | 'high'

/** 完美像素转换的每格取色方式 */
export type PixelSample = 'median' | 'majority' | 'center'

/** 抠图节点的背景方式 */
export type BgMode = 'alpha' | 'color' | 'image'

/** 图片节点挂着的图 */
export type NodeImage = {
  path: string   // 磁盘绝对路径（当图生图的源时要传给后端）
  url: string    // /files/... 展示用
  w: number      // 展示宽度（px）；高度按原图比例
  h: number      // 原图高度
}

/** 图生图的源：指向画布上另一个节点的产物，或直接给本地绝对路径 */
export type NodeSource = { nodeId?: string; path?: string }

/**
 * 角色迁移节点的两个输入：
 *   source     —— 动作参考图（真人照片，用来提骨架）
 *   charSource —— 角色图（用来定身份，外观由 IP-Adapter 带过去）
 */
export type NodeCharSource = { nodeId?: string; path?: string }

/**
 * GET /models 的 recommend 字段：这个底模在**精灵管线**上的推荐采样参数。
 *
 * 为什么要由后端给：实测 SDXL 在 1024² 反而更差（角色在画面里占比更大、更容易
 * 贴边，可交付率 92%→75%），加上生成慢一倍，每张可用帧贵 2.3 倍。前端如果按
 * "SDXL 就该 1024"的直觉填表单，用户会稳定拿到更差的结果，而且没有任何报错。
 */
export type ModelRecommend = {
  width: number
  height: number
  steps: number
  cfg: number
  pixel: number
  palette: number
}

export type GenParams = {
  prompt: string
  negative: string
  width: number
  height: number
  steps: number
  cfg: number
  seed: number
  preset: Preset
  pixel: number
  palette: number
  matte: boolean
  clipSkip: number        // 1=最后一层（默认）；2=跳过最后一层（多数动漫/像素 LoRA 的训练设置）
  strength: number        // 图生图的重绘幅度；文生图忽略
  bgMode: BgMode          // 抠图节点：透明 / 纯色 / 换背景图
  bgColor: string         // 抠图节点：纯色背景（#rrggbb）
  bgNodeId: string        // 抠图节点：当背景用的那个图片节点 id
  maskPath: string        // 局部重绘节点：蒙版文件路径（后端要读它）
  maskUrl: string         // 局部重绘节点：蒙版缩略图（给用户看涂了哪）
  maskCoverage: number    // 局部重绘节点：蒙版覆盖了多少比例
  rpContext: number       // AI 重绘：选区外扩多少上下文（0~1，按选区尺寸的比例）
  rpExpand: number        // AI 重绘：蒙版先外扩几 px（给模型留「往哪长」的余地）
  rpFeather: number       // AI 重绘：贴回羽化 px（0 = 硬边）
  rpMatchColor: boolean   // AI 重绘：接缝带做色彩校正
  rpPaletteSnap: boolean  // AI 重绘：把重绘区里接近源色的杂色吸回源图色板
  pxTarget: number        // 像素转换：目标原生像素边长（0 = 用自动检测的网格）
  pxColors: number        // 像素转换：共享色板色数（0 = 不量化）
  pxSample: PixelSample   // 像素转换：每格取色方式
  pxGridSnap: boolean     // 像素转换：是否做网格检测/吸附
  pxTransparentBg: boolean// 像素转换：把纯色背景转成透明
  pxSnapTo: number        // 像素转换：网格吸附到这个数的整数倍（0 = 不吸附）
  pcCnScale: number       // 角色迁移：ControlNet 姿态强度（越大越贴参考动作）
  pcIpaScale: number      // 角色迁移：IP-Adapter 身份强度（越大越像角色图）
  pcFill: number          // 角色迁移：人物在画布里占的高度比例
  pcRefMode: RefMode      // 角色迁移：动作参考图的解释方式（骨架/轮廓/自动）
  ssCols: number          // 序列帧替换：精灵图列数（0 = 自动检测）
  ssRows: number          // 序列帧替换：精灵图行数（0 = 自动检测）
  ssMode: SwapMode        // 序列帧替换：AI 重绘 还是 确定性换皮
  ssRefMode: RefMode      // 序列帧替换：动作参考图的解释方式
  ssCnScale: number       // 序列帧替换：动作强度
  ssIpaScale: number      // 序列帧替换：身份强度
  ssPalette: number       // 序列帧替换：共享色板色数
  ssCandidates: number    // 序列帧替换：每帧生成几个候选再挑最好的
  ssGateRetries: number   // 序列帧替换：候选都不合格时补提几次
  ssEngineMode: EngineMode // 序列帧替换：AnimateDiff 时序一致性 / 逐帧多候选
  ssVramProfile: VramProfile // 序列帧替换：显存档位（auto 自动降分辨率）
  lora: string            // LoRA 名（后端解析成路径）；空串表示不挂
  loraScale: number
}

/** 后端 /loras 返回的条目 */
export type LoraInfo = {
  name: string
  label: string
  trigger: string
  scale: number
  preset: Preset | null
  matte?: boolean | null  // 选中时联动勾选「抠图」；null/缺失表示不改
  clip_skip?: number | null
  base?: string          // 适配的底模；缺失按 sd15 处理
  note?: string
  source?: string
  size_mb?: number
  hidden?: boolean
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed'

export type JobState = {
  id: string
  status: JobStatus
  error?: string
  url?: string
  seed?: number
  seconds?: number
  peakVram?: number
  path?: string          // 产物绝对路径（图生图要拿它当 init_image）
  raw?: string           // 后处理之前的原图路径（优先用它当源图）
  subjects?: number      // 抠图后的前景块数；>1 说明画了多个主体
  startedAt?: number
}

export type NodeView = 'expanded' | 'image'

export type GenNode = {
  id: string
  kind: NodeKind
  x: number
  y: number
  params: GenParams
  view: NodeView
  source?: NodeSource
  /** 角色迁移节点：角色图（身份来源） */
  charSource?: NodeCharSource
  image?: NodeImage
  job?: JobState
}

export type Viewport = { x: number; y: number; z: number }

export type FavoritePrompt = {
  id: string
  text: string
  at: number
}
