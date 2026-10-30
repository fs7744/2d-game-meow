export type Preset = 'sprite' | 'portrait' | 'wuxia' | 'pixel' | 'anime' | 'none'

/** 节点类型：图生成（w1）/ 图生图（w2）/ 图片 / 抠图换背景（wbg）/ 局部重绘（w3）/ 局部 AI 重绘（w5） */
export type NodeKind =
  | 'gen' | 'img2img' | 'image' | 'bg' | 'inpaint' | 'airepaint' | 'pixelify'
  | 'poseclone' | 'spriteswap' | 'poseswap' | 'topixel' | 'poseedit'

/** 骨骼编辑器节点的一个关节：x/y 归一化 0~1，z 是前后深度（伪 3D） */
export type Joint3 = { x: number; y: number; z: number }

/** 骨骼编辑器节点的状态（可持久化，重新打开还能继续编辑） */
export type SkeletonState = {
  points: Joint3[]
  /** 方位角（度）：0 正面 / 90 右侧 / 180 背面 / 270 左侧 */
  azimuth: number
  preset: string
}

/** 角色迁移 / 序列帧替换：动作参考图的解释方式 */
/** 逐帧生成用的底色。白毛/白甲角色用白底会被抠图吃掉，换 dark 更稳 */
export type BgTone = 'white' | 'dark'

export type RefMode = 'auto' | 'pose' | 'outline' | 'skeleton'

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
  /** 建议的提示词契约（动漫底模 → 'anime'） */
  preset?: Preset
  /**
   * 这个底模在推荐尺寸下，W4（姿态序列）最多能批量几个候选。
   * 由后端按本机总显存算——Qwen@768² 单帧峰值 11.8~15.4GB，多一个候选再 +6GB，
   * 24G 卡上只有 1 个候选装得下。靠用户自己猜是猜不出来的。
   */
  max_candidates: number
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
  /** 产出风格档位（gen 节点）：一档定「契约 + 像素块 + 色数」三件事。
   *  用字符串而不是枚举，是为了加档位时不用改 schema。 */
  genStyle: string
  // 转像素（topixel）：任意图 → 像素画（可先出图再转）
  apTarget: number        // 目标原生像素边长（192 = 精灵常用）
  apColors: number        // 共享色板色数（32 = 16-bit 风格）
  apPreEnhance: number    // 量化前对比度/饱和度增强强度（0 = 关）
  apEdgeRebuild: boolean  // 轮廓描边（色数不变，压缩后不「融化」）
  apMatte: boolean        // 先抠背景（像素精灵通常要透明底）
  pcFill: number          // 角色迁移：人物在画布里占的高度比例
  pcRefMode: RefMode      // 角色迁移：动作参考图的解释方式（自动/骨架/轮廓/已画好的骨架图）
  bgTone: BgTone          // 逐帧生成用的底色：white=白底（默认）/ dark=深色底（白毛角色用）
  ssCols: number          // 序列帧替换：精灵图列数（0 = 自动检测）
  ssRows: number          // 序列帧替换：精灵图行数（0 = 自动检测）
  ssMode: SwapMode        // 序列帧替换：AI 重绘 还是 确定性换皮
  ssRefMode: RefMode      // 序列帧替换：动作参考图的解释方式
  ssPalette: number       // 序列帧替换：共享色板色数
  ssCandidates: number    // 序列帧替换：每帧生成几个候选再挑最好的
  ssGateRetries: number   // 序列帧替换：候选都不合格时补提几次
  ssEngineMode: EngineMode // 序列帧替换：AnimateDiff 时序一致性 / 逐帧多候选
  /**
   * 姿态序列（W4）：骨架帧**目录**。可以填目录，也可以填通配
   * （assets/_pose_walk_*.png）。后端展开并自然排序——帧名 frame_2 < frame_10，
   * 字典序会把 10 帧以上排乱，动画顺序就错了。
   */
  psDir: string
  ssVramProfile: VramProfile // 序列帧替换：显存档位（auto 自动降分辨率）
  /**
   * 透明背景：只有 Qwen-Image-2.1 原生支持（VAE 第 4 通道）。
   *
   * 与 `matte` 的区别：matte 是**生成后**用分割模型抠一遍，边缘会有
   * 半透明毛边、发丝/细剑这类结构容易抠断；transparent 是生成时就让模型
   * 直接画 alpha，边界更干净、细节不丢。SD 系列开这个无效（后端会忽略）。
   */
  transparent: boolean
  /**
   * 这个节点用哪个底模。空串 = 跟随侧栏的默认底模。
   *
   * **为什么改成节点级**：底模影响的是"画成什么样"，而一张画布上经常要混跑——
   * 角色立绘想用 Qwen（原生透明底 + 像素风直出），精灵帧还得用 SDXL（ControlNet
   * 硬约束）。全局一个底模时，切一次全画布跟着变，想混跑就得来回切、每次还把
   * 已调好的参数冲一遍。引擎本来就按 `f"{kind}:{model}"` 缓存管线，多底模并存
   * 只是多占显存，不是不能跑。
   */
  model: string
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
  /** 序列帧/姿态序列：每帧**实际**生成了几个候选（显存不足时会被后端降下来） */
  candidates?: number
  /** 候选数被降级的原因；为空表示按用户设定执行 */
  candidatesNote?: string
  /** 序列帧：一共生成/补提了几次（candidates + 补救次数） */
  attempts?: number
  /** 序列帧：过了质量闸门的帧数 / 总帧数 */
  gatePassed?: number
  frames?: number
  startedAt?: number
}

export type NodeView = 'expanded' | 'image'

export type GenNode = {
  id: string
  kind: NodeKind
  x: number
  y: number
  /** 节点宽度（px）。缺省用 NODE_W；图片节点的宽度在 image.w 上，与此无关。 */
  w?: number
  /**
   * 节点高度（px）。缺省 = 内容自适应。
   *
   * 为什么要能锁高度：参数面板长的节点（序列帧替换、姿态序列有十几个字段）在
   * 画布上会撑到 900px 以上，摆两张就把视野占满。锁高 + 内部滚动才能把画布
   * 排得开。只对参数视图生效；图片节点的尺寸由 image.w 决定，不参与。
   */
  h?: number
  params: GenParams
  view: NodeView
  source?: NodeSource
  /** 角色迁移节点：角色图（身份来源） */
  charSource?: NodeCharSource
  image?: NodeImage
  /** 骨骼编辑器节点：关节坐标 + 方位角（导出过图时 image 上还有产物路径） */
  skeleton?: SkeletonState
  job?: JobState
}

export type Viewport = { x: number; y: number; z: number }

export type FavoritePrompt = {
  id: string
  text: string
  at: number
}
