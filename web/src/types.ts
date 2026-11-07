export type Preset = 'sprite' | 'portrait' | 'wuxia' | 'pixel' | 'anime' | 'none'

/** 节点类型：图生成（w1）/ 图生图（w2）/ 图片 / 抠图换背景（wbg）/ 局部重绘（w3）/ 局部 AI 重绘（w5） */
export type NodeKind =
  | 'gen' | 'img2img' | 'image' | 'bg' | 'inpaint' | 'airepaint' | 'pixelify'
  | 'poseclone' | 'spriteswap' | 'poseswap' | 'topixel' | 'poseedit'
  | 'posebatch'
  | 'outpaint' | 'outfit' | 'faceswap'

/**
 * 3D 人偶编辑器（poseedit）节点的持久化状态。
 *
 * **存的是官方 PoseStudio 的 pose JSON，不是 18 个 2D 点**：同一组 2D 点反解
 * 出来的三维骨骼旋转有无穷多种（肩膀朝里还是朝外，深度全靠猜），每次重开都
 * 会差一点。官方 `getPose()` 给的是骨骼欧拉角 + IK effector + 极点，
 * `setPose()` 能精确到弧度地还原。
 */
export type PoseStudioState = {
  /** 官方 getPose() 的 JSON（不透明：格式归官网，我们不解析也不改） */
  pose?: unknown
  /** 导出机位（度）：0 正面 / 90 右侧 / 180 背面 / 270 左侧 */
  yaw: number
}

/**
 * 后端「从图片」提好的一次 2D 关节点。
 *
 * 只是**一次性投递**：节点编辑器收到后走官方转换器变成 3D 骨骼，之后真正存
 * 档的是运动员后的 pose JSON。每次提取换一个新引用，组件据此判断"有新东西"。
 */
export type PoseImportRequest = {
  points: (number[] | null)[]
  /** 递增的时间戳，用来区分"又一次导入" */
  tick: number
  /** 这组点是「3D 转正后重投影」来的：编辑器要把官方反推的 modelRotation 归零，
   *  否则会用投影压缩过的髋线再判一次骨盆倾斜，人偶还是躺着 */
  reprojected?: boolean
}

/** 角色迁移 / 序列帧替换：动作参考图的解释方式 */
/** 逐帧生成用的底色。白毛/白甲角色用白底会被抠图吃掉，换 dark 更稳 */
export type BgTone = 'white' | 'dark' | 'transparent'

export type RefMode = 'auto' | 'pose' | 'outline' | 'skeleton'

/**
 * 角色迁移的朝向。`auto` 跟着上游骨骼编辑器的方位角；其余 8 个是固定方向，
 * 写进指令（英文短语，见 App 的 VIEW_PHRASES）。
 */
export type PcView =
  | 'auto' | 'front' | 'q34front_r' | 'side_r' | 'q34back_r'
  | 'back' | 'q34back_l' | 'side_l' | 'q34front_l'

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
 *   source      —— 动作参考图（真人照片，用来提骨架）
 *   charSource  —— 角色图（**第一张**，身份来源）
 *   charSources —— 额外的角色图（正面/侧面/细节图一起喂，提升身份一致度）
 *
 * 为什么要有数组：Qwen 的多图条件是"同一个 vision context"，多给几张同一
 * 角色的不同角度，模型对发型、服装细节的把握明显更稳。后端 `_ref_image_list`
 * 会把 `charSources` 归一成列表，`charSource` 作为兼容旧数据的单张兜底。
 */
export type NodeCharSource = { nodeId?: string; path?: string }

/** 一个节点最多挂几张角色参考图（与后端 MAX_REF_IMAGES 保持一致）。 */
export const MAX_CHAR_SOURCES = 6

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
  /** 角色迁移引擎：
   *  - `wpc`    旧路线：角色图 + 姿态条件图 + 自写指令。姿态跟得紧，外观容易飘。
   *  - `wvnccs` 官方 VNCCS 工作流复刻：专用 LoRA + 固定指令 `replace the pose of
   *             <image 2> with the pose of <image 1>`。实测保角色一致性好得多
   *             （画风不变、头不丢、不多腿），代价是姿态精度略低。
   */
  pcEngine?: 'wpc' | 'wvnccs'
  /**
   * 角色迁移**批量**：用项目里的哪个姿势集（`assets/projects/<项目>/poses/<名字>/`）。
   * 空 = 走单张那张动作参考图。
   */
  pcPoseSet?: string
  /** 角色迁移批量：输出版（成片集）的文件夹名，落在 `pose_frames/<名字>/` */
  pcOut?: string
  /**
   * 角色迁移：人物的**朝向**（写进指令，是模型唯一可靠的朝向线索）。
   *
   * 为什么必须有这个输入：骨架图本身分不清角度——正/背面在 2D 投影里互为镜像，
   * 90° 侧面又收成一条竖线。实测（同 seed、同角色、VNCCS）：
   *   白底骨架 90° + 无方位词 → 出**正面**（错）；
   *   同一张图 + `side view, facing right` → 出**右侧面**（对）。
   * 也试过 VNCCS 的"人偶渲染"路线（3D 人形 / 人偶+骨架叠加），两种都被还原成
   * 正面——我们的渲染不是那个 LoRA 的训练格式，它读不出来。
   * 所以角度按 VNCCS 的本意做成**显式输入**，而不是让模型猜。
   * `auto` = 上游是骨骼编辑器时用它的方位角。
   */
  pcView?: PcView
  twoStage: boolean       // 姿态序列：两阶段（先出姿态草图再换皮），动作更准但慢一倍
  /** 底色：white=白底（默认）/ dark=深色底（白毛角色）/ transparent=模型直接出 alpha */
  bgTone: BgTone
  ssCols: number          // 序列帧替换：精灵图列数（0 = 自动检测）
  ssRows: number          // 序列帧替换：精灵图行数（0 = 自动检测）
  ssMode: SwapMode        // 序列帧替换：AI 重绘 还是 确定性换皮
  /** 快速模式：挂 Lightning 少步蒸馏 LoRA + 步数降到 8（实测 -46%，画质几乎不掉） */
  fastMode: boolean
  // 扩图（outpaint）：往哪个方向扩、扩多少
  outpDirection: 'right' | 'left' | 'up' | 'down' | 'all'
  outpExpand: number     // 相对对应边长的比例（0.5 = 扩一半）
  ssRefMode: RefMode      // 序列帧替换：动作参考图的解释方式
  ssPalette: number       // 序列帧替换：共享色板色数
  ssCandidates: number    // 序列帧替换：每帧生成几个候选再挑最好的
  ssGateRetries: number   // 序列帧替换：候选都不合格时补提几次
  ssEngineMode: EngineMode // 序列帧替换：**已废弃**（AnimateDiff 随 SD 一起删了）
  /** 序列帧替换的逐帧引擎：
   *  - `wpc`    角色图 + 轮廓/骨架 + 自写指令（旧路线）
   *  - `wvnccs` 逐帧挂 VNCCS LoRA、官方固定指令、条件图不灰度化。
   *             **帧间角色一致性明显更好**（8 帧不会换脸换衣服），单帧更慢。
   */
  ssEngine?: 'wpc' | 'wvnccs'
  /** 序列帧替换：VNCCS LoRA 权重（只在该引擎下生效） */
  ssLoraScale?: number
  /**
   * 姿态序列（W4）：骨架帧**目录**。可以填目录，也可以填通配
   * （assets/_pose_walk_*.png）。后端展开并自然排序——帧名 frame_2 < frame_10，
   * 字典序会把 10 帧以上排乱，动画顺序就错了。
   */
  psDir: string
  /** 姿态序列的逐帧引擎：
   *  - `wpc`（旧）—— 角色图 + 骨架图 + 自写指令。软引导，腿部常被角色图的站姿覆盖。
   *  - `wvnccs` —— 挂 VNCCS LoRA、官方固定指令 `replace the pose of <image 2>...`、
   *    图序 [骨架, 角色]。**帧间身份更稳**，但必须给角色图。
   */
  psEngine?: 'wpc' | 'wvnccs'
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
   * **模型原样**：勾上就一个像素都不动（默认关）。
   *
   * 默认会做两件很轻的事：
   *  1. `sanitize_alpha` —— alpha<16 清零、>240 拉满（模型的 alpha 会带
   *     1~15 的脏毛边，透明区 RGB 还是 VAE 噪声，忽略 alpha 的工具会露馅）；
   *  2. `_defringe_white_matte` —— 把半透明边缘的「白底混色」还原成前景色
   *     （实测过渡带亮度 237.8、实心区 206.4，87% 接近纯白；不还原的话
   *     深色背景上一圈白边）。**只改 RGB，alpha 一个像素都不动。**
   *
   * 勾上就是模型吐出来什么样、存什么样。
   */
  raw_output: boolean
  /**
   * **旧存档兼容字段，UI 已不提供节点级底模选择。**
   *
   * 底模是全局唯一的 Qwen-Image-2.1（SD 权重已归档），节点里再放一个只有
   * 一项的下拉纯属噪音。保留字段是为了旧画布 JSON 能原样反序列化；
   * `App.migrate()` 会把非 Qwen 的旧值（sd15 / sdxl …）清成空串，
   * 免得它们被提交后让 `engine._check_model` 直接抛错。
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
  /**
   * 批量任务的进度（表情批量生成用）。
   *
   * 一批几十张要跑几分钟，只给 `queued/running` 的话用户完全不知道还剩多久；
   * 后端 `emotiongen.run_batch` 每张更新一次。
   */
  /**
   * 任务进度。`step` 只有「一键出全套」会填（underwear / poses / emotions），
   * 因为那一条任务内部跑了好几步，前端要知道现在在哪一步。
   * `done/total` 在流水线里是**加权张数**（不是整数）。
   */
  progress?: { done: number; total: number; current?: string; step?: string }
  /** 批量任务的结果摘要（表情批量：`count` / `failed[]` / `items[]`） */
  result?: Record<string, unknown>
}

/** 节点视图。'result' = 出图后默认的结果优先视图：只给结果与摘要，
 *  参数收起来（ComfyUI 折叠节点 / n8n 画布节点只显示状态，参数进详情）。 */
export type NodeView = 'expanded' | 'image' | 'result'

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
  /** 角色迁移节点：角色图（身份来源，第一张） */
  charSource?: NodeCharSource
  /** 角色迁移节点：额外角色图（多角度参考）。空数组等同于只给 charSource */
  charSources?: NodeCharSource[]
  image?: NodeImage
  /**
   * 人偶编辑器节点：导出的**白底灰 3D 人偶图**（官方 PoseStudio capture）。
   *
   * VNCCS 的姿态 LoRA 就是在这种图上训的；以前那张 OpenPose-18 线框骨架
   * 不在它的分布里，姿态贴合明显更弱，所以现在只留人偶图这一个产物。
   *
   * 老工程里可能还留着这个字段（那时它指人偶图、image 是线框骨架），保留类型
   * 是为了能干净地读回旧存档，新代码不再往里写。
   */
  mannequin?: NodeImage
  /** 3D 人偶编辑器的姿势与机位（官方 pose JSON + yaw） */
  pose?: unknown
  /** 导出机位（度）。和 pose 一块存，重开时能还原当时选的朝向 */
  yaw?: number
  /** 「从图片」提好的 2D 关节点，投递给 iframe 用（收完就被 pose 取代） */
  poseImport?: PoseImportRequest | null
  /**
   * 外部（姿势库）套过来的姿势。**带目标 nodeId**：一次只给一个节点，
   * 别的人偶节点必须忽略它。`tick` 变化才触发应用（避免重复套用）。
   */
  poseApply?: { nodeId: string; pose: unknown; yaw: number; tick: number } | null
  /**
   * 角色迁移批量：当前在看的成片集名字（服务端 `pose_frames/<名字>/`）。
   * 帧候选与"选了哪一张"都在服务端的 render.json 里，这里只记名字。
   */
  poseRender?: string
  job?: JobState
}

export type Viewport = { x: number; y: number; z: number }

export type FavoritePrompt = {
  id: string
  /** 正向提示词 */
  text: string
  /**
   * 收藏时节点上的负向提示词（可空）。
   *
   * **一起收**：负向词和正向词是一套搭配（"no text, watermark" 只对某些主体有意义），
   * 只存正向、应用时把负向留成节点原样，等于收藏了个半成品。
   * 旧存档没有这个字段，读回来按空串处理。
   */
  negative?: string
  at: number
}

/**
 * 姿势库里的一条。
 *
 * 用户的**复用资产**，不是项目内容：同一个「挥剑姿势」要在十个角色上用，
 * 存进项目里等于每个工程复制一份。所以跟提示词收藏一样放 localStorage。
 */
export type PosePreset = {
  id: string
  name: string
  /** 官方 PoseStudio 的 pose JSON（不透明） */
  pose: unknown
  /** 存下当时的导出机位：某些姿势换个朝向就不是那个动作了 */
  yaw: number
  /** 缩略图（dataURL，可选）。没有也能存，只是浏览面板里没有预览 */
  preview?: string
  at: number
}
