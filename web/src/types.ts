export type Preset = 'sprite' | 'portrait' | 'wuxia' | 'pixel' | 'none'

/** 节点类型：图生成（w1）/ 图生图（w2）/ 图片（拖进来的本地图） */
export type NodeKind = 'gen' | 'img2img' | 'image'

/** 图片节点挂着的图 */
export type NodeImage = {
  path: string   // 磁盘绝对路径（当图生图的源时要传给后端）
  url: string    // /files/... 展示用
  w: number      // 展示宽度（px）；高度按原图比例
  h: number      // 原图高度
}

/** 图生图的源：指向画布上另一个节点的产物，或直接给本地绝对路径 */
export type NodeSource = { nodeId?: string; path?: string }

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
  image?: NodeImage
  job?: JobState
}

export type Viewport = { x: number; y: number; z: number }

export type FavoritePrompt = {
  id: string
  text: string
  at: number
}
