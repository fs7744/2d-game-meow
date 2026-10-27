export type Preset = 'sprite' | 'portrait' | 'wuxia' | 'none'

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
  startedAt?: number
}

export type NodeView = 'expanded' | 'image'

export type GenNode = {
  id: string
  kind: 'gen'
  x: number
  y: number
  params: GenParams
  view: NodeView
  job?: JobState
}

export type Viewport = { x: number; y: number; z: number }

export type FavoritePrompt = {
  id: string
  text: string
  at: number
}
