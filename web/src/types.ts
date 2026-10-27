export type Preset = 'sprite' | 'portrait' | 'none'

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
