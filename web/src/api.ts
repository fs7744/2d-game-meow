import type { GenParams, JobState, LoraInfo } from './types'

const base = ''                                   // 生产同源；开发由 vite proxy 转发

function basename(p: string): string {
  const parts = p.split(/[\\/]/)
  return parts[parts.length - 1] || p
}

export function fileUrl(localPath: string): string {
  return `${base}/files/${encodeURIComponent(basename(localPath))}`
}

export type SubmitResult = { job_id: string }

export async function submitTxt2Img(params: GenParams): Promise<SubmitResult> {
  const r = await fetch(`${base}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      workflow: 'w1',
      params: {
        prompt: params.prompt,
        negative: params.negative || undefined,
        width: params.width,
        height: params.height,
        steps: params.steps,
        cfg: params.cfg,
        seed: params.seed,
        preset: params.preset,
        pixel: params.pixel,
        palette: params.palette,
        matte: params.matte,
        // 空串表示不挂 LoRA；后端会把名字解析成绝对路径
        lora: params.lora || undefined,
        lora_scale: params.loraScale,
      },
    }),
  })
  if (!r.ok) throw new Error(`提交失败：HTTP ${r.status}`)
  return r.json()
}

type ApiJob = {
  job_id: string
  status: string
  error?: string
  result?: { file?: string; meta?: Record<string, unknown> }
}

export async function fetchJob(jobId: string): Promise<JobState> {
  const r = await fetch(`${base}/jobs/${jobId}`)
  if (!r.ok) throw new Error(`查询失败：HTTP ${r.status}`)
  const j = (await r.json()) as ApiJob
  const meta = j.result?.meta ?? {}
  const file = j.result?.file
  return {
    id: j.job_id,
    status: (j.status as JobState['status']) ?? 'queued',
    error: j.error,
    url: file ? fileUrl(file) : undefined,
    seed: typeof meta.seed === 'number' ? (meta.seed as number) : undefined,
    seconds: typeof meta.seconds === 'number' ? (meta.seconds as number) : undefined,
    peakVram:
      typeof meta.peak_vram_gb === 'number' ? (meta.peak_vram_gb as number) : undefined,
  }
}

export type TrainState = {
  train_id: string
  status: 'queued' | 'running' | 'done' | 'failed'
  stage?: string
  name?: string
  step?: number
  total?: number
  loss?: number
  elapsed?: number
  peak_vram_gb?: number
  trainable_params?: number
  samples?: number
  path?: string
  seconds?: number
  error?: string
}

export type TrainReq = {
  name: string
  data_dir: string
  caption?: string
  trigger?: string
  label?: string
  rank?: number
  steps?: number
  epochs?: number
  resolution?: number
}

export async function startTrain(req: TrainReq): Promise<{ train_id: string }> {
  const r = await fetch(`${base}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!r.ok) throw new Error(`提交失败：HTTP ${r.status}`)
  return r.json()
}

export async function fetchTrain(id: string): Promise<TrainState> {
  const r = await fetch(`${base}/train/${id}`)
  if (!r.ok) throw new Error(`查询失败：HTTP ${r.status}`)
  return (await r.json()) as TrainState
}

export async function fetchLoras(): Promise<LoraInfo[]> {
  const r = await fetch(`${base}/loras`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const j = await r.json()
  const items: LoraInfo[] = Array.isArray(j.items) ? j.items : []
  return items.filter((x) => !x.hidden)
}

export async function health(): Promise<{ ok: boolean; device?: string }> {
  const r = await fetch(`${base}/health`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const j = await r.json()
  return { ok: !!j.ok, device: j?.engine?.device }
}
