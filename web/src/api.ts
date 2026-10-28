import type { GenParams, JobState, LoraInfo } from './types'

const base = ''                                   // 生产同源；开发由 vite proxy 转发

function basename(p: string): string {
  const parts = p.split(/[\\/]/)
  return parts[parts.length - 1] || p
}

/**
 * 产物 URL。产物现在在 assets/projects/<项目id>/ 子目录里，只取 basename 会 404，
 * 所以从绝对路径里截出「assets/ 之后」的部分，逐段编码。
 */
export function fileUrl(localPath: string): string {
  const p = localPath.replace(/\\/g, '/')
  const i = p.lastIndexOf('/assets/')
  const rel = i >= 0 ? p.slice(i + '/assets/'.length) : basename(localPath)
  return `${base}/files/${rel.split('/').map(encodeURIComponent).join('/')}`
}

export type SubmitResult = { job_id: string }

export type UploadedImage = {
  path: string
  url: string
  w: number
  h: number
  format: string
}

/**
 * 上传一张本地图。请求体是**原始字节**（不是 multipart），后端省掉一个依赖。
 * 返回的 path 是磁盘绝对路径——图生图要拿它当 init_image。
 */
export async function uploadImage(file: File, project?: string): Promise<UploadedImage> {
  const q = project ? `?name=${encodeURIComponent(file.name)}&project=${encodeURIComponent(project)}`
    : `?name=${encodeURIComponent(file.name)}`
  const r = await fetch(`${base}/upload${q}`, {
    method: 'POST',
    headers: { 'Content-Type': file.type || 'application/octet-stream' },
    body: file,
  })
  if (!r.ok) throw new Error(`上传失败：HTTP ${r.status}`)
  return r.json() as Promise<UploadedImage>
}

export type DeleteResult = {
  deleted: string[]
  skipped: string[]
  not_found: string[]
  count: number
}

/**
 * 真删：把产物从磁盘上删掉（连带 sidecar 与 raw），并清掉资产索引里的条目。
 * 后端会拒绝 assets 目录之外的路径。
 */
export async function deleteAssets(paths: string[]): Promise<DeleteResult> {
  if (paths.length === 0) return { deleted: [], skipped: [], not_found: [], count: 0 }
  const r = await fetch(`${base}/assets`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paths }),
  })
  if (!r.ok) throw new Error(`删除失败：HTTP ${r.status}`)
  return r.json() as Promise<DeleteResult>
}

export type SubmitOpts = { model?: string; matteBackend?: string; scope?: string }

export type MaskInfo = {
  path: string
  url: string
  width: number
  height: number
  coverage: number
}

/** 保存局部重绘的蒙版（PNG 原始字节），同名覆盖，后端放在项目目录的 masks/ 下。 */
export async function saveMask(
  blob: Blob,
  name: string,
  project?: string,
): Promise<MaskInfo> {
  const q = `?name=${encodeURIComponent(name)}${project ? `&project=${encodeURIComponent(project)}` : ''}`
  const r = await fetch(`${base}/mask${q}`, {
    method: 'POST',
    headers: { 'Content-Type': 'image/png' },
    body: blob,
  })
  if (!r.ok) throw new Error(`蒙版保存失败：HTTP ${r.status}`)
  return r.json() as Promise<MaskInfo>
}

/** 两个工作流共用的参数。w2 不接受 width/height（尺寸由源图决定）。 */
function common(params: GenParams, opts: SubmitOpts) {
  return {
    prompt: params.prompt,
    negative: params.negative || undefined,
    steps: params.steps,
    cfg: params.cfg,
    seed: params.seed,
    preset: params.preset,
    pixel: params.pixel,
    palette: params.palette,
    matte: params.matte,
    clip_skip: params.clipSkip,
    // 底模与抠图后端是全局设置，不在节点参数里
    model: opts.model || 'sd15',
    matte_backend: opts.matteBackend || 'auto',
    // 空串表示不挂 LoRA；后端会把名字解析成绝对路径
    lora: params.lora || undefined,
    lora_scale: params.loraScale,
    // 项目作用域：产物写进 assets/projects/<scope>/，项目之间互不可见
    scope: opts.scope || undefined,
  }
}

async function post(workflow: 'w1' | 'w2' | 'w3' | 'wbg', params: Record<string, unknown>) {
  const r = await fetch(`${base}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workflow, params }),
  })
  if (!r.ok) throw new Error(`提交失败：HTTP ${r.status}`)
  return r.json() as Promise<SubmitResult>
}

export function submitTxt2Img(params: GenParams, opts: SubmitOpts = {}) {
  return post('w1', { ...common(params, opts), width: params.width, height: params.height })
}

/**
 * 图生图。initImage 必须是**磁盘上的绝对路径**（后端直接 Image.open）。
 * 调用方优先给 raw（未像素化的原图），拿成品精灵回炉会把色块重新糊成插画。
 */
export function submitImg2Img(params: GenParams, opts: SubmitOpts, initImage: string) {
  return post('w2', {
    ...common(params, opts),
    init_image: initImage,
    strength: params.strength,
  })
}

/**
 * 局部重绘（w3）。后端 inpaint() 只认这几个参数，别的别塞——多传一个就 TypeError。
 * 源图与蒙版都是磁盘路径：蒙版是笔刷画完上传到项目目录的那张 PNG。
 */
export function submitInpaint(
  params: GenParams,
  opts: SubmitOpts,
  image: string,
  mask: string,
) {
  return post('w3', {
    image,
    mask,
    prompt: params.prompt,
    negative: params.negative || undefined,
    strength: params.strength,
    steps: params.steps,
    cfg: params.cfg,
    seed: params.seed,
    preset: params.preset,
    lora: params.lora || undefined,
    lora_scale: params.loraScale,
    model: opts.model || 'sd15',
    scope: opts.scope || undefined,
  })
}

/**
 * 抠图去背景 / 换背景。纯图像处理，不占显存。
 * background 为空 = 透明底（去背景）；给了路径 = 换背景（前景叠上去）。
 */
export function submitMatte(
  _params: GenParams,
  opts: SubmitOpts,
  initImage: string,
  background?: string,
  bgColor?: string,
) {
  return post('wbg', {
    init_image: initImage,
    background: background || undefined,
    bg_color: bgColor || undefined,
    scope: opts.scope || undefined,
  })
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
    subjects:
      typeof meta.subjects === 'number' ? (meta.subjects as number) : undefined,
    path: typeof file === 'string' ? file : undefined,
    raw: typeof meta.raw === 'string' ? (meta.raw as string) : undefined,
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

export async function fetchModels(): Promise<string[]> {
  const r = await fetch(`${base}/models`)
  if (!r.ok) return ['sd15']
  const j = await r.json()
  return Array.isArray(j.available) ? j.available : ['sd15']
}

export async function fetchLoras(baseModel?: string): Promise<LoraInfo[]> {
  const r = await fetch(`${base}/loras`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const j = await r.json()
  const items: LoraInfo[] = Array.isArray(j.items) ? j.items : []
  // SD1.5 的 LoRA 装在 SDXL 上完全无效，会静默出废图 —— 按基座直接过滤掉
  return items.filter((x) => !x.hidden && (!baseModel || (x.base || 'sd15') === baseModel))
}

export async function health(): Promise<{ ok: boolean; device?: string }> {
  const r = await fetch(`${base}/health`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const j = await r.json()
  return { ok: !!j.ok, device: j?.engine?.device }
}
