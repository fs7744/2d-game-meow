import type { GenParams, JobState, ModelRecommend } from './types'

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
    // 底模与抠图后端是全局设置，不在节点参数里
    model: opts.model || 'qwen-image-2.1',
    matte_backend: opts.matteBackend || 'auto',
    // 透明背景：只有 Qwen-Image-2.1 原生支持（VAE 第 4 通道）。
    // 后端会在这个字段为真时给提示词加双段式声明，否则 alpha 不会激活。
    transparent: params.transparent || undefined,
    // 项目作用域：产物写进 assets/projects/<scope>/，项目之间互不可见
    scope: opts.scope || undefined,
  }
}

/**
 * 提交生成的超时（毫秒）。
 *
 * **必须有超时**：`fetch` 没有默认超时，服务端不响应时请求会挂到天荒地老。
 * 前端在提交前把节点标成「排队中」，请求挂住 = 节点永远排队中、按钮禁用。
 * 30 秒足够服务端把任务塞进队列（真出图是后面的事，不占这个连接）。
 */
const SUBMIT_TIMEOUT_MS = 30_000

async function post(
  workflow: 'w1' | 'w2' | 'w3' | 'w4' | 'w5' | 'wbg' | 'wpx' | 'wpc' | 'wss'
    | 'topixel' | 'wap' | 'wout' | 'woutfit' | 'wface' | 'wvnccs',
  params: Record<string, unknown>,
) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), SUBMIT_TIMEOUT_MS)
  let r: Response
  try {
    r = await fetch(`${base}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow, params }),
      signal: ctrl.signal,
    })
  } catch (e) {
    // 区分「超时」与「连不上」——用户看的提示不一样
    if (e instanceof Error && e.name === 'AbortError') {
      throw new Error('提交超时：服务端 30 秒没响应，请检查后端是否在运行')
    }
    throw new Error(`提交失败：${e instanceof Error ? e.message : '网络错误'}`)
  } finally {
    clearTimeout(timer)
  }
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
  })
}

/**
 * 局部重绘（w3）。源图与蒙版都是磁盘路径：蒙版是笔刷画完上传到项目目录的那张 PNG。
 *
 * **别再往载荷里塞后端没有的键**：`app._run` 会按目标函数签名过滤，多传的会被
 * 丢掉并写进产物的 `meta.params_ignored`——不会崩，但每张图都会挂一条没用的警告。
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
    steps: params.steps,
    cfg: params.cfg,
    seed: params.seed,
    preset: params.preset,
    model: opts.model || 'qwen-image-2.1',
    scope: opts.scope || undefined,
  })
}

/**
 * 局部 AI 重绘（w5）。裁剪「选区+上下文」→ 通用模型重画 → 贴回。
 * 与 w3 的区别：w5 多一步色板吸附，并能选 `strategy=crop` 做对照实验；
 * 两者在 Qwen 上走同一个「整图指令改图 + 蒙版贴回」机制。
 */
export function submitAiRepaint(
  params: GenParams,
  opts: SubmitOpts,
  initImage: string,
  mask: string,
) {
  return post('w5', {
    image: initImage,
    mask,
    prompt: params.prompt,
    negative: params.negative || undefined,
    steps: params.steps,
    cfg: params.cfg,
    seed: params.seed,
    preset: params.preset,
    model: opts.model || 'qwen-image-2.1',
    scope: opts.scope || undefined,
    context: params.rpContext,
    expand: params.rpExpand,
    feather: params.rpFeather,
    match_color: params.rpMatchColor,
    palette_snap: params.rpPaletteSnap,
  })
}

/**
 * 完美像素转换（wpx）。任意图 → 像素完美（每格一色、边界对齐）的像素画。
 * 不跑模型，纯确定性处理。
 */
export function submitPixelify(
  params: GenParams,
  opts: SubmitOpts,
  initImage: string,
) {
  return post('wpx', {
    init_image: initImage,
    // target 为 0 时后端用自动检测到的网格，不给 target
    target: params.pxTarget > 0 ? [params.pxTarget, params.pxTarget] : undefined,
    colors: params.pxColors,
    sample_method: params.pxSample,
    grid_snap: params.pxGridSnap,
    transparent_bg: params.pxTransparentBg,
    // snap_to=0 表示不吸附，保留检测到的原始网格
    snap_to: params.pxSnapTo > 0 ? params.pxSnapTo : null,
    scope: opts.scope || undefined,
  })
}

/**
 * 任意图 → 像素（topixel，旧名 wap）。
 *
 * 两步走：先出高质量大图，再压成像素画——比让 SD 直接画像素好得多
 * （实测 SD 直接画常裁到腿、颜色是连续渐变）。**不只动漫**：照片、插画、
 * 3D 渲染都能转；只有「没给源图、需要用描述先生成」时才会走动漫底模。
 */
export function submitToPixel(
  params: GenParams,
  opts: SubmitOpts,
  initImage?: string,
) {
  return post('topixel', {
    init_image: initImage || undefined,
    prompt: params.prompt,
    negative: params.negative || undefined,
    target: params.apTarget > 0 ? [params.apTarget, params.apTarget] : undefined,
    // 透传画布：后端默认 1024²，节点上的 768 不传就会被忽略（实测 45s vs 25s，
    // 而且 1024² 角色占比偏大、容易顶边）
    source_width: params.width || undefined,
    source_height: params.height || undefined,
    colors: params.apColors,
    pre_enhance: params.apPreEnhance,
    edge_rebuild: params.apEdgeRebuild,
    // 底模只有 qwen-image-2.1（SDXL 那条路早删了）
    model: opts.model || 'qwen-image-2.1',
    matte: params.apMatte,
    // 原生透明底：Qwen 直接出 alpha，比「生成后再抠」干净得多
    //（白猫白底这类场景，抠图会把身体一起切掉——实测过）
    transparent: params.transparent || undefined,
    seed: params.seed,
    scope: opts.scope || undefined,
  })
}

/**
 * 角色迁移（wpc）。动作参考图 → 骨架/轮廓（条件图）+ 角色图，两张一起进
 * Qwen 的 vision context，生成角色做出参考动作的图。
 */
/**
 * 扩图（wout）：把 `image` 往 `direction` 方向扩 `expand` 比例。
 *
 * 靠 Outpaint LoRA + 中性灰 padding；提示词由后端加 LoRA 触发句，
 * 这里传的 prompt 只用来描述**原图里看得见的**内容（给模型当上下文）。
 */
export function submitOutpaint(
  params: GenParams,
  opts: SubmitOpts,
  image: string,
) {
  return post('wout', {
    image,
    direction: params.outpDirection || 'right',
    expand: params.outpExpand ?? 0.5,
    prompt: params.prompt,
    negative: params.negative || undefined,
    steps: params.steps,
    cfg: params.cfg,
    seed: params.seed,
    ...opts,
  })
}

/** 换装（woutfit）：把 `outfit`（图2）的服装穿到 `image`（图1）的人物身上 */
export function submitOutfitSwap(
  params: GenParams,
  opts: SubmitOpts,
  image: string,
  outfit: string,
  /** 额外的服装参考图（同一套的背面/细节），细节会更准 */
  outfits?: string[],
) {
  return post('woutfit', {
    image, outfit,
    outfits: outfits && outfits.length > 1 ? outfits : undefined,
    prompt: params.prompt,
    negative: params.negative || undefined,
    // **不传 width/height**：后端默认跟随基底图尺寸。传死 768 会把
    // 非方形的基底图人物压扁（与函数语义相反）。
    steps: params.steps, cfg: params.cfg, seed: params.seed,
    ...opts,
  })
}

/** 换脸（wface）：把 `face`（图2）的脸/发型换到 `image`（图1）的人物上 */
export function submitFaceSwap(
  params: GenParams,
  opts: SubmitOpts,
  image: string,
  face: string,
  /** 额外的人脸参考图（正脸/侧脸），五官更稳 */
  faces?: string[],
) {
  return post('wface', {
    image, face,
    faces: faces && faces.length > 1 ? faces : undefined,
    prompt: params.prompt,
    negative: params.negative || undefined,
    // 同上：不传 width/height，输出跟随基底图
    steps: params.steps, cfg: params.cfg, seed: params.seed,
    ...opts,
  })
}

export function submitPoseClone(
  params: GenParams,
  opts: SubmitOpts,
  poseImage: string,
  charImage: string,
  /** 额外的角色参考图（正面/侧面/细节）。第一张仍走 char_image 保持兼容 */
  charImages?: string[],
) {
  return post('wpc', {
    pose_image: poseImage,
    char_image: charImage,
    char_images: charImages && charImages.length > 1 ? charImages : undefined,
    prompt: params.prompt,
    negative: params.negative || undefined,
    width: params.width,
    height: params.height,
    steps: params.steps,
    cfg: params.cfg,
    seed: params.seed,
    preset: params.preset,
    model: opts.model || 'qwen-image-2.1',
    scope: opts.scope || undefined,
    fill: params.pcFill,
    // 参考模式：auto=先试骨架，认不出人自动降级到轮廓
    ref_mode: params.pcRefMode,
    bg: params.bgTone,
  })
}

/**
 * 姿态迁移（VNCCS 路线）：挂专用 LoRA + 双图指令，**不用 ControlNet**。
 *
 * 与 `submitPoseClone` 的区别：那边是"角色图 + 姿态条件图"的自写指令，
 * 这边是官方工作流的固定模板 `replace the pose of <image 2> with the pose of
 * <image 1>`，图序相反（骨架在前），并且挂 VNCCS LoRA。实测保角色一致性
 * 好得多（画风不变、头不丢、不会多腿），代价是姿态精度略低。
 *
 * `width/height` 不传时后端按**骨架比例**自动定画布——方形画布会把人物
 * 撑满、头顶出画面。
 */
export function submitVnccsPose(
  params: GenParams,
  opts: SubmitOpts,
  poseImage: string,
  charImages: string[],
) {
  return post('wvnccs', {
    pose_image: poseImage,
    char_images: charImages,
    prompt: params.prompt,
    negative: params.negative || undefined,
    // **不传 width/height**：让后端按骨架比例自动定画布。传死 768 会变成
    // 方形画布，模型把人物撑满、头顶出画面（实测过，这是"头丢了"的元凶）。
    // 想手动指定时由后端 `_vnccs_canvas_size(pose, w, h)` 接住。
    // 步数**不低于 20**：官方示例是 25，实测少步会把姿态和细节一起崩掉
    // （VNCCS LoRA 是按 25 步的完整采样训的，不是少步蒸馏）。
    steps: Math.max(20, params.steps),
    // 透明底在 VNCCS 下**是可用的**（实测主体 16.6%、半透明 2.1%、背景全透），
    // 不像 ControlNet 那条路会被控制信号压成半透膜。
    transparent: params.transparent === true || undefined,
    cfg: params.cfg,
    seed: params.seed,
    model: opts.model || 'qwen-image-2.1',
    scope: opts.scope || undefined,
    pixel: 0,
    palette: 0,
  })
}

/** /pose/extract 的返回：18 个 COCO 关节点（缺失为 null）+ 原图朝向猜测 */
export type PoseExtract = {
  points: ([number, number] | null)[]
  joints: number
  /** 只用来提示：2D 图分不出正面/背面，只有肩宽很窄时才有把握说是侧面 */
  azimuth_guess: number
}

/** 骨骼编辑器「从图片提取」：上传后拿磁盘路径，让后端跑 DWPose 提 18 个关节 */
export async function extractPose(path: string): Promise<PoseExtract> {
  const r = await fetch(`${base}/pose/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  if (!r.ok) {
    let msg = `HTTP ${r.status}`
    try {
      const j = await r.json()
      if (j?.detail) msg = String(j.detail)
    } catch { /* 响应不是 JSON，用状态码兜底 */ }
    throw new Error(msg)
  }
  return r.json() as Promise<PoseExtract>
}

/** /loras 返回的一个 Qwen LoRA（只含 diffusers 原生格式的） */
export type QwenLora = {
  name: string
  file: string
  size_mb: number
  tensors: number
}

/** 当前生效的 LoRA */
export type ActiveLora = { name: string; scale: number; adapter?: string }

/**
 * 可用的 Qwen LoRA + 当前生效的。
 *
 * **只列 diffusers 原生格式的**（key 前缀 `transformer.`）——ComfyUI 格式
 * （`diffusion_model.` + `lora_A/B`）挂上去会**静默不生效**，比报错更坏。
 * 转换：`python tools/convert_lora.py in.safetensors out.safetensors`
 */
export async function fetchLoras(): Promise<{ items: QwenLora[]; active: ActiveLora[] }> {
  const r = await fetch(`${base}/loras`)
  if (!r.ok) return { items: [], active: [] }
  const j = await r.json()
  return {
    items: Array.isArray(j.items) ? j.items : [],
    active: Array.isArray(j.active) ? j.active : [],
  }
}

/**
 * 挂载 LoRA（状态式：一直生效，直到下次 apply 或 unload）。
 * 传空数组 = 卸载全部。
 */
export async function applyLoras(
  items: { name: string; scale: number }[],
): Promise<{ active: string[]; items: ActiveLora[] }> {
  const r = await fetch(`${base}/loras/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  })
  if (!r.ok) {
    let msg = `HTTP ${r.status}`
    try {
      const j = await r.json()
      if (j?.detail) msg = String(j.detail)
    } catch { /* 响应不是 JSON，用状态码兜底 */ }
    throw new Error(msg)
  }
  return r.json()
}

/** 卸载全部 LoRA，回到基座模型 */
export async function unloadLoras(): Promise<void> {
  await fetch(`${base}/loras/unload`, { method: 'POST' })
}

/**
 * 开关**快速模式**：挂 Lightning 少步蒸馏 LoRA + 步数压到 8。
 * 实测（768²/RTX4090）：24 步 17.3s → 8 步 9.3s（-46%），画质几乎不掉。
 */
export async function setFastMode(on: boolean): Promise<{
  fast_mode: boolean
  lora: string | null
  max_steps: number | null
}> {
  const r = await fetch(`${base}/fast_mode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ on }),
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export type PoseDir = { path: string; count: number; sample: string[]; name: string }

/** 可当 W4 输入的骨架帧目录（后端扫 assets 下两层） */
export async function fetchPoseDirs(): Promise<PoseDir[]> {
  const r = await fetch(`${base}/pose_dirs`)
  if (!r.ok) return []
  const j = await r.json()
  return Array.isArray(j.items) ? (j.items as PoseDir[]) : []
}

/**
 * 姿态序列（w4）：一批骨架图 → 逐帧生成（多候选择优 + 质量闸门）→ 锚点对齐 → 精灵图。
 *
 * 与 wss 的区别：wss 是"拿现成的精灵图换角色"，w4 是"从骨架从零生成"，
 * 所以它能吃候选/闸门这套产线机制，也是 README 里推荐的全身帧产线。
 * poseDir 可以给目录，后端会展开成排序后的帧列表。
 */
export function submitPoseSeq(
  params: GenParams,
  opts: SubmitOpts,
  poseDir: string,
  charImage?: string,
  /** 额外角色参考图（正面/侧面），多图时身份更稳 */
  charImages?: string[],
  /** 逐帧引擎：wpc（旧） / wvnccs（LoRA + 官方指令） */
  engine?: 'wpc' | 'wvnccs',
) {
  return post('w4', {
    pose_images: poseDir,
    char_images: charImages && charImages.length > 1 ? charImages : undefined,
    engine: engine || undefined,
    // 角色图（可选）：给了它帧间身份更一致，但也会「抢姿态」——
    // 想两全就开 two_stage（先出姿态草图，再拿角色图换皮）。
    ref_image: charImage || undefined,
    prompt: params.prompt,
    negative: params.negative || undefined,
    preset: params.preset,
    model: opts.model || 'qwen-image-2.1',
    scope: opts.scope || undefined,
    width: params.width,
    height: params.height,
    steps: params.steps,
    cfg: params.cfg,
    seed: params.seed,
    // 列数：0 时按帧数自动排成一行（后端 cols 默认 8，这里显式给）
    cols: params.ssCols > 0 ? params.ssCols : 0,
    palette: params.ssPalette,
    pixel: params.pixel,
    matte: params.matte,
    // 多候选择优 + 闸门：w4 的默认值就是开着的，这里把用户设定透传
    candidates: params.ssCandidates,
    gate_retries: params.ssGateRetries,
    // 底色：白毛角色用白底会被抠图吃掉，dark 有对比才抠得干净
    bg: params.bgTone,
    // 两阶段：每帧两次生成（草图 + 换皮），姿态更准
    two_stage: params.twoStage || undefined,
  })
}

/**
 * 序列帧角色替换（wss）。把一段游戏动画（精灵图）里的角色换成目标角色。
 *
 * mode=recolor 走确定性换皮（不跑模型，帧间零漂移）；
 * mode=generate 跑模型（轮廓条件 + 角色参考图），能换体型。
 */
export function submitSpriteSwap(
  params: GenParams,
  opts: SubmitOpts,
  sheetImage: string,
  charImage: string,
  /** 额外角色参考图（正面/侧面），逐帧复用同一组，身份更稳 */
  charImages?: string[],
  /**
   * 逐帧引擎：
   * - `wpc`（旧）—— 角色图 + 轮廓/骨架 + 自写指令
   * - `wvnccs` —— 逐帧挂 VNCCS LoRA，帧间角色一致性明显更好（不换脸不换衣）
   */
  engine?: 'wpc' | 'wvnccs',
) {
  return post('wss', {
    sheet_image: sheetImage,
    char_image: charImage,
    char_images: charImages && charImages.length > 1 ? charImages : undefined,
    prompt: params.prompt,
    negative: params.negative || undefined,
    preset: params.preset,
    model: opts.model || 'qwen-image-2.1',
    scope: opts.scope || undefined,
    // 0 = 后端自动检测精灵图网格
    cols: params.ssCols > 0 ? params.ssCols : undefined,
    rows: params.ssRows > 0 ? params.ssRows : undefined,
    mode: params.ssMode,
    ref_mode: params.ssRefMode,
    bg: params.bgTone,
    palette: params.ssPalette,
    // 多候选择优：每帧生成几个候选，按「动作贴合 + 身份贴合」挑最好的
    candidates: params.ssCandidates,
    gate_retries: params.ssGateRetries,
    // 引擎：animatediff=时序注意力（帧间更稳），frame=逐帧多候选（更省显存）
    engine_mode: params.ssEngineMode,
    // 逐帧引擎选 VNCCS 时会自动把步数提到 20 以上（LoRA 不是少步蒸馏）
    engine: engine || undefined,
    vram_profile: params.ssVramProfile,
    width: params.width,
    height: params.height,
    steps: params.steps,
    cfg: params.cfg,
    seed: params.seed,
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

/** meta 里的值类型是 unknown；只要数字，别把字符串混进去。 */
function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
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
    seconds: typeof meta.seconds === 'number' ? (meta.seconds as number)
      : typeof meta.seconds_total === 'number' ? (meta.seconds_total as number) : undefined,
    peakVram:
      typeof meta.peak_vram_gb === 'number' ? (meta.peak_vram_gb as number) : undefined,
    subjects:
      typeof meta.subjects === 'number' ? (meta.subjects as number) : undefined,
    path: typeof file === 'string' ? file : undefined,
    raw: typeof meta.raw === 'string' ? (meta.raw as string) : undefined,
    candidates: num(meta.candidates),
    candidatesNote: typeof meta.candidates_note === 'string'
      ? (meta.candidates_note as string) : undefined,
    attempts: num(meta.attempts),
    gatePassed: num(meta.gate_passed),
    frames: num(meta.frames),
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

export type ModelCaps = {
  /** 是否支持原生透明背景（Qwen-Image-2.1 的 VAE 有第 4 通道） */
  transparent: boolean
  /** 图生图是否走「指令改图」语义（没有 strength 渐变） */
  instruction: boolean
  /** 是否支持挂 LoRA（Qwen 底模不吃 SD 的 LoRA） */
  lora: boolean
  /**
   * 这个底模能跑哪些工作流（txt2img / pose_clone / sprite_swap …）。
   * 节点的底模下拉按它过滤——Qwen 没有 SD 的 UNet/CLIP，姿态类节点
   * 选上它就是必崩，不如直接不列出来。
   */
  workflows: string[]
  /**
   * workflows 里哪些是**近似实现**：硬约束在这个底模上不存在，靠软引导或
   * 后处理凑出同样的产物（例：Qwen 没有 mask 接口 → 局部重绘 = 整图指令改图
   * + 蒙版贴回；没有 ControlNet → 姿态序列靠骨架图软引导）。
   * 节点上要打标，用户据此判断能不能拿它替代 SD 的那条路。
   */
  approx?: string[]
}

export type ModelList = {
  /** 后端的默认底模（engine.DEFAULT_MODEL）。前端初始选它，而不是猜一个 */
  current: string
  available: string[]
  recommend: Record<string, ModelRecommend>
  /** 底模名 → 架构（sd15 / sdxl / qwen21）。LoRA 按架构兼容，不能按底模名比 */
  arch: Record<string, string>
  /** 底模名 → 能力表（透明底 / 指令改图 / LoRA） */
  caps: Record<string, ModelCaps>
  /** 恒 false：refiner 已随 SD 删除，字段留着给老前端做 `if` 判空 */
  refiner: boolean
}

/** 全部工作流名（与后端 engine._ARCH_WORKFLOWS 同一套词）。老后端不给
 *  workflows 时按它兜底，前端至少不会把节点锁死。 */
const ALL_WF = ['txt2img', 'img2img', 'inpaint', 'poseswap', 'matte',
  'ai_repaint', 'pixelify', 'pose_clone', 'sprite_swap', 'topixel',
  'outpaint', 'outfit_swap', 'face_swap', 'vnccs_pose']

/** Qwen 上属于「近似实现」的工作流（老后端不给 approx 时兜底） */
// inpaint / ai_repaint 已走**真蒙版路径**（模型看得见蒙版），不再是近似
const QWEN_APPROX = ['poseswap', 'sprite_swap']

export async function fetchModels(): Promise<ModelList> {
  const r = await fetch(`${base}/models`)
  // 后端不可达时退回 Qwen 单档（SD 权重已归档）：宁可少显示，也不要让下拉是空的
  if (!r.ok) {
    return {
      current: 'qwen-image-2.1',
      available: ['qwen-image-2.1'], recommend: {},
      arch: { 'qwen-image-2.1': 'qwen21' },
      caps: {
        'qwen-image-2.1': {
          transparent: true, instruction: true, lora: false,
          workflows: ALL_WF, approx: QWEN_APPROX,
        },
      },
      refiner: false,
    }
  }
  const j = await r.json()
  const available: string[] = Array.isArray(j.available)
    ? j.available : ['qwen-image-2.1']
  const recommend: Record<string, ModelRecommend> =
    j.recommend && typeof j.recommend === 'object' ? j.recommend : {}
  // 老后端没有 arch 字段：按名字兜底（含 xl 就是 sdxl），至少不比以前差
  const arch: Record<string, string> = {}
  const caps: Record<string, ModelCaps> = {}
  for (const m of available) {
    arch[m] = (j.arch && j.arch[m])
      || (/qwen/i.test(m) ? 'qwen21' : /xl/i.test(m) ? 'sdxl' : 'sd15')
    // 老后端没有 caps：按架构推。qwen 的能力是固定的，推出来的结果一致
    const c = (j.caps && j.caps[m]) || {}
    const isQwen = arch[m] === 'qwen21'
    caps[m] = {
      transparent: c.transparent ?? isQwen,
      instruction: c.instruction ?? isQwen,
      lora: c.lora ?? !isQwen,
      workflows: Array.isArray(c.workflows) ? c.workflows
        // 老后端不给 workflows：qwen 现在十个全支持（近似的那几个见 approx）
        : ALL_WF,
      approx: Array.isArray(c.approx) ? c.approx : (isQwen ? QWEN_APPROX : []),
    }
  }
  const refiner = j.refiner === true
  // current 缺失时退回第一个可用底模：宁可给个能跑的，也别让下拉是空的
  const current = (typeof j.current === 'string' && available.includes(j.current))
    ? j.current : (available[0] || 'qwen-image-2.1')
  return { current, available, recommend, arch, caps, refiner }
}

export async function health(): Promise<{ ok: boolean; device?: string }> {
  const r = await fetch(`${base}/health`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const j = await r.json()
  return { ok: !!j.ok, device: j?.engine?.device }
}

// ---------------- 项目导入 / 导出 / 模板 ----------------
//
// 项目（画布 + 节点）存在**服务端磁盘**（assets/projects/<id>/project.json），
// 素材也在服务端 assets/ 下。导入导出把两边拼成一个 `.mproj` 单文件：
// 拷走它就等于拷走了整个项目。

/** 打包时的项目内容。meta 是元信息，payload 是节点等主体数据。 */
export type ProjectArchive = {
  meta: Record<string, unknown>
  payload: Record<string, unknown>
}

/** 导入/实例化后返回的项目内容。 */
export type ImportedProject = {
  meta: Record<string, unknown>
  payload: Record<string, unknown>
  imported_files: number
}

/**
 * 导出项目 → `.mproj` 归档字节。
 *
 * 后端会把节点里引用到的素材（图片、目录、通配）一起打包进去，
 * 所以这个文件是**自包含**的——换台机器导入也能用。
 */
export async function exportProject(archive: ProjectArchive): Promise<Blob> {
  const r = await fetch(`${base}/projects/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(archive),
  })
  if (!r.ok) throw new Error(`导出失败：HTTP ${r.status}`)
  return r.blob()
}

/** 导入一个 `.mproj` → 新项目的内容。 */
export async function importProject(
  file: Blob,
  scope: string,
  name?: string,
): Promise<ImportedProject> {
  const buf = await file.arrayBuffer()
  // base64：JSON 请求体传二进制（与 /upload 的裸字节不同，这里要 JSON 化）
  let bin = ''
  const bytes = new Uint8Array(buf)
  const CHUNK = 0x8000                       // 分块，避免 apply 参数过多爆栈
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)))
  }
  const r = await fetch(`${base}/projects/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scope, name: name || '', data: btoa(bin) }),
  })
  if (!r.ok) {
    let msg = `导入失败：HTTP ${r.status}`
    try {
      const j = await r.json()
      if (j && j.detail) msg = String(j.detail)
    } catch {
      /* 后端没给 JSON 就用默认文案 */
    }
    throw new Error(msg)
  }
  return r.json() as Promise<ImportedProject>
}

export type TemplateInfo = {
  id: string
  name: string
  created: number | string
  bytes: number
  nodes: number
}

/** 可用模板列表。 */
export async function fetchTemplates(): Promise<TemplateInfo[]> {
  const r = await fetch(`${base}/templates`)
  if (!r.ok) return []
  const j = await r.json()
  return Array.isArray(j.items) ? (j.items as TemplateInfo[]) : []
}

/** 把当前项目存成模板。 */
export async function saveTemplate(archive: ProjectArchive): Promise<{ id: string; name: string }> {
  const r = await fetch(`${base}/templates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(archive),
  })
  if (!r.ok) throw new Error(`保存模板失败：HTTP ${r.status}`)
  return r.json()
}

/** 从模板实例化一个新项目（素材复制到新项目目录）。 */
export async function instantiateTemplate(
  tid: string,
  scope: string,
  name?: string,
): Promise<ImportedProject> {
  const r = await fetch(`${base}/templates/${encodeURIComponent(tid)}/instantiate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scope, name: name || '', data: '' }),
  })
  if (!r.ok) throw new Error(`从模板新建失败：HTTP ${r.status}`)
  return r.json() as Promise<ImportedProject>
}

/** 删模板。 */
export async function deleteTemplate(tid: string): Promise<void> {
  const r = await fetch(`${base}/templates/${encodeURIComponent(tid)}`, { method: 'DELETE' })
  if (!r.ok) throw new Error(`删除模板失败：HTTP ${r.status}`)
}

// ---------------- 项目落盘存储 ----------------
//
// 项目定义存在**服务端磁盘**：`assets/projects/<id>/project.json`。
// 换机器、换浏览器、清缓存都不丢。**浏览器里不留任何项目内容。**

/** 服务端返回的项目摘要（不带节点）。 */
export type ServerProjectMeta = {
  id: string
  name: string
  created: string
  updated: string
  count: number
}

/** 服务端返回的项目定义。 */
export type ServerProject = {
  id: string
  name: string
  created: string
  updated: string
  nodes: unknown[]
  viewport: Record<string, unknown>
}

/**
 * 拉磁盘上的项目列表。
 *
 * **失败返回空数组而不是抛错**：后端没连上时，前端要能用缓存继续工作
 * （离线兜底），不能因为拉列表失败就整个应用打不开。
 */
export async function fetchServerProjects(): Promise<{
  items: ServerProjectMeta[]
  current: string
  /** 后端是否可达。false = 拉不到（服务没起 / 网络断） */
  reachable: boolean
}> {
  try {
    const r = await fetch(`${base}/projects`)
    if (!r.ok) return { items: [], current: '', reachable: false }
    const j = await r.json()
    return {
      items: Array.isArray(j.items) ? (j.items as ServerProjectMeta[]) : [],
      current: typeof j.current === 'string' ? j.current : '',
      reachable: true,
    }
  } catch {
    return { items: [], current: '', reachable: false }
  }
}

/** 拉一个项目的完整定义。失败返回 null（本地缓存里可能还有）。 */
export async function fetchServerProject(id: string): Promise<ServerProject | null> {
  try {
    const r = await fetch(`${base}/projects/${encodeURIComponent(id)}`)
    if (!r.ok) return null
    return (await r.json()) as ServerProject
  } catch {
    return null
  }
}

/** 保存项目定义的结果。 */
export type SaveResult =
  | { ok: true }
  /** 409：id 被占用（新建撞车）或内容被别人改过 */
  | { ok: false; conflict: true; message: string }
  | { ok: false; conflict: false; message: string }

/**
 * 把项目定义保存到服务端磁盘。
 *
 * **409 冲突要区分开**：
 *  · `create=true` 且 id 已被占用 → 两个标签页撞了 id；
 *  · `expect_updated` 对不上 → 别人改过这个项目了。
 * 两种都不能默默覆盖，得让用户知道。
 */
export async function saveServerProject(p: {
  id: string
  name: string
  created: string
  updated: string
  nodes: unknown[]
  viewport: Record<string, unknown>
  /** 新建时为 true（id 已存在则 409） */
  create?: boolean
  /** 客户端读到的 updated；用于检出并发修改 */
  expectUpdated?: string
}): Promise<SaveResult> {
  try {
    const r = await fetch(`${base}/projects/${encodeURIComponent(p.id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: p.id, name: p.name, created: p.created, updated: p.updated,
        nodes: p.nodes, viewport: p.viewport,
        create: !!p.create,
        expect_updated: p.expectUpdated ?? null,
      }),
    })
    if (r.ok) return { ok: true }
    let msg = `保存失败：HTTP ${r.status}`
    try {
      const j = await r.json()
      if (j && j.detail) msg = String(j.detail)
    } catch {
      /* 后端没给 JSON 就用默认文案 */
    }
    return { ok: false, conflict: r.status === 409, message: msg }
  } catch (e) {
    return {
      ok: false, conflict: false,
      message: `保存失败：${e instanceof Error ? e.message : '网络错误'}`,
    }
  }
}

/**
 * 向服务端要一个**保证没用过**的项目 id。
 *
 * **为什么不能本地生成**：两个标签页在同一毫秒建项目会拿到完全相同的 id
 * （时间戳 + 序号，两边都是从 0 开始），后保存的会把先保存的覆盖掉。
 *
 * 服务端用随机后缀 + 存在性检查分配，撞了就换。
 */
export async function allocateProjectId(): Promise<string | null> {
  try {
    const r = await fetch(`${base}/projects/allocate`, { method: 'POST' })
    if (!r.ok) return null
    const j = await r.json()
    return typeof j.id === 'string' && j.id ? j.id : null
  } catch {
    return null
  }
}

/** 删服务端的项目。`files=true` 连产物目录一起删。 */
export async function deleteServerProject(id: string, files: boolean): Promise<boolean> {
  try {
    const r = await fetch(
      `${base}/projects/${encodeURIComponent(id)}?files=${files ? 1 : 0}`,
      { method: 'DELETE' },
    )
    return r.ok
  } catch {
    return false
  }
}

/** 记「当前打开的是哪个项目」——重开浏览器回到同一个画布。 */
export async function saveWorkspace(current: string): Promise<void> {
  try {
    await fetch(`${base}/workspace`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current }),
    })
  } catch {
    /* 记不住无所谓：下次仍会按项目列表的顺序选 */
  }
}
