/**
 * 影片工作区的离屏编码：逐帧用 Pose 采样 + drawScene 画到离屏 canvas，
 * 交给浏览器 WebCodecs 编码，最后用一比一搬入的 rigby2d MP4 / WebM(含 alpha)
 * 封装器写成文件。这里不重写任何骨骼/动画算法或容器格式。
 */
import { pose as P, mp4, webm, output } from '../../../rig/facade.js'
import { drawScene } from './rigRender'
import { poseSceneActors, sceneBounds } from './scenePose'

export type FilmFormat = 'mp4' | 'webm' | 'png'
export type FilmOptions = {
  width: number
  height: number
  fps: number
  format: FilmFormat
  transparent: boolean
  background: string | null
  start: number
  end: number
  quality: number
  animation: number
}

export const QUALITY = { high: 0.25, medium: 0.12, low: 0.06 }
/** UI 的画质档位是 0/1/2（低/中/高），一一对应上游 QUALITY 的三个 key。 */
const QUALITY_KEYS = ['low', 'medium', 'high'] as const

function bitrateFactor(quality: number): number {
  const i = Math.max(0, Math.min(QUALITY_KEYS.length - 1, Math.round(quality)))
  return QUALITY[QUALITY_KEYS[i]]
}

export function canEncodeVideo(): boolean {
  return typeof (window as any).VideoEncoder === 'function'
    && typeof (window as any).VideoFrame === 'function'
}

export function timecode(t: number, fps: number): string {
  const total = Math.max(0, Math.round(t * fps))
  const f = total % Math.round(fps)
  const s = Math.floor(total / fps)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(Math.floor(s / 3600))}:${p(Math.floor(s / 60) % 60)}:${p(s % 60)}:${p(f)}`
}

type Chunk = { data: Uint8Array; key: boolean; time: number; alpha?: Uint8Array | null }

async function pickConfig(candidates: any[]): Promise<any | null> {
  const VE = (window as any).VideoEncoder
  if (!VE) return null
  for (const c of candidates) {
    try {
      const r = await VE.isConfigSupported(c)
      if (r?.supported) return r.config || c
    } catch { /* next */ }
  }
  return null
}

function drained(e: any): Promise<void> {
  return new Promise((res) => {
    if (!e || e.encodeQueueSize <= 8 || e.state !== 'configured') return res()
    const h = () => {
      if (e.encodeQueueSize <= 8 || e.state !== 'configured') {
        e.removeEventListener?.('dequeue', h); res()
      }
    }
    e.addEventListener?.('dequeue', h)
    setTimeout(() => { e.removeEventListener?.('dequeue', h); res() }, 120)
  })
}

/**
 * 逐帧推进姿势，语义与上游 `render/offline.js` 的 `FrameRenderer.render` 一致：
 * 物理约束（摆动/惯性）靠跨帧累积的 dt 仿真，`Pose` 与它的 `physics` 状态必须跨帧复用；
 * 起始帧之前的预热帧也要走一遍，否则从静止状态起算会让摆动轨迹整体偏掉。
 */
export function makeFramePose(doc: any, anim: any, fps: number,
  actors?: Map<string, { doc: any; pose: any }>) {
  const pose = new P.Pose(doc)
  const dt = 1 / Math.max(1e-6, fps)
  let frame = -1
  let time = 0
  const shot = anim?.name ?? ''
  return (n: number) => {
    while (frame < n) {
      // 第一帧不推进：上游 `render()` 里 `if (this.frame >= 0) state.update(...)`。
      if (frame >= 0) { pose.physicsDelta = dt; time += dt }
      frame++
      pose.reset(doc)
      // 时间累加而不是 frame*dt：上游 `TrackEntry.time += step`，逐帧累加才与它逐位一致。
      if (anim) pose.apply(doc, anim, time, 1)
      pose.updateWorld(doc)
      // 场景演员：视差 + 各自片段，物理同样按帧推进（frame 0 不推进）。
      if (actors?.size) {
        poseSceneActors({
          doc, stagePose: pose, shot, time, dt: frame > 0 ? dt : 0,
          playing: true, actors,
        })
      }
    }
    return pose
  }
}

/** 渲染 PNG 序列（透明），返回 [{path, bytes}]，由调用方打包 ZIP。 */
export async function renderPngSequence(doc: any, images: any, o: FilmOptions,
  onProgress: (done: number, total: number) => void,
  actors?: Map<string, { doc: any; pose: any }>): Promise<{ path: string; bytes: Uint8Array }[]> {
  const cv = document.createElement('canvas')
  cv.width = o.width; cv.height = o.height
  const ctx = cv.getContext('2d')!
  const anim = doc.animations[o.animation]
  // 跟上游 frameRange 一致：帧号含首尾两端（0..round(duration*fps)），比时长×fps 多一帧。
  const frames = Math.max(1, Math.round((o.end - o.start) * o.fps) + 1)
  const startFrame = Math.max(0, Math.round(o.start * o.fps))
  const bounds = sceneBounds(doc, o.animation, o.fps, actors ?? new Map())
  const frame = output.outputFrame({ width: o.width, height: o.height, fps: o.fps,
    frame: null }, bounds)
  const zoom = 1 / Math.max(1e-6, frame.scale)
  const view = { zoom, x: -frame.cx * zoom, y: frame.cy * zoom }
  const sample = makeFramePose(doc, anim, o.fps, actors)
  const out: { path: string; bytes: Uint8Array }[] = []
  for (let i = 0; i < frames; i++) {
    const pose = sample(startFrame + i)
    drawScene(ctx, doc, pose, images, {
      width: o.width, height: o.height, view, showBones: false,
      showMesh: false, showImages: true, onion: false, sel: null, grid: false, actors,
    })
    const blob: Blob = await new Promise((r) => cv.toBlob((b) => r(b || new Blob()), 'image/png'))
    out.push({ path: `frames/frame_${String(i).padStart(4, '0')}.png`,
      bytes: new Uint8Array(await blob.arrayBuffer()) })
    onProgress(i + 1, frames)
  }
  return out
}

/** 编码 MP4(H.264) 或 WebM(VP9/VP8，可选透明)。返回文件字节。 */
export async function renderVideo(doc: any, images: any, o: FilmOptions,
  onProgress: (done: number, total: number) => void,
  actors?: Map<string, { doc: any; pose: any }>): Promise<Uint8Array> {
  if (!canEncodeVideo()) throw new Error('当前浏览器没有 WebCodecs，无法编码 MP4/WebM；请改用 PNG 序列')
  const VE = (window as any).VideoEncoder
  const VF = (window as any).VideoFrame
  const isMp4 = o.format === 'mp4'
  // H.264 要求偶数边长
  const width = isMp4 ? o.width + (o.width % 2) : o.width
  const height = isMp4 ? o.height + (o.height % 2) : o.height
  const transparent = !isMp4 && o.transparent
  const bitrate = Math.round(width * height * o.fps * bitrateFactor(o.quality))
  const base = { width, height, bitrate, framerate: o.fps }
  const config = isMp4
    ? await pickConfig(['avc1.640033', 'avc1.640028', 'avc1.4d0033', 'avc1.42e033', 'avc1.42e01f']
      .map((codec) => ({ ...base, codec, avc: { format: 'avc' } })))
    : await pickConfig(['vp09.00.51.08', 'vp09.00.41.08', 'vp8']
      .map((codec) => ({ ...base, codec })))
  if (!config) throw new Error(`${isMp4 ? 'H.264' : 'VP9/VP8'} 在 ${width}×${height} 不被当前浏览器支持`)

  const cv = document.createElement('canvas')
  cv.width = width; cv.height = height
  const ctx = cv.getContext('2d', { willReadFrequently: true })!
  const color: Chunk[] = []
  const alphaByTime = new Map<number, Uint8Array>()
  let description: Uint8Array | null = null
  let failure: any = null

  const encoder = new VE({
    output(chunk: any, meta: any) {
      const data = new Uint8Array(chunk.byteLength)
      chunk.copyTo(data)
      if (meta?.decoderConfig?.description && !description) {
        const d = meta.decoderConfig.description
        description = new Uint8Array(d instanceof ArrayBuffer ? d : d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength))
      }
      color.push({ data, key: chunk.type === 'key', time: chunk.timestamp / 1000 })
    },
    error(e: any) { failure = e },
  })
  encoder.configure(config)
  const alphaEncoder = transparent ? new VE({
    output(chunk: any) {
      const data = new Uint8Array(chunk.byteLength)
      chunk.copyTo(data)
      alphaByTime.set(chunk.timestamp, data)
    },
    error(e: any) { failure = e },
  }) : null
  alphaEncoder?.configure(config)

  const anim = doc.animations[o.animation]
  const total = Math.max(1, Math.round((o.end - o.start) * o.fps) + 1)
  const startFrame = Math.max(0, Math.round(o.start * o.fps))
  const sample = makeFramePose(doc, anim, o.fps, actors)
  const bounds = sceneBounds(doc, o.animation, o.fps, actors ?? new Map())
  const outFrame = output.outputFrame({ width, height, fps: o.fps, frame: null }, bounds)
  const zoom = 1 / Math.max(1e-6, outFrame.scale)
  const view = { zoom, x: -outFrame.cx * zoom, y: outFrame.cy * zoom }
  const keyEvery = Math.max(1, Math.round(o.fps * 2))
  const cw = Math.ceil(width / 2), ch = Math.ceil(height / 2)
  const planes = transparent ? new Uint8Array(width * height + 2 * cw * ch).fill(128) : null

  try {
    for (let i = 0; i < total; i++) {
      if (failure) throw failure
      const pose = sample(startFrame + i)
      drawScene(ctx, doc, pose, images, {
        width, height, view, showBones: false, showMesh: false,
        showImages: true, onion: false, sel: null, grid: false, actors,
      })
      if (!transparent && o.background) {
        ctx.save()
        ctx.globalCompositeOperation = 'destination-over'
        ctx.fillStyle = o.background
        ctx.fillRect(0, 0, width, height)
        ctx.restore()
      }
      const timestamp = Math.round(i * 1e6 / o.fps)
      const timing = { timestamp, duration: Math.round(1e6 / o.fps) }
      const key = { keyFrame: i === 0 || i % keyEvery === 0 }
      const rgba = transparent ? ctx.getImageData(0, 0, width, height).data : null
      const frame = rgba
        ? new VF(rgba, { format: 'RGBA', codedWidth: width, codedHeight: height, ...timing })
        : new VF(cv, { ...timing, alpha: 'discard' })
      encoder.encode(frame, key)
      frame.close()
      if (alphaEncoder && rgba && planes) {
        for (let p = 0, n = width * height; p < n; p++) planes[p] = rgba[4 * p + 3]
        const a = new VF(planes, { format: 'I420', codedWidth: width, codedHeight: height, ...timing })
        alphaEncoder.encode(a, key)
        a.close()
      }
      // 跟上游 encodeVideo 一样循环排空两个编码器的队列：只等一次在 4K 会让队列无限堆积。
      for (const e of [encoder, alphaEncoder].filter(Boolean) as any[]) {
        while (e.encodeQueueSize > 8 && !failure) await drained(e)
      }
      onProgress(i + 1, total)
    }
    await encoder.flush()
    if (alphaEncoder) await alphaEncoder.flush()
  } finally {
    try { encoder.close() } catch { /* noop */ }
    try { alphaEncoder?.close() } catch { /* noop */ }
  }
  if (failure) throw failure
  if (!color.length) throw new Error('编码器没有产出任何帧')
  // 编码器的 output 回调不保证按时间顺序到达（尤其加了 alpha 副流），上游也会排一次。
  color.sort((a, b) => a.time - b.time)

  const frames = color.map((c) => ({
    data: c.data, key: c.key, time: c.time,
    alpha: c.alpha ?? (alphaByTime.size ? alphaByTime.get(Math.round(c.time * 1000)) ?? null : null),
  }))
  if (isMp4) {
    if (!description) throw new Error('没有拿到 H.264 的 avcC 配置，无法封装 MP4')
    return mp4.writeMp4({ width, height, fps: o.fps, avcC: description, frames }) as Uint8Array
  }
  return webm.writeWebm({
    width, height, fps: o.fps,
    codec: String(config.codec).startsWith('vp8') ? 'V_VP8' : 'V_VP9',
    alpha: transparent, frames, app: 'Meow Rig Studio',
  }) as Uint8Array
}
