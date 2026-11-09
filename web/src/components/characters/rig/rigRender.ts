/**
 * 画布绘制：完全消费 rigby2d 运行时的求值结果（Pose.quads / Pose.world），
 * 这里只负责把世界坐标画到 Canvas2D，不参与任何骨骼或动画计算。
 *
 * 坐标系与运行时一致：Y 向上，角度逆时针为正。屏幕变换用 scale(zoom, -zoom)。
 */
import type { Sel, ViewState } from './types'

export type DrawOpts = {
  width: number
  height: number
  view: ViewState
  showBones: boolean
  showMesh: boolean
  showImages: boolean
  onion: boolean
  sel: Sel
  grid: boolean
}

/**
 * region 的「图片像素 → 世界坐标」仿射，与 rigby2d `editor/viewport.js` 的
 * `toLocal = Affine(att.width/img.width, 0, 0, -att.height/img.height, -att.width/2, att.height/2)`
 * 等价：图片 (0,0) 在左上，对应 Pose.quads() 的 positions[0]；(w,0) → positions[1]，
 * (0,h) → positions[3]。外层画布已有 scale(zoom,-zoom)（Y 向上），这里不能再翻一次，
 * 否则整张图上下颠倒。单独导出给 tools/verify_rig_alignment.mjs 做逐角核对。
 */
export function regionTransform(pos: number[][], w: number, h: number): number[] | null {
  if (!w || !h || pos.length !== 4) return null
  return affine3([[0, 0], [w, 0], [0, h]], [pos[0], pos[1], pos[3]])
}

/** 三个源点 → 三个目标点的仿射变换（canvas transform 的 a,b,c,d,e,f）。 */
function affine3(s: number[][], d: number[][]): number[] | null {
  const [s0, s1, s2] = s, [d0, d1, d2] = d
  const det = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s2[0] - s0[0]) * (s1[1] - s0[1])
  if (Math.abs(det) < 1e-9) return null
  const a = ((d1[0] - d0[0]) * (s2[1] - s0[1]) - (d2[0] - d0[0]) * (s1[1] - s0[1])) / det
  const c = ((d2[0] - d0[0]) * (s1[0] - s0[0]) - (d1[0] - d0[0]) * (s2[0] - s0[0])) / det
  const b = ((d1[1] - d0[1]) * (s2[1] - s0[1]) - (d2[1] - d0[1]) * (s1[1] - s0[1])) / det
  const d2c = ((d2[1] - d0[1]) * (s1[0] - s0[0]) - (d1[1] - d0[1]) * (s2[0] - s0[0])) / det
  return [a, b, c, d2c, d0[0] - a * s0[0] - c * s0[1], d0[1] - b * s0[0] - d2c * s0[1]]
}

function texturedTriangle(ctx: CanvasRenderingContext2D, img: CanvasImageSource,
  src: number[][], dst: number[][]) {
  const m = affine3(src, dst)
  if (!m) return
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(dst[0][0], dst[0][1])
  ctx.lineTo(dst[1][0], dst[1][1])
  ctx.lineTo(dst[2][0], dst[2][1])
  ctx.closePath()
  ctx.clip()
  ctx.transform(m[0], m[1], m[2], m[3], m[4], m[5])
  ctx.drawImage(img, 0, 0)
  ctx.restore()
}

/** 与上游 editor/viewport.js 的 BLEND_OPS 一致。 */
const BLEND_OPS: Record<string, GlobalCompositeOperation> = {
  normal: 'source-over', additive: 'lighter', multiply: 'multiply', screen: 'screen',
}

function drawQuad(ctx: CanvasRenderingContext2D, q: any, getImage: (n: string, c: number[]) => any, o: DrawOpts) {
  const tint = q.color || [1, 1, 1, 1]
  const img = q.image ? getImage(q.image, tint) : null
  const pos: number[][] = q.positions
  if (o.showImages && img) {
    // save/restore 包住 alpha 与混合模式，避免污染后续的骨骼线 / 网格线。
    ctx.save()
    ctx.globalAlpha = Math.min(1, Math.max(0, tint[3]))
    ctx.globalCompositeOperation = BLEND_OPS[q.blend] ?? 'source-over'
    if (q.kind === 'region' && pos.length === 4) {
      // positions 是 Pose.quads() 的 左上/右上/右下/左下（Y 向上）。
      const m = regionTransform(pos, img.width, img.height)
      if (m) {
        ctx.save()
        ctx.transform(m[0], m[1], m[2], m[3], m[4], m[5])
        ctx.drawImage(img, 0, 0)
        ctx.restore()
      }
    } else if (q.kind === 'mesh' && q.triangles && q.uvs) {
      const tri = q.triangles
      for (let i = 0; i + 2 < tri.length; i += 3) {
        const ids = [tri[i], tri[i + 1], tri[i + 2]]
        if (ids.some((k) => !pos[k])) continue
        const src = ids.map((k) => {
          const uv = q.uvs[k] || [0, 0]
          return [uv[0] * img.width, uv[1] * img.height]
        })
        texturedTriangle(ctx, img, src, ids.map((k) => pos[k]))
      }
    }
    ctx.restore()
  }
  if (q.kind === 'mesh' && o.showMesh) {
    ctx.strokeStyle = 'rgba(90, 209, 196, .55)'
    ctx.lineWidth = 0.7 / o.view.zoom
    const tri = q.triangles || []
    for (let i = 0; i + 2 < tri.length; i += 3) {
      const a = pos[tri[i]], b = pos[tri[i + 1]], c = pos[tri[i + 2]]
      if (!a || !b || !c) continue
      ctx.beginPath()
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.closePath()
      ctx.stroke()
    }
  }
}

function drawBones(ctx: CanvasRenderingContext2D, doc: any, pose: any, o: DrawOpts,
  alpha = 1, selectedOnly = false) {
  ctx.globalAlpha = alpha
  const w = pose.world || []
  for (let i = 0; i < doc.bones.length; i++) {
    const b = doc.bones[i]
    const m = w[i]
    if (!m) continue
    const from = m.apply([0, 0])
    const len = Math.max(12, b.length || 40)
    const to = m.apply([0, len])
    const on = o.sel?.kind === 'bone' && o.sel.index === i
    if (selectedOnly && !on) continue
    ctx.strokeStyle = on ? '#ffce5c' : '#5ad1c4'
    ctx.lineWidth = (on ? 3 : 1.8) / o.view.zoom
    ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke()
    ctx.fillStyle = on ? '#ffce5c' : '#8fe3d7'
    const r = (on ? 5 : 3.5) / o.view.zoom
    ctx.beginPath(); ctx.arc(from[0], from[1], r, 0, Math.PI * 2); ctx.fill()
    if (o.showBones && !selectedOnly) {
      // 上层变换是 scale(zoom, -zoom)（Y 向上），文字要单独翻回来否则是镜像。
      ctx.save()
      ctx.translate(to[0] + 4 / o.view.zoom, to[1])
      ctx.scale(1, -1)
      ctx.fillStyle = 'rgba(220,235,240,.75)'
      ctx.font = `${11 / o.view.zoom}px sans-serif`
      ctx.fillText(b.name, 0, 0)
      ctx.restore()
    }
  }
  ctx.globalAlpha = 1
}

export function drawScene(ctx: CanvasRenderingContext2D, doc: any, pose: any,
  images: any, o: DrawOpts, ghosts: { pose: any; alpha: number }[] = []) {
  ctx.clearRect(0, 0, o.width, o.height)
  if (o.grid) {
    ctx.strokeStyle = 'rgba(120,150,170,.09)'
    ctx.lineWidth = 1
    const step = 50 * o.view.zoom
    for (let x = (o.width / 2 + o.view.x) % step; x < o.width; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, o.height); ctx.stroke()
    }
    for (let y = (o.height / 2 + o.view.y) % step; y < o.height; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(o.width, y); ctx.stroke()
    }
  }
  ctx.save()
  // 上游 viewport.js：nearest 时关掉平滑，避免像素风素材被插值糊掉。
  ctx.imageSmoothingEnabled = doc.textureFilter !== 'nearest'
  ctx.translate(o.width / 2 + o.view.x, o.height / 2 + o.view.y)
  ctx.scale(o.view.zoom, -o.view.zoom)

  // 原点
  ctx.strokeStyle = 'rgba(160,190,205,.35)'
  ctx.lineWidth = 1 / o.view.zoom
  ctx.beginPath(); ctx.moveTo(-20, 0); ctx.lineTo(20, 0); ctx.moveTo(0, -20); ctx.lineTo(0, 20); ctx.stroke()

  // Images.tinted 处理 RGB 染色（白色时直接返回原图），与上游 drawQuad 一致。
  const getImage = (n: string, c: number[]) =>
    images?.tinted?.(n, c) || images?.get?.(n) || null
  for (const q of pose.quads(doc)) drawQuad(ctx, q, getImage, o)

  if (o.onion) {
    for (const g of ghosts) {
      const prev = o.showImages
      drawBones(ctx, doc, g.pose, { ...o, showImages: false, sel: null }, g.alpha)
      o.showImages = prev
    }
  }
  if (o.showBones || o.sel?.kind === 'bone') drawBones(ctx, doc, pose, o)
  ctx.restore()
}
