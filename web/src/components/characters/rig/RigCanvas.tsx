/** 视口：Canvas2D 渲染 + 选择 / 平移 / 缩放 / 拖拽变换骨骼。所有求值来自 useRig 的 poseRef。 */
import { useCallback, useEffect, useRef } from 'react'
import type { RigApi } from './useRig'
import { drawScene } from './rigRender'
import { toWorld, type RigTool } from './types'

function pointSegment(p: number[], a: number[], b: number[]): number {
  const vx = b[0] - a[0], vy = b[1] - a[1]
  const wx = p[0] - a[0], wy = p[1] - a[1]
  const len = vx * vx + vy * vy
  const t = len ? Math.max(0, Math.min(1, (wx * vx + wy * vy) / len)) : 0
  return Math.hypot(wx - t * vx, wy - t * vy)
}

type Drag =
  | { kind: 'pan'; sx: number; sy: number; vx: number; vy: number }
  | { kind: 'bone'; index: number; tool: RigTool; sx: number; sy: number; lastAngle: number; lastDist: number }
  | null

export function RigCanvas({ rig }: { rig: RigApi }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const cvRef = useRef<HTMLCanvasElement>(null)
  const dragRef = useRef<Drag>(null)
  const sizeRef = useRef({ w: 800, h: 560 })
  const shapeRef = useRef('')

  /** 按当前姿势的可见内容自动取景（导入 PSD / AutoRig 后调用）。 */
  const fit = useCallback(() => {
    const d = rig.docRef.current, p = rig.poseRef.current
    const pts: number[][] = []
    for (const q of p.quads(d)) for (const pt of q.positions) pts.push(pt)
    // 场景演员也纳入取景，否则加了演员后默认视图里看不到他们。
    for (const rt of rig.actorRef.current.values()) {
      for (const q of rt.pose.quads(rt.doc)) for (const pt of q.positions) pts.push(pt)
    }
    if (!pts.length) return
    const xs = pts.map((v) => v[0]), ys = pts.map((v) => v[1])
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    const { w, h } = sizeRef.current
    const zoom = Math.max(0.05, Math.min(4,
      Math.min(w / Math.max(1, maxX - minX), h / Math.max(1, maxY - minY)) * 0.82))
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
    rig.setView({ zoom, x: -cx * zoom, y: cy * zoom })
  }, [rig])

  // 结构变化（新增层 / AutoRig 加骨）才自动取景；普通编辑不打断用户的视图。
  useEffect(() => {
    const d = rig.docRef.current
    const key = `${d.slots.length}:${d.bones.length}:${d.name}`
    if (shapeRef.current !== key) {
      shapeRef.current = key
      if (d.slots.length) setTimeout(fit, 60)
    }
  }, [rig.rev, fit])

  useEffect(() => {
    const wrap = wrapRef.current, cv = cvRef.current
    if (!wrap || !cv) return
    if (typeof ResizeObserver === 'undefined') {
      // 老环境 / 测试 DOM 没有 ResizeObserver：按当前尺寸初始化一次即可。
      const r = wrap.getBoundingClientRect()
      cv.width = Math.max(1, Math.round(r.width || 800))
      cv.height = Math.max(1, Math.round(r.height || 560))
      sizeRef.current = { w: r.width || 800, h: r.height || 560 }
      return
    }
    const ro = new ResizeObserver(() => {
      const r = wrap.getBoundingClientRect()
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      cv.width = Math.max(1, Math.round(r.width * dpr))
      cv.height = Math.max(1, Math.round(r.height * dpr))
      cv.style.width = `${r.width}px`
      cv.style.height = `${r.height}px`
      sizeRef.current = { w: r.width, h: r.height }
    })
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [])

  // 独立 rAF：播放时 poseRef 被 useRig 更新，这里只负责画，不跟着 React 重渲染。
  // rigRef 保持最新状态但 effect 只挂一次，避免播放时每帧重建 rAF。
  const rigRef = useRef(rig)
  rigRef.current = rig
  useEffect(() => {
    const cv = cvRef.current
    if (!cv) return
    let raf = 0
    const loop = () => {
      const ctx = cv.getContext('2d')
      const r = rigRef.current
      const { w, h } = sizeRef.current
      const d = r.docRef.current, p = r.poseRef.current
      if (ctx && d && p) {
        const dpr = Math.min(2, window.devicePixelRatio || 1)
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        drawScene(ctx, d, p, r.imagesRef.current, {
          width: w, height: h, view: r.view,
          showBones: r.showBones, showMesh: r.showMesh, showImages: r.showImages,
          onion: r.onion, sel: r.sel, grid: r.grid,
          actors: r.actorRef.current,
        }, r.ghosts())
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  const worldAt = useCallback((e: React.PointerEvent) => {
    const r = cvRef.current!.getBoundingClientRect()
    const sx = e.clientX - r.left, sy = e.clientY - r.top
    const { w, h } = sizeRef.current
    return { p: toWorld(rig.view, w, h, sx, sy), sx, sy }
  }, [rig.view])

  const pickBone = useCallback((p: number[]): number => {
    const d = rig.docRef.current, pose = rig.poseRef.current
    let best = -1, dist = 12 / rig.view.zoom
    for (let i = 0; i < d.bones.length; i++) {
      const m = pose.world[i]
      if (!m) continue
      const a = m.apply([0, 0])
      const b = m.apply([0, Math.max(12, d.bones[i].length || 40)])
      const dd = pointSegment(p, a, b)
      if (dd < dist) { dist = dd; best = i }
    }
    return best
  }, [rig.docRef, rig.poseRef, rig.view.zoom])

  const onDown = (e: React.PointerEvent) => {
    const { p, sx, sy } = worldAt(e)
    const panning = e.button === 1 || e.shiftKey || e.altKey
    if (panning) {
      dragRef.current = { kind: 'pan', sx, sy, vx: rig.view.x, vy: rig.view.y }
    } else {
      const i = rig.sel?.kind === 'bone' ? rig.sel.index : pickBone(p)
      if (i >= 0) {
        rig.setSel({ kind: 'bone', index: i })
        const m = rig.poseRef.current.world[i]
        const o = m ? m.apply([0, 0]) : [0, 0]
        dragRef.current = {
          kind: 'bone', index: i, tool: rig.tool === 'select' ? 'rotate' : rig.tool,
          sx, sy,
          lastAngle: Math.atan2(p[1] - o[1], p[0] - o[0]),
          lastDist: Math.max(1, Math.hypot(p[0] - o[0], p[1] - o[1])),
        }
      } else if (rig.tool === 'create') {
        const parent = rig.sel?.kind === 'bone' ? rig.sel.index : 0
        rig.addBone(parent)
      } else {
        rig.setSel(null)
      }
    }
    ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
  }

  const onMove = (e: React.PointerEvent) => {
    const drag = dragRef.current
    if (!drag) return
    const { p, sx, sy } = worldAt(e)
    if (drag.kind === 'pan') {
      rig.setView({ ...rig.view, x: drag.vx + (sx - drag.sx), y: drag.vy + (sy - drag.sy) })
      return
    }
    const m = rig.poseRef.current.world[drag.index]
    const o = m ? m.apply([0, 0]) : [0, 0]
    const angle = Math.atan2(p[1] - o[1], p[0] - o[0])
    const distNow = Math.max(1, Math.hypot(p[0] - o[0], p[1] - o[1]))
    let da = angle - drag.lastAngle
    if (da > Math.PI) da -= 2 * Math.PI
    if (da < -Math.PI) da += 2 * Math.PI
    const scale = distNow / drag.lastDist
    // 位移用屏幕增量换算成世界增量（Y 向上）
    const dx = (e.movementX || 0) / rig.view.zoom
    const dy = -(e.movementY || 0) / rig.view.zoom
    rig.transformBone(drag.index, drag.tool, {
      dx, dy, rotation: da * 180 / Math.PI, scale,
    })
    dragRef.current = { ...drag, lastAngle: angle, lastDist: distNow }
  }

  const onUp = () => { dragRef.current = null }

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const k = e.deltaY < 0 ? 1.12 : 1 / 1.12
    rig.setView({ ...rig.view, zoom: Math.max(0.12, Math.min(8, rig.view.zoom * k)) })
  }

  return (
    <div className="rig-canvas-wrap" ref={wrapRef}>
      <canvas ref={cvRef} className="rig-canvas"
        onPointerDown={onDown} onPointerMove={onMove}
        onPointerUp={onUp} onPointerCancel={onUp} onWheel={onWheel} />
      <div className="rig-canvas-hud">
        <span>缩放 {Math.round(rig.view.zoom * 100)}%</span>
        <span>{rig.docRef.current.bones.length} 骨 · {rig.docRef.current.slots.length} 槽</span>
        <span>{rig.mode === 'animate' ? `动画 ${rig.time.toFixed(2)}s` : '设定模式'}</span>
        <button type="button" className="rig-mini" onClick={fit}>适配视图</button>
        <button type="button" className="rig-mini"
          onClick={() => rig.setView({ zoom: 1, x: 0, y: 0 })}>复位</button>
      </div>
    </div>
  )
}
