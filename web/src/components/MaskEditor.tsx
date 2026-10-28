/**
 * 蒙版笔刷（局部重绘用）。
 *
 * 思路照搬 pixelforge 的 MaskEditor（那边已经踩过一轮坑）：
 *
 * - **三层画布叠加**：图（源图）/ 蒙版（白=重绘区，导出给后端）/ 覆盖（半透明红，
 *   给眼睛看的）。擦除时覆盖层用 destination-out 只擦自己，下面的图不动。
 * - **撤销用「操作列表 + 全量重放」**，不存 ImageData 快照：1024² 的快照是
 *   4MB/步，撤 20 步就是 80MB；操作列表每步几十字节，重放一张图几毫秒。
 * - **笔刷一律硬边**：柔和边界交给后端的羽化（那边按像素算、可复现）；
 *   画布上做软笔刷只会得到不稳定的半透明，「选区外逐点不变」也没法验证。
 * - **覆盖层用不透明色 + CSS opacity**：如果直接画 rgba(0.4)，同一笔拖动时
 *   相邻线段会互相叠加（0.4→0.64→0.78…），看起来像「有的地方没涂上」。
 * - **H 键临时隐藏蒙版**：用原图核对「到底涂到哪了」。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { saveMask } from '../api'

type Tool = 'brush' | 'eraser' | 'rect'

type Op =
  | { t: 'brush'; pts: number[][]; size: number; erase: boolean }
  | { t: 'rect'; x0: number; y0: number; x1: number; y1: number; erase: boolean }

export type MaskInfo = {
  path: string
  url: string
  width: number
  height: number
  coverage: number
}

type Props = {
  open: boolean
  srcUrl: string
  /** 上次涂的蒙版（继续编辑） */
  initialMaskUrl?: string | null
  title?: string
  /** 蒙版文件名前缀（同一节点同名覆盖） */
  name: string
  project?: string
  onSaved: (m: MaskInfo) => void
  onClose: () => void
}

const OVERLAY_ALPHA_DEFAULT = 0.4
const OVERLAY_ALPHA_MIN = 0.08
const SWATCHES = ['#ff3b6b', '#00e08a', '#2f8bff', '#ffd400', '#ffffff', '#000000']

/** 拿 2d 上下文；某些环境（隐私模式 / 无 canvas 支持）会抛异常，包一层。 */
function ctx2d(c: HTMLCanvasElement | null | undefined): CanvasRenderingContext2D | null {
  if (!c) return null
  try {
    return ctx2d(c)
  } catch {
    return null
  }
}

function tint(hex: string, alpha = 1): string {
  const h = (hex || '').replace('#', '')
  const v = h.length === 3 ? h.split('').map((c) => c + c).join('') : h
  const n = Number.parseInt(v || 'ff3b6b', 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`
}

export default function MaskEditor({
  open, srcUrl, initialMaskUrl, title, name, project, onSaved, onClose,
}: Props) {
  const [tool, setTool] = useState<Tool>('brush')
  const [size, setSize] = useState(24)
  const [overlayHex, setOverlayHex] = useState(() => {
    try { return localStorage.getItem('meow.maskColor') || '#ff3b6b' } catch { return '#ff3b6b' }
  })
  const [overlayAlpha, setOverlayAlpha] = useState(() => {
    try {
      const v = Number(localStorage.getItem('meow.maskAlpha'))
      return Number.isFinite(v) && v >= OVERLAY_ALPHA_MIN && v <= 1 ? v : OVERLAY_ALPHA_DEFAULT
    } catch { return OVERLAY_ALPHA_DEFAULT }
  })
  const [peek, setPeek] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [tick, setTick] = useState(0)
  const [srcSize, setSrcSize] = useState({ w: 0, h: 0 })

  const imgRef = useRef<HTMLCanvasElement | null>(null)
  const maskRef = useRef<HTMLCanvasElement | null>(null)
  const overRef = useRef<HTMLCanvasElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const baseRef = useRef<HTMLCanvasElement | null>(null)
  const opsRef = useRef<Op[]>([])
  const undosRef = useRef<Op[]>([])
  const drawing = useRef<Op | null>(null)
  const panning = useRef<{ x: number; y: number; px: number; py: number } | null>(null)
  const spaceRef = useRef(false)

  const overlaySolid = useMemo(() => tint(overlayHex, 1), [overlayHex])

  useEffect(() => {
    try { localStorage.setItem('meow.maskColor', overlayHex) } catch { /* 隐私模式忽略 */ }
  }, [overlayHex])
  useEffect(() => {
    try { localStorage.setItem('meow.maskAlpha', String(overlayAlpha)) } catch { /* 同上 */ }
  }, [overlayAlpha])

  const ensureBase = useCallback((w: number, h: number) => {
    let b = baseRef.current
    if (!b) { b = document.createElement('canvas'); baseRef.current = b }
    if (b.width !== w || b.height !== h) { b.width = w; b.height = h }
    return b
  }, [])

  const paint = useCallback((ctx: CanvasRenderingContext2D, op: Op, color: string | null) => {
    ctx.save()
    ctx.globalCompositeOperation = color ? 'source-over' : 'destination-out'
    ctx.fillStyle = color || '#000'
    ctx.strokeStyle = color || '#000'
    if (op.t === 'brush') {
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.lineWidth = op.size
      const pts = op.pts
      if (pts.length === 1) {
        ctx.beginPath()
        ctx.arc(pts[0][0], pts[0][1], op.size / 2, 0, Math.PI * 2)
        ctx.fill()
      } else {
        ctx.beginPath()
        ctx.moveTo(pts[0][0], pts[0][1])
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1])
        ctx.stroke()
      }
    } else {
      const x = Math.min(op.x0, op.x1)
      const y = Math.min(op.y0, op.y1)
      ctx.fillRect(x, y, Math.abs(op.x1 - op.x0), Math.abs(op.y1 - op.y0))
    }
    ctx.restore()
  }, [])

  const redraw = useCallback((ops: Op[]) => {
    const mask = maskRef.current
    const over = overRef.current
    if (!mask || !over) return
    const mc = ctx2d(mask)
    const oc = ctx2d(over)
    if (!mc || !oc) return
    mc.clearRect(0, 0, mask.width, mask.height)
    oc.clearRect(0, 0, over.width, over.height)
    const base = baseRef.current
    if (base && base.width) {
      mc.drawImage(base, 0, 0)
      oc.save()
      oc.drawImage(base, 0, 0)
      oc.globalCompositeOperation = 'source-in'
      oc.fillStyle = overlaySolid
      oc.fillRect(0, 0, over.width, over.height)
      oc.restore()
    }
    for (const op of ops) {
      paint(mc, op, op.t === 'rect' ? '#fff' : op.erase ? null : '#fff')
      paint(oc, op, op.erase ? null : overlaySolid)
    }
  }, [paint, overlaySolid])

  useEffect(() => {
    if (!open) return
    redraw(opsRef.current)
  }, [overlaySolid, open, redraw, tick])

  // 打开：初始化画布尺寸 + 载入源图与已有蒙版
  useEffect(() => {
    if (!open) return
    opsRef.current = []
    undosRef.current = []
    drawing.current = null
    setDirty(false)
    setPan({ x: 0, y: 0 })
    setTool('brush')

    const img = new Image()
    img.onload = () => {
      const w = img.naturalWidth || 512
      const h = img.naturalHeight || 512
      setSrcSize({ w, h })
      setSize(Math.max(8, Math.round(Math.min(w, h) / 12)))
      const canvas = imgRef.current
      if (canvas) {
        canvas.width = w
        canvas.height = h
        const ctx = ctx2d(canvas)
        if (ctx) {
          ctx.imageSmoothingEnabled = false
          ctx.clearRect(0, 0, w, h)
          ctx.drawImage(img, 0, 0, w, h)
        }
      }
      for (const c of [maskRef.current, overRef.current]) {
        if (c) { c.width = w; c.height = h }
      }
      const stage = stageRef.current
      if (stage) {
        const fit = Math.min(1, (stage.clientWidth - 120) / w, (stage.clientHeight - 120) / h)
        setZoom(Math.max(0.05, fit))
      }
      if (initialMaskUrl) {
        const m = new Image()
        m.onload = () => {
          const mask = maskRef.current
          const over = overRef.current
          const mc = ctx2d(mask)
          if (!mask || !over || !mc) return
          // 历史蒙版可能是「灰阶不透明 PNG」（整张不透明），直接 source-in 会把
          // 整幅图上成红色。这里把亮度折进 alpha，两种格式就统一成一个意思。
          const flat = document.createElement('canvas')
          flat.width = w
          flat.height = h
          const fc = ctx2d(flat)
          if (!fc) return
          fc.drawImage(m, 0, 0, w, h)
          const id = fc.getImageData(0, 0, w, h)
          const d = id.data
          for (let i = 0; i < d.length; i += 4) {
            const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
            d[i + 3] = Math.round((d[i + 3] * lum) / 255)
            d[i] = 255; d[i + 1] = 255; d[i + 2] = 255
          }
          fc.putImageData(id, 0, 0)
          const base = ensureBase(w, h)
          const bc = ctx2d(base)
          if (!bc) return
          bc.clearRect(0, 0, w, h)
          bc.drawImage(flat, 0, 0)
          opsRef.current = []
          undosRef.current = []
          setTick((n) => n + 1)
        }
        m.src = initialMaskUrl
      } else {
        setTick((n) => n + 1)
      }
    }
    img.src = srcUrl
  }, [open, srcUrl, initialMaskUrl, ensureBase])

  const toImage = useCallback((clientX: number, clientY: number) => {
    const el = wrapRef.current
    if (!el || !srcSize.w) return null
    const r = el.getBoundingClientRect()
    // getBoundingClientRect 已含 transform，除以 zoom 回到图像素坐标
    const x = (clientX - r.left) / zoom
    const y = (clientY - r.top) / zoom
    return {
      x: Math.max(0, Math.min(srcSize.w, x)),
      y: Math.max(0, Math.min(srcSize.h, y)),
    }
  }, [zoom, srcSize])

  const commit = useCallback((op: Op) => {
    opsRef.current = [...opsRef.current, op]
    undosRef.current = []
    setDirty(true)
    setTick((n) => n + 1)
  }, [])

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button === 1 || spaceRef.current) {
      panning.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y }
      return
    }
    if (e.button !== 0) return
    const p = toImage(e.clientX, e.clientY)
    if (!p) return
    ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
    if (tool === 'rect') {
      drawing.current = { t: 'rect', x0: p.x, y0: p.y, x1: p.x, y1: p.y, erase: false }
    } else {
      drawing.current = { t: 'brush', pts: [[p.x, p.y]], size, erase: tool === 'eraser' }
      const mask = maskRef.current
      const over = overRef.current
      if (mask && over) {
        const op = drawing.current
        paint(ctx2d(mask)!, op, op.erase ? null : '#fff')
        paint(ctx2d(over)!, op, op.erase ? null : overlaySolid)
      }
    }
    setDirty(true)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const p = toImage(e.clientX, e.clientY)
    if (panning.current) {
      setPan({
        x: panning.current.px + (e.clientX - panning.current.x),
        y: panning.current.py + (e.clientY - panning.current.y),
      })
      return
    }
    const op = drawing.current
    if (!op || !p) return
    const mask = maskRef.current
    const over = overRef.current
    if (!mask || !over) return
    if (op.t === 'brush') {
      const last = op.pts[op.pts.length - 1]
      if (Math.hypot(p.x - last[0], p.y - last[1]) < Math.max(0.6, 1 / zoom)) return
      op.pts.push([p.x, p.y])
      // 增量画最后一段（全量重放只在撤销时做）
      const seg: Op = { ...op, pts: [last, [p.x, p.y]] }
      paint(ctx2d(mask)!, seg, op.erase ? null : '#fff')
      paint(ctx2d(over)!, seg, op.erase ? null : overlaySolid)
    } else {
      op.x1 = p.x
      op.y1 = p.y
      const mc = ctx2d(mask)!
      const oc = ctx2d(over)!
      mc.clearRect(0, 0, mask.width, mask.height)
      oc.clearRect(0, 0, over.width, over.height)
      const base = baseRef.current
      if (base && base.width) {
        mc.drawImage(base, 0, 0)
        oc.save()
        oc.drawImage(base, 0, 0)
        oc.globalCompositeOperation = 'source-in'
        oc.fillStyle = overlaySolid
        oc.fillRect(0, 0, over.width, over.height)
        oc.restore()
      }
      for (const o of opsRef.current) {
        paint(mc, o, o.t === 'rect' ? '#fff' : o.erase ? null : '#fff')
        paint(oc, o, o.erase ? null : overlaySolid)
      }
      paint(mc, op, '#fff')
      paint(oc, op, overlaySolid)
    }
  }

  const onPointerUp = () => {
    const op = drawing.current
    drawing.current = null
    if (op) commit(op)
  }

  const undo = useCallback(() => {
    const ops = opsRef.current
    if (ops.length === 0) return
    const last = ops[ops.length - 1]
    undosRef.current = [...undosRef.current, last]
    opsRef.current = ops.slice(0, -1)
    redraw(opsRef.current)
    setDirty(true)
    setTick((n) => n + 1)
  }, [redraw])

  const redo = useCallback(() => {
    const undos = undosRef.current
    if (undos.length === 0) return
    const last = undos[undos.length - 1]
    opsRef.current = [...opsRef.current, last]
    undosRef.current = undos.slice(0, -1)
    redraw(opsRef.current)
    setTick((n) => n + 1)
  }, [redraw])

  const clearMask = useCallback(() => {
    opsRef.current = []
    undosRef.current = []
    const base = baseRef.current
    if (base) {
      const bc = ctx2d(base)
      if (bc) bc.clearRect(0, 0, base.width, base.height)
    }
    redraw([])
    setDirty(true)
    setTick((n) => n + 1)
  }, [redraw])

  // 快捷键：撤销/重做/清空/H 预览/Esc 关闭
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      const typing =
        el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT'
      if (e.code === 'Space') spaceRef.current = true
      if (typing) return
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) redo()
        else undo()
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault()
        redo()
      } else if (e.key.toLowerCase() === 'h' && !e.ctrlKey && !e.metaKey) {
        setPeek((v) => !v)
      } else if (e.key === 'Escape') {
        onClose()
      }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') spaceRef.current = false
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [open, undo, redo, onClose])

  // 滚轮缩放（以光标为锚）
  useEffect(() => {
    if (!open) return
    const el = wrapRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const r = el.getBoundingClientRect()
      const px = e.clientX - r.left
      const py = e.clientY - r.top
      setZoom((z) => {
        const next = Math.max(0.1, Math.min(8, z * Math.exp(-e.deltaY * 0.0015)))
        const k = next / z
        setPan((p) => ({ x: px - (px - p.x) * k, y: py - (py - p.y) * k }))
        return next
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [open])

  const save = useCallback(async () => {
    const mask = maskRef.current
    if (!mask) return
    setSaving(true)
    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        mask.toBlob((b) => resolve(b), 'image/png'))
      if (!blob) return
      const info = await saveMask(blob, name, project)
      onSaved(info)
      onClose()
    } catch (e) {
      alert(`蒙版保存失败：${e instanceof Error ? e.message : '未知错误'}`)
    } finally {
      setSaving(false)
    }
  }, [name, project, onSaved, onClose])

  if (!open) return null

  return (
    <div className="mask-editor">
      <div className="mask-bar">
        <span className="mask-title">{title || '绘制蒙版'}</span>
        <span className="spacer" />
        <button className="mini" onClick={undo} title="撤销 (Ctrl+Z)">↶</button>
        <button className="mini" onClick={redo} title="重做 (Ctrl+Shift+Z)">↷</button>
        <button className="mini" onClick={clearMask} title="清空蒙版">✕ 清空</button>
        <button className="mini" onClick={onClose}>取消</button>
        <button className="mini primary" onClick={() => void save()} disabled={saving}>
          {saving ? '保存中…' : '保存蒙版'}
        </button>
      </div>

      <div className="mask-tools">
        <button
          className={`mini${tool === 'brush' ? ' on' : ''}`}
          onClick={() => setTool('brush')}
          title="画笔"
        >
          ✎ 画笔
        </button>
        <button
          className={`mini${tool === 'eraser' ? ' on' : ''}`}
          onClick={() => setTool('eraser')}
          title="橡皮"
        >
          ⌫ 橡皮
        </button>
        <button
          className={`mini${tool === 'rect' ? ' on' : ''}`}
          onClick={() => setTool('rect')}
          title="矩形框选"
        >
          ▭ 矩形
        </button>
        <label className="mask-size">
          笔刷
          <input
            type="range"
            min={4}
            max={120}
            step={1}
            value={size}
            onChange={(e) => setSize(Number(e.target.value))}
          />
          <b>{size}</b>
        </label>
        <label className="mask-size">
          覆盖
          <input
            type="range"
            min={OVERLAY_ALPHA_MIN}
            max={1}
            step={0.02}
            value={overlayAlpha}
            onChange={(e) => setOverlayAlpha(Number(e.target.value))}
          />
        </label>
        <span className="mask-swatches">
          {SWATCHES.map((c) => (
            <button
              key={c}
              className={`swatch${overlayHex === c ? ' on' : ''}`}
              style={{ background: c }}
              onClick={() => setOverlayHex(c)}
              title={`蒙版颜色 ${c}`}
            />
          ))}
        </span>
        <span className="mask-hint">
          滚轮缩放 · 空格拖动平移 · H 临时隐藏蒙版{dirty ? ' · 有未保存的修改' : ''}
        </span>
      </div>

      <div className="mask-stage" ref={stageRef}>
        <div
          className="mask-wrap"
          ref={wrapRef}
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            width: srcSize.w || 512,
            height: srcSize.h || 512,
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <canvas ref={imgRef} className="layer" />
          <canvas ref={maskRef} className="layer" />
          <canvas
            ref={overRef}
            className="layer overlay"
            style={{ opacity: peek ? 0 : overlayAlpha }}
          />
        </div>
      </div>
    </div>
  )
}
