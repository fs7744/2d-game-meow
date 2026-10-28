import { useCallback, useEffect, useRef, useState } from 'react'
import type { GenNode, LoraInfo, Viewport } from '../types'
import GenNodeCard, { NODE_W, type NodePatch } from '../nodes/GenNodeCard'

type Props = {
  nodes: GenNode[]
  model: string
  selectedId: string | null
  viewport: Viewport
  setViewport: (v: Viewport | ((prev: Viewport) => Viewport)) => void
  onSelect: (id: string | null) => void
  onPatch: (id: string, patch: NodePatch) => void
  onGenerate: (id: string) => void
  onDuplicate: (id: string) => void
  onDelete: (id: string) => void
  isFavorite: (text: string) => boolean
  onToggleFavorite: (id: string) => void
  loras: LoraInfo[]
  onDropType: (type: string, world: { x: number; y: number }) => void
}

const MIN_Z = 0.2
const MAX_Z = 2.4

export default function Canvas({
  nodes,
  selectedId,
  viewport,
  setViewport,
  onSelect,
  onPatch,
  onGenerate,
  onDuplicate,
  onDelete,
  isFavorite,
  onToggleFavorite,
  loras,
  model,
  onDropType,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [panning, setPanning] = useState(false)
  const panRef = useRef<{ sx: number; sy: number; vx: number; vy: number } | null>(null)
  const spaceRef = useRef(false)

  // 世界坐标 = (屏幕 - 平移) / 缩放
  const toWorld = useCallback(
    (clientX: number, clientY: number) => {
      const el = wrapRef.current
      if (!el) return { x: 0, y: 0 }
      const rect = el.getBoundingClientRect()
      return {
        x: (clientX - rect.left - viewport.x) / viewport.z,
        y: (clientY - rect.top - viewport.y) / viewport.z,
      }
    },
    [viewport],
  )

  // 滚轮缩放：以光标为锚点，缩放前后光标下的世界点不动
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      setViewport((v) => {
        const next = Math.min(MAX_Z, Math.max(MIN_Z, v.z * Math.exp(-e.deltaY * 0.0015)))
        const k = next / v.z
        return { z: next, x: px - (px - v.x) * k, y: py - (py - v.y) * k }
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [setViewport])

  // 空格按住 = 临时抓手
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space') spaceRef.current = true
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') spaceRef.current = false
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  const startPan = (e: React.PointerEvent) => {
    const target = e.target as HTMLElement
    // 原来靠 data-bg="1" 判断，但 .canvas-grid 是 pointer-events:none，
    // 点空白时 target 是 viewport 自己（没有 data-bg）→ 画布永远拖不动。
    // 改成反向排除：只要不落在节点/面板/控件上，就算可拖区域。
    const onUi = !!target.closest(
      '.node, button, input, textarea, select, a, label, .palette, .book, .train-panel',
    )
    const withSpace = spaceRef.current
    const middle = e.button === 1
    if (onUi && !withSpace && !middle) return
    if (e.button !== 0 && !middle) return
    e.preventDefault()
    onSelect(null)
    panRef.current = { sx: e.clientX, sy: e.clientY, vx: viewport.x, vy: viewport.y }
    setPanning(true)
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  }

  const movePan = (e: React.PointerEvent) => {
    const p = panRef.current
    if (!p) return
    setViewport((v) => ({ ...v, x: p.vx + (e.clientX - p.sx), y: p.vy + (e.clientY - p.sy) }))
  }

  const endPan = (e: React.PointerEvent) => {
    if (!panRef.current) return
    panRef.current = null
    setPanning(false)
    try {
      ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
    } catch {
      /* 指针已释放 */
    }
  }

  const gridStyle: React.CSSProperties = {
    backgroundPosition: `${viewport.x}px ${viewport.y}px, ${viewport.x}px ${viewport.y}px`,
    backgroundSize: `${24 * viewport.z}px ${24 * viewport.z}px, ${120 * viewport.z}px ${120 * viewport.z}px, ${120 * viewport.z}px ${120 * viewport.z}px`,
  }

  return (
    <div
      ref={wrapRef}
      className={`canvas-viewport${panning ? ' panning' : ''}`}
      onPointerDown={startPan}
      onPointerMove={movePan}
      onPointerUp={endPan}
      onPointerCancel={endPan}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes('application/x-node-type')) e.preventDefault()
      }}
      onDrop={(e) => {
        const t = e.dataTransfer.getData('application/x-node-type')
        if (!t) return
        e.preventDefault()
        onDropType(t, toWorld(e.clientX, e.clientY))
      }}
    >
      <div className="canvas-grid" data-bg="1" style={gridStyle} />
      <div
        className="canvas-world"
        style={{ transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.z})` }}
      >
        <div className="origin-mark" style={{ left: 0, top: 0 }} title="画布原点 (0,0)" />
        {nodes.map((n) => (
          <GenNodeCard
            key={n.id}
            node={n}
            selected={n.id === selectedId}
            zoom={viewport.z}
            onSelect={() => onSelect(n.id)}
            onPatch={(p) => onPatch(n.id, p)}
            onGenerate={() => onGenerate(n.id)}
            onDuplicate={() => onDuplicate(n.id)}
            onDelete={() => onDelete(n.id)}
            favorited={isFavorite(n.params.prompt)}
            onToggleFavorite={() => onToggleFavorite(n.id)}
            loras={loras}
            model={model}
          />
        ))}
      </div>
      {nodes.length === 0 && (
        <div className="canvas-empty" data-bg="1">
          <div className="empty-card" data-bg="1">
            <h2 data-bg="1">空白画布</h2>
            <p data-bg="1">
              从左侧节点列表拖入「图生成」，或点它右边的 <b>+</b> 放到原点。
            </p>
            <p className="dim" data-bg="1">
              拖空白处平移</p>
          </div>
        </div>
      )}
      <div className="canvas-hint" data-bg="1">
        <b>拖空白</b> 平移　<b>滚轮</b> 缩放　<b>Delete</b> 删除　<b>Ctrl+D</b> 复制
      </div>
      <div className="node-width-hint" style={{ display: 'none' }}>
        {NODE_W}
      </div>
    </div>
  )
}
