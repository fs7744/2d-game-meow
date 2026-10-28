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
  onSpawnImg2Img: (id: string) => void
  onDropFiles: (files: File[], world: { x: number; y: number }, targetId?: string) => void
  onConnect: (from: string, to: string) => void
  onPreview: (url: string, name: string) => void
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
  onSpawnImg2Img,
  onDropFiles,
  onConnect,
  onPreview,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [panning, setPanning] = useState(false)
  const panRef = useRef<{ sx: number; sy: number; vx: number; vy: number } | null>(null)
  const spaceRef = useRef(false)
  // 连线：从某个节点的「输出点」拖到另一个节点上
  const [linking, setLinking] = useState(false)
  const [linkTip, setLinkTip] = useState<{ x: number; y: number } | null>(null)
  const linkFrom = useRef<string | null>(null)
  const hoverRef = useRef<string | null>(null)
  const [hoverId, setHoverId] = useState<string | null>(null)
  // 文件拖拽时悬停在哪个节点上（drop 时要判断「落在节点上」还是「落在空白」）
  const dragOverRef = useRef<string | null>(null)

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

  // 图生图的源候选：出过图的节点 + 图片节点（它们没有 job，路径在 image 上）
  const sourceOptions = nodes
    .filter((n) => n.job?.path || n.job?.raw || (n.kind === 'image' && n.image?.path))
    .map((n) => ({
      id: n.id,
      label:
        n.kind === 'image'
          ? `图片 · ${n.image?.path.split(/[\\/]/).pop() || n.id}`
          : `${n.kind === 'img2img' ? '图生图' : '图生成'} · ${n.params.prompt.slice(0, 20) || n.id}`,
    }))

  const startLink = useCallback((id: string) => {
    linkFrom.current = id
    hoverRef.current = null
    setHoverId(null)
    setLinking(true)
    setLinkTip({ x: 0, y: 0 })
  }, [])

  useEffect(() => {
    if (!linking) return
    const move = (e: PointerEvent) => {
      const w = toWorld(e.clientX, e.clientY)
      setLinkTip({ x: w.x, y: w.y })
    }
    const up = () => {
      const from = linkFrom.current
      const to = hoverRef.current
      linkFrom.current = null
      hoverRef.current = null
      setHoverId(null)
      setLinking(false)
      setLinkTip(null)
      if (from && to) onConnect(from, to)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [linking, toWorld, onConnect])

  const setHover = useCallback((id: string | null) => {
    hoverRef.current = id
    setHoverId(id)
  }, [])

  const onDragOverNode = useCallback((id: string) => {
    dragOverRef.current = id
  }, [])

  // 连线几何：两端都锚在卡头（y+13），这样线永远贴着卡片顶部，不会穿过内容
  const widthOf = (n: GenNode) => (n.kind === 'image' ? n.image?.w ?? NODE_W : NODE_W)
  const edgePath = (from: GenNode, toX: number, toY: number) => {
    const x1 = from.x + widthOf(from)
    const y1 = from.y + 13
    return `M ${x1} ${y1} C ${x1 + 70} ${y1}, ${toX - 70} ${toY}, ${toX} ${toY}`
  }
  const edges = nodes
    .filter((n) => n.source?.nodeId)
    .map((n) => {
      const from = nodes.find((x) => x.id === n.source!.nodeId)
      if (!from) return null
      return { id: n.id, d: edgePath(from, n.x, n.y + 13) }
    })
    .filter((e): e is { id: string; d: string } => e !== null)

  // 可以当新背景用的节点：图片节点（拖进来的本地图）或任何已出图的节点
  const bgOptions = nodes
    .filter((n) => n.kind === 'image' || n.job?.path)
    .map((n) => ({
      id: n.id,
      label:
        n.kind === 'image'
          ? `图片 · ${n.image?.path.split(/[\\/]/).pop() || n.id}`
          : `${n.kind === 'bg' ? '抠图' : n.kind === 'img2img' ? '图生图' : '图生成'} · ${
              n.params.prompt.slice(0, 16) || n.id
            }`,
    }))

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
        // 本机图片：允许拖进画布（空白处建图片节点，节点上则作为该节点的源）
        if (e.dataTransfer.types.includes('Files')) {
          e.preventDefault()
          e.dataTransfer.dropEffect = 'copy'
        }
      }}
      onDrop={(e) => {
        const world = toWorld(e.clientX, e.clientY)
        const files = e.dataTransfer.files ? Array.from(e.dataTransfer.files) : []
        if (files.length) {
          e.preventDefault()
          const target = dragOverRef.current ?? undefined
          dragOverRef.current = null
          onDropFiles(files, world, target)
          return
        }
        const t = e.dataTransfer.getData('application/x-node-type')
        if (!t) return
        e.preventDefault()
        onDropType(t, world)
      }}
    >
      <div className="canvas-grid" data-bg="1" style={gridStyle} />
      <div
        className="canvas-world"
        style={{ transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.z})` }}
      >
        <div className="origin-mark" style={{ left: 0, top: 0 }} title="画布原点 (0,0)" />
        <svg className="links">
          {edges.map((e) => (
            <path key={e.id} className="link" d={e.d} />
          ))}
          {linking && linkTip && linkFrom.current && (() => {
            const from = nodes.find((x) => x.id === linkFrom.current)
            if (!from) return null
            return <path className="link pending" d={edgePath(from, linkTip.x, linkTip.y)} />
          })()}
        </svg>
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
            onSpawnImg2Img={() => onSpawnImg2Img(n.id)}
            sourceNode={nodes.find((x) => x.id === n.source?.nodeId)}
            sourceOptions={sourceOptions}
            bgOptions={bgOptions}
            onStartLink={() => startLink(n.id)}
            onHover={setHover}
            onDragOverNode={onDragOverNode}
            linkTarget={hoverId === n.id}
            onPreview={onPreview}
          />
        ))}
      </div>
      {nodes.length === 0 && (
        <div className="canvas-empty" data-bg="1">
          <div className="empty-card" data-bg="1">
            <h2 data-bg="1">空白画布</h2>
            <p data-bg="1">
              从左侧节点列表拖入「图生成」或「图生图」，或点它右边的 <b>+</b> 放到原点。
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
