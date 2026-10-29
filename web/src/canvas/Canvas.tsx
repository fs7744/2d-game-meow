import { useCallback, useEffect, useRef, useState } from 'react'
import type { ModelCaps, PoseDir } from '../api'
import { fileNameOf } from '../fileName'
import type { GenNode, LoraInfo, ModelRecommend, Viewport } from '../types'
import GenNodeCard, { NODE_W, PORT_CHAR_Y, PORT_SRC_Y, type NodePatch } from '../nodes/GenNodeCard'

/** 连线指向节点的哪个输入：source（源图/动作参考）还是 charSource（角色图） */
export type LinkField = 'source' | 'char'

type Props = {
  nodes: GenNode[]
  model: string
  /** 当前底模的候选数上限（后端按总显存算），透传给每个节点卡 */
  maxCandidates?: number
  /** W4 可用的骨架帧目录 */
  poseDirs?: PoseDir[]
  /** 本机是否装了 SDXL refiner 权重 */
  hasRefiner?: boolean
  /** 当前底模是否支持原生透明背景 */
  canTransparent?: boolean
  /** 可用底模清单（节点各自选，不再是全局唯一） */
  models?: string[]
  /** 底模能力表：节点按自己的 workflow 过滤可选项 */
  modelCaps?: Record<string, ModelCaps>
  /** 侧栏的「新节点默认底模」，节点下拉里显示成「跟随默认（xxx）」 */
  defaultModel?: string
  /** 各底模的推荐参数（候选上限要按**节点自己的**底模算，不能用全局的） */
  recommend?: Record<string, ModelRecommend>
  /** 底模名 → 架构（sd15 / sdxl / qwen21），节点据此过滤自己的 LoRA 列表 */
  modelArch?: Record<string, string>
  /** 给单个节点换底模（keepParams=true 时只改底模，不套推荐参数） */
  onNodeModel?: (id: string, next: string, keepParams?: boolean) => void
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
  /** 骨骼编辑器导出 PNG（Blob），由 App 上传并挂到节点上 */
  onExportSkeleton?: (id: string, blob: Blob) => void
  /** 骨骼编辑器「从图片提取」：上传 + DWPose 提关节 */
  onImportPose?: (id: string, file: File) => void | Promise<void>
  onDropFiles: (files: File[], world: { x: number; y: number }, targetId?: string) => void
  /** 连线落点：field 决定接到源图还是角色图 */
  onConnect: (from: string, to: string, field: LinkField) => void
  /** 断开一条连线（点连线中点的 ×，或右键） */
  onDisconnect: (to: string, field: LinkField) => void
  onPreview: (url: string, name: string) => void
  projectId: string
}

const MIN_Z = 0.2
const MAX_Z = 2.4

export default function Canvas({
  nodes,
  model,
  maxCandidates,
  poseDirs,
  hasRefiner,
  canTransparent,
  models,
  modelCaps,
  defaultModel,
  recommend,
  modelArch,
  onNodeModel,
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
  onDropType,
  onSpawnImg2Img,
  onExportSkeleton,
  onImportPose,
  onDropFiles,
  onConnect,
  onDisconnect,
  onPreview,
  projectId,
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

  /**
   * 可以当源图的节点：出过图的节点 + 图片节点（它们没有 job，路径在 image 上）。
   *
   * **已连上但还没出图的上游也要列进来**：连线只记 nodeId，节点可以先连后生成。
   * 如果下拉只列出"已有产物"的节点，先连好的那一路会显示成「未选择源图」——
   * 连线和下拉互相打脸，用户会以为连线没生效（实测复现）。
   */
  const referencedIds = new Set(
    nodes.flatMap((n) => [n.source?.nodeId, n.charSource?.nodeId]).filter(Boolean) as string[],
  )
  const sourceOptions = nodes
    .filter((n) => n.job?.path || n.job?.raw
      || ((n.kind === 'image' || n.kind === 'poseedit') && n.image?.path)
      || referencedIds.has(n.id))
    .map((n) => {
      const kindLabel = n.kind === 'image' ? '图片' : n.kind === 'poseedit' ? '骨骼'
        : n.kind === 'gen' ? '图生成' : '图'
      const name = (n.kind === 'image' || n.kind === 'poseedit')
        ? fileNameOf(n.image?.path) || n.id
        : n.params.prompt.slice(0, 20) || n.id
      // 还没出图要标出来，否则用户以为连上就能直接用
      const pending = !(n.job?.path || n.job?.raw || n.image?.path) ? '（未生成）' : ''
      return { id: n.id, label: `${kindLabel} · ${name}${pending}` }
    })

  /**
   * 当前视口/回调的最新值，供 startLink 里挂的原生监听读取。
   *
   * 为什么不直接在 effect 里挂：`setLinking(true)` 之后要等 React 渲染完才跑 effect，
   * 这中间有一段窗口。自动化拖拽（以及手速极快的人）会在这段窗口里就松手，
   * pointerup 落在监听挂上之前 = 连线静默丢失。改成 pointerdown 时**同步**挂监听，
   * 用 ref 读最新值避开闭包过期。
   */
  const vpRef = useRef(viewport)
  vpRef.current = viewport
  const connectRef = useRef(onConnect)
  connectRef.current = onConnect

  const endLink = useCallback((e: PointerEvent) => {
    const from = linkFrom.current
    // **先存再清**：hoverRef 在下面会被清空，清空之后再去读就永远是 null
    //（这个顺序踩过一次：连线在真实浏览器能用、在无布局的 jsdom 里全丢）
    const hoverAt = hoverRef.current
    linkFrom.current = null
    hoverRef.current = null
    setHoverId(null)
    setLinking(false)
    setLinkTip(null)
    if (!from) return
    /**
     * 落点判定：**先 elementFromPoint，再退回 hover**。
     *
     * 首选 elementFromPoint：只有它能说出"落在哪个输入端口"——角色迁移 /
     * 序列帧替换有两个图片输入（源图 = 动作参考，角色图 = 身份），节点级
     * hover 永远只会连到 source，角色图那一路根本连不上。
     *
     * 但它依赖布局，且会被覆盖层挡住（灯箱、蒙版编辑器、jsdom 无布局时
     * 直接返回 null）。所以拿不到就退回 hoverRef——至少把线连到节点上，
     * 掉进"拖半天没反应"比连错一路更让人困惑。
     */
    let el: HTMLElement | null = null
    try {
      // jsdom 没有 elementFromPoint（无布局），直接调会 TypeError 并把整个
      // pointerup 打断——表现就是"拖了线但什么都没连上"。特征检测 + try 兜住，
      // 拿不到就走下面的 hover 分支。
      if (typeof document.elementFromPoint === 'function') {
        el = document.elementFromPoint(e.clientX, e.clientY) as HTMLElement | null
      }
    } catch {
      el = null
    }
    const nodeEl = el?.closest('[data-node-id]') as HTMLElement | null
    let to = nodeEl?.dataset.nodeId
    let field = (el?.closest('[data-port]') as HTMLElement | null)?.dataset.port
    if (!to) {
      to = hoverAt ?? undefined
      field = 'source'
    }
    if (!to) return
    connectRef.current(from, to, (field as LinkField) || 'source')
  }, [])

  const moveLink = useCallback((e: PointerEvent) => {
    const v = vpRef.current
    const el = wrapRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    setLinkTip({
      x: (e.clientX - rect.left - v.x) / v.z,
      y: (e.clientY - rect.top - v.y) / v.z,
    })
  }, [])

  const startLink = useCallback((id: string) => {
    linkFrom.current = id
    hoverRef.current = null
    setHoverId(null)
    setLinking(true)
    setLinkTip({ x: 0, y: 0 })
    // 同步挂：不能等 effect，否则快速松手会丢
    window.addEventListener('pointermove', moveLink)
    window.addEventListener('pointerup', endLink)
    window.addEventListener('pointercancel', endLink)
  }, [moveLink, endLink])

  useEffect(() => () => {
    // 组件卸载时兜底摘掉，避免监听泄漏到 window 上
    window.removeEventListener('pointermove', moveLink)
    window.removeEventListener('pointerup', endLink)
    window.removeEventListener('pointercancel', endLink)
  }, [moveLink, endLink])

  const setHover = useCallback((id: string | null) => {
    hoverRef.current = id
    setHoverId(id)
  }, [])

  const onDragOverNode = useCallback((id: string) => {
    dragOverRef.current = id
  }, [])

  // 连线几何：两端锚在卡片左/右边缘，线贴着卡片顶部，不会穿过内容
  // 节点宽度现在可拖：连线端点必须跟着实际宽度走，否则线会插进卡片中间
  const widthOf = (n: GenNode) =>
    (n.kind === 'image' ? n.image?.w ?? NODE_W : n.w ?? NODE_W)
  /**
   * 输入端口相对节点顶部的实际 y。
   *
   * 为什么不用常量：端口挂在「源图区 / 角色图区」上，而这两个框在各类节点里的
   * 纵向位置并不一致（转像素节点在源图区上面多一行说明，序列帧替换有两块输入框）。
   * 写死偏移的后果是线端点和圆圈差十几像素，看起来像画错了。这里从 DOM 实测，
   * 只在节点集合/尺寸变化时重测。
   */
  const [portY, setPortY] = useState<Record<string, number>>({})
  const nodesKey = nodes
    .map((n) => `${n.id}:${n.kind}:${n.w ?? ''}:${n.h ?? ''}:${n.view}`)
    .join('|')
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const next: Record<string, number> = {}
    el.querySelectorAll<HTMLElement>('[data-node-id]').forEach((nodeEl) => {
      const id = nodeEl.dataset.nodeId!
      const top = nodeEl.getBoundingClientRect().top
      nodeEl.querySelectorAll<HTMLElement>('[data-port]').forEach((p) => {
        const r = p.getBoundingClientRect()
        // 用屏幕坐标差 → 世界坐标：除以缩放比
        next[`${id}:${p.dataset.port}`] = (r.top + r.height / 2 - top) / viewport.z
      })
    })
    setPortY(next)
  }, [nodesKey, viewport.z])

  const portAt = (n: GenNode, field: LinkField) =>
    portY[`${n.id}:${field === 'char' ? 'char' : 'source'}`] ??
    (field === 'char' ? PORT_CHAR_Y : PORT_SRC_Y)
  const edgePath = (from: GenNode, toX: number, toY: number) => {
    const x1 = from.x + widthOf(from)
    const y1 = from.y + PORT_SRC_Y
    return `M ${x1} ${y1} C ${x1 + 70} ${y1}, ${toX - 70} ${toY}, ${toX} ${toY}`
  }
  type Edge = { key: string; to: string; field: LinkField; d: string; mx: number; my: number }
  const edges: Edge[] = []
  for (const n of nodes) {
    // 一个节点可能有两条入边（源图 + 角色图），key 必须带 field 才不互相覆盖
    const targets: [LinkField, string | undefined][] = [
      ['source', n.source?.nodeId],
      ['char', n.charSource?.nodeId],
    ]
    for (const [field, fromId] of targets) {
      if (!fromId) continue
      const from = nodes.find((x) => x.id === fromId)
      if (!from) continue
      const ty = n.y + portAt(n, field)
      edges.push({ key: `${n.id}:${field}`, to: n.id, field,
        d: edgePath(from, n.x, ty),
        // 断开按钮贴在端口外侧 26px，而不是贝塞尔中点：
        // 中点经常正好压在节点身上（上下堆叠时尤其明显），会挡住参数控件。
        // 贴在端口旁边还有语义——它管的就是这一路输入。
        mx: n.x - 26, my: ty })
    }
  }

  // 可以当新背景用的节点：图片节点（拖进来的本地图）或任何已出图的节点
  const bgOptions = nodes
    .filter((n) => n.kind === 'image' || n.job?.path)
    .map((n) => ({
      id: n.id,
      label:
        n.kind === 'image'
          ? `图片 · ${fileNameOf(n.image?.path) || n.id}`
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
          {/* 箭头：一条线上只有一个终点，用 marker 而不是手画三角形——
              手画要算切线角度，曲线一变就得跟着改，迟早对不上 */}
          <defs>
            <marker id="link-arrow" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" className="link-arrow-head" />
            </marker>
            <marker id="link-arrow-char" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" className="link-arrow-head char" />
            </marker>
          </defs>
          {edges.map((e) => (
            <path key={e.key} className={`link ${e.field}`} d={e.d}
              markerEnd={`url(#link-arrow${e.field === 'char' ? '-char' : ''})`} />
          ))}
          {linking && linkTip && linkFrom.current && (() => {
            const from = nodes.find((x) => x.id === linkFrom.current)
            if (!from) return null
            return <path className="link pending" d={edgePath(from, linkTip.x, linkTip.y)} />
          })()}
        </svg>
        {/* 断开按钮：连线本身是 pointer-events:none 的细线，直接点很难命中，
            所以在端口外侧放真正的按钮。只在悬停/选中该节点时显示——
            常显会让满屏都是 ✕，反而看不清连线本身 */}
        {edges.filter((e) => hoverId === e.to || selectedId === e.to).map((e) => (
          <button
            key={`x-${e.key}`}
            className="link-cut"
            style={{ left: e.mx, top: e.my }}
            title={`断开这条连线（${e.field === 'char' ? '角色图' : '源图'}）`}
            onPointerDown={(ev) => ev.stopPropagation()}
            onClick={(ev) => {
              ev.stopPropagation()
              onDisconnect(e.to, e.field)
            }}
          >
            ✕
          </button>
        ))}
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
            loraBase={modelArch?.[n.params.model || model] || (n.params.model || model)}
            model={n.params.model || model}
            onModel={(m, opts) => onNodeModel?.(n.id, m, opts?.keepNodes)}
            nodesModels={models}
            modelCaps={modelCaps}
            defaultModel={defaultModel ?? model}
            maxCandidates={recommend?.[n.params.model || model]?.max_candidates
              ?? maxCandidates}
            poseDirs={poseDirs}
            hasRefiner={hasRefiner}
            canTransparent={
              // 按**这个节点自己的底模**判断，不是全局默认——一张画布上
              // 可能一个节点用 Qwen、另一个用 SDXL。modelCaps 缺失时退回
              // App 算好的全局值（老后端不返回 caps）
              modelCaps?.[n.params.model || model]?.transparent ?? canTransparent
            }
            onSpawnImg2Img={() => onSpawnImg2Img(n.id)}
            onExportSkeleton={(blob) => onExportSkeleton?.(n.id, blob)}
            onImportPose={(file) => onImportPose?.(n.id, file)}
            sourceNode={nodes.find((x) => x.id === n.source?.nodeId)}
            sourceOptions={sourceOptions}
            bgOptions={bgOptions}
            charThumbOf={(id) => {
              if (!id) return undefined
              const n = nodes.find((x) => x.id === id)
              if (!n) return undefined
              if (n.kind === 'image') return n.image?.url
              return n.job?.url
            }}
            onStartLink={() => startLink(n.id)}
            onHover={setHover}
            onDragOverNode={onDragOverNode}
            linkTarget={hoverId === n.id}
            onPreview={onPreview}
            projectId={projectId}
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
        <b>连线中点 ✕</b> 断开
      </div>
      <div className="node-width-hint" style={{ display: 'none' }}>
        {NODE_W}
      </div>
    </div>
  )
}
