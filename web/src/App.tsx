import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Canvas from './canvas/Canvas'
import NodePalette from './components/NodePalette'
import PromptBook from './components/PromptBook'
import { hasPrompt, loadPrompts, removePrompt, savePrompts, togglePrompt } from './store/prompts'
import { fetchJob, health, submitTxt2Img } from './api'
import type { FavoritePrompt, GenNode, GenParams, Viewport } from './types'
import type { NodePatch } from './nodes/GenNodeCard'

const STORAGE_KEY = 'meow.canvas.v1'

const DEFAULT_PARAMS: GenParams = {
  prompt: '',
  negative: '',
  width: 512,
  height: 512,
  steps: 25,
  cfg: 7.5,
  seed: -1,
  preset: 'sprite',
  pixel: 4,
  palette: 16,
  matte: false,
}

let seq = 0
const newId = () => `n${Date.now().toString(36)}${(seq++).toString(36)}`

function makeNode(x: number, y: number, params?: Partial<GenParams>): GenNode {
  return {
    id: newId(),
    kind: 'gen',
    x,
    y,
    view: 'expanded',
    params: { ...DEFAULT_PARAMS, ...params },
  }
}

type Saved = { nodes: GenNode[]; viewport: Viewport }

// 仅供截图/联调：?demo=1 铺一个示例节点，?demo=gen 同时触发一次生成
const DEMO = new URLSearchParams(location.search).get('demo')

export default function App() {
  const [nodes, setNodes] = useState<GenNode[]>(() => {
    if (DEMO) {
      return [{
        id: newId(), kind: 'gen', x: 0, y: 0, view: 'expanded',
        params: { ...DEFAULT_PARAMS, prompt: '白色猫骑士，红披风，武士刀，正面全身' },
      }]
    }
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) {
        const s = JSON.parse(raw) as Saved
        if (Array.isArray(s.nodes)) return s.nodes
      }
    } catch {
      /* 旧数据损坏就忽略 */
    }
    return []
  })
  const [viewport, setViewport] = useState<Viewport>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) {
        const s = JSON.parse(raw) as Saved
        if (s.viewport && typeof s.viewport.z === 'number') return s.viewport
      }
    } catch {
      /* 忽略 */
    }
    return { x: 120, y: 90, z: 1 }
  })
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [device, setDevice] = useState<string>('')
  const [favs, setFavs] = useState<FavoritePrompt[]>(() => loadPrompts())

  useEffect(() => savePrompts(favs), [favs])

  const nodesRef = useRef(nodes)
  nodesRef.current = nodes

  // 自动保存（防抖 400ms）
  useEffect(() => {
    const t = setTimeout(() => {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ nodes, viewport }))
      } catch {
        /* 存储满了就放弃 */
      }
    }, 400)
    return () => clearTimeout(t)
  }, [nodes, viewport])

  useEffect(() => {
    health()
      .then((h) => setDevice(h.device ?? ''))
      .catch(() => setDevice('后端未连接'))
  }, [])


  const patchNode = useCallback((id: string, patch: NodePatch) => {
    setNodes((list) => list.map((n) => (n.id === id ? { ...n, ...patch } : n)))
  }, [])

  const addNode = useCallback((_type: string, world?: { x: number; y: number }) => {
    const at = world ?? { x: 0, y: 0 }
    setNodes((list) => {
      // 原点放多个时阶梯错开，避免完全重叠
      const sameSpot = list.filter((n) => n.x === at.x && n.y === at.y).length
      const node = makeNode(at.x + sameSpot * 24, at.y + sameSpot * 24)
      setSelectedId(node.id)
      return [...list, node]
    })
  }, [])

  const duplicateNode = useCallback((id: string) => {
    setNodes((list) => {
      const src = list.find((n) => n.id === id)
      if (!src) return list
      const copy: GenNode = {
        ...src,
        id: newId(),
        x: src.x + 32,
        y: src.y + 32,
        job: undefined,
        view: 'expanded',
        params: { ...src.params },
      }
      setSelectedId(copy.id)
      return [...list, copy]
    })
  }, [])

  const deleteNode = useCallback((id: string) => {
    setNodes((list) => list.filter((n) => n.id !== id))
    setSelectedId((cur) => (cur === id ? null : cur))
  }, [])

  const generate = useCallback(async (id: string) => {
    const node = nodesRef.current.find((n) => n.id === id)
    if (!node || !node.params.prompt.trim()) return
    patchNode(id, { job: { id: 'pending', status: 'queued', startedAt: Date.now() } })
    try {
      const { job_id } = await submitTxt2Img(node.params)
      patchNode(id, { job: { id: job_id, status: 'queued', startedAt: Date.now() } })
    } catch (e) {
      patchNode(id, {
        job: { id: 'error', status: 'failed', error: e instanceof Error ? e.message : '提交失败' },
      })
    }
  }, [patchNode])

  // 截图/联调专用：demo=gen 时自动跑一次生成
  const demoFired = useRef(false)
  useEffect(() => {
    if (DEMO !== 'gen' || demoFired.current) return
    const first = nodesRef.current[0]
    if (!first) return
    demoFired.current = true
    void generate(first.id)
  }, [generate])

  // 轮询：只查还在排队/生成中的节点
  useEffect(() => {
    const timer = setInterval(async () => {
      const active = nodesRef.current.filter(
        (n) => n.job && (n.job.status === 'queued' || n.job.status === 'running') && n.job.id !== 'pending',
      )
      if (active.length === 0) return
      const results = await Promise.allSettled(active.map((n) => fetchJob(n.job!.id)))
      setNodes((list) =>
        list.map((n) => {
          const idx = active.findIndex((a) => a.id === n.id)
          if (idx < 0) return n
          const r = results[idx]
          if (r.status !== 'fulfilled') return n
          return { ...n, job: { ...n.job!, ...r.value, startedAt: n.job?.startedAt } }
        }),
      )
    }, 700)
    return () => clearInterval(timer)
  }, [])

  // 快捷键：Delete 删除、Ctrl/Cmd+D 复制（输入框内不拦截）
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      const typing =
        el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable
      if (!selectedId) return
      if ((e.key === 'Delete' || e.key === 'Backspace') && !typing) {
        e.preventDefault()
        deleteNode(selectedId)
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault()
        duplicateNode(selectedId)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedId, deleteNode, duplicateNode])

  const fit = useCallback(() => {
    const list = nodesRef.current
    if (list.length === 0) {
      setViewport({ x: 120, y: 90, z: 1 })
      return
    }
    const minX = Math.min(...list.map((n) => n.x))
    const minY = Math.min(...list.map((n) => n.y))
    const maxX = Math.max(...list.map((n) => n.x + 300))
    const maxY = Math.max(...list.map((n) => n.y + 420))
    const pad = 80
    const z = Math.min(1.2, Math.max(0.3, Math.min(
      (window.innerWidth - pad * 2) / (maxX - minX || 1),
      (window.innerHeight - 140 - pad * 2) / (maxY - minY || 1),
    )))
    setViewport({
      z,
      x: pad - minX * z,
      y: 96 - minY * z,
    })
  }, [])

  const zoomTo = useCallback((z: number) => {
    setViewport((v) => {
      const cx = window.innerWidth / 2
      const cy = window.innerHeight / 2
      const k = z / v.z
      return { z, x: cx - (cx - v.x) * k, y: cy - (cy - v.y) * k }
    })
  }, [])

  // 收藏：以「提示词文本」为键去重，节点上的星标据此点亮
  const isFavorite = useCallback((text: string) => hasPrompt(favs, text), [favs])

  const toggleFavorite = useCallback((id: string) => {
    const n = nodesRef.current.find((x) => x.id === id)
    if (!n) return
    setFavs((list) => togglePrompt(list, n.params.prompt))
  }, [])

  const applyFavorite = useCallback((text: string) => {
    const n = nodesRef.current.find((x) => x.id === selectedId)
    if (!n) return
    patchNode(n.id, { params: { ...n.params, prompt: text } })
  }, [patchNode, selectedId])

  const removeFavorite = useCallback((id: string) => {
    setFavs((list) => removePrompt(list, id))
  }, [])

  const zoomLabel = useMemo(() => `${Math.round(viewport.z * 100)}%`, [viewport.z])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">喵</span>
          <span className="title">喵叫工作台</span>
          <span className="sub">画布 · 文生图</span>
        </div>
        <div className="spacer" />
        <span className={`gpu ${device && !device.includes('未连接') ? 'ok' : 'bad'}`} title={device}>
          {device || '连接中…'}
        </span>
        <div className="zoom-group">
          <button onClick={() => zoomTo(Math.max(0.2, viewport.z / 1.2))}>−</button>
          <span className="zoom-val">{zoomLabel}</span>
          <button onClick={() => zoomTo(Math.min(2.4, viewport.z * 1.2))}>+</button>
          <button onClick={() => zoomTo(1)} title="回到 100%">1:1</button>
          <button onClick={fit} title="适应全部节点">适应</button>
        </div>
      </header>

      <main className="stage">
        <div className="left-col">
          <NodePalette onAdd={(t) => addNode(t)} />
          <PromptBook items={favs} canApply={!!selectedId}
                      onApply={applyFavorite} onRemove={removeFavorite} />
        </div>
        <Canvas
          nodes={nodes}
          selectedId={selectedId}
          viewport={viewport}
          setViewport={setViewport}
          onSelect={setSelectedId}
          onPatch={patchNode}
          onGenerate={generate}
          onDuplicate={duplicateNode}
          onDelete={deleteNode}
          isFavorite={isFavorite}
          onToggleFavorite={toggleFavorite}
          onDropType={(t, world) => addNode(t, world)}
        />
        <div className="badge-count">{nodes.length} 个节点</div>
      </main>
    </div>
  )
}
