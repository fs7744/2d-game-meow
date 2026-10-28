import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Canvas from './canvas/Canvas'
import Sidebar from './components/Sidebar'
import TrainPanel from './components/TrainPanel'
import { hasPrompt, loadPrompts, removePrompt, savePrompts, togglePrompt } from './store/prompts'
import {
  deleteAssets, fetchJob, fetchLoras, fetchModels, health, submitImg2Img, submitTxt2Img,
  uploadImage,
} from './api'
import ConfirmDialog from './components/ConfirmDialog'
import type {
  FavoritePrompt, GenNode, GenParams, LoraInfo, NodeImage, NodeKind, NodeSource, Viewport,
} from './types'
import type { NodePatch } from './nodes/GenNodeCard'

const STORAGE_KEY = 'meow.canvas.v1'

const DEFAULT_PARAMS: GenParams = {
  prompt: '',
  negative: '',
  // 实测：512² 出全身角色必糊；768² 是 SD1.5 的甜点（1024² 会开始重复肢体）。
  // pixel=6 / palette=32 正好把 768 降到 128 有效像素 —— 标准精灵尺寸。
  width: 768,
  height: 768,
  steps: 30,
  cfg: 7.0,
  seed: -1,
  preset: 'sprite',
  pixel: 6,
  palette: 32,
  matte: false,
  clipSkip: 1,
  // 图生图默认幅度：0.55 能明显改内容又不丢构图；低于 0.35 基本只是磨皮
  strength: 0.55,
  lora: '',
  loraScale: 0.8,
}

let seq = 0
const newId = () => `n${Date.now().toString(36)}${(seq++).toString(36)}`

function makeNode(
  x: number,
  y: number,
  kind: NodeKind = 'gen',
  params?: Partial<GenParams>,
  source?: NodeSource,
  image?: NodeImage,
): GenNode {
  return {
    id: newId(),
    kind,
    x,
    y,
    view: 'expanded',
    source,
    image,
    params: { ...DEFAULT_PARAMS, ...params },
  }
}

type Saved = { nodes: GenNode[]; viewport: Viewport }

/** 老存档没有 lora/loraScale/strength，读回来补默认值，否则下拉会显示空、强度变 NaN。 */
function migrate(n: GenNode): GenNode {
  return { ...n, kind: n.kind ?? 'gen', params: { ...DEFAULT_PARAMS, ...n.params } }
}

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
        if (Array.isArray(s.nodes)) return s.nodes.map(migrate)
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
  const [loras, setLoras] = useState<LoraInfo[]>([])
  const [trainOpen, setTrainOpen] = useState(false)
  const [models, setModels] = useState<string[]>(['sd15'])
  const [model, setModel] = useState<string>('sd15')
  const [matteBackend, setMatteBackend] = useState('auto')
  const [uploadMsg, setUploadMsg] = useState('')
  const [pendingDelete, setPendingDelete] = useState<GenNode | null>(null)
  const pickFiles = useRef<HTMLInputElement | null>(null)

  useEffect(() => savePrompts(favs), [favs])

  const reloadLoras = useCallback(() => {
    fetchLoras(model).then(setLoras).catch(() => setLoras([]))
  }, [model])

  useEffect(() => { reloadLoras() }, [reloadLoras])

  useEffect(() => {
    fetchModels().then((m) => {
      setModels(m)
      if (m.length && !m.includes(model)) setModel(m[0])
    })
  }, [model])

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

  const addNode = useCallback((type: string, world?: { x: number; y: number }) => {
    const at = world ?? { x: 0, y: 0 }
    if (type === 'image') {
      // 图片节点的内容来自本机文件：走文件选择器，选完直接落成节点
      pickFiles.current?.click()
      return
    }
    const kind: NodeKind = type === 'img2img' ? 'img2img' : 'gen'
    setNodes((list) => {
      // 原点放多个时阶梯错开，避免完全重叠
      const sameSpot = list.filter((n) => n.x === at.x && n.y === at.y).length
      const node = makeNode(at.x + sameSpot * 24, at.y + sameSpot * 24, kind)
      setSelectedId(node.id)
      return [...list, node]
    })
  }, [])

  /** 以某个节点的产物为源，在它右边新建一个图生图节点。 */
  const spawnImg2Img = useCallback((id: string) => {
    setNodes((list) => {
      const src = list.find((n) => n.id === id)
      if (!src) return list
      const node = makeNode(src.x + 340, src.y, 'img2img',
        { prompt: src.params.prompt }, { nodeId: src.id })
      setSelectedId(node.id)
      return [...list, node]
    })
  }, [])

  /**
   * 拖进来的本地图：落到空白处 → 变成一个图片节点；落到图生图节点上 → 直接当它的源。
   * 上传在后端做（图生图需要磁盘绝对路径，前端传 base64 没有意义）。
   */
  const handleFiles = useCallback(
    async (files: File[], world: { x: number; y: number }, targetId?: string) => {
      const imgs = files.filter((f) => f.type.startsWith('image/'))
      if (!imgs.length) return
      const target = targetId ? nodesRef.current.find((n) => n.id === targetId) : undefined
      let i = 0
      for (const f of imgs) {
        try {
          const up = await uploadImage(f)
          if (target && target.kind === 'img2img') {
            patchNode(target.id, { source: { path: up.path } })
            continue
          }
          const node = makeNode(world.x + i * 24, world.y + i * 24, 'image', undefined, undefined, {
            path: up.path, url: up.url, w: Math.min(320, up.w), h: up.h,
          })
          setSelectedId(node.id)
          setNodes((list) => [...list, node])
        } catch (e) {
          // 上传失败要看得见，不能静默丢文件
          setUploadMsg(`上传失败：${e instanceof Error ? e.message : '未知错误'}`)
          window.setTimeout(() => setUploadMsg(''), 5000)
        }
        i += 1
      }
    },
    [patchNode],
  )

  /** 连线：把 from 接到 to（图生图节点）上。 */
  const onConnect = useCallback((from: string, to: string) => {
    const target = nodesRef.current.find((n) => n.id === to)
    if (!target || target.kind !== 'img2img' || target.id === from) return
    patchNode(to, { source: { nodeId: from } })
  }, [patchNode])

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

  /** 删除前先弹确认框：删节点 ≠ 删文件，两件事分开。 */
  const requestDelete = useCallback((id: string) => {
    const n = nodesRef.current.find((x) => x.id === id)
    if (!n) return
    setPendingDelete(n)
  }, [])

  /** 待删节点的文件清单 + 还被谁引用。 */
  const deleteInfo = useCallback((n: GenNode) => {
    const files = [n.job?.path, n.job?.raw, n.image?.path]
      .filter((x): x is string => !!x)
      .map((path) => ({ path, name: path.split(/[\\/]/).pop() || path }))
    const refs = nodesRef.current.filter(
      (o) => o.id !== n.id && o.source?.nodeId === n.id,
    ).length
    return { files, refs }
  }, [])

  const finishDelete = useCallback(async (hard: boolean) => {
    const n = pendingDelete
    setPendingDelete(null)
    if (!n) return
    if (hard) {
      const { files } = deleteInfo(n)
      try {
        await deleteAssets(files.map((f) => f.path))
      } catch (e) {
        setUploadMsg(`删除失败：${e instanceof Error ? e.message : '未知错误'}`)
        window.setTimeout(() => setUploadMsg(''), 5000)
      }
    }
    // 从画布移除，并断开指向它的连线（否则下游图生图节点会悬空）
    setNodes((list) =>
      list
        .filter((x) => x.id !== n.id)
        .map((x) => (x.source?.nodeId === n.id ? { ...x, source: undefined } : x)),
    )
    setSelectedId((cur) => (cur === n.id ? null : cur))
  }, [pendingDelete, deleteInfo])

  /**
   * 图生图的源图绝对路径。优先用上游节点后处理**之前**的原图（raw）：
   * 拿已经像素化成 128 有效像素的成品回炉，模型只会把色块重新糊成插画。
   */
  const sourcePathOf = useCallback((n: GenNode): string | undefined => {
    if (n.source?.path) return n.source.path
    const up = n.source?.nodeId
      ? nodesRef.current.find((x) => x.id === n.source!.nodeId)
      : undefined
    if (!up) return undefined
    if (up.kind === 'image') return up.image?.path
    return up.job?.raw || up.job?.path
  }, [])

  const generate = useCallback(async (id: string) => {
    const node = nodesRef.current.find((n) => n.id === id)
    if (!node || !node.params.prompt.trim()) return
    let init: string | undefined
    if (node.kind === 'img2img') {
      init = sourcePathOf(node)
      if (!init) {
        // 没有源图就没法提交，直接在节点上报错，别让用户等一个必然失败的任务
        patchNode(id, {
          job: { id: 'error', status: 'failed',
            error: '还没有源图：先生成一张，或点生成节点下方的「↳ 图生图」' },
        })
        return
      }
    }
    patchNode(id, { job: { id: 'pending', status: 'queued', startedAt: Date.now() } })
    try {
      const { job_id } = init
        ? await submitImg2Img(node.params, { model, matteBackend }, init)
        : await submitTxt2Img(node.params, { model, matteBackend })
      patchNode(id, { job: { id: job_id, status: 'queued', startedAt: Date.now() } })
    } catch (e) {
      patchNode(id, {
        job: { id: 'error', status: 'failed', error: e instanceof Error ? e.message : '提交失败' },
      })
    }
    // 底模/抠图后端是全局设置，必须进依赖，否则切了底模还在用旧值提交
  }, [patchNode, model, matteBackend, sourcePathOf])

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
        requestDelete(selectedId)
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault()
        duplicateNode(selectedId)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedId, requestDelete, duplicateNode])

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
          <span className="sub">画布 · 文生图 / 图生图</span>
        </div>
        <div className="spacer" />
        <button className="train-btn" onClick={() => setTrainOpen(true)} title="训练自己的 LoRA">
          训练 LoRA
        </button>
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
        <Sidebar
          loras={loras}
          favs={favs}
          onAddNode={(kind) => addNode(kind ?? 'gen')}
          canApply={!!selectedId}
          onApplyFav={applyFavorite}
          onRemoveFav={removeFavorite}
          models={models}
          model={model}
          onModel={setModel}
          matteBackend={matteBackend}
          onMatteBackend={setMatteBackend}
        />
        <Canvas
          nodes={nodes}
          model={model}
          selectedId={selectedId}
          viewport={viewport}
          setViewport={setViewport}
          onSelect={setSelectedId}
          onPatch={patchNode}
          onGenerate={generate}
          onDuplicate={duplicateNode}
          onDelete={requestDelete}
          isFavorite={isFavorite}
          onToggleFavorite={toggleFavorite}
          loras={loras}
          onDropType={(t, world) => addNode(t, world)}
          onSpawnImg2Img={spawnImg2Img}
          onDropFiles={(files, world, targetId) => void handleFiles(files, world, targetId)}
          onConnect={onConnect}
        />
        <div className="badge-count">{nodes.length} 个节点</div>
      </main>

      <input
        ref={pickFiles}
        type="file"
        accept="image/*"
        multiple
        style={{ display: 'none' }}
        onChange={(e) => {
          const files = e.target.files ? Array.from(e.target.files) : []
          e.target.value = ''
          if (files.length) {
            void handleFiles(files, { x: 120, y: 120 })
          }
        }}
      />

      {uploadMsg && <div className="toast">{uploadMsg}</div>}

      {pendingDelete && (() => {
        const { files, refs } = deleteInfo(pendingDelete)
        const kindLabel =
          pendingDelete.kind === 'image' ? '图片' : pendingDelete.kind === 'img2img' ? '图生图' : '图生成'
        const summary = `${kindLabel}节点${pendingDelete.params.prompt
          ? `：「${pendingDelete.params.prompt.slice(0, 28)}」`
          : ''}`
        return (
          <ConfirmDialog
            title="删除这个节点？"
            summary={summary}
            files={files}
            note={
              files.length
                ? '「真删」会连磁盘上的产物（PNG / sidecar / raw）一起删掉，不可恢复。'
                : '这个节点还没有产出文件，删掉只是从画布上移除。'
            }
            blockReason={
              refs > 0
                ? `还有 ${refs} 个节点把它当源图，删了文件那些节点就没有源了——先断开连线或只移除节点。`
                : undefined
            }
            onCancel={() => setPendingDelete(null)}
            onRemove={() => void finishDelete(false)}
            onDelete={() => void finishDelete(true)}
          />
        )
      })()}

      {trainOpen && (
        <TrainPanel onClose={() => setTrainOpen(false)} onTrained={reloadLoras} />
      )}
    </div>
  )
}
