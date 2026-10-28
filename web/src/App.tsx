import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Canvas from './canvas/Canvas'
import Sidebar from './components/Sidebar'
import TrainPanel from './components/TrainPanel'
import { hasPrompt, loadPrompts, removePrompt, savePrompts, togglePrompt } from './store/prompts'
import {
  loadStore, makeProject, metaOf, saveStore, type Project, type ProjectMeta,
} from './store/projects'
import {
  deleteAssets, fetchJob, fetchLoras, fetchModels, health, submitAiRepaint, submitImg2Img,
  submitInpaint, submitMatte, submitPixelify, submitPoseClone, submitSpriteSwap, submitTxt2Img,
  uploadImage,
} from './api'
import ConfirmDialog from './components/ConfirmDialog'
import Lightbox from './components/Lightbox'
import type {
  FavoritePrompt, GenNode, GenParams, LoraInfo, ModelRecommend, NodeImage, NodeKind,
  NodeSource, Viewport,
} from './types'
import type { NodePatch } from './nodes/GenNodeCard'

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
  // 抠图节点默认：透明底（纯去背景）
  bgMode: 'alpha',
  bgColor: '#ffffff',
  bgNodeId: '',
  // 局部重绘的蒙版（笔刷画完上传到项目目录，节点只记路径）
  maskPath: '',
  maskUrl: '',
  maskCoverage: 0,
  // 局部 AI 重绘（w5）：默认值照 pixelforge 的实测结论
  rpContext: 0.35,      // 选区外带 35% 上下文
  rpExpand: 8,          // 蒙版外扩 8px
  rpFeather: 4,         // 贴回羽化 4px
  rpMatchColor: true,   // 接缝带色彩校正
  rpPaletteSnap: false, // 色板吸附：像素素材才需要
  // 完美像素转换：0 = 自动检测网格；32 色是像素画的常见上限
  pxTarget: 0,
  pxColors: 32,
  pxSample: 'median',
  pxGridSnap: true,
  pxTransparentBg: false,
  pxSnapTo: 16,           // 网格吸附到 16 的整数倍（精灵常用尺寸）
  // 角色迁移：姿态强度 / 身份强度 / 人物占画面高度
  pcCnScale: 1.0,
  pcIpaScale: 0.8,
  pcFill: 0.8,
  // 参考模式默认 auto：先试骨架，认不出人体自动降级到轮廓（游戏素材走这条路）
  pcRefMode: 'auto',
  // 序列帧替换：0 = 自动检测精灵图网格
  ssCols: 0,
  ssRows: 0,
  ssMode: 'generate',
  ssRefMode: 'outline',
  ssCnScale: 1.0,
  ssIpaScale: 0.85,
  ssPalette: 32,
  // 多候选择优：业界实测同 seed 也会漂 20~30%，靠生成多张挑一张
  ssCandidates: 2,
  ssGateRetries: 1,
  // 引擎：AnimateDiff 走时序注意力，帧间一致性更好（实测 8 帧 7.4s / 13.1GB）
  ssEngineMode: 'auto',
  ssVramProfile: 'auto',
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

/** 切换底模时会跟着改的参数。只动采样，不动 prompt / lora 这些创作内容。 */
const REC_KEYS = ['width', 'height', 'steps', 'cfg', 'pixel', 'palette'] as const

function pickRec(rec?: ModelRecommend): Partial<GenParams> {
  if (!rec) return {}
  return { width: rec.width, height: rec.height, steps: rec.steps,
           cfg: rec.cfg, pixel: rec.pixel, palette: rec.palette }
}

/** 这组参数是否仍等于某个底模的推荐值——等于就说明用户没手改过。 */
function matchesRec(p: GenParams, rec?: ModelRecommend): boolean {
  return !!rec && REC_KEYS.every((k) => p[k] === rec[k])
}

/** 老存档没有 lora/loraScale/strength，读回来补默认值，否则下拉会显示空、强度变 NaN。 */
function migrate(n: GenNode): GenNode {
  return { ...n, kind: n.kind ?? 'gen', params: { ...DEFAULT_PARAMS, ...n.params } }
}

// 仅供截图/联调：?demo=1 铺一个示例节点，?demo=gen 同时触发一次生成
const DEMO = new URLSearchParams(location.search).get('demo')

export default function App() {
  // 项目：每个项目是一套独立的节点 + 视口。store 里存全部项目的完整数据，
  // 内存里只留当前项目的（切项目就是把另一套换上）。
  const storeRef = useRef<{ current: string; projects: Project[] }>(loadStore())
  const [currentId, setCurrentId] = useState<string>(storeRef.current.current)
  const [projectMeta, setProjectMeta] = useState<ProjectMeta[]>(
    () => storeRef.current.projects.map(metaOf),
  )
  const initial = storeRef.current.projects.find((p) => p.id === storeRef.current.current)
  const [nodes, setNodes] = useState<GenNode[]>(() => {
    if (DEMO) {
      return [{
        id: newId(), kind: 'gen', x: 0, y: 0, view: 'expanded',
        params: { ...DEFAULT_PARAMS, prompt: '白色猫骑士，红披风，武士刀，正面全身' },
      }]
    }
    return (initial?.nodes ?? []).map(migrate)
  })
  const [viewport, setViewport] = useState<Viewport>(
    () => initial?.viewport ?? { x: 120, y: 90, z: 1 },
  )
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [device, setDevice] = useState<string>('')
  const [favs, setFavs] = useState<FavoritePrompt[]>(() => loadPrompts())
  const [loras, setLoras] = useState<LoraInfo[]>([])
  const [trainOpen, setTrainOpen] = useState(false)
  const [models, setModels] = useState<string[]>(['sd15'])
  // 每个底模的推荐采样参数，由后端 /models 给出（见 types.ts 的 ModelRecommend）
  const [recommend, setRecommend] = useState<Record<string, ModelRecommend>>({})
  const [model, setModel] = useState<string>('sd15')
  const [matteBackend, setMatteBackend] = useState('auto')
  const [uploadMsg, setUploadMsg] = useState('')
  const [notice, setNotice] = useState('')          // 底模切换之类的一次性提示
  const [pendingDelete, setPendingDelete] = useState<GenNode | null>(null)
  const [pendingProjectDelete, setPendingProjectDelete] = useState<string | null>(null)
  const [preview, setPreview] = useState<{ url: string; name: string } | null>(null)
  const pickFiles = useRef<HTMLInputElement | null>(null)

  useEffect(() => savePrompts(favs), [favs])

  const reloadLoras = useCallback(() => {
    fetchLoras(model).then(setLoras).catch(() => setLoras([]))
  }, [model])

  useEffect(() => { reloadLoras() }, [reloadLoras])

  useEffect(() => {
    fetchModels().then((m) => {
      setModels(m.available)
      setRecommend(m.recommend)
      if (m.available.length && !m.available.includes(model)) setModel(m.available[0])
    })
  }, [model])

  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(''), 3200)
    return () => clearTimeout(t)
  }, [notice])

  // addNode 的 useCallback 依赖是空数组，拿不到最新的 recommend/model，所以走 ref
  const recRef = useRef(recommend)
  recRef.current = recommend
  const modelRef = useRef(model)
  modelRef.current = model

  const nodesRef = useRef(nodes)
  nodesRef.current = nodes
  const viewportRef = useRef(viewport)
  viewportRef.current = viewport
  const currentIdRef = useRef(currentId)
  currentIdRef.current = currentId

  // 自动保存（防抖 400ms）：只写回**当前项目**，别的项目原封不动
  useEffect(() => {
    const t = setTimeout(() => {
      const st = storeRef.current
      const projects = st.projects.map((p) =>
        p.id === currentId
          ? { ...p, nodes, viewport, updated: new Date().toISOString() }
          : p,
      )
      storeRef.current = { current: currentId, projects }
      saveStore(currentId, projects)
      setProjectMeta(projects.map(metaOf))
    }, 400)
    return () => clearTimeout(t)
  }, [nodes, viewport, currentId])

  useEffect(() => {
    health()
      .then((h) => setDevice(h.device ?? ''))
      .catch(() => setDevice('后端未连接'))
  }, [])


  /** 把当前内存里的节点/视口写回 store（切项目、新建项目、删项目前都要先做） */
  const flushCurrent = useCallback(() => {
    const st = storeRef.current
    const projects = st.projects.map((p) =>
      p.id === currentIdRef.current
        ? { ...p, nodes: nodesRef.current, viewport: viewportRef.current,
            updated: new Date().toISOString() }
        : p,
    )
    storeRef.current = { ...st, projects }
    return projects
  }, [])

  /** 切换项目：把另一套节点/视口整个换上，互不干涉。 */
  const switchProject = useCallback((id: string) => {
    if (id === currentIdRef.current) return
    const projects = flushCurrent()
    const target = projects.find((p) => p.id === id)
    if (!target) return
    storeRef.current = { current: id, projects }
    saveStore(id, projects)
    setProjectMeta(projects.map(metaOf))
    setCurrentId(id)
    setNodes(target.nodes)
    setViewport(target.viewport)
    setSelectedId(null)
  }, [flushCurrent])

  const createProject = useCallback((name?: string) => {
    const projects = flushCurrent()
    const p = makeProject(name?.trim() || `项目 ${projects.length + 1}`)
    const next = [...projects, p]
    storeRef.current = { current: p.id, projects: next }
    saveStore(p.id, next)
    setProjectMeta(next.map(metaOf))
    setCurrentId(p.id)
    setNodes([])
    setViewport({ x: 120, y: 90, z: 1 })
    setSelectedId(null)
  }, [flushCurrent])

  const renameProject = useCallback((id: string, name: string) => {
    const projects = flushCurrent().map((p) =>
      p.id === id ? { ...p, name: name.trim() || p.name } : p,
    )
    storeRef.current = { ...storeRef.current, projects }
    saveStore(storeRef.current.current, projects)
    setProjectMeta(projects.map(metaOf))
  }, [flushCurrent])

  /** 删项目：hard=true 时连这个项目的产物文件一起删（不可恢复）。 */
  const deleteProject = useCallback(async (id: string, hard: boolean) => {
    const projects = flushCurrent()
    const target = projects.find((p) => p.id === id)
    if (!target) return
    if (hard) {
      // 整个项目目录删掉：产物、上传、帧序列都在里面
      try {
        await deleteAssets([`projects/${id}`])
      } catch (e) {
        setUploadMsg(`删除失败：${e instanceof Error ? e.message : '未知错误'}`)
        window.setTimeout(() => setUploadMsg(''), 5000)
      }
    }
    let next = projects.filter((p) => p.id !== id)
    if (next.length === 0) next = [makeProject('我的项目')]   // 至少留一个空项目
    const wasCurrent = currentIdRef.current === id
    const current = wasCurrent ? next[0].id : currentIdRef.current
    storeRef.current = { current, projects: next }
    saveStore(current, next)
    setProjectMeta(next.map(metaOf))
    if (wasCurrent) {
      setCurrentId(next[0].id)
      setNodes(next[0].nodes)
      setViewport(next[0].viewport)
      setSelectedId(null)
    }
  }, [flushCurrent])

  const patchNode = useCallback((id: string, patch: NodePatch) => {
    setNodes((list) => list.map((n) => (n.id === id ? { ...n, ...patch } : n)))
  }, [])

  const addNode = useCallback((type: string, world?: { x: number; y: number }) => {
    const at = world ?? { x: 0, y: 0 }
    if (type === 'airepaint') {
      // 局部 AI 重绘的默认幅度比图生图高：实测 0.8 时模型几乎保留原样，
      // 等于没重画（0.9 才开始明显改动）
      const node = makeNode(at.x, at.y, 'airepaint', { strength: 0.9 })
      setSelectedId(node.id)
      setNodes((list) => [...list, node])
      return
    }
    if (type === 'image') {
      // 图片节点的内容来自本机文件：走文件选择器，选完直接落成节点
      pickFiles.current?.click()
      return
    }
    const kind: NodeKind =
      type === 'img2img' ? 'img2img'
        : type === 'bg' ? 'bg'
          : type === 'inpaint' ? 'inpaint'
            : type === 'airepaint' ? 'airepaint'
              : type === 'pixelify' ? 'pixelify'
                : type === 'poseclone' ? 'poseclone'
                  : type === 'spriteswap' ? 'spriteswap'
                    : 'gen'
    setNodes((list) => {
      // 原点放多个时阶梯错开，避免完全重叠
      const sameSpot = list.filter((n) => n.x === at.x && n.y === at.y).length
      const node = makeNode(at.x + sameSpot * 24, at.y + sameSpot * 24, kind)
      // 新节点跟当前底模走：否则选了 SDXL 还是按 SD1.5 的 512/30 步出一张废图
      const rec = recRef.current[modelRef.current]
      if (rec) node.params = { ...node.params, ...pickRec(rec) }
      setSelectedId(node.id)
      return [...list, node]
    })
  }, [])

  /**
   * 切换底模：把推荐采样参数套到「用户没动过」的生成节点上。
   *
   * 为什么不是无条件覆盖：用户可能已经为某个节点手调过步数/尺寸。
   * 判据是「当前值仍等于旧底模的推荐值」——等于说明没动过，可以跟着换；
   * 不等就保留，避免把人调好的参数冲掉。
   *
   * 为什么需要这件事：后端实测 SDXL 在 1024² 反而更差（可交付率 92%→75%），
   * 前端若按直觉填 1024，用户会稳定拿到更差的结果且没有任何报错。
   */
  const switchModel = useCallback((next: string) => {
    const oldRec = recommend[model]
    const rec = recommend[next]
    setModel(next)
    if (!rec) return
    const cap = Math.max(1, Math.min(6, rec.max_candidates || 6))
    setNodes((list) => list.map((n) => {
      // 序列帧：候选数跟着新底模的显存上限走。不夹的话，用户先在 SD1.5
      // 设了 4 张、切到 SDXL（上限 3）仍会提交 4——运行时守卫会降，
      // 但界面上写着 4，等于骗人。
      if (n.kind === 'spriteswap' && n.params.ssCandidates > cap) {
        return { ...n, params: { ...n.params, ssCandidates: cap } }
      }
      if ((n.kind ?? 'gen') !== 'gen') return n
      if (oldRec && !matchesRec(n.params, oldRec)) return n
      return { ...n, params: { ...n.params, ...pickRec(rec) } }
    }))
    setNotice(`已套用 ${next} 推荐参数：${rec.width}×${rec.height} / ${rec.steps} 步` +
      `，候选上限 ${cap} 张/帧`)
  }, [recommend, model])

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
          const up = await uploadImage(f, currentId)
          if (target && (target.kind === 'img2img' || target.kind === 'bg'
            || target.kind === 'inpaint' || target.kind === 'airepaint'
            || target.kind === 'pixelify' || target.kind === 'poseclone'
            || target.kind === 'spriteswap')) {
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
    [patchNode, currentId],
  )

  /** 连线：把 from 接到 to（图生图节点）上。 */
  const onConnect = useCallback((from: string, to: string) => {
    const target = nodesRef.current.find((n) => n.id === to)
    // 能接源的节点：图生图（要底图）、抠图与局部重绘（要源图）；文生图没有「底」这回事
    if (!target || !['img2img', 'bg', 'inpaint', 'airepaint', 'pixelify', 'poseclone',
      'spriteswap'].includes(target.kind)) return
    if (target.id === from) return
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

  /** 角色迁移的「角色图」路径（身份来源）。角色图要用**成品**而不是 raw：
   *  IP-Adapter 学的是外观特征，raw 是未像素化的原图，学不到像素感。 */
  const charPathOf = useCallback((n: GenNode): string | undefined => {
    if (n.charSource?.path) return n.charSource.path
    const up = n.charSource?.nodeId
      ? nodesRef.current.find((x) => x.id === n.charSource!.nodeId)
      : undefined
    if (!up) return undefined
    if (up.kind === 'image') return up.image?.path
    return up.job?.path || up.job?.raw
  }, [])

  const generate = useCallback(async (id: string) => {
    const node = nodesRef.current.find((n) => n.id === id)
    if (!node) return
    // 抠图节点不需要提示词（纯图像处理）；文生图/图生图需要
    const needPrompt = node.kind === 'gen' || node.kind === 'img2img' || node.kind === 'inpaint'
      || node.kind === 'airepaint'
    if (needPrompt && !node.params.prompt.trim()) return
    let init: string | undefined
    if (node.kind === 'img2img' || node.kind === 'bg' || node.kind === 'inpaint'
      || node.kind === 'airepaint' || node.kind === 'pixelify' || node.kind === 'poseclone'
      || node.kind === 'spriteswap') {
      init = sourcePathOf(node)
      if (!init) {
        // 没有源图就没法提交，直接在节点上报错，别让用户等一个必然失败的任务
        patchNode(id, {
          job: { id: 'error', status: 'failed',
            error: '还没有源图：把别的节点连过来，或直接把本地图拖到这个节点上' },
        })
        return
      }
    }
    patchNode(id, { job: { id: 'pending', status: 'queued', startedAt: Date.now() } })
    try {
      let jobId: string
      if (node.kind === 'poseclone') {
        const posePath = sourcePathOf(node)
        const charPath = charPathOf(node)
        if (!posePath || !charPath) {
          patchNode(id, {
            job: { id: 'error', status: 'failed',
              error: !posePath
                ? '缺动作参考图：连线接一张，或直接把本地图拖到这个节点上'
                : '缺角色图：在下面「角色图」里选一个' },
          })
          return
        }
        const r = await submitPoseClone(
          node.params, { model, matteBackend, scope: currentId }, posePath, charPath)
        jobId = r.job_id
      } else if (node.kind === 'spriteswap') {
        const sheetPath = sourcePathOf(node)
        const charPath = charPathOf(node)
        if (!sheetPath || !charPath) {
          patchNode(id, {
            job: { id: 'error', status: 'failed',
              error: !sheetPath
                ? '缺动作序列帧：连线接一张精灵图，或直接把本地图拖到这个节点上'
                : '缺角色图：在下面「角色图」里选一个' },
          })
          return
        }
        const r = await submitSpriteSwap(
          node.params, { model, matteBackend, scope: currentId }, sheetPath, charPath)
        jobId = r.job_id
      } else if (node.kind === 'pixelify') {
        const r = await submitPixelify(
          node.params, { model, matteBackend, scope: currentId }, init!)
        jobId = r.job_id
      } else if (node.kind === 'airepaint') {
        const r = await submitAiRepaint(
          node.params, { model, matteBackend, scope: currentId }, init!, node.params.maskPath)
        jobId = r.job_id
      } else if (node.kind === 'inpaint') {
        const r = await submitInpaint(
          node.params, { model, matteBackend, scope: currentId }, init!, node.params.maskPath)
        jobId = r.job_id
      } else if (node.kind === 'bg') {
        // 换背景：背景图来自另一个图片节点
        const bgNode = node.params.bgMode === 'image'
          ? nodesRef.current.find((n) => n.id === node.params.bgNodeId)
          : undefined
        const bgPath = bgNode?.image?.path || bgNode?.job?.path
        const r = await submitMatte(
          node.params, { model, matteBackend, scope: currentId }, init!,
          bgPath, node.params.bgMode === 'color' ? node.params.bgColor : undefined)
        jobId = r.job_id
      } else if (init) {
        const r = await submitImg2Img(node.params, { model, matteBackend, scope: currentId }, init)
        jobId = r.job_id
      } else {
        const r = await submitTxt2Img(node.params, { model, matteBackend, scope: currentId })
        jobId = r.job_id
      }
      patchNode(id, { job: { id: jobId, status: 'queued', startedAt: Date.now() } })
    } catch (e) {
      patchNode(id, {
        job: { id: 'error', status: 'failed', error: e instanceof Error ? e.message : '提交失败' },
      })
    }
    // 底模/抠图后端是全局设置，必须进依赖，否则切了底模还在用旧值提交
  }, [patchNode, model, matteBackend, sourcePathOf, currentId])

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
      if (e.key === 'Escape' && preview) {
        setPreview(null)
        return
      }
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
  }, [selectedId, requestDelete, duplicateNode, preview])

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
          <span className="sub">
            {projectMeta.find((p) => p.id === currentId)?.name ?? '画布'} · 文生图 / 图生图
          </span>
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
          projects={projectMeta}
          currentProject={currentId}
          onSwitchProject={switchProject}
          onCreateProject={() => createProject()}
          onRenameProject={renameProject}
          onDeleteProject={(id) => setPendingProjectDelete(id)}
          onAddNode={(kind) => addNode(kind ?? 'gen')}
          canApply={!!selectedId}
          onApplyFav={applyFavorite}
          onRemoveFav={removeFavorite}
          models={models}
          recommend={recommend}
          model={model}
          onModel={switchModel}
          matteBackend={matteBackend}
          onMatteBackend={setMatteBackend}
        />
        <Canvas
          nodes={nodes}
          model={model}
          maxCandidates={recommend[model]?.max_candidates}
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
          onPreview={(url, name) => setPreview({ url, name })}
          projectId={currentId}
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

      {(uploadMsg || notice) && <div className="toast">{uploadMsg || notice}</div>}

      {preview && <Lightbox url={preview.url} name={preview.name} onClose={() => setPreview(null)} />}

      {pendingProjectDelete && (() => {
        const proj = projectMeta.find((p) => p.id === pendingProjectDelete)
        return (
          <ConfirmDialog
            title="删除这个项目？"
            summary={`项目「${proj?.name ?? '未命名'}」，${proj?.count ?? 0} 个节点`}
            removeLabel="仅从列表移除"
            note="「真删」会把这个项目里所有节点的产物文件（PNG / sidecar / raw）一起删掉，不可恢复。"
            onCancel={() => setPendingProjectDelete(null)}
            onRemove={() => { void deleteProject(pendingProjectDelete, false); setPendingProjectDelete(null) }}
            onDelete={() => { void deleteProject(pendingProjectDelete, true); setPendingProjectDelete(null) }}
          />
        )
      })()}

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
