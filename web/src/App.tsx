import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Canvas, { type LinkField } from './canvas/Canvas'
import Sidebar from './components/Sidebar'
import { hasPrompt, loadPrompts, removePrompt, savePrompts, togglePrompt } from './store/prompts'
import {
  loadStore, makeProject, metaOf, saveStore, type Project, type ProjectMeta,
} from './store/projects'
import {
  deleteAssets, fetchJob, fetchModels, health, submitAiRepaint, submitImg2Img,
  fetchPoseDirs, submitToPixel, submitInpaint, submitMatte,
  submitPixelify, submitPoseClone,
  submitPoseSeq,
  submitSpriteSwap, submitTxt2Img, uploadImage, extractPose,
} from './api'
import ConfirmDialog from './components/ConfirmDialog'
import Lightbox from './components/Lightbox'
import type { PoseDir } from './api'
import type { ModelCaps } from './api'
import type {
  FavoritePrompt, GenNode, GenParams, ModelRecommend, NodeImage, NodeKind,
  NodeSource, Viewport,
} from './types'
import type { NodePatch } from './nodes/GenNodeCard'
import { styleFromParams } from './genStyles'
import { defaultSkeleton, skeletonFromPoints } from './components/SkeletonEditor'
import { NODE_W } from './nodes/GenNodeCard'

/**
 * 画布**可见区域**（屏幕坐标）——必须扣掉左侧浮动侧栏。
 *
 * 侧栏是 `position: absolute` 浮在画布上的（展开 236px / 收起 42px），不属于
 * 布局流。所以 `window.innerWidth` 不等于可见宽度：用它算「中心」或「适应」
 * 会把节点摆到侧栏底下（实测新节点只露出右半截、「适应」之后依旧被挡住）。
 */
function canvasBox() {
  const rail = document.querySelector<HTMLElement>('.rail')
  const left = rail && !rail.classList.contains('collapsed')
    ? rail.offsetWidth : 0
  return {
    left,
    width: Math.max(240, window.innerWidth - left),
    height: window.innerHeight,
  }
}

/**
 * 可见区中心对应的世界坐标（再往回挪半个节点，让节点视觉居中）。
 *
 * 侧栏「+」不带坐标，以前一律落世界原点 (0,0)，而初始平移只有 120px——
 * 原点正好在 236px 宽的侧栏下面，用户点完什么都看不见。
 */
function centerWorld(v: Viewport): { x: number; y: number } {
  const b = canvasBox()
  return {
    x: (b.left + b.width / 2 - v.x) / v.z - NODE_W / 2,
    y: (b.height / 2 - v.y) / v.z - 200,
  }
}

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
  genStyle: 'sprite16',   // 产出风格：像素精灵 16-bit（最常用）
  // 转像素（topixel）：任意图 → 像素画（没有源图时先出图再转）
  apTarget: 192,          // 目标原生像素边长（192 装得下「全身 + 武器」）
  apColors: 32,           // 共享色板（32 色 = 16-bit 风格）
  apPreEnhance: 1.0,      // 量化前拉对比度——不开会发灰
  apEdgeRebuild: true,    // 轮廓描边——不描就「融化」
  apMatte: true,          // 先抠背景（转照片时可关掉保留背景）
  // 角色迁移：姿态强度 / 身份强度 / 人物占画面高度
  pcFill: 0.8,
  // 参考模式默认 auto：先试骨架，认不出人体自动降级到轮廓（游戏素材走这条路）
  pcRefMode: 'auto',
  // 底色：默认白底；角色本身偏白（白毛/白甲）时改 dark，抠图才不会啃掉脸
  bgTone: 'white',
  // 序列帧替换：0 = 自动检测精灵图网格
  ssCols: 0,
  ssRows: 0,
  ssMode: 'generate',
  ssRefMode: 'outline',
  ssPalette: 32,
  // 多候选择优：业界实测同 seed 也会漂 20~30%，靠生成多张挑一张
  ssCandidates: 2,
  ssGateRetries: 1,
  // 引擎：AnimateDiff 走时序注意力，帧间一致性更好（实测 8 帧 7.4s / 13.1GB）
  ssEngineMode: 'auto',
  ssVramProfile: 'auto',
  // 姿态序列（w4）的输入：骨架帧目录（也接受通配，如 assets/_pose_walk_*.png）
  psDir: '',
  // 透明底默认关：只有 Qwen 底模支持，SD 系列开了会被后端忽略，
  // 默认打开会让用户误以为选 SD 时也能出透明图
  transparent: false,
  // 空串 = 跟随侧栏的「新节点默认底模」。节点一旦自己选过就写具体值
  model: '',
}

let seq = 0
const newId = () => `n${Date.now().toString(36)}${(seq++).toString(36)}`

/**
 * 骨骼编辑节点的方位角 → 方向提示词。
 *
 * 骨架图本身已经带上朝向信息（肩膀叠合、鼻子偏一侧），但 Qwen 有时会把
 * "侧面骨架"仍理解成正面人物。方位角是编辑器里的确定值，直接补一句方向词
 * 是零成本的双保险——不改用户写的 prompt，只在提交时附加。
 */
const VIEW_PHRASES = [
  'front view, facing the viewer',
  'three-quarter front view from the right',
  'side view, facing right',
  'three-quarter rear view from the right',
  'back view, seen from behind',
  'three-quarter rear view from the left',
  'side view, facing left',
  'three-quarter front view from the left',
]
function viewPhrase(az: number): string {
  const a = ((Math.round(az) % 360) + 360) % 360
  return VIEW_PHRASES[Math.round(a / 45) % 8]
}

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
    // 骨骼编辑器节点：初始给一个「张手正面」骨架，打开就能拖
    ...(kind === 'poseedit' ? { skeleton: defaultSkeleton() } : {}),
  }
}

/** 切换底模时会跟着改的参数。只动采样，不动 prompt / lora 这些创作内容。 */
const REC_KEYS = ['width', 'height', 'steps', 'cfg', 'pixel', 'palette'] as const

function pickRec(rec?: ModelRecommend): Partial<GenParams> {
  if (!rec) return {}
  const out: Partial<GenParams> = {
    width: rec.width, height: rec.height, steps: rec.steps,
    cfg: rec.cfg, pixel: rec.pixel, palette: rec.palette,
  }
  // 底模可以额外带一个 preset：动漫底模给的是「质量词 + 简单背景 + 短负向」
  // 这套 danbooru 契约。只在有值时才覆盖，其它档位不受影响。
  if (rec.preset !== undefined) out.preset = rec.preset
  // **档位要跟着参数一起改**。只改参数不改 genStyle 的话，切到动漫底模后
  // 参数已经是「动漫立绘」（pixel/palette 归零），下拉却还停在「像素精灵」——
  // 显示与实物不一致，用户会以为下拉坏了（真踩过）。
  out.genStyle = styleFromParams(
    out.preset ?? 'sprite', out.pixel ?? 0, out.palette ?? 0)
  return out
}

/** 这组参数是否仍等于某个底模的推荐值——等于就说明用户没手改过。 */
function matchesRec(p: GenParams, rec?: ModelRecommend): boolean {
  return !!rec && REC_KEYS.every((k) => p[k] === rec[k])
}

/** 老存档可能缺后来新增的字段，读回来补齐默认值，否则节点渲染会拿到 undefined。 */
function migrate(n: GenNode): GenNode {
  // 旧存档兼容：'animepixel'（早期只做动漫人物）改名为 'topixel'（通用）
  const kind = (n.kind as string) === 'animepixel' ? 'topixel' : (n.kind ?? 'gen')
  return { ...n, kind: kind as NodeKind, params: { ...DEFAULT_PARAMS, ...n.params } }
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
  const [models, setModels] = useState<string[]>(['qwen-image-2.1'])
  // 每个底模的推荐采样参数，由后端 /models 给出（见 types.ts 的 ModelRecommend）
  const [recommend, setRecommend] = useState<Record<string, ModelRecommend>>({})
  // 底模名 → 架构（Qwen-only 之后恒为 qwen21）
  const [model, setModel] = useState<string>('qwen-image-2.1')
  // W4 的输入是"一组骨架帧"，后端扫 assets 给出可选目录，前端只做下拉
  const [poseDirs, setPoseDirs] = useState<PoseDir[]>([])
  // 底模能力表（透明底 / 指令改图 / LoRA）。当前底模支不支持透明底由它决定，
  // 不能按名字判断——底模名可以是任意目录名。
  const [caps, setCaps] = useState<Record<string, ModelCaps>>({})
  const [matteBackend, setMatteBackend] = useState('auto')
  const [uploadMsg, setUploadMsg] = useState('')
  const [notice, setNotice] = useState('')          // 底模切换之类的一次性提示
  const [pendingDelete, setPendingDelete] = useState<GenNode | null>(null)
  const [pendingProjectDelete, setPendingProjectDelete] = useState<string | null>(null)
  const [preview, setPreview] = useState<{ url: string; name: string } | null>(null)
  const pickFiles = useRef<HTMLInputElement | null>(null)

  useEffect(() => savePrompts(favs), [favs])

  // 骨架目录一次拉完即可（后端 20ms 扫完 assets 两层）；失败就空列表，
  // 节点里仍然可以手填路径，功能不因这个接口挂掉而消失。
  const reloadPoseDirs = useCallback(() => {
    fetchPoseDirs().then(setPoseDirs).catch(() => setPoseDirs([]))
  }, [])
  useEffect(() => { reloadPoseDirs() }, [reloadPoseDirs])

  useEffect(() => {
    fetchModels().then((m) => {
      setModels(m.available)
      setRecommend(m.recommend)
      setCaps(m.caps)
      // **跟随后端给的默认底模**（/models 的 current = engine.DEFAULT_MODEL）。
      //
      // 之前这里写死「有 SDXL 就用 SDXL」——那是 SD 时代的结论（SDXL@768 可交付率
      // 92% vs SD1.5 33%）。现在 SD 权重已归档、默认是 Qwen-Image-2.1，
      // 写死的规则会挑一个**用户根本没装**的底模出来。
      // 老后端没有 current 时退回「有 SDXL 用 SDXL」，保持向后兼容。
      const want = m.available.includes(m.current) ? m.current
        : (m.available.includes('sdxl') ? 'sdxl' : m.available[0])
      if (m.available.length && !m.available.includes(model)) setModel(want)
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
    setViewport({ x: canvasBox().left + 40, y: 90, z: 1 })
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
    // 不带坐标（侧栏「+」）→ 落到可见区中心，而不是世界原点：
    // 原点在侧栏底下，点完看不见节点（实测只露出右半截）。
    const at = world ?? centerWorld(viewportRef.current)
    if (type === 'airepaint') {
      const node = makeNode(at.x, at.y, 'airepaint')
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
                : type === 'topixel' ? 'topixel'
                : type === 'poseclone' ? 'poseclone'
                  : type === 'spriteswap' ? 'spriteswap'
                    : type === 'poseswap' ? 'poseswap'
                      : type === 'poseedit' ? 'poseedit'
                        : 'gen'
    setNodes((list) => {
      // 原点放多个时阶梯错开，避免完全重叠
      const sameSpot = list.filter((n) => n.x === at.x && n.y === at.y).length
      const node = makeNode(at.x + sameSpot * 24, at.y + sameSpot * 24, kind)
      // 新节点跟当前底模走：否则选了 SDXL 还是按 SD1.5 的 512/30 步出一张废图
      const rec = recRef.current[modelRef.current]
      // 骨骼编辑器不跑模型，别把采样参数写进去（没意义，还会误导）
      if (rec && kind !== 'poseedit') {
        node.params = { ...node.params, ...pickRec(rec) }
      }
      setSelectedId(node.id)
      return [...list, node]
    })
  }, [])

  /**
   * 给**单个节点**换底模，并套用新底模的推荐参数（只动这一个节点）。
   *
   * 底模是节点级属性（见 GenParams.model），画布上混跑 Qwen 立绘 + SDXL
   * 精灵帧是常态。套推荐参数仍然只在"用户没手改过"时生效——判据是当前值
   * 还等于旧底模的推荐值，否则保留手调结果。
   */
  const setNodeModel = useCallback((id: string, next: string, keepParams = false) => {
    const cur = nodesRef.current.find((n) => n.id === id)
    if (!cur) return
    if (!models.includes(next)) {
      setNotice(`未安装底模 ${next}，先下权重`)
      return
    }
    const rec = recommend[next]
    const oldRec = recommend[cur.params.model || model]
    setNodes((list) => list.map((n) => {
      if (n.id !== id) return n
      const patch: Partial<GenParams> = { model: next }
      // 新底模不支持原生透明底时，把勾着但无效的开关关掉——
      // 留着会让用户以为这次出的图有 alpha（实际没有）
      if (!caps[next]?.transparent) patch.transparent = false
      // keepParams：产出风格档位会自己把尺寸/像素块/LoRA patch 好，
      // 这里再套一遍推荐值会两边抢同一个节点（setNodes 读的是渲染时闭包的
      // 旧参数，会把刚 patch 的值覆盖掉——实测档位的 LoRA 和尺寸就是这么丢的）
      if (keepParams) {
        return { ...n, params: { ...n.params, ...patch } }
      }
      if (rec && (n.kind ?? 'gen') === 'gen' && (!oldRec || matchesRec(n.params, oldRec))) {
        Object.assign(patch, pickRec(rec))
      }
      // 角色迁移切到 Qwen：换一套适配参数。
      // 1024² 是 Qwen **文生图**的推荐，但角色迁移是「改图」路线（双图条件），
      // 768² / 28 步实测姿势跟随最准，也快得多（72s vs 150s+）。
      if (n.kind === 'poseclone' && next === 'qwen-image-2.1') {
        Object.assign(patch, { width: 768, height: 768, steps: 28, cfg: 4.0 })
      }
      // 序列帧候选上限按新底模的显存上限夹住，否则界面写着 4 实际只能出 3
      if (n.kind === 'spriteswap' && rec) {
        const cap = Math.max(1, Math.min(6, rec.max_candidates || 6))
        if (n.params.ssCandidates > cap) patch.ssCandidates = cap
      }
      return { ...n, params: { ...n.params, ...patch } }
    }))
    if (!keepParams && rec && (cur.kind ?? 'gen') === 'gen') {
      setNotice(`${next}：推荐 ${rec.width}×${rec.height} / ${rec.steps} 步`)
    }
  }, [models, recommend, caps, model])

  /**
   * 骨骼编辑器「导出骨架图」：前端画好 OpenPose-18 PNG → 上传 → 挂到节点 image 上。
   *
   * 挂 image 是关键：画布的连线与「源候选」只认有 image.path / job.path 的节点，
   * 有了它这张骨架图才能像普通图片节点一样连到角色迁移的「动作参考」输入。
   */
  const exportSkeleton = useCallback(async (id: string, blob: Blob) => {
    try {
      const f = new File([blob], 'skeleton.png', { type: 'image/png' })
      const up = await uploadImage(f, currentId)
      patchNode(id, {
        image: { path: up.path, url: up.url, w: 300, h: 300 },
      })
      setNotice('骨架图已导出：把它连到角色迁移节点的「动作参考」输入即可')
      window.setTimeout(() => setNotice(''), 5000)
    } catch (e) {
      setNotice(`导出骨架图失败：${e instanceof Error ? e.message : '未知错误'}`)
      window.setTimeout(() => setNotice(''), 5000)
    }
  }, [patchNode, currentId])

  /**
   * 骨骼编辑器「从图片提取」：上传原图 → 后端 DWPose 提 18 关节 → 2D 点补深度
   * 变成可旋转的 3D 骨架 → 写回节点。
   *
   * 不改 node.image：那个字段挂的是**导出的骨架图**（连线源）。把原图写进去
   * 会让下游角色迁移拿到原图而不是骨架，方向也就无从谈起。
   */
  const importPoseFromImage = useCallback(async (id: string, file: File) => {
    try {
      const up = await uploadImage(file, currentId)
      const r = await extractPose(up.path)
      patchNode(id, { skeleton: skeletonFromPoints(r.points, r.azimuth_guess) })
      const dir = r.azimuth_guess === 90 ? '右侧'
        : r.azimuth_guess === 270 ? '左侧' : '正面'
      setNotice(`已从图片提取 ${r.joints}/18 个关节（原图疑似${dir}）；`
        + '点上方方向按钮换视觉方向，再导出骨架图')
    } catch (e) {
      setNotice(`提取骨架失败：${e instanceof Error ? e.message : '未知错误'}`)
    }
    window.setTimeout(() => setNotice(''), 7000)
  }, [patchNode, currentId])

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
            || target.kind === 'spriteswap' || target.kind === 'topixel')) {
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

  /**
   * 连线：把 from 接到 to 的某个输入上。
   *
   * field 区分两个图片输入：
   *   source     —— 源图 / 动作参考（图生图、抠图、局部重绘…都用它）
   *   char       —— 角色图（身份来源），只有角色迁移 / 序列帧替换有
   * 落点由 Canvas 按端口 data-port 判定，用户拖到哪个圆点就接哪一路。
   */
  const onConnect = useCallback((from: string, to: string, field: LinkField) => {
    const target = nodesRef.current.find((n) => n.id === to)
    if (!target || target.id === from) return
    if (field === 'char') {
      // 只有这两类节点有角色图输入；别的节点拖过来静默忽略，别乱塞字段
      if (!['poseclone', 'spriteswap'].includes(target.kind)) return
      patchNode(to, { charSource: { nodeId: from } })
      return
    }
    // 能接源的节点：图生图（要底图）、抠图与局部重绘（要源图）；文生图没有「底」这回事
    if (!['img2img', 'bg', 'inpaint', 'airepaint', 'pixelify', 'poseclone',
      'spriteswap', 'topixel'].includes(target.kind)) return
    patchNode(to, { source: { nodeId: from } })
  }, [patchNode])

  /** 断开一条入边：只清对应字段，不影响另一路 */
  const onDisconnect = useCallback((to: string, field: LinkField) => {
    patchNode(to, field === 'char' ? { charSource: undefined } : { source: undefined })
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
    if (up.kind === 'image' || up.kind === 'poseedit') return up.image?.path
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
    /**
     * 这个节点用哪个底模：节点自己选过的优先，否则跟随侧栏默认。
     *
     * 底模是**节点级**的（见 GenParams.model）：一张画布上混跑 Qwen 立绘和
     * SDXL 精灵帧是常态，全局一个底模做不到。引擎按 `f"{kind}:{model}"`
     * 分别缓存管线，两套并存只是多占显存。
     */
    const mdl = node.params.model || model
    // 抠图节点不需要提示词（纯图像处理）；文生图/图生图需要
    const needPrompt = node.kind === 'gen' || node.kind === 'img2img' || node.kind === 'inpaint'
      || node.kind === 'airepaint'
    if (needPrompt && !node.params.prompt.trim()) return
    let init: string | undefined
    if (node.kind === 'img2img' || node.kind === 'bg' || node.kind === 'inpaint'
      || node.kind === 'airepaint' || node.kind === 'pixelify' || node.kind === 'poseclone'
      || node.kind === 'spriteswap'
      // **topixel 例外**：源图是可选的。没有源图时后端会用 prompt 先出
      // 一张图再转像素——这才是这条链路的默认用法（用户只写描述）。
      ) {
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
        // 动作参考来自「骨骼编辑」节点时：它导出的就是 OpenPose 骨架图，
        // 必须走 skeleton 模式（auto 会因检不到人体降级成剪影，把骨架毁掉），
        // 并把编辑器的方位角补成方向词，让「换视觉方向」稳定生效。
        const upNode = node.source?.nodeId
          ? nodesRef.current.find((x) => x.id === node.source!.nodeId)
          : undefined
        const fromEditor = upNode?.kind === 'poseedit'
        const pcParams: GenParams = { ...node.params }
        if (fromEditor) {
          pcParams.pcRefMode = 'skeleton'
          pcParams.prompt = [node.params.prompt.trim(),
            `viewing direction: ${viewPhrase(upNode?.skeleton?.azimuth ?? 0)}`]
            .filter(Boolean).join('. ')
        }
        const r = await submitPoseClone(
          pcParams, { model: mdl, matteBackend, scope: currentId }, posePath, charPath)
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
          node.params, { model: mdl, matteBackend, scope: currentId }, sheetPath, charPath)
        jobId = r.job_id
      } else if (node.kind === 'poseswap') {
        // W4 的输入是一组骨架帧（目录），不是单张源图——所以不走 init 那条路
        const dir = node.params.psDir.trim()
        if (!dir) {
          patchNode(id, {
            job: { id: 'error', status: 'failed',
              error: '先选一个骨架帧目录：用 tools/pose_from_image.py 从图片/视频提骨架，'
                + '或在下拉里选已有目录' },
          })
          return
        }
        const r = await submitPoseSeq(node.params, { model: mdl, matteBackend, scope: currentId }, dir)
        jobId = r.job_id
      } else if (node.kind === 'pixelify') {
        const r = await submitPixelify(
          node.params, { model: mdl, matteBackend, scope: currentId }, init!)
        jobId = r.job_id
      } else if (node.kind === 'topixel') {
        // 源图可选：有就直转，没有就让后端用 prompt 出图再转
        const r = await submitToPixel(
          node.params, { model: mdl, matteBackend, scope: currentId }, init)
        jobId = r.job_id
      } else if (node.kind === 'airepaint') {
        const r = await submitAiRepaint(
          node.params, { model: mdl, matteBackend, scope: currentId }, init!, node.params.maskPath)
        jobId = r.job_id
      } else if (node.kind === 'inpaint') {
        const r = await submitInpaint(
          node.params, { model: mdl, matteBackend, scope: currentId }, init!, node.params.maskPath)
        jobId = r.job_id
      } else if (node.kind === 'bg') {
        // 换背景：背景图来自另一个图片节点
        const bgNode = node.params.bgMode === 'image'
          ? nodesRef.current.find((n) => n.id === node.params.bgNodeId)
          : undefined
        const bgPath = bgNode?.image?.path || bgNode?.job?.path
        const r = await submitMatte(
          node.params, { model: mdl, matteBackend, scope: currentId }, init!,
          bgPath, node.params.bgMode === 'color' ? node.params.bgColor : undefined)
        jobId = r.job_id
      } else if (init) {
        const r = await submitImg2Img(node.params, { model: mdl, matteBackend, scope: currentId }, init)
        jobId = r.job_id
      } else {
        const r = await submitTxt2Img(node.params, { model: mdl, matteBackend, scope: currentId })
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
    const box = canvasBox()
    if (list.length === 0) {
      // 侧栏收起/展开宽度不同，空画布的初始平移要跟着走，否则原点又陷进侧栏
      setViewport({ x: box.left + 40, y: 90, z: 1 })
      return
    }
    const minX = Math.min(...list.map((n) => n.x))
    const minY = Math.min(...list.map((n) => n.y))
    const maxX = Math.max(...list.map((n) => n.x + 300))
    const maxY = Math.max(...list.map((n) => n.y + 420))
    const pad = 60
    const z = Math.min(1.2, Math.max(0.3, Math.min(
      (box.width - pad * 2) / (maxX - minX || 1),
      (box.height - 140 - pad * 2) / (maxY - minY || 1),
    )))
    // 水平起点要加侧栏宽度，否则「适应」后节点一半压在侧栏下面
    setViewport({
      z,
      x: box.left + pad - minX * z,
      y: 96 - minY * z,
    })
  }, [])

  const zoomTo = useCallback((z: number) => {
    setViewport((v) => {
      const box = canvasBox()
      const cx = box.left + box.width / 2
      const cy = box.height / 2
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
          matteBackend={matteBackend}
          onMatteBackend={setMatteBackend}
        />
        <Canvas
          nodes={nodes}
          model={model}
          maxCandidates={recommend[model]?.max_candidates}
          poseDirs={poseDirs}
          models={models}
          modelCaps={caps}
          recommend={recommend}
          defaultModel={model}
          onNodeModel={setNodeModel}
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
          onDropType={(t, world) => addNode(t, world)}
          onSpawnImg2Img={spawnImg2Img}
          onExportSkeleton={(id, blob) => void exportSkeleton(id, blob)}
          onImportPose={(id, file) => importPoseFromImage(id, file)}
          onDropFiles={(files, world, targetId) => void handleFiles(files, world, targetId)}
          onConnect={onConnect}
          onDisconnect={onDisconnect}
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
            void handleFiles(files, centerWorld(viewportRef.current))
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

    </div>
  )
}
