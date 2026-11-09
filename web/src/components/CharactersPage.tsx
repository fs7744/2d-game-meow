/**
 * 角色中心 —— 与画布同级的一级页面。
 *
 * 磁盘负责封面 / 精灵帧 / 生成用的属性 config；SQLite（workflows/metadb.py）
 * 是角色元数据的权威存储：标签、创建/编辑时间、自定义标签、备注、任务历史。
 * 文件与模型一律不进数据库。
 *
 * ── 2026-11-07 入口重构 ────────────────────────────────────────────────
 * 默认进入**全屏角色库**（大图 + 贴在图片上的标签 + 名字/tag 搜索、
 * tag 过滤、名字/创建时间/最后编辑时间排序），点卡片才进某个角色的
 * 四步工作台；工作台不再留一条窄角色库侧栏。
 *
 * ── 2026-11-06 四步工作台 ──────────────────────────────────────────────
 * 旧版是「5 个并列 tab + 右栏 4 组重复按钮 + 3 套互相竞争的引导」，
 * 同一个操作有 2~3 个入口、两个「当前服装」状态、四处不同的抠图默认值、
 * 9 条互相独立的小字消息，用户读完所有说明仍不知道第一步点哪里。
 *
 * 现在按主流 AI 游戏素材平台的共识重排（见 characters/model.ts）：
 *   ① 角色是持久对象，所有产物挂在它下面；
 *   ② 一条唯一的 4 步主线，完成状态由「磁盘上真有什么」算出来；
 *   ③ 成果预览常驻右栏，结果优先；
 *   ④ 结构化选择题优先，专业参数收进进阶设置；
 *   ⑤ 每个动作只有一个入口，前置条件、耗时、失败原因写在按钮旁边。
 *
 * 所有英文标识仍按原值传给服务端，中文只在渲染处映射。
 */
import {
  useCallback, useEffect, useMemo, useRef, useState,
  type CSSProperties, type PointerEvent as ReactPointerEvent,
} from 'react'
import {
  type CharacterDetail, type CharacterSummary, type StyleCatalog,
  type EmotionCatalog,
  deleteCharacter, deleteCostume, fetchCharacter,
  fetchCharacters, renameCharacter, saveCostume, updateCharacterInfo,
  fetchStyleCatalog, fetchEmotionCatalog,
  fetchCharacterPrompt, generateForCharacter, fetchJob, fetchCharacterTags,
  detectEmotionFaces, generateEmotions, type EmotionFrame,
  analyzeCharacter, wizardCharacter, designClothes, clothesWizard, uploadImage,
  fetchPortraits, fetchStages, saveStages,
  type PortraitItem, type StageItem, type StagesPayload,
  generatePoses, fetchPoseDirs, exportAtlas, fetchAtlasInfo, designUnderwear,
  runPipeline, type PoseDir, type AtlasInfo,
  type CharacterPrompt, type TagCatalog,
} from '../api'
import CharacterLibrary, { tagLabel } from './CharacterLibrary'
import CharacterPlaytest from './CharacterPlaytest'
import { PartsPanel } from './characters/PartsPanel'
import PromptZoom from './PromptZoom'
import TagPicker from './TagPicker'
import {
  BG_COLOR_LABEL, COSTUME_LABEL, STYLE_GROUP_LABEL, STYLE_LABEL, emotionLabel, zh,
} from '../labels'
import {
  Adv, Blocker, Card, CostumeForm, EmotionGrid, FieldChips, PortraitPane,
  StylePicker,
  type EmotionOption, type StyleOption,
} from './characters/Controls'
import {
  BG_COLORS, INFO_ADV_FIELDS, INFO_FIELDS,
  estimateSeconds, explainError,
  type TaskState,
} from './characters/model'

type Props = {
  onNotice: (msg: string) => void
  /** 全局抠图方式（顶栏 ⚙ 全局设置里改）。角色中心的幕布抠图会跟着它走。 */
  matteBackend?: string
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/** 取景（官方 framing）。**默认全身**：半身图切精灵帧会缺脚。值原样传给后端。 */
const FRAMING_OPTIONS: Array<[string, string]> = [
  ['full_body', '全身（游戏素材默认）'],
  ['cowboy_shot', '半身（到大腿）'],
  ['portrait', '头像 / 胸像'],
]

/** 全局抠图方式 → 勾选项后面那一小截说明（值原样来自 /settings）。 */
const MATTE_SHORT: Record<string, string> = {
  auto: '自动', u2net: '语义分割', color: '纯色键控',
}

function fmtStamp(ts?: number): string {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false })
}

export default function CharactersPage({ onNotice, matteBackend = 'auto' }: Props) {
  const matteMethod = MATTE_SHORT[matteBackend] ?? matteBackend
  /* ------------------------------------------------------------ 角色库 */
  const [items, setItems] = useState<CharacterSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')
  /** library=全屏角色库；stages=生命阶段列表；stage=角色设定；
   *  parts=拆部件（与角色设定同级）；pending=待处理页 */
  const [view, setView] = useState<
    'library' | 'stages' | 'stage' | 'parts' | 'pending'>('library')
  const [sel, setSel] = useState<string | null>(null)
  const [detail, setDetail] = useState<CharacterDetail | null>(null)
  const [styles, setStyles] = useState<StyleCatalog | null>(null)
  const [emotions, setEmotions] = useState<EmotionCatalog | null>(null)
  const [tagCat, setTagCat] = useState<TagCatalog | null>(null)
  const [poseDirs, setPoseDirs] = useState<PoseDir[]>([])
  const [poseDir, setPoseDir] = useState('')
  const [atlasInfo, setAtlasInfo] = useState<AtlasInfo | null>(null)
  /** 试玩验收台调好的帧率，写回图集导出（一个来源，别在三处各填一遍） */
  const [atlasFrameRate, setAtlasFrameRate] = useState(8)
  const [renameText, setRenameText] = useState('')
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

  /* --------------------------------------------------- 当前角色的配置 */
  const [draft, setDraft] = useState<Record<string, unknown>>({})
  const [cPrompt, setCPrompt] = useState<CharacterPrompt | null>(null)
  /** 单一「当前服装」：预览、换装、表情、姿势、流水线全都用它。
   *  旧版 costume / browseCostume 两套状态会让「服装页选 A、素材页跑 B」。 */
  const [costume, setCostume] = useState('Naked')
  /** 全部立绘（右侧「立绘」栏）。立绘在 Portraits/，跟动作帧分开。 */
  const [portraits, setPortraits] = useState<PortraitItem[]>([])

  /* ---------------- 生命阶段 ---------------- */
  const [stageData, setStageData] = useState<StagesPayload | null>(null)
  const [stageId, setStageId] = useState('')
  const [stageNewOpen, setStageNewOpen] = useState(false)
  const [stageNewTitle, setStageNewTitle] = useState('')
  const [stageNewDesc, setStageNewDesc] = useState('')
  const [dragId, setDragId] = useState<string | null>(null)
  const [importFrom, setImportFrom] = useState('')
  const [stageErr, setStageErr] = useState('')
  /** 顶部「修改」弹窗：改名 / 改描述 / 删除角色 / 只读档案 */
  const [editOpen, setEditOpen] = useState(false)
  const [descDraft, setDescDraft] = useState('')
  /** 单一「抠背景」开关，默认关 —— 模型自带透明底，跑分割是用户的选择。
   *  旧版出图 / 表情 / 流水线 / 后端四处默认值各不相同。 */
  const [matte, setMatte] = useState(false)
  /** Prompt Enhancement：**只有用户点按钮才跑**。中文角色出来就是中文描述。 */
  const [enhancing, setEnhancing] = useState(false)
  const doEnhancePrompt = useCallback(async () => {
    if (!sel || enhancing) return
    setEnhancing(true)
    try {
      const p = await fetchCharacterPrompt(sel, '', undefined, true)
      setCPrompt(p)
      onNotice(p.enhanced
        ? '提示词已润色（Qwen3-VL 改写），可以直接出图'
        : '润色没生效，已保留原始提示词')
    } catch (e) {
      setErr(`润色失败：${explainError(e)}`)
    } finally { setEnhancing(false) }
  }, [sel, enhancing, onNotice])

  /* ------------------------------------------------------- 步骤与任务 */
  /** 唯一任务状态，常驻右栏；替代旧版 9 个 xxxMsg + 一闪而过的 toast。 */
  const [task, setTask] = useState<TaskState | null>(null)
  const [doneFlash, setDoneFlash] = useState<{ text: string; at: number } | null>(null)

  /* --------------------------------------------------- 各步的输入草稿 */
  const [picking, setPicking] = useState<string | null>(null)
  const [styleQuery, setStyleQuery] = useState('')
  const [styleGroup, setStyleGroup] = useState('')
  const [styleOpen, setStyleOpen] = useState(false)
  /** 右栏「角色成果」宽度：可拖动，记在本地（不同机器窗口大小差别很大）。 */
  const [sideW, setSideW] = useState<number>(() => {
    const v = Number(localStorage.getItem('meow.cc.sideW'))
    return Number.isFinite(v) && v >= 280 ? v : 420
  })
  const splitRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<{ x: number; w: number } | null>(null)
  const startSideDrag = useCallback((e: ReactPointerEvent) => {
    if (window.innerWidth <= 1020) return        // 窄屏是单栏，别让人拖出个空条
    dragRef.current = { x: e.clientX, w: sideW }
    ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
    document.body.classList.add('is-resizing')
  }, [sideW])
  const moveSideDrag = useCallback((e: ReactPointerEvent) => {
    const st = dragRef.current
    if (!st) return
    const host = splitRef.current?.parentElement
    const total = host?.clientWidth ?? window.innerWidth
    // 右栏宽度 = 指针到容器右边的距离；主区至少留 460px，右栏 280~总宽-500
    const next = Math.round(st.w + (st.x - e.clientX))
    setSideW(Math.max(280, Math.min(next, Math.max(320, total - 460))))
  }, [])
  const endSideDrag = useCallback(() => {
    if (!dragRef.current) return
    dragRef.current = null
    document.body.classList.remove('is-resizing')
    localStorage.setItem('meow.cc.sideW', String(sideW))
  }, [sideW])
  const [wizText, setWizText] = useState('')
  const [anaResult, setAnaResult] = useState<Record<string, unknown> | null>(null)
  /** 「选本地图片反推」的隐藏 file input */
  const analyzeRef = useRef<HTMLInputElement | null>(null)
  const [clothText, setClothText] = useState('')
  const [costumeDraft, setCostumeDraft] = useState<Record<string, string>>({})
  const [newCostume, setNewCostume] = useState('')
  const [emoSel, setEmoSel] = useState<string[]>([])
  const [emoFrames, setEmoFrames] = useState<EmotionFrame[] | null>(null)
  /** 默认 16 步 = 实测质量下限，不再是页面自己都写着「约 27 分钟」的 25 步档。 */
  const [emoParams, setEmoParams] = useState({ target_size: 2048, steps: 16 })
  const [poseText, setPoseText] = useState('')
  const [pipeUnderwear, setPipeUnderwear] = useState(true)
  const [pipeEmotions, setPipeEmotions] = useState<string[]>([])

  const busy = task?.status === 'running'

  const reload = useCallback(async () => {
    try {
      const r = await fetchCharacters()
      setItems(r.items)
      setErr('')
    } catch (e) {
      setErr(`读角色库失败：${explainError(e)}`)
    } finally {
      setLoading(false)
    }
  }, [])

  /** 刷新当前角色的全部服务端事实（配置 / 图集 / 列表），但不覆盖用户手改的 prompt，
   *  也不改当前服装 —— 新建服装后立刻刷新若把选择弹回 Naked，用户会以为没建成功。
   *  服装的合法性只在「切角色」时校验（见下面的 useEffect）。 */
  const refresh = useCallback(async (name: string, stage = stageId) => {
    const d = await fetchCharacter(name)
    setDetail(d)
    setDraft(d.config.character_info ?? {})
    const info = await fetchAtlasInfo(name).catch(() => null)
    setAtlasInfo(info)
    setPortraits(await fetchPortraits(name, stage).catch(() => []))
    await reload()
    return d
  }, [reload, stageId])

  /** 悬浮工具箱新增 / 删除立绘后只刷画廊，不重启整页数据流。 */
  const refreshPortraits = useCallback(async () => {
    if (!sel) return
    setPortraits(await fetchPortraits(sel, stageId).catch(() => []))
  }, [sel, stageId])

  /**
   * 重新拼 prompt（**不跑 PE**）。
   *
   * PE 要跑一次 8B 自回归解码（10~40s），只能是用户点「润色」时才动 ——
   * 以前打开这一步就自动跑，页面像卡住，而且用户没得选。
   */
  const refreshPrompt = useCallback((name: string) => {
    void fetchCharacterPrompt(name, '', undefined, false)
      .then(setCPrompt).catch(() => setCPrompt(null))
  }, [])

  useEffect(() => {
    void reload()
    fetchStyleCatalog().then(setStyles).catch(() => setStyles(null))
    fetchEmotionCatalog().then(setEmotions).catch(() => setEmotions(null))
    fetchCharacterTags().then(setTagCat).catch(() => setTagCat(null))
    fetchPoseDirs().then((list) => {
      setPoseDirs(list)
      setPoseDir((cur) => cur || (list[0]?.path ?? ''))
    }).catch(() => setPoseDirs([]))
  }, [reload])

  useEffect(() => {
    if (!sel) {
      setDetail(null); setCPrompt(null); setDraft({}); setAtlasInfo(null)
      setPortraits([])
      return
    }
    let alive = true
    setTask(null); setDoneFlash(null); setAnaResult(null)
    setEmoFrames(null); setEmoSel([])
    void fetchCharacter(sel)
      .then((d) => {
        if (!alive) return
        setDetail(d); setDraft(d.config.character_info ?? {})
        setCostume((cur) => d.costumes.includes(cur) ? cur : (d.costumes[0] ?? 'Naked'))
      })
      .catch((e) => { if (alive) setErr(explainError(e)) })
    refreshPrompt(sel)
    void fetchAtlasInfo(sel).then((i) => { if (alive) setAtlasInfo(i) }).catch(() => {})
    void fetchPortraits(sel, stageId).then((p) => { if (alive) setPortraits(p) }).catch(() => {})
    return () => { alive = false }
  }, [sel, stageId, refreshPrompt])

  /** 阶段列表只在换角色时拉一次；排序 / 改名由本地改完后整包 PUT。 */
  useEffect(() => {
    if (!sel) { setStageData(null); setStageId(''); return }
    let alive = true
    void fetchStages(sel)
      .then((d) => { if (alive) { setStageData(d); setStageId(d.active_stage) } })
      .catch((e) => { if (alive) setStageErr(explainError(e)) })
    return () => { alive = false }
  }, [sel])

  // 服装字段草稿：换角色 / 换服装时从 config 载入（AI 拆服装要能直接灌进来）
  useEffect(() => {
    setCostumeDraft({ ...(detail?.config.costumes?.[costume] ?? {}) })
  }, [detail, costume])

  /* ------------------------------------------------------------ 派生 */
  const flatStyles = useMemo<StyleOption[]>(() => {
    const out: StyleOption[] = []
    for (const g of styles?.groups ?? []) {
      const groupLabel = g.label || g.name || ''
      for (const s of g.styles ?? []) {
        out.push({
          id: s.id, label: s.label, group: groupLabel,
          zhLabel: zh(STYLE_LABEL, s.label), zhGroup: zh(STYLE_GROUP_LABEL, groupLabel),
        })
      }
    }
    return out
  }, [styles])
  const styleGroups = useMemo(
    () => styles?.groups.map((g) => g.label || g.name || '') ?? [], [styles])

  const flatEmotions = useMemo<EmotionOption[]>(() => {
    const out: EmotionOption[] = []
    for (const [, list] of Object.entries(emotions ?? {})) {
      for (const e of Array.isArray(list) ? list : []) {
        out.push({
          key: e.key, safe_name: e.safe_name,
          label: emotionLabel(e.safe_name, e.key), prompt: e.natural_prompt,
        })
      }
    }
    return out
  }, [emotions])

  const currentStyle = String(draft.style ?? '')

  /** 步骤完成标记只认磁盘上的真实产物。 */
  const assetStats = useMemo(() => {
    const byCostume = detail?.assets ?? {}
    let frames = 0, groups = 0
    for (const byEmotion of Object.values(byCostume)) {
      const emos = Object.values(byEmotion ?? {})
      groups += emos.length
      for (const list of emos) frames += Array.isArray(list) ? list.length : 0
    }
    return { frames, groups }
  }, [detail])

  /** 「有立绘」= 能拿它当底图（换装 / 出姿势 / 白内衣起稿）。 */
  const hasPortrait = portraits.length > 0
  /** 「有动作帧」= 表情叠加 / 导出图集 / 试玩台才有东西可看。立绘不算动作帧。 */
  const hasFrames = assetStats.frames > 0
  const hasSprite = hasPortrait || hasFrames
  const poseFrameCount = poseDirs.find((d) => d.path === poseDir)?.count ?? 4
  const pipelineEstimate = estimateSeconds(poseFrameCount, pipeEmotions.length)

  /* ==================================================================
     统一任务执行器
     ------------------------------------------------------------------
     以前 7 个生成动作各写一遍轮询循环：间隔 800/1000/1200ms 各不相同、
     上限 400~3600 次各不相同、失败文案各不相同，超时还统一把用户赶去画布。
     现在一个 runTask 覆盖全部：互斥、统一进度、统一错误话术、统一完成反馈。
     ================================================================== */
  const runTask = useCallback(async (
    label: string,
    submit: () => Promise<unknown>,
    onDone?: (job: unknown, result: unknown) => void,
  ) => {
    if (!sel || busy) return
    const startedAt = Date.now()
    setDoneFlash(null)
    setTask({ label, status: 'running', message: '已提交，排队中…', startedAt })
    try {
      const result = await submit()
      const jobId = (result as { job_id?: string })?.job_id
      if (!jobId) throw new Error('服务端没返回任务号')
      for (let i = 0; i < 3600; i++) {
        const j = await fetchJob(jobId)
        if (j.status === 'done') {
          await refresh(sel)
          const secs = Math.round((Date.now() - startedAt) / 1000)
          setTask({ label, status: 'done', message: `完成，用时 ${secs} 秒`, startedAt })
          setDoneFlash({ text: `${label}完成，成果已更新`, at: Date.now() })
          onNotice(`${label}完成`)
          // onDone 放在统一成功消息之后：个别任务（表情批量）要补「N 张失败」的话术
          onDone?.(j, result)
          // 回到自动推导的步骤：刚完成的这一步会自然把主线推进一格
          return
        }
        if (j.status === 'failed') {
          setTask({ label, status: 'failed',
            message: explainError(j.error || '任务失败'), startedAt })
          return
        }
        setTask({
          label, status: 'running', startedAt,
          message: j.progress?.current || '正在生成…',
          // 一键出全套会带 step（underwear|poses|emotions），手工展开 progress
          // 时必须一起带上，否则 Controls 那边无从显示当前阶段。
          progress: j.progress
            ? { done: j.progress.done, total: j.progress.total,
                current: j.progress.current, step: j.progress.step }
            : undefined,
        })
        await sleep(1000)
      }
      setTask({ label, status: 'failed', startedAt,
        message: '任务还在后台跑，产物会自己进角色目录；可以稍后刷新查看' })
    } catch (e) {
      setTask({ label, status: 'failed', message: explainError(e), startedAt })
    }
  }, [sel, busy, refresh, onNotice])

  /** 本地 LLM 类的即时动作不走任务队列，但也用同一条状态条反馈。 */
  const runInline = useCallback(async (label: string, fn: () => Promise<string>) => {
    if (busy) return
    const startedAt = Date.now()
    setTask({ label, status: 'running', message: '正在处理（本地模型，约 20~60 秒）…', startedAt })
    try {
      const message = await fn()
      setTask({ label, status: 'done', message, startedAt })
    } catch (e) {
      setTask({ label, status: 'failed', message: explainError(e), startedAt })
    }
  }, [busy])

  /* ------------------------------------------------------- 视图切换 */
  /** 创建动作只在全屏角色库里发生；这里只负责「选中谁、进不进工作台」。 */
  const openWorkspace = useCallback((name: string) => {
    setSel(name)
    setView('stages')
  }, [])

  const backToLibrary = useCallback(() => {
    setView('library')
    setSel(null)
    setTask(null)
    setDoneFlash(null)
    setStageData(null); setStageId(''); setStageErr('')
  }, [])

  /** 整包保存阶段数组（数组顺序 = 展示顺序），顺带同步激活阶段。 */
  const persistStages = useCallback(async (next: StageItem[], active?: string) => {
    if (!sel || !stageData) return
    try {
      const r = await saveStages(sel, {
        stages: next,
        active_stage: active ?? stageData.active_stage,
        description: stageData.description,
      })
      setStageData(r); setStageErr('')
    } catch (e) { setStageErr(explainError(e)) }
  }, [sel, stageData])

  /** 进入阶段编辑：先设为激活阶段，后端会把它的设定写回顶层
   *  `character_info`，现有全部出图链路自动跟着这个阶段走。 */
  const openStage = useCallback(async (id: string) => {
    if (!sel || !stageData) return
    try {
      const r = await saveStages(sel, {
        stages: stageData.stages, active_stage: id,
        description: stageData.description,
      })
      setStageData(r); setStageId(id); setView('stage'); setStageErr('')
      await refresh(sel, id)
      refreshPrompt(sel)
    } catch (e) { setStageErr(explainError(e)) }
  }, [sel, stageData, refresh, refreshPrompt])

  const addStage = useCallback(async () => {
    if (!stageData) return
    const base = stageData.stages.find((x) => x.id === stageData.active_stage)
    const item: StageItem = {
      id: `stage_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
      title: stageNewTitle.trim() || `生命阶段 ${stageData.stages.length + 1}`,
      description: stageNewDesc.trim(),
      // 从当前阶段设定起步；进编辑页还能用「导入设定」换成别的阶段
      info: { ...(base?.info || {}) },
      portrait_count: 0, portraits: [],
    }
    await persistStages([...stageData.stages, item])
    setStageNewOpen(false); setStageNewTitle(''); setStageNewDesc('')
  }, [stageData, stageNewTitle, stageNewDesc, persistStages])

  const patchStage = useCallback((id: string, patch: Partial<StageItem>) => {
    setStageData((d) => d ? {
      ...d, stages: d.stages.map((x) => x.id === id ? { ...x, ...patch } : x),
    } : d)
  }, [])

  const commitStage = useCallback(async () => {
    if (stageData) await persistStages(stageData.stages)
  }, [stageData, persistStages])

  const removeStage = useCallback(async (id: string) => {
    if (!stageData) return
    if (stageData.stages.length <= 1) {
      setStageErr('至少要保留一个生命阶段'); return
    }
    const next = stageData.stages.filter((x) => x.id !== id)
    const active = stageData.active_stage === id ? next[0].id : stageData.active_stage
    await persistStages(next, active)
    if (stageId === id) setView('stages')
  }, [stageData, stageId, persistStages])

  /** 拖拽手柄落点：把源项挪到目标项位置，整包提交新顺序。 */
  const dropStage = useCallback(async (targetId: string) => {
    if (!stageData || !dragId || dragId === targetId) return
    const arr = [...stageData.stages]
    const from = arr.findIndex((x) => x.id === dragId)
    const to = arr.findIndex((x) => x.id === targetId)
    if (from < 0 || to < 0) return
    const [moved] = arr.splice(from, 1)
    arr.splice(to, 0, moved)
    await persistStages(arr)
    setDragId(null)
  }, [stageData, dragId, persistStages])

  /** 从别的阶段导入角色设定：**只复制 info，绝不碰立绘**。 */
  const importSettings = useCallback(async () => {
    if (!sel || !stageData || !stageId || !importFrom) return
    const src = stageData.stages.find((x) => x.id === importFrom)
    if (!src) return
    const next = stageData.stages.map((x) => x.id === stageId
      ? { ...x, info: { ...src.info } } : x)
    await persistStages(next, stageId)
    await refresh(sel, stageId)
    refreshPrompt(sel)
    onNotice(`已导入「${src.title}」的设定（立绘不导入）`)
    setImportFrom('')
  }, [sel, stageData, stageId, importFrom, persistStages, refresh, refreshPrompt, onNotice])

  /** 顶部「修改」：角色名与描述是角色级信息，不属于任何生命阶段。 */
  const openEdit = useCallback(() => {
    setRenameText(detail?.name || sel || '')
    setDescDraft(stageData?.description || '')
    setPendingDelete(null)
    setEditOpen(true)
  }, [detail, sel, stageData])

  const doRename = useCallback(async () => {
    if (!sel || !renameText.trim() || renameText.trim() === sel) return
    try {
      await renameCharacter(sel, renameText.trim())
      setEditOpen(false)
      await reload()
      setSel(renameText.trim())
    } catch (e) { setErr(`改名失败：${explainError(e)}`) }
  }, [sel, renameText, reload])

  const saveEdit = useCallback(async () => {
    if (!sel) return
    try {
      if (renameText.trim() && renameText.trim() !== sel) await doRename()
      if (stageData && descDraft.trim() !== stageData.description) {
        const r = await saveStages(sel, {
          stages: stageData.stages,
          active_stage: stageData.active_stage,
          description: descDraft.trim(),
        })
        setStageData(r)
      }
      setEditOpen(false)
      onNotice('角色信息已更新')
    } catch (e) { setStageErr(explainError(e)) }
  }, [sel, renameText, doRename, stageData, descDraft, onNotice])

  const doDelete = useCallback(async (name: string) => {
    try {
      await deleteCharacter(name)
      setPendingDelete(null)
      if (sel === name) backToLibrary()
      await reload()
    } catch (e) { setErr(`删除失败：${explainError(e)}`) }
  }, [sel, reload, backToLibrary])

  const saveInfo = useCallback(async (patch: Record<string, unknown>) => {
    if (!sel) return
    try {
      await updateCharacterInfo(sel, patch)
      const d = await fetchCharacter(sel)
      setDetail(d); setDraft(d.config.character_info ?? {})
      refreshPrompt(sel)
    } catch (e) { setErr(`保存失败：${explainError(e)}`) }
  }, [sel, refreshPrompt])

  /* --------------------------------------------------- 阶段编辑：角色设定 */
  const doWizard = useCallback(async () => {
    const text = wizText.trim()
    if (!sel || !text || busy) return
    await runInline('一句话生成设定', async () => {
      const r = await wizardCharacter(text)
      setAnaResult(r.fields)
      return `读好了（${r.seconds}s）——下面是它拆出来的设定，确认后再写进角色`
    })
  }, [sel, wizText, busy, runInline])

  /**
   * 从图片反推属性。
   *
   * `file` 传了就先上传这张本地图，用它反推；不传则用角色最新的一张立绘
   * （后端默认行为）。**两种都要支持**：用户可能拿别人的设定图来抄属性，
   * 也可能只是想重读自己刚出的图。
   */
  const doAnalyze = useCallback(async (file?: File) => {
    if (!sel || busy) return
    await runInline(file ? `从「${file.name}」反推属性` : '从图片反推属性', async () => {
      let image = ''
      if (file) {
        const up = await uploadImage(file)
        image = up.path
      }
      const r = await analyzeCharacter(sel, {
        background_color: String(draft.background_color ?? ''),
        ...(image ? { image } : {}),
      })
      setAnaResult(r.fields)
      return `读完了（${r.seconds}s）——确认无误再点「用这些属性」`
    })
  }, [sel, busy, draft, runInline])

  const applyAnaResult = useCallback(() => {
    if (!anaResult) return
    const patch: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(anaResult)) {
      if (v === '' || v === null || v === undefined) continue
      patch[k] = v
    }
    setDraft((d) => ({ ...d, ...patch }))
    setAnaResult(null)
    void saveInfo(patch)
  }, [anaResult, saveInfo])

  /** 生成立绘（w1）/ 基于最新立绘再画一张（w2）。 */
  const doGenerate = useCallback((workflow: 'w1' | 'w2') => {
    if (!sel) return
    void runTask(workflow === 'w1' ? '生成立绘' : '再画一张立绘',
      () => generateForCharacter(sel, {
        workflow, costume, emotion: 'neutral',
        prompt: cPrompt?.prompt || '', negative: cPrompt?.negative || '',
        // 界面上这段 prompt 就是最终要喂的（润色过就是润色版）；
        // 再让后端跑一次 PE 等于把已经改好的文字再改一遍，纯浪费 40 秒。
        enhance: false, matte,
      }),
      () => {})
  }, [sel, runTask, costume, cPrompt, matte])

  /* --------------------------------------------------- 待处理：出素材（内容保留） */
  const doCreateCostume = useCallback(async () => {
    const name = newCostume.trim()
    if (!sel || !name || busy) return
    try {
      // 新建即落盘：旧版只改本地 state，切个角色就悄悄消失。
      await saveCostume(sel, name, {})
      await refresh(sel)
      setCostume(name); setNewCostume('')
      onNotice(`服装「${name}」已建好`)
    } catch (e) { setErr(`建服装失败：${explainError(e)}`) }
  }, [sel, newCostume, busy, refresh, onNotice])

  const doDeleteCostume = useCallback(async () => {
    if (!sel || costume === 'Naked') return
    try {
      await deleteCostume(sel, costume)
      setCostume('Naked')
      await refresh(sel)
    } catch (e) { setErr(`删服装失败：${explainError(e)}`) }
  }, [sel, costume, refresh])

  const saveCostumeFields = useCallback(async (data: Record<string, string>) => {
    if (!sel) return
    try {
      await saveCostume(sel, costume, data)
      await refresh(sel)
      onNotice(`服装「${zh(COSTUME_LABEL, costume)}」已保存`)
    } catch (e) { setErr(`保存服装失败：${explainError(e)}`) }
  }, [sel, costume, refresh, onNotice])

  const doClothesWizard = useCallback(async () => {
    const text = clothText.trim()
    if (!sel || !text || busy) return
    await runInline('AI 拆服装', async () => {
      const r = await clothesWizard(sel, text)
      setCostumeDraft((d) => ({ ...d, ...(r.fields as Record<string, string>) }))
      return `拆好了（${r.seconds}s）——检查下面的字段，确认后点「保存服装」`
    })
  }, [sel, clothText, busy, runInline])

  const doDesignClothes = useCallback(() => {
    if (!sel) return
    void runTask('生成换装立绘', async () => {
      await saveCostume(sel, costume, costumeDraft)
      return designClothes(sel, { costume, matte })
    })
  }, [sel, runTask, costume, costumeDraft, matte])

  const doUnderwear = useCallback(() => {
    if (!sel) return
    void runTask('生成干净起稿', () => designUnderwear(sel, {}), (_j, r) => {
      const c = (r as { costume?: string })?.costume
      if (c) setCostume(c)
    })
  }, [sel, runTask])

  const doDetect = useCallback(async () => {
    if (!sel || busy) return
    await runInline('检出脸框', async () => {
      const r = await detectEmotionFaces(sel, { costume })
      setEmoFrames(r.items)
      const hit = r.items.filter((f) => f.detected).length
      if (r.count === 0) return '没有动作帧：表情是在动作帧上只改脸，先在待处理页出素材'
      return `${r.count} 张源帧，${hit} 张检出脸框`
        + (hit < r.count ? '（没检出的会整图重画）' : '')
    })
  }, [sel, busy, costume, runInline])

  const doGenerateEmotions = useCallback(() => {
    if (!sel || emoSel.length === 0) return
    void runTask('生成表情', () => generateEmotions(sel, {
      costume, emotions: emoSel,
      steps: emoParams.steps, target_size: emoParams.target_size, matte,
    }), (j) => {
      const res = ((j as { result?: { count?: number; failed?: { error: string }[] } })
        .result || {})
      if (res.failed?.length) {
        setTask((t) => t ? { ...t, message: `完成 ${res.count ?? 0} 张，`
          + `${res.failed!.length} 张失败：${res.failed![0]?.error || ''}` } : t)
      }
    })
  }, [sel, runTask, costume, emoSel, emoParams, matte])

  const doGeneratePoses = useCallback(() => {
    if (!sel || !poseDir) return
    void runTask('按姿势批量生成', () => generatePoses(sel, {
      costume, pose_dir: poseDir, prompts: poseText.trim(), matte,
    }))
  }, [sel, runTask, costume, poseDir, poseText, matte])

  /** 一键出全套：内衣 → 姿势 → 表情，顺序由后端固定，界面不再让用户记。 */
  const doRunPipeline = useCallback(() => {
    if (!sel || !poseDir) return
    void runTask('一键出全套', () => runPipeline(sel, {
      costume, underwear: pipeUnderwear, pose_dir: poseDir,
      emotions: pipeEmotions, matte,
    }))
  }, [sel, runTask, costume, pipeUnderwear, poseDir, pipeEmotions, matte])

  /* ----------------------------------------------------- 待处理：导出（内容保留） */
  const doExportAtlas = useCallback(() => {
    if (!sel) return
    void runTask('导出图集', () => exportAtlas(sel, { frame_rate: atlasFrameRate }))
  }, [sel, runTask, atlasFrameRate])

  /* ================================================================== 渲染 */
  const currentSummary = items.find((c) => c.name === sel)
  const stageList = stageData?.stages ?? []
  const currentStage = stageList.find((x) => x.id === stageId)

  /* 默认首页：整个页面就是角色库。工作台只是「点开某个角色」之后的第二层。 */
  if (view === 'library') {
    return (
      <CharacterLibrary
        items={items} loading={loading} err={err}
        onOpen={openWorkspace}
        onCreated={async (name) => { await reload(); openWorkspace(name) }}
        onReload={reload}
        onDelete={(name) => void doDelete(name)}
        onNotice={onNotice} />
    )
  }

  return (
    <div className={`chars cc is-workspace${view === 'stages' ? ' is-stages' : ''}`
      + `${view === 'parts' ? ' is-parts' : ''}`}
      style={{ '--cc-side-w': `${sideW}px` } as CSSProperties}>
      <main className="chars-detail cc-main">
        {/* 顶部只有角色名 + 更小的描述；最右是「修改」与编辑时间。
            服装 / 画风属于角色设定，不再出现在这一层。 */}
        <div className="cc-topbar">
          <button type="button" className="cc-btn ghost" onClick={backToLibrary}>← 角色库</button>
          <div className="cc-topbar-id">
            <b className="cc-topbar-name">{detail?.name || sel || ''}</b>
            <span className="cc-topbar-desc">{stageData?.description || '还没有角色描述'}</span>
          </div>
          <span className="spacer" />
          <span className="cc-topbar-meta">最后编辑 {fmtStamp(currentSummary?.updated_at)}</span>
          <button type="button" className="cc-btn ghost" onClick={openEdit}>修改</button>
        </div>

        {/* 面包屑只在进入某个生命阶段（或待处理页）后出现 */}
        {view !== 'stages' && (
          <div className="cc-crumb">
            <button type="button" className="cc-crumb-link"
              onClick={() => setView('stages')}>{detail?.name || sel || ''}</button>
            <span className="cc-crumb-sep">/</span>
            <b>{view === 'stage' ? (currentStage?.title || '角色设定')
              : view === 'parts' ? '拆部件' : '待处理'}</b>
          </div>
        )}
        {err && <div className="chars-err">⚠ {err}</div>}
        {stageErr && <div className="chars-err">⚠ {stageErr}</div>}

        {/* 只有两个入口。进入某个阶段后，第一项就是**当前所在的「角色设定」**，
            名字只在这里出现一次，正文不再重复一个大标题。列表页仍叫「生命阶段」，
            点它（或面包屑里的角色名 / 底部返回按钮）回到阶段列表。 */}
        <div className="cc-viewnav">
          <button type="button"
            className={(view === 'stages' || view === 'stage') ? 'is-on' : ''}
            onClick={() => { if (view !== 'stage') setView('stages') }}
            title={view === 'stage' ? '当前在角色设定；返回列表请点上方角色名'
              : '生命阶段列表'}>
            {view === 'stage' ? '角色设定' : '生命阶段'}
          </button>
          <button type="button" className={view === 'parts' ? 'is-on' : ''}
            disabled={!stageId}
            onClick={() => stageId && setView('parts')}>拆部件</button>
          <button type="button" className={view === 'pending' ? 'is-on' : ''}
            onClick={() => setView('pending')}>待处理 · 出素材 / 导出</button>
        </div>

        <div className="chars-body cc-panel">
          {/* ============================ 生命阶段列表 ============================ */}
          {view === 'stages' && (
            <>
              <div className="cc-stages-head">
                <div>
                  <h2>生命阶段</h2>
                  <p>一个角色可以有多套设定（幼年 / 成年 / 战损…）。每行是一个阶段，
                    拖拽 ⠿ 调整顺序，点「进入设定」编辑它的角色设定与立绘。</p>
                </div>
                <button type="button" className="cc-btn primary"
                  onClick={() => setStageNewOpen(true)}>＋ 新建阶段</button>
              </div>
              {stageNewOpen && (
                <div className="cc-stage-new">
                  <label>阶段名称
                    <input value={stageNewTitle} placeholder="例：成年 / 觉醒"
                      onChange={(e) => setStageNewTitle(e.target.value)} />
                  </label>
                  <label>阶段描述（可选）
                    <textarea rows={2} value={stageNewDesc}
                      onChange={(e) => setStageNewDesc(e.target.value)} />
                  </label>
                  <div className="cc-actions">
                    <button type="button" className="cc-btn primary"
                      onClick={() => void addStage()}>创建阶段</button>
                    <button type="button" className="cc-btn ghost"
                      onClick={() => setStageNewOpen(false)}>取消</button>
                  </div>
                </div>
              )}
              <div className="cc-stage-list">
                {stageList.map((st, i) => (
                  <div className="cc-stage-row" key={st.id}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={() => void dropStage(st.id)}>
                    <span className="cc-stage-drag" draggable
                      onDragStart={() => setDragId(st.id)}
                      onDragEnd={() => setDragId(null)} title="拖拽排序">⠿</span>
                    {/* 这一行以立绘为主：缩略图放大铺开，文字退到紧凑的一列 */}
                    <div className="cc-stage-thumbs">
                      {st.portraits.slice(0, 5).map((img) => (
                        <img key={img.url} src={img.url} alt={img.name} loading="lazy" />
                      ))}
                      {!st.portraits.length && (
                        <span className="cc-stage-nopreview">暂无立绘</span>
                      )}
                    </div>
                    <div className="cc-stage-info">
                      <div className="cc-stage-line">
                        <span className="cc-stage-idx">{i + 1}</span>
                        <input className="cc-stage-title" value={st.title}
                          aria-label="阶段名称"
                          onChange={(e) => patchStage(st.id, { title: e.target.value })}
                          onBlur={() => void commitStage()} />
                        {stageData?.active_stage === st.id && (
                          <span className="cc-stage-active">当前</span>
                        )}
                      </div>
                      <input className="cc-stage-desc" value={st.description}
                        aria-label="阶段描述" placeholder="补一句描述（可选）"
                        onChange={(e) => patchStage(st.id, { description: e.target.value })}
                        onBlur={() => void commitStage()} />
                      <span className="cc-stage-count">{st.portrait_count} 张立绘</span>
                    </div>
                    <div className="cc-stage-ops">
                      <button type="button" className="cc-btn primary"
                        onClick={() => void openStage(st.id)}>进入设定 →</button>
                      <button type="button" className="cc-btn ghost"
                        onClick={() => void removeStage(st.id)}>删除</button>
                    </div>
                  </div>
                ))}
                {!stageList.length && (
                  <div className="cc-preview-empty">
                    <p>还没有生命阶段，点右上角「＋ 新建阶段」开始。</p>
                  </div>
                )}
              </div>
              {/* 生命阶段列表页只放阶段本身：待处理在导航里，角色档案在「修改」弹窗里 */}
            </>
          )}

          {/* ===== 拆部件：选图 → 选拆法 → 分层 PSD + 预览（与角色设定同级） ===== */}
          {view === 'parts' && sel && detail && (
            <PartsPanel character={detail.name} stage={stageId}
              portraits={portraits} task={task} runTask={runTask} />
          )}

          {/* ===== 阶段编辑：角色设定（原「定形象」内容，用户要求改名） ===== */}
          {view === 'stage' && sel && detail && (
            <>
              <div className="cc-stagehead">
                <div>
                  {/* 「角色设定」这个名字已经在导航里了，这里不再重复一个大标题 */}
                  <p>阶段「{currentStage?.title || ''}」的外貌、画风与提示词；立绘在右栏。</p>
                </div>
                <div className="cc-import">
                  <select value={importFrom} aria-label="选择要导入的阶段"
                    onChange={(e) => setImportFrom(e.target.value)}>
                    <option value="">从其它阶段导入设定…</option>
                    {stageList.filter((x) => x.id !== stageId).map((x) => (
                      <option key={x.id} value={x.id}>{x.title}</option>
                    ))}
                  </select>
                  <button type="button" className="cc-btn" disabled={!importFrom}
                    onClick={() => void importSettings()}>导入</button>
                  <span className="cc-import-note">只复制角色设定，不导入立绘</span>
                </div>
              </div>
              <Card title="不想一项项选？直接说一句话" note="AI 会拆成结构化设定，先给你看，确认后才写进角色。">
                <div className="cc-row">
                  <input className="chars-wizard-input" value={wizText}
                    placeholder="例：一个苍白的精灵女剑士，银发红瞳，身材纤细"
                    onChange={(e) => setWizText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') void doWizard() }} />
                  <button type="button" className="cc-btn" onClick={() => void doWizard()}
                    disabled={busy || !wizText.trim()}>一句话生成</button>
                  <button type="button" className="cc-btn ghost" onClick={() => void doAnalyze()}
                    disabled={busy}>从图片反推属性</button>
                  {/* 图源自己选：当前立绘（上面那个按钮）或本地任意一张 */}
                  <button type="button" className="cc-btn ghost"
                    onClick={() => analyzeRef.current?.click()}
                    disabled={busy}>选本地图片反推…</button>
                  <input ref={analyzeRef} type="file" accept="image/*"
                    className="lora-file-input"
                    onChange={(e) => {
                      const f = e.target.files?.[0]
                      e.target.value = ''            // 允许重复选同一张
                      if (f) void doAnalyze(f)
                    }} />
                </div>
                <p className="chars-sub">
                  「从图片反推属性」读的是当前最新立绘；想用别的图（参考设定图、别处存的角色图）
                  就点「选本地图片反推…」。
                </p>
                {anaResult && (
                  <div className="chars-analyze">
                    <table><tbody>
                      {Object.entries(anaResult).map(([k, v]) => (
                        <tr key={k}><th>{k}</th><td>{String(v)}</td></tr>
                      ))}
                    </tbody></table>
                    <div className="cc-row">
                      <button type="button" className="cc-btn" onClick={applyAnaResult}>用这些属性（覆盖同名项）</button>
                      <button type="button" className="cc-btn ghost" onClick={() => setAnaResult(null)}>放弃</button>
                    </div>
                  </div>
                )}
              </Card>

              <Card title="外貌与基础设定" note="点「选」从标签库里挑，不必自己写英文；标签值会原样写进角色档案。">
                <div className="cc-row">
                  <label className="cc-inline-field">性别
                    <select value={String(draft.sex ?? 'female')}
                      onChange={(e) => {
                        setDraft((d) => ({ ...d, sex: e.target.value }))
                        void saveInfo({ sex: e.target.value })
                      }}>
                      <option value="female">女</option><option value="male">男</option>
                    </select>
                  </label>
                  <label className="cc-inline-field">年龄
                    <input type="number" value={Number(draft.age ?? 18)}
                      onChange={(e) => setDraft((d) => ({ ...d, age: Number(e.target.value) }))}
                      onBlur={(e) => void saveInfo({ age: Number(e.target.value) })} />
                  </label>
                  <label className="cc-inline-field">出图底色
                    <select value={String(draft.background_color ?? 'green')}
                      onChange={(e) => {
                        setDraft((d) => ({ ...d, background_color: e.target.value }))
                        void saveInfo({ background_color: e.target.value })
                      }}>
                      {BG_COLORS.map((c) => (
                        <option key={c} value={c}>{zh(BG_COLOR_LABEL, c)}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="cc-fields">
                  {INFO_FIELDS.filter(([k]) => !INFO_ADV_FIELDS.some(([a]) => a === k)).map(([k, label]) => (
                    <FieldChips key={k} label={label} value={String(draft[k] ?? '')}
                      onPick={() => setPicking(k)} />
                  ))}
                </div>
              </Card>

              <Card title="画风" note="画风决定整张图的笔触与配色，选定后同一角色所有素材都会跟随。">
                <StylePicker
                  options={flatStyles} groups={styleGroups} current={currentStyle}
                  query={styleQuery} group={styleGroup} open={styleOpen}
                  onQuery={setStyleQuery} onGroup={setStyleGroup}
                  onToggle={() => setStyleOpen((v) => !v)}
                  onPick={(id) => {
                    setDraft((d) => ({ ...d, style: id }))
                    void saveInfo({ style: id })
                  }} />
              </Card>

              <Adv title="进阶：提示词与生成参数">
                <p className="chars-sub">
                  下面是按角色档案拼好的结构化提示词（每个字段一行）。
                  改了就用你这份出图；不确定就别动。
                </p>
                {/* 取景 / 语言：以前取景只跟官方默认（半身到大腿），
                    游戏素材要的是全身；语言默认英文，但 Qwen-Image 中文原生支持。 */}
                <div className="cc-row">
                  <label className="cc-inline-field">取景
                    <select value={String(draft.framing || 'full_body')}
                      onChange={(e) => {
                        setDraft((d) => ({ ...d, framing: e.target.value }))
                        void saveInfo({ framing: e.target.value })
                      }}>
                      {FRAMING_OPTIONS.map(([v, label]) => (
                        <option key={v} value={v}>{label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="cc-inline-field">提示词语言
                    <select value={String(draft.prompt_language || 'zh')}
                      onChange={(e) => {
                        setDraft((d) => ({ ...d, prompt_language: e.target.value }))
                        void saveInfo({ prompt_language: e.target.value })
                      }}>
                      <option value="zh">中文（默认）</option>
                      <option value="en">英文（官方模板）</option>
                    </select>
                  </label>
                  <span className="chars-sub">
                    取景选全身，切精灵图才不会缺脚。中文模式连润色也出中文，
                    直接喂 Qwen-Image（它的文本编码器就是 Qwen3-VL）。
                  </span>
                </div>
                <div className="cc-row">
                  <span className="chars-sub">
                    分辨率 {cPrompt ? `${cPrompt.resolution[0]}×${cPrompt.resolution[1]}` : '—'}
                    · {cPrompt ? `${cPrompt.defaults.steps} 步 / cfg ${cPrompt.defaults.cfg}` : ''}
                  </span>
                </div>
                <label className="chars-field">正向 prompt
                  <textarea className="chars-prompt" value={cPrompt?.prompt ?? ''}
                    onChange={(e) => setCPrompt((p) => (p ? { ...p, prompt: e.target.value } : p))} />
                  <PromptZoom value={cPrompt?.prompt ?? ''} label="正向 prompt" disabled={!cPrompt}
                    hint="中英文都行：Qwen-Image 的文本编码器原生支持中文。改完直接出图，不用重拼。"
                    onChange={(v) => setCPrompt((p) => (p ? { ...p, prompt: v } : p))} />
                </label>
                <label className="chars-field">负向词
                  <textarea className="chars-prompt short" value={cPrompt?.negative ?? ''}
                    onChange={(e) => setCPrompt((p) => (p ? { ...p, negative: e.target.value } : p))} />
                  <PromptZoom value={cPrompt?.negative ?? ''} label="负向词" disabled={!cPrompt}
                    hint="不想出现的内容，一行一个。"
                    onChange={(v) => setCPrompt((p) => (p ? { ...p, negative: v } : p))} />
                </label>
                <div className="cc-row">
                  <button type="button" className="cc-btn"
                    onClick={() => void doEnhancePrompt()}
                    disabled={busy || enhancing || !sel}>
                    {enhancing ? '润色中…（10~40 秒）' : '✨ 润色提示词'}
                  </button>
                  <button type="button" className="cc-btn ghost"
                    onClick={() => sel && refreshPrompt(sel)} disabled={busy || enhancing}>
                    还原结构化原文
                  </button>
                  <span className="chars-sub">
                    {cPrompt?.enhanced
                      ? '当前是润色后的版本（再点一次会用原文重写）'
                      : '当前是结构化原文；润色只在点按钮时跑，用图像模型自带的 Qwen3-VL 改写'}
                  </span>
                </div>
                <label className="chars-check">
                  <input type="checkbox" checked={matte} onChange={(e) => setMatte(e.target.checked)} />
                  额外抠一次纯色背景
                  <span className="cc-inline-tag" title="在顶栏 ⚙ 全局设置里改">
                    方式：{matteMethod}
                  </span>
                  <span className="chars-sub">
                    默认关：模型本来就出透明底。只有底色不干净时才勾，
                    白毛 / 白甲角色勾了可能被一起判成背景。这个开关对后面所有生成都生效；
                    抠图算法跟全局设置走（顶栏 ⚙ 设置）。
                  </span>
                </label>
                <div className="cc-fields">
                  {INFO_ADV_FIELDS.map(([k, label, hint]) => (
                    <label className="chars-field" key={k} title={hint}>
                      {label}
                      <input value={String(draft[k] ?? '')} placeholder={hint}
                        onChange={(e) => setDraft((d) => ({ ...d, [k]: e.target.value }))}
                        onBlur={(e) => void saveInfo({ [k]: e.target.value })} />
                    </label>
                  ))}
                </div>
              </Adv>

              <div className="cc-actions">
                <button type="button" className="cc-btn primary cc-cta"
                  onClick={() => doGenerate('w1')} disabled={busy || !cPrompt}>
                  生成立绘
                </button>
                <button type="button" className="cc-btn ghost" onClick={() => doGenerate('w2')}
                  disabled={busy || !cPrompt || !hasPortrait}
                  title={hasPortrait ? '以最新那张立绘为基础再画一张' : '要先有一张立绘'}>
                  基于这张再生一张
                </button>
                <span className="cc-cost">单张约 25 秒 · 显存峰值 11~15GB</span>
              </div>
              {!hasPortrait && (
                <p className="cc-tip">还没有立绘。待处理页的整套动作素材都要以它为基准，先立一张。</p>
              )}
            </>
          )}

          {/* ===== 待处理页：原「出素材」内容原样保留 ===== */}
          {view === 'pending' && sel && detail && (
            <>
              <div className="cc-pending-note">
                待处理 · 供参考：出素材与导出的内容先原样放在这一页，暂不并入角色主线。
              </div>
              {!hasSprite && (
                <Blocker text="还没有立绘，整套姿势、换装、表情都没有基准。"
                  action="去角色设定" onAction={() => setView('stage')} />
              )}
              <Card title="一键出全套" tone="primary"
                note={`按固定顺序跑完「干净起稿 → 整套姿势 → 表情」，不用自己记步骤。`}>
                <div className="cc-row">
                  <label className="cc-inline-field">姿势素材
                    <select value={poseDir} onChange={(e) => setPoseDir(e.target.value)}>
                      {poseDirs.length === 0 && <option value="">（没有可用的姿势目录）</option>}
                      {poseDirs.map((d) => (
                        <option key={d.path} value={d.path}>{d.name}（{d.count} 张）</option>
                      ))}
                    </select>
                  </label>
                  <label className="cc-inline-field">顺带做的表情
                    <select multiple size={4} value={pipeEmotions}
                      onChange={(e) => setPipeEmotions(
                        [...e.target.selectedOptions].map((o) => o.value))}>
                      {flatEmotions.map((e2) => (
                        <option key={e2.safe_name} value={e2.safe_name}>{e2.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="cc-row">
                  <label className="chars-check">
                    <input type="checkbox" checked={pipeUnderwear}
                      onChange={(e) => setPipeUnderwear(e.target.checked)} />
                    先做干净起稿
                  </label>
                  <label className="chars-check">
                    <input type="checkbox" checked={matte} onChange={(e) => setMatte(e.target.checked)} />
                    额外抠一次背景
                    <span className="cc-inline-tag" title="在顶栏 ⚙ 全局设置里改">
                      {matteMethod}
                    </span>
                  </label>
                  <span className="cc-cost">
                    预计 {pipelineEstimate}
                    {pipeEmotions.length > 0 && ` · 含 ${pipeEmotions.length} 个表情（每个约 110 秒/帧）`}
                  </span>
                </div>
                <div className="cc-actions">
                  <button type="button" className="cc-btn primary cc-cta"
                    onClick={doRunPipeline} disabled={busy || !poseDir || !hasSprite}>
                    一键出全套
                  </button>
                  {!poseDir && <span className="cc-tip">没有姿势目录：先在画布里用「人偶姿势」存一组姿势集。</span>}
                </div>
              </Card>

              <Adv title="只重跑某一步（换装 / 表情 / 姿势）">
                <div className="cc-subcard">
                  <h4>换装</h4>
                  <p className="chars-sub">每套服装是一张独立立绘，之后的表情与姿势都从它出发。</p>
                  <div className="cc-row">
                    <label className="cc-inline-field">当前服装
                      <select value={costume} onChange={(e) => setCostume(e.target.value)}>
                        {detail.costumes.map((c) => (
                          <option key={c} value={c}>{zh(COSTUME_LABEL, c)}</option>
                        ))}
                      </select>
                    </label>
                    <input className="cc-costume-name" value={newCostume} placeholder="新服装名，例：护士服"
                      onChange={(e) => setNewCostume(e.target.value)} />
                    <button type="button" className="cc-btn" onClick={() => void doCreateCostume()}
                      disabled={busy || !newCostume.trim()}>新建服装</button>
                    <button type="button" className="cc-btn ghost" onClick={() => void doDeleteCostume()}
                      disabled={costume === 'Naked'}>删除</button>
                  </div>
                  <div className="cc-row">
                    <input className="chars-wizard-input" value={clothText}
                      placeholder="一句话描述服装，例：赛博朋克护士，白大褂 + 护目镜"
                      onChange={(e) => setClothText(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') void doClothesWizard() }} />
                    <button type="button" className="cc-btn ghost" onClick={() => void doClothesWizard()}
                      disabled={busy || !clothText.trim()}>AI 拆服装</button>
                  </div>
                  <CostumeForm key={`${detail.name}/${costume}`} fields={costumeDraft}
                    onChange={setCostumeDraft} onSave={(d) => void saveCostumeFields(d)} />
                  <div className="cc-row">
                    <button type="button" className="cc-btn" onClick={doDesignClothes}
                      disabled={busy || costume === 'Naked' || costume === 'Original' || !hasSprite}
                      title={costume === 'Naked' || costume === 'Original'
                        ? '先新建一套具体服装' : '按上面的字段生成这套服装的立绘'}>
                      生成这套服装的立绘
                    </button>
                    <button type="button" className="cc-btn ghost" onClick={doUnderwear}
                      disabled={busy || !hasSprite}>生成干净起稿（白内衣）</button>
                  </div>
                  <p className="chars-sub">
                    干净起稿是换装的统一起点：先换成白色内衣，新衣服才不会带着上一套的袖子残留。
                    上面的「一键出全套」已经自动包含这一步。
                  </p>
                </div>

                <div className="cc-subcard">
                  <h4>表情</h4>
                  <p className="chars-sub">
                    表情只在立绘的脸部区域重画，衣服和动作逐像素不变。
                    先点表情卡选中，再生成；建议一次别选太多。
                  </p>
                  <div className="cc-row">
                    <button type="button" className="cc-btn ghost" onClick={() => void doDetect()}
                      disabled={busy}>检出脸框</button>
                    <button type="button" className="cc-btn ghost"
                      onClick={() => setEmoSel(flatEmotions.map((e) => e.safe_name))}>全选</button>
                    <button type="button" className="cc-btn ghost"
                      onClick={() => setEmoSel([])}>清空</button>
                    <span className="chars-sub">已选 {emoSel.length} 个</span>
                    {emoFrames && <span className="chars-sub">
                      源帧 {emoFrames.length} 张 · 脸框 {emoFrames.filter((f) => f.box).length} 张
                      {emoFrames.filter((f) => f.box).length > 0 && `（${emoFrames.filter((f) => f.box)
                        .map((f) => `${f.box![2] - f.box![0]}×${f.box![3] - f.box![1]}`).join(' / ')}）`}
                    </span>}
                  </div>
                  <EmotionGrid options={flatEmotions} selected={emoSel}
                    onToggle={(name) => setEmoSel((list) => list.includes(name)
                      ? list.filter((x) => x !== name) : [...list, name])} />
                  <div className="cc-row">
                    <label className="cc-inline-field">画质档
                      <select value={emoParams.target_size}
                        onChange={(e) => setEmoParams((p) => ({ ...p, target_size: Number(e.target.value) }))}>
                        <option value={2048}>标准（2048）</option>
                        <option value={1024}>快速（1024，约快一倍）</option>
                      </select>
                    </label>
                    <label className="cc-inline-field">步数
                      <input type="number" min={8} max={60} value={emoParams.steps}
                        onChange={(e) => setEmoParams((p) => ({ ...p, steps: Number(e.target.value) || 16 }))} />
                    </label>
                    <span className="cc-cost">低于 16 步容易出糊块；一张约 1~2 分钟。</span>
                  </div>
                  <div className="cc-actions">
                    <button type="button" className="cc-btn" onClick={doGenerateEmotions}
                      disabled={busy || emoSel.length === 0 || !hasFrames}>
                      批量生成（{emoSel.length} 个表情）
                    </button>
                  </div>
                </div>

                <div className="cc-subcard">
                  <h4>姿势（整批替换）</h4>
                  <p className="chars-sub">
                    按上面「姿势素材」选的那组骨架帧，一次出整套精灵帧，
                    共用一个种子保证衣服发型不飘。⚠️ 这套服装现有的平静帧会归档到
                    <code>V1/</code>，需要单帧保留就别跑这一步。
                  </p>
                  <div className="cc-row">
                    <input className="chars-wizard-input" value={poseText}
                      placeholder="追加描述（可空）" onChange={(e) => setPoseText(e.target.value)} />
                    <button type="button" className="cc-btn" onClick={doGeneratePoses}
                      disabled={busy || !poseDir || !hasSprite}>按这批姿势生成</button>
                  </div>
                </div>
              </Adv>
            </>
          )}

          {/* ===== 待处理页：原「导出」内容原样保留 ===== */}
          {view === 'pending' && sel && detail && (
            <>
              <h2 className="cc-pending-title">导出（待处理）</h2>
              <Card title="导出前试玩验收" tone="plain"
                note="俯视四方向 / 俯视八方向 / 横板三种视角都能试；缺哪些方向、对应白模哪个机位，会直接列在下面。">
                <CharacterPlaytest character={sel} onApplyFps={setAtlasFrameRate} />
              </Card>
              <Card title="导出图集" tone="primary"
                note="所有帧会统一到同一画布并做水平居中 + 底边对齐，动画里角色才不会上下抖。">
                <div className="cc-actions">
                  <button type="button" className="cc-btn primary cc-cta"
                    onClick={doExportAtlas} disabled={busy || !hasFrames}>导出图集</button>
                  <span className="cc-cost">
                    输出 atlas.png + atlas.json · 帧率 {atlasFrameRate} fps
                    {atlasFrameRate !== 8 && '（来自试玩验收）'}
                  </span>
                </div>
                {atlasInfo?.exists && (
                  <>
                    <p className="chars-sub">
                      现有图集：{atlasInfo.frames} 帧 / {atlasInfo.animations} 组动画
                      {!!atlasInfo.directionAnimations && `（其中 ${atlasInfo.directionAnimations} 组按方向拆分）`} ·{' '}
                      {atlasInfo.size?.[0]}×{atlasInfo.size?.[1]} ·{' '}
                      单帧画布 {atlasInfo.canvas?.[0]}×{atlasInfo.canvas?.[1]} ·{' '}
                      {Math.max(1, Math.round((atlasInfo.bytes || 0) / 1024))} KB
                    </p>
                    {atlasInfo.image && (
                      <a className="chars-atlas-link" href={atlasInfo.image}
                        target="_blank" rel="noreferrer">
                        <img src={atlasInfo.image} alt="图集" loading="lazy" />
                        <span>打开图集（新标签页 · 右键可另存）</span>
                      </a>
                    )}
                    <div className="cc-row">
                      <a className="cc-btn ghost" href={atlasInfo.image?.replace(/\.png$/i, '.json')}
                        target="_blank" rel="noreferrer" download>下载坐标表 atlas.json</a>
                    </div>
                  </>
                )}
              </Card>
              <Card title="拿进游戏" note="图集与坐标表都在角色目录的 Atlas/ 下，整个文件夹可以直接拷进工程。">
                <div className="cc-how">
                  <ol>
                    <li>把 atlas.png 和 atlas.json 放进工程资源目录</li>
                    <li>坐标表按「服装 / 表情」分好了动画组，不用再手工归类</li>
                    <li>想继续加素材，点上面的步骤 ③ 回来单步重跑</li>
                  </ol>
                </div>
                <button type="button" className="cc-btn ghost"
                  onClick={backToLibrary}>回到角色库</button>
              </Card>
            </>
          )}

          {!sel && (
            <Blocker text="先在角色库里选一个角色。" action="回角色库"
              onAction={backToLibrary} />
          )}
        </div>

        {view !== 'stages' && (
          <div className="cc-next">
            <button type="button" className="cc-btn ghost"
              onClick={() => setView('stages')}>← 返回生命阶段</button>
          </div>
        )}
      </main>

      {/* 生命阶段列表页不要右栏：每行已经在显示该阶段的立绘，
          再放一个全局立绘面板反而分不清属于哪个阶段。
          立绘是「角色设定」的一部分，只在进入某个阶段后出现。 */}
      {/* 拆部件页也要单列：选图 + 分层预览占满宽度，右边不再挂立绘 */}
      {view !== 'stages' && view !== 'parts' && (
        <>
          <div ref={splitRef} className="cc-split" role="separator" aria-orientation="vertical"
            aria-label="拖动调整成果区宽度" title="拖动调整宽度（双击恢复默认）"
            onPointerDown={startSideDrag}
            onPointerMove={moveSideDrag}
            onPointerUp={endSideDrag}
            onPointerCancel={endSideDrag}
            onDoubleClick={() => { setSideW(420); localStorage.setItem('meow.cc.sideW', '420') }} />

          {sel && detail ? (
            <PortraitPane
              character={detail.name} portraits={portraits}
              task={task} doneFlash={doneFlash} onDismissFlash={() => setDoneFlash(null)}
              onChanged={() => void refreshPortraits()}
              stageId={stageId} runTask={runTask} />
          ) : (
            <aside className="cc-preview cc-preview-idle">
              <div className="cc-preview-head"><b>立绘</b></div>
              <div className="cc-preview-empty">
                <span className="cc-preview-empty-icon">◧</span>
                <p>立绘是角色设定的一部分，跟动作帧分开。</p>
                <p>进入某个生命阶段后，这里只显示该阶段的立绘设定图。</p>
              </div>
            </aside>
          )}
        </>
      )}

      {/* 「修改」弹窗：角色级信息（名字 / 描述 / 档案 / 删除），不属于任何生命阶段 */}
      {editOpen && (
        <div className="cc-edit-mask" onClick={() => setEditOpen(false)}>
          <div className="cc-edit-dialog" onClick={(e) => e.stopPropagation()}>
            <h2>修改角色</h2>
            <label className="cc-edit-field">角色名
              <input value={renameText}
                onChange={(e) => setRenameText(e.target.value)} />
            </label>
            <label className="cc-edit-field">描述
              <textarea rows={3} value={descDraft}
                placeholder="这个角色是谁…"
                onChange={(e) => setDescDraft(e.target.value)} />
            </label>
            <div className="cc-edit-meta">
              <span>创建 {fmtStamp(currentSummary?.created_at)}</span>
              <span>最后编辑 {fmtStamp(currentSummary?.updated_at)}</span>
              <span>已有素材 {assetStats.frames} 帧</span>
            </div>
            <p className="cc-edit-tags">标签：{[
              ...(currentSummary?.tags || []),
              ...(currentSummary?.custom_tags || []),
            ].map(tagLabel).join('、') || '—'}</p>
            {currentSummary?.note && (
              <p className="cc-edit-note">备注：{currentSummary.note}</p>
            )}
            {pendingDelete && (
              <div className="cc-confirm">
                <p>删掉「{pendingDelete}」？所有立绘、服装、表情会一起删。</p>
                <button type="button" className="cc-btn ghost"
                  onClick={() => setPendingDelete(null)}>取消</button>
                <button type="button" className="cc-btn danger"
                  onClick={() => void doDelete(pendingDelete)}>确认删除</button>
              </div>
            )}
            <div className="cc-edit-actions">
              <button type="button" className="cc-btn ghost"
                onClick={() => setPendingDelete(detail?.name || '')}>删除角色</button>
              <span className="spacer" />
              <button type="button" className="cc-btn ghost"
                onClick={() => setEditOpen(false)}>取消</button>
              <button type="button" className="cc-btn primary"
                onClick={() => void saveEdit()}>保存</button>
            </div>
          </div>
        </div>
      )}

      <TagPicker
        open={!!picking}
        field={picking || ''}
        label={INFO_FIELDS.find(([k]) => k === picking)?.[1] || ''}
        catalog={tagCat}
        value={String(draft[picking ?? ''] ?? '')}
        onClose={() => setPicking(null)}
        onApply={(v) => {
          const k = picking
          if (!k) return
          setDraft((d) => ({ ...d, [k]: v }))
          void saveInfo({ [k]: v })
        }} />
    </div>
  )
}
