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
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  type CharacterDetail, type CharacterSummary, type StyleCatalog,
  type EmotionCatalog,
  deleteCharacter, deleteCostume, fetchCharacter,
  fetchCharacters, renameCharacter, saveCostume, updateCharacterInfo,
  fetchStyleCatalog, fetchEmotionCatalog,
  fetchCharacterPrompt, generateForCharacter, fetchJob, fetchCharacterTags,
  detectEmotionFaces, generateEmotions, type EmotionFrame,
  analyzeCharacter, wizardCharacter, designClothes, clothesWizard,
  generatePoses, fetchPoseDirs, exportAtlas, fetchAtlasInfo, designUnderwear,
  runPipeline, type PoseDir, type AtlasInfo,
  type CharacterPrompt, type TagCatalog,
} from '../api'
import CharacterLibrary, { tagLabel } from './CharacterLibrary'
import CharacterPlaytest from './CharacterPlaytest'
import PromptZoom from './PromptZoom'
import TagPicker from './TagPicker'
import {
  BG_COLOR_LABEL, COSTUME_LABEL, STYLE_GROUP_LABEL, STYLE_LABEL, emotionLabel, zh,
} from '../labels'
import {
  Adv, Blocker, Card, CostumeForm, EmotionGrid, FieldChips, PreviewPane,
  StepBar, StepHead, StylePicker,
  type EmotionOption, type StyleOption,
} from './characters/Controls'
import {
  BG_COLORS, INFO_ADV_FIELDS, INFO_FIELDS, STEPS,
  computeProgress, estimateSeconds, explainError,
  type StepId, type TaskState,
} from './characters/model'

type Props = {
  onNotice: (msg: string) => void
  /** 全局抠图方式（顶栏 ⚙ 全局设置里改）。角色中心的幕布抠图会跟着它走。 */
  matteBackend?: string
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

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
  /** 'library' = 全屏角色库（默认首页）；'workspace' = 某个角色的四步工作台 */
  const [view, setView] = useState<'library' | 'workspace'>('library')
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
  const [renameOpen, setRenameOpen] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

  /* --------------------------------------------------- 当前角色的配置 */
  const [draft, setDraft] = useState<Record<string, unknown>>({})
  const [cPrompt, setCPrompt] = useState<CharacterPrompt | null>(null)
  /** 单一「当前服装」：预览、换装、表情、姿势、流水线全都用它。
   *  旧版 costume / browseCostume 两套状态会让「服装页选 A、素材页跑 B」。 */
  const [costume, setCostume] = useState('Naked')
  const [browseEmotion, setBrowseEmotion] = useState('neutral')
  const [browseFiles, setBrowseFiles] = useState<string[]>([])
  /** 单一「抠背景」开关，默认关 —— 模型自带透明底，跑分割是用户的选择。
   *  旧版出图 / 表情 / 流水线 / 后端四处默认值各不相同。 */
  const [matte, setMatte] = useState(false)
  /** Prompt Enhancement 官方默认开；代价写在进阶设置里，不再埋在折叠小字。 */
  const [enhance, setEnhance] = useState(true)
  const enhanceRef = useRef(enhance)
  enhanceRef.current = enhance

  /* ------------------------------------------------------- 步骤与任务 */
  const [userStep, setUserStep] = useState<StepId | null>(null)
  /** 唯一任务状态，常驻右栏；替代旧版 9 个 xxxMsg + 一闪而过的 toast。 */
  const [task, setTask] = useState<TaskState | null>(null)
  const [doneFlash, setDoneFlash] = useState<{ text: string; at: number } | null>(null)

  /* --------------------------------------------------- 各步的输入草稿 */
  const [picking, setPicking] = useState<string | null>(null)
  const [styleQuery, setStyleQuery] = useState('')
  const [styleGroup, setStyleGroup] = useState('')
  const [styleOpen, setStyleOpen] = useState(false)
  const [wizText, setWizText] = useState('')
  const [anaResult, setAnaResult] = useState<Record<string, unknown> | null>(null)
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
  const refresh = useCallback(async (name: string) => {
    const d = await fetchCharacter(name)
    setDetail(d)
    setDraft(d.config.character_info ?? {})
    const info = await fetchAtlasInfo(name).catch(() => null)
    setAtlasInfo(info)
    await reload()
    return d
  }, [reload])

  const refreshPrompt = useCallback((name: string, withEnhance = enhanceRef.current) => {
    void fetchCharacterPrompt(name, '', undefined, withEnhance)
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
      return
    }
    let alive = true
    setUserStep(null); setTask(null); setDoneFlash(null); setAnaResult(null)
    setEmoFrames(null); setEmoSel([]); setBrowseEmotion('neutral')
    void fetchCharacter(sel)
      .then((d) => {
        if (!alive) return
        setDetail(d); setDraft(d.config.character_info ?? {})
        setCostume((cur) => d.costumes.includes(cur) ? cur : (d.costumes[0] ?? 'Naked'))
      })
      .catch((e) => { if (alive) setErr(explainError(e)) })
    refreshPrompt(sel)
    void fetchAtlasInfo(sel).then((i) => { if (alive) setAtlasInfo(i) }).catch(() => {})
    return () => { alive = false }
  }, [sel, refreshPrompt])

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
  const currentStyleLabel = useMemo(
    () => flatStyles.find((s) => s.id === currentStyle)?.zhLabel ?? currentStyle,
    [flatStyles, currentStyle])

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

  const progress = useMemo(() => computeProgress({
    hasCharacter: !!sel && !!detail,
    frames: assetStats.frames,
    groups: assetStats.groups,
    hasAtlas: !!atlasInfo?.exists,
  }), [sel, detail, assetStats, atlasInfo])
  const activeStep: StepId = userStep ?? progress.firstUndone
  const hasSprite = assetStats.frames > 0
  const browseEmotions = useMemo(() => {
    const had = Object.keys(detail?.assets?.[costume] || {})
    return had.length ? had : ['neutral']
  }, [detail, costume])

  /** 必须稳定：SpritePreview 的 loadMeta 把它列进依赖，内联箭头函数会让
   *  每次 render 都重新拉 meta、把翻到的第二帧弹回第一帧。 */
  const handlePreviewMeta = useCallback(
    (m: { count: number; files: string[] }) => setBrowseFiles(m.files), [])

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
          setUserStep(null)
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
    setView('workspace')
    setUserStep(null)
  }, [])

  const backToLibrary = useCallback(() => {
    setView('library')
    setSel(null)
    setUserStep(null)
    setTask(null)
    setDoneFlash(null)
  }, [])

  const doRename = useCallback(async () => {
    if (!sel || !renameText.trim() || renameText.trim() === sel) return
    try {
      await renameCharacter(sel, renameText.trim())
      setRenameOpen(false)
      await reload()
      setSel(renameText.trim())
    } catch (e) { setErr(`改名失败：${explainError(e)}`) }
  }, [sel, renameText, reload])

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

  /* --------------------------------------------------- 第 ② 步：定形象 */
  const doWizard = useCallback(async () => {
    const text = wizText.trim()
    if (!sel || !text || busy) return
    await runInline('一句话生成设定', async () => {
      const r = await wizardCharacter(text)
      setAnaResult(r.fields)
      return `读好了（${r.seconds}s）——下面是它拆出来的设定，确认后再写进角色`
    })
  }, [sel, wizText, busy, runInline])

  const doAnalyze = useCallback(async () => {
    if (!sel || busy) return
    await runInline('从图片反推属性', async () => {
      const r = await analyzeCharacter(sel, {
        background_color: String(draft.background_color ?? ''),
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

  /** 出第一张立绘（w1）/ 以立绘为基础再画一张（w2）。 */
  const doGenerate = useCallback((workflow: 'w1' | 'w2') => {
    if (!sel) return
    void runTask(workflow === 'w1' ? '生成第一张立绘' : '再画一张立绘',
      () => generateForCharacter(sel, {
        workflow, costume, emotion: 'neutral',
        prompt: cPrompt?.prompt || '', negative: cPrompt?.negative || '',
        enhance, matte,
      }),
      () => { setBrowseEmotion('neutral') })
  }, [sel, runTask, costume, cPrompt, enhance, matte])

  /* --------------------------------------------------- 第 ③ 步：出素材 */
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
      if (r.count === 0) return '没有源帧：先生成第一张立绘，表情是在它上面只改脸'
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
      setBrowseEmotion(emoSel[0])
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

  /* ----------------------------------------------------- 第 ④ 步：导出 */
  const doExportAtlas = useCallback(() => {
    if (!sel) return
    void runTask('导出图集', () => exportAtlas(sel, { frame_rate: atlasFrameRate }))
  }, [sel, runTask, atlasFrameRate])

  /* ================================================================== 渲染 */
  /** 下一步永远指向第一个真正没做完的步骤（不是简单的 activeStep+1：
   *  用户手动回头看第 ② 步时，主线仍应指向真正缺的那一步）。 */
  const nextStep = progress.firstUndone
  const allDone = progress.done.every(Boolean)
  const currentSummary = items.find((c) => c.name === sel)

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
    <div className="chars cc is-workspace">
      {/* ---------------- 中：唯一主线 ------------- */}
      <main className="chars-detail cc-main">
        <div className="cc-backbar">
          <button type="button" className="cc-btn ghost" onClick={backToLibrary}>← 角色库</button>
          <span>角色库 / <b>{sel || ''}</b></span>
          <span className="spacer" />
          {currentSummary?.updated_at ? (
            <span className="cc-backbar-meta">
              最后编辑 {new Date(currentSummary.updated_at * 1000).toLocaleDateString('zh-CN')}
            </span>
          ) : null}
        </div>
        {err && <div className="chars-err">⚠ {err}</div>}
        <StepBar done={progress.done} active={activeStep} onPick={setUserStep} />

        {sel && detail && (
          <div className="cc-charhead">
            <b>{detail.name}</b>
            <span className="cc-charhead-meta">
              当前服装：{zh(COSTUME_LABEL, costume)} · 画风：{currentStyleLabel || '默认'}
            </span>
            <span className="spacer" />
            {renameOpen ? (
              <>
                <input className="cc-rename" value={renameText}
                  onChange={(e) => setRenameText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') void doRename() }} />
                <button type="button" className="cc-btn ghost" onClick={() => void doRename()}>保存</button>
              </>
            ) : (
              <button type="button" className="cc-btn ghost"
                onClick={() => { setRenameText(detail.name); setRenameOpen(true) }}>改名</button>
            )}
          </div>
        )}

        <div className="chars-body cc-panel">
          {/* ============================ ① 角色档案（创建已在全屏库完成） ============================ */}
          {activeStep === 1 && sel && detail && (
            <>
              <StepHead step={1} sub="角色已在角色库里建好。这里能改名、删除、查看标签与备注；要编辑标签，回角色库点卡片上的 ✎。" />
              <Card title="角色档案"
                note="标签、时间、自定义标签与备注来自本地 SQLite 元数据；生成用的属性 config 与所有图片仍在磁盘上。"
                tone="primary">
                <div className="cc-profile">
                  <div><span>名字</span><b>{detail.name}</b></div>
                  <div><span>创建时间</span><b>{fmtStamp(currentSummary?.created_at)}</b></div>
                  <div><span>最后编辑</span><b>{fmtStamp(currentSummary?.updated_at)}</b></div>
                  <div><span>服装</span><b>{detail.costumes.map((c) => zh(COSTUME_LABEL, c)).join(' / ') || '—'}</b></div>
                  <div><span>已有素材</span><b>{assetStats.frames} 帧</b></div>
                  <div><span>标签</span><b className="cc-profile-tags">
                    {[...(currentSummary?.tags || []), ...(currentSummary?.custom_tags || [])]
                      .map(tagLabel).join('、') || '—'}
                  </b></div>
                </div>
                {currentSummary?.note && <p className="cc-profile-note">备注：{currentSummary.note}</p>}
                <div className="cc-actions">
                  <button type="button" className="cc-btn primary cc-cta"
                    onClick={() => setUserStep(2)}>去定形象 →</button>
                  <button type="button" className="cc-btn ghost"
                    onClick={() => { setRenameText(detail.name); setRenameOpen(true) }}>改名</button>
                  <button type="button" className="cc-btn ghost"
                    onClick={() => setPendingDelete(detail.name)}>删除角色</button>
                </div>
              </Card>
              {pendingDelete && (
                <div className="cc-confirm">
                  <p>删掉「{pendingDelete}」？所有立绘、服装、表情会一起删。</p>
                  <button type="button" className="cc-btn ghost"
                    onClick={() => setPendingDelete(null)}>取消</button>
                  <button type="button" className="cc-btn danger"
                    onClick={() => void doDelete(pendingDelete)}>确认删除</button>
                </div>
              )}
              <div className="cc-how">
                <b>接下来会发生什么</b>
                <ol>
                  <li>定形象：选外貌与画风，出第一张立绘</li>
                  <li>出素材：一键生成整套姿势、表情、换装</li>
                  <li>导出：打包图集，直接进游戏引擎</li>
                </ol>
              </div>
            </>
          )}

          {/* ============================ ② 定形象 ============================ */}
          {activeStep === 2 && sel && detail && (
            <>
              <StepHead step={2} sub="这一步定的是「这个角色长什么样」。定完生成的立绘会作为后面所有素材的基准。" />
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
                </div>
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
                <label className="chars-field">正向 prompt
                  <textarea className="chars-prompt" value={cPrompt?.prompt ?? ''}
                    onChange={(e) => setCPrompt((p) => (p ? { ...p, prompt: e.target.value } : p))} />
                  <PromptZoom value={cPrompt?.prompt ?? ''} label="正向 prompt" disabled={!cPrompt}
                    hint="英文描述效果更稳。改完直接出图，不用重拼。"
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
                  <span className="chars-sub">
                    分辨率 {cPrompt ? `${cPrompt.resolution[0]}×${cPrompt.resolution[1]}` : '—'}
                    · {cPrompt ? `${cPrompt.defaults.steps} 步 / cfg ${cPrompt.defaults.cfg}` : ''}
                  </span>
                  <button type="button" className="cc-btn ghost" onClick={() => sel && refreshPrompt(sel)}>
                    重拼 prompt
                  </button>
                </div>
                <label className="chars-check">
                  <input type="checkbox" checked={enhance}
                    onChange={(e) => { setEnhance(e.target.checked); refreshPrompt(sel, e.target.checked) }} />
                  让模型自己润色提示词（Prompt Enhancement）
                  <span className="chars-sub">
                    用图像模型自带的 <b>Qwen3-VL</b> 改写，措辞更贴近训练分布；
                    代价是编码器要常驻显存、每次多花一点时间。失败会自动退回原始 prompt。
                  </span>
                </label>
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
                  生成第一张立绘
                </button>
                <button type="button" className="cc-btn ghost" onClick={() => doGenerate('w2')}
                  disabled={busy || !cPrompt || !hasSprite}
                  title={hasSprite ? '以现有立绘为基础再画一张' : '要先生成第一张立绘'}>
                  以立绘为基础再画一张
                </button>
                <span className="cc-cost">单张约 25 秒 · 显存峰值 11~15GB</span>
              </div>
              {!hasSprite && (
                <p className="cc-tip">还没有立绘。第 ③ 步的整套素材都要以它为基准，先生成这一张。</p>
              )}
            </>
          )}

          {/* ============================ ③ 出素材 ============================ */}
          {activeStep === 3 && sel && detail && (
            <>
              <StepHead step={3} sub="所有素材都从刚才那张立绘出发，自动落到这个角色的档案里，任何一步都能单独重跑。" />
              {!hasSprite && (
                <Blocker text="还没有立绘，整套姿势、换装、表情都没有基准。"
                  action="去生成立绘" onAction={() => setUserStep(2)} />
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
                      disabled={busy || emoSel.length === 0 || !hasSprite}>
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

          {/* ============================= ④ 导出 ============================= */}
          {activeStep === 4 && sel && detail && (
            <>
              <StepHead step={4} sub="先把素材放进场景里跑一遍，再打包成图集和坐标表，直接拿进 Godot / Unity / Aseprite。" />
              <Card title="导出前试玩验收" tone="plain"
                note="俯视四方向 / 俯视八方向 / 横板三种视角都能试；缺哪些方向、对应白模哪个机位，会直接列在下面。">
                <CharacterPlaytest character={sel} onApplyFps={setAtlasFrameRate} />
              </Card>
              <Card title="导出图集" tone="primary"
                note="所有帧会统一到同一画布并做水平居中 + 底边对齐，动画里角色才不会上下抖。">
                <div className="cc-actions">
                  <button type="button" className="cc-btn primary cc-cta"
                    onClick={doExportAtlas} disabled={busy || !hasSprite}>导出图集</button>
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

        {/* 下一步：永远给一个明确的落点 */}
        {sel && !allDone && (
          <div className="cc-next">
            <span>下一步：<b>{STEPS[nextStep - 1].label}</b> · {STEPS[nextStep - 1].goal}</span>
            <button type="button" className="cc-btn" onClick={() => setUserStep(nextStep)}>
              前往 →
            </button>
          </div>
        )}
        {sel && allDone && (
          <div className="cc-next is-done">✓ 这个角色已经走完全部四步，图集可以直接进游戏。</div>
        )}
      </main>

      {/* ---------------- 右：常驻成果 + 统一任务状态 ------------- */}
      {sel && detail ? (
        <PreviewPane
          character={detail.name} costume={costume} emotion={browseEmotion}
          costumes={detail.costumes} emotions={browseEmotions} files={browseFiles}
          task={task} doneFlash={doneFlash} onDismissFlash={() => setDoneFlash(null)}
          onCostume={setCostume} onEmotion={setBrowseEmotion}
          onMeta={handlePreviewMeta} atlasInfo={atlasInfo} />
      ) : (
        <aside className="cc-preview cc-preview-idle">
          <div className="cc-preview-head"><b>角色成果</b></div>
          <div className="cc-preview-empty">
            <span className="cc-preview-empty-icon">◧</span>
            <p>这里会一直显示这个角色的最新成果。</p>
            <p>从选定形象开始，生成的每一张立绘、每一套精灵帧都能立刻翻看。</p>
          </div>
          <div className="cc-how">
            <b>四步做出一整套素材</b>
            <ol>
              {STEPS.map((s) => <li key={s.id}>{s.label} —— {s.result}</li>)}
            </ol>
          </div>
        </aside>
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
