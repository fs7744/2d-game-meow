/**
 * 角色中心（与画布同级的一级页面）—— 移植自 VNCCS 的 Character Creator /
 * Cloner / Clothes Designer / Emotion Studio。
 *
 * **为什么是一个独立页面**：VNCCS 里这些是 ComfyUI 画布节点里塞的巨大 DOM
 * widget（`web/vnccs_character_creator_v2.js:1567` 的 registerExtension），
 * 我们没有 ComfyUI，也不该把角色管理塞进节点。做成与画布同级的一级页面，
 * 左边角色库、右边角色详情，是这类"素材库"最直接的形态。
 *
 * **数据契约跟官方一致**（见 workflows/characters.py 的模块注释）：
 *   <角色>/{角色}_config.json + Sprites/Faces/Sheets/<服装>/<表情>/sprite_*.png
 * 所以官方工作流产出的角色目录，我们能直接读；我们产出的它也能读。
 *
 * 官方的三个 profile 里只保留 **QI2（Qwen-Image-2.1）**；Flux Klein、MiniMax H3、
 * Anima、Illustrious/SDXL 一概不移植（见 docs/角色中心-VNCCS移植评估.md）。
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  type CharacterDetail, type CharacterSummary, type StyleCatalog,
  type StyleEntry, type EmotionCatalog, type EmotionEntry,
  createCharacter, deleteCharacter, deleteCostume, fetchCharacter,
  fetchCharacters, renameCharacter, saveCostume, updateCharacterInfo,
  fetchStyleCatalog, fetchEmotionCatalog, stylePreviewUrl, emotionIconUrl,
  fetchCharacterPrompt, generateForCharacter, fetchJob, fetchCharacterTags,
  detectEmotionFaces, generateEmotions, type EmotionFrame,
  analyzeCharacter, wizardCharacter, designClothes, clothesWizard,
  generatePoses, fetchPoseDirs, exportAtlas, fetchAtlasInfo, designUnderwear,
  type PoseDir, type AtlasInfo,
  type CharacterPrompt, type TagCatalog,
} from '../api'
import SpritePreview from './SpritePreview'
import TagPicker from './TagPicker'

/** 角色属性字段（官方 character_creator_v2.py:2138-2146 的字段清单） */
const INFO_TEXT: Array<[key: string, label: string, hint?: string]> = [
  ['race', '种族', 'human / elf / demon …'],
  ['skin_color', '肤色', ''],
  ['hair', '发型发色', '逗号分隔，如 "blonde hair, twin tails"'],
  ['eyes', '眼睛', ''],
  ['face', '脸部特征', '如 freckles'],
  ['body', '体型', ''],
  ['breast_size', '胸围', ''],
  ['additional_details', '附加细节', ''],
  ['aesthetics', '画质词', 'masterpiece 之类'],
  ['negative_prompt', '负向词', ''],
  ['lora_prompt', 'LoRA 触发词', ''],
]

/**
 * 能用官方标签库点选的字段（官方 `FIELD_CATEGORIES`，character_presets.mjs:1-9）。
 *
 * **画质词 / 负向词 / LoRA 触发词不在里面** —— 官方也不给这三个做标签选择：
 * 它们是整段的提示词工程，不是一个一个 trait。
 */
const INFO_TAGS = new Set([
  'race', 'skin_color', 'hair', 'eyes', 'face', 'body', 'breast_size', 'additional_details',
])

const INFO_NUM: Array<[key: string, label: string]> = [['age', '年龄']]

/** 背景色（官方 background_color，用于抠图与预览） */
const BG_COLORS = ['green', 'blue', 'white', 'grey', 'magenta']

type Props = {
  /** 一次性提示（走 App 里那条 toast） */
  onNotice: (msg: string) => void
}

export default function CharactersPage({ onNotice }: Props) {
  const [items, setItems] = useState<CharacterSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')
  const [sel, setSel] = useState<string | null>(null)
  const [detail, setDetail] = useState<CharacterDetail | null>(null)
  const [styles, setStyles] = useState<StyleCatalog | null>(null)
  const [emotions, setEmotions] = useState<EmotionCatalog | null>(null)
  const [tab, setTab] = useState<'info' | 'gen' | 'clothes' | 'emotions' | 'assets'>('info')
  const [newName, setNewName] = useState('')
  const [costume, setCostume] = useState('Naked')
  const [draft, setDraft] = useState<Record<string, unknown>>({})
  /** 后端按官方结构拼好的 prompt（不跑模型，改属性就能立刻看到） */
  const [cPrompt, setCPrompt] = useState<CharacterPrompt | null>(null)
  /** Prompt Enhancement：用图像模型自己的 Qwen3-VL 改写 prompt（官方默认开） */
  const [enhance, setEnhance] = useState(true)
  /** 抠背景：官方 bg_remove 阶段，纯 torch 零模型 */
  const [matte, setMatte] = useState(true)
  /** 官方属性标签库（creator_v2），随项目发行、按需拉一次 */
  const [tagCat, setTagCat] = useState<TagCatalog | null>(null)
  /** 正在用标签选择器编辑哪个字段 */
  const [picking, setPicking] = useState<string | null>(null)
  /** 素材页当前在翻哪一组（服装 / 表情） */
  const [browseCostume, setBrowseCostume] = useState('Naked')
  const [browseEmotion, setBrowseEmotion] = useState('neutral')
  const [browseFiles, setBrowseFiles] = useState<string[]>([])
  /** 翻看器汇报上来的 meta（连 Node 的数量与清单，避免二次请求） */
  const setBrowseMeta = useCallback(
    (m: { count: number; files: string[] }) => setBrowseFiles(m.files), [])
  const [genBusy, setGenBusy] = useState(false)
  const [genMsg, setGenMsg] = useState('')
  /** 表情批量生成：选中的表情（官方 safe_name）、源帧脸框信息、进度 */
  const [emoSel, setEmoSel] = useState<string[]>([])
  const [emoFrames, setEmoFrames] = useState<EmotionFrame[] | null>(null)
  const [emoBusy, setEmoBusy] = useState(false)
  const [emoMsg, setEmoMsg] = useState('')
  const [emoProg, setEmoProg] = useState<{ done: number; total: number } | null>(null)
  /** 表情生成参数（官方默认值；target_size 是「总像素档位」不是边长） */
  const [emoParams, setEmoParams] = useState({ target_size: 2048, steps: 25, matte: true })
  /** 看图反推属性（官方 Cloner）：结果先给用户看，确认了才写进 config */
  const [anaBusy, setAnaBusy] = useState(false)
  const [anaMsg, setAnaMsg] = useState('')
  const [anaResult, setAnaResult] = useState<Record<string, unknown> | null>(null)
  /** AI 向导：一句话造角色（同一块结果面板） */
  const [wizText, setWizText] = useState('')
  const [wizBusy, setWizBusy] = useState(false)
  /** 换装：一句话拆服装 / 出换装底图 */
  const [clothText, setClothText] = useState('')
  const [clothBusy, setClothBusy] = useState(false)
  const [clothMsg, setClothMsg] = useState('')
  /** 服装表单的草稿（CostumeForm 受控时会回填，这里只用来喂换装 prompt） */
  const [costumeDraft, setCostumeDraft] = useState<Record<string, string>>({})
  /** 多姿势批量：骨架帧目录 / 一句话 / 进度 */
  const [poseDirs, setPoseDirs] = useState<PoseDir[]>([])
  const [poseDir, setPoseDir] = useState('')
  const [poseText, setPoseText] = useState('')
  const [poseBusy, setPoseBusy] = useState(false)
  const [poseMsg, setPoseMsg] = useState('')
  const [poseProg, setPoseProg] = useState<{ done: number; total: number } | null>(null)
  /** 图集：导出中 / 上次导出的统计 / 已有图集的信息 */
  const [atlasBusy, setAtlasBusy] = useState(false)
  const [atlasMsg, setAtlasMsg] = useState('')
  const [atlasInfo, setAtlasInfo] = useState<AtlasInfo | null>(null)
  /** 内衣底图：出图中 / 提示 */
  const [uwBusy, setUwBusy] = useState(false)
  const [uwMsg, setUwMsg] = useState('')

  const reload = useCallback(async () => {
    try {
      const r = await fetchCharacters()
      setItems(r.items)
      setErr('')
    } catch (e) {
      setErr(`取角色列表失败：${e instanceof Error ? e.message : e}`)
    } finally {
      setLoading(false)
    }
  }, [])

  // 静态目录里的官方素材：风格库 358 条 + 表情库 157 条
  useEffect(() => {
    void reload()
    fetchStyleCatalog().then(setStyles).catch(() => setStyles(null))
    fetchEmotionCatalog().then(setEmotions).catch(() => setEmotions(null))
    fetchCharacterTags().then(setTagCat).catch(() => setTagCat(null))
  }, [reload])

  useEffect(() => {
    // 多姿势生成要用骨架帧目录清单（后端扫 assets 下两层）
    fetchPoseDirs().then((list) => {
      setPoseDirs(list)
      setPoseDir((cur) => cur || (list[0]?.path ?? ''))
    }).catch(() => setPoseDirs([]))
  }, [])

  // 切角色时顺手查一下图集导出了没有（只拿统计，不拉整份 JSON）
  useEffect(() => {
    setAtlasInfo(null)
    setAtlasMsg('')
    if (!sel) return
    let alive = true
    fetchAtlasInfo(sel).then((info) => {
      if (alive) setAtlasInfo(info)
    }).catch(() => { /* 没导出过就是 exists:false */ })
    return () => { alive = false }
  }, [sel])

  useEffect(() => {
    if (!sel) { setDetail(null); return }
    let alive = true
    // 换角色时把表情批量的中间态清掉（不然会显示上一个角色的源帧与选中）
    setEmoFrames(null); setEmoSel([]); setEmoMsg(''); setEmoProg(null)
    void fetchCharacter(sel)
      .then((d) => { if (alive) { setDetail(d); setDraft(d.config.character_info ?? {}) } })
      .catch((e) => { if (alive) setErr(String(e instanceof Error ? e.message : e)) })
    void fetchCharacterPrompt(sel, '', undefined, enhance)
      .then((p) => { if (alive) setCPrompt(p) })
      .catch(() => { if (alive) setCPrompt(null) })
    return () => { alive = false }
  }, [sel])

  const flatStyles = useMemo(() => {
    const out: Array<StyleEntry & { group: string }> = []
    for (const g of styles?.groups ?? []) {
      for (const s of g.styles ?? []) {
        out.push({ ...s, group: g.label || g.name || '' })
      }
    }
    return out
  }, [styles])

  const flatEmotions = useMemo(() => {
    const out: Array<EmotionEntry & { category: string }> = []
    for (const [cat, list] of Object.entries(emotions ?? {})) {
      for (const e of Array.isArray(list) ? list : []) {
        out.push({ ...e, category: cat })
      }
    }
    return out
  }, [emotions])

  const doCreate = useCallback(async () => {
    const name = newName.trim()
    if (!name) return
    try {
      await createCharacter(name)
      setNewName('')
      await reload()
      setSel(name)
      onNotice(`已建角色「${name}」`)
    } catch (e) {
      setErr(`建角色失败：${e instanceof Error ? e.message : e}`)
    }
  }, [newName, reload, onNotice])

  const doRename = useCallback(async () => {
    if (!sel) return
    const next = window.prompt('新名字', sel)
    if (!next || next === sel) return
    try {
      await renameCharacter(sel, next.trim())
      await reload()
      setSel(next.trim())
    } catch (e) {
      setErr(`改名失败：${e instanceof Error ? e.message : e}`)
    }
  }, [sel, reload])

  const doDelete = useCallback(async (name: string) => {
    if (!window.confirm(`删掉角色「${name}」？\n材质（所有 sprite / 服装 / 表情）会一起删。`)) return
    try {
      await deleteCharacter(name)
      if (sel === name) setSel(null)
      await reload()
    } catch (e) {
      setErr(`删除失败：${e instanceof Error ? e.message : e}`)
    }
  }, [sel, reload])

  const saveInfo = useCallback(async (patch: Record<string, unknown>) => {
    if (!sel) return
    try {
      await updateCharacterInfo(sel, patch)
      const d = await fetchCharacter(sel)
      setDetail(d)
      // prompt 是属性拼出来的，属性变了它就得跟着变
      const p = await fetchCharacterPrompt(sel, '', undefined, enhance).catch(() => null)
      if (p) setCPrompt(p)
      onNotice('已保存')
    } catch (e) {
      setErr(`保存失败：${e instanceof Error ? e.message : e}`)
    }
  }, [sel, onNotice])

  /**
   * 从一张图反推属性（官方 Cloner 旁路，要本地 LLM）。
   *
   * **只拿结果，不落盘**：拿到后摆在下面让用户逐条看一眼，点「用这些属性」
   * 才写进 config。属性错了会一路影响后面所有出图（prompt 是它拼的），
   * 不能替用户决定。
   */
  const doAnalyze = useCallback(async () => {
    if (!sel || anaBusy) return
    setAnaBusy(true)
    setAnaMsg('正在看图（本地 4B 模型，约 20~40 秒）…')
    setAnaResult(null)
    try {
      const r = await analyzeCharacter(sel, {
        background_color: String(draft.background_color ?? ''),
      })
      setAnaResult(r.fields)
      setAnaMsg(`读完了（${r.seconds}s）——下面是它看到的，确认无误再点「用这些属性」`)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      // 后端的 detail 是 {error: CODE, message}；按码给不同的下一步建议
      let hint = ''
      if (msg.includes('DEPENDENCY_MISSING')) hint = '（llama-cpp-python 装错了 fork，需要 JamePeng 版）'
      else if (msg.includes('MODEL_MISSING') || msg.includes('MMPROJ_MISSING')) hint = '（models/llm/Qwen3.5-4B 下的 gguf 没装齐）'
      else if (msg.includes('TIMEOUT')) hint = '（超时了，模型还占着，等半分钟再试）'
      else if (msg.includes('PARSE_ERROR') || msg.includes('INVALID_RESPONSE')) hint = '（模型没答成 JSON，重试一次通常就好）'
      setAnaMsg(`反推失败：${msg}${hint}`)
    } finally {
      setAnaBusy(false)
    }
  }, [sel, anaBusy, draft])

  /**
   * 一句话造角色（官方 AI 向导，纯文本）。
   *
   * 跟看图反推共用一块结果面板：生成 → 用户逐条看 → 确认才保存。
   */
  const doWizard = useCallback(async () => {
    const text = wizText.trim()
    if (!sel || wizBusy || !text) return
    setWizBusy(true)
    setAnaMsg('正在生成（本地 4B 模型，约 30~60 秒）…')
    setAnaResult(null)
    try {
      const r = await wizardCharacter(text)
      setAnaResult(r.fields)
      setAnaMsg(`生成好了（${r.seconds}s）——不满意可以改描述再来一次，或直接手改下面字段`)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setAnaMsg(`生成失败：${msg}`)
    } finally {
      setWizBusy(false)
    }
  }, [sel, wizBusy, wizText])

  /**
   * 换装：把当前服装的五个字段变成一张**换装底图**（官方 Clothes Designer）。
   *
   * 注意这不是"存描述下次拼 prompt"——它是真的出一张图，落到
   * `Sprites/<服装>/neutral/sprite_pose_0001.png`，后面出姿势/表情都从它走。
   */
  const doDesignClothes = useCallback(async () => {
    if (!sel || clothBusy) return
    setClothBusy(true)
    setClothMsg('已提交，排队中…（25 步，约 1 分钟）')
    try {
      // 先把表单里的字段存下来：换装读的是 config 里的服装信息
      await saveCostume(sel, costume, costumeDraft)
      const r = await designClothes(sel, { costume })
      for (let i = 0; i < 600; i++) {
        const j = await fetchJob(r.job_id)
        if (j.status === 'done') {
          const d = await fetchCharacter(sel)
          setDetail(d)
          setBrowseCostume(costume)
          setBrowseEmotion('neutral')
          setClothMsg('换装完成，已存成这套服装的底图（素材页可看）')
          onNotice(`「${costume}」换装完成`)
          return
        }
        if (j.status === 'failed') {
          setClothMsg(`换装失败：${j.error || '未知错误'}`)
          return
        }
        setClothMsg(j.progress?.current ? `换装中：${j.progress.current}` : '换装中…')
        await new Promise((res) => setTimeout(res, 1200))
      }
      setClothMsg('等太久了：任务还在跑，去画布那边看任务状态')
    } catch (e) {
      setClothMsg(`换装失败：${e instanceof Error ? e.message : e}`)
    } finally {
      setClothBusy(false)
    }
  }, [sel, costume, costumeDraft, clothBusy, onNotice])

  /** 一句话拆服装：五字段填进表单，用户改完再保存（AI 向导不落盘）。 */
  const doClothesWizard = useCallback(async () => {
    const text = clothText.trim()
    if (!sel || clothBusy || !text) return
    setClothBusy(true)
    setClothMsg('正在拆解（本地 4B 模型，约 20~40 秒）…')
    try {
      const r = await clothesWizard(sel, text)
      const fields = r.fields as Record<string, string>
      setCostumeDraft((d) => ({ ...d, ...fields }))
      setClothMsg(`拆好了（${r.seconds}s）——检查下面的字段，确认后点「保存服装」`)
    } catch (e) {
      setClothMsg(`拆解失败：${e instanceof Error ? e.message : e}`)
    } finally {
      setClothBusy(false)
    }
  }, [sel, clothBusy, clothText])

  /**
   * 多姿势批量：一个骨架帧目录 → 一整批精灵帧（官方 pose_generation）。
   *
   * ⚠️ **整批替换**：这套服装现有的 Neutral 帧会被推进 `V1/`（能找回）。
   * 跟表情那条（逐张追加）不是一回事，界面上要说清。
   */
  const doGeneratePoses = useCallback(async () => {
    if (!sel || poseBusy || !poseDir) return
    setPoseBusy(true)
    setPoseProg(null)
    setPoseMsg('已提交，排队中…（每张 20 步，约 20~30 秒/张）')
    try {
      const r = await generatePoses(sel, {
        costume: browseCostume,
        pose_dir: poseDir,
        prompts: poseText.trim(),
        matte: true,
      })
      setPoseProg({ done: 0, total: r.total })
      for (let i = 0; i < 1800; i++) {
        const j = await fetchJob(r.job_id)
        if (j.status === 'done') {
          const d = await fetchCharacter(sel)
          setDetail(d)
          setBrowseEmotion('neutral')
          setPoseMsg(`完成 ${r.total} 张：旧帧已进 V1，素材页可看`)
          onNotice('多姿势生成完成')
          return
        }
        if (j.status === 'failed') {
          setPoseMsg(`生成失败：${j.error || '未知错误'}`)
          return
        }
        if (j.progress) {
          setPoseProg({ done: j.progress.done, total: j.progress.total })
          setPoseMsg(j.progress.current || '生成中…')
        }
        await new Promise((res) => setTimeout(res, 1200))
      }
      setPoseMsg('等太久了：任务还在跑，去画布那边看任务状态')
    } catch (e) {
      setPoseMsg(`生成失败：${e instanceof Error ? e.message : e}`)
    } finally {
      setPoseBusy(false)
    }
  }, [sel, poseBusy, poseDir, poseText, browseCostume, onNotice])

  /**
   * 导出精灵图集（`atlas.png` + `atlas.json`）。
   *
   * 这是**接引擎那一步**：官方只管把帧出到 `Sprites/<服装>/<表情>/`，
   * 引擎要的是"一张大图 + 坐标表"。对齐规则是统一画布 + 底边对齐，
   * 否则动画里角色会上下抖。
   */
  const doExportAtlas = useCallback(async () => {
    if (!sel || atlasBusy) return
    setAtlasBusy(true)
    setAtlasMsg('打包中…')
    try {
      const r = await exportAtlas(sel)
      const kb = Math.max(1, Math.round(r.bytes / 1024))
      setAtlasMsg(
        `完成：${r.frames} 帧 / ${r.animations} 组动画，`
        + `图集 ${r.size[0]}×${r.size[1]}，单帧画布 ${r.canvas[0]}×${r.canvas[1]}，${kb} KB`)
      setAtlasInfo(await fetchAtlasInfo(sel))
      onNotice('图集已导出')
    } catch (e) {
      setAtlasMsg(`导出失败：${e instanceof Error ? e.message : e}`)
    } finally {
      setAtlasBusy(false)
    }
  }, [sel, atlasBusy, onNotice])

  /**
   * 换白色内衣底图（官方 `remove_clothes`）。
   *
   * **是一套具体服装，不是"去掉衣服"**：官方就是一条换装指令加 Clothes LoRA，
   * 产物给后续每套服装当统一起点（这样新衣服不会带着上一套的袖子残留）。
   */
  const doUnderwear = useCallback(async () => {
    if (!sel || uwBusy) return
    setUwBusy(true)
    setUwMsg('已提交，排队中…（20 步，约 40 秒）')
    try {
      const r = await designUnderwear(sel, {})
      for (let i = 0; i < 600; i++) {
        const j = await fetchJob(r.job_id)
        if (j.status === 'done') {
          const d = await fetchCharacter(sel)
          setDetail(d)
          setBrowseCostume(r.costume)
          setBrowseEmotion('neutral')
          setUwMsg(`完成：已存成「${r.costume}」底图（素材页可看）`)
          onNotice(`内衣底图已生成`)
          return
        }
        if (j.status === 'failed') {
          setUwMsg(`失败：${j.error || '未知错误'}`)
          return
        }
        setUwMsg(j.progress?.current ? `生成中：${j.progress.current}` : '生成中…')
        await new Promise((res) => setTimeout(res, 1200))
      }
      setUwMsg('等太久了：任务还在跑，去画布那边看任务状态')
    } catch (e) {
      setUwMsg(`失败：${e instanceof Error ? e.message : e}`)
    } finally {
      setUwBusy(false)
    }
  }, [sel, uwBusy, onNotice])

  /** 出图：提交 → 轮询 → 完成后后端已经把图收进角色目录了，刷新即可 */
  const doGenerate = useCallback(async (workflow: 'w1' | 'w2') => {
    if (!sel || genBusy) return
    setGenBusy(true)
    setGenMsg('已提交，排队中…')
    try {
      const r = await generateForCharacter(sel, {
        workflow,
        costume,
        emotion: 'neutral',
        prompt: cPrompt?.prompt || '',
        negative: cPrompt?.negative || '',
        enhance,
        matte,
      })
      for (let i = 0; i < 400; i++) {
        const j = await fetchJob(r.job_id)
        if (j.status === 'done') {
          const d = await fetchCharacter(sel)
          setDetail(d)
          setTab('assets')
          setGenMsg(d.assets[costume]?.neutral?.length
            ? '已收进角色目录（素材页可看）'
            : '出图完成了，但没收到角色目录里：'
              + '看任务结果里的 publish_error')
          onNotice('出图完成')
          return
        }
        if (j.status === 'failed') {
          setGenMsg(`生成失败：${j.error || '未知错误'}`)
          return
        }
        setGenMsg('生成中…')
        await new Promise((res) => setTimeout(res, 1000))
      }
      setGenMsg('等太久了：任务还在跑，去画布那边看任务状态')
    } catch (e) {
      setGenMsg(`出图失败：${e instanceof Error ? e.message : e}`)
    } finally {
      setGenBusy(false)
    }
  }, [sel, genBusy, costume, cPrompt, onNotice])

  /**
   * 表情批量生成：先检脸框（不跑生成模型），让用户看到"会在哪几张上改脸"。
   *
   * **先检再生成**是官方 Emotion Studio 的顺序：没有源帧或一张脸都没检出时
   * 要立刻说清楚，别等提交完再报错——一批几十张，等 3 分钟才被告知"没源帧"
   * 是最糟的体验。
   */
  const doDetect = useCallback(async () => {
    if (!sel || emoBusy) return
    setEmoMsg('检测脸框…')
    try {
      const r = await detectEmotionFaces(sel, { costume })
      setEmoFrames(r.items)
      const hit = r.items.filter((f) => f.detected).length
      setEmoMsg(r.count === 0
        ? `「${costume}」没有源帧：表情是从 Neutral 姿势帧上改脸的，先去「出图」页生成一张`
        : `${r.count} 张源帧，${hit} 张检出脸框` +
          (hit < r.count ? '（没检出的整图重画，官方行为）' : ''))
    } catch (e) {
      setEmoMsg(`检测失败：${e instanceof Error ? e.message : e}`)
    }
  }, [sel, costume, emoBusy])

  const doGenerateEmotions = useCallback(async () => {
    if (!sel || emoBusy || emoSel.length === 0) return
    setEmoBusy(true)
    setEmoProg(null)
    setEmoMsg('已提交，排队中…')
    try {
      const r = await generateEmotions(sel, {
        costume,
        emotions: emoSel,
        steps: emoParams.steps,
        target_size: emoParams.target_size,
        matte: emoParams.matte,
      })
      setEmoProg({ done: 0, total: r.total })
      setEmoMsg(`排队第 ${r.position} 位，共 ${r.total} 张`)
      for (let i = 0; i < 1800; i++) {
        const j = await fetchJob(r.job_id)
        if (j.status === 'done') {
          const d = await fetchCharacter(sel)
          setDetail(d)
          const res = (j.result || {}) as {
            count?: number; failed?: Array<{ error: string }>
          }
          const bad = res.failed?.length ?? 0
          setEmoMsg(bad
            ? `完成 ${res.count ?? 0} 张，${bad} 张失败：${res.failed?.[0]?.error || ''}`
            : `完成 ${res.count ?? 0} 张，已收进角色目录`)
          setBrowseCostume(costume)
          setBrowseEmotion(emoSel[0])
          setTab('assets')
          onNotice(`表情生成完成（${res.count ?? 0} 张）`)
          return
        }
        if (j.status === 'failed') {
          setEmoMsg(`生成失败：${j.error || '未知错误'}`)
          return
        }
        const p = j.progress
        if (p) {
          setEmoProg({ done: p.done, total: p.total })
          setEmoMsg(`${p.done}/${p.total}${p.current ? ` · ${p.current}` : ''}`)
        }
        await new Promise((res) => setTimeout(res, 1200))
      }
      setEmoMsg('等太久了：任务还在跑，去画布那边看任务状态')
    } catch (e) {
      setEmoMsg(`生成失败：${e instanceof Error ? e.message : e}`)
    } finally {
      setEmoBusy(false)
    }
  }, [sel, costume, emoSel, emoParams, emoBusy, onNotice])

  /** 素材页能选哪些表情：服务端该服装下真有的 + 保底 neutral（空素材也要能选） */
  const browseEmotions = useMemo(() => {
    const had = Object.keys(detail?.assets?.[browseCostume] || {})
    return had.length ? had : ['neutral']
  }, [detail, browseCostume])

  // 服装字段的草稿：换角色 / 换服装时从 config 载入。
  // **必须回填**：AI 向导把五字段灌进来之后，用户还要能在表单上改。
  useEffect(() => {
    setCostumeDraft({ ...(detail?.config.costumes?.[costume] ?? {}) })
  }, [detail, costume])

  const saveCostumeFields = useCallback(async (data: Record<string, string>) => {
    if (!sel) return
    try {
      await saveCostume(sel, costume, data)
      const d = await fetchCharacter(sel)
      setDetail(d)
      onNotice(`服装「${costume}」已保存`)
    } catch (e) {
      setErr(`保存服装失败：${e instanceof Error ? e.message : e}`)
    }
  }, [sel, costume, onNotice])

  const doDeleteCostume = useCallback(async () => {
    if (!sel || costume === 'Naked') return
    try {
      await deleteCostume(sel, costume)
      setCostume('Naked')
      const d = await fetchCharacter(sel)
      setDetail(d)
    } catch (e) {
      setErr(`删服装失败：${e instanceof Error ? e.message : e}`)
    }
  }, [sel, costume])

  const currentStyle = String(draft.style ?? '')

  return (
    <div className="chars">
      {/* ---------- 左：角色库 ---------- */}
      <div className="chars-list">
        <div className="chars-list-head">
          <b>角色库</b>
          <span className="chars-count">{items.length}</span>
        </div>
        <div className="chars-new">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="新角色名（字母/数字/空格）"
            onKeyDown={(e) => { if (e.key === 'Enter') void doCreate() }}
          />
          <button type="button" className="mini" onClick={() => void doCreate()}
            disabled={!newName.trim()}>
            新建
          </button>
        </div>
        {loading && <p className="chars-empty">正在读角色库…</p>}
        {!loading && items.length === 0 && (
          <p className="chars-empty">
            还没有角色。右上角新建一个，或者把官方工作流产出的角色目录直接拷进
            <code>assets/characters/</code>——结构是同一套，能直接读。
          </p>
        )}
        <div className="chars-grid">
          {items.map((c) => (
            <button
              type="button"
              key={c.name}
              className={`chars-card${sel === c.name ? ' on' : ''}`}
              onClick={() => setSel(c.name)}
            >
              {c.cover_url
                ? <img src={c.cover_url} alt={c.name} loading="lazy" />
                : <span className="chars-nocover">无封面</span>}
              <span className="chars-card-name">{c.name}</span>
              <span className="chars-card-meta">
                {c.costumes.length} 服装 · {c.sprites} 图
              </span>
              <span
                className="chars-card-del"
                title="删除角色"
                onClick={(e) => { e.stopPropagation(); void doDelete(c.name) }}
              >
                ✕
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* ---------- 右：角色详情 ---------- */}
      <div className="chars-detail">
        {err && <div className="chars-err">⚠ {err}</div>}
        {!sel && (
          <p className="chars-empty">
            左边选一个角色。这里是官方 Character Creator 的那套属性——
            性别/年龄/种族/发色/瞳色/身材 + 358 条画风 + 服装 + 157 条表情库。
          </p>
        )}

        {sel && detail && (
          <>
            <div className="chars-detail-head">
              <b>{detail.name}</b>
              <span className="chars-sub">
                {detail.config.config_version ? `config v${detail.config.config_version} · ` : ''}
                {detail.costumes.join(' / ')}
              </span>
              <span className="spacer" />
              <button type="button" className="mini" onClick={() => void doRename()}>改名</button>
            </div>
            <div className="chars-tabs">
              {([['info', '属性'], ['gen', '出图'], ['clothes', '服装'],
                ['emotions', '表情库'], ['assets', '素材']] as const).map(([k, label]) => (
                <button
                  type="button" key={k}
                  className={`chars-tab${tab === k ? ' on' : ''}`}
                  onClick={() => setTab(k)}
                >
                  {label}
                </button>
              ))}
            </div>

            {tab === 'info' && (
              <div className="chars-body">
                <div className="chars-row">
                  <button type="button" className="mini" onClick={() => void doAnalyze()}
                    disabled={anaBusy}
                    title="用本地 4B 模型看一张图，读出身高体型/发色瞳色这些可见特征">
                    {anaBusy ? '看图反推中…' : '从图片反推属性'}
                  </button>
                  <span className="chars-sub">
                    官方 Cloner 的做法：拿一张图读<b>看得见的身体特征</b>
                    （不含服装 / 姿势 / 表情），读完先给你看，确认了才写进角色。
                  </span>
                </div>
                <div className="chars-row">
                  <input
                    className="chars-wizard-input"
                    placeholder="一句话造角色，例：一个苍白的精灵女剑士，银发红瞳，身材纤细"
                    value={wizText}
                    onChange={(e) => setWizText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') void doWizard() }}
                  />
                  <button type="button" className="mini" onClick={() => void doWizard()}
                    disabled={wizBusy || !wizText.trim()}
                    title="官方 AI 向导：把一句话拆成 9 个结构化属性（不落盘，先给你看）">
                    {wizBusy ? '生成中…' : '用一句话生成'}
                  </button>
                </div>
                {anaMsg && <p className="chars-sub">{anaMsg}</p>}
                {anaResult && (
                  <div className="chars-analyze">
                    <table>
                      <tbody>
                        {Object.entries(anaResult).map(([k, v]) => (
                          <tr key={k}>
                            <th>{k}</th>
                            <td>{String(v)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="chars-row">
                      <button type="button" className="mini wide" onClick={() => {
                        const patch: Record<string, unknown> = {}
                        for (const [k, v] of Object.entries(anaResult)) {
                          if (v === '' || v === null || v === undefined) continue
                          patch[k] = v
                        }
                        setDraft((d) => ({ ...d, ...patch }))
                        setAnaResult(null)
                        setAnaMsg('')
                        void saveInfo(patch)
                      }}>
                        用这些属性（覆盖同名项）
                      </button>
                      <button type="button" className="mini" onClick={() => {
                        setAnaResult(null); setAnaMsg('')
                      }}>
                        放弃
                      </button>
                    </div>
                  </div>
                )}
                <div className="chars-row">
                  <label>
                    性别
                    <select
                      value={String(draft.sex ?? 'female')}
                      onChange={(e) => {
                        setDraft((d) => ({ ...d, sex: e.target.value }))
                        void saveInfo({ sex: e.target.value })
                      }}
                    >
                      <option value="female">女</option>
                      <option value="male">男</option>
                    </select>
                  </label>
                  {INFO_NUM.map(([k, label]) => (
                    <label key={k}>
                      {label}
                      <input
                        type="number"
                        value={Number(draft[k] ?? 18)}
                        onChange={(e) => setDraft((d) => ({ ...d, [k]: Number(e.target.value) }))}
                        onBlur={(e) => void saveInfo({ [k]: Number(e.target.value) })}
                      />
                    </label>
                  ))}
                  <label>
                    背景色
                    <select
                      value={String(draft.background_color ?? 'green')}
                      onChange={(e) => {
                        setDraft((d) => ({ ...d, background_color: e.target.value }))
                        void saveInfo({ background_color: e.target.value })
                      }}
                      title="出图时的幕布色，抠图按它取样（官方 background_color）"
                    >
                      {BG_COLORS.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </label>
                </div>

                {INFO_TEXT.map(([k, label, hint]) => (
                  <label className="chars-field" key={k} title={hint || ''}>
                    {label}
                    <span className="chars-field-row">
                      <input
                        value={String(draft[k] ?? '')}
                        placeholder={hint || ''}
                        onChange={(e) => setDraft((d) => ({ ...d, [k]: e.target.value }))}
                        onBlur={(e) => void saveInfo({ [k]: e.target.value })}
                      />
                      {/* 官方 Character Creator 的「Choose Presets」：手填容易中带来
                          一堆近义写法（"blue eyes" vs "blue_eyes"），点选能对齐官方措辞 */}
                      {INFO_TAGS.has(k) && (
                        <button
                          type="button" className="mini"
                          onClick={() => setPicking(k)}
                          title="从官方标签库里选（可多选）"
                        >
                          选
                        </button>
                      )}
                    </span>
                  </label>
                ))}

                <div className="chars-style">
                  <div className="chars-style-head">
                    <b>画风</b>
                    <span className="chars-sub">
                      {flatStyles.length} 条 · 官方 character_styles.json
                    </span>
                    <span className="spacer" />
                    <span className="chars-sub">当前：{currentStyle || '（无）'}</span>
                  </div>
                  <div className="chars-style-grid">
                    {flatStyles.slice(0, 60).map((s) => (
                      <button
                        type="button" key={s.id}
                        className={`chars-style-card${currentStyle === s.id ? ' on' : ''}`}
                        title={`${s.group} · ${s.description || s.prompt.slice(0, 60)}`}
                        onClick={() => {
                          setDraft((d) => ({ ...d, style: s.id }))
                          void saveInfo({ style: s.id })
                        }}
                      >
                        <img src={stylePreviewUrl(s.id)} alt={s.label} loading="lazy" />
                        <span>{s.label}</span>
                      </button>
                    ))}
                  </div>
                  {flatStyles.length > 60 && (
                    <p className="chars-sub">
                      先显示前 60 条（共 {flatStyles.length}）。
                      风格预览图是压缩过的 192px webp，原图在源仓库。
                    </p>
                  )}
                </div>
              </div>
            )}

            {tab === 'gen' && (
              <div className="chars-body">
                <p className="chars-sub">
                  这是<b>官方 QI2 的结构化 prompt</b>：每个字段一行，不是一句自然语言。
                  风格 LoRA 就是照这个格式调的，自己另写一句会明显偏。
                </p>
                <label className="chars-check">
                  <input type="checkbox" checked={enhance}
                    onChange={(e) => {
                      setEnhance(e.target.checked)
                      if (sel) {
                        fetchCharacterPrompt(sel, '', undefined, e.target.checked)
                          .then(setCPrompt).catch(() => {})
                      }
                    }} />
                  Prompt Enhancement（官方默认开）
                  <span className="chars-sub">
                    用<b>图像模型自己的 Qwen3-VL</b> 改写——不新增权重，且改写出来的
                    措辞跟模型训练时见到的文本同分布，风格跟随更准。
                    代价：编码器要常驻 GPU 跑自回归解码（显存/耗时，见评估文档 §4.1）。
                    失败会自动退回下面的原始 prompt。
                  </span>
                </label>
                {cPrompt?.enhanced ? (
                  <p className="chars-sub">✓ 已改写（下面是扩写后的版本）</p>
                ) : (
                  <p className="chars-sub">
                    · 当前是<b>未扩写</b>版（PE 关掉了，或这次没生效）
                  </p>
                )}
                <label className="chars-field">
                  正向 prompt（可改，改了就用你这份出图）
                  <textarea
                    className="chars-prompt"
                    value={cPrompt?.prompt ?? ''}
                    onChange={(e) => setCPrompt((p) => (p ? { ...p, prompt: e.target.value } : p))}
                  />
                </label>
                <label className="chars-field">
                  负向词
                  <textarea
                    className="chars-prompt short"
                    value={cPrompt?.negative ?? ''}
                    onChange={(e) => setCPrompt((p) => (p ? { ...p, negative: e.target.value } : p))}
                  />
                </label>
                <div className="chars-row">
                  <span className="chars-sub">
                    分辨率 {cPrompt ? `${cPrompt.resolution[0]}×${cPrompt.resolution[1]}` : '—'}
                    （官方公式：按总像素等比，对齐到 latent 步长 8）
                  </span>
                  <span className="chars-sub">
                    {cPrompt ? `${cPrompt.defaults.steps} 步 / cfg ${cPrompt.defaults.cfg}` : ''}
                  </span>
                </div>
                <div className="chars-row">
                  <button type="button" className="mini wide" onClick={() => void doGenerate('w1')}
                    disabled={genBusy || !cPrompt}>
                    {genBusy ? '生成中…' : '生成底图（文生图）'}
                  </button>
                  <button type="button" className="mini" onClick={() => void doGenerate('w2')}
                    disabled={genBusy || !cPrompt}
                    title="以当前这套服装的最新一张为底图，按服装描述改">
                    换装（图生图）
                  </button>
                  <button type="button" className="mini"
                    onClick={() => sel && fetchCharacterPrompt(sel, '', undefined, enhance)
                      .then(setCPrompt).catch(() => {})}>
                    重拼 prompt
                  </button>
                </div>
                <label className="chars-check">
                  <input type="checkbox" checked={matte}
                    onChange={(e) => setMatte(e.target.checked)} />
                  抠掉纯色幕布（官方 bg_remove）
                  <span className="chars-sub">
                    用官方那套纯 torch 抠像（边框估幕布色 → 去溢色 → 边缘净化），
                    <b>不加载任何模型</b>。抠完另存一份 <code>_matte.png</code>，
                    原图保留。白底/浅色角色慎用——幕布色是按边框取的，
                    白毛白甲会被一起判成背景。
                  </span>
                </label>

                {genMsg && <p className="chars-sub">{genMsg}</p>}
              </div>
            )}

            {tab === 'clothes' && (
              <div className="chars-body">
                <div className="chars-row">
                  <label>
                    服装
                    <select value={costume} onChange={(e) => setCostume(e.target.value)}>
                      {detail.costumes.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </label>
                  <button type="button" className="mini"
                    onClick={() => {
                      const n = window.prompt('新服装名')
                      if (n?.trim()) setCostume(n.trim())
                    }}>
                    新建服装
                  </button>
                  <button type="button" className="mini"
                    onClick={() => void doDeleteCostume()}
                    disabled={costume === 'Naked'}
                    title="Naked / Original 是基础集，不能删">
                    删除
                  </button>
                </div>
                <p className="chars-sub">
                  字段照抄官方 Clothes Designer（top/bottom/head/shoes/face/negative_prompt），
                  存进 <code>{(detail.name)}_config.json</code> 的 costumes。
                  <br />
                  <b>换装是"把描述变成一张图"</b>：把五个字段拼成
                  <code>Dress the character:</code> 指令，拿裸装底图重画一张，
                  存成这套服装的 <code>Sprites/&lt;服装&gt;/neutral/</code> 底图。
                  之后出姿势/表情都从它出发。
                </p>

                <div className="chars-row">
                  <input
                    className="chars-wizard-input"
                    placeholder="一句话拆服装，例：赛博朋克护士，白大褂 + 荧光绿内衬 + 护目镜"
                    value={clothText}
                    onChange={(e) => setClothText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') void doClothesWizard() }}
                  />
                  <button type="button" className="mini" onClick={() => void doClothesWizard()}
                    disabled={clothBusy || !clothText.trim()}
                    title="官方 Clothes Wizard：把一句话拆成五个服装字段（不落盘）">
                    {clothBusy ? '处理中…' : 'AI 拆服装'}
                  </button>
                </div>
                <CostumeForm
                  key={`${detail.name}/${costume}`}
                  fields={costumeDraft}
                  onChange={setCostumeDraft}
                  onSave={(data) => void saveCostumeFields(data)}
                />
                <div className="chars-row">
                  <button type="button" className="mini"
                    onClick={() => void doUnderwear()}
                    disabled={uwBusy}
                    title="官方 remove_clothes：换成白色内衣，得到一张干净底图（后续每套服装都从它出发）">
                    {uwBusy ? '处理中…' : '换白内衣底图'}
                  </button>
                  <span className="chars-sub">
                    <b>换白内衣</b>是官方的预处理一步：把角色换成
                    <code>Dress character: White underwear</code>，
                    得到一张干净底图（默认存成 <code>Underwear</code> 服装）。
                    后面每套衣服都从它出发，新衣服就不会带着上一套的袖子残留。
                  </span>
                </div>
                {uwMsg && <p className="chars-sub">{uwMsg}</p>}
                <div className="chars-row">
                  <button type="button" className="mini wide"
                    onClick={() => void doDesignClothes()}
                    disabled={clothBusy || costume === 'Naked' || costume === 'Original'}
                    title={costume === 'Naked' || costume === 'Original'
                      ? '裸装/原始是基础集，换装要挑一套具体的衣服'
                      : '挂 Clothes Core LoRA 出这张换装底图（25 步，约 1 分钟）'}>
                    {clothBusy ? '换装中…' : `出换装底图（${costume}）`}
                  </button>
                </div>
                {clothMsg && <p className="chars-sub">{clothMsg}</p>}
              </div>
            )}

            {tab === 'emotions' && (
              <div className="chars-body">
                <p className="chars-sub">
                  官方表情库 {flatEmotions.length} 条（9 个分类，全量原样入库）。
                  每条带 <code>natural_prompt</code>——QI2 直接用它当指令。
                  <br />
                  <b>表情是"只改脸"</b>：把脸框裁出来放大重画，再按羽化蒙版贴回原图，
                  所以脖子、衣服、动作逐像素不变。**点表情卡选中**，再点「批量生成」。
                </p>

                <div className="chars-row">
                  <button type="button" className="mini" onClick={() => void doDetect()}
                    disabled={emoBusy} title="跑 YOLO 找人脸框，不跑生成模型">
                    检出脸框
                  </button>
                  <span className="chars-sub">
                    {emoFrames
                      ? `源帧 ${emoFrames.length} 张（${costume}）`
                      : `尚未检测（${costume} 的 Neutral 姿势帧就是源帧）`}
                  </span>
                  {emoFrames && emoFrames.filter((f) => f.box).length > 0 && (
                    <span className="chars-sub">
                      脸框 {emoFrames.filter((f) => f.box).map(
                        (f) => `#${f.index} ${f.box![2] - f.box![0]}×${f.box![3] - f.box![1]}`).join(' / ')}
                    </span>
                  )}
                </div>

                <div className="chars-row">
                  <button type="button" className="mini"
                    onClick={() => setEmoSel(flatEmotions.map((e) => e.safe_name))}>
                    全选
                  </button>
                  <button type="button" className="mini" onClick={() => setEmoSel([])}>
                    清空
                  </button>
                  <span className="chars-sub">已选 {emoSel.length} 个</span>
                </div>

                <div className="chars-emo-grid">
                  {flatEmotions.map((e) => {
                    const on = emoSel.includes(e.safe_name)
                    return (
                      <button
                        type="button"
                        className={`chars-emo${on ? ' on' : ''}`}
                        key={`${e.category}/${e.key}`}
                        title={e.natural_prompt || e.key}
                        onClick={() => setEmoSel((list) => (on
                          ? list.filter((x) => x !== e.safe_name)
                          : [...list, e.safe_name]))}
                      >
                        <img src={emotionIconUrl(e.safe_name)} alt={e.key} loading="lazy" />
                        <span>{e.key}</span>
                      </button>
                    )
                  })}
                </div>

                <div className="chars-row">
                  <label>
                    目标像素档
                    <select
                      value={emoParams.target_size}
                      onChange={(ev) => setEmoParams((p) => ({
                        ...p, target_size: Number(ev.target.value) }))}
                    >
                      <option value={2048}>2048（官方默认，2.0 MP）</option>
                      <option value={1024}>1024（1.0 MP，快一倍）</option>
                    </select>
                  </label>
                  <label>
                    步数
                    <input
                      type="number" min={8} max={60} value={emoParams.steps}
                      onChange={(ev) => setEmoParams((p) => ({
                        ...p, steps: Number(ev.target.value) || 25 }))}
                    />
                  </label>
                  <label className="chars-check">
                    <input type="checkbox" checked={emoParams.matte}
                      onChange={(ev) => setEmoParams((p) => ({ ...p, matte: ev.target.checked }))} />
                    贴回后抠纯色幕布
                  </label>
                </div>
                <p className="chars-sub">
                  推理边长封顶 1024（24GB 单卡在 1440² 上单张 60s 以上且显存压线）。
                  目标像素档决定官方公式算出的裁切尺寸，封顶值再压一档。
                  <b>实测一张到底多久</b>：1024² / 8 步 ≈ 43s，25 步 ≈ 110s。
                  所以「姿势数 × 表情数」要算着来——3 帧 × 5 表情 ≈ 27 分钟。
                </p>

                <div className="chars-row">
                  <button type="button" className="mini wide"
                    onClick={() => void doGenerateEmotions()}
                    disabled={emoBusy || emoSel.length === 0}>
                    {emoBusy ? '生成中…' : `批量生成（${emoSel.length} 个表情）`}
                  </button>
                </div>
                {emoProg && emoProg.total > 0 && (
                  <div className="chars-prog">
                    <div className="chars-prog-bar"
                      style={{ width: `${Math.round((emoProg.done / emoProg.total) * 100)}%` }} />
                    <span>{emoProg.done}/{emoProg.total}</span>
                  </div>
                )}
                {emoMsg && <p className="chars-sub">{emoMsg}</p>}
              </div>
            )}

            {tab === 'assets' && (
              <div className="chars-body">
                <p className="chars-sub">
                  翻这一套角色用过哪些帧。路径
                  <code>assets/characters/{detail.name}/Sprites/&lt;服装&gt;/&lt;表情&gt;/</code>
                </p>
                <div className="chars-row">
                  <label>
                    服装
                    <select value={browseCostume} onChange={(e) => setBrowseCostume(e.target.value)}>
                      {detail.costumes.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </label>
                  <label>
                    表情
                    <select value={browseEmotion} onChange={(e) => setBrowseEmotion(e.target.value)}>
                      {browseEmotions.map((e) => <option key={e} value={e}>{e}</option>)}
                    </select>
                  </label>
                </div>
                <SpritePreview
                  character={detail.name}
                  costume={browseCostume}
                  emotion={browseEmotion}
                  onMeta={setBrowseMeta}
                />
                <div className="chars-posegen">
                  <b>多姿势批量</b>
                  <p className="chars-sub">
                    一个骨架帧目录 → 一整批 <code>sprite_pose_0001..N</code>。
                    整批<b>共用一个种子</b>（官方做法：每张随机会让衣服/发型飘）。
                    <br />
                    ⚠️ 这是<b>整批替换</b>：这套服装现有的 Neutral 帧会被推进
                    <code>V1/</code>（能找回）。表情那条是逐张追加，两者不一样。
                  </p>
                  <div className="chars-row">
                    <label>
                      骨架帧目录
                      <select value={poseDir} onChange={(e) => setPoseDir(e.target.value)}>
                        {poseDirs.length === 0 && <option value="">（没有可用目录）</option>}
                        {poseDirs.map((d) => (
                          <option key={d.path} value={d.path}>{d.name}（{d.count} 张）</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      追加描述
                      <input
                        className="chars-wizard-input"
                        placeholder="可空；会接在官方指令后面"
                        value={poseText}
                        onChange={(e) => setPoseText(e.target.value)}
                      />
                    </label>
                  </div>
                  <div className="chars-row">
                    <button type="button" className="mini wide"
                      onClick={() => void doGeneratePoses()}
                      disabled={poseBusy || !poseDir}>
                      {poseBusy ? '生成中…' : '按这批姿势生成'}
                    </button>
                  </div>
                  {poseProg && poseProg.total > 0 && (
                    <div className="chars-prog">
                      <div className="chars-prog-bar"
                        style={{ width: `${Math.round((poseProg.done / poseProg.total) * 100)}%` }} />
                      <span>{poseProg.done}/{poseProg.total}</span>
                    </div>
                  )}
                  {poseMsg && <p className="chars-sub">{poseMsg}</p>}
                </div>

                <div className="chars-posegen">
                  <b>导出图集</b>
                  <p className="chars-sub">
                    把这个角色的所有帧打成 <code>atlas.png</code> + <code>atlas.json</code>，
                    输出在角色目录的 <code>Atlas/</code> 下。
                    <br />
                    JSON 是 TexturePacker Hash（Aseprite 能直接导入），另有一层
                    <code>animations</code>：按「服装/表情」分好组，引擎侧不用自己归类。
                    <br />
                    <b>对齐规则</b>：所有帧统一到最大宽高的画布，<b>水平居中 + 底边对齐</b>
                    —— 脚底固定在底部，动画里角色才不会上下抖。
                  </p>
                  <div className="chars-row">
                    <button type="button" className="mini wide"
                      onClick={() => void doExportAtlas()}
                      disabled={atlasBusy}>
                      {atlasBusy ? '打包中…' : '导出图集'}
                    </button>
                  </div>
                  {atlasInfo?.exists && (
                    <>
                      <p className="chars-sub">
                        现有图集：{atlasInfo.frames} 帧 / {atlasInfo.animations} 组 ·{' '}
                        {atlasInfo.size?.[0]}×{atlasInfo.size?.[1]} ·{' '}
                        {Math.max(1, Math.round((atlasInfo.bytes || 0) / 1024))} KB
                      </p>
                      {/* **必须给入口**：只显示一行路径的话，用户导出完既看不到
                          图在哪、也没法把文件拿走（浏览器里打开 /files/ 才行）。 */}
                      {atlasInfo.image && (
                        <a className="chars-atlas-link" href={atlasInfo.image}
                          target="_blank" rel="noreferrer">
                          <img src={atlasInfo.image} alt="atlas" loading="lazy" />
                          <span>打开图集图（新标签页 · 右键可另存）</span>
                        </a>
                      )}
                      <div className="chars-row">
                        <a className="mini"
                          href={atlasInfo.image?.replace(/\.png$/i, '.json')}
                          target="_blank" rel="noreferrer" download>
                          看 atlas.json（坐标表）
                        </a>
                        <span className="chars-sub">
                          同目录还有 <code>atlas.json</code>；整个{' '}
                          <code>Atlas/</code> 文件夹可以直接拷进工程。
                        </span>
                      </div>
                    </>
                  )}
                  {atlasMsg && <p className="chars-sub">{atlasMsg}</p>}
                </div>
                {browseFiles.length > 0 && (
                  <div className="chars-asset-row">
                    <span className="chars-asset-label">全部 {browseFiles.length}</span>
                    {browseFiles.map((u, i) => (
                      <img
                        key={u} src={u} alt={`第 ${i + 1} 帧`} loading="lazy"
                        title={`${browseCostume}/${browseEmotion} 第 ${i + 1} 帧`}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}

        <TagPicker
          open={!!picking}
          field={picking || ''}
          label={INFO_TEXT.find(([k]) => k === picking)?.[1] || ''}
          catalog={tagCat}
          value={String(draft[picking ?? ''] ?? '')}
          onClose={() => setPicking(null)}
          onApply={(v) => {
            const k = picking
            if (!k) return
            setDraft((d) => ({ ...d, [k]: v }))
            void saveInfo({ [k]: v })
          }}
        />
      </div>
    </div>
  )
}

/**
 * 服装字段表单（官方 COSTUME_TEXT_FIELDS 六个字段）。
 *
 * **受控**：值由父组件持有，因为「AI 拆服装」要把五个字段直接灌进表单。
 * 自己存一份 state 的话，向导填出来的东西用户看不到（踩过这个坑）。
 */
function CostumeForm({
  fields, onChange, onSave,
}: {
  fields: Record<string, string>
  onChange: (data: Record<string, string>) => void
  onSave: (data: Record<string, string>) => void
}) {
  const d: Record<string, string> = {
    top: '', bottom: '', shoes: '', head: '', face: '', negative_prompt: '', ...fields,
  }
  const labels: Array<[string, string]> = [
    ['top', '上装'], ['bottom', '下装'], ['shoes', '鞋'],
    ['head', '头部'], ['face', '脸部'], ['negative_prompt', '负向词'],
  ]
  return (
    <div className="chars-costume">
      {labels.map(([k, label]) => (
        <label className="chars-field" key={k}>
          {label}
          <input
            value={d[k] ?? ''}
            onChange={(e) => onChange({ ...d, [k]: e.target.value })}
          />
        </label>
      ))}
      <button type="button" className="mini wide" onClick={() => onSave(d)}>保存服装</button>
    </div>
  )
}
