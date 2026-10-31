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
} from '../api'

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
  const [tab, setTab] = useState<'info' | 'clothes' | 'emotions' | 'assets'>('info')
  const [newName, setNewName] = useState('')
  const [costume, setCostume] = useState('Naked')
  const [draft, setDraft] = useState<Record<string, unknown>>({})

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
  }, [reload])

  useEffect(() => {
    if (!sel) { setDetail(null); return }
    let alive = true
    void fetchCharacter(sel)
      .then((d) => { if (alive) { setDetail(d); setDraft(d.config.character_info ?? {}) } })
      .catch((e) => { if (alive) setErr(String(e instanceof Error ? e.message : e)) })
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
      onNotice('已保存')
    } catch (e) {
      setErr(`保存失败：${e instanceof Error ? e.message : e}`)
    }
  }, [sel, onNotice])

  const costumeFields = useMemo(
    () => detail?.config.costumes?.[costume] ?? {}, [detail, costume])

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
              {([['info', '属性'], ['clothes', '服装'], ['emotions', '表情库'],
                ['assets', '素材']] as const).map(([k, label]) => (
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
                    <input
                      value={String(draft[k] ?? '')}
                      placeholder={hint || ''}
                      onChange={(e) => setDraft((d) => ({ ...d, [k]: e.target.value }))}
                      onBlur={(e) => void saveInfo({ [k]: e.target.value })}
                    />
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
                </p>
                <CostumeForm
                  key={`${detail.name}/${costume}`}
                  fields={costumeFields}
                  onSave={(data) => void saveCostumeFields(data)}
                />
              </div>
            )}

            {tab === 'emotions' && (
              <div className="chars-body">
                <p className="chars-sub">
                  官方表情库 {flatEmotions.length} 条（9 个分类，全量原样入库）。
                  每条带 <code>natural_prompt</code>——QI2 直接用它当指令。
                </p>
                <div className="chars-emo-grid">
                  {flatEmotions.map((e) => (
                    <div className="chars-emo" key={`${e.category}/${e.key}`} title={e.natural_prompt}>
                      <img src={emotionIconUrl(e.safe_name)} alt={e.key} loading="lazy" />
                      <span>{e.key}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {tab === 'assets' && (
              <div className="chars-body">
                <p className="chars-sub">
                  磁盘上的真实素材：<code>assets/characters/{detail.name}/Sprites/&lt;服装&gt;/&lt;表情&gt;/</code>
                </p>
                {Object.entries(detail.assets).map(([c, byEmotion]) => (
                  <div key={c} className="chars-asset-group">
                    <b>{c}</b>
                    {Object.entries(byEmotion).map(([emo, urls]) => (
                      <div key={emo} className="chars-asset-row">
                        <span className="chars-asset-label">{emo}</span>
                        {urls.map((u) => (
                          <img key={u} src={u} alt={`${c}/${emo}`} loading="lazy" />
                        ))}
                      </div>
                    ))}
                  </div>
                ))}
                {Object.keys(detail.assets).length === 0 && (
                  <p className="chars-empty">这个角色还没有出过图。</p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

/** 服装字段表单（官方 COSTUME_TEXT_FIELDS 六个字段） */
function CostumeForm({
  fields, onSave,
}: { fields: Record<string, string>; onSave: (data: Record<string, string>) => void }) {
  const [d, setD] = useState<Record<string, string>>({
    top: '', bottom: '', shoes: '', head: '', face: '', negative_prompt: '', ...fields,
  })
  const labels: Array<[string, string]> = [
    ['top', '上装'], ['bottom', '下装'], ['shoes', '鞋'],
    ['head', '头部'], ['face', '脸部'], ['negative_prompt', '负向词'],
  ]
  return (
    <div className="chars-costume">
      {labels.map(([k, label]) => (
        <label className="chars-field" key={k}>
          {label}
          <input value={d[k] ?? ''} onChange={(e) => setD((s) => ({ ...s, [k]: e.target.value }))} />
        </label>
      ))}
      <button type="button" className="mini wide" onClick={() => onSave(d)}>保存服装</button>
    </div>
  )
}
