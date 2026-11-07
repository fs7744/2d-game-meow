/**
 * 角色库 —— 角色中心的**默认首页**，占满整个页面。
 *
 * 旧版把角色库挤在左边 220px 的窄栏里，一进角色中心看到的是一堆表单；
 * 现在先看到所有角色：大图、名字、贴在图片上的标签，再点卡片进四步工作台。
 *
 * 数据分工：
 *   磁盘   = 封面 / 精灵帧 / 属性 config（生成产线用）
 *   SQLite = 派生标签、稳定的创建/编辑时间、自定义标签、备注、任务历史
 *          （见 workflows/metadb.py；**不存任何文件与模型**）
 *
 * ⚠️ 标签值原样英文传给后端（同时是 prompt 内容与目录名），中文只在
 *    tagLabel() 里做显示层映射，查不到就回落原值。
 */
import { useEffect, useMemo, useState } from 'react'
import {
  type CharacterSummary,
  createCharacter, createCharacterFromImage, uploadImage,
  getLibrary, savePrefsApi, updateCharacterMeta,
} from '../api'
import { COSTUME_LABEL, STYLE_LABEL, TAG_LABEL, zh } from '../labels'
import { explainError } from './characters/model'

type SortKey = 'name' | 'created' | 'updated'

type Props = {
  items: CharacterSummary[]
  loading: boolean
  err: string
  /** 点卡片进四步工作台 */
  onOpen: (name: string) => void
  /** 创建成功后：主组件刷新列表并进入工作台 */
  onCreated: (name: string) => void | Promise<void>
  /** 编辑元数据后刷新列表 */
  onReload: () => void | Promise<void>
  onDelete: (name: string) => void
  onNotice: (msg: string) => void
}

/** 标签中文显示：三个官方映射表依次查，查不到回落原值 */
export function tagLabel(tag: string): string {
  for (const map of [TAG_LABEL, STYLE_LABEL, COSTUME_LABEL]) {
    const v = zh(map, tag)
    if (v !== tag) return v
  }
  return tag
}

function fmtTime(ts?: number): string {
  if (!ts) return '—'
  const diff = Date.now() - ts * 1000
  if (diff < 60_000) return '刚刚'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`
  if (diff < 7 * 86_400_000) return `${Math.floor(diff / 86_400_000)} 天前`
  const d = new Date(ts * 1000)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
    + `-${String(d.getDate()).padStart(2, '0')}`
}

const SORT_LABEL: Record<SortKey, string> = {
  name: '名字', created: '创建时间', updated: '最后编辑时间',
}

export default function CharacterLibrary({
  items, loading, err, onOpen, onCreated, onReload, onDelete, onNotice,
}: Props) {
  /* ---------------- 搜索 / 过滤 / 排序 ---------------- */
  const [query, setQuery] = useState('')
  const [activeTags, setActiveTags] = useState<string[]>([])
  const [sortBy, setSortBy] = useState<SortKey>('updated')
  const [asc, setAsc] = useState(false)
  const [showAllTags, setShowAllTags] = useState(false)
  /** 偏好从服务端读回来之前不要回写，否则会把用户存的排序覆盖成默认值 */
  const [prefsReady, setPrefsReady] = useState(false)

  useEffect(() => {
    let alive = true
    void getLibrary().then((lib) => {
      if (!alive) return
      const p = (lib.prefs || {}) as Record<string, unknown>
      if (p.charSort === 'name' || p.charSort === 'created' || p.charSort === 'updated') {
        setSortBy(p.charSort)
      }
      if (typeof p.charAsc === 'boolean') setAsc(p.charAsc)
      if (Array.isArray(p.charFilterTags)) {
        setActiveTags(p.charFilterTags.filter((t): t is string => typeof t === 'string'))
      }
      setPrefsReady(true)
    }).catch(() => { if (alive) setPrefsReady(true) })
    return () => { alive = false }
  }, [])

  const savePrefs = (patch: Record<string, unknown>) => {
    if (!prefsReady) return
    void savePrefsApi(patch).catch(() => { /* 偏好写失败不影响浏览 */ })
  }

  /* ---------------- 创建角色 ---------------- */
  const [createOpen, setCreateOpen] = useState(false)
  const [newName, setNewName] = useState('')
  const [cloneFile, setCloneFile] = useState<File | null>(null)
  const [clonePreview, setClonePreview] = useState('')
  const [busy, setBusy] = useState(false)
  const [createErr, setCreateErr] = useState('')

  const openCreate = () => {
    setCreateErr('')
    setCreateOpen(true)
  }
  const closeCreate = () => {
    setCreateOpen(false); setCloneFile(null); setClonePreview('')
  }
  const doCreate = async () => {
    const name = newName.trim()
    if (!name || busy) return
    setBusy(true); setCreateErr('')
    try {
      if (cloneFile) {
        const up = await uploadImage(cloneFile)
        await createCharacterFromImage({
          name, image: up.path, background_color: 'green',
        })
      } else {
        await createCharacter(name)
      }
      setNewName(''); closeCreate()
      onNotice(`已建角色「${name}」`)
      await onCreated(name)
    } catch (e) {
      setCreateErr(explainError(e))
    } finally {
      setBusy(false)
    }
  }

  /* ---------------- 编辑元数据（自定义标签 / 备注） ---------------- */
  const [editing, setEditing] = useState<string | null>(null)
  const [editTags, setEditTags] = useState('')
  const [editNote, setEditNote] = useState('')
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

  const openEdit = (c: CharacterSummary) => {
    setEditing(c.name)
    setEditTags((c.custom_tags || []).join(', '))
    setEditNote(c.note || '')
  }
  const saveMeta = async () => {
    if (!editing || busy) return
    setBusy(true); setCreateErr('')
    try {
      await updateCharacterMeta(editing, {
        custom_tags: editTags.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
        note: editNote.trim(),
      })
      setEditing(null)
      onNotice('标签与备注已保存')
      await onReload()
    } catch (e) {
      setCreateErr(explainError(e))
    } finally {
      setBusy(false)
    }
  }

  /* ---------------- 派生数据 ---------------- */
  const allTags = useMemo(() => {
    const count = new Map<string, number>()
    for (const c of items) {
      for (const t of [...(c.tags || []), ...(c.custom_tags || [])]) {
        count.set(t, (count.get(t) || 0) + 1)
      }
    }
    return [...count.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([tag, n]) => ({ tag, n }))
  }, [items])

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = items.filter((c) => {
      if (q) {
        const hay = [c.name, c.note || '', ...(c.tags || []), ...(c.custom_tags || [])]
          .join(' ').toLowerCase()
        if (!hay.includes(q)) return false
      }
      if (activeTags.length > 0) {
        const tags = [...(c.tags || []), ...(c.custom_tags || [])]
        // 多选是「包含任一选中标签」（OR），比 AND 更符合"按标签挑角色"的直觉
        if (!activeTags.some((t) => tags.includes(t))) return false
      }
      return true
    })
    const dir = asc ? 1 : -1
    return [...filtered].sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name, 'zh-CN') * dir
      const av = sortBy === 'created' ? (a.created_at || 0) : (a.updated_at || 0)
      const bv = sortBy === 'created' ? (b.created_at || 0) : (b.updated_at || 0)
      return (av - bv) * dir
    })
  }, [items, query, activeTags, sortBy, asc])

  const visibleTags = showAllTags ? allTags : allTags.slice(0, 18)
  const clearFilters = () => { setQuery(''); setActiveTags([]); savePrefs({ charFilterTags: [] }) }

  return (
    <div className="charlib">
      <header className="charlib-head">
        <div className="charlib-head-title">
          <h1>角色库</h1>
          <span className="charlib-count">{items.length} 个角色</span>
          <span className="charlib-head-sub">点卡片进入四步工作台 · 素材按角色归档</span>
        </div>
        <button type="button" className="cc-btn primary charlib-new"
          onClick={openCreate}>＋ 新建角色</button>
      </header>

      {err && <div className="chars-err">⚠ {err}</div>}

      <div className="charlib-tools">
        <input className="charlib-search" type="search"
          placeholder="搜索角色名或标签…" value={query}
          onChange={(e) => setQuery(e.target.value)} />
        <div className="charlib-sort">
          <span>排序</span>
          <select value={sortBy} onChange={(e) => {
            const v = e.target.value as SortKey
            setSortBy(v); savePrefs({ charSort: v })
          }}>
            {(Object.keys(SORT_LABEL) as SortKey[]).map((k) => (
              <option key={k} value={k}>{SORT_LABEL[k]}</option>
            ))}
          </select>
          <button type="button" className="cc-btn ghost" onClick={() => {
            const v = !asc
            setAsc(v); savePrefs({ charAsc: v })
          }}>{asc ? '↑ 升序' : '↓ 降序'}</button>
        </div>
        <span className="charlib-result">显示 {shown.length} / {items.length}</span>
      </div>

      {allTags.length > 0 && (
        <div className="charlib-filters">
          <span className="charlib-filter-label">标签筛选</span>
          <button type="button"
            className={`charlib-filter${activeTags.length === 0 ? ' on' : ''}`}
            onClick={() => { setActiveTags([]); savePrefs({ charFilterTags: [] }) }}>全部</button>
          {visibleTags.map(({ tag, n }) => (
            <button type="button" key={tag}
              className={`charlib-filter${activeTags.includes(tag) ? ' on' : ''}`}
              onClick={() => {
                const next = activeTags.includes(tag)
                  ? activeTags.filter((t) => t !== tag)
                  : [...activeTags, tag]
                setActiveTags(next); savePrefs({ charFilterTags: next })
              }}>
              {tagLabel(tag)}<i>{n}</i>
            </button>
          ))}
          {allTags.length > 18 && (
            <button type="button" className="charlib-filter more"
              onClick={() => setShowAllTags((v) => !v)}>
              {showAllTags ? '收起' : `更多 ${allTags.length - 18}`}
            </button>
          )}
          {activeTags.length > 0 && (
            <button type="button" className="charlib-filter clear"
              onClick={clearFilters}>清除筛选 ✕</button>
          )}
        </div>
      )}

      {/* ---------------- 空库：一个足够大的创建入口 ---------------- */}
      {!loading && items.length === 0 && (
        <div className="charlib-empty">
          <div className="charlib-empty-art" aria-hidden="true">
            <span /><span /><span /><span />
          </div>
          <h2>还没有角色</h2>
          <p>建一个角色，之后所有立绘、服装、表情和图集都会挂在它下面。</p>
          <button type="button" className="charlib-big-create" onClick={openCreate}>
            <b>＋</b>
            <span>创建第一个角色</span>
            <i>给角色起个名字，或直接拿一张参考图开始</i>
          </button>
        </div>
      )}
      {loading && <div className="charlib-loading">正在读角色库…</div>}

      {/* ---------------- 卡片网格 ---------------- */}
      {items.length > 0 && shown.length > 0 && (
        <div className="charlib-grid">
          {shown.map((c) => {
            // 用户手打的自定义标签优先展示；派生标签太多时才被 +N 收起
            const tags = [...(c.custom_tags || []), ...(c.tags || [])]
            return (
              <article key={c.name} className="charlib-card"
                onClick={() => onOpen(c.name)} title={`打开「${c.name}」`}>
                <div className="charlib-thumb">
                  {c.cover_url
                    ? <img src={c.cover_url} alt={c.name} loading="lazy" />
                    : <div className="charlib-nocover"><span>◧</span>暂无立绘</div>}
                  {/* 标签贴在图片上：tag 可能很多，只铺前 5 个 + "+N" */}
                  <div className="charlib-tagwrap">
                    {tags.slice(0, 5).map((t) => (
                      <span className="charlib-tag" key={t}>{tagLabel(t)}</span>
                    ))}
                    {tags.length > 5 && (
                      <span className="charlib-tag is-more">+{tags.length - 5}</span>
                    )}
                  </div>
                  <div className="charlib-quick">
                    <button type="button" className="charlib-edit" title="编辑标签 / 备注"
                      onClick={(e) => { e.stopPropagation(); openEdit(c) }}>✎</button>
                    <button type="button" className="charlib-remove" title="删除角色"
                      onClick={(e) => { e.stopPropagation(); setPendingDelete(c.name) }}>✕</button>
                  </div>
                </div>
                <div className="charlib-meta">
                  <b className="charlib-name">{c.name}</b>
                  <span className="charlib-sub">
                    {(c.costumes || []).length} 服装 · {c.sprites} 帧
                  </span>
                  <span className="charlib-time">编辑于 {fmtTime(c.updated_at)}</span>
                </div>
                {c.note && <p className="charlib-note">{c.note}</p>}
              </article>
            )
          })}
        </div>
      )}

      {items.length > 0 && shown.length === 0 && !loading && (
        <div className="charlib-noresult">
          <b>没有匹配的角色</b>
          <p>换个名字 / 标签，或者清掉筛选条件。</p>
          <button type="button" className="cc-btn" onClick={clearFilters}>清除筛选</button>
        </div>
      )}

      {/* ---------------- 创建面板 ---------------- */}
      {createOpen && (
        <div className="charlib-mask" onClick={closeCreate}>
          <div className="charlib-create" onClick={(e) => e.stopPropagation()}>
            <h2>新建角色</h2>
            <p>角色会存成一个独立档案，之后所有素材都挂在它下面。</p>
            <label className="charlib-field">角色名
              <input value={newName} placeholder="例：白色猫骑士"
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void doCreate() }} />
            </label>
            <label className="charlib-field">参考图（可选）
              <input type="file" accept="image/png,image/jpeg,image/webp,image/bmp"
                onChange={(e) => {
                  const f = e.target.files?.[0] ?? null
                  setCloneFile(f)
                  setClonePreview(f ? URL.createObjectURL(f) : '')
                }} />
            </label>
            {clonePreview && (
              <img className="charlib-clone-preview" src={clonePreview} alt="参考图" />
            )}
            {createErr && <p className="charlib-err">{createErr}</p>}
            <div className="charlib-dialog-actions">
              <button type="button" className="cc-btn ghost" onClick={closeCreate}>取消</button>
              <button type="button" className="cc-btn primary"
                onClick={() => void doCreate()} disabled={busy || !newName.trim()}>
                {busy ? '创建中…' : '创建角色'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ---------------- 编辑元数据 ---------------- */}
      {editing && (
        <div className="charlib-mask" onClick={() => setEditing(null)}>
          <div className="charlib-create" onClick={(e) => e.stopPropagation()}>
            <h2>编辑「{editing}」</h2>
            <p>自定义标签和备注存在本地 SQLite 元数据里，不会写进生成用的属性 config。</p>
            <label className="charlib-field">自定义标签（逗号分隔）
              <input value={editTags} placeholder="例：主角, 待精修"
                onChange={(e) => setEditTags(e.target.value)} />
            </label>
            <label className="charlib-field">备注
              <textarea value={editNote} placeholder="想记点什么…"
                onChange={(e) => setEditNote(e.target.value)} />
            </label>
            {createErr && <p className="charlib-err">{createErr}</p>}
            <div className="charlib-dialog-actions">
              <button type="button" className="cc-btn ghost" onClick={() => setEditing(null)}>取消</button>
              <button type="button" className="cc-btn primary"
                onClick={() => void saveMeta()} disabled={busy}>保存</button>
            </div>
          </div>
        </div>
      )}

      {/* ---------------- 删除确认 ---------------- */}
      {pendingDelete && (
        <div className="charlib-confirm">
          <span>删掉「{pendingDelete}」？所有素材会一起删，不可恢复。</span>
          <button type="button" className="cc-btn ghost"
            onClick={() => setPendingDelete(null)}>取消</button>
          <button type="button" className="cc-btn danger" onClick={() => {
            onDelete(pendingDelete); setPendingDelete(null)
          }}>确认删除</button>
        </div>
      )}
    </div>
  )
}
