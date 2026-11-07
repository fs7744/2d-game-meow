/**
 * 画布库 —— 画布视图的默认首页，与角色中心的全屏角色库同一套骨架。
 *
 * 单位仍是磁盘上的 `assets/projects/<id>/project.json`（后端 scope 不变，
 * 改路径会牵动几十处调用），但**界面上一律叫「画布」，不再叫「项目」**。
 *
 * 元数据同样放在 SQLite（workflows/metadb.py 的 canvases 表）：
 * 节点类型派生标签、稳定时间、自定义标签、备注；封面只记路径字符串，
 * 图片 / 节点定义 / 模型一律留在磁盘，绝不进库。
 *
 * 样式直接复用 `.charlib-*`（角色库那套），两个库的视觉与交互保持一致。
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  type ServerProjectMeta, fetchServerProjects, getLibrary, savePrefsApi, updateCanvasMeta,
} from '../api'
import { explainError } from './characters/model'

type SortKey = 'name' | 'created' | 'updated'

type Props = {
  /** 点卡片进编辑器（传项目 scope，后端仍用这个 id） */
  onOpen: (id: string) => void
  /** 新建画布：App 负责分配 id / 命名 / 切进编辑器 */
  onCreate: (name: string) => void | Promise<void>
  onDelete: (id: string) => void | Promise<void>
  onNotice: (msg: string) => void
}

/** 节点 kind → 中文（数据标识仍按原值传给后端） */
const KIND_LABEL: Record<string, string> = {
  gen: '图生成', img2img: '图生图', outpaint: '扩图', outfit: '换装',
  faceswap: '换脸', poseedit: '人偶姿势', poseclone: '角色迁移',
  posebatch: '姿态批量', spriteswap: '序列帧替换', poseswap: '姿态序列',
  pixelify: '像素转换', topixel: '转像素', matte: '抠图', image: '图片',
  bg: '背景', delight: '打光', spritesheet: '图集',
}
const labelOf = (tag: string) => KIND_LABEL[tag] || tag

const SORT_LABEL: Record<SortKey, string> = {
  name: '名字', created: '创建时间', updated: '最后编辑时间',
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

export default function CanvasLibrary({
  onOpen, onCreate, onDelete, onNotice,
}: Props) {
  /** 列表自己拉，不和 App 的 projectMeta 抢状态：后者是编辑器用的精简版，
      这里需要封面 / 标签 / 时间戳等完整摘要。 */
  const [items, setItems] = useState<ServerProjectMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')
  const load = useCallback(async () => {
    const r = await fetchServerProjects()
    setItems(r.items)
    setErr(r.reachable ? '' : '连不上后端，显示的可能是旧数据')
    setLoading(false)
  }, [])
  useEffect(() => { void load() }, [load])

  const [query, setQuery] = useState('')
  const [activeTags, setActiveTags] = useState<string[]>([])
  const [sortBy, setSortBy] = useState<SortKey>('updated')
  const [asc, setAsc] = useState(false)
  const [showAllTags, setShowAllTags] = useState(false)
  const [prefsReady, setPrefsReady] = useState(false)

  useEffect(() => {
    let alive = true
    void getLibrary().then((lib) => {
      if (!alive) return
      const p = (lib.prefs || {}) as Record<string, unknown>
      if (p.canvasSort === 'name' || p.canvasSort === 'created' || p.canvasSort === 'updated') {
        setSortBy(p.canvasSort)
      }
      if (typeof p.canvasAsc === 'boolean') setAsc(p.canvasAsc)
      if (Array.isArray(p.canvasFilterTags)) {
        setActiveTags(p.canvasFilterTags.filter((t): t is string => typeof t === 'string'))
      }
      setPrefsReady(true)
    }).catch(() => { if (alive) setPrefsReady(true) })
    return () => { alive = false }
  }, [])

  const savePrefs = (patch: Record<string, unknown>) => {
    if (!prefsReady) return
    void savePrefsApi(patch).catch(() => {})
  }

  /* ---------------- 新建 / 编辑 / 删除 ---------------- */
  const [createOpen, setCreateOpen] = useState(false)
  const [newName, setNewName] = useState('')
  const [busy, setBusy] = useState(false)
  const [formErr, setFormErr] = useState('')
  const [editing, setEditing] = useState<ServerProjectMeta | null>(null)
  const [editTags, setEditTags] = useState('')
  const [editNote, setEditNote] = useState('')
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)

  const doCreate = async () => {
    const name = newName.trim()
    if (!name || busy) return
    setBusy(true); setFormErr('')
    try {
      await onCreate(name)
      setNewName(''); setCreateOpen(false)
      onNotice(`已建画布「${name}」`)
    } catch (e) {
      setFormErr(explainError(e))
    } finally { setBusy(false) }
  }

  const openEdit = (c: ServerProjectMeta) => {
    setEditing(c)
    setEditTags((c.custom_tags || []).join(', '))
    setEditNote(c.note || '')
  }
  const saveMeta = async () => {
    if (!editing || busy) return
    setBusy(true); setFormErr('')
    try {
      await updateCanvasMeta(editing.id, {
        custom_tags: editTags.split(/[,，]/).map((s) => s.trim()).filter(Boolean),
        note: editNote.trim(),
      })
      setEditing(null)
      onNotice('标签与备注已保存')
      await load()
    } catch (e) {
      setFormErr(explainError(e))
    } finally { setBusy(false) }
  }

  /* ---------------- 搜索 / 过滤 / 排序 ---------------- */
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
        if (!activeTags.some((t) => tags.includes(t))) return false
      }
      return true
    })
    const dir = asc ? 1 : -1
    return [...filtered].sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name, 'zh-CN') * dir
      const av = sortBy === 'created' ? (a.created_ts || 0) : (a.updated_ts || 0)
      const bv = sortBy === 'created' ? (b.created_ts || 0) : (b.updated_ts || 0)
      return (av - bv) * dir
    })
  }, [items, query, activeTags, sortBy, asc])

  const visibleTags = showAllTags ? allTags : allTags.slice(0, 18)
  const clearFilters = () => {
    setQuery(''); setActiveTags([])
    savePrefs({ canvasFilterTags: [] })
  }

  return (
    <div className="charlib">
      <header className="charlib-head">
        <div className="charlib-head-title">
          <h1>画布库</h1>
          <span className="charlib-count">{items.length} 张画布</span>
          <span className="charlib-head-sub">点卡片进入节点编辑器 · 每个画布是一套独立的节点与参数</span>
        </div>
        <button type="button" className="cc-btn primary charlib-new"
          onClick={() => setCreateOpen(true)}>＋ 新建画布</button>
      </header>

      {err && <div className="chars-err">⚠ {err}</div>}

      <div className="charlib-tools">
        <input className="charlib-search" type="search"
          placeholder="搜索画布名或标签…" value={query}
          onChange={(e) => setQuery(e.target.value)} />
        <div className="charlib-sort">
          <span>排序</span>
          <select value={sortBy} onChange={(e) => {
            const v = e.target.value as SortKey
            setSortBy(v); savePrefs({ canvasSort: v })
          }}>
            {(Object.keys(SORT_LABEL) as SortKey[]).map((k) => (
              <option key={k} value={k}>{SORT_LABEL[k]}</option>
            ))}
          </select>
          <button type="button" className="cc-btn ghost" onClick={() => {
            const v = !asc
            setAsc(v); savePrefs({ canvasAsc: v })
          }}>{asc ? '↑ 升序' : '↓ 降序'}</button>
        </div>
        <span className="charlib-result">显示 {shown.length} / {items.length}</span>
      </div>

      {allTags.length > 0 && (
        <div className="charlib-filters">
          <span className="charlib-filter-label">标签筛选</span>
          <button type="button"
            className={`charlib-filter${activeTags.length === 0 ? ' on' : ''}`}
            onClick={() => { setActiveTags([]); savePrefs({ canvasFilterTags: [] }) }}>全部</button>
          {visibleTags.map(({ tag, n }) => (
            <button type="button" key={tag}
              className={`charlib-filter${activeTags.includes(tag) ? ' on' : ''}`}
              onClick={() => {
                const next = activeTags.includes(tag)
                  ? activeTags.filter((t) => t !== tag) : [...activeTags, tag]
                setActiveTags(next); savePrefs({ canvasFilterTags: next })
              }}>{labelOf(tag)}<i>{n}</i></button>
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

      {!loading && items.length === 0 && (
        <div className="charlib-empty">
          <div className="charlib-empty-art" aria-hidden="true">
            <span /><span /><span /><span />
          </div>
          <h2>还没有画布</h2>
          <p>建一张画布，节点、参数、产物都会收在这张画布里，随时回来继续做。</p>
          <button type="button" className="charlib-big-create"
            onClick={() => setCreateOpen(true)}>
            <b>＋</b>
            <span>创建第一张画布</span>
            <i>给画布起个名字，然后从节点库里拖节点进来</i>
          </button>
        </div>
      )}
      {loading && <div className="charlib-loading">正在读画布库…</div>}

      {items.length > 0 && shown.length > 0 && (
        <div className="charlib-grid">
          {shown.map((c) => {
            // 用户手打的自定义标签优先展示；派生标签太多时才被 +N 收起
            const tags = [...(c.custom_tags || []), ...(c.tags || [])]
            return (
              <article key={c.id} className="charlib-card"
                onClick={() => onOpen(c.id)} title={`打开「${c.name}」`}>
                <div className="charlib-thumb">
                  {c.cover_url
                    ? <img src={c.cover_url} alt={c.name} loading="lazy" />
                    : <div className="charlib-nocover"><span>◧</span>还没有产物</div>}
                  <div className="charlib-tagwrap">
                    {tags.slice(0, 5).map((t) => (
                      <span className="charlib-tag" key={t}>{labelOf(t)}</span>
                    ))}
                    {tags.length > 5 && (
                      <span className="charlib-tag is-more">+{tags.length - 5}</span>
                    )}
                  </div>
                  <div className="charlib-quick">
                    <button type="button" className="charlib-edit" title="编辑标签 / 备注"
                      onClick={(e) => { e.stopPropagation(); openEdit(c) }}>✎</button>
                    <button type="button" className="charlib-remove" title="删除画布"
                      onClick={(e) => { e.stopPropagation(); setPendingDelete(c.id) }}>✕</button>
                  </div>
                </div>
                <div className="charlib-meta">
                  <b className="charlib-name">{c.name}</b>
                  <span className="charlib-sub">{c.count} 个节点</span>
                  <span className="charlib-time">编辑于 {fmtTime(c.updated_ts)}</span>
                </div>
                {c.note && <p className="charlib-note">{c.note}</p>}
              </article>
            )
          })}
        </div>
      )}

      {items.length > 0 && shown.length === 0 && !loading && (
        <div className="charlib-noresult">
          <b>没有匹配的画布</b>
          <p>换个名字 / 标签，或者清掉筛选条件。</p>
          <button type="button" className="cc-btn" onClick={clearFilters}>清除筛选</button>
        </div>
      )}

      {createOpen && (
        <div className="charlib-mask" onClick={() => setCreateOpen(false)}>
          <div className="charlib-create" onClick={(e) => e.stopPropagation()}>
            <h2>新建画布</h2>
            <p>一张画布 = 一套独立的节点、参数与产物。</p>
            <label className="charlib-field">画布名
              <input value={newName} placeholder="例：主角立绘产线"
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void doCreate() }} />
            </label>
            {formErr && <p className="charlib-err">{formErr}</p>}
            <div className="charlib-dialog-actions">
              <button type="button" className="cc-btn ghost"
                onClick={() => setCreateOpen(false)}>取消</button>
              <button type="button" className="cc-btn primary"
                onClick={() => void doCreate()} disabled={busy || !newName.trim()}>
                {busy ? '创建中…' : '创建画布'}
              </button>
            </div>
          </div>
        </div>
      )}

      {editing && (
        <div className="charlib-mask" onClick={() => setEditing(null)}>
          <div className="charlib-create" onClick={(e) => e.stopPropagation()}>
            <h2>编辑「{editing.name}」</h2>
            <p>自定义标签和备注存在本地 SQLite 元数据里，不会写进 project.json。</p>
            <label className="charlib-field">自定义标签（逗号分隔）
              <input value={editTags} placeholder="例：主角, 待精修"
                onChange={(e) => setEditTags(e.target.value)} />
            </label>
            <label className="charlib-field">备注
              <textarea value={editNote} placeholder="想记点什么…"
                onChange={(e) => setEditNote(e.target.value)} />
            </label>
            {formErr && <p className="charlib-err">{formErr}</p>}
            <div className="charlib-dialog-actions">
              <button type="button" className="cc-btn ghost"
                onClick={() => setEditing(null)}>取消</button>
              <button type="button" className="cc-btn primary"
                onClick={() => void saveMeta()} disabled={busy}>保存</button>
            </div>
          </div>
        </div>
      )}

      {pendingDelete && (
        <div className="charlib-confirm">
          <span>删掉这张画布？节点定义与产物目录会一起删，不可恢复。</span>
          <button type="button" className="cc-btn ghost"
            onClick={() => setPendingDelete(null)}>取消</button>
          <button type="button" className="cc-btn danger" onClick={() => {
            const id = pendingDelete
            setPendingDelete(null)
            void (async () => { await onDelete(id); await load() })()
          }}>确认删除</button>
        </div>
      )}
    </div>
  )
}
