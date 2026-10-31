/**
 * 资料库浏览面板：提示词收藏 + 姿势库，一个地方翻完。
 *
 * **为什么需要一个专门的浏览面板**（而不是继续挤在侧栏那几条里）：
 * 收藏是会**持续变多**的东西——侧栏那点高度，十条之后就只能靠滚动瞎翻；
 * 姿势库更是必须有预览图，光看名字「挥剑 2」猜不出是哪一版。所以这里给：
 *
 *   · 搜索框（正向/负向/名字都参与匹配，只想得起半个词也能翻出来）
 *   · 卡片网格（姿势有缩略图，一眼挑）
 *   · 每张卡片**明确写出「应用到哪」**（选中节点的类型决定能不能用，
 *     不能用时必须说为什么，而不是灰着一个按钮让人猜）
 *
 * 数据全部在服务端磁盘（workflows/library.py），这里只负责翻和应用。
 */
import { useEffect, useMemo, useState } from 'react'
import type { FavoritePrompt, PosePreset } from '../types'
import type { PresetPose } from '../api'
import { fetchPoseBody } from '../api'
import { filterPrompts } from '../store/prompts'
import { filterPoses } from '../store/poses'

export type LibraryTab = 'prompt' | 'pose'
/** 姿势页的两种来源：用户自己的 / 随项目发行的官方库 */
export type PoseSource = 'mine' | 'official'

const YAW_LABEL: Record<number, string> = {
  0: '正面', 45: '右前', 90: '右侧', 135: '右后',
  180: '背面', 225: '左后', 270: '左侧', 315: '左前',
}

type Props = {
  open: boolean
  tab: LibraryTab
  onTab: (t: LibraryTab) => void
  onClose: () => void
  prompts: FavoritePrompt[]
  poses: PosePreset[]
  /** 随项目发行的 VNCCS 官方姿势库（懒加载，没取到就是空数组） */
  presets: PresetPose[]
  presetCategories: string[]
  onLoadPresets: () => void
  /** 选中节点有没有提示词（=能不能把收藏填回去） */
  canApplyPrompt: boolean
  /** 选中节点是不是人偶编辑器（=能不能把姿势套上去） */
  canApplyPose: boolean
  /** 为什么不能用的提示（写清楚下一步） */
  applyHint: string
  onApplyPrompt: (f: FavoritePrompt) => void
  onRemovePrompt: (id: string) => void
  onApplyPose: (p: PosePreset) => void
  onRenamePose: (id: string, name: string) => void
  onRemovePose: (id: string) => void
  /** 把官方库里的一条收进「我存的」（存的是已经取回来的 pose JSON） */
  onSavePresetAsMine: (name: string, pose: unknown, yaw: number, preview: string) => void
}

export default function LibraryDrawer({
  open, tab, onTab, onClose, prompts, poses,
  presets, presetCategories, onLoadPresets,
  canApplyPrompt, canApplyPose, applyHint,
  onApplyPrompt, onRemovePrompt, onApplyPose, onRenamePose, onRemovePose,
  onSavePresetAsMine,
}: Props) {
  const [q, setQ] = useState('')
  const [src, setSrc] = useState<PoseSource>('official')
  const [cat, setCat] = useState('')
  const [renaming, setRenaming] = useState<string | null>(null)
  const [renameVal, setRenameVal] = useState('')
  const [busyId, setBusyId] = useState('')

  // Esc 关：浏览面板是**临时的**浮层，Esc 关不掉就只能去找那个 ×
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  // 官方库的清单 191 条，**只在真的要看时才拉**（切到姿势页 + 第一次打开）
  useEffect(() => {
    if (open && tab === 'pose' && presets.length === 0) onLoadPresets()
  }, [open, tab, presets.length, onLoadPresets])

  const shownPrompts = useMemo(() => filterPrompts(prompts, q), [prompts, q])
  const shownMine = useMemo(() => filterPoses(poses, q), [poses, q])
  /** 官方库搜索：名字 / 分类 / 标签都参与（标签里有 "side view" 这种角度词） */
  const shownPresets = useMemo(() => {
    const kw = q.trim().toLowerCase()
    return presets.filter((p) => {
      if (cat && p.category !== cat) return false
      if (!kw) return true
      return p.name.toLowerCase().includes(kw)
        || p.category.toLowerCase().includes(kw)
        || p.tags.some((t) => t.toLowerCase().includes(kw))
    })
  }, [presets, q, cat])

  if (!open) return null

  return (
    <div className="lib-mask" onClick={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal lib-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span>资料库</span>
          <span className="lib-path">library/ · 全部落在磁盘</span>
          <button onClick={onClose} title="关闭（Esc）">×</button>
        </div>

        <div className="lib-tabs">
          <button
            className={`lib-tab${tab === 'prompt' ? ' on' : ''}`}
            onClick={() => onTab('prompt')}
          >
            提示词 <b>{prompts.length}</b>
          </button>
          <button
            className={`lib-tab${tab === 'pose' ? ' on' : ''}`}
            onClick={() => onTab('pose')}
          >
            姿势 <b>{poses.length}</b>
          </button>
          <span className="spacer" />
          <input
            className="lib-search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={tab === 'prompt' ? '搜正向 / 负向…' : '搜名字 / 朝向…'}
          />
        </div>

        {tab === 'prompt' ? (
          <div className="lib-grid">
            {prompts.length === 0 ? (
              <p className="lib-empty">
                还没有收藏。在节点的提示词框点 <b>☆</b> 收一条（负向词会一起收进来）。
              </p>
            ) : shownPrompts.length === 0 ? (
              <p className="lib-empty">「{q}」没有匹配到收藏（负向词也参与搜索）。</p>
            ) : shownPrompts.map((f) => (
              <div className="lib-card" key={f.id}>
                <div className="lib-card-text">{f.text}</div>
                {f.negative
                  ? <div className="lib-card-sub">− {f.negative}</div>
                  : null}
                <div className="lib-card-actions">
                  <button
                    type="button"
                    className="mini wide"
                    disabled={!canApplyPrompt}
                    title={canApplyPrompt
                      ? '填入选中节点（正向与负向都会覆盖）'
                      : applyHint}
                    onClick={() => onApplyPrompt(f)}
                  >
                    应用
                  </button>
                  <button
                    type="button"
                    className="mini"
                    title="从库里删掉（磁盘上的 library/prompts.json）"
                    onClick={() => onRemovePrompt(f.id)}
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <>
            <div className="lib-src">
              <button
                className={`lib-src-btn${src === 'official' ? ' on' : ''}`}
                onClick={() => setSrc('official')}
              >
                VNCCS 官方库 <b>{presets.length}</b>
              </button>
              <button
                className={`lib-src-btn${src === 'mine' ? ' on' : ''}`}
                onClick={() => setSrc('mine')}
              >
                我存的 <b>{poses.length}</b>
              </button>
              {src === 'official' && (
                <select
                  className="lib-cat"
                  value={cat}
                  onChange={(e) => setCat(e.target.value)}
                  title="按分类筛（分类来自官方库自己的标签）"
                >
                  <option value="">全部分类</option>
                  {presetCategories.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              )}
            </div>

            {src === 'mine' ? (
              <div className="lib-grid">
                {poses.length === 0 ? (
                  <p className="lib-empty">
                    还没有存过姿势。在「人偶姿势」节点里摆好一个，点 <b>存入</b> 就能在这看到。
                  </p>
                ) : shownMine.length === 0 ? (
                  <p className="lib-empty">「{q}」没有匹配到姿势。</p>
                ) : shownMine.map((p) => (
              <div className="lib-card lib-pose" key={p.id}>
                <div className="lib-thumb">
                  {p.preview
                    ? <img src={p.preview} alt={p.name} draggable={false} />
                    : <span>无预览</span>}
                </div>
                {renaming === p.id ? (
                  <input
                    className="lib-rename"
                    value={renameVal}
                    autoFocus
                    onChange={(e) => setRenameVal(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        onRenamePose(p.id, renameVal)
                        setRenaming(null)
                      }
                    }}
                  />
                ) : (
                  <div className="lib-card-text">
                    {p.name}
                    <em>{p.yaw}° {YAW_LABEL[p.yaw] ?? ''}</em>
                  </div>
                )}
                <div className="lib-card-actions">
                  <button
                    type="button"
                    className="mini wide"
                    disabled={!canApplyPose}
                    title={canApplyPose
                      ? '套用到选中节点的人偶（连同当时的朝向）'
                      : applyHint}
                    onClick={() => onApplyPose(p)}
                  >
                    应用
                  </button>
                  <button
                    type="button"
                    className="mini"
                    onClick={() => {
                      if (renaming === p.id) {
                        onRenamePose(p.id, renameVal)
                        setRenaming(null)
                      } else {
                        setRenaming(p.id)
                        setRenameVal(p.name)
                      }
                    }}
                  >
                    {renaming === p.id ? '✓' : '改名'}
                  </button>
                  <button
                    type="button"
                    className="mini"
                    title="从库里删掉"
                    onClick={() => onRemovePose(p.id)}
                  >
                    删
                  </button>
                </div>
              </div>
            ))}
              </div>
            ) : presets.length === 0 ? (
              <p className="lib-empty">还没取到官方库清单（191 个）。确认服务在运行。</p>
            ) : shownPresets.length === 0 ? (
              <p className="lib-empty">「{q}」没有匹配到官方姿势（名字 / 分类 / 标签都参与搜索）。</p>
            ) : (
              <div className="lib-grid">
                {shownPresets.map((p) => (
                  <div className="lib-card lib-pose" key={p.id}>
                    <div className="lib-thumb">
                      {p.preview
                        ? <img src={p.preview} alt={p.name} draggable={false} loading="lazy" />
                        : <span>无预览</span>}
                    </div>
                    <div className="lib-card-text">
                      {p.name}
                      <em>{p.yaw}° {YAW_LABEL[p.yaw] ?? ''}</em>
                    </div>
                    <div className="lib-card-sub lib-tags">
                      {p.category ? <span className="lib-tag">{p.category}</span> : null}
                      {p.tags.slice(0, 3).map((t) => (
                        <span className="lib-tag" key={t}>{t}</span>
                      ))}
                    </div>
                    <div className="lib-card-actions">
                      <button
                        type="button"
                        className="mini wide"
                        disabled={!canApplyPose || !!busyId}
                        title={canApplyPose
                          ? `套用到选中节点的人偶（默认 ${p.yaw}° 机位，来自标签里的拍摄角度）`
                          : applyHint}
                        onClick={async () => {
                          setBusyId(p.id)
                          try {
                            const body = await fetchPoseBody(p.json)
                            // 官方 json 是 PoseStudio 原生格式，直接当 pose 套，
                            // 同时把它变成一条 PosePreset 交给 App 落 patch
                            onApplyPose({
                              id: p.id, name: p.name, pose: body,
                              yaw: p.yaw, preview: p.preview, at: 0,
                            })
                          } catch (e) {
                            window.alert(`取姿势失败：${e instanceof Error ? e.message : e}`)
                          } finally {
                            setBusyId('')
                          }
                        }}
                      >
                        {busyId === p.id ? '取中…' : '应用'}
                      </button>
                      <button
                        type="button"
                        className="mini"
                        title="把这条存到「我存的」（之后能在所有项目里复用）"
                        onClick={async () => {
                          setBusyId(`${p.id}#save`)
                          try {
                            const body = await fetchPoseBody(p.json)
                            onSavePresetAsMine(p.name, body, p.yaw, p.preview)
                          } catch (e) {
                            window.alert(`取姿势失败：${e instanceof Error ? e.message : e}`)
                          } finally {
                            setBusyId('')
                          }
                        }}
                      >
                        收藏
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        <div className="lib-foot">
          「我存的」在服务端磁盘 <b>library/</b>（换浏览器、清缓存都不丢，整个目录拷走就是备份）；
          「VNCCS 官方库」随项目发行，在 <b>web/public/vnccs-pose/PoseLibrary/</b>。
        </div>
      </div>
    </div>
  )
}
