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
import type { FavoritePrompt, PoseCharacter, PosePreset } from '../types'
import type { PresetPose } from '../api'
import { fetchPoseBody } from '../api'
import { filterPrompts } from '../store/prompts'
import { filterPoses } from '../store/poses'
import { POSE_CATEGORY_LABEL, POSE_NAME_LABEL, POSE_TAG_LABEL, zh } from '../labels'
import MotionTemplatePanel from './MotionTemplatePanel'

export type LibraryTab = 'prompt' | 'pose' | 'motion'
/** 姿势页的两种来源：用户自己的 / 随项目发行的官方库 */
export type PoseSource = 'mine' | 'official'

const YAW_LABEL: Record<number, string> = {
  0: '正面', 45: '右前', 90: '右侧', 135: '右后',
  180: '背面', 225: '左后', 270: '左侧', 315: '左前',
}

/** 官方 cameraParams 里带构图信息的数值键；其余字段与全是空串的 prompt 不带进场景。 */
const CAMERA_KEYS = ['zoom', 'offset_x', 'offset_y', 'yaw_deg', 'pitch_deg'] as const
/** 一条姿势最多带几个角色，和后端 poselib.py 的清单上限对齐 */
const MAX_CHARS = 8

function asRecord(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
    ? v as Record<string, unknown> : {}
}
function finite(v: unknown, fallback = 0): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback
}
function transformOf(v: unknown): { x: number; y: number; z: number; zoom: number } {
  const t = asRecord(v)
  return { x: finite(t.x), y: finite(t.y), z: finite(t.z), zoom: finite(t.zoom, 1) }
}
function rotationOf(v: unknown): [number, number, number] {
  const a = Array.isArray(v) ? v : []
  return [finite(a[0]), finite(a[1]), finite(a[2])]
}

/**
 * 从官方 pose JSON 里摘出构图参数与角色放置。body 是官网的不透明格式，
 * 逐字段收窄、坏值回退：缺字段/坏字段一律给空对象/空数组，别让一条老姿势
 * 把整套「应用」流程抛错。transform 的位移/缩放与 rotation 的欧拉角合并成
 * PosePreset.characters 要的 placement，App 按它直接初始化多角色场景。
 */
function poseExtras(body: unknown): {
  cameraParams: Record<string, number>
  characters: PoseCharacter[]
} {
  const doc = asRecord(body)
  const cameraParams: Record<string, number> = {}
  const cam = asRecord(doc.cameraParams)
  for (const k of CAMERA_KEYS) {
    const v = cam[k]
    if (typeof v === 'number' && Number.isFinite(v)) cameraParams[k] = v
  }
  const raw = Array.isArray(doc.characters) ? doc.characters.slice(0, MAX_CHARS) : []
  const characters: PoseCharacter[] = raw.map((c, i) => {
    const o = asRecord(c)
    const transform = transformOf(o.transform)
    const rotation = rotationOf(o.rotation)
    return {
      id: typeof o.id === 'string' && o.id ? o.id : `ch_${i + 1}`,
      name: typeof o.name === 'string' && o.name ? o.name : `角色 ${i + 1}`,
      color: '',
      placement: { ...transform, rotation },
    }
  })
  return { cameraParams, characters }
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
          <button
            className={`lib-tab${tab === 'motion' ? ' on' : ''}`}
            onClick={() => onTab('motion')}
            title="动作模板：方向 / 动作 / 帧率 / 预算，抽帧与 Godot 导出都从这里读"
          >
            动作模板
          </button>
          <span className="spacer" />
          <input
            className="lib-search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={tab === 'prompt' ? '搜正向 / 负向…'
              : tab === 'motion' ? '搜模板名 / id…' : '搜名字 / 朝向…'}
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
        ) : tab === 'pose' ? (
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
                  {presetCategories.map((c) => (
                    <option key={c} value={c}>{zh(POSE_CATEGORY_LABEL, c)}</option>
                  ))}
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
                        ? <img
                          src={p.preview}
                          alt={zh(POSE_NAME_LABEL, p.name)}
                          draggable={false}
                          loading="lazy"
                        />
                        : <span>无预览</span>}
                    </div>
                    <div className="lib-card-text">
                      {zh(POSE_NAME_LABEL, p.name)}
                      <em>{p.yaw}° {YAW_LABEL[p.yaw] ?? ''}</em>
                    </div>
                    <div className="lib-card-sub lib-tags">
                      {p.category ? (
                        <span className="lib-tag">{zh(POSE_CATEGORY_LABEL, p.category)}</span>
                      ) : null}
                      {p.tags.slice(0, 3).map((t) => (
                        <span className="lib-tag" key={t}>{zh(POSE_TAG_LABEL, t)}</span>
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
                            // 同时把它变成一条 PosePreset 交给 App 落 patch。
                            // 构图参数与角色放置也从同一份 body 里摘出来一起带上，
                            // App 按 PosePreset 接时才能还原多角色场景的机位与站位。
                            const extras = poseExtras(body)
                            onApplyPose({
                              id: p.id, name: p.name, pose: body,
                              yaw: p.yaw, preview: p.preview, at: 0,
                              cameraParams: extras.cameraParams,
                              characters: extras.characters,
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
                            // 落盘用**中文名**：「我存的」那一栏不翻译（那里显示的都是
                            // 用户自己起的名字），官方姿势收藏进来若存英文原名，
                            // 就会跟用户自建的名字混在一格里，一半中文一半英文。
                            // 后端 id 是时间戳、跟名字无关，同名会原地覆盖，重复收藏不会翻倍。
                            onSavePresetAsMine(
                              zh(POSE_NAME_LABEL, p.name), body, p.yaw, p.preview)
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
        ) : (
          <MotionTemplatePanel query={q} />
        )}

        <div className="lib-foot">
          {tab === 'motion' ? (
            <>
              动作模板在服务端磁盘 <b>library/motion_templates.json</b>，只存元数据不存图；
              内置三套是代码常量、不落盘。<b>成本按 t/(1-(1-p)^k) 估算</b>，横板镜像派生会少一半机位。
            </>
          ) : (
            <>
              「我存的」在服务端磁盘 <b>library/</b>（换浏览器、清缓存都不丢，整个目录拷走就是备份）；
              「VNCCS 官方库」随项目发行，在 <b>web/public/vnccs-pose/PoseLibrary/</b>。
            </>
          )}
        </div>
      </div>
    </div>
  )
}
