/**
 * 角色中心的展示型组件。
 *
 * 全部是纯展示：状态与业务动作留在 CharactersPage，这里只负责「长什么样」。
 * 拆出来的目的很实际 —— 以前 1900 行一个文件，改一个按钮要在一堆轮询循环里找 JSX。
 */
import { type ReactNode } from 'react'
import SpritePreview from '../SpritePreview'
import { stylePreviewUrl, emotionIconUrl, type AtlasInfo } from '../../api'
import {
  BG_COLOR_LABEL, COSTUME_LABEL, STYLE_GROUP_LABEL, emotionLabel, zh,
} from '../../labels'
import { BG_COLORS, STEPS, type StepId, type TaskState } from './model'

/* ---------------------------------------------------------------- 步骤条 */

export function StepBar({
  done, active, onPick,
}: {
  done: boolean[]
  active: StepId
  onPick: (s: StepId) => void
}) {
  return (
    <nav className="cc-steps" aria-label="角色制作步骤">
      {STEPS.map((s, i) => {
        const state = done[i] ? 'done' : (active === s.id ? 'on' : 'todo')
        return (
          <button
            type="button"
            key={s.id}
            className={`chars-tab cc-step is-${state}`}
            data-step={s.id}
            onClick={() => onPick(s.id)}
            title={s.result}
          >
            <span className="cc-step-no">{done[i] ? '✓' : `0${s.id}`}</span>
            <span className="cc-step-label">{s.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

/** 每一步顶上的一句话：做什么、做完拿到什么。不做成折叠说明，直接写。 */
export function StepHead({ step, sub }: { step: StepId; sub?: string }) {
  const s = STEPS[step - 1]
  return (
    <div className="cc-stephead">
      <div className="cc-stephead-kicker">第 {step} 步 / 共 4 步</div>
      <h2 className="cc-stephead-title">{s.label}</h2>
      <p className="cc-stephead-goal">{s.goal}</p>
      <div className="cc-stephead-result">完成后：{s.result}</div>
      {sub && <p className="cc-stephead-sub">{sub}</p>}
    </div>
  )
}

/* ------------------------------------------------------------ 统一任务状态 */

/** 一键出全套的阶段键（顺序即执行顺序），中文只在渲染处映射，值一律用原样。 */
const STEP_ORDER = ['underwear', 'poses', 'emotions']
const STEP_LABEL: Record<string, string> = {
  underwear: '底图', poses: '姿势', emotions: '表情',
}

/** 已知阶段给「第几步/共几步 · 阶段名」；后端若加了新键就原样显示，不猜序号也不翻译。 */
function stageText(step: string): string {
  const i = STEP_ORDER.indexOf(step)
  if (i < 0) return step
  return `阶段 ${i + 1}/${STEP_ORDER.length} · ${STEP_LABEL[step]}`
}

export function TaskBar({ task, onDismiss }: {
  task: TaskState | null
  /** 任务条默认常驻（结果要看得见）；给了关闭回调才渲染 ✕ */
  onDismiss?: () => void
}) {
  if (!task) return null
  const pct = task.progress && task.progress.total > 0
    ? Math.min(100, Math.round((task.progress.done / task.progress.total) * 100)) : 0
  const stage = task.progress?.step ? stageText(task.progress.step) : ''
  return (
    <div className={`cc-task is-${task.status}`} data-state={task.status}>
      <div className="cc-task-head">
        <span className="cc-task-dot" />
        <b>{task.label}</b>
        {stage && <span className="cc-task-step">{stage}</span>}
        {onDismiss && (
          <button type="button" className="cc-task-x" onClick={onDismiss} title="收起">✕</button>
        )}
      </div>
      <p className="cc-task-msg">{task.message}</p>
      {task.status === 'running' && task.progress && (
        <div className="chars-prog">
          <div className="chars-prog-bar" style={{ width: `${pct}%` }} />
          <span>{Math.round(task.progress.done)}/{Math.round(task.progress.total)}</span>
        </div>
      )}
      {task.status === 'running' && !task.progress && (
        <div className="chars-prog cc-prog-indet"><div className="chars-prog-bar" /></div>
      )}
    </div>
  )
}

/* --------------------------------------------------------------- 通用容器 */

/** 渐进披露：进阶参数一律收进 details，主流程只留必要的选择题。 */
export function Adv({ title, children, defaultOpen = false }: {
  title: string
  children: ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details className="cc-adv" open={defaultOpen || undefined}>
      <summary>{title}</summary>
      <div className="cc-adv-body">{children}</div>
    </details>
  )
}

export function Card({ title, note, children, tone }: {
  title: string
  note?: string
  children: ReactNode
  tone?: 'primary' | 'plain'
}) {
  return (
    <section className={`cc-card${tone === 'primary' ? ' is-primary' : ''}`}>
      <h3 className="cc-card-title">{title}</h3>
      {note && <p className="cc-card-note">{note}</p>}
      {children}
    </section>
  )
}

/** 前置条件没满足时的挡路卡：明确说清缺什么、一键回到缺的那一步。
 *  以前是按钮可点、点了才 400，用户根本不知道自己少了什么。 */
export function Blocker({ text, action, onAction }: {
  text: string
  action: string
  onAction: () => void
}) {
  return (
    <div className="cc-blocker">
      <span className="cc-blocker-icon">!</span>
      <p>{text}</p>
      <button type="button" className="cc-btn ghost" onClick={onAction}>{action}</button>
    </div>
  )
}

/* ------------------------------------------------------------ 属性字段 UI */

/** 结构化选择题：chips + 一个「选」按钮打开官方标签库。
 *  不再让用户对着中文界面手写英文（写了还跑偏）。 */
export function FieldChips({ label, value, onPick }: {
  label: string
  value: string
  onPick: () => void
}) {
  const chips = value.split(',').map((s) => s.trim()).filter(Boolean)
  return (
    <div className="chars-field cc-field" title={`${label}：点「选」从标签库里挑，也可以直接改`}>
      <div className="chars-field-label">{label}</div>
      <div className="chars-tags">
        {chips.length === 0 && <span className="chars-tags-empty">未设置</span>}
        {chips.map((c) => <span className="chars-tag" key={c}>{c}</span>)}
        <button type="button" className="mini" onClick={onPick} title="从标签库里选（可多选）">
          {chips.length ? '改' : '选'}
        </button>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------- 画风选择 */

export type StyleOption = { id: string; label: string; group: string; zhLabel: string; zhGroup: string }

export function StylePicker({
  options, groups, current, query, group, open,
  onQuery, onGroup, onToggle, onPick,
}: {
  options: StyleOption[]
  groups: string[]
  current: string
  query: string
  group: string
  open: boolean
  onQuery: (v: string) => void
  onGroup: (v: string) => void
  onToggle: () => void
  onPick: (id: string) => void
}) {
  const filtered = options.filter((s) => {
    if (group && s.group !== group) return false
    const q = query.trim().toLowerCase()
    if (!q) return true
    return s.label.toLowerCase().includes(q) || s.zhLabel.toLowerCase().includes(q)
      || s.group.toLowerCase().includes(q) || s.zhGroup.toLowerCase().includes(q)
  })
  const currentLabel = options.find((s) => s.id === current)?.zhLabel || current
  return (
    <div className="chars-style cc-style">
      <div className="chars-style-head">
        <div className="cc-style-current">
          <span className="cc-kicker">当前画风</span>
          <b>{currentLabel || '尚未选择（默认画风）'}</b>
        </div>
        <span className="spacer" />
        <button type="button" className="cc-btn ghost" onClick={onToggle}>
          {open ? '收起画风库' : '浏览画风'}
        </button>
      </div>
      <div className="cc-style-quick">
        {options.slice(0, 8).map((s) => (
          <button type="button" key={s.id}
            className={`cc-style-chip${current === s.id ? ' on' : ''}`}
            onClick={() => onPick(s.id)} title={`${s.zhGroup} · ${s.zhLabel}`}>
            {s.zhLabel}
          </button>
        ))}
      </div>
      {open && (
        <>
          <div className="chars-style-toolbar">
            <input className="chars-style-search" type="search"
              placeholder="搜索画风（中文名 / 英文名 / 分类）…"
              value={query} onChange={(e) => onQuery(e.target.value)} />
            <select value={group} onChange={(e) => onGroup(e.target.value)}>
              <option value="">全部分类</option>
              {groups.map((g) => <option key={g} value={g}>{zh(STYLE_GROUP_LABEL, g)}</option>)}
            </select>
            <span className="chars-sub">{filtered.length} 条</span>
          </div>
          <div className="chars-style-grid">
            {filtered.map((s) => (
              <button type="button" key={s.id}
                className={`chars-style-card${current === s.id ? ' on' : ''}`}
                title={`${s.zhGroup} · ${s.zhLabel}`}
                onClick={() => onPick(s.id)}>
                <img src={stylePreviewUrl(s.id)} alt={s.zhLabel} loading="lazy" />
                <span>{s.zhLabel}</span>
              </button>
            ))}
          </div>
          {filtered.length === 0 && <p className="chars-sub">没有匹配的画风，换个词试试。</p>}
        </>
      )}
    </div>
  )
}

/* -------------------------------------------------------------- 表情选择 */

export type EmotionOption = { key: string; safe_name: string; label: string; prompt: string }

export function EmotionGrid({ options, selected, onToggle }: {
  options: EmotionOption[]
  selected: string[]
  onToggle: (name: string) => void
}) {
  return (
    <div className="chars-emo-grid">
      {options.map((e) => {
        const on = selected.includes(e.safe_name)
        return (
          <button type="button" key={e.safe_name}
            className={`chars-emo${on ? ' on' : ''}`}
            title={e.prompt || e.key}
            onClick={() => onToggle(e.safe_name)}>
            <img src={emotionIconUrl(e.safe_name)} alt={e.label} loading="lazy" />
            <span>{e.label}</span>
          </button>
        )
      })}
    </div>
  )
}

/* -------------------------------------------------------------- 服装表单 */

/** 官方 COSTUME_TEXT_FIELDS 六个字段。受控：AI 拆服装要能直接灌进表单。 */
export function CostumeForm({ fields, onChange, onSave }: {
  fields: Record<string, string>
  onChange: (d: Record<string, string>) => void
  onSave: (d: Record<string, string>) => void
}) {
  const d: Record<string, string> = {
    top: '', bottom: '', shoes: '', head: '', face: '', negative_prompt: '', ...fields,
  }
  const labels: Array<[string, string]> = [
    ['top', '上装'], ['bottom', '下装'], ['shoes', '鞋'],
    ['head', '头部'], ['face', '脸部'], ['negative_prompt', '不要出现'],
  ]
  return (
    <div className="chars-costume">
      {labels.map(([k, label]) => (
        <label className="chars-field" key={k}>
          {label}
          <input value={d[k] ?? ''}
            onChange={(e) => onChange({ ...d, [k]: e.target.value })} />
        </label>
      ))}
      <button type="button" className="cc-btn" onClick={() => onSave(d)}>保存服装</button>
    </div>
  )
}

/* ------------------------------------------------------------ 右栏：预览 */

/** 常驻成果区。业界同类产品的共同点：画布/结果始终在视野里，
 *  不让用户在一堆表单里猜"我刚才到底生成出来没有"。 */
export function PreviewPane({
  character, costume, emotion, costumes, emotions, files, task,
  doneFlash, onDismissFlash, onCostume, onEmotion, onMeta, atlasInfo,
}: {
  character: string
  costume: string
  emotion: string
  costumes: string[]
  emotions: string[]
  files: string[]
  task: TaskState | null
  doneFlash: { text: string; at: number } | null
  onDismissFlash: () => void
  onCostume: (v: string) => void
  onEmotion: (v: string) => void
  onMeta: (m: { count: number; files: string[] }) => void
  atlasInfo: AtlasInfo | null
}) {
  return (
    <aside className="cc-preview">
      <div className="cc-preview-head">
        <b>角色成果</b>
        <span className="cc-preview-name" title={character}>{character}</span>
      </div>

      <div className="cc-preview-stage">
        <SpritePreview character={character} costume={costume} emotion={emotion} onMeta={onMeta} />
      </div>

      <div className="cc-preview-pick">
        <label>服装
          <select value={costume} onChange={(e) => onCostume(e.target.value)}>
            {costumes.map((c) => <option key={c} value={c}>{zh(COSTUME_LABEL, c)}</option>)}
          </select>
        </label>
        <label>表情
          <select value={emotion} onChange={(e) => onEmotion(e.target.value)}>
            {emotions.map((e) => <option key={e} value={e}>{emotionLabel(e, e)}</option>)}
          </select>
        </label>
      </div>

      {files.length > 0 && (
        <div className="chars-asset-row cc-asset-row">
          <span className="chars-asset-label">这套共 {files.length} 帧</span>
          {files.map((u, i) => (
            <img key={u} src={u} alt={`第 ${i + 1} 帧`} loading="lazy"
              title={`第 ${i + 1} 帧`} />
          ))}
        </div>
      )}

      <div className="cc-preview-stats">
        <div><b>{costumes.length}</b><span>套服装</span></div>
        <div><b>{files.length}</b><span>当前套帧数</span></div>
        <div><b>{atlasInfo?.exists ? '已导出' : '未导出'}</b><span>图集</span></div>
      </div>

      {doneFlash && (
        <div className="chars-done cc-done" key={doneFlash.at}>
          <span className="chars-done-icon">✓</span>
          <span className="chars-done-text">{doneFlash.text}</span>
          <button type="button" className="mini" onClick={onDismissFlash}>✕</button>
        </div>
      )}
      <TaskBar task={task} />
    </aside>
  )
}

/* --------------------------------------------------------------- 小工具 */

export function BgSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <label className="cc-inline-field">出图底色
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {BG_COLORS.map((c) => (
          <option key={c} value={c}>{zh(BG_COLOR_LABEL, c)}</option>
        ))}
      </select>
    </label>
  )
}
