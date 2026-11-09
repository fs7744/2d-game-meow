/**
 * 角色中心的表单控件：官方服装六字段 + 官方表情库网格。
 *
 * 抽成独立文件是为了让立绘工具箱（PortraitTools）也能直接用，而不必
 * 反向 import Controls（Controls 已经 import 了 PortraitTools，会成环）。
 */
import { emotionIconUrl } from '../../api'

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

