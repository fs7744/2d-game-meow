/**
 * 属性标签选择器（对应官方 Character Creator 里那排「Choose Presets」弹窗）。
 *
 * 多选 chips，**点一下选中/取消**，关闭时用官方 `presetSelection.value()`
 * 拼回逗号分隔的文本。两个细节照官方：
 *
 * - chip 上显示 `label`，`title` 放 `prompt`（官方 `character_creator_v2.js:4050`）
 *   —— tag 名很短，真正要看出图会怎么写就得看 prompt。
 * - 分组标题来自 `item.group`（如 "Humanoid peoples"），没有就用分类名。
 *   hair 这种跨 6 个分类的字段，没有分组标题就是一锅粥。
 */
import { useEffect, useMemo, useState } from 'react'
import { presetGroups, presetSelection, type TagCatalog, type TagItem } from '../store/characterTags'

type Props = {
  open: boolean
  field: string
  label: string
  catalog: TagCatalog | null
  value: string
  onClose: () => void
  onApply: (value: string) => void
}

export default function TagPicker({
  open, field, label, catalog, value, onClose, onApply,
}: Props) {
  const groups = useMemo(
    () => presetGroups(catalog, field), [catalog, field])
  const [local, setLocal] = useState(value)

  // 每次打开都**从当前字段值重新解析**：用户在输入框里手改过，
  // 弹窗再打开时必须显示改后的样子，否则会把手写的内容悄悄覆盖掉。
  useEffect(() => {
    if (open) setLocal(value)
  }, [open, value])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const sel = presetSelection(local, groups)
  const toggle = (item: TagItem) => {
    const next = presetSelection(local, groups)
    next.toggle(item)
    setLocal(next.value())
  }

  return (
    <div className="tag-mask" onClick={onClose}>
      <div className="tag-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <b>选择 {label}</b>
          <button type="button" className="modal-x" onClick={onClose} title="关闭">✕</button>
        </div>
        {groups.length === 0 ? (
          <p className="tag-empty">
            官方标签库里没有这一项的可选值（{field}），手填就行。
          </p>
        ) : (
          <div className="tag-grid">
            {groups.map((g) => (
              <div key={g.header} className="tag-group">
                <div className="tag-group-head">{g.header}</div>
                {g.items.map((item) => (
                  <button
                    key={item.tag}
                    type="button"
                    className={`tag-chip${sel.has(item) ? ' on' : ''}`}
                    title={item.prompt || item.tag.split('_').join(' ')}
                    onClick={() => toggle(item)}
                  >
                    {item.label || item.tag.split('_').join(' ')}
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
        <div className="tag-foot">
          {/* 这个值就是会写进 prompt 的样子 —— 让用户点「确定」之前先看见 */}
          <label className="tag-result">
            写入内容（也可直接改）
            <input
              value={local}
              onChange={(e) => setLocal(e.target.value)}
              placeholder="留空就是不要这一项"
            />
          </label>
          <div className="tag-act">
            <button type="button" className="mini" onClick={onClose}>取消</button>
            <button type="button" className="mini wide" onClick={() => {
              onApply(local)
              onClose()
            }}>
              确定
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
