/**
 * 提示词「放大编辑」按钮。
 *
 * 为什么需要：节点卡片和角色中心的输入框都只有 2~3 行高，写一段几百字的
 * 角色描述时**只能看到尾巴**，改中间几个词要来回滚。业界做法（Civitai /
 * SD WebUI / ComfyUI 的 prompt 弹窗）都是给一个放大入口，弹到屏幕中央编辑。
 *
 * 用法：放在 `position: relative` 的父容器里（如 `.prompt-wrap` / `.chars-field`），
 * 按钮自己绝对定位到右下角。弹窗走 Portal 到 `document.body` —— 节点卡片有
 * `overflow: hidden` 和 `transform`，直接渲染会被裁掉、也会跟着画布缩放。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

export type PromptZoomProps = {
  value: string
  onChange: (v: string) => void
  /** 弹窗标题，同时也是按钮的 title 提示 */
  label?: string
  placeholder?: string
  /** 弹窗里额外显示的说明（例如「英文描述效果更好」） */
  hint?: string
  disabled?: boolean
}

export default function PromptZoom({
  value, onChange, label = '提示词', placeholder, hint, disabled,
}: PromptZoomProps) {
  const [open, setOpen] = useState(false)
  const areaRef = useRef<HTMLTextAreaElement>(null)
  const returnFocus = useRef<HTMLElement | null>(null)

  const close = useCallback(() => {
    setOpen(false)
    // 焦点还给刚才那个输入框，免得用户关掉弹窗后键盘输入落到页面上
    returnFocus.current?.focus?.()
  }, [])

  useEffect(() => {
    if (open) areaRef.current?.focus()
  }, [open])

  // Esc 关闭：弹窗是覆盖式的，没有键盘出口会憋住用户
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); close() }
      // Ctrl/Cmd+Enter 也是「写完收工」的直觉动作
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); close() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, close])

  const lines = value ? value.split('\n').length : 0

  return (
    <>
      <button
        type="button"
        className="pzoom"
        disabled={disabled}
        title={`放大编辑${label}（弹窗里可以看全整段）`}
        onClick={(e) => {
          returnFocus.current = e.currentTarget.parentElement
            ?.querySelector('textarea') as HTMLElement | null
          setOpen(true)
        }}
      >
        ⤢
      </button>
      {open && createPortal(
        <div
          className="pzoom-mask"
          onMouseDown={(e) => { if (e.target === e.currentTarget) close() }}
        >
          <div className="pzoom-panel" role="dialog" aria-label={`编辑${label}`}>
            <div className="pzoom-head">
              <b>{label}</b>
              <span className="pzoom-meta">{value.length} 字{lines > 1 ? ` · ${lines} 行` : ''}</span>
              <span className="spacer" />
              <button type="button" className="pzoom-done" onClick={close}>完成</button>
            </div>
            <textarea
              ref={areaRef}
              className="pzoom-area"
              value={value}
              placeholder={placeholder}
              spellCheck={false}
              onChange={(e) => onChange(e.target.value)}
            />
            <div className="pzoom-foot">
              {hint ? <span className="pzoom-hint">{hint}</span> : <span />}
              <span className="pzoom-keys">Esc 或 Ctrl+Enter 关闭</span>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </>
  )
}
