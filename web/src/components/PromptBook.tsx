import type { FavoritePrompt } from '../types'

type Props = {
  items: FavoritePrompt[]
  canApply: boolean
  onApply: (text: string) => void
  onRemove: (id: string) => void
}

export default function PromptBook({ items, canApply, onApply, onRemove }: Props) {
  return (
    <div className="book">
      <div className="book-head">
        提示词收藏
        <span className="book-count">{items.length}</span>
      </div>

      {items.length === 0 ? (
        <div className="book-empty">
          在节点的提示词框右上角点 <b>☆</b> 收藏常用提示词（中英文都行）。
        </div>
      ) : (
        <div className="book-list">
          {items.map((p) => (
            <div key={p.id} className="book-item" title={canApply ? p.text : '先选中一个节点'}>
              <button
                className="book-text"
                disabled={!canApply}
                onClick={() => onApply(p.text)}
                title={canApply ? '应用到选中的节点' : '先选中一个节点'}
              >
                {p.text}
              </button>
              <button className="book-del" title="删除收藏" onClick={() => onRemove(p.id)}>
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="book-tip">
        {canApply ? '点条目即可填入选中节点的提示词' : '选中一个节点后，点条目填入提示词'}
      </div>
    </div>
  )
}
