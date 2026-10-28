/**
 * 删除确认框。
 *
 * 「删节点」和「删文件」是两件事，所以给两个出口：
 *   - 仅从画布移除：节点消失，磁盘上的产物留着（后悔了还能在 assets 里找回）
 *   - 真删：连磁盘上的 PNG / sidecar / raw 一起删（后端 DELETE /assets）
 *
 * 真删是**不可逆**的，所以按钮默认是危险色；如果这些文件还被别的节点当源图，
 * 真删会被禁用并说明原因——否则会默默把下游节点变成「没有源的图生图」。
 */
type FileItem = { name: string; path: string }

type Props = {
  title: string
  summary?: string
  files?: FileItem[]
  /** 补充说明（会被删掉的到底是什么） */
  note?: string
  /** 阻止真删的原因 */
  blockReason?: string
  /** 两个动作按钮的文案（删节点 / 删项目说法不一样） */
  removeLabel?: string
  deleteLabel?: string
  onCancel: () => void
  onRemove: () => void
  onDelete: () => void
}

export default function ConfirmDialog({
  title, summary, files, note, blockReason,
  removeLabel = '仅移除节点', deleteLabel = '真删（含文件）',
  onCancel, onRemove, onDelete,
}: Props) {
  return (
    <div className="modal-mask" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span className="node-kind danger">{title}</span>
        </div>
        {summary && <p className="modal-summary">{summary}</p>}
        {files && files.length > 0 && (
          <ul className="modal-files">
            {files.map((f) => (
              <li key={f.path} title={f.path}>
                {f.name}
              </li>
            ))}
          </ul>
        )}
        {note && <p className="modal-note">{note}</p>}
        {blockReason && <p className="modal-warn">⚠ {blockReason}</p>}
        <div className="modal-actions">
          <button className="mini wide" onClick={onCancel}>
            取消
          </button>
          <button className="mini wide" onClick={onRemove} title="只从画布移除，磁盘文件保留">
            {removeLabel}
          </button>
          <button
            className="mini wide danger"
            onClick={onDelete}
            disabled={!!blockReason}
            title={blockReason || '删除磁盘上的文件（不可恢复）'}
          >
            {deleteLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
