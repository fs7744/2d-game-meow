type Props = {
  onAdd: (type: string) => void
}

const NODE_TYPES = [
  { type: 'gen', label: '图生成', desc: '提示词 → 像素素材', accent: 'green' as const },
  { type: 'img2img', label: '图生图', desc: '已有图 → 改内容', accent: 'cyan' as const },
]

export default function NodePalette({ onAdd }: Props) {
  return (
    <div className="palette">
      <div className="palette-head">节点</div>
      {NODE_TYPES.map((n) => (
        <div
          key={n.type}
          className={`palette-item accent-${n.accent}`}
          draggable
          onDragStart={(e) => {
            e.dataTransfer.setData('application/x-node-type', n.type)
            e.dataTransfer.effectAllowed = 'copy'
          }}
          title="拖到画布，或点 + 放到原点"
        >
          <span className="dot" />
          <span className="name">{n.label}</span>
          <span className="desc">{n.desc}</span>
          <button
            className="add"
            onClick={(e) => {
              e.stopPropagation()
              onAdd(n.type)
            }}
            title="添加到画布原点 (0,0)"
          >
            +
          </button>
        </div>
      ))}
      <div className="palette-tip">
        拖拽到画布任意位置；点 <b>+</b> 放到原点
      </div>
    </div>
  )
}
