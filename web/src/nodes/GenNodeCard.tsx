import { useRef } from 'react'
import { hasCJK } from '../store/prompts'
import type { GenNode, Preset } from '../types'

export const NODE_W = 300

export type NodePatch = Partial<Pick<GenNode, 'x' | 'y' | 'params' | 'view' | 'job'>>

type Props = {
  node: GenNode
  selected: boolean
  zoom: number
  onSelect: () => void
  onPatch: (p: NodePatch) => void
  onGenerate: () => void
  onDuplicate: () => void
  onDelete: () => void
  favorited: boolean
  onToggleFavorite: () => void
}

const PRESETS: { v: Preset; label: string }[] = [
  { v: 'sprite', label: '像素精灵' },
  { v: 'portrait', label: '立绘' },
  { v: 'none', label: '不套契约' },
]

export default function GenNodeCard({
  node,
  selected,
  zoom,
  onSelect,
  onPatch,
  onGenerate,
  onDuplicate,
  onDelete,
  favorited,
  onToggleFavorite,
}: Props) {
  const dragRef = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null)
  const p = node.params
  const job = node.job
  const busy = job?.status === 'queued' || job?.status === 'running'
  const hasImage = !!job?.url
  const imageMode = node.view === 'image' && hasImage

  const startDrag = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    e.stopPropagation()
    onSelect()
    dragRef.current = { sx: e.clientX, sy: e.clientY, ox: node.x, oy: node.y }
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  }

  const moveDrag = (e: React.PointerEvent) => {
    const d = dragRef.current
    if (!d) return
    onPatch({
      x: Math.round(d.ox + (e.clientX - d.sx) / zoom),
      y: Math.round(d.oy + (e.clientY - d.sy) / zoom),
    })
  }

  const endDrag = (e: React.PointerEvent) => {
    if (!dragRef.current) return
    dragRef.current = null
    try {
      ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
    } catch {
      /* 已释放 */
    }
  }

  const setParam = <K extends keyof typeof p>(k: K, v: (typeof p)[K]) =>
    onPatch({ params: { ...p, [k]: v } })

  return (
    <div
      className={`node gen${selected ? ' selected' : ''}${imageMode ? ' image-mode' : ''}`}
      style={{ left: node.x, top: node.y, width: NODE_W }}
      onPointerDown={(e) => {
        e.stopPropagation()
        onSelect()
      }}
    >
      <div
        className="node-head"
        onPointerDown={startDrag}
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <span className="node-kind">图生成</span>
        {job && <span className={`node-state s-${job.status}`}>{statusLabel(job.status)}</span>}
        <span className="spacer" />
        {hasImage && (
          <button
            className="icon-btn"
            title={imageMode ? '切回参数视图' : '只看图'}
            onClick={(e) => {
              e.stopPropagation()
              onPatch({ view: imageMode ? 'expanded' : 'image' })
            }}
          >
            {imageMode ? '▤' : '▣'}
          </button>
        )}
        <button
          className="icon-btn"
          title="复制节点 (Ctrl+D)"
          onClick={(e) => {
            e.stopPropagation()
            onDuplicate()
          }}
        >
          ⧉
        </button>
        <button
          className="icon-btn danger"
          title="删除节点 (Delete)"
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
        >
          ✕
        </button>
      </div>

      {imageMode ? (
        <div className="node-image-only">
          <img src={job!.url} alt={p.prompt} draggable={false} />
          <div className="img-badge">
            种子 {job!.seed ?? p.seed}
            {typeof job!.seconds === 'number' ? ` · ${job!.seconds}s` : ''}
          </div>
        </div>
      ) : (
        <div className="node-body">
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={3}
              placeholder="English prompt, e.g. white cat knight, red cape, katana, front view"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
            <button
              className={`star${favorited ? ' on' : ''}`}
              title={favorited ? '取消收藏' : '收藏这条提示词'}
              disabled={!p.prompt.trim()}
              onClick={(e) => {
                e.stopPropagation()
                onToggleFavorite()
              }}
            >
              {favorited ? '★' : '☆'}
            </button>
          </div>
          <div className={`lang-hint${hasCJK(p.prompt) ? ' bad' : ''}`}>
            {hasCJK(p.prompt)
              ? '⚠ 检测到中文：SD1.5 的 CLIP 只认英文，中文会被拆成乱 token，出图基本靠猜'
              : 'SD1.5 只支持英文提示词；逗号分隔，主体 + 特征 + 视角'}
          </div>

          <div className="row">
            <label>
              宽
              <input
                type="number"
                value={p.width}
                step={64}
                min={128}
                max={1024}
                onChange={(e) => setParam('width', Number(e.target.value) || 512)}
              />
            </label>
            <label>
              高
              <input
                type="number"
                value={p.height}
                step={64}
                min={128}
                max={1024}
                onChange={(e) => setParam('height', Number(e.target.value) || 512)}
              />
            </label>
            <label>
              步数
              <input
                type="number"
                value={p.steps}
                min={4}
                max={50}
                onChange={(e) => setParam('steps', Number(e.target.value) || 25)}
              />
            </label>
          </div>

          <div className="row">
            <label className="grow">
              种子
              <input
                type="number"
                value={p.seed}
                onChange={(e) => setParam('seed', Number(e.target.value) || 0)}
              />
            </label>
            <button
              className="mini"
              title="随机种子"
              onClick={() => setParam('seed', -1)}
            >
              🎲
            </button>
            <label className="grow">
              风格
              <select
                value={p.preset}
                onChange={(e) => setParam('preset', e.target.value as Preset)}
              >
                {PRESETS.map((x) => (
                  <option key={x.v} value={x.v}>
                    {x.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <details className="fold">
            <summary>更多设置</summary>
            <div className="fold-body">
              <label className="block">
                反向提示词
                <textarea
                  rows={2}
                  placeholder="留空则用内置契约"
                  value={p.negative}
                  onChange={(e) => setParam('negative', e.target.value)}
                />
              </label>
              <div className="row">
                <label>
                  CFG
                  <input
                    type="number"
                    step={0.5}
                    value={p.cfg}
                    onChange={(e) => setParam('cfg', Number(e.target.value) || 7.5)}
                  />
                </label>
                <label>
                  像素块
                  <input
                    type="number"
                    min={0}
                    max={16}
                    value={p.pixel}
                    onChange={(e) => setParam('pixel', Number(e.target.value) || 0)}
                  />
                </label>
                <label>
                  色数
                  <input
                    type="number"
                    min={0}
                    max={64}
                    value={p.palette}
                    onChange={(e) => setParam('palette', Number(e.target.value) || 0)}
                  />
                </label>
              </div>
              <label className="check">
                <input
                  type="checkbox"
                  checked={p.matte}
                  onChange={(e) => setParam('matte', e.target.checked)}
                />
                抠图出透明底
              </label>
            </div>
          </details>

          {hasImage && (
            <div className="preview" onClick={() => onPatch({ view: 'image' })} title="点击只看图">
              <img src={job!.url} alt={p.prompt} draggable={false} />
            </div>
          )}

          <button className="gen-btn" disabled={busy || !p.prompt.trim()} onClick={onGenerate}>
            {busy ? (job?.status === 'queued' ? '排队中…' : '生成中…') : '生成'}
          </button>

          {job?.status === 'failed' && <div className="err">{job.error}</div>}
          {job?.status === 'done' && (
            <div className="meta">
              种子 {job.seed ?? p.seed}
              {typeof job.seconds === 'number' ? ` · ${job.seconds}s` : ''}
              {typeof job.peakVram === 'number' ? ` · ${job.peakVram}GB` : ''}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function statusLabel(s: string): string {
  switch (s) {
    case 'queued':
      return '排队'
    case 'running':
      return '生成中'
    case 'done':
      return '完成'
    case 'failed':
      return '失败'
    default:
      return s
  }
}
