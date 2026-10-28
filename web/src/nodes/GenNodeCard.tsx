import { useRef } from 'react'
import { hasCJK } from '../store/prompts'
import type { GenNode, LoraInfo, Preset } from '../types'

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
  loras: LoraInfo[]
  model: string
}

const PRESETS: { v: Preset; label: string }[] = [
  { v: 'sprite', label: '像素精灵' },
  { v: 'portrait', label: '立绘' },
  { v: 'wuxia', label: '武侠 / 国风' },
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
  loras,
  model,
}: Props) {
  const dragRef = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null)
  const p = node.params
  const loraInfo = loras.find((l) => l.name === p.lora)
  const job = node.job
  const busy = job?.status === 'queued' || job?.status === 'running'
  const hasImage = !!job?.url
  const imageMode = node.view === 'image' && hasImage

  const startDrag = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    // 交互控件上绝不启动拖拽。一旦 setPointerCapture，后续 pointerup 会被重定向到
    // 卡头，浏览器按「pointerdown 与 pointerup 的最近公共祖先」派发 click ——
    // click 就落到卡头上而不是按钮上，按钮永远点不动（切回参数视图失效即此因）。
    const el = e.target as HTMLElement
    if (el.closest('button, input, textarea, select, summary, a, label')) return
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

  /**
   * 选 LoRA：同时把「推荐强度」和「匹配的提示词契约」一起带过来。
   * 不这么做的话，挂着武侠 LoRA 却仍在用像素契约，两边打架、两头不到位。
   */
  const chooseLora = (name: string) => {
    const info = loras.find((l) => l.name === name)
    const next = { ...p, lora: name }
    if (info) {
      next.loraScale = info.scale
      if (info.preset && p.preset === 'sprite') next.preset = info.preset
      // 注册表声明的「必须抠图 / 推荐 clip skip」一起带过来：像素 LoRA 都会画彩色底，
      // 不勾抠图就没法用；这两项用户基本不会自己想到要改
      if (typeof info.matte === 'boolean') next.matte = info.matte
      if (typeof info.clip_skip === 'number') next.clipSkip = info.clip_skip
    }
    onPatch({ params: next })
  }

  /** 触发词缺了等于白装 LoRA，所以给个一键补入，已包含则不重复加。 */
  const addTrigger = () => {
    const t = (loraInfo?.trigger || '').trim()
    if (!t) return
    const lower = p.prompt.toLowerCase()
    const missing = t.split(',').map((x) => x.trim()).filter(Boolean)
      .filter((x) => !lower.includes(x.toLowerCase()))
    if (!missing.length) return
    setParam('prompt', `${missing.join(', ')}, ${p.prompt}`.replace(/^, /, ''))
  }

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
            className={`icon-btn view-toggle${imageMode ? ' on' : ''}`}
            title={imageMode ? '切回参数视图' : '只看图'}
            onClick={(e) => {
              e.stopPropagation()
              onPatch({ view: imageMode ? 'expanded' : 'image' })
            }}
          >
            {imageMode ? '▤ 参数' : '▣'}
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
              <label className="block">
                LoRA
                <select value={p.lora} onChange={(e) => chooseLora(e.target.value)}>
                  <option value="">不挂（纯底模）</option>
                  {loras.map((l) => (
                    <option key={l.name} value={l.name}>
                      {l.label}
                    </option>
                  ))}
                </select>
              </label>
              {loraInfo && (
                <div className="lora-box">
                  <div className="lora-trigger">
                    <span>
                      触发词 <b>{loraInfo.trigger || '（无）'}</b>
                    </span>
                    {loraInfo.trigger && (
                      <button className="mini" title="补进提示词开头" onClick={addTrigger}>
                        + 填入
                      </button>
                    )}
                  </div>
                  {loraInfo.note && <p className="lora-note">{loraInfo.note}</p>}
                </div>
              )}
              <div className="row">
                <label>
                  LoRA 强度
                  <input
                    type="number"
                    step={0.05}
                    min={0}
                    max={1.5}
                    value={p.loraScale}
                    disabled={!p.lora}
                    onChange={(e) => setParam('loraScale', Number(e.target.value))}
                  />
                </label>
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
                <label title="1=最后一层（默认）；2=跳过最后一层，多数动漫/像素 LoRA 的训练设置">
                  跳过层
                  <input
                    type="number"
                    min={1}
                    max={4}
                    value={p.clipSkip}
                    disabled={model === 'sdxl'}
                    onChange={(e) => setParam('clipSkip', Number(e.target.value) || 1)}
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
          {/* SD1.5 画小体型角色时爱复制一份，负向词压不住；抠图后能量出来就报出来 */}
          {typeof job?.subjects === 'number' && job.subjects > 1 && (
            <div className="warn">
              ⚠ 画面里有 {job.subjects} 个主体（精灵只要一个）：换种子，或把「single character」
              提到提示词最前面
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
