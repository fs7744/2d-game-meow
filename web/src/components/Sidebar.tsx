import { useState, type ReactNode } from 'react'
import type { FavoritePrompt, LoraInfo } from '../types'

type Props = {
  loras: LoraInfo[]
  favs: FavoritePrompt[]
  onAddNode: () => void
  canApply: boolean
  onApplyFav: (text: string) => void
  onRemoveFav: (id: string) => void
  models: string[]
  model: string
  onModel: (m: string) => void
  matteBackend: string
  onMatteBackend: (v: string) => void
}

const MODEL_LABEL: Record<string, string> = {
  sd15: 'SD1.5 · 快（默认）',
  sdxl: 'SDXL 1.0 · 更好但慢约 3×',
}

const MATTE_LABEL: Record<string, string> = {
  auto: '自动（有分割模型就抠得干净）',
  u2net: '语义分割 · 准但慢',
  color: '纯色键控 · 极快',
}

function Section(
  { icon, title, count, children, defaultOpen = true }:
  { icon: string; title: string; count?: number; children: ReactNode; defaultOpen?: boolean },
) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`rail-sec${open ? ' open' : ''}`}>
      <button className="rail-sec-head" onClick={() => setOpen(!open)}>
        <span className="rail-sec-arrow">{open ? '▾' : '▸'}</span>
        <span className="rail-sec-icon">{icon}</span>
        <span className="rail-sec-title">{title}</span>
        {count ? <span className="rail-count">{count}</span> : null}
      </button>
      {open && <div className="rail-sec-body">{children}</div>}
    </div>
  )
}

/**
 * 左边栏：菜单式层级，可整条收起。
 *
 * 之前节点列表和提示词收藏是两个各自浮在左上角的卡片，节点一多就把画布左上角
 * 占满、互相挤压。改成一条可收起的侧栏后，画布的可视区域稳定了，也能放下设置。
 */
export default function Sidebar({
  loras, favs, onAddNode, canApply, onApplyFav, onRemoveFav,
  models, model, onModel, matteBackend, onMatteBackend,
}: Props) {
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <aside className="rail collapsed">
        <button className="rail-toggle" title="展开侧栏" onClick={() => setCollapsed(false)}>
          »
        </button>
        <button className="rail-icon" title="节点" onClick={() => { setCollapsed(false); onAddNode() }}>
          ▣
        </button>
        <button className="rail-icon" title={`提示词收藏 ${favs.length}`} onClick={() => setCollapsed(false)}>
          ☆{favs.length ? <sup>{favs.length}</sup> : null}
        </button>
        <button className="rail-icon" title={`底模 ${model}`} onClick={() => setCollapsed(false)}>
          ⚙
        </button>
      </aside>
    )
  }

  return (
    <aside className="rail">
      <div className="rail-head">
        <span className="rail-brand">面板</span>
        <span className="spacer" />
        <button className="rail-toggle" title="收起侧栏" onClick={() => setCollapsed(true)}>
          «
        </button>
      </div>

      <div className="rail-scroll">
        <Section icon="▣" title="节点">
          <div className="rail-item">
            <span className="rail-item-main">
              <b>图生成</b>
              <em>文生图 · 可复制删除</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={onAddNode}>+</button>
          </div>
          <p className="rail-tip">拖这一项到画布任意位置，也能在那里新建节点。</p>
        </Section>

        <Section icon="☆" title="提示词收藏" count={favs.length}>
          {favs.length === 0 ? (
            <p className="rail-tip">
              在节点的提示词框点 <b>☆</b> 收藏。收藏后点这里可填回选中的节点。
            </p>
          ) : (
            favs.map((f) => (
              <div className="rail-item" key={f.id}>
                <button
                  className="rail-item-main"
                  disabled={!canApply}
                  title={canApply ? '填入选中的节点' : '先在画布上选中一个节点'}
                  onClick={() => onApplyFav(f.text)}
                >
                  {f.text}
                </button>
                <button className="rail-del" title="删除" onClick={() => onRemoveFav(f.id)}>
                  ✕
                </button>
              </div>
            ))
          )}
        </Section>

        <Section icon="⚙" title="设置">
          <label className="block">
            底模
            <select value={model} onChange={(e) => onModel(e.target.value)}>
              {models.length === 0 && <option value={model}>{model}（仅本机已装）</option>}
              {models.map((m) => (
                <option key={m} value={m}>{MODEL_LABEL[m] || m}</option>
              ))}
            </select>
          </label>
          <p className="rail-tip">
            {model === 'sdxl'
              ? 'SDXL 画质更好，但 SD1.5 的 LoRA 不能通用（下面会自动只留 SDXL 可用的），且 W3/W4 暂不支持。'
              : 'SD1.5 快、生态全（ControlNet / IP-Adapter / LoRA 都齐）。W3 局部重绘与 W4 姿态驱动只能用它。'}
          </p>

          <label className="block">
            抠图方式
            <select value={matteBackend} onChange={(e) => onMatteBackend(e.target.value)}>
              {Object.keys(MATTE_LABEL).map((k) => (
                <option key={k} value={k}>{MATTE_LABEL[k]}</option>
              ))}
            </select>
          </label>

          <div className="rail-stat">
            LoRA {loras.length} 个
          </div>
        </Section>
      </div>
    </aside>
  )
}
