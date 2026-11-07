/**
 * 多角色 / 整角色变换面板（PoseStudio.tsx 里的一块）。
 *
 * 用户要的是「两个角色同框摆互动姿势」，所以这里必须能：加/删/选角色、
 * 整角色移动/旋转（含 gizmo，在 iframe 里）、World/Local 轴、联动全部角色、
 * 每个姿势 tab 独立位置，以及直接输 XYZ / 拖 Radar 做精确可复现的摆放。
 *
 * 状态**不在组件里**：唯一来源是 iframe 回推的 `sceneState`，所有操作都
 * 变成 `{type:'scene', action}` 消息发出去，改完再回推。这样存档、撤销、
 * 导出看到的永远是同一份状态，不会出现「界面一套、引擎一套」。
 */
import { useEffect, useState } from 'react'
import type { PoseScene, PosePlacement } from '../types'

type Props = {
  scene: PoseScene | null
  history: { canUndo: boolean; canRedo: boolean }
  disabled: boolean
  onAction: (action: string, payload?: Record<string, unknown>) => void
}

/** 数字框：只在失焦 / 回车时提交，别每敲一个字符就写一次撤销栈。 */
function NumField({ label, value, step = 0.1, min, max, disabled, onCommit }: {
  label: string
  value: number
  step?: number
  min?: number
  max?: number
  disabled: boolean
  onCommit: (v: number) => void
}) {
  const [text, setText] = useState(String(value))
  useEffect(() => { setText(String(Number(value.toFixed(3)))) }, [value])
  const commit = () => {
    const n = Number(text)
    if (Number.isFinite(n) && n !== value) onCommit(n)
    else setText(String(Number(value.toFixed(3))))
  }
  return (
    <label className="pose-num" title={`${label}（回车或失焦生效）`}>
      <span>{label}</span>
      <input
        type="number" value={text} step={step} min={min} max={max} disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
      />
    </label>
  )
}

/** Radar：X 横轴 / Y 纵轴，跟官方一样可拖可输，范围 ±20 个世界单位。 */
function Radar({ placement, disabled, onMove }: {
  placement: PosePlacement
  disabled: boolean
  onMove: (x: number, y: number, commit: boolean) => void
}) {
  const R = 20
  const [dragging, setDragging] = useState(false)
  const px = ((placement.x + R) / (2 * R)) * 100
  const py = ((R - placement.y) / (2 * R)) * 100
  const move = (e: React.PointerEvent<HTMLDivElement>, commit: boolean) => {
    const box = e.currentTarget.getBoundingClientRect()
    const fx = Math.max(0, Math.min(1, (e.clientX - box.left) / box.width))
    const fy = Math.max(0, Math.min(1, (e.clientY - box.top) / box.height))
    onMove(Number((fx * 2 * R - R).toFixed(2)), Number((R - fy * 2 * R).toFixed(2)), commit)
  }
  return (
    <div className="pose-radar-wrap">
      <div
        className={`pose-radar${disabled ? ' is-off' : ''}`}
        title="拖动摆 X / Y（与下面的数值框同步）"
        onPointerDown={(e) => {
          if (disabled) return
          e.currentTarget.setPointerCapture(e.pointerId)
          setDragging(true)
          move(e, false)
        }}
        onPointerMove={(e) => { if (dragging) move(e, false) }}
        onPointerUp={(e) => { if (dragging) { move(e, true); setDragging(false) } }}
      >
        <span className="pose-radar-axis x" /><span className="pose-radar-axis y" />
        <span className="pose-radar-dot" style={{ left: `${px}%`, top: `${py}%` }} />
      </div>
      <div className="pose-radar-legend">X 横 / Y 纵 · ±20</div>
    </div>
  )
}

export default function PoseScenePanel({ scene, history, disabled, onAction }: Props) {
  if (!scene) return null
  const active = scene.characters.find((c) => c.id === scene.activeId) || scene.characters[0]
  const p = scene.placements[scene.activeTab]?.[active?.id] || {
    x: 0, y: 0, z: 0, zoom: 1, rotation: [0, 0, 0] as [number, number, number],
  }
  const put = (patch: Partial<PosePlacement>, record = true) =>
    onAction('transform', { id: active.id, patch, record })
  const rot = (i: number, v: number) => {
    const next = [...p.rotation] as [number, number, number]
    next[i] = v
    put({ rotation: next })
  }

  return (
    <div className="pose-scene">
      <div className="pose-scene-head">
        <b>多角色 · 整角色变换</b>
        <button type="button" className="mini" disabled={disabled || !history.canUndo}
          title="撤销（Ctrl+Z）：姿势与整角色位移一起退" onClick={() => onAction('undo')}>↶</button>
        <button type="button" className="mini" disabled={disabled || !history.canRedo}
          title="重做（Ctrl+Shift+Z）" onClick={() => onAction('redo')}>↷</button>
        <button type="button" className="mini" disabled={disabled || scene.characters.length >= 12}
          title="加一个角色（复制当前姿势，可各自摆）" onClick={() => onAction('add')}>＋角色</button>
      </div>

      <div className="pose-scene-chars">
        {scene.characters.map((c) => (
          <span key={c.id} className={`pose-char${c.id === scene.activeId ? ' on' : ''}`}>
            <button type="button" className="pose-char-pick" disabled={disabled}
              title={`选中「${c.name}」摆它的关节`}
              onClick={() => onAction('select', { id: c.id })}>
              <i style={{ background: c.color || '#e8eaf2' }} />{c.name}
            </button>
            {scene.characters.length > 1 && (
              <button type="button" className="pose-char-del" disabled={disabled}
                title={`移除「${c.name}」`} onClick={() => onAction('remove', { id: c.id })}>×</button>
            )}
          </span>
        ))}
      </div>

      <div className="pose-scene-modes">
        {(['pose', 'move', 'rotate'] as const).map((m) => (
          <button key={m} type="button" className={`pose-mode${scene.mode === m ? ' on' : ''}`}
            disabled={disabled} onClick={() => onAction('mode', { value: m })}>
            {m === 'pose' ? '摆姿势' : m === 'move' ? '移动' : '旋转'}
          </button>
        ))}
        <button type="button" className={`pose-toggle${scene.space === 'local' ? ' on' : ''}`}
          disabled={disabled} title="切换整角色变换的坐标轴"
          onClick={() => onAction('space', { value: scene.space === 'world' ? 'local' : 'world' })}>
          {scene.space === 'world' ? '世界' : '本地'}
        </button>
        <button type="button" className={`pose-toggle${scene.linkAll ? ' on' : ''}`}
          disabled={disabled} title="移动 / 旋转全部角色（保持相对间距）"
          onClick={() => onAction('linkAll', { value: !scene.linkAll })}>联动</button>
      </div>

      <div className="pose-scene-tabs">
        <span className="pose-scene-kicker">姿势位</span>
        {scene.tabs.map((t, i) => (
          <button key={t} type="button" className={`pose-tab${t === scene.activeTab ? ' on' : ''}`}
            disabled={disabled} onClick={() => onAction('tab', { tabId: t })}>{i + 1}</button>
        ))}
        <button type="button" className="pose-tab" disabled={disabled || scene.tabs.length >= 16}
          title="新增一个姿势位：每个姿势位有独立的位置，切换时各自还原"
          onClick={() => onAction('addTab')}>＋</button>
      </div>

      <div className="pose-scene-grid">
        <NumField label="X" value={p.x} disabled={disabled} onCommit={(v) => put({ x: v })} />
        <NumField label="Y" value={p.y} disabled={disabled} onCommit={(v) => put({ y: v })} />
        <NumField label="Z" value={p.z} disabled={disabled} onCommit={(v) => put({ z: v })} />
        <NumField label="缩放" value={p.zoom} step={0.05} min={0.2} max={4}
          disabled={disabled} onCommit={(v) => put({ zoom: v })} />
        <NumField label="转X" value={p.rotation[0]} step={1} disabled={disabled}
          onCommit={(v) => rot(0, v)} />
        <NumField label="转Y" value={p.rotation[1]} step={1} disabled={disabled}
          onCommit={(v) => rot(1, v)} />
        <NumField label="转Z" value={p.rotation[2]} step={1} disabled={disabled}
          onCommit={(v) => rot(2, v)} />
        <button type="button" className="mini" disabled={disabled}
          title="把位置与旋转归零" onClick={() => put({
            x: 0, y: 0, z: 0, zoom: 1, rotation: [0, 0, 0],
          })}>归零</button>
      </div>

      <Radar placement={p} disabled={disabled}
        onMove={(x, y, commit) => put({ x, y }, commit)} />

      <div className="pose-scene-hint">
        {scene.mode === 'pose'
          ? '拖橙色关节摆当前角色的姿势；切到「移动/旋转」可在画布上直接拖整个人。'
          : scene.linkAll
            ? '联动开：拖一个角色，其它角色保持相对间距一起动。'
            : '只动选中的角色；勾「联动」可一起移动。撤销会连位置一起退回。'}
      </div>
    </div>
  )
}
