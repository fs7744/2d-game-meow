/** 时间轴：摄影表、带贝塞尔手柄的曲线编辑器、播放控制与动画管理。 */
import { useRef, useState } from 'react'
import type { RigApi } from './useRig'
import { CURVES, fmtTime } from './types'

function scalar(v: any): number {
  if (Array.isArray(v)) return Number(v[0]) || 0
  return typeof v === 'number' ? v : 0
}

export function RigTimeline({ rig }: { rig: RigApi }) {
  const d = rig.docRef.current
  const [newName, setNewName] = useState('')
  const [track, setTrack] = useState(0)
  const [keyIdx, setKeyIdx] = useState(-1)
  const [curvePreset, setCurvePreset] = useState(0)
  const barRef = useRef<HTMLDivElement>(null)
  const tracks = rig.allKeys()
  const a = d.animations[rig.animation]
  const duration = a?.duration || 1
  const selected = tracks[track]
  const keys = selected?.keys || []
  const dragKey = useRef<{ idx: number; startX: number; startTime: number } | null>(null)
  const dragHandle = useRef<{ idx: number; handle: 0 | 1 } | null>(null)

  const px = (t: number) => `${(t / Math.max(0.001, duration)) * 100}%`
  const timeFromEvent = (e: React.PointerEvent | React.MouseEvent) => {
    const r = barRef.current!.getBoundingClientRect()
    return Math.max(0, Math.min(duration, ((e.clientX - r.left) / r.width) * duration))
  }

  // ---- 曲线：SVG 里画关键点与贝塞尔控制手柄，拖手柄改 key.curve ----
  const CW = 420, CH = 150, PAD = 16
  const vals = keys.map((k: any) => scalar(k.value))
  const min = Math.min(...vals, 0), max = Math.max(...vals, 1)
  const span = Math.max(1e-6, max - min)
  const cx = (t: number) => PAD + (t / Math.max(0.001, duration)) * (CW - PAD * 2)
  const cy = (v: number) => CH - PAD - ((v - min) / span) * (CH - PAD * 2)
  const bezierOf = (k: any) => (k.curve && typeof k.curve === 'object' && k.curve.bezier)
    ? k.curve.bezier : [0.42, 0, 0.58, 1]
  const curvePath = keys.map((k: any, i: number) => {
    if (i === 0) return ''
    const p = keys[i - 1], b = bezierOf(p)
    const x0 = cx(p.time), y0 = cy(scalar(p.value)), x1 = cx(k.time), y1 = cy(scalar(k.value))
    const dx = x1 - x0
    return `M ${x0} ${y0} C ${x0 + dx * b[0]} ${cy(min + span * (1 - b[1]))} ${x0 + dx * b[2]} ${cy(min + span * (1 - b[3]))} ${x1} ${y1}`
  }).filter(Boolean).join(' ')

  const onHandleMove = (e: React.PointerEvent) => {
    const h = dragHandle.current
    if (!h || !keys[h.idx]) return
    const r = (e.currentTarget as SVGElement).getBoundingClientRect()
    const nx = Math.max(0, Math.min(1, (e.clientX - r.left - PAD) / (CW - PAD * 2)))
    const ny = Math.max(0, Math.min(1, 1 - (e.clientY - r.top - PAD) / (CH - PAD * 2)))
    const b = [...bezierOf(keys[h.idx])]
    if (h.handle === 0) { b[0] = nx; b[1] = ny } else { b[2] = nx; b[3] = ny }
    rig.patchKey(keys, h.idx, { curve: { bezier: b } })
  }

  return (
    <div className="rig-timeline">
      <div className="rig-transport">
        <div className="rig-row">
          <button type="button" className={`rig-mini${rig.mode === 'setup' ? ' is-on' : ''}`}
            onClick={() => rig.setMode('setup')}>设定</button>
          <button type="button" className={`rig-mini${rig.mode === 'animate' ? ' is-on' : ''}`}
            onClick={() => rig.setMode('animate')}>动画</button>
          <select value={rig.animation} onChange={(e) => rig.setAnimation(Number(e.target.value))}>
            <option value={-1}>（无动画）</option>
            {d.animations.map((x: any, i: number) => <option key={i} value={i}>{x.name}</option>)}
          </select>
          <input className="rig-name" placeholder="新动画名" value={newName}
            onChange={(e) => setNewName(e.target.value)} />
          <button type="button" className="rig-mini"
            onClick={() => { rig.addAnimation(newName.trim()); setNewName('') }}>＋ 动画</button>
          {rig.animation >= 0 && <button type="button" className="rig-mini danger"
            onClick={() => rig.removeAnimation(rig.animation)}>删动画</button>}
        </div>
        <div className="rig-row">
          <button type="button" className="rig-mini primary"
            onClick={() => rig.setPlaying(!rig.playing)}>{rig.playing ? '⏸ 暂停' : '▶ 播放'}</button>
          <label className="rig-check"><input type="checkbox" checked={rig.loop}
            onChange={(e) => rig.setLoop(e.target.checked)} />循环</label>
          <label className="rig-check"><input type="checkbox" checked={rig.onion}
            onChange={(e) => rig.setOnion(e.target.checked)} />洋葱皮</label>
          <span className="rig-time">{fmtTime(rig.time)} / {fmtTime(duration)}</span>
          <label className="rig-field inline"><span>时长</span>
            <input type="number" min={0.1} step={0.1} value={duration}
              onChange={(e) => rig.patchAnimation(rig.animation, { duration: Number(e.target.value) })} />
          </label>
          <label className="rig-field inline"><span>FPS</span>
            <input type="number" min={1} max={120} value={rig.fps}
              onChange={(e) => rig.setFps(Number(e.target.value) || 30)} />
          </label>
          <button type="button" className="rig-mini primary"
            disabled={rig.mode !== 'animate' || rig.animation < 0}
            onClick={() => rig.keyCurrent()}>◆ 打关键帧</button>
        </div>
        <div className="rig-scrub" ref={barRef}
          onPointerDown={(e) => rig.setTime(timeFromEvent(e))}
          onPointerMove={(e) => { if (e.buttons) rig.setTime(timeFromEvent(e)) }}>
          <i style={{ left: px(rig.time) }} />
        </div>
      </div>

      <div className="rig-tl-body">
        <div className="rig-dopesheet">
          <div className="rig-dope-head">
            <span>轨道</span>
            <div className="rig-ruler">
              {[0, 0.25, 0.5, 0.75, 1].map((f) =>
                <span key={f} style={{ left: `${f * 100}%` }}>{fmtTime(duration * f)}</span>)}
            </div>
          </div>
          {tracks.length === 0 && <p className="rig-hint">还没有关键帧：切到「动画」模式，选中骨骼后在画布里拖动，或点「◆ 打关键帧」。</p>}
          {tracks.map((t: any, ti: number) => (
            <div className={`rig-dope-row${track === ti ? ' is-on' : ''}`} key={`${t.label}-${ti}`}
              onClick={() => { setTrack(ti); setKeyIdx(-1) }}>
              <span title={t.label}>{t.label}</span>
              <div className="rig-dope-lane">
                {t.keys.map((k: any, ki: number) => (
                  <button type="button" key={ki}
                    className={`rig-key${keyIdx === ki && track === ti ? ' is-on' : ''}`}
                    style={{ left: px(k.time) }}
                    title={`${fmtTime(k.time)} · ${JSON.stringify(k.value)}`}
                    onPointerDown={(e) => {
                      e.stopPropagation()
                      setTrack(ti); setKeyIdx(ki); rig.setTime(k.time)
                      dragKey.current = { idx: ki, startX: e.clientX, startTime: k.time }
                    }}
                    onPointerMove={(e) => {
                      const dr = dragKey.current
                      if (!dr || e.buttons === 0) return
                      const lane = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect()
                      const nt = Math.max(0, Math.min(duration,
                        dr.startTime + ((e.clientX - dr.startX) / lane.width) * duration))
                      rig.patchKey(t.keys, dr.idx, { time: nt })
                    }}
                    onPointerUp={() => { dragKey.current = null }}
                    onDoubleClick={(e) => { e.stopPropagation(); rig.removeKey(t.keys, ki) }} />
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="rig-graph">
          <div className="rig-graph-head">
            <b>曲线编辑器</b>
            <span>{selected?.label || '选一条轨道'}</span>
            <select value={curvePreset} onChange={(e) => {
              const i = Number(e.target.value)
              setCurvePreset(i)
              if (keyIdx >= 0 && keys[keyIdx]) rig.patchKey(keys, keyIdx, { curve: CURVES[i].value })
            }}>
              {CURVES.map((c, i) => <option key={i} value={i}>{c.label}</option>)}
            </select>
          </div>
          <svg viewBox={`0 0 ${CW} ${CH}`} className="rig-curve"
            onPointerMove={onHandleMove}
            onPointerUp={() => { dragHandle.current = null }}>
            <line x1={PAD} y1={CH - PAD} x2={CW - PAD} y2={CH - PAD} className="rig-axis" />
            <line x1={PAD} y1={PAD} x2={PAD} y2={CH - PAD} className="rig-axis" />
            <path d={curvePath} className="rig-curve-line" />
            {keys.map((k: any, i: number) => {
              const b = bezierOf(k), x0 = cx(k.time), y0 = cy(scalar(k.value))
              const dx = i + 1 < keys.length ? cx(keys[i + 1].time) - x0 : 0
              return (
                <g key={i}>
                  {i + 1 < keys.length && (
                    <>
                      <line x1={x0} y1={y0} x2={x0 + dx * b[0]} y2={cy(min + span * (1 - b[1]))}
                        className="rig-handle-line" />
                      <circle className="rig-handle" cx={x0 + dx * b[0]}
                        cy={cy(min + span * (1 - b[1]))} r={4}
                        onPointerDown={(e) => { e.stopPropagation(); dragHandle.current = { idx: i, handle: 0 } }} />
                      <circle className="rig-handle" cx={x0 + dx * b[2]}
                        cy={cy(min + span * (1 - b[3]))} r={4}
                        onPointerDown={(e) => { e.stopPropagation(); dragHandle.current = { idx: i, handle: 1 } }} />
                    </>
                  )}
                  <circle className={`rig-keydot${keyIdx === i ? ' is-on' : ''}`}
                    cx={x0} cy={y0} r={4}
                    onPointerDown={(e) => { e.stopPropagation(); setKeyIdx(i) }} />
                </g>
              )
            })}
          </svg>
          {keyIdx >= 0 && keys[keyIdx] && (
            <div className="rig-row rig-keyedit">
              <label className="rig-field inline"><span>时间</span>
                <input type="number" step={0.01} value={keys[keyIdx].time}
                  onChange={(e) => rig.patchKey(keys, keyIdx, { time: Number(e.target.value) })} />
              </label>
              <label className="rig-field inline"><span>值</span>
                <input type="number" step={0.1} value={scalar(keys[keyIdx].value)}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    const old = keys[keyIdx].value
                    rig.patchKey(keys, keyIdx, { value: Array.isArray(old) ? [v, ...old.slice(1)] : v })
                  }} />
              </label>
              <button type="button" className="rig-mini danger"
                onClick={() => { rig.removeKey(keys, keyIdx); setKeyIdx(-1) }}>删关键帧</button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
