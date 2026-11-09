/** 左侧结构树、右侧属性面板，以及原生 AutoRig 的部位/左右修正表格。 */
import { useMemo, useState } from 'react'
import { autorig } from '../../../rig/facade.js'
import type { RigApi } from './useRig'
import { ROLE_LABELS, SIDE_LABELS, CONSTRAINT_TYPES } from './types'

const ROLE_LIST: string[] = autorig.PART_ROLES

function Num({ label, value, onChange, step = 1 }: {
  label: string; value: number; onChange: (v: number) => void; step?: number
}) {
  return (
    <label className="rig-field">
      <span>{label}</span>
      <input type="number" step={step} value={Number.isFinite(value) ? Number(value.toFixed(4)) : 0}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

function Txt({ label, value, onChange }: {
  label: string; value: string; onChange: (v: string) => void
}) {
  return (
    <label className="rig-field">
      <span>{label}</span>
      <input value={value ?? ''} onChange={(e) => onChange(e.target.value)} />
    </label>
  )
}

export type TreeTab = 'bones' | 'slots' | 'constraints' | 'events' | 'skins' | 'poses'
const TABS: [TreeTab, string][] = [
  ['bones', '骨骼'], ['slots', '槽位'], ['constraints', '约束'],
  ['events', '事件'], ['skins', '皮肤'], ['poses', '姿势库'],
]

export function RigTree({ rig, tab, setTab }: {
  rig: RigApi; tab: TreeTab; setTab: (t: TreeTab) => void
}) {
  const d = rig.docRef.current
  const cons = rig.constraints()
  const depth = useMemo(() => {
    const out: number[] = []
    d.bones.forEach((b: any, i: number) => {
      out[i] = b.parent == null ? 0 : (out[b.parent] ?? 0) + 1
    })
    return out
  }, [d, rig.rev])
  const row = (kind: string, index: number, label: string, sub = '') => (
    <button type="button" key={`${kind}-${index}`}
      className={`rig-tree-row${rig.sel?.kind === kind && rig.sel.index === index ? ' is-on' : ''}`}
      onClick={() => rig.setSel({ kind, index } as any)}>
      <span>{label}</span><em>{sub}</em>
    </button>
  )

  return (
    <div className="rig-tree">
      <div className="rig-tabs">
        {TABS.map(([k, label]) => (
          <button type="button" key={k} className={tab === k ? 'is-on' : ''}
            onClick={() => setTab(k)}>{label}</button>
        ))}
      </div>
      <div className="rig-tree-body">
        {tab === 'bones' && (
          <>
            {d.bones.map((b: any, i: number) => (
              <div key={i} style={{ paddingLeft: `${Math.min(8, depth[i] || 0) * 10}px` }}>
                {row('bone', i, b.name, `${Math.round(b.length || 0)}px`)}
              </div>
            ))}
            <button type="button" className="rig-add"
              onClick={() => rig.addBone(rig.sel?.kind === 'bone' ? rig.sel.index : 0)}>＋ 加子骨</button>
          </>
        )}
        {tab === 'slots' && (
          <>
            {d.slots.map((s: any, i: number) => (
              <div key={i} className="rig-slot-row">
                {row('slot', i, s.name, d.bones[s.bone]?.name || '')}
                <span className="rig-updown">
                  <button type="button" onClick={() => rig.moveSlot(i, Math.max(0, i - 1))}>↑</button>
                  <button type="button" onClick={() => rig.moveSlot(i, Math.min(d.slots.length - 1, i + 1))}>↓</button>
                </span>
              </div>
            ))}
            <button type="button" className="rig-add" onClick={() => rig.addSlot(0)}>＋ 加槽位</button>
          </>
        )}
        {tab === 'constraints' && (
          <>
            {cons.map((c: any, i: number) => row('constraint', i, c.name, `${c.type} · #${c.order}`))}
            <ConstraintAdder rig={rig} />
          </>
        )}
        {tab === 'events' && (
          <>
            {d.events.map((e: any, i: number) => row('event', i, e.name, `int ${e.int ?? 0}`))}
            <button type="button" className="rig-add" onClick={() => rig.addEvent()}>＋ 加事件</button>
          </>
        )}
        {tab === 'skins' && (
          <>
            {d.skins.map((s: any, i: number) => row('skin', i, s.name, `${s.slots?.length || 0} 槽`))}
            <button type="button" className="rig-add" onClick={() => rig.addSkin()}>＋ 加皮肤</button>
          </>
        )}
        {tab === 'poses' && (
          <>
            {d.poses.map((p: any, i: number) => (
              <div key={i} className="rig-slot-row">
                {row('pose', i, p.name, `${Object.keys(p.bones || {}).length} 骨`)}
                <button type="button" className="rig-mini" onClick={() => rig.applyPose(i)}>应用</button>
              </div>
            ))}
            <button type="button" className="rig-add" onClick={() => rig.addPose()}>＋ 存当前姿势</button>
          </>
        )}
      </div>
    </div>
  )
}

/** transform/path/physics/slider 约束用 4.3 JSON 布局新增（真实 core API：CD.addConstraint）。 */
function ConstraintAdder({ rig }: { rig: RigApi }) {
  const [open, setOpen] = useState(false)
  const [type, setType] = useState<string>('transform')
  const [raw, setRaw] = useState('{}')
  if (!open) return <button type="button" className="rig-add" onClick={() => setOpen(true)}>＋ 加约束</button>
  return (
    <div className="rig-json-add">
      <select value={type} onChange={(e) => {
        setType(e.target.value)
        setRaw(JSON.stringify({ type: e.target.value, name: `${e.target.value}_${Date.now() % 1000}` }, null, 2))
      }}>
        {CONSTRAINT_TYPES.filter((t) => t !== 'ik').map((t) => <option key={t} value={t}>{t}</option>)}
      </select>
      <textarea rows={5} value={raw} onChange={(e) => setRaw(e.target.value)} />
      <div className="rig-row">
        <button type="button" className="rig-mini" onClick={() => {
          try {
            rig.addConstraint(JSON.parse(raw))
            setOpen(false)
          } catch (err) { rig.say(`约束 JSON 不合法：${err instanceof Error ? err.message : err}`) }
        }}>添加</button>
        <button type="button" className="rig-mini" onClick={() => setOpen(false)}>取消</button>
      </div>
    </div>
  )
}

export function RigInspector({ rig, tab }: { rig: RigApi; tab: TreeTab }) {
  const d = rig.docRef.current
  const sel = rig.sel
  if (!sel) return <div className="rig-inspector rig-empty">在左侧选一项，或点画布里的骨骼。</div>
  const del = (fn: () => void) => <button type="button" className="rig-mini danger" onClick={fn}>删除</button>

  if (sel.kind === 'bone') {
    const b = d.bones[sel.index]
    if (!b) return null
    return (
      <div className="rig-inspector">
        <h4>骨骼 · {b.name} {sel.index > 0 && del(() => rig.removeBone(sel.index))}</h4>
        <Txt label="名称" value={b.name} onChange={(v) => rig.patchBone(sel.index, { name: v })} />
        <label className="rig-field"><span>父骨</span>
          <select value={b.parent ?? ''} onChange={(e) => rig.reparentBone(sel.index, Number(e.target.value))}>
            <option value="">（根）</option>
            {d.bones.map((x: any, i: number) => i !== sel.index && <option key={i} value={i}>{x.name}</option>)}
          </select>
        </label>
        <div className="rig-grid">
          <Num label="X" value={b.setup.x} onChange={(v) => rig.patchBone(sel.index, { x: v })} />
          <Num label="Y" value={b.setup.y} onChange={(v) => rig.patchBone(sel.index, { y: v })} />
          <Num label="旋转°" value={b.setup.rotation} onChange={(v) => rig.patchBone(sel.index, { rotation: v })} />
          <Num label="长度" value={b.length || 0} onChange={(v) => rig.patchBone(sel.index, { length: v })} />
          <Num label="缩放 X" value={b.setup.scale_x} step={0.05} onChange={(v) => rig.patchBone(sel.index, { scale_x: v })} />
          <Num label="缩放 Y" value={b.setup.scale_y} step={0.05} onChange={(v) => rig.patchBone(sel.index, { scale_y: v })} />
          <Num label="斜切 X" value={b.setup.shear_x || 0} onChange={(v) => rig.patchBone(sel.index, { shear_x: v })} />
          <Num label="斜切 Y" value={b.setup.shear_y || 0} onChange={(v) => rig.patchBone(sel.index, { shear_y: v })} />
        </div>
        <label className="rig-field"><span>继承</span>
          <select value={b.inherit || 'normal'} onChange={(e) => rig.patchBone(sel.index, { inherit: e.target.value })}>
            {['normal', 'noRotationOrReflection', 'noScale', 'noScaleOrReflection', 'onlyTranslation'].map((x) =>
              <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
      </div>
    )
  }

  if (sel.kind === 'slot') {
    const s = d.slots[sel.index]
    if (!s) return null
    const a = s.attachments?.find((x: any) => x.name === s.attachment) || s.attachments?.[0]
    return (
      <div className="rig-inspector">
        <h4>槽位 · {s.name} {del(() => rig.removeSlot(sel.index))}</h4>
        <Txt label="名称" value={s.name} onChange={(v) => rig.patchSlot(sel.index, { name: v })} />
        <label className="rig-field"><span>绑定骨</span>
          <select value={s.bone} onChange={(e) => rig.patchSlot(sel.index, { bone: Number(e.target.value) })}>
            {d.bones.map((b: any, i: number) => <option key={i} value={i}>{b.name}</option>)}
          </select>
        </label>
        <label className="rig-field"><span>当前附件</span>
          <select value={s.attachment ?? ''} onChange={(e) => rig.patchSlot(sel.index, { attachment: e.target.value || null })}>
            <option value="">（隐藏）</option>
            {(s.attachments || []).map((x: any) => <option key={x.name} value={x.name}>{x.name}</option>)}
          </select>
        </label>
        <label className="rig-field"><span>混合</span>
          <select value={s.blend || 'normal'} onChange={(e) => rig.patchSlot(sel.index, { blend: e.target.value })}>
            {['normal', 'additive', 'multiply', 'screen'].map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        {a && (
          <>
            <h4>附件 · {a.name}</h4>
            <div className="rig-grid">
              <Num label="宽" value={a.width} onChange={(v) => rig.patchAttachment(sel.index, { width: v })} />
              <Num label="高" value={a.height} onChange={(v) => rig.patchAttachment(sel.index, { height: v })} />
              <Num label="偏移 X" value={a.offset?.x || 0} onChange={(v) => rig.patchAttachment(sel.index, { offset: { ...a.offset, x: v } })} />
              <Num label="偏移 Y" value={a.offset?.y || 0} onChange={(v) => rig.patchAttachment(sel.index, { offset: { ...a.offset, y: v } })} />
              <Num label="偏移旋转" value={a.offset?.rotation || 0} onChange={(v) => rig.patchAttachment(sel.index, { offset: { ...a.offset, rotation: v } })} />
            </div>
          </>
        )}
      </div>
    )
  }

  if (sel.kind === 'constraint') {
    const c = rig.constraints()[sel.index]
    if (!c) return null
    return (
      <div className="rig-inspector">
        <h4>约束 · {c.name} {del(() => rig.removeConstraint(c.name))}</h4>
        <p className="rig-hint">类型 {c.type} · 应用顺序 {c.order}</p>
        <pre className="rig-json">{JSON.stringify(c.data || { name: c.name, type: 'ik', ik: c.ik }, null, 2)}</pre>
        <p className="rig-hint">IK 约束由 AutoRig 生成；transform / path / physics / slider 用左侧 JSON 新增。</p>
      </div>
    )
  }

  if (sel.kind === 'event') {
    const e = d.events[sel.index]
    if (!e) return null
    return (
      <div className="rig-inspector">
        <h4>事件 · {e.name} {del(() => rig.removeEvent(sel.index))}</h4>
        <Txt label="名称" value={e.name} onChange={(v) => rig.patchEvent(sel.index, { name: v })} />
        <div className="rig-grid">
          <Num label="int" value={e.int || 0} onChange={(v) => rig.patchEvent(sel.index, { int: v })} />
          <Num label="float" value={e.float || 0} step={0.1} onChange={(v) => rig.patchEvent(sel.index, { float: v })} />
        </div>
        <Txt label="string" value={e.string || ''} onChange={(v) => rig.patchEvent(sel.index, { string: v })} />
        <Txt label="audio" value={e.audio || ''} onChange={(v) => rig.patchEvent(sel.index, { audio: v })} />
      </div>
    )
  }

  if (sel.kind === 'skin') {
    const s = d.skins[sel.index]
    if (!s) return null
    return (
      <div className="rig-inspector">
        <h4>皮肤 · {s.name} {del(() => rig.removeSkin(sel.index))}</h4>
        <Txt label="名称" value={s.name} onChange={(v) => { s.name = v; rig.commit() }} />
        <p className="rig-hint">{s.slots?.length || 0} 个槽位覆盖 · 双击画布里的部件可换附件。</p>
      </div>
    )
  }

  if (sel.kind === 'pose') {
    const p = d.poses[sel.index]
    if (!p) return null
    return (
      <div className="rig-inspector">
        <h4>姿势 · {p.name}</h4>
        <Txt label="名称" value={p.name} onChange={(v) => { p.name = v; rig.commit() }} />
        <button type="button" className="rig-mini" onClick={() => rig.applyPose(sel.index)}>应用到设定姿势</button>
      </div>
    )
  }

  return <div className="rig-inspector rig-empty">当前分类：{tab}</div>
}

/** AutoRig：直接用 core 的 findParts 列出每一层，允许逐行改部位与左右，再调 autoRig。 */
export function AutoRigDialog({ rig, onClose }: { rig: RigApi; onClose: () => void }) {
  const [roles, setRoles] = useState<Record<string, string>>({})
  const [sides, setSides] = useState<Record<string, string>>({})
  const [meshes, setMeshes] = useState('deforming')
  const [ik, setIk] = useState(true)
  const [err, setErr] = useState('')
  const parts = useMemo(() => {
    try { return rig.previewParts(roles, sides) as any[] }
    catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setErr(msg.includes('already has bones')
        ? '当前工程已有骨骼。AutoRig 只用于未绑定的分层图；要重来请先返回列表、新建空白工程并重新导入 PSD。'
        : msg)
      return []
    }
    // previewParts 是稳定回调：不要在生成后（rig 对象重建）再跑一次 findParts。
  }, [rig.previewParts, roles, sides])
  const run = () => {
    try {
      const r = rig.runAutoRig({
        meshes, ik, roles: Object.fromEntries(Object.entries(roles).filter(([, v]) => v)),
        sides: Object.fromEntries(Object.entries(sides).filter(([, v]) => v)),
      })
      rig.say(`AutoRig：生成 ${r.bones.length} 根骨、${r.parts.filter((p: any) => p.mesh).length} 个可弯曲网格、${r.ik.length} 个 IK 约束。${(r.warnings || []).join('；')}`)
      onClose()
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  return (
    <div className="rig-modal-backdrop" onClick={onClose}>
      <div className="rig-modal" onClick={(e) => e.stopPropagation()}>
        <header><b>AutoRig · 一步把分层图变成绑定</b>
          <button type="button" onClick={onClose}>✕</button></header>
        <p className="rig-hint">按层名或位置识别部位；这里可以逐行修正。层名写 head / torso / arm_l / upperarm_r 时会自动认出。</p>
        {err && <div className="rig-err">{err}</div>}
        <div className="rig-autorig-table">
          <div className="rig-autorig-head"><span>图层</span><span>部位</span><span>左右</span><span>识别</span></div>
          {parts.map((p: any) => (
            <div className="rig-autorig-row" key={p.slot}>
              <span title={p.name}>{p.name}</span>
              <select value={roles[p.name] || p.role || 'none'}
                onChange={(e) => setRoles((s) => ({ ...s, [p.name]: e.target.value }))}>
                <option value="none">不绑定</option>
                {ROLE_LIST.map((r) => <option key={r} value={r}>{ROLE_LABELS[r] || r}</option>)}
              </select>
              <select value={sides[p.name] || p.side || ''}
                onChange={(e) => setSides((s) => ({ ...s, [p.name]: e.target.value }))}>
                {Object.entries(SIDE_LABELS).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
              <em>{p.from === 'name' ? '层名' : p.from === 'manual' ? '手改' : '位置'}</em>
            </div>
          ))}
          {!parts.length && !err && <p className="rig-hint">还没有可识别的图层：先导入分层 PSD / PNG / SVG。</p>}
        </div>
        <div className="rig-row">
          <label className="rig-field"><span>可弯曲网格</span>
            <select value={meshes} onChange={(e) => setMeshes(e.target.value)}>
              <option value="deforming">需要弯曲的部件（推荐）</option>
              <option value="all">所有部件</option>
              <option value="none">不生成网格</option>
            </select>
          </label>
          <label className="rig-check"><input type="checkbox" checked={ik}
            onChange={(e) => setIk(e.target.checked)} /> 手臂与腿加 IK</label>
        </div>
        <div className="rig-row rig-modal-actions">
          <button type="button" className="rig-btn primary" disabled={!parts.length} onClick={run}>生成绑定</button>
          <button type="button" className="rig-btn" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  )
}
