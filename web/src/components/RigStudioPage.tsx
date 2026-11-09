/**
 * 骨骼动画工作台（顶栏一级视图，与「画布」「角色中心」平级）。
 *
 * 这里不是外部编辑器：整个工作台长在当前项目的 React 里，核心算法来自
 * web/src/rig（rigby2d 运行时一比一副本）。产物仍按「角色 / 生命阶段」归档，
 * 与角色中心「骨骼动画」页共用同一份后端数据。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchCharacters, fetchStages, fetchParts, fetchRigAnims, uploadRigPsd,
  deleteRigAnim, exportRigGodot,
  type CharacterSummary, type StageItem, type PartsSummary, type RigAnimSummary,
} from '../api'
import { useRig } from './characters/rig/useRig'
import { RigEditor } from './characters/rig/RigEditor'
import { RigWorkspace } from './characters/rig/RigWorkspace'
import { stamp, type RigSource } from './characters/rig/types'

const FORMATS: [string, string][] = [
  ['skeleton-3.8.json', '3.8'], ['skeleton-4.2.json', '4.2'],
  ['skeleton-4.3.json', '4.3'], ['skeleton.skel', '.skel'],
  ['skeleton.atlas', 'atlas'], ['skeleton.png', '图集'],
  ['project.rigby', '工程'],
]

export default function RigStudioPage({ onNotice }: { onNotice: (s: string) => void }) {
  const rig = useRig()
  const [characters, setCharacters] = useState<CharacterSummary[]>([])
  const [character, setCharacter] = useState('')
  const [stages, setStages] = useState<StageItem[]>([])
  const [stage, setStage] = useState('default')
  const [parts, setParts] = useState<PartsSummary[]>([])
  const [items, setItems] = useState<RigAnimSummary[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [delId, setDelId] = useState('')
  const [godotUrls, setGodotUrls] = useState<Record<string, string>>({})
  const psdRef = useRef<HTMLInputElement>(null)
  const workRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    void fetchCharacters().then((r) => {
      setCharacters(r.items || [])
      if (!character && r.items?.[0]) setCharacter(r.items[0].name)
    }).catch((e) => setErr(String(e)))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!character) return
    void fetchStages(character).then((d) => {
      setStages(d.stages || [])
      setStage(d.active_stage || 'default')
    }).catch((e) => setErr(String(e)))
  }, [character])

  const reload = useCallback(async () => {
    if (!character) return
    try {
      const [r, p] = await Promise.all([
        fetchRigAnims(character, stage),
        fetchParts(character, stage).catch(() => null),
      ])
      setItems(r.items || [])
      setParts(((p?.items || []) as PartsSummary[]).filter((x) => !!x.psd))
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }, [character, stage])
  useEffect(() => { void reload() }, [reload])

  const stageTitle = stages.find((s) => s.id === stage)?.title || ''

  // 自动化核对的探针：只有 `?rigdebug=1` 才挂到 window，页面自身从不读它。
  // `tools/rig_drag.mjs` 用真鼠标事件拖演员，靠它读锚骨世界坐标与选中态 ——
  // 比在截图里猜坐标可靠（也不必给产品加任何测试专用 API）。
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has('rigdebug')) return
    const w = window as unknown as { __rig?: unknown }
    w.__rig = rig
    return () => { delete w.__rig }
  })

  const newBlank = () => {
    rig.resetDoc(character || 'untitled', { kind: 'blank', label: '空白工程', name: '' })
    rig.say('新建空白工程：导入分层素材后点 AutoRig')
  }

  const importPsd = async (f: File) => {
    setBusy(true); setErr('')
    try {
      const up = await uploadRigPsd(character, f, stage)
      const r = await fetch(up.url)
      const src: RigSource = { kind: 'upload', label: '自己导入的 PSD', name: f.name }
      rig.resetDoc(character || f.name, src)
      const n = await rig.importPsd(f.name, new Uint8Array(await r.arrayBuffer()))
      rig.say(`已导入 ${n} 层；点「✨ AutoRig」生成骨骼`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false); if (psdRef.current) psdRef.current.value = '' }
  }

  const fromPart = async (p: PartsSummary) => {
    if (!p.psd) return
    setBusy(true); setErr('')
    try {
      const r = await fetch(p.psd)
      const src: RigSource = { kind: 'part', label: '拆部件 PSD', name: p.psdName || p.id }
      rig.resetDoc(character || 'untitled', src)
      const n = await rig.importPsd(p.psdName || 'layered.psd', new Uint8Array(await r.arrayBuffer()))
      rig.say(`已导入 ${n} 层；点「✨ AutoRig」生成骨骼`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  const openWork = async (f: File) => {
    setBusy(true); setErr('')
    try {
      const w = await rig.loadWorkfile(new Uint8Array(await f.arrayBuffer()), f.name,
        { kind: 'import', label: '本机工程', name: f.name })
      rig.say(`已打开工程${w.length ? `（${w.length} 条提示）` : ''}`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false); if (workRef.current) workRef.current.value = '' }
  }

  const resume = async (it: RigAnimSummary) => {
    const url = it.formats['project.rigby']
    if (!url) { setErr('这份产物没有可编辑工程'); return }
    setBusy(true); setErr('')
    try {
      const r = await fetch(url)
      const w = await rig.loadWorkfile(new Uint8Array(await r.arrayBuffer()), 'project.rigby',
        { kind: 'resume', label: '继续编辑', name: it.id })
      rig.say(`已恢复「${it.source?.label || it.id}」${w.length ? `（${w.length} 条提示）` : ''}`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  const doGodot = async (rid: string) => {
    try {
      const r = await exportRigGodot(character, rid, stage)
      setGodotUrls((s) => ({ ...s, [rid]: r.url }))
      await reload()
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  const removeOne = async (rid: string) => {
    try {
      await deleteRigAnim(character, rid, stage, true)
      setDelId(''); await reload()
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  return (
    <div className="rig-studio">
      <header className="rig-studio-head">
        <div className="rig-studio-title"><b>骨骼动画工作台</b>
          <span>绑定 · 动画 · 影片 · 口型 · 场景 · 重定向 · 外部工程</span></div>
        <label className="rig-field"><span>角色</span>
          <select value={character} onChange={(e) => setCharacter(e.target.value)}>
            <option value="">（未选）</option>
            {characters.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select></label>
        <label className="rig-field"><span>生命阶段</span>
          <select value={stage} onChange={(e) => setStage(e.target.value)}>
            {stages.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
          </select></label>
        <button type="button" className="rig-btn" onClick={newBlank}>＋ 空白工程</button>
        <button type="button" className="rig-btn" disabled={busy || !character}
          onClick={() => psdRef.current?.click()}>导入分层 PSD</button>
        <input ref={psdRef} type="file" accept=".psd" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) void importPsd(f) }} />
        <button type="button" className="rig-btn" onClick={() => workRef.current?.click()}>打开 .rigby</button>
        <input ref={workRef} type="file" accept=".rigby,.json,.skel" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) void openWork(f) }} />
        {!!parts.length && <label className="rig-field"><span>拆件 PSD</span>
          <select value="" onChange={(e) => {
            const p = parts.find((x) => x.id === e.target.value); if (p) void fromPart(p)
          }}>
            <option value="">选一份…</option>
            {parts.map((p) => <option key={p.id} value={p.id}>{p.psdName || p.id} · {p.layers} 层</option>)}
          </select></label>}
        <span className="spacer" />
        <span className="rig-hint">产物归档到 {character || '—'} / {stageTitle || stage}</span>
      </header>

      {err && <div className="rig-err">⚠ {err}</div>}

      <RigEditor rig={rig} character={character} stage={stage} stageTitle={stageTitle}
        source={rig.source} exitLabel="＋ 新建空白工程" onExit={newBlank}
        onPublished={() => { void reload(); onNotice('已发布到当前生命阶段') }} />

      <RigWorkspace rig={rig} items={items} character={character} stage={stage}
        onNotice={onNotice} />

      <section className="rig-studio-history">
        <h3>这个生命阶段的产物
          <span className="rig-hint">{items.length} 份 · RigAnim{stage && stage !== 'default' ? `/s_${stage}` : ''}/</span>
        </h3>
        {!items.length && <p className="rig-hint">还没有发布过。在编辑器里点「发布到生命阶段」后，这里会出现记录。</p>}
        {!!items.length && <div className="rig-history-list">
          {items.map((it) => {
            const zip = godotUrls[it.id] || it.godot
            return (
              <div className="rig-history-row" key={it.id}>
                <div className="rig-history-main">
                  <b>{it.source?.label || '骨骼动画'} · {stamp(it.created)}</b>
                  <span>{it.bones} 骨 / {it.slots} 槽 / {it.animations} 动画 / {it.constraints} 约束</span>
                </div>
                <div className="rig-row">
                  {FORMATS.filter(([k]) => !!it.formats[k]).map(([k, label]) => (
                    <a key={k} className="rig-mini" href={it.formats[k]} download={k}>{label}</a>
                  ))}
                </div>
                <div className="rig-row">
                  <button type="button" className="rig-mini" onClick={() => void resume(it)}>继续编辑</button>
                  <button type="button" className="rig-mini" onClick={() => void doGodot(it.id)}>导出 Godot</button>
                  {zip && <a className="rig-mini primary" href={zip}
                    download={`${character}_godot.zip`}>下载工程包</a>}
                  {delId === it.id ? (
                    <>
                      <button type="button" className="rig-mini danger"
                        onClick={() => void removeOne(it.id)}>确认删除</button>
                      <button type="button" className="rig-mini" onClick={() => setDelId('')}>取消</button>
                    </>
                  ) : <button type="button" className="rig-mini danger"
                    onClick={() => setDelId(it.id)}>删除</button>}
                </div>
              </div>
            )
          })}
        </div>}
      </section>
    </div>
  )
}
