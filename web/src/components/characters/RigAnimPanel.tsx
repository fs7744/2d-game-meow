/**
 * 角色中心 · 骨骼动画（与拆部件同级）。
 *
 * 编辑器是**当前项目内嵌的 React 工作台**（components/characters/rig/），
 * 核心算法来自 web/src/rig（rigby2d 运行时的一比一副本）。这里不再打开任何
 * 外部应用 / 新标签页：选 PSD → 进入编辑器 → AutoRig / 改骨骼 / 做动画 →
 * 发布，全部在这一页完成。产物按生命阶段落在 `RigAnim[/s_<阶段>]/`。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  deleteRigAnim, exportRigGodot, fetchParts, fetchRigAnims, uploadRigPsd,
  type PartsSummary, type RigAnimSummary,
} from '../../api'
import { useRig } from './rig/useRig'
import { RigEditor } from './rig/RigEditor'
import { stamp, type RigSource } from './rig/types'

type Props = { character: string; stage: string; stageTitle: string }
type Mode = 'part' | 'upload' | 'blank'

const FORMATS: [string, string][] = [
  ['skeleton-3.8.json', 'JSON 3.8'],
  ['skeleton-4.2.json', 'JSON 4.2'],
  ['skeleton-4.3.json', 'JSON 4.3'],
  ['skeleton.skel', '.skel'],
  ['skeleton.atlas', '图集描述'],
  ['skeleton.png', '图集页'],
  ['project.rigby', '可编辑工程'],
]

export function RigAnimPanel({ character, stage, stageTitle }: Props) {
  const rig = useRig()
  const [editing, setEditing] = useState(false)
  const [items, setItems] = useState<RigAnimSummary[]>([])
  const [parts, setParts] = useState<PartsSummary[]>([])
  const [mode, setMode] = useState<Mode>('part')
  const [selPart, setSelPart] = useState('')
  const [uploaded, setUploaded] = useState<{ id: string; name: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [delId, setDelId] = useState('')
  const [godotUrls, setGodotUrls] = useState<Record<string, string>>({})
  const [godotBusy, setGodotBusy] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const reload = useCallback(async () => {
    try {
      const [r, p] = await Promise.all([
        fetchRigAnims(character, stage),
        fetchParts(character, stage).catch(() => null),
      ])
      setItems(r.items)
      const withPsd = (p?.items || []).filter((x) => !!x.psd)
      setParts(withPsd)
      if (!withPsd.some((x) => x.id === selPart)) setSelPart(withPsd[0]?.id || '')
      setErr('')
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }, [character, stage, selPart])

  useEffect(() => { void reload() }, [reload])

  const begin = async () => {
    setBusy(true); setErr('')
    try {
      let source: RigSource = { kind: 'blank', label: '空白工程', name: '' }
      let psdUrl = '', psdName = ''
      if (mode === 'part') {
        const p = parts.find((x) => x.id === selPart)
        if (!p?.psd) throw new Error('先选一份带 PSD 的拆件产物')
        psdUrl = p.psd; psdName = p.psdName || p.id
        source = { kind: 'part', label: '拆部件 PSD', name: psdName }
      } else if (mode === 'upload') {
        if (!uploaded) throw new Error('先上传一个分层 PSD')
        source = { kind: 'upload', label: '自己导入的 PSD', name: uploaded.name }
      }
      rig.resetDoc(character, source)
      if (psdUrl) {
        const r = await fetch(psdUrl)
        if (!r.ok) throw new Error(`读取 PSD 失败：HTTP ${r.status}`)
        const n = await rig.importPsd(psdName, new Uint8Array(await r.arrayBuffer()))
        rig.say(`已导入 ${n} 层；点工具栏「✨ AutoRig」生成骨骼`)
      }
      setEditing(true)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  const resume = async (it: RigAnimSummary) => {
    setBusy(true); setErr('')
    try {
      const url = it.formats['project.rigby']
      if (!url) throw new Error('这份产物没有可编辑工程')
      const r = await fetch(url)
      if (!r.ok) throw new Error(`读取工程失败：HTTP ${r.status}`)
      const warnings = await rig.loadWorkfile(new Uint8Array(await r.arrayBuffer()), 'project.rigby',
        { kind: 'resume', label: '继续编辑', name: it.id })
      rig.say(`已恢复工程${warnings.length ? `（${warnings.length} 条提示）` : ''}`)
      setEditing(true)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  const onUpload = async (f: File) => {
    setBusy(true); setErr('')
    try {
      const r = await uploadRigPsd(character, f, stage)
      setUploaded({ id: r.id, name: r.name })
      setMode('upload')
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false); if (fileRef.current) fileRef.current.value = '' }
  }

  const doGodot = async (rid: string) => {
    setGodotBusy(rid); setErr('')
    try {
      const r = await exportRigGodot(character, rid, stage)
      setGodotUrls((s) => ({ ...s, [rid]: r.url }))
      await reload()
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setGodotBusy('') }
  }

  const removeOne = async (rid: string) => {
    try {
      await deleteRigAnim(character, rid, stage, true)
      setDelId(''); await reload()
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  if (editing) {
    return (
      <div className="cc-rig">
        <RigEditor rig={rig} character={character} stage={stage}
          stageTitle={stageTitle} source={rig.source}
          onPublished={() => void reload()}
          onExit={() => { setEditing(false); void reload() }} />
      </div>
    )
  }

  const picked = parts.find((x) => x.id === selPart)
  const canBegin = !busy && (mode === 'blank'
    || (mode === 'part' && !!picked?.psd) || (mode === 'upload' && !!uploaded))

  return (
    <div className="cc-rig">
      <div className="cc-rig-head">
        <div>
          <h2>骨骼动画</h2>
          <p>用分层 PSD 一步 AutoRig，在本页的编辑器里改骨骼、权重、IK、动画曲线与摄影表；
            发布后生成 3.8 / 4.2 / 4.3 无损骨骼 JSON、二进制 <code> .skel </code>
            与纹理图集，归到这个生命阶段下。</p>
        </div>
      </div>

      {err && <div className="chars-err">⚠ {err}</div>}

      <section className="cc-rig-sec">
        <h3>1. 选分层 PSD</h3>
        <div className="cc-rig-modes">
          <button type="button" className={`cc-rig-mode${mode === 'part' ? ' is-on' : ''}`}
            onClick={() => setMode('part')}><b>用拆部件的 PSD</b>
            <span>从本阶段已拆过的产物里挑一份标准分层 PSD</span></button>
          <button type="button" className={`cc-rig-mode${mode === 'upload' ? ' is-on' : ''}`}
            onClick={() => setMode('upload')}><b>自己导入 PSD</b>
            <span>本地分层 PSD（Photoshop / Affinity / Krita 导出）</span></button>
          <button type="button" className={`cc-rig-mode${mode === 'blank' ? ' is-on' : ''}`}
            onClick={() => setMode('blank')}><b>空白工程</b>
            <span>进编辑器后再导入分层素材</span></button>
        </div>
        {mode === 'part' && (parts.length === 0 ? (
          <p className="cc-rig-note">这个阶段还没有带 PSD 的拆件产物：先去「拆部件」出一份，或改用「自己导入 PSD」。</p>
        ) : (
          <div className="cc-rig-pick">
            {parts.map((p) => (
              <button type="button" key={p.id}
                className={`cc-rig-pick-item${selPart === p.id ? ' is-on' : ''}`}
                onClick={() => setSelPart(p.id)} title={p.psdName || p.id}>
                {p.preview ? <img src={p.preview} alt="" loading="lazy" />
                  : <span className="cc-rig-ph">PSD</span>}
                <span>{p.methodLabel} · {p.layers} 层</span>
              </button>
            ))}
          </div>
        ))}
        {mode === 'part' && picked && <p className="cc-rig-note">已选：{picked.psdName || picked.id}（{picked.layers} 层）</p>}
        {mode === 'upload' && (
          <div className="cc-rig-upload">
            <input ref={fileRef} type="file" accept=".psd"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void onUpload(f) }} />
            {uploaded && <span className="cc-rig-note">已上传：{uploaded.name}</span>}
          </div>
        )}
      </section>

      <section className="cc-rig-sec">
        <h3>2. 进入编辑器</h3>
        <div className="cc-rig-actions">
          <button type="button" className="cc-btn primary" disabled={!canBegin}
            onClick={() => void begin()}>{busy ? '载入中…' : '打开骨骼动画编辑器 →'}</button>
          <span className="cc-rig-note">编辑器内嵌在本页：画布拖拽改骨骼、结构树、属性面板、
            约束 / 皮肤 / 事件 / 姿势库、摄影表、贝塞尔曲线编辑器、洋葱皮、播放预览都在里面。</span>
        </div>
        <p className="cc-rig-note cc-rig-warn">改完要在编辑器工具栏点「发布到生命阶段」才会落盘；
          直接退出不保存会丢本次改动。</p>
      </section>

      <section className="cc-rig-sec cc-rig-out">
        <h3>这个阶段的骨骼动画
          <span className="cc-rig-meta">{items.length} 份 · 存在{' '}
            <code>RigAnim{stage && stage !== 'default' ? `/s_${stage}` : ''}/</code> 下</span>
        </h3>
        {items.length === 0 ? (
          <p className="cc-rig-note">还没有发布过。选 PSD 进编辑器，AutoRig 并发布后，这里会出现记录。</p>
        ) : (
          <ul className="cc-rig-list">
            {items.map((it) => {
              const zip = godotUrls[it.id] || it.godot
              return (
                <li key={it.id}>
                  <div className="cc-rig-item">
                    {it.formats['skeleton.png']
                      ? <img src={it.formats['skeleton.png']} alt="" loading="lazy" />
                      : <span className="cc-rig-ph">SKEL</span>}
                    <div className="cc-rig-item-txt">
                      <b>{it.source?.label || '骨骼动画'} · {stamp(it.created)}</b>
                      <span>{it.bones} 骨 · {it.slots} 槽 · {it.animations} 动画 · {it.constraints} 约束
                        {it.animationNames?.length ? ` · ${it.animationNames.slice(0, 3).join(' / ')}` : ''}</span>
                      <span className="cc-rig-src">{it.source?.name || ''}</span>
                    </div>
                  </div>
                  <div className="cc-rig-formats">
                    {FORMATS.filter(([k]) => !!it.formats[k]).map(([k, label]) => (
                      <a key={k} className="cc-btn ghost" href={it.formats[k]} download={k}>{label}</a>
                    ))}
                  </div>
                  <div className="cc-rig-ops">
                    <button type="button" className="cc-btn ghost"
                      disabled={!it.formats['project.rigby']}
                      onClick={() => void resume(it)}>继续编辑 →</button>
                    <button type="button" className="cc-btn" disabled={godotBusy === it.id}
                      onClick={() => void doGodot(it.id)}>
                      {godotBusy === it.id ? '打包中…' : '导出 Godot'}</button>
                    {zip && <a className="cc-btn primary" href={zip}
                      download={`${character}_godot.zip`}>下载 Godot 工程</a>}
                    {delId === it.id ? (
                      <>
                        <button type="button" className="cc-btn danger"
                          onClick={() => void removeOne(it.id)}>确认删除</button>
                        <button type="button" className="cc-btn ghost" onClick={() => setDelId('')}>取消</button>
                      </>
                    ) : (
                      <button type="button" className="cc-btn ghost"
                        onClick={() => setDelId(it.id)}>删除</button>
                    )}
                  </div>
                  {!!it.warnings?.length && <p className="cc-rig-note">导出提示：{it.warnings.slice(0, 3).join('；')}</p>}
                </li>
              )
            })}
          </ul>
        )}
        <p className="cc-rig-note cc-rig-warn">Godot 工程包含 addon 配置与示例场景，但运行时是 Rust
          GDExtension，需按包内 README-GODOT.md 放入对应平台二进制（Godot 4.3+）。</p>
      </section>
    </div>
  )
}
