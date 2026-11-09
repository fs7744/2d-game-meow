/** 编辑器工作台：把画布 / 结构树 / 属性 / 时间轴组装成一个页面，并负责导入与发布。 */
import { useRef, useState } from 'react'
import { publishRigAnim } from '../../../api'
import type { RigApi } from './useRig'
import type { RigSource, RigTool } from './types'
import { RigCanvas } from './RigCanvas'
import { RigTree, RigInspector, AutoRigDialog, type TreeTab } from './RigPanels'
import { RigTimeline } from './RigTimeline'

const TOOLS: [RigTool, string, string][] = [
  ['select', '✋', '选择 / 旋转'], ['rotate', '⟳', '旋转'], ['translate', '✥', '位移'],
  ['scale', '⤢', '缩放'], ['create', '🦴', '点空白处加子骨'],
]

function download(name: string, bytes: Uint8Array | Blob) {
  const blob = bytes instanceof Blob ? bytes : new Blob([bytes as BlobPart])
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = name; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 4000)
}

export function RigEditor({ rig, character, stage, stageTitle, source, onPublished, onExit, exitLabel = '← 返回列表' }: {
  rig: RigApi
  character: string
  stage: string
  stageTitle: string
  source: RigSource
  onPublished: () => void
  onExit: () => void
  exitLabel?: string
}) {
  const [tab, setTab] = useState<TreeTab>('bones')
  const [autorig, setAutorig] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [frames, setFrames] = useState(24)
  const fileRef = useRef<HTMLInputElement>(null)

  const importFiles = async (files: FileList | null) => {
    if (!files?.length) return
    setBusy(true); setErr('')
    try {
      const list = [...files]
      const psds = list.filter((f) => /\.psd$/i.test(f.name))
      const svgs = list.filter((f) => /\.svg$/i.test(f.name))
      const pngs = list.filter((f) => /\.png$/i.test(f.name))
      for (const f of psds) await rig.importPsd(f.name, new Uint8Array(await f.arrayBuffer()))
      if (svgs.length) await rig.importSvg(svgs[0].name, new Uint8Array(await svgs[0].arrayBuffer()))
      if (pngs.length) await rig.importPngs(await Promise.all(pngs.map(async (f) => ({
        name: f.name, bytes: new Uint8Array(await f.arrayBuffer()),
      }))))
      rig.say(`导入完成：${psds.length} PSD / ${svgs.length} SVG / ${pngs.length} PNG`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false); if (fileRef.current) fileRef.current.value = '' }
  }

  const doPublish = async () => {
    if (!character) { setErr('先在顶部选择一个角色，才能发布到它的生命阶段'); return }
    setBusy(true); setErr('')
    try {
      const r = await rig.buildExports()
      await publishRigAnim(character, stage, source,
        { ...r.summary, warnings: r.warnings }, r.files)
      rig.say(`已发布：${Object.keys(r.files).length} 个文件归到当前生命阶段`)
      onPublished()
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  const doFrames = async () => {
    setBusy(true); setErr('')
    try {
      const bytes = await rig.renderFrames(frames)
      download(`${rig.docRef.current.name || 'anim'}_frames.zip`, bytes)
      rig.say(`已渲染 ${frames} 帧 PNG 序列`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  return (
    <div className="rig-editor">
      <div className="rig-toolbar">
        <button type="button" className="rig-btn ghost" onClick={onExit}>{exitLabel}</button>
        <span className="rig-sep" />
        <button type="button" className="rig-btn" disabled={busy}
          onClick={() => fileRef.current?.click()}>导入 PSD / PNG / SVG</button>
        <input ref={fileRef} type="file" multiple accept=".psd,.png,.svg" hidden
          onChange={(e) => void importFiles(e.target.files)} />
        <button type="button" className="rig-btn" onClick={() => setAutorig(true)}>✨ AutoRig</button>
        <span className="rig-sep" />
        {TOOLS.map(([t, icon, title]) => (
          <button type="button" key={t} title={title}
            className={`rig-tool${rig.tool === t ? ' is-on' : ''}`}
            onClick={() => rig.setTool(t)}>{icon}</button>
        ))}
        <span className="rig-sep" />
        <label className="rig-check"><input type="checkbox" checked={rig.showBones}
          onChange={(e) => rig.setShowBones(e.target.checked)} />骨骼</label>
        <label className="rig-check"><input type="checkbox" checked={rig.showMesh}
          onChange={(e) => rig.setShowMesh(e.target.checked)} />权重网格</label>
        <label className="rig-check"><input type="checkbox" checked={rig.showImages}
          onChange={(e) => rig.setShowImages(e.target.checked)} />贴图</label>
        <label className="rig-check"><input type="checkbox" checked={rig.grid}
          onChange={(e) => rig.setGrid(e.target.checked)} />背景网格</label>
        <span className="spacer" />
        <label className="rig-field inline"><span>帧数</span>
          <input type="number" min={2} max={240} value={frames}
            onChange={(e) => setFrames(Number(e.target.value) || 24)} /></label>
        <button type="button" className="rig-btn" disabled={busy} onClick={() => void doFrames()}>PNG 序列</button>
        <button type="button" className="rig-btn primary" disabled={busy || !character}
          title={character ? '把全部无损格式归档到当前生命阶段' : '先在顶部选一个角色'}
          onClick={() => void doPublish()}>{busy ? '处理中…' : '发布到生命阶段'}</button>
      </div>

      {err && <div className="rig-err">⚠ {err}</div>}
      <div className="rig-status">
        <span>{rig.status || '拖动画布里的骨骼编辑；Shift/Alt + 拖动平移，滚轮缩放。'}</span>
        <span>{character} · {stageTitle || stage} · {rig.docRef.current.name} · {source.label}</span>
      </div>

      <div className="rig-work">
        <RigTree rig={rig} tab={tab} setTab={setTab} />
        <RigCanvas rig={rig} />
        <RigInspector rig={rig} tab={tab} />
      </div>
      <RigTimeline rig={rig} />
      {autorig && <AutoRigDialog rig={rig} onClose={() => setAutorig(false)} />}
    </div>
  )
}
