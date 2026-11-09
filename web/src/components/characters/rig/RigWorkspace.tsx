/**
 * 高级工作区：影片、口型、场景、重定向、外部工程导入。
 * 全部调用已一比一搬入的 core / io 真实 API，不在这里重写骨骼动画算法。
 */
import { useState } from 'react'
import {
  output, lipsync, scene as sceneNs, retarget, io, zip as zipNs,
  images as imagesNs, interop,
} from '../../../rig/facade.js'
import type { RigApi } from './useRig'
import type { RigAnimSummary } from '../../../api'
import { renderVideo, renderPngSequence, canEncodeVideo, timecode, type FilmOptions } from './videoRender'

type Tab = 'film' | 'lipsync' | 'scene' | 'retarget' | 'import'
const TABS: [Tab, string][] = [
  ['film', '影片工作区'], ['lipsync', '视位口型'], ['scene', '场景 / 摄影机'],
  ['retarget', '动画重定向'], ['import', 'Spine / DragonBones 导入'],
]

function download(name: string, data: Uint8Array | Blob | string, mime = 'application/octet-stream') {
  const blob = typeof data === 'string' ? new Blob([data], { type: mime })
    : data instanceof Blob ? data : new Blob([data as BlobPart], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = name; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 4000)
}

function basename(p: string): string {
  return String(p).replace(/\\/g, '/').split('/').pop() || p
}

type VisemeCue = { time: number; viseme: string }

/**
 * core 的 readTiming 只认 Papagayo / Rhubarb(TSV|JSON) / ElevenLabs / 词时间 JSON。
 * UI 额外接受 SRT 字幕和「单词 起始秒 结束秒」每行格式：在这里转成词时间数组，
 * 再交给同一个 readTiming —— 不改一比一搬入的 core。
 */
function normalizeTimingText(src: string): string {
  const text = String(src).replace(/^\uFEFF/, '').trim()
  if (!text) return text
  if (/^MohoSwitch1/i.test(text) || /^[[{]/.test(text)) return text
  const lines = text.split(/\r?\n/).map((l) => l.trim())
  // Rhubarb TSV：每行「秒 单字母」
  if (lines.length && lines.every((l) => /^\d+(?:\.\d+)?\s+\S+$/.test(l))) return text

  const clock = (v: string): number | null => {
    const m = /^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})$/.exec(v)
    if (!m) return null
    return Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]) + Number('0.' + m[4])
  }
  const words: { word: string; start: number; end: number }[] = []
  if (/^\d+$/.test(lines[0]) && lines.some((l) => l.includes('-->'))) {
    for (let i = 0; i < lines.length; i++) {
      const m = /^(\S+)\s*-->\s*(\S+)/.exec(lines[i])
      if (!m) continue
      const a = clock(m[1]), b = clock(m[2])
      if (a === null || b === null || b <= a) continue
      const body: string[] = []
      for (let j = i + 1; j < lines.length && lines[j]; j++) body.push(lines[j])
      i += body.length
      const toks = body.join(' ').trim().split(/\s+/).filter(Boolean)
      if (!toks.length) continue
      toks.forEach((w, k) => words.push({
        word: w, start: a + (b - a) * k / toks.length,
        end: a + (b - a) * (k + 1) / toks.length,
      }))
    }
  } else {
    for (const l of lines) {
      const m = /^(\S+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)$/.exec(l)
      if (!m) continue
      const a = Number(m[2]), b = Number(m[3])
      if (b > a) words.push({ word: m[1], start: a, end: b })
    }
  }
  return words.length ? JSON.stringify(words) : text
}

export type WorkspaceProps = {
  rig: RigApi
  items: RigAnimSummary[]
  character: string
  stage: string
  onNotice: (s: string) => void
}

export function RigWorkspace(props: WorkspaceProps) {
  const [tab, setTab] = useState<Tab>('film')
  return (
    <section className="rig-ws">
      <div className="rig-ws-tabs">
        {TABS.map(([k, label]) => (
          <button type="button" key={k} className={tab === k ? 'is-on' : ''}
            onClick={() => setTab(k)}>{label}</button>
        ))}
        <span className="spacer" />
        <span className="rig-hint">核心能力来自 web/src/rig（rigby2d 运行时一比一副本）</span>
      </div>
      <div className="rig-ws-body">
        {tab === 'film' && <FilmPanel {...props} />}
        {tab === 'lipsync' && <LipSyncPanel {...props} />}
        {tab === 'scene' && <ScenePanel {...props} />}
        {tab === 'retarget' && <RetargetPanel {...props} />}
        {tab === 'import' && <ImportPanel {...props} />}
      </div>
    </section>
  )
}

/* ------------------------------ 影片工作区 ------------------------------ */

function FilmPanel({ rig, onNotice }: WorkspaceProps) {
  const sizes: any[] = output.SIZE_PRESETS
  const [sizeIdx, setSizeIdx] = useState(3)
  const [fps, setFps] = useState(30)
  const [format, setFormat] = useState<'mp4' | 'webm' | 'png'>('webm')
  const [transparent, setTransparent] = useState(true)
  const [background, setBackground] = useState('#0d1525')
  const [quality, setQuality] = useState(1)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState('')
  const [err, setErr] = useState('')
  const [w, h] = sizes[sizeIdx] || [1280, 720]
  const anims = rig.docRef.current.animations || []

  const run = async () => {
    setBusy(true); setErr(''); setProgress('准备中…')
    try {
      const a = rig.docRef.current.animations[rig.animation]
      if (!a) throw new Error('先在时间轴里选一个动画')
      const o: FilmOptions = {
        width: w, height: h, fps, format, transparent: format === 'webm' && transparent,
        background: transparent ? null : background,
        start: 0, end: a.duration || 1, quality, animation: rig.animation,
      }
      const onP = (done: number, total: number) =>
        setProgress(`${done}/${total} 帧 · ${timecode(done / fps, fps)}`)
      if (format === 'png') {
        const files = await renderPngSequence(rig.docRef.current, rig.imagesRef.current, o, onP, rig.actorRef.current)
        download(`${rig.docRef.current.name}_png.zip`, zipNs.zip(files) as Uint8Array, 'application/zip')
      } else {
        const bytes = await renderVideo(rig.docRef.current, rig.imagesRef.current, o, onP, rig.actorRef.current)
        download(`${rig.docRef.current.name || 'anim'}.${format}`, bytes,
          format === 'mp4' ? 'video/mp4' : 'video/webm')
      }
      onNotice(`影片渲染完成：${w}×${h} · ${fps}fps · ${format.toUpperCase()}`)
      setProgress('完成')
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  return (
    <div className="rig-ws-panel">
      <div className="rig-ws-grid">
        <label className="rig-field"><span>分辨率</span>
          <select value={sizeIdx} onChange={(e) => setSizeIdx(Number(e.target.value))}>
            {sizes.map((s: any, i: number) => <option key={i} value={i}>{s[2]} · {s[0]}×{s[1]}</option>)}
          </select>
        </label>
        <label className="rig-field"><span>帧率</span>
          <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
            {(output.FPS_PRESETS as number[]).map((f) => <option key={f} value={f}>{f} fps</option>)}
          </select>
        </label>
        <label className="rig-field"><span>格式</span>
          <select value={format} onChange={(e) => setFormat(e.target.value as any)}>
            <option value="webm">WebM · VP9（可透明）</option>
            <option value="mp4">MP4 · H.264</option>
            <option value="png">PNG 序列（透明 ZIP）</option>
          </select>
        </label>
        <label className="rig-field"><span>画质</span>
          <select value={quality} onChange={(e) => setQuality(Number(e.target.value))}>
            <option value={0}>低</option><option value={1}>中</option><option value={2}>高</option>
          </select>
        </label>
        <label className="rig-check"><input type="checkbox" checked={transparent}
          disabled={format === 'mp4'} onChange={(e) => setTransparent(e.target.checked)} />透明通道</label>
        {!transparent && <label className="rig-field"><span>背景</span>
          <input type="color" value={background} onChange={(e) => setBackground(e.target.value)} /></label>}
      </div>
      <div className="rig-row">
        <span className="rig-hint">时间码 {timecode(rig.time, fps)} · 当前 {w}×{h} · {fps}fps</span>
        <span className="rig-hint">{anims.length} 个动画</span>
        <span className="spacer" />
        <button type="button" className="rig-btn primary" disabled={busy || !anims.length}
          onClick={() => void run()}>{busy ? '渲染中…' : '开始渲染'}</button>
      </div>
      {!canEncodeVideo() && <p className="rig-hint">当前浏览器没有 WebCodecs；MP4/WebM 不可用，PNG 序列仍可渲染。</p>}
      {w >= 3840 && <p className="rig-hint">4K 编码很慢且占内存，建议先用 HD 预览。</p>}
      {progress && <div className="rig-status">{progress}</div>}
      {err && <div className="rig-err">{err}</div>}
    </div>
  )
}

/* ------------------------------ 视位口型同步 ------------------------------ */

function LipSyncPanel({ rig, onNotice }: WorkspaceProps) {
  const [text, setText] = useState('')
  const [words, setWords] = useState<{ word: string; start: number; end: number }[]>([])
  const [directCues, setDirectCues] = useState<VisemeCue[]>([])
  const [slotIdx, setSlotIdx] = useState(0)
  const [fps, setFps] = useState(30)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')
  const d = rig.docRef.current
  const mouths = (() => {
    const s = d.slots[slotIdx]
    return s ? lipsync.matchMouths((s.attachments || []).map((a: any) => a.name)) : {}
  })()

  const parse = () => {
    try {
      const r = lipsync.readTiming(normalizeTimingText(text), { papagayoFps: fps })
      const w = r.words || [], c = (r.cues || []) as VisemeCue[]
      if (!w.length && !c.length) throw new Error('没解析出词或视位')
      setWords(w); setDirectCues(c)
      setMsg(c.length
        ? `解析 ${r.kind}：${c.length} 条视位，可直接生成关键帧`
        : `解析 ${r.kind}：${w.length} 个带时间戳的词`)
      setErr('')
    } catch (e) { setErr(`解析失败：${e instanceof Error ? e.message : e}`) }
  }

  const transcribe = async (f: File) => {
    setBusy(true); setErr('')
    try {
      const r = await fetch(`/rig/transcribe?filename=${encodeURIComponent(f.name)}`,
        { method: 'POST', body: f })
      const j = await r.json()
      if (!r.ok) throw new Error(j.detail || `HTTP ${r.status}`)
      setWords(j.words || [])
      setDirectCues([])
      setText(j.text || '')
      setMsg(`本地 Whisper 转写完成：${(j.words || []).length} 个词`)
    } catch (e) { setErr(`转写失败：${e instanceof Error ? e.message : e}。可改用下方粘贴时间戳文本。`); }
    finally { setBusy(false) }
  }

  const apply = () => {
    try {
      const a = d.animations[rig.animation]
      if (!a) throw new Error('先在时间轴里新建一个动画')
      // Papagayo / Rhubarb 已经给了视位时间线，直接用；词时间才需要经过 visemeCues。
      const cues: VisemeCue[] = directCues.length
        ? directCues
        : lipsync.visemeCues(words, { fps, lead: 2, minHold: 2, restGap: 0.14 })
      const n = lipsync.keyLipSync(a, slotIdx, cues, mouths,
        { start: 0, end: a.duration, offset: 0 })
      rig.commit()
      setMsg(`已写入 ${n} 个口型关键帧（${cues.length} 个视位）`)
      onNotice(`口型同步完成：${n} 个关键帧`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  return (
    <div className="rig-ws-panel">
      <div className="rig-ws-grid">
        <label className="rig-field"><span>口型槽</span>
          <select value={slotIdx} onChange={(e) => setSlotIdx(Number(e.target.value))}>
            {d.slots.map((s: any, i: number) => <option key={i} value={i}>{s.name}（{s.attachments?.length || 0} 张）</option>)}
          </select>
        </label>
        <label className="rig-field"><span>FPS</span>
          <input type="number" value={fps} onChange={(e) => setFps(Number(e.target.value) || 30)} /></label>
        <label className="rig-field"><span>本地 Whisper</span>
          <input type="file" accept="audio/*" onChange={(e) => {
            const f = e.target.files?.[0]; if (f) void transcribe(f)
          }} /></label>
        <button type="button" className="rig-btn" onClick={parse}>解析时间戳文本</button>
        <button type="button" className="rig-btn primary"
          disabled={!words.length && !directCues.length}
          onClick={apply}>生成视位关键帧</button>
      </div>
      <textarea className="rig-ws-text" rows={5} value={text}
        placeholder={'粘贴 Whisper / ElevenLabs 词时间 JSON、Papagayo .dat、Rhubarb TSV/JSON、SRT 字幕，或每行「单词 起始秒 结束秒」。\n也可以上传音频用本地 Whisper 转写（首次会下载模型、固定 CPU 不抢显存）。'}
        onChange={(e) => setText(e.target.value)} />
      <div className="rig-row">
        <span className="rig-hint">视位：{(lipsync.VISEMES as string[]).join(' / ')}</span>
        <span className="spacer" />
        <span className="rig-hint">{words.length} 个词 · {directCues.length} 条视位 · {busy ? '转写中…' : ''}</span>
      </div>
      <div className="rig-visemes">
        {(lipsync.VISEMES as string[]).map((v) => (
          <span key={v} className={(mouths as any)[v] ? 'is-on' : ''}>
            <b>{v}</b>{(mouths as any)[v] || '—'}
          </span>
        ))}
      </div>
      {msg && <div className="rig-ok">{msg}</div>}
      {err && <div className="rig-err">{err}</div>}
    </div>
  )
}

/* ------------------------------ 场景 / 摄影机 ------------------------------ */

function ScenePanel({ rig, items, character, stage }: WorkspaceProps) {
  const d = rig.docRef.current
  const sc: any = d.scene || null
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')
  const [actorName, setActorName] = useState('演员 1')
  const [actorSrc, setActorSrc] = useState('')
  const [busy, setBusy] = useState('')

  const init = () => {
    d.scene = sceneNs.emptyScene()
    rig.commit(); setMsg('已建立场景（当前骨架作为舞台）')
  }
  /**
   * 放到舞台：选了来源工程就把它的 `.rigby` 真读进来（文档 + 带 `@<id>/` 前缀的图片），
   * 这样画布 / 影片渲染里真的能看到这个演员；没选来源就只建锚骨与槽位（纯元数据）。
   */
  const addActor = async () => {
    setErr(''); setMsg('')
    if (!d.scene) d.scene = sceneNs.emptyScene()
    const it = actorSrc ? items.find((x) => x.id === actorSrc) : null
    const url = it?.formats?.['project.rigby']
    // 同一个来源可以放多次：id 是演员图片的前缀与运行时键，必须唯一。
    const base = actorSrc || 'actor'
    let id = `${base}_${d.scene.actors.length + 1}`
    for (let n = 2; d.scene.actors.some((x: any) => x.id === id); n++) id = `${base}_${n}`
    if (actorSrc && !url) { setErr('这份产物没有可编辑工程（project.rigby），只能建空演员'); return }
    setBusy(url ? '载入演员工程…' : '')
    try {
      let actorDoc: any = null
      let warnings: string[] = []
      if (url) ({ doc: actorDoc, warnings } = await rig.loadActorDoc(id, url))
      const at = [d.scene.actors.length * 260 - 130, 0]
      const a = sceneNs.addActorToStage(d, {
        id, name: actorName || id, source: it?.source?.label || actorSrc || null,
        at, ground: actorDoc ? sceneNs.groundOffset(actorDoc) : 0,
      })
      if (actorDoc) rig.attachActor(a.id, actorDoc)
      else rig.commit()
      setMsg(actorDoc
        ? `已载入「${a.name}」：${actorDoc.bones.length} 骨 / ${actorDoc.animations.length} 动画，锚骨 ${a.bone}`
        : `已把「${a.name}」放到舞台上（未载入工程，仅锚骨 ${a.bone} / 槽 ${a.slot}）`)
      if (warnings.length) setErr(warnings.slice(0, 3).join('；'))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally { setBusy('') }
  }
  // 片段是按「镜头」分组的（镜头 = 舞台当前动画名），写错键名就一个都播不出来。
  const shotName = d.animations[rig.animation]?.name || ''
  const setClip = (ai: number, patch: any) => {
    if (!shotName) { setErr('先在时间轴里选一个舞台动画（它就是这个镜头）'); return }
    const a: any = d.scene.actors[ai]
    const list = (a.clips[shotName] ||= [])
    // 新片段默认用演员自己的第一个动画（选舞台动画会一个都播不出来）。
    const own = rig.actorAnimations(a.id)
    if (!list.length) list.push(sceneNs.readClip({ animation: own[0] || 'idle' }))
    Object.assign(list[0], patch)
    rig.commit()
  }
  const exportManifest = async () => {
    if (!d.scene) { setErr('先初始化场景'); return }
    setErr('')
    try {
      const r = await rig.exportSceneBundle()
      download(`${d.name || 'scene'}_scene.zip`, r.zip, 'application/zip')
      setMsg(`场景已导出：${r.files.length} 个文件（清单 + 舞台骨架/图集 + 演员骨架/图集）`)
      if (r.missing) setErr(`${r.missing} 个演员未载入工程，清单里没有他们的骨架文件`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }
  const clipOf = (a: any) => (shotName ? (a.clips?.[shotName]?.[0] || null) : null)
  const camBone = rig.cameraBone()
  const camPos = camBone >= 0 ? rig.cameraPos() : null

  return (
    <div className="rig-ws-panel">
      <div className="rig-ws-grid">
        {!sc && <button type="button" className="rig-btn primary" onClick={init}>初始化场景</button>}
        {sc && <>
          <label className="rig-field"><span>焦距</span>
            <input type="number" min={100} step={50} value={sc.focal}
              onChange={(e) => { sc.focal = Number(e.target.value) || 1000; rig.commit() }} /></label>
          <div className="rig-camera">
            <span className="rig-camera-head">摄影机</span>
            {camBone < 0 ? (
              <button type="button" className="rig-mini" onClick={() => rig.ensureCamera()}>
                ＋ 建摄影机骨
              </button>
            ) : (
              <>
                <label className="rig-field inline"><span>X</span>
                  <input type="number" step={20} value={Math.round(camPos?.[0] ?? 0)}
                    onChange={(e) => rig.setCameraPos(Number(e.target.value), camPos?.[1] ?? 0)} /></label>
                <label className="rig-field inline"><span>Y</span>
                  <input type="number" step={20} value={Math.round(camPos?.[1] ?? 0)}
                    onChange={(e) => rig.setCameraPos(camPos?.[0] ?? 0, Number(e.target.value))} /></label>
                <button type="button" className="rig-mini" onClick={() => rig.setSel({ kind: 'bone', index: camBone })}>
                  选中
                </button>
                <button type="button" className="rig-mini primary" onClick={() => rig.keyCamera()}>
                  打镜头关键帧
                </button>
              </>
            )}
          </div>
          <label className="rig-field"><span>新演员</span>
            <input value={actorName} onChange={(e) => setActorName(e.target.value)} /></label>
          <label className="rig-field"><span>来源工程</span>
            <select value={actorSrc} onChange={(e) => setActorSrc(e.target.value)}>
              <option value="">（外部 / 手动）</option>
              {items.filter((it) => it.formats['project.rigby'])
                .map((it) => <option key={it.id} value={it.id}>{it.source?.label || it.id} · {it.bones} 骨</option>)}
            </select>
          </label>
          <button type="button" className="rig-btn" disabled={!!busy} onClick={() => void addActor()}>
            {busy || '＋ 放到舞台'}
          </button>
          <button type="button" className="rig-btn primary" onClick={exportManifest}>导出场景清单</button>
        </>}
      </div>
      {sc && <>
        <div className="rig-ws-actors">
          {sc.actors.map((a: any, i: number) => {
            const c = clipOf(a)
            // 片段只能选演员自己的动画；没载入工程时退回舞台动画名（仅清单用途）。
            const anims: string[] = rig.actorAnimations(a.id)
            const choices = anims.length ? anims : d.animations.map((x: any) => x.name)
            const loaded = anims.length > 0
            return (
              <div className={`rig-actor${rig.sel?.kind === 'actor' && rig.sel.index === i ? ' is-sel' : ''}`}
                key={a.id} onClick={() => rig.setSel({ kind: 'actor', index: i })}>
                <b>{a.name}</b>
                <em>{a.id} · 深度 {sc.depth[a.bone] ?? 0} · {loaded ? '已载入' : '未载入工程'}{a.flip ? ' · 已翻转' : ''}</em>
                <div className="rig-row">
                  <label className="rig-field inline"><span>深度</span>
                    <input type="number" value={sc.depth[a.bone] ?? 0}
                      onChange={(e) => { sc.depth[a.bone] = Number(e.target.value); rig.commit() }} /></label>
                  <label className="rig-field inline"><span>动画</span>
                    <select value={c?.animation || ''} onChange={(e) => setClip(i, { animation: e.target.value })}>
                      <option value="">（无）</option>
                      {choices.map((n: string) => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </label>
                  <label className="rig-field inline"><span>起</span>
                    <input type="number" step={0.1} value={c?.start ?? 0}
                      onChange={(e) => setClip(i, { start: Number(e.target.value) })} /></label>
                  <label className="rig-field inline"><span>入</span>
                    <input type="number" step={0.1} value={c?.in ?? 0}
                      onChange={(e) => setClip(i, { in: Number(e.target.value) })} /></label>
                  <label className="rig-field inline"><span>出</span>
                    <input type="number" step={0.1} value={c?.out ?? 0}
                      onChange={(e) => setClip(i, { out: Number(e.target.value) || null })} /></label>
                  <label className="rig-field inline"><span>速度</span>
                    <input type="number" step={0.1} value={c?.speed ?? 1}
                      onChange={(e) => setClip(i, { speed: Number(e.target.value) || 1 })} /></label>
                  <label className="rig-field inline"><span>循环</span>
                    <input type="number" value={c?.loops ?? 1}
                      onChange={(e) => setClip(i, { loops: Number(e.target.value) })} /></label>
                  <label className="rig-field inline"><span>交叉淡入</span>
                    <input type="number" step={0.05} value={c?.blend ?? 0.2}
                      onChange={(e) => setClip(i, { blend: Number(e.target.value) })} /></label>
                  <button type="button" className="rig-mini" onClick={() => rig.flipActor(i)}>
                    {a.flip ? '取消翻转' : '翻转'}
                  </button>
                  <button type="button" className="rig-mini danger" onClick={() => rig.removeActor(i)}>移除</button>
                </div>
              </div>
            )
          })}
          {!sc.actors.length && <p className="rig-hint">还没有演员。加进来后每个演员是一个锚骨 + 槽位，片段支持裁剪 / 速度 / 循环 / 交叉淡入，深度控制视差。</p>}
          {!!sc.actors.length && <p className="rig-hint">在画布上点演员即可整具选中（黄色虚线框）：拖动 = 移动，切到旋转 / 缩放工具 = 转 / 缩，锚骨仍可用骨骼工具单独调。</p>}
        </div>
        <p className="rig-hint">舞台骨架：{d.bones.length} 骨 · 当前镜头 {shotName || '（未选动画）'} · 摄影机深度层 {Object.keys(sc.depth).length} 个 · 角色 {character} / 阶段 {stage}</p>
        <p className="rig-hint">镜头运动 = 给摄影机骨打关键帧（写进当前镜头动画）。深度 0 的层完全不跟镜头走，数越大跟得越多（深度 = 焦距时跟一半）。</p>
      </>}
      {msg && <div className="rig-ok">{msg}</div>}
      {err && <div className="rig-err">{err}</div>}
    </div>
  )
}

/* ------------------------------ 动画重定向 ------------------------------ */

function RetargetPanel({ rig, items, onNotice }: WorkspaceProps) {
  const [target, setTarget] = useState<any>(null)
  const [mapping, setMapping] = useState<Record<string, string>>({})
  const [animName, setAnimName] = useState('')
  const [mode, setMode] = useState<'relative' | 'world'>('relative')
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')
  const src = rig.docRef.current
  const pairs = target ? retarget.mapBones(src, target, mapping).pairs : []

  const loadTarget = async (f: File) => {
    setErr('')
    try {
      const bytes = new Uint8Array(await f.arrayBuffer())
      const imgs = new imagesNs.Images()
      const readFile = async () => bytes
      const r = await io.openProject(f.name, {
        readFile, readText: async () => new TextDecoder().decode(await readFile()),
        list: async () => [],
      }, imgs)
      setTarget(r.doc)
      setAnimName(src.animations[0]?.name || '')
      setMsg(`已载入目标骨架：${r.doc.bones.length} 骨 / ${r.doc.animations.length} 动画`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  const loadFromHistory = async (it: RigAnimSummary) => {
    const url = it.formats['project.rigby']
    if (!url) { setErr('这份产物没有可编辑工程'); return }
    try {
      const r = await fetch(url)
      const bytes = new Uint8Array(await r.arrayBuffer())
      const imgs = new imagesNs.Images()
      const readFile = async () => bytes
      const res = await io.openProject('project.rigby', {
        readFile, readText: async () => new TextDecoder().decode(await readFile()),
        list: async () => [],
      }, imgs)
      setTarget(res.doc); setAnimName(src.animations[0]?.name || '')
      setMsg(`已载入历史产物「${it.id}」作为目标骨架`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  const run = () => {
    if (!target) { setErr('先载入目标骨架'); return }
    try {
      // core 的 retarget() 已经把结果写进 target.animations（replace=true 覆盖同名），
      // 这里只负责导出，不要再自己 push 一次。报告里是 unmatched，不是 skipped。
      const r = retarget.retarget(src, animName, target,
        { mapping, mode, replace: true })
      download(`${target.name || 'target'}.retarget.json`,
        JSON.stringify(interop.writeSkeletonJson(target, { version: '4.2' }).json, null, 2),
        'application/json')
      setMsg(`重定向完成：映射 ${pairs.length} 对骨骼，未匹配 ${r.report?.unmatched?.length ?? 0} 根骨，`
        + `写入 ${r.report?.keys ?? 0} 个关键帧，跳过 ${r.report?.skipped?.length ?? 0} 项`
        + (r.report?.skipped?.length ? `（${r.report.skipped.slice(0, 2).join('；')}）` : ''))
      onNotice('动画重定向已导出为 4.2 JSON')
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  return (
    <div className="rig-ws-panel">
      <div className="rig-ws-grid">
        <label className="rig-field"><span>目标骨架</span>
          <input type="file" accept=".rigby,.json,.skel" onChange={(e) => {
            const f = e.target.files?.[0]; if (f) void loadTarget(f)
          }} /></label>
        <label className="rig-field"><span>或历史产物</span>
          <select value="" onChange={(e) => {
            const it = items.find((x) => x.id === e.target.value); if (it) void loadFromHistory(it)
          }}>
            <option value="">选择…</option>
            {items.filter((x) => x.formats['project.rigby']).map((it) =>
              <option key={it.id} value={it.id}>{it.source?.label || it.id} · {it.bones} 骨</option>)}
          </select>
        </label>
        <label className="rig-field"><span>源动画</span>
          <select value={animName} onChange={(e) => setAnimName(e.target.value)}>
            {src.animations.map((a: any) => <option key={a.name} value={a.name}>{a.name}</option>)}
          </select>
        </label>
        <label className="rig-field"><span>模式</span>
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="relative">相对（保留目标 rest pose）</option>
            <option value="world">世界（补偿不同 rest pose）</option>
          </select>
        </label>
        <button type="button" className="rig-btn primary" disabled={!target || !animName}
          onClick={run}>重定向并导出</button>
      </div>
      {target && <div className="rig-ws-map">
        <div className="rig-map-head"><span>源骨</span><span>目标骨（可改）</span><span>匹配方式</span></div>
        {pairs.map((p: any) => (
          <div className="rig-map-row" key={p.source}>
            <span>{src.bones[p.source]?.name}</span>
            <select value={mapping[src.bones[p.source]?.name] ?? target.bones[p.target]?.name ?? ''}
              onChange={(e) => setMapping((m) => ({ ...m, [src.bones[p.source].name]: e.target.value }))}>
              <option value="">（跳过）</option>
              {target.bones.map((b: any) => <option key={b.name} value={b.name}>{b.name}</option>)}
            </select>
            <em>{p.how}</em>
          </div>
        ))}
        {!pairs.length && <p className="rig-hint">没有自动匹配到骨骼，可以手动指定目标骨名。</p>}
      </div>}
      {msg && <div className="rig-ok">{msg}</div>}
      {err && <div className="rig-err">{err}</div>}
    </div>
  )
}

/* ------------------------------ 外部工程导入 ------------------------------ */

function ImportPanel({ rig, onNotice }: WorkspaceProps) {
  const [files, setFiles] = useState<{ name: string; bytes: Uint8Array }[]>([])
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')
  const [warnings, setWarnings] = useState<string[]>([])

  const pick = async (list: FileList | null) => {
    if (!list) return
    setFiles(await Promise.all([...list].map(async (f) => ({
      name: f.name, bytes: new Uint8Array(await f.arrayBuffer()),
    }))))
    setMsg(''); setErr(''); setWarnings([])
  }

  const doImport = async () => {
    setErr(''); setMsg(''); setWarnings([])
    try {
      if (!files.length) throw new Error('先选文件')
      const map = new Map<string, Uint8Array>()
      for (const f of files) map.set(basename(f.name), f.bytes)
      const get = async (p: string) => {
        const k = basename(p)
        const b = map.get(k) ?? map.get(p)
        if (!b) throw new Error(`缺少文件 ${p}`)
        return b
      }
      const fs = {
        readFile: get,
        readText: async (p: string) => new TextDecoder().decode(await get(p)),
        list: async () => [...map.keys()],
      }
      const names = [...map.keys()]
      const main = names.find((n) => /\.rigby\d?$/i.test(n))
        ?? names.find((n) => /\.skel$/i.test(n))
        ?? names.find((n) => /_ske\.json$/i.test(n))
        ?? names.find((n) => /\.json$/i.test(n) && !/atlas|_tex|_ske/i.test(n))
      if (!main) throw new Error('没找到主文件（.rigby / .skel / .json / _ske.json）')
      const imgs = new imagesNs.Images()
      const r = await io.openProject(main, fs, imgs)
      const missing = io.missingFiles(r.warnings || [])
      rig.adoptDoc(r.doc, imgs, { kind: 'import', label: `外部工程 ${main}`, name: main })
      setWarnings(r.warnings || [])
      const parts = [`已导入「${main}」：${r.doc.bones.length} 骨 / ${r.doc.slots.length} 槽 / ${r.doc.animations.length} 动画`]
      if (missing.names.length) parts.push(`迁移检查器：仍缺 ${missing.names.slice(0, 6).join('、')}`)
      if (missing.dragonBonesAtlas) parts.push('DragonBones 纹理图集缺失，图片按名称查找')
      setMsg(parts.join('；'))
      onNotice(`外部工程导入完成：${main}`)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
  }

  return (
    <div className="rig-ws-panel">
      <div className="rig-ws-grid">
        <label className="rig-field"><span>工程文件</span>
          <input type="file" multiple
            accept=".rigby,.json,.skel,.atlas,.png,.jpg,.jpeg"
            onChange={(e) => void pick(e.target.files)} /></label>
        <button type="button" className="rig-btn primary" disabled={!files.length}
          onClick={() => void doImport()}>导入到编辑器</button>
      </div>
      <p className="rig-hint">支持 Spine 导出的 .json / 二进制 .skel + .atlas + 图集页，
        DragonBones 的 _ske.json + _tex.json + 图片，以及 Rigby2D 的 .rigby 工程。请一次选中全部文件。</p>
      {!!files.length && <div className="rig-filelist">
        {files.map((f) => <span key={f.name}>{f.name}<em>{(f.bytes.length / 1024).toFixed(0)}KB</em></span>)}
      </div>}
      {msg && <div className="rig-ok">{msg}</div>}
      {!!warnings.length && <ul className="rig-warnlist">{warnings.slice(0, 8).map((w, i) => <li key={i}>{w}</li>)}</ul>}
      {err && <div className="rig-err">{err}</div>}
      <p className="rig-hint">导入会替换当前编辑器里的工程；发布时仍会写回当前角色 / 生命阶段。</p>
    </div>
  )
}
