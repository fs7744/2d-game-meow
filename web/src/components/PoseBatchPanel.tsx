/**
 * 角色迁移的**批量**面板：一个姿势集 → 一整套成片。
 *
 * 三个必须做对的点（都是用户明确要求的）：
 *
 * 1. **候选只增不删**：某一帧不满意就重新生成，新图**追加**成候选；
 *    旧候选原地留着 —— 用户很可能看完新的又觉得上一版更好。所以每帧存的是
 *    `candidates[]` + `selected`，点哪个用哪个。
 * 2. **预览要看得见**：整页网格 + 每帧的候选条，不用一次只翻一张。
 * 3. **能像帧动画那样播**：按顺序翻图就是"动作预览"（真实播放要等成片接进
 *    引擎，这里先解决"这套动作连起来顺不顺"的判断）。
 *
 * 数据全在服务端（`assets/projects/<项目>/pose_frames/<名字>/`），
 * 组件只负责读和发请求 —— 关掉浏览器再打开，候选和选择都还在。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  createPoseRender, fetchPoseRender, fetchProjectPoseSets, runPoseRender,
  selectPoseCandidate,
  type PoseRender, type ProjectPoseSet,
} from '../api'
import type { GenNode } from '../types'
import type { NodePatch } from '../nodes/GenNodeCard'

type Props = {
  projectId: string
  node: GenNode
  onPatch: (p: NodePatch) => void
  /** 底模 / 抠图后端是全局设置 */
  model?: string
  matteBackend?: string
}

/** 播放在一帧上停留多久（毫秒）。8fps 够看出动作连不连贯，又不至于像幻灯片。 */
const PLAY_MS = 125

export default function PoseBatchPanel({ projectId, node, onPatch, model, matteBackend }: Props) {
  const [sets, setSets] = useState<ProjectPoseSet[]>([])
  const [setName, setSetName] = useState(node.params.pcPoseSet || '')
  const [outName, setOutName] = useState(node.params.pcOut || '')
  const [render, setRender] = useState<PoseRender | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [pickFrame, setPickFrame] = useState(1)
  const [playing, setPlaying] = useState(false)
  const [playIndex, setPlayIndex] = useState(0)
  const [candidates, setCandidates] = useState(1)
  const playTimer = useRef<number | null>(null)

  const job = node.job
  const running = job?.status === 'queued' || job?.status === 'running'
  /** 当前显示的成片集名字：节点记着，或刚生成的那个 */
  const renderName = node.poseRender || ''

  const loadSets = useCallback(async () => {
    if (!projectId) return
    try {
      const r = await fetchProjectPoseSets(projectId)
      setSets(r.items || [])
    } catch (e) {
      setErr(e instanceof Error ? e.message : '读姿势集失败')
    }
  }, [projectId])

  const loadRender = useCallback(async () => {
    if (!projectId || !renderName) {
      setRender(null)
      return
    }
    try {
      const r = await fetchPoseRender(projectId, renderName)
      setRender(r.render)
    } catch {
      setRender(null)            // 文件夹被删了/还没生成：不当作错误
    }
  }, [projectId, renderName])

  useEffect(() => { void loadSets() }, [loadSets])
  useEffect(() => { void loadRender() }, [loadRender])

  // 任务结束时刷新（候选是服务端追加的，本地那份已经过期）
  useEffect(() => {
    if (job?.status === 'done') void loadRender()
  }, [job?.status, job?.id, loadRender])

  useEffect(() => () => {
    if (playTimer.current) window.clearInterval(playTimer.current)
  }, [])

  /** 角色图：和单张生成用的是同一份来源 */
  const charPath = node.source?.path || node.charSource?.path || ''
  const charPaths = useMemo(() => {
    const extra = (node.charSources || [])
      .map((c) => c.path)
      .filter((x): x is string => !!x)
    return [charPath, ...extra].filter((x): x is string => !!x)
  }, [charPath, node.charSources])

  const run = useCallback(async (frames: number[]) => {
    if (!projectId) { setErr('没有项目 id'); return }
    const set = (setName || '').trim()
    const out = (outName || '').trim() || `${set}-成片`
    if (!set) { setErr('先选一个姿势集'); return }
    if (!charPath) { setErr('先接一张角色图（角色迁移的来源）'); return }
    setErr('')
    setBusy(true)
    try {
      // 同名先当成"继续往这个文件夹里加"；但如果它是**另一套姿势集**的，
      // 帧表对不上，必须让用户换个名字（否则会出现张冠李戴的成片）
      try {
        await createPoseRender(projectId, out, set, node.params.prompt || '',
          { model, matteBackend })
      } catch (e) {
        const cur = await fetchPoseRender(projectId, out).catch(() => null)
        if (!cur) throw e
        if (cur.render.poseSet !== set) {
          throw new Error(`「${out}」这个文件夹是姿势集「${cur.render.poseSet}」的成片，`
            + '换一个输出名，或者把姿势集选回那一套')
        }
      }
      const r = await runPoseRender(projectId, out, {
        frames,
        candidates,
        char_image: charPath,
        char_images: charPaths,
        prompt: node.params.prompt || '',
        negative: node.params.negative || '',
        steps: node.params.steps,
        cfg: node.params.cfg,
        width: node.params.width,
        height: node.params.height,
        seed: node.params.seed,
        ref_mode: node.params.pcRefMode || 'auto',
        bg: node.params.bgTone === 'dark' ? 'dark' : 'white',
        matte: !!node.params.matte,
        matte_backend: matteBackend || 'auto',
        pixel: node.params.pixel,
        palette: node.params.palette,
        model,
      })
      onPatch({
        poseRender: out,
        params: { ...node.params, pcPoseSet: set, pcOut: out },
        job: { id: r.job_id, status: 'queued', progress: { done: 0, total: r.total } },
      })
    } catch (e) {
      setErr(e instanceof Error ? e.message : '提交失败')
    } finally {
      setBusy(false)
    }
  }, [projectId, setName, outName, charPath, charPaths, candidates, node.params,
    model, matteBackend, onPatch])

  const frames = render?.frames || []
  const picked = frames.find((f) => f.index === pickFrame) || frames[0]

  const onSelect = useCallback(async (frame: number, cand: number) => {
    if (!projectId || !renderName) return
    try {
      const r = await selectPoseCandidate(projectId, renderName, frame, cand)
      setRender(r.render)
    } catch (e) {
      setErr(e instanceof Error ? e.message : '选择失败')
    }
  }, [projectId, renderName])

  /** 播放：按顺序翻"当前选中的那一张" */
  const togglePlay = useCallback(() => {
    if (playTimer.current) { window.clearInterval(playTimer.current); playTimer.current = null }
    if (playing) { setPlaying(false); return }
    if (!frames.length) return
    setPlaying(true)
    let i = 0
    setPlayIndex(0)
    playTimer.current = window.setInterval(() => {
      i = (i + 1) % frames.length
      setPlayIndex(i)
    }, PLAY_MS)
  }, [playing, frames.length])

  useEffect(() => {
    if (!playing) return
    if (!frames.length) { setPlaying(false); return }
    if (playIndex >= frames.length) setPlayIndex(0)
  }, [playing, playIndex, frames.length])

  const shown = playing ? (frames[playIndex] || picked) : picked
  const shownUrl = shown
    ? (shown.candidates[shown.selected]?.url || '')
    : ''

  if (!projectId) {
    return <div className="src-note">批量需要项目 id（先保存项目）</div>
  }

  return (
    <div className="pose-batch">
      <div className="pose-batch-bar">
        <select
          value={setName}
          onChange={(e) => { setSetName(e.target.value); setPickFrame(1) }}
          title="人偶编辑器里「存为姿势集」存下来的那一套帧"
        >
          <option value="">
            {sets.length ? '（选姿势集）' : '（这个项目还没有姿势集）'}
          </option>
          {sets.map((s) => (
            <option key={s.name} value={s.name}>{s.name}（{s.count} 帧）</option>
          ))}
        </select>
        <button
          type="button" className="mini" onClick={() => void loadSets()}
          disabled={running} title="重新扫一遍项目里的姿势集"
        >↻</button>
      </div>

      <div className="pose-batch-bar">
        <input
          className="pose-batch-out"
          value={outName}
          placeholder={setName ? `${setName}-成片` : '输出文件夹名'}
          onChange={(e) => setOutName(e.target.value)}
          title={'存到本项目的 pose_frames/<这个名字>/。已存在就**接着往里加**（候选只增不删）'}
        />
        <label className="pose-batch-n" title="每帧出几个候选（多了能挑，但每张约 25 秒）">
          候选
          <input
            type="number" min={1} max={4} value={candidates}
            onChange={(e) => setCandidates(Math.max(1, Math.min(4, Number(e.target.value) || 1)))}
          />
        </label>
        <button
          type="button" className="pose-batch-btn"
          disabled={!setName || running || busy}
          onClick={() => void run([])}
          title="把整个姿势集逐帧生成，进度在节点底部"
        >
          {running ? '生成中…' : '全部生成'}
        </button>
      </div>

      {err && <div className="err" title={err}>⚠ {err}</div>}
      {/* 部分候选失败时**必须说出来**：否则用户看到"6/8 帧"以为是自己没点到 */}
      {(() => {
        const f = (job?.result as { failed?: unknown[] } | undefined)?.failed
        if (!Array.isArray(f) || !f.length) return null
        const first = f[0] as { frame?: number; error?: string } | undefined
        return (
          <div className="warn" title={String(first?.error || '')}>
            ⚠ {f.length} 张没出来（第 {first?.frame} 帧起）；再点「重生成」会继续追加，不会丢已有的
          </div>
        )
      })()}
      {/* 部分候选失败时**必须说出来**：否则用户看到"6/8 帧"以为是自己没点到 */}
      {(() => {
        const f = (job?.result as { failed?: unknown[] } | undefined)?.failed
        if (!Array.isArray(f) || !f.length) return null
        const first = f[0] as { frame?: number; error?: string } | undefined
        return (
          <div className="warn" title={String(first?.error || '')}>
            ⚠ {f.length} 张没出来（第 {first?.frame} 帧起）；再点「重生成」会继续追加，不会丢已有的
          </div>
        )
      })()}

      {renderName && render && (
        <>
          <div className="pose-batch-head">
            <b>{render.name}</b>
            <span>{render.done}/{render.total} 帧</span>
            <button
              type="button" className="mini"
              onClick={togglePlay}
              disabled={!frames.some((f) => f.selected >= 0)}
              title="按顺序播一遍，看这套动作连不连贯"
            >{playing ? '■' : '▶'}</button>
          </div>

          {/* 大图：当前帧选中的那张 */}
          <div className="pose-batch-main">
            {shownUrl
              ? <img src={shownUrl} alt={`第 ${shown?.index} 帧`} draggable={false} />
              : <span className="pose-batch-empty">还没出图：点「全部生成」</span>}
            {shown && shownUrl && (
              <span className="pose-batch-tag">
                第 {shown.index} 帧 · 候选 {(shown.selected ?? 0) + 1}/{shown.candidates.length || 0}
              </span>
            )}
          </div>

          {/* 缩略网格：一眼看完整套，点哪帧编哪帧 */}
          <div className="pose-batch-grid">
            {frames.map((f) => {
              const url = f.candidates[f.selected]?.url || ''
              return (
                <button
                  key={f.index}
                  type="button"
                  className={`pose-batch-cell${f.index === (shown?.index ?? -1) ? ' on' : ''}${f.selectedIndex < 0 ? ' empty' : ''}`}
                  onClick={() => { setPlaying(false); setPickFrame(f.index) }}
                  title={`第 ${f.index} 帧（${f.candidates.length} 个候选）`}
                >
                  {url ? <img src={url} alt={`第 ${f.index} 帧`} draggable={false} /> : <span>—</span>}
                  <i>{f.index}</i>
                </button>
              )
            })}
          </div>

          {/* 当前帧的候选条：**旧的一个都不删**，点哪个用哪个 */}
          {picked && (
            <div className="pose-batch-cands">
              <div className="pose-batch-cands-head">
                <span>第 {picked.index} 帧的候选（{picked.candidates.length}）</span>
                <button
                  type="button" className="pose-batch-btn"
                  disabled={running || busy}
                  onClick={() => void run([picked.index])}
                  title="按当前的「候选」数再出一组，**追加**在旧的后面（旧的一张都不会删）"
                >重生成</button>
              </div>
              <div className="pose-batch-strip">
                {picked.candidates.map((c) => (
                  <button
                    key={c.index}
                    type="button"
                    className={`pose-batch-cell${c.index === picked.selectedIndex ? ' on' : ''}`}
                    onClick={() => void onSelect(picked.index, c.index)}
                    title={c.seed != null ? `候选 ${c.index}（seed ${c.seed}）` : `候选 ${c.index}`}
                  >
                    <img src={c.url} alt={`候选 ${c.index}`} draggable={false} />
                    <i>{c.index}</i>
                  </button>
                ))}
                {!picked.candidates.length && <span className="pose-batch-empty">这一帧还没出图</span>}
              </div>
            </div>
          )}
        </>
      )}

      {!renderName && (
        <div className="src-note">
          选一套姿势集 → 取个文件夹名 → 「全部生成」。每帧的候选都会留着，
          哪张不满意就单独重生成，旧的那张随时能点回来。
        </div>
      )}
    </div>
  )
}
