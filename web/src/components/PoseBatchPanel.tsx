/**
 * 姿态批量（`posebatch` 节点）：**一个姿势集 → 一整套精灵帧**。
 *
 * 为什么单独成节点，而不是塞进「角色迁移」里当个折叠项：
 *
 * - **单张和批量是两种模式**。混在一张卡片里，用户永远要先想"我现在是在批量
 *   还是单张"，折叠起来还会出现"选了姿势集却没接角色图"这种半截状态（实测
 *   报错文案把锅甩给了角色图，而角色图明明接了 —— 那是路径没解析对）。
 * - 批量要**先看清这套动作是什么**（缩略条 + 播放），再决定要不要花一小时出图。
 * - 出完要**逐帧和原始姿势对照**，才知道哪一帧没换好、该单独重生成哪一帧。
 *
 * 三条硬要求（用户提的，也是这个节点存在的理由）：
 *
 * 1. **先看清**：选完姿势集立刻铺出整组源帧，`▶` 能连着播；
 * 2. **能对比**：左边原始姿势帧、右边选中的结果，一屏内左右对照；
 * 3. **候选只增不删**：不满意就重生成，新图**追加**，旧的原地留着随时点回来
 *    —— 用户很可能看完新的又觉得上一版更好。
 *
 * 数据全在服务端（`assets/projects/<项目>/pose_frames/<名字>/`），
 * 组件只负责读和发请求 —— 关掉浏览器再打开，候选和选择都还在。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createPoseRender, exportPoseRenderGif, exportPoseRenderGodot, exportPoseSetGif,
  fetchPoseRender, fetchProjectPoseSet, fetchProjectPoseSets, runPoseRender,
  selectPoseCandidate,
  type GodotBundle, type PoseGif, type PoseRender, type ProjectPoseSet,
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
  /** 角色图（磁盘路径，第一张是身份来源，后面是额外角度） */
  charPaths: string[]
  /** 有没有接过角色图（只用来提示"还没接"） */
  hasChar?: boolean
  onPreview?: (url: string, name: string) => void
}

/** 播放在一帧上停留多久（毫秒）。8fps 够看出动作连不连贯，又不至于像幻灯片。 */
const PLAY_MS = 125

export default function PoseBatchPanel({
  projectId, node, onPatch, model, matteBackend, charPaths, hasChar, onPreview,
}: Props) {
  const [sets, setSets] = useState<ProjectPoseSet[]>([])
  const [setName, setSetName] = useState(node.params.pcPoseSet || '')
  const [outName, setOutName] = useState(node.params.pcOut || '')
  /** 选中的姿势集的源帧 —— 用来"先看清要做的是哪套动作" */
  const [source, setSource] = useState<ProjectPoseSet | null>(null)
  const [render, setRender] = useState<PoseRender | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [pickFrame, setPickFrame] = useState(1)
  const [playing, setPlaying] = useState(false)
  const [playIndex, setPlayIndex] = useState(0)
  const [candidates, setCandidates] = useState(1)
  /** 对照视图：左原始 / 右结果。关掉就只看结果大图 */
  const [compare, setCompare] = useState(true)
  /** 导出的 GIF：源帧一份、成片一份（团队审核用，拿到文件就能看动作连贯性） */
  const [srcGif, setSrcGif] = useState<PoseGif | null>(null)
  const [resGif, setResGif] = useState<PoseGif | null>(null)
  const [gifBusy, setGifBusy] = useState<'src' | 'res' | ''>('')
  /** Godot 工程包（逐帧 PNG + .tres + .tscn + zip） */
  const [godot, setGodot] = useState<GodotBundle | null>(null)
  const [godotBusy, setGodotBusy] = useState(false)
  const [gifFps, setGifFps] = useState(12)
  const playTimer = useRef<number | null>(null)

  const job = node.job
  const running = job?.status === 'queued' || job?.status === 'running'
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

  /** 拉源帧：选完姿势集立刻能看到这套动作长什么样 */
  const loadSource = useCallback(async (name: string) => {
    if (!projectId || !name) { setSource(null); return }
    try {
      const r = await fetchProjectPoseSet(projectId, name)
      setSource(r.set)
    } catch {
      setSource(null)
    }
  }, [projectId])

  const loadRender = useCallback(async () => {
    if (!projectId || !renderName) { setRender(null); return }
    try {
      const r = await fetchPoseRender(projectId, renderName)
      setRender(r.render)
    } catch {
      setRender(null)          // 还没生成过 / 文件夹被删：不当作错误
    }
  }, [projectId, renderName])

  useEffect(() => { void loadSets() }, [loadSets])
  useEffect(() => { void loadSource(setName) }, [loadSource, setName])
  useEffect(() => { void loadRender() }, [loadRender])
  // 任务结束时刷新（候选是服务端追加的，本地那份已经过期）
  useEffect(() => {
    if (job?.status === 'done') void loadRender()
  }, [job?.status, job?.id, loadRender])

  useEffect(() => () => {
    if (playTimer.current) window.clearInterval(playTimer.current)
  }, [])

  const run = useCallback(async (frames: number[]) => {
    if (!projectId) { setErr('没有项目 id'); return }
    const set = (setName || '').trim()
    const out = (outName || '').trim() || `${set}-成片`
    if (!set) { setErr('先选一个姿势集'); return }
    if (!charPaths.length) { setErr('先接一张角色图（左侧「角色」端口）'); return }
    setErr('')
    setBusy(true)
    try {
      // 同名先当成"继续往这个文件夹里加候选"；但如果里面是**另一套姿势集**的
      // 成片，帧表对不上，必须让用户换个名字（否则会张冠李戴）
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
        char_image: charPaths[0],
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
  }, [projectId, setName, outName, charPaths, candidates, node.params,
    model, matteBackend, onPatch])

  const onSelect = useCallback(async (frame: number, cand: number) => {
    if (!projectId || !renderName) return
    try {
      const r = await selectPoseCandidate(projectId, renderName, frame, cand)
      setRender(r.render)
    } catch (e) {
      setErr(e instanceof Error ? e.message : '选择失败')
    }
  }, [projectId, renderName])

  /** 导出 GIF：`kind` 决定导原始帧还是成片（成片取当前选中的候选） */
  const exportGif = useCallback(async (kind: 'src' | 'res') => {
    if (!projectId) return
    const name = kind === 'src' ? (setName || '').trim() : renderName
    if (!name) { setErr(kind === 'src' ? '先选一个姿势集' : '还没有成片可导'); return }
    setErr('')
    setGifBusy(kind)
    try {
      const r = kind === 'src'
        ? await exportPoseSetGif(projectId, name)
        : await exportPoseRenderGif(projectId, name)
      if (kind === 'src') setSrcGif(r.gif)
      else setResGif(r.gif)
      if (r.gif.skipped > 0) {
        setErr(`GIF 已生成，但有 ${r.gif.skipped} 帧还没有选中的候选，被跳过了`)
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : '导出 GIF 失败')
    } finally {
      setGifBusy('')
    }
  }, [projectId, setName, renderName])

  /** 导出 Godot 4 工程包：解到工程根目录就能播，不用在引擎里拼 */
  const exportGodot = useCallback(async () => {
    if (!projectId || !renderName) return
    setErr('')
    setGodotBusy(true)
    try {
      const r = await exportPoseRenderGodot(projectId, renderName, {
        stem: renderName, anim: renderName, fps: gifFps, loop: true,
      })
      setGodot(r.bundle)
      if (r.bundle.warnings?.length) setErr(r.bundle.warnings.join('；'))
    } catch (e) {
      setErr(e instanceof Error ? e.message : '导出 Godot 失败')
    } finally {
      setGodotBusy(false)
    }
  }, [projectId, renderName, gifFps])

  /** 播放：按顺序翻"当前选中的那一张"。源帧与结果共用一条时间轴 */
  const togglePlay = useCallback(() => {
    if (playTimer.current) { window.clearInterval(playTimer.current); playTimer.current = null }
    if (playing) { setPlaying(false); return }
    const n = Math.max(source?.frames?.length || 0, render?.frames.length || 0)
    if (!n) return
    setPlaying(true)
    let i = 0
    setPlayIndex(0)
    playTimer.current = window.setInterval(() => {
      i = (i + 1) % n
      setPlayIndex(i)
    }, PLAY_MS)
  }, [playing, source, render])

  const frames = render?.frames || []
  const srcFrames = source?.frames || []
  /** 播放时按索引同步走；不播放时跟着点选的那一帧 */
  const shownFrame = playing
    ? (frames[playIndex] || null)
    : (frames.find((f) => f.index === pickFrame) || frames[0] || null)
  const shownSrc = playing
    ? (srcFrames[playIndex] || null)
    : (srcFrames.find((f) => f.index === pickFrame) || srcFrames[0] || null)
  const resultUrl = shownFrame ? (shownFrame.candidates[shownFrame.selected]?.url || '') : ''

  if (!projectId) {
    return <div className="src-note">需要项目 id（先把项目保存一次）</div>
  }

  return (
    <div className="pose-batch">
      {/* ---- 1) 选姿势集 ---- */}
      <div className="pose-batch-bar">
        <select
          value={setName}
          onChange={(e) => { setSetName(e.target.value); setPlaying(false) }}
          title="人偶编辑器里「存为姿势集」存下来的那一套"
        >
          <option value="">{sets.length ? '（选姿势集）' : '（这个项目还没有姿势集）'}</option>
          {sets.map((s) => (
            <option key={s.name} value={s.name}>{s.name}（{s.count} 帧）</option>
          ))}
        </select>
        <button
          type="button" className="pose-batch-btn" onClick={() => void loadSets()}
          disabled={running} title="重新扫一遍项目里的姿势集"
        >刷新</button>
      </div>

      {/* ---- 2) 先看清是哪套动作：源帧缩略条 + 播放 ---- */}
      {source && srcFrames.length > 0 && (
        <div className="pose-batch-source">
          <div className="pose-batch-head">
            <b>{source.name}</b>
            <span>{srcFrames.length} 帧 · 原始姿势</span>
            <button
              type="button" className="pose-batch-btn"
              onClick={togglePlay}
              title="按顺序播一遍，先确认这是不是你要的那套动作"
            >{playing ? '■ 停' : '▶ 播'}</button>
            <button
              type="button" className="pose-batch-btn"
              disabled={gifBusy === 'src'}
              onClick={() => void exportGif('src')}
              title="把这套原始帧导成 GIF，发群里/发给同事看动作连不连贯"
            >{gifBusy === 'src' ? '合成中…' : '导出 GIF'}</button>
          </div>
          {srcGif && (
            <div className="pose-batch-gif">
              <a href={srcGif.url} target="_blank" rel="noreferrer"
                title="点开看动图（浏览器里直接播）">预览 GIF</a>
              <a href={srcGif.url} download={`${srcGif.name}-原始.gif`}
                title="下载后可以直接发到群里审核">下载</a>
              <span>{srcGif.frames} 帧 · {Math.round(srcGif.bytes / 1024)}KB</span>
            </div>
          )}
          <div className="pose-batch-strip">
            {srcFrames.map((f, i) => (
              <button
                key={f.index}
                type="button"
                className={`pose-batch-cell${(playing ? i === playIndex : f.index === pickFrame) ? ' on' : ''}`}
                onClick={() => { setPlaying(false); setPickFrame(f.index) }}
                onDoubleClick={() => onPreview?.(f.url, `第 ${f.index} 帧 · 原始`)}
                title={`第 ${f.index} 帧${f.t ? ` · ${f.t.toFixed(2)}s` : ''}（双击放大）`}
              >
                <img src={f.url} alt={`源 ${f.index}`} draggable={false} />
                <i>{f.index}</i>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ---- 3) 出图设置 ---- */}
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
          disabled={!setName || running || busy || !charPaths.length}
          onClick={() => void run([])}
          title={!charPaths.length ? '先接一张角色图（左侧「角色」端口）'
            : '把整个姿势集逐帧生成，进度在节点底部'}
        >
          {running ? '生成中…' : '全部生成'}
        </button>
      </div>

      {!charPaths.length && !hasChar && (
        <div className="src-note">
          还没接角色图：把上游节点的输出点拖到左边的「角色」端口，再点「全部生成」。
        </div>
      )}
      {err && <div className="err" title={err}>⚠ {err}</div>}
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

      {/* ---- 4) 结果：左原始 / 右结果，逐帧对照 ---- */}
      {renderName && render && (
        <>
          <div className="pose-batch-head">
            <b>{render.name}</b>
            <span>{render.done}/{render.total} 帧</span>
            <label className="pose-batch-toggle" title="左边原始姿势、右边生成结果，一屏对照">
              <input
                type="checkbox" checked={compare}
                onChange={(e) => setCompare(e.target.checked)}
              />
              对照原始
            </label>
          </div>

          <div className={`pose-batch-pair${compare ? '' : ' single'}`}>
            {compare && (
              <figure>
                <img src={shownSrc?.url || ''} alt="原始姿势" draggable={false}
                  onDoubleClick={() => shownSrc && onPreview?.(shownSrc.url, '原始姿势')} />
                <figcaption>原始 · 第 {shownSrc?.index ?? '—'} 帧</figcaption>
              </figure>
            )}
            <figure>
              {resultUrl
                ? <img src={resultUrl} alt="生成结果" draggable={false}
                  onDoubleClick={() => onPreview?.(resultUrl, '生成结果')} />
                : <div className="pose-batch-empty">还没出图：点「全部生成」</div>}
              <figcaption>
                结果{shownFrame
                  ? ` · 第 ${shownFrame.index} 帧 · 候选 ${(shownFrame.selected ?? 0) + 1}/${shownFrame.candidates.length || 0}`
                  : ''}
              </figcaption>
            </figure>
          </div>

          {/* 导出：GIF（给团队审核）+ Godot 工程包（直接进引擎） */}
          <div className="pose-batch-export">
            <button
              type="button" className="pose-batch-btn"
              disabled={gifBusy === 'res' || !render.done}
              onClick={() => void exportGif('res')}
              title="把成片导成 GIF（每帧取你当前选中的那张），发出去给团队审核"
            >{gifBusy === 'res' ? '合成中…' : '导出 GIF'}</button>
            <button
              type="button" className="pose-batch-btn"
              disabled={godotBusy || !render.done}
              onClick={() => void exportGodot()}
              title={'导出 Godot 4 工程包：逐帧 PNG + SpriteFrames(.tres) + '
                + 'AnimatedSprite2D(.tscn) + 说明 + zip。解到工程根目录就能播'}
            >{godotBusy ? '打包中…' : '导出 Godot'}</button>
            <label className="pose-batch-n" title="动画帧率（图集和 .tres 里都按它写）">
              帧率
              <input
                type="number" min={1} max={60} value={gifFps}
                onChange={(e) => setGifFps(Math.max(1, Math.min(60, Number(e.target.value) || 12)))}
              />
            </label>
          </div>

          {resGif && (
            <div className="pose-batch-gif">
              <a href={resGif.url} target="_blank" rel="noreferrer"
                title="点开看动图（浏览器里直接播）">预览 GIF</a>
              <a href={resGif.url} download={`${resGif.name}.gif`}
                title="下载后可以直接发到群里审核">下载</a>
              <span>{resGif.frames} 帧 · {Math.round(resGif.bytes / 1024)}KB
                {resGif.skipped ? ` · 跳过 ${resGif.skipped} 帧` : ''}</span>
            </div>
          )}

          {godot && (
            <div className="pose-batch-gif">
              <a href={godot.zip.url} download={godot.zip.name}
                title={godot.resPaths.scene}>下载工程包 zip</a>
              {godot.files.map((f) => (
                <a key={f.name} href={f.url} download={f.name}>{f.name}</a>
              ))}
              <span>{godot.frames} 帧 · {Math.round(godot.zip.bytes / 1024)}KB
                · {godot.resPaths.scene}</span>
            </div>
          )}

          {/* 缩略网格：一眼看完整套，点哪帧编哪帧 */}
          <div className="pose-batch-grid">
            {frames.map((f) => {
              const url = f.candidates[f.selected]?.url || ''
              return (
                <button
                  key={f.index}
                  type="button"
                  className={`pose-batch-cell${f.index === (shownFrame?.index ?? -1) ? ' on' : ''}${f.selectedIndex < 0 ? ' empty' : ''}`}
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
          {shownFrame && (
            <div className="pose-batch-cands">
              <div className="pose-batch-cands-head">
                <span>第 {shownFrame.index} 帧的候选（{shownFrame.candidates.length}）</span>
                <button
                  type="button" className="pose-batch-btn"
                  disabled={running || busy}
                  onClick={() => void run([shownFrame.index])}
                  title="按当前的「候选」数再出一组，**追加**在旧的后面（旧的一张都不会删）"
                >重生成</button>
              </div>
              <div className="pose-batch-strip">
                {shownFrame.candidates.map((c) => (
                  <button
                    key={c.index}
                    type="button"
                    className={`pose-batch-cell${c.index === shownFrame.selectedIndex ? ' on' : ''}`}
                    onClick={() => void onSelect(shownFrame.index, c.index)}
                    title={c.seed != null ? `候选 ${c.index}（seed ${c.seed}）` : `候选 ${c.index}`}
                  >
                    <img src={c.url} alt={`候选 ${c.index}`} draggable={false} />
                    <i>{c.index}</i>
                  </button>
                ))}
                {!shownFrame.candidates.length && <span className="pose-batch-empty">这一帧还没出图</span>}
              </div>
            </div>
          )}
        </>
      )}

      {!renderName && (
        <div className="src-note">
          选一套姿势集 → 看缩略条确认是这套动作 → 取个文件夹名 → 「全部生成」。
          每帧的候选都会留着：哪张不满意就单独「重生成」，旧的那张随时能点回来。
        </div>
      )}
    </div>
  )
}
