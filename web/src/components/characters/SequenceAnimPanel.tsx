/**
 * 角色中心 · 序列帧动画（与「角色设定 / 拆部件 / 骨骼动画」同级）。
 *
 * 这里放的就是画布上那两个节点的**全部内容**：
 *
 *   1. `poseedit`（3D 人偶）：官方 VNCCS PoseStudio 内嵌，摆姿势 / 摆场景 /
 *      导入 glb 动作 / 抽帧 / 存姿势集；
 *   2. `posebatch`（参考图批量帧）：选姿势集 → 逐帧套到本阶段的立绘上 →
 *      候选只增不删 → GIF / Godot 导出。
 *
 * **产物属于这个角色的这个生命阶段**，不是某个画布项目：
 * `assets/characters/<角色>/Sequence[/s_<阶段>]/`。所以姿势集 / 成片集接口都传
 * `{character, stage}` 作用域（后端同一份 `poseset` 实现，只换根目录）。
 *
 * 人偶编辑器是个 37MB 的 iframe，切子页**不能卸载**：两份内容常驻，只用
 * `.is-hidden` 切换（项目约定，见侧栏 3 tab 的注释）。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import PoseStudio, { type PoseSetFrameIn } from '../PoseStudio'
import PoseBatchPanel from '../PoseBatchPanel'
import Lightbox from '../Lightbox'
import {
  addPoseApi, createProjectPoseSet, extractPose, fetchJob, fetchSequence,
  fetchSequenceStudio, getLibrary, saveSequenceStudio, uploadImage,
  uploadSequenceMannequin,
  type PoseRenderSummary, type PortraitItem, type SequenceSummary,
} from '../../api'
import { DEFAULT_PARAMS } from '../../nodes/defaults'
import type { NodePatch } from '../../nodes/GenNodeCard'
import type {
  GenNode, JobState, PosePreset, PoseScene,
} from '../../types'

type Props = {
  character: string
  stage: string
  stageTitle: string
  /** 本阶段的立绘（批量帧拿它当身份来源） */
  portraits: PortraitItem[]
  matteBackend?: string
  onNotice: (msg: string) => void
}

type StudioState = {
  pose?: unknown
  yaw: number
  scene?: PoseScene | null
}

/** 抽帧面板投递的 2D 关节点（从图片提姿势用，收完就被真正的 pose 取代） */
type ImportPoints = {
  points: (number[] | null)[]
  tick: number
  reprojected?: boolean
}

function makeBatchNode(): GenNode {
  return {
    id: 'seq-batch',
    kind: 'posebatch',
    x: 0,
    y: 0,
    w: 400,
    view: 'expanded',
    params: {
      ...DEFAULT_PARAMS,
      // 批量链路跟着人偶编辑器走：VNCCS 引擎 + 灰人偶直接当条件图。
      // 和画布节点同一套默认值（`DEFAULT_PARAMS` 是全项目唯一一份）。
      steps: 25,
      pcEngine: 'wvnccs',
      pcRefMode: 'skeleton',
      bgTone: 'white',
    },
  }
}

const TONE_OPTIONS: Array<[string, string]> = [
  ['white', '白底'],
  ['dark', '深色底（白毛 / 白甲）'],
  ['transparent', '透明底'],
]

export default function SequenceAnimPanel({
  character, stage, stageTitle, portraits, matteBackend = 'auto', onNotice,
}: Props) {
  const [tab, setTab] = useState<'studio' | 'batch'>('studio')
  const [summary, setSummary] = useState<SequenceSummary | null>(null)
  const [studio, setStudio] = useState<StudioState>({ yaw: 0 })
  /** 存档读回来之前别回写，否则一进页面就把 studio.json 覆盖成空姿势 */
  const [ready, setReady] = useState(false)
  /** PoseStudio 的姿势库（跨项目复用，和画布同一个 library/poses.json） */
  const [poses, setPoses] = useState<PosePreset[]>([])
  const [poseImport, setPoseImport] = useState<ImportPoints | null>(null)
  const [importing, setImporting] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [preview, setPreview] = useState<{ url: string; name: string } | null>(null)
  const [err, setErr] = useState('')

  /** 概览第一次回来时把「最近一套姿势集 / 成片」灌进批量节点：
   *  面板的输入框只在挂载时读一次 node.params，所以必须**在它挂载前**灌好
   *  （下面的面板用 `summary &&` 门控）。 */
  const seeded = useRef(false)
  /** 种子灌进节点之后再挂载批量面板（它只在挂载时读一次 node.params） */
  const [seedReady, setSeedReady] = useState(false)

  /** 姿势集存完要让批量面板重新扫一遍（它自己只在挂载时扫） */
  const [setsVersion, setSetsVersion] = useState(0)

  /** 批量节点：**不入画布**，只用来喂 PoseBatchPanel（它要一份 node 状态） */
  const [node, setNode] = useState<GenNode>(makeBatchNode)
  const patchNode = useCallback((p: NodePatch) => {
    setNode((n) => ({ ...n, ...p, params: p.params ?? n.params }))
  }, [])

  /** 切子页时把内容区滚回顶部：人偶页滚到底再切到批量页，设置条会被卷到
   *  视口上方（截图核对时发现的）。 */
  const switchTab = useCallback((t: 'studio' | 'batch') => {
    setTab(t)
    window.requestAnimationFrame(() => {
      const box = document.querySelector<HTMLElement>('.chars-body.cc-panel')
      // jsdom 没实现 Element.scrollTo，测试环境退回 scrollTop
      if (box && typeof box.scrollTo === 'function') box.scrollTo({ top: 0 })
      else if (box) box.scrollTop = 0
    })
  }, [])

  /* ------------------------------------------------------------ 概览 */
  const reloadSummary = useCallback(async () => {
    try {
      setSummary(await fetchSequence(character, stage))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [character, stage])

  useEffect(() => { void reloadSummary() }, [reloadSummary])

  useEffect(() => {
    if (seeded.current || !summary) return
    seeded.current = true
    const firstSet = summary.poseSets[0]?.name || ''
    const firstRender = summary.renders[0]?.name || ''
    if (firstSet || firstRender) {
      setNode((n) => ({
        ...n,
        poseRender: firstRender || n.poseRender,
        params: {
          ...n.params,
          pcPoseSet: n.params.pcPoseSet || firstSet,
          pcOut: n.params.pcOut || firstRender,
        },
      }))
    }
    setSeedReady(true)
  }, [summary])

  /* ---------------------------------------------------- 人偶状态存档 */
  useEffect(() => {
    let alive = true
    setReady(false)
    fetchSequenceStudio(character, stage)
      .then((r) => {
        if (!alive) return
        setStudio({
          pose: r.studio?.pose,
          yaw: Number(r.studio?.yaw) || 0,
          scene: (r.studio?.scene as PoseScene) || null,
        })
        setReady(true)
      })
      .catch(() => { if (alive) setReady(true) })   // 读不到就当空的开始
    return () => { alive = false }
  }, [character, stage])

  /** 姿势 / 机位 / 场景变了就 debounce 存一次（拖关节时别每帧发请求） */
  useEffect(() => {
    if (!ready) return
    const t = window.setTimeout(() => {
      void saveSequenceStudio(character, stage, {
        pose: studio.pose,
        yaw: studio.yaw,
        scene: studio.scene,
        updatedAt: Date.now(),
      }).catch(() => { /* 自动存档失败不打扰用户，下次改动还会再存 */ })
    }, 900)
    return () => window.clearTimeout(t)
  }, [ready, studio, character, stage])

  /* -------------------------------------------------------------- 姿势库 */
  useEffect(() => {
    getLibrary()
      .then((l) => setPoses(Array.isArray(l?.poses) ? l.poses : []))
      .catch(() => setPoses([]))
  }, [])

  const savePoseToLibrary = useCallback(async (
    name: string, pose: unknown, yaw: number, previewUrl: string,
  ) => {
    try {
      const r = await addPoseApi(name, pose, yaw, previewUrl)
      setPoses(r.poses ?? [])
      onNotice(`姿势「${name}」已存入资料库`)
    } catch (e) {
      onNotice(`存姿势失败：${e instanceof Error ? e.message : '未知错误'}`)
    }
  }, [onNotice])

  /* -------------------------------------------------- 抽帧 → 姿势集 */
  const savePoseSet = useCallback(async (
    name: string, frames: PoseSetFrameIn[], source: Record<string, unknown> = {},
  ) => {
    try {
      const r = await createProjectPoseSet({ character, stage }, name, frames, {
        count: frames.length,
        at: Date.now(),
        character,
        stage: stage || 'default',
        ...source,
      })
      onNotice(`已存成姿势集「${r.set.name}」（${r.set.count} 帧，`
        + `在本阶段 Sequence 目录下）`)
      // 新姿势集直接选进批量节点：重挂面板后输入框就是这一套，不用再手选
      setNode((n) => ({
        ...n,
        params: {
          ...n.params,
          pcPoseSet: r.set.name,
          pcOut: `${r.set.name}-成片`,
        },
      }))
      setSetsVersion((v) => v + 1)
      void reloadSummary()
      return true
    } catch (e) {
      setErr(e instanceof Error ? e.message : '存姿势集失败')
      return false
    }
  }, [character, stage, onNotice, reloadSummary])

  const exportMannequin = useCallback(async (blob: Blob) => {
    setExporting(true)
    try {
      const r = await uploadSequenceMannequin(character, stage, blob)
      onNotice('人偶图已存进本阶段（可在下方「人偶图」里下载）')
      setPreview({ url: r.url, name: r.name })
      void reloadSummary()
    } catch (e) {
      setErr(e instanceof Error ? e.message : '导出人偶图失败')
    } finally {
      window.setTimeout(() => setExporting(false), 600)
    }
  }, [character, stage, onNotice, reloadSummary])

  const importPoseFromImage = useCallback(async (file: File) => {
    setImporting(true)
    try {
      const up = await uploadImage(file)
      const r = await extractPose(up.path)
      setPoseImport({ points: r.points, tick: Date.now(), reprojected: r.reprojected })
      onNotice(`已从图片提取 ${r.joints}/18 个关节，可在人偶上微调`)
    } catch (e) {
      setErr(e instanceof Error ? e.message : '提取姿势失败')
    } finally {
      setImporting(false)
    }
  }, [onNotice])

  /* ------------------------------------------- 批量任务：轮询到结束 */
  const jobId = node.job?.id
  const jobStatus = node.job?.status
  useEffect(() => {
    if (!jobId || (jobStatus !== 'queued' && jobStatus !== 'running')) return
    let alive = true
    const tick = async () => {
      try {
        const j = await fetchJob(jobId)
        if (!alive) return
        setNode((n) => (n.job?.id === jobId
          ? { ...n, job: { ...(n.job as JobState), ...j } }
          : n))
        if (j.status === 'done') {
          onNotice('序列帧生成完成')
          void reloadSummary()
        } else if (j.status === 'failed') {
          setErr(`批量失败：${j.error || '未知原因'}`)
        }
      } catch { /* 单次查询失败不致命，下一轮继续 */ }
    }
    const timer = window.setInterval(() => { void tick() }, 1500)
    void tick()
    return () => { alive = false; window.clearInterval(timer) }
  }, [jobId, jobStatus, onNotice, reloadSummary])

  /* ------------------------------------------------------------ 立绘 */
  const [charPath, setCharPath] = useState('')
  useEffect(() => {
    if (!portraits.length) { setCharPath(''); return }
    if (!portraits.some((p) => p.path === charPath)) setCharPath(portraits[0].path)
  }, [portraits, charPath])
  const charPaths = useMemo(
    () => (charPath ? [charPath] : []), [charPath])

  const renders: PoseRenderSummary[] = summary?.renders || []
  const poseSets = summary?.poseSets || []
  const mannequins = summary?.mannequins || []

  return (
    <div className="cc-seq">
      <div className="cc-seq-head">
        <div>
          <h2>序列帧动画</h2>
          <p>
            3D 人偶摆姿势 → 抽成一整套帧 → 用本阶段立绘批量换皮。
            产物按生命阶段归档（「{stageTitle || '当前阶段'}」）。
          </p>
        </div>
        <div className="cc-seq-counts">
          <span>{poseSets.length} 套姿势集</span>
          <span>{renders.length} 套成片</span>
        </div>
      </div>

      <div className="cc-seq-tabs">
        <button type="button" className={tab === 'studio' ? 'is-on' : ''}
          onClick={() => switchTab('studio')}>① 3D 人偶 · 抽帧</button>
        <button type="button" className={tab === 'batch' ? 'is-on' : ''}
          onClick={() => switchTab('batch')}>② 参考图批量帧</button>
        <span className="spacer" />
        <span className="cc-seq-hint">
          {tab === 'studio'
            ? '摆好姿势 / 导入动作 → 抽帧 → 「存为姿势集」'
            : '选姿势集 + 立绘 → 全部生成 → 逐帧挑候选'}
        </span>
      </div>

      {err && <div className="chars-err">⚠ {err}</div>}

      {/* ================= ① 3D 人偶（iframe 常驻，别卸载） ================= */}
      <div className={`cc-seq-pane${tab === 'studio' ? '' : ' is-hidden'}`}>
        {portraits.length === 0 && (
          <p className="cc-tip">还没有立绘：批量帧要以本阶段的立绘当身份来源，先去「角色设定」出一张。</p>
        )}
        <PoseStudio
          pose={studio.pose}
          yaw={studio.yaw}
          onPose={(pose) => setStudio((s) => ({ ...s, pose }))}
          onYaw={(yaw) => setStudio((s) => ({ ...s, yaw }))}
          scene={studio.scene ?? null}
          onScene={(scene) => setStudio((s) => ({ ...s, scene }))}
          onExport={(blob) => void exportMannequin(blob)}
          exporting={exporting}
          onImportImage={(f) => void importPoseFromImage(f)}
          importing={importing}
          importPoints={poseImport?.points ?? null}
          importReprojected={poseImport?.reprojected}
          poses={poses}
          onSavePose={(name, pose, yaw, preview) =>
            void savePoseToLibrary(name, pose, yaw, preview)}
          onSavePoseSet={savePoseSet}
          onPreview={(url, name) => setPreview({ url, name })}
          projectId={character}
        />
        {mannequins.length > 0 && (
          <div className="cc-seq-mannequins">
            <b>本阶段的人偶图（{mannequins.length}）</b>
            <div className="cc-seq-thumbs">
              {mannequins.map((m) => (
                <button key={m.name} type="button"
                  onClick={() => setPreview({ url: m.url, name: m.name })}
                  title={`${m.name}（点击放大）`}>
                  <img src={m.url} alt={m.name} loading="lazy" />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ================= ② 参考图批量帧 ================= */}
      <div className={`cc-seq-pane${tab === 'batch' ? '' : ' is-hidden'}`}>
        {portraits.length === 0 ? (
          <p className="cc-tip">缺角色图：先在「角色设定」出立绘，再回来批量生成。</p>
        ) : (
          <>
            <div className="cc-seq-cfg">
              <label>角色图
                <select value={charPath} onChange={(e) => setCharPath(e.target.value)}
                  title="用本阶段的哪张立绘当身份来源">
                  {portraits.map((p) => (
                    <option key={p.path} value={p.path}>
                      {p.name}{p.costume && p.costume !== 'Naked' ? ` · ${p.costume}` : ''}
                    </option>
                  ))}
                </select>
              </label>
              <label className="cc-seq-prompt">提示词
                <input value={node.params.prompt}
                  placeholder="可选：补充外观描述（身份由角色图决定）"
                  onChange={(e) => patchNode({
                    params: { ...node.params, prompt: e.target.value },
                  })} />
              </label>
              <label>引擎
                <select value={node.params.pcEngine ?? 'wvnccs'}
                  onChange={(e) => patchNode({
                    params: { ...node.params, pcEngine: e.target.value as 'wpc' | 'wvnccs' },
                  })}>
                  <option value="wvnccs">VNCCS（跟住姿势）</option>
                  <option value="wpc">旧版 wpc（快，容易照抄站姿）</option>
                </select>
              </label>
              <label>底色
                <select value={node.params.bgTone}
                  onChange={(e) => patchNode({
                    params: { ...node.params, bgTone: e.target.value as typeof node.params.bgTone },
                  })}>
                  {TONE_OPTIONS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
                </select>
              </label>
              <label className="cc-seq-check">
                <input type="checkbox" checked={!!node.params.matte}
                  onChange={(e) => patchNode({
                    params: { ...node.params, matte: e.target.checked },
                  })} />
                抠图
              </label>
              <label>像素块
                <input type="number" min={0} max={64} value={node.params.pixel}
                  title="0 = 不像素化；6 ~ 8 是精灵帧常用值"
                  onChange={(e) => patchNode({
                    params: {
                      ...node.params,
                      pixel: Math.max(0, Math.min(64, Number(e.target.value) || 0)),
                    },
                  })} />
              </label>
            </div>

            {renders.length > 0 && (
              <div className="cc-seq-history">
                <b>本阶段的成片</b>
                {renders.map((r) => (
                  <button key={r.name} type="button"
                    className={node.poseRender === r.name ? 'is-on' : ''}
                    onClick={() => patchNode({
                    poseRender: r.name,
                    params: { ...node.params, pcPoseSet: r.poseSet, pcOut: r.name },
                  })}
                    title={`打开「${r.name}」继续挑帧 / 补生成`}>
                    {r.cover && <img src={r.cover} alt="" loading="lazy" />}
                    <span>{r.name}</span>
                    <em>{r.done}/{r.total}</em>
                  </button>
                ))}
              </div>
            )}

            {node.job && (node.job.status === 'queued' || node.job.status === 'running') && (
              <div className="cc-seq-progress">
                <b>生成中 {node.job.progress?.done ?? 0}/{node.job.progress?.total ?? '?'}</b>
                <span>{node.job.progress?.current || '排队中…'}</span>
                <div className="cc-seq-bar">
                  <i style={{
                    width: `${Math.min(100, Math.max(0,
                      ((node.job.progress?.done ?? 0)
                        / Math.max(1, node.job.progress?.total ?? 1)) * 100))}%`,
                  }} />
                </div>
              </div>
            )}
            {seedReady && (
            <PoseBatchPanel
              key={`${character}:${stage}:${setsVersion}`}
              character={character}
              stage={stage}
              scopeLabel="本阶段"
              node={node}
              onPatch={patchNode}
              matteBackend={matteBackend}
              charPaths={charPaths}
              hasChar={charPaths.length > 0}
              onPreview={(url, name) => setPreview({ url, name })}
            />
            )}
          </>
        )}
      </div>

      {preview && (
        <Lightbox url={preview.url} name={preview.name}
          onClose={() => setPreview(null)} />
      )}
    </div>
  )
}
