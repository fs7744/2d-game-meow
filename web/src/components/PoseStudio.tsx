/**
 * 3D 人偶编辑器（poseedit 节点的正文）—— 官方 VNCCS PoseStudio 的内嵌壳。
 *
 * 为什么是 iframe 而不是直接 import：
 * - 官方 core.js 靠 `import.meta.url` 相对路径动态加载 vendor/three 与网格资产，
 *   放在 public/ 下由浏览器原样 serve，路径才成立；
 * - PoseStudio 自带 WebGL 上下文与 rAF 循环，塞进 React 树会跟主画布抢上下文；
 * - 隔离后官方代码的更新不会污染我们的构建。
 *
 * **它替代了原先自写的 2D 伪 3D 骨架编辑器**：官方那边有 MakeHuman 网格、53 根骨、
 * 两骨 IK 与极点约束，摆出来的是真三维姿势；导出的白底灰人偶正是 VNCCS 姿态
 * LoRA 的训练分布（详见 docs/VNCCS-REAL.md），比线框骨架贴合得多。
 *
 * 存档用官方自己的 `getPose()` JSON（骨骼欧拉角 + IK effector + 极点），
 * 重开节点能精确还原。姿势库存的也是同一份 JSON（在服务端磁盘），
 * 所以「复用动作」拿到的是分毫不差的同一个姿势。
 *
 * 协议见 web/public/vnccs-pose/entry.js 头注释。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { PosePreset } from '../types'
import { BUILTIN_POSES, uniquePoseName } from '../store/poses'
import {
  applyFilters, diffStats, hashDiff, loopSeam, type SampledFrame,
} from '../store/animFrames'

/** 8 个精灵素材常用的朝向（导出机位，绕 Y 的角度） */
export const YAWS: Array<[number, string]> = [
  [0, '正面'], [45, '右前'], [90, '右侧'], [135, '右后'],
  [180, '背面'], [225, '左后'], [270, '左侧'], [315, '左前'],
]

/** 嵌入模式：只留画布，控制条由宿主提供 */
const SRC = '/vnccs-pose/studio.html?embed=1'
/** 网格 37MB + morph 解算 + 首帧渲染，给足时间（首次可能要十几秒） */
const LOAD_HINT_MS = 8_000
const EXPORT_TIMEOUT_MS = 60_000
/** 问子窗要当前姿势 / 缩略图的等待上限 */
const ASK_TIMEOUT_MS = 5_000

/** dataURL → Blob。用 atob 而不是 fetch(data:)：jsdom 不支持 fetch data URL。 */
function dataUrlToBlob(dataURL: string): Blob {
  const comma = dataURL.indexOf(',')
  const head = comma > 0 ? dataURL.slice(0, comma) : ''
  const body = comma > 0 ? dataURL.slice(comma + 1) : ''
  const mime = /:(.*?);/.exec(head)?.[1] || 'image/png'
  const bin = atob(body)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Blob([bytes], { type: mime })
}

type Props = {
  /** 官方 getPose() 的 JSON；重开节点时用它还原姿势 */
  pose?: unknown
  /** 导出机位（度）：0 正面 / 90 右侧 / 180 背面 / 270 左侧 */
  yaw: number
  onPose: (pose: unknown) => void
  onYaw: (yaw: number) => void
  /** 导出白底灰人偶 PNG */
  onExport: (blob: Blob) => void
  exporting?: boolean
  /** 从一张真人图自动提姿势（上传 + DWPose 由 App 层负责） */
  onImportImage?: (file: File) => void
  importing?: boolean
  /**
   * App 层提好的 2D 关节点（0~1）。每次「从图片」都换一个新引用，
   * 变化时推给子窗走官方 convertOpenPoseToPose。
   */
  importPoints?: (number[] | null)[] | null
  /** 这组点是 3D 转正后重投影的（后端 MediaPipe 给的），见 applyPoints 的注释 */
  importReprojected?: boolean
  /** 姿势库（App 持有，落在服务端磁盘） */
  poses: PosePreset[]
  /** 外部套过来的姿势（带 tick，变化才应用） */
  poseApply?: { pose: unknown; yaw: number; tick: number } | null
  /** 把当前姿势存进库（缩略图一并提交） */
  onSavePose?: (name: string, pose: unknown, yaw: number, preview: string) => void
  /** 打开资料库的姿势页 */
  onOpenPoseLibrary?: () => void
  /** 当前项目 id：抽出来的帧要存进**这个项目**的 `poses/<名字>/` 下 */
  projectId?: string
  /** 双击帧墙里的某一帧 → 全屏放大（灯箱在 App 那边） */
  onPreview?: (url: string, name: string) => void
  /** 把一个动作抽出来的 N 帧存成项目里的姿势集（同名会被后端拒绝） */
  onSavePoseSet?: (name: string, frames: PoseSetFrameIn[]) => Promise<boolean> | boolean
}

/** 抽帧结果的一帧（`png` 是 iframe 出的人偶图 dataURL） */
export type PoseSetFrameIn = {
  t: number
  yaw: number
  pose: unknown
  png: string
}

export default function PoseStudio({
  pose, yaw, onPose, onYaw, onExport, exporting,
  onImportImage, importing, importPoints, importReprojected, poses,
  poseApply, onSavePose, onOpenPoseLibrary, projectId, onSavePoseSet, onPreview,
}: Props) {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [status, setStatus] = useState('正在加载人偶模型（37MB，首次约十几秒）…')
  const [ready, setReady] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [err, setErr] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveName, setSaveName] = useState('')
  const [libSel, setLibSel] = useState('')
  /**
   * 导入的动作文件（glb/gltf/fbx）。`clips` 是文件里的动画轨道，
   * 一个 glb 可能同时带 Idle/Walk/Run（three 的 Soldier.glb 就是 3 条）。
   */
  const [anim, setAnim] = useState<{
    clips: { name: string; duration: number }[]
    index: number
    /** 骨架识别结果（家族名 + 认出几根），给用户一个"这份文件是哪种骨架"的交代 */
    rig?: { familyLabel: string; mapped: number }
  } | null>(null)
  const [animT, setAnimT] = useState(0)
  const [animBusy, setAnimBusy] = useState(false)
  /** 抽帧存姿势集：帧数（用户自己填）/ 名字 / 进行中 */
  const [frameCount, setFrameCount] = useState(8)
  const [setName, setSetName] = useState('')
  const [sampling, setSampling] = useState(false)
  const [savingSet, setSavingSet] = useState(false)
  /**
   * **抽出来但还没存的帧**（照 pixelforge「拆解帧 · 挑帧」的做法）：
   * 先把整条动作抽成一摞缩略图给用户挑，而不是"点了就闷头存盘"。
   * 关掉视图/切节点就丢 —— 它是过渡态，不写进 project.json（十几张
   * 100KB 的 base64 会让自动保存变慢）。
   */
  const [sampled, setSampled] = useState<SampledFrame[] | null>(null)
  /** 已勾选的帧（按 sampled 的**下标**，不是点击顺序 —— 存出去要帧序） */
  const [picks, setPicks] = useState<number[]>([])
  /** 两把筛子：跳帧步长 + 近似度（只改"显示哪些"，不动勾选） */
  const [step, setStep] = useState(1)
  const [simPct, setSimPct] = useState(0)
  const [showAll, setShowAll] = useState(false)
  /** 正在播放（iframe 那边在推进时间轴） */
  const [playing, setPlaying] = useState(false)
  const animFileRef = useRef<HTMLInputElement>(null)
  /** 拖滑块时的节流器：每 60ms 才真采样一次，否则一帧一次 WebGL 解算会卡 */
  const animTimer = useRef<number | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  // 导出/取姿势没有回执就没法收尾：超时要明确告诉用户，别让按钮一直转
  const exportTimer = useRef<number | null>(null)
  // 子窗一直在推消息，回调要用最新的；放 ref 避免 effect 反复重建
  const onPoseRef = useRef(onPose)
  onPoseRef.current = onPose
  const onSavePoseSetRef = useRef(onSavePoseSet)
  onSavePoseSetRef.current = onSavePoseSet
  const yawRef = useRef(yaw)
  yawRef.current = yaw
  const poseRef = useRef<unknown>(pose)
  const lastPointsRef = useRef<unknown>(null)
  const lastApplyRef = useRef<number>(0)
  const askPoseRef = useRef<((p: unknown) => void) | null>(null)
  const askPreviewRef = useRef<((d: string) => void) | null>(null)

  const post = useCallback((msg: Record<string, unknown>) => {
    iframeRef.current?.contentWindow?.postMessage(msg, '*')
  }, [])

  /** 向子窗要当前姿势（存库要最新的，不能拿过期引用） */
  const askPose = useCallback(() => new Promise<unknown>((resolve) => {
    askPoseRef.current = resolve
    post({ type: 'getPose' })
    window.setTimeout(() => {
      if (!askPoseRef.current) return
      askPoseRef.current = null
      resolve(poseRef.current)
    }, ASK_TIMEOUT_MS)
  }), [post])

  /** 向子窗要一张小缩略图（存姿势库用） */
  const askPreview = useCallback(() => new Promise<string>((resolve) => {
    askPreviewRef.current = resolve
    post({ type: 'preview', size: 256, yaw: yawRef.current })
    window.setTimeout(() => {
      if (!askPreviewRef.current) return
      askPreviewRef.current = null
      resolve('')
    }, ASK_TIMEOUT_MS)
  }), [post])

  // ---- 子窗消息 ----
  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      // 同页可能有别的人在 postMessage（别的 iframe、devtools），只认自己那个子窗
      if (!iframeRef.current || e.source !== iframeRef.current.contentWindow) return
      const d = e.data as {
        type?: string; dataURL?: string; message?: string
        text?: string; pose?: unknown; yaw?: number
        clips?: { name: string; duration: number }[]
        boneCount?: number; format?: string; switched?: boolean
        rig?: { familyLabel?: string; mapped?: number; missing?: string[] }
        frames?: { t: number; pose: unknown; png?: string }[]
        clip?: string
        t?: number
        on?: boolean
      } | null
      if (!d || typeof d.type !== 'string') return
      if (d.type === 'ready') {
        setReady(true)
        setErr('')
        setStatus('就绪：拖橙色关节标记摆姿势，右键拖动转视角')
      } else if (d.type === 'progress') {
        setStatus(String(d.text || '加载中…'))
      } else if (d.type === 'preview') {
        const cb = askPreviewRef.current
        askPreviewRef.current = null
        cb?.(String(d.dataURL || ''))
      } else if (d.type === 'pose') {
        // 拖动松手后官方才派发（syncMode:'end'），一次编辑一条，正好当存档点
        poseRef.current = d.pose
        const cb = askPoseRef.current
        askPoseRef.current = null
        cb?.(d.pose)
        onPoseRef.current(d.pose)
      } else if (d.type === 'animLoaded') {
        setAnimBusy(false)
        const clips = d.clips || []
        if (!clips.length) return
        const rig = d.rig as { familyLabel?: string; mapped?: number } | undefined
        setAnim({
          clips,
          index: 0,
          rig: rig ? { familyLabel: String(rig.familyLabel || ''), mapped: Number(rig.mapped) || 0 } : undefined,
        })
        setAnimT(0)
        setStatus(`动作已载入：${clips[0].name}（${d.boneCount} 根骨，${d.format}`
          + `${rig?.familyLabel ? `，${rig.familyLabel} 骨架` : ''}）——拖滑块选帧`)
        post({ type: 'animTime', t: 0 })
      } else if (d.type === 'animTick') {
        // 播放中每 ~80ms 推一次时间，滑块跟着走（播放时不再推 pose，见 entry.js）
        if (typeof d.t === 'number') setAnimT(d.t)
      } else if (d.type === 'animPlaying') {
        setPlaying(!!d.on)
      } else if (d.type === 'animFrames') {
        // 抽帧回来了：**先摆出来给用户挑**，点「存为姿势集」才落盘
        setSampling(false)
        const frames = (d.frames || []) as SampledFrame[]
        if (!frames.length) {
          setErr('抽帧没拿到内容')
          return
        }
        setSampled(frames)
        // 默认全选：用户的预期是"我抽了 12 帧就存 12 帧"，
        // 筛子是帮他去重的工具，不是默认就要他重挑一遍
        setPicks(frames.map((_, i) => i))
        setStep(1)
        setSimPct(0)
        setShowAll(false)
        setStatus(`抽到 ${frames.length} 帧（还没存）：挑完点「存为姿势集」`)
      } else if (d.type === 'error') {
        setAnimBusy(false)
        setSampling(false)
        setPlaying(false)
        setStatus('')
        setErr(String(d.message || '未知错误'))
      } else if (d.type === 'result') {
        if (exportTimer.current) {
          window.clearTimeout(exportTimer.current)
          exportTimer.current = null
        }
        try {
          onExport(dataUrlToBlob(String(d.dataURL || '')))
        } catch (er) {
          setErr(`导出结果解析失败：${(er as Error).message}`)
        }
      }
    }
    window.addEventListener('message', onMsg)
    return () => window.removeEventListener('message', onMsg)
  }, [onExport])

  // ---- 就绪后把存档的姿势与朝向灌回去 ----
  // 只在 ready 由假变真（或 iframe 重载）时做一次：之后 pose 由子窗单向推过来，
  // 回头再 setPose 会和正在进行的拖动打架。
  useEffect(() => {
    if (!ready) return
    post({ type: 'init', pose: poseRef.current ?? null, yaw: yawRef.current })
  }, [ready, post])

  // ---- 「从图片」提好的 2D 点 → 官方转换器（真 3D 骨骼） ----
  useEffect(() => {
    if (!ready || !importPoints) return
    if (lastPointsRef.current === importPoints) return
    lastPointsRef.current = importPoints
    post({ type: 'joints', points: importPoints, reprojected: !!importReprojected })
  }, [ready, importPoints, importReprojected, post])

  // ---- 姿势库套过来的（外部请求） ----
  useEffect(() => {
    if (!ready || !poseApply) return
    if (lastApplyRef.current === poseApply.tick) return
    lastApplyRef.current = poseApply.tick
    post({ type: 'setPose', pose: poseApply.pose })
    const n = ((Math.round(poseApply.yaw) % 360) + 360) % 360
    onYaw(n)
    post({ type: 'setYaw', yaw: n })
  }, [ready, poseApply, post, onYaw])

  // ---- 加载兜底：脚本没起来时给明确提示，而不是让用户盯着黑框 ----
  useEffect(() => {
    if (ready || err) return
    const t = window.setTimeout(() => {
      setStatus((s) => (s.startsWith('就绪') ? s : '还在加载人偶模型…第一次较慢'))
    }, LOAD_HINT_MS)
    return () => window.clearTimeout(t)
  }, [ready, err])

  useEffect(() => () => {
    if (exportTimer.current) window.clearTimeout(exportTimer.current)
  }, [])

  /** 读动作文件 → 字节发给子窗解析（子窗拿不到 File 对象，只能传 Byte） */
  const loadAnimFile = useCallback((f: File) => {
    setErr('')
    setAnimBusy(true)
    if (!iframeRef.current?.contentWindow) {
      setAnimBusy(false)
      setErr('编辑器还没起来，等一下再试')
      return
    }
    f.arrayBuffer().then((buf) => {
      iframeRef.current?.contentWindow?.postMessage(
        { type: 'animLoad', buffer: buf, name: f.name }, '*', [buf])
    }).catch((e: Error) => {
      setAnimBusy(false)
      setErr(`读文件失败：${e.message}`)
    })
  }, [])

  /** 拖滑块选帧。节流 60ms：每帧都要跑一次关键点提取 + 全骨 IK，不解节流会卡 */
  const scrubAnim = useCallback((t: number) => {
    setAnimT(t)
    if (animTimer.current) window.clearTimeout(animTimer.current)
    animTimer.current = window.setTimeout(() => {
      animTimer.current = null
      post({ type: 'animTime', t })
    }, 60)
  }, [post])

  const doExport = useCallback(() => {
    if (!ready) return
    if (exportTimer.current) window.clearTimeout(exportTimer.current)
    exportTimer.current = window.setTimeout(() => {
      exportTimer.current = null
      setErr('导出超时：人偶渲染器没回图，看看它是不是还在加载')
    }, EXPORT_TIMEOUT_MS)
    post({ type: 'export', size: 768 })   // 导出当前视角（所见即所得），不传 yaw
  }, [ready, post])

  const setYaw = useCallback((deg: number) => {
    const n = ((Math.round(deg) % 360) + 360) % 360
    onYaw(n)
    post({ type: 'setYaw', yaw: n })
  }, [onYaw, post])

  // ---- 姿势库 ----
  const applyBuiltin = useCallback((name: string) => {
    const hit = BUILTIN_POSES.find((p) => p.name === name)
    if (hit) post({ type: 'joints', points: hit.points })
  }, [post])

  const applyPreset = useCallback((id: string) => {
    const hit = poses.find((p) => p.id === id)
    if (!hit) return
    post({ type: 'setPose', pose: hit.pose })
    setYaw(hit.yaw)
  }, [poses, post, setYaw])

  const onLibPick = useCallback((v: string) => {
    if (!v) return
    if (v.startsWith('b:')) applyBuiltin(v.slice(2))
    else if (v.startsWith('u:')) applyPreset(v.slice(2))
  }, [applyBuiltin, applyPreset])

  const startSave = useCallback(async () => {
    if (!ready) return
    const p = await askPose()
    if (!p) {
      setErr('还没读到姿势：先在人偶上拖一下，或等模型加载完')
      return
    }
    poseRef.current = p
    onPoseRef.current(p)
    setSaveName(uniquePoseName(poses, '姿势'))
    setSaving(true)
  }, [ready, askPose, poses])

  const doSave = useCallback(async () => {
    const p = poseRef.current
    if (!p || !onSavePose) return
    // 缩略图和 pose 一并提交：浏览面板靠它认人。
    // 拿不到缩略图也要把姿势存下来（缩略图只是锦上添花）。
    const preview = await askPreview().catch(() => '')
    onSavePose(saveName.trim() || '姿势', p, yawRef.current, preview)
    setSaving(false)
  }, [askPreview, onSavePose, saveName])

  /* ---- 挑帧：显示什么由筛子决定，存什么由勾选决定 ---- */
  const allIdx = useMemo(
    () => (sampled ? sampled.map((_, i) => i) : []),
    [sampled],
  )
  const filtered = useMemo(
    () => (sampled ? applyFilters(sampled, { step, simPct }) : []),
    [sampled, step, simPct],
  )
  /** 筛选结果对应的下标（"应用筛选"要的是下标，不是帧本体） */
  const filteredIdx = useMemo(() => {
    if (!sampled) return []
    const keep = new Set(filtered)
    return sampled.map((f, i) => (keep.has(f) ? i : -1)).filter((i) => i >= 0)
  }, [sampled, filtered])
  /** 帧墙上要画的：带下标 + 与前一张**显示帧**的差异（tooltip 用） */
  const shown = useMemo(() => {
    const list = showAll ? (sampled || []) : filtered
    return list.map((frame) => {
      const i = sampled ? sampled.indexOf(frame) : 0
      const prev = i > 0 && sampled ? sampled[i - 1] : null
      return { frame, i, diff: prev ? hashDiff(prev.hash, frame.hash) : 0 }
    })
  }, [sampled, filtered, showAll])
  /** 勾选的帧，**按帧序**（存出去要稳定顺序，不能按点击顺序） */
  const pickedFrames = useMemo(
    () => (sampled ? sampled.map((frame, i) => ({ frame, i })).filter((x) => picks.includes(x.i)) : []),
    [sampled, picks],
  )
  const stats = useMemo(() => diffStats(pickedFrames.map((p) => p.frame)), [pickedFrames])
  const seam = useMemo(() => loopSeam(pickedFrames.map((p) => p.frame)), [pickedFrames])

  return (
    <div className="pose-editor" onPointerDown={(e) => e.stopPropagation()}>
      <div className={`pose-stage${expanded ? ' expanded' : ''}`}>
        <iframe
          ref={iframeRef}
          src={SRC}
          title="VNCCS PoseStudio 3D 人偶编辑器"
        />
        {expanded && (
          <div className="pose-expanded-bar">
            <span>3D 人偶编辑器 · 官方 VNCCS PoseStudio</span>
            <button type="button" onClick={() => setExpanded(false)}>收起</button>
          </div>
        )}
      </div>

      {err && <div className="pose-err" title={err}>⚠ {err}</div>}
      {!ready && !err && <div className="pose-status">{status}</div>}

      <div className="pose-lib">
        <select
          className="pose-lib-sel"
          value={libSel}
          onChange={(e) => { setLibSel(e.target.value); onLibPick(e.target.value) }}
          disabled={!ready}
          title="姿势库：内置几个开箱即用，下面是你自己存的（存在磁盘，跨项目可复用）"
        >
          <option value="">姿势…</option>
          <optgroup label="内置">
            {BUILTIN_POSES.map((p) => (
              <option key={p.name} value={`b:${p.name}`}>{p.name}</option>
            ))}
          </optgroup>
          {poses.length > 0 && (
            <optgroup label={`我的（${poses.length}）`}>
              {poses.map((p) => (
                <option key={p.id} value={`u:${p.id}`}>{p.name}</option>
              ))}
            </optgroup>
          )}
        </select>
        <button
          type="button" className="mini"
          onClick={() => onOpenPoseLibrary?.()}
          disabled={!onOpenPoseLibrary}
          title="打开资料库：带预览图的姿势总览，还能改名 / 删除 / 搜索"
        >
          库
        </button>
        <button
          type="button" className="mini"
          onClick={() => void startSave()}
          disabled={!ready || saving}
          title="把当前姿势存进姿势库（存的是官方 pose JSON，套用时分毫不差）"
        >
          {saving ? '…' : '存入'}
        </button>
      </div>

      {/* 动作文件（glb/gltf/fbx）：**比「从图片提姿势」可靠**——动画里的 3D 是
          真的，不用从 2D 猜深度。载入后拖滑块逐帧选，停在哪帧人偶就是哪帧。 */}
      <div className="pose-lib pose-anim">
        <button
          type="button" className="mini"
          onClick={() => animFileRef.current?.click()}
          disabled={!ready || animBusy}
          title={'导入 .glb / .gltf / .fbx 动画。Mixamo、Quaternius/KayKit、Unreal、\n'
            + 'VRM 命名的骨架都能认（按骨名 + 层级自动对到人偶骨上）。\n'
            + '动画自带真实 3D，比「从图片提姿势」准得多：不会有"人偶躺倒"这类深度歧义。'}
        >
          {animBusy ? '读取中…' : '导入动作'}
        </button>
        {anim && anim.clips.length > 1 && (
          <select
            className="pose-lib-sel"
            value={anim.index}
            onChange={(e) => {
              const i = Number(e.target.value)
              setAnim({ ...anim, index: i })
              setAnimT(0)
              post({ type: 'animClip', index: i })
            }}
            title="这个文件里有多个动作（Idle / Walk / Run…）"
          >
            {anim.clips.map((c, i) => (
              <option key={i} value={i}>{c.name}</option>
            ))}
          </select>
        )}
        {anim && (
          <>
            <button
              type="button" className="mini"
              onClick={() => {
                setErr('')
                post({ type: 'animPlay', on: !playing, t: playing ? 0 : animT })
              }}
              disabled={!ready || animBusy}
              title={playing ? '停止播放（回到滑块那一帧）'
                : '从头播到尾（按真实时长，慢的时候自动跳帧，不会越播越拖）'}
            >
              {playing ? '■' : '▶'}
            </button>
            <input
              className="pose-anim-range"
              type="range"
              min={0}
              max={Math.max(0.02, anim.clips[anim.index]?.duration ?? 1)}
              step={0.01}
              value={Math.min(animT, Math.max(0.02, anim.clips[anim.index]?.duration ?? 1))}
              onChange={(e) => scrubAnim(Number(e.target.value))}
              disabled={playing}
              title="拖到想要的那一帧停住（人偶实时跟帧）"
            />
            <b className="pose-anim-t">{animT.toFixed(2)}s</b>
          </>
        )}
      </div>

      {/* ---- 抽帧 → 挑帧 → 存姿势集 ----
          交互照 pixelforge 的「拆解帧 · 挑帧」：**先抽出来看**，再用两把筛子
          （跳帧步长 / 近似度）压掉重复姿势，勾选是显式状态 —— 拖滑杆只改
          "显示哪些"，不会把你已经挑好的帧悄悄换掉，点「应用筛选」才采纳。 */}
      {anim && (
        <>
          <div className="pose-lib pose-frames">
            <label className="pose-frames-n" title="沿这条动作等分抽几帧（1~240）">
              抽
              <input
                type="number"
                min={1}
                max={240}
                value={frameCount}
                disabled={sampling}
                onChange={(e) => {
                  const n = Math.max(1, Math.min(240, Math.round(Number(e.target.value) || 8)))
                  setFrameCount(n)
                }}
              />
              帧
            </label>
            <button
              type="button" className="pose-batch-btn"
              disabled={!ready || sampling}
              onClick={() => {
                setErr('')
                setSampling(true)
                post({ type: 'animSample', count: frameCount })
              }}
              title={'按当前机位把这条动作等分成 N 帧，抽出来**先给你看**，'
                + '挑完再存（不点存就不会落盘）'}
            >
              {sampling ? '抽帧中…' : (sampled ? '重新抽帧' : '抽帧看看')}
            </button>
            {anim.rig && (
              <span className="pose-frames-rig" title="骨骼是按名字 + 层级认出来的">
                {anim.rig.familyLabel}
              </span>
            )}
          </div>

          {sampled && (
            <>
              {/* 两把筛子：即时生效，但只改「显示哪些」 */}
              <div className="anim-filters">
                <label className="anim-slider">
                  <span>跳帧步长 <b>每 {step} 帧取 1</b></span>
                  <input
                    type="range" min={1} max={10} step={1} value={step}
                    onChange={(e) => setStep(Number(e.target.value))}
                  />
                </label>
                <label className="anim-slider">
                  <span>
                    近似度过滤 <b>{simPct.toFixed(0)}%</b>
                    <em title="与上一张保留帧的差异低于这个值就当作重复帧">低于此视为重复</em>
                  </span>
                  {/* 上限 40：差异按「改动格数 / 两帧墨迹并集」算，走路的相邻帧
                      正常在 10~30%，留到 40 才够用（0 = 不过滤） */}
                  <input
                    type="range" min={0} max={40} step={1} value={simPct}
                    onChange={(e) => setSimPct(Number(e.target.value))}
                  />
                </label>
              </div>

              <div className="anim-toolbar">
                <button
                  type="button" className="pose-batch-btn"
                  onClick={() => setPicks(filteredIdx)}
                  title="把当前筛选结果设为已选（已选的会被替换，但可以随时点格子改回来）"
                >应用筛选（{filtered.length}）</button>
                <button
                  type="button" className="pose-batch-btn"
                  onClick={() => setPicks(allIdx.filter((i) => !picks.includes(i)))}
                >反选</button>
                <button
                  type="button" className="pose-batch-btn"
                  onClick={() => setPicks([])}
                >清空</button>
                <label className="anim-showall" title="显示被筛掉的帧（灰显），方便手动捞回来">
                  <input
                    type="checkbox" checked={showAll}
                    onChange={(e) => setShowAll(e.target.checked)}
                  />
                  显示全部
                </label>
              </div>

              <div className="anim-count">
                已选 <b>{pickedFrames.length}</b> / 共 {sampled.length} 帧
                {pickedFrames.length >= 2 && (
                  <>
                    <span className="anim-stat" title="相邻已选帧的差异中位数：太低说明还有重复姿势">
                      间距中位 {stats.median.toFixed(1)}%
                    </span>
                    <span className={`anim-stat${seam < 1.5 ? ' warn' : ''}`}
                      title="末帧与首帧的差异：太小的话循环播放会卡一下（末帧≈首帧）">
                      首尾差 {seam.toFixed(1)}%
                    </span>
                    {stats.nearDup > 0 && (
                      <span className="anim-stat warn"
                        title="相邻两帧几乎一样，连着播会停一拍">
                        近重复 {stats.nearDup} 对
                      </span>
                    )}
                  </>
                )}
              </div>

              {/* 帧墙：点一下勾选/取消，双击放大看细节 */}
              <div className="anim-wall">
                {shown.map((f) => {
                  const on = picks.includes(f.i)
                  return (
                    <button
                      key={f.i}
                      type="button"
                      className={`anim-cell${on ? ' on' : ''}`}
                      title={`第 ${f.i + 1} 帧 · ${f.frame.t.toFixed(2)}s · 与前帧差 ${f.diff.toFixed(1)}%`
                        + `\n点击${on ? '取消' : '选中'}；双击放大`}
                      onClick={() => setPicks((cur) => (
                        cur.includes(f.i) ? cur.filter((x) => x !== f.i) : [...cur, f.i].sort((a, b) => a - b)
                      ))}
                      onDoubleClick={() => onPreview?.(f.frame.png, `第 ${f.i + 1} 帧`)}
                    >
                      <img src={f.frame.png} alt={`第 ${f.i + 1} 帧`} draggable={false} />
                      <span className="anim-idx">{f.i + 1}</span>
                      {on && (
                        <span className="anim-order">
                          {pickedFrames.findIndex((x) => x.i === f.i) + 1}
                        </span>
                      )}
                    </button>
                  )
                })}
                {!shown.length && (
                  <div className="src-note" style={{ padding: 6 }}>
                    当前筛选把帧全滤掉了：把「近似度过滤」调低，或勾「显示全部」手动挑。
                  </div>
                )}
              </div>

              <div className="pose-lib pose-frames" style={{ marginTop: 4 }}>
                <input
                  className="pose-set-name"
                  value={setName}
                  placeholder={anim.clips[anim.index]?.name
                    ? `姿势集名（默认 ${anim.clips[anim.index].name}）` : '姿势集名'}
                  disabled={savingSet}
                  onChange={(e) => setSetName(e.target.value)}
                  title="存到本项目的 poses/<这个名字>/ 下。**同名会被拒绝**（不会覆盖上一次的）"
                />
                <button
                  type="button" className="pose-batch-btn"
                  disabled={!projectId || !onSavePoseSet || savingSet || !pickedFrames.length}
                  onClick={() => {
                    setErr('')
                    const name = (setName.trim() || anim.clips[anim.index]?.name || '动作').trim()
                    if (!projectId) { setErr('没有项目 id，存不了（先保存项目）'); return }
                    const items: PoseSetFrameIn[] = pickedFrames.map((p) => ({
                      t: p.frame.t, yaw: yawRef.current, pose: p.frame.pose, png: p.frame.png,
                    }))
                    setSavingSet(true)
                    Promise.resolve(onSavePoseSetRef.current?.(name, items))
                      .then((ok) => {
                        if (ok) {
                          setStatus(`已存成姿势集「${name}」（${items.length} 帧）`)
                          setSampled(null)
                          setPicks([])
                        }
                      })
                      .catch(() => { /* App 已经提示过错误，这里不重复报 */ })
                      .finally(() => setSavingSet(false))
                  }}
                  title={pickedFrames.length
                    ? `只存勾选的这 ${pickedFrames.length} 帧`
                    : '还没勾任何帧'}
                >
                  {savingSet ? '保存中…' : `存为姿势集（${pickedFrames.length}）`}
                </button>
              </div>
            </>
          )}
        </>
      )}

      <input
        ref={animFileRef}
        type="file"
        accept=".glb,.gltf,.fbx"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) loadAnimFile(f)
          e.target.value = ''
        }}
      />

      {saving && (
        <div className="pose-save">
          <input
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="姿势名"
            autoFocus
            onKeyDown={(e) => { if (e.key === 'Enter') void doSave() }}
          />
          <button type="button" className="mini" onClick={() => void doSave()}>保存</button>
          <button type="button" className="mini" onClick={() => setSaving(false)}>×</button>
        </div>
      )}

      <label className="pose-row">
        朝向
        <select
          value={YAWS.some(([d]) => d === yaw) ? String(yaw) : ''}
          onChange={(e) => setYaw(Number(e.target.value))}
          disabled={!ready}
          title={'快捷对齐：点一下把相机摆到标准角度（0° 正面 / 90° 右侧 / 180° 背面 / 270° 左侧）。\n'
            + '导出永远以编辑器当前视角为准——右键拖动转到的角度就是导出的角度。'}
        >
          {!YAWS.some(([d]) => d === yaw) && <option value="">{yaw}°</option>}
          {YAWS.map(([d, label]) => (
            <option key={d} value={d}>{d}° {label}</option>
          ))}
        </select>
        <button
          type="button" className="mini"
          onClick={() => setExpanded((v) => !v)}
          title="放大到整屏摆姿势（拖关节更准）；画面按新尺寸重新渲染"
        >
          {expanded ? '收起' : '放大'}
        </button>
      </label>

      <label className="pose-row pose-yaw-slider">
        机位
        <input
          type="range"
          min={0}
          max={359}
          step={1}
          value={yaw}
          disabled={!ready}
          onChange={(e) => setYaw(Number(e.target.value))}
        />
        <b className="pose-yaw-val">{yaw}°</b>
      </label>

      <div className="pose-cam-hint-row">
        导出用编辑器当前视角（右键拖动换角度）；上面的朝向/机位只是快速对齐。
      </div>

      <div className="pose-actions">
        <button type="button" className="mini" onClick={() => post({ type: 'resetView' })}
          disabled={!ready} title="把观察视角拉回正面（不改姿势）">重置视角</button>
        <button type="button" className="mini" onClick={() => post({ type: 'resetPose' })}
          disabled={!ready} title="回到初始站姿（清掉摆好的关节）">重置姿势</button>
        <button
          type="button" className="mini"
          onClick={() => fileRef.current?.click()}
          disabled={!ready || importing || !onImportImage}
          title="从一张真人/写实人像自动提姿势（走官方 OpenPose → 骨骼转换器）"
        >
          {importing ? '提取中…' : '从图片提姿势'}
        </button>
        <button
          type="button" className="mini wide"
          onClick={doExport}
          disabled={!ready || exporting}
          title={'导出白底灰 3D 人偶 PNG（768²）——VNCCS 姿态 LoRA 的训练分布，\n'
            + '给角色迁移当动作参考'}
        >
          {exporting ? '导出中…' : '导出人偶图'}
        </button>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onImportImage?.(f)
          e.target.value = ''        // 允许连续选同一张图
        }}
      />
      <div className="src-note">
        拖橙色关节标记摆姿势（官方两骨 IK + 肘膝极点约束），滚轮缩放看细节。
        <b>导出的是白底灰 3D 人偶</b>，把它连到「角色迁移」的<b>动作参考</b>输入。
        摆好的姿势点 <b>存入</b> 就进资料库（存在磁盘 library/ 下），
        别的节点和项目都能原样复用。要现成的整套动作（idle / walk / run），
        用 <b>导入动作</b> 读 Mixamo 的 glb/fbx（CC0 的 Quaternius / KayKit 也认），
        拖滑块选帧，或者直接 <b>存入 N 帧</b> 一次灌进姿势库。
      </div>
    </div>
  )
}
