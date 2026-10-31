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
import { useCallback, useEffect, useRef, useState } from 'react'
import type { PosePreset } from '../types'
import { BUILTIN_POSES, uniquePoseName } from '../store/poses'

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
  /** 姿势库（App 持有，落在服务端磁盘） */
  poses: PosePreset[]
  /** 外部套过来的姿势（带 tick，变化才应用） */
  poseApply?: { pose: unknown; yaw: number; tick: number } | null
  /** 把当前姿势存进库（缩略图一并提交） */
  onSavePose?: (name: string, pose: unknown, yaw: number, preview: string) => void
  /** 打开资料库的姿势页 */
  onOpenPoseLibrary?: () => void
}

export default function PoseStudio({
  pose, yaw, onPose, onYaw, onExport, exporting,
  onImportImage, importing, importPoints, poses,
  poseApply, onSavePose, onOpenPoseLibrary,
}: Props) {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [status, setStatus] = useState('正在加载人偶模型（37MB，首次约十几秒）…')
  const [ready, setReady] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [err, setErr] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveName, setSaveName] = useState('')
  const [libSel, setLibSel] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  // 导出/取姿势没有回执就没法收尾：超时要明确告诉用户，别让按钮一直转
  const exportTimer = useRef<number | null>(null)
  // 子窗一直在推消息，回调要用最新的；放 ref 避免 effect 反复重建
  const onPoseRef = useRef(onPose)
  onPoseRef.current = onPose
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
      } else if (d.type === 'error') {
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
    post({ type: 'joints', points: importPoints })
  }, [ready, importPoints, post])

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

  const doExport = useCallback(() => {
    if (!ready) return
    if (exportTimer.current) window.clearTimeout(exportTimer.current)
    exportTimer.current = window.setTimeout(() => {
      exportTimer.current = null
      setErr('导出超时：人偶渲染器没回图，看看它是不是还在加载')
    }, EXPORT_TIMEOUT_MS)
    post({ type: 'export', yaw: yawRef.current, size: 768 })
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
          title={'这是**导出机位**：0° 正面 / 90° 右侧面 / 180° 背面 / 270° 左侧面。\n'
            + '画布里的视角（右键拖动）只是用来看的，不影响导出。'}
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

      <div className="pose-actions">
        <button type="button" className="mini" onClick={() => post({ type: 'resetView' })}
          disabled={!ready} title="把观察视角拉回正面（不改姿势）">视角</button>
        <button type="button" className="mini" onClick={() => post({ type: 'resetPose' })}
          disabled={!ready} title="回到初始站姿">站立</button>
        <button
          type="button" className="mini"
          onClick={() => fileRef.current?.click()}
          disabled={!ready || importing || !onImportImage}
          title="从一张真人/写实人像自动提姿势（走官方 OpenPose → 骨骼转换器）"
        >
          {importing ? '提取中…' : '从图片'}
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
        拖橙色关节标记摆姿势（官方两骨 IK + 肘膝极点约束）；右键拖动转视角、滚轮缩放。
        <b>导出的是白底灰 3D 人偶</b>，把它连到「角色迁移」的<b>动作参考</b>输入。
        摆好的姿势点 <b>存入</b> 就进资料库（存在磁盘 library/ 下），
        别的节点和项目都能原样复用。
      </div>
    </div>
  )
}
