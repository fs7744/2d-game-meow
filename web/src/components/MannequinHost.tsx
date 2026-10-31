/**
 * 3D 人偶编辑器宿主：把官方 VNCCS PoseStudio 当 iframe 打开，做成弹窗。
 *
 * 为什么是 iframe 而不是直接 import：
 * - 官方 core.js 靠 `import.meta.url` 相对路径动态加载 vendor/three 与网格资产，
 *   放在 public/ 下由浏览器原样 serve，路径才成立；
 * - PoseStudio 自带 WebGL 上下文与 rAF 循环，塞进 React 树会跟主画布抢上下文；
 * - 隔离后官方代码的更新不会污染我们的构建。
 *
 * 交互与官方 UniCanvas 的 pose layer 一致：**编辑会话**（弹窗）→ 摆姿势 → 导出。
 * 编辑器里拖的是官方给的橙色关节点（每个骨一个），走官方 IK。
 *
 * 协议（entry.js 侧）：
 *   父 → 子：{ type:'init', joints, azimuth }
 *   子 → 父：{ type:'ready' } | { type:'progress', text }
 *           | { type:'result', dataURL } | { type:'error', message }
 */
import { useCallback, useEffect, useRef, useState } from 'react'

/** 18 关节（编辑器 0~1 坐标，y 向下），顺序同 COCO-18 */
export type MannequinJoints = { x: number; y: number }[]

export type MannequinHandle = {
  /** 打开编辑会话；用户导出后回调 onResult（PNG Blob） */
  open: (opts: {
    joints?: MannequinJoints
    azimuth?: number
    onResult: (blob: Blob) => void
  }) => void
  close: () => void
  /** 是否正在编辑 */
  active: boolean
}

const SRC = '/vnccs-pose/studio.html'
/** 网格 37MB + 解析 + 渲染，给足时间（首次可能要十几秒） */
const READY_TIMEOUT = 180_000

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

export function MannequinHost({
  open,
  onClose,
  onResult,
  joints,
  azimuth,
}: {
  open: boolean
  onClose: () => void
  onResult: (blob: Blob) => void
  joints?: MannequinJoints
  azimuth?: number
}) {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [status, setStatus] = useState('')

  const post = useCallback((msg: Record<string, unknown>) => {
    iframeRef.current?.contentWindow?.postMessage(msg, '*')
  }, [])

  useEffect(() => {
    if (!open) return
    const onMsg = (e: MessageEvent) => {
      const d = e.data as { type?: string; dataURL?: string; message?: string; text?: string } | null
      if (!d || typeof d.type !== 'string') return
      if (d.type === 'ready') {
        setStatus('就绪：拖橙色关节点摆姿势，右键拖动转视角')
        post({ type: 'init', joints, azimuth: azimuth ?? 0 })
      } else if (d.type === 'progress') {
        setStatus(String(d.text || '加载中…'))
      } else if (d.type === 'error') {
        setStatus(`出错：${d.message || '未知错误'}`)
      } else if (d.type === 'result') {
        try {
          onResult(dataUrlToBlob(String(d.dataURL || '')))
        } catch (err) {
          setStatus(`导出结果解析失败：${(err as Error).message}`)
        }
      }
    }
    window.addEventListener('message', onMsg)
    return () => window.removeEventListener('message', onMsg)
  }, [open, joints, azimuth, post, onResult])

  // 打开时如果一直没 ready（脚本没加载 / 网格拉不到），给出明确提示而不是无限等
  useEffect(() => {
    if (!open) return
    setStatus('正在加载人偶模型（37MB，首次约十几秒）…')
    const t = window.setTimeout(() => {
      setStatus((s) => (s && s.startsWith('就绪')) ? s
        : '等待人偶渲染器响应…（若一直如此，多半是脚本没加载成功）')
    }, READY_TIMEOUT)
    return () => window.clearTimeout(t)
  }, [open])

  if (!open) return null

  return (
    <div
      className="modal-mask"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="modal mannequin-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span>3D 人偶编辑器（官方 PoseStudio）</span>
          <button onClick={onClose} title="关闭">×</button>
        </div>
        <div className="mannequin-frame">
          <iframe
            ref={iframeRef}
            src={SRC}
            title="VNCCS PoseStudio 3D 人偶编辑器"
          />
        </div>
        <div className="modal-actions">
          <span className="mannequin-status">{status}</span>
          <button onClick={onClose}>完成</button>
        </div>
      </div>
    </div>
  )
}
