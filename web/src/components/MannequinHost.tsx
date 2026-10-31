/**
 * 人偶渲染桥接：把官方 VNCCS PoseStudio（web/public/vnccs-pose/）当作
 * 独立 iframe 载入，用 postMessage 驱动它渲染。
 *
 * 为什么走 iframe 而不是直接 import：
 * - 官方 core.js 靠 `import.meta.url` 相对路径动态加载 vendor/three 与网格资产，
 *   放在 public/ 下由浏览器原样 serve，路径才成立；
 * - PoseStudio 自带 WebGL 上下文与 rAF 循环，塞进 React 树会跟主画布抢上下文；
 * - 隔离后官方代码的更新不会污染我们的构建。
 *
 * 协议（entry.mjs 侧）：
 *   父 → 子：{ type:'render', id, joints, azimuth, opts }
 *   子 → 父：{ type:'ready' } | { type:'result', id, dataURL } | { type:'error', message }
 */
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'

/** 18 关节（编辑器 0~1 坐标，y 向下），顺序同 COCO-18 */
export type MannequinJoints = { x: number; y: number }[]

export type MannequinHandle = {
  /** 网格加载完成，可以渲染了 */
  ready: boolean
  /** 加载/渲染错误 */
  error: string
  /** 渲染人偶 → PNG Blob */
  render: (
    joints: MannequinJoints,
    azimuthDeg: number,
    opts?: { width?: number; height?: number },
  ) => Promise<Blob>
}

const SRC = '/vnccs-pose/studio.html'
const RENDER_TIMEOUT = 60_000

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

export const MannequinHost = forwardRef<MannequinHandle>(function MannequinHost(_props, ref) {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const pending = useRef(new Map<string, { resolve: (v: string) => void; reject: (e: Error) => void }>())
  // pending 里存的是 dataURL 的 resolve；对外承诺的是 Blob（见 render 的类型）
  const readyWaiters = useRef<Array<() => void>>([])

  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      const d = e.data as { type?: string; id?: string; dataURL?: string; message?: string } | null
      if (!d || typeof d.type !== 'string') return
      if (d.type === 'ready') {
        setReady(true)
        setError('')
        readyWaiters.current.splice(0).forEach((fn) => fn())
        return
      }
      if (d.type === 'error') {
        setError(String(d.message || '人偶渲染器出错'))
        // 失败也要唤醒等待者，否则点按钮会一直转
        readyWaiters.current.splice(0).forEach((fn) => fn())
        return
      }
      if (d.type === 'result' && d.id) {
        const p = pending.current.get(d.id)
        if (p) {
          pending.current.delete(d.id)
          p.resolve(String(d.dataURL || ''))
        }
      }
    }
    window.addEventListener('message', onMsg)
    return () => {
      window.removeEventListener('message', onMsg)
      // 卸载时把还在等的人叫醒，别留悬挂的 promise
      pending.current.forEach((p) => p.reject(new Error('人偶渲染器已卸载')))
      pending.current.clear()
    }
  }, [])

  const render = useCallback(
    (joints: MannequinJoints, azimuthDeg: number, opts?: { width?: number; height?: number }) => {
      const win = iframeRef.current?.contentWindow
      if (!win) return Promise.reject(new Error('人偶渲染器未挂载'))
      // 等 ready（第一次要等 37.8MB 网格下载+解析）
      const waitReady = ready
        ? Promise.resolve()
        : new Promise<void>((resolve) => {
            readyWaiters.current.push(resolve)
            setTimeout(resolve, RENDER_TIMEOUT)
          })
      return waitReady.then(() => {
        const id = `m${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`
        return new Promise<Blob>((resolve, reject) => {
          pending.current.set(id, {
            resolve: (url) => {
              try { resolve(dataUrlToBlob(url)) } catch (e) { reject(e as Error) }
            },
            reject,
          })
          win.postMessage({ type: 'render', id, joints, azimuth: azimuthDeg, opts }, '*')
          setTimeout(() => {
            if (pending.current.has(id)) {
              pending.current.delete(id)
              reject(new Error('人偶渲染超时（60 秒）'))
            }
          }, RENDER_TIMEOUT)
        })
      })
    },
    [ready],
  )

  useImperativeHandle(ref, () => ({ ready, error, render }), [ready, error, render])

  // 隐藏但保持布局（display:none 的 iframe 在部分浏览器里不会跑 rAF）
  return (
    <iframe
      ref={iframeRef}
      src={SRC}
      title="VNCCS PoseStudio 人偶渲染器"
      aria-hidden
      style={{
        position: 'fixed', left: -10000, top: 0, width: 64, height: 64,
        border: 0, opacity: 0, pointerEvents: 'none',
      }}
    />
  )
})
