/**
 * 全屏预览（灯箱）+ 缩放平移。
 *
 * 卡片里的图只是缩略图，看不清像素细节；点一下就全屏，能放大到单个像素。
 *
 * **缩放的语义按真实像素算**（`zoom` = 屏幕像素 / 图像像素）：
 * 100% 就是 1:1，看得到模型真正画了什么；初始值是「适应窗口」，通常对
 * 1024² 的图是 70~90%。业界图片查看器都是这套（Photoshop / 系统预览 /
 * Figma），用户不用猜百分比是什么意思。
 *
 * 为什么不给 `<img>` 挂 CSS `max-width` 了：一旦要缩放，浏览器得知道
 * 这张图**实际占多少像素**，`max-width` 撑出来的尺寸和 zoom 对不上，
 * 缩放锚点就会漂。现在宽高直接写成 `natural × zoom`，光标的锚点算得准。
 *
 * 渲染方式跟着缩放走（本项目的规矩，见 styles.css 的注释）：
 *   - `zoom >= 1`（放大或 1:1）→ `pixelated`，像素画的方块才不会被插值抹圆；
 *   - `zoom < 1`（缩小看整图）→ `auto`，双线性缩小比最近邻清楚得多。
 *   以前这里恒开 `pixelated`，1024 的图缩到 800 宽显示全是锯齿。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

type Props = {
  url: string
  name?: string
  onClose: () => void
}

/** 放大上限：1024 的图放到 16 倍已经很够看单像素了 */
const ZOOM_MAX = 16
/** 一倍点按的倍率（按钮 / 键盘） */
const STEP = 1.25

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

export default function Lightbox({ url, name, onClose }: Props) {
  const stageRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [nat, setNat] = useState({ w: 0, h: 0 })
  const [fit, setFit] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [pos, setPos] = useState({ x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)

  /**
   * 视图状态放在 ref 里做「唯一真相」，state 只用来触发重渲染。
   *
   * 缩放时 `zoom / tx / ty` 必须一起算（锚点公式里三个都用到），而
   * `setState` 是异步的、读回渲染时的闭包值会拿到上一帧——滚轮连发时
   * 就会抖动、光标下的点会漂。ref 同步可读，事件里直接拿最新值。
   */
  const view = useRef({ zoom: 1, tx: 0, ty: 0, fit: 1, nw: 0, nh: 0 })

  /** 把 ref 里的最新视图落到 state（这是唯一写 state 的地方） */
  const commit = useCallback((next: Partial<{ zoom: number; tx: number; ty: number }>) => {
    const stage = stageRef.current
    const v = { ...view.current, ...next }
    v.zoom = clamp(v.zoom, v.fit, ZOOM_MAX)
    // 平移不能把图拖出可视区：最多拖到图边贴住窗口边
    const cw = stage?.clientWidth ?? 0
    const ch = stage?.clientHeight ?? 0
    const maxX = Math.max(0, (v.nw * v.zoom - cw) / 2)
    const maxY = Math.max(0, (v.nh * v.zoom - ch) / 2)
    v.tx = clamp(v.tx, -maxX, maxX)
    v.ty = clamp(v.ty, -maxY, maxY)
    view.current = v
    setZoom(v.zoom)
    setPos({ x: v.tx, y: v.ty })
  }, [])

  /** 重新量「适应窗口」的比例。窗口 resize、图加载完都要调一次。 */
  const measure = useCallback(() => {
    const stage = stageRef.current
    const img = imgRef.current
    if (!stage || !img || !img.naturalWidth || !img.naturalHeight) return
    const cw = stage.clientWidth
    const ch = stage.clientHeight
    const f = Math.min(1, cw / img.naturalWidth, ch / img.naturalHeight) || 1
    view.current.fit = f
    view.current.nw = img.naturalWidth
    view.current.nh = img.naturalHeight
    setFit(f)
    setNat({ w: img.naturalWidth, h: img.naturalHeight })
    return f
  }, [])

  const onImgLoad = useCallback(() => {
    const f = measure()
    if (f) {
      view.current.zoom = f
      view.current.tx = 0
      view.current.ty = 0
      setZoom(f)
      setPos({ x: 0, y: 0 })
    }
  }, [measure])

  /**
   * 缩放到 `next`，并让「光标下的那个点」留在原地。
   *
   * 屏幕坐标以舞台中心为原点。图像上的点 p = (m - t) / zoom；
   * 缩放后要它还在 m：t' = m - p·zoom'。不给锚点就以中心为准（m=0）。
   */
  const zoomTo = useCallback((next: number, clientX?: number, clientY?: number) => {
    const stage = stageRef.current
    if (!stage) return
    const v = view.current
    const z = clamp(next, v.fit, ZOOM_MAX)
    if (Math.abs(z - v.zoom) < 1e-6) return
    const box = stage.getBoundingClientRect()
    const mx = clientX === undefined ? 0 : clientX - box.left - box.width / 2
    const my = clientY === undefined ? 0 : clientY - box.top - box.height / 2
    const px = (mx - v.tx) / v.zoom
    const py = (my - v.ty) / v.zoom
    commit({ zoom: z, tx: mx - px * z, ty: my - py * z })
  }, [commit])

  const zoomBy = useCallback((factor: number) => {
    zoomTo(view.current.zoom * factor)
  }, [zoomTo])

  // 滚轮缩放：**必须自己绑 addEventListener**，React 的 onWheel 是被动监听，
  // 里面 preventDefault 不生效（页面会跟着滚，缩放看着像卡住）。
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      // exp 让"滚同样的距离"在大小倍率下手感一致；0.0015 是调过的（一格 ≈ 1.15 倍）
      const factor = Math.exp(-e.deltaY * 0.0015)
      zoomTo(view.current.zoom * factor, e.clientX, e.clientY)
    }
    stage.addEventListener('wheel', onWheel, { passive: false })
    return () => stage.removeEventListener('wheel', onWheel)
  }, [zoomTo])

  // 键盘：+/- 缩放、0 适应、1 一比一、Esc 关闭
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      const typing = !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA'
        || el.isContentEditable)
      if (typing) return
      if (e.key === 'Escape') { onClose(); return }
      if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomBy(STEP) }
      else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomBy(1 / STEP) }
      else if (e.key === '0') { e.preventDefault(); zoomTo(view.current.fit) }
      else if (e.key === '1') { e.preventDefault(); zoomTo(1) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, zoomBy, zoomTo])

  // 窗口变化时「适应窗口」的比例要重算，否则从大窗口缩到小窗口后，
  // 图会溢出、平移范围也算错。
  useEffect(() => {
    const stage = stageRef.current
    if (!stage || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => {
      const wasFit = Math.abs(view.current.zoom - view.current.fit) < 0.01
      const f = measure()
      if (!f) return
      // 原本就是"适应窗口"的话跟着新尺寸走；已经手动放大过的别乱动它
      if (wasFit) commit({ zoom: f, tx: 0, ty: 0 })
      else commit({})
    })
    ro.observe(stage)
    return () => ro.disconnect()
  }, [measure, commit])

  /** 拖拽平移 */
  const drag = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null)

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    drag.current = { x: e.clientX, y: e.clientY, tx: view.current.tx, ty: view.current.ty, moved: false }
    setDragging(true)
    // jsdom / 老浏览器不一定实现了指针捕获，失败也不该让拖拽整段挂掉
    try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* 没有就算了 */ }
  }

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (!d) return
    const dx = e.clientX - d.x
    const dy = e.clientY - d.y
    if (!d.moved && Math.hypot(dx, dy) > 3) d.moved = true
    commit({ tx: d.tx + dx, ty: d.ty + dy })
  }

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return
    drag.current = null
    setDragging(false)
    try { e.currentTarget.releasePointerCapture(e.pointerId) } catch { /* 已释放 */ }
  }

  const atFit = zoom <= fit + 1e-6
  // 放大/1:1 用最近邻（像素方块要方），缩小用平滑（否则满屏锯齿）
  const rendering = zoom >= 1 ? 'pixelated' : 'auto'

  // **走 Portal 到 body**：调用点可能在 `.sprite-stage`（overflow:hidden）或
  // 节点卡片（overflow:hidden + transform）里面，就地渲染会被裁掉、跟着缩放。
  return createPortal(
    <div className="lightbox" role="dialog" aria-label="图片预览" onClick={onClose}>
      <div className="lightbox-bar" onClick={(e) => e.stopPropagation()}>
        <span className="lightbox-name">{name || '预览'}</span>
        <span className="spacer" />
        <div className="lightbox-zoomctl" onClick={(e) => e.stopPropagation()}>
          <button
            className="mini"
            onClick={() => zoomBy(1 / STEP)}
            disabled={atFit}
            title="缩小（-）"
          >
            −
          </button>
          <span className="lightbox-pct" title="显示像素 / 图像像素">{Math.round(zoom * 100)}%</span>
          <button
            className="mini"
            onClick={() => zoomBy(STEP)}
            disabled={zoom >= ZOOM_MAX - 1e-6}
            title="放大（+）"
          >
            +
          </button>
          <button
            className="mini wide"
            onClick={() => zoomTo(fit)}
            disabled={atFit}
            title="适应窗口（0）"
          >
            适应
          </button>
          <button
            className="mini wide"
            onClick={() => zoomTo(1)}
            disabled={Math.abs(zoom - 1) < 0.01}
            title="1:1 实际像素（1）—— 看模型真正画了什么"
          >
            1:1
          </button>
        </div>
        <a
          className="mini wide"
          href={`${url}${url.includes('?') ? '&' : '?'}download=1`}
          download={name}
          title="下载原图"
        >
          ⭳ 下载
        </a>
        <button className="mini wide" onClick={onClose} title="关闭 (Esc)">
          ✕ 关闭
        </button>
      </div>

      <div
        ref={stageRef}
        className={`lightbox-stage${dragging ? ' dragging' : ''}`}
        onClick={(e) => {
          // 点空白关掉；点到图上不关（放大后满屏都是图，误触太容易）
          if (e.target === e.currentTarget) onClose()
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={(e) => {
          // 双击：适应窗口 ↔ 4 倍（以双击点为中心），业界查看器都是这个手感
          if (zoom <= fit * 1.01) zoomTo(fit * 4, e.clientX, e.clientY)
          else zoomTo(fit)
        }}
        style={{ touchAction: 'none' }}
      >
        <img
          ref={imgRef}
          className="lightbox-img"
          src={url}
          alt={name || '预览'}
          draggable={false}
          onLoad={onImgLoad}
          style={{
            // 宽高按真实像素算（见文件头注释）；图还没加载完时退回 CSS 撑满
            ...(nat.w ? { width: Math.round(nat.w * zoom), height: Math.round(nat.h * zoom) } : {}),
            maxWidth: nat.w ? 'none' : '100%',
            maxHeight: nat.w ? 'none' : '100%',
            transform: `translate(${pos.x}px, ${pos.y}px)`,
            imageRendering: rendering,
          }}
        />
      </div>

      <div className="lightbox-hint" onClick={(e) => e.stopPropagation()}>
        滚轮缩放 · 拖动平移 · 双击放大 4 倍 · +/− 微调 · Esc 关闭
      </div>
    </div>,
    document.body,
  )
}
