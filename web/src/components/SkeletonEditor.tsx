/**
 * 2D 骨骼编辑器：拖关节摆姿势 + 绕竖轴旋转换视觉方向。
 *
 * 为什么是「伪 3D」：只有 2D 坐标的话，「侧面 / 背面」只能靠左右镜像硬凑，
 * 肩膀和髋部一转就叠成一个点，模型读不懂。这里给每个关节存一个 z（前后深度），
 * 渲染时绕竖轴旋转再正交投影：
 *
 *     rx = 0.5 + (x - 0.5) * cos(a) + z * sin(a)
 *     ry = y
 *
 * 于是 0° 是正面、90° 是侧面、180° 是背面，中间角度自然过渡。
 * 拖动时把屏幕横向位移按 (cos a, sin a) 分解回 x/z —— 侧面视角下横向拖
 * 动的语义自动变成「改深度」，符合三维直觉。
 *
 * 导出的 PNG 严格复刻 tools/make_pose.render 的 OpenPose-18 画法
 * （18 色、19 条边、黑底、彩色肢体 + 关节圆点），因为 Qwen/ControlNet
 * 认的是这个格式，不是随便画的白线骨架。
 */
import { useCallback, useRef, useState } from 'react'
import type { Joint3, SkeletonState } from '../types'
export type { Joint3, SkeletonState }

/** COCO-18 关节顺序（与 tools/make_pose.py 完全一致，别改） */
export const JOINT_NAMES = [
  '鼻', '颈', '右肩', '右肘', '右腕', '左肩', '左肘', '左腕',
  '右髋', '右膝', '右踝', '左髋', '左膝', '左踝',
  '右眼', '左眼', '右耳', '左耳',
]

/** 官方 18 色（BODY_25 之前的 COCO-18 版本），ControlNet 训练时约定的编码 */
export const COLORS = [
  [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
  [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
  [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],
]

/** 标准 19 条边，颜色取边的序号（不是端点序号） */
export const LIMBS = [
  [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
  [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
  [2, 16], [5, 17],
]

type Props = {
  state: SkeletonState
  onChange: (s: SkeletonState) => void
  /** 导出 PNG（Blob，不含文件名）；由节点负责上传 */
  onExport: (blob: Blob) => void
  exporting?: boolean
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const rad = (deg: number) => (deg * Math.PI) / 180

/** 投影：3D 关节 → 2D 屏幕（SVG 用户坐标 0~100） */
export function project(p: Joint3, azimuth: number) {
  const a = rad(azimuth)
  const x = 0.5 + (p.x - 0.5) * Math.cos(a) + p.z * Math.sin(a)
  return { x: x * 100, y: p.y * 100 }
}

/** 预设姿势。z 是「正面站立」时的前后深度：躯干 0，右肩/右髋略前，左侧略后。 */
function pose(pts: Array<[number, number, number?]>): Joint3[] {
  return pts.map(([x, y, z]) => ({ x, y, z: z ?? 0 }))
}

export const PRESETS: Record<string, Joint3[]> = {
  站立: pose([
    [0.50, 0.10], [0.50, 0.20],
    [0.42, 0.22, -0.05], [0.39, 0.36, -0.05], [0.38, 0.50, -0.05],
    [0.58, 0.22, 0.05], [0.61, 0.36, 0.05], [0.62, 0.50, 0.05],
    [0.46, 0.52, -0.04], [0.45, 0.68, -0.04], [0.44, 0.86, -0.04],
    [0.54, 0.52, 0.04], [0.55, 0.68, 0.04], [0.56, 0.86, 0.04],
    [0.47, 0.09], [0.53, 0.09], [0.45, 0.10], [0.55, 0.10],
  ]),
  张手: pose([
    [0.50, 0.10], [0.50, 0.20],
    [0.42, 0.22, -0.04], [0.32, 0.24, -0.08], [0.20, 0.20, -0.14],
    [0.58, 0.22, 0.04], [0.68, 0.24, 0.08], [0.80, 0.20, 0.14],
    [0.46, 0.52, -0.04], [0.45, 0.68, -0.04], [0.44, 0.86, -0.04],
    [0.54, 0.52, 0.04], [0.55, 0.68, 0.04], [0.56, 0.86, 0.04],
    [0.47, 0.09], [0.53, 0.09], [0.45, 0.10], [0.55, 0.10],
  ]),
  行走: pose([
    [0.50, 0.10], [0.50, 0.20],
    [0.42, 0.22, -0.05], [0.38, 0.35, -0.05], [0.42, 0.48, -0.05],
    [0.58, 0.22, 0.05], [0.62, 0.34, 0.05], [0.58, 0.46, 0.05],
    [0.46, 0.52, -0.04], [0.40, 0.68, -0.04], [0.36, 0.86, -0.04],
    [0.54, 0.52, 0.04], [0.60, 0.68, 0.04], [0.64, 0.86, 0.04],
    [0.47, 0.09], [0.53, 0.09], [0.45, 0.10], [0.55, 0.10],
  ]),
  跑步: pose([
    [0.50, 0.10], [0.50, 0.20],
    [0.43, 0.23, -0.05], [0.34, 0.30, -0.05], [0.42, 0.36, -0.05],
    [0.57, 0.23, 0.05], [0.66, 0.29, 0.05], [0.58, 0.38, 0.05],
    [0.46, 0.52, -0.04], [0.36, 0.66, -0.04], [0.30, 0.80, -0.04],
    [0.54, 0.52, 0.04], [0.62, 0.64, 0.04], [0.70, 0.76, 0.04],
    [0.47, 0.09], [0.53, 0.09], [0.45, 0.10], [0.55, 0.10],
  ]),
  跳跃: pose([
    [0.50, 0.10], [0.50, 0.20],
    [0.42, 0.22, -0.05], [0.30, 0.14, -0.05], [0.22, 0.06, -0.05],
    [0.58, 0.22, 0.05], [0.70, 0.14, 0.05], [0.78, 0.06, 0.05],
    [0.46, 0.52, -0.04], [0.38, 0.66, -0.04], [0.34, 0.80, -0.04],
    [0.54, 0.52, 0.04], [0.62, 0.66, 0.04], [0.66, 0.80, 0.04],
    [0.47, 0.09], [0.53, 0.09], [0.45, 0.10], [0.55, 0.10],
  ]),
  拳击: pose([
    [0.50, 0.10], [0.50, 0.20],
    [0.42, 0.22, -0.05], [0.34, 0.22, -0.05], [0.24, 0.16, -0.05],
    [0.58, 0.22, 0.05], [0.66, 0.26, 0.05], [0.60, 0.14, 0.05],
    [0.46, 0.52, -0.04], [0.42, 0.68, -0.04], [0.40, 0.86, -0.04],
    [0.54, 0.52, 0.04], [0.58, 0.68, 0.04], [0.60, 0.86, 0.04],
    [0.47, 0.09], [0.53, 0.09], [0.45, 0.10], [0.55, 0.10],
  ]),
  举手: pose([
    [0.50, 0.10], [0.50, 0.20],
    [0.42, 0.22, -0.05], [0.38, 0.10, -0.05], [0.36, 0.02, -0.05],
    [0.58, 0.22, 0.05], [0.62, 0.10, 0.05], [0.64, 0.02, 0.05],
    [0.46, 0.52, -0.04], [0.45, 0.68, -0.04], [0.44, 0.86, -0.04],
    [0.54, 0.52, 0.04], [0.55, 0.68, 0.04], [0.56, 0.86, 0.04],
    [0.47, 0.09], [0.53, 0.09], [0.45, 0.10], [0.55, 0.10],
  ]),
}

export function defaultSkeleton(): SkeletonState {
  return { points: PRESETS['张手'].map((p) => ({ ...p })), azimuth: 0, preset: '张手' }
}

/** 8 个方向：精灵素材常用的方位。名字里的「右」指人物自己的右侧。 */
export const DIRECTIONS: Array<[string, number]> = [
  ['正面', 0], ['右前', 45], ['右侧', 90], ['右后', 135],
  ['背面', 180], ['左后', 225], ['左侧', 270], ['左前', 315],
]

const rgb = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`

/** 把当前骨架画成 OpenPose-18 PNG。和后端 tools/make_pose.render 对齐。 */
export function renderSkeletonPNG(state: SkeletonState, size = 768): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const cv = document.createElement('canvas')
    cv.width = cv.height = size
    const g = cv.getContext('2d')
    if (!g) { reject(new Error('拿不到 2d context')); return }
    g.fillStyle = '#000'
    g.fillRect(0, 0, size, size)
    const pts = state.points.map((p) => project(p, state.azimuth))
    const stick = Math.max(4, Math.round(size / 64))
    const dot = Math.max(3, Math.round(size / 96))
    g.lineCap = 'round'
    g.lineJoin = 'round'
    LIMBS.forEach(([a, b], i) => {
      if (!pts[a] || !pts[b]) return
      g.strokeStyle = rgb(COLORS[i % COLORS.length])
      g.lineWidth = stick
      g.beginPath()
      g.moveTo((pts[a].x / 100) * size, (pts[a].y / 100) * size)
      g.lineTo((pts[b].x / 100) * size, (pts[b].y / 100) * size)
      g.stroke()
    })
    pts.forEach((p, k) => {
      g.fillStyle = rgb(COLORS[k % COLORS.length])
      g.beginPath()
      g.arc((p.x / 100) * size, (p.y / 100) * size, dot, 0, Math.PI * 2)
      g.fill()
    })
    cv.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('toBlob 失败'))), 'image/png')
  })
}

export default function SkeletonEditor({ state, onChange, onExport, exporting }: Props) {
  const [dragIdx, setDragIdx] = useState<number | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const dragRef = useRef<{ i: number; sx: number; sy: number; ox: number; oy: number; oz: number } | null>(null)

  const pts = state.points.map((p) => project(p, state.azimuth))
  const a = rad(state.azimuth)

  const pointFromEvent = (e: React.PointerEvent) => {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const r = svg.getBoundingClientRect()
    return { x: ((e.clientX - r.left) / r.width) * 100, y: ((e.clientY - r.top) / r.height) * 100 }
  }

  const startDragJoint = (i: number, e: React.PointerEvent) => {
    e.stopPropagation()
    e.preventDefault()
    const pt = pointFromEvent(e)
    const j = state.points[i]
    dragRef.current = { i, sx: pt.x, sy: pt.y, ox: j.x, oy: j.y, oz: j.z }
    setDragIdx(i)
    setSelected(i)
    try { (e.currentTarget as Element).setPointerCapture(e.pointerId) } catch { /* 某些环境不支持 */ }
  }

  const moveDragJoint = (e: React.PointerEvent) => {
    const d = dragRef.current
    if (!d) return
    const pt = pointFromEvent(e)
    const dx = (pt.x - d.sx) / 100
    const dy = (pt.y - d.sy) / 100
    // 屏幕横向位移分解回 3D 的水平轴 (cos a, 0, sin a)：
    // 正面时改 x，侧面时自动变成改 z（深度），符合三维直觉
    const next = state.points.map((p, k) => {
      if (k !== d.i) return p
      return {
        x: clamp(d.ox + dx * Math.cos(a), 0, 1),
        y: clamp(d.oy + dy, 0, 1),
        z: clamp(d.oz + dx * Math.sin(a), -0.35, 0.35),
      }
    })
    onChange({ ...state, points: next, preset: '自定义' })
  }

  const endDragJoint = (e: React.PointerEvent) => {
    if (!dragRef.current) return
    dragRef.current = null
    setDragIdx(null)
    try { (e.currentTarget as Element).releasePointerCapture(e.pointerId) } catch { /* 已释放 */ }
  }

  const setPreset = (name: string) => {
    onChange({ ...state, points: PRESETS[name].map((p) => ({ ...p })), preset: name })
  }
  const setAzimuth = (deg: number) => onChange({ ...state, azimuth: deg })
  const setDepth = (v: number) => {
    if (selected === null) return
    const next = state.points.map((p, k) => (k === selected ? { ...p, z: v } : p))
    onChange({ ...state, points: next, preset: '自定义' })
  }

  const doExport = useCallback(async () => {
    const blob = await renderSkeletonPNG(state)
    onExport(blob)
  }, [state, onExport])

  return (
    <div className="skel-editor" onPointerDown={(e) => e.stopPropagation()}>
      <div className="skel-row">
        <label>
          姿势
          <select value={state.preset in PRESETS ? state.preset : '自定义'}
            onChange={(e) => e.target.value in PRESETS && setPreset(e.target.value)}>
            {Object.keys(PRESETS).map((k) => <option key={k} value={k}>{k}</option>)}
            {!(state.preset in PRESETS) && <option value="自定义">自定义</option>}
          </select>
        </label>
        <span className="skel-size">OpenPose-18</span>
      </div>

      <div className="skel-dirs">
        {DIRECTIONS.map(([name, deg]) => (
          <button
            key={name}
            className={`mini skel-dir${state.azimuth === deg ? ' on' : ''}`}
            onClick={() => setAzimuth(deg)}
            title={`${name}（${deg}°）`}
          >
            {name}
          </button>
        ))}
      </div>
      <label className="skel-range">
        方位 <b>{state.azimuth}°</b>
        <input type="range" min={0} max={359} step={1} value={state.azimuth}
          onChange={(e) => setAzimuth(Number(e.target.value))} />
      </label>

      <svg
        ref={svgRef}
        className="skel-canvas"
        viewBox="0 0 100 100"
        onPointerMove={moveDragJoint}
        onPointerUp={endDragJoint}
        onPointerCancel={endDragJoint}
      >
        <rect x="0" y="0" width="100" height="100" fill="#08080b" />
        {/* 参考线：脚底线与中轴，方便对齐 */}
        <line x1="0" y1="86" x2="100" y2="86" stroke="#1c1c24" strokeWidth="0.4" />
        <line x1="50" y1="0" x2="50" y2="100" stroke="#1c1c24" strokeWidth="0.4" />
        {LIMBS.map(([p, q], i) => (
          pts[p] && pts[q] && (
            <line key={`l${i}`} x1={pts[p].x} y1={pts[p].y} x2={pts[q].x} y2={pts[q].y}
              stroke={rgb(COLORS[i % COLORS.length])} strokeWidth="1.8"
              strokeLinecap="round" opacity={0.95} />
          )
        ))}
        {pts.map((p, k) => (
          <g key={`j${k}`} onPointerDown={(e) => startDragJoint(k, e)}>
            <circle cx={p.x} cy={p.y} r={selected === k ? 2.6 : 1.9}
              fill={rgb(COLORS[k % COLORS.length])}
              stroke={selected === k ? '#fff' : 'none'} strokeWidth="0.5" />
            {/* 透明大热区：11px 的圆点在画布缩放后很难点中 */}
            <circle cx={p.x} cy={p.y} r="4.5" fill="transparent" style={{ cursor: 'grab' }} />
          </g>
        ))}
        {dragIdx !== null && (
          <text x="50" y="97" fill="#7a7a88" fontSize="3" textAnchor="middle">
            {JOINT_NAMES[dragIdx]}
          </text>
        )}
      </svg>

      <label className="skel-range">
        深度 {selected === null ? '（先点一个关节）' : JOINT_NAMES[selected]}
        <input type="range" min={-0.35} max={0.35} step={0.01}
          value={selected === null ? 0 : state.points[selected].z}
          disabled={selected === null}
          onChange={(e) => setDepth(Number(e.target.value))} />
      </label>

      <div className="skel-actions">
        <button className="mini" onClick={() => onChange(defaultSkeleton())}>重置</button>
        <button className="mini wide" onClick={() => void doExport()} disabled={exporting}>
          {exporting ? '导出中…' : '导出骨架图'}
        </button>
      </div>
      <div className="src-note">
        拖圆点摆关节；方位角绕竖轴旋转（0° 正面 / 180° 背面）。导出的骨架图可直接连到角色迁移节点。
      </div>
    </div>
  )
}
