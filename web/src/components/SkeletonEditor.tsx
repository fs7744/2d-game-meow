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
  /** 从用户提供的图片自动提骨架（上传 + DWPose 由节点侧负责） */
  onImportImage?: (file: File) => void
  importing?: boolean
  /**
   * 导出 3D 人偶图（VNCCS 官方条件图格式）。
   *
   * 官方 PoseStudio 用 MakeHuman 网格渲染「白底灰色人偶」当 image1，LoRA 是在
   * 这种图上训的；线框骨架不在它的分布里，姿态跟随会弱很多（见 docs/VNCCS-REAL.md）。
   * 这里的关节是编辑器的 0~1 坐标（x 右 / y 下），方位角决定朝向。
   * 由节点侧调官方渲染器（iframe）完成，可能要等几秒。
   */
  onExportMannequin?: (
    joints: { x: number; y: number }[],
    azimuth: number,
  ) => Promise<void>
  mannequinExporting?: boolean
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
  // **比例按真人重映射过**：旧版是「大头短腿」（头占身高 22%、腿只 34%），
  // 人形看着像吉祥物，导出的骨架比例失真也会让模型的姿态贴合变差。
  // 锚点：头顶 0.02 / 鼻 0.075 / 颈 0.155 / 肩 0.18 / 髋 0.52 /
  //       膝 0.72 / 踝 0.93（8 头身）。
  站立: pose([
    [0.5, 0.075],     [0.5, 0.155],     [0.42, 0.18, -0.05],     [0.39, 0.339, -0.05],
    [0.38, 0.497, -0.05],     [0.58, 0.18, 0.05],     [0.61, 0.339, 0.05],     [0.62, 0.497, 0.05],
    [0.46, 0.52, -0.04],     [0.45, 0.72, -0.04],     [0.44, 0.93, -0.04],     [0.54, 0.52, 0.04],
    [0.55, 0.72, 0.04],     [0.56, 0.93, 0.04],     [0.47, 0.069],     [0.53, 0.069],
    [0.45, 0.075],     [0.55, 0.075],   ]),
  张手: pose([
    [0.5, 0.075],     [0.5, 0.155],     [0.42, 0.18, -0.04],     [0.32, 0.203, -0.08],
    [0.2, 0.155, -0.14],     [0.58, 0.18, 0.04],     [0.68, 0.203, 0.08],     [0.8, 0.155, 0.14],
    [0.46, 0.52, -0.04],     [0.45, 0.72, -0.04],     [0.44, 0.93, -0.04],     [0.54, 0.52, 0.04],
    [0.55, 0.72, 0.04],     [0.56, 0.93, 0.04],     [0.47, 0.069],     [0.53, 0.069],
    [0.45, 0.075],     [0.55, 0.075],   ]),
  行走: pose([
    [0.5, 0.075],     [0.5, 0.155],     [0.42, 0.18, -0.05],     [0.38, 0.327, -0.05],
    [0.42, 0.475, -0.05],     [0.58, 0.18, 0.05],     [0.62, 0.316, 0.05],     [0.58, 0.452, 0.05],
    [0.46, 0.52, -0.04],     [0.4, 0.72, -0.04],     [0.36, 0.93, -0.04],     [0.54, 0.52, 0.04],
    [0.6, 0.72, 0.04],     [0.64, 0.93, 0.04],     [0.47, 0.069],     [0.53, 0.069],
    [0.45, 0.075],     [0.55, 0.075],   ]),
  跑步: pose([
    [0.5, 0.075],     [0.5, 0.155],     [0.43, 0.191, -0.05],     [0.34, 0.271, -0.05],
    [0.42, 0.339, -0.05],     [0.57, 0.191, 0.05],     [0.66, 0.259, 0.05],     [0.58, 0.361, 0.05],
    [0.46, 0.52, -0.04],     [0.36, 0.695, -0.04],     [0.3, 0.86, -0.04],     [0.54, 0.52, 0.04],
    [0.62, 0.67, 0.04],     [0.7, 0.813, 0.04],     [0.47, 0.069],     [0.53, 0.069],
    [0.45, 0.075],     [0.55, 0.075],   ]),
  跳跃: pose([
    [0.5, 0.075],     [0.5, 0.155],     [0.42, 0.18, -0.05],     [0.3, 0.107, -0.05],
    [0.22, 0.053, -0.05],     [0.58, 0.18, 0.05],     [0.7, 0.107, 0.05],     [0.78, 0.053, 0.05],
    [0.46, 0.52, -0.04],     [0.38, 0.695, -0.04],     [0.34, 0.86, -0.04],     [0.54, 0.52, 0.04],
    [0.62, 0.695, 0.04],     [0.66, 0.86, 0.04],     [0.47, 0.069],     [0.53, 0.069],
    [0.45, 0.075],     [0.55, 0.075],   ]),
  拳击: pose([
    [0.5, 0.075],     [0.5, 0.155],     [0.42, 0.18, -0.05],     [0.34, 0.18, -0.05],
    [0.24, 0.123, -0.05],     [0.58, 0.18, 0.05],     [0.66, 0.225, 0.05],     [0.6, 0.107, 0.05],
    [0.46, 0.52, -0.04],     [0.42, 0.72, -0.04],     [0.4, 0.93, -0.04],     [0.54, 0.52, 0.04],
    [0.58, 0.72, 0.04],     [0.6, 0.93, 0.04],     [0.47, 0.069],     [0.53, 0.069],
    [0.45, 0.075],     [0.55, 0.075],   ]),
  举手: pose([
    [0.5, 0.075],     [0.5, 0.155],     [0.42, 0.18, -0.05],     [0.38, 0.075, -0.05],
    [0.36, 0.031, -0.05],     [0.58, 0.18, 0.05],     [0.62, 0.075, 0.05],     [0.64, 0.031, 0.05],
    [0.46, 0.52, -0.04],     [0.45, 0.72, -0.04],     [0.44, 0.93, -0.04],     [0.54, 0.52, 0.04],
    [0.55, 0.72, 0.04],     [0.56, 0.93, 0.04],     [0.47, 0.069],     [0.53, 0.069],
    [0.45, 0.075],     [0.55, 0.075],   ]),
}

export function defaultSkeleton(): SkeletonState {
  return { points: PRESETS['张手'].map((p) => ({ ...p })), azimuth: 0, preset: '张手' }
}

/**
 * 2D 骨架（DWPose 从图里提的）→ 编辑器用的 3D 骨架。
 *
 * **为什么需要推断 z**：2D 图里没有深度，直接把 z 全设 0 的话，一转到侧面
 * 肩膀和髋部就叠成一个点，模型读不懂。这里按"人体左右侧"给每个关节一个
 * 基础深度（右侧在前、左侧在后），手腕/脚踝再加大一点前后差，于是换方向时
 * 至少能正确表现"一前一后"的四肢。数值是保守的启发式，不是从图里测出来的——
 * 用户还能用深度滑块再改。
 */
const SIDE_Z: number[] = [
  0, 0,
  -0.04, -0.06, -0.12,   // 右肩 / 右肘 / 右腕
  0.04, 0.06, 0.12,      // 左肩 / 左肘 / 左腕
  -0.04, -0.06, -0.10,   // 右髋 / 右膝 / 右踝
  0.04, 0.06, 0.10,      // 左髋 / 左膝 / 左踝
  -0.03, 0.03, -0.03, 0.03,
]

export function skeletonFromPoints(
  points: ([number, number] | null)[], azimuthGuess = 0,
): SkeletonState {
  const base = PRESETS['站立']
  const pts: Joint3[] = base.map((b, i) => {
    const q = points[i]
    if (!q) return { ...b }        // 没检到的关节用站立预设补，保证 18 点齐全
    return {
      x: clamp(q[0], 0, 1), y: clamp(q[1], 0, 1),
      z: SIDE_Z[i] ?? 0,
    }
  })
  // 2D 分不出正/背面；把后端给的朝向猜测只用于提示，导入一律按正面，
  // 用户点方向按钮切换（避免对已经是侧面的图做二次旋转）
  void azimuthGuess
  return { points: pts, azimuth: 0, preset: '自定义' }
}

/** 8 个方向：精灵素材常用的方位。名字里的「右」指人物自己的右侧。 */
export const DIRECTIONS: Array<[string, number]> = [
  ['正面', 0], ['右前', 45], ['右侧', 90], ['右后', 135],
  ['背面', 180], ['左后', 225], ['左侧', 270], ['左前', 315],
]

const rgb = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`

/**
 * 头骨相关的 7 条边（鼻-颈、头骨的四个角、两耳到肩）。
 * 人形模式下**不画**——它们正好落在脸上，把五官全糊住；
 * 骨架格式本身不变，导出照样有这几条边。
 */
const HEAD_LIMBS = new Set([12, 13, 14, 15, 16, 17, 18])

/** 头部的关节点（鼻/眼/耳）。人形模式下淡化——它们正好在脸上，
 *  不淡化就是一团彩色糊住五官。照样可以拖。 */
const HEAD_JOINTS = new Set([0, 14, 15, 16, 17])

/* ------------------------------------------------------------------ *
 * 人形体（3D 感）
 *
 * 为什么要有这一层：纯线条骨架看不出"这个姿势像不像人"——手脚的位置对了，
 * 但躯干是扁的、朝向是猜的。这里把同一份 18 个关节渲染成**有体积的人形**：
 * 躯干是面、四肢是带锥度的柱、头是椭球，并按深度排序遮挡，转到侧面时
 * 身体会自然变窄。
 *
 * **导出的 PNG 仍然是原来的 OpenPose-18 线骨架**（见 renderSkeletonPNG）——
 * Qwen / ControlNet 认的是那个格式，人形体只是编辑时的参考显示。
 * ------------------------------------------------------------------ */

/** 人体各肢段的半径（相对身高；viewBox 高 100 ≈ 身高）。[起点, 终点, r0, r1] */
const HUMAN_SEGS: Array<[number, number, number, number]> = [
  [2, 3, 2.7, 2.0], [3, 4, 2.0, 1.4],      // 右 上臂 / 前臂
  [5, 6, 2.7, 2.0], [6, 7, 2.0, 1.4],      // 左 上臂 / 前臂
  [8, 9, 3.5, 2.4], [9, 10, 2.4, 1.6],     // 右 大腿 / 小腿
  [11, 12, 3.5, 2.4], [12, 13, 2.4, 1.6],  // 左 大腿 / 小腿
]

/** 躯干配色（正面 / 背面 / 侧面） */
const SKIN_FRONT = '#b6c2d6'
const SKIN_BACK = '#8b97ad'
const SKIN_EDGE = '#e7eef8'

/** 深度（正交旋转后的 z）：值越小越靠近镜头。 */
function depthOf(p: Joint3, azimuth: number) {
  const a = rad(azimuth)
  return -(p.x - 0.5) * Math.sin(a) + p.z * Math.cos(a)
}

/** 带锥度的肢体：一个四边形，两端半径不同。 */
function limbPath(p0: { x: number; y: number }, p1: { x: number; y: number },
                  r0: number, r1: number) {
  const dx = p1.x - p0.x
  const dy = p1.y - p0.y
  const len = Math.hypot(dx, dy) || 1
  const nx = -dy / len
  const ny = dx / len
  const P = (p: { x: number; y: number }, r: number, s: number) =>
    `${(p.x + nx * r * s).toFixed(2)},${(p.y + ny * r * s).toFixed(2)}`
  return `M${P(p0, r0, 1)} L${P(p1, r1, 1)} L${P(p1, r1, -1)} L${P(p0, r0, -1)} Z`
}

/**
 * 画一个人形体。返回按「远 → 近」排好序的 SVG 片段。
 *
 * 深度排序是体积感的关键：侧面时两条腿一前一后，近的那条要盖住远的那条；
 * 不排序的话两条腿会画成同一个平面，看着像纸片人。
 */
function renderHuman(state: SkeletonState) {
  const az = state.azimuth
  const J = state.points
  const S = J.map((p) => project(p, az))
  const D = J.map((p) => depthOf(p, az))
  const back = Math.abs(((az % 360) + 360) % 360 - 180) < 45
  const side = Math.abs(((az % 360) + 360) % 360 - 90) < 45
  const base = back ? SKIN_BACK : SKIN_FRONT

  type Item = { d: number; el: JSX.Element }
  const items: Item[] = []

  // ---- 躯干：肩-腰-髋 的六边形（侧面自然收窄）----
  const shR = S[2], shL = S[5], hipR = S[8], hipL = S[11]
  const waistR = { x: (shR.x + hipR.x) / 2 + (hipR.x - shR.x) * 0.18,
                   y: (shR.y + hipR.y) / 2 }
  const waistL = { x: (shL.x + hipL.x) / 2 + (hipL.x - shL.x) * 0.18,
                   y: (shL.y + hipL.y) / 2 }
  const torso = [
    `${shR.x},${shR.y}`, `${shL.x},${shL.y}`, `${waistL.x},${waistL.y}`,
    `${hipL.x},${hipL.y}`, `${hipR.x},${hipR.y}`, `${waistR.x},${waistR.y}`,
  ].join(' ')
  items.push({
    d: (D[2] + D[5] + D[8] + D[11]) / 4,
    el: <polygon key="torso" points={torso} fill={base}
      stroke={SKIN_EDGE} strokeWidth="0.35" strokeOpacity="0.5" />,
  })

  // ---- 头：颈与鼻的中点，横向半径随视角变 ----
  const nose = S[0], neck = S[1]
  const headC = { x: (nose.x + neck.x) / 2, y: (nose.y + neck.y) / 2 - 0.6 }
  // 头高约占身高的 1/7（viewBox 100 → 头高 ≈ 14，ry ≈ 7）。
  // 之前按「鼻到颈的距离」直接乘 1.15 会得到 ry≈13.6 的巨头，
  // 整个人看着像吉祥物而不是人。
  const headR = clamp((neck.y - nose.y) * 0.62, 3.4, 7.2)
  const a = rad(az)
  const headW = headR * (0.80 * Math.abs(Math.cos(a)) + 0.66 * Math.abs(Math.sin(a)))
  const headH = headR * 1.12
  items.push({
    d: D[0],
    el: <g key="head">
      <ellipse cx={headC.x} cy={headC.y} rx={headW} ry={headH}
        fill={back ? SKIN_BACK : SKIN_FRONT} stroke={SKIN_EDGE}
        strokeWidth="0.35" strokeOpacity="0.5" />
      {back ? (
        // 背面：发际线，没有五官——一眼看出朝向
        <path d={`M${headC.x - headW * 0.72},${headC.y - headH * 0.15}
                  Q${headC.x},${headC.y - headH * 0.62} ${headC.x + headW * 0.72},${headC.y - headH * 0.15}`}
          fill="none" stroke={SKIN_EDGE} strokeWidth="0.5" opacity="0.55" />
      ) : side ? (
        // 侧面：只留一只眼与鼻子的一点
        <circle cx={headC.x + headW * 0.25} cy={headC.y - headH * 0.05} r="0.55"
          fill="#2a2f3a" />
      ) : (
        <>
          <circle cx={headC.x - headW * 0.34} cy={headC.y - headH * 0.08} r="0.6" fill="#2a2f3a" />
          <circle cx={headC.x + headW * 0.34} cy={headC.y - headH * 0.08} r="0.6" fill="#2a2f3a" />
          <path d={`M${headC.x - headW * 0.3},${headC.y + headH * 0.38}
                    Q${headC.x},${headC.y + headH * 0.56} ${headC.x + headW * 0.3},${headC.y + headH * 0.38}`}
            fill="none" stroke="#2a2f3a" strokeWidth="0.45" opacity="0.8" />
        </>
      )}
    </g>,
  })

  // ---- 脖子 ----
  items.push({
    d: D[1],
    el: <path key="neck" d={limbPath(neck, { x: headC.x, y: headC.y + headH * 0.5 }, 1.2, 1.5)}
      fill={base} stroke={SKIN_EDGE} strokeWidth="0.3" strokeOpacity="0.45" />,
  })

  // ---- 四肢：带锥度的柱 + 高光 ----
  HUMAN_SEGS.forEach(([i, j, r0, r1], k) => {
    const p0 = S[i], p1 = S[j]
    items.push({
      d: (D[i] + D[j]) / 2,
      el: <g key={`seg${k}`}>
        <path d={limbPath(p0, p1, r0, r1)} fill={base}
          stroke={SKIN_EDGE} strokeWidth="0.3" strokeOpacity="0.45" />
        {/* 高光：沿肢体中轴偏上一条亮线，暗示圆柱受光 */}
        <path d={limbPath(
          { x: p0.x - r0 * 0.28, y: p0.y - r0 * 0.28 },
          { x: p1.x - r1 * 0.28, y: p1.y - r1 * 0.28 },
          r0 * 0.34, r1 * 0.34)}
          fill={SKIN_EDGE} opacity="0.4" />
      </g>,
    })
  })

  // 远 → 近
  items.sort((x, y) => y.d - x.d)
  return items.map((it, k) => <g key={`h${k}`}>{it.el}</g>)
}


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

export default function SkeletonEditor({
  state, onChange, onExport, exporting, onImportImage, importing,
  onExportMannequin, mannequinExporting,
}: Props) {
  const [dragIdx, setDragIdx] = useState<number | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  // 显示层：人形（3D 感）/ 线框 / 叠加。**导出永远是线框骨架**
  // （"人偶渲染带朝向"实测无效，见 types.ts 的 PcView 注释）。
  const [view, setView] = useState<'human' | 'wire' | 'both'>('both')
  const svgRef = useRef<SVGSVGElement | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)
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

  /**
   * 导出 3D 人偶图。官方 PoseStudio 只吃 2D 关键点（OpenPose 的 COCO-18），
   * 所以这里把编辑器的 x/y 原样传过去；编辑器的 z（前后深度）在官方那套
   * 2D→骨旋转里用不到（朝向由方位角单独给）。
   */
  const doExportMannequin = useCallback(async () => {
    if (!onExportMannequin) return
    await onExportMannequin(state.points.map((p) => ({ x: p.x, y: p.y })), state.azimuth)
  }, [state.points, state.azimuth, onExportMannequin])

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

      <div className="skel-views">
        {([['human', '人形'], ['wire', '线框'], ['both', '叠加']] as const).map(([k, label]) => (
          <button key={k} className={`mini${view === k ? ' on' : ''}`}
            onClick={() => setView(k)}
            title={k === 'human' ? '只看 3D 人形体（体积感，能看清姿势像不像人）'
              : k === 'wire' ? '只看线框骨架（= 导出的 OpenPose-18）'
              : '人形 + 线框叠加'}>
            {label}
          </button>
        ))}
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
        {/* 人形体在底层：它只是"长什么样"的参考，拖拽靠上面的关节点 */}
        {view !== 'wire' && renderHuman(state)}
        {view !== 'human' && LIMBS.map(([p, q], i) => (
          pts[p] && pts[q] && !HEAD_LIMBS.has(i) && (
            <line key={`l${i}`} x1={pts[p].x} y1={pts[p].y} x2={pts[q].x} y2={pts[q].y}
              stroke={rgb(COLORS[i % COLORS.length])} strokeWidth="1.8"
              strokeLinecap="round" opacity={0.95} />
          )
        ))}
        {pts.map((p, k) => (
          <g key={`j${k}`} onPointerDown={(e) => startDragJoint(k, e)}>
            <circle className="skel-joint" cx={p.x} cy={p.y}
              r={(selected === k ? 2.6 : 1.9) - (HEAD_JOINTS.has(k) ? 0.5 : 0)}
              fill={rgb(COLORS[k % COLORS.length])}
              opacity={HEAD_JOINTS.has(k) && view !== 'wire' ? 0.45 : 1}
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
        <button
          className="mini"
          onClick={() => fileRef.current?.click()}
          disabled={importing || !onImportImage}
          title="从一张真人/写实人像自动提取 18 个关节，再手动换方向"
        >
          {importing ? '提取中…' : '📷 从图片'}
        </button>
        <button className="mini wide" onClick={() => void doExport()} disabled={exporting}>
          {exporting ? '导出中…' : '导出骨架图'}
        </button>
        <button
          className="mini wide"
          onClick={() => void doExportMannequin()}
          disabled={mannequinExporting || !onExportMannequin}
          title={'导出 3D 人偶图（白底灰人偶）。VNCCS 的姿态 LoRA 是在这种图上训的，\n'
            + '比线框骨架贴合得好；朝向 = 当前的方位角（0° 正面 / 90° 侧面 / 180° 背面）。\n'
            + '会调起官方 PoseStudio 渲染，第一次要等网格加载。'}
        >
          {mannequinExporting ? '渲染人偶…' : '导出人偶图（3D）'}
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
        拖圆点摆关节；「从图片」自动提真人姿势（像素/插画认不出，用预设手动摆）。
        方位角绕竖轴旋转（0° 正面 / 90° 侧面 / 180° 背面）即可换视觉方向。
        「人形」是带体积的 3D 感显示，方便看清姿势像不像人、朝向对不对
        （背面会有发际线、没有五官）。
        <b>两种导出</b>：「导出骨架图」是 OpenPose-18 线骨架（ControlNet 系要这个）；
        「导出人偶图」是白底灰 3D 人偶——VNCCS 的姿态 LoRA 就是在这种图上训的，
        姿态贴合明显更好，做角色迁移请优先用它。
      </div>
    </div>
  )
}
