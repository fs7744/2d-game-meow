/** 内嵌骨骼编辑器的业务类型与显示用常量。核心数据结构本身由 rig/facade 里的 JS 运行时持有。 */
export type Sel =
  | { kind: 'bone' | 'slot' | 'constraint' | 'event' | 'skin' | 'pose' | 'animation'; index: number }
  | null

export type RigMode = 'setup' | 'animate'
export type RigTool = 'select' | 'rotate' | 'translate' | 'scale' | 'create'
export type RigSource = { kind: string; label: string; name: string; url?: string }

export type ViewState = { zoom: number; x: number; y: number }

export const ROLE_LABELS: Record<string, string> = {
  head: '头', neck: '颈', torso: '躯干', pelvis: '骨盆',
  upperarm: '上臂', forearm: '前臂', hand: '手', arm: '整条手臂',
  thigh: '大腿', shin: '小腿', foot: '脚', leg: '整条腿',
  eye: '眼', pupil: '瞳孔', brow: '眉', mouth: '嘴', nose: '鼻',
  jaw: '下颌', ear: '耳', hair: '头发', tail: '尾巴', wing: '翅膀', extra: '其它',
  none: '不参与绑定',
}

export const SIDE_LABELS: Record<string, string> = {
  '': '无', l: '左', r: '右',
}

export const CONSTRAINT_TYPES = ['ik', 'transform', 'path', 'physics', 'slider'] as const

export const CURVES: { label: string; value: any }[] = [
  { label: '线性', value: 'linear' },
  { label: '阶跃', value: 'stepped' },
  { label: '缓入缓出', value: { bezier: [0.42, 0, 0.58, 1] } },
  { label: '缓入', value: { bezier: [0.42, 0, 1, 1] } },
  { label: '缓出', value: { bezier: [0, 0, 0.58, 1] } },
]

export function basename(p: string): string {
  return String(p).replace(/\\/g, '/').split('/').pop() || 'file'
}

export function fmtTime(t: number): string {
  return `${t.toFixed(2)}s`
}

export function stamp(sec: number): string {
  if (!sec) return '—'
  const d = new Date(sec * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getMonth() + 1}/${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`
}

/** 屏幕像素 → 世界坐标（Y 向上，原点在画布中心）。 */
export function toWorld(view: ViewState, w: number, h: number, sx: number, sy: number): [number, number] {
  return [(sx - w / 2 - view.x) / view.zoom, -(sy - h / 2 - view.y) / view.zoom]
}
