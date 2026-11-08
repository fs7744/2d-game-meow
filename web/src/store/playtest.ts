/**
 * 试玩验收台的纯逻辑层（无 React、无 DOM 依赖，便于单测）。
 *
 * 借鉴 ai_game_workbench 的「角色预览」页（SpriteAnimator.tsx:391-402 键位向量、
 * :5009-5011 fps 换算、:4107-4171 键盘层），但修掉它三个硬伤：
 *
 *  1. 方向只支持 4 个、且 `.at(-1)` 只取最后按下的键 → 这里做**双键向量合成**，
 *     四方向 / 八方向 / 横板各有一张方向表；
 *  2. 播放用 setInterval、位移用另一个 rAF，两套时钟必然滑步 → 这里规定
 *     **每推进一帧就走固定步长 `speed / fps`**，位移与帧天然联动；
 *  3. 5 个动作名和 4 个方向写死 → 这里方向集由 viewMode 参数化。
 *
 * 视角模式是「一等概念」：模板、试玩、导出都得读同一张表，别先做四方向再返工。
 */

export type ViewMode = 'topdown-4' | 'topdown-8' | 'side-2'

export const VIEW_MODES: Array<{ id: ViewMode; label: string; hint: string }> = [
  { id: 'topdown-4', label: '俯视 · 四方向', hint: 'WASD 上下左右，适合俯视 RPG' },
  { id: 'topdown-8', label: '俯视 · 八方向', hint: '双键组合出斜向（W+D = 右上），白模转 8 个机位' },
  { id: 'side-2', label: '横板 · 左右', hint: 'A/D 移动、空格跳跃、右向由左向镜像派生' },
]

export const DIRECTIONS: Record<ViewMode, string[]> = {
  'topdown-4': ['down', 'up', 'left', 'right'],
  'topdown-8': [
    'down', 'down-right', 'right', 'up-right',
    'up', 'up-left', 'left', 'down-left',
  ],
  'side-2': ['left', 'right'],
}

export const DIRECTION_LABELS: Record<string, string> = {
  down: '下', up: '上', left: '左', right: '右',
  'down-right': '右下', 'up-right': '右上',
  'up-left': '左上', 'down-left': '左下',
}

/** 与 PoseStudio.tsx 的 YAWS 对齐：0 正面 / 45 右前 / 90 右侧 … */
export const YAW_MAP: Record<string, number> = {
  down: 0, 'down-right': 45, right: 90, 'up-right': 135,
  up: 180, 'up-left': 225, left: 270, 'down-left': 315,
}

/** 屏幕坐标（+y 向下）；对角线已归一化，避免斜向移动比正向快 41%。 */
export const VECTORS: Record<string, [number, number]> = {
  down: [0, 1], up: [0, -1], left: [-1, 0], right: [1, 0],
  'down-right': [0.7071, 0.7071], 'up-right': [0.7071, -0.7071],
  'up-left': [-0.7071, -0.7071], 'down-left': [-0.7071, 0.7071],
}

const KEYS: Record<string, [number, number]> = {
  w: [0, -1], s: [0, 1], a: [-1, 0], d: [1, 0],
  arrowup: [0, -1], arrowdown: [0, 1], arrowleft: [-1, 0], arrowright: [1, 0],
}

/** 按键集合 → 方向。八方向靠向量合成后取点积最大的方向，不是「最后按下的键」。 */
export function directionFromKeys(keys: Set<string>, mode: ViewMode): string | null {
  if (mode === 'side-2') {
    const x = (keys.has('d') || keys.has('arrowright') ? 1 : 0)
      - (keys.has('a') || keys.has('arrowleft') ? 1 : 0)
    return x < 0 ? 'left' : x > 0 ? 'right' : null
  }
  let x = 0
  let y = 0
  for (const k of keys) {
    const v = KEYS[k]
    if (v) { x += v[0]; y += v[1] }
  }
  if (!x && !y) return null
  const len = Math.hypot(x, y)
  x /= len
  y /= len
  let best = DIRECTIONS[mode][0]
  let bestDot = -Infinity
  for (const dir of DIRECTIONS[mode]) {
    const [vx, vy] = VECTORS[dir]
    const dot = x * vx + y * vy
    if (dot > bestDot) { bestDot = dot; best = dir }
  }
  return best
}

/** fps → 每帧毫秒。 */
export function playbackMs(fps: number): number {
  return Math.max(1, Math.round(1000 / Math.max(1, Math.min(300, fps))))
}

/** 每推进一帧走多远。位移与帧率联动，避免「脚在动、人在飘」。 */
export function movePerFrame(speed: number, fps: number): number {
  return speed / Math.max(1, Math.min(300, fps))
}

/** 下一帧；单次动作播到末尾返回 null（由调用方结束动作）。 */
export function nextFrame(index: number, count: number, loop: boolean): number | null {
  if (count <= 1) return 0
  const next = index + 1
  if (next >= count) return loop ? 0 : null
  return next
}

export function isTextEntryTarget(el: EventTarget | null): boolean {
  const node = el as HTMLElement | null
  if (!node || !node.tagName) return false
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(node.tagName)
    || node.isContentEditable === true
}

/** 舞台边界（像素，相对中心）。 */
export function clampPosition(x: number, y: number, mode: ViewMode): [number, number] {
  if (mode === 'side-2') return [Math.max(-260, Math.min(260, x)), y]
  return [Math.max(-220, Math.min(220, x)), Math.max(-150, Math.min(150, y))]
}

/** 跳跃抛物线：t∈[0,1] → 屏幕 y 偏移（负值在地面之上）。 */
export function jumpOffset(t: number, height = 90): number {
  const c = Math.max(0, Math.min(1, t))
  return -4 * height * c * (1 - c)
}
