/**
 * 界面偏好（蒙版颜色 / 透明度这类"记住上次选的什么"）。
 *
 * 以前写在 localStorage，现在跟着资料库一起**落到服务端磁盘**
 * （用户要求 localStorage 一个字不留）。
 *
 * 为什么不用 React 状态一路透传：MaskEditor 在节点卡片深处，为两个标量套一套
 * context/props 会把调用链拖长；偏好本身也不需要即时同步到别处。
 * 所以这里做**内存缓存 + 写接口**：
 *   - 应用启动时 `hydratePrefs()` 灌入服务端那份；
 *   - `getPref()` 同步读缓存（渲染要用，不能等网络）；
 *   - `setPref()` 立刻改缓存并异步写服务端（失败只在控制台留一句，不打断操作）。
 */
import { savePrefsApi } from '../api'

const KEY_MASK_COLOR = 'meow.maskColor'
const KEY_MASK_ALPHA = 'meow.maskAlpha'

let cache: Record<string, unknown> = {}
let hydrated = false

/** 服务端那份到位后调用一次。已有的键不被覆盖（本地可能有用户刚改的）。 */
export function hydratePrefs(prefs: Record<string, unknown>): void {
  cache = hydrated ? { ...prefs, ...cache } : { ...prefs }
  hydrated = true
}

export function getPref<T>(key: string, fallback: T): T {
  const v = cache[key]
  return v === undefined || v === null ? fallback : v as T
}

/** 改偏好：立刻生效（缓存）+ 异步落盘。失败不影响本次操作，但会留痕。 */
export function setPref(key: string, value: unknown): void {
  cache[key] = value
  void savePrefsApi({ [key]: value }).catch((e) => {
    console.warn(`[prefs] 写偏好 ${key} 失败：`, e)
  })
}

/** 蒙版画笔颜色（默认品红：在大多数图上都有足够对比） */
export const PREF_MASK_COLOR = KEY_MASK_COLOR
/** 蒙版画笔透明度（0~1） */
export const PREF_MASK_ALPHA = KEY_MASK_ALPHA

/** 旧 localStorage 键（蒙版偏好以前存在这里），迁移完由 store/legacy.ts 删 */
export const LEGACY_MASK_KEYS = [KEY_MASK_COLOR, KEY_MASK_ALPHA]

/**
 * 读出 localStorage 里的旧偏好（**只看不删**）。
 *
 * 迁移要发一次网络请求，失败了数据还得留在浏览器里等下次重试；删由
 * store/legacy.ts 在服务端确认收到之后统一做。
 *
 * @returns 键 → 值。蒙版透明度这种数值键会自动转成 number。
 */
export function readLegacyPrefs(): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  const pairs: Array<[string, boolean]> = [
    [KEY_MASK_COLOR, false], [KEY_MASK_ALPHA, true],
  ]
  for (const [k, numeric] of pairs) {
    try {
      const v = localStorage.getItem(k)
      if (v === null) continue
      const parsed = numeric ? Number(v) : v
      if (numeric && !Number.isFinite(parsed)) continue
      out[k] = parsed
    } catch { /* 隐私模式：读不到就算了 */ }
  }
  return out
}
