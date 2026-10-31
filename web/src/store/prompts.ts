/** 提示词收藏：**只有去重/检索这些纯函数**，存取全部走服务端。
 *
 * 以前整份列表存在 localStorage 里。用户明确要求「localStorage 不存任何东西」——
 * 清一次站点数据，攒了半年的提示词就没了。现在权威副本在服务端
 * `library/prompts.json`（见 workflows/library.py），这里只剩：
 *  - 去重键的计算（前端要判断「这条是不是已经收过」，不能为了判一次发个请求）；
 *  - 旧 localStorage 存档的一次性接管（读完立即删，从此一个字不留）。
 */
import type { FavoritePrompt } from '../types'

/** 旧的 localStorage 键名。迁移完由 store/legacy.ts 统一删。 */
export const LEGACY_KEY = 'meow.prompts.v1'

/**
 * 去重键：正向 + 负向**一起**算。
 *
 * 为什么不只按正向：负向词和正向词是一套搭配。同一句"白猫骑士"配
 * "no text" 和配 "blurry, bad anatomy" 是两条不同的收藏，用户会同时留着。
 * 用 `\u0000` 分隔，避免 `("a", "b")` 与 `("a\u0000b", "")` 撞键。
 */
export function keyOf(text: string, negative: string): string {
  return `${text.trim()}\u0000${negative.trim()}`
}

/** 是不是已经收藏过这一套（正向 + 负向都相同）。 */
export function hasPrompt(
  list: FavoritePrompt[],
  text: string,
  negative = '',
): boolean {
  const t = text.trim()
  if (!t) return false
  const k = keyOf(t, negative)
  return list.some((x) => keyOf(x.text, x.negative || '') === k)
}

/** 命中收藏则返回它的 id（用于「取消收藏」时知道删哪条）。 */
export function findPromptId(
  list: FavoritePrompt[],
  text: string,
  negative = '',
): string | null {
  const t = text.trim()
  if (!t) return null
  const k = keyOf(t, negative)
  return list.find((x) => keyOf(x.text, x.negative || '') === k)?.id ?? null
}

/**
 * 过滤（浏览面板的搜索框）。
 *
 * 正向和负向都参与匹配：用户常常只记得负向里那句 "no text"，
 * 只想得起半个词时也能把那条翻出来。
 */
export function filterPrompts(list: FavoritePrompt[], q: string): FavoritePrompt[] {
  const kw = q.trim().toLowerCase()
  if (!kw) return list
  return list.filter((x) => x.text.toLowerCase().includes(kw)
    || (x.negative || '').toLowerCase().includes(kw))
}

/**
 * 读出 localStorage 里的旧存档（**只看不删**）。
 *
 * 为什么不动手删：迁移要发一次网络请求，失败了数据还得留在浏览器里等着下次
 * 重试 —— 读完就删的话，请求一失败这批收藏就彻底没了。删由 store/legacy.ts
 * 在服务端确认收到之后统一做。
 */
export function readLegacyPrompts(): FavoritePrompt[] {
  try {
    const raw = localStorage.getItem(LEGACY_KEY)
    if (!raw) return []
    const arr = JSON.parse(raw)
    if (!Array.isArray(arr)) return []
    return arr
      .filter((x) => x && typeof x.text === 'string' && x.text.trim())
      .map((x) => ({
        id: String(x.id || x.text),
        text: String(x.text).trim(),
        negative: typeof x.negative === 'string' ? x.negative.trim() : '',
        at: Number(x.at) || 0,
      }))
  } catch {
    return []
  }
}
