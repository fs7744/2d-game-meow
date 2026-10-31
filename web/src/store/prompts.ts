/** 提示词收藏：纯函数 + localStorage，便于单测与复用。 */
import type { FavoritePrompt } from '../types'

const KEY = 'meow.prompts.v1'

/** 去重键：正向 + 负向**一起**算。
 *
 * 为什么不只按正向：负向词和正向词是一套搭配。同一句"白猫骑士"配
 * "no text" 和配 "blurry, bad anatomy" 是两条不同的收藏，用户会同时留着。
 * 用 `\u0000` 分隔，避免 `("a", "b")` 与 `("a\u0000b", "")` 撞键。
 */
function keyOf(text: string, negative: string): string {
  return `${text.trim()}\u0000${negative.trim()}`
}

export function loadPrompts(): FavoritePrompt[] {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return []
    const arr = JSON.parse(raw)
    if (!Array.isArray(arr)) return []
    return arr
      .filter((x) => x && typeof x.text === 'string' && x.text.trim())
      .map((x) => ({
        id: String(x.id || x.text),
        text: String(x.text).trim(),
        // 旧存档没有 negative，读成空串而不是 undefined，后面比较不用再判空
        negative: typeof x.negative === 'string' ? x.negative.trim() : '',
        at: Number(x.at) || 0,
      }))
  } catch {
    return []
  }
}

export function savePrompts(list: FavoritePrompt[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(list))
  } catch {
    /* 存储满/隐私模式：忽略 */
  }
}

/** 收藏（正向+负向都相同才算重复，重复收藏返回原列表）。 */
export function addPrompt(
  list: FavoritePrompt[],
  text: string,
  negative = '',
  now = Date.now(),
): FavoritePrompt[] {
  const t = text.trim()
  if (!t) return list
  const n = negative.trim()
  const k = keyOf(t, n)
  if (list.some((x) => keyOf(x.text, x.negative || '') === k)) return list
  return [{ id: `p${now.toString(36)}`, text: t, negative: n, at: now }, ...list]
}

export function removePrompt(list: FavoritePrompt[], id: string): FavoritePrompt[] {
  return list.filter((x) => x.id !== id)
}

export function togglePrompt(
  list: FavoritePrompt[],
  text: string,
  negative = '',
  now = Date.now(),
): FavoritePrompt[] {
  const t = text.trim()
  if (!t) return list
  const k = keyOf(t, negative)
  const hit = list.find((x) => keyOf(x.text, x.negative || '') === k)
  return hit ? removePrompt(list, hit.id) : addPrompt(list, t, negative, now)
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

/**
 * 含中日韩字符。
 *
 * **只用来换提示文案，不是用来告警的**：SD 时代（CLIP 词表没有中文）它会把
 * 输入框转红警告「会被拆成乱 token」；现在底模是 Qwen-Image（文本编码器
 * Qwen3-VL），中文可以直接写，那个结论已随 SD 管线作废。
 */
export function hasCJK(text: string): boolean {
  return /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/.test(text)
}
