/** 提示词收藏：纯函数 + localStorage，便于单测与复用。 */
import type { FavoritePrompt } from '../types'

const KEY = 'meow.prompts.v1'

export function loadPrompts(): FavoritePrompt[] {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return []
    const arr = JSON.parse(raw)
    if (!Array.isArray(arr)) return []
    return arr
      .filter((x) => x && typeof x.text === 'string' && x.text.trim())
      .map((x) => ({ id: String(x.id || x.text), text: String(x.text).trim(), at: Number(x.at) || 0 }))
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

/** 收藏（按文本去重，重复收藏返回原列表）。 */
export function addPrompt(list: FavoritePrompt[], text: string, now = Date.now()): FavoritePrompt[] {
  const t = text.trim()
  if (!t) return list
  if (list.some((x) => x.text === t)) return list
  return [{ id: `p${now.toString(36)}`, text: t, at: now }, ...list]
}

export function removePrompt(list: FavoritePrompt[], id: string): FavoritePrompt[] {
  return list.filter((x) => x.id !== id)
}

export function togglePrompt(list: FavoritePrompt[], text: string, now = Date.now()): FavoritePrompt[] {
  const t = text.trim()
  if (!t) return list
  const hit = list.find((x) => x.text === t)
  return hit ? removePrompt(list, hit.id) : addPrompt(list, t, now)
}

export function hasPrompt(list: FavoritePrompt[], text: string): boolean {
  const t = text.trim()
  return !!t && list.some((x) => x.text === t)
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
