/**
 * localStorage 迁移的统一入口：**读完服务端确认收到了，才抹掉浏览器里的旧存档**。
 *
 * 用户要求「localStorage 不存任何东西」。但历史版本里攒下了提示词收藏和界面偏好，
 * 直接删等于把用户半年攒的东西扔掉 —— 所以做法是：
 *
 *   1. `readLegacy()` —— **只读**，不动 localStorage（见各 store 的 readLegacy*）；
 *   2. App 把它 POST 给 `/library/seed`（服务端只补不覆盖，两个浏览器的存档都能留）；
 *   3. 服务端返回成功 → `clearLegacy()` 才把浏览器里的键删掉。
 *
 * 第 2 步失败时数据还在浏览器里，下次启动还能重试一次；反过来（读完就删、
 * 请求失败）就是不可逆丢失。所以**顺序不能换**。
 */
import type { FavoritePrompt, PosePreset } from '../types'
import { LEGACY_KEY as PROMPT_KEY, readLegacyPrompts } from './prompts'
import { LEGACY_KEY as POSE_KEY, readLegacyPoses } from './poses'
import { LEGACY_MASK_KEYS, readLegacyPrefs } from './prefs'

export type LegacyBundle = {
  prompts: FavoritePrompt[]
  poses: PosePreset[]
  prefs: Record<string, unknown>
}

export function readLegacy(): LegacyBundle {
  return {
    prompts: readLegacyPrompts(),
    poses: readLegacyPoses(),
    prefs: readLegacyPrefs(),
  }
}

export function hasLegacy(b: LegacyBundle): boolean {
  return b.prompts.length > 0 || b.poses.length > 0
    || Object.keys(b.prefs).length > 0
}

/** 抹掉所有已知的旧键。**只在服务端确认收到之后调用。 */
export function clearLegacy(): void {
  for (const k of [PROMPT_KEY, POSE_KEY, ...LEGACY_MASK_KEYS]) {
    try { localStorage.removeItem(k) } catch { /* 隐私模式忽略 */ }
  }
}
