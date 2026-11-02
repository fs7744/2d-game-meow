/**
 * 路径 / 文件名的小工具。单独成文件是为了避免循环依赖——
 * Canvas 与 GenNodeCard 互相 import，谁也不能从对方那里拿工具函数。
 */

/**
 * 从磁盘路径取「给人看的文件名」。
 *
 * 上传时后端会拼一个 10 位十六进制前缀防覆盖/防路径穿越
 * （`0638ae05ff_Clipboard_Screenshot.png`）。直接取 basename 的话，
 * 用户在连线下拉里看到的是 `0638ae05ff_Clipboard_Scr…`——
 * 前缀把真正有用的名字挤掉了，等于没显示。
 */
export function fileNameOf(p?: string): string {
  const base = (p || '').split(/[\\/]/).pop() || ''
  return base.replace(/^[0-9a-f]{10}_/i, '')
}

/**
 * 动画轨道的**显示名**。
 *
 * glb / fbx 里的 track 名经常是导出工具留下的脏数据，最典型的是 Mixamo：
 * 它把整段动画命名为 `mixamo.com`（FBX 的 take 名），于是界面上就变成
 * 「姿势集名（默认 mixamo.com）」，存出来一个叫 `mixamo.com` 的文件夹。
 * 同类垃圾还有 `Armature|mixamo.com`、`Take 001`、`Action`、`Scene`、
 * `mixamorig|...`、空串。
 *
 * 这些名字对用户没有任何信息量，所以**认不出就退回「未命名动作」**（多条时
 * 加序号区分）。认得出（Idle / Walk / Run / 自定义名）就原样显示 —— 那才是
 * 用户在 Mixamo 上挑动作时看到的词。
 *
 * 这个值同时是**姿势集的默认名**：不填名字直接存，文件夹就叫「未命名动作」
 * 或「Idle」，而不是 `mixamo.com` 这种导出工具的残留。
 */
export function clipLabel(name: string | undefined | null, index = 0): string {
  const raw = String(name || '').trim()
  // 去掉导出工具的前缀：`Armature|Idle` → `Idle`
  const short = raw.replace(/^[^|]*\|/, '').trim()
  const junk = /^(mixamo\.?com|mixamo|mixamorig|take\s*\d*|action|anim(ation)?|track|scene|armature|root|untitled|null|none)$/i
  if (!short || junk.test(short)) return unnamed(index)
  // 名字里带域名/路径的，同样当垃圾（`mixamo.com|Layer0` 这种见过）
  if (/\.(com|net|org)\b/i.test(short) || /[\\/]/.test(short)) return unnamed(index)
  return short
}

/** 认不出时的兜底名。第一条就叫「未命名动作」，多条加序号（下拉里要能区分） */
function unnamed(index: number): string {
  const n = Number(index) || 0
  return n <= 0 ? '未命名动作' : `未命名动作 ${n + 1}`
}
