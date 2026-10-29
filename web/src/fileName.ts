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
