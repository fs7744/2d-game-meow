/**
 * 检查每个 `hint('X')` 在 PARAM_HINTS 里都有定义。
 *
 * 为什么要这个脚本：hint() 查不到键时界面上只是**没有说明**，不报错、不崩溃，
 * 也不会有任何测试失败——「像素块」就这样漏了一整轮（表里 37 个键、代码引用 38 个），
 * 最后是用户发现的。
 *
 * 静态比对源码比跑 DOM 更彻底：有些分支（局部重绘的"更多设置"）默认折叠，
 * 冒烟用例不一定会展开。
 */
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const file = resolve(here, 'src/nodes/GenNodeCard.tsx')
const src = readFileSync(file, 'utf8')

const tableStart = src.indexOf('const PARAM_HINTS')
const tableEnd = src.indexOf('function hint(')
if (tableStart < 0 || tableEnd < 0) {
  console.error('✗ 找不到 PARAM_HINTS 或 hint()，检查脚本需要更新')
  process.exit(1)
}
const table = src.slice(tableStart, tableEnd)

// 表里的键：两个空格缩进的 `键名: {`（键名可能带引号，如 'LoRA 强度'）
const keys = new Set(
  [...table.matchAll(/^\s{2}([\w\u4e00-\u9fff' ]+):\s*\{/gm)]
    .map((m) => m[1].trim().replace(/^'|'$/g, '')),
)

// 代码里的调用（不含 hint 定义自身）
const used = new Set([...src.matchAll(/hint\('([^']+)'\)/g)].map((m) => m[1]))

const missing = [...used].filter((k) => !keys.has(k)).sort()
const unused = [...keys].filter((k) => !used.has(k)).sort()

console.log(`参数说明：表 ${keys.size} 条，引用 ${used.size} 个`)

if (missing.length) {
  console.error(`✗ 这些参数调用了 hint() 但没有说明：${missing.join(', ')}`)
  console.error('  补进 PARAM_HINTS，否则界面上 hover 只有一句"未定义说明"')
  process.exit(1)
}
// 表里有、代码没用：不算错误（可能是给即将接的参数先写好），但要提示
if (unused.length) {
  console.log(`  提示：这些说明暂时没用上（${unused.join(', ')}）`)
}
console.log('✓ 每个参数都有说明')
