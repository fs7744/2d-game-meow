/**
 * 把骨骼编辑器渲染成静态 SVG —— 用来**肉眼检查**人形体的效果，
 * 不必启动前端、不必点界面。
 *
 * 用法（在 web 目录下）：
 *   npx vite build --ssr src/components/SkeletonEditor.tsx --outDir ssr-tmp
 *   node skeleton-preview.mjs [姿势,姿势] [角度,角度]
 *
 * 产物：assets/_skel_preview.svg
 * 转 PNG（本机 Edge 无头模式，不需要装任何渲染库）：
 *   "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless \
 *     --disable-gpu --screenshot=assets/_skel_preview.png --window-size=1040,340 \
 *     "file:///D:/code/2d-game-meow/assets/_skel_preview.svg"
 */
import { renderToStaticMarkup } from 'react-dom/server'
import React from 'react'
import { writeFileSync } from 'node:fs'
import SkeletonEditor, { PRESETS } from './ssr-tmp/SkeletonEditor.js'

function svgOf(state) {
  const html = renderToStaticMarkup(
    React.createElement(SkeletonEditor, { state, onChange() {}, onExport() {} }))
  const m = html.match(/<svg[\s\S]*?<\/svg>/)
  return m ? m[0] : ''
}

const W = 220, H = 290
const poses = (process.argv[2] || '行走').split(',')
const azis = (process.argv[3] || '0,45,90,180').split(',').map(Number)
// 网格：**行 = 姿势，列 = 朝向**
const out = []
poses.forEach((pz, r) => {
  azis.forEach((az, c) => {
    const st = { points: PRESETS[pz].map((q) => ({ ...q })), azimuth: az, preset: pz }
    const svg = svgOf(st).replace('<svg ', `<svg width="${W}" height="${H}" `)
    out.push(`<g transform="translate(${c * W},${r * H})">${svg}</g>`)
  })
})
const sheet = `<svg xmlns="http://www.w3.org/2000/svg" width="${W * azis.length}" height="${H * poses.length}">
<rect width="100%" height="100%" fill="#111"/>${out.join('')}</svg>`
writeFileSync('../assets/_skel_preview.svg', sheet)
console.log('→ assets/_skel_preview.svg', W * azis.length, 'x', H * poses.length)
