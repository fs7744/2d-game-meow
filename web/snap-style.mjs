/**
 * 把真实组件渲染成静态 HTML（jsdom），内联 CSS，交给无头浏览器截图。
 *
 * 用途：样式改完要"看见"，不能靠推理。jsdom 不渲染像素，但它能给出**真实的 DOM
 * 结构和 className**；把这棵树配上真实 CSS，用 Edge 截图就是所见即所得。
 */
import { readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { JSDOM, VirtualConsole } from 'jsdom'

const here = dirname(fileURLToPath(import.meta.url))
const dist = resolve(here, 'dist')

const css = readdirSync(resolve(dist, 'assets'))
  .filter((f) => f.endsWith('.css'))
  .map((f) => readFileSync(resolve(dist, 'assets', f), 'utf8'))
  .join('\n')

function makeFetch() {
  return async (url, init = {}) => {
    const u = String(url)
    const ok = (obj) => ({ ok: true, json: async () => obj })
    if (u.endsWith('/health')) return ok({ ok: true, engine: { device: 'Fake GPU' } })
    if (u.endsWith('/models')) {
      return ok({
        current: 'sd15', available: ['sd15', 'sdxl'],
        recommend: {
          sd15: { width: 768, height: 768, steps: 30, cfg: 7, pixel: 6, palette: 32, max_candidates: 6 },
          sdxl: { width: 768, height: 768, steps: 24, cfg: 6.5, pixel: 6, palette: 32, max_candidates: 3 },
        },
      })
    }
    if (u.endsWith('/loras')) {
      return ok({ count: 2, items: [
        { name: 'wuxia_v2', label: '武侠 Wuxia V2', trigger: 'wuxia', scale: 0.8, preset: 'wuxia', base: 'sd15' },
        { name: 'px_illustrious', label: '像素', trigger: '', scale: 0.75, preset: 'pixel', base: 'sd15' },
      ] })
    }
    if (u.endsWith('/pose_dirs')) {
      return ok({ count: 1, items: [{ path: 'D:\\x\\assets\\_pose', count: 4, sample: ['a.png'], name: '_pose' }] })
    }
    if (u.endsWith('/assets')) return ok({ count: 0, items: [] })
    return { ok: false, status: 404, json: async () => ({}) }
  }
}

const html = readFileSync(resolve(dist, 'index.html'), 'utf8')
const vc = new VirtualConsole()
const dom = new JSDOM(html, {
  url: 'http://localhost/',
  runScripts: 'dangerously',
  resources: 'usable',
  pretendToBeVisual: true,
  virtualConsole: vc,
  beforeParse(win) {
    win.fetch = makeFetch()
    win.matchMedia = () => ({ matches: false, addListener() {}, removeListener() {} })
    win.HTMLElement.prototype.setPointerCapture = () => {}
    win.HTMLElement.prototype.releasePointerCapture = () => {}
    win.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }
    // canvas 在 jsdom 里没有实现，屏蔽掉噪音
    win.HTMLCanvasElement.prototype.getContext = () => null
  },
})

const bundle = html.match(/src="([^"]+\.js)"/)[1]
const code = readFileSync(resolve(dist, bundle.replace(/^\//, '')), 'utf8')
const script = dom.window.document.createElement('script')
script.textContent = code
dom.window.document.body.appendChild(script)

const wait = (ms) => new Promise((r) => setTimeout(r, ms))
await wait(600)

const doc = dom.window.document
const win = dom.window

// 造 4 个节点：覆盖 select 最多的几种
const items = [...doc.querySelectorAll('.rail-item')]
const addByLabel = async (label) => {
  const it = items.find((x) => (x.querySelector('b')?.textContent || '') === label)
  it?.querySelector('.rail-add')?.dispatchEvent(new win.MouseEvent('click', { bubbles: true }))
  await wait(150)
}
await addByLabel('图生成')
await addByLabel('序列帧替换')
await addByLabel('姿态序列')
await addByLabel('局部 AI 重绘')

// 点开「更多设置」，让折叠区里的控件也进画面
for (const d of doc.querySelectorAll('details.adv')) d.open = true
await wait(200)

// 摆开：导出后是死 HTML，直接改 style 不影响任何逻辑
const nodes = [...doc.querySelectorAll('.node')]
const pos = [[20, 20], [360, 20], [700, 20], [1040, 20]]
nodes.forEach((n, i) => {
  n.style.left = `${pos[i]?.[0] ?? 20}px`
  n.style.top = `${pos[i]?.[1] ?? 20}px`
})

const page = `<!doctype html><html><head><meta charset="utf-8"><style>${css}
body{margin:0;background:#0a0a0a;width:1600px;height:1000px}
.canvas-wrap,.canvas,.world{width:1600px!important;height:1000px!important;transform:none!important}
</style></head><body>${doc.body.innerHTML}</body></html>`

writeFileSync(resolve(here, '../../tmp-ui-snap.html'), page)
console.log('已写出 tmp-ui-snap.html，节点数', nodes.length)
