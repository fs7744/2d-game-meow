/**
 * UI review 截图工具（真机 / 脱离 GPU）。
 *
 *   node tools/shot/capture.mjs before            # 截现状
 *   UI_DIST=web/dist_shot node tools/shot/capture.mjs after
 *
 * 自己起一个静态服务托管前端产物，凡是静态文件不存在的路径一律反代到
 * 127.0.0.1:8000 的后端，这样侧栏 / 项目 / 资料库拿到的是真数据，
 * 不需要 GPU，也不需要改 app.py 的挂载目录。
 *
 * 产物：tests/_out/ui_review/<tag>/*.png + diagnostics.json
 */
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..', '..')
const DIST = resolve(ROOT, process.env.UI_DIST || 'web/dist')
const OUT = join(ROOT, 'tests', '_out', 'ui_review', process.argv[2] || 'before')
const BACKEND = 'http://127.0.0.1:8000'
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
const requireFromShot = createRequire(join(HERE, 'package.json'))
const puppeteer = requireFromShot('puppeteer-core')

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
  '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.woff': 'font/woff',
  '.glb': 'model/gltf-binary', '.wasm': 'application/wasm', '.bin': 'application/octet-stream',
  '.gif': 'image/gif', '.ico': 'image/x-icon', '.map': 'application/json',
}

async function serve(port) {
  const srv = createServer(async (req, res) => {
    const raw = req.url || '/'
    const pathname = decodeURIComponent(raw.split('?')[0])
    const file = resolve(DIST, '.' + (pathname === '/' ? '/index.html' : pathname))
    if (file.startsWith(DIST + sep) && existsSync(file) && !file.endsWith(sep)) {
      const body = readFileSync(file)
      res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' })
      res.end(body)
      return
    }
    // 反代后端（含 query string）
    try {
      const headers = { ...req.headers }
      delete headers.host
      const chunks = []
      for await (const c of req) chunks.push(c)
      const r = await fetch(BACKEND + raw, {
        method: req.method,
        headers,
        body: ['GET', 'HEAD'].includes(req.method) ? undefined : Buffer.concat(chunks),
      })
      const buf = Buffer.from(await r.arrayBuffer())
      const h = {}
      const ct = r.headers.get('content-type')
      if (ct) h['Content-Type'] = ct
      res.writeHead(r.status, h)
      res.end(buf)
    } catch (e) {
      res.writeHead(502).end(String(e))
    }
  })
  return new Promise((ok) => srv.listen(port, '127.0.0.1', () => ok(srv)))
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function main() {
  mkdirSync(OUT, { recursive: true })
  const srv = await serve(8123)
  const browser = await puppeteer.launch({
    executablePath: EDGE, headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--hide-scrollbars', '--force-device-scale-factor=1'],
    defaultViewport: { width: 1600, height: 950, deviceScaleFactor: 1 },
  })
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e)))
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })

  const shot = async (name, wait = 900) => {
    await sleep(wait)
    await page.screenshot({ path: join(OUT, name + '.png') })
    console.log('  ✓', name)
  }
  const clickText = async (sel, text) => {
    const ok = await page.evaluate((s, t) => {
      const el = [...document.querySelectorAll(s)].find((x) => (x.textContent || '').includes(t))
      if (!el) return false
      el.click()
      return true
    }, sel, text)
    if (!ok) console.log('  ! 没找到按钮:', sel, text)
    return ok
  }

  console.log('截图目录:', OUT)
  await page.goto('http://127.0.0.1:8123/index.html?demo=1', { waitUntil: 'networkidle2', timeout: 60000 })
  await shot('01-canvas', 1400)

  // 展开左侧全部折叠段 + 节点里的高级项
  await page.evaluate(() => {
    document.querySelectorAll('details').forEach((d) => { d.open = true })
  })
  await page.evaluate(() => {
    document.querySelectorAll('.rail-sec-head').forEach((b) => {
      const sec = b.closest('.rail-sec')
      if (sec && sec.getAttribute('data-open') !== 'true') { /* 由 React 状态管，见下 */ }
    })
  })
  await shot('02-canvas-details', 700)

  // 视图切换：角色中心
  await clickText('.view-tab', '角色中心')
  await shot('03-characters', 1600)

  // 回到画布，打开资料库抽屉
  await clickText('.view-tab', '画布')
  await sleep(400)
  const libOpened = await clickText('button', '资料库')
  if (!libOpened) {
    await page.evaluate(() => {
      const b = [...document.querySelectorAll('button, .rail-item-main')]
        .find((x) => /资料库|收藏|姿势库/.test(x.textContent || ''))
      if (b) b.click()
    })
  }
  await shot('04-library', 1200)

  // 侧栏重构后的四个一级面板 + 搜索 + 收起态
  await page.keyboard.press('Escape')   // 关掉资料库抽屉，别挡住侧栏
  await sleep(300)
  const clickTab = async (label) => {
    await page.evaluate((t) => {
      const b = [...document.querySelectorAll('.rail-tab')]
        .find((x) => (x.textContent || '').includes(t))
      b?.click()
    }, label)
  }
  for (const [name, label] of [
    ['05-tab-projects', '项目'],
    ['06-tab-library', '资料库'],
    ['07-tab-settings', '设置'],
    ['08-tab-nodes', '节点'],
  ]) {
    await clickTab(label)
    await shot(name, 500)
  }
  await page.click('.rail-search-input')
  await page.type('.rail-search-input', '抠图')
  await shot('09-node-search', 600)

  // 诊断：字体是否真的加载、层级颜色、主要元素尺寸
  const diag = await page.evaluate(() => {
    const cs = (sel) => {
      const el = document.querySelector(sel)
      if (!el) return null
      const s = getComputedStyle(el)
      return {
        font: s.fontFamily, size: s.fontSize, color: s.color,
        bg: s.backgroundColor, border: s.borderColor,
        rect: (() => { const r = el.getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height)] })(),
      }
    }
    return {
      fontsLoaded: document.fonts ? [...document.fonts].map((f) => f.family) : [],
      body: cs('body'), title: cs('.title'), sub: cs('.sub'),
      rail: cs('.rail'), railHead: cs('.rail-sec-head'),
      node: cs('.node'), nodeKind: cs('.node-kind'),
      tab: cs('.view-tab'), paletteHead: cs('.palette-head'),
      button: cs('.mini'),
      counts: {
        nodes: document.querySelectorAll('.node').length,
        railItems: document.querySelectorAll('.rail-item').length,
        buttons: document.querySelectorAll('button').length,
        details: document.querySelectorAll('details').length,
        tabs: document.querySelectorAll('.rail-tab').length,
        sections: document.querySelectorAll('.rail-sec').length,
        nodeEntries: document.querySelectorAll('[data-page="nodes"] .rail-item').length,
      },
      sidebar: {
        activeTab: document.querySelector('.rail-tab.on')?.textContent,
        activePage: [...document.querySelectorAll('.rail-page')]
          .filter((p) => !p.classList.contains('is-hidden'))
          .map((p) => p.dataset.page),
        searchValue: document.querySelector('.rail-search-input')?.value,
        visibleNodeItems: [...document.querySelectorAll('[data-page="nodes"] .rail-item')]
          .filter((x) => !x.classList.contains('filtered-out')).length,
        favChips: document.querySelectorAll('.rail-fav-chip').length,
      },
      scroll: { w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight },
    }
  })
  writeFileSync(join(OUT, 'diagnostics.json'), JSON.stringify({ diag, errors }, null, 2))
  console.log('诊断:', JSON.stringify(diag.counts), JSON.stringify(diag.sidebar), '错误', errors.length, '条')

  // 收起态放最后拍，避免影响上面的诊断（收起后 tab 不在 DOM 里）
  await page.evaluate(() => document.querySelector('.rail-toggle')?.click())
  await shot('10-collapsed', 500)

  // 结果优先视图：另开一页加载 ?demo=result（前端假数据，不跑生成）。
  // 依次验证「默认只看结果 → 展开参数 → 收回结果」这条闭环。
  const rp = await browser.newPage()
  await rp.setViewport({ width: 1600, height: 950, deviceScaleFactor: 1 })
  await rp.goto('http://127.0.0.1:8123/index.html?demo=result',
    { waitUntil: 'networkidle2', timeout: 60000 })
  await sleep(1100)
  const resultDiag = await rp.evaluate(() => ({
    hasResultPanel: !!document.querySelector('.node-result'),
    focusMode: !!document.querySelector('.node.result-focus'),
    paramsStillInDom: !!document.querySelector('.node .prompt'),
    actions: [...document.querySelectorAll('.node-result-btn')].map((b) => b.textContent?.trim()),
  }))
  await rp.screenshot({ path: join(OUT, '11-node-result.png') })
  await rp.evaluate(() => {
    const b = [...document.querySelectorAll('.node-result-btn')]
      .find((x) => (x.textContent || '').includes('参数'))
    b?.click()
  })
  await sleep(400)
  await rp.screenshot({ path: join(OUT, '12-node-params.png') })
  await rp.evaluate(() => document.querySelector('.icon-btn.result-toggle')?.click())
  await sleep(400)
  await rp.screenshot({ path: join(OUT, '13-node-result-again.png') })
  console.log('  ✓ 11/12/13 节点结果视图', JSON.stringify(resultDiag))
  await rp.close()

  await browser.close()
  srv.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
