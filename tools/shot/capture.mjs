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
    // CC_FAKE：精灵帧 / 图集是 <img src> 直接发的请求，不走页面里的假 fetch，
    // 这里用本地图标顶上，好真机验证「成果常驻右栏」这一块的排版。
    if (process.env.CC_FAKE === '1'
      && (pathname.endsWith('/poses') || pathname.startsWith('/files/'))) {
      const icon = join(DIST, 'icon.png')
      if (existsSync(icon)) {
        res.writeHead(200, { 'Content-Type': 'image/png' })
        res.end(readFileSync(icon))
        return
      }
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
  await page.goto('http://127.0.0.1:8123/index.html?demo=1&editor=1', { waitUntil: 'networkidle2', timeout: 60000 })
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

  // 回到画布：从角色中心切回来默认落在「画布库」首页，而 04~13 拍的是
  // 编辑器里的侧栏 / 资料库 / 节点。直接重载 ?editor=1 深链最稳，
  // 不用在库里再点一次卡片（后端没起时库里可能只有空库大按钮）。
  await page.goto('http://127.0.0.1:8123/index.html?demo=1&editor=1',
    { waitUntil: 'networkidle2', timeout: 60000 })
  await sleep(900)
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
    ['05-tab-projects', '画布'],
    ['06-tab-library', '资料库'],
    ['07-tab-settings', '设置'],
    ['08-tab-nodes', '节点'],
  ]) {
    await clickTab(label)
    await shot(name, 500)
  }
  const searchOk = await page.evaluate(() => {
    const el = document.querySelector('.rail-search-input')
    if (!el) return false
    el.focus()
    return true
  })
  if (searchOk) await page.keyboard.type('抠图')
  else console.log('  ! 节点搜索框未渲染（跳过输入）')
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
  await rp.goto('http://127.0.0.1:8123/index.html?demo=result&editor=1',
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

  // 离线假数据：画布库 → 画布编辑器 → 角色库 → 四步工作台。
  // CC_FAKE=1 时全部走页面内假接口，不接后端 / 不占 GPU。
  if (process.env.CC_FAKE === '1') {
    const fp = await browser.newPage()
    await fp.setViewport({ width: 1600, height: 950, deviceScaleFactor: 1 })
    await fp.evaluateOnNewDocument(() => {
      const real = window.fetch.bind(window)
      const json = (o) => new Response(JSON.stringify(o), {
        status: 200, headers: { 'Content-Type': 'application/json' },
      })
      const NOW = 1793500000
      const chars = [
        { name: '白色猫骑士', cover_url: '/icon.png', sex: 'female', style: 'ghibli_miyazaki',
          nsfw: false, costumes: ['Naked', 'Knight'], sprites: 12, config_version: '2.0',
          tags: ['ghibli_miyazaki', 'human', 'silver hair', 'Knight'],
          custom_tags: ['主角'], note: '主角，立绘待精修',
          created_at: NOW - 86400 * 9, updated_at: NOW - 3600 },
        { name: '精灵弓手', cover_url: '/icon.png', sex: 'female', style: 'anime',
          nsfw: false, costumes: ['Naked'], sprites: 6, config_version: '2.0',
          tags: ['elf', 'green eyes'], custom_tags: [], note: '',
          created_at: NOW - 86400 * 5, updated_at: NOW - 86400 },
        { name: '机械法师', cover_url: '/icon.png', sex: 'male', style: 'pixel_art',
          nsfw: false, costumes: ['Naked', 'Battle'], sprites: 20, config_version: '2.0',
          tags: ['gen', 'topixel'], custom_tags: ['Boss'], note: '',
          created_at: NOW - 86400 * 2, updated_at: NOW - 86400 * 2 },
      ]
      const canvases = [
        { id: 'c1', name: '主角立绘产线', cover_url: '/icon.png', created: '', updated: '',
          count: 6, tags: ['gen', 'topixel'], custom_tags: ['主角'], note: '',
          created_ts: NOW - 86400 * 9, updated_ts: NOW - 3600 },
        { id: 'c2', name: '敌人动作批量', cover_url: '/icon.png', created: '', updated: '',
          count: 9, tags: ['posebatch', 'poseclone'], custom_tags: [], note: '',
          created_ts: NOW - 86400 * 5, updated_ts: NOW - 86400 },
        { id: 'c3', name: '场景扩图实验', cover_url: '/icon.png', created: '', updated: '',
          count: 3, tags: ['outpaint'], custom_tags: [], note: '待评审',
          created_ts: NOW - 86400 * 2, updated_ts: NOW - 86400 * 2 },
      ]
      window.fetch = async (url, init = {}) => {
        const u = String(url)
        const m = init.method || 'GET'
        if (u.endsWith('/health')) return json({ ok: true, engine: { device: '离线预览' } })
        if (u.endsWith('/characters') && m === 'GET') return json({ count: chars.length, items: chars })
        if (u.endsWith('/projects') && m === 'GET') return json({ count: canvases.length,
          items: canvases, current: 'c1' })
        if (u.endsWith('/library') && m === 'GET') return json({ prompts: [], poses: [], prefs: {} })
        if (u.endsWith('/meta') && m === 'PATCH') return json({ ok: true })
        if (u.endsWith('/character-tasks')) return json({ items: [] })
        if (u.includes('/characters/') && u.endsWith('/prompt')) return json({
          name: chars[0].name, style: 'ghibli_miyazaki',
          prompt: 'The image contains one character.\nGender: female\nHair: silver hair\nEyes: red eyes',
          negative: 'blurry', resolution: [792, 1408],
          defaults: { steps: 25, cfg: 3, sampler: 'euler', scheduler: 'simple' },
        })
        if (u.includes('/characters/') && u.endsWith('/atlas')) return json({
          character: chars[0].name, exists: true, frames: 12, animations: 3,
          size: [512, 768], canvas: [128, 192], bytes: 82000, image: '/icon.png',
        })
        if (u.includes('/poses/meta')) return json({
          count: 3, files: ['/icon.png', '/icon.png', '/icon.png'],
        })
        if (u.includes('/poses?')) return real('/icon.png')
        if (u.endsWith('/pose_dirs')) return json({ items: [
          { path: 'D:/x/walk', name: '步行循环', count: 4, sample: [] },
        ] })
        if (u.includes('/projects/') && m === 'GET' && !u.includes('/pose_')) return json({
          id: 'c1', name: '主角立绘产线', created: '', updated: '',
          nodes: [], viewport: { x: 300, y: 96, z: 1 },
        })
        if (u.includes('/characters/') && m === 'GET') return json({
          name: chars[0].name,
          config: { character_info: { name: chars[0].name, sex: 'female', age: 18,
            race: 'human', eyes: 'red eyes', hair: 'silver hair', style: 'ghibli_miyazaki' },
          costumes: {}, costume_ids: {}, config_version: '2.0' },
          costumes: ['Naked', 'Knight'],
          assets: { Naked: { neutral: ['/icon.png', '/icon.png', '/icon.png'],
            angry: ['/icon.png'] }, Knight: { neutral: ['/icon.png'] } },
          cover_url: '/icon.png',
        })
        if (u.startsWith('/files/')) return real('/icon.png')
        return real(url, init)
      }
    })
    await fp.goto('http://127.0.0.1:8123/index.html',
      { waitUntil: 'networkidle2', timeout: 60000 })
    await sleep(1000)
    const fshot = async (name, wait = 700) => {
      await sleep(wait)
      await fp.screenshot({ path: join(OUT, name + '.png') })
      console.log('  ✓', name)
    }
    // 默认首页 = 全屏画布库
    await fshot('20-canvas-library')
    await fp.evaluate(() => document.querySelector('.charlib-card')?.click())
    await fshot('21-canvas-editor', 900)
    await fp.evaluate(() => [...document.querySelectorAll('button')]
      .find((b) => (b.textContent || '').includes('← 画布库'))?.click())
    await sleep(400)
    // 角色中心同样先看到全屏角色库
    await fp.evaluate(() => [...document.querySelectorAll('.view-tab')]
      .find((b) => (b.textContent || '').includes('角色中心'))?.click())
    await fshot('22-character-library')
    await fp.evaluate(() => document.querySelector('.charlib-card')?.click())
    await sleep(800)
    await fshot('23-cc-workspace')
    await fp.evaluate(() => document.querySelector('.cc-step[data-step="3"]')?.click())
    await fshot('24-cc-step3')
    await fp.evaluate(() => {
      document.querySelectorAll('.cc-adv').forEach((d) => { d.open = true })
    })
    await fshot('25-cc-step3-adv')
    await fp.evaluate(() => document.querySelector('.cc-step[data-step="4"]')?.click())
    await fshot('26-cc-step4')
    await fp.close()
  }

  await browser.close()
  srv.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
