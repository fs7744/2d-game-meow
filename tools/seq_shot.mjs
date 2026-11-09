/**
 * 角色中心「序列帧动画」页的真机排版核对（真后端 + 真 Edge + 真 iframe）。
 *
 * 冒烟测试（jsdom）能钉住 DOM 结构与请求，但钉不住"东西有没有互相盖住 / 撑出屏幕 /
 * 人偶舞台有没有被压扁"。这个脚本补那一层：
 *
 *   角色库 → 角色 → 进入设定 → 点「序列帧动画」→ 截两块内容
 *
 * 断言：
 *   ① 导航项在「角色设定」之后、「拆部件」之前；
 *   ② 页面没有横向溢出（scrollWidth <= clientWidth + 1）；
 *   ③ 人偶舞台高度落在 40%~80% 视口内（不被压成一条，也不顶穿整屏）；
 *   ④ 两个子页切换时，另一块 `display:none`（iframe 不重载）；
 *   ⑤ 设置条里的控件都在视口内、不互相重叠；
 *   ⑥ 页面无 JS 报错。
 *
 * 资产完全隔离在 `tests/_out/seq_ui/assets`，不碰真实的 assets/。
 *
 *   node tools/seq_shot.mjs
 */
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { createServer } from 'node:http'
import {
  copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync,
} from 'node:fs'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..')
const DIST = join(ROOT, 'web', 'dist')
const SANDBOX = join(ROOT, 'tests', '_out', 'seq_ui')
const ASSETS = join(SANDBOX, 'assets')
const CHAR = 'uitest'
const CHAR_DIR = join(ASSETS, 'characters', CHAR)
const OUT = join(ROOT, 'tests', '_out', 'ui_review', 'seq')
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
const PY = join(ROOT, '.venv', 'Scripts', 'python.exe')
const API_PORT = 8146
const WEB_PORT = 8147
const requireFromShot = createRequire(join(ROOT, 'tools', 'shot', 'package.json'))
const puppeteer = requireFromShot('puppeteer-core')

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let failures = 0
const ok = (m) => console.log('  ok   ' + m)
const bad = (m) => { failures++; console.log('  FAIL ' + m) }
const check = (cond, m, extra = '') => (cond ? ok(m) : bad(`${m}  ${extra}`))

// ---------------------------------------------------------------- 资产

function buildAssets() {
  rmSync(SANDBOX, { recursive: true, force: true })
  mkdirSync(join(CHAR_DIR, 'Portraits'), { recursive: true })
  const icon = join(DIST, 'icon.png')
  for (const n of ['p1.png', 'p2.png']) copyFileSync(icon, join(CHAR_DIR, 'Portraits', n))

  writeFileSync(join(CHAR_DIR, `${CHAR}_config.json`), JSON.stringify({
    config_version: '9',
    character_info: {
      name: CHAR, sex: 'female', age: 18,
      hair: 'silver hair', eyes: 'red eyes', race: 'elf',
    },
    description: '截图用角色',
  }, null, 2))

  const frames = [1, 2, 3, 4]
  const poses = join(CHAR_DIR, 'Sequence', 'poses', '走路')
  mkdirSync(poses, { recursive: true })
  for (const i of frames) copyFileSync(icon, join(poses, `f_00${i}.png`))
  writeFileSync(join(poses, 'set.json'), JSON.stringify({
    name: '走路', created: 1794220000000, source: { templateId: 'walk' },
    frames: frames.map((i) => ({
      index: i, t: (i - 1) * 0.15, yaw: 0, pose: {}, file: `f_00${i}.png`,
    })),
  }, null, 2))

  const render = join(CHAR_DIR, 'Sequence', 'pose_frames', '走路-成片')
  mkdirSync(render, { recursive: true })
  for (const i of frames) copyFileSync(icon, join(render, `f_00${i}_c01.png`))
  writeFileSync(join(render, 'render.json'), JSON.stringify({
    name: '走路-成片', created: 1794220100000, poseSet: '走路', prompt: '',
    params: {},
    frames: frames.map((i) => ({
      index: i, source: `/files/characters/${CHAR}/Sequence/poses/走路/f_00${i}.png`,
      candidates: [{ index: 1, file: `f_00${i}_c01.png`, at: 1794220100000, seed: 42, size: [128, 128] }],
      selected: 0,
    })),
  }, null, 2))

  writeFileSync(join(CHAR_DIR, 'Sequence', 'studio.json'), JSON.stringify({
    pose: null, yaw: 0, scene: null,
  }))
}

// ---------------------------------------------------------------- 服务

function startApi() {
  const logs = []
  const child = spawn(PY, ['-m', 'uvicorn', 'app:app', '--host', '127.0.0.1',
    '--port', String(API_PORT), '--log-level', 'warning'], {
    cwd: ROOT,
    env: {
      ...process.env,
      MEOW_ASSETS_DIR: ASSETS,
      MEOW_TEMPLATES_DIR: join(SANDBOX, 'templates'),
      MEOW_LIBRARY_DIR: join(SANDBOX, 'library'),
      HF_HUB_OFFLINE: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  const take = (b) => logs.push(String(b))
  child.stdout.on('data', take)
  child.stderr.on('data', take)
  return { child, logs }
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
  '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.woff': 'font/woff',
  '.glb': 'model/gltf-binary', '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream', '.gif': 'image/gif', '.ico': 'image/x-icon',
}

async function startWeb() {
  const srv = createServer(async (req, res) => {
    const raw = req.url || '/'
    const pathname = decodeURIComponent(raw.split('?')[0])
    const file = resolve(DIST, '.' + (pathname === '/' ? '/index.html' : pathname))
    if (file.startsWith(DIST + sep) && existsSync(file) && !file.endsWith(sep)) {
      res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' })
      res.end(readFileSync(file))
      return
    }
    try {
      const headers = { ...req.headers }
      delete headers.host
      const chunks = []
      for await (const c of req) chunks.push(c)
      const r = await fetch(`http://127.0.0.1:${API_PORT}${raw}`, {
        method: req.method, headers,
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
  return new Promise((ok2) => srv.listen(WEB_PORT, '127.0.0.1', () => ok2(srv)))
}

// 固定端口，绝不能连到上一轮残留的进程上（否则断言全对、页面其实跑在旧资产目录）
async function assertPortFree(port, label) {
  try {
    const r = await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(2000) })
    throw new Error(`${label}端口 ${port} 已被占用（应答 ${r.status}），先关掉残留进程再跑`)
  } catch (e) {
    if (String(e.message).includes('已被占用')) throw e
    // 连接被拒 = 端口空闲
  }
}

async function waitHealthy(child, ms = 90000) {
  const t0 = Date.now()
  while (Date.now() - t0 < ms) {
    if (child.exitCode !== null) throw new Error(`后端退出（code ${child.exitCode}）`)
    try {
      const r = await fetch(`http://127.0.0.1:${API_PORT}/characters`)
      if (r.ok) return
    } catch { /* 还没起来 */ }
    await sleep(400)
  }
  throw new Error('后端启动超时')
}

// ---------------------------------------------------------------- 页面

async function main() {
  buildAssets()
  mkdirSync(OUT, { recursive: true })
  await assertPortFree(API_PORT, '后端')
  await assertPortFree(WEB_PORT, '静态')
  const api = startApi()
  const web = await startWeb()
  const browser = await puppeteer.launch({
    executablePath: EDGE, headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--hide-scrollbars',
      '--force-device-scale-factor=1', '--use-gl=swiftshader'],
    defaultViewport: { width: 1600, height: 950, deviceScaleFactor: 1 },
  })
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e)))
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
  page.on('response', (r) => {
    if (r.status() >= 400) errors.push(`HTTP ${r.status()} ${r.request().method()} ${r.url()}`)
  })
  const click = async (sel, text) => page.evaluate((s, t) => {
    const el = [...document.querySelectorAll(s)].find((x) => (x.textContent || '').includes(t))
    if (!el) return false
    el.click()
    return true
  }, sel, text)
  const shot = async (name, wait = 800) => {
    await sleep(wait)
    await page.screenshot({ path: join(OUT, name + '.png') })
    console.log('  ✓', name)
  }

  try {
    await waitHealthy(api.child)
    await page.goto(`http://127.0.0.1:${WEB_PORT}/index.html?editor=1`,
      { waitUntil: 'networkidle2', timeout: 60000 })
    await sleep(800)

    if (!(await click('.view-tab', '角色中心'))) throw new Error('找不到「角色中心」')
    await page.waitForSelector('.charlib-card', { timeout: 20000 })
    await page.evaluate(() => document.querySelector('.charlib-card')?.click())
    await sleep(700)
    if (!(await click('.cc-stage-row button', '进入设定'))) throw new Error('进不了阶段')
    await sleep(800)

    const navText = await page.evaluate(() =>
      [...document.querySelectorAll('.cc-viewnav button')].map((b) => b.textContent?.trim()).join('/'))
    check(navText.includes('序列帧动画'), `导航含序列帧动画（${navText}）`)
    check(navText.indexOf('序列帧动画') > navText.indexOf('角色设定')
      && navText.indexOf('序列帧动画') < navText.indexOf('拆部件'),
    '序列帧动画排在角色设定之后、拆部件之前', navText)

    if (!(await click('.cc-viewnav button', '序列帧动画'))) throw new Error('点不到序列帧动画')
    await sleep(2600)
    await shot('01-studio', 400)

    const studio = await page.evaluate(() => {
      const stage = document.querySelector('.cc-seq .pose-stage')
      const main = document.querySelector('.chars-body.cc-panel')
      const r = stage?.getBoundingClientRect()
      return {
        hasSeq: !!document.querySelector('.cc-seq'),
        stageH: r ? Math.round(r.height) : 0,
        stageW: r ? Math.round(r.width) : 0,
        vh: window.innerHeight,
        overflowX: main ? main.scrollWidth - main.clientWidth : 0,
        panes: document.querySelectorAll('.cc-seq-pane').length,
        hidden: document.querySelectorAll('.cc-seq-pane.is-hidden').length,
        scrollable: !!main && main.scrollHeight > main.clientHeight + 4,
        canScroll: !!main && (main.scrollTop = 200, main.scrollTop > 0),
      }
    })
    check(studio.hasSeq, '序列帧动画页渲染出来了')
    check(studio.overflowX <= 1, '页面没有横向溢出', `overflowX=${studio.overflowX}`)
    check(studio.stageH >= studio.vh * 0.4 && studio.stageH <= studio.vh * 0.8,
      '人偶舞台高度合理（40%~80% 视口）', `${studio.stageH}/${studio.vh}`)
    check(studio.stageW > 600, '人偶舞台铺开了（不是节点宽度）', `${studio.stageW}px`)
    check(studio.panes === 2 && studio.hidden === 1, '两块内容常驻，只隐藏一块')
    check(studio.scrollable && studio.canScroll, '人偶页可以上下滚动（抽帧控制条在下方）',
      `scrollable=${studio.scrollable} canScroll=${studio.canScroll}`)
    // 滚到底拍一张：确认姿态库 / 抽帧控制条真的能看到、没被 fixed 元素盖住
    await page.evaluate(() => {
      const m = document.querySelector('.chars-body.cc-panel')
      if (m) m.scrollTop = m.scrollHeight
    })
    await shot('03-studio-controls', 500)

    await click('.cc-seq-tabs button', '参考图批量帧')
    await sleep(1000)
    await shot('02-batch', 400)
    const batch = await page.evaluate(() => {
      const cfg = document.querySelector('.cc-seq-cfg')
      const panel = document.querySelector('.cc-seq .pose-batch')
      const rects = [...(cfg?.querySelectorAll('label') || [])].map((el) => {
        const r = el.getBoundingClientRect()
        return { x: r.x, y: r.y, w: r.width, h: r.height, text: (el.textContent || '').slice(0, 8) }
      })
      let overlap = 0
      for (let i = 0; i < rects.length; i++) {
        for (let j = i + 1; j < rects.length; j++) {
          const a = rects[i], b = rects[j]
          if (a.x < b.x + b.w - 1 && b.x < a.x + a.w - 1
            && a.y < b.y + b.h - 1 && b.y < a.y + a.h - 1) overlap++
        }
      }
      const pr = panel?.getBoundingClientRect()
      return {
        cfg: !!cfg, panel: !!panel,
        labels: rects.length,
        overlap,
        panelW: pr ? Math.round(pr.width) : 0,
        outOfView: rects.filter((r) => r.x < 0 || r.x + r.w > window.innerWidth + 1).length,
        history: document.querySelectorAll('.cc-seq-history button').length,
        poseSetOptions: document.querySelectorAll('.pose-batch select option').length,
        setName: document.querySelector('.pose-batch select')?.value || '',
        srcCells: document.querySelectorAll('.pose-batch-strip .pose-batch-cell').length,
        outName: document.querySelector('.pose-batch-out')?.value || '',
        stageImgs: [...document.querySelectorAll('.pose-batch-pair img')].map((im) => {
          const r = im.getBoundingClientRect()
          return { src: im.getAttribute('src'), nw: im.naturalWidth,
            x: Math.round(r.x), w: Math.round(r.width), op: getComputedStyle(im).opacity }
        }),
      }
    })
    check(batch.cfg && batch.panel, '批量帧：设置条与姿势集面板都在')
    check(batch.labels >= 5 && batch.overlap === 0, '设置条控件换行排布、不互相覆盖',
      `labels=${batch.labels} overlap=${batch.overlap}`)
    check(batch.outOfView === 0, '设置条控件都在视口内', `out=${batch.outOfView}`)
    check(batch.panelW > 800, '批量面板铺满内容区', `${batch.panelW}px`)
    check(batch.history >= 1, '本阶段成片列表出现在页面上', `${batch.history}`)
    check(batch.poseSetOptions >= 2, '姿势集下拉读到本阶段的姿势集', `${batch.poseSetOptions}`)
    check(batch.stageImgs.length >= 2
      && batch.stageImgs.every((i) => i.nw > 0 && i.w > 10 && i.op !== '0'),
    '左右对照两张图都真的画出来了', JSON.stringify(batch.stageImgs))
    check(batch.setName === '走路', '默认选中最近一套姿势集', batch.setName)
    check(batch.srcCells >= 4, '源帧缩略条出来了（先看清这套动作）', `${batch.srcCells}`)
    check(batch.outName === '走路-成片', '输出文件夹名预填最近一套成片', batch.outName)

    check(errors.length === 0, '页面无 JS 报错', errors[0] || '')
  } finally {
    if (failures) console.log('\n--- 后端日志 ---\n' + api.logs.join('').slice(-4000))
    await browser.close()
    web.close()
    api.child.kill()
  }
  if (failures) {
    console.log(`\n${failures} 项不通过`)
    process.exit(1)
  }
  console.log('\n全部通过。截图：' + OUT)
}

main().catch((e) => { console.error(e); process.exit(1) })
