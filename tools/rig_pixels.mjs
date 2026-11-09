/**
 * 场景渲染的浏览器像素核对（真 Edge / 真 Canvas2D）。
 *
 *   cd web && npx vite build --config ../tools/rigshot/vite.config.mjs
 *   node tools/rig_pixels.mjs
 *
 * 页面里跑的是产品自己的模块（io / core/scene / scenePose / rigRender），所以这里拿到的
 * 像素就是编辑器画布上的像素。核对四件事：
 *   ① 舞台 + 演员真的画出来了（不透明像素量、包围盒、通道不单一）；
 *   ② 演员在画面上的位置与几何投影一致（轮廓框 vs 光栅包围盒）；
 *   ③ 点选：点演员身上命中它，点空白不命中（走产品的 pickActorIndex）；
 *   ④ 演员平移 150 → 质心按 zoom 平移（走锚骨换算）。
 *
 * 产物：tests/_out/ui_review/rig/<tag>/NN-*.png + report.json
 */
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..')
const DIST = resolve(ROOT, 'tests', '_out', 'rig_harness')
const DEMO = resolve(ROOT, 'tests', '_out', 'rig_examples')
const TAG = process.argv[2] || 'scene'
const OUT = join(ROOT, 'tests', '_out', 'ui_review', 'rig', TAG)
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
// puppeteer 装在 tools/shot（那里的 node_modules 不进库），这里借它的解析根。
const requireFromShot = createRequire(join(ROOT, 'tools', 'shot', 'package.json'))
const puppeteer = requireFromShot('puppeteer-core')

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg',
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

function send(res, file, type) {
  res.writeHead(200, { 'Content-Type': type || 'application/octet-stream' })
  res.end(readFileSync(file))
}

async function serve(port) {
  const srv = createServer((req, res) => {
    const pathname = decodeURIComponent((req.url || '/').split('?')[0])
    const pick = (base, p) => {
      const f = resolve(base, '.' + (p === '/' ? '/index.html' : p))
      return f.startsWith(base + sep) && existsSync(f) && !f.endsWith(sep) ? f : null
    }
    if (pathname.startsWith('/demo/')) {
      const f = pick(DEMO, pathname.slice('/demo'.length))
      if (f) return send(res, f, MIME[extname(f)])
      return res.writeHead(404).end('no demo file')
    }
    const f = pick(DIST, pathname)
    if (f) return send(res, f, MIME[extname(f)])
    res.writeHead(404).end('not found')
  })
  return new Promise((ok) => srv.listen(port, '127.0.0.1', () => ok(srv)))
}

const near = (a, b, tol) => Math.abs(a - b) <= tol

async function main() {
  if (!existsSync(join(DIST, 'index.html'))) {
    console.log('先构建核对页：cd web && npx vite build --config ../tools/rigshot/vite.config.mjs')
    process.exit(2)
  }
  for (const d of ['rigbytron', 'nila']) {
    if (!existsSync(join(DEMO, d, 'project.json'))) {
      console.log(`[skip] 缺少示例工程：${join(DEMO, d)}`)
      process.exit(0)
    }
  }
  mkdirSync(OUT, { recursive: true })
  const srv = await serve(8137)
  const browser = await puppeteer.launch({
    executablePath: EDGE, headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--hide-scrollbars', '--force-device-scale-factor=1'],
    defaultViewport: { width: 940, height: 680, deviceScaleFactor: 1 },
  })
  const page = await browser.newPage()
  const errors = []
  const missing = []
  page.on('pageerror', (e) => errors.push(String(e)))
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
  page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()) })

  await page.goto('http://127.0.0.1:8137/index.html', { waitUntil: 'networkidle2', timeout: 60000 })
  await page.waitForFunction('window.__rigshot && window.__rigshot.info', { timeout: 30000 })
  const failures = []
  const bad = (m) => { failures.push(m); console.log('  FAIL', m) }
  const ok = (m) => console.log('  ok  ', m)

  const info = await page.evaluate(() => window.__rigshot.info())
  console.log(`舞台 ${info.stage.bones} 骨 / ${info.stage.slots} 槽，镜头 ${info.stage.shot}；`
    + `演员 ${info.actor.bones} 骨 / ${info.actor.slots} 槽，片段 ${info.actor.clip}`)
  if (info.warnings.length) console.log('  警告:', info.warnings.join('；'))
  if (!info.actor.bones) bad('演员没有载入（0 骨）')

  // ① 基础画面（舞台 + 演员）
  const fit = await page.evaluate(() => window.__rigshot.fit())
  const canvasShot = async (name) => {
    await sleep(120)
    const el = await page.$('#c')
    await el.screenshot({ path: join(OUT, name + '.png') })
    console.log('  ✓', name + '.png')
  }
  const m0 = await page.evaluate(() => window.__rigshot.metrics())
  await canvasShot('01-scene')
  if (m0.pixels < 5000) bad(`画面几乎空白（不透明像素 ${m0.pixels}）`)
  else ok(`画面有内容：${m0.pixels} 像素，包围盒 ${JSON.stringify(m0.box)}`)
  const spread = (m0.rgb.r[1] - m0.rgb.r[0]) + (m0.rgb.g[1] - m0.rgb.g[0]) + (m0.rgb.b[1] - m0.rgb.b[0])
  if (spread < 60) bad(`颜色几乎单一（通道跨度 ${spread}），可能只有一块纯色`)
  else ok(`颜色跨度 ${spread}（不是纯色块）`)

  // 只画演员：量它自己的位置（全局质心会被不动的舞台稀释）
  const a0 = await page.evaluate(() => { window.__rigshot.renderActorOnly(0); return window.__rigshot.metrics() })
  if (a0.pixels < 2000) bad(`演员几乎没画出来（${a0.pixels} 像素）`)
  else ok(`演员单独有 ${a0.pixels} 像素，包围盒 ${JSON.stringify(a0.box)}`)

  // ② 几何投影 vs 实际光栅：贴图有透明边，光栅只会比几何框小，不会超出
  const sil = await page.evaluate(() => window.__rigshot.actorSilhouette())
  if (!sil.box) bad('演员没有任何可见四边形')
  else {
    const [ax0, ay0, ax1, ay1] = sil.box
    const [bx0, by0, bx1, by1] = a0.box
    // 贴图有透明边、加上 Canvas2D 在仿射变换下的边缘抗锯齿，允许几像素出入。
    const tol = 4
    const inside = bx0 >= ax0 - tol && by0 >= ay0 - tol && bx1 <= ax1 + tol && by1 <= ay1 + tol
    const ratio = ((bx1 - bx0) * (by1 - by0)) / Math.max(1, (ax1 - ax0) * (ay1 - ay0))
    const dcx = (bx0 + bx1) / 2 - (ax0 + ax1) / 2
    const dcy = (by0 + by1) / 2 - (ay0 + ay1) / 2
    if (!inside) bad(`演员光栅 ${JSON.stringify(a0.box)} 超出了几何投影 ${JSON.stringify(sil.box)}`)
    else if (Math.hypot(dcx, dcy) > 6) bad(`演员光栅中心偏离几何中心 ${Math.hypot(dcx, dcy).toFixed(1)}px`)
    else if (ratio < 0.35) bad(`演员光栅只占几何投影的 ${(ratio * 100).toFixed(0)}%，可能贴图没贴上`)
    else ok(`几何投影 ${Math.round(ax1 - ax0)}×${Math.round(ay1 - ay0)} 包含光栅（占 ${(ratio * 100).toFixed(0)}%，中心偏 ${Math.hypot(dcx, dcy).toFixed(1)}px）`)
  }

  // ③ 点选：点演员身上命中它，画面角落不命中
  const cx = (a0.box[0] + a0.box[2]) / 2
  const cy = (a0.box[1] + a0.box[3]) / 2
  const hit = await page.evaluate((x, y) => window.__rigshot.pickAtScreen(x, y), cx, cy)
  const miss = await page.evaluate(() => window.__rigshot.pickAtScreen(4, 4))
  if (hit !== 0) bad(`点演员中心没命中（返回 ${hit}）`)
  else if (miss !== -1) bad(`点空白误命中（返回 ${miss}）`)
  else ok(`点选：演员中心 → ${hit}，空白 → ${miss}`)

  // ④ 选中态要画出来（黄色选框），并且比不选中时多出像素
  await page.evaluate(() => window.__rigshot.render(0, 0))
  const mSel = await page.evaluate(() => window.__rigshot.metrics())
  await canvasShot('02-actor-selected')
  if (mSel.pixels <= m0.pixels) bad('选中后画面没有多出选框（像素数没变化）')
  else ok(`选中框已绘制（+${mSel.pixels - m0.pixels} 像素）`)

  // ⑤ 平移：锚骨 +150 → 演员自己的质心按 zoom 平移
  await page.evaluate(() => {
    window.__rigshot.moveActorBy(150, 0)
    window.__rigshot.render(0, null)
  })
  const mFull = await page.evaluate(() => window.__rigshot.metrics())
  await canvasShot('03-actor-moved')
  const a1 = await page.evaluate(() => { window.__rigshot.renderActorOnly(0); return window.__rigshot.metrics() })
  const zoom = fit?.zoom ?? info.view.zoom
  const dx = a1.centroid[0] - a0.centroid[0]
  if (!near(dx, 150 * zoom, 1.5)) bad(`演员平移后质心偏移 ${dx.toFixed(2)}px，期望 ${(150 * zoom).toFixed(2)}px`)
  else ok(`演员平移 150 → 画面偏移 ${dx.toFixed(2)}px（zoom ${zoom.toFixed(3)}，全图 ${mFull.pixels} 像素）`)

  // ⑥ 换一帧：动画在播时演员姿势必须有变化（否则是「定格」）
  await page.evaluate(() => window.__rigshot.moveActorBy(-150, 0))
  const f0 = await page.evaluate(() => { window.__rigshot.renderActorOnly(0); return window.__rigshot.metrics() })
  const f1 = await page.evaluate(() => { window.__rigshot.renderActorOnly(0.9); return window.__rigshot.metrics() })
  const moved = Math.hypot(f1.centroid[0] - f0.centroid[0], f1.centroid[1] - f0.centroid[1])
  const sizeDelta = Math.abs(f1.pixels - f0.pixels)
  if (moved < 0.5 && sizeDelta < 200) bad('t=0 与 t=0.9 画面几乎一致（疑似定格）')
  else ok(`换帧有变化：质心 ${moved.toFixed(1)}px、像素差 ${sizeDelta}`)

  writeFileSync(join(OUT, 'report.json'), JSON.stringify({
    info, fit, frame0: m0, actorOnly: a0, selected: mSel, moved: a1, silhouette: sil,
    pick: { hit, miss }, missing, pageErrors: errors,
  }, null, 2))

  if (errors.length) {
    console.log('\n页面报错:')
    for (const e of errors.slice(0, 5)) console.log('  ', e)
  }
  await browser.close()
  srv.close()
  console.log(`\n产物：${OUT}`)
  console.log(failures.length ? `\n不通过（${failures.length} 项）` : '\n通过：浏览器里的场景画面、演员位置、点选与平移都对得上')
  process.exit(failures.length || errors.length ? 1 : 0)
}

main().catch((e) => { console.error(e); process.exit(1) })
