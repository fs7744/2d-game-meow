/**
 * 骨骼编辑器「演员拖拽」的全链路核对（真后端 + 真 Edge + 真鼠标事件）。
 *
 * 之前只验过几何：`rig_pixels.mjs` 直接调 `pickActorIndex` / `transformActor`，等于
 * 把手从指针链路上拿掉了。这个脚本补的就是那一段：
 *
 *   ?rigdebug=1 打开编辑器 → 点顶栏「骨骼动画」→ 选角色 / 生命阶段
 *   → 「继续编辑」恢复舞台工程 → 「场景 / 摄影机」初始化场景
 *   → 选来源工程 + 「＋ 放到舞台」（真的 fetch 那份 .rigby）
 *   → 「适配视图」→ 在画布上按下 / 移动 / 抬起
 *
 * 断言：
 *   ① 按下落在演员身上，选中态是 {kind:'actor'}（不是骨骼）；
 *   ② 拖 dx 屏幕像素 → 锚骨世界坐标按 zoom 换算移动，误差 < 0.5 世界单位；
 *   ③ 拖动过程没顺手把舞台骨骼转掉（12 根骨 rotation 一个都没变）；
 *   ④ 画面像素真的跟着动：现在这一帧的每个演员像素，都能在「上一帧按 (dx,dy) 搬回」
 *      的位置找到（差 < 1%），且原地仍相同的不足 90%；
 *   ⑤ 点空白不再选中演员；
 *   ⑥ 「翻转」改了 actor.flip 且画面重绘；
 *   ⑦ 换「旋转」工具再拖 30° → 锚骨原地转，原点位移 < 0.01。
 *
 * 两个踩过的坑（写在这里免得下次重踩）：
 *   - 背景网格是**屏幕空间**辅助线，不跟着内容平移，比对前先关掉；
 *   - 快照要在「已选中」之后拍，否则选中框 / 名字只在一边有，凭空多出几万像素差。
 *
 * 资产完全隔离在 `tests/_out/rig_ui/assets`：用两个 rigby2d 官方示例拼出
 * 「舞台」和「演员」两份 RigAnim 产物，不碰 `assets/` 里的真实角色。
 *
 *   node tools/rig_drag.mjs
 */
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..')
const DEMO = join(ROOT, 'tests', '_out', 'rig_examples')
const SANDBOX = join(ROOT, 'tests', '_out', 'rig_ui')
const ASSETS = join(SANDBOX, 'assets')
const CHAR = 'uitest'
const CHAR_DIR = join(ASSETS, 'characters', CHAR)
const OUT = join(ROOT, 'tests', '_out', 'ui_review', 'rig', 'drag')
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
const PY = join(ROOT, '.venv', 'Scripts', 'python.exe')
const PORT = 8145
const BASE = `http://127.0.0.1:${PORT}`
const requireFromShot = createRequire(join(ROOT, 'tools', 'shot', 'package.json'))
const puppeteer = requireFromShot('puppeteer-core')

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let failures = 0
const ok = (m) => console.log('  ok   ' + m)
const bad = (m) => { failures++; console.log('  FAIL ' + m) }

// ---------------------------------------------------------------- 造资产

/** `project.json` + `images/*.png` → 一份 `.rigby`（工作文件就是 zip，见 io.saveWorkfile）。 */
async function workfileFromDemo(name, zip) {
  const src = join(DEMO, name)
  const doc = JSON.parse(readFileSync(join(src, 'project.json'), 'utf8'))
  const entries = [{
    path: 'project.json',
    bytes: new Uint8Array(readFileSync(join(src, 'project.json'))),
  }]
  for (const f of readdirSync(join(src, 'images'))) {
    if (!f.endsWith('.png')) continue
    entries.push({ path: `images/${f}`, bytes: new Uint8Array(readFileSync(join(src, 'images', f))) })
  }
  return { bytes: zip(entries), doc }
}

/**
 * 两产物：`stage_rigbytron`（当舞台）+ `actor_nila`（当演员）。
 * UI 只读 `formats['project.rigby']`，其余六个文件放占位内容即可（下载链接用得到）。
 */
async function buildFixture() {
  rmSync(join(ASSETS, 'characters', CHAR), { recursive: true, force: true })
  mkdirSync(join(CHAR_DIR, 'RigAnim'), { recursive: true })
  writeFileSync(join(CHAR_DIR, `${CHAR}_config.json`), JSON.stringify({
    config_version: '9', character_info: { sex: 'female', style: 'test' }, stages: [],
  }, null, 2))

  const zip = (await import(pathToFileURL(join(ROOT, 'web', 'src', 'rig', 'io', 'zip.js')).href)).zip
  const made = []
  for (const [id, demo, label] of [
    ['stage_rigbytron', 'rigbytron', '舞台 · rigbytron'],
    ['actor_nila', 'nila', '演员 · NILA'],
  ]) {
    const { bytes, doc } = await workfileFromDemo(demo, zip)
    const dir = join(CHAR_DIR, 'RigAnim', id)
    mkdirSync(dir, { recursive: true })
    writeFileSync(join(dir, 'project.rigby'), bytes)
    const extras = {
      'skeleton-3.8.json': '{}', 'skeleton-4.2.json': '{}', 'skeleton-4.3.json': '{}',
      'skeleton.skel': '', 'skeleton.atlas': '', 'skeleton.png': '',
    }
    for (const [n, body] of Object.entries(extras)) writeFileSync(join(dir, n), body)
    const formats = Object.fromEntries(
      ['skeleton-3.8.json', 'skeleton-4.2.json', 'skeleton-4.3.json', 'skeleton.skel',
        'skeleton.atlas', 'skeleton.png', 'project.rigby']
        .map((n) => [n, `/files/characters/${CHAR}/RigAnim/${id}/${n}`]))
    writeFileSync(join(dir, 'manifest.json'), JSON.stringify({
      id, created: Math.floor(Date.now() / 1000), stage: 'default',
      source: { kind: 'demo', label, name: demo },
      bones: (doc.bones || []).length, slots: (doc.slots || []).length,
      animations: (doc.animations || []).length, constraints: (doc.ik || []).length,
      animationNames: (doc.animations || []).map((a) => a.name),
      formats,
    }, null, 2))
    made.push({ id, label, bones: (doc.bones || []).length })
  }
  return made
}

// ---------------------------------------------------------------- 起后端

function startServer() {
  mkdirSync(SANDBOX, { recursive: true })
  const logs = []
  const child = spawn(PY, ['-m', 'uvicorn', 'app:app', '--host', '127.0.0.1',
    '--port', String(PORT), '--log-level', 'warning'], {
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

async function waitHealthy(child, ms = 90000) {
  const t0 = Date.now()
  while (Date.now() - t0 < ms) {
    if (child.exitCode !== null) throw new Error(`后端退出（code ${child.exitCode}）`)
    try {
      const r = await fetch(`${BASE}/characters`)
      if (r.ok) return
    } catch { /* 还没起来 */ }
    await sleep(400)
  }
  throw new Error('后端启动超时')
}

// ---------------------------------------------------------------- 页面操作

const mark = (page, selector, text, tag) => page.evaluate((sel, t, key) => {
  for (const el of document.querySelectorAll(sel)) {
    if (!el.textContent.includes(t)) continue
    el.setAttribute('data-t', key)
    return true
  }
  return false
}, selector, text, tag)

/** 记一份画布快照（整张 ImageData 留在页面里，后面做区域比对）。 */
const TAKE = `window.__prev = (() => {
  const c = document.querySelector('.rig-canvas')
  const im = c.getContext('2d').getImageData(0, 0, c.width, c.height)
  window.__prevIM = im
  return { w: im.width, h: im.height }
})()`

/**
 * 把「当前画面」的某个矩形，跟「上一张快照平移 (dx, dy)」比：
 * 演员整体平移时，把上一帧按 (dx, dy) 搬回来应该和现在几乎重合 → IoU 接近 1；
 * 同时跟「同坐标不搬」比 → IoU 应该明显低（说明真的动了，不是只改了数据）。
 */
const IOU = `window.__iou = (box, dx, dy) => {
  const c = document.querySelector('.rig-canvas')
  const cur = c.getContext('2d').getImageData(0, 0, c.width, c.height)
  const A = cur.data, B = window.__prevIM.data, W = cur.width, H = cur.height
  const [x0, y0, x1, y1] = box.map(Math.round)
  let inter = 0, uni = 0, same = 0, tot = 0, curN = 0, prevN = 0
  for (let y = Math.max(0, y0); y <= Math.min(H - 1, y1); y++) {
    for (let x = Math.max(0, x0); x <= Math.min(W - 1, x1); x++) {
      const i = (y * W + x) * 4 + 3
      const a = A[i] > 8
      const bx = x - dx, by = y - dy
      const b = bx >= 0 && bx < W && by >= 0 && by < H ? B[(by * W + bx) * 4 + 3] > 8 : false
      if (a) curN++
      if (b) prevN++
      if (a && b) inter++
      if (a || b) uni++
      if (a === (B[i] > 8)) same++
      tot++
    }
  }
  return {
    iou: uni ? inter / uni : 0, same: tot ? same / tot : 0, pixels: tot,
    curN, prevN, miss: curN - inter,
  }
}
`

/** 演员锚骨的世界坐标 / 视图变换 / 画布矩形（都由 ?rigdebug=1 的探针读真值）。 */
const GEOM = `(() => {
  const r = window.__rig
  const d = r.docRef.current
  const a = d.scene && d.scene.actors[0]
  if (!a) return null
  const bi = r.actorBoneIndex(0)
  const m = r.poseRef.current.world[bi]
  const o = m ? m.apply([0, 0]) : null
  const c = document.querySelector('.rig-canvas')
  const rect = c.getBoundingClientRect()
  const dpr = c.width / Math.max(1, rect.width)
  // 演员最结实的那个四边形（面积最大）的中心，用来当「按下的那一点」。
  const rt = r.actorRef.current.get(a.id)
  let best = null
  if (rt) for (const q of rt.pose.quads(rt.doc)) {
    if ((q.color ? q.color[3] : 1) <= 0) continue
    const p = q.positions
    let area = 0, cx = 0, cy = 0
    for (let i = 0; i < p.length; i++) {
      const [ax, ay] = p[i], [bx, by] = p[(i + 1) % p.length]
      const cr = ax * by - bx * ay
      area += cr; cx += (ax + bx) * cr; cy += (ay + by) * cr
    }
    area *= 0.5
    if (Math.abs(area) < 1) continue
    const centroid = [cx / (6 * area), cy / (6 * area)]
    if (!best || Math.abs(area) > best.area) best = { area: Math.abs(area), centroid }
  }
  // 演员的光栅包围盒（设备像素）：像素核对只在这个矩形里比，舞台是静止的、别掺进来。
  let wb = [Infinity, Infinity, -Infinity, -Infinity]
  if (rt) for (const q of rt.pose.quads(rt.doc)) {
    if ((q.color ? q.color[3] : 1) <= 0) continue
    for (const [x, y] of q.positions) {
      if (x < wb[0]) wb[0] = x; if (y < wb[1]) wb[1] = y
      if (x > wb[2]) wb[2] = x; if (y > wb[3]) wb[3] = y
    }
  }
  const toDev = (x, y) => [
    (x * r.view.zoom + rect.width / 2 + r.view.x) * dpr,
    (-y * r.view.zoom + rect.height / 2 + r.view.y) * dpr,
  ]
  const screenBox = wb[2] > wb[0]
    ? [toDev(wb[0], wb[3])[0], toDev(wb[0], wb[3])[1],
      toDev(wb[2], wb[1])[0], toDev(wb[2], wb[1])[1]]
    : null
  return {
    bone: bi, origin: o, zoom: r.view.zoom, view: { x: r.view.x, y: r.view.y },
    sel: r.sel ? { kind: r.sel.kind, index: r.sel.index } : null,
    actors: d.scene.actors.length, loaded: r.actorRef.current.size,
    flip: !!a.flip, pick: best ? best.centroid : null,
    rect: { left: rect.left, top: rect.top, w: rect.width, h: rect.height },
    dpr, screenBox,
    canvas: { w: c.width, h: c.height },
    rotations: d.bones.map((b) => b.setup.rotation),
    names: d.bones.map((b) => b.name),
  }
})()`

/** 只截画布本身（整页截图会带上一堆无关面板，还会被页面滚动位置影响）。 */
async function shotCanvas(page, name) {
  const el = await page.$('.rig-canvas')
  await el.screenshot({ path: join(OUT, name + '.png') })
}

/** 世界 → 页面坐标（与 types.ts::toWorld 互逆）。 */
function toScreen(g, p) {
  const w = g.rect.w, h = g.rect.h
  return {
    x: g.rect.left + (p[0] * g.zoom + w / 2 + g.view.x),
    y: g.rect.top + (-p[1] * g.zoom + h / 2 + g.view.y),
  }
}

async function main() {
  if (!existsSync(join(ROOT, 'web', 'dist', 'index.html')))
    throw new Error('web/dist 不存在，先构建：cd web && npx vite build --outDir dist')

  const made = await buildFixture()
  console.log(`资产：${made.map((m) => `${m.id}(${m.bones} 骨)`).join(' / ')}`)

  const { child, logs } = startServer()
  const stop = () => { try { child.kill() } catch { /* 已退出 */ } }
  process.on('exit', stop)

  let browser = null
  try {
    await waitHealthy(child)
    console.log(`后端 ${BASE} 就绪`)

    browser = await puppeteer.launch({
      executablePath: EDGE, headless: true,
      args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars'],
    })
    const page = await browser.newPage()
    await page.setViewport({ width: 1680, height: 1000, deviceScaleFactor: 1 })
    const errors = []
    page.on('pageerror', (e) => errors.push(String(e)))
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })

    await page.goto(`${BASE}/?editor=1&rigdebug=1`, { waitUntil: 'networkidle2' })
    await page.waitForSelector('.view-tab', { timeout: 30000 })
    if (!await mark(page, '.view-tab', '骨骼动画', 'tab')) throw new Error('找不到顶栏「骨骼动画」')
    await page.click('[data-t=tab]')
    await page.waitForSelector('.rig-studio', { timeout: 20000 })

    // 角色 / 阶段：临时资产里只有 uitest，阶段自动补「初始设定」。
    await page.waitForFunction(() => {
      const sel = document.querySelector('.rig-studio-head select')
      return !!sel && sel.value === 'uitest'
    }, { timeout: 20000 })
    ok('角色列表来自后端，默认选中 uitest')

    await page.waitForFunction(() => document.querySelectorAll('.rig-history-row').length >= 2,
      { timeout: 20000 })
    if (!await page.evaluate(() => {
      for (const row of document.querySelectorAll('.rig-history-row')) {
        if (!row.textContent.includes('舞台')) continue
        for (const b of row.querySelectorAll('button')) {
          if (b.textContent.trim() === '继续编辑') { b.setAttribute('data-t', 'resume'); return true }
        }
      }
      return false
    })) throw new Error('找不到舞台产物的「继续编辑」')
    await page.click('[data-t=resume]')
    await page.waitForSelector('.rig-canvas', { timeout: 20000 })
    await page.waitForFunction(() => window.__rig && window.__rig.docRef.current.slots.length > 0,
      { timeout: 20000 })
    ok('「继续编辑」把舞台工作文件读进了编辑器')

    // 场景：初始化 → 选来源工程 → 放到舞台（真的去 fetch 那份 .rigby）。
    if (!await mark(page, '.rig-ws-tabs button', '场景', 'wstab'))
      throw new Error('找不到「场景 / 摄影机」')
    await page.click('[data-t=wstab]')
    if (!await mark(page, '.rig-ws-panel button', '初始化场景', 'init'))
      throw new Error('找不到「初始化场景」')
    await page.click('[data-t=init]')
    await page.waitForFunction(() => !!window.__rig.docRef.current.scene, { timeout: 10000 })

    const hasSelect = await page.evaluate(() => {
      for (const label of document.querySelectorAll('.rig-ws-panel label')) {
        if (!label.textContent.includes('来源工程')) continue
        const s = label.querySelector('select')
        if (!s) continue
        s.setAttribute('data-t', 'src')
        return true
      }
      return false
    })
    if (!hasSelect) throw new Error('找不到「来源工程」下拉')
    await page.select('[data-t=src]', 'actor_nila')
    if (!await mark(page, '.rig-ws-panel button', '放到舞台', 'add'))
      throw new Error('找不到「＋ 放到舞台」')
    await page.click('[data-t=add]')
    await page.waitForFunction(
      () => window.__rig.actorRef.current.size === 1
        && window.__rig.docRef.current.scene.actors.length === 1,
      { timeout: 30000 })
    await sleep(400)
    ok('演员工程已载入并挂到舞台上（真 .rigby → 文档 + 图片）')

    // 背景网格默认开着。它是**屏幕空间**的辅助线，不跟着内容平移，
    // 会让「把上一帧按位移搬回来」到处对不上 —— 先关掉（用户平时也会关）。
    await page.evaluate(() => {
      for (const l of document.querySelectorAll('.rig-check')) {
        if (!l.textContent.includes('背景网格')) continue
        const c = l.querySelector('input')
        if (c && c.checked) c.click()
      }
    })
    await sleep(150)

    // 适配视图：演员是后加的，画布的结构变化钩子不会自动重取景。
    if (!await mark(page, '.rig-canvas-hud button', '适配视图', 'fit')) throw new Error('找不到「适配视图」')
    await page.click('[data-t=fit]')
    await sleep(300)

    let g = await page.evaluate(GEOM)
    if (!g) throw new Error('读不到演员几何（探针没挂上？）')
    if (!g.pick) throw new Error('演员没有任何可见四边形')
    // 画布可能在折叠线以下：先滚到视口中央，否则鼠标事件落在窗口外（坐标读了也没用）。
    await page.evaluate(() => document.querySelector('.rig-canvas').scrollIntoView({ block: 'center' }))
    await sleep(250)

    // 滚动后再读一次几何（rect 变了）。
    g = await page.evaluate(GEOM)
    const first = toScreen(g, g.pick)
    const vp = page.viewport()
    if (first.x < g.rect.left || first.x > g.rect.left + g.rect.w ||
        first.y < g.rect.top || first.y > g.rect.top + g.rect.h) {
      throw new Error(`演员不在可视画布内（点 ${first.x.toFixed(0)},${first.y.toFixed(0)}）`)
    }
    if (first.y < 0 || first.y > vp.height || first.x < 0 || first.x > vp.width) {
      throw new Error(`点击点 (${first.x.toFixed(0)},${first.y.toFixed(0)}) 在视口外`
        + `（视口 ${vp.width}×${vp.height}，画布 top ${g.rect.top.toFixed(0)}）`)
    }
    console.log(`  演员锚骨 (${g.origin.map((v) => v.toFixed(1)).join(', ')})`
      + ` · zoom ${g.zoom.toFixed(3)} · 画布 ${Math.round(g.rect.w)}×${Math.round(g.rect.h)}`
      + ` · 按下点 (${first.x.toFixed(0)}, ${first.y.toFixed(0)})`)

    // ① 按下就选中演员（而不是它附近的骨骼）
    await page.mouse.move(first.x, first.y)
    await sleep(120)
    await page.mouse.down()
    await sleep(250)
    g = await page.evaluate(GEOM)
    if (g.sel?.kind !== 'actor' || g.sel.index !== 0) {
      bad(`按下后选中的是 ${JSON.stringify(g.sel)}，不是演员`)
    } else ok(`按下命中演员：sel = ${JSON.stringify(g.sel)}`)

    // 快照要在「已选中」之后拍：选中框 / 四角把手 / 名字都画在画布上，
    // 两张快照带着同一套装饰，平移比对才是同图对比。
    await page.evaluate(TAKE)
    await page.evaluate(IOU)
    await shotCanvas(page, '01-before-drag')
    await sleep(500)
    const still = await page.evaluate((box) => window.__iou(box, 0, 0), g.screenBox)
    if (still.iou < 0.999) bad(`不动的时候画面就不稳（IoU ${still.iou.toFixed(3)}），平移比对无效`)
    else ok(`静止画面稳定（IoU ${still.iou.toFixed(3)}）`)

    // ② 拖动：屏幕 dx → 锚骨世界 dx/zoom
    // **只横向拖**：适配后 zoom 只有 17%，演员高度撑满画布、下沿被裁掉一截；纵向位移
    // 会让「按位移搬回上一帧」在裁切边上对不齐。纵向正确性由下面的世界坐标断言覆盖。
    const DX = -170, DY = 0
    const g0 = g
    await page.mouse.move(first.x + DX, first.y + DY, { steps: 15 })
    await sleep(200)
    await page.mouse.up()
    await sleep(300)
    g = await page.evaluate(GEOM)
    const wantX = g0.origin[0] + DX / g0.zoom
    const wantY = g0.origin[1] - DY / g0.zoom
    const errX = Math.abs(g.origin[0] - wantX), errY = Math.abs(g.origin[1] - wantY)
    if (errX > 0.5 || errY > 0.5) {
      bad(`锚骨移动到 (${g.origin.map((v) => v.toFixed(2)).join(', ')})，`
        + `期望 (${wantX.toFixed(2)}, ${wantY.toFixed(2)})，误差 ${errX.toFixed(2)}/${errY.toFixed(2)}`)
    } else ok(`拖 ${DX},${DY}px → 锚骨 ${g.origin.map((v) => v.toFixed(2)).join(', ')}`
      + `（误差 ${Math.max(errX, errY).toFixed(3)} 世界单位）`)

    // ③ 没把舞台骨骼转掉
    const rots = g.rotations.map((v, i) => Math.abs(v - g0.rotations[i]))
    const turned = rots.map((d, i) => (d > 1e-9 ? `${g.names[i]} ${d.toFixed(3)}°` : null)).filter(Boolean)
    if (turned.length) bad(`拖动演员顺手改了骨骼旋转：${turned.slice(0, 3).join('、')}`)
    else ok(`${g.rotations.length} 根骨的 setup.rotation 一个都没变`)

    // ④ 画面真的把演员平移了：演员矩形内「上一帧按 (dx,dy) 搬回来」与现在几乎重合
    await sleep(150)
    await shotCanvas(page, '02-after-drag')
    const inCanvas = g.screenBox
      && g.screenBox[0] > 2 && g.screenBox[2] < g.canvas.w - 2
      && g.screenBox[3] > 2 && g.screenBox[1] < g.canvas.h - 2
    if (!inCanvas) {
      bad(`演员光栅包围盒 ${JSON.stringify((g.screenBox || []).map(Math.round))} 横向贴边，`
        + `无法做平移比对（画布 ${g.canvas.w}×${g.canvas.h}）`)
    } else {
      const ddx = DX * g.dpr, ddy = DY * g.dpr
      const r = await page.evaluate((box, dx, dy) => window.__iou(box, dx, dy), g.screenBox, ddx, ddy)
      // 现在这一帧的每个演员像素，都应该能在上一帧「按 (dx,dy) 搬回来」的位置找到。
      // 反过来不一定成立：演员挪开后，原先被它挡住的舞台骨骼露了出来（那是应该露的）。
      if (r.miss > Math.max(20, r.curN * 0.01)) {
        bad(`当前帧有 ${r.miss}/${r.curN} 个像素在上一帧的平移位置找不到（应 < 1%）`)
      } else if (r.same > 0.9) {
        bad(`不搬的同一坐标仍有 ${(r.same * 100).toFixed(1)}% 相同（画面其实没动）`)
      } else {
        ok(`画面平移一致：${r.curN} 个演员像素里只有 ${r.miss} 个对不上，`
          + `原地仍相同的只剩 ${(r.same * 100).toFixed(1)}%`)
      }
    }

    // ⑤ 点空白不该再选中演员（截图可能把页面滚走，先重新取几何）
    await page.evaluate(() => document.querySelector('.rig-canvas').scrollIntoView({ block: 'center' }))
    await sleep(200)
    g = await page.evaluate(GEOM)
    await page.mouse.click(g.rect.left + 10, g.rect.top + 10)
    await sleep(150)
    g = await page.evaluate(GEOM)
    if (g.sel?.kind === 'actor') bad('点画布左上角仍选中演员（命中判定过宽）')
    else ok(`点空白后 sel = ${JSON.stringify(g.sel)}`)

    // ⑥ 翻转：改 actor.flip，且画面确实重绘（不动画布就只是改了个布尔值）
    if (!await mark(page, '.rig-ws-actors button', '翻转', 'flip')) throw new Error('找不到「翻转」')
    const countPixels = `(() => {
      const c = document.querySelector('.rig-canvas')
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data
      let n = 0
      for (let i = 3; i < d.length; i += 4) if (d[i] > 8) n++
      return n
    })()`
    const preFlip = await page.evaluate(countPixels)
    await page.click('[data-t=flip]')
    await sleep(400)
    g = await page.evaluate(GEOM)
    const postFlip = await page.evaluate(countPixels)
    await shotCanvas(page, '03-flipped')
    if (!g.flip) bad('点了「翻转」但 actor.flip 没变')
    else if (postFlip === preFlip) bad('翻转后可见像素数完全没变（可能没重绘）')
    else ok(`翻转生效：actor.flip = ${g.flip}，可见像素 ${preFlip} → ${postFlip}`)

    // ⑦ 旋转工具：绕锚骨转 30°（换工具后再按一次，走同一条指针链路）
    const hasRot = await page.evaluate(() => {
      for (const b of document.querySelectorAll('.rig-tool')) {
        if (b.getAttribute('title') !== '旋转') continue
        b.setAttribute('data-t', 'rot')
        return true
      }
      return false
    })
    if (!hasRot) throw new Error('找不到「旋转」工具')
    await page.click('[data-t=rot]')
    await sleep(120)
    g = await page.evaluate(GEOM)
    const rot0 = g.rotations[g.bone]
    const c0 = toScreen(g, g.origin)
    const RAD = 90
    const a0 = -Math.PI / 2
    const a1 = a0 + Math.PI / 6
    await page.mouse.move(c0.x + RAD * Math.cos(a0), c0.y + RAD * Math.sin(a0))
    await sleep(80)
    await page.mouse.down()
    await sleep(120)
    await page.mouse.move(c0.x + RAD * Math.cos(a1), c0.y + RAD * Math.sin(a1), { steps: 20 })
    await sleep(150)
    await page.mouse.up()
    await sleep(250)
    const g2 = await page.evaluate(GEOM)
    const drot = ((g2.rotations[g2.bone] - rot0) % 360 + 540) % 360 - 180
    const moved = Math.hypot(g2.origin[0] - g.origin[0], g2.origin[1] - g.origin[1])
    if (Math.abs(Math.abs(drot) - 30) > 4) {
      bad(`旋转工具拖 30° → 锚骨转了 ${drot.toFixed(2)}°（选中的是 ${JSON.stringify(g2.sel)}）`)
    } else if (moved > 0.01) {
      bad(`旋转把锚骨原点挪了 ${moved.toFixed(3)} 世界单位（应该原地转）`)
    } else {
      ok(`旋转工具：绕锚骨转 ${drot.toFixed(2)}°，原点没动，sel = ${JSON.stringify(g2.sel)}`)
    }
    await shotCanvas(page, '04-rotated')

    writeFileSync(join(OUT, 'report.json'), JSON.stringify({
      fixture: made, actors: g.actors, loaded: g.loaded,
      geometry: { bone: g.bone, zoom: g.zoom, rect: g.rect, screenBox: g.screenBox },
      drag: { dx: DX, dy: DY, originBefore: g0.origin, originAfter: g.origin },
      flip: { on: g.flip, pixels: [preFlip, postFlip] },
      pageErrors: errors,
    }, null, 2))

    if (errors.length) bad(`页面报错 ${errors.length} 条：${errors.slice(0, 2).join(' | ')}`)
    else ok('页面无 JS 报错')
  } catch (e) {
    bad(String(e && e.stack ? e.stack.split('\n').slice(0, 3).join(' ') : e))
    if (logs.length) console.log('\n后端日志：\n' + logs.join('').split('\n').slice(-12).join('\n'))
  } finally {
    if (browser) await browser.close().catch(() => {})
    stop()
  }

  console.log(failures ? `\n不通过（${failures} 项）` : '\n通过：真指针拖拽全链路一致')
  process.exit(failures ? 1 : 0)
}

mkdirSync(OUT, { recursive: true })
await main()
