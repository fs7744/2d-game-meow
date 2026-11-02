/**
 * 真机验证「抽帧 → 挑帧」这条交互（人偶编辑器里）。
 *
 * 照 pixelforge 的「拆解帧 · 挑帧」改的，验证的就是那几条**只有真浏览器能验**的语义：
 *
 *  1. 抽帧**不落盘**，先摆成帧墙给你看（以前是点了就闷头存盘，什么也看不见）；
 *  2. 两把筛子（跳帧步长 / 近似度）**只改「显示哪些」，不动勾选**；
 *  3. 「应用筛选」才把筛选结果变成勾选；
 *  4. 点格子能勾/取消，序号按**帧序**显示；
 *  5. 「存为姿势集（N）」上的 N 跟着勾选变。
 *
 * 怎么跑：起一个静态服务（只有 `web/dist`），用 puppeteer 驱动系统 Edge，
 * 把一段真动画（three 的 Soldier.glb）喂给节点里的 iframe，然后**像用户一样点界面**。
 *
 *     cd web && npm run build
 *     node tools/verify_frame_pick.mjs
 *
 * 依赖 puppeteer-core（隔离装在 `tools/shot/node_modules`，驱动系统 Edge，不下载 Chromium）。
 */
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..')
const DIST = join(ROOT, 'web', 'dist')
const OUT = join(ROOT, 'tests', '_out', 'anim_import')
const MESH = join(OUT, 'mesh')
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
const MODEL_URL = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/Soldier.glb'
/** 这条的轨道名是 `mixamo.com` / `Take 001`（导出工具的残留），用来验显示名清洗 */
const FBX_URL = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/fbx/Samba%20Dancing.fbx'

const requireFromShot = createRequire(join(ROOT, 'tools', 'shot', 'package.json'))
let puppeteer
try {
  puppeteer = requireFromShot('puppeteer-core')
} catch {
  console.error('缺少 puppeteer-core：先 cd tools/shot && npm install puppeteer-core')
  process.exit(2)
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.webp': 'image/webp', '.glb': 'model/gltf-binary',
  '.wasm': 'application/wasm', '.bin': 'application/octet-stream',
  '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
}

async function download(dst, url) {
  if (existsSync(dst) && readFileSync(dst).length > 1024) return
  mkdirSync(dirname(dst), { recursive: true })
  console.log(`  下载 ${dst.split(/[\\/]/).pop()} …`)
  const r = await fetch(url)
  if (!r.ok) throw new Error(`取模型失败 ${r.status}`)
  writeFileSync(dst, Buffer.from(await r.arrayBuffer()))
}

function serve(port) {
  const srv = createServer((req, res) => {
    const url = decodeURIComponent((req.url || '/').split('?')[0])
    if (url.startsWith('/__model/')) {
      const p = join(MESH, url.slice('/__model/'.length))
      if (p.startsWith(MESH) && existsSync(p)) {
        const body = readFileSync(p)
        res.writeHead(200, { 'Content-Type': 'application/octet-stream', 'Content-Length': body.length })
        res.end(body)
        return
      }
      res.writeHead(404).end('no model')
      return
    }
    let file = join(DIST, url === '/' ? 'index.html' : url)
    if (!file.startsWith(DIST + sep) && file !== join(DIST, 'index.html')) {
      res.writeHead(403).end('forbidden')
      return
    }
    if (!existsSync(file)) { res.writeHead(404).end('not found'); return }
    const body = readFileSync(file)
    res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' })
    res.end(body)
  })
  return new Promise((ok) => srv.listen(port, '127.0.0.1', () => ok(srv)))
}

/** 在页面里设置 range 的值（React 只认原生 setter + input 事件） */
const SET_RANGE = `(sel, v) => {
  const el = document.querySelector(sel)
  if (!el) return false
  const set = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
  set.call(el, String(v))
  el.dispatchEvent(new Event('input', { bubbles: true }))
  el.dispatchEvent(new Event('change', { bubbles: true }))
  return true
}`

async function main() {
  if (!existsSync(join(DIST, 'index.html'))) {
    console.error('先 cd web && npm run build（web/dist 不存在）')
    return 2
  }
  mkdirSync(MESH, { recursive: true })
  await download(join(MESH, 'Soldier.glb'), MODEL_URL)
  await download(join(MESH, 'SambaDancing.fbx'), FBX_URL)

  const port = 8161
  const srv = await serve(port)
  const browser = await puppeteer.launch({
    executablePath: EDGE,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-swiftshader',
      '--use-gl=angle', '--use-angle=swiftshader', '--window-size=1400,1100'],
  })
  let bad = 0
  const results = []
  const expect = (cond, label, detail) => {
    results.push({ label, ok: !!cond, detail })
    if (!cond) bad++
    console.log(`${cond ? '✓' : '✗'} ${label}${detail ? '  ' + detail : ''}`)
  }

  try {
    const page = await browser.newPage()
    await page.setViewport({ width: 1400, height: 1100 })
    page.on('pageerror', (e) => console.log('  [pageerror]', e.message))

    // 只 mock 后端必需的几个端点（这个用例不碰生成，也不需要真项目）
    await page.setRequestInterception(true)
    page.on('request', (r) => {
      const u = new URL(r.url())
      if (u.origin !== `http://127.0.0.1:${port}`) return r.continue()
      const p = u.pathname
      const json = (o) => r.respond({
        status: 200, contentType: 'application/json', body: JSON.stringify(o),
      })
      if (p === '/projects') return json({ items: [{ id: 'pdemo', name: '演示项目', count: 0, at: 1 }] })
      if (p === '/projects/pdemo') {
        return json({ id: 'pdemo', name: '演示项目', viewport: { x: 0, y: 0, z: 1 }, nodes: [] })
      }
      if (p === '/projects/pdemo/pose_sets') return json({ count: 0, items: [] })
      if (p === '/workspace') return json({ ok: true })
      if (p === '/library') return json({ prompts: [], poses: [], prefs: {} })
      if (p === '/models') return json({ refiner: false, items: [], recommend: {}, caps: {} })
      if (p === '/pose_dirs') return json({ count: 0, items: [] })
      if (p.startsWith('/pose_library')) return json({ count: 0, items: [] })
      return r.continue()
    })

    console.log('打开宿主页…')
    await page.goto(`http://127.0.0.1:${port}/?demo=1`, { waitUntil: 'networkidle2' })
    await new Promise((r) => setTimeout(r, 1200))

    // 加一个「人偶姿势」节点
    const added = await page.evaluate(() => {
      const b = [...document.querySelectorAll('.rail-add')]
        .find((x) => (x.closest('.rail-item')?.textContent || '').includes('人偶姿势'))
      if (!b) return false
      b.click()
      return true
    })
    expect(added, '能加「人偶姿势」节点')
    await new Promise((r) => setTimeout(r, 800))

    // 等 iframe 里的编辑器就绪（37MB 人偶模型，软件渲染十几秒）
    console.log('等人偶编辑器就绪…')
    await page.waitForFunction(() => {
      const f = document.querySelector('.pose-stage iframe')
      return !!f && !!f.contentWindow
    }, { timeout: 60000 })
    // 宿主在 `!ready && !err` 时挂一个 `.pose-status`；它消失 = ready（或报错）
    await page.waitForFunction(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))
      return !!n && !n.querySelector('.pose-status')
    }, { timeout: 180000 })

    // 喂一段真动画给 iframe（宿主那层文件上传在 jsdom 里测过，这里直接走协议）
    const fed = await page.evaluate(async () => {
      const f = document.querySelector('.pose-stage iframe')
      const buf = await fetch('/__model/Soldier.glb').then((r) => r.arrayBuffer())
      return await new Promise((resolve) => {
        const onMsg = (e) => {
          if (e.source !== f.contentWindow) return
          const d = e.data || {}
          if (d.type === 'animLoaded' || d.type === 'error') {
            window.removeEventListener('message', onMsg)
            resolve(d.type === 'animLoaded' ? { clips: (d.clips || []).length } : { err: d.message })
          }
        }
        window.addEventListener('message', onMsg)
        f.contentWindow.postMessage({ type: 'animLoad', buffer: buf, name: 'Soldier.glb' }, '*', [buf])
        setTimeout(() => resolve({ err: '超时' }), 120000)
      })
    })
    expect(fed.clips >= 1, '动画载入 iframe', JSON.stringify(fed))
    await new Promise((r) => setTimeout(r, 600))

    // ---- 抽帧 ----
    const clickedSample = await page.evaluate(() => {
      const b = [...document.querySelectorAll('.pose-frames .pose-batch-btn')]
        .find((x) => (x.textContent || '').includes('抽帧看看'))
      if (!b) return false
      b.click()
      return true
    })
    expect(clickedSample, '有「抽帧看看」按钮（抽了先看，不落盘）')
    await page.waitForFunction(() => document.querySelectorAll('.anim-wall .anim-cell').length > 0,
      { timeout: 120000 })
    await new Promise((r) => setTimeout(r, 300))

    const wall = await page.evaluate(() => ({
      cells: document.querySelectorAll('.anim-wall .anim-cell').length,
      picked: document.querySelectorAll('.anim-wall .anim-cell.on').length,
      orders: document.querySelectorAll('.anim-wall .anim-order').length,
      count: (document.querySelector('.anim-count')?.textContent || '').replace(/\s+/g, ' ').trim(),
      saveText: ([...document.querySelectorAll('.pose-frames .pose-batch-btn')]
        .map((b) => b.textContent || '').find((t) => t.includes('存为姿势集')) || ''),
    }))
    expect(wall.cells === 8, '默认抽 8 帧、帧墙 8 格', String(wall.cells))
    expect(wall.picked === 8, '默认全选（用户预期是"抽多少存多少"）', String(wall.picked))
    expect(wall.orders === 8, '每格显示"第几帧"的顺序号', String(wall.orders))
    expect(wall.count.includes('已选 8 / 共 8'), '统计行显示已选/共', wall.count)
    expect(wall.saveText.includes('（8）'), '保存按钮上带勾选数', wall.saveText)

    // ---- 筛子只改显示，不动勾选 ----
    await page.evaluate(`(${SET_RANGE})('.anim-slider input[type=range]', 2)`)
    await new Promise((r) => setTimeout(r, 300))
    const afterStep = await page.evaluate(() => ({
      cells: document.querySelectorAll('.anim-wall .anim-cell').length,
      count: (document.querySelector('.anim-count')?.textContent || '').replace(/\s+/g, ' ').trim(),
      saveText: ([...document.querySelectorAll('.pose-frames .pose-batch-btn')]
        .map((b) => b.textContent || '').find((t) => t.includes('存为姿势集')) || ''),
    }))
    expect(afterStep.cells === 4, '步长 2 → 帧墙只显示 4 格', String(afterStep.cells))
    // **这条是这套交互的核心**：筛子只改"显示哪些"，勾选是显式状态，不会被悄悄改
    expect(afterStep.count.includes('已选 8'), '筛选不动勾选（统计行仍是 8）', afterStep.count)
    expect(afterStep.saveText.includes('（8）'), '保存按钮仍是 8', afterStep.saveText)

    // 「显示全部」把滤掉的放回来（灰显），勾选不变
    await page.evaluate(() => {
      const cb = document.querySelector('.anim-showall input')
      if (cb) cb.click()
    })
    await new Promise((r) => setTimeout(r, 250))
    const afterAll = await page.evaluate(() => ({
      cells: document.querySelectorAll('.anim-wall .anim-cell').length,
      picked: document.querySelectorAll('.anim-wall .anim-cell.on').length,
    }))
    expect(afterAll.cells === 8, '「显示全部」把 8 格都放回来', String(afterAll.cells))
    expect(afterAll.picked === 8, '勾选没被筛子改动', String(afterAll.picked))

    // 点格子能取消勾选
    await page.evaluate(() => {
      document.querySelectorAll('.anim-wall .anim-cell')[2]?.click()
    })
    await new Promise((r) => setTimeout(r, 200))
    const afterToggle = await page.evaluate(() => ({
      picked: document.querySelectorAll('.anim-wall .anim-cell.on').length,
      saveText: ([...document.querySelectorAll('.pose-frames .pose-batch-btn')]
        .map((b) => b.textContent || '').find((t) => t.includes('存为姿势集')) || ''),
    }))
    expect(afterToggle.picked === 7, '点格子能取消勾选', String(afterToggle.picked))
    expect(afterToggle.saveText.includes('（7）'), '保存按钮跟着变', afterToggle.saveText)

    // 清空 → 保存禁用
    await page.evaluate(() => {
      const b = [...document.querySelectorAll('.anim-toolbar .pose-batch-btn')]
        .find((x) => (x.textContent || '').trim() === '清空')
      b?.click()
    })
    await new Promise((r) => setTimeout(r, 200))
    const afterClear = await page.evaluate(() => ({
      picked: document.querySelectorAll('.anim-wall .anim-cell.on').length,
      disabled: ([...document.querySelectorAll('.pose-frames .pose-batch-btn')]
        .find((b) => (b.textContent || '').includes('存为姿势集')) || {}).disabled,
    }))
    expect(afterClear.picked === 0, '「清空」清掉勾选')
    expect(afterClear.disabled === true, '一帧没勾时保存按钮禁用（不给无声的失败）')

    // 应用筛选 → 勾选回到 4
    await page.evaluate(() => {
      const b = [...document.querySelectorAll('.anim-toolbar .pose-batch-btn')]
        .find((x) => (x.textContent || '').includes('应用筛选'))
      b?.click()
    })
    await new Promise((r) => setTimeout(r, 200))
    const afterApply = await page.evaluate(() => ({
      picked: document.querySelectorAll('.anim-wall .anim-cell.on').length,
    }))
    expect(afterApply.picked === 4, '「应用筛选」把筛选结果变成勾选', String(afterApply.picked))

    // 截图（肉眼复核排版：小格子 + 序号 + 勾选）
    await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))
      n.scrollIntoView({ block: 'center' })
    })
    await new Promise((r) => setTimeout(r, 400))
    const clip = await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))
      const r = n.getBoundingClientRect()
      return { x: Math.max(0, r.x - 8), y: Math.max(0, r.y - 8), width: r.width + 16, height: r.height + 16 }
    })
    mkdirSync(OUT, { recursive: true })
    const shot = join(OUT, 'frame_pick.png')
    await page.screenshot({ path: shot, clip })
    console.log(`  → ${shot}`)

    // ---- 轨道名的显示清洗（用户报的 "默认 mixamo.com 是啥鬼"） ----
    // SambaDancing.fbx 的两条轨道就叫 `mixamo.com` / `Take 001`，那是导出
    // 工具留在文件里的 take 名，对用户没有任何信息量。显示层必须洗成
    // 「未命名动作」/「未命名动作 2」，否则默认姿势集就叫 `mixamo.com`。
    const fed2 = await page.evaluate(async () => {
      const f = document.querySelector('.pose-stage iframe')
      const buf = await fetch('/__model/SambaDancing.fbx').then((r) => r.arrayBuffer())
      return await new Promise((resolve) => {
        const onMsg = (e) => {
          if (e.source !== f.contentWindow) return
          const d = e.data || {}
          if (d.type === 'animLoaded' || d.type === 'error') {
            window.removeEventListener('message', onMsg)
            resolve(d.type === 'animLoaded' ? { clips: (d.clips || []).map((c) => c.name) } : { err: d.message })
          }
        }
        window.addEventListener('message', onMsg)
        f.contentWindow.postMessage({ type: 'animLoad', buffer: buf, name: 'SambaDancing.fbx' }, '*', [buf])
        setTimeout(() => resolve({ err: '超时' }), 120000)
      })
    })
    expect(fed2.clips?.length === 2, 'FBX 载入（两条轨道）', JSON.stringify(fed2))
    expect(!!fed2.clips?.includes('mixamo.com'), '这条素材的轨道名确实是脏的（mixamo.com）',
      JSON.stringify(fed2.clips))
    await new Promise((r) => setTimeout(r, 600))

    const label = await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))
      const sel = n.querySelector('.pose-anim select')
      const opts = sel ? [...sel.options].map((o) => o.textContent || '') : []
      const nameInput = n.querySelector('.pose-set-name')
      const status = n.querySelector('.pose-status')?.textContent || ''
      return {
        opts,
        placeholder: nameInput?.getAttribute('placeholder') || '',
        leaks: (n.textContent || '').includes('mixamo.com'),
        wall: n.querySelectorAll('.anim-wall .anim-cell').length,
        status,
      }
    })
    expect(label.opts.length === 2, '轨道下拉有两条', JSON.stringify(label.opts))
    expect(label.opts[0] === '未命名动作', '脏轨道名洗成「未命名动作」', label.opts[0])
    expect(label.opts[1] === '未命名动作 2', '第二条加序号区分', label.opts[1])
    // 姿势集默认名：抽一帧出来才有"存为姿势集"那一行（没抽就没得存，输入框不渲染）
    await page.evaluate(() => {
      const b = [...document.querySelectorAll('.pose-frames .pose-batch-btn')]
        .find((x) => (x.textContent || '').includes('抽帧看看'))
      b?.click()
    })
    await page.waitForFunction(() => document.querySelectorAll('.anim-wall .anim-cell').length > 0,
      { timeout: 120000 })
    const ph = await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))
      return n.querySelector('.pose-set-name')?.getAttribute('placeholder') || ''
    })
    expect(ph.includes('未命名动作'), '姿势集默认名是「未命名动作」', ph)
    expect(label.leaks === false, '界面上不该再出现 mixamo.com', '')
    // 换文件后旧帧墙必须清掉，否则会被当成新动作的帧存出去
    expect(label.wall === 0, '换动作后旧帧墙清空（避免张冠李戴）', String(label.wall))

    // 留一张「脏名字被洗干净」的截图，方便肉眼复核
    const clip2 = await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('人偶姿势'))
      n.scrollIntoView({ block: 'center' })
      const r = n.getBoundingClientRect()
      return { x: Math.max(0, r.x - 8), y: Math.max(0, r.y - 8), width: r.width + 16, height: r.height + 16 }
    })
    const shot2 = join(OUT, 'clip_name_clean.png')
    await page.screenshot({ path: shot2, clip: clip2 })
    console.log(`  → ${shot2}`)
  } finally {
    await browser.close()
    srv.close()
  }

  console.log(bad === 0 ? '\n全部通过 ✓' : `\n${bad} 项不达标 ✗`)
  return bad === 0 ? 0 : 1
}

main().then((c) => process.exit(c)).catch((e) => {
  console.error('验证脚本异常：', e)
  process.exit(1)
})
