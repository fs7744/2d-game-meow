/**
 * 真机验证「姿态批量」节点（`posebatch`）。
 *
 * 这个节点把三件事凑在一起，jsdom 一件都验不到：
 *
 *  1. **先看清**：选完姿势集立刻铺出源帧缩略条，`▶` 能连着播；
 *  2. **能对比**：左边原始姿势帧、右边生成结果，一屏内左右对照；
 *  3. **能交付**：GIF（给团队审核）与 Godot 4 工程包（直接进引擎）两个导出。
 *
 * 做法：静态服务 `web/dist`，用 puppeteer 驱动系统 Edge，把 `/projects*` 这几个
 * 端点 mock 成一个"已经生成过"的项目，然后**像用户一样点界面**。
 *
 *     cd web && npm run build
 *     node tools/verify_pose_batch.mjs
 */
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..')
const DIST = join(ROOT, 'web', 'dist')
const OUT = join(ROOT, 'tests', '_out', 'pose_batch')
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'

const requireFromShot = createRequire(join(ROOT, 'tools', 'shot', 'package.json'))
let puppeteer
try {
  puppeteer = requireFromShot('puppeteer-core')
} catch {
  console.error('缺少 puppeteer-core：先 cd tools/shot && npm install puppeteer-core')
  process.exit(2)
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.webp': 'image/webp',
  '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
}

/** 占位图：用 dist 里的 favicon（真 PNG，界面上能看见东西） */
const IMG = '/favicon.png'

function serve(port) {
  const srv = createServer((req, res) => {
    const url = decodeURIComponent((req.url || '/').split('?')[0])
    let file = join(DIST, url === '/' ? 'index.html' : url)
    if (existsSync(file) && statSync(file).isDirectory()) file = join(file, 'index.html')
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

const FRAMES = [1, 2, 3, 4].map((i) => ({ index: i, t: (i - 1) * 0.2, yaw: 0, url: IMG }))
const RENDER = {
  name: '走路-成片', poseSet: '走路', prompt: 'a cat knight', created: 1,
  done: 3, total: 4,
  frames: FRAMES.map((f) => ({
    index: f.index, source: f.url,
    candidates: f.index <= 3
      ? [{ index: 1, url: IMG, at: 1, seed: 11 }, { index: 2, url: IMG, at: 2, seed: 22 }]
      : [],
    selected: f.index <= 3 ? 1 : -1,
    selectedIndex: f.index <= 3 ? 2 : -1,
  })),
}

async function main() {
  if (!existsSync(join(DIST, 'index.html'))) {
    console.error('先 cd web && npm run build（web/dist 不存在）')
    return 2
  }
  const port = 8165
  const srv = await serve(port)
  const browser = await puppeteer.launch({
    executablePath: EDGE,
    headless: 'new',
    args: ['--no-sandbox', '--window-size=1400,1100'],
  })
  let bad = 0
  const expect = (cond, label, detail) => {
    if (!cond) bad++
    console.log(`${cond ? '✓' : '✗'} ${label}${detail ? '  ' + detail : ''}`)
  }

  try {
    const page = await browser.newPage()
    await page.setViewport({ width: 1400, height: 1100 })
    page.on('pageerror', (e) => console.log('  [pageerror]', e.message))

    await page.setRequestInterception(true)
    page.on('request', (r) => {
      const u = new URL(r.url())
      if (u.origin !== `http://127.0.0.1:${port}`) return r.continue()
      // **必须 decode**：姿势集名是中文，encodeURIComponent 后 pathname 里是 %E8…
      const p = decodeURIComponent(u.pathname)
      const json = (o) => r.respond({
        status: 200, contentType: 'application/json', body: JSON.stringify(o),
      })
      if (p === '/projects') return json({ items: [{ id: 'pdemo', name: '演示项目', count: 2, at: 1 }] })
      if (p === '/projects/pdemo') {
        return json({
          id: 'pdemo', name: '演示项目', viewport: { x: 120, y: 60, z: 1 },
          nodes: [
            {
              id: 'n0', kind: 'image', x: 40, y: 40, view: 'expanded', params: {},
              image: { path: 'assets/demo.png', url: IMG },
            },
            {
              id: 'n1', kind: 'posebatch', x: 420, y: 60, w: 400, view: 'expanded',
              poseRender: '走路-成片', charSource: { nodeId: 'n0' },
              params: { pcPoseSet: '走路', pcOut: '走路-成片', prompt: 'a cat knight' },
            },
          ],
        })
      }
      if (p === '/projects/pdemo/pose_sets') {
        return json({ count: 1, items: [{ name: '走路', count: 4, cover: IMG, created: 1 }] })
      }
      if (p === '/projects/pdemo/pose_sets/走路') {
        return json({ set: { name: '走路', count: 4, cover: IMG, created: 1, frames: FRAMES } })
      }
      if (p === '/projects/pdemo/pose_renders/走路-成片') {
        return json({ render: RENDER, items: [{ name: RENDER.name, poseSet: '走路', created: 1, total: 4, done: 3, cover: IMG }] })
      }
      if (p === '/projects/pdemo/pose_renders/走路-成片/gif') {
        return json({ gif: { name: '走路-成片', kind: 'render', url: IMG, frames: 3, skipped: 1, frame_ms: 120, bytes: 20480, size: [512, 512] } })
      }
      if (p === '/projects/pdemo/pose_renders/走路-成片/godot') {
        return json({
          bundle: {
            ok: true, stem: 'walk', folder: 'sprites', frames: 3, skipped: 1,
            fps: 12, loop: true, animName: '走路-成片', size: [768, 768],
            origin: 'bottom', dir: 'x',
            files: [
              { name: 'walk.tres', bytes: 900, url: IMG },
              { name: 'walk.tscn', bytes: 400, url: IMG },
            ],
            zip: { name: 'walk.zip', bytes: 40960, url: IMG },
            resPaths: { tres: 'res://sprites/walk.tres', scene: 'res://sprites/walk.tscn' },
            checks: { pathsMatch: true, missing: [], refsChecked: 4 },
            warnings: [],
          },
        })
      }
      if (p === '/workspace') return json({ ok: true })
      if (p === '/library') return json({ prompts: [], poses: [], prefs: {} })
      if (p === '/models') return json({ refiner: false, items: [], recommend: {}, caps: {} })
      if (p === '/pose_dirs') return json({ count: 0, items: [] })
      if (p.startsWith('/pose_library')) return json({ count: 0, items: [] })
      if (p === '/assets') return json({ count: 0, items: [] })
      return r.continue()
    })

    console.log('打开宿主页（不带 demo，让服务端的项目结构生效）…')
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle2' })
    await new Promise((r) => setTimeout(r, 1500))

    const node = await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('姿态批量'))
      if (!n) return null
      n.scrollIntoView({ block: 'center' })
      return { title: n.querySelector('.node-kind')?.textContent }
    })
    expect(!!node, '画布上有「姿态批量」节点', node?.title || '（没有）')
    if (!node) throw new Error('节点没渲染出来')

    await new Promise((r) => setTimeout(r, 800))
    const view = await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('姿态批量'))
      return {
        srcCells: n.querySelectorAll('.pose-batch-source .pose-batch-cell').length,
        pairFigures: n.querySelectorAll('.pose-batch-pair figure').length,
        pairImgs: [...n.querySelectorAll('.pose-batch-pair img')].map((i) => i.getAttribute('src')),
        captions: [...n.querySelectorAll('.pose-batch-pair figcaption')].map((c) => c.textContent || ''),
        gridCells: n.querySelectorAll('.pose-batch .pose-batch-grid .pose-batch-cell').length,
        exportBtns: [...n.querySelectorAll('.pose-batch-export .pose-batch-btn')].map((b) => b.textContent || ''),
      }
    })
    expect(view.srcCells === 4, '源帧缩略条 4 格（先看清是哪套动作）', String(view.srcCells))
    expect(view.pairFigures === 2, '对照视图两栏（原始 / 结果）', String(view.pairFigures))
    expect(view.pairImgs.length === 2 && view.pairImgs.every(Boolean), '两栏都出了图',
      JSON.stringify(view.pairImgs))
    expect(view.captions[0]?.includes('原始'), '左栏标注「原始」', view.captions[0] || '')
    expect(view.captions[1]?.includes('结果'), '右栏标注「结果」', view.captions[1] || '')
    expect(view.gridCells === 4, '结果缩略网格 4 格', String(view.gridCells))
    expect(view.exportBtns.some((t) => t.includes('GIF')), '有「导出 GIF」',
      view.exportBtns.join(' | '))
    expect(view.exportBtns.some((t) => t.includes('Godot')), '有「导出 Godot」',
      view.exportBtns.join(' | '))

    // 点第 3 帧 → 对照跟着换
    await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('姿态批量'))
      n.querySelectorAll('.pose-batch-grid .pose-batch-cell')[2]?.click()
    })
    await new Promise((r) => setTimeout(r, 300))
    const after = await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('姿态批量'))
      return {
        captions: [...n.querySelectorAll('.pose-batch-pair figcaption')].map((c) => c.textContent || ''),
        cands: n.querySelectorAll('.pose-batch-cands .pose-batch-cell').length,
      }
    })
    expect(after.captions[1]?.includes('第 3 帧'), '点第 3 帧 → 结果栏跟着换', after.captions[1] || '')
    expect(after.cands === 2, '第 3 帧有两个候选可选', String(after.cands))

    // 两个导出：点完要出现下载链接
    for (const [label, key] of [['导出 GIF', 'gif'], ['导出 Godot', 'godot']]) {
      await page.evaluate((text) => {
        const n = [...document.querySelectorAll('.node')]
          .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('姿态批量'))
        const b = [...n.querySelectorAll('.pose-batch-export .pose-batch-btn')]
          .find((x) => (x.textContent || '').includes(text))
        b?.click()
      }, label)
      await new Promise((r) => setTimeout(r, 700))
      const links = await page.evaluate(() => {
        const n = [...document.querySelectorAll('.node')]
          .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('姿态批量'))
        return [...n.querySelectorAll('.pose-batch-gif a')].map((a) => a.textContent || '')
      })
      expect(links.length > 0, `点「${label}」后出现下载链接`, links.join(' | '))
    }

    const clip = await page.evaluate(() => {
      const n = [...document.querySelectorAll('.node')]
        .find((x) => (x.querySelector('.node-kind')?.textContent || '').includes('姿态批量'))
      const r = n.getBoundingClientRect()
      return { x: Math.max(0, r.x - 8), y: Math.max(0, r.y - 8), width: r.width + 16, height: r.height + 16 }
    })
    mkdirSync(OUT, { recursive: true })
    const shot = join(OUT, 'pose_batch_node.png')
    await page.screenshot({ path: shot, clip })
    console.log(`  → ${shot}`)
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
