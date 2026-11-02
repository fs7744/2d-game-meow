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

const FRAMES = [1, 2, 3, 4].map((i) => ({
  index: i, t: (i - 1) * 0.2, yaw: 0,
  // 每帧**不同的 URL**（同一个文件挂个 query）：播放要验"大图真的换了"，
  // 四帧都指同一个 src 的话，断言永远是绿的却什么都没动
  url: `${IMG}?f=${i}`,
}))
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
  /** 点「全部生成」时抓到 `/run` 的请求体，断言前端把引擎/参考模式带对了 */
  let runBody = null
  const expect = (cond, label, detail) => {
    if (!cond) bad++
    console.log(`${cond ? '✓' : '✗'} ${label}${detail ? '  ' + detail : ''}`)
  }

  try {
    const page = await browser.newPage()
    await page.setViewport({ width: 1600, height: 1400 })
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
            {
              // **只选了姿势集**：没接角色图、没出过图。用来验
              // ① 输出点能不能手动拖线接进来；② 没出图时舞台也要有内容可播。
              id: 'n2', kind: 'posebatch', x: 900, y: 60, w: 400, view: 'expanded',
              params: { pcPoseSet: '走路' },
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
      if (p === '/projects/pdemo/pose_renders/走路-成片/run') {
        runBody = JSON.parse(r.postData() || '{}')
        return json({ job_id: 'j1', status: 'queued', total: 4, frames: [1, 2, 3, 4] })
      }
      if (p === '/projects/pdemo/pose_renders') {
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
        srcCells: n.querySelectorAll('.pose-batch > .pose-batch-strip .pose-batch-cell').length,
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

    /* ---------------- A) 手动连线：n0 的输出点 → n2 的角色端口 ----------------
       用户报的"下拉能选、拖线不行"：`onConnect` 的角色图白名单漏了 posebatch，
       于是拖过去被静默忽略。这里**真的用鼠标拖**（pointerdown → pointerup），
       走的是 Canvas 的 elementFromPoint 落点判定，不是直接改 state。 */
    const pt = (sel, id) => page.evaluate((s, i) => {
      const root = i ? document.querySelector(`[data-node-id="${i}"]`) : document
      const el = root?.querySelector(s)
      if (!el) return null
      const r = el.getBoundingClientRect()
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 }
    }, sel, id)

    const genBtn = (id) => page.evaluate((nid) => {
      const n = document.querySelector(`[data-node-id="${nid}"]`)
      const b = n?.querySelector('.posebatch-body .gen-btn')
      if (!b) return null
      const cs = getComputedStyle(b)
      return {
        disabled: b.disabled, title: b.getAttribute('title') || '',
        text: (b.textContent || '').trim(),
        bg: cs.backgroundColor, color: cs.color, width: Math.round(b.getBoundingClientRect().width),
        height: Math.round(b.getBoundingClientRect().height),
      }
    }, id)
    const beforeBtn = await genBtn('n2')
    expect(beforeBtn?.disabled === true, '没接角色图时「全部生成」是禁用的',
      JSON.stringify(beforeBtn))
    // 主操作必须是**整行绿色大按钮**（和「迁移角色」同一套 `gen-btn`）：
    // 底色要是不透明绿、宽度撑满、高度 32 —— 之前用 `.pose-batch-btn` 时
    // 底色 `--bg` 比卡片还暗，被用户当成"透明了"
    expect(beforeBtn?.text.includes('全部生成'), '按钮文案是「全部生成」', beforeBtn?.text || '')
    expect((beforeBtn?.height || 0) >= 30, '主按钮是大号（≥30px 高）', String(beforeBtn?.height))
    expect((beforeBtn?.width || 0) >= 300, '主按钮整行撑满', String(beforeBtn?.width))
    // 禁用态也必须是**实色**（透明的话在深色卡片上就是一个洞，正是被报的现象）
    expect(/^rgb\(/.test(beforeBtn?.bg || ''), '禁用态也是实色填充，不是透明',
      beforeBtn?.bg || '')

    const beforeChar = await page.evaluate(
      () => document.querySelectorAll('path.link.char').length)
    const from = await pt('.conn.out', 'n0')
    const to = await pt('[data-port="char"]', 'n2')
    expect(!!from && !!to, '找得到输出点与角色端口', JSON.stringify({ from, to }))
    if (from && to) {
      await page.mouse.move(from.x, from.y)
      await page.mouse.down()
      await page.mouse.move(from.x + 40, from.y + 10, { steps: 4 })
      await page.mouse.move(to.x, to.y, { steps: 12 })
      await page.mouse.up()
      await new Promise((r) => setTimeout(r, 400))
    }
    const afterChar = await page.evaluate(() => {
      const n = document.querySelector('[data-node-id="n2"]')
      return {
        edges: document.querySelectorAll('path.link.char').length,
        thumb: !!n?.querySelector('.src-thumb img'),
        note: (n?.textContent || '').includes('还没接角色图'),
      }
    })
    expect(afterChar.edges === beforeChar + 1, '拖一条角色图连线（图上多一条 char 边）',
      `${beforeChar} → ${afterChar.edges}`)
    expect(afterChar.thumb, '拖完角色图缩略图出现在节点里')
    expect(!afterChar.note, '「还没接角色图」提示消失')
    // **这条是用户报的那个 bug 的正面证据**：接上之后按钮真的能点，
    // 说明 `charPaths` 被解析出来了（不只是"界面上有个缩略图"）。
    const afterBtn = await genBtn('n2')
    expect(afterBtn?.disabled === false, '接上角色图后「全部生成」可点',
      afterBtn?.title || '')
    expect(/^rgb\(0, 255, 136\)/.test(afterBtn?.bg || ''),
      '可点时是品牌绿主按钮（gen-btn），不是暗色小按钮', afterBtn?.bg || '')


    /* ---------------- B) 播放：没出图也要有内容 ----------------
       上一版只在"已经有成片"时才渲染大图，抽完帧点 ▶ 屏幕上什么都不动 —— 这次
       舞台必须独立渲染，播放时大图要跟着换（验的是 img src 变、不是格子里高亮）。 */
    // 底色三档：白 / 深 / **透明**（用户报的"更多设置里没有透明选项"）
    const bgOpts = await page.evaluate(() => {
      const n = document.querySelector('[data-node-id="n2"]')
      const sel = [...n.querySelectorAll('select')]
        .find((x) => [...x.options].some((o) => o.value === 'dark'))
      return sel ? { values: [...sel.options].map((o) => o.value),
        labels: [...sel.options].map((o) => o.textContent || ''),
        disabled: sel.disabled } : null
    })
    expect(bgOpts?.values.join(',') === 'white,dark,transparent',
      '底色有「透明底」选项', bgOpts?.values.join('|') || '（没找到底色下拉）')
    expect(bgOpts?.disabled === false, '底色下拉可改（没被别的开关锁住）',
      String(bgOpts?.disabled))

    const stage = await page.evaluate(() => {
      const n = document.querySelector('[data-node-id="n2"]')
      return {
        hasStage: !!n?.querySelector('.pose-batch-stage'),
        figures: n?.querySelectorAll('.pose-batch-stage .pose-batch-pair figure').length,
        playBtn: [...(n?.querySelectorAll('.pose-batch-stage-bar .pose-batch-btn') || [])]
          .map((b) => b.textContent || ''),
        label: n?.querySelector('.pose-batch-stage-bar b')?.textContent || '',
        img: n?.querySelector('.pose-batch-stage figure img')?.getAttribute('src') || '',
      }
    })
    expect(stage.hasStage, '没出图也有舞台（播放才有内容）')
    expect(stage.figures === 1, '只有源帧时舞台单栏（不摆空相框）', String(stage.figures))
    expect(stage.playBtn.some((t) => t.includes('播')), '舞台上有 ▶ 播', stage.playBtn.join('|'))
    expect(stage.label.includes('/ 4 帧'), '舞台显示"第 N / 4 帧"', stage.label)

    await page.evaluate(() => {
      document.querySelector('[data-node-id="n2"] .pose-batch-stage-bar .pose-batch-btn')?.click()
    })
    // 等一个多 tick（125ms/帧），看大图有没有真的换
    await new Promise((r) => setTimeout(r, 300))
    const playing = await page.evaluate(() => {
      const n = document.querySelector('[data-node-id="n2"]')
      return {
        label: n?.querySelector('.pose-batch-stage-bar b')?.textContent || '',
        btn: n?.querySelector('.pose-batch-stage-bar .pose-batch-btn')?.textContent || '',
        onCells: n?.querySelectorAll('.pose-batch > .pose-batch-strip .pose-batch-cell.on').length,
        img: n?.querySelector('.pose-batch-stage figure img')?.getAttribute('src') || '',
        caption: n?.querySelector('.pose-batch-stage figcaption')?.textContent || '',
      }
    })
    expect(playing.label.includes('播放中'), '点播后标签变成"播放中 N/M"', playing.label)
    expect(playing.btn.includes('停'), '播放时按钮变成「■ 停」', playing.btn)
    expect(playing.onCells === 1, '缩略条同步高亮当前帧', String(playing.onCells))
    // **这条才是"播放有内容"的硬证据**：大图 src 换了、帧号跟着走
    expect(!!stage.img && playing.img !== stage.img, '播放时舞台大图真的在换',
      `${stage.img} → ${playing.img}`)
    expect(/第 [2-4] 帧/.test(playing.caption), '舞台标注跟着走到后面的帧', playing.caption)

    // 趁还在播，留一张"播放中"的截图 —— 用户报的就是"点了播看不出在播"
    {
      mkdirSync(OUT, { recursive: true })
      const box = await page.evaluate(() => {
        const n = document.querySelector('[data-node-id="n2"]')
        const r = n.getBoundingClientRect()
        return { x: Math.max(0, r.x - 8), y: Math.max(0, r.y - 8),
          width: Math.min(r.width + 16, 1600), height: Math.min(r.height + 16, 1400) }
      })
      await page.screenshot({ path: join(OUT, 'pose_batch_playing.png'), clip: box })
      console.log(`  → ${join(OUT, 'pose_batch_playing.png')}`)
    }

    // 停下 → 回到静止态；再点一次从第 1 帧开始（不能停在哪就从哪开始）
    await page.evaluate(() => {
      document.querySelector('[data-node-id="n2"] .pose-batch-stage-bar .pose-batch-btn')?.click()
    })
    await new Promise((r) => setTimeout(r, 200))
    const stopped = await page.evaluate(() => {
      const n = document.querySelector('[data-node-id="n2"]')
      return {
        label: n?.querySelector('.pose-batch-stage-bar b')?.textContent || '',
        btn: n?.querySelector('.pose-batch-stage-bar .pose-batch-btn')?.textContent || '',
      }
    })
    expect(stopped.label.includes('第 ') && !stopped.label.includes('播放中'),
      '停下后回到"第 N / M 帧"', stopped.label)
    expect(stopped.btn.includes('播'), '停下后按钮变回「▶ 播」', stopped.btn)

    // 点「全部生成」→ 前端必须把引擎/参考模式**原样**带上：
    // 引擎默认 VNCCS（以前批量写死 wpc，节点选 VNCCS 也没用）；参考模式
    // 默认 skeleton（灰人偶不能走 auto 的 DWPose 提取，否则条件图被画成
    // 几乎空白的细线，12 帧全照抄角色图的站姿）。
    await page.evaluate(() => {
      document.querySelector('[data-node-id="n2"] .posebatch-body .gen-btn')?.click()
    })
    await new Promise((r) => setTimeout(r, 600))
    expect(runBody?.engine === 'wvnccs', '请求体带 engine=wvnccs（和角色迁移节点一致）',
      runBody?.engine || '（没发出去）')
    expect(runBody?.ref_mode === 'skeleton', '请求体带 ref_mode=skeleton（人偶图不能走 auto）',
      runBody?.ref_mode || '')

        mkdirSync(OUT, { recursive: true })
    for (const [id, file] of [['n1', 'pose_batch_node.png'], ['n2', 'pose_batch_source.png']]) {
      await page.evaluate((nid) => {
        const d = document.querySelector(`[data-node-id="${nid}"] details.adv`)
        if (d) d.open = true
      }, id)
      const box = await page.evaluate((nid) => {
        const n = document.querySelector(`[data-node-id="${nid}"]`)
        if (!n) return null
        const r = n.getBoundingClientRect()
        return { x: Math.max(0, r.x - 8), y: Math.max(0, r.y - 8),
          width: Math.min(r.width + 16, 1600), height: Math.min(r.height + 16, 1400) }
      }, id)
      if (!box) continue
      const shot = join(OUT, file)
      await page.screenshot({ path: shot, clip: box })
      console.log(`  → ${shot}`)
    }
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
