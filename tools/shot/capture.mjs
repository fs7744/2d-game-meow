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

  // 资料库第三个 tab：动作模板管理（真实后端数据，3 套内置预设）
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('.lib-tab')]
      .find((x) => (x.textContent || '').includes('动作模板'))
    b?.click()
  })
  await shot('35-motion-templates', 800)
  const mtDiag = await page.evaluate(() => ({
    tabs: [...document.querySelectorAll('.lib-tab')].map((b) => b.textContent?.trim()),
    cards: document.querySelectorAll('.mt-card').length,
    builtin: document.querySelectorAll('.mt-card.builtin').length,
    cost: document.querySelector('.mt-cost')?.textContent?.trim(),
    foot: document.querySelector('.lib-foot')?.textContent?.trim().slice(0, 60),
  }))
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('.mt-btn')]
      .find((x) => (x.textContent || '').includes('新建模板'))
    b?.click()
  })
  await shot('36-motion-editor', 500)
  const mtEditor = await page.evaluate(() => ({
    open: !!document.querySelector('.mt-editor'),
    actionRows: document.querySelectorAll('.mt-action-row').length,
    advanced: !!document.querySelector('.mt-adv'),
  }))
  if (mtDiag.cards < 3 || mtDiag.builtin !== 3 || !mtEditor.open || !mtEditor.advanced) {
    errors.push(`动作模板面板不完整: ${JSON.stringify({ mtDiag, mtEditor })}`)
  }
  console.log('  ✓ 35/36 动作模板管理', JSON.stringify({ mtDiag, mtEditor }))
  // 切回默认 tab 再关，别影响后面 06 场景的截图
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('.lib-tab')]
      .find((x) => (x.textContent || '').includes('提示词'))
    b?.click()
  })

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
      const gsettings = { base_model: 'qwen-image-2.1', matte_backend: 'auto',
        loras: [{ name: 'qwen21_anyangle', scale: 0.8 }], fast_mode: false }
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
        if (u.endsWith('/motion-templates') && m === 'GET') return json({
          version: 1, count: 3,
          items: [
            { id: 'topdown-walk-4', name: '俯视 RPG · 四方向基础', viewMode: 'topdown-4',
              builtin: true, directions: ['down', 'up', 'left', 'right'],
              actions: [{ name: 'idle', frames: 4, fps: 8, loop: true, oneShot: false }],
              mirrorFrom: {}, pivot: { x: 0.5, y: 1 }, camera: { pitch: 45, ortho: false },
              source: { kind: 'glb', file: '', clip: 'walk' }, yawMap: {},
              qwenPromptContract: '', subjects: 1, note: '',
              budget: { candidates: 1, singleFrameSeconds: 26, passRate: 0.75 } },
            { id: 'topdown-full-8', name: '俯视 RPG · 八方向全套', viewMode: 'topdown-8',
              builtin: true, directions: ['down', 'down-right', 'right', 'up-right',
                'up', 'up-left', 'left', 'down-left'],
              actions: [{ name: 'idle', frames: 4, fps: 8, loop: true, oneShot: false }],
              mirrorFrom: {}, pivot: { x: 0.5, y: 1 }, camera: { pitch: 45, ortho: false },
              source: { kind: 'glb', file: '', clip: 'walk' }, yawMap: {},
              qwenPromptContract: '', subjects: 1, note: '',
              budget: { candidates: 1, singleFrameSeconds: 26, passRate: 0.75 } },
            { id: 'side-platformer-2', name: '横板 · 左右 + 镜像', viewMode: 'side-2',
              builtin: true, directions: ['left', 'right'], mirrorFrom: { right: 'left' },
              actions: [{ name: 'idle', frames: 4, fps: 8, loop: true, oneShot: false }],
              pivot: { x: 0.5, y: 1 }, camera: { pitch: 0, ortho: true },
              source: { kind: 'glb', file: '', clip: 'walk' }, yawMap: {},
              qwenPromptContract: '', subjects: 1, note: '',
              budget: { candidates: 1, singleFrameSeconds: 26, passRate: 0.75 } },
          ],
          estimates: {
            'topdown-walk-4': { images: 104, totalMinutes: 60.1, mirroredDirections: [] },
            'topdown-full-8': { images: 208, totalMinutes: 120.2, mirroredDirections: [] },
            'side-platformer-2': { images: 34, totalMinutes: 19.6, mirroredDirections: ['right'] },
          },
        })
        if (u.endsWith('/settings')) {
          if (m === 'PUT') Object.assign(gsettings, JSON.parse(init.body || '{}'))
          return json({ ...gsettings })
        }
        // 假数据刻意贴近真实目录（8 个可挂 + 2 个格式不对），
        // 别只放两条——上一版截图让人以为程序只认出了 2 个 LoRA。
        if (u.endsWith('/loras') && m === 'GET') return json({
          items: [
            { name: 'qwen21_anyangle', size_mb: 120.1, tensors: 448, format: 'diffusers',
              file: 'models/lora/qwen21_anyangle.safetensors' },
            { name: 'qwen21_lightning', size_mb: 64.0, tensors: 256, format: 'diffusers',
              file: 'models/lora/qwen21_lightning.safetensors' },
            { name: 'qwen21_outpaint', size_mb: 160.1, tensors: 448, format: 'diffusers',
              file: 'models/lora/qwen21_outpaint.safetensors' },
            { name: 'vnccs_pose', size_mb: 160.1, tensors: 448, format: 'diffusers',
              file: 'models/lora/vnccs_pose.safetensors' },
            { name: 'vnccs_pose_v2', size_mb: 160.1, tensors: 448, format: 'diffusers',
              file: 'models/lora/vnccs_pose_v2.safetensors' },
            { name: 'VNCCS_QI2_ClothesCoreV2.6', size_mb: 160.1, tensors: 448,
              format: 'diffusers', file: 'models/lora/VNCCS_QI2_ClothesCoreV2.6.safetensors' },
            { name: 'VNCCS_QI2_AnimeOverhaulV2', size_mb: 160.1, tensors: 448,
              format: 'diffusers', file: 'models/lora/VNCCS_QI2_AnimeOverhaulV2.safetensors' },
            { name: 'style_lora_v3', size_mb: 96.4, tensors: 448, format: 'diffusers',
              file: 'models/lora/style_lora_v3.safetensors' },
          ],
          active: gsettings.loras,
          skipped: [
            { name: 'comfy_style_v1', size_mb: 120, tensors: 384, format: 'comfyui',
              need_convert: true, convertible: true,
              reason: '这是 ComfyUI 格式（diffusion_model. + lora_A/B）',
              file: 'models/lora/comfy_style_v1.safetensors' },
            { name: 'sdxl_style_x', size_mb: 144, tensors: 300, format: 'unknown',
              need_convert: true, convertible: false,
              reason: '没找到 Qwen-Image 的 LoRA 键（可能不是本底模的权重）',
              file: 'models/lora/sdxl_style_x.safetensors' },
          ],
          count: 10,
          dir: 'D:/code/2d-game-meow/models/lora',
        })
        if (u.includes('/loras/inspect')) return json({
          token: 'previewtoken01', name: 'my_style',
          original_name: 'my_style.safetensors', size_mb: 120, tensors: 384,
          format: 'comfyui', need_convert: true, convertible: true,
          reason: '这是 ComfyUI 格式（diffusion_model. + lora_A/B），当前底模需要 diffusers 格式',
          sample_keys: ['diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight'],
        })
        if (u.endsWith('/meta') && m === 'PATCH') return json({ ok: true })
        if (u.endsWith('/character-tasks')) return json({ items: [] })
        if (u.includes('/characters/') && u.endsWith('/prompt')) return json({
          name: chars[0].name, style: 'ghibli_miyazaki',
          prompt: 'The image contains one character.\nGender: female\nHair: silver hair\nEyes: red eyes',
          negative: 'blurry', resolution: [792, 1408],
          defaults: { steps: 25, cfg: 3, sampler: 'euler', scheduler: 'simple' },
        })
        // 生命阶段（GET 列表 / PUT 全量保存），截图里能看到三行不同阶段的立绘
        if (u.includes('/characters/') && u.includes('/stages')) return json({
          name: chars[0].name,
          description: '白毛猫耳骑士，幼年与成年是两套不同设定',
          active_stage: 'default',
          stages: [
            { id: 'default', title: '初始设定', description: '银发红瞳 · 常服', info: {},
              portrait_count: 3,
              portraits: [0, 1, 2].map((i) => ({ url: '/icon.png', name: `p${i}.png` })) },
            { id: 'child', title: '幼年', description: '短发，还没拿到铠甲', info: {},
              portrait_count: 2,
              portraits: [0, 1].map((i) => ({ url: '/icon.png', name: `c${i}.png` })) },
            { id: 'adult', title: '成年 · 觉醒', description: '', info: {},
              portrait_count: 0, portraits: [] },
          ],
        })
        // 带 ?stage= 也要命中（阶段编辑页只取该阶段的立绘）
        // 拆部件：列表 / 详情（必须排在 /characters/ 宽泛分支之前）
        if (/\/characters\/[^/]+\/parts$/.test(u)) return json({
          available: { seethrough: { ok: true, reason: '' },
            explode: { ok: true, reason: '' }, psd: { ok: true, reason: '' } },
          items: [{ id: 'p1', method: 'seethrough',
            methodLabel: 'See-Through 语义分层', created: 1794200000,
            seconds: 168, layers: 7, psd: '/icon.png', psdName: 'model.psd',
            preview: '/icon.png', canvas: [512, 768], source: '/icon.png',
            notice: '' }],
        })
        if (/\/characters\/[^/]+\/parts\/[^/]+$/.test(u)) return json({
          id: 'p1', method: 'seethrough', source: '/icon.png',
          sourceName: 'portrait_0.png', canvas: [512, 768], psd: '/icon.png',
          psdName: 'model.psd', preview: '/icon.png', created: 1794200000,
          seconds: 168, notice: '', options: {},
          layers: [
            ['back hair', 60, 40, 392, 700],
            ['legwear', 150, 420, 212, 330],
            ['topwear', 140, 240, 232, 220],
            ['handwear-l', 80, 250, 70, 200],
            ['handwear-r', 362, 250, 70, 200],
            ['face', 190, 60, 132, 180],
            ['front hair', 150, 30, 212, 220],
          ].map(([name, x, y, width, height], i) => ({
            name, file: `layers/${i}.png`, url: '/icon.png',
            x, y, width, height, order: i,
          })),
        })
        // 立绘悬浮工具箱（必须排在 /portraits 之前，否则被 includes 抢走）
        if (u.includes('/portraits/process')) {
          let b = {}
          try { b = JSON.parse(init.body || '{}') } catch { /* 忽略 */ }
          const op = String(b.op || 'matte')
          return json({ before: '/icon.png', after: '/icon.png',
            path: '/icon.png', name: `portrait_x_${op === 'pixel' ? 'px' : 'bg'}.png`,
            meta: { op } })
        }
        if (u.includes('/portraits/delete')) return json({ ok: true, deleted: 'x.png' })
        if (u.includes('/characters/') && u.includes('/portraits')) return json({
          count: 3,
          items: ['最新一版', 'Knight', 'Naked'].map((c, i) => ({
            url: '/icon.png', path: `/icon.png#${i}`, name: `portrait_${i}.png`,
            costume: i === 1 ? 'Knight' : 'Naked', emotion: 'neutral', mtime: 3 - i,
          })),
        })
        if (u.includes('/characters/') && u.endsWith('/atlas')) return json({
          character: chars[0].name, exists: true, frames: 12, animations: 7,
          directionAnimations: 4,
          size: [512, 768], canvas: [128, 192], bytes: 82000, image: '/icon.png',
        })
        // 导出前试玩验收台（只读）。必须给完整形状，否则会落到下面那条
        // 「按角色名读详情」的宽泛 GET 分支，组件拿到 CharacterDetail 就空了。
        if (u.includes('/characters/') && u.endsWith('/playtest')) {
          const yaw = { down: 0, 'down-right': 45, right: 90, 'up-right': 135,
            up: 180, 'up-left': 225, left: 270, 'down-left': 315 }
          const have = ['down', 'up', 'left', 'right']
          const vm = (dirs, mirrored = []) => {
            const missing = dirs.filter((d) => !have.includes(d) && !mirrored.includes(d))
            return {
              ready: missing.length === 0, required: dirs, missing, mirrored,
              yawMap: Object.fromEntries(dirs.map((d) => [d, yaw[d]])),
              hasFrames: true,
              note: missing.length
                ? `已有 ${have.length} 个方向帧；缺的按白模机位补出。`
                : '方向资产齐全，可以按当前视角模式验收并导出。',
            }
          }
          // 每个方向给独立 URL（query 只用于断言「换朝向真的换了帧组」），
          // 静态服务会忽略 query 照常返回 icon.png。
          const dirFrames = Object.fromEntries(have.map((d) => [
            d, [`/icon.png?dir=${d}`, `/icon.png?dir=${d}&f=2`],
          ]))
          return json({
            character: chars[0].name,
            clips: [
              { id: 'Naked/neutral', label: 'Naked · neutral', costume: 'Naked',
                emotion: 'neutral',
                frames: dirFrames.down, directionFrames: dirFrames,
                directions: have, defaultDirection: 'down', totalFrames: 8,
                frameRate: 8, loop: true },
              { id: 'Naked/angry', label: 'Naked · angry', costume: 'Naked',
                emotion: 'angry', frames: ['/icon.png'], frameRate: 8, loop: true },
            ],
            clipCount: 2, frameCount: 9,
            atlas: { exists: true, frameRate: 8, loop: true,
              pivot: { x: 0.5, y: 1 }, canvas: [128, 192], source: 'atlas',
              directions: have },
            directions: { present: have, characterDirs: have, atlasDirs: have,
              projectDirs: [], convention: 'directory-or-filename',
              templates: [{
                id: 'topdown-walk-4', name: '俯视 RPG · 四方向基础',
                viewMode: 'topdown-4', directions: have,
                actionCount: 4, fps: 8, loop: true,
                estimate: { images: 104, totalMinutes: 60.1, mirroredDirections: [] },
                project: 'c1', poseSet: '四向走路',
              }] },
            viewModes: {
              'topdown-4': vm(['down', 'up', 'left', 'right']),
              'topdown-8': vm(Object.keys(yaw)),
              'side-2': vm(['left', 'right'], ['right']),
            },
          })
        }
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
    // 进入某个生命阶段的「角色设定」（不再有四步步骤条）
    await fp.evaluate(() => {
      const row = document.querySelector('.cc-stage-row')
      ;[...(row?.querySelectorAll('button') || [])]
        .find((b) => (b.textContent || '').includes('进入设定'))?.click()
    })
    await fshot('23b-cc-step2')
    // 立绘悬浮工具箱：真实 hover 第一张卡，三个动作应该浮出来
    const portraitBox = await fp.evaluate(() => {
      const r = document.querySelector('.cc-portrait-view')?.getBoundingClientRect()
      return r ? { x: r.x + r.width / 2, y: r.y + r.height / 2 } : null
    })
    if (portraitBox) {
      await fp.mouse.move(portraitBox.x, portraitBox.y)
      await fshot('23e-portrait-tools', 350)
    } else {
      console.log('  ✗ 没找到立绘卡片')
    }
    // 换装弹窗：六字段 + AI 拆服装 + 保存服装（与待处理页同源）
    await fp.evaluate(() => [...document.querySelectorAll('.cc-portrait-tools button')]
      .find((b) => (b.textContent || '').includes('换装'))?.click())
    await fshot('23g-portrait-outfit', 600)
    await fp.evaluate(() => document.querySelector('.cc-tool-x')?.click())
    await sleep(220)

    // 换表情弹窗：官方表情库卡片（多选，出「已选 N 个表情」标签）+ 画质档
    await fp.evaluate(() => [...document.querySelectorAll('.cc-portrait-tools button')]
      .find((b) => (b.textContent || '').includes('换表情'))?.click())
    await sleep(500)
    await fp.evaluate(() => {
      const cards = [...document.querySelectorAll('.cc-tool-dialog .chars-emo')]
      cards[2]?.click()
      cards[4]?.click()
    })
    await fshot('23h-portrait-emotion', 600)
    await fp.evaluate(() => document.querySelector('.cc-tool-x')?.click())
    await sleep(220)

    // 右栏可拖：往左拖 160px，成果区应变宽（真实 pointer 事件走一遍）
    const sideBefore = await fp.evaluate(() =>
      document.querySelector('.cc-preview')?.getBoundingClientRect().width || 0)
    const grip = await fp.evaluate(() => {
      const r = document.querySelector('.cc-split')?.getBoundingClientRect()
      return r ? { x: r.x + r.width / 2, y: r.y + 220 } : null
    })
    if (grip) {
      await fp.mouse.move(grip.x, grip.y)
      await fp.mouse.down()
      await fp.mouse.move(grip.x - 160, grip.y, { steps: 8 })
      await fp.mouse.up()
      await fshot('23d-cc-side-drag', 350)
      const sideAfter = await fp.evaluate(() =>
        document.querySelector('.cc-preview')?.getBoundingClientRect().width || 0)
      console.log('  右栏宽度', Math.round(sideBefore), '→', Math.round(sideAfter))
      // 双击恢复默认，后面的场景保持一致
      await fp.evaluate(() => document.querySelector('.cc-split')
        ?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })))
      await sleep(200)
    } else {
      console.log('  ✗ 没找到 .cc-split 拖动条')
    }
    // 画风库弹窗：缩略图要够大才能比较笔触
    await fp.evaluate(() => [...document.querySelectorAll('button')]
      .find((b) => (b.textContent || '').includes('浏览画风库'))?.click())
    await fshot('23c-cc-style-dialog', 900)
    await fp.evaluate(() => document.querySelector('.cc-style-close')?.click())
    await sleep(200)
    // 出素材 / 导出都原样保留在「待处理」页（同一页同时渲染两段内容）
    // 拆部件页：与角色设定同级，切过去右栏立绘应隐藏、分层预览铺满
    await fp.evaluate(() => [...document.querySelectorAll('.cc-viewnav button')]
      .find((b) => (b.textContent || '').includes('拆部件'))?.click())
    await fshot('23f-cc-parts', 700)
    await fp.evaluate(() => [...document.querySelectorAll('.cc-viewnav button')]
      .find((b) => (b.textContent || '').includes('待处理'))?.click())
    await fshot('24-cc-step3')
    await fp.evaluate(() => {
      document.querySelectorAll('.cc-adv').forEach((d) => { d.open = true })
    })
    await fshot('25-cc-step3-adv')
    await fshot('26-cc-step4')
    // 试玩验收台：横板 → 八方向双键 → 横板跳跃
    const ptDiag = await fp.evaluate(() => ({
      hasPlaytest: !!document.querySelector('.pt-root'),
      modes: [...document.querySelectorAll('.pt-mode')].map((b) => b.textContent?.trim()),
      warning: document.querySelector('.pt-warning')?.textContent?.trim().slice(0, 90),
      reportRows: document.querySelectorAll('.pt-report-row').length,
      avatar: !!document.querySelector('.pt-avatar'),
      hud: document.querySelector('.pt-hud')?.textContent?.trim(),
    }))
    await fshot('30-playtest-side')
    await fp.evaluate(() => [...document.querySelectorAll('.pt-mode')]
      .find((b) => (b.textContent || '').includes('八方向'))?.click())
    await fp.keyboard.down('d')
    await fp.keyboard.down('w')
    await fshot('31-playtest-8dir', 700)
    await fp.keyboard.up('w')
    await fp.keyboard.up('d')
    await fp.evaluate(() => [...document.querySelectorAll('.pt-mode')]
      .find((b) => (b.textContent || '').includes('横板'))?.click())
    await fp.keyboard.press('Space')
    await fshot('32-playtest-jump', 450)
    // 方向帧：按 W 要切到 up 帧组（HUD 与 <img src> 都得变），
    // 不能还是拿同一组帧占位却假装按方向播。
    await fp.evaluate(() => [...document.querySelectorAll('.pt-mode')]
      .find((b) => (b.textContent || '').includes('八方向'))?.click())
    await fp.keyboard.down('w')
    await fshot('33-playtest-up', 500)
    const dirDiag = await fp.evaluate(() => ({
      hud: document.querySelector('.pt-hud')?.textContent?.trim(),
      frameSrc: document.querySelector('.pt-avatar')?.getAttribute('src'),
    }))
    await fp.keyboard.up('w')
    if (!String(dirDiag.frameSrc || '').includes('dir=up')
        || !String(dirDiag.hud || '').includes('帧组 上')) {
      errors.push(`方向帧切换失败: ${JSON.stringify(dirDiag)}`)
    }
    // 验收参数写回图集导出：点了必须给可见反馈
    await fp.evaluate(() => document.querySelector('.pt-apply .pt-btn')?.click())
    // 验收清单 + 写回按钮在舞台下方，滚进去才拍得到（不然只有 DOM 断言、没有视觉证据）
    await fp.evaluate(() => document.querySelector('.pt-report')
      ?.scrollIntoView({ block: 'center' }))
    await fshot('34-playtest-apply', 400)
    const appliedText = await fp.evaluate(() =>
      document.querySelector('.pt-applied')?.textContent?.trim() || '')
    if (!appliedText) errors.push('写入导出参数没有可见反馈')
    const templateText = await fp.evaluate(() =>
      document.querySelector('.pt-templates')?.textContent?.trim() || '')
    if (!templateText.includes('俯视 RPG')) errors.push('关联动作模板没显示')
    console.log('  ✓ 30~34 试玩验收台',
      JSON.stringify({ ...ptDiag, dirDiag, appliedText, templateText: templateText.slice(0, 90) }))
    // 角色中心也能开全局设置（顶栏 ⚙，不依赖画布侧栏）
    await fp.evaluate(() => [...document.querySelectorAll('.top-action')]
      .find((b) => (b.textContent || '').includes('设置'))?.click())
    await fshot('27-gs-from-characters')
    await fp.evaluate(() => document.querySelector('.gs-close')?.click())
    await sleep(300)
    // 回画布编辑器，拍「全局设置 / LoRA 导入」
    await fp.goto('http://127.0.0.1:8123/index.html?editor=1',
      { waitUntil: 'networkidle2', timeout: 60000 })
    await sleep(900)
    // 设置已经从侧栏搬到顶栏 ⚙ 弹窗（侧栏只留画布/模板/节点/资料库）
    await fp.evaluate(() => [...document.querySelectorAll('.top-action')]
      .find((b) => (b.textContent || '').includes('设置'))?.click())
    await fshot('28-settings-global')
    // 选一个 ComfyUI 格式的 LoRA → 应弹出「需要转换」确认框
    await fp.evaluate(() => {
      const input = document.querySelector('.lora-file-input')
      if (!input) return
      const file = new File([new Uint8Array([1, 2, 3])], 'my_style.safetensors')
      Object.defineProperty(input, 'files', { configurable: true, get: () => [file] })
      input.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await fshot('29-lora-convert', 1000)
    await fp.close()
  }

  await browser.close()
  srv.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
