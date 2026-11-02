/**
 * 真机验证「人偶姿势」的动作导入（glb / fbx / 非 Mixamo 骨架 / 认不出的骨架）。
 *
 * **为什么必须真浏览器**：这条链上的四件事 jsdom 一律验不到 ——
 *
 *  1. `vendor/three/jsm` 的 loader 能不能真解析开（studio.html 的 importmap 对不对）；
 *  2. FBX 那条路（依赖 fflate 解压 + NURBS 曲线）通不通；
 *  3. 换成 Unreal / KayKit 命名后，`anim-skeleton.js` 还能不能把 17 个锚点对到
 *     真实骨头上 —— 这是"支持 CC0 资产"的唯一证据；
 *  4. 认不出的骨架有没有**明确报错**（而不是"点了没反应"）。
 *
 * 一个浏览器进程跑完全部用例（人偶模型 37MB，起一次就够）：宿主页顺序给 iframe
 * 发 `animLoad` → `animTime` → `export` → `animSample`，每步结果攒成 JSON 交给
 * 本脚本断言。
 *
 * 非 Mixamo 命名的用例是**把 Soldier.glb 的节点名改掉**生成的 —— 层级不变、
 * 只换名字，正好只考骨名映射这一件事（真实资产里名字和层级会一起变，
 * 反而说不清是哪一环救了它）。
 *
 * 依赖 puppeteer-core（隔离装在 `tools/shot/node_modules`，不污染项目 package.json），
 * 驱动系统 Edge，不需要下载 Chromium。
 *
 * 用法（先 `cd web && npm run build`）：
 *
 *     node tools/verify_anim_import.mjs
 *     node tools/verify_anim_import.mjs --only fbx
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

/** puppeteer-core 只装在 tools/shot（隔离），这里按那个目录解析 */
const requireFromShot = createRequire(join(ROOT, 'tools', 'shot', 'package.json'))
let puppeteer
try {
  puppeteer = requireFromShot('puppeteer-core')
} catch {
  console.error('缺少 puppeteer-core：先 cd tools/shot && npm install puppeteer-core')
  process.exit(2)
}

const SOURCES = {
  'Soldier.glb': 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/Soldier.glb',
  'SambaDancing.fbx': 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/fbx/Samba%20Dancing.fbx',
}

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.glb': 'model/gltf-binary', '.fbx': 'application/octet-stream', '.wasm': 'application/wasm', '.bin': 'application/octet-stream' }

/* ------------------------------------------------------------------ *
 * 测试素材
 * ------------------------------------------------------------------ */

async function download(dst, url) {
  if (existsSync(dst) && readFileSync(dst).length > 1024) return
  mkdirSync(dirname(dst), { recursive: true })
  console.log(`  下载 ${dst.split(/[\\/]/).pop()} …`)
  const r = await fetch(url)
  if (!r.ok) throw new Error(`取模型失败 ${r.status} ${url}`)
  writeFileSync(dst, Buffer.from(await r.arrayBuffer()))
}

/** 改 glb 里的节点名（动画/皮肤按**下标**引用，改名字不影响绑定） */
function rewriteGlbNames(src, dst, mapper) {
  const data = readFileSync(src)
  const magic = data.readUInt32LE(0)
  const ver = data.readUInt32LE(4)
  const jlen = data.readUInt32LE(12)
  const jtype = data.readUInt32LE(16)
  const doc = JSON.parse(data.subarray(20, 20 + jlen).toString('utf8'))
  const rest = data.subarray(20 + jlen)
  for (const n of doc.nodes || []) if (n.name) n.name = mapper(n.name)
  let raw = Buffer.from(JSON.stringify(doc), 'utf8')
  const pad = (4 - (raw.length % 4)) % 4
  if (pad) raw = Buffer.concat([raw, Buffer.alloc(pad, 0x20)])
  const head = Buffer.alloc(20)
  head.writeUInt32LE(magic, 0)
  head.writeUInt32LE(ver, 4)
  head.writeUInt32LE(12 + 8 + raw.length + rest.length, 8)
  head.writeUInt32LE(raw.length, 12)
  head.writeUInt32LE(jtype, 16)
  writeFileSync(dst, Buffer.concat([head, raw, rest]))
}

const UNREAL = {
  Hips: 'pelvis', Spine: 'spine_01', Spine1: 'spine_02', Spine2: 'spine_03',
  Neck: 'neck_01', Head: 'head', HeadTop_End: 'head_end',
  LeftShoulder: 'clavicle_l', RightShoulder: 'clavicle_r',
  LeftArm: 'upperarm_l', RightArm: 'upperarm_r',
  LeftForeArm: 'lowerarm_l', RightForeArm: 'lowerarm_r',
  LeftHand: 'hand_l', RightHand: 'hand_r',
  LeftUpLeg: 'thigh_l', RightUpLeg: 'thigh_r',
  LeftLeg: 'calf_l', RightLeg: 'calf_r',
  LeftFoot: 'foot_l', RightFoot: 'foot_r',
  LeftToeBase: 'ball_l', RightToeBase: 'ball_r',
}

const KAYKIT = {
  Hips: 'hips', Spine: 'spine', Spine1: 'chest', Spine2: 'chest_upper',
  Neck: 'neck', Head: 'head', HeadTop_End: 'head_end',
  LeftShoulder: 'shoulder_l', RightShoulder: 'shoulder_r',
  LeftArm: 'upper_arm_l', RightArm: 'upper_arm_r',
  LeftForeArm: 'lower_arm_l', RightForeArm: 'lower_arm_r',
  LeftHand: 'hand_l', RightHand: 'hand_r',
  LeftUpLeg: 'upper_leg_l', RightUpLeg: 'upper_leg_r',
  LeftLeg: 'lower_leg_l', RightLeg: 'lower_leg_r',
  LeftFoot: 'foot_l', RightFoot: 'foot_r',
  LeftToeBase: 'toe_l', RightToeBase: 'toe_r',
}

const mapper = (table) => (name) => {
  const short = name.includes(':') ? name.split(':').pop() : name
  return table[short] || name
}
const junk = (() => {
  let i = 0
  const seen = new Map()
  return (name) => {
    if (!seen.has(name)) seen.set(name, `b${i++}`)
    return seen.get(name)
  }
})()

/* ------------------------------------------------------------------ *
 * 宿主页（真浏览器里驱动 iframe，跑完把结论写进 <pre id="json">）
 * ------------------------------------------------------------------ */

const HARNESS = (cases) => `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body { margin:0; background:#121216; color:#eee; font:12px monospace; }
  #stage { width:420px; height:420px; }
  #stage iframe { display:block; width:100%; height:100%; border:0; }
  #log { white-space:pre-wrap; }
</style></head><body>
<div id="stage"><iframe id="f" src="/vnccs-pose/studio.html?embed=1"></iframe></div>
<div id="log">…</div>
<script>
const f = document.getElementById('f')
const logEl = document.getElementById('log')
const CASES = ${JSON.stringify(cases)}
window.__errors = []
window.addEventListener('error', (e) => window.__errors.push('error: ' + e.message))
window.addEventListener('unhandledrejection', (e) => window.__errors.push('rej: ' + ((e.reason && e.reason.message) || e.reason)))

/** 等一条子窗消息；同时记录最后一条 error（用例失败时要能读出原因） */
function waitFor(type, ms) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => {
      window.removeEventListener('message', h)
      reject(new Error('超时等 ' + type + (window.__lastErr ? ('，最后一个错误：' + window.__lastErr) : '')))
    }, ms)
    const h = (e) => {
      if (e.source !== f.contentWindow) return
      const d = e.data || {}
      if (d.type === 'error') window.__lastErr = d.message
      if (d.type !== type) return
      clearTimeout(t); window.removeEventListener('message', h); resolve(d)
    }
    window.addEventListener('message', h)
  })
}

const send = (msg, transfer) => f.contentWindow.postMessage(msg, '*', transfer || [])

async function waitReady() {
  for (let i = 0; i < 1200; i++) {
    if (window.__ready) return
    await new Promise((r) => setTimeout(r, 100))
  }
  throw new Error('人偶模型没加载完')
}

window.addEventListener('message', (e) => {
  if (e.source !== f.contentWindow) return
  const d = e.data || {}
  if (d.type === 'ready') window.__ready = true
  if (d.type === 'error') window.__lastErr = d.message
  if (d.type === 'progress') logEl.textContent += d.text + '\\n'
})

async function one(c) {
  delete window.__lastErr
  const buf = await fetch(c.model).then((r) => r.arrayBuffer())
  // 用 Promise.all 同时挂监听：animLoad 失败时会推 error 而不是 animLoaded
  const loadedP = Promise.race([
    waitFor('animLoaded', 120000),
    waitFor('error', 120000).then((d) => ({ type: 'error', message: d.message })),
  ])
  send({ type: 'animLoad', buffer: buf, name: c.file || c.model.split('/').pop() }, [buf])
  const d = await loadedP
  if (d.type === 'error') {
    const hit = c.expect === 'reject' && d.message.includes(c.expectText || '')
    return { name: c.name, ok: hit, expect: c.expect || '', msg: d.message }
  }
  const clips = d.clips || []
  if (c.expect === 'reject') return { name: c.name, ok: false, why: '本该报错却载入成功' }
  if (!clips.length) return { name: c.name, ok: false, why: '没有动画轨道' }

  const dur = clips[0].duration || 1
  const out = { name: c.name, ok: true, clips: clips.length, bones: d.boneCount, fmt: d.format, rig: d.rig }
  send({ type: 'animTime', t: dur * 0.4 })
  const pose = await waitFor('pose', 60000)
  out.posed = !!pose.pose

  if (c.export) {
    send({ type: 'export', size: 384 })
    const r = await waitFor('result', 60000)
    const url = String(r.dataURL || '')
    out.pngOk = url.startsWith('data:image/png')
    out.pngBytes = Math.floor((url.split(',')[1] || '').length * 3 / 4)
    out.pngB64 = url.split(',')[1] || ''       // 落到 tests/_out，肉眼复核用
  }
  if (c.play) {
    // **播放**：从 0 播 1.2 秒，看 tick 有没有单调推进（iframe 按 wall clock 算 t）
    const ticks = []
    const onTick = (e) => {
      if (e.source !== f.contentWindow) return
      const d = e.data || {}
      if (d.type === 'animTick' && typeof d.t === 'number') ticks.push(d.t)
      if (d.type === 'animPlaying') out.playEvents = (out.playEvents || 0) + 1
    }
    window.addEventListener('message', onTick)
    send({ type: 'animPlay', on: true, t: 0 })
    await new Promise((r) => setTimeout(r, 1200))
    send({ type: 'animPlay', on: false })
    await new Promise((r) => setTimeout(r, 500))
    window.removeEventListener('message', onTick)
    out.ticks = ticks.length
    out.tickMax = ticks.length ? Math.max(...ticks) : 0
    out.tickMonotonic = ticks.every((v, i) => i === 0 || v >= ticks[i - 1] - 1e-6)
  }
  if (c.sample) {
    send({ type: 'animSample', count: c.sample })
    const fr = await waitFor('animFrames', 180000)
    const frames = fr.frames || []
    out.frames = frames.length
    out.framesWithPreview = frames.filter((x) => String(x.png || '').startsWith('data:image/png')).length
    out.framesWithPose = frames.filter((x) => x.pose && typeof x.pose === 'object').length
    out.clipOfFrames = fr.clip
    // 抽出来的帧必须**互不相同**：全一样说明采样没生效（时间没推进到 mixer）
    const sig = new Set(frames.map((x) => JSON.stringify(x.pose).length))
    out.distinctFrames = sig.size
  }
  return out
}

;(async () => {
  const out = []
  try {
    await waitReady()
    for (const c of CASES) {
      logEl.textContent += '---- ' + c.name + '\\n'
      let r
      try { r = await one(c) } catch (e) { r = { name: c.name, ok: false, why: String(e.message || e) } }
      if (window.__lastErr && !r.msg) r.lastErr = window.__lastErr
      out.push(r)
      logEl.textContent += JSON.stringify(r) + '\\n'
    }
  } catch (e) {
    out.push({ name: 'harness', ok: false, why: String(e.message || e) })
  }
  out.push({ name: '__errors', ok: true, errors: window.__errors })
  const pre = document.createElement('pre')
  pre.id = 'json'
  pre.textContent = JSON.stringify(out)
  document.body.appendChild(pre)
  const bad = out.filter((x) => !x.ok && x.name !== '__errors')
  document.title = bad.length
    ? 'ANIM_FAIL ' + bad.map((x) => x.name + ':' + (x.why || x.msg || '')).join(' | ')
    : 'ANIM_OK ' + out.length
})()
</script></body></html>`

/* ------------------------------------------------------------------ *
 * 静态服务（dist + 内存里的宿主页 + 模型）
 * ------------------------------------------------------------------ */

function serve(port, harness) {
  const srv = createServer((req, res) => {
    const url = decodeURIComponent((req.url || '/').split('?')[0])
    if (url === '/__anim_check.html') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
      res.end(harness)
      return
    }
    if (url.startsWith('/__model/')) {
      const p = join(MESH, url.slice('/__model/'.length))
      if (p.startsWith(MESH) && existsSync(p)) {
        const body = readFileSync(p)
        res.writeHead(200, { 'Content-Type': MIME[extname(p)] || 'application/octet-stream', 'Content-Length': body.length })
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
    if (!existsSync(file)) {
      res.writeHead(404).end('not found')
      return
    }
    const body = readFileSync(file)
    res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' })
    res.end(body)
  })
  return new Promise((ok) => srv.listen(port, '127.0.0.1', () => ok(srv)))
}

/* ------------------------------------------------------------------ *
 * 主流程
 * ------------------------------------------------------------------ */

async function main() {
  const only = process.argv.includes('--only')
    ? process.argv[process.argv.indexOf('--only') + 1] : ''

  if (!existsSync(join(DIST, 'index.html'))) {
    console.error('先 cd web && npm run build（web/dist 不存在）')
    return 2
  }
  if (!existsSync(EDGE)) {
    console.error('找不到系统 Edge：' + EDGE)
    return 2
  }

  console.log('准备测试模型…')
  mkdirSync(MESH, { recursive: true })
  for (const [name, url] of Object.entries(SOURCES)) await download(join(MESH, name), url)

  const soldier = join(MESH, 'Soldier.glb')
  rewriteGlbNames(soldier, join(MESH, 'Soldier.unreal.glb'), mapper(UNREAL))
  rewriteGlbNames(soldier, join(MESH, 'Soldier.kaykit.glb'), mapper(KAYKIT))
  rewriteGlbNames(soldier, join(MESH, 'Soldier.junk.glb'), junk)

  let cases = [
    { name: 'glb-mixamo', model: '/__model/Soldier.glb', file: 'Soldier.glb',
      export: true, sample: 4, play: true },
    { name: 'fbx-mixamo', model: '/__model/SambaDancing.fbx', file: 'SambaDancing.fbx', export: true },
    { name: 'glb-unreal', model: '/__model/Soldier.unreal.glb', file: 'Soldier.glb' },
    { name: 'glb-kaykit', model: '/__model/Soldier.kaykit.glb', file: 'Soldier.glb' },
    // 认不出的骨架：必须**明确报错**（"骨骼认不出"+缺哪几根），不能沉默
    { name: 'glb-junk', model: '/__model/Soldier.junk.glb', file: 'junk.glb',
      expect: 'reject', expectText: '骨骼认不出' },
  ]
  if (only) {
    cases = cases.filter((c) => c.name.includes(only))
    if (!cases.length) {
      console.error(`没有匹配 --only ${only} 的用例`)
      return 2
    }
  }

  const port = 8137
  const srv = await serve(port, HARNESS(cases))
  const browser = await puppeteer.launch({
    executablePath: EDGE,
    headless: 'new',
    args: [
      '--enable-unsafe-swiftshader', '--no-sandbox', '--disable-gpu-sandbox',
      '--use-gl=angle', '--use-angle=swiftshader', '--window-size=520,760',
    ],
  })

  let results = []
  /** 被拦下的外部请求（用来证明"断网也能导入"） */
  const blocked = []
  try {
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.log('  [pageerror]', e.message))
    page.on('console', (m) => {
      if (m.type() === 'error') console.log('  [console]', m.text().slice(0, 200))
    })
    // **断网模拟**：只放行本地静态服务 / blob / data。解析器要是还挂在 esm.sh 上，
    // 这一步就会直接失败 —— 那正是 §6-1「断网也能导入动作」的验收方式。
    await page.setRequestInterception(true)
    page.on('request', (req) => {
      const u = req.url()
      if (u.startsWith(`http://127.0.0.1:${port}/`) || u.startsWith('blob:') || u.startsWith('data:')) {
        req.continue()
        return
      }
      blocked.push(u)
      req.abort()
    })
    console.log('打开宿主页（人偶模型 37MB，软件渲染约 15~60 秒）…')
    const t0 = Date.now()
    await page.goto(`http://127.0.0.1:${port}/__anim_check.html`, { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(
      () => document.title.startsWith('ANIM_') || !!document.getElementById('json'),
      { timeout: 600000, polling: 1000 })
    console.log(`跑完，耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s`)
    const raw = await page.evaluate(() => document.getElementById('json')?.textContent || '[]')
    results = JSON.parse(raw)
  } finally {
    await browser.close()
    srv.close()
  }

  /* ---------------- 断言 ---------------- */
  let bad = 0
  const expect = (cond, label, detail) => {
    if (cond) return
    bad++
    console.log(`   ✗ ${label}${detail ? '：' + detail : ''}`)
  }

  // 断网：loader 必须走本地副本，一个外部请求都不该发出去
  const external = [...new Set(blocked)].filter((u) => !u.startsWith('http://127.0.0.1'))
  console.log(external.length
    ? `断网模拟：拦下 ${external.length} 个外部请求 → ${external.slice(0, 3).join(', ')}`
    : '断网模拟：全程没发外部请求（解析器确实是本地的）✓')
  expect(external.length === 0, '不该联网取解析器（本地方案失效了）', external.slice(0, 3).join(', '))

  for (const r of results) {
    if (r.name === '__errors') {
      if ((r.errors || []).length) {
        bad++
        console.log('   ✗ 宿主页有脚本错误：', r.errors.join(' | '))
      }
      continue
    }
    const flag = r.ok ? '✓' : '✗'
    const detail = { ...r }
    delete detail.name
    delete detail.ok
    delete detail.pngB64          // 十几 KB 的 base64 不往控制台倒
    console.log(`${flag} ${r.name}: ${JSON.stringify(detail)}`)
    if (!r.ok) { bad++; continue }

    // 内容层面的验收 —— "跑通了"不等于"画对了"
    if (r.name === 'glb-mixamo') {
      expect(r.clips === 4, '轨数应为 4（Idle/Run/TPose/Walk）', String(r.clips))
      expect(r.posed === true, '40% 处应套出姿势')
      expect(r.pngOk === true && r.pngBytes > 2000, '导出的人偶图应为非空 PNG', String(r.pngBytes))
      expect(r.frames === 4 && r.framesWithPose === 4, '抽 4 帧应拿到 4 份 pose')
      expect(r.framesWithPreview === 4, '每帧都应带缩略图', String(r.framesWithPreview))
      // 播放：滑块要真在走（tick 单调递增、能推进到 1 秒附近）
      expect(r.ticks >= 4, '播放应持续推出 animTick', String(r.ticks))
      expect(r.tickMax > 0.5, '播放应真的推进时间轴', String(r.tickMax))
      expect(r.tickMonotonic === true, 'animTick 必须单调（不能来回跳）', String(r.tickMonotonic))
      expect(r.playEvents >= 2, '播放要回 animPlaying on/off', String(r.playEvents))
      expect(r.distinctFrames >= 3, '抽出来的帧应互不相同（时间真在走）', String(r.distinctFrames))
    }
    if (r.name === 'fbx-mixamo') {
      expect(r.posed === true, 'FBX 40% 处应套出姿势')
      expect(r.pngOk === true && r.pngBytes > 2000, 'FBX 导出应为非空 PNG', String(r.pngBytes))
    }
    if (r.name.startsWith('glb-unreal') || r.name.startsWith('glb-kaykit')) {
      // 关键：换名字之后**仍然是 17 个锚点全认出**（rig.missing 空）
      expect((r.rig?.missing || []).length === 0, '锚点应全部认出', JSON.stringify(r.rig?.missing))
      expect(r.posed === true, '换名字后 40% 处应套出姿势')
      expect(r.rig?.mapped === 17, '应映射 17 个锚点', String(r.rig?.mapped))
      // 而且**对到的是对的那根骨头**，不是"凑够 17 个"
      const a = r.rig?.anchors || {}
      const want = r.name.startsWith('glb-unreal')
        ? { Hips: 'pelvis', LeftArm: 'upperarm_l', LeftForeArm: 'lowerarm_l', LeftHand: 'hand_l', LeftUpLeg: 'thigh_l', LeftLeg: 'calf_l', LeftFoot: 'foot_l' }
        : { Hips: 'hips', LeftArm: 'upper_arm_l', LeftForeArm: 'lower_arm_l', LeftHand: 'hand_l', LeftUpLeg: 'upper_leg_l', LeftLeg: 'lower_leg_l', LeftFoot: 'foot_l' }
      for (const [k, v] of Object.entries(want)) {
        expect(a[k] === v, `${k} 应对到 ${v}`, String(a[k]))
      }
    }
    if (r.name === 'glb-junk') {
      expect(String(r.msg || '').includes('骨骼认不出'), '应给出「骨骼认不出」的明确报错', r.msg)
    }
  }

  // 把人偶图落盘：数值对了不等于画面对（踩过"假对象测试会骗人"的亏）
  for (const r of results) {
    if (!r.pngB64) continue
    const p = join(OUT, `${r.name}.png`)
    writeFileSync(p, Buffer.from(r.pngB64, 'base64'))
    console.log(`  → ${p}`)
  }

  console.log(bad === 0 ? '\n全部通过 ✓' : `\n${bad} 项不达标 ✗`)
  return bad === 0 ? 0 : 1
}

main().then((c) => process.exit(c)).catch((e) => {
  console.error('验证脚本异常：', e)
  process.exit(1)
})
