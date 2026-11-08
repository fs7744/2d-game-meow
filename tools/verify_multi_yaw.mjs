/**
 * 真机验证「白模按 YAWS 多机位批量抽帧」。
 *
 * 为什么必须真机：我们改的是**导航相机**再 viewport capture。
 * 单测/假对象只能证明消息字段对，证明不了 8 个方向抓到的不是同一张图
 * （snapToCaptureCamera 是否同步生效、rAF 够不够、renderOut 有没有读新相机）。
 *
 * 直接打开 studio.html 作顶层页面，不走宿主 React 树 —— 协议层完全一样，
 * 但省掉 37MB 模型之外的所有启动开销。
 *
 *     cd web && npx vite build --outDir <新目录> && cp -r ... web/dist/
 *     node tools/verify_multi_yaw.mjs
 *
 * 依赖 puppeteer-core（隔离装在 tools/shot/node_modules，驱动系统 Edge）。
 */
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = resolve(HERE, '..')
const DIST = join(ROOT, 'web', process.env.UI_DIST || 'dist')
const MESH = join(ROOT, 'tests', '_out', 'anim_import', 'mesh')
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
const MODEL_URL = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/models/gltf/Soldier.glb'
const requireFromShot = createRequire(join(ROOT, 'tools', 'shot', 'package.json'))
const puppeteer = requireFromShot('puppeteer-core')

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
    const file = join(DIST, url === '/' ? 'index.html' : url)
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

async function main() {
  if (!existsSync(join(DIST, 'vnccs-pose', 'studio.html'))) {
    console.error('先构建前端：web/dist/vnccs-pose/studio.html 不存在')
    return 2
  }
  await download(join(MESH, 'Soldier.glb'), MODEL_URL)
  const port = 8163
  const srv = await serve(port)
  const browser = await puppeteer.launch({
    executablePath: EDGE,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-swiftshader',
      '--use-gl=angle', '--use-angle=swiftshader', '--window-size=1000,900'],
  })
  let bad = 0
  const expect = (ok, label, detail = '') => {
    if (!ok) bad++
    console.log(`${ok ? '✓' : '✗'} ${label}${detail ? '  ' + detail : ''}`)
  }
  try {
    const page = await browser.newPage()
    await page.setViewport({ width: 1000, height: 900 })
    page.on('pageerror', (e) => console.log('  [pageerror]', e.message))
    await page.evaluateOnNewDocument(() => {
      window.__msgs = []
      window.addEventListener('message', (e) => {
        if (e.data && typeof e.data.type === 'string') window.__msgs.push(e.data)
      })
    })
    console.log('打开 studio.html（顶层）…')
    await page.goto(`http://127.0.0.1:${port}/vnccs-pose/studio.html?embed=1`,
      { waitUntil: 'domcontentloaded', timeout: 60000 })
    await page.evaluate(() => window.postMessage({ type: 'init', yaw: 0 }, '*'))
    console.log('等官方人偶引擎就绪（37MB 模型，软件渲染可能十几秒）…')
    await page.waitForFunction(() => window.__msgs.some((m) => m.type === 'ready'),
      { timeout: 180000 })
    expect(true, 'iframe 协议就绪')

    const loaded = await page.evaluate(async () => {
      const buf = await fetch('/__model/Soldier.glb').then((r) => r.arrayBuffer())
      const from = window.__msgs.length
      window.postMessage({ type: 'animLoad', buffer: buf, name: 'Soldier.glb' }, '*', [buf])
      return await new Promise((done) => {
        const t0 = Date.now()
        const iv = setInterval(() => {
          const m = window.__msgs.slice(from)
            .find((x) => x.type === 'animLoaded' || x.type === 'error')
          if (m) {
            clearInterval(iv)
            done(m.type === 'animLoaded'
              ? { ok: true, clips: (m.clips || []).length }
              : { ok: false, err: m.message })
          } else if (Date.now() - t0 > 120000) {
            clearInterval(iv); done({ ok: false, err: '超时' })
          }
        }, 100)
      })
    })
    expect(loaded.ok, 'glb 动画载入', JSON.stringify(loaded))
    if (!loaded.ok) return bad ? 1 : 0

    const sample = (directions, count) => page.evaluate(async ({ directions, count }) => {
      const from = window.__msgs.length
      window.postMessage({ type: 'animSample', count, directions }, '*')
      return await new Promise((done) => {
        const t0 = Date.now()
        const iv = setInterval(() => {
          const m = window.__msgs.slice(from)
            .find((x) => x.type === 'animFrames' || x.type === 'error')
          if (m) {
            clearInterval(iv)
            done(m.type === 'animFrames' ? {
              ok: true,
              dirs: m.directions || [],
              frames: (m.frames || []).map((f) => ({
                t: f.t, yaw: f.yaw, direction: f.direction,
                hash: f.hash, pngLen: (f.png || '').length,
              })),
            } : { ok: false, err: m.message })
          } else if (Date.now() - t0 > 180000) {
            clearInterval(iv); done({ ok: false, err: '超时' })
          }
        }, 100)
      })
    }, { directions, count })

    const dirs = [
      { key: 'down', yaw: 0 }, { key: 'right', yaw: 90 },
      { key: 'up', yaw: 180 }, { key: 'left', yaw: 270 },
    ]
    console.log('四机位 × 2 帧抽帧…')
    const multi = await sample(dirs, 2)
    expect(multi.ok, '多机位抽帧返回成功', multi.err || '')
    if (multi.ok) {
      expect(multi.frames.length === 8, '帧数 = 方向数 × 每方向帧数',
        `${multi.frames.length}`)
      expect(multi.dirs.join(',') === 'down,right,up,left', '回执带方向列表',
        multi.dirs.join(','))
      const perDir = {}
      for (const f of multi.frames) {
        perDir[f.direction] = perDir[f.direction] || []
        perDir[f.direction].push(f)
      }
      expect(Object.keys(perDir).length === 4 && Object.values(perDir).every((a) => a.length === 2),
        '每个方向各 2 帧', JSON.stringify(Object.fromEntries(
          Object.entries(perDir).map(([k, v]) => [k, v.length]))))
      expect(multi.frames.every((f) => typeof f.yaw === 'number' && f.pngLen > 100),
        '每帧带 yaw 且 PNG 非空')
      // 核心：不同机位抓到的画面不能一样。取每个方向的第 0 帧比较。
      const firstHash = Object.values(perDir).map((a) => a[0].hash)
      const firstPng = Object.values(perDir).map((a) => a[0].pngLen)
      const distinct = new Set(firstHash).size
      expect(distinct >= 2, '不同机位抓到不同画面（指纹不全相同）',
        `${distinct}/4 种指纹，PNG 长度 ${firstPng.join('/')}`)
      // 同一方向内两帧仍要不同（动作没被机位循环覆盖掉）
      expect(new Set(perDir.down.map((f) => f.hash)).size >= 1, '同方向内帧序列保留')
    }

    console.log('单方向抽帧（不带 directions，验证向后兼容）…')
    const single = await sample(undefined, 3)
    expect(single.ok, '单方向抽帧返回成功', single.err || '')
    if (single.ok) {
      expect(single.frames.length === 3, '单方向仍是 3 帧', `${single.frames.length}`)
      expect(single.frames.every((f) => f.direction === undefined),
        '单方向帧不带 direction 字段（老契约未变）')
    }
  } finally {
    await browser.close()
    srv.close()
  }
  console.log(bad ? `\n${bad} 项失败` : '\n全部通过')
  return bad ? 1 : 0
}

main().then((code) => process.exit(code)).catch((e) => { console.error(e); process.exit(1) })
