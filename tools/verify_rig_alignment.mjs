/**
 * 骨骼编辑器「贴图不会错位」的回归验证。
 *
 * 对照的是 rigby2d 上游的真实实现：
 *   - Pose.quads() 给出 region 四角（左上/右上/右下/左下，Y 向上）、mesh 顶点与 UV。
 *   - editor/viewport.js 的 drawQuad 用 toLocal = (w/img.w, -h/img.h, -hw, +hh)，
 *     也就是「图片像素 (0,0) → 世界 positions[0]」。
 *   - 我们的 web/src/components/characters/rig/rigRender.ts 必须给出同一个仿射。
 *
 * 默认拿上游 demo 工程（tests/_out/rig_examples/，专有素材只用于本地验证、不入库）
 * 逐帧逐 quad 核对；demo 不在时用程序构造的最小骨架也能跑，方便 CI。
 *
 * 用法：node tools/verify_rig_alignment.mjs [demoDir ...]
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(ROOT, 'web', 'package.json'))
const esbuild = require('esbuild')

const rigDir = path.join(ROOT, 'web', 'src', 'rig')
const model = await import(pathToFile(rigDir, 'core/model.js'))
const poseNs = await import(pathToFile(rigDir, 'core/pose.js'))
const math = await import(pathToFile(rigDir, 'core/math.js'))

// rigRender.ts 是 TS，但只有类型注解与 import type；用 esbuild 转成 ESM 后从 data URL 载入，
// 这样测的是 UI 真正调用的那个 regionTransform，不是复制品。
const built = await esbuild.build({
  entryPoints: [path.join(ROOT, 'web/src/components/characters/rig/rigRender.ts')],
  bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
})
const { regionTransform } = await import(
  'data:text/javascript;base64,' + Buffer.from(built.outputFiles[0].text).toString('base64'))

function pathToFile(...parts) {
  return 'file://' + path.join(...parts).replace(/\\/g, '/')
}

const apply = (m, p) => [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]]
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1])

/** 旧实现（错误）：把 (0,0) 放到 pos[3]，会让 region 上下颠倒。留着当回归哨兵。 */
function legacyRegionTransform(pos, w, h) {
  return affine3([[0, 0], [w, 0], [0, h]], [pos[3], pos[2], pos[0]])
}

/** 与 rigRender.ts 的 affine3 同式（独立抄一份当哨兵，避免自己验证自己）。 */
function affine3(s, d) {
  const [s0, s1, s2] = s, [d0, d1, d2] = d
  const det = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s2[0] - s0[0]) * (s1[1] - s0[1])
  if (Math.abs(det) < 1e-9) return null
  const a = ((d1[0] - d0[0]) * (s2[1] - s0[1]) - (d2[0] - d0[0]) * (s1[1] - s0[1])) / det
  const c = ((d2[0] - d0[0]) * (s1[0] - s0[0]) - (d1[0] - d0[0]) * (s2[0] - s0[0])) / det
  const b = ((d1[1] - d0[1]) * (s2[1] - s0[1]) - (d2[1] - d0[1]) * (s1[1] - s0[1])) / det
  const e = ((d2[1] - d0[1]) * (s1[0] - s0[0]) - (d1[1] - d0[1]) * (s2[0] - s0[0])) / det
  return [a, b, c, e, d0[0] - a * s0[0] - c * s0[1], d0[1] - b * s0[0] - e * s0[1]]
}

function fixtureSkeleton() {
  const sk = new model.Skeleton()
  sk.name = 'alignment-fixture'
  const img = { name: 'probe', path: null, width: 120, height: 80 }
  sk.images.push(img)
  const region = model.makeAttachment('probe', 'probe', math.transform(18, -9, 0.35), 120, 80)
  sk.slots.push(model.makeSlot('region-slot', 0, region))
  const mesh = {
    ...model.makeAttachment('probe', 'probe', math.transform(-12, 22, -0.6), 120, 80),
    type: 'mesh', uvs: [0, 0, 1, 0, 1, 1, 0, 1], triangles: [0, 1, 2, 0, 2, 3],
    vertices: [-60, 40, 60, 40, 60, -40, -60, -40],
  }
  sk.slots.push(model.makeSlot('mesh-slot', 0, mesh))
  return sk
}

function checkDoc(sk, label) {
  const pose = new poseNs.Pose(sk)
  let regions = 0, meshes = 0, maxErr = 0, legacyCaught = 0
  const frames = []
  // setup pose + 每个动画的中间帧，覆盖非单位变换（旋转/缩放/位移）下的贴图。
  const sample = (tag) => {
    pose.reset(sk)
    pose.updateWorld(sk)
    frames.push(tag)
    for (const q of pose.quads(sk)) {
      const im = sk.images.find((i) => i.name === q.image)
      const w = im?.width || 120, h = im?.height || 80
      if (q.kind === 'region') {
        regions++
        const m = regionTransform(q.positions, w, h)
        if (!m) throw new Error(`${label}/${tag}: regionTransform 返回 null`)
        const corners = [[0, 0], [w, 0], [w, h], [0, h]]
        for (let k = 0; k < 4; k++) {
          const e = dist(apply(m, corners[k]), q.positions[k])
          maxErr = Math.max(maxErr, e)
          if (e > 1e-6) throw new Error(
            `${label}/${tag}: region 第 ${k} 角错位 ${e.toFixed(4)}（图片 ${corners[k]} → ${apply(m, corners[k])}，应为 ${q.positions[k]}）`)
        }
        // 哨兵：旧映射必须被这组数据判为错，否则测试本身没抓错能力。
        const old = legacyRegionTransform(q.positions, w, h)
        if (old && dist(apply(old, [0, 0]), q.positions[0]) > 1e-6) legacyCaught++
      } else if (q.kind === 'mesh' && q.triangles && q.uvs) {
        meshes++
        for (let i = 0; i + 2 < q.triangles.length; i += 3) {
          const ids = [q.triangles[i], q.triangles[i + 1], q.triangles[i + 2]]
          if (ids.some((k) => !q.positions[k])) continue
          const src = ids.map((k) => [q.uvs[k][0] * w, q.uvs[k][1] * h])
          const dst = ids.map((k) => q.positions[k])
          const m = affine3(src, dst)
          if (!m) continue
          for (let k = 0; k < 3; k++) {
            const e = dist(apply(m, src[k]), dst[k])
            maxErr = Math.max(maxErr, e)
            if (e > 1e-6) throw new Error(`${label}/${tag}: mesh 三角第 ${k} 点错位 ${e}`)
          }
        }
      }
    }
  }
  sample('setup')
  for (const a of sk.animations || []) {
    pose.reset(sk)
    pose.apply(sk, a, (a.duration || 1) * 0.45, 1)
    sample(`anim:${a.name}`)
  }
  return { label, regions, meshes, maxErr, legacyCaught, frames: frames.length }
}

const args = process.argv.slice(2)
const defaults = ['nila', 'rigbytron'].map((n) => path.join(ROOT, 'tests/_out/rig_examples', n))
const dirs = (args.length ? args : defaults).filter((d) => fs.existsSync(path.join(d, 'project.json')))
if (!dirs.length) {
  console.log('未找到上游 demo，使用内置最小骨架（region + mesh）验证。')
  const r = checkDoc(fixtureSkeleton(), 'fixture')
  console.log(`  ${r.label}: ${r.regions} region / ${r.meshes} mesh, 最大误差 ${r.maxErr.toExponential(2)}`)
} else {
  let totalR = 0, totalM = 0, worst = 0, caught = 0
  for (const d of dirs) {
    const txt = fs.readFileSync(path.join(d, 'project.json'), 'utf8')
    const sk = model.Skeleton.fromJson(txt)
    const r = checkDoc(sk, path.basename(d))
    totalR += r.regions; totalM += r.meshes; worst = Math.max(worst, r.maxErr); caught += r.legacyCaught
    console.log(`  ${r.label}: ${r.frames} 帧 / ${r.regions} region / ${r.meshes} mesh`
      + ` / 最大角误差 ${r.maxErr.toExponential(2)} / 旧映射被判错 ${r.legacyCaught} 次`)
  }
  console.log(`合计 ${totalR} region + ${totalM} mesh，最大误差 ${worst.toExponential(2)}，`
    + `错误映射被捕获 ${caught}/${totalR}`)
}
console.log('PASS: 贴图坐标与 rigby2d 上游契约一致，没有上下颠倒或 UV 错位。')
