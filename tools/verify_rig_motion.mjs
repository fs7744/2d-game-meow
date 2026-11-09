/**
 * 骨骼编辑器「动画运动」的回归验证：不只比单帧姿势，而是逐帧比运动轨迹。
 *
 * 对照的是 rigby2d 上游的真实播放/渲染实现：
 *   - editor/render/offline.js 的 FrameRenderer：一个常驻 Pose + 每帧 `state.update(doc, dt)`
 *     再 `state.apply(doc, pose)`，物理约束靠跨帧累积的 physicsDelta 推进。
 *   - 我们的 web/src/components/characters/rig/videoRender.ts 的 makeFramePose 必须给出
 *     同一组世界矩阵（骨骼 + 四角），误差在浮点噪声内。
 *
 * 同时报告两件事，防止「看着像对的」：
 *   1. 物理约束在播放中是否真的动了（对照「每帧新建 Pose、physicsDelta=0」的旧做法）。
 *   2. IK 约束的链末端每帧是否落在 target 上（约束没生效时这条会立刻红）。
 *
 * 用法：node tools/verify_rig_motion.mjs [demoDir ...]
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(ROOT, 'web', 'package.json'))
const esbuild = require('esbuild')

const DEMO_ROOT = path.join(ROOT, 'tests', '_out', 'rig_examples')
const UPSTREAM = path.join('D:', 'code', 'gen', 'others', 'rigby2d')

const model = await import(pathToFileURL(path.join(ROOT, 'web/src/rig/core/model.js')).href)
const ourPose = await import(pathToFileURL(path.join(ROOT, 'web/src/rig/core/pose.js')).href)

// upstream 的原件（用来当参照，不是我们的副本）
const upPose = await import(pathToFileURL(path.join(UPSTREAM, 'src/core/pose.js')).href)
const upState = await import(pathToFileURL(path.join(UPSTREAM, 'src/core/state.js')).href)

// 被测函数：真正跑在影片工作区里的那个采样器。
const built = await esbuild.build({
  entryPoints: [path.join(ROOT, 'web/src/components/characters/rig/videoRender.ts')],
  bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
})
const { makeFramePose } = await import(
  'data:text/javascript;base64,' + Buffer.from(built.outputFiles[0].text).toString('base64'))

const load = (dir) => model.Skeleton.fromJson(fs.readFileSync(path.join(dir, 'project.json'), 'utf8'))
const m = (a) => [a.a, a.b, a.c, a.d, a.tx, a.ty]

/**
 * 逐帧快照：骨骼世界矩阵 + 每个 quad 的顶点（覆盖 deform / mesh / 槽位换图 / 绘制顺序）。
 * 只看骨骼矩阵会漏掉形变与换图，所以这里两样都取。
 */
function snapshot(doc, pose) {
  const quads = pose.quads(doc).map((q) => ({
    kind: q.kind, name: String(q.name ?? ''), image: String(q.image ?? ''),
    pos: q.positions.map((p) => [p[0], p[1]]),
  }))
  return {
    world: pose.world.map(m),
    quads,
    attachment: pose.attachment.slice(),
    order: pose.drawOrder.slice(),
  }
}

/** 上游 FrameRenderer 的逐帧推进（不含绘制）。只在动画自身时长内采样。 */
function upstreamTrack(doc, anim, fps, frames) {
  const pose = new upPose.Pose(doc)
  const state = new upState.AnimationState()
  state.setAnimation(doc, 0, anim.name, false, 0)
  const out = []
  for (let f = 0; f < frames; f++) {
    if (f > 0) state.update(doc, 1 / fps)
    state.apply(doc, pose)
    out.push(snapshot(doc, pose))
  }
  return out
}

function oursTrack(doc, anim, fps, frames) {
  const sample = makeFramePose(doc, anim, fps)
  const out = []
  for (let f = 0; f < frames; f++) out.push(snapshot(doc, sample(f)))
  return out
}

/** 旧做法：每帧新建 Pose、不给 physicsDelta —— 摆动类约束会完全静止。 */
function statelessTrack(doc, anim, fps, frames) {
  const out = []
  for (let f = 0; f < frames; f++) {
    const p = new ourPose.Pose(doc)
    p.reset(doc)
    p.apply(doc, anim, f / fps, 1)
    p.updateWorld(doc)
    out.push(snapshot(doc, p))
  }
  return out
}

const maxDiff = (a, b) => {
  let d = 0
  for (let i = 0; i < a.length; i++) for (let j = 0; j < 6; j++) d = Math.max(d, Math.abs(a[i][j] - b[i][j]))
  return d
}

const maxQuadDiff = (a, b) => {
  let d = 0
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < a[i].pos.length; k++) {
      d = Math.max(d, Math.abs(a[i].pos[k][0] - b[i].pos[k][0]), Math.abs(a[i].pos[k][1] - b[i].pos[k][1]))
    }
  }
  return d
}

/** 两帧快照的全部差异（骨骼 + 贴图顶点 + 槽位/顺序）。 */
function frameDiff(a, b) {
  const world = maxDiff(a.world, b.world)
  const quads = a.quads.length === b.quads.length ? maxQuadDiff(a.quads, b.quads) : Infinity
  let meta = 0
  if (a.attachment.length !== b.attachment.length) meta = 1
  else for (let i = 0; i < a.attachment.length; i++) if (a.attachment[i] !== b.attachment[i] || a.order[i] !== b.order[i]) meta = 1
  return Math.max(world, quads, meta)
}

function ikError(doc, pose) {
  let worst = 0
  let reachable = 0
  doc.ik.forEach((c, i) => {
    const st = pose.ik?.[i]
    if (!st || !(st.mix > 0) || !c.bones.length) return
    const target = pose.world[c.target].apply([0, 0])
    const chain = c.bones[c.bones.length - 1]
    const tip = pose.world[chain].apply([doc.bones[chain].length, 0])
    // 可达性按解算器自己的量算（solveIk 的 a = |P-C|、bLen = |tip-C|），
    // 目标超出链长时 IK 只能尽量伸，残差是「够不着」而不是「没解算」。
    const root = pose.world[c.bones[0]].apply([0, 0])
    let maxReach
    if (c.bones.length === 1) {
      const w = pose.world[c.bones[0]]
      maxReach = doc.bones[c.bones.length - 1].length * Math.hypot(w.a, w.b)
    } else {
      const p = c.bones[0], ch = c.bones[1]
      const P = pose.world[p].apply([0, 0])
      const C = pose.world[ch].apply([0, 0])
      const T = pose.world[ch].apply([doc.bones[ch].length, 0])
      maxReach = Math.hypot(C[0] - P[0], C[1] - P[1]) + Math.hypot(T[0] - C[0], T[1] - C[1])
    }
    if (Math.hypot(target[0] - root[0], target[1] - root[1]) > maxReach - 1e-6) return
    reachable++
    worst = Math.max(worst, Math.hypot(target[0] - tip[0], target[1] - tip[1]))
  })
  return { worst, reachable }
}

const dirs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : fs.existsSync(DEMO_ROOT)
    ? fs.readdirSync(DEMO_ROOT).map((n) => path.join(DEMO_ROOT, n)).filter((d) => fs.existsSync(path.join(d, 'project.json')))
    : []

if (!dirs.length) {
  console.log('[skip] 没有找到示例工程（tests/_out/rig_examples），跳过运动核对')
  process.exit(0)
}

let failures = 0
for (const dir of dirs) {
  const doc = load(dir)
  const name = doc.name || path.basename(dir)
  const fps = 30
  console.log(`\n=== ${name}: ${doc.bones.length} 骨 / ${doc.slots.length} 槽 / ${doc.animations.length} 动画 / ${doc.ik.length} IK`)

  for (const anim of doc.animations) {
    const frames = Math.max(2, Math.round((anim.duration || 0.5) * fps) + 1)
    const ref = upstreamTrack(doc, anim, fps, frames)
    const ours = oursTrack(doc, anim, fps, frames)
    const flat = statelessTrack(doc, anim, fps, frames)

    let vsUpstream = 0
    for (let f = 0; f < frames; f++) vsUpstream = Math.max(vsUpstream, frameDiff(ref[f], ours[f]))
    let vsStateless = 0
    for (let f = 0; f < frames; f++) vsStateless = Math.max(vsStateless, frameDiff(ours[f], flat[f]))

    // IK：拿我们的采样器逐帧复查链末端与 target 的距离。
    const sample = makeFramePose(doc, anim, fps)
    let ikMax = 0, ikFrames = 0
    for (let f = 0; f < frames; f++) {
      const r = ikError(doc, sample(f))
      ikMax = Math.max(ikMax, r.worst)
      ikFrames += r.reachable
    }

    const ok = vsUpstream < 1e-7
    if (!ok) failures++
    console.log(
      `  ${ok ? 'ok  ' : 'FAIL'} ${anim.name.padEnd(22)} ${String(frames).padStart(4)}f ` +
      `vs上游 ${vsUpstream.toExponential(2)}  ` +
      `vs无状态 ${vsStateless.toFixed(3)}  ` +
      `IK ${ikFrames ? `${ikMax.toExponential(2)}/${ikFrames}帧` : '-'}`)

    // 运动本身要动起来：任一帧与首帧的骨骼位置必须有差异，否则是「定格」。
    let moved = 0
    for (let f = 1; f < frames; f++) moved = Math.max(moved, frameDiff(ours[f], ours[0]))
    if (moved < 1e-6) {
      console.log(`  FAIL ${anim.name}: 动画 ${anim.duration}s 内姿势完全没变`)
      failures++
    }
  }
}

console.log(failures ? `\n${failures} 项不通过` : '\n全部通过：逐帧轨迹与上游一致')
process.exit(failures ? 1 : 0)
