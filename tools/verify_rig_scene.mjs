/**
 * 场景（多角色同框）的运动核对：舞台 + 演员各自的片段、交叉淡入、视差与物理。
 *
 * 参照仍是 rigby2d 上游原件：`core/scene.js` 的 `poseActor()` + `applyParallax()`，
 * 上游离线渲染里就是这么逐帧摆的（`render/offline.js::FrameRenderer.poseScene`）。
 * 被测的是我们的 `scenePose.ts::poseSceneActors`（画布与影片渲染共用它）。
 *
 * 用法：node tools/verify_rig_scene.mjs [stageDir] [actorDir]
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(ROOT, 'web', 'package.json'))
const esbuild = require('esbuild')

const DEMO = path.join(ROOT, 'tests', '_out', 'rig_examples')
const UPSTREAM = path.join('D:', 'code', 'gen', 'others', 'rigby2d')

const our = {
  model: await import(pathToFileURL(path.join(ROOT, 'web/src/rig/core/model.js')).href),
  pose: await import(pathToFileURL(path.join(ROOT, 'web/src/rig/core/pose.js')).href),
  scene: await import(pathToFileURL(path.join(ROOT, 'web/src/rig/core/scene.js')).href),
}
const up = {
  pose: await import(pathToFileURL(path.join(UPSTREAM, 'src/core/pose.js')).href),
  scene: await import(pathToFileURL(path.join(UPSTREAM, 'src/core/scene.js')).href),
  state: await import(pathToFileURL(path.join(UPSTREAM, 'src/core/state.js')).href),
}

const built = await esbuild.build({
  entryPoints: [path.join(ROOT, 'web/src/components/characters/rig/scenePose.ts')],
  bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
})
const { poseSceneActors } = await import(
  'data:text/javascript;base64,' + Buffer.from(built.outputFiles[0].text).toString('base64'))

const args = process.argv.slice(2)
const stageDir = args[0] || path.join(DEMO, 'rigbytron')
const actorDir = args[1] || path.join(DEMO, 'nila')
for (const d of [stageDir, actorDir]) {
  if (!fs.existsSync(path.join(d, 'project.json'))) {
    console.log(`[skip] 缺少示例工程：${d}`)
    process.exit(0)
  }
}

const load = (dir) => our.model.Skeleton.fromJson(fs.readFileSync(path.join(dir, 'project.json'), 'utf8'))
const m = (a) => [a.a, a.b, a.c, a.d, a.tx, a.ty]

const stage = load(stageDir)
const actorDoc = load(actorDir)
if (!stage.animations.length) { console.log('[skip] 舞台没有动画'); process.exit(0) }

// 舞台：一个动画片段作为 shot；演员：从自己的动画里挑一段，带交叉淡入 + 循环。
const shot = stage.animations[0]
stage.scene = our.scene.emptyScene()
const at = [120, 0]
const actor = our.scene.addActorToStage(stage, {
  id: 'nila', name: '演员 NILA', at, ground: our.scene.groundOffset(actorDoc),
})
const pick = (n) => actorDoc.animations.find((a) => a.name === n) || actorDoc.animations[0]
actor.clips[shot.name] = [
  our.scene.readClip({ animation: pick('wave').name, start: 0, in: 0, out: null, speed: 1, loops: 0, blend: 0.2 }),
].filter(Boolean)
// 视差：舞台根骨给个深度，镜头动起来时深层要漂得少。
stage.scene.depth[stage.bones[0].name] = 300

const fps = 30
const frames = Math.max(2, Math.min(240, Math.round((shot.duration || 1) * fps) + 1))

/** 我们的路径：和 editor / 离屏渲染一样，逐帧推进舞台 + 演员。 */
function ourTrack() {
  const out = []
  const stagePose = new our.pose.Pose(stage)
  const actors = new Map([['nila', { doc: actorDoc, pose: new our.pose.Pose(actorDoc) }]])
  let time = 0
  for (let f = 0; f < frames; f++) {
    if (f > 0) { time += 1 / fps; stagePose.physicsDelta = 1 / fps }
    stagePose.reset(stage)
    stagePose.apply(stage, shot, time, 1)
    stagePose.updateWorld(stage)
    poseSceneActors({
      doc: stage, stagePose, shot: shot.name, time, dt: f > 0 ? 1 / fps : 0,
      playing: true, actors,
    })
    out.push({ stage: stagePose.world.map(m), actor: actors.get('nila').pose.world.map(m) })
  }
  return out
}

/** 上游路径：AnimationState 推时间 + FrameRenderer.poseScene 的摆法。 */
function upTrack() {
  const out = []
  const stagePose = new up.pose.Pose(stage)
  const state = new up.state.AnimationState()
  state.setAnimation(stage, 0, shot.name, false, 0)
  const actorPose = new up.pose.Pose(actorDoc)
  const setupCamera = up.scene.setupCameraPosition(stage)
  for (let f = 0; f < frames; f++) {
    if (f > 0) state.update(stage, 1 / fps)
    state.apply(stage, stagePose)
    const t = f / fps
    up.scene.applyParallax(stage, stage.scene, stagePose, setupCamera)
    up.scene.poseActor(stage, stagePose, actor, actorDoc, shot.name, t, actorPose, f > 0 ? 1 / fps : 0)
    out.push({ stage: stagePose.world.map(m), actor: actorPose.world.map(m) })
  }
  return out
}

const diff = (a, b) => {
  let d = 0
  for (let i = 0; i < a.length; i++) for (let j = 0; j < 6; j++) d = Math.max(d, Math.abs(a[i][j] - b[i][j]))
  return d
}

const ours = ourTrack()
const ref = upTrack()
let stageErr = 0, actorErr = 0
for (let f = 0; f < frames; f++) {
  stageErr = Math.max(stageErr, diff(ours[f].stage, ref[f].stage))
  actorErr = Math.max(actorErr, diff(ours[f].actor, ref[f].actor))
}

// 演员必须真的跟着锚骨走：没有 placement 时会画在原点。
const placementOk = Math.abs(ours[0].actor[0][4] - 0) > 1e-6 || Math.abs(ours[0].actor[0][5]) > 1e-6
// 演员必须在动（片段在播）。
let moved = 0
for (let f = 1; f < frames; f++) moved = Math.max(moved, diff(ours[f].actor, ours[0].actor))

console.log(`舞台 ${stage.bones.length} 骨 / 演员 ${actorDoc.bones.length} 骨，${frames} 帧`)
console.log(`  舞台  vs 上游 ${stageErr.toExponential(2)}`)
console.log(`  演员  vs 上游 ${actorErr.toExponential(2)}  位移 ${moved.toFixed(2)}`)
console.log(`  锚骨落位 ${placementOk ? 'ok' : 'FAIL'}（演员被放到 x=${at[0]}）`)

const bad = stageErr > 1e-7 || actorErr > 1e-7 || !placementOk || moved < 1e-6
if (bad) {
  console.log('\n不通过')
  process.exit(1)
}

// ---- 绘制顺序：演员必须画在自己的槽位处（前后关系由舞台 drawOrder 决定） ----------------
const rendered = await esbuild.build({
  entryPoints: [path.join(ROOT, 'web/src/components/characters/rig/rigRender.ts')],
  bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
})
const { drawScene } = await import(
  'data:text/javascript;base64,' + Buffer.from(rendered.outputFiles[0].text).toString('base64'))

function fixture() {
  const sk = new our.model.Skeleton()
  sk.name = 'scene-fixture'
  for (const name of ['back', 'front']) {
    sk.images.push({ name, path: null, width: 10, height: 10 })
    sk.slots.push(our.model.makeSlot(name, 0, our.model.makeAttachment(name, name, our.model.makeBone('b', null, { x: 0, y: 0, rotation: 0, scale_x: 1, scale_y: 1 }, 0).setup, 10, 10)))
  }
  sk.scene = our.scene.emptyScene()
  our.scene.addActorToStage(sk, { id: 'a1', name: '演员', at: [0, 0] })
  return sk
}

const fx = fixture()
// 把演员槽挪到中间：back | 演员 | front
const actorSlot = fx.slots.pop()
fx.slots.splice(1, 0, actorSlot)
const actorForDraw = fx.scene.actors[0]
actorForDraw.slot = actorSlot.name

const logged = []
const ctx = new Proxy({}, {
  get(_t, k) {
    if (k === 'drawImage') return (img) => logged.push(img?.tag ?? '?')
    if (k === 'canvas') return { width: 100, height: 100 }
    return () => {}
  },
  set() { return true },
})
const store = {
  get: (n) => ({ width: 10, height: 10, tag: n }),
  tinted: (n) => ({ width: 10, height: 10, tag: n }),
}
const stagePose = new our.pose.Pose(fx)
const aPose = new our.pose.Pose(actorDoc)
aPose.updateWorld(actorDoc)
drawScene(ctx, fx, stagePose, store, {
  width: 100, height: 100, view: { zoom: 1, x: 0, y: 0 },
  showBones: false, showMesh: false, showImages: true, onion: false, sel: null, grid: false,
  actors: new Map([['a1', { doc: actorDoc, pose: aPose }]]),
})
const iBack = logged.indexOf('back')
const iFront = logged.indexOf('front')
const actorDraws = logged.filter((t) => t.startsWith('@a1/')).length
const inBetween = iBack === 0 && logged.indexOf('front') > iBack && actorDraws > 0
  && logged.slice(iBack + 1, iFront).every((t) => t.startsWith('@a1/'))
console.log(`  绘制顺序 ${inBetween ? 'ok' : 'FAIL'}（${logged.length} 次 drawImage：back → 演员 ${actorDraws} 张 → front）`)

console.log(inBetween ? '\n通过：场景逐帧与上游一致，演员按槽位插画' : '\n不通过（绘制顺序）')
process.exit(inBetween ? 0 : 1)
