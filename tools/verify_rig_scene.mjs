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
  math: await import(pathToFileURL(path.join(ROOT, 'web/src/rig/core/math.js')).href),
  anim: await import(pathToFileURL(path.join(ROOT, 'web/src/rig/core/animation.js')).href),
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
const { poseSceneActors, pickActorIndex } = await import(
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

// ---- 演员点选：点到演员身上必须选中它，点空白不能误选 ----------------------------------
// 用同一份 fixture（舞台 drawOrder：back | 演员 | front），取演员一个可见 region 的中心点。
const pickMap = new Map([['a1', { doc: actorDoc, pose: aPose }]])
let inside = null
for (const q of aPose.quads(actorDoc)) {
  if ((q.color?.[3] ?? 1) <= 0 || q.positions.length !== 4) continue
  inside = [
    q.positions.reduce((s, p) => s + p[0], 0) / 4,
    q.positions.reduce((s, p) => s + p[1], 0) / 4,
  ]
  break
}
const hitIdx = inside ? pickActorIndex(fx, pickMap, inside) : -2
const missIdx = inside ? pickActorIndex(fx, pickMap, [inside[0] + 1e5, inside[1] + 1e5]) : -3
// 后加的演员在上面：两个演员叠在一起时点中的是最后一个。
const stack = [fx.scene.actors[0], { ...fx.scene.actors[0], id: 'a2' }]
fx.scene.actors = stack
pickMap.set('a2', { doc: actorDoc, pose: aPose })
const topIdx = inside ? pickActorIndex(fx, pickMap, inside) : -4
fx.scene.actors = [stack[0]]
const pickOk = hitIdx === 0 && missIdx === -1 && topIdx === 1
console.log(`  演员点选 ${pickOk ? 'ok' : 'FAIL'}（命中 ${hitIdx} / 空白 ${missIdx} / 叠放 ${topIdx}）`)

// ---- 镜头运动：给 `camera` 骨打关键帧后，深度层按焦距视差（深层漂得少） ----------------
// 上游的摄影机就是舞台上一根叫 camera 的顶层骨，镜头运动 = 这根骨的关键帧。这里验的是
// 「关键帧真的驱动视差」以及幅度符合 depthFactor 的闭式解，防止 UI 那边写错骨名或通道。
const camDoc = new our.model.Skeleton()
camDoc.name = 'camera-fixture'
camDoc.addBone(our.model.makeBone('near', null, our.math.transform(0, 0, 0), 40))
camDoc.addBone(our.model.makeBone('far', null, our.math.transform(0, 200, 0), 40))
const camIdx = camDoc.addBone(our.model.makeBone('camera', null, our.math.transform(0, 0, 0), 0))
camDoc.scene = our.scene.emptyScene()
camDoc.scene.focal = 1000
camDoc.scene.depth = { near: 300, far: 900 }
const camAnim = new our.anim.Animation('shot', 0.6)
camDoc.animations.push(camAnim)
const camTl = camAnim.boneTimelineMut(camIdx)
our.anim.setKey(camTl.translate, 0, [0, 0], 'linear')
our.anim.setKey(camTl.translate, 0.5, [200, 0], 'linear')

const camPose = new our.pose.Pose(camDoc)
const camRef = our.scene.setupCameraPosition(camDoc)
const camPos = (t) => {
  camPose.reset(camDoc)
  camPose.apply(camDoc, camAnim, t, 1)
  camPose.updateWorld(camDoc)
  our.scene.applyParallax(camDoc, camDoc.scene, camPose, camRef)
  const at = (n) => {
    const i = camDoc.bones.findIndex((b) => b.name === n)
    return camPose.world[i].apply([0, 0])[0]
  }
  return [at('near'), at('far')]
}
const c0 = camPos(0), c5 = camPos(0.5)
const dNear = c5[0] - c0[0], dFar = c5[1] - c0[1]
const expNear = 200 * (1 - 1000 / (1000 + 300))
const expFar = 200 * (1 - 1000 / (1000 + 900))
const camOk = Math.abs(dNear - expNear) < 1e-9 && Math.abs(dFar - expFar) < 1e-9
  && Math.abs(dFar) > Math.abs(dNear)
console.log(`  镜头视差 ${camOk ? 'ok' : 'FAIL'}（深度 300 → ${dNear.toFixed(2)} / 深度 900 → ${dFar.toFixed(2)}，`
  + `闭式解 ${expNear.toFixed(2)} / ${expFar.toFixed(2)}）`)

const ok = inBetween && pickOk && camOk
console.log(ok ? '\n通过：场景逐帧与上游一致，演员按槽位插画，演员可点选，镜头关键帧驱动视差' : '\n不通过')
process.exit(ok ? 0 : 1)
