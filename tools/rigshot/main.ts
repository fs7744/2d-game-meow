/**
 * 场景渲染的浏览器端核对页（不是产品页面，只给 tools/shot/rig_pixels.mjs 用）。
 *
 * 它调的是产品里**同一份代码**：`io.openProject` 读工程、`core/scene` 落位、
 * `scenePose.poseSceneActors` 摆姿势、`rigRender.drawScene` + `pickActorIndex` 绘制与点选。
 * 所以画在 Edge 里的像素就是编辑器画布上的像素。
 *
 * 和 `tools/verify_rig_scene.mjs`（Node 里比世界矩阵）互补：那边比数值，这边比真实光栅结果。
 */
import { pose as P, images, io, model, scene as sceneNs } from '../../web/src/rig/facade.js'
import { pickActorIndex, poseSceneActors } from '../../web/src/components/characters/rig/scenePose'
import { drawScene } from '../../web/src/components/characters/rig/rigRender'

const W = 900, H = 640
const cv = document.getElementById('c') as HTMLCanvasElement
cv.width = W
cv.height = H
const ctx = cv.getContext('2d', { willReadFrequently: true })!

/**
 * 读一个 rigby2d 示例工程：`project.json` 是工作文件里的项目数据（`Skeleton.fromJson`），
 * 图片是 `images/<name>.png` 的独立文件。与 `io.openWorkfile` 的读法一致（它读的是 zip 里的
 * 同名条目），只是这里的数据在磁盘上散着放。
 */
async function loadDemo(dir: string, store: any): Promise<{ doc: any; warnings: string[] }> {
  const r = await fetch(`/demo/${dir}/project.json`)
  if (!r.ok) throw new Error(`读取 ${dir}/project.json 失败（HTTP ${r.status}）`)
  const doc: any = model.Skeleton.fromJson(await r.text())
  const warnings: string[] = []
  for (const im of doc.images || []) {
    const name = im.name
    const png = await fetch(`/demo/${dir}/images/${name}.png`)
    if (!png.ok) { warnings.push(`缺图片 ${name}.png`); continue }
    store.insert(name, await io.decodeImage(new Uint8Array(await png.arrayBuffer())))
  }
  return { doc, warnings }
}

const stageImages = new images.Images()
const stageOpen = await loadDemo('rigbytron', stageImages)
const stage: any = stageOpen.doc

const actorImages = new images.Images()
const actorOpen = await loadDemo('nila', actorImages)
const actorDoc: any = actorOpen.doc
// 演员图片按 `@<id>/<name>` 并进同一个图像库（与 useRig.loadActorDoc 一致）。
for (const [name, entry] of actorImages.map as Map<string, any>) {
  stageImages.insert(`@nila/${name}`, entry.canvas)
}

const shot = stage.animations[0]
const clipAnim = actorDoc.animations.find((a: any) => a.name === 'wave') || actorDoc.animations[0]
stage.scene = sceneNs.emptyScene()
// 演员放到舞台右侧：两人叠在一起看不清前后关系，也量不准「各画各的」。
const setupPose = new P.Pose(stage)
setupPose.updateWorld(stage)
let stageRight = 0
let stageLeft = 0
let stageTop = 0
let stageBottom = 0
for (const q of setupPose.quads(stage)) {
  if ((q.color?.[3] ?? 1) <= 0) continue
  for (const [x, y] of q.positions) {
    stageRight = Math.max(stageRight, x)
    stageLeft = Math.min(stageLeft, x)
    stageTop = Math.max(stageTop, y)
    stageBottom = Math.min(stageBottom, y)
  }
}
const stageBox = [stageLeft, stageBottom, stageRight, stageTop]
const actorAt: [number, number] = [Math.round(stageRight) + 100, 0]
const actor = sceneNs.addActorToStage(stage, {
  id: 'nila', name: 'NILA', at: actorAt, ground: sceneNs.groundOffset(actorDoc),
})
actor.clips[shot.name] = [
  sceneNs.readClip({ animation: clipAnim.name, speed: 1, loops: 0, blend: 0.2 }),
].filter(Boolean)

const actors = new Map([['nila', { doc: actorDoc, pose: new P.Pose(actorDoc) }]])
const stagePose = new P.Pose(stage)

/** 自动取景：舞台 + 演员的可见顶点（跟画布「适配视图」同一套算法）。 */
const bounds = () => {
  let b: number[] | null = null
  const push = (pts: number[][]) => {
    for (const [x, y] of pts) {
      if (!b) b = [x, y, x, y]
      else { b[0] = Math.min(b[0], x); b[1] = Math.min(b[1], y); b[2] = Math.max(b[2], x); b[3] = Math.max(b[3], y) }
    }
  }
  for (const q of stagePose.quads(stage)) if ((q.color?.[3] ?? 1) > 0) push(q.positions)
  for (const rt of actors.values()) {
    for (const q of rt.pose.quads(rt.doc)) if ((q.color?.[3] ?? 1) > 0) push(q.positions)
  }
  return b as number[] | null
}

let view = { zoom: 0.35, x: 0, y: 60 }

function frame(t = 0, dt = 0) {
  stagePose.reset(stage)
  stagePose.apply(stage, shot, t, 1)
  stagePose.updateWorld(stage)
  poseSceneActors({
    doc: stage, stagePose, shot: shot.name, time: t, dt, playing: dt > 0, actors,
  })
}

function render(t = 0, selActor: number | null = null, dt = 0) {
  frame(t, dt)
  drawScene(ctx, stage, stagePose, stageImages, {
    width: W, height: H, view,
    showBones: false, showMesh: false, showImages: true,
    onion: false, sel: null, grid: false,
    actors, selActor,
  })
}

/**
 * 只画演员（舞台的图一律返回 null）。用来把「演员的位置」从台子里孤立出来量 ——
 * 全局质心会被不动的舞台稀释，量不出平移量。
 */
const actorOnlyStore = {
  tinted: (n: string, c: number[]) => (n.startsWith('@nila/') ? stageImages.tinted(n, c) : null),
  get: (n: string) => (n.startsWith('@nila/') ? stageImages.get(n) : null),
}

function renderActorOnly(t = 0) {
  frame(t, 0)
  drawScene(ctx, stage, stagePose, actorOnlyStore, {
    width: W, height: H, view,
    showBones: false, showMesh: false, showImages: true,
    onion: false, sel: null, grid: false,
    actors, selActor: null,
  })
}

/** 统计不透明像素（alpha > 8）的数量 / 包围盒 / 质心 —— 就是「画面上到底有没有东西」。 */
function metrics() {
  const d = ctx.getImageData(0, 0, W, H).data
  let n = 0, x0 = W, y0 = H, x1 = -1, y1 = -1, sx = 0, sy = 0
  const lo = [255, 255, 255], hi = [0, 0, 0]
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4
      if (d[i + 3] <= 8) continue
      n++
      sx += x; sy += y
      if (x < x0) x0 = x
      if (x > x1) x1 = x
      if (y < y0) y0 = y
      if (y > y1) y1 = y
      for (let c = 0; c < 3; c++) {
        if (d[i + c] < lo[c]) lo[c] = d[i + c]
        if (d[i + c] > hi[c]) hi[c] = d[i + c]
      }
    }
  }
  return {
    pixels: n,
    box: n ? [x0, y0, x1, y1] : null,
    centroid: n ? [sx / n, sy / n] : null,
    rgb: { r: [lo[0], hi[0]], g: [lo[1], hi[1]], b: [lo[2], hi[2]] },
  }
}

/** 屏幕坐标 → 世界的点选（产品里 RigCanvas 的 toWorld + pickActorIndex）。 */
function pickAtScreen(sx: number, sy: number) {
  const wx = (sx - W / 2 - view.x) / view.zoom
  const wy = -(sy - H / 2 - view.y) / view.zoom
  return pickActorIndex(stage, actors, [wx, wy])
}

/** 演员锚骨平移（与 useRig.transformActor 的 translate 分支同一换算）。 */
function moveActorBy(dx: number, dy: number) {
  const bi = stage.bones.findIndex((b: any) => b.name === actor.bone)
  const b = stage.bones[bi]
  const pw = b.parent >= 0 ? stagePose.world[b.parent] : null
  const v = pw ? pw.inverse().applyVector([dx, dy]) : [dx, dy]
  b.setup.x += v[0]
  b.setup.y += v[1]
}

const fit = () => {
  frame(0, 0)
  const b = bounds()
  if (!b) return null
  const zoom = Math.max(0.05, Math.min(4,
    Math.min(W / Math.max(1, b[2] - b[0]), H / Math.max(1, b[3] - b[1])) * 0.82))
  view = { zoom, x: -(b[0] + b[2]) / 2 * zoom, y: (b[1] + b[3]) / 2 * zoom }
  // 改完视图立刻重画，否则画布上还是旧的取景（调用方接着量像素会量到上一帧）。
  render(0, null)
  return { zoom, view }
}

const api = {
  info: () => ({
    stage: { bones: stage.bones.length, slots: stage.slots.length, shot: shot.name, duration: shot.duration },
    actor: {
      bones: actorDoc.bones.length, slots: actorDoc.slots.length, clip: clipAnim.name,
      ground: actor.ground, bone: actor.bone,
    },
    view,
    warnings: [...(stageOpen.warnings || []), ...(actorOpen.warnings || [])].slice(0, 6),
  }),
  fit, render, renderActorOnly, metrics, pickAtScreen, moveActorBy,
  setView: (v: any) => { view = v },
  png: () => cv.toDataURL('image/png'),
  /**
   * 只画演员自己的四边形轮廓（关掉图片），用来给「演员在画面里的位置」建立独立参照：
   * 这是纯几何，和光栅化的贴图无关。
   */
  actorSilhouette: () => {
    frame(0, 0)
    const pts: number[][] = []
    for (const q of actors.get('nila')!.pose.quads(actorDoc)) {
      if ((q.color?.[3] ?? 1) <= 0) continue
      for (const [x, y] of q.positions) {
        pts.push([W / 2 + view.x + x * view.zoom, H / 2 + view.y - y * view.zoom])
      }
    }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
    return {
      n: pts.length,
      box: pts.length
        ? [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
        : null,
    }
  },
}

;(window as any).__rigshot = api
render(0, null)
