/**
 * 场景演员的逐帧摆姿势：视差 → 每个演员按自己的片段走（带交叉淡入），物理用真实 dt 推进。
 *
 * 与 rigby2d `editor/actors.js` 的 `poseActors()` 同语义：动画名（shot）取舞台当前动画，
 * 时间用编辑器当前时间，暂停时 `resetPhysics()` 让摆动重新起振。
 * 计算全部走 core/scene.js，这里只是把「我们这边的状态」翻成它的入参。
 */
import { pose as P, scene as sceneNs, output } from '../../../rig/facade.js'

export type ActorRuntime = { doc: any; pose: any }

export type ScenePoseArgs = {
  doc: any
  stagePose: any
  /** 舞台当前动画名（不在动画模式时为空串）。 */
  shot: string
  /** 场景时间；< 0 表示不播片段（演员停在设定姿势）。 */
  time: number
  /** 距上一帧的秒数，只在播放时推进物理。 */
  dt: number
  playing: boolean
  actors: Map<string, ActorRuntime>
}

/** 摆好所有演员的姿势（就地写入各自的 pose），没有场景或没有已载入演员时什么都不做。 */
export function poseSceneActors(a: ScenePoseArgs): void {
  const scene = a.doc.scene
  if (!scene?.actors?.length || !a.actors.size) return
  // 被移出场景（或换了文档）的演员顺手清掉，免得一直占着图片与 Pose。
  const live = new Set<string>(scene.actors.map((x: any) => x.id))
  for (const id of [...a.actors.keys()]) if (!live.has(id)) a.actors.delete(id)

  // 视差：深层在镜头移动时漂得少。没有深度层就不必每帧多算一遍 setup 摄影机。
  if (Object.keys(scene.depth ?? {}).length) {
    sceneNs.applyParallax(a.doc, scene, a.stagePose, sceneNs.setupCameraPosition(a.doc))
  }
  for (const actor of scene.actors) {
    const rt = a.actors.get(actor.id)
    if (!rt) continue
    if (!a.playing) rt.pose.resetPhysics()
    sceneNs.poseActor(a.doc, a.stagePose, actor, rt.doc, a.shot, a.time, rt.pose, a.playing ? a.dt : 0)
  }
}

/**
 * 自动取景的世界包围盒 [x0, y0, x1, y1]：上游 `output.animationBounds` 只看舞台，
 * 演员被放到舞台旁边时会被裁出画面。有演员时把他们的顶点并进来（算法没改，仍是
 * output.js 的 outputFrame 做取景换算）。没有演员时原样返回，行为与上游一致。
 */
export function sceneBounds(doc: any, animIndex: number | null, fps: number,
  actors: Map<string, { doc: any; pose: any }>): number[] | null {
  let b: number[] | null = output.animationBounds(doc, animIndex, fps)
  const scene = doc.scene
  if (!scene?.actors?.length || !actors.size) return b
  const anim = animIndex === null || animIndex === undefined ? null : doc.animations[animIndex]
  const frames = anim ? Math.max(1, Math.round(anim.duration * fps)) : 0
  const stagePose = new P.Pose(doc)
  const shot = anim?.name ?? ''
  for (let f = 0; f <= frames; f++) {
    const t = Math.min(anim?.duration ?? 0, f / fps)
    stagePose.reset(doc)
    if (anim) stagePose.apply(doc, anim, t, 1)
    stagePose.updateWorld(doc)
    poseSceneActors({ doc, stagePose, shot, time: t, dt: 0, playing: false, actors })
    for (const rt of actors.values()) {
      for (const q of rt.pose.quads(rt.doc)) {
        if (q.color[3] <= 0) continue
        for (const [x, y] of q.positions) {
          if (!b) b = [x, y, x, y]
          else {
            if (x < b[0]) b[0] = x
            if (y < b[1]) b[1] = y
            if (x > b[2]) b[2] = x
            if (y > b[3]) b[3] = y
          }
        }
      }
    }
  }
  return b
}
