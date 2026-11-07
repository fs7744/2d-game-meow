/**
 * 采样 / 导入失败时把人偶还原到「进这一步之前」的姿势。
 *
 * 官方 0.7.1 修过同类问题：FBX retarget 无论成功失败都要还原 viewer pose。
 * 我们这边的坑更直接 —— `applyTimeInner()` 在任何校验**之前**就
 * `viewer.resetPose()`，一旦关键点算不出或骨架映射失败，用户摆了半天的
 * 姿势就被清成站姿，而且界面只报一句「这一帧套不上人偶」。
 *
 * 单独成模块是为了能拿一个假 viewer 在 Node 里真跑单元测：浏览器里那套
 * 37MB 网格不该成为验证「失败会不会丢姿势」的前提。
 */

/** 取一份可还原的快照。getPose() 带骨骼旋转 / IK effector / 极点 / 相机。 */
export function snapshotPose(viewer) {
  try {
    return viewer?.getPose?.() ?? null
  } catch {
    return null
  }
}

/** 还原成功后关节标记要重新露出来，否则用户会以为人偶不见了。 */
export function restorePose(viewer, snap) {
  if (!snap || !viewer) return false
  try {
    viewer.setPose?.(snap, true)          // preserveCamera：别顺手把视角也换了
    viewer.updateMarkers?.()
    viewer.jointMarkers?.forEach?.((m) => { m.visible = true })
    viewer.requestRender?.()
    return true
  } catch {
    return false
  }
}

/**
 * `fn()` 返回 false 或抛异常 → 还原；返回 true → 保留它的结果。
 * 用「返回值」判定而不是只靠异常：`applyTimeInner` 的失败分支是 return false。
 */
export async function withPoseRollback(viewer, fn) {
  const snap = snapshotPose(viewer)
  let ok = false
  try {
    ok = await fn()
  } catch (e) {
    restorePose(viewer, snap)
    throw e
  }
  if (!ok) restorePose(viewer, snap)
  return ok
}
