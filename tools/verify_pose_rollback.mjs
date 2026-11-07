/**
 * 失败回滚的**真单元测**（不用浏览器、不用 37MB 网格）。
 *
 * pose_guard.js 是零依赖纯模块，用假 viewer 就能验到最关键的一条：
 * applyTime 里的 `resetPose()` 发生在校验之前，一旦关键点算不出，
 * 用户摆了半天的姿势必须原样回来，而不是被清成站姿。
 *
 *     node tools/verify_pose_rollback.mjs
 */
import assert from 'node:assert/strict'
import {
  snapshotPose, restorePose, withPoseRollback,
} from '../web/public/vnccs-pose/pose_guard.js'

const POSE = {
  bones: { pelvis: [0, 15, 0], upperarm_l: [30, 0, 45] },
  modelRotation: [0, 10, 0],
  camera: { posX: 1, posY: 2, posZ: 20 },
}

function fakeViewer(initial = POSE) {
  const calls = []
  let pose = structuredClone(initial)
  const marker = { visible: false }
  return {
    calls,
    marker,
    getPose: () => structuredClone(pose),
    setPose: (p, preserve) => {
      calls.push(['setPose', structuredClone(p), preserve])
      pose = structuredClone(p)
    },
    updateMarkers: () => calls.push(['updateMarkers']),
    jointMarkers: [marker],
    requestRender: () => calls.push(['requestRender']),
    /** 模拟 applyTimeInner 里那句「先 reset 再算」 */
    resetPose: () => { calls.push(['resetPose']); pose = { bones: {} } },
    current: () => structuredClone(pose),
  }
}

let passed = 0
function check(name, fn) {
  fn()
  passed += 1
  console.log('  ✓', name)
}

await (async () => {
  check('成功：不调用 setPose，姿势停在采样结果', async () => {
    const v = fakeViewer()
    const ok = await withPoseRollback(v, async () => {
      v.resetPose()
      v.setPose({ bones: { pelvis: [1, 2, 3] } }, true)
      return true
    })
    assert.equal(ok, true)
    assert.deepEqual(v.current().bones, { pelvis: [1, 2, 3] })
    // 只有 fn 自己那一次 setPose，没有额外的还原调用
    assert.equal(v.calls.filter((c) => c[0] === 'setPose').length, 1)
  })

  check('返回 false：还原成进入前的完整姿势（含 modelRotation / camera）', async () => {
    const v = fakeViewer()
    const ok = await withPoseRollback(v, async () => {
      v.resetPose()                 // 关键点算不出的那条分支
      return false
    })
    assert.equal(ok, false)
    assert.deepEqual(v.current(), POSE)
    const restore = v.calls.find((c) => c[0] === 'setPose' && c[2] === true)
    assert.ok(restore, '还原必须带 preserveCamera=true，别把视角也换了')
    assert.deepEqual(restore[1], POSE)
    assert.equal(v.marker.visible, true, '关节标记要重新露出来')
  })

  check('抛异常：还原后原样抛出，不吞错', async () => {
    const v = fakeViewer()
    await assert.rejects(
      withPoseRollback(v, async () => { v.resetPose(); throw new Error('骨架映射失败') }),
      /骨架映射失败/,
    )
    assert.deepEqual(v.current(), POSE)
  })

  check('getPose 抛错时不会二次崩溃', async () => {
    const v = fakeViewer()
    v.getPose = () => { throw new Error('viewer 还没就绪') }
    assert.equal(snapshotPose(v), null)
    await assert.rejects(
      withPoseRollback(v, async () => { throw new Error('采样失败') }),
      /采样失败/,
    )
  })

  check('没有快照时 restorePose 是安全的 no-op', () => {
    const v = fakeViewer()
    assert.equal(restorePose(v, null), false)
    assert.equal(v.calls.length, 0)
    assert.equal(restorePose(null, POSE), false)
  })

  check('setPose 抛错也不影响调用方（返回 false）', () => {
    const v = fakeViewer()
    v.setPose = () => { throw new Error('还原失败') }
    assert.equal(restorePose(v, POSE), false)
  })
})()

console.log(`\n${passed}/6 通过`)
