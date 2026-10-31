/**
 * 把骨骼编辑器的 18 个关节（COCO 顺序）映射成 MakeHuman 53 根骨的旋转。
 *
 * 原理与坑（见 docs/VNCCS-REAL.md 与实测）：
 * - 我们的骨架已给出肩/肘/腕、髋/膝/踝、颈/鼻这些中间点，不需要解 IK，只需把每根骨
 *   的「额状面方向」对齐到对应关节段。**朝向不烘焙进骨骼**——由 renderMannequin 的
 *   相机绕圈（yaw）决定，姿势与视角解耦（和 VNCCS 官方一致）。
 * - 只做**额状面 2D 对齐**：只改骨在 x-y 平面里的角度，保持它原本的前后（z）分量。
 *   早先试过 3D `setFromUnitVectors`，编辑器的 z（±0.05 的微深度）会被当成真实深度，
 *   让锁骨绕 Y 轴扭转、手被甩到胸口——mesh 在肩部撕成碎片。所以这里**忽略编辑器 z**，
 *   用网格自带的 z 保持自然前后。
 * - 只绕世界 Z 轴转，天然不引入绕骨轴的滚转（twist），肘/膝不会反向弯。
 */
import * as THREE from 'three'
import type { Mannequin } from './mannequin'

export type Joint3 = { x: number; y: number; z: number }

/** COCO 18 关节索引 */
const NOSE = 0, NECK = 1
const RS = 2, RE = 3, RW = 4
const LS = 5, LE = 6, LW = 7
const RH = 8, RK = 9, RA = 10
const LH = 11, LK = 12, LA = 13

/** 编辑器关节 → 网格额状面（x 右、y 上）。z 忽略（用网格自带 z）。 */
function frontal(j: Joint3): { x: number; y: number } {
  return { x: j.x - 0.5, y: 0.5 - j.y }
}

export function solvePose(m: Mannequin, joints: Joint3[]): void {
  const P = joints.map(frontal)
  const mid = (i: number, k: number) => ({ x: (P[i].x + P[k].x) / 2, y: (P[i].y + P[k].y) / 2 })
  const hipMid = mid(LH, RH)

  // 每根骨的目标额状面方向（null = 跟随父骨保持静息）
  const target: Record<string, { x: number; y: number } | null> = {}
  const D = (name: string, a: { x: number; y: number }, b: { x: number; y: number }) => {
    const dx = b.x - a.x, dy = b.y - a.y
    target[name] = (dx * dx + dy * dy) < 1e-12 ? null : { x: dx, y: dy }
  }

  D('pelvis', hipMid, P[NECK])
  D('spine_01', hipMid, P[NECK])
  D('spine_02', hipMid, P[NECK])
  D('spine_03', hipMid, P[NECK])
  D('head', P[NECK], P[NOSE])
  D('clavicle_l', P[NECK], P[LS])
  D('clavicle_r', P[NECK], P[RS])
  D('upperarm_l', P[LS], P[LE]); D('lowerarm_l', P[LE], P[LW])
  D('upperarm_r', P[RS], P[RE]); D('lowerarm_r', P[RE], P[RW])
  D('thigh_l', P[LH], P[LK]); D('calf_l', P[LK], P[LA])
  D('thigh_r', P[RH], P[RK]); D('calf_r', P[RK], P[RA])

  for (const b of m.boneList) b.quaternion.identity()

  const worldQuat = new Map<THREE.Bone, THREE.Quaternion>()
  const Z = new THREE.Vector3(0, 0, 1)
  const visit = (bone: THREE.Bone, parentWorld: THREE.Quaternion) => {
    const t = target[bone.name]
    let w = parentWorld
    if (t) {
      const r = m.restDir[bone.name]
      const aRest = Math.atan2(r.y, r.x)
      const aTgt = Math.atan2(t.y, t.x)
      const q = new THREE.Quaternion().setFromAxisAngle(Z, aTgt - aRest)
      bone.quaternion.copy(parentWorld.clone().invert().multiply(q))
      w = q
    } else {
      bone.quaternion.identity()
    }
    worldQuat.set(bone, w)
    for (const child of bone.children) {
      if (child instanceof THREE.Bone) visit(child, w)
    }
  }
  for (const b of m.boneList) {
    if (!(b.parent instanceof THREE.Bone)) visit(b, new THREE.Quaternion())
  }
}
