// Node 侧验证 mannequin 的网格解析 + 骨骼/蒙皮构建（不含 WebGL 渲染）。
// 用法：node --experimental-strip-types mannequin-node-test.mjs
import { readFileSync } from 'node:fs'
import * as THREE from 'three'
import { parseMorphPack, buildMannequin } from './src/vnccs/mannequin.ts'

const gz = readFileSync('public/vnccs/pose_studio_makehuman.v2.bin.gz')
const raw = await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()

const pack = parseMorphPack(raw)
console.log('顶点', pack.baseVertices.length / 3, '三角', pack.baseIndices.length / 3,
  '骨', pack.bones.length, 'skinIndices', pack.skinIndices.length / 4)

const m = buildMannequin(pack)
console.log('boneList', m.boneList.length, 'bones 对象', Object.keys(m.bones).length)

// bbox 归一化后高度应 ≈ 2
m.group.updateMatrixWorld(true)
const box = new THREE.Box3().setFromObject(m.group)
const size = box.getSize(new THREE.Vector3())
console.log('归一化后 bbox 尺寸', size.x.toFixed(3), size.y.toFixed(3), size.z.toFixed(3))

// restDir 都是单位向量
let bad = 0
for (const [k, v] of Object.entries(m.restDir)) if (Math.abs(v.length() - 1) > 1e-3) { bad++; console.log('非单位', k, v.length()) }
console.log('非单位 restDir 数', bad)

// 蒙皮数据一致性：取几个顶点，验证 skinIndex < 骨数 且权重和≈1
const verts = pack.baseVertices
let wbad = 0
for (let i = 0; i < pack.skinWeights.length; i += 4) {
  const s = pack.skinWeights[i] + pack.skinWeights[i + 1] + pack.skinWeights[i + 2] + pack.skinWeights[i + 3]
  if (Math.abs(s - 1) > 0.02) wbad++
  for (let k = 0; k < 4; k++) if (pack.skinIndices[i + k] >= m.boneList.length) wbad++
}
console.log('蒙皮权重/索引异常顶点数', wbad)

// 骨骼层级：没有孤儿（除了 Root），每个非根骨都有 parent
let orphan = 0
for (const b of m.boneList) { if (b.name !== 'Root' && !b.parent) orphan++ }
console.log('孤儿骨数（应 0）', orphan)
console.log('OK')
