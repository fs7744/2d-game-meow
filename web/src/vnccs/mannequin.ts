/**
 * VNCCS 人偶渲染：在浏览器里用 three.js 把 MakeHuman v2 网格渲染成条件图。
 *
 * 依据（见 docs/VNCCS-REAL.md）：VNCCS 的 image1 是「白底灰色阴影 3D 人偶、
 * 相机机位定朝向」，不是线框骨架。实测真网格人偶一次就能让 LoRA 认朝向
 * （0°/90°/180° → 正面/侧面/背面），线框和手绘人偶都不行。
 *
 * 网格是 CC0 的 MakeHuman v2（`pose_studio_makehuman.v2.bin.gz`，VNMHJS02 格式），
 * 放在 `web/public/vnccs/`（gitignore）。材质/灯光/相机照抄官方：
 *   MeshPhongMaterial(color 0xaaaaaa, specular 0x111111, shininess 5)
 *   ambient #404040/0.5 + directional #ffffff/1.0
 *   PerspectiveCamera(45°)，白色背景，1024²
 */
import * as THREE from 'three'

// ---------------------------------------------------------------------------
// 网格解析（移植自 ComfyUI_VNCCS_Utils 的 morph_runtime.mjs）
// ---------------------------------------------------------------------------

const MAGIC = 'VNMHJS02'

type MorphPack = {
  baseVertices: Float32Array
  baseIndices: Uint32Array
  skinIndices: Uint16Array
  skinWeights: Float32Array
  joints: Record<string, number[]>
  bones: { name: string; parent: string | null; head_joint: string; tail_joint: string }[]
}

function typedView<T>(
  ctor: { new (b: ArrayBuffer, o: number, l: number): T },
  buffer: ArrayBuffer, dataOffset: number, desc?: { offset: number; length: number },
): T {
  if (!desc || !Number.isInteger(desc.offset) || !Number.isInteger(desc.length)) {
    return new ctor(buffer, 0, 0)
  }
  return new ctor(buffer, dataOffset + desc.offset, desc.length)
}

export function parseMorphPack(buffer: ArrayBuffer): MorphPack {
  const bytes = new Uint8Array(buffer)
  if (bytes.length < 12 || new TextDecoder().decode(bytes.subarray(0, 8)) !== MAGIC) {
    throw new Error('无效的 MakeHuman 资产头')
  }
  const view = new DataView(buffer)
  const headerLength = view.getUint32(8, true)
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + headerLength)))
  const dataOffset = 12 + headerLength + ((4 - ((12 + headerLength) % 4)) % 4)
  const joints: Record<string, number[]> = {}
  for (const [name, indices] of Object.entries(header.joints || {})) {
    joints[name] = Array.from(indices as number[])
  }
  return {
    baseVertices: typedView(Float32Array, buffer, dataOffset, header.base_vertices),
    baseIndices: typedView(Uint32Array, buffer, dataOffset, header.base_indices),
    skinIndices: typedView(Uint16Array, buffer, dataOffset, header.skin_indices),
    skinWeights: typedView(Float32Array, buffer, dataOffset, header.skin_weights),
    joints,
    bones: header.bones || [],
  }
}

function avgVertices(vertices: Float32Array, indices: number[]): [number, number, number] | null {
  if (!indices?.length) return null
  let x = 0, y = 0, z = 0, n = 0
  for (const v of indices) {
    const o = v * 3
    if (o < 0 || o + 2 >= vertices.length) continue
    x += vertices[o]; y += vertices[o + 1]; z += vertices[o + 2]; n++
  }
  return n ? [x / n, y / n, z / n] : null
}

export type Mannequin = {
  /** 含人偶的 group，几何已归一化（中心在原点、高度约 2 单位） */
  group: THREE.Group
  bones: Record<string, THREE.Bone>
  boneList: THREE.Bone[]
  boneHeads: Record<string, THREE.Vector3>     // 网格坐标（未归一化）里的骨 head/tail
  boneTails: Record<string, THREE.Vector3>
  restDir: Record<string, THREE.Vector3>       // 单位方向
}

export async function loadMannequin(url = '/vnccs/pose_studio_makehuman.v2.bin.gz'): Promise<Mannequin> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`加载人偶网格失败：HTTP ${res.status}`)
  const gz = await res.arrayBuffer()
  const stream = new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))
  const raw = await new Response(stream).arrayBuffer()
  return buildMannequin(parseMorphPack(raw))
}

export function buildMannequin(pack: MorphPack): Mannequin {
  const verts = pack.baseVertices
  const boneHeads: Record<string, THREE.Vector3> = {}
  const boneTails: Record<string, THREE.Vector3> = {}
  for (const b of pack.bones) {
    boneHeads[b.name] = new THREE.Vector3(...(avgVertices(verts, pack.joints[b.head_joint]) || [0, 0, 0]))
    boneTails[b.name] = new THREE.Vector3(...(avgVertices(verts, pack.joints[b.tail_joint]) || [0, 0, 0]))
  }

  // 骨层级
  const bones: Record<string, THREE.Bone> = {}
  const boneList: THREE.Bone[] = []
  const rootBones: THREE.Bone[] = []
  for (const b of pack.bones) {
    const bone = new THREE.Bone()
    bone.name = b.name
    bone.userData = { parentName: b.parent }
    bones[b.name] = bone
    boneList.push(bone)
  }
  for (const bone of boneList) {
    const pName = bone.userData.parentName as string | null
    if (pName && bones[pName]) {
      bones[pName].add(bone)
      const ph = boneHeads[pName], ch = boneHeads[bone.name]
      bone.position.copy(ch.clone().sub(ph))
    } else {
      rootBones.push(bone)
    }
  }

  const restDir: Record<string, THREE.Vector3> = {}
  for (const b of pack.bones) {
    restDir[b.name] = boneTails[b.name].clone().sub(boneHeads[b.name]).normalize()
  }

  // 几何体
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3))
  geometry.setAttribute('skinIndex', new THREE.BufferAttribute(pack.skinIndices, 4))
  geometry.setAttribute('skinWeight', new THREE.BufferAttribute(pack.skinWeights, 4))
  geometry.setIndex(new THREE.BufferAttribute(pack.baseIndices, 1))

  const skeleton = new THREE.Skeleton(boneList)
  const material = new THREE.MeshPhongMaterial({
    color: 0xaaaaaa, specular: 0x111111, shininess: 5, side: THREE.DoubleSide,
  })
  const mesh = new THREE.SkinnedMesh(geometry, material)
  mesh.add(...rootBones)
  mesh.bind(skeleton)

  // 归一化：把整个 mesh 缩放到高度 ≈ 2、中心在原点（bone 随 mesh 一起动）
  const box = new THREE.Box3().setFromObject(mesh)
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  const s = 2.0 / Math.max(size.x, size.y, size.z)
  mesh.position.sub(center.multiplyScalar(s))
  mesh.scale.setScalar(s)
  mesh.updateMatrixWorld(true)

  const group = new THREE.Group()
  group.add(mesh)

  return { group, bones, boneList, boneHeads, boneTails, restDir }
}

// ---------------------------------------------------------------------------
// 渲染：灯光 / 相机 / 白底 → dataURL
// ---------------------------------------------------------------------------

export type RenderOptions = {
  width?: number
  height?: number
  yawDeg?: number      // 相机绕 Y 的水平角（0=正面，90=侧面）
  pitchDeg?: number
  fill?: number        // 人物高度占画面高度的比例
}

let renderer: THREE.WebGLRenderer | null = null

export function renderMannequin(m: Mannequin, opts: RenderOptions = {}): string {
  const W = opts.width ?? 1024
  const H = opts.height ?? 1024
  const fill = opts.fill ?? 0.85

  if (!renderer) {
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false })
  }
  renderer.setPixelRatio(1)
  renderer.setSize(W, H)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0xffffff)
  scene.add(new THREE.AmbientLight(0x404040, 0.5))
  const dir = new THREE.DirectionalLight(0xffffff, 1.0)
  dir.position.set(1, 2, 3)
  scene.add(dir)
  scene.add(m.group)

  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 1000)
  const yaw = THREE.MathUtils.degToRad(opts.yawDeg ?? 0)
  const pitch = THREE.MathUtils.degToRad(opts.pitchDeg ?? 0)
  // 人偶高度归一化到 2，高度要占画面 fill 比例：
  // 相机会看到的高度 = 2 * dist * tan(fov/2)，令其等于 2/fill → dist = 1 / (fill * tan(fov/2))
  const dist = 1 / (fill * Math.tan(THREE.MathUtils.degToRad(22.5)))
  camera.position.set(
    Math.sin(yaw) * Math.cos(pitch) * dist,
    Math.sin(pitch) * dist,
    Math.cos(yaw) * Math.cos(pitch) * dist,
  )
  camera.lookAt(0, 0, 0)
  camera.updateMatrixWorld(true)

  renderer.render(scene, camera)
  return renderer.domElement.toDataURL('image/png')
}
