/**
 * VNCCS PoseStudio 人偶渲染入口（整套移植官方，不写任何自己的 IK/渲染）。
 *
 * 链路（全部官方现成）：loadMorphPack → solveMorph → buildStaticModelData →
 * PoseViewerCore.loadData → convertOpenPoseToPose(18 关节) → setPose →
 * capture(yawDeg=朝向)。
 *
 * 供两种方式调用：
 *  1. postMessage（iframe 集成）：父页 post { type:'render', joints, azimuth, ... }
 *     本页回 { type:'result', dataURL } 或 { type:'error', message }。
 *  2. 直接访问（?test=1）：渲染「站立」预设的 0/90/180/270 四个朝向，便于自检。
 */
import { PoseViewerCore } from './pose_studio/core.js'
import { loadMorphPack, solveMorph, buildStaticModelData } from './pose_studio/morph_runtime.mjs'
import { convertOpenPoseToPose } from './pose_studio/imports/openpose.js'

const canvas = document.getElementById('stage')
const status = document.getElementById('status')
const out = document.getElementById('out')

const COCO_NAMES = [
  'nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist',
  'l_shoulder', 'l_elbow', 'l_wrist', 'r_hip', 'r_knee',
  'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye',
  'l_eye', 'r_ear', 'l_ear',
]

const STAND = [
  { x: 0.5, y: 0.075 }, { x: 0.5, y: 0.155 }, { x: 0.42, y: 0.18 }, { x: 0.39, y: 0.339 }, { x: 0.38, y: 0.497 },
  { x: 0.58, y: 0.18 }, { x: 0.61, y: 0.339 }, { x: 0.62, y: 0.497 }, { x: 0.46, y: 0.52 }, { x: 0.45, y: 0.72 },
  { x: 0.44, y: 0.93 }, { x: 0.54, y: 0.52 }, { x: 0.55, y: 0.72 }, { x: 0.56, y: 0.93 }, { x: 0.47, y: 0.069 },
  { x: 0.53, y: 0.069 }, { x: 0.45, y: 0.075 }, { x: 0.55, y: 0.075 },
]

let viewerPromise = null

function buildModelData(result, staticData) {
  const bones = staticData.bones.map((bone, i) => {
    const o = i * 6
    const hp = Array.from(result.bonePositions.subarray(o, o + 3))
    const tp = Array.from(result.bonePositions.subarray(o + 3, o + 6))
    const dx = tp[0] - hp[0], dy = tp[1] - hp[1], dz = tp[2] - hp[2]
    return { name: bone.name, parent: bone.parent || null, headPos: hp, tailPos: tp, length: Math.hypot(dx, dy, dz) }
  })
  return {
    vertices: result.vertices, uvs: staticData.uvs, indices: staticData.indices,
    bones, skinIndices: staticData.skinIndices, skinWeights: staticData.skinWeights,
    landmarks: result.landmarks || {}, landmark_indices: result.landmarkIndices || {},
  }
}

async function getViewer(skinMode = 'flat_color') {
  if (viewerPromise) return viewerPromise
  viewerPromise = (async () => {
    const morphPack = await loadMorphPack()
    const result = solveMorph(morphPack, {})
    const staticData = buildStaticModelData(morphPack, result.includeGenitals)
    const viewer = new PoseViewerCore(canvas, { skinMode })
    await viewer.init()
    viewer.loadData(buildModelData(result, staticData))
    return viewer
  })()
  return viewerPromise
}

/** 18 关节（0~1 坐标）+ 朝向（度）→ PNG dataURL */
export async function renderMannequin(joints, azimuthDeg = 0, opts = {}) {
  const { width = 768, height = 768, skinMode = 'flat_color' } = opts
  const viewer = await getViewer(skinMode)
  const j = (i) => ({ x: joints[i].x * 512, y: joints[i].y * 512, c: 1 })
  const parsedJoints = {}
  COCO_NAMES.forEach((name, i) => { parsedJoints[name] = j(i) })
  const pose = convertOpenPoseToPose({ joints: parsedJoints, canvasWidth: 512, canvasHeight: 512 }, viewer)
  if (pose) viewer.setPose(pose, true)
  return viewer.capture(width, height, 1.0, [255, 255, 255], 0, 0, azimuthDeg, 0)
}

// ---- postMessage 集成（iframe）----
window.addEventListener('message', async (e) => {
  const msg = e.data
  if (!msg || msg.type !== 'render') return
  try {
    const dataURL = await renderMannequin(msg.joints, msg.azimuth ?? 0, msg.opts || {})
    e.source?.postMessage({ type: 'result', dataURL }, '*')
  } catch (err) {
    e.source?.postMessage({ type: 'error', message: String(err?.message || err) }, '*')
  }
})

// ---- 自检模式：?test=1 ----
if (new URLSearchParams(location.search).get('test') === '1') {
  ;(async () => {
    try {
      status.textContent = '加载中…'
      const viewer = await getViewer('naked')
      for (const az of [0, 90, 180, 270]) {
        const url = await renderMannequin(STAND, az, { skinMode: 'naked' })
        const img = document.createElement('img')
        img.src = url
        img.title = `azimuth ${az}°`
        out.appendChild(img)
      }
      status.textContent = 'done'
    } catch (e) {
      status.textContent = 'ERROR: ' + (e?.stack || e?.message || e)
    }
  })()
}
