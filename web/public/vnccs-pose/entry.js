/**
 * VNCCS PoseStudio —— 3D 人偶编辑器（官方引擎 + 最小宿主外壳）
 *
 * 引擎全是官方的（web/public/vnccs-pose/pose_studio/core.js 的 PoseViewerCore）：
 * 网格、53 根骨、IK、关节标记（每根骨一个可拖的球体）、相机轨道、渲染。
 * **这里只写外壳**：一块画布、一条工具栏、两个按钮、一套 postMessage 协议。
 *
 * 交互与官方 UniCanvas 的 pose layer 一致：
 * - 直接在画布上拖橙色关节点摆姿势（官方 raycast + IK，肘/膝的弯向有极点约束）
 * - 右键拖 = 轨道旋转视角，中键平移，滚轮缩放（官方 OrbitControls）
 * - 视角只是「检查视图」，不影响输出：导出用的是**朝向下拉**定的机位
 *   （官方同理：capture framing 与导航相机分离）
 * - 「导出人偶图」输出白底灰人偶 PNG —— VNCCS 的姿态 LoRA 就是在这个格式上训的
 *
 * 协议：
 *   父 → 子：{ type:'init', joints?, azimuth? }   设定初始姿态（可选）
 *   子 → 父：{ type:'ready' } | { type:'result', dataURL }
 *           | { type:'error', message } | { type:'progress', text }
 */
import { PoseViewerCore } from './pose_studio/core.js'
import { loadMorphPack, solveMorph, buildStaticModelData } from './pose_studio/morph_runtime.js'
import { convertOpenPoseToPose } from './pose_studio/imports/openpose.js'

const statusEl = document.getElementById('status')
const outEl = document.getElementById('out')
const canvas = document.getElementById('stage')
const yawSel = document.getElementById('yaw')

/** COCO-18 顺序，与骨骼编辑器的 18 个关节完全一致 */
const COCO_NAMES = [
  'nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist',
  'l_shoulder', 'l_elbow', 'l_wrist', 'r_hip', 'r_knee',
  'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye',
  'l_eye', 'r_ear', 'l_ear',
]

/** 朝向（相机机位绕 Y 的角度，度）。数值与 2D 骨架编辑器的方位角同号同义。 */
const YAW_LABELS = [
  [0, '正面'], [45, '右前 3/4'], [90, '右侧面'], [135, '右后 3/4'],
  [180, '背面'], [225, '左后 3/4'], [270, '左侧面'], [315, '左前 3/4'],
]

let viewerPromise = null
let viewer = null
let currentYaw = 0

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

async function getViewer() {
  if (viewerPromise) return viewerPromise
  viewerPromise = (async () => {
    const morphPack = await loadMorphPack()
    const result = solveMorph(morphPack, {})
    const staticData = buildStaticModelData(morphPack, result.includeGenitals)
    const v = new PoseViewerCore(canvas, {
      skinMode: 'flat_color',
      // 编辑时要看得见骨架线与关节点（导出时官方 capture 会自动藏起来）
      showSkeletonHelper: true,
      ikEnabled: true,
    })
    await v.init()
    v.loadData(buildModelData(result, staticData))
    // 关节标记是官方给的（每根骨一个球体），确保可见——拖它们就能摆姿势
    v.updateMarkers?.()
    v.jointMarkers?.forEach((m) => { m.visible = true })
    frameView(v)
    v.requestRender()
    return v
  })()
  return viewerPromise
}

/** 18 关节（编辑器 0~1 坐标）→ 官方骨骼欧拉角 */
function applyJoints(v, joints) {
  const j = (i) => ({ x: joints[i].x * 512, y: joints[i].y * 512, c: 1 })
  const parsedJoints = {}
  COCO_NAMES.forEach((name, i) => { parsedJoints[name] = j(i) })
  const pose = convertOpenPoseToPose({ joints: parsedJoints, canvasWidth: 512, canvasHeight: 512 }, v)
  if (pose) v.setPose(pose, true)
  v.requestRender()
}

/** 导出：官方 capture（自动隐藏骨架线/关节标记、白底、按 yawDeg 定机位） */
function exportMannequin() {
  if (!viewer) return null
  return viewer.capture(768, 768, 1.0, [255, 255, 255], 0, 0, currentYaw, 0)
}

/**
 * 把编辑相机摆到「人偶刚好填满画布」的位置。
 *
 * 官方 setupScene 的初始机位是 (0,10,30)，那是给 MakeHuman 原始尺度
 * （模型高约 14 单位）准备的通用视角，人偶只占画布一半多。编辑时要看得清
 * 关节，所以拉近到 meshCenter + 20 单位（fov 45° 下可见高度 ≈16.5 单位，
 * 人偶占八成）。
 *
 * 目标点用官方的 meshCenter（capture 的 framing 也用它），所以**编辑视图与
 * 导出视图的中心一致**——所见即所得。
 */
function frameView(v) {
  if (!v) return
  const center = v.meshCenter || new v.THREE.Vector3(0, 0, 0)
  v.camera.position.set(center.x, center.y, center.z + 20)
  v.orbit.target.copy(center)
  v.orbit.update()
}

function resetCamera() {
  if (!viewer) return
  frameView(viewer)
  viewer.requestRender()
}

function post(msg) { window.parent?.postMessage(msg, '*') }

function setStatus(text, isError = false) {
  if (statusEl) {
    statusEl.textContent = text
    statusEl.style.color = isError ? '#ff8a8a' : ''
  }
}

// ---- 朝向下拉 ----
if (yawSel) {
  for (const [deg, label] of YAW_LABELS) {
    const opt = document.createElement('option')
    opt.value = String(deg)
    opt.textContent = `${deg}° ${label}`
    yawSel.appendChild(opt)
  }
  yawSel.value = '0'
  yawSel.addEventListener('change', () => { currentYaw = Number(yawSel.value) || 0 })
}

// ---- 工具栏按钮 ----
document.getElementById('btn-export')?.addEventListener('click', () => {
  const url = exportMannequin()
  if (url) post({ type: 'result', dataURL: url })
})
document.getElementById('btn-reset')?.addEventListener('click', resetCamera)

// ---- 父页消息 ----
window.addEventListener('message', async (e) => {
  const msg = e.data
  if (!msg || typeof msg.type !== 'string') return
  try {
    if (msg.type === 'init' || msg.type === 'render') {
      const v = await getViewer()
      viewer = v
      if (Array.isArray(msg.joints) && msg.joints.length >= 18) {
        applyJoints(v, msg.joints)
      }
      if (typeof msg.azimuth === 'number' && yawSel) {
        currentYaw = msg.azimuth
        yawSel.value = String(currentYaw)
      }
      post({ type: 'ready' })
    }
  } catch (err) {
    post({ type: 'error', message: String(err?.message || err) })
  }
})

// ---- 自举：直接打开这个页面时也可用 ----
;(async () => {
  try {
    setStatus('正在加载人偶模型（37MB，首次约十几秒）…')
    post({ type: 'progress', text: '正在加载人偶模型（37MB，首次约十几秒）…' })
    viewer = await getViewer()
    setStatus('就绪：拖橙色关节点摆姿势，右键拖动转视角')
    post({ type: 'ready' })
    if (outEl) outEl.style.display = 'none'
  } catch (e) {
    setStatus('加载失败：' + (e?.message || e), true)
    post({ type: 'error', message: String(e?.message || e) })
  }
})()

// ---- 自检：?selftest=1 ----
// 走**和父页完全一样的通路**：ready → 收到 init → 应用关节 → 点导出 → 收到 result。
// 用来在真实浏览器里验证 iframe 侧整条链（网格加载 → 骨骼 → 渲染 → 回 PNG）。
if (new URLSearchParams(location.search).get('selftest') === '1') {
  window.addEventListener('message', (e) => {
    const d = e.data
    if (d?.type !== 'result') return
    const img = document.createElement('img')
    img.src = d.dataURL
    img.style.width = '256px'
    if (outEl) outEl.appendChild(img)
    setStatus(`selftest: OK dataURL ${String(d.dataURL).length} 字节`)
  })
  ;(async () => {
    // 等 viewer 就绪后，自己给自己发一条 init（模拟父页）
    for (let i = 0; i < 600; i++) {
      if (viewer) break
      await new Promise((r) => setTimeout(r, 100))
    }
    if (!viewer) {
      setStatus('selftest ERROR: viewer 未就绪', true)
      return
    }
    const STAND = [
      { x: 0.5, y: 0.075 }, { x: 0.5, y: 0.155 }, { x: 0.42, y: 0.18 }, { x: 0.39, y: 0.339 }, { x: 0.38, y: 0.497 },
      { x: 0.58, y: 0.18 }, { x: 0.61, y: 0.339 }, { x: 0.62, y: 0.497 }, { x: 0.46, y: 0.52 }, { x: 0.45, y: 0.72 },
      { x: 0.44, y: 0.93 }, { x: 0.54, y: 0.52 }, { x: 0.55, y: 0.72 }, { x: 0.56, y: 0.93 }, { x: 0.47, y: 0.069 },
      { x: 0.53, y: 0.069 }, { x: 0.45, y: 0.075 }, { x: 0.55, y: 0.075 },
    ]
    window.postMessage({ type: 'init', joints: STAND, azimuth: 90 }, '*')
    await new Promise((r) => setTimeout(r, 400))
    // 模拟用户点「导出人偶图」
    document.getElementById('btn-export')?.click()
  })()
}
