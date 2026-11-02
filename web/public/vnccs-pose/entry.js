/**
 * VNCCS PoseStudio —— 3D 人偶编辑器（官方引擎 + 最小宿主外壳）
 *
 * 引擎全是官方的（web/public/vnccs-pose/pose_studio/core.js 的 PoseViewerCore）：
 * 网格、53 根骨、IK、关节标记（每根骨一个可拖的球体）、相机轨道、渲染。
 * **这里只写外壳**：一块画布、一套 postMessage 协议、两个模式。
 *
 * 两种用法：
 *  - 独立页（studio.html）：自带状态栏 / 朝向下拉 / 导出按钮，直接开页也能用；
 *  - 嵌入模式（studio.html?embed=1）：只留一块填满 iframe 的画布，
 *    朝向与导出由宿主页（节点卡片）提供 —— 这是产线上的用法。
 *
 * 交互与官方 UniCanvas 的 pose layer 一致：
 * - 直接在画布上拖橙色关节点摆姿势（官方 raycast + IK，肘/膝的弯向有极点约束）
 * - 右键拖 = 轨道旋转视角，中键平移，滚轮缩放（官方 OrbitControls）
 * - 视角只是「检查视图」，不影响输出：导出用的是**朝向下拉**定的机位
 *   （官方同理：capture framing 与导航相机分离）
 * - 「导出人偶图」输出白底灰人偶 PNG —— VNCCS 的姿态 LoRA 就是在这个格式上训的
 *
 * 协议：
 *   父 → 子：
 *     { type:'init', pose?, joints?, yaw? }  初始化（pose = 官方 getPose() 的 JSON）
 *     { type:'setPose', pose }               套用官方姿势 JSON
 *     { type:'joints', points }              2D 关节点（0~1）→ 官方 convertOpenPoseToPose
 *     { type:'setYaw', yaw }                 改导出机位（度）
 *     { type:'export', yaw?, size? }         导出人偶图 → 回 result
 *     { type:'preview', yaw?, size? }        只要一张小缩略图（存姿势库用）→ 回 preview
 *     { type:'getPose' }                     回 { type:'pose', pose, yaw }
 *     { type:'resetPose' } / { type:'resetView' }
 *     { type:'animLoad', buffer, name }     动作文件（glb/gltf/fbx 的 ArrayBuffer）
 *                                           → 回 { type:'animLoaded', clips:[{name,duration}],
 *                                                  boneCount, format, rig:{family,familyLabel,mapped,missing} }
 *     { type:'animClip', index }            切动画轨道（一个文件里可能有多条）
 *     { type:'animTime', t }                采样 t 秒并套到人偶上 → 回 pose
 *     { type:'animSample', count }          等分抽 count 帧（入库用）
 *                                           → 回 { type:'animFrames', frames:[{t,pose,preview}], clip, yaw }
 *   子 → 父：
 *     { type:'ready' } | { type:'progress', text }
 *     | { type:'result', dataURL } | { type:'pose', pose, yaw }
 *     | { type:'animLoaded', clips, boneCount, format, rig }
 *     | { type:'animFrames', frames, clip, yaw }
 *     | { type:'error', message }
 *
 * **为什么动作文件走 ArrayBuffer 而不是路径**：父子是同源 iframe，但子窗拿不到
 * 用户 `<input type=file>` 选的 File 对象（跨文档不行）。宿主读成字节发过来，
 * 子窗自己造 Blob 解析。见 anim-import.js。
 */
import { PoseViewerCore } from './pose_studio/core.js'
import { loadMorphPack, solveMorph, buildStaticModelData } from './pose_studio/morph_runtime.js'
import { convertOpenPoseToPose } from './pose_studio/imports/openpose.js'

const EMBED = new URLSearchParams(location.search).get('embed') === '1'

const statusEl = document.getElementById('status')
const outEl = document.getElementById('out')
const canvas = document.getElementById('stage')
const stageWrap = document.getElementById('stagewrap')

/** COCO-18 顺序，与后端 /pose/extract 返回的 points 完全一致 */
const COCO_NAMES = [
  'nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist',
  'l_shoulder', 'l_elbow', 'l_wrist', 'r_hip', 'r_knee',
  'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye',
  'l_eye', 'r_ear', 'l_ear',
]

/**
 * 站立参考（0~1）。DWPose 漏检的关节用它补 —— 官方转换器里算角度要
 * 成对的关键点，缺一个会让整条肢体摆到奇怪的方向。
 */
const STAND = [
  [0.50, 0.075], [0.50, 0.155], [0.42, 0.180], [0.39, 0.339], [0.38, 0.497],
  [0.58, 0.180], [0.61, 0.339], [0.62, 0.497], [0.46, 0.520], [0.45, 0.720],
  [0.44, 0.930], [0.54, 0.520], [0.55, 0.720], [0.56, 0.930], [0.47, 0.069],
  [0.53, 0.069], [0.45, 0.075], [0.55, 0.075],
]

let viewerPromise = null
let viewer = null
let currentYaw = 0

/* ------------------------------------------------------------------ *
 * 画布尺寸
 *
 * 嵌入模式下画布跟着 iframe 走（dam 卡片里 280px、放大后 ~900px），
 * 所以要看容器，不能写死 768。
 *
 * 官方 resize() 只会分配一次 WebGL 缓冲（_viewportBufferLocked），
 * 放大到全屏时画面会糊 —— 这里在调用前临时解锁，让它按当前尺寸重分配。
 * ------------------------------------------------------------------ */
function measure() {
  if (!EMBED || !stageWrap) return { w: 768, h: 768 }
  const r = stageWrap.getBoundingClientRect()
  if (!(r.width > 0 && r.height > 0)) return { w: 768, h: 768 }
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  return {
    w: Math.max(64, Math.round(r.width * dpr)),
    h: Math.max(64, Math.round(r.height * dpr)),
  }
}

/** 把画布的内部缓冲调到容器尺寸。返回是否变化。 */
function applyCanvasSize() {
  const { w, h } = measure()
  if (canvas.width === w && canvas.height === h) return false
  canvas.width = w
  canvas.height = h
  return true
}

let sizeRaf = 0
function syncViewerSize() {
  if (!viewer) return
  const { w, h } = measure()
  if (viewer.width === w && viewer.height === h) return
  canvas.width = w
  canvas.height = h
  viewer._viewportBufferLocked = false
  viewer.resize(w, h)
}

if (stageWrap && typeof ResizeObserver !== 'undefined') {
  const ro = new ResizeObserver(() => {
    // 宿主把编辑器从卡片放大到全屏时会连续触发，一帧一次够了
    if (sizeRaf) return
    sizeRaf = requestAnimationFrame(() => {
      sizeRaf = 0
      if (applyCanvasSize()) syncViewerSize()
    })
  })
  ro.observe(stageWrap)
}

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
    applyCanvasSize()
    const morphPack = await loadMorphPack()
    const result = solveMorph(morphPack, {})
    const staticData = buildStaticModelData(morphPack, result.includeGenitals)
    const v = new PoseViewerCore(canvas, {
      skinMode: 'flat_color',
      // 编辑时要看得见骨架线与关节点（导出时官方 capture 会自动藏起来）
      showSkeletonHelper: true,
      ikEnabled: true,
      // 'end' = 拖动过程中不打扰宿主，松手才算一次变更（宿主据此存档）
      syncMode: 'end',
      onPoseChange: (pose) => post({ type: 'pose', pose, yaw: currentYaw }),
    })
    await v.init()
    // **方向天穹（彩虹网格球）在嵌入模式下一律关掉**，理由有两条：
    //  · 编辑时：节点卡片只有 280px，人身后糊一层彩虹网格纯属干扰；
    //  · 导出时：官方 `capture()` 会隐藏 gridHelper / skeletonHelper / 关节标记，
    //    但**没有**隐藏它 —— 不关就是每张条件图背景都带一层彩色网格
    //    （实测占画面 4%，直接把白底灰人偶毁了）。导出那条路再兜一次底。
    if (EMBED) v.setDirectionalSkydomeVisible?.(false)
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

/** 官方姿势 JSON → 人偶。`{}` 会把所有骨归零（= 初始站姿）。 */
function applyPose(v, pose) {
  if (!pose || typeof pose !== 'object') return
  v.setPose(pose, true)
  v.updateMarkers?.()
  v.jointMarkers?.forEach((m) => { m.visible = true })
  v.requestRender()
}

/**
 * 2D 关节点（0~1，后端 DWPose 提的）→ 官方骨骼欧拉角。
 *
 * 走的是官方的 convertOpenPoseToPose：它会先用两骨 IK 解四肢，再按
 * 父子段配躯干（含自动补 mid_hip），不是把 2D 点直接投影到骨头上。
 * 漏检的点用站立参考补，避免某条肢体角度算歪。
 */
function applyPoints(v, points, opts = {}) {
  const joints = {}
  COCO_NAMES.forEach((name, i) => {
    const p = Array.isArray(points?.[i]) ? points[i] : null
    const src = p || STAND[i]
    joints[name] = { x: src[0] * 512, y: src[1] * 512, c: p ? 1 : 0.2 }
  })
  const pose = convertOpenPoseToPose(
    { joints, canvasWidth: 512, canvasHeight: 512 }, v)
  if (pose) {
    // **3D 转正后重投影的点，必须把官方反推的 modelRotation 归零。**
    // 转换器会从 2D 点反推两件事：髋线角度 → 骨盆倾斜(Z)，肩宽比 → 躯干 Y 旋转。
    // 侧身照片的两髋投影几乎重合（实测只隔身高 1.5%），它据此判出 81° 骨盆
    // 倾斜，整个人偶绕 Z 转倒 —— 用户看到的是"躺着的麻花"。
    // 后端已经用 MediaPipe 的真实 3D 把身体转到正面再投影，朝向信息是准的，
    // 这里再让它按 2D 猜一次只会帮倒忙。实测归零后人偶才立得住。
    if (opts.reprojected) {
      pose.modelRotation = [0, 0, 0]
      pose.cameraYRotation = 0
    }
    v.setPose(pose, true)
    v.updateMarkers?.()
    v.jointMarkers?.forEach((m) => { m.visible = true })
    v.requestRender()
  }
  return !!pose
}

/**
 * 出图（导出 / 缩略图都走这里）。
 *
 * **导出以编辑器当前视角为准（所见即所得）**：走官方 `capture` 的
 * `viewport: true`，用**导航相机**（`this.camera`）而不是 `captureCamera`
 * 渲染。这样用户在画布里右键转到什么角度，导出的就是什么角度，不被
 * 「朝向」选项锁死。朝向下拉只是**快捷对齐**（点一下把相机摆到标准角度），
 * 摆完还能继续右键微调。
 *
 * **天穹必须在 capture 期间藏起来**：官方只会隐藏 gridHelper /
 * skeletonHelper / 关节标记，彩虹网格球不在那份清单里。capture 是同步的
 * （`canvas.toDataURL`），所以 try/finally 兜得住。
 */
function renderOut(size) {
  if (!viewer) return null
  const n = Math.max(256, Math.min(2048, Math.round(size) || 768))
  const prevSky = viewer.directionalSkydomeVisible
  const cam = viewer.camera
  const prevAspect = cam ? cam.aspect : null
  try {
    viewer.setDirectionalSkydomeVisible?.(false)
    // 导出是正方形，把导航相机 aspect 临时对齐到 1:1，避免编辑器非方形时拉伸
    if (cam) {
      cam.aspect = 1
      cam.updateProjectionMatrix?.()
    }
    // viewport:true —— 用当前导航相机（所见即所得），隐藏骨架线/关节标记、白底
    return viewer.capture(n, n, 1.0, [255, 255, 255], 0, 0, 0, 0, { viewport: true })
  } finally {
    if (cam && prevAspect != null) {
      cam.aspect = prevAspect
      cam.updateProjectionMatrix?.()
    }
    viewer.setDirectionalSkydomeVisible?.(prevSky)
  }
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

function post(msg) { window.parent?.postMessage(msg, '*') }

function setStatus(text, isError = false) {
  if (statusEl) {
    statusEl.textContent = text
    statusEl.style.color = isError ? '#ff8a8a' : ''
  }
}

function setYaw(deg) {
  currentYaw = ((Math.round(Number(deg) || 0) % 360) + 360) % 360
  // 「朝向」是**快捷对齐**：把导航相机摆到标准角度，摆完用户还能右键继续微调。
  // 导出始终以导航相机为准（renderOut 用 viewport:true），所以这里只是摆相机。
  if (viewer) snapCamera(viewer)
}

/**
 * 把导航相机（视口相机 + OrbitControls）对齐到当前选中的标准角度（快捷对齐）。
 *
 * 走官方 `snapToCaptureCamera`：它把 capture 相机的 position / orbit.target
 * 拷到视口相机，但**不复制 fov/aspect**（capture frame 会以盒子的形式显示在
 * 3D 空间里）。点「朝向」下拉就是摆一下相机，摆完用户还能右键继续自由转。
 * 导出始终用导航相机（renderOut 的 viewport:true），不依赖这个值。
 */
function snapCamera(v) {
  if (!v || !v.snapToCaptureCamera) return
  const { w, h } = measure()
  try {
    v.snapToCaptureCamera(w, h, 1.0, 0, 0, currentYaw, 0)
  } catch (e) {
    // 某些初始帧 captureCamera 还没就绪，忽略这一次
  }
  v.requestRender?.()
}

function resetCamera() {
  if (!viewer) return
  frameView(viewer)
  viewer.requestRender()
}

// ---- 这里**不再有**朝向下拉和工具栏按钮 ----
// 它们只在宿主（PoseStudio.tsx）那一份：朝向、机位滑块、重置视角/姿势、
// 从图片提姿势、导出人偶图。iframe 这一侧只认 postMessage：setYaw / resetView /
// resetPose / export / getPose。以前这里也有一套，用户在卡片里看到的就是
// "相同的东西里面又外面都有"。

// ---- 父页消息 ----
window.addEventListener('message', async (e) => {
  const msg = e.data
  if (!msg || typeof msg.type !== 'string') return
  try {
    if (msg.type === 'init' || msg.type === 'render') {
      const v = await getViewer()
      viewer = v
      syncViewerSize()
      if (typeof msg.yaw === 'number') setYaw(msg.yaw)
      if (msg.pose) applyPose(v, msg.pose)
      else if (Array.isArray(msg.joints) && msg.joints.length >= 18) {
        applyPoints(v, msg.joints.map((p) => [p.x, p.y]), { reprojected: !!msg.reprojected })
      }
      post({ type: 'ready', yaw: currentYaw })
      return
    }
    // 下面这些都要 viewer：还没就绪就排队等，别丢消息
    const v = await getViewer()
    viewer = v
    if (msg.type === 'setPose') {
      applyPose(v, msg.pose)
    } else if (msg.type === 'joints') {
      applyPoints(v, msg.points, { reprojected: !!msg.reprojected })
    } else if (msg.type === 'setYaw') {
      setYaw(msg.yaw)
      post({ type: 'pose', pose: v.getPose(), yaw: currentYaw })
    } else if (msg.type === 'getPose') {
      post({ type: 'pose', pose: v.getPose(), yaw: currentYaw })
    } else if (msg.type === 'resetPose') {
      applyPose(v, { bones: {}, bonePositions: {} })
      post({ type: 'pose', pose: v.getPose(), yaw: currentYaw })
    } else if (msg.type === 'resetView') {
      resetCamera()
    } else if (msg.type === 'snap') {
      snapCamera(v)
    } else if (msg.type === 'preview') {
      // 存库时要一张小缩略图，浏览面板得靠它认人。**走和导出同一条通路**
      // （同一个 viewport capture），只是尺寸小、不算正式产物，所以单独回一个类型。
      const url = renderOut(msg.size || 256)
      if (url) post({ type: 'preview', dataURL: url })
    } else if (msg.type === 'animLoad') {
      // 动作文件：宿主读好字节传进来（子窗拿不到 File 对象）
      try {
        const m = await import('./anim-import.js')
        const info = await m.loadAnimation(msg.buffer, msg.name, v)
        post({ type: 'animLoaded', ...info })
      } catch (e) {
        post({ type: 'error', message: String(e?.message || e) })
      }
    } else if (msg.type === 'animClip') {
      // 切轨道后立刻采第 0 帧，宿主只管更新下拉，不用再等一轮消息
      const m = await import('./anim-import.js')
      if (m.selectClip(msg.index || 0)) {
        const ok = await m.applyTime(0, v)
        if (ok) post({ type: 'pose', pose: v.getPose(), yaw: currentYaw })
      }
    } else if (msg.type === 'animTime') {
      const m = await import('./anim-import.js')
      const ok = await m.applyTime(msg.t, v)
      if (ok) {
        // 每帧都回一次 pose：宿主拿它当存档点（和拖动松手的语义一致）
        post({ type: 'pose', pose: v.getPose(), yaw: currentYaw })
      } else {
        post({ type: 'error', message: m.errorState() || '这一帧套不到人偶上' })
      }
    } else if (msg.type === 'animSample') {
      // 批量抽帧（入库用）：一条动画等分 N 帧，逐帧套到人偶上、连缩略图一起回。
      // 缩略图走**和导出同一条通路**（renderOut → viewport capture），
      // 所以存进姿势库的预览和「导出人偶图」长得一样，只是小一号。
      try {
        const m = await import('./anim-import.js')
        const st = m.animationState()
        if (!st) throw new Error('还没载入动作')
        const count = Math.max(1, Math.min(48, Math.round(Number(msg.count) || 8)))
        // 首尾都是端点帧（站立/收势），等分时避开两头一点点，取中段更"有动作"
        const dur = Math.max(0.001, st.duration)
        const frames = []
        for (let i = 0; i < count; i++) {
          const t = count === 1 ? dur * 0.5 : (dur * (i + 0.5)) / count
          const ok = await m.applyTime(t, v)
          if (!ok) throw new Error(m.errorState() || `第 ${i + 1} 帧套不上人偶`)
          const preview = renderOut(192) || ''
          frames.push({ t: Number(t.toFixed(3)), pose: v.getPose(), preview })
        }
        // 抽完把时间轴拉回第 0 帧，别让编辑器停在最后一帧
        await m.applyTime(0, v)
        post({ type: 'animFrames', frames, clip: st.name, yaw: currentYaw })
        post({ type: 'pose', pose: v.getPose(), yaw: currentYaw })
      } catch (e) {
        post({ type: 'error', message: String(e?.message || e) })
      }
    } else if (msg.type === 'export') {
      // 导出**当前编辑器视角**（所见即所得）：不做任何对齐，直接 viewport capture。
      const url = renderOut(msg.size)
      if (url) post({ type: 'result', dataURL: url, yaw: currentYaw })
      else post({ type: 'error', message: 'capture 没出图（渲染器还没就绪？）' })
    }
  } catch (err) {
    post({ type: 'error', message: String(err?.message || err) })
  }
})

// ---- 自举：直接打开这个页面时也可用 ----
;(async () => {
  try {
    applyCanvasSize()
    setStatus('正在加载人偶模型（37MB，首次约十几秒）…')
    post({ type: 'progress', text: '正在加载人偶模型（37MB，首次约十几秒）…' })
    viewer = await getViewer()
    syncViewerSize()
    setStatus(EMBED ? '' : '就绪：拖橙色关节点摆姿势，右键拖动转视角')
    post({ type: 'ready', yaw: currentYaw })
    if (outEl) outEl.style.display = 'none'
  } catch (e) {
    setStatus('加载失败：' + (e?.message || e), true)
    post({ type: 'error', message: String(e?.message || e) })
  }
})()

// ---- 自检：?selftest=1 ----
// 走**和父页完全一样的通路**：ready → 收到 init → 应用关节 → 导出 → 收到 result。
// 用来在真实浏览器里验证 iframe 侧整条链（网格加载 → 骨骼 → 渲染 → 回 PNG）。
if (new URLSearchParams(location.search).get('selftest') === '1') {
  window.addEventListener('message', (e) => {
    const d = e.data
    if (d?.type !== 'result') return
    const img = document.createElement('img')
    img.src = d.dataURL
    img.style.width = '256px'
    if (outEl) { outEl.style.display = 'flex'; outEl.appendChild(img) }
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
    window.postMessage({ type: 'init', joints: STAND.map(([x, y]) => ({ x, y })),
      yaw: 90 }, '*')
    await new Promise((r) => setTimeout(r, 400))
    // 模拟宿主索要导出图（**走和宿主按钮同一条消息**）。
    // 以前这里是 `#btn-export.click()`，但那个按钮已经删掉了——工具条只在宿主一份。
    window.postMessage({ type: 'export', size: 768 }, '*')
  })()
}
