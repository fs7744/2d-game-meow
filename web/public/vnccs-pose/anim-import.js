/**
 * 动作文件（glTF / FBX 动画）→ 3D 人偶姿势。
 *
 * **为什么这条路比"从图片提姿势"可靠**：图片反推的 3D 是估出来的（有深度歧义，
 * 侧身照片常被判成 80° 骨盆倾斜 → 人偶躺倒）；动画里 3D 是**自带**的，
 * 骨长、朝向、遮挡全都真实，套上去就是自然人形。
 *
 * 骨骼映射不用我们写：官方 `pose_studio/imports/mixamo.js` 里有 65 根
 * Mixamo 骨 → MakeHuman 骨的对照表，并且 export 了 `buildMixamoWorldKeypoints`。
 * 这个模块只做三件事：**加载 → 按时间采样 → 交给 viewer**。
 *
 * 为什么不用官方的 `importMixamoFBXAnimation`：那是"一次性抽 N 帧返回一整组
 * pose"，适合批量导入；我们要的是"拖滑块实时改时间点"，必须自己持有 mixer。
 *
 * 支持的格式：`.glb` / `.gltf`（GLTFLoader）、`.fbx`（FBXLoader）。
 * **注意**：解析器是从 esm.sh 动态 import 的（vendor 里只有 three 本体，
 * 没有 loader），所以第一次导入需要联网；断网会给出明确报错而不是静默失败。
 */
const THREE_VER = '0.160.0'
const THREE_URL = `https://esm.sh/three@${THREE_VER}`
const FBX_LOADER_URL = `${THREE_URL}/examples/jsm/loaders/FBXLoader.js`
const GLTF_LOADER_URL = `${THREE_URL}/examples/jsm/loaders/GLTFLoader.js`

/** { THREE, buildMixamoWorldKeypoints } —— 只加载一次 */
let deps = null
let depsPromise = null

async function loadDeps() {
  if (deps) return deps
  if (!depsPromise) {
    depsPromise = (async () => {
      const THREE = await import(THREE_URL)
      const { buildMixamoWorldKeypoints } = await import('./pose_studio/imports/mixamo.js')
      deps = { THREE, buildMixamoWorldKeypoints }
      return deps
    })().catch((e) => {
      depsPromise = null
      throw new Error('加载动画解析器失败（需要联网）：' + (e?.message || e))
    })
  }
  return depsPromise
}

/** 当前动作状态：{ root, clips, clip, mixer, sourceBones, time } */
let state = null

/** 把 root 下所有骨骼按"原名 / 去掉 mixamorig: 前缀 / 归一化名"三种方式登记 */
function collectBones(root) {
  const bones = {}
  const normalizedBones = {}
  const reg = (o) => {
    if (!o?.name) return
    bones[o.name] = o
    const short = o.name.includes(':') ? o.name.split(':').pop() : o.name
    if (!bones[short]) bones[short] = o
    const norm = short.replace(/[^a-z0-9]/gi, '').toLowerCase()
    if (norm && !normalizedBones[norm]) normalizedBones[norm] = o
  }
  root.traverse((o) => {
    if (o?.isBone) reg(o)
    if (o?.isSkinnedMesh && Array.isArray(o.skeleton?.bones)) o.skeleton.bones.forEach(reg)
  })
  return { bones, normalizedBones }
}

/**
 * 读一个动作文件。
 *
 * `buffer` 是宿主读好的 ArrayBuffer（iframe 里拿不到用户选的 File —— 跨文档
 * 只能传字节）。返回 `{ clips: [{name, duration}] }` 供宿主列出来选。
 */
export async function loadAnimation(buffer, fileName) {
  const { THREE } = await loadDeps()
  const url = URL.createObjectURL(new Blob([buffer]))
  const isFbx = /\.fbx$/i.test(fileName || '')
  try {
    let root
    let clips = []
    if (isFbx) {
      const m = await import(FBX_LOADER_URL)
      root = await new m.FBXLoader().loadAsync(url)
      clips = root.animations || []
    } else {
      const m = await import(GLTF_LOADER_URL)
      const gltf = await new m.GLTFLoader().loadAsync(url)
      root = gltf.scene
      clips = gltf.animations || []
    }
    if (!clips.length) {
      throw new Error('这个文件里没有动画轨道（可能是静态模型）')
    }
    const sourceBones = collectBones(root)
    if (!Object.keys(sourceBones.bones).length) {
      throw new Error('文件里没有骨骼（不是骨架动画）')
    }
    const clip = clips[0]
    const mixer = new THREE.AnimationMixer(root)
    mixer.clipAction(clip).play()
    state = { root, clips, clip, mixer, sourceBones, time: 0 }
    return {
      clips: clips.map((c) => ({ name: c.name || '(未命名)', duration: c.duration })),
      boneCount: Object.keys(sourceBones.bones).length,
      format: isFbx ? 'fbx' : 'gltf',
    }
  } finally {
    URL.revokeObjectURL(url)
  }
}

/** 切换动画轨道（一个文件里可能有多条，如 Soldier.glb 的 Idle/Walk/Run） */
export function selectClip(index) {
  if (!state) return null
  const clip = state.clips[index]
  if (!clip) return null
  state.mixer.stopAllAction()
  state.clip = clip
  state.mixer.clipAction(clip).play()
  state.time = 0
  return { name: clip.name || '(未命名)', duration: clip.duration }
}

/** 有没有加载过动作 */
export function animationState() {
  return state ? { name: state.clip.name, duration: state.clip.duration, time: state.time } : null
}

/**
 * 采样 `t` 秒处的姿势并套到人偶上。
 *
 * 采样前先 `resetPose()`：上一帧摆过的关节如果这一帧没被驱动，会停在旧位置，
 * 看起来像"动作有残留"。官方 FBX 导入也是每帧先 reset。
 */
export async function applyTime(t, viewer) {
  if (!state || !viewer) return false
  const { buildMixamoWorldKeypoints } = await loadDeps()
  const { root, mixer, clip } = state
  const time = Math.max(0, Math.min(clip.duration, Number(t) || 0))
  state.time = time

  mixer.setTime(time)
  root.updateMatrixWorld(true)

  viewer.resetPose()
  const kp = buildMixamoWorldKeypoints(state.sourceBones, viewer)
  if (!kp?.worldKps) return false
  return !!viewer.applyWorldKeypointImport(kp.worldKps, {
    includeSpine: false,
    normalizeLimbs: true,
    drawFigure: false,
    updateMarkers: true,
    dispatchPoseChange: false,
  })
}

/** 清掉当前动作（不删人偶姿势，只是不再按动画驱动） */
export function clearAnimation() {
  if (state?.mixer) state.mixer.stopAllAction()
  state = null
}
