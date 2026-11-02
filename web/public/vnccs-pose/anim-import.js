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
 *
 * **解析器在本地**（`vendor/three/jsm/`，r160，和 vendor 的 three 同版本）——
 * 断网也能导入。只有本地那份加载失败时才回落 esm.sh（老行为），
 * 回落时会明确告诉用户「用的是联网解析器」。
 *
 * **非 Mixamo 骨架**（Quaternius / KayKit / Unreal / VRM 命名）走
 * `anim-skeleton.js` 把真实骨头挂到 Mixamo 名下，官方代码一行不改。
 * 认不出来时抛明确错误（缺哪几根 + 文件里的骨名样本），不再"点了没反应"。
 */
import {
  readSkeleton, resolveRig, buildSourceBones, describeRig,
} from './anim-skeleton.js'

const THREE_VER = '0.160.0'
const CDN_BASE = `https://esm.sh/three@${THREE_VER}`
const LOCAL = {
  gltf: new URL('./vendor/three/jsm/loaders/GLTFLoader.js', import.meta.url).href,
  fbx: new URL('./vendor/three/jsm/loaders/FBXLoader.js', import.meta.url).href,
}
const CDN = {
  gltf: `${CDN_BASE}/examples/jsm/loaders/GLTFLoader.js`,
  fbx: `${CDN_BASE}/examples/jsm/loaders/FBXLoader.js`,
}

/** 解析器只加载一次（本地优先）。`{ Loader, THREE, source }` */
const loaderCache = new Map()

async function getLoader(kind) {
  const hit = loaderCache.get(kind)
  if (hit) return hit
  let out
  try {
    const m = await import(LOCAL[kind])
    out = {
      Loader: kind === 'fbx' ? m.FBXLoader : m.GLTFLoader,
      // 本地 loader 里的 `from 'three'` 由 studio.html 的 importmap 指到 vendor 的
      // three.module.js —— 与 viewer 同一份实例，所以这里不用另取 THREE
      THREE: null,
      source: 'local',
    }
  } catch (localErr) {
    try {
      const m = await import(CDN[kind])
      const THREE = await import(CDN_BASE)
      out = {
        Loader: kind === 'fbx' ? m.FBXLoader : m.GLTFLoader,
        THREE,
        source: 'cdn',
      }
    } catch (cdnErr) {
      throw new Error('动画解析器加载失败（本地副本读不到，联网回落也失败）：'
        + (cdnErr?.message || cdnErr))
    }
  }
  loaderCache.set(kind, out)
  return out
}

/** 当前动作状态 */
let state = null
/** 最近一次失败的原因（宿主要拿到具体文案，不能只说"没反应"） */
let lastError = ''

/**
 * 读一个动作文件。
 *
 * `buffer` 是宿主读好的 ArrayBuffer（iframe 里拿不到用户选的 File —— 跨文档
 * 只能传字节）。`viewer` 只用来借它的 THREE 实例（保证与 loader 同一份）。
 */
export async function loadAnimation(buffer, fileName, viewer) {
  const isFbx = /\.fbx$/i.test(fileName || '')
  const kind = isFbx ? 'fbx' : 'gltf'
  const { Loader, THREE: loaderTHREE, source } = await getLoader(kind)
  const THREE = loaderTHREE || viewer?.THREE
  if (!THREE) throw new Error('渲染器还没就绪，读不了动画')

  const url = URL.createObjectURL(new Blob([buffer]))
  try {
    const loader = new Loader()
    const loaded = await loader.loadAsync(url)
    // FBXLoader 直接返回 Object3D；GLTFLoader 返回 { scene, animations }
    const root = isFbx ? loaded : loaded.scene
    const clips = (isFbx ? loaded.animations : loaded.animations) || []
    if (!clips.length) {
      throw new Error('这个文件里没有动画轨道（可能是静态模型，或者动作没烘进文件）')
    }

    // ---- 骨骼：解析成官方认的 17 个锚点 ----
    const rig = readSkeleton(root)
    const map = resolveRig(rig)
    if (map.missing.length) {
      throw new Error(describeRig(map))
    }
    const sourceBones = buildSourceBones(rig, map)

    const clip = clips[0]
    const mixer = new THREE.AnimationMixer(root)
    mixer.clipAction(clip).play()
    state = { root, clips, clip, mixer, sourceBones, rig, map, time: 0 }
    lastError = ''
    return {
      clips: clips.map((c) => ({ name: c.name || '(未命名)', duration: c.duration })),
      boneCount: rig.nodes.filter((n) => n.obj?.isBone).length,
      format: source === 'cdn' ? `${kind}（联网解析器）` : kind,
      rig: {
        family: map.family,
        familyLabel: map.familyLabel,
        mapped: Object.keys(map.alias).length,
        missing: map.missing,
        // 每个锚点对到文件里的哪根骨头 —— 界面不显示，但用户报"套歪了"时
        // 一眼就能看出是映射错了还是 IK 错了
        anchors: Object.fromEntries(
          Object.entries(map.alias).map(([k, i]) => [k, rig.nodes[i]?.name || '']),
        ),
      },
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

/** 当前动作的信息 */
export function animationState() {
  if (!state) return null
  return {
    name: state.clip.name,
    duration: state.clip.duration,
    time: state.time,
    rig: state.map ? { family: state.map.family, label: state.map.familyLabel } : null,
  }
}

/** 最近一次失败原因（空串 = 没失败） */
export function errorState() {
  return lastError
}

/**
 * 采样 `t` 秒处的姿势并套到人偶上。
 *
 * 采样前先 `resetPose()`：上一帧摆过的关节如果这一帧没被驱动，会停在旧位置，
 * 看起来像"动作有残留"。官方 FBX 导入也是每帧先 reset。
 */
export async function applyTime(t, viewer) {
  try {
    return await applyTimeInner(t, viewer)
  } catch (e) {
    lastError = String(e?.message || e)
    throw e
  }
}

async function applyTimeInner(t, viewer) {
  if (!state || !viewer) return false
  const { buildMixamoWorldKeypoints } = await import('./pose_studio/imports/mixamo.js')
  const { root, mixer, clip } = state
  const time = Math.max(0, Math.min(clip.duration, Number(t) || 0))
  state.time = time

  mixer.setTime(time)
  root.updateMatrixWorld(true)

  viewer.resetPose()
  const kp = buildMixamoWorldKeypoints(state.sourceBones, viewer)
  if (!kp?.worldKps) {
    lastError = '这一帧的关键点算不出来：骨架里缺骨盆或肩，套不到人偶上'
    return false
  }
  const ok = !!viewer.applyWorldKeypointImport(kp.worldKps, {
    includeSpine: false,
    normalizeLimbs: true,
    drawFigure: false,
    updateMarkers: true,
    dispatchPoseChange: false,
  })
  lastError = ok ? '' : '这一帧套不到人偶上（骨骼映射对不上 MakeHuman 骨架）'
  return ok
}

/** 清掉当前动作（不删人偶姿势，只是不再按动画驱动） */
export function clearAnimation() {
  if (state?.mixer) state.mixer.stopAllAction()
  state = null
  lastError = ''
}
