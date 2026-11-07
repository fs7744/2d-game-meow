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
import { withPoseRollback } from './pose_guard.js'

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

/**
 * 脚底参考高度：脚掌/脚骨世界 y 的最小值（y 向上，越小越靠地）。
 *
 * 用骨头而不是网格包围盒：`Box3.setFromObject` 对 SkinnedMesh 用的是**未蒙皮**
 * 的几何包围盒，摆完姿势后完全不准。
 */
function footY(viewer) {
  const THREE = viewer?.THREE
  const bones = viewer?.bones
  if (!THREE || !bones) return null
  const v = new THREE.Vector3()
  let min = Infinity
  for (const n of ['ball_l', 'ball_r', 'foot_l', 'foot_r', 'toe_l', 'toe_r']) {
    const b = bones[n]
    if (!b) continue
    b.getWorldPosition(v)
    if (v.y < min) min = v.y
  }
  return Number.isFinite(min) ? min : null
}

/**
 * 人偶的根节点（scene 的直接子节点）。
 *
 * 必须平移**它**而不是 pelvis：pelvis 只是根骨，单独挪会让蒙皮网格和骨架错位
 * （网格挂在 Root 下面）。实测骨盆的父链是 `pelvis → Root → scene`。
 */
function characterRoot(viewer) {
  let o = viewer?.bones?.pelvis
  if (!o) return null
  while (o.parent && o.parent !== viewer.scene) o = o.parent
  return o.parent === viewer.scene ? o : null
}
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

    // loadAnimation 全程不动 viewer 的骨骼（只读 characterRoot 取根节点 y），
    // 所以失败时**保留上一次成功载入的动作 state**，别清空用户当前的工作。
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
    state = {
      root, clips, clip, mixer, sourceBones, rig, map, time: 0,
      // 根节点初始 y：脚下对齐时以它为准重算，避免一次次 += 漂走
      rootY: characterRoot(viewer)?.position.y ?? 0,
      // 地面高度：**只量一次**（见 applyTimeInner 里的注释）
      groundY: null,
    }
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
  // **失败别把用户摆好的姿势清掉**：applyTimeInner 一开始就 resetPose()，
  // 后面任何一步失败（关键点算不出 / 骨架映射不上 / 抛异常）都要还原。
  try {
    return await withPoseRollback(viewer, () => applyTimeInner(t, viewer))
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
  // **脚下对齐的基准只能量一次**。
  // resetPose 只重置骨骼旋转，**不动根节点的 position** —— 如果每帧都量，
  // 上一帧已经平移过的根节点会让"地面"跟着漂，越走越低/越高。
  const rootNode = characterRoot(viewer)
  if (rootNode) rootNode.position.y = state.rootY
  viewer.scene?.updateMatrixWorld?.(true)
  if (state.groundY == null) state.groundY = footY(viewer)
  const groundY = state.groundY

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
  if (ok) groundCharacter(viewer, groundY)
  return ok
}

/**
 * 把这一帧的脚底对齐到地面高度。
 *
 * **为什么必须做**：导入的关键点里骨盆随动作上下起伏（走路的 bob、换支撑脚），
 * 而相机是固定的 —— 于是**每帧脚底在画面里的高度都不一样**，抽出来的一批帧
 * 拼成动画就"一下高一下低"。实测 256² 画布上 Walk 的脚底在 228~244 之间跳
 * （16px，占身高 7%），Run 跳 31px（14%）。
 *
 * 对齐之后脚底固定，**身高的变化（抬腿、屈膝、上身起伏）仍然保留** —— 那是
 * 动作本身，不该被抹平。腾空帧（双脚离地）会被贴回地面，这正是精灵表的惯例：
 * 垂直位移交给引擎，不烘进帧里。
 */
function groundCharacter(viewer, groundY) {
  if (groundY == null || !state) return
  const now = footY(viewer)
  if (now == null) return
  const root = characterRoot(viewer)
  if (!root) return
  root.position.y = state.rootY + (groundY - now)
  root.updateMatrixWorld(true)
  state.lastShift = groundY - now
}

/**
 * 找出这张图里**最低的一行墨迹**（人偶的脚底）。白底图，按"三通道都接近 255"判空。
 */
function lowestRow(dataURL) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      try {
        const c = document.createElement('canvas')
        c.width = img.width
        c.height = img.height
        const g = c.getContext('2d')
        g.drawImage(img, 0, 0)
        const d = g.getImageData(0, 0, c.width, c.height).data
        for (let y = c.height - 1; y >= 0; y--) {
          for (let x = 0; x < c.width; x++) {
            const o = (y * c.width + x) * 4
            if (d[o] < 235 || d[o + 1] < 235 || d[o + 2] < 235) {
              resolve({ y, w: c.width, h: c.height })
              return
            }
          }
        }
        resolve(null)
      } catch { resolve(null) }
    }
    img.onerror = () => resolve(null)
    img.src = dataURL
  })
}

/** 整张图垂直平移 dy 像素（正 = 向下），空出来的补白 */
function shiftPng(dataURL, dy) {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      try {
        const c = document.createElement('canvas')
        c.width = img.width
        c.height = img.height
        const g = c.getContext('2d')
        g.fillStyle = '#fff'
        g.fillRect(0, 0, c.width, c.height)
        g.drawImage(img, 0, dy)
        resolve(c.toDataURL('image/png'))
      } catch { resolve(dataURL) }
    }
    img.onerror = () => resolve(dataURL)
    img.src = dataURL
  })
}

/**
 * 帧间「底部对齐」：把所有帧的最低墨迹行压到同一条水平线上。
 *
 * **为什么世界空间对齐还不够**：`groundCharacter` 只把脚**骨**的世界 y 对齐了，
 * 但导出用的是**透视相机** —— 走/跑时脚在前后方向（z）移动，同样的 y 在屏幕上
 * 落点不同（近大远小）。实测 Walk 对齐骨骼后，画面里的脚底仍在 226~249 之间跳
 * （23px，占画布 9%）。
 *
 * 所以最后一步放在**屏幕空间**：量每帧最低的墨迹行，统一平移到最"低"那一帧的
 * 位置。纯像素平移，不改渲染方式（保持和单张导出一致的透视外观）。
 *
 * 基准取"最低那帧"，但**留出 5% 底边距**：不留的话最矮的那帧会贴到画布边缘
 * （实测 252/256），看着挤，也怕某些帧真的贴到 256 被裁。
 */
export async function alignFrames(frames) {
  if (!Array.isArray(frames) || frames.length < 2) return frames
  const boxes = []
  for (const f of frames) boxes.push(await lowestRow(f.png))
  const valid = boxes.filter(Boolean)
  if (!valid.length) return frames
  const h = valid[0].h || 256
  const margin = Math.max(4, Math.round(h * 0.05))
  const base = Math.min(Math.max(...valid.map((b) => b.y)), h - margin - 1)
  const out = []
  for (let i = 0; i < frames.length; i++) {
    const b = boxes[i]
    if (!b) { out.push(frames[i]); continue }
    const dy = base - b.y
    out.push(dy === 0 ? frames[i] : { ...frames[i], png: await shiftPng(frames[i].png, dy) })
  }
  return out
}

/** 清掉当前动作（不删人偶姿势，只是不再按动画驱动） */
export function clearAnimation() {
  if (state?.mixer) state.mixer.stopAllAction()
  state = null
  lastError = ''
}
