/**
 * 多角色场景 / 整角色变换控制器（iframe 侧，宿主 UI 在 PoseStudio.tsx）。
 *
 * 引擎本来就有「1 个可拖关节的 active 人偶 + 任意个 passive 人偶」：
 * `passiveCharacters` Map、`upsertPassiveCharacterFromActive()`、
 * `setPassiveCharacterState()`，而且 `capture()` 会把 passive mesh 一起渲染。
 * 官方 0.7.4 新增的只是「整角色 gizmo + World/Local + 联动 + 每 tab 位置」
 * 这一层**状态与 UI**，纯算法在 `pose_studio/character_transform.js`（原样
 * vendored 自 Utils 0.7.5）。所以这里不换引擎、不激活官方那套编辑器 UI。
 *
 * 撤销/重做复用引擎预留的扩展点：`options.captureHistoryContext()` 会把
 * 「每个角色的放置 + 姿势」一起塞进 history，`options.onHistoryRestore()`
 * 再取回来 —— 这样一次 Ctrl+Z 能同时退回姿势和整角色位移，而不是只退骨骼。
 *
 * 协议（父 → 子）：统一 `{ type:'scene', action, ... }`，见 handle()。
 * 子 → 父：`{ type:'sceneState', scene, pose, yaw, history, commit }`。
 * `commit=false` 表示拖动中间态，宿主只更新显示、别写 project.json。
 */
import {
  clampCharacterPosition,
  translateCharacterPosition,
  modelRotationQuaternion,
  quaternionToModelRotation,
  rotateCharacterAboutPivot,
  setCharacterRotationAboutPivot,
  wrapDegrees,
} from './pose_studio/character_transform.js'

const clone = (o) => JSON.parse(JSON.stringify(o))
const RAD = Math.PI / 180
const PALETTE = ['#e8eaf2', '#6cc4ef', '#4ade9b', '#ff7a90', '#ffd34d', '#b28cff']

/** 一个角色在**某个 tab** 下的放置。默认站姿、缩放 1、无整角色旋转。 */
function zeroPlacement() {
  return { x: 0, y: 0, z: 0, zoom: 1, rotation: [0, 0, 0] }
}

function normPlacement(raw) {
  const src = raw && typeof raw === 'object' ? raw : {}
  const p = clampCharacterPosition(src)
  const zoom = Number(src.zoom)
  const r = Array.isArray(src.rotation) ? src.rotation : [0, 0, 0]
  return {
    x: p.x, y: p.y, z: p.z,
    zoom: Number.isFinite(zoom) ? Math.max(0.2, Math.min(4, zoom)) : 1,
    rotation: [0, 1, 2].map((i) => wrapDegrees(Number(r[i]) || 0)),
  }
}

export function defaultScene() {
  return {
    characters: [{ id: 'c1', name: '角色 1', color: PALETTE[0], pose: null, placement: zeroPlacement() }],
    activeId: 'c1',
    tabs: ['t1'],
    activeTab: 't1',
    placements: { t1: { c1: zeroPlacement() } },
    mode: 'pose',
    space: 'world',
    linkAll: false,
    cameraParams: null,
  }
}

/** 外部传进来的场景一律过一遍：老存档缺字段、坏数值都不该让 iframe 崩。 */
export function normalizeScene(raw) {
  const base = defaultScene()
  if (!raw || typeof raw !== 'object') return base
  const chars = Array.isArray(raw.characters) && raw.characters.length
    ? raw.characters.slice(0, 12).map((c, i) => ({
      id: String(c?.id || `c${i + 1}`),
      name: String(c?.name || `角色 ${i + 1}`),
      color: String(c?.color || PALETTE[i % PALETTE.length]),
      pose: c?.pose && typeof c.pose === 'object' ? c.pose : null,
      placement: normPlacement(c?.placement ?? c?.transform),
    }))
    : base.characters
  const ids = new Set(chars.map((c) => c.id))
  const tabs = Array.isArray(raw.tabs) && raw.tabs.length
    ? raw.tabs.slice(0, 16).map(String) : base.tabs
  const placements = {}
  for (const t of tabs) {
    placements[t] = {}
    const src = raw.placements?.[t] || {}
    for (const c of chars) placements[t][c.id] = normPlacement(src[c.id] || c.placement)
  }
  const activeTab = tabs.includes(raw.activeTab) ? String(raw.activeTab) : tabs[0]
  const cp = raw.cameraParams && typeof raw.cameraParams === 'object' ? raw.cameraParams : null
  return {
    characters: chars,
    activeId: ids.has(raw.activeId) ? String(raw.activeId) : chars[0].id,
    tabs, activeTab, placements,
    mode: ['pose', 'move', 'rotate'].includes(raw.mode) ? raw.mode : 'pose',
    space: ['world', 'local'].includes(raw.space) ? raw.space : 'world',
    linkAll: !!raw.linkAll,
    cameraParams: cp,
  }
}

export function createSceneController({ getViewer, canvas, post, getYaw, measure }) {
  let scene = defaultScene()
  let gizmo = null
  let gizmoViewer = null
  let drag = null
  let applying = false

  /* ------------------------------------------------------------ 小工具 */
  function tabPlacements() {
    if (!scene.placements[scene.activeTab]) scene.placements[scene.activeTab] = {}
    return scene.placements[scene.activeTab]
  }
  function placementOf(id) {
    const tab = tabPlacements()
    if (!tab[id]) tab[id] = zeroPlacement()
    return tab[id]
  }
  function activeChar() {
    return scene.characters.find((c) => c.id === scene.activeId) || scene.characters[0]
  }
  function meshFor(v, id) {
    return id === scene.activeId ? v.skinnedMesh : v.passiveCharacters?.get(String(id))?.mesh
  }
  /** 切 active / 增删角色前，先把当前那个可拖关节的人偶姿势收回来，否则会被覆盖。 */
  function captureActivePose(v) {
    const c = activeChar()
    if (c && v?.getPose) c.pose = v.getPose()
  }

  /** 官方姿势库带来的构图参数：直接走它自己的 capture 相机计算，别自己猜距离。 */
  function applyCameraParams(v, cp) {
    if (!cp || !v?.snapToCaptureCamera) return
    const { w, h } = measure()
    const zoom = Number(cp.zoom) || 1
    try {
      v.setCameraParams?.({ ...cp })
      v.snapToCaptureCamera(w, h, Math.max(0.2, Math.min(4, zoom)),
        Number(cp.offset_x) || 0, Number(cp.offset_y) || 0,
        Number(cp.yaw_deg) || 0, Number(cp.pitch_deg) || 0)
    } catch { /* captureCamera 还没就绪就忽略，下次 load 会再来 */ }
    v.requestRender?.()
  }

  /* ------------------------------------------------------------ gizmo */
  function ensureGizmo(v) {
    if (gizmo && gizmoViewer === v) return gizmo
    if (gizmo) {
      gizmo.detach()
      gizmo.parent?.remove(gizmo)
      gizmo.dispose?.()
    }
    // 用引擎同款 TransformControls 类（core.js 已经动态加载过同一 URL），
    // 保证 THREE 是同一个实例；另起一个而不是复用 v.transform，是为了不跟
    // 「拖关节 / IK effector」那个 gizmo 抢 attach 目标。
    gizmo = new v.TransformControls(v.camera, canvas)
    gizmo.setSize(0.85)
    v.scene.add(gizmo)
    gizmoViewer = v

    gizmo.addEventListener('dragging-changed', (e) => {
      v.orbit.enabled = !e.value
      if (e.value) {
        drag = { starts: clone(scene.placements[scene.activeTab] || {}) }
        v.recordState?.()
      } else {
        drag = null
        void emit(true)
      }
    })
    gizmo.addEventListener('objectChange', () => {
      if (drag && !applying) readObject(v)
    })
    gizmo.addEventListener('change', () => v.requestRender?.())
    return gizmo
  }

  function attachGizmo(v) {
    const g = ensureGizmo(v)
    if (scene.mode === 'pose') { g.detach(); return }
    const mesh = meshFor(v, scene.activeId)
    if (!mesh) { g.detach(); return }
    v.transform?.detach?.()                 // 关节 gizmo 让位，避免两套手柄重叠
    g.attach(mesh)
    g.setMode(scene.mode === 'move' ? 'translate' : 'rotate')
    g.setSpace(scene.space === 'local' ? 'local' : 'world')
  }

  /** 把 gizmo 拖出来的结果读回 scene，并按 linkAll 把同一 delta 分给其它角色。 */
  function readObject(v) {
    const obj = gizmo?.object
    if (!obj || !drag) return
    const id = scene.activeId
    const from = drag.starts[id] || zeroPlacement()
    const p = placementOf(id)
    const pos = clampCharacterPosition({ x: obj.position.x, y: obj.position.y, z: obj.position.z })
    let rotation = p.rotation
    if (scene.mode === 'rotate') {
      const r = quaternionToModelRotation(v.THREE, obj.quaternion,
        { x: p.rotation[0], y: p.rotation[1], z: p.rotation[2] })
      rotation = [r.x, r.y, r.z]
    }
    const zoom = Math.max(0.2, Math.min(4, Number(obj.scale?.x) || 1))
    tabPlacements()[id] = { x: pos.x, y: pos.y, z: pos.z, zoom, rotation }

    if (scene.linkAll) {
      const delta = { x: pos.x - from.x, y: pos.y - from.y, z: pos.z - from.z }
      let deltaQ = null
      if (scene.mode === 'rotate') {
        deltaQ = modelRotationQuaternion(v.THREE, rotation)
          .multiply(modelRotationQuaternion(v.THREE, from.rotation).invert())
      }
      const pivot = v.meshCenter
        ? { x: v.meshCenter.x, y: v.meshCenter.y, z: v.meshCenter.z }
        : { x: 0, y: 0, z: 0 }
      const linkedStarts = scene.characters
        .filter((c) => c.id !== id).map((c) => drag.starts[c.id] || zeroPlacement())
      for (const c of scene.characters) {
        if (c.id === id) continue
        const start = drag.starts[c.id] || zeroPlacement()
        if (scene.mode === 'rotate' && deltaQ) {
          const out = rotateCharacterAboutPivot(v.THREE, {
            position: start, rotation: start.rotation, pivot, delta: deltaQ,
          })
          if (out) tabPlacements()[c.id] = {
            ...start, x: out.position.x, y: out.position.y, z: out.position.z,
            rotation: [out.rotation.x, out.rotation.y, out.rotation.z],
          }
        } else {
          const out = translateCharacterPosition(start, delta, linkedStarts)
          tabPlacements()[c.id] = { ...start, x: out.x, y: out.y, z: out.z }
        }
      }
    }
    applyPlacements(v)
  }

  /** 只刷位置/旋转，不重建骨架、不动 pose —— 拖动时每帧都走。 */
  function applyPlacements(v) {
    applying = true
    try {
      for (const c of scene.characters) {
        const p = placementOf(c.id)
        if (c.id === scene.activeId) {
          v.setActiveCharacterAppearance?.({ transform: p })
          v.setModelRotation?.(p.rotation[0], p.rotation[1], p.rotation[2])
        } else {
          v.setPassiveCharacterPlacement?.(c.id, { transform: p, rotation: p.rotation })
        }
      }
      v.requestRender?.()
    } finally { applying = false }
  }

  /** 全量同步：角色集合变化、切换 active/tab、载入场景时才走（会重建 passive mesh）。 */
  function syncToViewer(v, { camera = false } = {}) {
    const ids = new Set(scene.characters.map((c) => c.id))
    for (const key of Array.from(v.passiveCharacters?.keys?.() || [])) {
      if (!ids.has(key) || key === scene.activeId) v.removePassiveCharacter(key)
    }
    const active = activeChar()
    if (active?.pose) v.setPose(active.pose, true)
    const ap = placementOf(active.id)
    v.setActiveCharacterAppearance?.({ color: active.color || undefined, transform: ap })
    v.setModelRotation?.(ap.rotation[0], ap.rotation[1], ap.rotation[2])

    for (const c of scene.characters) {
      if (c.id === active.id) continue
      const p = placementOf(c.id)
      if (!v.passiveCharacters?.has(c.id)) {
        v.upsertPassiveCharacterFromActive(c.id, {
          pose: c.pose || {}, color: c.color || undefined, transform: p,
        })
      } else {
        v.setPassiveCharacterState(c.id, {
          pose: c.pose || {}, color: c.color || undefined, transform: p,
        })
      }
      v.setPassiveCharacterPlacement?.(c.id, { rotation: p.rotation })
    }
    if (camera && scene.cameraParams) applyCameraParams(v, scene.cameraParams)
    attachGizmo(v)
    v.requestRender?.()
  }

  async function emit(commit = true) {
    const v = await getViewer()
    post({
      type: 'sceneState',
      scene: clone(scene),
      pose: v.getPose?.(),
      yaw: getYaw(),
      history: { canUndo: (v.history?.length || 0) > 0, canRedo: (v.future?.length || 0) > 0 },
      commit,
    })
  }

  /* ------------------------------------------------------------ 撤销扩展点 */
  /** 引擎 getHistoryPose() 会把它合并进 history，undo 时原样回给 onHistoryRestore。 */
  function captureHistoryContext() {
    return {
      placements: clone(scene.placements),
      activeTab: scene.activeTab,
      activeId: scene.activeId,
      characters: scene.characters.map((c) => ({ id: c.id, pose: c.pose })),
    }
  }
  function onHistoryRestore(pose) {
    const ed = pose?.editorState
    if (!ed) return
    if (ed.placements) scene.placements = clone(ed.placements)
    if (ed.activeTab && scene.tabs.includes(ed.activeTab)) scene.activeTab = ed.activeTab
    if (Array.isArray(ed.characters)) {
      for (const item of ed.characters) {
        const c = scene.characters.find((x) => x.id === item.id)
        if (c) c.pose = item.pose
      }
    }
  }

  /* ------------------------------------------------------------ 消息入口 */
  async function handle(msg) {
    const v = await getViewer()
    const action = msg.action
    if (action === 'load') {
      scene = normalizeScene(msg.scene)
      syncToViewer(v, { camera: true })
      await emit(true)
      return
    }
    if (action === 'get') { await emit(true); return }
    if (action === 'select') {
      if (msg.id === scene.activeId) return
      captureActivePose(v)
      scene.activeId = String(msg.id)
      syncToViewer(v)
      await emit(true)
      return
    }
    if (action === 'add') {
      captureActivePose(v)
      const used = new Set(scene.characters.map((c) => c.id))
      let i = scene.characters.length + 1
      while (used.has(`c${i}`)) i += 1
      const id = `c${i}`
      scene.characters.push({
        id, name: `角色 ${i}`, color: PALETTE[(i - 1) % PALETTE.length],
        // 新人偶复制当前姿势，用户一加就能看到两个同姿势角色，再各自摆
        pose: clone(activeChar().pose || {}), placement: zeroPlacement(),
      })
      for (const t of scene.tabs) scene.placements[t][id] = zeroPlacement()
      scene.activeId = id
      syncToViewer(v)
      await emit(true)
      return
    }
    if (action === 'remove') {
      if (scene.characters.length <= 1) return
      captureActivePose(v)
      const id = String(msg.id || scene.activeId)
      scene.characters = scene.characters.filter((c) => c.id !== id)
      for (const t of scene.tabs) delete scene.placements[t][id]
      if (scene.activeId === id) scene.activeId = scene.characters[0].id
      syncToViewer(v)
      await emit(true)
      return
    }
    if (action === 'rename') {
      const c = scene.characters.find((x) => x.id === msg.id)
      if (c) c.name = String(msg.name || c.name).slice(0, 24)
      await emit(true)
      return
    }
    if (action === 'mode' || action === 'space' || action === 'linkAll') {
      if (action === 'mode' && ['pose', 'move', 'rotate'].includes(msg.value)) scene.mode = msg.value
      if (action === 'space' && ['world', 'local'].includes(msg.value)) scene.space = msg.value
      if (action === 'linkAll') scene.linkAll = !!msg.value
      attachGizmo(v)
      await emit(true)
      return
    }
    if (action === 'addTab' || action === 'tab') {
      if (action === 'addTab') {
        let i = scene.tabs.length + 1
        while (scene.tabs.includes(`t${i}`)) i += 1
        const id = `t${i}`
        scene.tabs.push(id)
        // 新 tab 从当前 tab 复制一份位置：用户要的是「每个 tab 独立」，
        // 但从全 0 开始会让人以为刚才摆的位置丢了。
        scene.placements[id] = clone(scene.placements[scene.activeTab] || {})
        scene.activeTab = id
      } else if (scene.tabs.includes(String(msg.tabId))) {
        scene.activeTab = String(msg.tabId)
      }
      applyPlacements(v)
      await emit(true)
      return
    }
    if (action === 'transform') {
      const id = String(msg.id || scene.activeId)
      const c = scene.characters.find((x) => x.id === id)
      if (!c) return
      const cur = placementOf(id)
      const patch = msg.patch || {}
      const next = {
        ...cur,
        x: Number.isFinite(Number(patch.x)) ? Number(patch.x) : cur.x,
        y: Number.isFinite(Number(patch.y)) ? Number(patch.y) : cur.y,
        z: Number.isFinite(Number(patch.z)) ? Number(patch.z) : cur.z,
        zoom: Number.isFinite(Number(patch.zoom)) ? Number(patch.zoom) : cur.zoom,
        rotation: Array.isArray(patch.rotation) ? patch.rotation : cur.rotation,
      }
      // 精确输入旋转时保持骨盆不动（官方 setCharacterRotationAboutPivot）
      if (Array.isArray(patch.rotation) && v.THREE && v.meshCenter) {
        const out = setCharacterRotationAboutPivot(v.THREE, {
          position: cur, rotation: cur.rotation,
          pivot: { x: v.meshCenter.x, y: v.meshCenter.y, z: v.meshCenter.z },
          nextRotation: { x: next.rotation[0], y: next.rotation[1], z: next.rotation[2] },
        })
        if (out) {
          next.x = out.position.x; next.y = out.position.y; next.z = out.position.z
          next.rotation = [out.rotation.x, out.rotation.y, out.rotation.z]
        }
      }
      if (msg.record !== false) v.recordState?.()
      tabPlacements()[id] = normPlacement(next)
      applyPlacements(v)
      attachGizmo(v)                 // 刷新 gizmo 手柄位置（数值输入后要跟手）
      await emit(true)
      return
    }
    if (action === 'camera') {
      scene.cameraParams = msg.cameraParams || null
      if (scene.cameraParams) applyCameraParams(v, scene.cameraParams)
      await emit(true)
      return
    }
    if (action === 'undo' || action === 'redo') {
      if (action === 'undo') v.undo?.()
      else v.redo?.()
      // onHistoryRestore 已经把 placements 换回历史快照，这里再全量刷一次
      syncToViewer(v)
      post({ type: 'pose', pose: v.getPose?.(), yaw: getYaw() })
      await emit(true)
      return
    }
  }

  /** capture 是同步的 canvas.toDataURL，出图前必须把 gizmo 手柄藏掉。 */
  function hideForCapture() {
    if (!gizmo) return
    gizmo._meowWasVisible = gizmo.visible
    gizmo.visible = false
  }
  function showAfterCapture() {
    if (!gizmo || gizmo._meowWasVisible === undefined) return
    gizmo.visible = gizmo._meowWasVisible
    delete gizmo._meowWasVisible
  }

  return {
    handle, emit, captureHistoryContext, onHistoryRestore,
    hideForCapture, showAfterCapture,
    getScene: () => clone(scene),
    /** 关节拖动 / setPose / init 之后把最新 pose 记进当前角色，
     *  否则切到别的角色时用的还是上一次收下来的旧姿势。 */
    notePose: (pose) => { const c = activeChar(); if (c && pose) c.pose = pose },
  }
}
