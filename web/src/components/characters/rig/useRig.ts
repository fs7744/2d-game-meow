/**
 * 内嵌编辑器的状态与全部编辑命令。
 *
 * 所有骨骼 / 动画 / 约束的真实计算都调用 facade 里一比一搬入的 rigby2d 运行时：
 * Skeleton、Pose、Animation、setKey、autoRig、constraintdata、interop、binary、io。
 * 本文件只管「当前选中谁、现在第几秒、界面要不要重画」，不重新实现任何算法。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  model, pose as P, anim, autorig, io, images, math,
  interop, binary, constraintdata as CD, zip as zipNs, output,
} from '../../../rig/facade.js'
import { drawScene } from './rigRender'
import {
  basename, type Sel, type RigMode, type RigTool, type ViewState, type RigSource,
} from './types'

const enc = new TextEncoder()

function memFs(out: Record<string, Uint8Array>) {
  return {
    writeFile: async (p: string, b: Uint8Array) => {
      out[basename(p)] = b instanceof Uint8Array ? b : new Uint8Array(b)
    },
    writeText: async (p: string, t: string) => { out[basename(p)] = enc.encode(String(t)) },
    list: async () => [],
    finish: async () => {},
  }
}

export type RigApi = ReturnType<typeof useRig>

export function useRig() {
  const docRef = useRef<any>(null)
  const imagesRef = useRef<any>(null)
  const poseRef = useRef<any>(null)
  if (!docRef.current) {
    docRef.current = new model.Skeleton()
    imagesRef.current = new images.Images()
    poseRef.current = new P.Pose(docRef.current)
  }

  const timeRef = useRef(0)
  const animRef = useRef(-1)
  const modeRef = useRef<RigMode>('setup')
  const loopRef = useRef(true)
  const playingRef = useRef(false)
  /**
   * 物理约束的模拟状态必须跨帧保留（上游 app.physics），否则摆动类约束在播放/离屏渲染里
   * 永远是静止的：`solvePhysics` 只在 delta > 0 时才推进仿真。
   */
  const physicsRef = useRef<Map<string, any>>(new Map())
  const physicsDeltaRef = useRef(0)
  const [rev, setRev] = useState(0)
  const bump = useCallback(() => setRev((v) => v + 1), [])

  const [sel, setSel] = useState<Sel>(null)
  const [mode, setModeState] = useState<RigMode>('setup')
  const [animation, setAnimationState] = useState(-1)
  const [time, setTimeState] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [loop, setLoopState] = useState(true)
  const [fps, setFps] = useState(30)
  const [view, setView] = useState<ViewState>({ zoom: 1, x: 0, y: 0 })
  const [showBones, setShowBones] = useState(true)
  const [showMesh, setShowMesh] = useState(false)
  const [showImages, setShowImages] = useState(true)
  const [onion, setOnion] = useState(false)
  const [grid, setGrid] = useState(true)
  const [tool, setTool] = useState<RigTool>('select')
  const [status, setStatus] = useState('')
  const [source, setSource] = useState<RigSource>({ kind: 'blank', label: '空白工程', name: '' })

  /** 把 setup pose + 当前动画在 t 秒的采样合成为 pose.world / quads。
   *  `dt` 是距上一帧的秒数：物理约束据此推进（与上游 `app.pose()` / `AnimationState.apply` 同语义）。 */
  const recompute = useCallback((t = timeRef.current, dt = 0) => {
    const d = docRef.current, p = poseRef.current
    if (!d || !p) return
    p.reset(d)
    const ai = animRef.current
    if (modeRef.current === 'animate' && ai >= 0 && d.animations[ai]) {
      if (dt > 0 && playingRef.current) physicsDeltaRef.current += dt
      // 暂停时上游会清空物理状态（app.pose(): `else if (this.physics.size) this.physics.clear()`），
      // 这样再次播放时摆动从动画姿势重新起振，而不是接着上次的残留。
      if (!playingRef.current && physicsRef.current.size) physicsRef.current.clear()
      p.physics = physicsRef.current
      p.physicsDelta = physicsDeltaRef.current
      physicsDeltaRef.current = 0
      p.apply(d, d.animations[ai], t, 1)
    } else if (physicsRef.current.size) {
      physicsRef.current.clear()
      physicsDeltaRef.current = 0
    }
    p.updateWorld(d)
  }, [])

  const commit = useCallback(() => { recompute(); bump() }, [recompute, bump])
  const say = useCallback((s: string) => setStatus(s), [])

  /** 洋葱皮：前后各两帧的骨骼轮廓，用独立的 Pose 计算，不碰主 pose。 */
  const ghosts = useCallback(() => {
    const d = docRef.current
    if (!onion || modeRef.current !== 'animate' || animRef.current < 0) return []
    const a = d.animations[animRef.current]
    if (!a) return []
    return [-0.2, -0.1, 0.1, 0.2].map((dt) => {
      const p = new P.Pose(d)
      p.reset(d)
      p.apply(d, a, Math.max(0, Math.min(a.duration || 1, timeRef.current + dt)), 1)
      p.updateWorld(d)
      return { pose: p, alpha: Math.abs(dt) < 0.15 ? 0.35 : 0.18 }
    })
  }, [onion])

  // 播放状态同步：暂停时清空物理模拟，避免下次播放接着上次的残留速度。
  useEffect(() => {
    playingRef.current = playing
    if (!playing) { physicsRef.current.clear(); physicsDeltaRef.current = 0 }
  }, [playing])

  // 播放：单一 rAF 驱动时间与姿势（双时钟会滑步）。画布自己另跑 rAF 读 poseRef。
  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    let lastUi = 0
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      const d = docRef.current
      const a = d.animations[animRef.current]
      const dur = (a?.duration || 1)
      let t = timeRef.current + dt
      if (t > dur) t = loopRef.current ? t % dur : dur
      timeRef.current = t
      // 传真实帧间隔：物理约束要逐帧仿真，按时间 t 直接采样会让摆动/惯性完全停住。
      recompute(t, dt)
      if (now - lastUi > 90) { setTimeState(t); bump(); lastUi = now }
      if (!loopRef.current && t >= dur) { setPlaying(false); return }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, recompute, bump])

  // ---------------------------------------------------------------- 文档

  const resetDoc = useCallback((name = 'untitled', src?: RigSource) => {
    const d = new model.Skeleton()
    d.name = name
    docRef.current = d
    imagesRef.current = new images.Images()
    poseRef.current = new P.Pose(d)
    timeRef.current = 0; animRef.current = -1
    setTimeState(0); setAnimationState(-1); setSel(null); setPlaying(false)
    if (src) setSource(src)
    commit()
  }, [commit])

  const loadWorkfile = useCallback(async (bytes: Uint8Array, name = 'project.rigby', src?: RigSource) => {
    const imgs = new images.Images()
    const r = await io.openProject(name, {
      readFile: async () => bytes,
      readText: async () => new TextDecoder().decode(bytes),
      list: async () => [],
    }, imgs)
    docRef.current = r.doc
    imagesRef.current = imgs
    poseRef.current = new P.Pose(r.doc)
    timeRef.current = 0; animRef.current = -1
    setTimeState(0); setAnimationState(-1); setSel(null); setPlaying(false)
    if (src) setSource(src)
    commit()
    return (r.warnings || []) as string[]
  }, [commit])

  /** 用外部工程导入 / 场景切换得到的 Skeleton 直接替换当前文档。 */
  const adoptDoc = useCallback((d: any, imgs: any, src?: RigSource) => {
    docRef.current = d
    imagesRef.current = imgs
    poseRef.current = new P.Pose(d)
    timeRef.current = 0; animRef.current = -1
    setTimeState(0); setAnimationState(-1); setSel(null); setPlaying(false)
    if (src) setSource(src)
    commit()
  }, [commit])

  const importPsd = useCallback(async (name: string, bytes: Uint8Array) => {
    const n = await io.importPsd({ name, bytes }, docRef.current, imagesRef.current)
    if (docRef.current.name === 'untitled') docRef.current.name = basename(name).replace(/\.psd$/i, '')
    commit()
    return n as number
  }, [commit])

  const importPngs = useCallback(async (files: { name: string; bytes: Uint8Array }[]) => {
    const n = await io.importPngs(files, 0, docRef.current, imagesRef.current)
    commit()
    return n as number
  }, [commit])

  const importSvg = useCallback(async (name: string, bytes: Uint8Array) => {
    const n = await io.importSvg({ name, bytes }, 0, docRef.current, imagesRef.current, { scale: 1 })
    if (docRef.current.name === 'untitled') docRef.current.name = basename(name).replace(/\.svg$/i, '')
    commit()
    return n as number
  }, [commit])

  // ---------------------------------------------------------------- AutoRig

  const alphaOf = useCallback((imgName: string) => {
    const im = imagesRef.current
    if (!imgName || !im.contains(imgName) || im.alphaAt(imgName, 0, 0) === null) return null
    const s = im.map.get(imgName)
    return { width: s.canvas.width, height: s.canvas.height, data: s.alpha }
  }, [])

  const previewParts = useCallback((roles: Record<string, string> = {},
    sides: Record<string, string> = {}) =>
    autorig.findParts(docRef.current, alphaOf, { roles, sides }), [alphaOf])

  const runAutoRig = useCallback((options: any) => {
    const report = autorig.autoRig(docRef.current, alphaOf, options)
    poseRef.current = new P.Pose(docRef.current)
    setSel(null)
    commit()
    return report
  }, [alphaOf, commit])

  // ---------------------------------------------------------------- 骨骼 / 槽

  const rebuildPose = useCallback(() => {
    poseRef.current = new P.Pose(docRef.current)
    commit()
  }, [commit])

  const addBone = useCallback((parent: number | null) => {
    const d = docRef.current
    const name = d.uniqueBoneName('bone')
    const y = parent == null ? 0 : (d.bones[parent]?.length || 40)
    const b = model.makeBone(name, parent, math.transform(0, y, 0), 40)
    const idx = d.addBone(b)
    rebuildPose()
    setSel({ kind: 'bone', index: idx })
    return idx
  }, [rebuildPose])

  const removeBone = useCallback((i: number) => {
    if (i <= 0) return
    docRef.current.removeBone(i)
    setSel(null)
    rebuildPose()
  }, [rebuildPose])

  const reparentBone = useCallback((i: number, parent: number) => {
    docRef.current.reparentBone(i, parent)
    rebuildPose()
  }, [rebuildPose])

  const patchBone = useCallback((i: number, patch: Record<string, any>) => {
    const b = docRef.current.bones[i]
    if (!b) return
    for (const [k, v] of Object.entries(patch)) {
      if (['name', 'parent', 'length', 'inherit', 'color', 'skinRequired'].includes(k)) b[k] = v
      else b.setup[k] = v
    }
    commit()
  }, [commit])

  const addSlot = useCallback((bone = 0) => {
    const d = docRef.current
    const s = model.makeSlot(d.uniqueSlotName('slot'), bone, null)
    d.slots.push(s)
    setSel({ kind: 'slot', index: d.slots.length - 1 })
    commit()
  }, [commit])

  const removeSlot = useCallback((i: number) => {
    docRef.current.removeSlot(i)
    setSel(null)
    commit()
  }, [commit])

  const moveSlot = useCallback((from: number, to: number) => {
    docRef.current.moveSlot(from, to)
    setSel({ kind: 'slot', index: to })
    commit()
  }, [commit])

  const patchSlot = useCallback((i: number, patch: Record<string, any>) => {
    const s = docRef.current.slots[i]
    if (!s) return
    Object.assign(s, patch)
    commit()
  }, [commit])

  const patchAttachment = useCallback((slot: number, patch: Record<string, any>) => {
    const s = docRef.current.slots[slot]
    const a = s?.attachments?.find((x: any) => x.name === s.attachment) || s?.attachments?.[0]
    if (!a) return
    Object.assign(a, patch)
    commit()
  }, [commit])

  /** 拖动骨骼：setup 模式改 rest 变换，animate 模式在当前动画里写关键帧（增量）。 */
  const transformBone = useCallback((i: number, t: RigTool,
    delta: { dx: number; dy: number; rotation: number; scale: number }) => {
    const d = docRef.current
    const b = d.bones[i]
    if (!b) return
    const p = poseRef.current
    const L = { ...(p.local[i] || math.cloneTransform(b.setup)) }
    if (t === 'rotate') L.rotation += delta.rotation
    else if (t === 'translate') { L.x += delta.dx; L.y += delta.dy }
    else if (t === 'scale') {
      L.scale_x *= delta.scale; L.scale_y *= delta.scale
    } else return
    if (modeRef.current === 'setup') {
      Object.assign(b.setup, L)
    } else {
      const a = d.animations[animRef.current]
      if (!a) { say('先在时间轴里新建一个动画'); return }
      const tl = a.boneTimelineMut(i)
      anim.setKey(tl.rotate, timeRef.current, L.rotation - b.setup.rotation, 'linear')
      anim.setKey(tl.translate, timeRef.current, [L.x - b.setup.x, L.y - b.setup.y], 'linear')
      anim.setKey(tl.scale, timeRef.current,
        [L.scale_x / (b.setup.scale_x || 1), L.scale_y / (b.setup.scale_y || 1)], 'linear')
    }
    commit()
  }, [commit, say])

  // ---------------------------------------------------------------- 动画 / 关键帧

  const setMode = useCallback((m: RigMode) => {
    modeRef.current = m
    setModeState(m)
    if (m === 'animate' && animRef.current < 0 && docRef.current.animations.length) {
      animRef.current = 0; setAnimationState(0)
    }
    commit()
  }, [commit])

  const setAnimation = useCallback((i: number) => {
    animRef.current = i
    setAnimationState(i)
    const a = docRef.current.animations[i]
    if (a) { timeRef.current = Math.min(timeRef.current, a.duration || 1); setTimeState(timeRef.current) }
    commit()
  }, [commit])

  const setTime = useCallback((t: number) => {
    timeRef.current = Math.max(0, t)
    setTimeState(timeRef.current)
    recompute()
    bump()
  }, [recompute, bump])

  const addAnimation = useCallback((name: string) => {
    const d = docRef.current
    const a = new anim.Animation(name || `动画 ${d.animations.length + 1}`, 1)
    d.animations.push(a)
    const i = d.animations.length - 1
    animRef.current = i; setAnimationState(i)
    modeRef.current = 'animate'; setModeState('animate')
    commit()
    return i
  }, [commit])

  const removeAnimation = useCallback((i: number) => {
    docRef.current.animations.splice(i, 1)
    animRef.current = -1; setAnimationState(-1)
    commit()
  }, [commit])

  const patchAnimation = useCallback((i: number, patch: Record<string, any>) => {
    const a = docRef.current.animations[i]
    if (!a) return
    Object.assign(a, patch)
    commit()
  }, [commit])

  /** 把当前姿势写成一个关键帧（旋转 / 位移 / 缩放三条通道）。 */
  const keyCurrent = useCallback(() => {
    const d = docRef.current
    const a = d.animations[animRef.current]
    if (!a) { say('先新建一个动画'); return }
    const i = sel?.kind === 'bone' ? sel.index : 0
    const b = d.bones[i]
    if (!b) return
    const L = poseRef.current.local[i] || math.cloneTransform(b.setup)
    const tl = a.boneTimelineMut(i)
    anim.setKey(tl.rotate, timeRef.current, L.rotation - b.setup.rotation, 'linear')
    anim.setKey(tl.translate, timeRef.current, [L.x - b.setup.x, L.y - b.setup.y], 'linear')
    anim.setKey(tl.scale, timeRef.current,
      [L.scale_x / (b.setup.scale_x || 1), L.scale_y / (b.setup.scale_y || 1)], 'linear')
    a.duration = Math.max(a.duration || 0, timeRef.current + 1 / fps)
    commit()
    say(`已在 ${timeRef.current.toFixed(2)}s 给「${b.name}」打关键帧`)
  }, [sel, fps, commit, say])

  const allKeys = useCallback(() => {
    const a = docRef.current.animations[animRef.current]
    if (!a) return []
    const out: { label: string; keys: any[]; channel: string; owner: string }[] = []
    a.bones.forEach((tl: any) => {
      for (const ch of anim.BONE_CHANNELS) {
        if (tl[ch]?.length) out.push({ label: `${docRef.current.bones[tl.bone]?.name || tl.bone} · ${ch}`,
          keys: tl[ch], channel: ch, owner: 'bone' })
      }
    })
    a.slots.forEach((tl: any) => {
      for (const ch of anim.SLOT_CHANNELS) {
        if (tl[ch]?.length) out.push({ label: `${docRef.current.slots[tl.slot]?.name || tl.slot} · ${ch}`,
          keys: tl[ch], channel: ch, owner: 'slot' })
      }
    })
    return out
  }, [])

  const patchKey = useCallback((keys: any[], idx: number, patch: Record<string, any>) => {
    const k = keys[idx]
    if (!k) return
    Object.assign(k, patch)
    if (patch.time !== undefined) anim.resort(keys, idx)
    commit()
  }, [commit])

  const removeKey = useCallback((keys: any[], idx: number) => {
    keys.splice(idx, 1)
    commit()
  }, [commit])

  // ---------------------------------------------------------------- 皮肤 / 事件 / 约束 / 姿势库

  const addEvent = useCallback(() => {
    const d = docRef.current
    d.events.push({ name: `event_${d.events.length + 1}`, int: 0, float: 0, string: '', audio: '', volume: 1, balance: 0 })
    setSel({ kind: 'event', index: d.events.length - 1 })
    commit()
  }, [commit])

  const removeEvent = useCallback((i: number) => {
    docRef.current.events.splice(i, 1); setSel(null); commit()
  }, [commit])

  const patchEvent = useCallback((i: number, patch: Record<string, any>) => {
    Object.assign(docRef.current.events[i], patch); commit()
  }, [commit])

  const addSkin = useCallback(() => {
    const d = docRef.current
    d.skins.push({ name: `skin_${d.skins.length + 1}`, slots: [], bones: [], ik: [], color: null, extra: {} })
    setSel({ kind: 'skin', index: d.skins.length - 1 })
    commit()
  }, [commit])

  const removeSkin = useCallback((i: number) => {
    docRef.current.skins.splice(i, 1); setSel(null); commit()
  }, [commit])

  const constraints = useCallback(() => CD.listConstraints(docRef.current), [rev]) // eslint-disable-line react-hooks/exhaustive-deps

  const addConstraint = useCallback((data: any) => {
    CD.addConstraint(docRef.current, data)
    commit()
  }, [commit])

  const removeConstraint = useCallback((name: string) => {
    CD.removeConstraint(docRef.current, name)
    setSel(null); commit()
  }, [commit])

  const addPose = useCallback(() => {
    const d = docRef.current
    const bones: Record<string, any> = {}
    d.bones.forEach((b: any, i: number) => { bones[b.name] = math.cloneTransform(poseRef.current.local[i]) })
    d.poses.push({ name: `姿势 ${d.poses.length + 1}`, bones, slots: {} })
    setSel({ kind: 'pose', index: d.poses.length - 1 })
    commit()
  }, [commit])

  const applyPose = useCallback((i: number) => {
    const d = docRef.current, p = d.poses[i]
    if (!p) return
    d.bones.forEach((b: any) => {
      const t = p.bones?.[b.name]
      if (t) Object.assign(b.setup, t)
    })
    commit()
  }, [commit])

  // ---------------------------------------------------------------- 导出

  /** 一次生成用户要的全部无损格式 + 可继续编辑的工程文件。 */
  const buildExports = useCallback(async () => {
    const d = docRef.current, imgs = imagesRef.current
    const out: Record<string, Uint8Array> = {}
    const fs = memFs(out)
    const warnings: string[] = []
    const r38 = await io.exportSkeletonJson('skeleton.json', fs, d, imgs, '3.8')
    warnings.push(...(r38.warnings || []))
    out['skeleton-3.8.json'] = out['skeleton.json']
    delete out['skeleton.json']
    for (const v of ['4.2', '4.3']) {
      const r = interop.writeSkeletonJson(d, { version: v })
      out[`skeleton-${v}.json`] = enc.encode(JSON.stringify(r.json, null, 2))
      warnings.push(...(r.warnings || []).map((w: string) => `${v}: ${w}`))
    }
    const rb = interop.writeSkeletonJson(d, { version: '4.3', binary: true })
    out['skeleton.skel'] = binary.writeSkeletonBinary(rb.json)
    const rw = await io.saveWorkfile('project.rigby', fs, d, imgs, [])
    warnings.push(...(rw.warnings || []))
    return { files: out, warnings,
      summary: { bones: d.bones.length, slots: d.slots.length,
        animations: d.animations.length, constraints: CD.listConstraints(d).length,
        animationNames: d.animations.map((a: any) => a.name) } }
  }, [])

  /** PNG 序列（影片工作区的离屏渲染路线）：逐帧采样姿势并打包 ZIP。 */
  const renderFrames = useCallback(async (count: number, width = 640, height = 640) => {
    const d = docRef.current, imgs = imagesRef.current
    const a = d.animations[animRef.current]
    if (!a) throw new Error('先选一个动画')
    const cv = document.createElement('canvas')
    cv.width = width; cv.height = height
    const ctx = cv.getContext('2d')!
    const entries: { path: string; bytes: Uint8Array }[] = []
    // 跟影片工作区的离屏渲染用同一套自动取景（core/output.js），否则高个子角色会被裁掉。
    const bounds = output.animationBounds(d, animRef.current, 30)
    const fr = output.outputFrame({ width, height, fps: 30, frame: null }, bounds)
    const zoom = 1 / Math.max(1e-6, fr.scale)
    const view = { zoom, x: -fr.cx * zoom, y: fr.cy * zoom }
    // 物理约束要逐帧推进：单个 Pose + 累加 dt，跟上游离线渲染一样跨帧保留模拟状态。
    const p = new P.Pose(d)
    const step = (a.duration || 1) / Math.max(1, count - 1)
    for (let f = 0; f < count; f++) {
      const t = (a.duration || 1) * (f / Math.max(1, count - 1))
      p.reset(d)
      p.physicsDelta = f > 0 ? step : 0
      p.apply(d, a, t, 1); p.updateWorld(d)
      drawScene(ctx, d, p, imgs, {
        width, height, view,
        showBones: false, showMesh: false, showImages: true, onion: false,
        sel: null, grid: false,
      })
      const blob: Blob = await new Promise((res) => cv.toBlob((b) => res(b || new Blob()), 'image/png'))
      entries.push({ path: `frames/frame_${String(f).padStart(4, '0')}.png`,
        bytes: new Uint8Array(await blob.arrayBuffer()) })
    }
    return zipNs.zip(entries) as Uint8Array
  }, [])

  const setLoop = useCallback((v: boolean) => { loopRef.current = v; setLoopState(v) }, [])

  return {
    docRef, imagesRef, poseRef, rev, bump, commit, recompute, ghosts, say, status,
    sel, setSel, mode, setMode, animation, setAnimation, time, setTime,
    playing, setPlaying, loop, setLoop, fps, setFps,
    view, setView, showBones, setShowBones, showMesh, setShowMesh,
    showImages, setShowImages, onion, setOnion, grid, setGrid, tool, setTool,
    source, setSource,
    resetDoc, loadWorkfile, adoptDoc, importPsd, importPngs, importSvg,
    previewParts, runAutoRig,
    addBone, removeBone, reparentBone, patchBone,
    addSlot, removeSlot, moveSlot, patchSlot, patchAttachment,
    transformBone,
    addAnimation, removeAnimation, patchAnimation, keyCurrent, allKeys, patchKey, removeKey,
    addEvent, removeEvent, patchEvent, addSkin, removeSkin,
    constraints, addConstraint, removeConstraint, addPose, applyPose,
    buildExports, renderFrames,
  }
}
