/**
 * 角色中心 · 导出前试玩验收台。
 *
 * 借鉴 ai_game_workbench 的「角色预览」页（导航组「预览与导出」第一项，
 * SpriteAnimator.tsx:3972-4379），但按我们自己的资产形态重做：
 *
 *  - **三种视角模式是一等参数**：俯视四方向 / 俯视八方向 / 横板左右，
 *    而不是把四方向写死。八方向用双键向量合成；横板有地面、跳跃、镜像。
 *  - **单一 rAF 时钟**：每推进一帧就走 `speed / fps`，位移与帧天然联动，
 *    不重演对方「setInterval 播放 + 另一个 rAF 位移」导致的滑步。
 *  - **pivot 从 atlas.json 读**，渲染时作为 transform-origin；对方完全没有
 *    pivot，只是 CSS 居中。
 *  - **缺资源如实报告**：角色目录按方向落帧（``sprite_<方向>_NNNN.png``）
 *    后，验收台按当前朝向选对应帧组；没补出来的方向退回同一组帧占位预览，
 *    并在 HUD 标「占位」，绝不让用户以为素材已经能进游戏。
 *  - **验收写回导出**：调好的 fps 可以写回图集导出参数，不用在另一个卡片
 *    里再填一遍（对方项目三套 fps 互不相通，这里只有一个来源）。
 *
 * 仍未做（别假装做完）：动作模板 registry、碰撞盒、方向帧的自动补生成。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchPlaytestAssets, type PlaytestAssets } from '../api'
import {
  VIEW_MODES, DIRECTION_LABELS, VECTORS,
  directionFromKeys, playbackMs, movePerFrame, nextFrame,
  isTextEntryTarget, clampPosition, jumpOffset, type ViewMode,
} from '../store/playtest'

type Props = {
  character: string
  /** 把验收调好的 fps 写回图集导出参数（角色中心持有导出状态） */
  onApplyFps?: (fps: number) => void
}

const JUMP_MS = 650

export default function CharacterPlaytest({ character, onApplyFps }: Props) {
  const [data, setData] = useState<PlaytestAssets | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')
  const [mode, setMode] = useState<ViewMode>('side-2')
  const [clipId, setClipId] = useState('')
  const [playing, setPlaying] = useState(true)
  const [frameIndex, setFrameIndex] = useState(0)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [air, setAir] = useState(0)
  const [direction, setDirection] = useState<string | null>(null)
  const [facing, setFacing] = useState('right')
  const [action, setAction] = useState<string | null>(null)
  const [fps, setFps] = useState(8)
  const [speed, setSpeed] = useState(120)
  const [size, setSize] = useState(168)
  const [background, setBackground] = useState<'grid' | 'plain' | 'scene'>('grid')
  const [placeholder, setPlaceholder] = useState(true)
  const [appliedFps, setAppliedFps] = useState<number | null>(null)

  const keysRef = useRef(new Set<string>())
  const dirRef = useRef<string | null>(null)
  const frameRef = useRef(0)
  const oneShotRef = useRef(false)
  const jumpStartRef = useRef<number | null>(null)

  useEffect(() => {
    let alive = true
    setLoading(true)
    setErr('')
    fetchPlaytestAssets(character)
      .then((raw) => {
        if (!alive) return
        // 后端字段缺失 / 老版本 / jsdom mock 只回一个空对象时，也要能把壳渲染出来。
        // 以前直接 `data.clips.find(...)`，clips 缺失就会整棵 React 树崩掉。
        const d = (raw ?? {}) as Partial<PlaytestAssets>
        const clips = Array.isArray(d.clips) ? d.clips : []
        const safe: PlaytestAssets = {
          character: d.character ?? character,
          clips,
          clipCount: d.clipCount ?? clips.length,
          frameCount: d.frameCount ?? clips.reduce((n, c) => n + (c.frames?.length ?? 0), 0),
          atlas: {
            exists: false, frameRate: 8, loop: true,
            pivot: { x: 0.5, y: 1.0 }, canvas: [0, 0], source: 'default',
            ...(d.atlas ?? {}),
          },
          directions: { present: [], convention: 'none', ...(d.directions ?? {}) },
          viewModes: d.viewModes ?? {},
        }
        setData(safe)
        setFps(safe.atlas.frameRate)
        setClipId(clips[0]?.id ?? '')
        frameRef.current = 0
        setFrameIndex(0)
      })
      .catch((e) => { if (alive) setErr(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [character])

  const clip = useMemo(
    () => data?.clips?.find((c) => c.id === clipId) ?? data?.clips?.[0] ?? null,
    [data, clipId],
  )
  /** 方向 key → 该方向的帧；老目录 / 老 mock 没有这个字段时为空对象。 */
  const byDir = useMemo(() => clip?.directionFrames ?? {}, [clip])
  const hasDirs = Object.keys(byDir).length > 0
  const activeDirection = useMemo(() => {
    if (!hasDirs) return ''
    // 横板松开方向键后保持最后的朝向，别一下跳回默认帧组
    if (mode === 'side-2') return direction || facing
    return direction || clip?.defaultDirection || Object.keys(byDir)[0] || ''
  }, [hasDirs, mode, direction, facing, clip?.defaultDirection, byDir])
  /** 八个方向的帧绝不能串成一条动画：只播当前朝向那一组。 */
  const frames = useMemo(() => {
    if (hasDirs && byDir[activeDirection]?.length) return byDir[activeDirection]
    return clip?.frames ?? []
  }, [hasDirs, byDir, activeDirection, clip])
  const usingRealDirection = hasDirs && !!byDir[activeDirection]?.length
  const readiness = data?.viewModes[mode]
  const canPlay = frames.length > 0 && (!!readiness?.ready || placeholder)

  // 换方向 / 换帧组时从头播，否则会从上一组的中段突然跳过去
  useEffect(() => {
    frameRef.current = 0
    setFrameIndex(0)
  }, [activeDirection, clipId])

  const reset = useCallback(() => {
    setPosition({ x: 0, y: 0 })
    setAir(0)
    jumpStartRef.current = null
    frameRef.current = 0
    setFrameIndex(0)
    keysRef.current.clear()
    dirRef.current = null
    setDirection(null)
  }, [])

  /** 单次动作：从第一帧播一遍，方向锁定到当前朝向，播完回到循环。 */
  const fireOneShot = useCallback((kind: 'jump' | 'attack') => {
    if (!frames.length) return
    oneShotRef.current = true
    setAction(kind)
    frameRef.current = 0
    setFrameIndex(0)
    setPlaying(true)
    if (kind === 'jump' && mode === 'side-2') jumpStartRef.current = performance.now()
  }, [frames.length, mode])

  /** 键盘层：输入框内不抢键；失焦清空，防止「人一直往左走」的卡键。 */
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (isTextEntryTarget(e.target)) return
      const k = e.key.toLowerCase()
      if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) {
        keysRef.current.add(k)
        e.preventDefault()
      } else if (k === 'shift') {
        keysRef.current.add('shift')
      } else if (k === ' ' || e.code === 'Space') {
        e.preventDefault()
        fireOneShot('jump')
        return
      } else if (k === 'j') {
        fireOneShot('attack')
        return
      } else {
        return
      }
      // 单次动作（跳 / 攻击）期间锁向：对方项目也是这么做的，
      // 否则一按斜向键，跳到半空人物突然换朝向。
      if (!oneShotRef.current) {
        const dir = directionFromKeys(keysRef.current, mode)
        dirRef.current = dir
        setDirection(dir)
        if (dir) setFacing(dir.includes('left') ? 'left' : dir.includes('right') ? 'right' : facing)
      }
    }
    const up = (e: KeyboardEvent) => {
      keysRef.current.delete(e.key.toLowerCase())
      if (!oneShotRef.current) {
        const dir = directionFromKeys(keysRef.current, mode)
        dirRef.current = dir
        setDirection(dir)
      }
    }
    const blur = () => {
      keysRef.current.clear()
      dirRef.current = null
      setDirection(null)
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
    }
  }, [mode, fireOneShot, facing])

  /** 单一时钟：帧推进、位移、跳跃抛物线全在同一个 rAF 里。 */
  useEffect(() => {
    if (!playing || !canPlay) return
    let raf = 0
    let last = performance.now()
    let acc = 0
    const ms = playbackMs(fps)
    const tick = (now: number) => {
      const dt = Math.min(50, Math.max(0, now - last))
      last = now
      acc += dt
      const loop = !oneShotRef.current && (clip?.loop ?? true)
      while (acc >= ms) {
        acc -= ms
        const next = nextFrame(frameRef.current, frames.length, loop)
        if (next === null) {
          oneShotRef.current = false
          setAction(null)
          frameRef.current = 0
          setFrameIndex(0)
          dirRef.current = directionFromKeys(keysRef.current, mode)
          setDirection(dirRef.current)
          break
        }
        frameRef.current = next
        setFrameIndex(next)
        const dir = dirRef.current
        if (dir) {
          const boost = keysRef.current.has('shift') ? 1.6 : 1
          const step = movePerFrame(speed * boost, fps)
          const [vx, vy] = VECTORS[dir]
          setPosition((p) => {
            const nx = p.x + vx * step
            const ny = mode === 'side-2' ? 0 : p.y + vy * step
            const [cx, cy] = clampPosition(nx, ny, mode)
            return { x: cx, y: cy }
          })
        }
      }
      if (jumpStartRef.current !== null) {
        const t = (now - jumpStartRef.current) / JUMP_MS
        if (t >= 1) { jumpStartRef.current = null; setAir(0) }
        else setAir(jumpOffset(t))
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, canPlay, fps, speed, frames, mode, clip?.loop])

  if (loading) return <div className="pt-empty">正在读角色素材…</div>
  if (err) return <div className="pt-empty">试玩资产读取失败：{err}</div>
  if (!data) return null

  const pivot = data.atlas.pivot
  // 有真实方向帧时按方向选图；只有左向、右向要派生时才做水平镜像。
  const mirror = mode === 'side-2'
    ? (hasDirs
      ? (activeDirection === 'right' && !byDir.right?.length && !!byDir.left?.length ? -1 : 1)
      : (facing === 'left' ? -1 : 1))
    : 1
  const frame = frames.length ? frames[frameIndex % frames.length] : ''
  const keysHint = mode === 'side-2'
    ? 'A / D 移动 · 空格跳跃 · J 攻击 · Shift 加速'
    : mode === 'topdown-8'
      ? 'WASD 移动 · 双键组合出斜向（W+D = 右上）· J 攻击 · Shift 加速'
      : 'WASD 移动 · J 攻击 · Shift 加速'

  return (
    <div className="pt-root">
      <div className="pt-toolbar">
        <div className="pt-modes">
          {VIEW_MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              className={`pt-mode${mode === m.id ? ' on' : ''}`}
              title={m.hint}
              onClick={() => { setMode(m.id); reset() }}
            >
              {m.label}
            </button>
          ))}
        </div>
        <label className="pt-field">帧组
          <select value={clipId} onChange={(e) => {
            setClipId(e.target.value)
            frameRef.current = 0
            setFrameIndex(0)
          }}>
            {data.clips.length === 0 && <option value="">没有精灵帧</option>}
            {data.clips.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}（{c.totalFrames ?? c.frames.length} 帧
                {!!c.directions?.length && ` · ${c.directions.length} 向`}）
              </option>
            ))}
          </select>
        </label>
        <label className="pt-field">FPS
          <input type="number" min={1} max={300} value={fps}
            onChange={(e) => setFps(Number(e.target.value) || 1)} />
        </label>
        <label className="pt-field">速度
          <input type="number" min={0} max={360} value={speed}
            onChange={(e) => setSpeed(Number(e.target.value) || 0)} />
        </label>
        <label className="pt-field">尺寸
          <input type="range" min={96} max={320} value={size}
            onChange={(e) => setSize(Number(e.target.value))} />
        </label>
        <label className="pt-field">背景
          <select value={background} onChange={(e) => setBackground(e.target.value as typeof background)}>
            <option value="grid">棋盘 / 网格</option>
            <option value="plain">中性灰</option>
            <option value="scene">游戏场景色</option>
          </select>
        </label>
        <button type="button" className="pt-btn" onClick={() => setPlaying((v) => !v)}>
          {playing ? '暂停' : '播放'}
        </button>
        <button type="button" className="pt-btn" onClick={reset}>回中心</button>
      </div>

      {readiness && !readiness.ready && (
        <div className="pt-warning">
          <div>
            <b>还缺 {readiness.missing.length} 个方向</b>
            {readiness.missing.length > 0 && (
              <span>：{readiness.missing.map((d) => DIRECTION_LABELS[d] ?? d).join('、')}</span>
            )}
            {readiness.mirrored.length > 0 && (
              <span> · {readiness.mirrored.map((d) => DIRECTION_LABELS[d] ?? d).join('、')} 由镜像派生</span>
            )}
            <p>{readiness.note}</p>
            <p className="pt-warning-detail">
              该模式需要的白模机位：
              {Object.entries(readiness.yawMap ?? {})
                .map(([d, yaw]) => `${DIRECTION_LABELS[d] ?? d} ${yaw}°`).join(' · ')}
            </p>
          </div>
          <label className="pt-placeholder">
            <input type="checkbox" checked={placeholder}
              onChange={(e) => setPlaceholder(e.target.checked)} />
            用现有帧占位预览
          </label>
        </div>
      )}

      <div className={`pt-stage pt-bg-${background} pt-mode-${mode}`}>
        {mode === 'side-2' && <div className="pt-ground" />}
        {canPlay && frame ? (
          <img
            className="pt-avatar"
            src={frame}
            alt={`${clip?.label ?? ''} 第 ${frameIndex + 1} 帧`}
            draggable={false}
            style={{
              width: size,
              transform: `translate(calc(-50% + ${position.x}px), calc(${mode === 'side-2' ? '-100%' : '-50%'} + ${position.y + air}px)) scaleX(${mirror})`,
              transformOrigin: `${pivot.x * 100}% ${pivot.y * 100}%`,
            }}
          />
        ) : (
          <div className="pt-empty">
            {frames.length === 0
              ? '这个角色还没有精灵帧。先回到步骤 ③ 出素材。'
              : '该模式的方向帧未齐，且未开启占位预览。'}
          </div>
        )}
        <div className="pt-hud">
          <span>{VIEW_MODES.find((m) => m.id === mode)?.label}</span>
          <span>面朝 {direction ? (DIRECTION_LABELS[direction] ?? direction) : '—'}</span>
          <span>帧 {frames.length ? frameIndex + 1 : 0} / {frames.length}</span>
          <span>{action === 'jump' ? '跳跃' : action === 'attack' ? '攻击' : clip?.label ?? '—'}</span>
          <span>{hasDirs
            ? `帧组 ${DIRECTION_LABELS[activeDirection] ?? activeDirection}${usingRealDirection ? '' : ' · 占位'}`
            : '单一帧组'}</span>
          <span>{fps} fps</span>
        </div>
        <div className="pt-keys">{keysHint}</div>
      </div>

      <div className="pt-report">
        <h4>导出前验收清单</h4>
        <div className="pt-report-grid">
          {VIEW_MODES.map((m) => {
            const r = data.viewModes[m.id]
            return (
              <div key={m.id} className={`pt-report-row${r?.ready ? ' ok' : ''}`}>
                <b>{m.label}</b>
                <span>
                  {r?.ready
                    ? '方向齐全'
                    : `缺 ${r?.missing.length ?? 0} 向${r?.mirrored.length ? `（${r.mirrored.length} 向可镜像）` : ''}`}
                </span>
                <span className="pt-report-yaw">
                  {Object.keys(r?.yawMap ?? {}).length} 个白模机位
                </span>
              </div>
            )
          })}
        </div>
        {!!data.directions.templates?.length && (
          <div className="pt-templates">
            <h4>关联动作模板</h4>
            {data.directions.templates.map((t) => (
              <div key={t.id} className="pt-template-row">
                <b>{t.name}</b>
                <span>
                  {t.directions?.length ?? 0} 向 · {t.actionCount} 动作 · {t.fps} fps · {t.loop ? '循环' : '单次'}
                </span>
                <span className="pt-template-cost">
                  估算 {t.estimate?.images ?? '—'} 张 / 约 {t.estimate?.totalMinutes ?? '—'} 分钟
                  {!!t.estimate?.mirroredDirections?.length
                    && `（${t.estimate.mirroredDirections.join('、')} 可镜像派生）`}
                </span>
                <span className="pt-template-src">{t.project}/{t.poseSet}</span>
              </div>
            ))}
          </div>
        )}
        <p className="pt-source">
          fps / loop / pivot 来源：
          {data.atlas.source === 'atlas'
            ? 'atlas.json（已导出图集，轴心在底边中点）'
            : '默认值（还没导出图集，这里的 fps=8 / pivot=(0.5, 1) 不是验收值）'}
          {' · '}现有方向资产：{data.directions.present.length
            ? data.directions.present.join('、') : '无'}
          {!!data.directions.poseSets?.length && (
            <span>
              {' '}（来自 {data.directions.poseSets.length} 组画布姿势集：
              {data.directions.poseSets.slice(0, 2)
                .map((s) => `${s.name} ${s.directions.length} 向`).join('、')}）
            </span>
          )}
        </p>
        <p className="pt-note">
          方向帧由 3D 白模按 YAWS 机位 × 动作模板补出，验收台按朝向选帧组；
          缺的方向用同一组帧占位并标出来。官方 VNCCS 没有原生八方向功能，
          这里的八机位是我们自己转相机，机位与方向名的对应首次真机出帧时要校准一次。
          动作模板 registry、碰撞盒仍未做。
        </p>
        <div className="pt-apply">
          <button
            type="button" className="pt-btn"
            disabled={!onApplyFps}
            onClick={() => { onApplyFps?.(fps); setAppliedFps(fps) }}
          >
            把试玩参数写入导出（{fps} fps）
          </button>
          {appliedFps !== null && (
            <span className="pt-applied">
              已写入：点下方「导出图集」将用 {appliedFps} fps
              {appliedFps !== fps && '（试玩台已改动，再点一次同步）'}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
