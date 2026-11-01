/**
 * 精灵帧翻看器（移植自官方 `web/vnccs_common.js:739` `createSpritePreviewNavigator`）。
 *
 * 官方那套的关键点，每一条都是踩出来的：
 *
 * 1. **先看 meta 拿 count，再按 index 单张取**：一次把所有帧塞进 JSON 的话，
 *    精灵帧一多（几十上百张）首屏就卡死。
 * 2. **requestId 竞态保护**：快速换服装/表情时，晚到的旧请求不能覆盖新的。
 *    没有它就会出现"明明切了服装，显示的还是上一套的图"。
 * 3. **相邻预取** `index±1`：翻页手感全靠它。
 * 4. **三态齐全**：加载中 / 有图 / 没有素材。缺 material 时必须明说，
 *    不能留一块空白让用户以为是 bug。
 * 5. `index % count` 取模 → 翻到头绕回第一张，不会走到"第 0 张之前"。
 *
 * 只做**看**：后续要在节点画布里引用某一帧时再加工具条。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { base } from '../api'

type Props = {
  character: string
  /** 当前服装（可选：不给就退到 Naked/Original 的帧） */
  costume?: string
  /** 当前表情（可选：不给就看 neutral） */
  emotion?: string
  /**
   * 把 meta 汇报给上层（官方 navigator 的 `onLoaded` / `onMissing` 同理）。
   * **不让上层自己再拉一次 meta**——同一个 URL 打两遍，一边的缓存还破了另一边。
   */
  onMeta?: (m: { count: number; files: string[] }) => void
}

export default function SpritePreview({ character, costume, emotion, onMeta }: Props) {
  const [count, setCount] = useState(0)
  const [index, setIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  /** 三种状态：`idle`（还没问）/`empty`（确认真没有）/`ready` */
  const [state, setState] = useState<'idle' | 'empty' | 'ready'>('idle')
  const [error, setError] = useState('')

  // 竞态守卫：**每次换目标都让在途请求作废**（官方 `invalidate()`）
  const requestRef = useRef(0)

  const query = useCallback((q: Record<string, string | number>) => {
    const ps = new URLSearchParams()
    for (const [k, v] of Object.entries(q)) ps.set(k, String(v))
    // `t` 破 Cache-Control 之外还破浏览器磁盘缓存：同名文件被替换后，
    // 没有它浏览器会拿旧的那张（用户的现象是"重新生成了但图没变"）
    ps.set('t', String(Date.now()))
    return `${base}/characters/${encodeURIComponent(character)}`
      + `/poses?${ps.toString()}`
  }, [character, base])

  /** 问后端这个序列有多少帧 */
  const loadMeta = useCallback(async () => {
    const rid = ++requestRef.current
    setLoading(true)
    setError('')
    try {
      const ps = new URLSearchParams()
      if (costume) ps.set('costume', costume)
      if (emotion) ps.set('emotion', emotion)
      ps.set('t', String(Date.now()))
      const r = await fetch(
        `${base}/characters/${encodeURIComponent(character)}/poses/meta?${ps.toString()}`)
      const meta = (await r.json()) as { count?: number; files?: string[] }
      if (rid !== requestRef.current) return          // 晚到的旧请求，丢掉
      const n = Number(meta.count || 0)
      setCount(n)
      onMeta?.({ count: n, files: meta.files || [] })
      if (n <= 0) {
        setState('empty')
        setLoading(false)
      } else {
        setState('ready')
        setIndex(0)
      }
    } catch (e) {
      if (rid !== requestRef.current) return
      setError(e instanceof Error ? e.message : String(e))
      setLoading(false)
    }
  }, [base, character, costume, emotion, onMeta])

  useEffect(() => {
    void loadMeta()
  }, [loadMeta])

  /** 相邻帧预取：翻页手感全靠这一步 */
  const prefetch = useCallback((i: number) => {
    if (count <= 1) return
    const n = ((i % count) + count) % count
    const img = new Image()
    img.src = query({ index: n, ...(costume ? { costume } : {}), ...(emotion ? { emotion } : {}) })
  }, [count, query, costume, emotion])

  const go = useCallback((delta: number) => {
    if (count <= 0) return
    const next = ((index + delta) % count + count) % count
    setIndex(next)
    prefetch(next + delta)
  }, [count, index, prefetch])

  const src = useMemo(() => {
    if (state !== 'ready' || count <= 0) return ''
    return query({
      index,
      ...(costume ? { costume } : {}),
      ...(emotion ? { emotion } : {}),
    })
  }, [state, count, index, query, costume, emotion])

  return (
    <div className="sprite-preview">
      <div className="sprite-stage">
        {state === 'empty' ? (
          <p className="sprite-missing">
            这个角色还没有精灵帧素材。
            <br />
            去「出图」页生成一张，收编后就能在这儿翻。
          </p>
        ) : state === 'idle' || !src ? (
          <p className="sprite-missing">读取素材…</p>
        ) : (
          <img
            key={src}
            src={src}
            alt={`${character} 第 ${index + 1} 帧`}
            onLoad={() => setLoading(false)}
            onError={() => { setLoading(false); setError('这一帧读不出来（文件可能已被删掉）') }}
          />
        )}
        {loading && <span className="sprite-loading">…</span>}
      </div>

      {/* 翻页：**必须显式 told 用户还有没有第二张**——count<=1 时不显示
          导航条（官方 `updateNav`：`count > 1` 才可见），否则用户点了没反应。 */}
      {count > 1 && (
        <div className="sprite-nav">
          <button type="button" onClick={() => go(-1)} title="上一帧">‹</button>
          <span className="sprite-count">{index + 1}/{count}</span>
          <button type="button" onClick={() => go(1)} title="下一帧">›</button>
        </div>
      )}
      {count === 1 && (
        <p className="chars-sub">当前只有这一帧（{costume || 'Naked'}/{emotion || 'neutral'}）。</p>
      )}
      {error && <p className="sprite-err">{error}</p>}
    </div>
  )
}
