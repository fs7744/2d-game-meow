/**
 * 拆部件（与「角色设定」同级的 tab）：选图 → 选拆法 → 出**标准分层 PSD**
 * + PSD 标准的分层预览（层列表 + 叠合画布 + 显隐）。
 *
 * 两种拆法：
 *   · See-Through 语义分层：Live2D 语义层、每层补全遮挡、按深度排序，
 *     要独立环境跑 2~3 分钟（走单卡队列）；
 *   · 爆炸图 · 几何切件：按人形比例切 6 块，纯 PIL、几秒，质量上限低。
 *
 * 产物清单（`PartsManifest`）由后端统一，两种方法在前端长得一模一样：
 * `layers` 从底到顶，`order` = 索引，坐标是画布像素。
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  fetchPart, fetchParts, splitParts,
  type PartsAvailability, type PartsManifest, type PartsSummary,
  type PortraitItem,
} from '../../api'
import type { TaskState } from './model'

type Props = {
  character: string
  /** 当前生命阶段的立绘（拆部件的输入只能从这里选） */
  portraits: PortraitItem[]
  task: TaskState | null
  runTask: (label: string, submit: () => Promise<unknown>,
    onDone?: (job: unknown, result: unknown) => void) => Promise<void>
}

type Method = 'seethrough' | 'explode'

function stamp(sec: number): string {
  if (!sec) return '—'
  const d = new Date(sec * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getMonth() + 1}/${d.getDate()} ${p(d.getHours())}:${p(d.getMinutes())}`
}

export function PartsPanel({ character, portraits, task, runTask }: Props) {
  const [avail, setAvail] = useState<PartsAvailability | null>(null)
  const [items, setItems] = useState<PartsSummary[]>([])
  const [selPath, setSelPath] = useState('')
  const [method, setMethod] = useState<Method>('seethrough')
  const [res, setRes] = useState(1280)
  const [steps, setSteps] = useState(30)
  const [seed, setSeed] = useState(42)
  const [lrsplit, setLrsplit] = useState(false)
  const [overlap, setOverlap] = useState(3)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [mf, setMf] = useState<PartsManifest | null>(null)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [active, setActive] = useState('')

  const openResult = useCallback(async (rid: string) => {
    try {
      const m = await fetchPart(character, rid)
      setMf(m); setHidden(new Set()); setActive('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [character])

  const reload = useCallback(async (openLatest = true) => {
    try {
      const r = await fetchParts(character)
      setAvail(r.available)
      setItems(r.items)
      if (openLatest && r.items[0]) await openResult(r.items[0].id)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }, [character, openResult])

  useEffect(() => { void reload() }, [reload])

  // 默认选第一张立绘；换角色 / 换阶段时重选
  useEffect(() => {
    if (!portraits.some((p) => p.path === selPath)) {
      setSelPath(portraits[0]?.path || '')
    }
  }, [portraits, selPath])

  const stOk = avail?.seethrough.ok ?? false
  const psdOk = avail?.psd.ok ?? false
  const canRun = !!selPath && !busy && (method === 'explode' || stOk)

  const run = () => {
    if (!canRun) return
    setBusy(true); setErr('')
    const params = method === 'seethrough'
      ? { resolution: res, steps, seed, lrsplit }
      : { overlap }
    void runTask('拆部件',
      () => splitParts(character, { path: selPath, method, params }),
      async () => {
        // wparts 的 job.result 就是 manifest，但 fetchJob 只透传有限字段；
        // 统一从磁盘再拉一次列表 + 详情，保证和历史产物同一条路径。
        await reload(true)
      }).catch((e) => {
        setErr(e instanceof Error ? e.message : String(e))
      }).finally(() => setBusy(false))
  }

  const toggle = (name: string) => {
    setHidden((s) => {
      const n = new Set(s)
      if (n.has(name)) n.delete(name); else n.add(name)
      return n
    })
  }

  const canvas = mf?.canvas || [0, 0]
  const [cw, ch] = canvas
  // 层列表按 PSD 面板习惯**顶层在上**（manifest 是从底到顶）
  const layersTopFirst = useMemo(
    () => (mf ? [...mf.layers].slice().reverse() : []), [mf])

  return (
    <div className="cc-parts">
      <div className="cc-parts-head">
        <div>
          <h2>拆部件</h2>
          <p>
            把一张立绘拆成可分层编辑的部件，产出<strong>业界通用分层 PSD</strong>
            （Photoshop / Affinity / Krita 直接打开），每层带 alpha 与原始位置。
          </p>
        </div>
      </div>

      {err && <div className="chars-err">⚠ {err}</div>}

      {/* ---------------- 选图 ---------------- */}
      <section className="cc-parts-sec">
        <h3>1. 选图</h3>
        {portraits.length === 0 ? (
          <p className="cc-parts-note">
            当前阶段还没有立绘：先回「角色设定」生成立绘，再来拆。
          </p>
        ) : (
          <div className="cc-parts-pick">
            {portraits.map((p) => (
              <button type="button" key={p.path}
                className={`cc-parts-pick-item${selPath === p.path ? ' is-on' : ''}`}
                onClick={() => setSelPath(p.path)}
                title={`${p.name}（${p.costume || '默认'} / ${p.emotion || 'neutral'}）`}>
                <img src={p.url} alt={p.name} loading="lazy" />
              </button>
            ))}
          </div>
        )}
      </section>

      {/* ---------------- 选拆法 ---------------- */}
      <section className="cc-parts-sec">
        <h3>2. 选拆法</h3>
        <div className="cc-parts-methods">
          <button type="button"
            className={`cc-parts-method${method === 'seethrough' ? ' is-on' : ''}`}
            disabled={!stOk} onClick={() => setMethod('seethrough')}>
            <b>See-Through 语义分层</b>
            <span>Live2D 语义层（脸 / 上衣 / 手臂 / 头发…），每层补全被遮挡区域，
              按深度排序。约 2~3 分钟。</span>
            {!stOk && <em>{avail?.seethrough.reason || '本机未安装'}</em>}
          </button>
          <button type="button"
            className={`cc-parts-method${method === 'explode' ? ' is-on' : ''}`}
            onClick={() => setMethod('explode')}>
            <b>爆炸图 · 几何切件</b>
            <span>按人形比例切头 / 躯干 / 左右臂 / 左右腿 6 块，几秒出。
              长发大裙摆会裂、手持道具会丢。</span>
          </button>
        </div>
      </section>

      {/* ---------------- 参数 ---------------- */}
      <section className="cc-parts-sec">
        <h3>3. 参数</h3>
        <div className="cc-parts-params">
          {method === 'seethrough' ? (
            <>
              <label className="cc-inline-field">分辨率
                <select value={res} onChange={(e) => setRes(Number(e.target.value))}>
                  <option value={1024}>1024（快）</option>
                  <option value={1280}>1280（推荐）</option>
                  <option value={1536}>1536（慢）</option>
                </select>
              </label>
              <label className="cc-inline-field">步数
                <input type="number" min={10} max={60} value={steps}
                  onChange={(e) => setSteps(Number(e.target.value) || 30)} />
              </label>
              <label className="cc-inline-field">种子
                <input type="number" value={seed}
                  onChange={(e) => setSeed(Number(e.target.value) || 0)} />
              </label>
              <label className="cc-tool-check">
                <input type="checkbox" checked={lrsplit}
                  onChange={(e) => setLrsplit(e.target.checked)} />
                额外拆左右（眼 / 耳 / 手饰 / 头发）
              </label>
            </>
          ) : (
            <label className="cc-inline-field">切区重叠（像素）
              <input type="number" min={0} max={32} value={overlap}
                onChange={(e) => setOverlap(Number(e.target.value) || 0)} />
            </label>
          )}
        </div>
        <div className="cc-parts-actions">
          <button type="button" className="cc-btn primary" disabled={!canRun}
            onClick={run}>{busy ? '拆解中…' : '开始拆'}</button>
          <span className="cc-parts-note">
            {method === 'seethrough'
              ? '跑在 see-through 的独立环境里，会占单卡队列。'
              : '纯图像处理，不跑模型。'}
            {!psdOk && ' PSD 合成不可用，只能出分层 PNG。'}
          </span>
        </div>
        {busy && task && (
          <div className="cc-parts-progress">
            <b>{task.label}</b>
            <span>{task.message}</span>
            {!!task.progress?.total && (
              <div className="cc-parts-bar">
                <i style={{ width: `${Math.min(100,
                  Math.round((task.progress.done / task.progress.total) * 100))}%` }} />
              </div>
            )}
          </div>
        )}
      </section>

      {/* ---------------- 结果：分层预览 ---------------- */}
      {mf && (
        <section className="cc-parts-sec">
          <h3>分层预览
            <span className="cc-parts-meta">
              {mf.method === 'seethrough' ? 'See-Through' : '爆炸图'} ·
              {' '}{mf.layers.length} 层 · {cw}×{ch} · {mf.seconds}s
            </span>
            <span className="spacer" />
            {mf.psd ? (
              <a className="cc-btn primary" href={mf.psd}
                download={mf.psdName || undefined}>下载 PSD</a>
            ) : (
              <span className="cc-parts-note">{mf.notice || '没有生成 PSD'}</span>
            )}
          </h3>

          <div className="cc-parts-view">
            {/* 高度上限 520px，宽度按画布比例反推 —— 只有这样层的百分比定位
                才和画布严格同比例（object-fit 对绝对定位的层没用）。 */}
            <div className="cc-parts-canvas"
              style={{
                aspectRatio: `${cw || 1} / ${ch || 1}`,
                maxWidth: `${Math.round((520 * (cw || 1)) / (ch || 1))}px`,
              }}>
              {mf.layers.map((l) => {
                const off = hidden.has(l.name)
                const dim = !!active && active !== l.name
                return (
                  <img key={l.name} src={l.url} alt={l.name} loading="lazy"
                    className="cc-parts-layer"
                    style={{
                      left: `${(l.x / (cw || 1)) * 100}%`,
                      top: `${(l.y / (ch || 1)) * 100}%`,
                      width: `${(l.width / (cw || 1)) * 100}%`,
                      height: `${(l.height / (ch || 1)) * 100}%`,
                      zIndex: l.order,
                      opacity: off ? 0 : dim ? 0.22 : 1,
                    }} />
                )
              })}
              {mf.layers.every((l) => hidden.has(l.name)) && (
                <span className="cc-parts-empty">所有层都被隐藏了</span>
              )}
            </div>

            <ul className="cc-parts-layers">
              {layersTopFirst.map((l) => (
                <li key={l.name}
                  className={active === l.name ? 'is-on' : ''}
                  onClick={() => setActive((a) => a === l.name ? '' : l.name)}>
                  <button type="button" className="cc-parts-eye"
                    title={hidden.has(l.name) ? '显示这层' : '隐藏这层'}
                    onClick={(e) => { e.stopPropagation(); toggle(l.name) }}>
                    {hidden.has(l.name) ? '◻' : '◼'}
                  </button>
                  <img src={l.url} alt="" loading="lazy" />
                  <span>{l.name}</span>
                  <em>{l.width}×{l.height}</em>
                </li>
              ))}
            </ul>
          </div>
          <p className="cc-parts-note">
            点层可以在画布上单独高亮；◼ / ◻ 控制显隐。PSD 里的层名与顺序和这里一致。
          </p>

          {items.length > 1 && (
            <div className="cc-parts-history">
              <b>历史产物</b>
              {items.map((it) => (
                <button type="button" key={it.id}
                  className={mf.id === it.id ? 'is-on' : ''}
                  onClick={() => void openResult(it.id)}>
                  {stamp(it.created)} · {it.methodLabel} · {it.layers} 层
                </button>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
