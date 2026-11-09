/**
 * 立绘悬浮工具箱的弹窗：扣背景 / 完美像素化 / 删除。
 *
 * 三条硬规矩（用户 2026-11-09 明确）：
 *   1. 两个处理都在弹窗里挑参数，处理完先给**滑动比对**，拖动分割线就能看差异；
 *   2. 处理结果是**新的一份**，原图一个字节都不改；
 *   3. 删除必须确认，避免手抖误删。
 *   4. 像素化产出的是「格子图」，交付时要按 NEAREST 放大回原尺寸，
 *      否则 768×1024 的立绘会变成 48×64，画廊里小得没法看（也可选保留小图）。
 *
 * 「处理并比对」时后端已经把新文件落盘（同目录、不覆盖）；用户若不满意可以
 * 「放弃这份」把它删掉，或「改参数重做」自动丢弃上一份再调。
 */
import { useEffect, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  clothesWizard, deletePortrait, fetchCharacter, fetchEmotionCatalog,
  processPortrait, saveCostume, submitPortraitVariant,
  type PortraitItem, type PortraitOp, type PortraitProcessResult,
  type PortraitVariantKind,
} from '../../api'
import { emotionLabel } from '../../labels'
import { CostumeForm, EmotionGrid, type EmotionOption } from './Forms'
import type { TaskState } from './model'

type CommonProps = {
  character: string
  item: PortraitItem
  onClose: () => void
  onChanged: () => void
}

const MATTE_DEFAULTS = {
  backend: 'auto', tol: 32, fringe: true, holes: true,
}
const PIXEL_DEFAULTS = {
  target: 'auto', colors: 32, grid_snap: true, sample_method: 'median',
  transparent_bg: false, pre_enhance: '0', edge_rebuild: false,
  /** restore = 放大回原图尺寸（立绘交付）；native = 保留格子小图（精灵帧素材） */
  output: 'restore',
}

/** meta 里的尺寸是 [w, h]，展示成 "768 × 1024"；缺失就返回空串。 */
function sizeText(meta: Record<string, unknown> | undefined, key: string): string {
  const v = meta?.[key]
  return Array.isArray(v) && v.length === 2 ? `${v[0]} × ${v[1]}` : ''
}

/** 拖拽分割线：左边原图、右边结果，range 覆盖整个画面，拖动即比对。 */
function CompareSlider({ before, after, pos, onPos }: {
  before: string
  after: string
  pos: number
  onPos: (v: number) => void
}) {
  return (
    <div className="cc-compare">
      <img className="cc-compare-img" src={after} alt="处理后" />
      <div className="cc-compare-clip" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img className="cc-compare-img" src={before} alt="处理前" />
      </div>
      <span className="cc-compare-tag is-before">处理前</span>
      <span className="cc-compare-tag is-after">处理后</span>
      <div className="cc-compare-divider" style={{ left: `${pos}%` }}>
        <span className="cc-compare-knob">↔</span>
      </div>
      <input type="range" className="cc-compare-range" min={0} max={100}
        value={pos} aria-label="拖动比对处理前后"
        onChange={(e) => onPos(Number(e.target.value))} />
    </div>
  )
}

function Field({ label, hint, children }: {
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <label className="cc-tool-field">
      <span>{label}{hint && <i>{hint}</i>}</span>
      {children}
    </label>
  )
}

export function PortraitProcessDialog({ character, item, op, onClose, onChanged }:
  CommonProps & { op: PortraitOp }) {
  const [matte, setMatte] = useState(MATTE_DEFAULTS)
  const [pixel, setPixel] = useState(PIXEL_DEFAULTS)
  const [result, setResult] = useState<PortraitProcessResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [pos, setPos] = useState(50)

  const buildParams = (): Record<string, unknown> => op === 'matte'
    ? { backend: matte.backend, tol: matte.tol,
      fringe: matte.fringe, holes: matte.holes }
    : {
      // 只给「长边」，短边由后端按原图比例推导 —— 以前写死 [N, N] 会把
      // 3:4 的立绘压成正方形，人物变形。
      long_edge: pixel.target === 'auto' ? null : Number(pixel.target),
      colors: pixel.colors, grid_snap: pixel.grid_snap,
      sample_method: pixel.sample_method,
      transparent_bg: pixel.transparent_bg,
      pre_enhance: Number(pixel.pre_enhance),
      edge_rebuild: pixel.edge_rebuild,
      output: pixel.output,
    }

  const run = async () => {
    setBusy(true); setErr('')
    try {
      const r = await processPortrait(character,
        { path: item.path, op, params: buildParams() })
      setResult(r); setPos(50)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally { setBusy(false) }
  }

  /** 丢弃刚生成的那一份（不需要二次确认：它本来就是这次预览的产物）。 */
  const discard = async () => {
    if (!result) return
    try { await deletePortrait(character, result.path) } catch { /* 忽略 */ }
    setResult(null)
  }
  const finish = () => { onChanged(); onClose() }

  return createPortal(
    <div className="cc-tool-mask" onClick={finish}>
      <div className="cc-tool-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="cc-tool-head">
          <b>{op === 'matte' ? '扣背景' : '完美像素化'}</b>
          <span className="cc-tool-file" title={item.name}>{item.name}</span>
          <span className="spacer" />
          <button type="button" className="cc-tool-x" onClick={finish}>✕</button>
        </div>
        {err && <p className="cc-tool-err">⚠ {err}</p>}

        {!result ? (
          <>
            <div className="cc-tool-body">
              <div className="cc-tool-preview">
                <img src={item.url} alt={item.name} />
                <span>原图</span>
              </div>
              <div className="cc-tool-form">
                {op === 'matte' ? (
                  <>
                    <Field label="抠图方式" hint="自动会跟随全局设置">
                      <select value={matte.backend}
                        onChange={(e) => setMatte({ ...matte, backend: e.target.value })}>
                        <option value="auto">自动（推荐）</option>
                        <option value="color">纯色键控（快，适合绿幕）</option>
                        <option value="u2net">AI 语义分割</option>
                      </select>
                    </Field>
                    <Field label="颜色容差" hint={`${matte.tol}`}>
                      <input type="range" min={8} max={120} step={4} value={matte.tol}
                        onChange={(e) => setMatte({ ...matte, tol: Number(e.target.value) })} />
                    </Field>
                    <label className="cc-tool-check">
                      <input type="checkbox" checked={matte.fringe}
                        onChange={(e) => setMatte({ ...matte, fringe: e.target.checked })} />
                      去白边 / 绿边
                    </label>
                    <label className="cc-tool-check">
                      <input type="checkbox" checked={matte.holes}
                        onChange={(e) => setMatte({ ...matte, holes: e.target.checked })} />
                      补回主体内部空洞
                    </label>
                  </>
                ) : (
                  <>
                    <Field label="目标像素尺寸" hint="填的是长边格数，短边按原图比例算">
                      <select value={pixel.target}
                        onChange={(e) => setPixel({ ...pixel, target: e.target.value })}>
                        <option value="auto">自动检测</option>
                        <option value="64">长边 64 格</option>
                        <option value="96">长边 96 格</option>
                        <option value="128">长边 128 格</option>
                        <option value="192">长边 192 格</option>
                      </select>
                    </Field>
                    <Field label="输出尺寸" hint="像素块本身仍是低分辨率格子">
                      <select value={pixel.output}
                        onChange={(e) => setPixel({ ...pixel, output: e.target.value })}>
                        <option value="restore">恢复原图尺寸（推荐）</option>
                        <option value="native">保留像素小图（做精灵帧）</option>
                      </select>
                    </Field>
                    <Field label="色板色数" hint="0 = 只做网格对齐">
                      <select value={pixel.colors}
                        onChange={(e) => setPixel({ ...pixel, colors: Number(e.target.value) })}>
                        <option value={0}>不量化</option>
                        <option value={8}>8 色</option>
                        <option value={16}>16 色</option>
                        <option value={32}>32 色</option>
                        <option value={64}>64 色</option>
                      </select>
                    </Field>
                    <Field label="每格取色">
                      <select value={pixel.sample_method}
                        onChange={(e) => setPixel({ ...pixel, sample_method: e.target.value })}>
                        <option value="median">中位色（最稳）</option>
                        <option value="majority">多数表决</option>
                        <option value="center">中心取点</option>
                      </select>
                    </Field>
                    <Field label="量化前增强" hint="换算素材建议开">
                      <select value={pixel.pre_enhance}
                        onChange={(e) => setPixel({ ...pixel, pre_enhance: e.target.value })}>
                        <option value="0">关闭</option>
                        <option value="0.5">轻度</option>
                        <option value="1">标准</option>
                      </select>
                    </Field>
                    <label className="cc-tool-check">
                      <input type="checkbox" checked={pixel.grid_snap}
                        onChange={(e) => setPixel({ ...pixel, grid_snap: e.target.checked })} />
                      网格吸附（每格一色）
                    </label>
                    <label className="cc-tool-check">
                      <input type="checkbox" checked={pixel.transparent_bg}
                        onChange={(e) => setPixel({ ...pixel, transparent_bg: e.target.checked })} />
                      纯色背景转透明
                    </label>
                    <label className="cc-tool-check">
                      <input type="checkbox" checked={pixel.edge_rebuild}
                        onChange={(e) => setPixel({ ...pixel, edge_rebuild: e.target.checked })} />
                      色块交界补描边
                    </label>
                  </>
                )}
              </div>
            </div>
            <p className="cc-tool-hint">
              处理会生成一份<strong>新立绘</strong>，原图保持不变；处理完可以拖动分割线比对。
            </p>
            <div className="cc-tool-actions">
              <button type="button" className="cc-btn ghost" onClick={onClose}>取消</button>
              <span className="spacer" />
              <button type="button" className="cc-btn primary" disabled={busy}
                onClick={() => void run()}>{busy ? '处理中…' : '处理并比对'}</button>
            </div>
          </>
        ) : (
          <>
            <CompareSlider before={result.before} after={result.after}
              pos={pos} onPos={setPos} />
            <p className="cc-tool-hint">
              左：原图 · 右：处理后的新立绘 <b>{result.name}</b>（已保存为独立一份）
              {op === 'pixel' && (
                <> · 格子 {sizeText(result.meta, 'native_size')}
                  → 交付 {sizeText(result.meta, 'output_size')}
                  {result.meta?.output === 'native' && '（保留小图）'}</>
              )}
            </p>
            <div className="cc-tool-actions">
              <button type="button" className="cc-btn ghost" disabled={busy}
                onClick={() => void discard()}>改参数重做</button>
              <button type="button" className="cc-btn ghost" disabled={busy}
                onClick={() => void discard().then(() => { onChanged(); onClose() })}>
                放弃这份
              </button>
              <span className="spacer" />
              <button type="button" className="cc-btn primary" onClick={finish}>完成</button>
            </div>
          </>
        )}
      </div>
    </div>, document.body)
}


type VariantProps = CommonProps & {
  kind: PortraitVariantKind
  stage: string
  task: TaskState | null
  /** 复用角色中心统一的 runTask 轮询，不另起一套时钟 */
  runTask: (label: string, submit: () => Promise<unknown>,
    onDone?: (job: unknown, result: unknown) => void) => Promise<void>
}

type VariantResult = {
  before: string
  after: string
  path?: string
  name?: string
  /** 批量换表情时这张对应哪个表情（做缩略图切换条） */
  emotion?: string
}

/** 官方六个服装字段（跟待处理页的 CostumeForm 是同一组） */
const EMPTY_CLOTHES: Record<string, string> = {
  top: '', bottom: '', shoes: '', head: '', face: '', negative_prompt: '',
}
/** 表情库加载失败时的兜底（官方 safe_name） */
const FALLBACK_EMOTIONS = ['smile', 'angry', 'sad', 'surprised', 'wink',
  'laugh', 'cry', 'neutral']

/** 换装 / 换表情：跟待处理页那两块**同源**。
 *
 *  · 换装：六字段 + 一句话 AI 拆服装 + 保存服装，后端挂官方
 *    Clothes Core LoRA（`VNCCS_QI2_ClothesCoreV2.6`），25 步 / cfg 3.0；
 *  · 换表情：官方表情库卡片 + 画质档，后端走**脸部裁剪局部重绘后贴回**，
 *    衣服 / 姿势 / 背景逐像素不变。
 *  两者都产出同一个阶段下的新一份立绘，原图不动。 */
export function PortraitVariantDialog({ character, item, kind, stage, task,
  runTask, onClose, onChanged }: VariantProps) {
  const [fields, setFields] = useState<Record<string, string>>({ ...EMPTY_CLOTHES })
  const [clothText, setClothText] = useState('')
  const [costume, setCostume] = useState(item.costume || 'Naked')
  // 表情库支持多选：批量时一个任务里逐个出图，每张单独收编
  const [emoSel, setEmoSel] = useState<string[]>([])
  const [emoOptions, setEmoOptions] = useState<EmotionOption[]>([])
  const [steps, setSteps] = useState(25)
  const [cfg, setCfg] = useState(3)
  const [seed, setSeed] = useState(-1)
  const [loraScale, setLoraScale] = useState(1)
  const [targetSize, setTargetSize] = useState(2048)
  const [busy, setBusy] = useState(false)
  const [wiz, setWiz] = useState(false)
  const [saved, setSaved] = useState('')
  const [results, setResults] = useState<VariantResult[]>([])
  // 多张结果时当前在比对哪一张（单张恒为 0，UI 跟以前一样）
  const [cur, setCur] = useState(0)
  const [pos, setPos] = useState(50)
  const [err, setErr] = useState('')

  // 预填这套服装已有的六个字段（跟待处理页看到的一致）
  useEffect(() => {
    let alive = true
    void fetchCharacter(character).then((d) => {
      if (!alive) return
      const cur = (d.config?.costumes?.[item.costume] || {}) as Record<string, string>
      setFields({ ...EMPTY_CLOTHES, ...cur })
    }).catch(() => { /* 预填失败就空表单 */ })
    return () => { alive = false }
  }, [character, item.costume])

  // 官方表情库（图标 + 中文名）；取不到就用常用 safe_name 兜底
  useEffect(() => {
    if (kind !== 'emotion') return
    let alive = true
    void fetchEmotionCatalog().then((cat) => {
      if (!alive) return
      const out: EmotionOption[] = []
      for (const [, list] of Object.entries(cat)) {
        for (const e of Array.isArray(list) ? list : []) {
          out.push({
            key: e.key, safe_name: e.safe_name,
            label: emotionLabel(e.safe_name, e.key), prompt: e.natural_prompt,
          })
        }
      }
      setEmoOptions(out)
      // 库里不一定有 smile：默认落到第一项，避免「一个都没选中」
      setEmoSel((prev) => {
        const keep = prev.filter((x) => out.some((e) => e.safe_name === x))
        return keep.length ? keep : (out[0] ? [out[0].safe_name] : [])
      })
    }).catch(() => {
      if (!alive) return
      const fb = FALLBACK_EMOTIONS.map((s2) => ({
        key: s2, safe_name: s2, label: s2, prompt: '',
      }))
      setEmoOptions(fb)
      setEmoSel((prev) => {
        const keep = prev.filter((x) => fb.some((e) => e.safe_name === x))
        return keep.length ? keep : [fb[0].safe_name]
      })
    })
    return () => { alive = false }
  }, [kind])

  const label = kind === 'outfit' ? '换装' : '换表情'
  const filled = Object.entries(fields)
    .some(([k, v]) => k !== 'negative_prompt' && !!v.trim())
  const ready = kind === 'outfit' ? filled : emoSel.length > 0

  const doWizard = async () => {
    if (!clothText.trim() || wiz) return
    setWiz(true); setErr('')
    try {
      const r = await clothesWizard(character, clothText.trim())
      const got: Record<string, string> = {}
      for (const k of Object.keys(EMPTY_CLOTHES)) {
        const v = (r.fields as Record<string, unknown>)[k]
        if (typeof v === 'string') got[k] = v
      }
      setFields((f) => ({ ...f, ...got }))
      setSaved('')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally { setWiz(false) }
  }

  const saveFields = async (d: Record<string, string>) => {
    try {
      await saveCostume(character, costume.trim() || item.costume, d)
      setFields(d); setSaved('已保存到这套服装')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    }
  }

  const run = () => {
    if (!ready || busy) return
    setBusy(true); setErr(''); setSaved('')
    const params = kind === 'outfit'
      ? { fields, costume: costume.trim() || item.costume,
          emotion_tag: item.emotion || 'neutral',
          steps, cfg, seed, lora_scale: loraScale }
      : { emotions: emoSel, emotion_name: emoSel[0],
          costume: item.costume, emotion_tag: emoSel[0] || 'neutral',
          steps, cfg, seed, target_size: targetSize }
    void runTask(label,
      () => submitPortraitVariant(character,
        { path: item.path, kind, params, stage }),
      (job) => {
        const j = job as {
          url?: string
          published?: { path?: string; file?: string }
          result?: {
            items?: Array<{ emotion?: string; url?: string
              path?: string; file?: string }>
            count?: number
          }
        }
        const items = j.result?.items || []
        if (items.length) {
          // 批量：每个表情一条，弹窗里用缩略图切换逐张比对
          setResults(items.map((it) => ({
            before: item.url, after: it.url || item.url,
            path: it.path, name: it.file || it.emotion, emotion: it.emotion,
          })))
          setCur(0)
        } else {
          setResults([{ before: item.url, after: j.url || item.url,
            path: j.published?.path, name: j.published?.file }])
          setCur(0)
        }
        onChanged()
      }).catch((e) => {
        setErr(e instanceof Error ? e.message : String(e))
      }).finally(() => setBusy(false))
  }

  const discard = async () => {
    // 批量时把这一整批都撤掉（都是刚生成的）；逐张保留可以去画廊里单独删
    for (const r of results) {
      if (!r.path) continue
      try { await deletePortrait(character, r.path) } catch { /* 忽略 */ }
    }
    onChanged(); onClose()
  }

  return createPortal(
    <div className="cc-tool-mask" onClick={onClose}>
      <div className="cc-tool-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="cc-tool-head">
          <b>{label}</b>
          <span className="cc-tool-file" title={item.name}>{item.name}</span>
          <span className="spacer" />
          <button type="button" className="cc-tool-x" onClick={onClose}>✕</button>
        </div>
        {err && <p className="cc-tool-err">⚠ {err}</p>}

        {results.length === 0 ? (
          <>
            {kind === 'outfit' ? (
              <div className="cc-variant-body">
                <Field label="一句话描述" hint="AI 拆成下面六个字段，先给你看">
                  <div className="cc-variant-row">
                    <input value={clothText}
                      placeholder="例：赛博朋克护士，白大褂 + 护目镜"
                      onChange={(e) => setClothText(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') void doWizard() }} />
                    <button type="button" className="cc-btn ghost"
                      disabled={wiz || !clothText.trim()}
                      onClick={() => void doWizard()}>
                      {wiz ? '拆解中…' : 'AI 拆服装'}
                    </button>
                  </div>
                </Field>
                <CostumeForm fields={fields} onChange={setFields}
                  onSave={(d) => void saveFields(d)} />
                {saved && <p className="cc-parts-note">{saved}</p>}
                <Field label="归入服装" hint="留空 = 沿用当前服装">
                  <input value={costume} onChange={(e) => setCostume(e.target.value)} />
                </Field>
              </div>
            ) : (
              <div className="cc-variant-body">
                <div className="cc-variant-sel">
                  已选 <b>{emoSel.length}</b> 个表情
                  {emoSel.length > 1 && '（一个任务里逐个出图，各自收进对应表情目录）'}
                  {emoSel.length === 0 && '（点头像选，可多选）'}
                </div>
                <EmotionGrid options={emoOptions} selected={emoSel}
                  onToggle={(name) => setEmoSel((prev) => prev.includes(name)
                    ? prev.filter((x) => x !== name)
                    : [...prev, name])} />
                {emoSel.length > 0 && (
                  <div className="cc-variant-picked">
                    {emoSel.map((name) => (
                      <button type="button" key={name} className="cc-variant-tag"
                        title={`点一下取消「${name}」`}
                        onClick={() => setEmoSel((prev) => prev.filter((x) => x !== name))}>
                        {emoOptions.find((e) => e.safe_name === name)?.label || name} ✕
                      </button>
                    ))}
                  </div>
                )}
                <Field label="画质档" hint="官方 target_size（2.0MP = 2048）">
                  <select value={targetSize}
                    onChange={(e) => setTargetSize(Number(e.target.value))}>
                    <option value={2048}>标准（2048）</option>
                    <option value={1024}>快速（1024，约快一倍）</option>
                  </select>
                </Field>
              </div>
            )}

            <div className="cc-variant-nums">
              <Field label="步数">
                <input type="number" min={8} max={60} value={steps}
                  onChange={(e) => setSteps(Number(e.target.value) || 25)} />
              </Field>
              <Field label="CFG">
                <input type="number" min={0.5} max={8} step={0.5} value={cfg}
                  onChange={(e) => setCfg(Number(e.target.value) || 3)} />
              </Field>
              <Field label="种子" hint="-1 随机">
                <input type="number" value={seed}
                  onChange={(e) => setSeed(Number(e.target.value) || -1)} />
              </Field>
              {kind === 'outfit' && (
                <Field label="LoRA 强度" hint="Clothes Core">
                  <input type="number" min={0} max={2} step={0.1} value={loraScale}
                    onChange={(e) => setLoraScale(Number(e.target.value))} />
                </Field>
              )}
            </div>

            {busy && (
              <p className="cc-tool-progress">
                生成中… {task?.message || '单卡队列排队中'}
              </p>
            )}
            {!busy && task?.status === 'failed' && (
              <p className="cc-tool-err">⚠ {task.message}</p>
            )}
            <p className="cc-tool-hint">
              {kind === 'outfit'
                ? <>挂官方 Clothes Core LoRA 生成<strong>一张新立绘</strong>
                  （原图不变），自动收进当前阶段。</>
                : <>只重画脸框内：衣服 / 姿势 / 背景逐像素不变；选几个表情
                  就生成<strong>几张新立绘</strong>，原图不变。</>}
            </p>
            <div className="cc-tool-actions">
              <button type="button" className="cc-btn ghost" onClick={onClose}>取消</button>
              <span className="spacer" />
              <button type="button" className="cc-btn primary" disabled={busy || !ready}
                onClick={run}>{busy ? '生成中…' : '生成并比对'}</button>
            </div>
          </>
        ) : (
          <>
            <CompareSlider before={results[cur].before} after={results[cur].after}
              pos={pos} onPos={setPos} />
            {results.length > 1 && (
              <div className="cc-variant-strip">
                {results.map((r, i) => (
                  <button type="button" key={`${r.emotion || ''}${i}`}
                    className={`cc-variant-thumb${i === cur ? ' is-on' : ''}`}
                    title={r.emotion || r.name}
                    onClick={() => { setCur(i); setPos(50) }}>
                    <img src={r.after} alt={r.emotion || r.name} loading="lazy" />
                    <span>{r.emotion || `第 ${i + 1} 张`}</span>
                  </button>
                ))}
              </div>
            )}
            <p className="cc-tool-hint">
              左：原图 · 右：新生成的立绘{' '}
              <b>{results[cur].emotion || results[cur].name || '（已收进角色库）'}</b>
              {results.length > 1 && `（共 ${results.length} 张，上面切换逐张看）`}
            </p>
            <div className="cc-tool-actions">
              <button type="button" className="cc-btn ghost"
                onClick={() => void discard()}>
                {results.length > 1 ? `放弃这 ${results.length} 张` : '放弃这份'}
              </button>
              <span className="spacer" />
              <button type="button" className="cc-btn primary" onClick={onClose}>完成</button>
            </div>
          </>
        )}
      </div>
    </div>, document.body)
}

export function PortraitDeleteDialog({ character, item, onClose, onChanged }: CommonProps) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const doDelete = async () => {
    setBusy(true); setErr('')
    try {
      await deletePortrait(character, item.path, true)
      onChanged(); onClose()
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally { setBusy(false) }
  }
  return createPortal(
    <div className="cc-tool-mask" onClick={onClose}>
      <div className="cc-tool-dialog is-small" onClick={(e) => e.stopPropagation()}>
        <div className="cc-tool-head">
          <b>删除立绘</b>
          <span className="spacer" />
          <button type="button" className="cc-tool-x" onClick={onClose}>✕</button>
        </div>
        <div className="cc-tool-delbody">
          <img src={item.url} alt={item.name} />
          <div>
            <p className="cc-tool-warn">确定删除「{item.name}」吗？</p>
            <p className="cc-tool-sub">这个操作不能撤销。原图与这份立绘是独立文件，删掉不会影响别的阶段。</p>
          </div>
        </div>
        {err && <p className="cc-tool-err">⚠ {err}</p>}
        <div className="cc-tool-actions">
          <button type="button" className="cc-btn ghost" onClick={onClose}>取消</button>
          <span className="spacer" />
          <button type="button" className="cc-btn danger" disabled={busy}
            onClick={() => void doDelete()}>{busy ? '删除中…' : '确认删除'}</button>
        </div>
      </div>
    </div>, document.body)
}
