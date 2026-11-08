/**
 * 资料库 · 动作模板管理（第三个 tab）。
 *
 * **为什么单独一个组件**：模板是「会持续变多的用户资产」，跟提示词 / 姿势的
 * 浏览逻辑不同——它要能增删改、要显示逐帧成本、还要区分内置只读与"我的"。
 * 面板只负责编辑与展示，**校验和归一化的权威在后端**
 * `workflows/motiontemplates.normalize()`，这里不复制第二套规则。
 *
 * 表单只暴露用户真正要改的字段：名字 / 视角模式 / 动作表 / 预算与提示词契约。
 * `directions` / `yawMap` / `camera` / `mirrorFrom` / `pivot` 一律留空，
 * 由后端按 viewMode 推导——手填反而容易和抽帧机位对不上。
 *
 * 内置三套不可覆盖、不可删除；想改就「另存为我的」生成一份副本。
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  fetchMotionTemplates, saveMotionTemplate, updateMotionTemplate,
  deleteMotionTemplate,
  type MotionEstimate, type MotionTemplate,
} from '../api'
import { VIEW_MODES } from '../store/playtest'

type Props = {
  /** 抽屉顶部共用的搜索词（名字 / id 都参与匹配） */
  query: string
}

type DraftAction = { name: string; frames: number; fps: number; oneShot: boolean }

type Draft = {
  id: string
  name: string
  viewMode: string
  actions: DraftAction[]
  note: string
  qwenPromptContract: string
  subjects: number
  candidates: number
  singleFrameSeconds: number
  passRate: number
}

function emptyDraft(): Draft {
  return {
    id: '', name: '', viewMode: 'topdown-4',
    actions: [{ name: 'idle', frames: 4, fps: 8, oneShot: false }],
    note: '', qwenPromptContract: '', subjects: 1,
    candidates: 1, singleFrameSeconds: 26, passRate: 0.75,
  }
}

function draftOf(t: MotionTemplate, copy: boolean): Draft {
  return {
    id: copy ? '' : t.id,
    name: copy ? `${t.name}（我的）` : t.name,
    viewMode: t.viewMode,
    actions: t.actions.map((a) => ({
      name: a.name, frames: a.frames, fps: a.fps, oneShot: a.oneShot,
    })),
    note: t.note ?? '',
    qwenPromptContract: t.qwenPromptContract ?? '',
    subjects: t.subjects ?? 1,
    candidates: t.budget?.candidates ?? 1,
    singleFrameSeconds: t.budget?.singleFrameSeconds ?? 26,
    passRate: t.budget?.passRate ?? 0.75,
  }
}

function payloadOf(d: Draft): Record<string, unknown> {
  return {
    ...(d.id ? { id: d.id } : {}),
    name: d.name.trim(),
    viewMode: d.viewMode,
    actions: d.actions.map((a) => ({
      name: a.name.trim(), clip: a.name.trim(),
      frames: Number(a.frames) || 1, fps: Number(a.fps) || 12,
      loop: !a.oneShot, oneShot: a.oneShot,
    })),
    note: d.note,
    qwenPromptContract: d.qwenPromptContract,
    subjects: Number(d.subjects) || 1,
    budget: {
      candidates: Number(d.candidates) || 1,
      singleFrameSeconds: Number(d.singleFrameSeconds) || 26,
      passRate: Number(d.passRate) || 0.75,
    },
  }
}

const MODE_LABEL: Record<string, string> = Object.fromEntries(
  VIEW_MODES.map((m) => [m.id, m.label]),
)

export default function MotionTemplatePanel({ query }: Props) {
  const [items, setItems] = useState<MotionTemplate[]>([])
  const [estimates, setEstimates] = useState<Record<string, MotionEstimate>>({})
  const [draft, setDraft] = useState<Draft | null>(null)
  const [confirmId, setConfirmId] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [notice, setNotice] = useState('')

  const load = useCallback(() => {
    fetchMotionTemplates()
      .then((r) => {
        setItems(Array.isArray(r?.items) ? r.items : [])
        setEstimates(r?.estimates ?? {})
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => { load() }, [load])

  const shown = useMemo(() => {
    const kw = query.trim().toLowerCase()
    if (!kw) return items
    return items.filter((t) => t.name.toLowerCase().includes(kw)
      || t.id.toLowerCase().includes(kw))
  }, [items, query])

  const patch = (p: Partial<Draft>) => setDraft((d) => d ? { ...d, ...p } : d)
  const patchAction = (i: number, p: Partial<DraftAction>) => setDraft((d) => {
    if (!d) return d
    const actions = d.actions.map((a, n) => (n === i ? { ...a, ...p } : a))
    return { ...d, actions }
  })

  const save = async () => {
    if (!draft) return
    if (!draft.name.trim()) { setErr('模板要有名字'); return }
    if (!draft.actions.length) { setErr('至少留一个动作'); return }
    if (draft.actions.some((a) => !a.name.trim())) { setErr('动作名不能为空'); return }
    setBusy(true); setErr(''); setNotice('')
    try {
      const body = payloadOf(draft)
      const r = draft.id
        ? await updateMotionTemplate(draft.id, body)
        : await saveMotionTemplate(body)
      setItems(r.items ?? [])
      setEstimates(r.estimates ?? {})
      setDraft(null)
      setNotice(draft.id ? '模板已更新' : '模板已保存，抽帧面板里就能选到')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const remove = async (id: string) => {
    setBusy(true); setErr(''); setNotice('')
    try {
      const r = await deleteMotionTemplate(id)
      setItems(r.items ?? [])
      setEstimates(r.estimates ?? {})
      setConfirmId('')
      setNotice('模板已删除')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mt-root">
      <div className="mt-bar">
        <button
          type="button" className="mt-btn primary"
          onClick={() => { setDraft(emptyDraft()); setErr(''); setNotice('') }}
        >
          + 新建模板
        </button>
        <span className="mt-hint">
          模板只存元数据（方向 / 动作 / 帧率 / 预算），不存图片与权重。
          内置三套只读，想改先「另存为我的」。
        </span>
      </div>

      {err && <p className="mt-err">{err}</p>}
      {notice && <p className="mt-ok">{notice}</p>}

      {draft && (
        <div className="mt-editor">
          <div className="mt-editor-head">
            <b>{draft.id ? '编辑模板' : '新建模板'}</b>
            <span className="spacer" />
            <button type="button" className="mt-btn" onClick={() => setDraft(null)}>取消</button>
            <button type="button" className="mt-btn primary" disabled={busy} onClick={() => void save()}>
              {busy ? '保存中…' : '保存'}
            </button>
          </div>
          <div className="mt-fields">
            <label className="mt-field">名字
              <input value={draft.name} placeholder="例：俯视 · 施法四方向"
                onChange={(e) => patch({ name: e.target.value })} />
            </label>
            <label className="mt-field">视角模式
              <select value={draft.viewMode}
                onChange={(e) => patch({ viewMode: e.target.value })}>
                {VIEW_MODES.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </label>
          </div>

          <div className="mt-actions-head">
            <b>动作表</b>
            <span>方向由视角模式推导，不用在这里填</span>
            <button type="button" className="mt-btn"
              onClick={() => patch({ actions: [...draft.actions,
                { name: '', frames: 6, fps: 12, oneShot: false }] })}>
              + 加动作
            </button>
          </div>
          <div className="mt-action-rows">
            {draft.actions.map((a, i) => (
              <div className="mt-action-row" key={i}>
                <input className="mt-aname" value={a.name} placeholder="动作名（idle / walk…）"
                  onChange={(e) => patchAction(i, { name: e.target.value })} />
                <label>帧
                  <input type="number" min={1} max={240} value={a.frames}
                    onChange={(e) => patchAction(i, { frames: Number(e.target.value) || 1 })} />
                </label>
                <label>fps
                  <input type="number" min={0.1} max={60} step={0.5} value={a.fps}
                    onChange={(e) => patchAction(i, { fps: Number(e.target.value) || 12 })} />
                </label>
                <label className="mt-check">
                  <input type="checkbox" checked={a.oneShot}
                    onChange={(e) => patchAction(i, { oneShot: e.target.checked })} />
                  单次（不循环）
                </label>
                <button type="button" className="mt-btn danger" title="删掉这个动作"
                  onClick={() => patch({ actions: draft.actions.filter((_, n) => n !== i) })}>
                  ×
                </button>
              </div>
            ))}
          </div>

          <details className="mt-adv">
            <summary>进阶：成本估算 / 主体数 / 提示词契约</summary>
            <div className="mt-fields">
              <label className="mt-field">候选数
                <input type="number" min={1} max={8} value={draft.candidates}
                  onChange={(e) => patch({ candidates: Number(e.target.value) || 1 })} />
              </label>
              <label className="mt-field">单帧秒数
                <input type="number" min={1} value={draft.singleFrameSeconds}
                  onChange={(e) => patch({ singleFrameSeconds: Number(e.target.value) || 26 })} />
              </label>
              <label className="mt-field">通过率
                <input type="number" min={0.01} max={1} step={0.05} value={draft.passRate}
                  onChange={(e) => patch({ passRate: Number(e.target.value) || 0.75 })} />
              </label>
              <label className="mt-field">主体数
                <input type="number" min={1} max={8} value={draft.subjects}
                  onChange={(e) => patch({ subjects: Number(e.target.value) || 1 })} />
              </label>
            </div>
            <label className="mt-field wide">Qwen 提示词契约
              <input value={draft.qwenPromptContract}
                placeholder="例：top-down 45 degree view, full body, consistent outfit"
                onChange={(e) => patch({ qwenPromptContract: e.target.value })} />
            </label>
            <label className="mt-field wide">备注
              <input value={draft.note} placeholder="给以后的自己留一句"
                onChange={(e) => patch({ note: e.target.value })} />
            </label>
          </details>
        </div>
      )}

      <div className="mt-grid">
        {shown.length === 0 && (
          <p className="lib-empty">
            {items.length === 0 ? '模板还没读到（确认服务在运行）。' : `「${query}」没有匹配的模板。`}
          </p>
        )}
        {shown.map((t) => {
          const e = estimates[t.id]
          return (
            <div className={`mt-card${t.builtin ? ' builtin' : ''}`} key={t.id}>
              <div className="mt-card-head">
                <b>{t.name}</b>
                <span className={`mt-badge${t.builtin ? '' : ' mine'}`}>
                  {t.builtin ? '内置' : '我的'}
                </span>
              </div>
              <div className="mt-card-sub">
                {MODE_LABEL[t.viewMode] ?? t.viewMode} · {t.directions?.length ?? 0} 向
                {Object.keys(t.mirrorFrom ?? {}).length > 0 && ' · 含镜像派生'}
              </div>
              <div className="mt-chips">
                {(t.actions ?? []).map((a) => (
                  <span className="mt-chip" key={a.name}>
                    {a.name} · {a.frames}帧 · {a.fps}fps{a.oneShot ? ' · 单次' : ''}
                  </span>
                ))}
              </div>
              {e && (
                <div className="mt-cost">
                  估算 {e.images} 张 · 约 {e.totalMinutes} 分钟
                  {e.mirroredDirections?.length
                    ? `（${e.mirroredDirections.join('、')} 镜像派生）` : ''}
                </div>
              )}
              {t.note && <div className="mt-note">{t.note}</div>}
              <div className="mt-card-actions">
                {t.builtin ? (
                  <button type="button" className="mt-btn"
                    onClick={() => { setDraft(draftOf(t, true)); setErr(''); setNotice('') }}>
                    另存为我的
                  </button>
                ) : (
                  <>
                    <button type="button" className="mt-btn"
                      onClick={() => { setDraft(draftOf(t, false)); setErr(''); setNotice('') }}>
                      编辑
                    </button>
                    {confirmId === t.id ? (
                      <>
                        <button type="button" className="mt-btn danger" disabled={busy}
                          onClick={() => void remove(t.id)}>确认删除</button>
                        <button type="button" className="mt-btn"
                          onClick={() => setConfirmId('')}>取消</button>
                      </>
                    ) : (
                      <button type="button" className="mt-btn danger"
                        onClick={() => setConfirmId(t.id)}>删除</button>
                    )}
                  </>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
