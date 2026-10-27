import { useEffect, useRef, useState } from 'react'
import { fetchTrain, startTrain, type TrainState } from '../api'

type Props = {
  onClose: () => void
  onTrained: () => void
}

/**
 * LoRA 训练面板。
 *
 * 训练走独立接口（不占生成队列），所以这里只做「提交 + 轮询」。
 * 数据集就是磁盘上一个目录：图片 + 同名 .txt；没有 .txt 时用统一 caption。
 */
export default function TrainPanel({ onClose, onTrained }: Props) {
  const [form, setForm] = useState({
    data_dir: 'dataset/px_sprites',
    name: 'my_lora',
    trigger: '',
    caption: '',
    rank: 8,
    steps: 0,
    epochs: 8,
    resolution: 512,
  })
  const [state, setState] = useState<TrainState | null>(null)
  const [error, setError] = useState('')
  const timer = useRef<number | null>(null)

  const set = <K extends keyof typeof form>(k: K, v: (typeof form)[K]) =>
    setForm((f) => ({ ...f, [k]: v }))

  // 轮询训练进度，直到 done/failed
  useEffect(() => {
    if (!state || state.status === 'done' || state.status === 'failed') return
    timer.current = window.setTimeout(async () => {
      try {
        const next = await fetchTrain(state.train_id)
        setState(next)
        if (next.status === 'done') onTrained()
      } catch (e) {
        setError(String(e))
      }
    }, 1200)
    return () => {
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [state, onTrained])

  const submit = async () => {
    setError('')
    try {
      const { train_id } = await startTrain({
        ...form,
        label: form.name,
        trigger: form.trigger,
      })
      setState({ train_id, status: 'queued', step: 0, total: form.steps })
    } catch (e) {
      setError(String(e))
    }
  }

  const busy = state?.status === 'queued' || state?.status === 'running'
  const pct =
    busy && state?.total ? Math.min(100, Math.round(((state.step || 0) / state.total) * 100)) : 0

  return (
    <div className="train-mask" onClick={onClose}>
      <div className="train-panel" onClick={(e) => e.stopPropagation()}>
        <div className="train-head">
          <span className="node-kind">LoRA 训练</span>
          <span className="spacer" />
          <button className="icon-btn" title="关闭" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="train-body">
          <label className="block">
            数据集目录
            <input
              value={form.data_dir}
              onChange={(e) => set('data_dir', e.target.value)}
              placeholder="dataset/my_character"
            />
          </label>
          <p className="train-tip">
            图片 + 同名 <code>.txt</code>（每张一句 caption）。没 .txt 就填下面的统一 caption。
            角色 LoRA 建议 15~30 张同一角色不同姿势；风格 LoRA 建议 20~40 张同风格图。
          </p>

          <div className="row">
            <label className="grow">
              输出名
              <input value={form.name} onChange={(e) => set('name', e.target.value)} />
            </label>
            <label className="grow">
              触发词
              <input
                value={form.trigger}
                onChange={(e) => set('trigger', e.target.value)}
                placeholder="如 mk_knight"
              />
            </label>
          </div>
          <p className="train-tip">
            触发词没填就等于白训——生成时不知道要写什么词才能唤出它。
          </p>

          <label className="block">
            统一 caption（没有 .txt 时用）
            <input
              value={form.caption}
              onChange={(e) => set('caption', e.target.value)}
              placeholder="my_style, pixel art game sprite, white background"
            />
          </label>

          <div className="row">
            <label>
              rank
              <input
                type="number"
                min={1}
                max={64}
                value={form.rank}
                onChange={(e) => set('rank', Number(e.target.value) || 8)}
              />
            </label>
            <label>
              步数（0=自动）
              <input
                type="number"
                min={0}
                max={4000}
                value={form.steps}
                onChange={(e) => set('steps', Number(e.target.value) || 0)}
              />
            </label>
            <label>
              轮数
              <input
                type="number"
                min={1}
                max={40}
                value={form.epochs}
                onChange={(e) => set('epochs', Number(e.target.value) || 8)}
              />
            </label>
          </div>

          <button className="gen-btn" disabled={busy || !form.name || !form.data_dir} onClick={submit}>
            {busy ? '训练中…' : '开始训练'}
          </button>

          {error && <div className="err">{error}</div>}

          {state && (
            <div className="train-status">
              <div className="train-bar">
                <div className="train-bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <div className="train-line">
                <b>{state.stage || state.status}</b>
                {state.total ? ` · ${state.step || 0}/${state.total} 步（${pct}%）` : ''}
                {typeof state.loss === 'number' ? ` · loss ${state.loss}` : ''}
              </div>
              <div className="train-line dim">
                {typeof state.elapsed === 'number' ? `${state.elapsed}s` : ''}
                {typeof state.peak_vram_gb === 'number' ? ` · ${state.peak_vram_gb}GB` : ''}
                {typeof state.trainable_params === 'number'
                  ? ` · ${(state.trainable_params / 1e6).toFixed(2)}M 可训练参数`
                  : ''}
              </div>
              {state.status === 'done' && (
                <div className="train-line ok">
                  完成：{state.path}（{state.seconds}s）· 已加入 LoRA 列表，关掉面板即可选用
                </div>
              )}
              {state.status === 'failed' && <div className="err">{state.error}</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
