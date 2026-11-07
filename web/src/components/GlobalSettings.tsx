/**
 * 全局设置：底模（只读）/ 抠图方式 / LoRA / 快速模式。
 *
 * ── 为什么单独成一个组件 ───────────────────────────────────────────────
 * 这些设置**不是画布私有的**：LoRA 挂在引擎上、抠图方式由后端读取，
 * 角色中心和画布用的是同一份状态。但以前它们的 UI 只长在画布侧栏里，
 * 进角色中心就再也改不到，看起来像「画布的设置」。
 *
 * 现在同一份内容有两个挂载点，**共用一套实现与同一份 App 状态**：
 *   1. 画布侧栏的「LoRA」「设置」两个 `.rail-sec`（保持 6 段 DOM 契约不变）
 *   2. 顶栏的 ⚙ 全局设置弹窗（画布 & 角色中心都能开）
 *
 * 不要在两边各写一份：以前就是同一个动作两个入口、两套状态，最后不一致。
 */
import { useRef, useState, type ChangeEvent } from 'react'
import { createPortal } from 'react-dom'
import type { QwenLora, ActiveLora, LoraInspect } from '../api'
import { inspectLora, commitLora, deleteLora, convertLora } from '../api'

/** 抠图方式的可选值。值原样传给后端（`matte.matte_auto` 的 backend 参数）。 */
export const MATTE_LABEL: Record<string, string> = {
  auto: '自动（有分割模型就抠得干净）',
  u2net: '语义分割 · 准但慢',
  color: '纯色键控 · 极快',
}

export type LoraSettingsProps = {
  loras: QwenLora[]
  activeLoras: ActiveLora[]
  onLorasChange: (items: { name: string; scale: number }[]) => void
  fastMode: boolean
  onFastMode: (on: boolean) => void
  /** 导入 / 删除 / 转换之后刷新列表（App 的 reloadLoras） */
  onLorasChanged?: () => void
  /** 目录里存在但格式不对的（需要转换） */
  skipped?: QwenLora[]
  /** 目录里一共扫到几个文件 */
  count?: number
  /** 扫描目录（给用户看，免得他不知道往哪拷） */
  dir?: string
  /** 重新扫描目录（App 的 reloadLoras） */
  onRefresh?: () => void | Promise<void>
}

export function LoraSettings({
  loras, activeLoras, onLorasChange, fastMode, onFastMode, onLorasChanged,
  skipped = [], count, dir = '', onRefresh,
}: LoraSettingsProps) {
  const [refreshing, setRefreshing] = useState(false)
  const [converting, setConverting] = useState('')
  const [importBusy, setImportBusy] = useState(false)
  const [importMsg, setImportMsg] = useState('')
  /** 需要转换、等用户点确认的检测结果 */
  const [pendingConvert, setPendingConvert] = useState<LoraInspect | null>(null)
  const [pendingDelete, setPendingDelete] = useState('')
  const fileRef = useRef<HTMLInputElement | null>(null)

  const refresh = async () => {
    if (!onRefresh) return
    setRefreshing(true)
    try { await onRefresh() } finally { setRefreshing(false) }
  }

  /** 目录里格式不对的那个，就地转（后端会先备份原文件）。 */
  const convertOne = async (name: string) => {
    setConverting(name); setImportMsg('')
    try {
      const out = await convertLora(name)
      setImportMsg(out.converted
        ? `已把「${out.item.name}」转成 diffusers 格式，现在可以挂了`
        : `「${out.item.name}」本来就是 diffusers 格式，刷新即可`)
      await onRefresh?.()
    } catch (err) {
      setImportMsg(`转换失败：${err instanceof Error ? err.message : '未知错误'}`)
    } finally { setConverting('') }
  }

  const onPick = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''            // 允许重复选同名文件
    if (!file) return
    setImportBusy(true); setImportMsg('')
    try {
      const info = await inspectLora(file)
      if (!info.convertible) {
        setImportMsg(`无法导入：${info.reason}。头几个键：${info.sample_keys.join('、')}`)
        return
      }
      if (info.need_convert) { setPendingConvert(info); return }
      const out = await commitLora(info.token, false, info.name)
      setImportMsg(`已导入「${out.name}」· ${out.item.tensors} 个张量`)
      onLorasChanged?.()
    } catch (err) {
      setImportMsg(`导入失败：${err instanceof Error ? err.message : '未知错误'}`)
    } finally { setImportBusy(false) }
  }

  const confirmConvert = async () => {
    if (!pendingConvert) return
    setImportBusy(true)
    try {
      const out = await commitLora(pendingConvert.token, true, pendingConvert.name)
      setImportMsg(`已转换并导入「${out.name}」· ${out.item.tensors} 个张量`
        + (out.skipped ? `，跳过 ${out.skipped} 个无关键` : ''))
      setPendingConvert(null)
      onLorasChanged?.()
    } catch (err) {
      setImportMsg(`转换失败：${err instanceof Error ? err.message : '未知错误'}`)
    } finally { setImportBusy(false) }
  }

  const doDelete = async (name: string) => {
    setPendingDelete('')
    try {
      await deleteLora(name)
      setImportMsg(`已删除「${name}」`)
      onLorasChanged?.()
    } catch (err) {
      setImportMsg(`删除失败：${err instanceof Error ? err.message : '未知错误'}`)
    }
  }

  const shortDir = dir ? dir.split(/[\\/]/).slice(-2).join('/') : 'models/lora'
  const total = count ?? (loras.length + skipped.length)

  return (
    <>
      {/* 目录是权威：用户把文件拷进 models/lora/ 后点这里刷新即可，不用重启 */}
      <div className="lora-head">
        <span className="lora-head-info">
          扫到 <b>{total}</b> 个 · 可挂 <b>{loras.length}</b>
          {skipped.length > 0 && <> · 待转换 <b className="warn">{skipped.length}</b></>}
        </span>
        <button type="button" className="lora-refresh" disabled={refreshing}
          onClick={() => void refresh()}
          title={`重新扫描 ${dir || 'models/lora/'}`}>
          {refreshing ? '扫描中…' : '↻ 刷新'}
        </button>
      </div>
      <div className="lora-dir" title={dir || 'models/lora/'}>目录：{shortDir}/</div>

      <label className="lora-row fast-toggle">
        <input
          type="checkbox"
          checked={fastMode}
          onChange={(e) => onFastMode(e.target.checked)}
        />
        <span className="lora-name">快速模式（少步蒸馏）</span>
        <span className="lora-size">-46%</span>
      </label>
      <div className="src-note">
        挂 Lightning 少步蒸馏 LoRA，步数压到 8。实测 24 步 17.3s → 8 步 9.3s，
        画质几乎不掉。逐帧路线（姿态序列 / 序列帧替换）建议开着。
      </div>

      {/* 导入入口刻意不用 .lora-row：冒烟按 .lora-row 的行数断言可用 LoRA 数。 */}
      <div className="lora-import">
        <button type="button" className="lora-import-btn" disabled={importBusy}
          onClick={() => fileRef.current?.click()}>
          {importBusy ? '处理中…' : '＋ 导入 LoRA（.safetensors）'}
        </button>
        <input ref={fileRef} type="file" accept=".safetensors"
          className="lora-file-input" onChange={(e) => void onPick(e)} />
        <span className="lora-import-hint">先检测格式，不匹配会问你要不要转换</span>
      </div>
      {importMsg && <div className="lora-import-msg">{importMsg}</div>}

      {loras.length === 0 ? (
        <div className="src-note">
          这个目录里还没有可挂的 LoRA。点上面的「导入」选一个 .safetensors，
          或者直接把文件拷进 <code>{shortDir}/</code> 再点「↻ 刷新」。
        </div>
      ) : (
        <>
          {loras.map((l) => {
            const act = activeLoras.find((a) => a.name === l.name)
            return (
              <div className="lora-item" key={l.name}>
                <label className="lora-row">
                  <input
                    type="checkbox"
                    checked={!!act}
                    onChange={(e) => {
                      const next = e.target.checked
                        ? [...activeLoras.filter((a) => a.name !== l.name),
                          { name: l.name, scale: act?.scale ?? 0.8 }]
                        : activeLoras.filter((a) => a.name !== l.name)
                      onLorasChange(next.map((x) => ({ name: x.name, scale: x.scale })))
                    }}
                  />
                  <span className="lora-name" title={l.file}>{l.name}</span>
                  <span className="lora-size">{l.size_mb}MB</span>
                </label>
                <button type="button" className="lora-del" title={`删除 ${l.name}`}
                  onClick={() => setPendingDelete(l.name)}>✕</button>
                {/* 强度**每一行都显示**：以前只有勾上之后才出现，用户会以为
                    这个功能没了。没勾时显示默认值 0.8 并置灰，勾上就能拖。 */}
                <label className={`lora-scale${act ? '' : ' is-off'}`}
                  title={act ? `${l.name} 的强度` : '先勾上左边的框，强度才会生效'}>
                  <span>强度</span>
                  <input
                    type="range" min={0.1} max={1.5} step={0.05}
                    value={act?.scale ?? 0.8}
                    disabled={!act}
                    onChange={(e) => {
                      const sc = Number(e.target.value)
                      onLorasChange(activeLoras.map((x) =>
                        x.name === l.name ? { name: x.name, scale: sc } : x))
                    }}
                  />
                  <b>{(act?.scale ?? 0.8).toFixed(2)}</b>
                </label>
              </div>
            )
          })}
          {pendingDelete && (
            <div className="lora-del-confirm">
              <span>删掉「{pendingDelete}」？文件会从磁盘移除。</span>
              <button type="button" onClick={() => setPendingDelete('')}>取消</button>
              <button type="button" className="danger"
                onClick={() => void doDelete(pendingDelete)}>确认删除</button>
            </div>
          )}
          {activeLoras.length > 0 && (
            <div className="src-note">
              每个 LoRA 单独调强度；挂上的 LoRA 对**所有**生成生效（画布与角色中心，
              全局状态），重开应用也会自动恢复，直到在这里取消。
            </div>
          )}
        </>
      )}

      {/* 格式不对的文件也要露出来：以前被后端静默过滤，用户只会觉得「少了几个」 */}
      {skipped.length > 0 && (
        <div className="lora-skip">
          <div className="lora-skip-head">
            目录里还有 {skipped.length} 个不能直接挂的（格式不对）
          </div>
          {skipped.map((sk) => (
            <div className="lora-skip-item" key={sk.name}>
              <span className="lora-skip-name" title={sk.file}>{sk.name}</span>
              <span className="lora-skip-meta">
                {sk.size_mb}MB · {sk.format === 'comfyui' ? 'ComfyUI 格式' : '格式未知'}
              </span>
              {sk.convertible ? (
                <button type="button" className="lora-convert-one"
                  disabled={converting === sk.name}
                  onClick={() => void convertOne(sk.name)}>
                  {converting === sk.name ? '转换中…' : '转换'}
                </button>
              ) : (
                <span className="lora-skip-no" title={sk.reason}>无法转换</span>
              )}
            </div>
          ))}
          <div className="src-note">
            转换只改键名与张量布局，不改权重；原文件会备份到
            <code>library/lora_backup/</code>。
          </div>
        </div>
      )}

      {/* 格式不符时先问用户：直接挂 ComfyUI 格式会「静默不生效」，比报错更难查。
          用 Portal 挂到 body，别被侧栏的滚动容器裁掉。 */}
      {pendingConvert && createPortal(
        <div className="lora-convert-mask"
          onClick={() => { if (!importBusy) setPendingConvert(null) }}>
          <div className="lora-convert-dialog" onClick={(e) => e.stopPropagation()}>
            <h3>这个 LoRA 需要转换</h3>
            <p className="lora-convert-lead">
              「{pendingConvert.original_name}」是 <b>ComfyUI 格式</b>；当前底模
              Qwen-Image 2.1 只能挂载 <b>diffusers 原生格式</b>。直接导入会
              「挂上了但完全没效果」，而且不会报任何错。
            </p>
            <p className="lora-convert-reason">{pendingConvert.reason}</p>
            <div className="lora-convert-facts">
              <span>张量 {pendingConvert.tensors}</span>
              <span>{pendingConvert.size_mb} MB</span>
              <span>存为 {pendingConvert.name}.safetensors</span>
            </div>
            <p className="lora-convert-keys">
              键名示例：{pendingConvert.sample_keys.join('、')}
            </p>
            <div className="src-note">
              转换只改键名与张量布局（含 img_mlp 拆分），不改权重数值；
              原始文件保持不动，转换结果写进 models/lora/。
            </div>
            <div className="lora-convert-acts">
              <button type="button" className="cc-btn ghost" disabled={importBusy}
                onClick={() => setPendingConvert(null)}>取消</button>
              <button type="button" className="cc-btn primary" disabled={importBusy}
                onClick={() => void confirmConvert()}>
                {importBusy ? '转换中…' : '确认转换并导入'}
              </button>
            </div>
          </div>
        </div>, document.body)}
    </>
  )
}

export type GeneralSettingsProps = {
  baseModel: string
  matteBackend: string
  onMatteBackend: (v: string) => void
}

export function GeneralSettings({
  baseModel, matteBackend, onMatteBackend,
}: GeneralSettingsProps) {
  return (
    <>
      <div className="settings-global">
        <span className="settings-global-k">底模（全局）</span>
        <b className="settings-global-v">
          {baseModel === 'qwen-image-2.1' ? 'Qwen-Image 2.1' : baseModel}
        </b>
        <span className="settings-global-note">
          整站只有一个底模，节点里不用再选；挂上的 LoRA 同样全局生效。
        </span>
      </div>
      <label className="block">
        抠图方式（全局）
        <select className="matte-backend-select" value={matteBackend}
          onChange={(e) => onMatteBackend(e.target.value)}>
          {Object.keys(MATTE_LABEL).map((k) => (
            <option key={k} value={k}>{MATTE_LABEL[k]}</option>
          ))}
        </select>
      </label>
      <div className="src-note">
        对**画布节点和角色中心**同时生效（角色中心要在那一步勾了「抠一次背景」才用得上）。
        设置保存在本地，重开应用不丢。
      </div>
    </>
  )
}

export type GlobalSettingsDialogProps = LoraSettingsProps & GeneralSettingsProps & {
  open: boolean
  onClose: () => void
}

/** 顶栏 ⚙ 打开的弹窗。画布与角色中心共用同一个入口。 */
export default function GlobalSettingsDialog({
  open, onClose, ...rest
}: GlobalSettingsDialogProps) {
  if (!open) return null
  return createPortal(
    <div className="lora-convert-mask gs-mask" onClick={onClose}>
      <div className="gs-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="gs-dialog-head">
          <h2>全局设置</h2>
          <button type="button" className="gs-close" onClick={onClose}
            title="关闭（Esc）">✕</button>
        </div>
        <p className="gs-dialog-lead">
          这些设置跟视图无关：在画布和角色中心是同一份，改一次两边都生效，
          并且存在本地，重开应用不丢。
        </p>

        <h3 className="gs-section">通用</h3>
        <GeneralSettings
          baseModel={rest.baseModel}
          matteBackend={rest.matteBackend}
          onMatteBackend={rest.onMatteBackend} />

        <h3 className="gs-section">LoRA（{rest.loras.length}）</h3>
        <LoraSettings
          loras={rest.loras}
          activeLoras={rest.activeLoras}
          onLorasChange={rest.onLorasChange}
          fastMode={rest.fastMode}
          onFastMode={rest.onFastMode}
          onLorasChanged={rest.onLorasChanged}
          skipped={rest.skipped}
          count={rest.count}
          dir={rest.dir}
          onRefresh={rest.onRefresh} />
      </div>
    </div>, document.body)
}
