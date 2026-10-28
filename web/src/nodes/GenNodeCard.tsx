import { useRef, useState } from 'react'
import { hasCJK } from '../store/prompts'
import MaskEditor from '../components/MaskEditor'
import type { GenNode, LoraInfo, Preset } from '../types'

export const NODE_W = 300

export type NodePatch = Partial<
  Pick<GenNode, 'x' | 'y' | 'params' | 'view' | 'job' | 'source' | 'image' | 'charSource'>
>

type Props = {
  node: GenNode
  selected: boolean
  zoom: number
  onSelect: () => void
  onPatch: (p: NodePatch) => void
  onGenerate: () => void
  onDuplicate: () => void
  onDelete: () => void
  favorited: boolean
  onToggleFavorite: () => void
  loras: LoraInfo[]
  model: string
  onSpawnImg2Img: () => void
  /** 图生图的上游节点（从 source.nodeId 解析而来） */
  sourceNode?: GenNode
  /** 可以作为源图的节点列表 */
  sourceOptions: { id: string; label: string }[]
  /** 可以当新背景用的图片节点 */
  bgOptions: { id: string; label: string }[]
  /** 按节点 id 取缩略图 URL（角色迁移的角色图用） */
  charThumbOf?: (nodeId?: string) => string | undefined
  /** 从本节点的输出点拉一条连线 */
  onStartLink?: () => void
  /** 连线时鼠标进入/离开某个节点（用来判断连线落在哪） */
  onHover?: (id: string | null) => void
  /** 拖本机文件经过某个节点（落下时把它当源） */
  onDragOverNode?: (id: string) => void
  linkTarget?: boolean
  /** 全屏预览（灯箱） */
  onPreview?: (url: string, name: string) => void
  /** 当前项目 id（蒙版要存进项目目录） */
  projectId?: string
}

const PRESETS: { v: Preset; label: string }[] = [
  { v: 'sprite', label: '像素精灵' },
  { v: 'portrait', label: '立绘' },
  { v: 'wuxia', label: '武侠 / 国风' },
  { v: 'none', label: '不套契约' },
]

export default function GenNodeCard({
  node,
  selected,
  zoom,
  onSelect,
  onPatch,
  onGenerate,
  onDuplicate,
  onDelete,
  favorited,
  onToggleFavorite,
  loras,
  model,
  onSpawnImg2Img,
  sourceNode,
  sourceOptions,
  bgOptions,
  charThumbOf,
  onStartLink,
  onHover,
  onDragOverNode,
  linkTarget,
  onPreview,
  projectId,
}: Props) {
  const dragRef = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null)
  const p = node.params
  const loraInfo = loras.find((l) => l.name === p.lora)
  const job = node.job
  const busy = job?.status === 'queued' || job?.status === 'running'
  const hasImage = !!job?.url
  const imageMode = node.view === 'image' && hasImage
  const isI2I = node.kind === 'img2img'
  const isImg = node.kind === 'image'
  const isBg = node.kind === 'bg'
  const isInpaint = node.kind === 'inpaint'
  const isAiRepaint = node.kind === 'airepaint'
  const isPixelify = node.kind === 'pixelify'
  const isPoseClone = node.kind === 'poseclone'
  const isSpriteSwap = node.kind === 'spriteswap'
  const [maskOpen, setMaskOpen] = useState(false)
  // 源图可以是上游节点的产物、图片节点的本地图，也可以是节点上直接记的绝对路径
  const hasSource = !!(
    sourceNode?.job?.raw ||
    sourceNode?.job?.path ||
    (sourceNode?.kind === 'image' && sourceNode.image?.path) ||
    node.source?.path
  )
  // 缩略图：图片节点用自己的图，其它节点用出图结果
  const sourceThumb =
    sourceNode?.kind === 'image' ? sourceNode.image?.url : sourceNode?.job?.url
  // 角色迁移的角色图缩略图（角色图节点或本地图）
  const charThumb = charThumbOf?.(node.charSource?.nodeId)
  // 可全屏预览的图（图片节点的本地图 / 生成节点的产物）
  const previewUrl = isImg ? node.image?.url : job?.url
  const previewName = previewUrl
    ? decodeURIComponent(previewUrl.split('/').pop() || 'image.png')
    : ''

  const startDrag = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    // 交互控件上绝不启动拖拽。一旦 setPointerCapture，后续 pointerup 会被重定向到
    // 卡头，浏览器按「pointerdown 与 pointerup 的最近公共祖先」派发 click ——
    // click 就落到卡头上而不是按钮上，按钮永远点不动（切回参数视图失效即此因）。
    const el = e.target as HTMLElement
    if (el.closest('button, input, textarea, select, summary, a, label')) return
    e.stopPropagation()
    onSelect()
    dragRef.current = { sx: e.clientX, sy: e.clientY, ox: node.x, oy: node.y }
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  }

  const moveDrag = (e: React.PointerEvent) => {
    const d = dragRef.current
    if (!d) return
    onPatch({
      x: Math.round(d.ox + (e.clientX - d.sx) / zoom),
      y: Math.round(d.oy + (e.clientY - d.sy) / zoom),
    })
  }

  const endDrag = (e: React.PointerEvent) => {
    if (!dragRef.current) return
    dragRef.current = null
    try {
      ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
    } catch {
      /* 已释放 */
    }
  }

  const setParam = <K extends keyof typeof p>(k: K, v: (typeof p)[K]) =>
    onPatch({ params: { ...p, [k]: v } })

  /** 图片节点的展示宽度（高度按原图比例自动算） */
  const setImageW = (w: number) => {
    if (!node.image) return
    const cw = Math.max(60, Math.min(900, Math.round(w)))
    onPatch({ image: { ...node.image, w: cw } })
  }

  /** 拖右下角的手柄改大小：屏幕像素差要除以缩放比才是世界坐标 */
  const resizeRef = useRef<{ x: number; w: number } | null>(null)
  const startResize = (e: React.PointerEvent) => {
    e.stopPropagation()
    e.preventDefault()
    resizeRef.current = { x: e.clientX, w: node.image?.w ?? 160 }
    const move = (ev: PointerEvent) => {
      const st = resizeRef.current
      if (!st) return
      setImageW(st.w + (ev.clientX - st.x) / zoom)
    }
    const up = () => {
      resizeRef.current = null
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  /**
   * 选 LoRA：同时把「推荐强度」和「匹配的提示词契约」一起带过来。
   * 不这么做的话，挂着武侠 LoRA 却仍在用像素契约，两边打架、两头不到位。
   */
  const chooseLora = (name: string) => {
    const info = loras.find((l) => l.name === name)
    const next = { ...p, lora: name }
    if (info) {
      next.loraScale = info.scale
      if (info.preset && p.preset === 'sprite') next.preset = info.preset
      // 注册表声明的「必须抠图 / 推荐 clip skip」一起带过来：像素 LoRA 都会画彩色底，
      // 不勾抠图就没法用；这两项用户基本不会自己想到要改
      if (typeof info.matte === 'boolean') next.matte = info.matte
      if (typeof info.clip_skip === 'number') next.clipSkip = info.clip_skip
    }
    onPatch({ params: next })
  }

  /** 触发词缺了等于白装 LoRA，所以给个一键补入，已包含则不重复加。 */
  const addTrigger = () => {
    const t = (loraInfo?.trigger || '').trim()
    if (!t) return
    const lower = p.prompt.toLowerCase()
    const missing = t.split(',').map((x) => x.trim()).filter(Boolean)
      .filter((x) => !lower.includes(x.toLowerCase()))
    if (!missing.length) return
    setParam('prompt', `${missing.join(', ')}, ${p.prompt}`.replace(/^, /, ''))
  }

  /** 源图区：图生图与抠图节点共用（连线接上来的源，或直接拖一张图） */
  const sourceBox = (
    <div className="src-box">
      <div className="src-thumb" title={sourceNode ? '源图' : '还没有选源图'}>
        {sourceThumb ? (
          <img src={sourceThumb} alt="源图" draggable={false} />
        ) : (
          <span>无源图</span>
        )}
      </div>
      <div className="src-side">
        <select
          value={node.source?.nodeId ?? ''}
          onChange={(e) =>
            onPatch({ source: e.target.value ? { nodeId: e.target.value } : undefined })
          }
        >
          <option value="">（未选择源图）</option>
          {sourceOptions.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
        {isI2I && (
          <label className="strength">
            <span>重绘</span>
            <input
              type="range"
              min={0.15}
              max={0.9}
              step={0.05}
              value={p.strength}
              onChange={(e) => setParam('strength', Number(e.target.value))}
            />
            <b>{p.strength.toFixed(2)}</b>
          </label>
        )}
        <div className="src-note">
          {sourceNode?.kind === 'image'
            ? '用这张本地图'
            : sourceNode?.job?.raw
              ? '用原图（未像素化），保留细节'
              : hasSource
                ? '用成品，像素块会被重新糊开'
                : '把别的节点连过来，或直接把本地图拖到这个节点上'}
        </div>
      </div>
    </div>
  )

  return (
    <div
      className={`node gen${selected ? ' selected' : ''}${imageMode ? ' image-mode' : ''}${
        isImg ? ' is-img' : ''
      }${linkTarget ? ' link-target' : ''}`}
      data-node-id={node.id}
      style={{
        left: node.x,
        top: node.y,
        width: isImg ? node.image?.w ?? NODE_W : NODE_W,
      }}
      onPointerDown={(e) => {
        e.stopPropagation()
        onSelect()
      }}
      onPointerEnter={() => onHover?.(node.id)}
      onPointerLeave={() => onHover?.(null)}
      onDragOver={(e) => {
        // 本机图片拖到节点上：落下时把它当该节点的源
        if (e.dataTransfer.types.includes('Files')) {
          e.preventDefault()
          onDragOverNode?.(node.id)
        }
      }}
    >
      <div
        className="node-head"
        onPointerDown={startDrag}
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <span
          className={`node-kind${isI2I ? ' i2i' : ''}${isImg ? ' img' : ''}${
            isBg ? ' bg' : ''
          }${isInpaint ? ' inpaint' : ''}${isAiRepaint ? ' airepaint' : ''}${
            isPixelify ? ' pixelify' : ''
          }${isPoseClone ? ' poseclone' : ''}${isSpriteSwap ? ' spriteswap' : ''}`}
        >
          {isImg ? '图片' : isBg ? '抠图' : isPixelify ? '像素转换'
            : isPoseClone ? '角色迁移'
              : isSpriteSwap ? '序列帧替换'
                : isAiRepaint ? '局部AI重绘'
                  : isInpaint ? '局部重绘' : isI2I ? '图生图' : '图生成'}
        </span>
        {job && <span className={`node-state s-${job.status}`}>{statusLabel(job.status)}</span>}
        <span className="spacer" />
        {hasImage && (
          <button
            className={`icon-btn view-toggle${imageMode ? ' on' : ''}`}
            title={imageMode ? '切回参数视图' : '只看图'}
            onClick={(e) => {
              e.stopPropagation()
              onPatch({ view: imageMode ? 'expanded' : 'image' })
            }}
          >
            {imageMode ? '▤ 参数' : '▣'}
          </button>
        )}
        {previewUrl && (
          <button
            className="icon-btn"
            title="全屏预览 / 下载"
            onClick={(e) => {
              e.stopPropagation()
              onPreview?.(previewUrl, previewName)
            }}
          >
            ⤢
          </button>
        )}
        <button
          className="icon-btn"
          title="复制节点 (Ctrl+D)"
          onClick={(e) => {
            e.stopPropagation()
            onDuplicate()
          }}
        >
          ⧉
        </button>
        <button
          className="icon-btn danger"
          title="删除节点 (Delete)"
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
        >
          ✕
        </button>
        <span
          className="conn out"
          title="作为源图：从这里拖一条线到图生图节点"
          onPointerDown={(e) => {
            e.stopPropagation()
            e.preventDefault()
            onStartLink?.()
          }}
        />
      </div>

      {isSpriteSwap ? (
        <div className="node-body poseclone-body">
          {sourceBox}
          <div className="src-box char-box">
            <div className="src-thumb" title="角色图（身份来源）">
              {charThumb ? (
                <img src={charThumb} alt="角色" draggable={false} />
              ) : (
                <span>无角色</span>
              )}
            </div>
            <div className="src-side">
              <select
                value={node.charSource?.nodeId ?? ''}
                onChange={(e) =>
                  onPatch({
                    charSource: e.target.value
                      ? { nodeId: e.target.value }
                      : undefined,
                  })
                }
              >
                <option value="">（选角色图）</option>
                {sourceOptions.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
              <div className="src-note">
                {charThumb ? '这个角色的外观会套到每一帧上' : '缺角色图'}
              </div>
            </div>
          </div>
          <label className="bg-mode">
            方式
            <select
              value={p.ssMode}
              onChange={(e) => setParam('ssMode', e.target.value as typeof p.ssMode)}
            >
              <option value="generate">AI 重绘（可换体型）</option>
              <option value="recolor">换皮（只换配色，不跑模型）</option>
            </select>
          </label>
          {p.ssMode === 'generate' && (
            <label className="bg-mode">
              引擎
              <select
                value={p.ssEngineMode}
                onChange={(e) => setParam('ssEngineMode', e.target.value as typeof p.ssEngineMode)}
              >
                <option value="auto">自动（优先 AnimateDiff）</option>
                <option value="animatediff">AnimateDiff（帧间更稳）</option>
                <option value="frame">逐帧多候选（更省显存）</option>
              </select>
            </label>
          )}
          {p.ssMode === 'generate' && (
            <div className="prompt-wrap">
              <textarea
                className="prompt"
                rows={2}
                placeholder="角色描述？e.g. a blue slime monster with big eyes"
                value={p.prompt}
                onChange={(e) => setParam('prompt', e.target.value)}
              />
            </div>
          )}
          <details className="adv">
            <summary>更多设置</summary>
            <label className="strength">
              <span>列</span>
              <input
                type="number"
                min={0}
                max={32}
                value={p.ssCols}
                onChange={(e) => setParam('ssCols', Number(e.target.value) || 0)}
              />
              <b>{p.ssCols > 0 ? '列' : '自动'}</b>
            </label>
            <label className="strength">
              <span>行</span>
              <input
                type="number"
                min={0}
                max={32}
                value={p.ssRows}
                onChange={(e) => setParam('ssRows', Number(e.target.value) || 0)}
              />
              <b>{p.ssRows > 0 ? '行' : '自动'}</b>
            </label>
            {p.ssMode === 'generate' && (
              <>
                <label className="bg-mode">
                  参考
                  <select
                    value={p.ssRefMode}
                    onChange={(e) => setParam('ssRefMode', e.target.value as typeof p.ssRefMode)}
                  >
                    <option value="outline">轮廓（游戏素材）</option>
                    <option value="auto">自动（骨架→轮廓）</option>
                    <option value="pose">骨架（真人照片）</option>
                  </select>
                </label>
                <label className="strength">
                  <span>动作</span>
                  <input
                    type="range"
                    min={0.2}
                    max={1.5}
                    step={0.05}
                    value={p.ssCnScale}
                    onChange={(e) => setParam('ssCnScale', Number(e.target.value))}
                  />
                  <b>{p.ssCnScale.toFixed(2)}</b>
                </label>
                <label className="strength">
                  <span>身份</span>
                  <input
                    type="range"
                    min={0}
                    max={1.5}
                    step={0.05}
                    value={p.ssIpaScale}
                    onChange={(e) => setParam('ssIpaScale', Number(e.target.value))}
                  />
                  <b>{p.ssIpaScale.toFixed(2)}</b>
                </label>
              </>
            )}
            <label className="strength">
              <span>色板</span>
              <input
                type="number"
                min={0}
                max={64}
                value={p.ssPalette}
                onChange={(e) => setParam('ssPalette', Number(e.target.value) || 0)}
              />
              <b>{p.ssPalette > 0 ? '色' : '关'}</b>
            </label>
            {p.ssMode === 'generate' && p.ssEngineMode !== 'animatediff' && (
              <>
                <label className="strength">
                  <span>候选</span>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={p.ssCandidates}
                    onChange={(e) => setParam('ssCandidates', Math.max(1, Number(e.target.value) || 1))}
                  />
                  <b>张/帧</b>
                </label>
                <label className="strength">
                  <span>补提</span>
                  <input
                    type="number"
                    min={0}
                    max={5}
                    value={p.ssGateRetries}
                    onChange={(e) => setParam('ssGateRetries', Math.max(0, Number(e.target.value) || 0))}
                  />
                  <b>次</b>
                </label>
                <div className="src-note">
                  候选越多越稳（每帧生成几张按「动作贴合 + 身份贴合」挑一张），
                  但更慢。
                </div>
              </>
            )}
            {p.ssMode === 'generate' && p.ssEngineMode !== 'frame' && (
              <div className="src-note">
                AnimateDiff 一次生成整段，靠时序注意力锁住帧间一致性——
                实测 8 帧比逐帧方案快 2.6× 且漂移少 36%，但显存更高（13GB）。
              </div>
            )}
            <div className="src-note">
              源图是精灵图：横向排了几帧就填几列，纵向排了几帧就填几行；
              填 0 由后端自动检测。
            </div>
          </details>
          {hasImage && (
            <div
              className="preview"
              onClick={(e) => {
                e.stopPropagation()
                onPreview?.(job!.url!, previewName)
              }}
              title="点击全屏预览"
            >
              <img src={job!.url} alt="序列帧替换结果" draggable={false} />
            </div>
          )}
          <button
            className="gen-btn"
            disabled={
              busy || !hasSource || !charThumb ||
              (p.ssMode === 'generate' && !p.prompt.trim())
            }
            onClick={onGenerate}
            title={
              !hasSource ? '先接一张动作序列帧（精灵图）'
                : !charThumb ? '再选一张角色图'
                  : p.ssMode === 'generate' && !p.prompt.trim()
                    ? '写上角色长什么样' : undefined
            }
          >
            {busy ? '替换中…' : '替换序列帧'}
          </button>
        </div>
      ) : isPoseClone ? (
        <div className="node-body poseclone-body">
          {sourceBox}
          <div className="src-box char-box">
            <div className="src-thumb" title="角色图（身份来源）">
              {charThumb ? (
                <img src={charThumb} alt="角色" draggable={false} />
              ) : (
                <span>无角色</span>
              )}
            </div>
            <div className="src-side">
              <select
                value={node.charSource?.nodeId ?? ''}
                onChange={(e) =>
                  onPatch({
                    charSource: e.target.value
                      ? { nodeId: e.target.value }
                      : undefined,
                  })
                }
              >
                <option value="">（选角色图）</option>
                {sourceOptions.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
              <div className="src-note">
                {charThumb ? '角色外观由这张图决定（IP-Adapter）' : '缺角色图'}
              </div>
            </div>
          </div>
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={2}
              placeholder="角色描述？e.g. a blue slime monster with big eyes"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>
          <details className="adv">
            <summary>更多设置</summary>
            <label className="strength">
              <span>姿态</span>
              <input
                type="range"
                min={0.2}
                max={1.5}
                step={0.05}
                value={p.pcCnScale}
                onChange={(e) => setParam('pcCnScale', Number(e.target.value))}
              />
              <b>{p.pcCnScale.toFixed(2)}</b>
            </label>
            <label className="strength">
              <span>身份</span>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={p.pcIpaScale}
                onChange={(e) => setParam('pcIpaScale', Number(e.target.value))}
              />
              <b>{p.pcIpaScale.toFixed(2)}</b>
            </label>
            <label className="strength">
              <span>占高</span>
              <input
                type="range"
                min={0.3}
                max={1}
                step={0.05}
                value={p.pcFill}
                onChange={(e) => setParam('pcFill', Number(e.target.value))}
              />
              <b>{p.pcFill.toFixed(2)}</b>
            </label>
            <label className="bg-mode">
              参考
              <select
                value={p.pcRefMode}
                onChange={(e) => setParam('pcRefMode', e.target.value as typeof p.pcRefMode)}
              >
                <option value="auto">自动（骨架→轮廓）</option>
                <option value="pose">骨架（真人照片）</option>
                <option value="outline">轮廓（游戏素材）</option>
              </select>
            </label>
            <div className="src-note">
              参考模式「自动」会先试骨架，认不出人体就改用轮廓——游戏像素素材也能用
            </div>
          </details>
          {hasImage && (
            <div
              className="preview"
              onClick={(e) => {
                e.stopPropagation()
                onPreview?.(job!.url!, previewName)
              }}
              title="点击全屏预览"
            >
              <img src={job!.url} alt="角色迁移结果" draggable={false} />
            </div>
          )}
          <button
            className="gen-btn"
            disabled={busy || !hasSource || !charThumb || !p.prompt.trim()}
            onClick={onGenerate}
            title={
              !hasSource ? '先接一张动作参考图'
                : !charThumb ? '再选一张角色图'
                  : !p.prompt.trim() ? '写上角色长什么样' : undefined
            }
          >
            {busy ? '迁移中…' : '迁移角色'}
          </button>
          {hasImage && (
            <button
              className="mini wide"
              onClick={(e) => {
                e.stopPropagation()
                onSpawnImg2Img()
              }}
              title="以这张为源新建图生图节点"
            >
              ↳ 图生图
            </button>
          )}
        </div>
      ) : isPixelify ? (
        <div className="node-body pixelify-body">
          {sourceBox}
          <div className="px-opts">
            <label className="strength">
              <span>边长</span>
              <input
                type="number"
                min={0}
                max={512}
                value={p.pxTarget}
                onChange={(e) => setParam('pxTarget', Number(e.target.value) || 0)}
              />
              <b>{p.pxTarget > 0 ? 'px' : '自动'}</b>
            </label>
            <label className="strength">
              <span>色数</span>
              <input
                type="number"
                min={0}
                max={256}
                value={p.pxColors}
                onChange={(e) => setParam('pxColors', Number(e.target.value) || 0)}
              />
              <b>色</b>
            </label>
            <label className="bg-mode">
              取色
              <select
                value={p.pxSample}
                onChange={(e) => setParam('pxSample', e.target.value as typeof p.pxSample)}
              >
                <option value="median">中位色（最稳）</option>
                <option value="majority">多数表决</option>
                <option value="center">中心点</option>
              </select>
            </label>
            <label className="strength">
              <span>吸附</span>
              <input
                type="number"
                min={0}
                max={64}
                value={p.pxSnapTo}
                onChange={(e) => setParam('pxSnapTo', Number(e.target.value) || 0)}
              />
              <b>{p.pxSnapTo > 0 ? '倍' : '关'}</b>
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={p.pxGridSnap}
                onChange={(e) => setParam('pxGridSnap', e.target.checked)}
              />
              网格检测（关掉=普通降采样）
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={p.pxTransparentBg}
                onChange={(e) => setParam('pxTransparentBg', e.target.checked)}
              />
              纯色背景转透明
            </label>
          </div>
          {hasImage && (
            <div
              className="preview"
              onClick={(e) => {
                e.stopPropagation()
                onPreview?.(job!.url!, previewName)
              }}
              title="点击全屏预览"
            >
              <img src={job!.url} alt="像素画" draggable={false} />
            </div>
          )}
          <button
            className="gen-btn"
            disabled={busy || !hasSource}
            onClick={onGenerate}
            title={hasSource ? undefined : '先接一张源图（连线或直接拖图）'}
          >
            {busy ? '转换中…' : '转成像素画'}
          </button>
          {hasImage && (
            <button
              className="mini wide"
              onClick={(e) => {
                e.stopPropagation()
                onSpawnImg2Img()
              }}
              title="以这张为源新建图生图节点"
            >
              ↳ 图生图
            </button>
          )}
        </div>
      ) : isAiRepaint ? (
        <div className="node-body inpaint-body">
          {sourceBox}
          <div className="mask-box">
            <div className="mask-thumb">
              {p.maskUrl ? (
                <img src={p.maskUrl} alt="蒙版" draggable={false} />
              ) : (
                <span>无蒙版</span>
              )}
            </div>
            <div className="src-side">
              <button
                className="mini wide"
                onClick={(e) => {
                  e.stopPropagation()
                  setMaskOpen(true)
                }}
                disabled={!sourceThumb}
                title={sourceThumb ? '画笔涂抹要重绘的区域' : '先接一张源图'}
              >
                ✎ {p.maskUrl ? '重画蒙版' : '绘制蒙版'}
              </button>
              <div className="src-note">
                {p.maskUrl
                  ? `已涂 ${(p.maskCoverage * 100).toFixed(1)}% · 白=重绘区`
                  : '用画笔涂出要重绘的区域（白色部分会被重画）'}
              </div>
            </div>
          </div>
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={2}
              placeholder="要改成什么？e.g. give the knight a golden helmet"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>
          <label className="strength">
            <span>重绘</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={p.strength}
              onChange={(e) => setParam('strength', Number(e.target.value))}
            />
            <b>{p.strength.toFixed(2)}</b>
          </label>
          <details className="adv">
            <summary>更多设置</summary>
            <label className="strength">
              <span>上下文</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={p.rpContext}
                onChange={(e) => setParam('rpContext', Number(e.target.value))}
              />
              <b>{p.rpContext.toFixed(2)}</b>
            </label>
            <label className="strength">
              <span>外扩</span>
              <input
                type="number"
                min={0}
                max={64}
                value={p.rpExpand}
                onChange={(e) => setParam('rpExpand', Number(e.target.value) || 0)}
              />
              <b>px</b>
            </label>
            <label className="strength">
              <span>羽化</span>
              <input
                type="number"
                min={0}
                max={24}
                value={p.rpFeather}
                onChange={(e) => setParam('rpFeather', Number(e.target.value) || 0)}
              />
              <b>px</b>
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={p.rpMatchColor}
                onChange={(e) => setParam('rpMatchColor', e.target.checked)}
              />
              接缝色彩校正（只在边界带）
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={p.rpPaletteSnap}
                onChange={(e) => setParam('rpPaletteSnap', e.target.checked)}
              />
              色板吸附（像素素材用）
            </label>
          </details>
          {hasImage && (
            <div
              className="preview"
              onClick={(e) => {
                e.stopPropagation()
                onPreview?.(job!.url!, previewName)
              }}
              title="点击全屏预览"
            >
              <img src={job!.url} alt="重绘结果" draggable={false} />
            </div>
          )}
          <button
            className="gen-btn"
            disabled={busy || !hasSource || !p.maskPath || !p.prompt.trim()}
            onClick={onGenerate}
            title={
              !hasSource ? '先接一张源图'
                : !p.maskPath ? '先绘制蒙版'
                  : !p.prompt.trim() ? '写上要改成什么' : undefined
            }
          >
            {busy ? '重绘中…' : 'AI 重绘'}
          </button>
          {hasImage && (
            <button
              className="mini wide"
              onClick={(e) => {
                e.stopPropagation()
                onSpawnImg2Img()
              }}
              title="以这张为源新建图生图节点"
            >
              ↳ 图生图
            </button>
          )}
          <MaskEditor
            open={maskOpen}
            srcUrl={sourceThumb || ''}
            initialMaskUrl={p.maskUrl || null}
            title="局部 AI 重绘 · 涂出要重绘的区域"
            name={`mask_${node.id}`}
            project={projectId}
            onClose={() => setMaskOpen(false)}
            onSaved={(m) =>
              onPatch({
                params: {
                  ...p,
                  maskPath: m.path,
                  maskUrl: m.url,
                  maskCoverage: m.coverage,
                },
              })
            }
          />
        </div>
      ) : isInpaint ? (
        <div className="node-body inpaint-body">
          {sourceBox}
          <div className="mask-box">
            <div className="mask-thumb">
              {p.maskUrl ? (
                <img src={p.maskUrl} alt="蒙版" draggable={false} />
              ) : (
                <span>无蒙版</span>
              )}
            </div>
            <div className="src-side">
              <button
                className="mini wide"
                onClick={(e) => {
                  e.stopPropagation()
                  setMaskOpen(true)
                }}
                disabled={!sourceThumb}
                title={sourceThumb ? '画笔涂抹要重绘的区域' : '先接一张源图'}
              >
                ✎ {p.maskUrl ? '重画蒙版' : '绘制蒙版'}
              </button>
              <div className="src-note">
                {p.maskUrl
                  ? `已涂 ${(p.maskCoverage * 100).toFixed(1)}% · 白=重绘区`
                  : '用画笔涂出要重绘的区域（白色部分会被重画）'}
              </div>
            </div>
          </div>
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={2}
              placeholder="要改成什么？e.g. give the knight a golden helmet"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>
          <label className="strength">
            <span>重绘</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={p.strength}
              onChange={(e) => setParam('strength', Number(e.target.value))}
            />
            <b>{p.strength.toFixed(2)}</b>
          </label>
          {hasImage && (
            <div
              className="preview"
              onClick={(e) => {
                e.stopPropagation()
                onPreview?.(job!.url!, previewName)
              }}
              title="点击全屏预览"
            >
              <img src={job!.url} alt="重绘结果" draggable={false} />
            </div>
          )}
          <button
            className="gen-btn"
            disabled={busy || !hasSource || !p.maskPath || !p.prompt.trim()}
            onClick={onGenerate}
            title={
              !hasSource ? '先接一张源图' : !p.maskPath ? '先绘制蒙版' : !p.prompt.trim() ? '写上要改成什么' : undefined
            }
          >
            {busy ? '重绘中…' : '局部重绘'}
          </button>
          {hasImage && (
            <button
              className="mini wide"
              onClick={(e) => {
                e.stopPropagation()
                onSpawnImg2Img()
              }}
              title="以这张为源新建图生图节点"
            >
              ↳ 图生图
            </button>
          )}
          <MaskEditor
            open={maskOpen}
            srcUrl={sourceThumb || ''}
            initialMaskUrl={p.maskUrl || null}
            title="局部重绘 · 涂出要重绘的区域"
            name={`mask_${node.id}`}
            project={projectId}
            onClose={() => setMaskOpen(false)}
            onSaved={(m) =>
              onPatch({
                params: {
                  ...p,
                  maskPath: m.path,
                  maskUrl: m.url,
                  maskCoverage: m.coverage,
                },
              })
            }
          />
        </div>
      ) : isBg ? (
        <div className="node-body bg-body">
          {sourceBox}
          <div className="bg-opts">
            <label className="bg-mode">
              <input
                type="radio"
                checked={p.bgMode === 'alpha'}
                onChange={() => setParam('bgMode', 'alpha')}
              />
              透明（去背景）
            </label>
            <label className="bg-mode">
              <input
                type="radio"
                checked={p.bgMode === 'color'}
                onChange={() => setParam('bgMode', 'color')}
              />
              纯色
              <input
                type="color"
                value={p.bgColor}
                disabled={p.bgMode !== 'color'}
                onChange={(e) => setParam('bgColor', e.target.value)}
              />
            </label>
            <label className="bg-mode">
              <input
                type="radio"
                checked={p.bgMode === 'image'}
                onChange={() => setParam('bgMode', 'image')}
              />
              换背景图
              <select
                value={p.bgNodeId}
                disabled={p.bgMode !== 'image'}
                onChange={(e) => setParam('bgNodeId', e.target.value)}
              >
                <option value="">（选一个图片节点）</option>
                {bgOptions.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {hasImage && (
            <div
              className="preview"
              onClick={(e) => {
                e.stopPropagation()
                onPreview?.(job!.url!, previewName)
              }}
              title="点击全屏预览"
            >
              <img src={job!.url} alt="抠图结果" draggable={false} />
            </div>
          )}
          <button
            className="gen-btn"
            disabled={busy || !hasSource}
            onClick={onGenerate}
            title={hasSource ? undefined : '先接一张源图（连线或直接拖图）'}
          >
            {busy ? '处理中…' : p.bgMode === 'alpha' ? '去背景' : '换背景'}
          </button>
          {hasImage && (
            <button
              className="mini wide"
              onClick={(e) => {
                e.stopPropagation()
                onSpawnImg2Img()
              }}
              title="以这张为源新建图生图节点"
            >
              ↳ 图生图
            </button>
          )}
        </div>
      ) : isImg ? (
        <div className="node-body img-body">
          {node.image ? (
            <>
              <div className="img-holder">
                <img
                  src={node.image.url}
                  alt=""
                  draggable={false}
                  title="点击全屏预览"
                  onClick={(e) => {
                    e.stopPropagation()
                    onPreview?.(node.image!.url, previewName)
                  }}
                />
                <span
                  className="resize"
                  title="拖动改变大小（也可以直接改下面的宽度）"
                  onPointerDown={startResize}
                />
              </div>
              <div className="img-tools">
                <label>
                  宽
                  <input
                    type="number"
                    min={60}
                    max={900}
                    value={Math.round(node.image.w)}
                    onChange={(e) => setImageW(Number(e.target.value) || 160)}
                  />
                </label>
                <button
                  className="mini wide"
                  onClick={(e) => {
                    e.stopPropagation()
                    onSpawnImg2Img()
                  }}
                  title="以此图为源新建图生图节点"
                >
                  ↳ 图生图
                </button>
              </div>
            </>
          ) : (
            <div className="img-empty">图片没了（可能文件被删）</div>
          )}
        </div>
      ) : imageMode ? (
        <div className="node-image-only">
          <img
            src={job!.url}
            alt={p.prompt}
            draggable={false}
            title="点击全屏预览"
            onClick={(e) => {
              e.stopPropagation()
              onPreview?.(job!.url!, previewName)
            }}
          />
          <div className="img-badge">
            种子 {job!.seed ?? p.seed}
            {typeof job!.seconds === 'number' ? ` · ${job!.seconds}s` : ''}
          </div>
        </div>
      ) : (
        <div className="node-body">
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={3}
              placeholder="English prompt, e.g. white cat knight, red cape, katana, front view"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
            <button
              className={`star${favorited ? ' on' : ''}`}
              title={favorited ? '取消收藏' : '收藏这条提示词'}
              disabled={!p.prompt.trim()}
              onClick={(e) => {
                e.stopPropagation()
                onToggleFavorite()
              }}
            >
              {favorited ? '★' : '☆'}
            </button>
          </div>
          <div className={`lang-hint${hasCJK(p.prompt) ? ' bad' : ''}`}>
            {hasCJK(p.prompt)
              ? '⚠ 检测到中文：SD1.5 的 CLIP 只认英文，中文会被拆成乱 token，出图基本靠猜'
              : 'SD1.5 只支持英文提示词；逗号分隔，主体 + 特征 + 视角'}
          </div>

          {(isI2I || isBg) && sourceBox}

          <div className="row">
            {!isI2I && (
              <>
                <label>
                  宽
                  <input
                    type="number"
                    value={p.width}
                    step={64}
                    min={128}
                    max={1024}
                    onChange={(e) => setParam('width', Number(e.target.value) || 512)}
                  />
                </label>
                <label>
                  高
                  <input
                    type="number"
                    value={p.height}
                    step={64}
                    min={128}
                    max={1024}
                    onChange={(e) => setParam('height', Number(e.target.value) || 512)}
                  />
                </label>
              </>
            )}
            <label>
              步数
              <input
                type="number"
                value={p.steps}
                min={4}
                max={50}
                onChange={(e) => setParam('steps', Number(e.target.value) || 25)}
              />
            </label>
          </div>

          <div className="row">
            <label className="grow">
              种子
              <input
                type="number"
                value={p.seed}
                onChange={(e) => setParam('seed', Number(e.target.value) || 0)}
              />
            </label>
            <button
              className="mini"
              title="随机种子"
              onClick={() => setParam('seed', -1)}
            >
              🎲
            </button>
            <label className="grow">
              风格
              <select
                value={p.preset}
                onChange={(e) => setParam('preset', e.target.value as Preset)}
              >
                {PRESETS.map((x) => (
                  <option key={x.v} value={x.v}>
                    {x.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <details className="fold">
            <summary>更多设置</summary>
            <div className="fold-body">
              <label className="block">
                反向提示词
                <textarea
                  rows={2}
                  placeholder="留空则用内置契约"
                  value={p.negative}
                  onChange={(e) => setParam('negative', e.target.value)}
                />
              </label>
              <label className="block">
                LoRA
                <select value={p.lora} onChange={(e) => chooseLora(e.target.value)}>
                  <option value="">不挂（纯底模）</option>
                  {loras.map((l) => (
                    <option key={l.name} value={l.name}>
                      {l.label}
                    </option>
                  ))}
                </select>
              </label>
              {loraInfo && (
                <div className="lora-box">
                  <div className="lora-trigger">
                    <span>
                      触发词 <b>{loraInfo.trigger || '（无）'}</b>
                    </span>
                    {loraInfo.trigger && (
                      <button className="mini" title="补进提示词开头" onClick={addTrigger}>
                        + 填入
                      </button>
                    )}
                  </div>
                  {loraInfo.note && <p className="lora-note">{loraInfo.note}</p>}
                </div>
              )}
              <div className="row">
                <label>
                  LoRA 强度
                  <input
                    type="number"
                    step={0.05}
                    min={0}
                    max={1.5}
                    value={p.loraScale}
                    disabled={!p.lora}
                    onChange={(e) => setParam('loraScale', Number(e.target.value))}
                  />
                </label>
                <label>
                  CFG
                  <input
                    type="number"
                    step={0.5}
                    value={p.cfg}
                    onChange={(e) => setParam('cfg', Number(e.target.value) || 7.5)}
                  />
                </label>
                <label>
                  像素块
                  <input
                    type="number"
                    min={0}
                    max={16}
                    value={p.pixel}
                    onChange={(e) => setParam('pixel', Number(e.target.value) || 0)}
                  />
                </label>
                <label>
                  色数
                  <input
                    type="number"
                    min={0}
                    max={64}
                    value={p.palette}
                    onChange={(e) => setParam('palette', Number(e.target.value) || 0)}
                  />
                </label>
                <label title="1=最后一层（默认）；2=跳过最后一层，多数动漫/像素 LoRA 的训练设置">
                  跳过层
                  <input
                    type="number"
                    min={1}
                    max={4}
                    value={p.clipSkip}
                    disabled={model === 'sdxl'}
                    onChange={(e) => setParam('clipSkip', Number(e.target.value) || 1)}
                  />
                </label>
              </div>
              <label className="check">
                <input
                  type="checkbox"
                  checked={p.matte}
                  onChange={(e) => setParam('matte', e.target.checked)}
                />
                抠图出透明底
              </label>
            </div>
          </details>

          {hasImage && (
            <div
              className="preview"
              onClick={(e) => {
                e.stopPropagation()
                onPreview?.(job!.url!, previewName)
              }}
              title="点击全屏预览"
            >
              <img src={job!.url} alt={p.prompt} draggable={false} />
            </div>
          )}

          <button
            className="gen-btn"
            disabled={busy || !p.prompt.trim() || (isI2I && !hasSource)}
            onClick={onGenerate}
            title={isI2I && !hasSource ? '先选源图' : undefined}
          >
            {busy ? (job?.status === 'queued' ? '排队中…' : '生成中…') : '生成'}
          </button>

          {hasImage && (
            <button
              className="mini wide"
              onClick={(e) => {
                e.stopPropagation()
                onSpawnImg2Img()
              }}
              title="以这张为源新建图生图节点"
            >
              ↳ 图生图
            </button>
          )}

          {job?.status === 'failed' && <div className="err">{job.error}</div>}
          {job?.status === 'done' && (
            <div className="meta">
              种子 {job.seed ?? p.seed}
              {typeof job.seconds === 'number' ? ` · ${job.seconds}s` : ''}
              {typeof job.peakVram === 'number' ? ` · ${job.peakVram}GB` : ''}
            </div>
          )}
          {/* SD1.5 画小体型角色时爱复制一份，负向词压不住；抠图后能量出来就报出来 */}
          {typeof job?.subjects === 'number' && job.subjects > 1 && (
            <div className="warn">
              ⚠ 抠图后还剩 {job.subjects} 块前景（精灵只要一个）：换种子，或把「single
              character」提到提示词最前面
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function statusLabel(s: string): string {
  switch (s) {
    case 'queued':
      return '排队'
    case 'running':
      return '生成中'
    case 'done':
      return '完成'
    case 'failed':
      return '失败'
    default:
      return s
  }
}
