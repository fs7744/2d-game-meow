import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { FavoritePrompt } from '../types'
import type { QwenLora, ActiveLora } from '../api'
import { LoraSettings, GeneralSettings } from './GlobalSettings'
import type { TemplateInfo } from '../api'
import type { ProjectMeta } from '../store/projects'

type Props = {
  favs: FavoritePrompt[]
  projects: ProjectMeta[]
  currentProject: string
  onSwitchProject: (id: string) => void
  onCreateProject: () => void
  onRenameProject: (id: string, name: string) => void
  onDeleteProject: (id: string) => void
  /** 导出当前项目为 .mproj 单文件 */
  onExportProject: (id: string) => void
  /** 在文件管理器里打开项目目录 */
  onRevealProject: (id: string) => void
  /** 把当前项目存成预设模板 */
  onSaveTemplate: (id: string) => void
  /** 打开 .mproj 文件导入成一个新项目 */
  onImportProject: (file: File) => void
  /** 可用模板 */
  templates: TemplateInfo[]
  /** 从模板新建一个项目 */
  onUseTemplate: (tid: string) => void
  onDeleteTemplate: (tid: string) => void
  onAddNode: (
    kind?: 'gen' | 'img2img' | 'image' | 'bg' | 'inpaint' | 'airepaint' | 'pixelify'
    | 'poseclone' | 'spriteswap' | 'topixel' | 'poseswap' | 'poseedit'
    | 'posebatch'
    | 'outpaint' | 'outfit' | 'faceswap',
  ) => void
  canApply: boolean
  onApplyFav: (item: FavoritePrompt) => void
  onRemoveFav: (id: string) => void
  /** 打开资料库浏览面板（提示词 / 姿势两个 tab） */
  onOpenLibrary: (tab: 'prompt' | 'pose') => void
  matteBackend: string
  onMatteBackend: (v: string) => void
  scanning?: boolean
  /** 可用的 Qwen LoRA（只含 diffusers 原生格式的） */
  loras: QwenLora[]
  /** 当前生效的 LoRA */
  activeLoras: ActiveLora[]
  /** 挂载 / 卸载 LoRA（状态式：一直生效直到下次改） */
  onLorasChange: (items: { name: string; scale: number }[]) => void
  /** 快速模式（少步蒸馏）：挂 Lightning + 步数压到 8 */
  fastMode: boolean
  onFastMode: (on: boolean) => void
  /** 全局底模（只读展示：节点里不再提供选择） */
  baseModel: string
  /** 导入 / 删除 / 转换 LoRA 之后刷新列表 */
  onLorasChanged?: () => void
  /** 目录里存在但格式不对的（显示为「待转换」） */
  lorasSkipped?: QwenLora[]
  lorasDir?: string
  lorasCount?: number
  /** 重新扫描 models/lora/ */
  onRefreshLoras?: () => void | Promise<void>
}

/* ==========================================================================
   侧栏信息架构：Activity Bar（图标 tab）+ 每 tab 独立面板
   --------------------------------------------------------------------------
   以前是 6 个分组全部默认展开、堆在一条滚动条里：18 个节点条目把「资料库」
   「设置」挤到屏幕外，找东西只能靠翻。业界同类产品的共识不是这样：

   - ComfyUI：左侧是 Nodes / Workflows / Models / Queue 等**图标 tab**；
     节点库面板里有 Essentials（按用途分组的精选）、All nodes、搜索框、
     Bookmarked 收藏置顶，节点靠拖到画布或点一下添加。
   - n8n：`Tab` 或 `Cmd/Ctrl+K` 唤起节点面板，面板顶部有搜索框，
     `Enter` 插入所选节点、`Esc` 关闭。
   - VS Code / Figma：Activity Bar 图标导航，每个面板独立滚动，
     面板底部有状态条，不把「资源」和「设置」混在一条长列表里。

   这里取同一个骨架：4 个一级 tab（节点 / 项目 / 资料库 / 设置），
   节点 tab 内做「常用置顶 → 搜索 → 分类列表」，并支持 Ctrl+K 聚焦搜索、
   Enter 添加第一个匹配、Esc 清空。
   ========================================================================== */
type RailTab = 'nodes' | 'projects' | 'library' | 'settings'
const RAIL_TABS: { id: RailTab; icon: string; label: string; hint: string }[] = [
  { id: 'nodes', icon: '▣', label: '节点', hint: '节点库：搜索、分类、拖到画布' },
  { id: 'projects', icon: '▤', label: '画布', hint: '画布与模板' },
  { id: 'library', icon: '☆', label: '资料库', hint: '提示词收藏与姿势库' },
  { id: 'settings', icon: '⚙', label: '设置', hint: 'LoRA 与全局选项' },
]

/** 节点清单。以前是 16 段复制粘贴的 JSX，改一个标题要动 16 处。
 *  **顺序不能随便重排**：下游冒烟用 `.rail-add` 的索引取「图生图」，
 *  其它用例按节点名查找。分类只是在这个顺序上插小标题。 */
type NodeKindArg = NonNullable<Parameters<Props['onAddNode']>[0]>
type NodeEntry = {
  kind: NodeKindArg
  name: string
  desc: string
  title: string
  /** 搜索别名：中文说法、英文 kind、用户可能输入的同义词 */
  keys: string
  group: 0 | 1 | 2
  addTitle?: string
  /** 是否进「常用」区（ComfyUI 的 Essentials / Bookmarked） */
  fav?: boolean
}
const NODE_GROUPS = ['生成与改写', '角色 · 姿势 · 序列', '修图与像素化'] as const
const NODE_ENTRIES: NodeEntry[] = [
  {
    kind: 'gen', name: '图生成', desc: '文生图 · 提示词 → 精灵', group: 0, fav: true,
    title: '拖到画布任意位置，或点 + 放到原点',
    keys: 'text2img t2i 文生图 出图 新建 一句话',
  },
  {
    kind: 'img2img', name: '图生图', desc: '拿已有图改内容 / 改画风', group: 0, fav: true,
    title: '拖到画布任意位置，或点 + 放到原点',
    keys: 'i2i 改图 参考图 画风 重绘',
  },
  {
    kind: 'outpaint', name: '扩图', desc: '场景/背景延展', group: 0,
    title: '扩图：把图往左/右/上/下/四周扩展，新区域按场景连续性补全',
    keys: 'outpaint 外扩 画布 背景延展',
  },
  {
    kind: 'outfit', name: '换装', desc: '同角色不同装备', group: 1,
    title: '换装：把服装图的衣服穿到人物图上，保留身份与姿势',
    keys: 'outfit 服装 衣服 装备',
  },
  {
    kind: 'faceswap', name: '换脸', desc: '换主角 / 捏脸落图', group: 1,
    title: '换脸：把身份图的脸与发型换到场景人物上，保留姿势与场景',
    keys: 'face 换脸 捏脸 身份',
  },
  {
    kind: 'poseedit', name: '人偶姿势', desc: '3D 人偶 · 摆姿势导出', group: 1, fav: true,
    title: '人偶姿势：官方 VNCCS PoseStudio 的 3D 人偶，拖关节摆姿势、选朝向，导出白底灰人偶图（姿态 LoRA 的训练分布）',
    keys: 'mannequin 人偶 3d 骨骼 摆姿势 朝向',
  },
  {
    kind: 'poseclone', name: '角色迁移', desc: '参考动作 · 换角色', group: 1, fav: true,
    title: '角色迁移：动作参考图定姿态 + 角色图定身份',
    keys: 'poseclone 姿态迁移 动作 换角色',
  },
  {
    kind: 'posebatch', name: '姿态批量', desc: '姿势集 · 整套换角色', group: 1,
    title: '姿态批量：选一个姿势集（人偶编辑器抽帧存下来的整套），一次出完整套精灵帧；每帧保留所有候选，不满意单独重生成，旧的那张随时选回来',
    keys: '批量 姿势集 整套 帧 候选',
  },
  {
    kind: 'spriteswap', name: '序列帧替换', desc: '精灵图 · 逐帧换角色', group: 1,
    title: '序列帧替换：把一段游戏动画（精灵图）里的角色换成目标角色，动作保持不变',
    keys: '序列帧 精灵图 动画 逐帧',
  },
  {
    kind: 'poseswap', name: '姿态序列', desc: '骨架帧 · 逐帧生成全身', group: 1,
    title: '姿态序列：从一组骨架图逐帧生成（多候选择优 + 质量闸门），自动对齐脚底并打包成精灵图',
    keys: 'w4 骨架帧 走路帧 逐帧生成',
  },
  {
    kind: 'pixelify', name: '像素转换', desc: '网格对齐 · 每格一色', group: 2,
    title: '完美像素转换：任意图 → 每格一色、边界对齐的像素画',
    keys: 'pixel 网格 每格一色 完美像素',
  },
  {
    kind: 'topixel', name: '转像素', desc: '任意图 · 一键像素化', group: 2, fav: true,
    title: '转像素：任意图（动漫/照片/插画/3D）→ 像素画。没源图就先出图再转',
    keys: 'pixel art 像素化 像素风',
  },
  {
    kind: 'airepaint', name: '局部AI重绘', desc: '裁剪重画 · 支持 LoRA', group: 2,
    title: '局部 AI 重绘：裁剪选区交给通用模型重画，能挂 LoRA、选区外逐点不变',
    keys: '局部重绘 ai 选区 lora 裁剪',
  },
  {
    kind: 'inpaint', name: '局部重绘', desc: '笔刷选区 · 选区外不变', group: 2,
    title: '局部重绘：用笔刷涂出要重绘的区域，只改选区',
    keys: 'inpaint 笔刷 蒙版 mask',
  },
  {
    kind: 'bg', name: '抠图', desc: '去背景 / 换背景 · 不用模型', group: 2, fav: true,
    title: '抠图去背景 / 换背景；源图用连线接过来，或直接拖图片上去',
    keys: 'matte 去背景 透明底 换背景 键控',
  },
  {
    kind: 'image', name: '图片', desc: '本地图 · 可当图生图的源', group: 2,
    title: '拖一张本机图片进画布，或点 + 选择文件',
    keys: '本地 上传 素材 文件',
    addTitle: '选择本机图片',
  },
]
const FAV_ENTRIES = NODE_ENTRIES.filter((e) => e.fav)

function nodeMatches(e: NodeEntry, q: string): boolean {
  const s = q.trim().toLowerCase()
  if (!s) return true
  return `${e.name} ${e.desc} ${e.kind} ${e.keys}`.toLowerCase().includes(s)
}

/** 一个节点条目：DOM 结构、类名、拖拽协议都与重构前一致。
 *  搜索只是给它加 `filtered-out` class（CSS 隐藏），**不从 DOM 移除**——
 *  这样键盘/无障碍能保留顺序，已有冒烟用例也不受影响。 */
function NodeItem(
  { entry, hidden, onAdd }:
  { entry: NodeEntry; hidden: boolean; onAdd: Props['onAddNode'] },
) {
  return (
    <div
      className={`rail-item${hidden ? ' filtered-out' : ''}`}
      draggable
      data-node-kind={entry.kind}
      onDragStart={(ev) => {
        ev.dataTransfer.setData('application/x-node-type', entry.kind)
        ev.dataTransfer.effectAllowed = 'copy'
      }}
      title={entry.title}
    >
      <span className="rail-item-main">
        <b>{entry.name}</b>
        <em>{entry.desc}</em>
      </span>
      <button
        className="rail-add"
        title={entry.addTitle ?? '放到画布原点'}
        onClick={() => onAdd(entry.kind)}
      >
        +
      </button>
    </div>
  )
}

/** forceOpen：搜索时强制展开节点分组，避免「搜到了却因为折叠看不见」。 */
function Section(
  { icon, title, count, children, defaultOpen = true, forceOpen }:
  {
    icon: string; title: string; count?: number; children: ReactNode;
    defaultOpen?: boolean; forceOpen?: boolean
  },
) {
  const [open, setOpen] = useState(defaultOpen)
  const shown = !!forceOpen || open
  return (
    <div className={`rail-sec${shown ? ' open' : ''}`}>
      <button className="rail-sec-head" onClick={() => setOpen(!shown)}>
        <span className="rail-sec-arrow">{shown ? '▾' : '▸'}</span>
        <span className="rail-sec-icon">{icon}</span>
        <span className="rail-sec-title">{title}</span>
        {count ? <span className="rail-count">{count}</span> : null}
      </button>
      {shown && <div className="rail-sec-body">{children}</div>}
    </div>
  )
}

/** 项目行：名字可双击改成输入框；右侧有「打开文件夹」「导出」「存模板」「删除」四个动作。 */
function ProjectRow({
  p, current, onSwitch, onRename, onDelete, onExport, onTemplate, onReveal,
}: {
  p: ProjectMeta
  current: boolean
  onSwitch: () => void
  onRename: (name: string) => void
  onDelete: () => void
  onExport: () => void
  onTemplate: () => void
  /** 在系统文件管理器里打开 `assets/projects/<id>/` */
  onReveal: () => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(p.name)
  return (
    <div className={`rail-item project-item${current ? ' current' : ''}`}>
      {editing ? (
        <input
          className="project-rename"
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => { setEditing(false); onRename(draft) }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { setEditing(false); onRename(draft) }
            if (e.key === 'Escape') { setEditing(false); setDraft(p.name) }
          }}
        />
      ) : (
        <button
          className="rail-item-main"
          onClick={onSwitch}
          onDoubleClick={() => { setDraft(p.name); setEditing(true) }}
          title={`切换到「${p.name}」（双击改名）`}
        >
          <b>{p.name}</b>
          <em>{p.count} 个节点{current ? ' · 当前' : ''}</em>
        </button>
      )}
      {!editing && (
        <>
          <button
            className="rail-act"
            title="在文件管理器里打开画布文件夹（产物、上传、蒙版都在里面）"
            onClick={onReveal}
          >
            ⌂
          </button>
          <button className="rail-act" title="导出为 .mproj 单文件（含引用的素材）" onClick={onExport}>
            ↧
          </button>
          <button className="rail-act" title="存为预设模板（可反复新建画布）" onClick={onTemplate}>
            ★
          </button>
          <button className="rail-del" title="删除画布" onClick={onDelete}>✕</button>
        </>
      )}
    </div>
  )
}

/**
 * 左边栏：一级 tab 导航 + 独立面板，整条可收起成图标条。
 *
 * **兼容性约定**：6 个 `.rail-sec` 的 DOM 顺序仍是
 * 「项目 / 模板 / 节点 / 资料库 / LoRA / 设置」。切 tab 只用 class 控制
 * `display:none`，不条件卸载——jsdom 冒烟用例仍能在任何 tab 下查询和点击
 * 面板里的按钮（真实浏览器里不可见，但 DOM 始终完整）。
 */
export default function Sidebar({
  favs, onAddNode, canApply, onApplyFav, onRemoveFav, onOpenLibrary,
  matteBackend, onMatteBackend,
  loras, activeLoras, onLorasChange, fastMode, onFastMode,
  baseModel, onLorasChanged,
  lorasSkipped, lorasDir, lorasCount, onRefreshLoras,
  projects, currentProject, onSwitchProject, onCreateProject, onRenameProject, onDeleteProject,
  onExportProject, onRevealProject, onSaveTemplate, onImportProject,
  templates, onUseTemplate, onDeleteTemplate,
}: Props) {
  const [collapsed, setCollapsed] = useState(false)
  const [tab, setTab] = useState<RailTab>('nodes')
  const [query, setQuery] = useState('')

  const importRef = useRef<HTMLInputElement | null>(null)
  const searchRef = useRef<HTMLInputElement | null>(null)

  /** Ctrl/⌘+K 唤起节点搜索（n8n Command Bar / ComfyUI 节点库搜索的通行做法）。
   *  用 setTimeout 而不是 requestAnimationFrame：jsdom 里 rAF 可能不存在。 */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return
      if (String(e.key ?? '').toLowerCase() !== 'k') return
      e.preventDefault()
      setCollapsed(false)
      setTab('nodes')
      window.setTimeout(() => searchRef.current?.focus(), 0)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const matchedKinds = useMemo(
    () => new Set(NODE_ENTRIES.filter((e) => nodeMatches(e, query)).map((e) => e.kind)),
    [query],
  )
  const matchCount = matchedKinds.size
  const current = projects.find((p) => p.id === currentProject)
  const searching = query.trim().length > 0

  if (collapsed) {
    return (
      <aside className="rail collapsed" data-tab={tab}>
        <button className="rail-toggle" title="展开侧栏" onClick={() => setCollapsed(false)}>
          »
        </button>
        <button className="rail-icon" title="新建图生成节点" onClick={() => onAddNode('gen')}>
          +
        </button>
        {RAIL_TABS.map((t) => (
          <button
            key={t.id}
            className={`rail-icon${tab === t.id ? ' on' : ''}`}
            title={t.hint}
            onClick={() => { setTab(t.id); setCollapsed(false) }}
          >
            {t.icon}
          </button>
        ))}
      </aside>
    )
  }

  const pageClass = (id: RailTab) => `rail-page${tab === id ? '' : ' is-hidden'}`

  return (
    <aside className="rail" data-tab={tab}>
      <div className="rail-head">
        <span className="rail-brand">工作台</span>
        <span className="spacer" />
        <button className="rail-toggle" title="收起侧栏" onClick={() => setCollapsed(true)}>
          «
        </button>
      </div>

      {/* 一级导航：Activity Bar。每个 tab 一块独立面板，不再共用一条长滚动条。 */}
      <div className="rail-tabs" role="tablist">
        {RAIL_TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={`rail-tab${tab === t.id ? ' on' : ''}`}
            title={t.hint}
            onClick={() => setTab(t.id)}
          >
            <span className="rail-tab-icon">{t.icon}</span>
            <span className="rail-tab-label">{t.label}</span>
            {t.id === 'library' && favs.length > 0
              ? <span className="rail-tab-badge">{favs.length}</span>
              : null}
          </button>
        ))}
      </div>

      {/* 节点搜索：常驻 DOM（其它 tab 只是 CSS 隐藏），Ctrl+K 才能稳定聚焦。 */}
      <div className={`rail-search${tab === 'nodes' ? '' : ' is-hidden'}`}>
        <span className="rail-search-icon" aria-hidden="true">⌕</span>
        <input
          ref={searchRef}
          className="rail-search-input"
          type="search"
          value={query}
          placeholder="搜索节点…"
          aria-label="搜索节点"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              const first = NODE_ENTRIES.find((it) => nodeMatches(it, query))
              if (first) {
                onAddNode(first.kind)
                setQuery('')
                e.currentTarget.blur()
              }
            }
            if (e.key === 'Escape') { setQuery(''); e.currentTarget.blur() }
          }}
        />
        {searching
          ? <button className="rail-search-clear" title="清空搜索" onClick={() => setQuery('')}>✕</button>
          : <kbd className="rail-search-kbd">Ctrl K</kbd>}
      </div>

      <div className="rail-scroll">
        {/* ---------- 面板 1：项目 + 模板（保持 Section 顺序第一、第二） ---------- */}
        <div className={pageClass('projects')} data-page="projects">
          <Section icon="▤" title="画布" count={projects.length}>
            {projects.map((p) => (
              <ProjectRow
                key={p.id}
                p={p}
                current={p.id === currentProject}
                onSwitch={() => onSwitchProject(p.id)}
                onRename={(name) => onRenameProject(p.id, name)}
                onDelete={() => onDeleteProject(p.id)}
                onExport={() => onExportProject(p.id)}
                onTemplate={() => onSaveTemplate(p.id)}
                onReveal={() => onRevealProject(p.id)}
              />
            ))}
            <button className="rail-new" onClick={onCreateProject} title="新建一个空画布">
              + 新建画布
            </button>
            <button
              className="rail-new"
              onClick={() => importRef.current?.click()}
              title="导入 .mproj 归档（别人发来的画布 / 换台机器继续做）"
            >
              ↑ 导入画布
            </button>
            <input
              ref={importRef}
              type="file"
              accept=".mproj"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0]
                // **清空 value**：同一个文件连续选两次不会触发 change，
                // 用户会以为「导入没反应」。
                e.target.value = ''
                if (f) onImportProject(f)
              }}
            />
            <p className="rail-tip">每张画布是一套独立的节点与参数，互不干涉。</p>
          </Section>

          <Section icon="★" title="模板" count={templates.length}>
            {templates.length === 0 ? (
              <p className="rail-tip">
                在画布上点 <b>★</b> 存成模板。之后点模板就能一键开一张
                同样的新画布——同一套产线要开好几张画布时用得上。
              </p>
            ) : (
              templates.map((t) => (
                <div className="rail-item" key={t.id}>
                  <button
                    className="rail-item-main"
                    onClick={() => onUseTemplate(t.id)}
                    title={`从「${t.name}」新建一张画布`}
                  >
                    <b>{t.name}</b>
                    <em>{t.nodes} 个节点</em>
                  </button>
                  <button className="rail-del" title="删除模板" onClick={() => onDeleteTemplate(t.id)}>
                    ✕
                  </button>
                </div>
              ))
            )}
          </Section>
        </div>

        {/* ---------- 面板 2：节点库（常用置顶 → 搜索结果 → 分类列表） ---------- */}
        <div className={pageClass('nodes')} data-page="nodes">
          <Section
            icon="▣"
            title="节点"
            count={NODE_ENTRIES.length}
            forceOpen={searching}
          >
            {/* ComfyUI 的 Essentials / Bookmarked：高频节点不用往下翻。 */}
            {/* 搜索时只给结果：常用区与底部长说明都收起来，否则「搜到了 1 个」
                却仍铺着 6 个常用按钮，看起来像搜索没生效。 */}
            <div className={`rail-fav${searching ? ' is-hidden' : ''}`}>
              <div className="rail-group-label">常用<span className="rail-group-count">{FAV_ENTRIES.length}</span></div>
              <div className="rail-fav-grid">
                {FAV_ENTRIES.map((e) => (
                  <button
                    key={e.kind}
                    className="rail-fav-chip"
                    title={`${e.name}：${e.desc}（点一下添加）`}
                    onClick={() => onAddNode(e.kind)}
                  >
                    <span className="rail-fav-dot" aria-hidden="true" />
                    {e.name}
                  </button>
                ))}
              </div>
            </div>

            {searching && matchCount === 0 && (
              <div className="rail-no-result">
                没有匹配「{query}」的节点。<br />
                试试「抠图」「姿势」「像素」，或直接输英文 kind（如 inpaint）。
              </div>
            )}
            {/* n8n 的 Enter 插入：不写出来的话没人知道这个入口存在。 */}
            {searching && matchCount > 0 && (
              <div className="rail-search-hint">
                找到 {matchCount} 个节点 · 按 <kbd>↵</kbd> 添加第一个
              </div>
            )}

            {NODE_GROUPS.map((label, gi) => {
              const items = NODE_ENTRIES.filter((e) => e.group === gi)
              const shown = items.filter((e) => matchedKinds.has(e.kind)).length
              return (
                <div key={label} className={`rail-group${shown ? '' : ' filtered-out'}`}>
                  <div className="rail-group-label">
                    {label}<span className="rail-group-count">{shown}</span>
                  </div>
                  {items.map((e) => (
                    <NodeItem
                      key={e.kind}
                      entry={e}
                      hidden={!matchedKinds.has(e.kind)}
                      onAdd={onAddNode}
                    />
                  ))}
                </div>
              )
            })}

            <p className={`rail-tip${searching ? ' is-hidden' : ''}`}>
              拖这一项到画布任意位置，也能在那里新建节点。图生图要先有源图：
              在生成节点下方点「↳ 图生图」会自动接上。
              <br />
              把本机图片<b>直接拖进画布</b>会变成图片节点；拖到图生图节点上则直接当它的源。
            </p>
          </Section>
        </div>

        {/* ---------- 面板 3：资料库 ---------- */}
        <div className={pageClass('library')} data-page="library">
          <Section icon="☆" title="资料库" count={favs.length}>
            <div className="rail-lib">
              <button className="rail-item-main" onClick={() => onOpenLibrary('prompt')}>
                提示词收藏 <b>{favs.length}</b>
                <em>浏览 / 搜索 / 填入选中节点</em>
              </button>
              <button className="rail-item-main" onClick={() => onOpenLibrary('pose')}>
                姿势库
                <em>复用官方 3D 人偶姿势</em>
              </button>
            </div>
            {favs.length === 0 ? (
              <p className="rail-tip">
                在节点的提示词框点 <b>☆</b> 收藏。收藏后可以在这里填回选中的节点。
              </p>
            ) : (
              favs.slice(0, 4).map((f) => (
                <div className="rail-item" key={f.id}>
                  <button
                    className="rail-item-main"
                    disabled={!canApply}
                    title={canApply
                      ? '填入选中节点（正向与负向都会覆盖）'
                      : '先在画布上选中一个节点'}
                    onClick={() => onApplyFav(f)}
                  >
                    {f.text}
                    {/* 负向词跟着正向一起收、一起填；缩成第二行灰字，
                        不抢正向的位置。没有负向就不占行。 */}
                    {f.negative ? <em>− {f.negative}</em> : null}
                  </button>
                  <button className="rail-del" title="删除" onClick={() => onRemoveFav(f.id)}>
                    ✕
                  </button>
                </div>
              ))
            )}
            {favs.length > 4 && (
              <button className="rail-new" onClick={() => onOpenLibrary('prompt')}>
                看全部 {favs.length} 条 →
              </button>
            )}
          </Section>
        </div>

        {/* ---------- 面板 4：LoRA + 设置（保持 Section 顺序第五、第六） ---------- */}
        <div className={pageClass('settings')} data-page="settings">
          <Section icon="◈" title="LoRA" count={loras.length}>
            <LoraSettings
              loras={loras}
              activeLoras={activeLoras}
              onLorasChange={onLorasChange}
              fastMode={fastMode}
              onFastMode={onFastMode}
              onLorasChanged={onLorasChanged}
              skipped={lorasSkipped}
              count={lorasCount}
              dir={lorasDir}
              onRefresh={onRefreshLoras} />
          </Section>

          <Section icon="⚙" title="设置">
            <GeneralSettings
              baseModel={baseModel}
              matteBackend={matteBackend}
              onMatteBackend={onMatteBackend} />
          </Section>
        </div>
      </div>

      {/* 状态条：一眼看到「我现在在哪个项目」，以及 Ctrl+K 这个入口。 */}
      <div className="rail-status">
        <span className="rail-status-dot" aria-hidden="true" />
        <span className="rail-status-name" title={current?.name}>{current?.name ?? '未命名画布'}</span>
        <span className="rail-status-count">{current?.count ?? 0} 节点</span>
      </div>
    </aside>
  )
}
