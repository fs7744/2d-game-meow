import { useState, type ReactNode } from 'react'
import type { FavoritePrompt, LoraInfo, ModelRecommend } from '../types'
import type { ProjectMeta } from '../store/projects'

type Props = {
  loras: LoraInfo[]
  favs: FavoritePrompt[]
  projects: ProjectMeta[]
  currentProject: string
  onSwitchProject: (id: string) => void
  onCreateProject: () => void
  onRenameProject: (id: string, name: string) => void
  onDeleteProject: (id: string) => void
  onAddNode: (
    kind?: 'gen' | 'img2img' | 'image' | 'bg' | 'inpaint' | 'airepaint' | 'pixelify'
    | 'poseclone' | 'spriteswap' | 'poseswap',
  ) => void
  canApply: boolean
  onApplyFav: (text: string) => void
  onRemoveFav: (id: string) => void
  models: string[]
  recommend: Record<string, ModelRecommend>
  model: string
  onModel: (m: string) => void
  matteBackend: string
  onMatteBackend: (v: string) => void
}

const MODEL_LABEL: Record<string, string> = {
  sd15: 'SD1.5 · 省显存（默认）',
  sdxl: 'SDXL 1.0 · 结构更稳',
}

const MATTE_LABEL: Record<string, string> = {
  auto: '自动（有分割模型就抠得干净）',
  u2net: '语义分割 · 准但慢',
  color: '纯色键控 · 极快',
}

function Section(
  { icon, title, count, children, defaultOpen = true }:
  { icon: string; title: string; count?: number; children: ReactNode; defaultOpen?: boolean },
) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`rail-sec${open ? ' open' : ''}`}>
      <button className="rail-sec-head" onClick={() => setOpen(!open)}>
        <span className="rail-sec-arrow">{open ? '▾' : '▸'}</span>
        <span className="rail-sec-icon">{icon}</span>
        <span className="rail-sec-title">{title}</span>
        {count ? <span className="rail-count">{count}</span> : null}
      </button>
      {open && <div className="rail-sec-body">{children}</div>}
    </div>
  )
}

/** 项目行：名字可双击改成输入框。 */
function ProjectRow({
  p, current, onSwitch, onRename, onDelete,
}: {
  p: ProjectMeta
  current: boolean
  onSwitch: () => void
  onRename: (name: string) => void
  onDelete: () => void
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
        <button className="rail-del" title="删除项目" onClick={onDelete}>✕</button>
      )}
    </div>
  )
}

/**
 * 左边栏：菜单式层级，可整条收起。
 *
 * 之前节点列表和提示词收藏是两个各自浮在左上角的卡片，节点一多就把画布左上角
 * 占满、互相挤压。改成一条可收起的侧栏后，画布的可视区域稳定了，也能放下设置。
 */
export default function Sidebar({
  loras, favs, onAddNode, canApply, onApplyFav, onRemoveFav,
  models, recommend, model, onModel, matteBackend, onMatteBackend,
  projects, currentProject, onSwitchProject, onCreateProject, onRenameProject, onDeleteProject,
}: Props) {
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <aside className="rail collapsed">
        <button className="rail-toggle" title="展开侧栏" onClick={() => setCollapsed(false)}>
          »
        </button>
        <button className="rail-icon" title="图生成" onClick={() => { setCollapsed(false); onAddNode('gen') }}>
          ▣
        </button>
        <button className="rail-icon" title="图生图" onClick={() => { setCollapsed(false); onAddNode('img2img') }}>
          ↳
        </button>
        <button className="rail-icon" title={`提示词收藏 ${favs.length}`} onClick={() => setCollapsed(false)}>
          ☆{favs.length ? <sup>{favs.length}</sup> : null}
        </button>
        <button className="rail-icon" title={`底模 ${model}`} onClick={() => setCollapsed(false)}>
          ⚙
        </button>
      </aside>
    )
  }

  return (
    <aside className="rail">
      <div className="rail-head">
        <span className="rail-brand">面板</span>
        <span className="spacer" />
        <button className="rail-toggle" title="收起侧栏" onClick={() => setCollapsed(true)}>
          «
        </button>
      </div>

      <div className="rail-scroll">
        <Section icon="▤" title="项目" count={projects.length}>
          {projects.map((p) => (
            <ProjectRow
              key={p.id}
              p={p}
              current={p.id === currentProject}
              onSwitch={() => onSwitchProject(p.id)}
              onRename={(name) => onRenameProject(p.id, name)}
              onDelete={() => onDeleteProject(p.id)}
            />
          ))}
          <button className="rail-new" onClick={onCreateProject} title="新建一个空画布">
            + 新建项目
          </button>
          <p className="rail-tip">每个项目是一套独立的画布，互不干涉。</p>
        </Section>

        <Section icon="◈" title="底模">
          <label className="block">
            底模
            <select value={model} onChange={(e) => onModel(e.target.value)}>
              {models.length === 0 && <option value={model}>{model}（仅本机已装）</option>}
              {models.map((m) => (
                <option key={m} value={m}>{MODEL_LABEL[m] || m}</option>
              ))}
            </select>
          </label>
          {/* 底模影响画布上**所有**节点，不是单个节点的参数——这一点必须写明，
              否则用户会以为只改了当前节点。引擎按 model 缓存管线，
              两个底模同时驻留要多花十几 GB。 */}
          <p className="rail-tip">全局生效：画布上所有节点都用它（引擎按底模缓存管线，不混跑）。</p>
          <p className="rail-tip">
            {model === 'sdxl'
              ? '结构更稳（实测可交付率 92% vs 33%），同分辨率只慢约 40%；但 W4 显存很紧（单帧约 15GB），且 SD1.5 的 LoRA 不能通用。'
              : '显存友好（W4 单帧约 6.6GB）、LoRA 生态全；代价是构图控制弱，需要多出候选让闸门挑。'}
          </p>
          {recommend[model] && (
            <p className="rail-tip">
              推荐 {recommend[model].width}×{recommend[model].height} /{' '}
              {recommend[model].steps} 步（切底模时自动套用，手改过的节点不动）
            </p>
          )}
        </Section>

        <Section icon="▣" title="节点">
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'gen')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="拖到画布任意位置，或点 + 放到原点"
          >
            <span className="rail-item-main">
              <b>图生成</b>
              <em>文生图 · 提示词 → 精灵</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('gen')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'img2img')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="拖到画布任意位置，或点 + 放到原点"
          >
            <span className="rail-item-main">
              <b>图生图</b>
              <em>拿已有图改内容 / 改画风</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('img2img')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'poseclone')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="角色迁移：动作参考图定姿态 + 角色图定身份"
          >
            <span className="rail-item-main">
              <b>角色迁移</b>
              <em>参考动作 · 换角色</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('poseclone')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'spriteswap')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="序列帧替换：把一段游戏动画（精灵图）里的角色换成目标角色，动作保持不变"
          >
            <span className="rail-item-main">
              <b>序列帧替换</b>
              <em>精灵图 · 逐帧换角色</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('spriteswap')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'poseswap')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="姿态序列：从一组骨架图逐帧生成（多候选择优 + 质量闸门），自动对齐脚底并打包成精灵图"
          >
            <span className="rail-item-main">
              <b>姿态序列</b>
              <em>骨架帧 · 逐帧生成全身</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('poseswap')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'pixelify')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="完美像素转换：任意图 → 每格一色、边界对齐的像素画"
          >
            <span className="rail-item-main">
              <b>像素转换</b>
              <em>网格对齐 · 每格一色</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('pixelify')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'airepaint')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="局部 AI 重绘：裁剪选区交给通用模型重画，能挂 LoRA、选区外逐点不变"
          >
            <span className="rail-item-main">
              <b>局部AI重绘</b>
              <em>裁剪重画 · 支持 LoRA</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('airepaint')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'inpaint')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="局部重绘：用笔刷涂出要重绘的区域，只改选区"
          >
            <span className="rail-item-main">
              <b>局部重绘</b>
              <em>笔刷选区 · 选区外不变</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('inpaint')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'bg')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="抠图去背景 / 换背景；源图用连线接过来，或直接拖图片上去"
          >
            <span className="rail-item-main">
              <b>抠图</b>
              <em>去背景 / 换背景 · 不用模型</em>
            </span>
            <button className="rail-add" title="放到画布原点" onClick={() => onAddNode('bg')}>+</button>
          </div>
          <div
            className="rail-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/x-node-type', 'image')
              e.dataTransfer.effectAllowed = 'copy'
            }}
            title="拖一张本机图片进画布，或点 + 选择文件"
          >
            <span className="rail-item-main">
              <b>图片</b>
              <em>本地图 · 可当图生图的源</em>
            </span>
            <button className="rail-add" title="选择本机图片" onClick={() => onAddNode('image')}>+</button>
          </div>
          <p className="rail-tip">
            拖这一项到画布任意位置，也能在那里新建节点。图生图要先有源图：
            在生成节点下方点「↳ 图生图」会自动接上。
            <br />
            把本机图片<b>直接拖进画布</b>会变成图片节点；拖到图生图节点上则直接当它的源。
          </p>
        </Section>

        <Section icon="☆" title="提示词收藏" count={favs.length}>
          {favs.length === 0 ? (
            <p className="rail-tip">
              在节点的提示词框点 <b>☆</b> 收藏。收藏后点这里可填回选中的节点。
            </p>
          ) : (
            favs.map((f) => (
              <div className="rail-item" key={f.id}>
                <button
                  className="rail-item-main"
                  disabled={!canApply}
                  title={canApply ? '填入选中的节点' : '先在画布上选中一个节点'}
                  onClick={() => onApplyFav(f.text)}
                >
                  {f.text}
                </button>
                <button className="rail-del" title="删除" onClick={() => onRemoveFav(f.id)}>
                  ✕
                </button>
              </div>
            ))
          )}
        </Section>

        <Section icon="⚙" title="设置">
          <label className="block">
            抠图方式
            <select value={matteBackend} onChange={(e) => onMatteBackend(e.target.value)}>
              {Object.keys(MATTE_LABEL).map((k) => (
                <option key={k} value={k}>{MATTE_LABEL[k]}</option>
              ))}
            </select>
          </label>

          <div className="rail-stat">
            LoRA {loras.length} 个
          </div>
        </Section>
      </div>
    </aside>
  )
}
