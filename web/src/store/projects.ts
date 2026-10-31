/**
 * 项目（= 独立画布）的本地存储。
 *
 * 一个项目就是一套自己的节点 + 自己的视口，切项目互不干涉。
 * 全部存在 localStorage 一个键里（原子读写，不会出现「切了一半」的中间态）。
 *
 * 老存档（meow.canvas.v1，只有一份 nodes+viewport）会自动迁移成第一个项目，
 * 不丢用户已有的画布。
 */
import type { GenNode, Viewport } from '../types'

export type Project = {
  id: string
  name: string
  created: string
  updated: string
  nodes: GenNode[]
  viewport: Viewport
}

/** 侧栏列表用的摘要（不带节点，省得每次渲染都序列化几千个节点） */
export type ProjectMeta = {
  id: string
  name: string
  created: string
  updated: string
  count: number
}

const KEY = 'meow.projects.v2'
const LEGACY_KEY = 'meow.canvas.v1'
const DEFAULT_VIEWPORT: Viewport = { x: 120, y: 90, z: 1 }

let seq = 0
const newId = () => `p${Date.now().toString(36)}${(seq++).toString(36)}`

export function makeProject(
  name: string,
  nodes: GenNode[] = [],
  viewport: Viewport = DEFAULT_VIEWPORT,
): Project {
  const now = new Date().toISOString()
  return { id: newId(), name, created: now, updated: now, nodes, viewport }
}

function normalize(p: Partial<Project>): Project {
  const now = new Date().toISOString()
  return {
    id: p.id || newId(),
    name: p.name || '未命名项目',
    created: p.created || now,
    updated: p.updated || now,
    nodes: Array.isArray(p.nodes) ? p.nodes : [],
    viewport: p.viewport && typeof p.viewport.z === 'number' ? p.viewport : DEFAULT_VIEWPORT,
  }
}

/** 读取全部项目；没有就新建一个默认项目（并顺手迁移老存档）。 */
export function loadStore(): { current: string; projects: Project[] } {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) {
      const s = JSON.parse(raw)
      if (s && Array.isArray(s.projects) && s.projects.length > 0) {
        const projects = (s.projects as Partial<Project>[]).map(normalize)
        const current = projects.some((p) => p.id === s.current) ? s.current : projects[0].id
        return { current, projects }
      }
    }
  } catch {
    /* 存档损坏就当没有 */
  }

  // 老存档迁移：只有一份 nodes + viewport 的时代
  let nodes: GenNode[] = []
  let viewport: Viewport = DEFAULT_VIEWPORT
  try {
    const raw = localStorage.getItem(LEGACY_KEY)
    if (raw) {
      const s = JSON.parse(raw)
      if (Array.isArray(s.nodes)) nodes = s.nodes
      if (s.viewport && typeof s.viewport.z === 'number') viewport = s.viewport
    }
  } catch {
    /* ignore */
  }
  const p = makeProject('我的项目', nodes, viewport)
  return { current: p.id, projects: [p] }
}

export function saveStore(current: string, projects: Project[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify({ current, projects }))
  } catch {
    /* 存储满了就放弃：不能因为存不下就把用户的画布搞坏 */
  }
}

export function metaOf(p: Project): ProjectMeta {
  return {
    id: p.id,
    name: p.name,
    created: p.created,
    updated: p.updated,
    count: p.nodes.length,
  }
}

/**
 * 把项目打成「可导出」的归档内容。
 *
 * **为什么 meta 和 payload 分开**：meta 是项目自己的元信息（名字、创建时间），
 * payload 是主体数据（节点、视口）。后端要靠 payload 里引用的素材路径决定
 * 打包哪些图，分开才能只扫 payload。
 */
export function archiveOf(p: Project): {
  meta: Record<string, unknown>
  payload: Record<string, unknown>
} {
  return {
    meta: {
      name: p.name,
      created: p.created,
      updated: p.updated,
    },
    payload: {
      nodes: p.nodes,
      viewport: p.viewport,
    },
  }
}

/**
 * 从导入/模板实例化出来的内容，还原成一个**新项目**。
 *
 * **为什么要换 id**：导入的项目必须是新项目，不能覆盖已有的——
 * 素材落在 `assets/projects/<id>/`，用同一个 id 会跟原项目的产物混在一起。
 */
export function projectFromImported(
  imp: { meta: Record<string, unknown>; payload: Record<string, unknown> },
  fallbackName: string,
): Project {
  const now = new Date().toISOString()
  const meta = imp.meta || {}
  const payload = imp.payload || {}
  const nodes = Array.isArray(payload.nodes) ? (payload.nodes as GenNode[]) : []
  const viewport = (payload.viewport as Viewport) || DEFAULT_VIEWPORT
  return {
    id: newId(),
    name: String(meta.name || fallbackName || '导入的项目'),
    created: String(meta.created || now),
    updated: now,
    nodes,
    viewport,
  }
}
