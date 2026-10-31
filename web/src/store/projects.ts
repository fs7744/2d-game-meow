/**
 * 项目（= 独立画布）的存储。
 *
 * 一个项目就是一套自己的节点 + 自己的视口，切项目互不干涉。
 *
 * **项目内容只存在服务端磁盘上**（`assets/projects/<id>/project.json`）。
 *
 * **浏览器里什么都不存**——不存 localStorage、不存 sessionStorage、
 * 不写 IndexedDB。之前把整个项目（几千个节点的 JSON）塞进 localStorage，
 * 问题很实际：
 *
 *   · 容量就 5MB 左右，一个大项目直接写不进去（`saveStore` 只能默默吞掉
 *     异常，用户毫无察觉，改动全丢）；
 *   · 换了浏览器 / 换了机器 / 清一次缓存，项目就没了；
 *   · 每次自动保存都要把整个项目序列化一遍塞进去，大项目会卡顿；
 *   · 浏览器里存着项目的明文 JSON，出于任何理由都不合适。
 *
 * 现在浏览器只保留**极小的界面状态**（提示词收藏之类），项目一律走服务端。
 * 老存档（meow.canvas.v1）只在**首次迁移**时读一次，迁到服务端后就删掉。
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

const DEFAULT_VIEWPORT: Viewport = { x: 120, y: 90, z: 1 }

let seq = 0
const newId = () => `p${Date.now().toString(36)}${(seq++).toString(36)}`

/**
 * 造一个新项目。
 *
 * **id 建议由服务端分配**（`allocateProjectId()`）：本地用「时间戳 + 序号」
 * 生成的话，两个标签页在同一毫秒建项目会拿到完全相同的 id（两边序号都从 0
 * 开始），后保存的直接覆盖先保存的——用户的画布无声消失。
 *
 * 传了 `id` 就用它；没传才本地生成（只在离线兜底时用）。
 */
export function makeProject(
  name: string,
  nodes: GenNode[] = [],
  viewport: Viewport = DEFAULT_VIEWPORT,
  id?: string,
): Project {
  const now = new Date().toISOString()
  return { id: id || newId(), name, created: now, updated: now, nodes, viewport }
}

/**
 * 老存档迁移：读一次 `meow.canvas.v1`（只有一份 nodes+viewport 的时代）。
 *
 * **只在迁移时读**：内容交给服务端保存后，浏览器里那份就该删掉——
 * 留着就等于「项目内容还存了一份在浏览器」。
 *
 * @returns 迁移用的项目；没有老存档则返回 null。
 */
export function takeLegacyProject(): Project | null {
  const LEGACY_KEY = 'meow.canvas.v1'
  try {
    const raw = localStorage.getItem(LEGACY_KEY)
    if (!raw) return null
    const s = JSON.parse(raw)
    const nodes: GenNode[] = Array.isArray(s.nodes) ? s.nodes : []
    const viewport: Viewport = (s.viewport && typeof s.viewport.z === 'number')
      ? s.viewport : DEFAULT_VIEWPORT
    // 读完就删：内容已经在服务端了，浏览器不留副本
    localStorage.removeItem(LEGACY_KEY)
    if (!nodes.length) return null
    return makeProject('我的项目', nodes, viewport)
  } catch {
    try { localStorage.removeItem(LEGACY_KEY) } catch { /* ignore */ }
    return null
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
 * **id 必须由调用方指定**：素材已经落在 `assets/projects/<id>/imported/`
 * 了（后端按这个 id 解包的），项目定义也写在这个 id 下。前端若另起一个 id，
 * 素材和项目定义就分家了——产物写到 A 目录，引用图在 B 目录。
 */
export function projectFromImported(
  imp: { meta: Record<string, unknown>; payload: Record<string, unknown> },
  id: string,
  fallbackName: string,
): Project {
  const now = new Date().toISOString()
  const meta = imp.meta || {}
  const payload = imp.payload || {}
  const nodes = Array.isArray(payload.nodes) ? (payload.nodes as GenNode[]) : []
  const viewport = (payload.viewport as Viewport) || DEFAULT_VIEWPORT
  return {
    id,
    name: String(meta.name || fallbackName || '导入的项目'),
    created: String(meta.created || now),
    updated: now,
    nodes,
    viewport,
  }
}
