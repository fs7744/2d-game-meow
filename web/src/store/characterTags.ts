/**
 * 角色属性的标签选择器（移植自官方 `web/character_presets.mjs`）。
 *
 * 官方 Character Creator 里那一排「选择一个特征」的弹窗就是这两个函数驱动的：
 *
 * - `presetGroups`：一个属性字段（如 `hair`）对应**若干组标签**
 *   （hair_color / hair_length / hairstyles / …），每组再按 `item.group` 二级分组。
 * - `presetSelection`：把字段里现有的文本（逗号分隔）解析成"选中集合"，
 *   toggle 之后再拼回文本。
 *
 * **两个必须照搬的细节**（改动会让官文献变的 flavor 直接跑偏）：
 *
 * 1. `toggle` 写进字段的是 `item.tag.replaceAll('_', ' ')` —— **空格形式**，
 *    不是下划线 tag。所以 prompt 里出现的是 "blue eyes" 而不是 "blue_eyes"。
 * 2. 已选项用 **tag / label / synonyms 三种别名**匹配，且**保留用户自定义文本**：
 *    `aliases.get(key) || key`，没匹上的 token 原样留着。
 *    否则用户手写的描述会被"选一次标签"整段清掉。
 *
 * 对照官方版本去掉的部分：`LEGACY_FIELD_CATEGORIES` 与 hair 的 legacy 兼容分支。
 * 那两个服务于 SDXL 时代的 `character_tags.json`（我们没入库，那在国内也没有
 * 对应的模型了）。我们只用 QI2 的 `character_presets_v2.json`
 * （官方 `vnccs_character_creator_v2.js:4000`：`mode === 'qi2'` 时强制 creator_v2）。
 */

export type TagItem = {
  tag: string
  label?: string
  /** 写进 prompt 的完整描述（只有种族这类需要额外解释的才有） */
  prompt?: string
  /** 二级分组标题，如 "Humanoid peoples" */
  group?: string
  synonyms?: string[]
  count?: number
}

export type TagGroup = { header: string; items: TagItem[] }

export type TagCatalog = { tags?: Record<string, TagItem[] | Record<string, TagItem[]>> }

/** 字段 → 标签分类。官方 `FIELD_CATEGORIES`（character_presets.mjs:1-9）。 */
const FIELD_CATEGORIES: Record<string, string[]> = {
  race: ['races'],
  skin_color: ['skin_color'],
  body: ['body_type', 'breast_size'],
  face: ['face_shape', 'face_details'],
  hair: ['hair_color', 'hair_pattern', 'hair_length', 'hair_texture', 'hairstyles', 'hair_framing'],
  eyes: ['eye_color', 'eye_features'],
  additional_details: ['details'],
}

/**
 * 某字段的标签分组（官方 `presetGroups` + 调用处的两个特例，:4018）。
 *
 * `body` 弹窗只显示体型、`breast_size` 弹窗只显示胸围——两边分开，
 * 但它们其实共用一个 `body_type`/`breast_size` 分类，所以这里各取一半。
 */
export function presetGroups(catalog: TagCatalog | null, field: string): TagGroup[] {
  // 官方把这两个做成调用处的特例（`field === 'body' ? 'body_type' : ...`）
  const categories = field === 'body' ? ['body_type']
    : field === 'breast_size' ? ['breast_size']
      : FIELD_CATEGORIES[field]
  if (!categories) return []
  const groups: TagGroup[] = []
  for (const category of categories) {
    const items = (catalog?.tags as Record<string, TagItem[]>)?.[category]
    if (!items) continue
    const byGroup = new Map<string, TagItem[]>()
    for (const item of items) {
      const header = item.group || category.split('_').join(' ')
      const list = byGroup.get(header) ?? []
      list.push(item)
      byGroup.set(header, list)
    }
    for (const [header, list] of byGroup) groups.push({ header, items: list })
  }
  return groups
}

/** 归一化：下划线换空格 + 小写 + 去多余空格（官方 `presetKey`）。 */
const presetKey = (value: string) =>
  String(value).split('_').join(' ').toLowerCase().trim().replace(/\s+/g, ' ')

export type PresetSelection = {
  has: (item: TagItem) => boolean
  toggle: (item: TagItem) => void
  value: () => string
}

/**
 * 把字段现有文本解析成选中集合（官方 `presetSelection`）。
 *
 * 现有文本是**逗号分隔的多个 trait**（"blue eyes, freckles"）；
 * `value()` 再拼回成同一种写法，跟后端 ok 的字段格式对得上。
 */
export function presetSelection(value: string, groups: TagGroup[]): PresetSelection {
  const aliases = new Map<string, string>()
  for (const { items } of groups) {
    for (const item of items) {
      for (const alias of [item.tag, item.label, ...(item.synonyms || [])]) {
        if (alias) aliases.set(presetKey(alias), presetKey(item.tag))
      }
    }
  }
  // key → 用户原本写的那个词。留原词是为了**不要偷偷改写用户手写的描述**。
  const selected = new Map<string, string>()
  for (const token of String(value || '').split(',').map((p) => p.trim()).filter(Boolean)) {
    const key = aliases.get(presetKey(token)) || presetKey(token)
    if (!selected.has(key)) selected.set(key, token)
  }
  return {
    has: (item) => selected.has(presetKey(item.tag)),
    toggle: (item) => {
      const key = presetKey(item.tag)
      if (selected.has(key)) selected.delete(key)
      else selected.set(key, item.tag.split('_').join(' '))
    },
    value: () => [...selected.values()].join(', '),
  }
}
