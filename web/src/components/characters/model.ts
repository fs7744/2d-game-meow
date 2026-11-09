/**
 * 角色中心的「数据模型 + 文案」层。
 *
 * 这里只放常量和纯函数，不放 React 状态。重构前这些东西散在 1900 行的
 * CharactersPage.tsx 里，改一句文案要翻半天；现在集中在一处。
 *
 * 交互范式取自主流 AI 游戏素材平台的共识（不绑定某一家）：
 *   1. 角色是**带持久 ID 的对象**，不是一段提示词 —— 所有后续产物都挂在它下面；
 *   2. 用户面对的是**一条唯一的线性主线**，不是一堆并列的 tab；
 *   3. **结果优先**：出图成果常驻可见，任何一步都知道"我现在有什么"；
 *   4. **结构化选择题**优先于自由输入框，专业参数渐进披露；
 *   5. 把**能力边界与代价**（要多久、需要什么前置）写在按钮旁边，而不是报错里。
 *
 * ⚠️ 中文化规矩：下面所有 key / value（race、Naked、neutral…）都是数据标识，
 * 同时是磁盘目录名、接口键和写进 prompt 的内容。**值传原样英文，只在渲染处显示中文**。
 */

/** 角色属性字段（官方 character_creator_v2.py:2138-2146 的字段清单）。
 *  第三列是给用户看的中文说明，不再让用户对着中文界面猜"该填什么英文"。 */
export const INFO_FIELDS: Array<[key: string, label: string, hint: string]> = [
  ['race', '种族', '例：human / elf / demon'],
  ['skin_color', '肤色', '例：fair skin'],
  ['hair', '发型发色', '例：blonde hair, twin tails'],
  ['eyes', '眼睛', '例：blue eyes'],
  ['face', '脸部特征', '例：freckles'],
  ['body', '体型', '例：slim build'],
  ['breast_size', '胸围', '例：medium breasts'],
  ['additional_details', '附加细节', '例：long hair'],
  ['aesthetics', '画质词', '进阶：整段画质提示词'],
  ['negative_prompt', '负向词', '进阶：不想出现的内容'],
  ['lora_prompt', 'LoRA 触发词', '进阶：自定义风格模型'],
]

/** 能用官方标签库点选的字段（官方 FIELD_CATEGORIES）。
 *  画质词 / 负向词 / LoRA 触发词是整段提示词工程，不是一个一个 trait，
 *  所以官方也不给它们做标签选择 —— 我们把这三个收进「进阶设置」。 */
export const INFO_TAG_FIELDS = new Set([
  'race', 'skin_color', 'hair', 'eyes', 'face', 'body', 'breast_size', 'additional_details',
])
export const INFO_ADV_FIELDS = INFO_FIELDS.filter(([k]) => !INFO_TAG_FIELDS.has(k))

/** 背景色（官方 background_color）。用户语言叫「出图底色」 */
export const BG_COLORS = ['green', 'blue', 'white', 'grey', 'magenta', 'transparent']

export type StepId = 1 | 2 | 3 | 4

export type StepDef = {
  id: StepId
  label: string
  /** 一句话说清这一步用户要做什么、做完会拿到什么 */
  goal: string
  /** 这一步做完的标志（给用户看的，不是内部状态） */
  result: string
}

/** 唯一主线。以前是 5 个并列 tab + 右栏 4 组 + 3 套互相竞争的 ol，
 *  用户读完所有引导仍不知道第一步点哪里。现在只有这 4 步。 */
export const STEPS: StepDef[] = [
  {
    id: 1, label: '创建角色',
    goal: '给角色起个名字，或直接拿一张参考图开始。',
    result: '得到一个可反复使用的角色档案',
  },
  {
    id: 2, label: '定形象',
    goal: '选外貌标签与画风，生成立绘（可以连着出好几版）。',
    result: '立绘，后面所有动作素材都以它为准',
  },
  {
    id: 3, label: '出素材',
    goal: '一键生成整套走路 / 表情 / 换装精灵帧，也可以只重跑某一步。',
    result: '按服装 × 表情 × 姿势整理好的精灵帧',
  },
  {
    id: 4, label: '导出',
    goal: '打包成图集和坐标表，直接拿进游戏引擎。',
    result: 'atlas.png + atlas.json，可直接导入工程',
  },
]

/** 统一任务状态：以前 9 个 xxxMsg 各写一行 10.5px 灰小字，
 *  切个页签就再也找不到结果了。现在只有一个状态，常驻右栏。 */
export type TaskState = {
  label: string
  status: 'running' | 'done' | 'failed'
  message: string
  /** step：一键出全套的阶段键（underwear|poses|emotions），由后端 pipeline.py 发布 */
  progress?: { done: number; total: number; current?: string; step?: string }
  startedAt: number
}

/** 后端错误码 → 用户能照做的话术（以前只有看图反推做了这件事）。 */
export function explainError(raw: unknown): string {
  const msg = raw instanceof Error ? raw.message : String(raw ?? '未知错误')
  if (msg.includes('DEPENDENCY_MISSING')) return `${msg}（本地语言模型装错了版本，需要 JamePeng 版 llama-cpp-python）`
  if (msg.includes('MODEL_MISSING') || msg.includes('MMPROJ_MISSING')) return `${msg}（models/llm/Qwen3.5-4B 下的权重没装齐）`
  if (msg.includes('TIMEOUT')) return `${msg}（模型还占着显存，等半分钟再试）`
  if (msg.includes('PARSE_ERROR') || msg.includes('INVALID_RESPONSE')) return `${msg}（模型这次没答成结构化结果，重试一次通常就好）`
  if (msg.includes('NO_BASE') || msg.includes('NO_SOURCE') || msg.includes('没有')) {
    return `${msg}。先回到第 ② 步生成立绘，再来做这一步。`
  }
  return msg
}

/** 角色当前产线状态。步骤条的完成标记只由「磁盘上真有什么」决定，
 *  不靠用户自己记得点到哪了。 */
export function computeProgress(o: {
  hasCharacter: boolean
  /** 立绘张数（`Portraits/`）。第 ② 步只看这个 —— 立绘不是动作帧。 */
  portraits: number
  frames: number
  groups: number
  hasAtlas: boolean
}): { done: boolean[]; firstUndone: StepId } {
  const done = [
    o.hasCharacter,
    o.hasCharacter && o.portraits > 0,
    o.hasCharacter && (o.frames >= 2 || o.groups >= 2),
    o.hasCharacter && o.hasAtlas,
  ]
  const idx = done.findIndex((d) => !d)
  return { done, firstUndone: (idx < 0 ? 4 : (idx + 1)) as StepId }
}

/** 估算耗时：把代价写在按钮旁边（业界共识：明码标价 / 预期管理）。
 *  数字来自本项目实测：单帧 24~28s、表情一张约 110s。 */
export function estimateSeconds(frames: number, emotions: number): string {
  const pose = frames * 26
  const emo = frames * emotions * 110
  const total = pose + emo
  if (total <= 0) return '约 1 分钟'
  if (total < 120) return `约 ${total} 秒`
  return `约 ${Math.round(total / 60)} 分钟`
}
