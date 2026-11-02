/**
 * 骨骼识别：把**任意**骨架的骨名解析成官方 `mixamo.js` 认得的那 17 个锚点
 * （Hips / Neck / Head / LeftArm / LeftForeArm / LeftHand / LeftUpLeg / LeftLeg /
 * LeftFoot / LeftToeBase …… 左右各一份）。
 *
 * **为什么需要这一层**：官方 `buildMixamoWorldKeypoints(sourceBones, viewer)` 只会
 * 用 `findSourceBone(sourceBones, 'LeftArm')` 这种方式按**名字**取骨。Mixamo 导出的
 * glb/fbx 天然符合，但 Quaternius / KayKit / Unreal / VRM 这些 CC0 或引擎资产命名
 * 各不相同（`upperarm_l` / `upper_arm.L` / `J_Bip_L_UpperArm` / `LeftUpperArm`），
 * 直接喂进去一个锚点都找不到 → 整帧套不上人偶，用户只看到"没反应"。
 *
 * 这里**不改官方文件**：解析结果是"别名登记表"，把真实骨头挂到 Mixamo 名下，
 * 官方那段代码一行不用动：
 *
 *     const rig = readSkeleton(root)
 *     const map = resolveRig(rig)                      // { family, alias, missing }
 *     const sourceBones = buildSourceBones(rig, map)
 *     buildMixamoWorldKeypoints(sourceBones, viewer)   // 官方原样调用
 *
 * **策略：结构优先，名字兜底。**
 *
 * 人形骨骼的**层级**是标准的：肩 → 肘 → 腕、髋 → 膝 → 踝，一定是祖孙三代。
 * 名字则五花八门。所以只要求"四个根锚点"（Hips / 两条上臂 / 两条大腿）认得出来，
 * 肘腕膝踝**沿子链往下走**就行 —— 这比再堆一百条别名可靠。
 *
 * 名字匹配只用于取根锚点，覆盖 4 套命名：Mixamo / Unreal / VRM / KayKit 风格，
 * 另有通用角色词兜底。
 *
 * 认不出**骨盆**就是彻底没救（人偶的中心），这时 `missing` 会带上缺失清单和源文件
 * 里的骨名样本，宿主据此给出"这个文件的骨骼认不出，缺 X / Y"的具体原因，
 * 而不是沉默（§6-6）。
 */

/** 官方 mixamo.js 会去找的全部锚点名（顺序即报错时的展示顺序） */
export const ANCHOR_NAMES = [
  'Hips', 'Neck', 'Head',
  'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
  'LeftHand', 'RightHand',
  'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg',
  'LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase',
]

/** 少一个就套不上（= 官方 buildMixamoWorldKeypoints 的 required 列表） */
const REQUIRED = [
  'Hips', 'Neck',
  'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm',
  'LeftHand', 'RightHand',
  'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg',
  'LeftFoot', 'RightFoot',
]

/**
 * 角色词表：`角色 → 归一化后的词`（小写、去分隔符、**已剥掉左右标记**）。
 *
 * 只做**精确相等**匹配。两三个字母的短词（`arm` / `leg` / `palm`）单独列在
 * `WEAK_WORDS` 里 —— 否则 `forearm` 会被 `arm` 抢走，`lowerleg` 会被 `leg` 抢走。
 */
const ROLE_WORDS = {
  hips: ['hips', 'pelvis', 'bip01pelvis', 'hip'],
  neck: ['neck', 'neck01', 'neck1'],
  head: ['head', 'head01', 'head1'],
  clavicle: ['clavicle', 'shoulder', 'collar'],
  upperArm: ['upperarm', 'uparm', 'armupper', 'armup'],
  foreArm: ['forearm', 'lowerarm', 'armlower', 'lowerforearm'],
  hand: ['hand', 'wrist', 'palm1', 'palm2', 'palm3', 'palm01'],
  upperLeg: ['upperleg', 'upleg', 'thigh', 'legupper'],
  lowerLeg: ['lowerleg', 'calf', 'shin', 'leglower'],
  foot: ['foot', 'ankle'],
  toe: ['toebase', 'toes', 'toe01', 'ball', 'toe'],
}

/** 只在没有强候选时才认的弱词（Mixamo 的 `LeftArm` / `LeftLeg` 走这条） */
const WEAK_WORDS = {
  upperArm: ['arm'],
  upperLeg: ['leg'],
  lowerLeg: ['leg'],
  hand: ['palm'],
}

/** 认出来也不该当锚点的名字（末端标记、眼球、IK 目标、扭骨等） */
const REJECT = /(end|nub|top|tip|target|twist|helper|leaf|eye)/i
/** 允许但降权：`ik` 单独判，避免误伤 `Pinky` 这类词 */
const REJECT_IK = /\bik\b|ik_|_ik|ik$/i

/* ------------------------------------------------------------------ *
 * 1. 读骨架：把 Object3D 树拍平成纯数据
 * ------------------------------------------------------------------ */

/**
 * 拍平骨架。**遍历全部子节点，不只是 isBone** ——
 * 不少 glTF 的末端骨头（`LowerLeg.L_end`、`Head_end`）不在 skin.joints 里，
 * GLTFLoader 把它建成普通 Object3D，但它的位置就是脚踝/头顶，
 * 缺了它腿会短一截。`isBone` 只在打分时作为加分项。
 */
export function readSkeleton(root) {
  const nodes = []
  const at = new Map()
  const add = (obj, parentIdx) => {
    const i = nodes.length
    nodes.push({
      name: obj.name || '',
      parent: parentIdx,
      children: [],
      obj,
      isBone: !!obj.isBone,
    })
    at.set(obj, i)
    if (parentIdx >= 0) nodes[parentIdx].children.push(i)
    const kids = obj.children || []
    for (let k = 0; k < kids.length; k++) add(kids[k], i)
    return i
  }
  add(root, -1)
  return { nodes, at, size: nodes.length }
}

/* ------------------------------------------------------------------ *
 * 2. 名字处理
 * ------------------------------------------------------------------ */

/** 去掉命名空间（`mixamorig:` / `Armature|`）与非字母数字，全部小写 */
function normalize(name) {
  if (!name) return ''
  let s = String(name)
  const colon = s.lastIndexOf(':')
  if (colon >= 0) s = s.slice(colon + 1)
  const pipe = s.lastIndexOf('|')
  if (pipe >= 0) s = s.slice(pipe + 1)
  return s.replace(/[^a-z0-9]/gi, '').toLowerCase()
}

/**
 * 判断左右并从名字里**剥掉**左右标记，返回 `[side, role]`。
 *
 * 覆盖的写法：`LeftArm` / `left_arm` / `upper_arm.L` / `upper_arm_l` /
 * `arml` / `Bip01 L UpperArm` / `J_Bip_L_UpperArm` / `Bip01_L_UpperArm`。
 */
export function splitSide(name) {
  const raw = String(name || '')
  let s = raw
  let side = ''

  // 1) `Left*` / `Right*`（前面允许有命名空间或分隔符，后面直接接词也可以：
  //    `mixamorig:LeftArm`、`LeftUpperArm` 都没有分隔符）
  const lr = /(left|right)/i.exec(s)
  if (lr) {
    side = lr[1].toLowerCase() === 'left' ? 'l' : 'r'
    s = s.slice(0, lr.index) + s.slice(lr.index + lr[0].length)
  } else {
    // 2) 分隔开的单字母：`_l` / `.L` / `-l` / ` L ` / `_L_`
    const m = /(?:^|[_.\-\s])([lr])(?=$|[_.\-\s\d])/i.exec(s)
    if (m) {
      side = m[1].toLowerCase()
      s = s.slice(0, m.index) + s.slice(m.index + m[0].length)
    }
  }

  let role = normalize(s)
  // 3) 还粘在词尾的 `arml` / `calfl` / `toel`：去掉最后一个字母。
  //    **必须剥完还在词表里才算**，否则 `shoulder` 会被剥成 `shoulde` + 右侧 —— 
  //    假左右比认不出左右更糟。
  if (!side && role.length >= 4) {
    const tail = role.slice(-1)
    const head = role.slice(0, -1)
    if ((tail === 'l' || tail === 'r') && KNOWN_ROLES.has(head)) {
      side = tail
      role = head
    }
  }
  return [side, role]
}

/** 词表全集（判断"剥掉末字母后还是不是一个角色词"） */
const KNOWN_ROLES = new Set([
  ...Object.values(ROLE_WORDS).flat(),
  ...Object.values(WEAK_WORDS).flat(),
])

/** 角色词 → 匹配强度（0 = 不匹配）。
 *
 * 先试**精确相等**；不中再试**后缀**（`J_Bip_C_Hips` / `Armature|Spine` 这类
 * 带前缀的命名，归一化后角色词被粘在前面）。后缀只认 4 个字母以上的词 ——
 * `arm` / `leg` 这种太短，粘在任何一个词尾上都可能是巧合。
 */
function roleScore(role, words) {
  if (!role) return 0
  const exact = words.indexOf(role)
  if (exact >= 0) return 100 - exact * 4 + Math.min(role.length, 20)
  for (let k = 0; k < words.length; k++) {
    const w = words[k]
    if (w.length >= 4 && role.endsWith(w)) return 55 - k * 3
  }
  return 0
}

/* ------------------------------------------------------------------ *
 * 3. 解析
 * ------------------------------------------------------------------ */

/**
 * 解析骨架 → `{ family, familyLabel, alias, missing, samples }`
 *
 * `alias` 只放**认出来的**锚点（键是 Mixamo 名，值是节点下标）。
 * `missing` 是 REQUIRED 里没认出来的那些。
 */
export function resolveRig(rig) {
  const { nodes } = rig
  const alias = {}
  const scores = {}

  // 每个节点先算好 (side, role)，只算一次
  const info = nodes.map((n) => {
    const [side, role] = splitSide(n.name)
    return { side, role, rejected: REJECT.test(n.name) || REJECT_IK.test(n.name) }
  })

  // ---- 名字过一遍：给每个锚点挑分最高的节点 ----
  const best = {}
  const weak = {}
  const put = (table, anchor, idx, base, bonus) => {
    // base 是**名字**给的匹配强度；0 表示名字根本不像，这时只有一个 isBone 加分
    // 不算数 —— 早期版本就是这里漏判，把 `LeftShoulder` 当成了 `LeftArm`
    if (!(base > 0)) return
    const score = base + bonus
    if (table[anchor] == null || score > table[anchor].score) table[anchor] = { idx, score }
  }
  for (let i = 0; i < nodes.length; i++) {
    const { side, role, rejected } = info[i]
    if (rejected || !role) continue
    const bonus = nodes[i].isBone ? 3 : 0

    put(best, 'Hips', i, roleScore(role, ROLE_WORDS.hips), bonus)
    put(best, 'Neck', i, roleScore(role, ROLE_WORDS.neck), bonus)
    put(best, 'Head', i, roleScore(role, ROLE_WORDS.head), bonus)
    if (!side) continue

    const S = side === 'l' ? 'Left' : 'Right'
    put(best, `${S}Arm`, i, roleScore(role, ROLE_WORDS.upperArm), bonus)
    put(best, `${S}ForeArm`, i, roleScore(role, ROLE_WORDS.foreArm), bonus)
    put(best, `${S}Hand`, i, roleScore(role, ROLE_WORDS.hand), bonus)
    put(best, `${S}UpLeg`, i, roleScore(role, ROLE_WORDS.upperLeg), bonus)
    put(best, `${S}Leg`, i, roleScore(role, ROLE_WORDS.lowerLeg), bonus)
    put(best, `${S}Foot`, i, roleScore(role, ROLE_WORDS.foot), bonus)
    put(best, `${S}ToeBase`, i, roleScore(role, ROLE_WORDS.toe), bonus)
    put(weak, `Clavicle${S}`, i, roleScore(role, ROLE_WORDS.clavicle), bonus)

    put(weak, `${S}Arm`, i, roleScore(role, WEAK_WORDS.upperArm), bonus)
    put(weak, `${S}UpLeg`, i, roleScore(role, WEAK_WORDS.upperLeg), bonus)
    put(weak, `${S}Leg`, i, roleScore(role, WEAK_WORDS.lowerLeg), bonus)
    put(weak, `${S}Hand`, i, roleScore(role, WEAK_WORDS.hand), bonus)
  }

  for (const anchor of ANCHOR_NAMES) {
    const b = best[anchor] || weak[anchor]
    if (!b) continue
    // 弱词（`arm`/`leg`/`palm`）先记着，结构修复时会被链上的真关节顶掉
    const isWeak = !best[anchor]
    alias[anchor] = b.idx
    scores[anchor] = isWeak ? -b.score : b.score
  }

  // ---- 结构修复：肘腕膝踝以层级为准 ----
  repairArmsAndLegs(rig, alias, scores, weak)

  // ---- 骨盆：名字认不出时用"两条腿的共同祖先" ----
  if (alias.Hips == null) {
    const p = commonAncestorOfLegs(rig, alias)
    if (p != null) alias.Hips = p
  }
  // ---- 脖子：名字认不出时取"头往下的第一个祖先" ----
  if (alias.Neck == null && alias.Head != null) {
    const p = rig.nodes[alias.Head].parent
    if (p != null && p >= 0 && p !== alias.Hips) alias.Neck = p
  }

  const missing = REQUIRED.filter((k) => alias[k] == null)
  // 家族按**锚点骨头的名字**判，不按全文件扫 —— 很多 CC0 资产是从 Mixamo
  // 转出来的，手指骨还带着 `mixamorig:` 前缀，全文件扫会把 KayKit 报成 Mixamo。
  const family = detectFamily(nodes, alias)
  return {
    family,
    familyLabel: FAMILY_LABELS[family] || family,
    alias,
    missing,
    samples: nodes.filter((n) => n.isBone).map((n) => n.name).filter(Boolean).slice(0, 8),
    nodeCount: nodes.length,
  }
}

const FAMILY_LABELS = {
  mixamo: 'Mixamo',
  unreal: 'Unreal / Manny',
  vrm: 'VRM (J_Bip)',
  kaykit: 'KayKit / Quaternius',
  generic: '通用人形',
}

/** 粗判骨架家族（只用于界面展示与报错文案，解析本身不依赖它） */
function detectFamily(nodes, alias) {
  const names = Object.values(alias || {}).map((i) => nodes[i]?.name || '')
  if (!names.length) return 'generic'
  const has = (re) => names.some((n) => re.test(n))
  if (has(/mixamorig/i)) return 'mixamo'
  if (has(/^J_Bip_/i)) return 'vrm'
  // Unreal：`pelvis` / `spine_01` / `upperarm_l` / `calf_r` 这种下划线小写
  if (has(/^(pelvis|spine_0\d)$/i) || has(/^(clavicle|upperarm|lowerarm|thigh|calf)_[lr]$/i)) {
    return 'unreal'
  }
  // KayKit / Quaternius：`upper_arm_l` / `upper_leg_l` / `shoulder_l`
  if (has(/^(upper_arm|lower_arm|upper_leg|lower_leg|shoulder)[._][lr]$/i)) return 'kaykit'
  // Mixamo 原生：`LeftArm` / `LeftUpLeg`（驼峰、无分隔符）
  if (has(/^(Left|Right)(Arm|ForeArm|Hand|UpLeg|Leg|Foot|ToeBase)$/)) return 'mixamo'
  return 'generic'
}

/* ------------------------------------------------------------------ *
 * 4. 层级工具
 * ------------------------------------------------------------------ */

const _sizeCache = new WeakMap()

/** 子树节点数（用来判断"哪一支才是主干"，腿和手臂总是比装饰骨大） */
function subtreeSize(rig, idx) {
  let cache = _sizeCache.get(rig)
  if (!cache) { cache = new Map(); _sizeCache.set(rig, cache) }
  const hit = cache.get(idx)
  if (hit != null) return hit
  let n = 1
  for (const k of rig.nodes[idx].children) n += subtreeSize(rig, k)
  cache.set(idx, n)
  return n
}

/** 挂在关节上的"杂物"：裙子/披风/头发/武器/挂点，子树再大也不是主干 */
const NOISE = /(skirt|cape|cloth|hair|weapon|prop|socket|attach|shield|sword|bag|quiver|wing|tail|ponytail)/i

/** 子节点里"最有分量"的那个（子树最大，且不是挂在身上的杂物） */
function mainChild(rig, idx) {
  const kids = rig.nodes[idx]?.children || []
  let pick = null
  let size = -1
  for (const k of kids) {
    if (NOISE.test(rig.nodes[k].name || '')) continue
    const s = subtreeSize(rig, k)
    if (s > size) { size = s; pick = k }
  }
  return pick
}

/** 沿主干往下走 hops 跳，返回途中的节点下标 */
function chainDown(rig, idx, hops) {
  const out = []
  let cur = idx
  for (let h = 0; h < hops; h++) {
    const nxt = mainChild(rig, cur)
    if (nxt == null) break
    out.push(nxt)
    cur = nxt
  }
  return out
}

/**
 * 结构修复：肩 → 肘 → 腕、髋 → 膝 → 踝，各走两跳。
 *
 * **只在名字不可信时动手**：某个锚点没认出来（`null`），或者只是靠弱词
 * （`arm` / `leg` / `palm`，score < 0）粘上的。名字强匹配（`upperarm` /
 * `lowerarm` / `thigh` / `calf`）一律尊重 —— 名字和层级都清楚的时候，
 * 层级不会提供额外信息，改它只会引入风险。
 *
 * 之所以还需要这一步：同一个词在不同骨架里意思不同。Mixamo 的 `LeftLeg`
 * 是小腿，KayKit 的 `Leg` 是大腿，RobotExpressive 的 `Leg.L` 干脆不是关节。
 * 名字分不清的时候，只有**层级**不会骗人。
 */
function repairArmsAndLegs(rig, alias, scores, weak) {
  const set = (anchor, idx) => { alias[anchor] = idx }
  /** 名字不可信（没认出来，或只靠弱词粘的） */
  const shaky = (anchor) => alias[anchor] == null || (scores[anchor] ?? -1) < 0

  for (const s of ['Left', 'Right']) {
    // ---- 上臂：名字没认出来时，退到「锁骨的第一跳」 ----
    if (alias[`${s}Arm`] == null) {
      const clav = weak[`Clavicle${s}`]?.idx
      if (clav != null) {
        const c = chainDown(rig, clav, 1)[0]
        if (c != null) set(`${s}Arm`, c)
      }
    }

    // ---- 手臂：肩 → 肘 → 腕（两跳） ----
    const armRoot = alias[`${s}Arm`]
    if (armRoot != null && (shaky(`${s}ForeArm`) || shaky(`${s}Hand`))) {
      const [elbow, wrist] = chainDown(rig, armRoot, 2)
      if (elbow != null && shaky(`${s}ForeArm`)) set(`${s}ForeArm`, elbow)
      if (wrist != null && shaky(`${s}Hand`)) set(`${s}Hand`, wrist)
    }

    // ---- 腿：髋 → 膝 → 踝（两跳） ----
    const legRoot = alias[`${s}UpLeg`]
    if (legRoot != null && (shaky(`${s}Leg`) || shaky(`${s}Foot`))) {
      const [knee, ankle] = chainDown(rig, legRoot, 2)
      if (knee != null && shaky(`${s}Leg`)) set(`${s}Leg`, knee)
      if (ankle != null && shaky(`${s}Foot`)) set(`${s}Foot`, ankle)
    }

    // 脚趾：名字里有就用名字（`toes` 常在踝外侧另起一根），否则踝的下一跳
    if (shaky(`${s}ToeBase`) && alias[`${s}Foot`] != null) {
      const toe = chainDown(rig, alias[`${s}Foot`], 1)[0]
      if (toe != null) set(`${s}ToeBase`, toe)
    }
  }

  // 同一个节点被挂到"根 + 末端"两个锚点上（链只有一跳的退化骨架），末端清掉，
  // 否则官方会拿零长度的段去算方向，整条肢体翻到背面
  for (const s of ['Left', 'Right']) {
    for (const [root, tip] of [
      [`${s}Arm`, `${s}ForeArm`], [`${s}ForeArm`, `${s}Hand`],
      [`${s}UpLeg`, `${s}Leg`], [`${s}Leg`, `${s}Foot`],
    ]) {
      if (alias[root] != null && alias[root] === alias[tip]) delete alias[tip]
    }
  }
}

/** 两条大腿（或两条小腿）的共同祖先 = 骨盆 */
function commonAncestorOfLegs(rig, alias) {
  const roots = [alias.LeftUpLeg, alias.RightUpLeg].filter((x) => x != null)
  if (roots.length < 2) return null
  const own = new Set()
  let cur = rig.nodes[roots[0]].parent
  let guard = 0
  while (cur != null && cur >= 0 && guard++ < 256) {
    own.add(cur)
    cur = rig.nodes[cur].parent
  }
  cur = rig.nodes[roots[1]].parent
  guard = 0
  while (cur != null && cur >= 0 && guard++ < 256) {
    if (own.has(cur)) return cur
    cur = rig.nodes[cur].parent
  }
  return null
}

/* ------------------------------------------------------------------ *
 * 5. 产出官方能吃的 sourceBones
 * ------------------------------------------------------------------ */

/**
 * 把解析结果包成官方 `findSourceBone` 认识的形状：
 * `{ bones: { Hips: Object3D, ... }, normalizedBones: {...} }`。
 *
 * 先把**原始骨名**全部登记进去（手指链之类按原名找），再把别名**覆盖**上去。
 * 所以 Mixamo 文件走这条路和以前完全等价 —— 别名只是把 `LeftArm` 指向了
 * 本来就叫 `LeftArm` 的那根骨头。
 */
export function buildSourceBones(rig, map) {
  const bones = {}
  const normalizedBones = {}
  for (const node of rig.nodes) {
    if (!node.name) continue
    bones[node.name] = node.obj
    const norm = normalize(node.name)
    if (norm && !normalizedBones[norm]) normalizedBones[norm] = node.obj
  }
  for (const [anchor, idx] of Object.entries(map?.alias || {})) {
    const node = rig.nodes[idx]
    if (!node) continue
    bones[anchor] = node.obj
    normalizedBones[normalize(anchor)] = node.obj
  }
  return { bones, normalizedBones }
}

/** 报错文案：说清"缺哪几根"和"文件里的骨名长什么样" */
export function describeRig(map) {
  if (!map) return '没有读到骨骼'
  const missing = map.missing || []
  if (!missing.length) return ''
  const sample = (map.samples || []).slice(0, 6).join(', ')
  return `这个文件的骨骼认不出：缺 ${missing.join(' / ')}`
    + `（判为「${map.familyLabel}」骨架）。`
    + (sample ? `文件里的骨名是 ${sample} ……。` : '')
    + '要用 Mixamo 导出的骨架，或标准人形命名（Hips / Spine / LeftArm / LeftUpLeg 这类）。'
}
