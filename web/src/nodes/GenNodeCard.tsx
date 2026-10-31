import { useRef, useState } from 'react'
import { hasCJK } from '../store/prompts'
import MaskEditor from '../components/MaskEditor'
import MaskPreview from '../components/MaskPreview'
import SkeletonEditor, { defaultSkeleton } from '../components/SkeletonEditor'
import type { ModelCaps, PoseDir } from '../api'
import { fileNameOf } from '../fileName'
import { MAX_CHAR_SOURCES, type GenNode, type GenParams } from '../types'
import { GEN_STYLES, styleFromParams } from '../genStyles'

/**
 * 每个参数的说明：做什么 / 影响什么 / 推荐值。
 *
 * 为什么集中放一张表而不是散在 JSX 里：同一个参数（比如「候选」）在姿态序列和
 * 序列帧两处都有，散着写一定会分叉——改一处忘一处，最后两处说法不一致。
 *
 * 推荐值尽量写**本项目实测过**的数字，不写"视情况而定"这种废话。
 */
const PARAM_HINTS: Record<string, { what: string; effect: string; rec: string }> = {
  宽: {
    what: '输出画面的宽度（px）',
    effect: '决定细节量与构图余量。太小画不出全身，太大反而更容易顶边',
    rec: '768。Qwen 的 1024² token 数是 768² 的 1.78 倍（单帧 45s vs 25s），'
      + '而精灵成品最后要缩到 32~128px，那点细节看不见；画立绘再手动调大',
  },
  高: { what: '输出画面的高度（px）', effect: '同宽，精灵帧要留出头顶与脚底的空隙', rec: '768（与宽一致，方形最好对齐）' },
  图片宽: {
    what: '这张图在画布上的显示宽度',
    effect: '只影响画布排版，不影响生成结果',
    rec: '200~320；拖右下角也能改',
  },
  步数: {
    what: '去噪步数（采样迭代次数）',
    effect: '太少会糊、结构不成形；太多只是变慢，画质提升很小',
    rec: '16（默认）。8~12 步浅色主体会发白；白猫/白甲建议 20~24；'
      + '开快速模式走 8 步蒸馏（Lightning LoRA）',
  },
  CFG: {
    what: '真 CFG 强度（classifier-free guidance）',
    effect: '大于 1 时管线每个步跑两遍 transformer（有条件 + 无条件），'
      + '时间翻倍，负向词这时才生效；等于 1 只跑一遍，快 40%',
    rec: '1.0（默认，快）。**别照 SD 的习惯填 7**——Qwen 的 CFG 语义不同，'
      + '这里 >1 就已经是"真 CFG"（每步跑两遍 transformer，时间翻倍），'
      + '7 属于强引导、容易过饱和与背景起雾。要更强的负向约束，从 2~4 试起',
  },
  种子: {
    what: '随机种子。-1 表示每次随机',
    effect: '固定种子才能复现同一张图；换种子是产线里最便宜的"重抽"手段',
    rec: '调参时固定；批量出候选时用 seed、seed+1000、seed+2000…',
  },
  风格: {
    what: '提示词契约（predefined 的构图/背景约定）',
    effect: '自动在你的描述后面追加「扁平背景、硬描边、全身」这类产线约束；'
      + '**不截断你的描述**（Qwen 没有 token 上限），写多长都原样送进去',
    rec: '精灵用 sprite、立绘用 portrait、武侠 LoRA 配 wuxia、像素用 pixel',
  },
  反向提示词: {
    what: '不希望出现的东西',
    effect: '**默认 cfg=1.0 时不会发给模型**（省掉一半前向），写了也不起作用；'
      + '把 CFG 调到 >1 才会真的生效，代价是时间翻倍。生效时写具体名词比堆形容词有效'
      + '（如 character sheet 压多主体）',
    rec: '留空用内置默认即可。多数情况下把「主体 + 构图」写清楚比加负向词更有效',
  },
  列: {
    what: '精灵图一行放几帧',
    effect: '决定打包后的网格形状，不影响每帧画面',
    rec: '0=按帧数自动排一行。8 帧走路常用 8 列（或 4×2）',
  },
  行: {
    what: '精灵图的行数',
    effect: '同上，只影响打包排版',
    rec: '0=自动。帧数不是列数整数倍时才需要手填',
  },
  占高: {
    what: '人物占画面高度的比例',
    effect: '太大会顶到画布边缘（脚被切），太小则像素化后没细节',
    rec: '0.75~0.85，留出头顶与脚底的安全边',
  },
  参考: {
    what: '参考图的解释方式',
    effect: '骨架=只取关节（真人照片适用）；轮廓=取外形（游戏素材适用）；自动=先试骨架，认不出人再退轮廓',
    rec: '游戏精灵图用「轮廓」；真人动作照用「骨架」；不确定用「自动」',
  },
  候选: {
    what: '每帧生成几张再挑一张',
    effect: '同 seed 也会漂，多张按「关节数+前景占比+贴边+漂移」挑最好的。越多越稳、越慢',
    rec: '1~2。上限由显存算出：Qwen 在 768² 上多一个候选就是多 6GB 峰值与一倍时间，'
      + '24G 卡只够 1 个',
  },
  补提: {
    what: '候选都不合格时再补生成几次',
    effect: '只对"过了闸门才要"的帧生效；0=不补，直接留最好的那张',
    rec: '1~2 次。开销集中在废帧上，通常划算',
  },
  色板: {
    what: '量化的调色板色数（0=不量化）',
    effect: '色数越少越"像素"；太少丢明暗层次，太多会出现脏色阶',
    rec: '32。复古风可到 16，细节多的角色用 48',
  },
  方式: {
    what: '序列帧的处理方式',
    effect: 'AI 重绘能换体型但帧间会漂；换皮只换配色，帧间完全一致',
    rec: '要改体型用 AI 重绘；只是换颜色（省时且零漂移）用换皮',
  },
  引擎: {
    what: '生成路线（序列帧替换 / 姿态序列都走这个下拉）',
    effect: 'VNCCS = 骨架 + 角色双图指令，专用 LoRA 锁住画风与服装（帧间不漂，推荐）；'
      + '旧版 = 每帧单独跑一次角色迁移再过质量闸门（慢，但每帧能挑最好的）',
    rec: '先用 VNCCS。腿部姿态被参考图的站姿覆盖时，再改用旧版逐帧',
  },
  边长: {
    what: '目标像素画的边长（原生像素数）',
    effect: '决定最终像素密度；0=自动检测输入图的网格',
    rec: '0（自动）。已知原图是 128px 像素画时直接填 128',
  },
  像素化: {
    what: '要不要把出图变成像素画（把画面按色块栅格化）',
    effect: '关掉就是普通插画/立绘；打开才做像素化。像素化会**丢掉大量细节**——'
      + '同一句提示词，1024² 生成再砍成 128 有效像素，比 768² 生成再砍成 128 明显更精细',
    rec: '要游戏素材就开（128px 有效像素是标准精灵尺寸）；要立绘/宣传图就关',
  },
  像素块: {
    what: '像素化时每个色块的边长（0 = 不像素化）',
    effect: '块越大，有效像素越少、越"像素"；块太小会保留 AI 图的软边和噪点，'
      + '看着像插画而不是像素画',
    rec: '768² 用 6（正好降到 128 有效像素，标准精灵尺寸）；512² 用 4；'
      + '要 64px 的素材就用 12',
  },
  色数: {
    what: '量化的调色板色数（0=不量化）',
    effect: '越少越像素；太少丢层次，太多脏',
    rec: '32；像素素材常用 16~48',
  },
  取色: {
    what: '每一格取哪个颜色',
    effect: 'median=中位数（抗噪）、majority=众数（色块更纯）、center=取中心点（保细节但易偏）',
    rec: '像素画用 majority；照片转像素用 median',
  },
  吸附: {
    what: '网格吸附到多少的整数倍（0=不吸附）',
    effect: '让像素格子对齐到标准尺寸（如 4 的倍数），方便引擎里缩放不糊',
    rec: '0；要出 2 的幂尺寸素材时填 4 或 8',
  },
  上下文: {
    what: '选区外扩多少上下文给模型看',
    effect: '大=模型知道周围是什么，接缝更自然；小=更听提示词但可能接不上',
    rec: '0.3~0.5（按选区尺寸的比例）',
  },
  外扩: {
    what: '蒙版先外扩几像素',
    effect: '给模型留"往哪长"的余地；不扩容易在原边界留一圈旧像素',
    rec: '8~16px（像素素材用 2~4，否则会吃掉邻格）',
  },
  羽化: {
    what: '贴回时的边缘羽化半径',
    effect: '大=接缝柔和但会糊；小=硬边（像素素材要的就是硬边）',
    rec: '像素素材用 0；插画用 2~4',
  },
  网格: {
    what: '是否检测并吸附到输入图的原生像素网格',
    effect: '命中时能保证"每格单色"，是完美像素转换的关键；输入不是像素画时会自动回退',
    rec: '开。只有输入本来就是连续的 AI 图、不需要网格时才关',
  },
  透明底: {
    what: '把纯色背景转成透明',
    effect: '只对纯色背景有效；复杂背景要靠「抠图」而不是这个开关',
    rec: '开（背景是纯色/白底时）',
  },
  色彩校正: {
    what: '在重绘区接缝处做色彩匹配',
    effect: '开=接缝更自然但可能轻微偏色；关=完全按模型输出的颜色',
    rec: '开。接缝明显时它比调强度更有效',
  },
  色板吸附: {
    what: '把重绘区里接近原图的杂色吸回原色板',
    effect: '保持整张图的色数不暴涨（像素素材尤其明显）',
    rec: '开（像素素材必开）',
  },
  抠图: {
    what: '生成后自动抠图（去背景）',
    effect: '精灵帧必须抠（要对齐脚底）；不抠则输出仍是带背景的方图',
    rec: '精灵/立绘开；只要一张图不做管线时关掉省时间',
  },
  背景方式: {
    what: '抠图节点要把背景换成什么',
    effect: '透明=出 PNG 带 alpha；纯色=叠一个色底；换背景图=叠到另一张图上',
    rec: '要进引擎用「透明」；要做展示图用「纯色」或「换背景图」',
  },
  像素边长: {
    what: '转像素后成品的原生边长（px）',
    effect: '这是**成品**尺寸，不是画布尺寸：192 就是 192×192 的像素画，放大看是色块',
    rec: '精灵 64~192；头像 96；宣传图 256~384。小于 64 细节必然丢',
  },
  对比度增强: {
    what: '量化前先拉对比度/饱和度的强度',
    effect: '不开时量化会整体发灰、红变暗红、金发变土黄；开 1.0 后颜色才鲜',
    rec: '1.0（默认体验最好）；原图本身已经高饱和时给 0.5，避免溢出',
  },
  描边: {
    what: '量化后给主体补一圈 1px 轮廓线',
    effect: '量化后相邻色块只有色差没有结构线，缩小看会糊成一坨（melt）；描边后轮廓立住',
    rec: '开。只描外轮廓——对内部结构边全描会把脸糊成一块灰',
  },
  透明背景: {
    what: '让模型直接输出带 alpha 的 RGBA 图（原生透明底）',
    effect: '与「抠图」不同：抠图是画完再用分割模型切一遍，发丝、细剑、半透明布料常被切断；'
      + '这个是从生成时就有 alpha，边缘更干净、细节不丢',
    rec: 'Qwen-Image 底模出精灵/立绘时开。开它就别再开「抠图」——重复处理反而把 alpha 弄脏',
  },
}

/** 拼成原生 title：换行分三段，扫一眼就知道"是什么/影响什么/填多少" */
function hint(k: string): string {
  const h = PARAM_HINTS[k]
  // **查不到就返回显眼文案，不要返回空串**：空串的失败模式是静默的——
  // 界面上只是没有说明，不报错，没人会发现漏了。实测「像素块」就是这样
  // 漏了一整轮（表里 37 个键、代码引用 38 个）。
  if (!h) return `未定义说明：${k}（在 PARAM_HINTS 里补）`
  return `${h.what}\n影响：${h.effect}\n推荐：${h.rec}`
}

export const NODE_W = 300
/** 节点可拖宽度范围：窄于 240 参数会换行，宽于 720 画布上就不好摆 */
export const WIDE_MIN = 240
export const WIDE_MAX = 720
/** 节点可拖高度范围。低于 160 连卡头和一行参数都放不下 */
export const TALL_MIN = 160
export const TALL_MAX = 1400

export type NodePatch = Partial<
  Pick<GenNode,
    'x' | 'y' | 'w' | 'h' | 'params' | 'view' | 'job' | 'source' | 'image'
    | 'charSource' | 'charSources' | 'skeleton'>
>

/**
 * 两个输入端口在节点左边缘的纵向位置（相对节点顶部，px）。
 *
 * 连线端点和端口圆点**共用这两个常量**：只要各写一份，改样式时必然对不上——
 * 表现是线插在端口上方/下方十几像素，看着像画错了。`char` 那个值对的是
 * 「源图区 + 间距」之后角色图区的中心（头部 37 + body padding 10 + 源图区 78
 * + gap 8，再往下 39 是角色图区中线）。
 */
export const PORT_SRC_Y = 13
export const PORT_CHAR_Y = 172

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
  model: string
  /** 切底模：产出风格档位可能要求特定底模（如「像素精灵·Illustrious」要 sdxl-anime）。
   *  keepNodes=true 表示参数由档位负责，别套推荐值（否则会覆盖刚 patch 的 LoRA/尺寸）。 */
  onModel?: (m: string, opts?: { keepNodes?: boolean }) => void
  /** 当前底模在 W4 上能批量几个候选（后端按总显存算）；缺省不限制 */
  maxCandidates?: number
  /** 可当 W4 输入的骨架帧目录（后端扫出来的） */
  poseDirs?: PoseDir[]
  /** 本机是否装了 SDXL refiner 权重（没有就别让人勾） */
  /** 当前底模支不支持原生透明背景（只有 Qwen-Image-2.1 支持） */
  canTransparent?: boolean
  /** 可用底模清单（底模是节点级的，这里给下拉用） */
  nodesModels?: string[]
  /** 底模能力表：按本节点的 workflow 过滤可选项 */
  modelCaps?: Record<string, ModelCaps>
  /** 侧栏的「新节点默认底模」，下拉里显示成「跟随默认（xxx）」 */
  defaultModel?: string
  onSpawnImg2Img: () => void
  /** 骨骼编辑器导出的 PNG（Blob）→ 上传并挂到节点 image 上 */
  onExportSkeleton?: (blob: Blob) => void
  /** 骨骼编辑器「从图片提取」，返回 Promise 以便节点显示"提取中" */
  onImportPose?: (file: File) => void | Promise<void>
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


/**
 * 当前参数落在哪一档风格上（用来回显下拉）。
 *
 * 规则按优先级：
 *  1. 用户明确选了「自定义参数」→ 保持自定义。这一档不改任何参数，靠反推
 *     必然落回某个预设档（参数恰好没变），不特判的话下拉会「弹回去」。
 *  2. 记录的档位与实际参数一致 → 用它（正常路径）。
 *  3. 其余一律按参数反推。
 *
 * 第 3 条是为了**不骗人**：切底模（动漫底模会改 preset/pixel/palette）之后
 * 参数被外部改掉，`genStyle` 却停在旧值——只信它就会显示「像素精灵」而参数
 * 实际是动漫立绘。
 */
function styleOf(p: GenParams): string {
  if (p.genStyle === 'custom') return 'custom'
  const byParams = styleFromParams(p.preset, p.pixel, p.palette)
  if (p.genStyle && byParams === p.genStyle) return p.genStyle
  return byParams
}

/**
 * 节点 kind → 工作流名。**必须和 engine._ARCH_WORKFLOWS 用同一套词**，
 * 否则前端过滤和后端 require() 会各说各话（界面允许选、提交后被拒）。
 */
const WORKFLOW_OF: Record<string, string> = {
  gen: 'txt2img',
  img2img: 'img2img',
  bg: 'matte',
  inpaint: 'inpaint',
  airepaint: 'ai_repaint',
  pixelify: 'pixelify',
  poseswap: 'poseswap',
  poseclone: 'pose_clone',
  spriteswap: 'sprite_swap',
  topixel: 'topixel',
  // 这三个是 2026-10-30/31 加的，漏了会让它们的 `wf` 为空 →
  // 底模下拉整块不渲染、isApprox 判断也失效
  outpaint: 'outpaint',
  outfit: 'outfit_swap',
  faceswap: 'face_swap',
}

/**
 * 底模下拉里的短名：目录名（qwen-image-2.1）太长，撑爆 260px 的卡片。
 *
 * **别再把 sd15 / sdxl 加回这张表**：SD 权重早就归档了（见 README「底模」），
 * 下拉选项本身来自后端 `/models`，这张表只做「目录名 → 人看的名字」。
 * 留着旧映射的坏处不是显示错——是让人以为"列表里可能会冒出一个 SD"，
 * 排查模型相关问题时白查一遍。
 */
function shortModel(m: string): string {
  const map: Record<string, string> = { 'qwen-image-2.1': 'Qwen-Image 2.1' }
  return map[m] || m
}


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
  model,
  onModel,
  maxCandidates,
  poseDirs,
  canTransparent,
  nodesModels,
  modelCaps,
  defaultModel,
  onSpawnImg2Img,
  onExportSkeleton,
  onImportPose,
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
  const job = node.job
  const busy = job?.status === 'queued' || job?.status === 'running'
  const hasImage = !!job?.url
  const imageMode = node.view === 'image' && hasImage
  const isI2I = node.kind === 'img2img'
  const isImg = node.kind === 'image'
  const isBg = node.kind === 'bg'
  const isInpaint = node.kind === 'inpaint'
  const isAiRepaint = node.kind === 'airepaint'
  /**
   * 像素化之后**还剩多少有效像素**：边长 / 像素块。
   *
   * 必须显示出来：这个字段很容易被理解成"想要的成品尺寸"（比如填 64 想要 64px 的
   * 像素画），实际它是"每个色块多少像素"——512 配 64 得到的是 **8×8**，
   * 细节必然全没。只靠 tooltip 拦不住这种误用。
   */
  const effPx = p.pixel > 1
    ? Math.max(1, Math.round(Math.max(p.width, p.height) / p.pixel))
    : Math.max(p.width, p.height)
  // 参考图那种像素角色大约 128px；低到 64px 以下细节就必然糊掉
  const pxWarn = p.pixel > 1 && (effPx < 64 || (p.palette > 0 && p.palette < 8))
  // 候选数上限：后端按"本机总显存"给（SDXL@768 是 3、@1024 只有 1）。
  // 用户填超了就夹住——不是不让他填，是那个配置会让耗时崩掉且不报错。
  const candCap = Math.max(1, Math.min(6, maxCandidates ?? 6))
  // 序列帧的三个统计值（可能缺失：单图工作流没有这些字段）
  const sCand = job?.candidates
  const sFrames = job?.frames
  const sAttempts = job?.attempts
  const sGate = job?.gatePassed
  const isPixelify = node.kind === 'pixelify'
  // 转像素：源图可选（没有就用 prompt 先生成），所以不能算「必须源图」
  const isToPixel = node.kind === 'topixel'
  const isPoseClone = node.kind === 'poseclone'
  const isSpriteSwap = node.kind === 'spriteswap'
  const isPoseSwap = node.kind === 'poseswap'
  const isPoseEdit = node.kind === 'poseedit'
  const isOutpaint = node.kind === 'outpaint'
  const isOutfit = node.kind === 'outfit'
  const isFaceSwap = node.kind === 'faceswap'
  /** 换装/换脸：都是「两图 + 结构化指令」，共用一套正文 */
  const isSwap = isOutfit || isFaceSwap
  /** 是否锁了高度（只有参数视图能锁；图片等比显示，锁高会拉伸） */
  const lockH = !isImg && !imageMode && !!node.h
  /**
   * 卡头上显示的文件名。
   *
   * 图片节点取它载入的那张；生成节点只在「只看图」视图下有意义
   * （此时正文被图占满，没有别的地方能写名字）。参数视图里生成的图
   * 由正文的提示词与状态说明身份，卡头不再重复。
   */
  const headFilePath = isImg || isPoseEdit ? node.image?.path || ''
    : imageMode ? job?.path || '' : ''
  const headFile = headFilePath ? fileNameOf(headFilePath) : ''
  /** 本节点跑的工作流名（和 /models 的 caps.workflows 同一套词） */
  const wf = WORKFLOW_OF[node.kind] || ''
  /**
   * 这个底模跑本节点的工作流是**近似实现**吗。
   *
   * 「近似」= 原生硬约束不存在，靠软引导或后处理凑（Qwen：没有 mask 接口 →
   * 局部重绘是「整图指令改图 + 蒙版贴回」；没有 ControlNet → 姿态靠骨架图软引导）。
   * 必须在卡头上标出来：不标的话用户会拿它当 SD 同款用，然后怪「改了没生效」。
   */
  const isApprox = !!(modelCaps?.[model]?.approx || []).includes(wf)
  /** 图片节点不跑模型，不显示底模下拉 */
  const modelAble = !isImg && !!wf
  /**
   * 本节点能选的底模：按 caps.workflows 过滤。
   *
   * 这是**正确性**过滤而不是洁癖：Qwen 管线没有 unet/tokenizer，ControlNet
   * 类节点选它会在建管线时崩成 "Pipeline expected ['unet', ...]"。
   * 老后端不返回 workflows 时（列表为空）不过滤，退回旧行为。
   */
  const usableModels = (nodesModels || []).filter((m) => {
    const list = modelCaps?.[m]?.workflows
    return !list || !list.length || !wf || list.includes(wf)
  })
  // 当前选中的骨架目录（可能是手填的通配，那就匹配不到，只是没有帧数提示）
  const poseDirInfo = (poseDirs || []).find((d) => d.path === p.psDir)
  const [maskOpen, setMaskOpen] = useState(false)
  const [skelExporting, setSkelExporting] = useState(false)
  const [skelImporting, setSkelImporting] = useState(false)
  // 源图可以是上游节点的产物、图片节点的本地图，也可以是节点上直接记的绝对路径
  //
  // **上游的图有两个落点，必须都认**：
  //   · `job.raw / job.path` —— 各类生成节点的产物
  //   · `image.path`         —— 图片节点，以及**骨骼编辑节点导出的骨架图**
  // 以前这里写成 `sourceNode?.kind === 'image' && sourceNode.image?.path`，漏了
  // poseedit：骨架图明明连上了，卡片却显示「无源图」，`迁移角色` 按钮一直是禁用的
  // ——用户看到的现象就是「骨骼编辑连线，无法迁移角色」。
  // 判断改成看「有没有图」而不是「是不是某一种节点」，以后新增存 image 的节点也不会再漏。
  const upImagePath = sourceNode?.image?.path
  const hasSource = !!(
    sourceNode?.job?.raw ||
    sourceNode?.job?.path ||
    upImagePath ||
    node.source?.path
  )
  // 缩略图同理：image 优先（图片节点 / 骨架图），没有再看生成结果
  const sourceThumb = sourceNode?.image?.url ?? sourceNode?.job?.url
  // 角色迁移的角色图缩略图（角色图节点或本地图）
  const charThumb = charThumbOf?.(node.charSource?.nodeId)
  // 上游是骨骼编辑器时，它的方位角就是默认朝向（"朝向"下拉里显示出来）
  const upAzimuth = sourceNode?.kind === 'poseedit'
    ? (sourceNode.skeleton?.azimuth ?? 0)
    : undefined
  // 角色迁移的引擎：只有 VNCCS 的提交带 `transparent`（旧版 wpc 没有这个字段）
  const isVnccs = (node.params.pcEngine ?? 'wvnccs') === 'wvnccs'
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
    /**
     * 可选中的文字（卡头上的文件名）同样不能启动拖拽。
     *
     * 一旦 setPointerCapture，指针事件全被卡头接管，浏览器原生的
     * 拖选/双击选词就再也收不到事件——表现是「文字选不中，只有节点被拖走」。
     * 这里是**唯一**放行文字选择的地方，别在别处再开 user-select。
     */
    if (el.closest('[data-selectable]')) return
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

  /**
   * 一次改多个参数。**要改 ≥2 个字段时必须用它，不能连调 setParam。**
   *
   * 原因：`setParam` 的展开源是**渲染时闭包里的 `p`**，连调 N 次各自基于
   * 同一份旧对象造新对象，React 合并后只有最后一次生效，前几次静默丢失。
   * 风格下拉一次要改「契约 + 像素块 + 色数 + 档位」4 个字段，连调的后果是
   * 只有 palette 落地——`styleOf()` 发现参数偏离档位就回退成「自定义参数」，
   * 用户看到的现象是**下拉怎么点都弹回去**。
   *
   * 顺带记一条：给这种多字段联动写测试时别只测「自定义」档（它只改一个字段，
   * 恰好绕过 bug），要测真会改多个字段的档位。
   */
  const setParams = (patch: Partial<GenParams>) =>
    onPatch({ params: { ...p, ...patch } })

  /** 图片节点的展示宽度（高度按原图比例自动算） */
  const setImageW = (w: number) => {
    if (!node.image) return
    const cw = Math.max(60, Math.min(900, Math.round(w)))
    onPatch({ image: { ...node.image, w: cw } })
  }

  /**
   * 拖右边缘改宽度 ——— **所有节点统一走这一个手柄**。
   *
   * 图片节点以前用的是 `.img-holder` 里那个小角标（只在图片右下角、只有图片节点有），
   * 于是"怎么变宽"在不同节点上是两套手感：参数节点拖右边缘、图片节点拖图片角落，
   * 而且图片角标跟 `.conn.out` 输出点在同一个位置附近，经常误点到连线。
   * 现在统一成：右边缘 = 只改宽，右下角 = 改宽（+参数节点改高）。
   *
   * 图片节点的宽度存在 `image.w`（不是 `node.w`）——它的高度按原图比例推，
   * 拉宽就是等比放大，不会变形。
   */
  const startWiden = (e: React.PointerEvent) => {
    e.stopPropagation()
    e.preventDefault()
    const startX = e.clientX
    // 图片节点宽度在 image.w；图片视图与参数节点都在 node.w
    const startW = isImg ? node.image?.w ?? NODE_W : node.w ?? NODE_W
    const move = (ev: PointerEvent) => {
      const next = Math.round(startW + (ev.clientX - startX) / zoom)
      if (isImg) {
        setImageW(next)
        return
      }
      onPatch({ w: Math.min(WIDE_MAX, Math.max(WIDE_MIN, next)) })
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  /**
   * 拖右下角：参数节点同时改宽和高，图片类节点（含「只看图」视图）只改宽。
   *
   * **高度是这次补的**：早先只能拖右边缘改宽度，`h` 根本没实现，所以像
   * 「序列帧替换」这种十几个字段的节点会一路撑到 900px 以上，两张就把画布占满，
   * 用户只能靠缩小画布看全，参数又小到看不清。锁高 + 内部滚动才排得开。
   *
   * 图片节点 / 图片视图不锁高：高度由图片比例决定，硬改高会把图压扁——
   * 而且 `h` 在 `lockH` 为假时根本不会写进 style，拖了也不动，等于假手柄。
   */
  const startCorner = (e: React.PointerEvent) => {
    e.stopPropagation()
    e.preventDefault()
    const sx = e.clientX
    const sy = e.clientY
    const el = (e.currentTarget as HTMLElement).closest('.node') as HTMLElement | null
    if (isImg || imageMode) {
      const sw = isImg ? node.image?.w ?? NODE_W : node.w ?? NODE_W
      const move = (ev: PointerEvent) => {
        const next = sw + (ev.clientX - sx) / zoom
        if (isImg) {
          setImageW(next)
          return
        }
        onPatch({ w: Math.min(WIDE_MAX, Math.max(WIDE_MIN, Math.round(next))) })
      }
      const up = () => {
        window.removeEventListener('pointermove', move)
        window.removeEventListener('pointerup', up)
      }
      window.addEventListener('pointermove', move)
      window.addEventListener('pointerup', up)
      return
    }
    const sw = node.w ?? NODE_W
    // 起始高度取**实际渲染高度**：h 未设置时节点是内容撑开的，读不到就是 0
    const sh = node.h ?? (el ? Math.round(el.getBoundingClientRect().height / zoom) : 320)
    const move = (ev: PointerEvent) => {
      const dw = (ev.clientX - sx) / zoom
      const dh = (ev.clientY - sy) / zoom
      onPatch({
        w: Math.min(WIDE_MAX, Math.max(WIDE_MIN, Math.round(sw + dw))),
        h: Math.min(TALL_MAX, Math.max(TALL_MIN, Math.round(sh + dh))),
      })
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }


  /** 源图区：图生图与抠图节点共用（连线接上来的源，或直接拖一张图） */
  /** 「额外参考图」列表：角色迁移与换装/换脸共用。角色迁移里是同一角色的
   *  不同角度，换装/换脸里是同一套服装/同一个人的不同角度——模型看到多
   *  张会更稳，指令里也会显式声明"这是同一件/同一人"（swap.multi_swap_prompt）。 */
  const extraRefBox = (
      <div className="pc-extra">
        <div className="pc-extra-head">
          额外参考图
          <span className="pc-extra-count">
            {(node.charSources ?? []).length}/{MAX_CHAR_SOURCES - 1}
          </span>
        </div>
        {(node.charSources ?? []).map((c, i) => (
          <div className="pc-extra-row" key={i}>
            <select
              value={c.nodeId ?? ''}
              onChange={(e) => {
                const next = [...(node.charSources ?? [])]
                next[i] = e.target.value ? { nodeId: e.target.value } : {}
                onPatch({ charSources: next.filter((x) => x.nodeId) })
              }}
            >
              <option value="">（选参考图）</option>
              {sourceOptions.map((o) => (
                <option key={o.id} value={o.id}>{o.label}</option>
              ))}
            </select>
            <button
              className="pc-extra-del"
              title="移除这张"
              onClick={() => {
                const next = [...(node.charSources ?? [])]
                next.splice(i, 1)
                onPatch({ charSources: next })
              }}
            >
              ×
            </button>
          </div>
        ))}
        {(node.charSources ?? []).length < MAX_CHAR_SOURCES - 1 && (
          <button
            className="pc-extra-add"
            onClick={() => onPatch({
              charSources: [...(node.charSources ?? []), {}],
            })}
          >
            + 追加参考图
          </button>
        )}
        <div className="src-note">
          {(node.charSources ?? []).length
            ? '同一角色的不同角度一起喂，发型和服装细节更稳'
            : '可选：加一张侧面或细节图，身份会更稳（最多 '
              + (MAX_CHAR_SOURCES - 1) + ' 张）'}
        </div>
      </div>
  )

  const sourceBox = (
    <div className="src-box">
      {/*
        输入端口挂在这里，而不是节点绝对定位的某个 top 值。
        原因：不同节点的正文块数不一样（转像素在源图区上面还有一行说明），
        写死偏移后端口就会悬在说明文字旁边、和它管的输入框错开一截。
        位置由 Canvas 实测 DOM 得到，线端点跟着端口走。
      */}
      <span
        className="conn in src"
        data-port="source"
        title="源图 / 动作参考：把上游节点的输出点拖到这里"
      />
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
          /* 图生图是**指令改图**：没有 strength 语义，改多少全看提示词。
             留个滑块在这儿只会让人以为调小就改得少（实际完全无效）。 */
          <p className="src-note">
            走「指令改图」：改成什么样由提示词决定，没有重绘幅度可调。
          </p>
        )}
        <div className="src-note">
          {sourceNode?.kind === 'poseedit'
            ? '用导出的骨架图（OpenPose 18 点）'
            : sourceNode?.kind === 'image'
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
    // pointerover / pointerout 用**冒泡**事件而不是 enter / leave：
    // enter/leave 是 React 拿 over/out + relatedTarget 模拟的，不冒泡，
    // 某些派发路径下（jsdom、合成事件）会静默不触发，现象是"拖了线松手却
    // 什么都没连上"。over/out 一定冒泡到根，再按 relatedTarget 是否仍在元素内
    // 滤掉子元素间的抖动。
    <div
      className={`node gen${selected ? ' selected' : ''}${imageMode ? ' image-mode' : ''}${
        isImg ? ' is-img' : ''
      }${lockH ? ' fixed-h' : ''}${linkTarget ? ' link-target' : ''}`}
      data-node-id={node.id}
      style={{
        left: node.x,
        top: node.y,
        width: isImg ? node.image?.w ?? NODE_W : node.w ?? NODE_W,
        // 高度只对参数视图生效：图片/图片视图是等比显示，锁高会把图压扁
        height: lockH ? node.h : undefined,
      }}
      onPointerDown={(e) => {
        e.stopPropagation()
        onSelect()
      }}
      onPointerOver={(e) => {
        const to = e.relatedTarget as Node | null
        if (to && (e.currentTarget as Node).contains(to)) return
        onHover?.(node.id)
      }}
      onPointerOut={(e) => {
        const to = e.relatedTarget as Node | null
        if (to && (e.currentTarget as Node).contains(to)) return
        onHover?.(null)
      }}
      onDragOver={(e) => {
        // 本机图片拖到节点上：落下时把它当该节点的源
        if (e.dataTransfer.types.includes('Files')) {
          e.preventDefault()
          onDragOverNode?.(node.id)
        }
      }}
    >
      {/*
        输入端口不在这里画——它们挂在各自的输入框上（源图区 / 角色图区），
        这样端口永远挨着它管的那块 UI，不会因为节点正文多一行说明就漂走。
        位置由 Canvas 实测（见 portTops），连线端点跟着走。
      */}
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
          }${isPoseClone ? ' poseclone' : ''}${isSpriteSwap ? ' spriteswap' : ''}${
            isPoseSwap ? ' poseswap' : ''
          }${isToPixel ? ' topixel' : ''}`}
        >
          {isOutpaint ? '扩图' : isOutfit ? '换装' : isFaceSwap ? '换脸'
            : isPoseEdit ? '骨骼编辑' : isImg ? '图片' : isBg ? '抠图' : isPixelify ? '像素转换'
            : isToPixel ? '转像素'
            : isPoseClone ? '角色迁移'
              : isPoseSwap ? '姿态序列'
              : isSpriteSwap ? '序列帧替换'
                : isAiRepaint ? '局部AI重绘'
                  : isInpaint ? '局部重绘' : isI2I ? '图生图' : '图生成'}
        </span>
        {/* 文件名放卡头：画布上常同时摆着好几张图/好几帧，只有缩略图时
            根本分不清哪张是哪张。放在正文底部要滚动才看得到，卡头一眼就能看见，
            而且 data-selectable 让它能被鼠标选中复制（见 startDrag 的绕过）。
            名字剥掉上传时拼的 10 位 hash 前缀，否则有用部分被挤没。 */}
        {headFile && (
          <span className="node-file" data-selectable="1" title={headFilePath || headFile}>
            {headFile}
          </span>
        )}
        {job && <span className={`node-state s-${job.status}`}>{statusLabel(job.status)}</span>}
        {/* 有文件名时不留 spacer：让文件名自己长满中间那段，
            否则 spacer 会把剩余宽度全吃掉，名字只剩「w1_2026…」几个字符。 */}
        {!headFile && <span className="spacer" />}
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
        {/* 图片视图下不给预览按钮：正文里的图本身就点得开全屏，
            省下的 22px 让给卡头的文件名（名字比"多一个入口"更值钱） */}
        {previewUrl && !imageMode && (
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

      {/*
        底模：**节点级选择**。放在正文最前面（不是塞进「更多设置」）。
        原因：底模决定"画成什么样"，是仅次于提示词的决定项；一张画布上
        经常要混跑（Qwen 出立绘 + SDXL 出精灵帧），藏在二级菜单里等于逼用户
        全局来回切，而每次全局切都会把各节点调好的参数冲一遍。
        只列出**本节点真能跑**的底模（Qwen 没有 SD 的 UNet/CLIP，
        角色迁移/序列帧替换选它就是必崩，不如不列）。
      */}
      {modelAble && usableModels.length > 0 && (
        <div className="node-model">
          <span className="node-model-label">底模</span>
          <select
            value={node.params.model || ''}
            title={'这个节点用哪个底模。\n'
              + '「跟随默认」= 跟着侧栏的新节点默认值走；选了具体值就固定住，'
              + '以后侧栏再改也不影响它。\n'
              + '切换会套用新底模的推荐尺寸/步数（手调过的参数不会被冲掉）。'}
            onChange={(e) => {
              const v = e.target.value
              if (!v) {
                // 清空 = 回到「跟随默认」。只改字段，不动参数——
                // 用户只是想让它跟默认走，不是想重置参数
                onPatch({ params: { ...p, model: '' } })
                return
              }
              if (v === model) return
              onModel?.(v)
            }}
          >
            <option value="">跟随默认（{shortModel(defaultModel || model)}）</option>
            {usableModels.map((m) => (
              <option key={m} value={m}>{shortModel(m)}</option>
            ))}
          </select>
          {isApprox && (
            <span
              className="node-model-approx"
              title={'这个底模跑这个节点是「近似实现」：\n'
                + '它没有该工作流原本依赖的硬约束（Qwen 没有 mask 接口、'
                + '也没有 ControlNet），改由「整图指令改图 + 蒙版贴回」或'
                + '「骨架图当参考 + 指令」实现。\n'
                + '· 局部重绘：选区外保证逐点不变，选区内不保证只改那一点\n'
                + '· 姿态/序列帧：姿势是软引导，上身跟随较好、腿部常被参考图的站姿覆盖，'
                + '动作幅度偏保守（详见 docs/PLAN-QWEN-ONLY.md §2.8）'}
            >
              近似
            </span>
          )}
          {usableModels.length < (nodesModels?.length ?? 0) && (
            <span
              className="node-model-note"
              title={'有些底模没列出来，是因为它们跑不了这个节点：\n'
                + 'Qwen-Image 的管线没有 SD 的 UNet / CLIP 文本编码器，'
                + '而 ControlNet / IP-Adapter 这套组件只有 SD 系有。'}
            >
              ⓘ
            </span>
          )}
        </div>
      )}

      {isPoseEdit ? (
        <div className="node-body">
          <SkeletonEditor
            state={node.skeleton ?? defaultSkeleton()}
            onChange={(st) => onPatch({ skeleton: st })}
            onExport={(blob) => {
              setSkelExporting(true)
              try { onExportSkeleton?.(blob) } finally {
                window.setTimeout(() => setSkelExporting(false), 600)
              }
            }}
            exporting={skelExporting}
            importing={skelImporting}
            onImportImage={async (f) => {
              setSkelImporting(true)
              try { await onImportPose?.(f) } finally { setSkelImporting(false) }
            }}
          />
          {node.image?.url && (
            <div className="skel-preview">
              <img src={node.image.url} alt="导出的骨架图" />
              <span>已导出 · {fileNameOf(node.image.path)}</span>
            </div>
          )}
        </div>
      ) : isPoseSwap ? (
        <div className="node-body poseclone-body">
          <div className="src-box">
            <div className="src-side" style={{ flex: 1 }}>
              <select
                value={poseDirInfo ? p.psDir : ''}
                onChange={(e) => setParam('psDir', e.target.value)}
                title="用 tools/pose_from_image.py 从图片/视频/GIF 提骨架，产物是一个目录"
              >
                <option value="">（选骨架帧目录）</option>
                {(poseDirs || []).map((d) => (
                  <option key={d.path} value={d.path}>
                    {d.name}（{d.count} 帧）
                  </option>
                ))}
              </select>
              <input
                className="dir-input"
                value={p.psDir}
                placeholder="或手填目录 / 通配：assets/_pose_walk_*.png"
                onChange={(e) => setParam('psDir', e.target.value)}
              />
              <div className="src-note">
                {poseDirInfo
                  ? `这个目录有 ${poseDirInfo.count} 帧 → 出 ${poseDirInfo.count} 帧精灵图`
                  : p.psDir
                    ? '手填路径（后端展开成排序后的帧列表，按文件名自然序）'
                    : '缺骨架帧：先跑 tools/pose_from_image.py，或选已有目录'}
              </div>
            </div>
          </div>
          <div className="src-box char-box">
            <span
              className="conn in char"
              data-port="char"
              title="角色图（身份来源，可选）：把上游节点的输出点拖到这里"
            />
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
                <option value="">（选角色图，可选）</option>
                {sourceOptions.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
              <div className="src-note">
                {charThumb
                  ? '给了角色图：帧间身份更一致，但它也会「抢姿态」（动作偏保守）'
                  : '不选也能跑：角色只靠文字描述，帧间身份可能漂'}
              </div>
            </div>
          </div>

          {/* 逐帧引擎：VNCCS（帧间身份更稳）/ 旧版 */}
          <div className="pc-engine">
            <label className="pc-engine-label">引擎</label>
            <select
              value={p.psEngine ?? 'wvnccs'}
              onChange={(e) => setParam('psEngine', e.target.value as 'wpc' | 'wvnccs')}
            >
              <option value="wvnccs">VNCCS（帧间身份稳 · 推荐）</option>
              <option value="wpc">旧版（动作更保守）</option>
            </select>
            <div className="src-note">
              {p.psEngine === 'wpc'
                ? '角色图 + 骨架图 + 自写指令：软引导，腿部常被角色图的站姿覆盖'
                : '专用 LoRA + 官方指令：图序 [骨架, 角色]，帧间身份更稳'}
            </div>
            {(p.psEngine ?? 'wvnccs') === 'wvnccs' && !charThumb && (
              <div className="src-note pc-warn">
                VNCCS 需要角色图（它的指令是「保留角色的形象」）——先在上面选一张
              </div>
            )}
            {(p.psEngine ?? 'wvnccs') === 'wvnccs' && charThumb && (
              <div className="src-note pc-warn">
                角色外观要写进提示词：模型会让 prompt 压过参考图
              </div>
            )}
          </div>

          {/* 同一角色的不同角度可以一起喂，身份更稳 */}
          {extraRefBox}
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={2}
              placeholder="角色描述？e.g. white cat knight, red cape, katana, front view"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>
          <details className="adv">
            <summary>更多设置</summary>
            <label
              className="check"
              title={'两阶段：先用骨架出一张姿态草图，再用角色图把它换皮。\n'
                + '实测姿态偏差 102px → 77px（动作更准），代价是每帧多一次生成'
                + '（33s → 50s），而且服装细节会漂（白袍可能被画成金甲）。\n'
                + '需要先在上面选一个角色图。'}
            >
              <input
                type="checkbox"
                checked={p.twoStage}
                disabled={!charThumb}
                onChange={(e) => setParam('twoStage', e.target.checked)}
              />
              两阶段（动作更准，慢一倍）
            </label>
            {/* 底色：逐帧生成时用。白毛/白甲角色配白底，抠图会把脸和腿当成
                背景一起吃掉（实测白猫骑士丢头顶+胸口，或丢两条腿）。 */}
            <label className="bg-mode" title={'生成时的底色。\n'
              + '角色偏白（白毛、白甲）时白底会让抠图分不清主体与背景，'
              + '选「深色底」靠对比度抠更干净。深色系角色反过来仍用白底。'}>
              底色
              <select
                value={p.bgTone}
                onChange={(e) => setParam('bgTone', e.target.value as typeof p.bgTone)}
              >
                <option value="white">白底（默认）</option>
                <option value="dark">深色底（白毛角色）</option>
              </select>
            </label>
            <label className="strength" title={hint('列')}>
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
            <label className="strength" title={hint('候选')}>
              <span>候选</span>
              <input
                type="number"
                min={1}
                max={candCap}
                value={p.ssCandidates}
                onChange={(e) => setParam('ssCandidates',
                  Math.min(candCap, Math.max(1, Number(e.target.value) || 1)))}
              />
              <b>张/帧</b>
            </label>
            <label className="strength" title={hint('补提')}>
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
            <label className="strength" title={hint('色板')}>
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
            <div className="src-note">
              每帧生成 {candCap >= p.ssCandidates ? p.ssCandidates : candCap} 个候选，
              按「关节数 + 前景占比 + 贴边 + 帧间漂移」挑一张；
              都没过闸门才补提。上限 {candCap} 张/帧由显存算出。
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
              <img src={job!.url} alt="姿态序列结果" draggable={false} />
            </div>
          )}
          <button
            className="gen-btn"
            disabled={busy || !p.psDir.trim() || !p.prompt.trim()}
            onClick={onGenerate}
            title={
              !p.psDir.trim() ? '先选一个骨架帧目录'
                : !p.prompt.trim() ? '写上角色长什么样' : undefined
            }
          >
            {busy ? '生成中…' : '生成精灵帧'}
          </button>
          {job?.status === 'failed' && <div className="err">{job.error}</div>}
          {job?.status === 'done' && (
            <div className="meta">
              种子 {job.seed ?? p.seed}
              {typeof job.seconds === 'number' ? ` · ${job.seconds}s` : ''}
              {typeof sCand === 'number' ? ` · 候选 ${sCand}/帧` : ''}
              {typeof sAttempts === 'number' && typeof sFrames === 'number'
                && typeof sCand === 'number' && sAttempts > sFrames * sCand
                ? ` · 补提 ${sAttempts - sFrames * sCand} 次` : ''}
              {typeof sGate === 'number' && typeof sFrames === 'number'
                ? ` · 闸门 ${sGate}/${sFrames}` : ''}
            </div>
          )}
          {job?.status === 'done' && job.candidatesNote && (
            <div className="warn">⚠ {job.candidatesNote}</div>
          )}
        </div>
      ) : isSpriteSwap ? (
        <div className="node-body poseclone-body">
          {sourceBox}
          <div className="src-box char-box">
            <span
              className="conn in char"
              data-port="char"
              title="角色图（身份来源）：把上游节点的输出点拖到这里"
            />
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

          {/* 序列帧也可以给多张角色图：正面 + 侧面，逐帧复用同一组 */}
          {extraRefBox}
          <label className="bg-mode" title={hint('方式')}>
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
            <label className="bg-mode" title={hint('引擎')}>
              引擎
              <select
                value={p.ssEngine ?? 'wvnccs'}
                onChange={(e) => setParam('ssEngine', e.target.value as 'wpc' | 'wvnccs')}
              >
                <option value="wvnccs">VNCCS（帧间角色一致 · 推荐）</option>
                <option value="wpc">旧版（单帧更快 · 易换脸）</option>
              </select>
            </label>
          )}
          {p.ssMode === 'generate' && (p.ssEngine ?? 'wvnccs') === 'wvnccs' && (
            <div className="src-note pc-warn">
              逐帧挂 VNCCS LoRA：单帧更慢（25 步起步），换来整段动画里角色
              不换脸、不换衣服。角色外观一定要写进提示词——模型会让 prompt
              压过参考图。
            </div>
          )}
          {p.ssMode === 'generate' && (
            <div className="prompt-wrap">
              <textarea
                className="prompt"
                rows={2}
                placeholder="可选：补充描述（外观已由角色图决定）"
                value={p.prompt}
                onChange={(e) => setParam('prompt', e.target.value)}
              />
            </div>
          )}
          <details className="adv">
            <summary>更多设置</summary>
            <label className="strength" title={hint('列')}>
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
            <label className="strength" title={hint('行')}>
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
                <label className="bg-mode" title={hint('参考')}>
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
                <label className="bg-mode" title={'生成时的底色：角色偏白（白毛、白甲）时，白底会让抠图把脸和腿一起判成背景；选「深色底」靠对比度抠更干净。深色系角色反过来仍用白底。'}>
                  底色
                  <select
                    value={p.bgTone}
                    onChange={(e) => setParam('bgTone', e.target.value as typeof p.bgTone)}
                  >
                    <option value="white">白底</option>
                    <option value="dark">深色底</option>
                  </select>
                </label>
              </>
            )}
            <label className="strength" title={hint('色板')}>
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
            {p.ssMode === 'generate' && (
              <>
                <label className="strength" title={hint('候选')}>
                  <span>候选</span>
                  <input
                    type="number"
                    min={1}
                    /**
                     * 上限来自后端（按本机总显存算），不是写死的 6。
                     * 显存打满时不报 OOM，但耗时能翻 3 倍——那个坑靠用户猜是猜不到的。
                     */
                    max={candCap}
                    value={p.ssCandidates}
                    onChange={(e) => setParam('ssCandidates',
                      Math.min(candCap, Math.max(1, Number(e.target.value) || 1)))}
                  />
                  <b>张/帧</b>
                </label>
                <label className="strength" title={hint('补提')}>
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
                  但更慢。上限 {candCap} 张/帧由当前底模的显存占用算出——
                  超了不会报错，只会静默变慢（显存打满时耗时能翻 3 倍）。
                </div>
              </>
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
            disabled={busy || !hasSource || !charThumb}
            onClick={onGenerate}
            title={
              !hasSource ? '先接一张动作序列帧（精灵图）'
                : !charThumb ? '再选一张角色图' : undefined
            }
          >
            {busy ? '替换中…' : '替换序列帧'}
          </button>
          {/* 序列帧节点走的是独立正文分支，通用结果区不会渲染到这里。
              而这三个数正是"要不要重跑"的判据：候选数不等于设定值 = 被显存降级了。 */}
          {job?.status === 'done' && (
            <div className="meta">
              种子 {job.seed ?? p.seed}
              {typeof sCand === 'number' ? ` · 候选 ${sCand}/帧` : ''}
              {typeof sAttempts === 'number' && typeof sFrames === 'number'
                && typeof sCand === 'number' && sAttempts > sFrames * sCand
                ? ` · 补提 ${sAttempts - sFrames * sCand} 次` : ''}
              {typeof sGate === 'number' && typeof sFrames === 'number'
                ? ` · 闸门 ${sGate}/${sFrames}` : ''}
            </div>
          )}
          {job?.status === 'done' && job.candidatesNote && (
            <div className="warn">⚠ {job.candidatesNote}</div>
          )}
        </div>
      ) : isOutpaint ? (
        <div className="node-body">
          {sourceBox}
          <div className="skel-dirs" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
            {([['right', '右'], ['left', '左'], ['down', '下'], ['up', '上'],
              ['all', '四周']] as const).map(([v, label]) => (
              <button
                key={v}
                className={`skel-dir${p.outpDirection === v ? ' on' : ''}`}
                onClick={() => setParam('outpDirection', v)}
                title={`往${label}扩展`}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="strength" title="扩展比例：0.5 = 往该方向扩一半">
            <span>扩展</span>
            <input
              type="range" min={0.15} max={1.5} step={0.05}
              value={p.outpExpand}
              onChange={(e) => setParam('outpExpand', Number(e.target.value))}
            />
            <b>{Math.round(p.outpExpand * 100)}%</b>
          </label>
          <div className="prompt-wrap">
            <textarea
              className="prompt" rows={2}
              placeholder="可选：描述原图里看得见的内容（给模型当上下文，别描述要扩出来的部分）"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>
          <div className="src-note">
            靠 Outpaint LoRA + 中性灰填充：原图区域贴回，新区域按场景连续性补全。
            一次别扩太大（超过 1.0 容易跑偏），要更大就分两次扩。
          </div>
        </div>
      ) : isSwap ? (
        <div className="node-body poseclone-body">
          {sourceBox}
          <div className="src-box char-box">
            <span
              className="conn in char"
              data-port="char"
              title={isOutfit ? '服装图：把上游输出拖到这里' : '人脸图：把上游输出拖到这里'}
            />
            <div className="src-thumb" title={isOutfit ? '服装图' : '人脸图'}>
              {charThumb ? (
                <img src={charThumb} alt="参考" draggable={false} />
              ) : (
                <span>{isOutfit ? '无服装' : '无人脸'}</span>
              )}
            </div>
            <div className="src-side">
              <select
                value={node.charSource?.nodeId ?? ''}
                onChange={(e) =>
                  onPatch({
                    charSource: e.target.value ? { nodeId: e.target.value } : undefined,
                  })
                }
              >
                <option value="">{isOutfit ? '（选服装图）' : '（选人脸图）'}</option>
                {sourceOptions.map((o) => (
                  <option key={o.id} value={o.id}>{o.label}</option>
                ))}
              </select>
              <div className="src-note">
                {charThumb
                  ? (isOutfit ? '服装的剪裁/颜色/材质来自这张' : '脸/发型来自这张')
                  : (isOutfit ? '缺服装图' : '缺人脸图')}
              </div>
            </div>
          </div>

          {/* 换装/换脸也可以给多张：同一件套的背面/细节，或同一个人的正脸/侧脸 */}
          {extraRefBox}

          <div className="prompt-wrap">
            <textarea
              className="prompt" rows={2}
              placeholder="可选：补充细节（如「外套改成皮夹克」「笑得自然些」）"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>
          <div className="src-note">
            {isOutfit
              ? '换装：保留图 1 的身份/姿势/背景，穿上图 2 的服装。'
              : '换脸：保留图 1 的姿势/服装/场景，换成图 2 的脸与发型。'}
          </div>
        </div>
      ) : isPoseClone ? (
        <div className="node-body poseclone-body">
          {sourceBox}
          <div className="src-box char-box">
            <span
              className="conn in char"
              data-port="char"
              title="角色图（身份来源）：把上游节点的输出点拖到这里"
            />
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
                {charThumb ? '第一张角色图（身份来源）' : '缺角色图'}
              </div>
            </div>
          </div>

          {/* 迁移引擎：两条路线的差别见 api.ts 的 submitVnccsPose 注释 */}
          <div className="pc-engine">
            <label className="pc-engine-label">引擎</label>
            <select
              value={node.params.pcEngine ?? 'wvnccs'}
              onChange={(e) => setParam('pcEngine', e.target.value as 'wpc' | 'wvnccs')}
            >
              <option value="wvnccs">VNCCS（保角色 · 推荐）</option>
              <option value="wpc">旧版（姿态更硬 · 外观易飘）</option>
            </select>
            <div className="src-note">
              {node.params.pcEngine === 'wvnccs'
                ? '专用 LoRA + 双图指令：画风/发型/服装一致，头不丢、不多腿；姿态精度略逊'
                : '自写指令的软引导：姿态贴得更紧，但画风和细节容易跑；方形画布下容易丢头'}
            </div>
          </div>

          {extraRefBox}

          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={2}
              placeholder="可选：补充描述（外观已由角色图决定）"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>

          {/* 朝向：**必选输入，不是可选项**。
              骨架图本身分不清角度——正/背面在 2D 投影里互为镜像，90° 侧面收成一条
              竖线，实测同一张 90° 骨架不加方位词会被画成正面、加一句 side view 才是
              侧面。所以这里把它做成显式选择，默认跟骨骼编辑器的方位角。 */}
          <label className="bg-mode" title={'人物的朝向。骨架图本身看不出角度'
            + '（正/背面互为镜像、侧面收成一条线），必须明确告诉模型。\n'
            + '「自动」= 跟随上游骨骼编辑器里选的方位角；上游不是骨骼编辑器时'
            + '自动等于「不指定」，模型会按正面画。'}>
            朝向
            <select
              value={p.pcView ?? 'auto'}
              onChange={(e) => setParam('pcView', e.target.value as typeof p.pcView)}
            >
              <option value="auto">
                自动{upAzimuth !== undefined ? `（骨骼 ${upAzimuth}°）` : ''}
              </option>
              <option value="front">正面</option>
              <option value="q34front_r">右前 3/4</option>
              <option value="side_r">右侧面</option>
              <option value="q34back_r">右后 3/4</option>
              <option value="back">背面</option>
              <option value="q34back_l">左后 3/4</option>
              <option value="side_l">左侧面</option>
              <option value="q34front_l">左前 3/4</option>
            </select>
          </label>
          <details className="adv">
            <summary>更多设置</summary>
            {/* 「姿态」「身份」两根强度滑块是 SD 那条路（ControlNet / IP-Adapter）
                专属的，已随 SD 一起删除。现在是**双图条件**：角色图 + 骨架图进同一个
                vision context，姿势与身份由模型自己分配。 */}
            <div className="src-note">
              双图条件：角色图 + 骨架图一起进模型，姿势与身份由它自己分配，
              没有姿态/身份强度可调。动作跟随的实际水平见 docs/PLAN-QWEN-ONLY.md §2.8。
            </div>
            <label className="strength" title={hint('占高')}>
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
            <label className="bg-mode" title={hint('参考')}>
              参考
              <select
                value={p.pcRefMode}
                onChange={(e) => setParam('pcRefMode', e.target.value as typeof p.pcRefMode)}
              >
                <option value="auto">自动（骨架→轮廓）</option>
                <option value="pose">骨架（真人照片）</option>
                <option value="outline">轮廓（游戏素材）</option>
                <option value="skeleton">已画好的骨架图（骨骼编辑节点）</option>
              </select>
            </label>
            <div className="src-note">
              参考模式「自动」会先试骨架，认不出人体就改用轮廓——游戏像素素材也能用
            </div>
            {/* 底色：白毛/白甲角色配白底，抠图会把脸和腿当成背景一起吃掉
                （实测白猫骑士丢头顶+胸口，或丢两条腿）。换深底就靠对比度抠。 */}
            <label className="bg-mode" title={'生成时的底色。\n'
              + '角色本身偏白（白毛、白甲）时白底会让抠图分不清主体与背景，'
              + '把脸/腿一起判成背景 —— 这种情况选「深色底」，靠对比度抠干净。\n'
              + '深色系角色（黑袍、黑甲）反过来仍用白底。\n'
              + '勾了「透明背景」之后这个选项就没意义了（模型直接出 alpha）。'}>
              底色
              <select
                value={p.bgTone}
                disabled={!!p.transparent && isVnccs}
                onChange={(e) => setParam('bgTone', e.target.value as typeof p.bgTone)}
              >
                <option value="white">白底（默认）</option>
                <option value="dark">深色底（白毛角色）</option>
              </select>
            </label>
            {/* 透明底：Qwen 原生 RGBA。VNCCS 下实测可用（主体 16.6%、
                半透明 2.1%、背景全透），不像 ControlNet 那条路会被控制信号压成半透膜。
                旧版引擎（wpc）的提交里没有这个字段，所以只在 VNCCS 下可勾。 */}
            <label
              className="check"
              title={'模型直接输出 alpha 通道，不用再抠图（抠图会把白鞋、发丝一起吃掉）。\n'
                + 'VNCCS 引擎实测：主体 16.6% / 半透明 2.1% / 背景全透。\n'
                + '旧版引擎不支持原生 alpha，需要透明底时请用 VNCCS。'}
            >
              <input
                type="checkbox"
                checked={!!p.transparent && isVnccs}
                disabled={!isVnccs}
                onChange={(e) => setParam('transparent', e.target.checked)}
              />
              透明背景（原生 RGBA）
              {!isVnccs && <span className="dim"> 仅 VNCCS 引擎</span>}
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
              <img src={job!.url} alt="角色迁移结果" draggable={false} />
            </div>
          )}
          <button
            className="gen-btn"
            disabled={busy || !hasSource || !charThumb}
            onClick={onGenerate}
            title={
              !hasSource ? '先接一张动作参考图'
                : !charThumb ? '再选一张角色图' : undefined
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
      ) : isToPixel ? (
        <div className="node-body animepixel-body">
          {/* 说明只留两行：300px 宽的卡片里塞四行长文会把整个面板顶满，
              下面的源图区和提示词框全被挤走（实测截图确认）。细节放 tooltip。 */}
          <div className="src-note">任意图都行：照片 / 插画 / 3D / 动漫。有源图直接转，没有就按描述先出一张。</div>
          {sourceBox}
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={3}
              placeholder={
                hasSource
                  ? '（有源图，提示词可留空）'
                  : '1girl, red cape, holding sword, full body'
              }
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>
          <div className="px-opts">
            <label className="strength" title={hint('像素边长')}>
              <span>边长</span>
              <input
                type="number"
                min={16}
                max={512}
                value={p.apTarget}
                onChange={(e) => setParam('apTarget', Number(e.target.value) || 192)}
              />
              <b>px</b>
            </label>
            <label className="strength" title={hint('色数')}>
              <span>色数</span>
              <input
                type="number"
                min={2}
                max={256}
                value={p.apColors}
                onChange={(e) => setParam('apColors', Number(e.target.value) || 32)}
              />
              <b>色</b>
            </label>
            <label className="strength" title={hint('对比度增强')}>
              <span>增强</span>
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={p.apPreEnhance}
                onChange={(e) => setParam('apPreEnhance', Number(e.target.value) || 0)}
              />
              <b>{p.apPreEnhance > 0 ? '×' : '关'}</b>
            </label>
            <label className="check" title={hint('描边')}>
              <input
                type="checkbox"
                checked={p.apEdgeRebuild}
                onChange={(e) => setParam('apEdgeRebuild', e.target.checked)}
              />
              轮廓描边
            </label>
            <label className="check" title={hint('透明背景')}>
              <input
                type="checkbox"
                checked={p.transparent && canTransparent}
                disabled={!canTransparent}
                onChange={(e) => setParam('transparent', e.target.checked)}
              />
              透明背景{p.transparent && canTransparent ? '（原生）' : ''}
            </label>
            <label
              className="check"
              title="先抠掉背景，得到透明底的精灵；转风景/照片时建议关掉"
            >
              <input
                type="checkbox"
                checked={p.apMatte && !(p.transparent && canTransparent)}
                disabled={p.transparent && canTransparent}
                onChange={(e) => setParam('apMatte', e.target.checked)}
              />
              抠背景{p.transparent && canTransparent ? '（已用原生 alpha）' : ''}
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
            disabled={busy || (!hasSource && !p.prompt.trim())}
            onClick={onGenerate}
            title={
              !hasSource && !p.prompt.trim()
                ? '接一张图，或写上角色描述（会先生成动漫全身图再转像素）'
                : undefined
            }
          >
            {busy ? '转换中…' : hasSource ? '转成像素' : '生成像素画'}
          </button>
        </div>
      ) : isPixelify ? (
        <div className="node-body pixelify-body">
          {sourceBox}
          <div className="px-opts">
            <label className="strength" title={hint('边长')}>
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
            <label className="strength" title={hint('色数')}>
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
            <label className="bg-mode" title={hint('取色')}>
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
            <label className="strength" title={hint('吸附')}>
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
            <label className="check" title={hint('网格')}>
              <input
                type="checkbox"
                checked={p.pxGridSnap}
                onChange={(e) => setParam('pxGridSnap', e.target.checked)}
              />
              网格检测（关掉=普通降采样）
            </label>
            <label className="check" title={hint('透明底')}>
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
                    <p className="src-note">
            Qwen 没有重绘幅度，改成什么样由提示词里的指令决定。它是
            <b>近似实现</b>：整图作条件、蒙版只负责贴回——选区外保证逐点不变，
            选区内不保证只改那一点（猜偏的部分会被蒙版裁掉，表现是「改得不够」）。
          </p>
          <details className="adv">
            <summary>更多设置</summary>
            <label className="strength" title={hint('上下文')}>
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
            <label className="strength" title={hint('外扩')}>
              <span>外扩</span>
              {/* 拖滑块比点数字框快得多，而且下面的预览会跟着变——
                  改这两个参数不用再点一次生成才知道效果 */}
              <input
                type="range"
                min={0}
                max={32}
                step={1}
                value={p.rpExpand}
                onChange={(e) => setParam('rpExpand', Number(e.target.value))}
              />
              <b>{p.rpExpand}px</b>
            </label>
            <label className="strength" title={hint('羽化')}>
              <span>羽化</span>
              <input
                type="range"
                min={0}
                max={16}
                step={1}
                value={p.rpFeather}
                onChange={(e) => setParam('rpFeather', Number(e.target.value))}
              />
              <b>{p.rpFeather}px</b>
            </label>
            <label className="check" title={hint('色彩校正')}>
              <input
                type="checkbox"
                checked={p.rpMatchColor}
                onChange={(e) => setParam('rpMatchColor', e.target.checked)}
              />
              接缝色彩校正（只在边界带）
            </label>
            <label className="check" title={hint('色板吸附')}>
              <input
                type="checkbox"
                checked={p.rpPaletteSnap}
                onChange={(e) => setParam('rpPaletteSnap', e.target.checked)}
              />
              色板吸附（像素素材用）
            </label>
          </details>
          {/* 几何预览：外扩/羽化是像素级参数，改完不用再点一次生成 */}
          <MaskPreview
            srcUrl={sourceNode?.job?.url || sourceNode?.image?.url}
            maskUrl={p.maskUrl || undefined}
            expand={p.rpExpand}
            feather={p.rpFeather}
          />
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
                    <p className="src-note">
            Qwen 没有重绘幅度，改成什么样由提示词里的指令决定。它是
            <b>近似实现</b>：整图作条件、蒙版只负责贴回——选区外保证逐点不变，
            选区内不保证只改那一点（猜偏的部分会被蒙版裁掉，表现是「改得不够」）。
          </p>
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
            <label className="bg-mode" title={hint('背景方式')}>
              <input
                type="radio"
                checked={p.bgMode === 'alpha'}
                onChange={() => setParam('bgMode', 'alpha')}
              />
              透明（去背景）
            </label>
            <label className="bg-mode" title={hint('背景方式')}>
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
            <label className="bg-mode" title={hint('背景方式')}>
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
              </div>
              <div className="img-tools">
                <label title={hint('图片宽')}>
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
            {/* 图片视图是序列帧生成完最常见的状态，候选/闸门也得在这显示：
                切到只看图之后才发现"候选被降到 1"已经太晚了。 */}
            {typeof sCand === 'number' && sCand > 1 ? ` · 候选 ${sCand}/帧` : ''}
            {typeof sGate === 'number' && typeof sFrames === 'number'
              ? ` · 闸门 ${sGate}/${sFrames}` : ''}
          </div>
          {/* 降级原因不能只在展开视图里说：用户看图时就不会展开 */}
          {job!.candidatesNote && (
            <div className="img-warn" title={job!.candidatesNote}>
              ⚠ 候选已降级（{sCand} 张/帧）
            </div>
          )}
        </div>
      ) : (
        <div className="node-body">
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={3}
              placeholder="中英文都行，如：白猫骑士，红披风，武士刀，正面全身"
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
          {/* **中文可以用**：Qwen-Image 的文本编码器是 Qwen3-VL，不是 SD 的
              CLIP。SD 时代这里会转红警告"中文会被拆成乱 token"——那条结论
              随 SD 管线一起作废了，别再按它把中文劝退。
              两段文案都不带告警色：中文不是错误，只是"契约词由后端补"这一点
              值得提一句。 */}
          <div className="lang-hint">
            {hasCJK(p.prompt)
              ? '中文没问题（Qwen 的编码器支持中文）；全身/纯背景这类构图契约由后端自动补'
              : '中英文都行；用一句话描述主体 + 特征 + 视角（整句或逗号分隔都可以）'}
          </div>

          {(isI2I || isBg) && sourceBox}

          <div className="row">
            {!isI2I && (
              <>
                <label title={hint('宽')}>
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
                <label title={hint('高')}>
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
            <label title={hint('步数')}>
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
            <label className="grow" title={hint('种子')}>
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
          </div>

          {/* 产出风格单独占一行。**别和种子挤半行**：这一档的标签很长
              （「像素精灵·高清」「动漫立绘（不像素化）」），300px 宽的卡片里
              挤半行会直接把文字顶出 select 边框，看起来像坏掉。 */}
          <div className="row">
            <label className="grow" title={hint('风格')}>
              产出风格
              <select
                value={styleOf(p)}
                onChange={(e) => {
                  const st = GEN_STYLES.find((x) => x.v === e.target.value)
                  if (!st) return
                  // **一次 patch 改完，不能连调 setParam**（那会只有最后一个生效）。
                  // 档位本身也要记：自定义档不改任何别的字段，不记档位
                  // 受控 select 会被 React 弹回原值。
                  const patch: Partial<GenParams> = { genStyle: st.v }
                  if (st.preset !== undefined) patch.preset = st.preset
                  if (st.pixel !== undefined) patch.pixel = st.pixel
                  if (st.palette !== undefined) patch.palette = st.palette
                  // 尺寸与像素块必须配套：有效像素 = 宽 ÷ 块，1024/8=128 才对。
                  // 只改块不改尺寸的话，从「Illustrious 档」(1024) 切回
                  // 「16-bit 档」会变成 1024÷6=170 有效像素，跟档位说明对不上。
                  if (st.width !== undefined) patch.width = st.width
                  if (st.height !== undefined) patch.height = st.height
                  setParams(patch)
                  // 底模不属于节点参数（是全局状态）：档位要求 sdxl-anime 就切过去，
                  // 否则用户漏一步就白挂 LoRA——Illustrious 的 LoRA 挂 SD1.5 会直接报错
                  if (st.model && st.model !== model) onModel?.(st.model, { keepNodes: true })
                }}
              >
                {GEN_STYLES.map((x) => (
                  <option key={x.v} value={x.v}>
                    {x.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {(() => {
            const st = GEN_STYLES.find((x) => x.v === styleOf(p))
            return st ? (
              <p className="src-note" style={{ margin: '4px 0 0' }}>{st.note}</p>
            ) : null
          })()}
          {/* 不乘契约是"看起来无害"的默认值，实际等于放弃全身/居中/纯背景三条约束。
              实测不选契约时 SD1.5 大量出半身特写——用户会以为是模型不行。 */}
          {p.preset === 'none' && (
            <p className="warn" style={{ margin: '4px 0 0' }}>
              ⚠ 没选提示词契约＝没有「全身 / 居中 / 纯背景」约束，SD1.5 多半出半身特写。
              在「产出风格」里选一档像素风格即可。
            </p>
          )}

          <details className="fold">
            <summary>更多设置</summary>
            <div className="fold-body">
              <label className="block" title={hint('反向提示词')}>
                反向提示词
                <textarea
                  rows={2}
                  placeholder="留空则用内置契约"
                  value={p.negative}
                  onChange={(e) => setParam('negative', e.target.value)}
                />
              </label>
              <div className="row">
                <label title={hint('CFG')}>
                  CFG
                  <input
                    type="number"
                    step={0.5}
                    min={1}
                    max={8}
                    value={p.cfg}
                    onChange={(e) => setParam('cfg', Math.max(1, Number(e.target.value) || 1))}
                  />
                </label>
                {/* 透明背景：Qwen-Image-2.1 原生 RGBA。跟「抠图」不是一回事——
                    抠图是生成后用分割模型切一遍（发丝/细剑常被切断），
                    这个是让模型直接画 alpha，边界更干净。 */}
                <label
                  className="check"
                  title={hint('透明背景')}
                >
                  <input
                    type="checkbox"
                    checked={p.transparent && canTransparent}
                    disabled={!canTransparent}
                    onChange={(e) => setParam('transparent', e.target.checked)}
                  />
                  透明背景（原生 RGBA）
                  {!canTransparent && <span className="dim"> 仅 Qwen-Image</span>}
                </label>
                {canTransparent && p.transparent && (
                  <p className="src-note">
                    模型直接输出 alpha，不用再抠图；此时「抠图」会被跳过。
                  </p>
                )}
                {/* 像素块 / 色数**只在「自定义」风格下暴露**：
                    平时由产出风格一档决定（用户不用算「768² 配 6px = 128 有效像素」），
                    填错不报错、只是出图看着不对，所以默认藏起来。 */}
                {styleOf(p) === 'custom' && (
                  <>
                    <label className="check" title={hint('像素化')}>
                      <input
                        type="checkbox"
                        checked={p.pixel > 1}
                        onChange={(e) => setParam('pixel',
                          e.target.checked ? (p.pixel > 1 ? p.pixel : 6) : 0)}
                      />
                      像素化
                    </label>
                    <label title={hint('像素块')}>
                      像素块
                      <input
                        type="number"
                        min={0}
                        max={16}
                        value={p.pixel}
                        disabled={p.pixel <= 1}
                        // max 挡不住手输，所以 onChange 也要夹一下：曾经手输 64，
                        // 512/64 = 8px 有效像素，出图细节全被抹平
                        onChange={(e) => setParam('pixel',
                          Math.min(16, Math.max(0, Number(e.target.value) || 0)))}
                      />
                      <b>{p.pixel > 1 ? `${effPx}px` : '关'}</b>
                    </label>
                    <label title={hint('色数')}>
                      色数
                      <input
                        type="number"
                        min={0}
                        max={64}
                        value={p.palette}
                        onChange={(e) => setParam('palette',
                          Math.min(64, Math.max(0, Number(e.target.value) || 0)))}
                      />
                    </label>
                  </>
                )}
              </div>
              {/* 极端值直接说后果：这类设置不会报错，只是出图"看着不对"，
                  而用户会归因为模型不行（实测有人填了 64/4 → 8px 有效像素）。 */}
              {pxWarn && (
                <p className="warn" style={{ margin: '4px 0 0' }}>
                  ⚠ 现在是 {effPx}px 有效像素{p.palette > 0 ? ` / ${p.palette} 色` : ''}——
                  细节会被抹平。要像参考图那样的像素角色，用 768² 配像素块 6
                  （→128px）、色数 32。
                </p>
              )}
              <label className="check" title={hint('抠图')}>
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
              {/* 序列帧：候选/补提/闸门是这套产线最需要盯的三个数——
                  候选数不等于设定值就说明被显存降级了（下面的 warn 会说明原因）。 */}
              {typeof sCand === 'number' ? ` · 候选 ${sCand}/帧` : ''}
              {typeof sAttempts === 'number' && typeof sFrames === 'number'
                && typeof sCand === 'number' && sAttempts > sFrames * sCand
                ? ` · 补提 ${sAttempts - sFrames * sCand} 次` : ''}
              {typeof sGate === 'number' && typeof sFrames === 'number'
                ? ` · 闸门 ${sGate}/${sFrames}` : ''}
            </div>
          )}
          {/* 候选被降级：不说清楚的话，用户只会觉得"设了 4 张怎么一样" */}
          {job?.status === 'done' && job.candidatesNote && (
            <div className="warn">⚠ {job.candidatesNote}</div>
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
      {/* 右边缘拖宽：**所有节点统一**（含图片节点，改的是 image.w）。
          以前图片节点用的是图片右下角那个小角标，两种节点两套手感，
          而且角标紧挨着输出连接点，经常误触连线。 */}
      <div
        className="node-resize"
        title="拖动改变宽度（参数挤的时候拖宽一点）"
        onPointerDown={startWiden}
      />
      {/* 右下角：参数节点改宽×高，图片节点等比改宽。
          图片/图片视图不锁高——那是等比显示，硬拉会变形。 */}
      <div
        className="node-corner"
        title={isImg || imageMode
          ? '拖动等比缩放（高度跟着原图比例）'
          : '拖动改变大小（宽 × 高；拉矮后正文可滚动；双击恢复自适应高度）'}
        onPointerDown={startCorner}
        onDoubleClick={(e) => {
          e.stopPropagation()
          if (!isImg && !imageMode) onPatch({ h: undefined })  // 恢复自适应高度
        }}
      />
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
