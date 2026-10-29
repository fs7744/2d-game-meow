import { useRef, useState } from 'react'
import { hasCJK } from '../store/prompts'
import MaskEditor from '../components/MaskEditor'
import MaskPreview from '../components/MaskPreview'
import SkeletonEditor, { defaultSkeleton } from '../components/SkeletonEditor'
import type { ModelCaps, PoseDir } from '../api'
import { fileNameOf } from '../fileName'
import type { GenNode, GenParams, LoraInfo } from '../types'
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
    rec: '768。SD1.5 在 512 上必糊；SDXL 用 1024 反而更差（角色占比大、可交付率 92%→75%）',
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
    rec: 'SD1.5 用 30、SDXL 用 24。用了蒸馏（lcm）可降到 4~8',
  },
  CFG: {
    what: '提示词服从度（classifier-free guidance）',
    effect: '低=自由但容易跑题；高=听话但僵硬、过饱和、易出伪影',
    rec: '6.5~7.5。蒸馏模型要 1.0~2.0（高了必糊）',
  },
  种子: {
    what: '随机种子。-1 表示每次随机',
    effect: '固定种子才能复现同一张图；换种子是产线里最便宜的"重抽"手段',
    rec: '调参时固定；批量出候选时用 seed、seed+1000、seed+2000…',
  },
  风格: {
    what: '提示词契约（predefined 的构图/背景约定）',
    effect: '自动补「扁平背景、硬描边、全身」这类产线约束，并截断到 token 预算内',
    rec: '精灵用 sprite、立绘用 portrait、武侠 LoRA 配 wuxia、像素用 pixel',
  },
  反向提示词: {
    what: '不希望出现的东西',
    effect: 'SD1.5 上作用很大；写具体名词比堆形容词有效（如 character sheet 压多主体）',
    rec: '留空用内置默认即可。多主体压不住时加 character sheet, multiple characters',
  },
  LoRA: {
    what: '附加的风格/角色权重（LoRA）',
    effect: '会大幅改变画风；必须与底模架构匹配（SD1.5 的 LoRA 不能用于 SDXL）',
    rec: '像素风挂 px_illustrious(0.75)；武侠挂 wuxia_v2。不挂就是纯底模',
  },
  'LoRA 强度': {
    what: 'LoRA 的作用强度',
    effect: '过高会过拟合：构图僵、出伪影、多主体；过低则看不出风格',
    rec: '0.6~0.85。像素 LoRA 实测 0.75 最稳，0.5 就掉风格了',
  },
  重绘: {
    what: '重绘幅度（denoise strength）',
    effect: '越低越接近原图；越高越自由、越可能丢原有结构',
    rec: '保持原样微调 0.3；改内容 0.55~0.7；几乎重画 0.85+',
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
  动作: {
    what: 'ControlNet 条件强度（贴不贴参考动作）',
    effect: '高=姿态准但僵硬、易崩；低=自由但动作走形',
    rec: '0.8~1.2。参考图本身不规范时降到 0.7',
  },
  身份: {
    what: 'IP-Adapter 强度（像不像角色图）',
    effect: '高=更像角色，但会压制姿态、画面变糊；低=姿态准但换了个角色',
    rec: '0.7~0.9。动作复杂时降到 0.65',
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
  姿态: {
    what: 'ControlNet 姿态强度（同「动作」）',
    effect: '高=严格贴合参考骨架；低=允许自由发挥',
    rec: '1.0；参考图比例不对时降到 0.8',
  },
  候选: {
    what: '每帧生成几张再挑一张',
    effect: '同 seed 也会漂，多张按「关节数+前景占比+贴边+漂移」挑最好的。越多越稳、越慢',
    rec: '2~3 张。上限由显存算出（SDXL@768 是 3、@1024 只有 1）',
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
    what: '生成引擎',
    effect: 'AnimateDiff 一次出整段、靠时序注意力锁一致性；逐帧多候选更省显存、更容易挑到好帧',
    rec: '帧数多且要连贯用 AnimateDiff；显存紧或帧数少用逐帧',
  },
  显存: {
    what: '显存档位（不够时降分辨率，帧数不砍）',
    effect: '帧数是你的动画内容不能动，所以只能牺牲分辨率',
    rec: 'auto。OOM 频繁时手动选 low（384²）',
  },
  边长: {
    what: '目标像素画的边长（原生像素数）',
    effect: '决定最终像素密度；0=自动检测输入图的网格',
    rec: '0（自动）。已知原图是 128px 像素画时直接填 128',
  },
  精修: {
    what: 'SDXL Refiner：底模出图后再走一遍精修（接在最后一段去噪上）',
    effect: '它是个**照片化**精修器：喂插画会把画面往写实拉（实测把动漫剑客'
      + '变成"墙上挂的一幅画"），喂写实图才会提升发丝/布料/金属的细节',
    rec: '**写实图**才值得开；像素/动漫素材别开——后处理本来就会抹平细节，'
      + 'refiner 只会把风格带偏',
  },
  精修步数: {
    what: 'refiner 自己的去噪步数',
    effect: '越多越干净也越慢；refiner 只处理最后一段，步数需求比底模低',
    rec: '20~30',
  },
  精修强度: {
    what: 'refiner 的 CFG（提示词服从度）',
    effect: '高=更贴合提示词但可能过饱和；低=更自然',
    rec: '5~7，与底模 CFG 接近',
  },
  精修起点: {
    what: 'refiner 接手收尾的那一段，起点越高改得越少',
    effect: '0.8 = 只修最后 20% 的幅度（不毁构图）；'
      + '0.6 会明显改画风，但也更容易把脸和手改坏。'
      + '注意实际执行步数 = 精修步数 ×(1−起点)，20 步 × 0.8 只剩 4 步',
    rec: '0.8。refiner 只微调质感，**不是画质增强**——'
      + '实测清晰度不升反略降（VAE 编解码有损），别指望它救糊图',
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
    | 'charSource' | 'skeleton'>
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
  loras: LoraInfo[]
  /** 这个节点的底模架构（sd15 / sdxl / qwen21）：LoRA 按它过滤，不能按全局默认 */
  loraBase?: string
  model: string
  /** 切底模：产出风格档位可能要求特定底模（如「像素精灵·Illustrious」要 sdxl-anime）。
   *  keepNodes=true 表示参数由档位负责，别套推荐值（否则会覆盖刚 patch 的 LoRA/尺寸）。 */
  onModel?: (m: string, opts?: { keepNodes?: boolean }) => void
  /** 当前底模在 W4 上能批量几个候选（后端按总显存算）；缺省不限制 */
  maxCandidates?: number
  /** 可当 W4 输入的骨架帧目录（后端扫出来的） */
  poseDirs?: PoseDir[]
  /** 本机是否装了 SDXL refiner 权重（没有就别让人勾） */
  hasRefiner?: boolean
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
 * 第 3 条是为了**不骗人**：切底模（动漫底模会改 preset/pixel/palette）或挂
 * LoRA（LoRA 自带契约）之后参数被外部改掉，`genStyle` 却停在旧值——只信它
 * 就会显示「像素精灵」而参数实际是动漫立绘。
 */
/** 取文件名 stem：lora 字段可能是名字，也可能是绝对路径。 */
function loraStem(v: string): string {
  return (v || '').split(/[\\/]/).pop()!.replace(/\.safetensors$/i, '')
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
}

/** 底模下拉里的短名：目录名（qwen-image-2.1）太长，撑爆 260px 的卡片 */
function shortModel(m: string): string {
  const map: Record<string, string> = {
    sd15: 'SD1.5',
    sdxl: 'SDXL',
    'sdxl-anime': 'SDXL 动漫',
    'sd15-inpaint': 'SD1.5 重绘',
    'qwen-image-2.1': 'Qwen-Image 2.1',
  }
  return map[m] || m
}

function styleOf(p: GenParams): string {
  if (p.genStyle === 'custom') return 'custom'
  const byParams = styleFromParams(p.preset, p.pixel, p.palette, loraStem(p.lora))
  if (p.genStyle && byParams === p.genStyle) return p.genStyle
  return byParams
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
  loras,
  loraBase,
  model,
  onModel,
  maxCandidates,
  poseDirs,
  hasRefiner,
  canTransparent,
  nodesModels,
  modelCaps,
  defaultModel,
  onSpawnImg2Img,
  onExportSkeleton,
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
  /**
   * LoRA 是**按节点底模**过滤的，不是按全局默认。
   *
   * 之前在 App 里 fetchLoras(arch[全局model]) 一次拉完，节点改成节点级底模后
   * 就对不上了：节点切到 SDXL，下拉里仍全是 SD1.5 的 LoRA（装上去是废图），
   * 而档位要挂的 SDXL LoRA 又根本不在列表里，value 落空。
   * 现在 App 拉全量，每个节点按自己的 arch 过滤。
   */
  const visibleLoras = loraBase
    ? loras.filter((l) => (l.base || 'sd15') === loraBase)
    : loras
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
  /**
   * 这个节点的底模是不是 Qwen。
   *
   * 判据用 caps.instruction 而不是硬编码底模名：instruction（图生图走「指令改图」）
   * 目前只有 Qwen 有，但底模名是可以自己放的目录，硬编码名字迟早对不上。
   */
  const isQwenModel = !!modelCaps?.[model]?.instruction
  /** 本节点跑的工作流名（和 /models 的 caps.workflows 同一套词） */
  const wf = WORKFLOW_OF[node.kind] || ''
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

  /**
   * 选 LoRA：同时把「推荐强度」和「匹配的提示词契约」一起带过来。
   * 不这么做的话，挂着武侠 LoRA 却仍在用像素契约，两边打架、两头不到位。
   */
  const extras = p.extraLoras || []
  const setExtras = (next: { name: string; scale: number }[]) =>
    onPatch({ params: { ...p, extraLoras: next } })

  /**
   * 选 LoRA：同时把「推荐强度」和「匹配的提示词契约」一起带过来。
   * 不这么做的话，挂着武侠 LoRA 却仍在用像素契约，两边打架、两头不到位。
   *
   * `slot` 指定是主槽（-1）还是叠加的第几个。副槽只带强度，不改
   * preset / 抠图 / clip skip —— 那几项是"这个 LoRA 的训练约定"，
   * 让第二个 LoRA 覆盖第一个的约定只会更乱。
   */
  const chooseLora = (name: string, slot = -1) => {
    const info = loras.find((l) => l.name === name)
    if (slot >= 0) {
      const next = extras.map((x, i) => (i === slot
        ? { name, scale: info?.scale ?? x.scale } : x))
      setExtras(next)
      return
    }
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

  /** 叠加一个 LoRA。上限 4 个：再多效果互相抵消，而且显存里堆叠的副本也变多。 */
  const addExtra = () => {
    if (extras.length >= 4) return
    setExtras([...extras, { name: '', scale: 0.8 }])
  }
  const removeExtra = (i: number) => setExtras(extras.filter((_, k) => k !== i))
  const setExtraScale = (i: number, v: number) =>
    setExtras(extras.map((x, k) => (k === i ? { ...x, scale: v } : x)))

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
          <label className="strength" title={hint('重绘')}>
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
          {isPoseEdit ? '骨骼编辑' : isImg ? '图片' : isBg ? '抠图' : isPixelify ? '像素转换'
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
            <label className="strength" title={hint('动作')}>
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
                value={p.ssEngineMode}
                onChange={(e) => setParam('ssEngineMode', e.target.value as typeof p.ssEngineMode)}
              >
                <option value="auto">自动（优先 AnimateDiff）</option>
                <option value="animatediff">AnimateDiff（帧间更稳）</option>
                <option value="frame">逐帧多候选（更省显存）</option>
              </select>
            </label>
          )}
          {p.ssMode === 'generate' && p.ssEngineMode !== 'frame' && (
            <label className="bg-mode" title={hint('显存')}>
              显存
              <select
                value={p.ssVramProfile}
                onChange={(e) => setParam('ssVramProfile', e.target.value as typeof p.ssVramProfile)}
              >
                <option value="auto">自动（不够就降分辨率）</option>
                <option value="low">省显存（384²）</option>
                <option value="medium">平衡（512²）</option>
                <option value="high">高质量（512²，最多 32 帧）</option>
              </select>
            </label>
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
                <label className="strength" title={hint('动作')}>
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
                <label className="strength" title={hint('身份')}>
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
            {p.ssMode === 'generate' && p.ssEngineMode !== 'animatediff' && (
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
            {p.ssMode === 'generate' && p.ssEngineMode !== 'frame' && (
              <div className="src-note">
                AnimateDiff 一次生成整段，靠时序注意力锁住帧间一致性——
                实测 8 帧比逐帧方案快 2.6× 且漂移少 36%。
                显存已优化（VAE 逐帧解码 + 参考图编码器用完即退）：
                8 帧 5.8GB、16 帧 7.1GB，比基线省 41%。
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
                {charThumb ? '角色外观由这张图决定（IP-Adapter）' : '缺角色图'}
              </div>
            </div>
          </div>
          <div className="prompt-wrap">
            <textarea
              className="prompt"
              rows={2}
              placeholder="可选：补充描述（外观已由角色图决定）"
              value={p.prompt}
              onChange={(e) => setParam('prompt', e.target.value)}
            />
          </div>
          <details className="adv">
            <summary>更多设置</summary>
            {/* 「姿态」「身份」两根滑块是 SD 那条路（ControlNet 强度 / IP-Adapter
                强度）专属的。Qwen 走的是**双图条件**（角色图 + 骨架图进同一个
                vision context），模型自己分配两张图各管什么，没有强度旋钮——
                留着只会让人以为调了有用。 */}
            {isQwenModel ? (
              <div className="src-note">
                Qwen 底模走「双图条件」：角色图 + 骨架图一起进模型，姿势与身份由它
                自己分配，没有姿态/身份强度可调（SD 那两个滑块已隐藏）。
              </div>
            ) : (
              <>
                <label className="strength" title={hint('姿态')}>
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
                <label className="strength" title={hint('身份')}>
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
              </>
            )}
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
          <label className="strength" title={hint('重绘')}>
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
          <label className="strength" title={hint('重绘')}>
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
                  // LoRA 是「档位的装备」：
                  //  · 目标档位自带 LoRA → 挂上
                  //  · 目标档位不带，而当前这个 LoRA 是**上一个档位挂上的** → 摘掉
                  //    （不摘的话参数永远比不中任何档位，下拉立刻弹回「自定义」，
                  //     而且之后每一档都弹回去——等于进了 Illustrious 档就出不来）
                  //  · 用户自己手动选的 LoRA（不属于任何档位）→ 保留，别抢
                  const cur = GEN_STYLES.find((x) => x.v === styleOf(p))
                  const owned = cur?.lora && cur.lora === loraStem(p.lora)
                  if (st.lora !== undefined) {
                    patch.lora = st.lora
                    patch.loraScale = st.loraScale ?? 0.8
                  } else if (owned) {
                    patch.lora = ''
                  }
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
              <label className="block" title={hint('LoRA')}>
                LoRA
                <select value={p.lora} onChange={(e) => chooseLora(e.target.value)}>
                  <option value="">不挂（纯底模）</option>
                  {visibleLoras.map((l) => (
                    <option key={l.name} value={l.name}>
                      {l.label}
                    </option>
                  ))}
                </select>
              </label>
              {/* LoRA 列表按**这个节点自己的底模架构**过滤，所以 Qwen 底下
                  本来就只剩「不挂」。但空列表不解释的话用户会以为扫坏了。 */}
              {!visibleLoras.length && (
                <p className="src-note">
                  当前底模下没有可用 LoRA：Qwen-Image 不吃 SD 的 LoRA（不同的主干与
                  文本编码器），能挂的只有它自己的微调。
                </p>
              )}
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
              {/* 叠加多个 LoRA：效果线性叠加（A1111 的 <lora:a:0.8> <lora:b:0.5>）。
                  常见组合是「风格 + 角色」——单个 LoRA 很难同时给两样。 */}
              <div className="lora-stack">
                <div className="stack-head">
                  <span>叠加 LoRA</span>
                  <button
                    className="mini"
                    onClick={addExtra}
                    disabled={!visibleLoras.length || extras.length >= 4}
                    title="再加一个 LoRA（与主 LoRA 线性叠加）"
                  >
                    + 叠加
                  </button>
                </div>
                {extras.map((it, i) => (
                  <div className="stack-row" key={i}>
                    <select
                      value={it.name}
                      onChange={(e) => chooseLora(e.target.value, i)}
                      title="选择要叠加的 LoRA（必须与主 LoRA 同架构）"
                    >
                      <option value="">（选一个）</option>
                      {visibleLoras.map((l) => (
                        <option key={l.name} value={l.name}>{l.label}</option>
                      ))}
                    </select>
                    <input
                      type="number"
                      step={0.05}
                      min={0}
                      max={1.5}
                      value={it.scale}
                      disabled={!it.name}
                      onChange={(e) => setExtraScale(i, Number(e.target.value))}
                      title="这个 LoRA 的强度"
                    />
                    <button
                      className="mini"
                      onClick={() => removeExtra(i)}
                      title="移除这一层"
                    >
                      ×
                    </button>
                  </div>
                ))}
                {!!extras.length && (
                  <p className="stack-note">
                    强度各自独立；总强度过高会过拟合（构图僵、出伪影），
                    叠两层时建议都在 0.5~0.8。
                  </p>
                )}
              </div>
              <div className="row">
                <label title={hint('LoRA 强度')}>
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
                <label title={hint('CFG')}>
                  CFG
                  <input
                    type="number"
                    step={0.5}
                    value={p.cfg}
                    onChange={(e) => setParam('cfg', Number(e.target.value) || 7.5)}
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
                <label className="check" title={hint('精修')}>
                  <input
                    type="checkbox"
                    checked={p.refiner}
                    disabled={model !== 'sdxl' || !hasRefiner}
                    onChange={(e) => setParam('refiner', e.target.checked)}
                  />
                  应用 Refiner 精修
                </label>
                {p.refiner && (
                  <div className="lora-stack">
                    <label className="strength" title={hint('精修强度')}>
                      <span>精修强度</span>
                      <input
                        type="range"
                        min={1}
                        max={12}
                        step={0.5}
                        value={p.refinerCfg}
                        onChange={(e) => setParam('refinerCfg', Number(e.target.value))}
                      />
                      <b>{p.refinerCfg}</b>
                    </label>
                    <label className="strength" title={hint('精修步数')}>
                      <span>精修步数</span>
                      <input
                        type="number"
                        min={5}
                        max={60}
                        value={p.refinerSteps}
                        onChange={(e) => setParam('refinerSteps',
                          Math.min(60, Math.max(5, Number(e.target.value) || 20)))}
                      />
                      <b>步</b>
                    </label>
                    <label className="strength" title={hint('精修起点')}>
                      <span>精修起点</span>
                      <input
                        type="range"
                        min={0.4}
                        max={0.95}
                        step={0.05}
                        value={p.refinerEnd}
                        onChange={(e) => setParam('refinerEnd', Number(e.target.value))}
                      />
                      <b>
                        {p.refinerEnd.toFixed(2)}
                        <em className="sub-val">
                          ≈{Math.max(1, Math.round(p.refinerSteps * (1 - p.refinerEnd)))} 步
                        </em>
                      </b>
                    </label>
                  </div>
                )}
                {model !== 'sdxl' && (
                  <p className="src-note">Refiner 只支持 SDXL；SD1.5 底模下不可用。</p>
                )}
                {model === 'sdxl' && !hasRefiner && (
                  <p className="src-note">
                    本机没有 refiner 权重（models/checkpoints/sdxl-refiner），勾了也不会生效。
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
                <label title="1=最后一层（默认）；2=跳过最后一层，多数动漫/像素 LoRA 的训练设置">
                  跳过层
                  <input
                    type="number"
                    min={1}
                    max={4}
                    value={p.clipSkip}
                    disabled={model === 'sdxl'}
                    // 灰掉要给原因：SDXL 有两个文本编码器 + pooled embeds，
                    // clip skip 不是"改个层号"就能对的事，后端会直接忽略
                    title={model === 'sdxl'
                      ? 'SDXL 不支持 clip skip（两个文本编码器要一起改），切回 SD1.5 才生效'
                      : '1 = 最后一层（默认）；2 = 跳过最后一层，多数动漫/像素 LoRA 的训练设置'}
                    onChange={(e) => setParam('clipSkip', Number(e.target.value) || 1)}
                  />
                </label>
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
