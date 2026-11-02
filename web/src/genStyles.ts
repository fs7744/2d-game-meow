/**
 * 「产出风格」档位表：**一档决定「契约 + 像素块 + 色数」三件事**。
 *
 * 为什么不让用户直接填像素块和色数：那两个参数强耦合——
 * 「768² 配 6px 块 = 128 有效像素」这种换算要用户在脑子里做，而填错
 * 不会报错、只是出图「看着不对」（实测有人填 64 → 8px 有效像素，细节全没）。
 * 产线上真正要选的是「要什么样的成品」，不是「块多大」。
 *
 * 放在独立模块而不是节点组件里：换底模套推荐参数时 App 也要用
 * `styleFromParams` 反推档位——两边必须用同一张表，否则下拉回显会和
 * 实际参数对不上。
 */
import type { Preset } from './types'

export type GenStyle = {
  v: string
  label: string
  preset?: Preset
  pixel?: number
  palette?: number
  /** 尺寸也要跟着档位定：像素块是相对尺寸算的（有效像素 = 宽 ÷ 块），
   *  1024 配 8 = 128 有效像素，换 768 就变成 96，档位必须自洽。 */
  width?: number
  height?: number
  /** 这一档建议的底模；选了档位就自动切过去，避免用户漏掉。 */
  model?: string
  note: string
}

export const GEN_STYLES: GenStyle[] = [
  {
    v: 'sprite16', label: '像素精灵（16-bit）',
    preset: 'sprite', pixel: 6, palette: 32, width: 768, height: 768,
    note: '最常用。768² 生成后每 6px 一格，约 128 有效像素，32 色。',
  },
  {
    v: 'spriteHi', label: '像素精灵·高清',
    preset: 'sprite', pixel: 4, palette: 48, width: 768, height: 768,
    note: '细节更多：每 4px 一格，约 192 有效像素；色数放宽到 48 保住层次。',
  },
  {
    v: 'spriteIll', label: '像素精灵·Illustrious（真像素）',
    preset: 'anime', pixel: 8, palette: 32, width: 1024, height: 1024,
    note: '像素块 8（1024÷8 = 128 有效像素）+ anime 契约。'
      + 'SD 时代这一档还要挂 Pixelizer LoRA 并切 sdxl-anime 底模；'
      + 'Qwen 之后 LoRA 与底模切换都已经删掉，只剩参数档位。',
  },
  {
    v: 'pixel8', label: '像素·8-bit 粗颗粒',
    preset: 'pixel', pixel: 8, palette: 16, width: 768, height: 768,
    note: '颗粒更粗、色数更少（16 色），复古 8-bit 味。',
  },
  {
    v: 'anime', label: '动漫立绘（1024²，不像素化）',
    preset: 'anime', pixel: 0, palette: 0, width: 1024, height: 1024,
    note: '纯插画、保画质。**立绘档位用 1024²**：精灵档是 768²，因为成品要缩到 '
      + '32~128px；立绘不缩图，1024² 的笔触细节才留得住。代价是每张约 45s（768² 约 25s）。',
  },
  {
    v: 'realistic', label: '写实插画（1024²，不像素化）',
    preset: 'portrait', pixel: 0, palette: 0, width: 1024, height: 1024,
    note: '厚涂/写实风，不像素化。跟立绘一样用 1024²。',
  },
  {
    v: 'ink', label: '水墨国风（1024²，不像素化）',
    preset: 'wuxia', pixel: 0, palette: 0, width: 1024, height: 1024,
    note: '水墨线稿风。提示词契约里有压印章与水墨背景的负向词'
      + '（SD 时代还要配 wuxia_v2 LoRA，Qwen 之后 LoRA 已删）。',
  },
  {
    v: 'custom', label: '自定义参数',
    pixel: undefined, palette: undefined,
    note: '自己填像素块与色数（在「更多设置」里）。',
  },
]

/**
 * 按「契约 + 像素块 + 色数」反推档位，不依赖记录的 `genStyle`。
 *
 * 换底模、切 LoRA 时参数是被外部改掉的，`genStyle` 不会跟着动——那种情况下
 * 下拉必须显示真实档位（或者老实说「自定义」），不能拿着过期的 genStyle 骗人。
 * App 套推荐参数时也调它，保证「参数」与「下拉回显」始终一致。
 *
 * `custom` 档没有特征参数，永远推不出来——那是它应有的行为，由调用方
 * 用显式的档位字段兜住。
 */
export function styleFromParams(
  preset: Preset, pixel: number, palette: number,
): string {
  const hit = GEN_STYLES.find(
    (x) => x.v !== 'custom' && x.preset === preset
      && x.pixel === pixel && x.palette === palette,
  )
  return hit ? hit.v : 'custom'
}

/* ==========================================================================
   尺寸预设
   ========================================================================== */

/**
 * 常见出图尺寸。
 *
 * 为什么要有：之前只有两个 number 输入框，用户得自己记住「768 是精灵甜点」
 * 「竖版立绘要 832×1216」这些换算——填错不报错，只是出图糊或比例怪。
 * 业界（SD WebUI / ComfyUI / Civitai）都给一串常用档位，选一下就行。
 *
 * 约束：**总像素不超过 1024²**（≈1.05MP）。这是实测过的成本上限——再大
 * 单帧要 60s+ 且容易 OOM。所以 2:3 与 3:2 用 832×1216 / 1216×832，
 * 而不是 SDXL 那套 1216×1664（后者会到 2MP）。宽高都是 8 的倍数
 * （Qwen 的 latent 步长）。
 */
export type SizePreset = {
  key: string
  w: number
  h: number
  label: string
  note: string
}

export const SIZE_PRESETS: SizePreset[] = [
  {
    key: '1024x1024', w: 1024, h: 1024,
    label: '1024×1024 · 方形', note: '细节最多，立绘/头像（约 45s）',
  },
  {
    key: '768x768', w: 768, h: 768,
    label: '768×768 · 方形', note: '精灵甜点档，也是默认值（约 25s）',
  },
  {
    key: '512x512', w: 512, h: 512,
    label: '512×512 · 小图', note: '试提示词/草稿，最快',
  },
  {
    key: '768x1024', w: 768, h: 1024,
    label: '768×1024 · 竖 3:4', note: '半身、立绘',
  },
  {
    key: '832x1216', w: 832, h: 1216,
    label: '832×1216 · 竖 2:3', note: '全身立绘（SDXL 常用比例）',
  },
  {
    key: '768x1344', w: 768, h: 1344,
    label: '768×1344 · 竖 9:16', note: '手机壁纸比例',
  },
  {
    key: '1024x768', w: 1024, h: 768,
    label: '1024×768 · 横 4:3', note: '横构图、双人',
  },
  {
    key: '1216x832', w: 1216, h: 832,
    label: '1216×832 · 横 3:2', note: '场景/横版立绘',
  },
]

/** 反查：当前宽高命中哪个预设；没命中就是自定义。 */
export function sizePresetKey(w: number, h: number): string {
  const hit = SIZE_PRESETS.find((p) => p.w === w && p.h === h)
  return hit ? hit.key : 'custom'
}
