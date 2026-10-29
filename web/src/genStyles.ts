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
  /** 这一档需要的 LoRA（键名 = models/lora 下的文件名 stem）。
   *  触发词不用管：后端会从 loras.json 读出来自动前置。 */
  lora?: string
  loraScale?: number
  /** 这一档建议的底模；选了档位就自动切过去，避免用户漏掉。 */
  model?: string
  note: string
}

export const GEN_STYLES: GenStyle[] = [
  {
    v: 'sprite16', label: '像素精灵（16-bit）',
    preset: 'sprite', pixel: 6, palette: 32,
    note: '最常用。768² 生成后每 6px 一格，约 128 有效像素，32 色。',
  },
  {
    v: 'spriteHi', label: '像素精灵·高清',
    preset: 'sprite', pixel: 4, palette: 48,
    note: '细节更多：每 4px 一格，约 192 有效像素；色数放宽到 48 保住层次。',
  },
  {
    v: 'spriteIll', label: '像素精灵·Illustrious（真像素）',
    preset: 'anime', pixel: 8, palette: 32, width: 1024, height: 1024,
    lora: 'Illustrious_PixelArt_Shader_M_01_epoch_4', loraScale: 1,
    model: 'sdxl-anime',
    note: '自动切 sdxl-anime 底模 + 挂 Pixelizer（触发词后端自动补）。'
      + '实测可交付 89%、硬边 0.407（产线真精灵帧 0.447，不挂 LoRA 只有 0.198）。'
      + '提示词用 danbooru tag，必须带 solo 与 full body——它只管像素化，不管构图。',
  },
  {
    v: 'pixel8', label: '像素·8-bit 粗颗粒',
    preset: 'pixel', pixel: 8, palette: 16,
    note: '颗粒更粗、色数更少（16 色），复古 8-bit 味。',
  },
  {
    v: 'anime', label: '动漫立绘（不像素化）',
    preset: 'anime', pixel: 0, palette: 0,
    note: '纯插画，保画质。建议底模选 sdxl-anime，尺寸 1024²。',
  },
  {
    v: 'realistic', label: '写实插画（不像素化）',
    preset: 'portrait', pixel: 0, palette: 0,
    note: '厚涂/写实风，不像素化。',
  },
  {
    v: 'ink', label: '水墨国风（不像素化）',
    preset: 'wuxia', pixel: 0, palette: 0,
    note: '水墨线稿风，建议配武侠 LoRA（wuxia_v2）。',
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
  preset: Preset, pixel: number, palette: number, lora: string = '',
): string {
  const hit = GEN_STYLES.find(
    (x) => x.v !== 'custom' && x.preset === preset
      && x.pixel === pixel && x.palette === palette
      // LoRA 也是档位的组成部分：档位挂着 Pixelizer，用户手动换掉就该回退
      // 成「自定义参数」，不能还显示「像素精灵·Illustrious」
      && (x.lora ?? '') === lora,
  )
  return hit ? hit.v : 'custom'
}
