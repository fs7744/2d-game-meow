import { useEffect, useRef, useState } from 'react'

type Props = {
  /** 源图 URL（局部重绘的输入） */
  srcUrl?: string
  /** 蒙版 URL（白 + 透明，alpha 决定选区） */
  maskUrl?: string
  /** 蒙版先外扩多少 px（原图像素） */
  expand: number
  /** 贴回时羽化多少 px（原图像素） */
  feather: number
  /** 显示宽度上限（px） */
  width?: number
}

/** 把 url 读成可画的 Image；失败返回 null（节点上就只是不显示预览） */
function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => resolve(null)
    img.src = url
  })
}

/**
 * 方形膨胀：每边外扩 r px（切比雪夫距离）。
 *
 * **必须和后端一致**：`workflows/repaint.dilate` 用的是 PIL `MaxFilter`，核是方的，
 * px 表示"每边外扩多少"。前端如果做成圆形核，预览会比实际窄一圈角。
 *
 * 可分离实现（先横向再纵向），复杂度 O(w·h)，与 r 无关——拖动滑块时不会卡。
 * 一开始我用「blur + 阈值化」近似，实测那根本不膨胀：blur 会把边缘灰度拉低，
 * 阈值化后反而**缩小**（r=16 时边界退了 3px）。
 */
function dilateSquare(src: Uint8Array, w: number, h: number, r: number): Uint8Array {
  if (r <= 0) return src
  const tmp = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    const row = y * w
    let count = 0
    for (let k = 0; k < Math.min(r, w); k++) if (src[row + k]) count++
    for (let x = 0; x < w; x++) {
      const add = x + r
      if (add < w && src[row + add]) count++
      const del = x - r - 1
      if (del >= 0 && src[row + del]) count--
      if (count > 0) tmp[row + x] = 1
    }
  }
  const out = new Uint8Array(w * h)
  for (let x = 0; x < w; x++) {
    let count = 0
    for (let k = 0; k < Math.min(r, h); k++) if (tmp[k * w + x]) count++
    for (let y = 0; y < h; y++) {
      const add = y + r
      if (add < h && tmp[add * w + x]) count++
      const del = y - r - 1
      if (del >= 0 && tmp[del * w + x]) count--
      if (count > 0) out[y * w + x] = 1
    }
  }
  return out
}

/** 把 0/1 掩码涂成纯色，返回一张可直接 drawImage 的图 */
function tintMask(mask: Uint8Array, w: number, h: number, color: string,
                  only?: Uint8Array): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d')
  if (!ctx) return c
  const id = ctx.createImageData(w, h)
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    if (only && only[i]) continue                 // 用来扣掉"原选区"得到外扩环
    const o = i * 4
    id.data[o] = parseInt(color.slice(1, 3), 16)
    id.data[o + 1] = parseInt(color.slice(3, 5), 16)
    id.data[o + 2] = parseInt(color.slice(5, 7), 16)
    id.data[o + 3] = 255
  }
  ctx.putImageData(id, 0, 0)
  return c
}

/**
 * 局部重绘的**几何预览**：源图 + 选区 + 外扩环 + 羽化边。
 *
 * 为什么需要它：外扩和羽化都是"像素级"参数，改完只能点一次生成才知道效果，
 * 一轮十几秒。而这两个参数的后果很具体——外扩太大会吃掉邻格（像素素材直接废），
 * 羽化太大接缝会糊。这些都能在本地算出来，不需要模型。
 *
 * **它不预测画面内容**：重绘幅度（strength）改变的是"模型画成什么样"，
 * 那部分必须生成才知道。所以这里只画几何范围，并在下面标清楚。
 */
export default function MaskPreview({ srcUrl, maskUrl, expand, feather, width = 200 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [note, setNote] = useState('')

  useEffect(() => {
    let cancelled = false
    if (!srcUrl || !maskUrl) {
      setNote('')
      return
    }
    void (async () => {
      const [src, msk] = await Promise.all([loadImage(srcUrl), loadImage(maskUrl)])
      if (cancelled || !src || !msk || !ref.current) {
        if (!cancelled) setNote(src ? '蒙版读不出来' : '')
        return
      }
      const cv = ref.current
      const scale = Math.min(1, width / (src.naturalWidth || width))
      const w = Math.max(16, Math.round((src.naturalWidth || width) * scale))
      const h = Math.max(16, Math.round((src.naturalHeight || width) * scale))
      cv.width = w
      cv.height = h
      const ctx = cv.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, w, h)
      ctx.drawImage(src, 0, 0, w, h)

      // 蒙版缩到同一尺寸，取 alpha 当选区
      const pick = document.createElement('canvas')
      pick.width = w
      pick.height = h
      const pc = pick.getContext('2d')
      if (!pc) return
      pc.drawImage(msk, 0, 0, w, h)
      const id = pc.getImageData(0, 0, w, h)
      // 两种历史格式要分开处理（与后端 repaint.load_mask 同一套判断）：
      //   有透明像素 → 这就是"白+透明"蒙版，按 alpha 判
      //   整张不透明 → 是老的灰阶蒙版，按亮度判
      // 一起混着用（alpha<128 就跳过）会把灰阶图的暗区全判成非选区。
      let minA = 255
      for (let i = 3; i < id.data.length; i += 4) {
        if (id.data[i] < minA) minA = id.data[i]
        if (minA === 0) break
      }
      const useAlpha = minA < 128
      for (let i = 0; i < id.data.length; i += 4) {
        const lum = 0.299 * id.data[i] + 0.587 * id.data[i + 1] + 0.114 * id.data[i + 2]
        const on = useAlpha ? id.data[i + 3] >= 128 : lum > 127
        id.data[i] = id.data[i + 1] = id.data[i + 2] = 255
        id.data[i + 3] = on ? 255 : 0
      }
      pc.putImageData(id, 0, 0)

      const any = id.data.some((v, i) => i % 4 === 3 && v > 0)
      if (!any) {
        setNote('还没涂选区')
        return
      }

      // 原图像素 → 显示像素的比例：参数（px）要按它换算，否则 4px 的外扩
      // 在缩略图上就看不见，64px 的又糊满整张
      const ds = scale
      // 掩码 → 0/1 数组
      const rs = Math.max(1, Math.round(expand * ds))
      const fs = Math.max(1, Math.round(feather * ds))
      const flat = new Uint8Array(w * h)
      for (let k = 0; k < flat.length; k++) {
        flat[k] = id.data[k * 4 + 3] >= 128 ? 1 : 0
      }

      // 原选区：红
      ctx.globalAlpha = 0.52
      ctx.drawImage(tintMask(flat, w, h, '#ff5a5a'), 0, 0)
      ctx.globalAlpha = 1

      // 外扩环（橙）：膨胀后的形状扣掉原选区，只剩"多出来的那一圈"
      if (expand > 0) {
        const grown = dilateSquare(flat, w, h, rs)
        ctx.globalAlpha = 0.7
        ctx.drawImage(tintMask(grown, w, h, '#ffb020', flat), 0, 0)
        ctx.globalAlpha = 1
      }

      // 羽化带（黄）：贴回时的过渡区，画在外扩环之外
      if (feather > 0) {
        const outer = dilateSquare(flat, w, h, rs + fs)
        const inner = rs > 0 ? dilateSquare(flat, w, h, rs) : flat
        ctx.globalAlpha = 0.45
        ctx.drawImage(tintMask(outer, w, h, '#ffcf7a', inner), 0, 0)
        ctx.globalAlpha = 1
      }

      const px = (n: number) => `${Math.round(n)}px`
      setNote(`红=选区　橙=外扩 ${px(expand)}　黄=羽化 ${px(feather)}（贴回过渡带）`)
    })()

    return () => {
      cancelled = true
    }
  }, [srcUrl, maskUrl, expand, feather, width])

  if (!srcUrl) return null

  return (
    <div className="mask-preview">
      <canvas ref={ref} />
      <div className="mp-note">{note || '（选区和源图都齐了才有预览）'}</div>
      <div className="mp-note mp-dim">只显示几何范围；画成什么样要生成才知道</div>
    </div>
  )
}
