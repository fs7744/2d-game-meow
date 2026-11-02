/**
 * 抽帧之后的「挑帧」两把筛子（照 pixelforge 的 `lib/frameSelect.ts` 那套语义）。
 *
 * **为什么要有这一步**：等分抽 24 帧，其中一半可能是"人偶几乎没动"的重复姿势。
 * 让用户从 24 张缩略图里手点掉 12 张不现实，但**自动替他决定**又不行 ——
 * 他可能就是想全要。所以做成「筛子只改**显示**，勾选是显式状态」：
 * 拖滑杆能立刻看到候选怎么变，但已经勾好的帧不会被悄悄换掉，点「应用筛选」
 * 才是采纳。
 *
 * 指纹（16×16 **剪影**，base64，256 字节）由 iframe 在抽帧时算好带过来 ——
 * 宿主里再解一遍 PNG 纯属浪费（几十张 256² 的 base64 解码会卡住主线程）。
 *
 * 判据和 pixelforge 的 `diff_percent` 思路一致（"差异百分比，0 = 一样"），
 * 但**分母换成了两帧墨迹的并集**：那边的视频帧是满幅的，我们的人偶图是
 * 白底细人形，按 256 格平均会被白底稀释到 0.2%（实测），滑杆就废了。
 */

/** 从 iframe 抽回来的一帧 */
export type SampledFrame = {
  /** 在这条动作里的时间（秒） */
  t: number
  /** 官方 pose JSON（存姿势集时原样落盘） */
  pose: unknown
  /** 人偶图 dataURL */
  png: string
  /** 16×16 灰度指纹（base64，256 字节）；空串 = 算指纹失败，按"完全不同"处理 */
  hash: string
}

const cache = new Map<string, Uint8Array>()

function decode(hash: string): Uint8Array | null {
  if (!hash) return null
  const hit = cache.get(hash)
  if (hit) return hit
  try {
    const bin = atob(hash)
    const out = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
    // 用户会反复重抽，指纹会一直涨；超了就整个丢掉（比 LRU 简单且不会长歪）
    if (cache.size > 4000) cache.clear()
    cache.set(hash, out)
    return out
  } catch {
    return null
  }
}

/**
 * 两个指纹的差异百分比（0~100）。算不出来时视为"完全不同"。
 *
 * 指纹是**剪影**（0 = 墨迹 / 255 = 背景），所以口径是
 * `改动格数 / 两帧墨迹的并集`：
 *
 * - 分母不用 256，否则白底会把差异稀释掉（实测两个完全不同的走路姿势
 *   按"平均灰度差"只有 0.2%，滑杆等于没用）；
 * - 用**并集**而不是各自的面积：一帧站着、一帧张臂时，各自的面积差很多，
 *   按各自面积算会得到一个不直观的比例。
 *
 * 两帧都是空白（拍出两张白图）→ 0，是对的：它们确实一样。
 */
export function hashDiff(a: string, b: string): number {
  const ba = decode(a)
  const bb = decode(b)
  if (!ba || !bb) return 100
  if (ba.length !== bb.length) return 100
  let changed = 0
  let union = 0
  for (let i = 0; i < ba.length; i++) {
    const inkA = ba[i] < 128
    const inkB = bb[i] < 128
    if (inkA || inkB) union++
    if (inkA !== inkB) changed++
  }
  return union ? (changed / union) * 100 : 0
}

export type FilterOptions = {
  /** 跳帧步长：每 N 帧取 1 帧（1 = 全保留） */
  step: number
  /** 近似度阈值（%）：与**上一张保留帧**差异低于它就算重复，丢掉 */
  simPct: number
}

/**
 * 两道筛子：先按步长等间距取，再按近似度与前一张**保留帧**比。
 *
 * 「与保留帧比」而不是「与前一帧比」是关键：动作里有连续几帧几乎不动
 * （起步、定格、收势），逐帧比只会丢掉其中一部分，留下好几张一样的；
 * 跟保留帧比才能把整段静止压成 1 张。
 *
 * 阈值 0 时不过滤（只按步长）。第一帧永远保留：没有"上一帧"可比，
 * 而且空序列会让后面"存姿势集"直接报错。
 */
export function applyFilters<T extends { hash: string }>(
  frames: T[], opt: FilterOptions,
): T[] {
  if (!frames.length) return []
  const step = Math.max(1, Math.floor(opt.step || 1))
  const sim = Math.max(0, Math.min(100, opt.simPct || 0))
  const out: T[] = []
  let lastHash = ''
  for (let i = 0; i < frames.length; i += step) {
    const f = frames[i]
    if (!out.length) { out.push(f); lastHash = f.hash; continue }
    if (sim > 0 && hashDiff(lastHash, f.hash) < sim) continue
    out.push(f)
    lastHash = f.hash
  }
  return out
}

/** 筛完之后「相邻两帧差异」的分布：看这套参数是不是筛过头了 */
export function diffStats<T extends { hash: string }>(frames: T[]): {
  median: number; min: number; max: number
  /** 差异 < 3% 的相邻对：基本是同一个姿势，连播会停一拍 */
  nearDup: number
} {
  if (frames.length < 2) return { median: 0, min: 0, max: 0, nearDup: 0 }
  const ds: number[] = []
  let nearDup = 0
  for (let i = 1; i < frames.length; i++) {
    const d = hashDiff(frames[i - 1].hash, frames[i].hash)
    ds.push(d)
    if (d < 3) nearDup++
  }
  ds.sort((a, b) => a - b)
  return {
    median: ds[Math.floor(ds.length / 2)],
    min: ds[0],
    max: ds[ds.length - 1],
    nearDup,
  }
}

/** 首尾帧差异：循环动作里这个值太小会「卡一下」（末帧≈首帧） */
export function loopSeam<T extends { hash: string }>(frames: T[]): number {
  if (frames.length < 2) return 0
  return hashDiff(frames[frames.length - 1].hash, frames[0].hash)
}
