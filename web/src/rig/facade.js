// 一比一搬入的 rigby2d 运行时门面。
//
// ./core 是上游 src/core 的逐字副本（25 文件 / 8945 行），./io 是上游编辑器里
// 不依赖 DOM/Tauri 的纯逻辑（PSD/SVG/PNG 导入、Spine JSON/.skel/图集导出、
// 工作文件 ZIP、图集装箱）。这里只做命名空间聚合，避免不同模块的同名导出打架，
// 也避免我们在 TS 里重新实现任何核心算法。
//
// 许可：MIT OR Apache-2.0，见同目录 LICENSE-MIT / LICENSE-APACHE /
// THIRD_PARTY_NOTICES.md。品牌与演示角色素材未引入。

export * as model from './core/model.js'
export * as pose from './core/pose.js'
export * as state from './core/state.js'
export * as anim from './core/animation.js'
export * as curve from './core/curve.js'
export * as math from './core/math.js'
export * as autorig from './core/autorig.js'
export * as interop from './core/interop.js'
export * as binary from './core/binary.js'
export * as atlas from './core/atlas.js'
export * as draw from './core/draw.js'
export * as constraints from './core/constraints.js'
export * as constraintdata from './core/constraintdata.js'
export * as constraintkeys from './core/constraintkeys.js'
export * as meshgen from './core/meshgen.js'
export * as meshgeom from './core/meshgeom.js'
export * as clipping from './core/clipping.js'
export * as scene from './core/scene.js'
export * as poses from './core/poses.js'
export * as retarget from './core/retarget.js'
export * as lipsync from './core/lipsync.js'
export * as sequence from './core/sequence.js'
export * as bodyparts from './core/bodyparts.js'

export * as images from './io/images.js'
export * as io from './io/io.js'
export * as pack from './io/pack.js'
export * as zip from './io/zip.js'
export * as sample from './io/sample.js'
export * as output from './io/output.js'
export * as raster from './io/raster.js'
// 影片工作区：上游自写的 MP4 / WebM(含 alpha) / 音频封装器，纯逻辑一比一搬入。
export * as mp4 from './io/mp4.js'
export * as webm from './io/webm.js'
export * as audio from './io/audio.js'
