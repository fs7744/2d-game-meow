/**
 * rigby2d 运行时是一比一搬入的 JS（未改成 TS）。这里给门面一组宽松声明，
 * 让 TS 组件能引用而不必把 9k 行核心重写成带类型的版本；具体字段结构由
 * components/characters/rig/types.ts 里的业务接口描述。
 */
export const model: any
export const pose: any
export const state: any
export const anim: any
export const curve: any
export const math: any
export const autorig: any
export const interop: any
export const binary: any
export const atlas: any
export const draw: any
export const constraints: any
export const constraintdata: any
export const constraintkeys: any
export const meshgen: any
export const meshgeom: any
export const clipping: any
export const scene: any
export const poses: any
export const retarget: any
export const lipsync: any
export const sequence: any
export const bodyparts: any
export const images: any
export const io: any
export const pack: any
export const zip: any
export const sample: any
export const output: any
export const raster: any
export const mp4: any
export const webm: any
export const audio: any
