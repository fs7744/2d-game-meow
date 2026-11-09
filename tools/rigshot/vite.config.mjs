/**
 * 浏览器端核对页的构建配置（配合 tools/shot/rig_pixels.mjs 使用）。
 *
 *   cd web && npx vite build --config ../tools/rigshot/vite.config.mjs
 *
 * 产物落在 tests/_out/rig_harness/（tests/_out 不入库，可随时重建）。
 */
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const repo = resolve(here, '..', '..')

export default {
  root: here,
  base: './',
  configFile: false,
  build: {
    outDir: resolve(repo, 'tests', '_out', 'rig_harness'),
    emptyOutDir: true,
    target: 'es2022',
    // 核对页不追求体积，别把产物拆成一堆小 chunk 方便排查。
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
  server: { fs: { allow: [repo] } },
}
