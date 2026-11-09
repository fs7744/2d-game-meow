import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时把 /api 与文件请求代理到 FastAPI（8000），生产时同源由 FastAPI 托管 dist
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/generate': 'http://127.0.0.1:8000',
      '/jobs': 'http://127.0.0.1:8000',
      '/assets': 'http://127.0.0.1:8000',
      '/files': 'http://127.0.0.1:8000',
      '/upload': 'http://127.0.0.1:8000',
      '/mask': 'http://127.0.0.1:8000',
      '/characters': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // PSD/SVG 读取是动态 import。内联成单文件：本地工具没有按需加载的收益，
    // 而且 web/smoke.mjs 用 jsdom 把入口 bundle 当普通脚本注入，多 chunk 的
    // 顶层 import/export 会直接 SyntaxError。
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
})
