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
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
