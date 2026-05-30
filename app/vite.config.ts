import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      port: 5173,
      proxy: {
        '/api': {
          target: env.RECALL_API_PROXY || env.VITE_RECALL_API_BASE || 'http://127.0.0.1:8765',
          changeOrigin: true,
        },
      },
    },
  }
})
