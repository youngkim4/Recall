import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// the origins this dev server is reached at
const OWN_ORIGINS = new Set(['http://127.0.0.1:5173', 'http://localhost:5173'])

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.RECALL_API_PROXY || env.VITE_RECALL_API_BASE || 'http://127.0.0.1:8765'
  return {
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      port: 5173,
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
          // The server takes a write (POST, DELETE) only from its own origin.
          // changeOrigin rewrites Host, not Origin, so a write from this page
          // is given the server's origin; a page elsewhere keeps its own.
          configure: (proxy) =>
            proxy.on('proxyReq', (request) => {
              if (OWN_ORIGINS.has(String(request.getHeader('origin')))) request.setHeader('origin', target)
            }),
        },
      },
    },
  }
})
