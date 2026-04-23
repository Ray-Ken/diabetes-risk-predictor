import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub project pages: set VITE_BASE_PATH at build (e.g. /my-repo/). Default is / for custom domain or user site.
const base = process.env.VITE_BASE_PATH?.replace(/\/?$/, '/') || '/'

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // ADD THIS LINE - bind to all interfaces
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
