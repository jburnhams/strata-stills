import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `npm run build` writes into the Python package, so `./stratastills serve` serves the built app.
// `npm run dev` runs the dev server with hot reload and proxies /api to a running `./stratastills serve` (port 8361).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: '../src/stratastills/server/static', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8361' } },
})
