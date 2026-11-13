import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

/** Алиас `@/` → `src/` (тот же путь прописан в tsconfig.json → paths). */
const srcDir = decodeURIComponent(new URL('./src', import.meta.url).pathname)

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': srcDir },
  },
  server: { port: 5173, strictPort: true },
  preview: { port: 4173 },
})
