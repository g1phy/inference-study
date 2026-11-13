import { defineConfig } from 'vitest/config'

/** Отдельный конфиг без плагинов Vite: тесты чистые (движок, утилиты), DOM не нужен. */
const srcDir = decodeURIComponent(new URL('./src', import.meta.url).pathname)

export default defineConfig({
  resolve: {
    alias: { '@': srcDir },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
    // запас по времени: тяжёлые проверочные тесты (attention.verify, токенизатор) при параллельной нагрузке идут дольше 5 с
    testTimeout: 30_000,
  },
})
