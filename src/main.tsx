import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/onest'
import '@fontsource-variable/jetbrains-mono'
import 'katex/dist/katex.min.css'
import './styles/index.css'
import './state/theme' // применяет тему до первого рендера
import App from './App'

/*
 * Сайт обновили, пока вкладка была открыта: старые чанки глав отвечают 404, и динамический импорт
 * падает. Один раз перезагружаем страницу сами (якорь и настройки сохранятся), дальше глава
 * покажет «Обновить страницу». Метка в sessionStorage не даёт уйти в цикл перезагрузок.
 */
if (import.meta.env.PROD) {
  const KEY = 'inference-study:chunk-reload'
  window.addEventListener('vite:preloadError', (event) => {
    try {
      const last = Number(window.sessionStorage.getItem(KEY) ?? 0)
      if (Date.now() - last < 60_000) return
      window.sessionStorage.setItem(KEY, String(Date.now()))
    } catch {
      return
    }
    event.preventDefault()
    window.location.reload()
  })
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
