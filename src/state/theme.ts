import { useSyncExternalStore } from 'react'

/**
 * Тема сайта: светлая, тёмная или как в системе.
 * Хранится в localStorage под ключом 'inference-study:theme' ('light' | 'dark'; для 'system' ключ удаляется).
 * index.html читает этот ключ до отрисовки, поэтому мигания темы нет.
 */
export type ThemePreference = 'light' | 'dark' | 'system'
export type ResolvedTheme = 'light' | 'dark'

export interface ThemeState {
  /** Что выбрал пользователь. */
  theme: ThemePreference
  /** Что реально показано сейчас. */
  resolved: ResolvedTheme
}

export const THEME_STORAGE_KEY = 'inference-study:theme'
const DARK_QUERY = '(prefers-color-scheme: dark)'
const THEME_COLORS: Record<ResolvedTheme, string> = { light: '#f5f5f2', dark: '#0e0e0d' }
const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined'

function readStored(): ThemePreference {
  if (!isBrowser) return 'system'
  try {
    const v = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (v === 'light' || v === 'dark') return v
  } catch {
    /* localStorage недоступен (приватный режим) */
  }
  return 'system'
}

const media = isBrowser && typeof window.matchMedia === 'function' ? window.matchMedia(DARK_QUERY) : null

let state: ThemeState = compute(readStored())
const listeners = new Set<() => void>()

function compute(theme: ThemePreference): ThemeState {
  const resolved: ResolvedTheme = theme === 'system' ? (media?.matches ? 'dark' : 'light') : theme
  return { theme, resolved }
}

function applyToDocument() {
  if (!isBrowser) return
  const root = document.documentElement
  if (state.theme === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', state.theme)
  // index.html ставит два theme-color с media (светлая и тёмная система) для первой отрисовки;
  // дальше остаётся один, без media, по реально показанной теме (в том числе выбранной вручную)
  const metas = Array.from(document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]'))
  let meta = metas[0]
  metas.slice(1).forEach((m) => m.remove())
  if (!meta) {
    meta = document.createElement('meta')
    meta.name = 'theme-color'
    document.head.appendChild(meta)
  }
  meta.removeAttribute('media')
  meta.content = THEME_COLORS[state.resolved]
}

function emit(next: ThemeState) {
  if (next.theme === state.theme && next.resolved === state.resolved) return
  state = next
  applyToDocument()
  listeners.forEach((l) => l())
}

if (isBrowser) {
  applyToDocument()
  media?.addEventListener('change', () => emit(compute(state.theme)))
  window.addEventListener('storage', (e) => {
    if (e.key === THEME_STORAGE_KEY) emit(compute(readStored()))
  })
}

export function getTheme(): ThemeState {
  return state
}

export function setTheme(theme: ThemePreference): void {
  try {
    if (theme === 'system') window.localStorage.removeItem(THEME_STORAGE_KEY)
    else window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch {
    /* не страшно: тема просто не запомнится */
  }
  emit(compute(theme))
}

/** Переключить светлую и тёмную (от того, что видно сейчас). */
export function toggleTheme(): void {
  setTheme(state.resolved === 'dark' ? 'light' : 'dark')
}

export function subscribeTheme(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/**
 * Хук темы. `resolved` нужен редко: все цвета на сайте берутся из CSS-переменных
 * и переключаются сами. Используйте для canvas или логики, зависящей от темы.
 */
export function useTheme(): ThemeState & { setTheme: typeof setTheme; toggle: typeof toggleTheme } {
  const s = useSyncExternalStore(subscribeTheme, getTheme, getTheme)
  return { ...s, setTheme, toggle: toggleTheme }
}
