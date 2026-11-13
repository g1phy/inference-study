/**
 * Ленивый загрузчик словаря Qwen3 (public/tokenizer/qwen3.json) с кэшем на уровне модуля.
 * Без React: хук useTokenizer лежит в ./useTokenizer.ts и подписывается на это состояние.
 */
import type { Tokenizer, TokenizerStatus } from '../types'
import { parseBpeAsset } from './asset'
import { ByteLevelBpeTokenizer } from './bpe'

/** Состояние загрузки, которое отдаёт useTokenizer(). */
export interface TokenizerState {
  tokenizer: Tokenizer | null
  status: TokenizerStatus
  error?: string
}

let promise: Promise<ByteLevelBpeTokenizer> | null = null
let state: TokenizerState = { tokenizer: null, status: 'idle' }
const listeners = new Set<() => void>()

function setState(next: TokenizerState): void {
  state = next
  for (const l of listeners) l()
}

/** URL словаря с учётом base из vite.config (import.meta.env.BASE_URL). */
export function tokenizerUrl(): string {
  const base = import.meta.env.BASE_URL ?? '/'
  return `${base}tokenizer/qwen3.json`
}

/**
 * Загружает и собирает токенизатор Qwen3 один раз на страницу (повторные вызовы получают тот же
 * Promise). После ошибки следующий вызов пробует снова.
 */
export function loadTokenizer(): Promise<Tokenizer> {
  return loadQwen3Tokenizer()
}

/** То же, что loadTokenizer, но с конкретным типом (доступ к trace/mergeOf/encodeIds). */
export function loadQwen3Tokenizer(): Promise<ByteLevelBpeTokenizer> {
  if (promise) return promise
  const url = tokenizerUrl()
  setState({ tokenizer: null, status: 'loading' })
  const p = fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`Не удалось загрузить словарь токенизатора (${url}): HTTP ${r.status}`)
      return r.json()
    })
    .then((json) => new ByteLevelBpeTokenizer(parseBpeAsset(json)))
  promise = p
  p.then(
    (tokenizer) => setState({ tokenizer, status: 'ready' }),
    (e: unknown) => {
      promise = null
      setState({ tokenizer: null, status: 'error', error: e instanceof Error ? e.message : String(e) })
    },
  )
  return p
}

/** Текущее состояние загрузки (снимок для useSyncExternalStore). */
export function getTokenizerState(): TokenizerState {
  return state
}

/** Уже загруженный токенизатор или null (без запуска загрузки). */
export function getLoadedTokenizer(): Tokenizer | null {
  return state.tokenizer
}

/** Подписка на изменения состояния загрузки; возвращает функцию отписки. */
export function subscribeTokenizer(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** Только для тестов: сбросить кэш модуля. */
export function resetTokenizerCacheForTests(): void {
  promise = null
  state = { tokenizer: null, status: 'idle' }
}
