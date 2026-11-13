/**
 * React-хук ленивой загрузки токенизатора Qwen3. Единственный файл движка, который знает о React.
 * Состояние общее на страницу (кэш в ./loader.ts): сколько бы компонентов ни вызвали хук,
 * словарь качается и собирается один раз.
 */
import { useEffect, useSyncExternalStore } from 'react'
import { getTokenizerState, loadTokenizer, subscribeTokenizer, type TokenizerState } from './loader'

export { loadTokenizer } from './loader'
export type { TokenizerState } from './loader'

/**
 * { tokenizer, status, error }: при первом использовании запускает загрузку
 * `${BASE_URL}tokenizer/qwen3.json`; пока status !== 'ready', tokenizer = null (UI рисует скелетон).
 */
export function useTokenizer(): TokenizerState {
  const s = useSyncExternalStore(subscribeTokenizer, getTokenizerState, getTokenizerState)
  useEffect(() => {
    if (s.status === 'idle') loadTokenizer().catch(() => {})
  }, [s.status])
  return s
}
