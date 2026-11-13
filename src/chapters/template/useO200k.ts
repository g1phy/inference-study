import { useEffect, useState } from 'react'
import { loadO200kTokenizer, type O200kTokenizer } from '@/engine/tokenizer'

export type O200kState = { tokenizer: O200kTokenizer | null; status: 'idle' | 'loading' | 'ready' | 'error' }

/**
 * Словарь o200k_harmony (gpt-oss) грузится отдельным чанком около 1 МБ gzip, поэтому только по требованию:
 * когда enabled стал true (фигура на экране или выбрано семейство gpt-oss). Promise кэшируется на всю страницу
 * (loadO200kTokenizer), так что повторные вызовы ничего не качают заново.
 */
export function useO200k(enabled: boolean): O200kState {
  const [state, setState] = useState<O200kState>({ tokenizer: null, status: 'idle' })
  useEffect(() => {
    if (!enabled) return
    let alive = true
    setState((s) => (s.status === 'ready' ? s : { tokenizer: null, status: 'loading' }))
    loadO200kTokenizer('o200k_harmony').then(
      (tokenizer) => {
        if (alive) setState({ tokenizer, status: 'ready' })
      },
      () => {
        if (alive) setState({ tokenizer: null, status: 'error' })
      },
    )
    return () => {
      alive = false
    }
  }, [enabled])
  return state
}
