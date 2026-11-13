/**
 * Прогон запроса для hero и симулятора: настройки сайта → buildRun на настоящем токенизаторе Qwen3.
 * Ввод текста пересчитывается с задержкой 150 мс (печать не тормозит), переключатели сразу.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getSettings, useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { buildRun, DEFAULT_MAX_TOKENS, type FlowRunEx } from '@/engine/flow'
import type { FlowSettings, TokenizerStatus } from '@/engine/types'

export interface FlowState {
  run: FlowRunEx | null
  status: TokenizerStatus
  error?: string
  settings: FlowSettings
  patch: (p: Partial<FlowSettings>) => void
  /**
   * Применить отложенный ввод сразу (кнопка «Отправить»). Читает настройки прямо из стора, поэтому видит
   * правку, сделанную в том же обработчике. true, если прогон пересчитается.
   */
  flush: () => boolean
  /** max_tokens запроса: живёт только в симуляторе (в настройках сайта такого поля нет). */
  maxTokens: number
  setMaxTokens: (n: number) => void
}

const TEXT_KEYS: ReadonlySet<keyof FlowSettings> = new Set<keyof FlowSettings>(['userText', 'systemText'])

function onlyTextChanged(a: FlowSettings, b: FlowSettings): boolean {
  return (Object.keys(a) as Array<keyof FlowSettings>).every((k) => TEXT_KEYS.has(k) || a[k] === b[k])
}

/** Одинаковые настройки по содержимому (снимок useSettings и объект стора бывают разными объектами). */
function sameSettings(a: FlowSettings, b: FlowSettings): boolean {
  if (a === b) return true
  return (Object.keys(a) as Array<keyof FlowSettings>).every((k) => Object.is(a[k], b[k]))
}

/** Настройки с задержкой для текстовых полей. */
export function useDebouncedSettings(settings: FlowSettings, delay = 150): [FlowSettings, () => boolean] {
  const [value, setValue] = useState(settings)
  const latest = useRef(settings)
  latest.current = settings
  const valueRef = useRef(value)
  valueRef.current = value
  useEffect(() => {
    if (sameSettings(value, settings)) return
    if (!onlyTextChanged(settings, value)) {
      setValue(settings)
      return
    }
    const t = window.setTimeout(() => setValue(latest.current), delay)
    return () => window.clearTimeout(t)
  }, [settings, value, delay])
  const flush = useCallback(() => {
    const cur = getSettings()
    if (sameSettings(valueRef.current, cur)) return false
    valueRef.current = cur
    setValue(cur)
    return true
  }, [])
  return [value, flush]
}

export function useFlowRun(): FlowState {
  const [settings, patch] = useSettings()
  const { tokenizer, status, error } = useTokenizer()
  const [debounced, flush] = useDebouncedSettings(settings)
  const [maxTokens, setMaxTokens] = useState(DEFAULT_MAX_TOKENS)
  const run = useMemo(() => {
    if (!tokenizer) return null
    try {
      return buildRun(debounced, tokenizer, { maxTokens })
    } catch (e) {
      console.error('buildRun:', e)
      return null
    }
  }, [debounced, tokenizer, maxTokens])
  return { run, status, error, settings, patch, flush, maxTokens, setMaxTokens }
}
