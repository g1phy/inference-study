import type { ChatMessage, Tokenizer } from '@/engine/types'
import { WEATHER_TOOL } from '@/engine/scenarios'
import { renderTemplate } from '@/engine/templates'
import { useReaderPromptTokenCount } from '@/state/prompt'

/** Поля настроек симулятора, от которых зависит промпт. */
export interface ReaderPromptInput {
  systemText: string
  userText: string
  thinking: boolean
  tools: boolean
}

/**
 * Длина промпта читателя в токенах так же, как её считает симулятор (engine/flow.ts): системный промпт,
 * вопрос и (если включены tools) описание инструмента в шаблоне чата Qwen3, токенизатор Qwen3.
 * Поэтому число совпадает с «N токенов промпта» в hero.
 */
export function readerPromptTokenCount(tokenizer: Tokenizer, s: ReaderPromptInput): number {
  const messages: ChatMessage[] = []
  if (s.systemText.trim()) messages.push({ role: 'system', content: s.systemText })
  messages.push({ role: 'user', content: s.userText })
  const input = { family: 'qwen3' as const, messages, enableThinking: s.thinking, addGenerationPrompt: true }
  const text = renderTemplate(s.tools ? { ...input, tools: [WEATHER_TOOL] } : input).text
  return tokenizer.encode(text, { allowSpecial: true }).length
}

/**
 * Сколько токенов в запросе читателя из симулятора; null, пока токенизатор не загружен.
 * Считает общий кэш страницы (@/state/prompt, тот же шаблон с tools): один пересчёт на текст для всех глав.
 */
export function useReaderPromptTokens(): number | null {
  return useReaderPromptTokenCount()
}
