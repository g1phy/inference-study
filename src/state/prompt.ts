import { useMemo } from 'react'
import { WEATHER_TOOL } from '@/engine/scenarios'
import { renderTemplate } from '@/engine/templates'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { ChatMessage, Token, Tokenizer, TokenizerStatus } from '@/engine/types'
import { useSetting } from './settings'

/**
 * Промпт читателя так, как его видит модель: шаблон Qwen3 (system + user, описание инструмента get_weather,
 * если в симуляторе включены tools, thinking из настроек, add_generation_prompt) и настоящая токенизация
 * словарём Qwen3. Число токенов совпадает с «N токенов промпта» в симуляторе (engine/flow.ts, buildRun).
 *
 * Один общий кэш на страницу: сколько бы фигур ни спросили токены промпта, для одного и того же
 * текста он считается один раз. Внутри глав (SettingsScope committed) текст приходит с задержкой
 * после набора, поэтому пересчёт идёт раз на паузу в наборе, а не на каждую букву.
 *
 *   const { tokens, status } = useReaderPromptTokens()
 *   const n = useReaderPromptTokenCount()   // только длина (null, пока словарь грузится)
 */

export interface ReaderPromptTokens {
  /** Токены промпта (null, пока токенизатор не загружен). */
  tokens: Token[] | null
  status: TokenizerStatus
}

function promptMessages(systemText: string, userText: string): ChatMessage[] {
  const messages: ChatMessage[] = []
  if (systemText.trim()) messages.push({ role: 'system', content: systemText })
  messages.push({ role: 'user', content: userText })
  return messages
}

/** Текст промпта Qwen3 целиком (шаблон чата с приглашением ассистенту; с tools описание инструмента в system). */
export function renderReaderPrompt(systemText: string, userText: string, thinking: boolean, tools = false): string {
  return renderTemplate({
    family: 'qwen3',
    messages: promptMessages(systemText, userText),
    enableThinking: thinking,
    addGenerationPrompt: true,
    ...(tools ? { tools: [WEATHER_TOOL] } : {}),
  }).text
}

/* небольшой кэш: живой и принятый текст могут отличаться, держим несколько последних вариантов */
const CACHE_MAX = 4
const cache: Array<{ tokenizer: Tokenizer; key: string; tokens: Token[] }> = []

/** Токены промпта с общим кэшем (один пересчёт на текст, сколько бы фигур ни спросили). */
export function readerPromptTokens(
  tokenizer: Tokenizer,
  systemText: string,
  userText: string,
  thinking: boolean,
  tools = false,
): Token[] {
  const key = `${thinking ? 1 : 0}${tools ? 1 : 0}\u0000${systemText}\u0000${userText}`
  const i = cache.findIndex((c) => c.tokenizer === tokenizer && c.key === key)
  if (i >= 0) {
    const hit = cache[i]
    if (i > 0) {
      cache.splice(i, 1)
      cache.unshift(hit)
    }
    return hit.tokens
  }
  const tokens = tokenizer.encode(renderReaderPrompt(systemText, userText, thinking, tools))
  cache.unshift({ tokenizer, key, tokens })
  if (cache.length > CACHE_MAX) cache.length = CACHE_MAX
  return tokens
}

/** Токены промпта читателя (системный текст, вопрос, tools и thinking из настроек). */
export function useReaderPromptTokens(): ReaderPromptTokens {
  const systemText = useSetting('systemText')
  const userText = useSetting('userText')
  const thinking = useSetting('thinking')
  const tools = useSetting('tools')
  const { tokenizer, status } = useTokenizer()
  const tokens = useMemo(
    () => (tokenizer ? readerPromptTokens(tokenizer, systemText, userText, thinking, tools) : null),
    [tokenizer, systemText, userText, thinking, tools],
  )
  return { tokens, status }
}

/** Только длина промпта в токенах: для корней глав, которым не нужны сами токены. */
export function useReaderPromptTokenCount(): number | null {
  return useReaderPromptTokens().tokens?.length ?? null
}
