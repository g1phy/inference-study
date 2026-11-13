import { useMemo } from 'react'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { ChatMessage, Token, Tokenizer, TokenizerStatus } from '@/engine/types'
import { readerPromptTokens } from '@/state/prompt'
import { useSetting } from '@/state/settings'

/**
 * Промпт читателя так, как его видит модель: шаблон Qwen3 (system + user, описание инструмента, если
 * в симуляторе включены tools, thinking из настроек) и настоящая токенизация словарём Qwen3.
 * Пока словарь грузится, tokens = null.
 */
export interface PromptTokens {
  status: TokenizerStatus
  tokens: Token[] | null
  tokenizer: Tokenizer | null
  thinking: boolean
  /** Вопрос по умолчанию про KV cache (для условного ответа). */
  aboutKv: boolean
}

export function promptMessages(systemText: string, userText: string): ChatMessage[] {
  const messages: ChatMessage[] = []
  if (systemText.trim()) messages.push({ role: 'system', content: systemText })
  messages.push({ role: 'user', content: userText })
  return messages
}

export function usePromptTokens(): PromptTokens {
  const systemText = useSetting('systemText')
  const userText = useSetting('userText')
  const thinking = useSetting('thinking')
  const tools = useSetting('tools')
  const { tokenizer, status } = useTokenizer()
  return useMemo(() => {
    const aboutKv = /kv|кэш|кеш|cache/i.test(userText)
    if (!tokenizer) return { status, tokens: null, tokenizer: null, thinking, aboutKv }
    // общий кэш страницы: тот же шаблон Qwen3, один пересчёт на текст для всех глав
    return { status, tokens: readerPromptTokens(tokenizer, systemText, userText, thinking, tools), tokenizer, thinking, aboutKv }
  }, [tokenizer, status, systemText, userText, thinking, tools])
}

/**
 * Условный короткий ответ модели (модель в браузере не запускается). С thinking ответ
 * начинается с короткого блока рассуждения, как у Qwen3; заканчивается стоп-токеном <|im_end|>.
 */
export function illustrativeAnswer(thinking: boolean, aboutKv: boolean): string {
  const think = thinking ? '<think>\nКоротко.\n</think>\n\n' : ''
  const body = aboutKv ? 'KV cache хранит ключи и значения прочитанных токенов.' : 'Хороший вопрос. Отвечу коротко и по делу.'
  return `${think}${body}<|im_end|>`
}
