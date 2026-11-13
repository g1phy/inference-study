import type { ChatMessage, Token, TokenizerStatus } from '@/engine/types'
import { renderTemplate } from '@/engine/templates'
import { useReaderPromptTokens } from '@/state/prompt'

/** Промпт Qwen3 целиком (шаблон чата, как его видит модель) для системного и пользовательского текста. */
export function renderQwen3Prompt(system: string, user: string, thinking = true): string {
  const messages: ChatMessage[] = []
  if (system.trim()) messages.push({ role: 'system', content: system })
  messages.push({ role: 'user', content: user })
  return renderTemplate({ family: 'qwen3', messages, enableThinking: thinking, addGenerationPrompt: true }).text
}

export interface PromptTokensState {
  /** Токены отрендеренного промпта читателя (null, пока токенизатор не загружен). */
  tokens: Token[] | null
  status: TokenizerStatus
}

/**
 * Настоящие токены запроса читателя: системный промпт и вопрос из настроек симулятора,
 * шаблон Qwen3, токенизатор Qwen3. Общий кэш страницы (@/state/prompt): один пересчёт на текст.
 */
export function usePromptTokens(): PromptTokensState {
  return useReaderPromptTokens()
}
