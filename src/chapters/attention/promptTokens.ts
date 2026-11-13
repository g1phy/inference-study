/**
 * Токены для фигуры «Ваш промпт»: реплика пользователя так, как её видит модель в шаблоне Qwen3
 * («<|im_start|>user\n» + текст). Заголовок кодируется со спецтокенами, сам текст без них:
 * из пользовательского текста спецтокен не получается (глава «Токенизация»).
 */
import type { Token, Tokenizer } from '@/engine/types'

export const USER_HEADER = '<|im_start|>user\n'

/** Пример с повтором: на нём видно induction и «тот же токен раньше». */
export const REPEAT_EXAMPLE = 'KV cache хранит ключи. KV cache хранит значения, и KV cache растёт.'

export interface PromptTokens {
  tokens: Token[]
  /** Сколько токенов во всей реплике (до обрезки). */
  total: number
  truncated: boolean
}

export function buildPromptTokens(tokenizer: Tokenizer, userText: string, limit: number): PromptTokens {
  const head = tokenizer.encode(USER_HEADER)
  const body = tokenizer.encode(userText, { allowSpecial: false })
  const all = [...head, ...body]
  return { tokens: all.slice(0, limit), total: all.length, truncated: all.length > limit }
}

/** Подпись токена для осей: пробел «·», перевод строки «↵», таб «→», спецтокен литералом. */
export function tokenLabel(t: Token): string {
  if (t.special) return t.raw
  const s = t.text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→').replace(/\r/g, '␍')
  return s.length ? s : '∅'
}
