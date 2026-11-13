import type { Token, Tokenizer } from '@/engine/types'

/** Условный ответ модели на вопрос по умолчанию (текст выдуман, токены настоящие, словарь Qwen3). */
export const ANSWER_TEXT =
  'KV cache хранит ключи и значения всех прошлых токенов. Без него модель на каждом шаге пересчитывала бы весь контекст, а с ним считает только новый токен.'

export const MIN_TOKENS = 16
export const MAX_TOKENS = 48

/**
 * Ответ для анимации: начало заготовленного ответа симулятора на запрос читателя (первый абзац, целыми
 * предложениями, до 48 токенов; если первое предложение короткое, а со вторым уже больше 48, то начало абзаца
 * по токенам). Вызов инструмента (body = null) и слишком короткие ответы («2 + 2 = 4.») заменяем примером.
 */
export function pickAnswer(tokenizer: Tokenizer, body: string | null): { tokens: Token[]; fromRun: boolean } {
  const encode = (t: string) => tokenizer.encode(t, { allowSpecial: false })
  if (body) {
    const para = body.trim().split(/\n\s*\n/)[0].trim()
    const sentences = para.split(/(?<=[.!?])\s+/)
    let text = ''
    for (const sentence of sentences) {
      const next = text ? `${text} ${sentence}` : sentence
      if (text && encode(next).length > MAX_TOKENS) break
      text = next
    }
    let tokens = encode(text)
    // первое предложение короткое, а второе целиком не влезает: берём начало абзаца по токенам
    if (tokens.length < MIN_TOKENS) tokens = encode(sentences.join(' '))
    if (tokens.length > MAX_TOKENS) tokens = tokens.slice(0, MAX_TOKENS)
    if (tokens.length >= MIN_TOKENS) return { tokens, fromRun: true }
  }
  return { tokens: encode(ANSWER_TEXT), fromRun: false }
}
