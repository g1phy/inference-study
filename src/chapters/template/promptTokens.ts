/**
 * Подсчёт токенов отрендеренного шаблона для разных семейств (общий для глав «Запрос» и «Шаблон чата»).
 *
 * В браузере есть два настоящих словаря: Qwen3 (useTokenizer) и o200k_harmony (gpt-oss, loadO200kTokenizer).
 * Для Llama 3.1 и DeepSeek-R1 своих словарей нет, поэтому счёт условный: каждый спецтокен семейства
 * считается одним токеном (так и есть в их словарях), а остальной текст режется словарём Qwen3.
 */
import type { ModelFamily, TemplateSegment, Tokenizer } from '@/engine/types'

export type CountMethod = 'exact-qwen3' | 'exact-o200k' | 'approx-qwen3'

export interface PromptCount {
  total: number
  /** Токены текста сообщений (сегменты content), приблизительно: границы кусков считаются отдельно. */
  content: number
  /** Токены описания инструментов (сегменты tools). */
  tools: number
  method: CountMethod
}

/** Каким словарём считаем семейство. */
export function countMethodOf(family: ModelFamily): CountMethod {
  if (family === 'qwen3') return 'exact-qwen3'
  if (family === 'gpt-oss') return 'exact-o200k'
  return 'approx-qwen3'
}

/** Сколько токенов в куске обычного текста (без спецтокенов). */
function plainCount(tk: Tokenizer, text: string): number {
  return text ? tk.encode(text, { allowSpecial: false }).length : 0
}

/**
 * Токены промпта по сегментам шаблона. qwen: словарь Qwen3; o200k: словарь gpt-oss (нужен только для 'gpt-oss').
 * Возвращает null, если нужного словаря ещё нет.
 */
export function countPrompt(
  family: ModelFamily,
  segments: TemplateSegment[],
  qwen: Tokenizer | null,
  o200k: Tokenizer | null,
): PromptCount | null {
  const method = countMethodOf(family)
  const tk = method === 'exact-o200k' ? o200k : qwen
  if (!tk) return null
  const text = segments.map((s) => s.text).join('')
  let total: number
  if (method === 'approx-qwen3') {
    // спецтокены по одному, всё между ними целыми кусками словарём Qwen3
    total = 0
    let run = ''
    for (const s of segments) {
      if (s.kind === 'special') {
        total += plainCount(tk, run) + 1
        run = ''
      } else if (s.kind === 'gen-prompt') {
        // хвост приглашения содержит спецтокены литералами: делим его сами
        for (const part of splitByLiterals(s.text, family)) {
          if (part.special) {
            total += plainCount(tk, run) + 1
            run = ''
          } else run += part.text
        }
      } else run += s.text
    }
    total += plainCount(tk, run)
  } else {
    total = tk.encode(text).length
  }
  let content = 0
  let tools = 0
  for (const s of segments) {
    if (s.kind === 'content') content += plainCount(tk, s.text)
    else if (s.kind === 'tools') tools += plainCount(tk, s.text)
  }
  return { total, content: Math.min(content, total), tools: Math.min(tools, total), method }
}

/** Литералы спецтокенов, которые встречаются в хвостах generation prompt (для условного счёта). */
const GEN_LITERALS: Record<ModelFamily, string[]> = {
  qwen3: ['<|im_start|>', '<think>', '</think>'],
  llama3: ['<|start_header_id|>', '<|end_header_id|>'],
  'deepseek-r1': ['<｜Assistant｜>', '<think>'],
  'gpt-oss': ['<|start|>'],
}

/** Делит строку на литералы спецтокенов семейства и обычный текст. */
export function splitByLiterals(text: string, family: ModelFamily): Array<{ text: string; special: boolean }> {
  const lits = GEN_LITERALS[family]
  const out: Array<{ text: string; special: boolean }> = []
  let i = 0
  let buf = ''
  while (i < text.length) {
    const lit = lits.find((l) => text.startsWith(l, i))
    if (lit) {
      if (buf) out.push({ text: buf, special: false })
      buf = ''
      out.push({ text: lit, special: true })
      i += lit.length
    } else {
      buf += text[i]
      i++
    }
  }
  if (buf) out.push({ text: buf, special: false })
  return out
}
