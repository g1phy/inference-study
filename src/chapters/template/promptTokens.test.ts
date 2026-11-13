import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { loadO200kTokenizer } from '@/engine/tokenizer/o200k'
import { renderTemplate } from '@/engine/templates'
import type { ChatMessage } from '@/engine/types'
import { countPrompt, splitByLiterals } from './promptTokens'
import { WEATHER_TOOL } from '../request/requestModel'

// эталонный диалог из заметок (notes-tokens §2.3): Qwen3 32 токена, gpt-oss 89
const MSGS: ChatMessage[] = [
  { role: 'system', content: 'Ты полезный ассистент.' },
  { role: 'user', content: 'Сколько будет 2+2?' },
]

describe('countPrompt', () => {
  it('Qwen3 и gpt-oss совпадают с замерами заметок (32 и 89)', async () => {
    const qwen = loadQwen3ForTests()
    const o200k = await loadO200kTokenizer()
    const q = countPrompt('qwen3', renderTemplate({ family: 'qwen3', messages: MSGS }).segments, qwen, null)!
    expect(q.total).toBe(32)
    expect(q.method).toBe('exact-qwen3')
    const g = countPrompt('gpt-oss', renderTemplate({ family: 'gpt-oss', messages: MSGS }).segments, qwen, o200k)!
    expect(g.total).toBe(89)
    expect(countPrompt('gpt-oss', renderTemplate({ family: 'gpt-oss', messages: MSGS }).segments, qwen, null)).toBeNull()
  })

  it('Llama 3.1 условно: спецтокены по одному, порядок величины как у замера (53)', () => {
    const qwen = loadQwen3ForTests()
    const l = countPrompt('llama3', renderTemplate({ family: 'llama3', messages: MSGS }).segments, qwen, null)!
    expect(l.method).toBe('approx-qwen3')
    expect(l.total).toBeGreaterThan(40)
    expect(l.total).toBeLessThan(70)
  })

  it('описание get_weather добавляет 147 токенов к промпту Qwen3 (notes-tokens §5, п. 27)', () => {
    const qwen = loadQwen3ForTests()
    const msgs: ChatMessage[] = [
      { role: 'system', content: 'Ты полезный ассистент.' },
      { role: 'user', content: 'Объясни простыми словами, что такое KV cache и зачем он нужен.' },
    ]
    const a = countPrompt('qwen3', renderTemplate({ family: 'qwen3', messages: msgs }).segments, qwen, null)!
    const b = countPrompt('qwen3', renderTemplate({ family: 'qwen3', messages: msgs, tools: [WEATHER_TOOL] }).segments, qwen, null)!
    expect(b.total - a.total).toBe(147)
    expect(b.tools).toBeGreaterThan(100)
  })

  it('splitByLiterals делит хвост приглашения на спецтокены и текст', () => {
    expect(splitByLiterals('<|im_start|>assistant\n<think>\n\n</think>\n\n', 'qwen3')).toEqual([
      { text: '<|im_start|>', special: true },
      { text: 'assistant\n', special: false },
      { text: '<think>', special: true },
      { text: '\n\n', special: false },
      { text: '</think>', special: true },
      { text: '\n\n', special: false },
    ])
  })
})
