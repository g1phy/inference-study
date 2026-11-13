import { describe, expect, it } from 'vitest'
import { piecesText, scriptFor } from '@/engine/scenarios'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import type { FlowSettings } from '@/engine/types'
import { ANSWER_TEXT, MAX_TOKENS, MIN_TOKENS, pickAnswer } from './answer'

const tk = loadQwen3ForTests()
const encode = (t: string) => tk.encode(t, { allowSpecial: false })

/** Настройки по умолчанию как в src/state/settings.ts (DEFAULT_SETTINGS). */
const DEFAULTS: FlowSettings = {
  scenarioId: 'thinking',
  systemText: 'Ты полезный ассистент.',
  userText: 'Объясни простыми словами, что такое KV cache и зачем он нужен.',
  thinking: true,
  tools: false,
  json: false,
  deployment: 'agg',
  cacheWarm: false,
  modelId: 'qwen3-8b',
  gpuId: 'h100-sxm',
  temperature: 0.7,
}

describe('ответ для анимации спекуляции', () => {
  it('целые предложения, пока влезают в 48 токенов', () => {
    const s1 = 'KV cache хранит ключи и значения всех прошлых токенов.'
    const s2 = 'Поэтому decode считает только новый токен.'
    const r = pickAnswer(tk, `${s1} ${s2}\n\nВторой абзац не берём.`)
    expect(r.fromRun).toBe(true)
    expect(r.tokens.map((t) => t.id)).toEqual(encode(`${s1} ${s2}`).map((t) => t.id))
  })

  it('короткое первое предложение и длинное второе: начало абзаца по токенам, а не пример', () => {
    const s1 = 'Да, можно.'
    const s2 =
      'Квантизация уменьшает число байт на каждый вес, поэтому шаг decode, который упирается в чтение весов из памяти, становится заметно короче, а длинный prefill ускоряется, только если сами умножения идут в FP8 на тензорных ядрах.'
    // условие теста: одно первое предложение короче минимума, а вместе со вторым длиннее максимума
    expect(encode(s1).length).toBeLessThan(MIN_TOKENS)
    expect(encode(`${s1} ${s2}`).length).toBeGreaterThan(MAX_TOKENS)
    const r = pickAnswer(tk, `${s1} ${s2}`)
    expect(r.fromRun).toBe(true)
    expect(r.tokens).toHaveLength(MAX_TOKENS)
    expect(r.tokens.map((t) => t.id)).toEqual(encode(`${s1} ${s2}`).slice(0, MAX_TOKENS).map((t) => t.id))
  })

  it('запрос из ревью про квантизацию берёт начало ответа симулятора', () => {
    const script = scriptFor({ ...DEFAULTS, scenarioId: 'custom', userText: 'Почему квантизация ускоряет decode, но не всегда prefill? Объясни на примере FP8.' })
    const body = script.bodyKind === 'tool_call' ? null : piecesText(script.body)
    const r = pickAnswer(tk, body)
    expect(r.fromRun).toBe(true)
    expect(r.tokens.length).toBeGreaterThanOrEqual(MIN_TOKENS)
    expect(r.tokens.length).toBeLessThanOrEqual(MAX_TOKENS)
  })

  it('слишком короткий ответ и вызов инструмента заменяются примером', () => {
    const example = encode(ANSWER_TEXT).map((t) => t.id)
    for (const body of ['2 + 2 = 4.', null]) {
      const r = pickAnswer(tk, body)
      expect(r.fromRun).toBe(false)
      expect(r.tokens.map((t) => t.id)).toEqual(example)
    }
  })
})
