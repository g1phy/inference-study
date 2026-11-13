import { describe, expect, it } from 'vitest'
import { incrementalDetokenize, naiveDetokenize, stopPrefixHoldback } from './detok'
import { loadQwen3ForTests } from './tokenizer/testing'

const tok = loadQwen3ForTests()

describe('incrementalDetokenize: неполный UTF-8 (аналог notes-tokens §3.8 на словаре Qwen3)', () => {
  const prompt = tok.encode('Вопрос:')
  const out = tok.encode(' Ответ: 4 🫠 готово')

  it('эмодзи 🫠 у Qwen3 режется на 3 токена, наивный декодер печатает «�»', () => {
    const texts = out.map((t) => t.text)
    expect(texts.filter((t) => t.includes('�'))).toHaveLength(3)
    expect(naiveDetokenize(out, tok)).toBe(' Ответ: 4 ��� готово')
  })

  it('офсеты придерживают байты, пока символ не соберётся', () => {
    const steps = incrementalDetokenize(out, tok, { promptTokens: prompt })
    expect(steps.map((s) => s.emitted).join('')).toBe(' Ответ: 4 🫠 готово')
    const i = out.findIndex((t) => t.text === ' �')
    expect(steps[i].emitted).toBe('')
    expect(steps[i].held).toBe(' �')
    expect(steps[i].heldBytes).toEqual([0x20, 0xf0, 0x9f])
    expect(steps[i + 1].emitted).toBe('')
    expect(steps[i + 1].heldBytes).toEqual([0x20, 0xf0, 0x9f, 0xab])
    expect(steps[i + 2].emitted).toBe(' 🫠')
    expect(steps[i + 2].heldBytes).toEqual([])
    // во время удержания офсеты не двигаются
    expect(steps[i + 1].readOffset).toBe(steps[i].readOffset)
    expect(steps[i + 2].readOffset).toBe(i + 3)
  })

  it('окно заряжено токенами промпта: стартовые офсеты отрицательные (в промпте)', () => {
    const longPrompt = tok.encode('Это длинный промпт из многих токенов. Вопрос:')
    expect(longPrompt.length).toBeGreaterThanOrEqual(7)
    const steps = incrementalDetokenize(out, tok, { promptTokens: longPrompt })
    // после первого шага: prefix_offset = старый read_offset (0), read_offset = 1
    expect(steps[0].prefixOffset).toBe(0)
    expect(steps[0].readOffset).toBe(1)
    const noPrompt = incrementalDetokenize(out, tok)
    expect(noPrompt.map((s) => s.emitted).join('')).toBe(' Ответ: 4 🫠 готово')
  })

  it('кириллическая буква, разрезанная пополам (Ѣ)', () => {
    const t = tok.encode('Ѣѣ')
    expect(t[0].text).toBe('�')
    const steps = incrementalDetokenize(t, tok)
    expect(steps.map((s) => s.emitted)).toEqual(['', 'Ѣ', 'ѣ'])
  })
})

describe('skip_special_tokens и стоп-токены', () => {
  const answer = tok.encode('<think>\nМысль.\n</think>\n\nОтвет<|im_end|>')

  it('<think> (hfSpecial=false) остаётся, <|im_end|> выкидывается', () => {
    const steps = incrementalDetokenize(answer, tok)
    expect(steps.map((s) => s.emitted).join('')).toBe('<think>\nМысль.\n</think>\n\nОтвет')
    const raw = incrementalDetokenize(answer, tok, { skipSpecial: false })
    expect(raw.map((s) => s.emitted).join('')).toBe('<think>\nМысль.\n</think>\n\nОтвет<|im_end|>')
  })

  it('стоп-токен заканчивает поток и сам текста не даёт', () => {
    const steps = incrementalDetokenize(answer, tok, { skipSpecial: false, stopTokenIds: [151645] })
    const last = steps.at(-1)!
    expect(last.stopped).toBe(true)
    expect(last.stopReason).toBe(151645)
    expect(last.output).toBe('<think>\nМысль.\n</think>\n\nОтвет')
  })
})

describe('stop-строки поперёк токенов (notes-tokens §3.10)', () => {
  const t = tok.encode('Ответ: 4</answer> и ещё мусор')

  it("токены: 'От' 'вет' ':' ' ' '4' '</' 'answer' '>' ' и' …", () => {
    expect(t.slice(0, 9).map((x) => x.text)).toEqual(['От', 'вет', ':', ' ', '4', '</', 'answer', '>', ' и'])
  })

  it('Dynamo jail: держим только возможное начало stop-строки', () => {
    const steps = incrementalDetokenize(t, tok, { stop: ['</answer>'] })
    expect(steps).toHaveLength(8)
    expect(steps.map((s) => s.emitted)).toEqual(['От', 'вет', ':', ' ', '4', '', '', ''])
    expect(steps[5].stopHeld).toBe('</')
    expect(steps[6].stopHeld).toBe('</answer')
    expect(steps[7].stopped).toBe(true)
    expect(steps[7].stopReason).toBe('</answer>')
    expect(steps[7].output).toBe('Ответ: 4')
  })

  it('vLLM: фиксированный хвост max(len) − 1 = 8 символов', () => {
    const steps = incrementalDetokenize(t, tok, { stop: ['</answer>'], stopMode: 'vllm' })
    expect(steps.map((s) => s.emitted)).toEqual(['', '', '', '', '', 'От', 'вет: 4', ''])
    expect(steps[4].stopHeld).toBe('Ответ: 4')
    expect(steps[7].stopped).toBe(true)
    expect(steps[7].output).toBe('Ответ: 4')
  })

  it('include_stop_str_in_output', () => {
    const steps = incrementalDetokenize(t, tok, { stop: ['</answer>'], includeStopStr: true })
    expect(steps.at(-1)!.output).toBe('Ответ: 4</answer>')
  })

  it('частичное совпадение оборвалось → придержанное уходит одним куском (notes-dynamo §3.13)', () => {
    const toks = tok.encode('Ответ: 4.\nUnit')
    const steps = incrementalDetokenize(toks, tok, { stop: ['\nUser:'], finished: false })
    const iNl = toks.findIndex((x) => x.text.includes('\n'))
    expect(steps[iNl].stopHeld.startsWith('\n')).toBe(true)
    expect(steps.map((s) => s.emitted).join('')).toBe('Ответ: 4.\nUnit')
    expect(steps.some((s) => s.stopped)).toBe(false)
  })

  it('без finished хвост под stop-строку остаётся придержанным; с finished — выдаётся', () => {
    const toks = tok.encode('Итог: <')
    const open = incrementalDetokenize(toks, tok, { stop: ['</answer>'], finished: false })
    expect(open.at(-1)!.stopHeld).toBe('<')
    expect(open.map((s) => s.emitted).join('')).toBe('Итог: ')
    const done = incrementalDetokenize(toks, tok, { stop: ['</answer>'] })
    expect(done.map((s) => s.emitted).join('')).toBe('Итог: <')
  })

  it('stopPrefixHoldback', () => {
    expect(stopPrefixHoldback('abc</ans', ['</answer>'])).toBe(5)
    expect(stopPrefixHoldback('abc', ['</answer>'])).toBe(0)
    expect(stopPrefixHoldback('x\nUs', ['\nUser:', 'Usage'])).toBe(3)
  })
})
