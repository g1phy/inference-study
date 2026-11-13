import { describe, expect, it } from 'vitest'
import { incrementalDetokenize } from './detok'
import {
  harmonyToChannels,
  partialTagOverlap,
  runStreamingParser,
  streamableArguments,
  toSseChunks,
  toSseText,
  type ParserEventEx,
  type ParserOptionsEx,
} from './parsers'
import { loadQwen3ForTests } from './tokenizer/testing'

const tok = loadQwen3ForTests()
/** Текст ответа модели → куски, как их отдаёт детокенизатор (по токену, <|im_end|> пропущен). */
const chunksOf = (text: string) => incrementalDetokenize(tok.encode(text), tok).map((s) => s.emitted)

const lanes = (events: ParserEventEx[]) => {
  const r = { reasoning: '', content: '', tool_call: '' }
  for (const e of events) for (const d of e.deltas) r[d.channel] += d.text
  return r
}
const toolArgs = (events: ParserEventEx[], index = 0) =>
  events.flatMap((e) => e.toolDeltas.filter((d) => d.index === index && d.arguments !== undefined).map((d) => d.arguments)).join('')

const QWEN: ParserOptionsEx = { reasoning: 'qwen3', tools: 'hermes' }
const META = { id: 'chatcmpl-7f3a', created: 1762950000, model: 'Qwen/Qwen3-8B' }

describe('Сценарий A: мысли и ответ (notes-tokens §3.11)', () => {
  const chunks = chunksOf('<think>\nПользователь спрашивает 2+2. Это 4.\n</think>\n\n2 + 2 = 4.<|im_end|>')
  const events = runStreamingParser(chunks, QWEN)

  it('дорожки reasoning и content', () => {
    expect(lanes(events)).toEqual({ reasoning: '\nПользователь спрашивает 2+2. Это 4.\n', content: '\n\n2 + 2 = 4.', tool_call: '' })
    expect(events[0].stateBefore).toBe('content')
    expect(events[0].stateAfter).toBe('reasoning')
    expect(events[0].deltas).toEqual([])
    expect(events[0].note).toContain('<think>')
  })

  it('SSE: роль, 17 reasoning-чанков, 9 content-чанков, stop, usage, [DONE]', () => {
    const sse = toSseChunks(events, { ...META, includeUsage: true, usage: { promptTokens: 32, completionTokens: 29 } })
    expect(sse).toHaveLength(30)
    expect(sse[0].line).toBe(
      'data: {"id":"chatcmpl-7f3a","object":"chat.completion.chunk","created":1762950000,"model":"Qwen/Qwen3-8B","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}',
    )
    const reasoning = sse.filter((c) => (c.json?.choices as any[])?.[0]?.delta?.reasoning_content !== undefined)
    const content = sse.filter((c) => typeof (c.json?.choices as any[])?.[0]?.delta?.content === 'string' && (c.json!.choices as any[])[0].delta.role === undefined)
    expect(reasoning).toHaveLength(17)
    expect(content.map((c) => (c.json!.choices as any[])[0].delta.content)).toEqual(['\n\n', '2', ' +', ' ', '2', ' =', ' ', '4', '.'])
    expect(sse[27].json!.choices).toEqual([{ index: 0, delta: {}, logprobs: null, finish_reason: 'stop' }])
    expect(sse[28].json).toEqual({ ...META, object: 'chat.completion.chunk', choices: [], usage: { prompt_tokens: 32, completion_tokens: 29, total_tokens: 61 } })
    expect(sse[29]).toEqual({ line: 'data: [DONE]', json: null })
    expect(toSseText(sse).endsWith('data: [DONE]\n\n')).toBe(true)
  })

  it("поле 'reasoning' (vLLM ≥ 0.16) и stop_reason", () => {
    const sse = toSseChunks(events, { ...META, reasoningField: 'reasoning', stopReason: null })
    expect(sse[1].line).toContain('"delta":{"reasoning":"\\n"}')
    expect(sse.at(-2)!.line).toContain('"finish_reason":"stop","stop_reason":null')
  })
})

describe('Сценарий B: мысли и вызов инструмента (notes-tokens §3.11)', () => {
  const text =
    '<think>\nНужна погода, вызову инструмент.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Москва"}}\n</tool_call><|im_end|>'
  const chunks = chunksOf(text)
  const events = runStreamingParser(chunks, QWEN)

  it('имя уходит один раз, затем приращения arguments без внешней }', () => {
    const named = events.flatMap((e) => e.toolDeltas.filter((d) => d.name))
    expect(named).toHaveLength(1)
    expect(named[0]).toMatchObject({ index: 0, type: 'function', name: 'get_weather' })
    expect(named[0].id).toMatch(/^chatcmpl-tool-[0-9a-f]{8}$/)
    const pieces = events.flatMap((e) => e.toolDeltas.filter((d) => d.arguments !== undefined).map((d) => d.arguments))
    expect(pieces).toEqual(['{"', 'city', '":', ' "', 'М', 'ос', 'ква', '"}'])
    expect(toolArgs(events)).toBe('{"city": "Москва"}')
    // токен '"}}\n' закрывает строку и оба объекта одним куском
    expect(chunks).toContain('"}}\n')
  })

  it('на </tool_call> готовый ToolCall, content пустой', () => {
    const call = events.find((e) => e.toolCall)!
    expect(call.toolCall).toMatchObject({ name: 'get_weather', arguments: { city: 'Москва' } })
    expect(lanes(events).content).toBe('\n\n')
  })

  it('SSE: tool_calls и finish_reason "tool_calls"', () => {
    const sse = toSseChunks(events, META)
    const first = sse.find((c) => (c.json?.choices as any[])?.[0]?.delta?.tool_calls)!
    const tc = (first.json!.choices as any[])[0].delta.tool_calls[0]
    expect(tc).toMatchObject({ index: 0, type: 'function', function: { name: 'get_weather' } })
    expect(tc.function.arguments).toBeUndefined()
    expect((sse.at(-2)!.json!.choices as any[])[0].finish_reason).toBe('tool_calls')
  })
})

describe('частичные теги: любое место разреза даёт тот же результат', () => {
  const text = 'abc<think>xyz < 5</think>def<tool_call>{"name": "f", "arguments": {"a": "<b>"}}</tool_call>\n<tool_call>{"name": "g", "arguments": {}}</tool_call>tail'
  const whole = runStreamingParser([text], QWEN)
  const ref = lanes(whole)

  it('эталон: разбор цельной строки', () => {
    expect(ref.reasoning).toBe('xyz < 5')
    expect(ref.content).toBe('abcdeftail')
    expect(whole[0].toolCalls!.map((c) => c.name)).toEqual(["f", "g"]) // оба вызова закрылись в одном событии
    expect(toolArgs(whole, 0)).toBe('{"a": "<b>"}')
    expect(toolArgs(whole, 1)).toBe('{}')
  })

  for (let i = 1; i < text.length; i++) {
    it(`разрез на позиции ${i}`, () => {
      const ev = runStreamingParser([text.slice(0, i), text.slice(i)], QWEN)
      const l = lanes(ev)
      expect(l.reasoning).toBe(ref.reasoning)
      expect(l.content).toBe(ref.content)
      expect(toolArgs(ev, 0)).toBe('{"a": "<b>"}')
      expect(toolArgs(ev, 1)).toBe('{}')
      expect(ev.flatMap((e) => e.toolDeltas.filter((d) => d.name).map((d) => d.name))).toEqual(['f', 'g'])
    })
  }

  it('по одному символу', () => {
    const ev = runStreamingParser([...text], QWEN)
    expect(lanes(ev).content).toBe(ref.content)
    expect(lanes(ev).reasoning).toBe(ref.reasoning)
    const calls = ev.filter((e) => e.toolCall).map((e) => e.toolCall!.name)
    expect(calls).toEqual(['f', 'g'])
  })

  it("«<thi» придерживается, состояние maybe_tag", () => {
    const ev = runStreamingParser(['Привет <thi', 'nk>мысль'], QWEN)
    expect(ev[0].buffer).toBe('<thi')
    expect(ev[0].stateAfter).toBe('maybe_tag')
    expect(ev[0].channelAfter).toBe('content')
    expect(ev[0].deltas).toEqual([{ channel: 'content', text: 'Привет ' }])
    expect(ev[0].note).toContain('<thi')
    expect(ev[1].stateBefore).toBe('maybe_tag')
    expect(ev[1].stateAfter).toBe('reasoning')
    expect(ev[1].deltas).toEqual([{ channel: 'reasoning', text: 'мысль' }])
  })

  it('«<» оказался текстом', () => {
    const ev = runStreamingParser(['a <', ' b'], QWEN)
    expect(ev[0].buffer).toBe('<')
    expect(lanes(ev).content).toBe('a < b')
    expect(ev[1].note).toContain('оказалось обычным текстом')
  })

  it('конец потока на частичном теге: финальное событие отдаёт хвост', () => {
    const ev = runStreamingParser(['ответ <tool_'], QWEN)
    expect(ev).toHaveLength(2)
    expect(ev[1].step).toBe(1)
    expect(lanes(ev).content).toBe('ответ <tool_')
    expect(runStreamingParser(['ответ <tool_'], { ...QWEN, final: false })).toHaveLength(1)
  })
})

describe('семейства и крайние случаи', () => {
  it('DeepSeek-R1: forced reasoning, вывод начинается внутри рассуждения', () => {
    const ev = runStreamingParser(['Пользователь', ' спрашивает.', '\n', '</think>', '\n\n', 'Ответ'], { reasoning: 'deepseek-r1', tools: 'none' })
    expect(ev[0].stateBefore).toBe('reasoning')
    expect(ev[0].note).toContain('шаблон уже открыл <think>')
    expect(lanes(ev)).toMatchObject({ reasoning: 'Пользователь спрашивает.\n', content: '\n\nОтвет' })
  })

  it('Qwen3 с startInReasoning (Thinking-2507: <think> в промпте)', () => {
    const ev = runStreamingParser(['мысль', '</think>', 'ответ'], { reasoning: 'qwen3', tools: 'none', startInReasoning: true })
    expect(lanes(ev)).toMatchObject({ reasoning: 'мысль', content: 'ответ' })
  })

  it('reasoning=none: <think> — обычный текст', () => {
    const ev = runStreamingParser(['<think>x</think>'], { reasoning: 'none', tools: 'none' })
    expect(lanes(ev).content).toBe('<think>x</think>')
  })

  it('<tool_call> внутри рассуждения не разбирается', () => {
    const ev = runStreamingParser(['<think>может, <tool_call>?</think>нет'], QWEN)
    expect(lanes(ev).reasoning).toBe('может, <tool_call>?')
    expect(ev.some((e) => e.toolDeltas.length > 0)).toBe(false)
  })

  it('битый JSON без отправленных кусков → текст уходит в content', () => {
    const ev = runStreamingParser(['<tool_call>{oops}</tool_call>'], QWEN)
    expect(lanes(ev).content).toBe('<tool_call>{oops}</tool_call>')
    expect(ev[0].note).toContain('не разобрался')
    const sse = toSseChunks(ev, META)
    expect((sse.at(-2)!.json!.choices as any[])[0].finish_reason).toBe('stop')
  })

  it('вызов без </tool_call> в конце потока, но с валидным JSON', () => {
    const ev = runStreamingParser(['<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n'], QWEN)
    expect(ev.at(-1)!.toolCall).toMatchObject({ name: 'f', arguments: { x: 1 } })
  })

  it('finish_reason length', () => {
    const ev = runStreamingParser(['обрыв'], QWEN)
    expect((toSseChunks(ev, { ...META, finishReason: 'length' }).at(-2)!.json!.choices as any[])[0].finish_reason).toBe('length')
  })

  it('usage с cached_tokens', () => {
    const sse = toSseChunks([], { ...META, includeUsage: true, usage: { promptTokens: 10, completionTokens: 2, cachedTokens: 8 } })
    expect(sse.at(-2)!.json!.usage).toEqual({ prompt_tokens: 10, completion_tokens: 2, total_tokens: 12, prompt_tokens_details: { cached_tokens: 8 } })
  })
})

describe('помощники', () => {
  it('partialTagOverlap (как в vLLM: «<tool_» → 6)', () => {
    expect(partialTagOverlap('text <tool_', ['<tool_call>'])).toBe(6)
    expect(partialTagOverlap('text', ['<tool_call>'])).toBe(0)
    expect(partialTagOverlap('<think>', ['<think>'])).toBe(0) // полный тег — не частичный
  })

  it('streamableArguments', () => {
    expect(streamableArguments('{"name": "f", "argu')).toBeNull()
    expect(streamableArguments('{"name": "f", "arguments": ')).toBe('')
    expect(streamableArguments('{"name": "f", "arguments": {"a": "}"')).toBe('{"a": "}"')
    expect(streamableArguments('{"name": "f", "arguments": {"a": 1}}\n')).toBe('{"a": 1}')
    expect(streamableArguments('{"name": "f", "arguments": "x\\"y"}')).toBe('"x\\"y"')
  })
})

describe('harmonyToChannels (gpt-oss, упрощённо)', () => {
  it('analysis → reasoning, final → content (notes-tokens §3.9)', () => {
    const msgs = harmonyToChannels(
      '<|channel|>analysis<|message|>Simple arithmetic.<|end|><|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>',
    )
    expect(msgs).toEqual([
      { role: 'assistant', channel: 'analysis', content: 'Simple arithmetic.', terminator: '<|end|>', lane: 'reasoning' },
      { role: 'assistant', channel: 'final', content: '2 + 2 = 4.', terminator: '<|return|>', lane: 'content' },
    ])
  })

  it('вызов: получатель в заголовке роли (HF) или канала (спецификация)', () => {
    const hf = harmonyToChannels('<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city": "Москва"}<|call|>')
    expect(hf[0]).toMatchObject({ recipient: 'functions.get_weather', channel: 'commentary', contentType: 'json', lane: 'tool_call', terminator: '<|call|>' })
    const spec = harmonyToChannels(
      '<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>',
    )
    expect(spec[0]).toMatchObject({ recipient: 'functions.get_weather', contentType: 'json', lane: 'tool_call', content: '{"location":"San Francisco"}' })
  })
})
