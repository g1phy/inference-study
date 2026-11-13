import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { runStreamingParser, toSseChunks } from '@/engine/parsers'
import { FAMILY_CASES, FAMILY_ORDER, SCENARIOS, buildStream, compactLine, lanesAt, parseFamilyOutput, promptTokens, splitByChars, splitSpecial, sseKind } from './scenarios'

const tk = loadQwen3ForTests()

describe('поток для парсера', () => {
  it('по токенам и по 3 символа дают одинаковые поля', () => {
    for (const sc of SCENARIOS) {
      const a = buildStream(tk, sc.text, 'tokens')
      const b = buildStream(tk, sc.text, 'chars')
      expect(a.eos?.id).toBe(151645)
      expect(a.tokens?.length).toBe(a.chunks.length)
      const ea = runStreamingParser(a.chunks, { reasoning: 'qwen3', tools: 'hermes' })
      const eb = runStreamingParser(b.chunks, { reasoning: 'qwen3', tools: 'hermes' })
      const la = lanesAt(ea, ea.length - 1)
      const lb = lanesAt(eb, eb.length - 1)
      expect(lb.reasoning).toBe(la.reasoning)
      expect(lb.content).toBe(la.content)
      expect(lb.toolName).toBe(la.toolName)
      expect(lb.toolArgs).toBe(la.toolArgs)
    }
  })

  it('кусками по 3 символа «<th» попадает в буфер', () => {
    const b = buildStream(tk, SCENARIOS[0].text, 'chars')
    expect(b.chunks[0]).toBe('<th')
    const ev = runStreamingParser(b.chunks, { reasoning: 'qwen3', tools: 'hermes' })
    expect(ev[0].buffer).toBe('<th')
    expect(ev[0].stateAfter).toBe('maybe_tag')
  })

  it('сценарий с вызовом: имя и аргументы собираются, finish_reason tool_calls', () => {
    const s = buildStream(tk, SCENARIOS[1].text, 'tokens')
    const ev = runStreamingParser(s.chunks, { reasoning: 'qwen3', tools: 'hermes' })
    const l = lanesAt(ev, ev.length - 1)
    expect(l.toolName).toBe('kv_cache_size')
    expect(JSON.parse(l.toolArgs)).toEqual({ context: 8192 })
    expect(l.reasoning.trim()).toBe('Нужен точный расчёт.')
    const sse = toSseChunks(ev, { id: 'x', created: 1, model: 'm', includeUsage: true, usage: { promptTokens: 1, completionTokens: 2 } })
    const kinds = sse.map((c) => sseKind(c.json))
    expect(kinds[0]).toBe('role')
    expect(kinds.at(-1)).toBe('done')
    expect(kinds.at(-2)).toBe('usage')
    expect(kinds.at(-3)).toBe('finish')
    expect(kinds).toContain('tool-head')
    expect(kinds).toContain('tool-args')
    const finish = sse.at(-3)!.json as { choices: Array<{ finish_reason: string }> }
    expect(finish.choices[0].finish_reason).toBe('tool_calls')
  })

  it('compactLine сворачивает служебные поля', () => {
    expect(compactLine(null)).toBe('data: [DONE]')
    const line = compactLine({ id: 'x', object: 'chat.completion.chunk', created: 1, model: 'm', choices: [{ index: 0, delta: { content: 'a' }, logprobs: null, finish_reason: null }] })
    expect(line).toBe('data: {…,"choices":[{"delta":{"content":"a"},…}]}')
  })

  it('splitByChars режет по code point', () => {
    expect(splitByChars('аб🚀вг', 2)).toEqual(['аб', '🚀в', 'г'])
  })
})

describe('семейства моделей', () => {
  it('каждое семейство разбирается в ожидаемые поля', () => {
    for (const f of FAMILY_ORDER) {
      const answer = parseFamilyOutput(f, FAMILY_CASES[f].raw.answer)
      expect(answer.content).toContain('KV cache хранит ключи и значения')
      expect(answer.toolCalls).toHaveLength(0)
      const tool = parseFamilyOutput(f, FAMILY_CASES[f].raw.tool)
      expect(tool.toolCalls).toHaveLength(1)
      expect(tool.toolCalls[0].name).toBe('kv_cache_size')
      expect(JSON.parse(tool.toolCalls[0].arguments)).toEqual({ context: 8192 })
      if (f !== 'llama3') expect(answer.reasoning.trim().length).toBeGreaterThan(0)
      else expect(answer.reasoning).toBe('')
    }
  })

  it('DeepSeek-R1: рассуждение без открывающего тега', () => {
    const r = parseFamilyOutput('deepseek-r1', FAMILY_CASES['deepseek-r1'].raw.answer)
    expect(r.reasoning.trim()).toBe('Нужна аналогия.')
    expect(r.content).not.toContain('</think>')
  })

  it('DeepSeek-R1-0528: вызов инструмента, открывающий <think> модель печатает сама', () => {
    const c = FAMILY_CASES['deepseek-r1']
    expect(c.raw.tool.startsWith('<think>\n')).toBe(true)
    const r = parseFamilyOutput('deepseek-r1', c.raw.tool)
    expect(r.reasoning.trim()).toBe('Нужен расчёт.')
    expect(r.reasoning).not.toContain('<think>')
    expect(r.content).toBe('')
    expect(c.toolFlags?.vllm).toContain('--chat-template')
  })

  it('splitSpecial выделяет литералы спецтокенов', () => {
    const parts = splitSpecial('<|im_start|>user<|im_end|>', ['<|im_start|>', '<|im_end|>'])
    expect(parts).toEqual([
      { text: '<|im_start|>', special: true },
      { text: 'user', special: false },
      { text: '<|im_end|>', special: true },
    ])
  })
})

describe('usage и флаги', () => {
  it('prompt_tokens по шаблону Qwen3: описание kv_cache_size добавляет к промпту больше сотни токенов', () => {
    const a = promptTokens(tk, 'answer')
    const t = promptTokens(tk, 'tool')
    expect(a.tools).toBe(0)
    expect(a.total).toBeGreaterThan(10)
    expect(t.tools).toBeGreaterThan(100)
    expect(t.total).toBeGreaterThan(t.tools)
  })

  it('DeepSeek-R1: в варианте «Ответ» только парсер рассуждений, deepseek_v3 только у R1-0528', () => {
    const c = FAMILY_CASES['deepseek-r1']
    expect(c.answerFlags?.vllm).toBe('--reasoning-parser deepseek_r1')
    expect(c.answerFlags?.dynamo).toBe('--dyn-reasoning-parser deepseek_r1')
    expect(c.toolFlags?.vllm).toContain('--tool-call-parser deepseek_v3')
    expect(c.toolFlags?.dynamo).toContain('--dyn-tool-call-parser deepseek_v3')
  })
})
