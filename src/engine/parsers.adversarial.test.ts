/**
 * Независимая проверка стримингового парсера и SSE:
 *  - результат не зависит от того, как текст порезан на куски (каждый офсет, по символу, случайные разрезы,
 *    по токенам Qwen3) — теги, разрезанные в любом месте, битый JSON вызова, текст после вызовов;
 *  - SSE-чанки по форме OpenAI chat.completion.chunk, клиентская склейка восстанавливает дорожки и аргументы.
 */
import { describe, expect, it } from 'vitest'
import { incrementalDetokenize } from './detok'
import { runStreamingParser, toSseChunks, toSseText, type ParserEventEx, type ParserOptionsEx } from './parsers'
import { mulberry32 } from './rng'
import { loadQwen3ForTests } from './tokenizer/testing'

const tok = loadQwen3ForTests()

interface Result {
  reasoning: string
  content: string
  calls: Array<{ id?: string; name: string; args: string }>
  closed: Array<{ name: string; arguments: Record<string, unknown> }>
}

function summarize(events: ParserEventEx[]): Result {
  const r: Result = { reasoning: '', content: '', calls: [], closed: [] }
  for (const e of events) {
    for (const it of e.out) {
      if (it.kind !== 'tool') {
        if (it.kind === 'reasoning') r.reasoning += it.text
        else r.content += it.text
      } else {
        const d = it.delta
        const slot = (r.calls[d.index] ??= { name: '', args: '' })
        if (d.id) slot.id = d.id
        if (d.name) slot.name += d.name
        if (d.arguments) slot.args += d.arguments
      }
    }
    for (const c of e.toolCalls ?? []) r.closed.push({ name: c.name, arguments: c.arguments })
    // deltas и out описывают одно и то же
    const lanes = { reasoning: '', content: '' }
    for (const d of e.deltas) if (d.channel !== 'tool_call') lanes[d.channel] += d.text
    const outLanes = { reasoning: '', content: '' }
    for (const it of e.out) if (it.kind !== 'tool') outLanes[it.kind] += it.text
    expect(outLanes).toEqual(lanes)
  }
  return r
}

const run = (chunks: string[], opts: ParserOptionsEx) => summarize(runStreamingParser(chunks, opts))

function chunkings(text: string, rng: () => number): string[][] {
  const cps = [...text]
  const out: string[][] = [[text], cps]
  for (let i = 1; i < cps.length; i++) out.push([cps.slice(0, i).join(''), cps.slice(i).join('')])
  for (let n = 0; n < 40; n++) {
    const cuts = new Set<number>()
    const k = 1 + Math.floor(rng() * 6)
    for (let j = 0; j < k; j++) cuts.add(1 + Math.floor(rng() * Math.max(1, cps.length - 1)))
    const sorted = [...cuts].sort((a, b) => a - b)
    const parts: string[] = []
    let prev = 0
    for (const c of sorted) {
      parts.push(cps.slice(prev, c).join(''))
      prev = c
    }
    parts.push(cps.slice(prev).join(''))
    out.push(parts.filter((p) => p !== ''))
  }
  // по токенам Qwen3, как отдаёт детокенизатор (спецтокены вроде <|im_end|> не попадают в текст)
  out.push(incrementalDetokenize(tok.encode(text), tok).map((s) => s.emitted))
  return out
}

const QWEN: ParserOptionsEx = { reasoning: 'qwen3', tools: 'hermes', idSeed: 's' }
const R1: ParserOptionsEx = { reasoning: 'deepseek-r1', tools: 'hermes', idSeed: 's' }

const CASES: Array<{ name: string; text: string; opts: ParserOptionsEx; expect?: Partial<Result> }> = [
  {
    name: 'мысли и ответ',
    text: '<think>\nМысль.\n</think>\n\nОтвет.',
    opts: QWEN,
    expect: { reasoning: '\nМысль.\n', content: '\n\nОтвет.' },
  },
  {
    name: 'мысли и вызов',
    text: '<think>a</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Москва"}}\n</tool_call>',
    opts: QWEN,
    expect: { reasoning: 'a', content: '\n\n', calls: [{ id: expect.any(String) as unknown as string, name: 'get_weather', args: '{"city": "Москва"}' }] },
  },
  {
    name: 'два вызова, пробелы между ними и текст после',
    text: '<tool_call>\n{"name": "a", "arguments": {"x": [1, {"y": "}"}]}}\n</tool_call>\n\n<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>\nГотово.\n',
    opts: QWEN,
  },
  {
    name: 'битый JSON (нет закрывающей скобки), имя успело уйти',
    text: '<tool_call>\n{"name": "f", "arguments": {"x": 1}\n</tool_call>',
    opts: QWEN,
    expect: { closed: [] },
  },
  {
    name: 'битый JSON без имени: текст возвращается в content',
    text: 'до <tool_call>не json</tool_call> после',
    opts: QWEN,
    expect: { content: 'до <tool_call>не json</tool_call> после', calls: [], closed: [] },
  },
  {
    name: 'поток оборвался внутри аргументов',
    text: '<tool_call>{"name": "f", "arguments": {"x": "Мос',
    opts: QWEN,
  },
  {
    name: 'поток оборвался после валидного JSON без </tool_call>',
    text: '<tool_call>{"name": "f", "arguments": {"q": 1.5e3, "ok": true, "n": null}}',
    opts: QWEN,
    expect: { closed: [{ name: 'f', arguments: { q: 1500, ok: true, n: null } }] },
  },
  { name: 'хвост — начало тега <thi', text: 'текст <thi', opts: QWEN, expect: { content: 'текст <thi' } },
  { name: 'хвост — начало тега <tool_', text: 'текст <tool_', opts: QWEN, expect: { content: 'текст <tool_' } },
  { name: 'хвост — начало </think>', text: '<think>abc</thi', opts: QWEN, expect: { reasoning: 'abc</thi', content: '' } },
  {
    name: 'похожее на теги, но не теги',
    text: 'a <thinking> b <tool_calls> c < think> <<think>> </tool_call>',
    opts: QWEN,
  },
  { name: 'forced reasoning (R1)', text: 'мысли</think>\n\nответ', opts: R1, expect: { reasoning: 'мысли', content: '\n\nответ' } },
  {
    name: 'forced reasoning: модель всё же печатает <think> после перевода строки',
    text: '\n<think>мысли</think>ответ',
    opts: R1,
    expect: { reasoning: '\nмысли', content: 'ответ' },
  },
  { name: 'forced reasoning: <think> в самом начале', text: '<think>\nмысли</think>ответ', opts: R1, expect: { reasoning: '\nмысли', content: 'ответ' } },
  {
    name: 'экранированное имя и скобки в строках аргументов',
    text: '<tool_call>{"name": "a\\"b", "arguments": {"s": "}{\\"[", "t": "\\\\"}}</tool_call>',
    opts: QWEN,
    expect: { closed: [{ name: 'a"b', arguments: { s: '}{"[', t: '\\' } }] },
  },
  {
    name: 'arguments раньше name',
    text: '<tool_call>{"arguments": {"city": "Казань"}, "name": "get_weather"}</tool_call>',
    opts: QWEN,
    expect: { closed: [{ name: 'get_weather', arguments: { city: 'Казань' } }] },
  },
  { name: 'reasoning выключен: <think> — обычный текст', text: '<think>x</think>y', opts: { reasoning: 'none', tools: 'hermes' }, expect: { content: '<think>x</think>y' } },
  {
    name: 'tools выключены: <tool_call> — обычный текст',
    text: 'a<tool_call>{"name": "f"}</tool_call>',
    opts: { reasoning: 'qwen3', tools: 'none' },
    expect: { content: 'a<tool_call>{"name": "f"}</tool_call>' },
  },
  {
    name: 'текст, мысли посреди content, снова текст',
    text: 'Привет<think>скрытое</think> мир',
    opts: QWEN,
    expect: { reasoning: 'скрытое', content: 'Привет мир' },
  },
  {
    name: 'вызов внутри рассуждения не ищется',
    text: '<think>может <tool_call>{"name": "x"}</tool_call>?</think>нет',
    opts: QWEN,
    expect: { reasoning: 'может <tool_call>{"name": "x"}</tool_call>?', content: 'нет', calls: [] },
  },
]

describe('стриминговый парсер: результат не зависит от разбиения на куски', () => {
  const rng = mulberry32(42)
  for (const c of CASES) {
    it(c.name, () => {
      const ref = run([c.text], c.opts)
      if (c.expect) expect(ref).toMatchObject(c.expect)
      for (const chunks of chunkings(c.text, rng)) {
        const got = run(chunks, c.opts)
        expect(got, JSON.stringify(chunks)).toEqual(ref)
      }
    })
  }

  it('буфер придерживает только строгий префикс тега и отдаёт его, как только ясно, что это текст', () => {
    const ev = runStreamingParser(['a <', 'thi', 'ng'], QWEN)
    expect(ev.map((e) => e.buffer)).toEqual(['<', '<thi', ''])
    expect(ev.map((e) => e.stateAfter)).toEqual(['maybe_tag', 'maybe_tag', 'content'])
    expect(summarize(ev).content).toBe('a <thing')
  })
})

describe('SSE: форма chat.completion.chunk и клиентская склейка (OpenAI streaming reference, notes-tokens §1.13)', () => {
  const text =
    '<think>\nНужна погода.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Москва"}}\n</tool_call>\n<tool_call>\n{"name": "get_time", "arguments": {"tz": "MSK"}}\n</tool_call>'
  const chunks = incrementalDetokenize(tok.encode(text), tok).map((s) => s.emitted)
  const events = runStreamingParser(chunks, QWEN)
  const meta = { id: 'chatcmpl-x', created: 1762950000, model: 'Qwen/Qwen3-8B', includeUsage: true, usage: { promptTokens: 10, completionTokens: 5 } }
  const sse = toSseChunks(events, meta)

  it('каждая строка — «data: JSON», одинаковые id/created/model/object, в конце [DONE]', () => {
    expect(sse.at(-1)).toEqual({ line: 'data: [DONE]', json: null })
    for (const c of sse.slice(0, -1)) {
      expect(c.line).toBe('data: ' + JSON.stringify(c.json))
      expect(JSON.parse(c.line.slice(6))).toEqual(c.json)
      expect(c.json).toMatchObject({ id: 'chatcmpl-x', object: 'chat.completion.chunk', created: 1762950000, model: 'Qwen/Qwen3-8B' })
      expect(Array.isArray(c.json!.choices)).toBe(true)
    }
    expect(toSseText(sse).split('\n\n').filter(Boolean)).toHaveLength(sse.length)
  })

  it('choices[0]: index 0, delta, logprobs null, finish_reason null до финала; первый чанк — роль', () => {
    const body = sse.slice(0, -2)
    expect((body[0].json!.choices as any[])[0]).toEqual({ index: 0, delta: { role: 'assistant', content: '' }, logprobs: null, finish_reason: null })
    for (const c of body.slice(1, -1)) {
      const ch = (c.json!.choices as any[])[0]
      expect(Object.keys(ch).sort()).toEqual(['delta', 'finish_reason', 'index', 'logprobs'])
      expect(ch.finish_reason).toBeNull()
      const keys = Object.keys(ch.delta)
      expect(keys.length).toBe(1) // ровно одно поле дельты в чанке
      expect(['reasoning_content', 'content', 'tool_calls']).toContain(keys[0])
    }
    const fin = (body.at(-1)!.json!.choices as any[])[0]
    expect(fin).toEqual({ index: 0, delta: {}, logprobs: null, finish_reason: 'tool_calls' })
    expect(sse.at(-2)!.json).toMatchObject({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } })
  })

  it('tool_calls: первый фрагмент с index/id/type/name, дальше только index + function.arguments', () => {
    const tcs = sse.flatMap((c) => ((c.json?.choices as any[]) ?? []).flatMap((ch) => ch.delta.tool_calls ?? []))
    const byIndex = new Map<number, any[]>()
    for (const t of tcs) byIndex.set(t.index, [...(byIndex.get(t.index) ?? []), t])
    expect([...byIndex.keys()]).toEqual([0, 1])
    for (const [, list] of byIndex) {
      expect(list[0]).toMatchObject({ type: 'function', function: { name: expect.any(String) } })
      expect(list[0].id).toMatch(/^chatcmpl-tool-/)
      for (const t of list.slice(1)) {
        expect(Object.keys(t).sort()).toEqual(['function', 'index'])
        expect(Object.keys(t.function)).toEqual(['arguments'])
      }
    }
  })

  it('клиент склеивает поток (псевдокод из notes-tokens §3.11) и получает исходные дорожки и валидные аргументы', () => {
    let reasoning = ''
    let content = ''
    const calls: Record<number, { id: string | null; name: string; arguments: string }> = {}
    let finish: string | null = null
    for (const c of sse) {
      if (c.line === 'data: [DONE]') break
      const ch = c.json!
      if (!(ch.choices as any[]).length) continue
      const d = (ch.choices as any[])[0].delta
      reasoning += d.reasoning ?? d.reasoning_content ?? ''
      content += d.content ?? ''
      for (const tc of d.tool_calls ?? []) {
        const slot = (calls[tc.index] ??= { id: null, name: '', arguments: '' })
        slot.id = tc.id ?? slot.id
        slot.name += tc.function?.name ?? ''
        slot.arguments += tc.function?.arguments ?? ''
      }
      finish = (ch.choices as any[])[0].finish_reason ?? finish
    }
    expect(reasoning).toBe('\nНужна погода.\n')
    expect(content).toBe('\n\n')
    expect(Object.values(calls).map((c) => [c.name, JSON.parse(c.arguments)])).toEqual([
      ['get_weather', { city: 'Москва' }],
      ['get_time', { tz: 'MSK' }],
    ])
    expect(finish).toBe('tool_calls')
  })
})

describe('фузз: случайные склейки тегов, JSON и текста, случайные разрезы', () => {
  it('3000 текстов × 6 разбиений × 4 конфигурации парсеров: результат одинаков', () => {
    const rng = mulberry32(5)
    const FR = ['<think>', '</think>', '<tool_call>', '</tool_call>', 'text', ' ', '\n', '{"name": "f", "arguments": {"a": 1}}', '{"name": "g"', '}', '<thi', '<tool_', '"', '\\', 'Я🙂', '<', '>', '{"arguments": {"b": [1]}, "name": "h"}', '\n\n', 'nk>']
    const OPTS: ParserOptionsEx[] = [
      { reasoning: 'qwen3', tools: 'hermes' },
      { reasoning: 'deepseek-r1', tools: 'hermes' },
      { reasoning: 'none', tools: 'hermes' },
      { reasoning: 'qwen3', tools: 'none' },
    ]
    for (let t = 0; t < 3000; t++) {
      let text = ''
      for (let i = 0, k = 1 + Math.floor(rng() * 10); i < k; i++) text += FR[Math.floor(rng() * FR.length)]
      const opts = OPTS[Math.floor(rng() * OPTS.length)]
      const ref = run([text], opts)
      const cps = [...text]
      const chunkings: string[][] = [cps]
      for (let j = 0; j < 5; j++) {
        const parts: string[] = []
        let cur = ''
        for (const ch of cps) {
          cur += ch
          if (rng() < 0.3) {
            parts.push(cur)
            cur = ''
          }
        }
        if (cur) parts.push(cur)
        chunkings.push(parts)
      }
      for (const ch of chunkings) expect(run(ch, opts), JSON.stringify({ opts, ch })).toEqual(ref)
    }
  })
})
