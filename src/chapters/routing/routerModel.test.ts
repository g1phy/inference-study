import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { LONG_DOC_SYSTEM, WEATHER_TOOL } from '@/engine/scenarios'
import { AGENT_SYSTEM, agentSwitchAt, commit, costTexFor, decide, initialWorkers, nextRequest, readerRequest, summarize, texNum } from './routerModel'

const tok = loadQwen3ForTests()
const base = { systemText: 'Ты полезный ассистент.', userText: 'Объясни простыми словами, что такое KV cache и зачем он нужен.', thinking: true }

describe('сценарии главной фигуры роутинга', () => {
  it('пример документации: стоимости 18 / 10 / 11, выбран W2', () => {
    const ws = initialWorkers('docs', base, tok)
    const d = decide('kv', readerRequest('docs', base, tok), ws, { overlapCredit: 1, temperature: 0 })
    expect(d.scores.map((s) => s.cost)).toEqual([18, 10, 11])
    expect(d.chosen).toBe('worker-1')
  })

  it('агент: тёплый Воркер 1 выигрывает, при c = 0 решает нагрузка (пустой Воркер 4)', () => {
    const ws = initialWorkers('agent', base, tok)
    const req = readerRequest('agent', base, tok)
    expect(req.isl).toBeGreaterThan(400)
    const d = decide('kv', req, ws, { overlapCredit: 1, temperature: 0 })
    expect(d.chosen).toBe('worker-0')
    expect(d.scores[0].overlapBlocks).toBeGreaterThan(20)
    expect(d.scores[3].overlapBlocks).toBe(0)
    expect(decide('kv', req, ws, { overlapCredit: 0, temperature: 0 }).chosen).toBe('worker-3')
  })

  it('поток запросов агента: тёплый воркер набирает нагрузку, и роутер уходит на другой', () => {
    let ws = initialWorkers('agent', base, tok)
    let req = readerRequest('agent', base, tok)
    const chosen: string[] = []
    for (let n = 0; n < 20; n++) {
      const d = decide('kv', req, ws, { overlapCredit: 1, temperature: 0 })
      chosen.push(d.chosen)
      ws = commit(ws, d, req)
      req = nextRequest('agent', n + 1, base, tok)
    }
    expect(chosen[0]).toBe('worker-0')
    expect(new Set(chosen).size).toBeGreaterThan(1)
    // при сообщении по умолчанию смена на четвёртом запросе; подпись фигуры берёт номер из agentSwitchAt
    expect(chosen.indexOf('worker-3')).toBe(3)
    expect(agentSwitchAt(base, tok)).toBe(4)
    expect(ws[3].role).toContain('был пустым')
    expect(ws[0].activeBlocks).toBeGreaterThan(67)
    // prefill-токены сняты после каждого запроса, очередь Воркера 2 осталась исходной
    expect(ws[0].queuedPrefillTokens).toBe(0)
    expect(ws[1].queuedPrefillTokens).toBe(1024)
  })

  it('у агента свой системный промпт: «Длинный документ» в симуляторе не раздувает префикс и не отменяет смену воркера', () => {
    const longDoc = { systemText: LONG_DOC_SYSTEM, userText: 'По регламенту: как безопасно перезапустить узел?', thinking: false }
    const a = readerRequest('agent', base, tok)
    const b = readerRequest('agent', longDoc, tok)
    expect(b.isl).toBeLessThan(700)
    // общее начало (системный промпт агента и tools) одинаковое при любом system prompt читателя
    expect(b.blocks.slice(0, 30).map((x) => x.hash)).toEqual(a.blocks.slice(0, 30).map((x) => x.hash))
    expect(tok.decode(b.blocks[0].tokenIds) + tok.decode(b.blocks[1].tokenIds)).toContain(AGENT_SYSTEM.slice(0, 12))
    for (const parts of [longDoc, { ...base, systemText: '' }, { ...base, thinking: false, tools: [WEATHER_TOOL] }]) {
      const n = agentSwitchAt(parts, tok)
      expect(n).toBeGreaterThanOrEqual(3)
      expect(n).toBeLessThanOrEqual(5)
    }
    // длинное сообщение читателя сразу добавляет Воркеру 1 много активных блоков: смена раньше, но не на первом запросе
    const long = agentSwitchAt({ ...base, userText: 'Расскажи подробно '.repeat(30) }, tok)
    expect(long).toBeGreaterThan(1)
    expect(long).toBeLessThan(4)
  })

  it('«Ваш запрос»: короткий промпт уходит на пустой Воркер 4, длинный остаётся у Воркера 1 ради кэша', () => {
    const route = (parts: typeof base & { tools?: (typeof WEATHER_TOOL)[]; systemText: string }) =>
      decide('kv', readerRequest('plain', parts, tok), initialWorkers('plain', parts, tok), { overlapCredit: 1, temperature: 0 }).chosen
    expect(route(base)).toBe('worker-3')
    expect(route({ ...base, userText: 'Сколько будет 2+2?', thinking: false })).toBe('worker-3')
    expect(route({ ...base, userText: 'Извлеки данные: Анна, 29 лет, живёт в Казани', thinking: false })).toBe('worker-3')
    expect(route({ ...base, userText: 'Какая погода сейчас в Берлине?', thinking: false, tools: [WEATHER_TOOL] })).toBe('worker-0')
    expect(route({ systemText: LONG_DOC_SYSTEM, userText: 'По регламенту: как безопасно перезапустить узел?', thinking: false })).toBe('worker-0')
  })

  it('round-robin идёт по кругу, KV-aware даёт больше токенов из кэша на том же потоке', () => {
    const run = (strategy: 'kv' | 'round-robin') => {
      let ws = initialWorkers('agent', base, tok)
      let req = readerRequest('agent', base, tok)
      const sent = []
      for (let n = 0; n < 12; n++) {
        const d = decide(strategy, req, ws, { overlapCredit: 1, temperature: 0 })
        const s = d.scores.find((x) => x.workerId === d.chosen)!
        sent.push({ n, label: req.label, worker: d.chosen, workerName: s.name, isl: req.isl, cachedTokens: s.cachedTokens, strategy })
        ws = commit(ws, d, req)
        req = nextRequest('agent', n + 1, base, tok)
      }
      return sent
    }
    const rr = run('round-robin')
    expect(rr.slice(0, 4).map((r) => r.worker)).toEqual(['worker-0', 'worker-1', 'worker-2', 'worker-3'])
    expect(summarize(run('kv')).hitRate).toBeGreaterThan(summarize(rr).hitRate)
  })

  it('формула с числами: десятичная запятая в TeX и итог совпадает со стоимостью', () => {
    expect(texNum(2.4375)).toBe('2{,}44')
    expect(texNum(58)).toBe('58')
    const ws = initialWorkers('docs', base, tok)
    const d = decide('kv', readerRequest('docs', base, tok), ws, { overlapCredit: 1, temperature: 0 })
    expect(costTexFor(d.scores[1], { overlapCredit: 1, temperature: 0 }, d.isl)).toContain('= 5 + 5 = 10')
  })
})
