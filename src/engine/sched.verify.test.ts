/**
 * Независимая проверка sched.ts (верификатор A2): детерминизм (тот же seed → тот же результат),
 * монотонность хвостов по нагрузке (для FCFS-очереди — строго, через «связанные» потоки: при том же
 * seed интервалы прихода масштабируются как 1 / rate), сохранение токенов (prefill-куски и decode-шаги
 * сходятся с промптом и ответом), бюджет и лимиты шага, граничные случаи.
 */
import { describe, expect, it } from 'vitest'
import { getGpu } from './data/gpus'
import { getModel } from './data/models'
import { getInterconnect } from './data/interconnects'
import { makeWorkload, simulateContinuous, simulateDisagg, simulateQueue, simulateStatic, type SimResultExt } from './sched'
import type { SimRequest } from './types'

const model = getModel('qwen3-8b')
const gpu = getGpu('h100-sxm')

/** Сколько токенов prefill и decode-шагов получил каждый запрос по записям шагов. */
function tally(res: SimResultExt) {
  const prefill = new Map<string, number>()
  const decode = new Map<string, number>()
  for (const s of res.steps) {
    for (const p of s.prefill) prefill.set(p.reqId, (prefill.get(p.reqId) ?? 0) + p.tokens)
    for (const id of s.decode) decode.set(id, (decode.get(id) ?? 0) + 1)
  }
  return { prefill, decode }
}

describe('makeWorkload', () => {
  it('durationMs при rate = 0 не зацикливается (все приходят в startMs, count по умолчанию 64)', () => {
    const w = makeWorkload({ rate: 0, promptDist: 100, outputDist: 10, durationMs: 1000, seed: 1 })
    expect(w).toHaveLength(64)
    expect(w.every((r) => r.arrivalMs === 0)).toBe(true)
    expect(makeWorkload({ rate: Infinity, promptDist: 100, outputDist: 10, durationMs: 1000, count: 5 })).toHaveLength(5)
  })
  it('связанные потоки: при том же seed длины те же, а моменты прихода масштабируются как 1 / rate', () => {
    const a = makeWorkload({ rate: 4, promptDist: { kind: 'lognormal', median: 900, sigma: 0.6 }, outputDist: { kind: 'uniform', min: 10, max: 90 }, count: 50, seed: 7 })
    const b = makeWorkload({ rate: 8, promptDist: { kind: 'lognormal', median: 900, sigma: 0.6 }, outputDist: { kind: 'uniform', min: 10, max: 90 }, count: 50, seed: 7 })
    a.forEach((r, i) => {
      expect(b[i].promptTokens).toBe(r.promptTokens)
      expect(b[i].outputTokens).toBe(r.outputTokens)
      expect(b[i].arrivalMs).toBeCloseTo(r.arrivalMs / 2, 9)
    })
  })
})

describe('детерминизм: тот же вход → тот же результат', () => {
  const w = makeWorkload({ rate: 12, promptDist: { kind: 'lognormal', median: 1500, sigma: 0.7, max: 12000 }, outputDist: { kind: 'uniform', min: 20, max: 300 }, count: 80, seed: 21, sharedPrefixes: { count: 4, tokens: 700 } })
  it.each([
    ['continuous', () => simulateContinuous(w, { model, gpu, tokenBudget: 2048, kvCapacityTokens: 60000 })],
    ['continuous ×3', () => simulateContinuous(w, { model, gpu, workers: 3 })],
    ['static', () => simulateStatic(w, { model, gpu, batchSize: 8 })],
    ['dynamic', () => simulateStatic(w, { model, gpu, batchSize: 8, maxWaitMs: 50 })],
    ['disagg', () => simulateDisagg(w, { model, gpu, prefillWorkers: 2, decodeWorkers: 3, link: getInterconnect('ib-ndr') })],
    ['queue', () => simulateQueue(w, { model, gpu, servers: 2 })],
  ])('%s', (_name, run) => {
    expect(run()).toEqual(run())
  })
})

describe('монотонность по нагрузке', () => {
  const at = (rate: number) =>
    makeWorkload({ rate, promptDist: { kind: 'lognormal', median: 1200, sigma: 0.6, max: 8000 }, outputDist: 60, count: 200, seed: 13 })
  it.each([1, 3])('FCFS-очередь (%i реплик): TTFT каждого запроса не убывает с ростом rate', (servers) => {
    let prev: number[] | null = null
    for (const rate of [2, 5, 10, 20, 40]) {
      const res = simulateQueue(at(rate), { model, gpu, servers })
      const ttft = res.requests.map((r) => r.ttftMs) // запросы в порядке прихода, у всех rate тот же порядок
      if (prev) ttft.forEach((x, i) => expect(x).toBeGreaterThanOrEqual(prev![i] - 1e-9))
      prev = ttft
    }
  })
  it('continuous batching: средний и p99 TTFT растут с нагрузкой, throughput не падает до насыщения', () => {
    const rows = [2, 8, 16, 32].map((rate) => simulateContinuous(at(rate), { model, gpu }).summary)
    for (let i = 1; i < rows.length; i++) {
      expect(rows[i].ttftMean).toBeGreaterThan(rows[i - 1].ttftMean)
      expect(rows[i].ttftP99).toBeGreaterThan(rows[i - 1].ttftP99)
      expect(rows[i].throughputTokS).toBeGreaterThan(rows[i - 1].throughputTokS * 0.95)
    }
  })
})

describe('сохранение токенов и лимиты шага', () => {
  const w = makeWorkload({ rate: 15, promptDist: { kind: 'uniform', min: 50, max: 5000 }, outputDist: { kind: 'uniform', min: 1, max: 120 }, count: 70, seed: 3, sharedPrefixes: { count: 3, tokens: 512 } })
  it('continuous без preemption: prefill = промпт − кэш, decode-шагов = ответ − 1, бюджет и max_num_seqs соблюдены', () => {
    const res = simulateContinuous(w, { model, gpu, tokenBudget: 1024, maxSeqs: 24 })
    expect(res.summary.preemptions).toBe(0)
    const { prefill, decode } = tally(res)
    for (const r of res.requests) {
      expect(prefill.get(r.id)).toBe(r.promptTokens - r.cachedTokens)
      expect(decode.get(r.id) ?? 0).toBe(r.outputTokens - 1)
      expect(r.itlMs).toHaveLength(r.outputTokens - 1)
      expect(r.queueMs).toBeGreaterThanOrEqual(0)
      expect(r.firstTokenMs).toBeGreaterThan(r.arrivalMs)
    }
    for (const s of res.steps) {
      expect(s.prefill.reduce((a, p) => a + p.tokens, 0) + s.decode.length).toBeLessThanOrEqual(1024)
      expect(new Set([...s.decode, ...s.prefill.map((p) => p.reqId)]).size).toBeLessThanOrEqual(24)
      expect(s.durationMs).toBeGreaterThan(0)
    }
  })
  it('continuous с preemption: все запросы завершаются, пересчёт виден как лишние prefill-токены', () => {
    const reqs: SimRequest[] = Array.from({ length: 10 }, (_, i) => ({ id: `p${i}`, arrivalMs: i, promptTokens: 300, outputTokens: 250 }))
    const res = simulateContinuous(reqs, { model, gpu, kvCapacityTokens: 2500 })
    expect(res.summary.preemptions).toBeGreaterThan(0)
    const { prefill } = tally(res)
    for (const r of res.requests) {
      expect(r.itlMs).toHaveLength(r.outputTokens - 1)
      expect(prefill.get(r.id)!).toBeGreaterThanOrEqual(r.promptTokens)
      if (r.preemptions === 0) expect(prefill.get(r.id)).toBe(r.promptTokens)
      else expect(prefill.get(r.id)!).toBeGreaterThan(r.promptTokens) // промпт + сгенерированное пересчитаны
    }
  })
  it('disagg: prefill-воркеры считают промпт целиком, decode-воркеры делают ответ (первый токен на decode)', () => {
    const res = simulateDisagg(w, { model, gpu, prefillWorkers: 2, decodeWorkers: 2, link: getInterconnect('roce-400') })
    const { prefill, decode } = tally(res)
    for (const r of res.requests) {
      expect(prefill.get(r.id)).toBe(r.promptTokens - r.cachedTokens)
      expect(decode.get(r.id)).toBe(r.outputTokens) // vLLM/Dynamo: первый токен сэмплирует decode
    }
    expect(res.transfers).toHaveLength(w.length)
  })
  it('disagg с первым токеном на prefill (SGLang): decode делает ответ − 1; ответ из 1 токена не едет на decode', () => {
    const res = simulateDisagg(w, { model, gpu, prefillWorkers: 2, decodeWorkers: 2, link: getInterconnect('roce-400'), firstTokenOn: 'prefill' })
    const { decode } = tally(res)
    const oneToken = res.requests.filter((r) => r.outputTokens === 1)
    for (const r of res.requests) expect(decode.get(r.id) ?? 0).toBe(r.outputTokens - 1)
    expect(res.transfers).toHaveLength(w.length - oneToken.length)
  })
  it('static: один prefill на batch, decode-шагов = самый длинный ответ − 1, batch-и не перекрываются', () => {
    const reqs: SimRequest[] = Array.from({ length: 12 }, (_, i) => ({ id: `s${i}`, arrivalMs: i * 10, promptTokens: 100 + 50 * i, outputTokens: 5 + ((i * 7) % 20) }))
    const res = simulateStatic(reqs, { model, gpu, batchSize: 4 })
    const prefillSteps = res.steps.filter((s) => s.prefill.length > 0)
    expect(prefillSteps).toHaveLength(3)
    for (let b = 0; b < 3; b++) {
      const batch = reqs.slice(b * 4, b * 4 + 4)
      const maxOut = Math.max(...batch.map((r) => r.outputTokens))
      const start = res.steps.indexOf(prefillSteps[b])
      const end = b < 2 ? res.steps.indexOf(prefillSteps[b + 1]) : res.steps.length
      expect(end - start - 1).toBe(maxOut - 1)
      const firsts = new Set(batch.map((r) => res.requests.find((x) => x.id === r.id)!.firstTokenMs))
      expect(firsts.size).toBe(1) // у всех в batch первый токен одновременно
    }
  })
  it('очередь: загрузка ≤ 1, TTFT = ожидание + обслуживание', () => {
    const res = simulateQueue(w, { model, gpu, servers: 2 })
    expect(res.utilization).toBeGreaterThan(0)
    expect(res.utilization).toBeLessThanOrEqual(1)
    for (const r of res.requests) expect(r.ttftMs).toBeGreaterThanOrEqual(r.queueMs)
  })
})

describe('ёмкость KV соблюдается и в agg, и на decode-воркерах disagg', () => {
  // Длинные ответы при коротких промптах: KV растёт уже после допуска, нужен recompute-preemption.
  const reqs: SimRequest[] = Array.from({ length: 40 }, (_, i) => ({ id: `g${i}`, arrivalMs: i * 20, promptTokens: 500, outputTokens: 2000 }))
  const cap = 12000
  const check = (res: SimResultExt, prefix: string) => {
    for (const s of res.steps) {
      if (!(s.worker ?? '').startsWith(prefix)) continue
      const participants = new Set([...s.decode, ...s.prefill.map((p) => p.reqId)]).size
      expect(s.kvTokens).toBeDefined()
      if (participants > 1) expect(s.kvTokens!).toBeLessThanOrEqual(cap)
    }
    for (const r of res.requests) expect(r.itlMs).toHaveLength(r.outputTokens - 1)
  }
  it('agg: пик KV ≤ ёмкости, есть preemption', () => {
    const res = simulateContinuous(reqs, { model, gpu, kvCapacityTokens: cap, tokenBudget: 100000 })
    check(res, 'gpu')
    expect(res.summary.preemptions).toBeGreaterThan(0)
  })
  it('disagg: decode-воркер тоже вытесняет и пересчитывает, а не держит KV сверх ёмкости', () => {
    const res = simulateDisagg(reqs, { model, gpu, prefillWorkers: 1, decodeWorkers: 1, link: 50, kvCapacityTokens: cap })
    check(res, 'decode')
    expect(res.summary.preemptions).toBeGreaterThan(0)
    // пересчёт на decode-воркере виден как prefill-кусок там
    expect(res.steps.some((s) => s.worker!.startsWith('decode') && s.prefill.length > 0)).toBe(true)
  })
})

describe('граничные случаи', () => {
  it('ответ из 1 токена, кэш больше промпта, промпт из 1 токена', () => {
    const reqs: SimRequest[] = [
      { id: 'a', arrivalMs: 0, promptTokens: 1, outputTokens: 1 },
      { id: 'b', arrivalMs: 0, promptTokens: 10, outputTokens: 1, cachedTokens: 50 },
      { id: 'c', arrivalMs: 5, promptTokens: 64, outputTokens: 3, cachedTokens: 64 },
    ]
    for (const res of [simulateContinuous(reqs, { model, gpu }), simulateDisagg(reqs, { model, gpu, prefillWorkers: 1, decodeWorkers: 1, link: 50 }), simulateQueue(reqs, { model, gpu }), simulateStatic(reqs, { model, gpu, batchSize: 2 })]) {
      expect(res.requests).toHaveLength(3)
      for (const r of res.requests) {
        expect(r.ttftMs).toBeGreaterThan(0) // даже при полном попадании в кэш считается последний токен
        expect(r.finishMs).toBeGreaterThanOrEqual(r.firstTokenMs)
        expect(r.itlMs).toHaveLength(r.outputTokens - 1)
        expect(r.cachedTokens).toBeLessThan(r.promptTokens)
      }
    }
  })
  it('пустой поток', () => {
    const res = simulateContinuous([], { model, gpu })
    expect(res.requests).toEqual([])
    expect(res.summary.throughputTokS).toBe(0)
    expect(simulateQueue([], { model, gpu }).utilization).toBe(0)
  })
  it('бюджет меньше числа декодирующих: никто не голодает навсегда', () => {
    const reqs: SimRequest[] = Array.from({ length: 6 }, (_, i) => ({ id: `t${i}`, arrivalMs: 0, promptTokens: 20, outputTokens: 30 }))
    const res = simulateContinuous(reqs, { model, gpu, tokenBudget: 4, maxSeqs: 6 })
    expect(res.requests.every((r) => r.finishMs > 0 && r.itlMs.length === 29)).toBe(true)
    for (const s of res.steps) expect(s.prefill.reduce((a, p) => a + p.tokens, 0) + s.decode.length).toBeLessThanOrEqual(4)
  })
})
