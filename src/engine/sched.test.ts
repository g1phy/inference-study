import { describe, expect, it } from 'vitest'
import { getGpu } from './data/gpus'
import { getModel } from './data/models'
import { getInterconnect } from './data/interconnects'
import { kvBytesForSeq } from './perf'
import { mulberry32 } from './rng'
import {
  goodput,
  histogram,
  makeWorkload,
  mean,
  percentile,
  quantile,
  sampleDist,
  simulateContinuous,
  simulateDisagg,
  simulateQueue,
  simulateStatic,
  summarize,
  vllmDefaultLimits,
  type SimResultExt,
} from './sched'
import type { SimRequest } from './types'

const model = getModel('qwen3-8b')
const gpu = getGpu('h100-sxm')

/** Общие инварианты любого результата симуляции. */
function checkInvariants(res: SimResultExt, reqs: SimRequest[]) {
  expect(res.requests).toHaveLength(reqs.length)
  for (const r of res.requests) {
    expect(r.firstTokenMs).toBeGreaterThanOrEqual(r.arrivalMs)
    expect(r.finishMs).toBeGreaterThanOrEqual(r.firstTokenMs)
    expect(r.ttftMs).toBeCloseTo(r.firstTokenMs - r.arrivalMs, 9)
    expect(r.itlMs).toHaveLength(r.outputTokens - 1)
    expect(r.itlMs.every((x) => x > 0)).toBe(true)
    const sumItl = r.itlMs.reduce((a, b) => a + b, 0)
    expect(r.firstTokenMs + sumItl).toBeCloseTo(r.finishMs, 6)
  }
  // шаги одного воркера не перекрываются
  const byWorker = new Map<string, typeof res.steps>()
  for (const s of res.steps) {
    const list = byWorker.get(s.worker ?? '') ?? []
    list.push(s)
    byWorker.set(s.worker ?? '', list)
  }
  for (const list of byWorker.values()) {
    for (let i = 1; i < list.length; i++) expect(list[i].tMs).toBeGreaterThanOrEqual(list[i - 1].tMs + list[i - 1].durationMs - 1e-6)
  }
}

describe('статистика', () => {
  // notes-foundations-production.md §3.3: 20 замеров TTFT
  const ttft = [180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 240, 250, 265, 280, 300, 340, 420, 900, 2400]
  it('перцентили: линейная интерполяция и nearest-rank', () => {
    expect(mean(ttft)).toBeCloseTo(382.5, 6)
    expect(percentile(ttft, 50)).toBeCloseTo(227.5, 6)
    expect(percentile(ttft, 90)).toBeCloseTo(468, 6)
    expect(percentile(ttft, 95)).toBeCloseTo(975, 6)
    expect(percentile(ttft, 99)).toBeCloseTo(2115, 6)
    expect(percentile(ttft, 90, 'nearest-rank')).toBe(420)
    expect(percentile(ttft, 95, 'nearest-rank')).toBe(900)
    expect(percentile(ttft, 99, 'nearest-rank')).toBe(2400)
    expect(quantile([], 0.5)).toBe(0)
    expect(quantile([7], 0.99)).toBe(7)
  })
  it('гистограмма покрывает все значения', () => {
    const h = histogram(ttft, { bins: 10 })
    expect(h).toHaveLength(10)
    expect(h.reduce((a, b) => a + b.count, 0)).toBe(20)
    expect(h[0].x0).toBe(180)
    expect(h[9].x1).toBeCloseTo(2400, 6)
  })
  it('распределения длин: целые ≥ 1 и в границах', () => {
    const rng = mulberry32(5)
    for (let i = 0; i < 200; i++) {
      const u = sampleDist({ kind: 'uniform', min: 10, max: 20 }, rng)
      expect(u).toBeGreaterThanOrEqual(10)
      expect(u).toBeLessThanOrEqual(20)
      const l = sampleDist({ kind: 'lognormal', median: 100, sigma: 1, max: 300 }, rng)
      expect(l).toBeGreaterThanOrEqual(1)
      expect(l).toBeLessThanOrEqual(300)
      expect([1, 5]).toContain(sampleDist({ kind: 'choice', values: [1, 5], weights: [1, 1] }, rng))
    }
    expect(sampleDist(42, rng)).toBe(42)
    expect(sampleDist(0, rng)).toBe(1)
  })
  it('дефолтные лимиты vLLM зависят от GPU', () => {
    expect(vllmDefaultLimits(getGpu('h100-sxm'))).toEqual({ tokenBudget: 8192, maxSeqs: 1024 })
    expect(vllmDefaultLimits(getGpu('b200'))).toEqual({ tokenBudget: 16384, maxSeqs: 1024 })
    expect(vllmDefaultLimits(getGpu('a100-80g'))).toEqual({ tokenBudget: 2048, maxSeqs: 256 })
    expect(vllmDefaultLimits(getGpu('l40s'))).toEqual({ tokenBudget: 2048, maxSeqs: 256 })
  })
})

describe('makeWorkload', () => {
  const opts = { rate: 10, promptDist: { kind: 'uniform', min: 100, max: 300 }, outputDist: 50, count: 40, seed: 9 } as const
  it('детерминирован по seed и упорядочен по времени', () => {
    const a = makeWorkload(opts)
    const b = makeWorkload(opts)
    expect(a).toEqual(b)
    expect(a).toHaveLength(40)
    for (let i = 1; i < a.length; i++) expect(a[i].arrivalMs).toBeGreaterThanOrEqual(a[i - 1].arrivalMs)
    expect(makeWorkload({ ...opts, seed: 10 })).not.toEqual(a)
  })
  it('интенсивность: в среднем rate запросов в секунду', () => {
    const w = makeWorkload({ ...opts, count: 2000 })
    const perSec = (w.length - 1) / ((w[w.length - 1].arrivalMs - w[0].arrivalMs) / 1000)
    expect(perSec).toBeGreaterThan(9)
    expect(perSec).toBeLessThan(11)
  })
  it('rate = 0: все приходят сразу; durationMs ограничивает поток', () => {
    expect(makeWorkload({ ...opts, rate: 0 }).every((r) => r.arrivalMs === 0)).toBe(true)
    const w = makeWorkload({ ...opts, count: undefined, durationMs: 2000 })
    expect(w.every((r) => r.arrivalMs < 2000)).toBe(true)
    expect(w.length).toBeGreaterThan(5)
  })
  it('общие префиксы: повторная встреча префикса даёт cachedTokens (целые блоки)', () => {
    const w = makeWorkload({ ...opts, sharedPrefixes: { count: 3, tokens: 500 } })
    const cached = w.filter((r) => (r.cachedTokens ?? 0) > 0)
    expect(cached.length).toBeGreaterThan(30)
    expect(cached.every((r) => r.cachedTokens === 496)).toBe(true) // floor(500 / 16) · 16
    expect(w.every((r) => r.promptTokens >= 600)).toBe(true)
  })
})

describe('continuous batching (vLLM V1)', () => {
  it('рост нагрузки → растёт хвост TTFT (p99)', () => {
    const run = (rate: number) => {
      const w = makeWorkload({ rate, promptDist: { kind: 'lognormal', median: 1000, sigma: 0.5, max: 8000 }, outputDist: { kind: 'uniform', min: 50, max: 250 }, count: 150, seed: 11 })
      const res = simulateContinuous(w, { model, gpu })
      checkInvariants(res, w)
      return res.summary
    }
    const low = run(2)
    const mid = run(10)
    const high = run(30)
    expect(mid.ttftP99).toBeGreaterThan(low.ttftP99)
    expect(high.ttftP99).toBeGreaterThan(mid.ttftP99 * 2)
    expect(high.throughputTokS).toBeGreaterThan(low.throughputTokS)
  })
  it('бюджет токенов соблюдается в каждом шаге; decode идёт первым', () => {
    const w = makeWorkload({ rate: 20, promptDist: { kind: 'uniform', min: 200, max: 3000 }, outputDist: 64, count: 60, seed: 4 })
    const res = simulateContinuous(w, { model, gpu, tokenBudget: 1024, maxSeqs: 32 })
    checkInvariants(res, w)
    for (const s of res.steps) {
      const tokens = s.prefill.reduce((a, p) => a + p.tokens, 0) + s.decode.length
      expect(tokens).toBeLessThanOrEqual(1024)
      expect(s.decode.length + s.prefill.length).toBeLessThanOrEqual(32)
    }
    // хотя бы один смешанный шаг: prefill-кусок и decode вместе
    expect(res.steps.some((s) => s.prefill.length > 0 && s.decode.length > 0)).toBe(true)
  })
  it('игрушка из notes-engines §3.7: бюджет 10, A (6) и B (9) — B режется на куски', () => {
    const reqs: SimRequest[] = [
      { id: 'A', arrivalMs: 0, promptTokens: 6, outputTokens: 3 },
      { id: 'B', arrivalMs: 0, promptTokens: 9, outputTokens: 3 },
    ]
    const res = simulateContinuous(reqs, { model, gpu, tokenBudget: 10, maxSeqs: 4 })
    expect(res.steps[0].prefill).toEqual([
      { reqId: 'A', tokens: 6 },
      { reqId: 'B', tokens: 4 },
    ])
    expect(res.steps[1].decode).toEqual(['A'])
    expect(res.steps[1].prefill).toEqual([{ reqId: 'B', tokens: 5 }])
    const a = res.requests.find((r) => r.id === 'A')!
    const b = res.requests.find((r) => r.id === 'B')!
    expect(b.firstTokenMs).toBeGreaterThan(a.firstTokenMs) // TTFT у B на шаг позже из-за чанка
  })
  it('continuous ≥ static по throughput на смешанных длинах ответов', () => {
    const mixed: SimRequest[] = Array.from({ length: 32 }, (_, i) => ({ id: `m${i}`, arrivalMs: 0, promptTokens: 256, outputTokens: 16 + ((i * 97) % 497) }))
    const st = simulateStatic(mixed, { model, gpu, batchSize: 8 })
    const ct = simulateContinuous(mixed, { model, gpu, maxSeqs: 8 })
    checkInvariants(st, mixed)
    checkInvariants(ct, mixed)
    expect(ct.summary.throughputTokS).toBeGreaterThan(st.summary.throughputTokS * 1.2)
    expect(ct.summary.makespanMs).toBeLessThan(st.summary.makespanMs)
    expect(ct.summary.ttftP50).toBeLessThan(st.summary.ttftP50)
  })
  it('chunked prefill снижает p99 ITL, когда длинные промпты приходят среди decode', () => {
    const reqs: SimRequest[] = [
      ...Array.from({ length: 8 }, (_, i) => ({ id: `s${i}`, arrivalMs: i * 5, promptTokens: 128, outputTokens: 300 })),
      ...[300, 900, 1500, 2100].map((t, i) => ({ id: `L${i}`, arrivalMs: t, promptTokens: 8000, outputTokens: 20 })),
    ]
    const plain = simulateContinuous(reqs, { model, gpu, chunkedPrefill: false, tokenBudget: 8192 })
    const chunked = simulateContinuous(reqs, { model, gpu, chunkedPrefill: true, tokenBudget: 512 })
    checkInvariants(plain, reqs)
    checkInvariants(chunked, reqs)
    expect(plain.summary.itlP99).toBeGreaterThan(150) // один огромный шаг на 8000 токенов
    expect(chunked.summary.itlP99).toBeLessThan(plain.summary.itlP99 / 4)
    // без чанков 8000 токенов идут одним шагом
    expect(plain.steps.some((s) => s.prefill.some((p) => p.tokens === 8000))).toBe(true)
    expect(chunked.steps.every((s) => s.prefill.every((p) => p.tokens <= 512))).toBe(true)
  })
  it('prefix cache сокращает TTFT', () => {
    const base: SimRequest[] = Array.from({ length: 10 }, (_, i) => ({ id: `p${i}`, arrivalMs: i * 400, promptTokens: 6000, outputTokens: 10 }))
    const cold = simulateContinuous(base, { model, gpu })
    const warm = simulateContinuous(
      base.map((r) => ({ ...r, cachedTokens: 5600 })),
      { model, gpu },
    )
    expect(warm.summary.ttftP50).toBeLessThan(cold.summary.ttftP50 / 3)
    expect(warm.requests[0].cachedTokens).toBe(5600)
  })
  it('нехватка KV: preemption (recompute), но все запросы завершаются', () => {
    const reqs: SimRequest[] = Array.from({ length: 6 }, (_, i) => ({ id: `q${i}`, arrivalMs: 0, promptTokens: 400, outputTokens: 300 }))
    const res = simulateContinuous(reqs, { model, gpu, kvCapacityTokens: 2000 })
    checkInvariants(res, reqs)
    expect(res.summary.preemptions).toBeGreaterThan(0)
  })
  it('несколько реплик: запросы распределяются', () => {
    const w = makeWorkload({ rate: 20, promptDist: 800, outputDist: 100, count: 40, seed: 2 })
    const res = simulateContinuous(w, { model, gpu, workers: 3 })
    checkInvariants(res, w)
    expect(new Set(res.requests.map((r) => r.worker)).size).toBe(3)
    expect(res.summary.gpuCount).toBe(3)
  })
})

describe('static и dynamic batching', () => {
  it('static ждёт полный batch; dynamic стартует по таймеру', () => {
    const reqs: SimRequest[] = [0, 50, 100, 240].map((t, i) => ({ id: `R${i + 1}`, arrivalMs: t, promptTokens: 200, outputTokens: 40 }))
    const st = simulateStatic(reqs, { model, gpu, batchSize: 4 })
    checkInvariants(st, reqs)
    expect(st.steps[0].tMs).toBe(240)
    const dyn = simulateStatic(reqs, { model, gpu, batchSize: 4, maxWaitMs: 150 })
    checkInvariants(dyn, reqs)
    expect(dyn.steps[0].tMs).toBe(150)
    expect(dyn.steps[0].prefill).toHaveLength(3) // R4 «batched separately»
  })
})

describe('дезагрегация', () => {
  const heavy = makeWorkload({ rate: 6, promptDist: { kind: 'lognormal', median: 4000, sigma: 0.4, max: 16000 }, outputDist: { kind: 'uniform', min: 100, max: 200 }, count: 120, seed: 5 })
  it('при равном числе GPU disagg снижает p99 ITL для prefill-heavy нагрузки', () => {
    const agg = simulateContinuous(heavy, { model, gpu, workers: 4, tokenBudget: 2048 })
    const dis = simulateDisagg(heavy, { model, gpu, prefillWorkers: 2, decodeWorkers: 2, link: getInterconnect('ib-ndr') })
    checkInvariants(agg, heavy)
    checkInvariants(dis, heavy)
    expect(dis.summary.gpuCount).toBe(agg.summary.gpuCount)
    expect(dis.summary.itlP99).toBeLessThan(agg.summary.itlP99 / 2)
    // decode-воркеры не делают prefill, prefill-воркеры не делают decode
    for (const s of dis.steps) {
      if (s.worker!.startsWith('decode')) expect(s.prefill).toHaveLength(0)
      else expect(s.decode).toHaveLength(0)
    }
  })
  it('передача KV: размер = KV промпта, время по каналу; первый токен на decode после передачи', () => {
    const dis = simulateDisagg(heavy, { model, gpu, prefillWorkers: 2, decodeWorkers: 2, link: getInterconnect('ib-ndr') })
    expect(dis.transfers).toHaveLength(heavy.length)
    const tr = dis.transfers[0]
    const req = heavy.find((r) => r.id === tr.reqId)!
    expect(tr.bytes).toBe(kvBytesForSeq(model, req.promptTokens))
    expect(tr.endMs - tr.startMs).toBeCloseTo((tr.bytes / 50e9) * 1000, 6)
    const st = dis.requests.find((r) => r.id === tr.reqId)!
    expect(st.firstTokenMs).toBeGreaterThan(tr.endMs)
    expect(st.prefillWorker).toMatch(/^prefill-/)
    expect(st.worker).toMatch(/^decode-/)
  })
  it('медленный канал (TCP 25G) увеличивает TTFT; первый токен на prefill (SGLang) — раньше', () => {
    const w = heavy.slice(0, 40)
    const fast = simulateDisagg(w, { model, gpu, prefillWorkers: 2, decodeWorkers: 2, link: getInterconnect('nvlink4') })
    const slow = simulateDisagg(w, { model, gpu, prefillWorkers: 2, decodeWorkers: 2, link: getInterconnect('tcp-25g') })
    expect(slow.summary.ttftP50).toBeGreaterThan(fast.summary.ttftP50)
    const sg = simulateDisagg(w, { model, gpu, prefillWorkers: 2, decodeWorkers: 2, link: getInterconnect('tcp-25g'), firstTokenOn: 'prefill' })
    checkInvariants(sg, w)
    expect(sg.summary.ttftP50).toBeLessThan(slow.summary.ttftP50)
  })
})

describe('очередь и хвосты', () => {
  it('хоккейная клюшка: при загрузке → 1 хвост TTFT растёт резко', () => {
    const at = (rate: number) => {
      const w = makeWorkload({ rate, promptDist: { kind: 'lognormal', median: 1500, sigma: 0.5, max: 8000 }, outputDist: 100, count: 300, seed: 2 })
      const q = simulateQueue(w, { model, gpu })
      checkInvariants(q, w)
      return q
    }
    const a = at(5)
    const b = at(20)
    const c = at(28)
    expect(a.utilization).toBeLessThan(0.5)
    expect(b.utilization).toBeGreaterThan(0.85)
    expect(b.summary.ttftP99).toBeGreaterThan(a.summary.ttftP99 * 2)
    expect(c.summary.ttftP99).toBeGreaterThan(b.summary.ttftP99 * 2)
    expect(a.summary.ttftP99).toBeGreaterThan(a.summary.ttftP50)
  })
  it('больше реплик → меньше ожидание; своё время обслуживания', () => {
    const w = makeWorkload({ rate: 50, promptDist: 1000, outputDist: 10, count: 100, seed: 3 })
    const one = simulateQueue(w, { model, gpu, servers: 1, serviceMs: () => 40 })
    const two = simulateQueue(w, { model, gpu, servers: 2, serviceMs: () => 40 })
    expect(two.summary.ttftP99).toBeLessThan(one.summary.ttftP99)
    expect(one.requests[0].ttftMs).toBe(40)
  })
  it('goodput: доля запросов в SLO', () => {
    const w = makeWorkload({ rate: 10, promptDist: 1000, outputDist: 50, count: 50, seed: 8 })
    const res = simulateContinuous(w, { model, gpu })
    const g = goodput(res, { ttftMs: 1e9, tpotMs: 1e9 })
    expect(g.fraction).toBe(1)
    expect(goodput(res, { ttftMs: 0, tpotMs: 1e9 }).fraction).toBe(0)
    expect(g.reqPerSec).toBeGreaterThan(0)
  })
  it('summarize считает throughput и TPOT', () => {
    const s = summarize(
      [{ id: 'a', arrivalMs: 0, firstTokenMs: 100, finishMs: 1100, ttftMs: 100, itlMs: [100, 900], promptTokens: 10, outputTokens: 3, cachedTokens: 0, e2eMs: 1100, queueMs: 0, preemptions: 0 }],
      2,
    )
    expect(s.tpotMean).toBe(500)
    expect(s.throughputTokS).toBeCloseTo(3 / 1.1, 6)
    expect(s.throughputPerGpuTokS).toBeCloseTo(3 / 2.2, 6)
  })
})
