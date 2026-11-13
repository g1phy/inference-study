import { describe, expect, it } from 'vitest'
import { getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'
import { prefillMs } from '@/engine/perf'
import { makeWorkload, quantileSorted, simulateQueue } from '@/engine/sched'
import { makeHardware } from '@/chapters/prefill-decode/shared/hw-core'
import { simulateLevel } from './goodput-sim'
import {
  POPULATION,
  lindleyStrided,
  lindleyStridedMulti,
  lindleyTtft,
  meanServiceMs,
  queueCurve,
  requestStream,
  sampleTtft,
  steadyTtft,
} from './queue-sim'

const hw = makeHardware(getModel('qwen3-8b'), getGpu('h100-sxm'), 'bf16')

describe('queue-sim', () => {
  it('рекурсия Линдли совпадает с simulateQueue для одной реплики', () => {
    const reqs = makeWorkload({ rate: 20, promptDist: { kind: 'lognormal', median: 1000, sigma: 0.8, min: 32, max: 16000 }, outputDist: 2, seed: 5, count: 800 })
    const svc = (n: number) => prefillMs(hw.model, hw.gpu, n, hw.perf)
    const ref = simulateQueue(reqs, { model: hw.model, gpu: hw.gpu, ...hw.perf, servers: 1 }).requests.map((r) => r.ttftMs)
    const mine = lindleyTtft(
      reqs.map((r) => r.arrivalMs),
      reqs.map((r) => svc(r.promptTokens)),
    )
    expect(mine.length).toBe(ref.length)
    for (let i = 0; i < ref.length; i++) expect(mine[i]).toBeCloseTo(ref[i], 6)
  })

  it('хвост TTFT растёт с нагрузкой («хоккейная клюшка»), кривая монотонна', () => {
    expect(meanServiceMs(hw)).toBeGreaterThan(5)
    const curve = queueCurve(hw)
    const at = (rho: number) => curve.find((c) => Math.abs(c.rho - rho) < 1e-9)!
    expect(at(0.9).p99).toBeGreaterThan(at(0.5).p99 * 2)
    expect(at(0.9).p50).toBeGreaterThan(at(0.5).p50)
    for (const c of curve) expect(c.p99).toBeGreaterThanOrEqual(c.p50)
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].p50).toBeGreaterThanOrEqual(curve[i - 1].p50)
      expect(curve[i].p99).toBeGreaterThanOrEqual(curve[i - 1].p99)
    }
  })

  it('прогон по интервалам с шагом stride берёт каждый stride-й запрос той же рекурсии', () => {
    const { gaps, prompts } = requestStream(3000, 9)
    const svc = Array.from(prompts, (n) => prefillMs(hw.model, hw.gpu, n, hw.perf))
    const scale = 40
    const arrivals: number[] = []
    let t = 0
    for (const g of gaps) arrivals.push((t += g * scale))
    const ref = lindleyTtft(arrivals, svc, 500)
    const all = lindleyStrided(gaps, svc, scale, 500, 1)
    expect(all.length).toBe(ref.length)
    for (let i = 0; i < ref.length; i++) expect(all[i]).toBeCloseTo(ref[i], 3)
    const every3 = lindleyStrided(gaps, svc, scale, 500, 3)
    expect(every3.length).toBe(Math.ceil(ref.length / 3))
    for (let i = 0; i < every3.length; i++) expect(every3[i]).toBeCloseTo(ref[3 * i], 3)
    // один проход сразу на несколько нагрузок даёт то же, что отдельные прогоны
    const multi = lindleyStridedMulti(gaps, svc, [30, scale, 55], 500, 3)
    for (const [k, sc] of [30, scale, 55].entries()) expect(Array.from(multi[k])).toEqual(Array.from(lindleyStrided(gaps, svc, sc, 500, 3)))
  })

  it('поток запросов тот же, что у makeWorkload', () => {
    const { gaps, prompts } = requestStream(2000, 3)
    const reqs = makeWorkload({ rate: 1000, promptDist: { kind: 'lognormal', median: 1000, sigma: 0.8, min: 32, max: 16000 }, outputDist: 1, seed: 3, count: 2000 })
    let t = 0
    for (let i = 0; i < reqs.length; i++) {
      t += gaps[i]
      expect(prompts[i]).toBe(reqs[i].promptTokens)
      expect(t).toBeCloseTo(reqs[i].arrivalMs, 1)
    }
  })

  it('в устойчивом режиме среднее больше медианы: длинный правый хвост', () => {
    for (const rho of [0.5, 0.7, 0.9, 0.95, 0.98]) {
      const s = steadyTtft(hw, rho)
      expect(s.length).toBe(POPULATION)
      let sum = 0
      for (const v of s) sum += v
      expect(sum / s.length).toBeGreaterThan(quantileSorted(s, 0.5))
    }
    // при 98 % хвост не схлопывается в один всплеск: среднее заметно больше медианы, p99 в разы дальше
    const s98 = steadyTtft(hw, 0.98)
    let sum98 = 0
    for (const v of s98) sum98 += v
    expect(sum98 / s98.length).toBeGreaterThan(1.25 * quantileSorted(s98, 0.5))
    expect(quantileSorted(s98, 0.99)).toBeGreaterThan(4 * quantileSorted(s98, 0.5))
  })

  it('p99 маленькой выборки прыгает, большой почти стоит на месте', () => {
    const p99 = (n: number) => [1, 2, 3, 4, 5].map((seed) => sampleTtft(hw, 0.7, n, seed).p99)
    const spread = (xs: number[]) => (Math.max(...xs) - Math.min(...xs)) / Math.min(...xs)
    expect(spread(p99(200))).toBeGreaterThan(0.15)
    expect(spread(p99(10000))).toBeLessThan(0.08)
    // тот же seed даёт ту же выборку
    expect(sampleTtft(hw, 0.7, 200, 9).p99).toBe(sampleTtft(hw, 0.7, 200, 9).p99)
  })
})

describe('goodput-sim', () => {
  it('больше concurrency: больше throughput и длиннее ITL, пока не упёрлись в память', () => {
    const l1 = simulateLevel(hw, 1, 11)
    const l8 = simulateLevel(hw, 8, 12)
    const l64 = simulateLevel(hw, 64, 13)
    expect(l8.reqPerSec).toBeGreaterThan(l1.reqPerSec * 4)
    expect(l64.reqPerSec).toBeGreaterThan(l8.reqPerSec)
    expect(l64.tpotP90).toBeGreaterThan(l8.tpotP90)
    expect(l8.tpotP90).toBeGreaterThan(l1.tpotP90)
    // в окне измерения есть запросы, массивы для пересчёта goodput согласованы
    expect(l64.ttft.length).toBe(l64.tpot.length)
    expect(l64.ttft.length).toBeGreaterThan(10)
  })

  it('детерминированно при том же seed', () => {
    const a = simulateLevel(hw, 16, 7)
    const b = simulateLevel(hw, 16, 7)
    expect(a.reqPerSec).toBe(b.reqPerSec)
    expect(a.ttftP90).toBe(b.ttftP90)
  })
})
