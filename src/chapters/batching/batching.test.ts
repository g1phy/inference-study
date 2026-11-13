import { describe, expect, test } from 'vitest'
import { getModel } from '@/engine/data/models'
import { getGpu } from '@/engine/data/gpus'
import { simulateContinuous, simulateStatic } from '@/engine/sched'
import type { SimRequest } from '@/engine/types'
import { TOY_REQUESTS, toySchedule } from './toyScheduler'
import { idleShare, itlTimeline, phaseAt, phaseCounts, requestTracks } from './timeline'
import { minTp, simConfig } from './config'

describe('toySchedule (vLLM V1, notes-engines §3.7)', () => {
  test('пример из заметок: бюджет 10, max_num_seqs 4', () => {
    const steps = toySchedule(
      [
        { id: 'A', arrive: 1, prompt: 6, output: 5 },
        { id: 'B', arrive: 1, prompt: 9, output: 5 },
        { id: 'C', arrive: 2, prompt: 3, output: 5 },
      ],
      { budget: 10, maxSeqs: 4 },
    )
    // шаг 1: A целиком (6), B урезан до 4 (chunked prefill); токен только у A
    expect(steps[0].allocs.map((a) => [a.id, a.kind, a.tokens])).toEqual([
      ['A', 'prefill', 6],
      ['B', 'prefill', 4],
    ])
    expect(steps[0].allocs[1].chunked).toBe(true)
    expect(steps[0].emitted.map((e) => e.id)).toEqual(['A'])
    // шаг 2: decode A + хвост B (5) + весь C (3) = 9 токенов
    expect(steps[1].allocs.map((a) => [a.id, a.kind, a.tokens])).toEqual([
      ['A', 'decode', 1],
      ['B', 'prefill', 5],
      ['C', 'prefill', 3],
    ])
    expect(steps[1].used).toBe(9)
    expect(steps[1].emitted.map((e) => `${e.id}${e.index}`)).toEqual(['A2', 'B1', 'C1'])
    // шаг 3: три decode
    expect(steps[2].used).toBe(3)
  })

  test('max_num_seqs держит запрос в очереди при свободном бюджете', () => {
    const steps = toySchedule(TOY_REQUESTS, { budget: 10, maxSeqs: 4 })
    const s3 = steps[2]
    expect(s3.runningAfter.length).toBeLessThanOrEqual(4)
    const all = steps.flatMap((s) => s.finished)
    expect(new Set(all)).toEqual(new Set(['A', 'B', 'C', 'D', 'E']))
    // каждый запрос выдал ровно output токенов
    for (const r of TOY_REQUESTS) expect(steps.flatMap((s) => s.emitted).filter((e) => e.id === r.id).length).toBe(r.output)
    // бюджет никогда не превышен
    for (const s of steps) expect(s.used).toBeLessThanOrEqual(10)
  })

  test('большой бюджет: B входит целиком на первом шаге', () => {
    const steps = toySchedule(TOY_REQUESTS, { budget: 16, maxSeqs: 4 })
    expect(steps[0].allocs.find((a) => a.id === 'B')!.tokens).toBe(9)
    expect(steps[0].emitted.map((e) => e.id)).toEqual(['A', 'B'])
  })
})

describe('timeline', () => {
  const m = getModel('qwen3-8b')
  const g = getGpu('h100-sxm')
  const reqs: SimRequest[] = [0, 30, 60, 90, 170, 230, 280, 340].map((a, i) => ({
    id: 'ABCDEFGH'[i],
    arrivalMs: a,
    promptTokens: [420, 760, 300, 610, 520, 380, 700, 450][i],
    outputTokens: [34, 8, 26, 12, 10, 38, 16, 22][i],
  }))

  test('static: простой слотов и ожидание набора batch', () => {
    const res = simulateStatic(reqs, { model: m, gpu: g, batchSize: 4 })
    const tracks = requestTracks(res, { static: true })
    const a = tracks.find((t) => t.id === 'A')!
    expect(a.segments[0].kind).toBe('wait')
    const b = tracks.find((t) => t.id === 'B')!
    expect(b.segments.at(-1)!.kind).toBe('idle')
    expect(b.batchEnd).toBeCloseTo(a.stats.finishMs, 6)
    const share = idleShare(tracks)
    expect(share).toBeGreaterThan(0.1)
    expect(share).toBeLessThan(0.9)
    expect(phaseAt(b, b.stats.finishMs + 1)).toBe('idle')
    // padding: в первом batch (A, B, C, D) все промпты дополняются до самого длинного, 760 токенов у B
    const pre = (id: string) => tracks.find((t) => t.id === id)!.segments.find((x) => x.kind === 'prefill')!
    expect(pre('A').padTo).toBe(760)
    expect(pre('B').padTo).toBeUndefined()
  })

  test('continuous: нет простоя, токены по шагам', () => {
    const res = simulateContinuous(reqs, { model: m, gpu: g })
    const tracks = requestTracks(res)
    expect(idleShare(tracks)).toBe(0)
    const c = phaseCounts(tracks, 100)
    expect(c.work + c.wait + c.done + c.future + c.idle).toBe(8)
    for (const tr of tracks) {
      const dec = tr.segments.filter((s) => s.kind === 'decode').length
      expect(dec).toBe(tr.stats.outputTokens - 1)
    }
  })

  test('ITL во времени совпадает с ITL запросов', () => {
    const res = simulateContinuous(reqs, { model: m, gpu: g })
    const pts = itlTimeline(res)
    const maxPts = Math.max(...pts.map((p) => p.itl))
    const maxReq = Math.max(...res.requests.flatMap((r) => r.itlMs))
    expect(maxPts).toBeCloseTo(maxReq, 6)
    expect(pts.reduce((a, p) => a + p.count, 0)).toBe(res.requests.reduce((a, r) => a + r.itlMs.length, 0))
  })
})

describe('simConfig', () => {
  test('TP подбирается по памяти', () => {
    expect(minTp(getModel('qwen3-8b'), getGpu('h100-sxm'))).toBe(1)
    expect(minTp(getModel('llama-3.1-70b'), getGpu('h100-sxm'))).toBe(4)
    expect(minTp(getModel('llama-3.1-70b'), getGpu('h200'))).toBe(2)
    expect(simConfig('qwen3-8b', 'h100-sxm').timeScale).toBeCloseTo(1, 6)
    expect(simConfig('qwen3-32b', 'h100-sxm').tp).toBe(2)
    expect(simConfig('nope', 'nope').model.id).toBe('qwen3-8b')
  })
})
