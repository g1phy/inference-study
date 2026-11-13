import { describe, expect, it } from 'vitest'
import { simulateAutoscale, trafficAt, violationBands, type AutoscaleOptions } from './autoscale'

const base: AutoscaleOptions = {
  preset: 'chat',
  spike: false,
  signal: 'concurrency',
  coldStartMin: 3,
  scaleDownDelayMin: 10,
  minReplicas: 1,
  maxReplicas: 32,
}

describe('трафик', () => {
  it('чат: ночью минимум, вечером пик; рабочие часы: ночью ноль', () => {
    expect(trafficAt(4 * 60, 'chat', false)).toBeLessThan(trafficAt(21 * 60, 'chat', false) / 4)
    expect(trafficAt(3 * 60, 'b2b', false)).toBe(0)
    expect(trafficAt(11 * 60, 'b2b', false)).toBeGreaterThan(80)
    expect(trafficAt(14 * 60 + 20, 'chat', true)).toBeGreaterThan(trafficAt(14 * 60 + 20, 'chat', false) * 1.9)
  })
})

describe('автоскейлинг', () => {
  it('детерминирован, сутки по минутам', () => {
    const a = simulateAutoscale(base)
    const b = simulateAutoscale(base)
    expect(a.minutes).toHaveLength(1440)
    expect(a.gpuHours).toBe(b.gpuHours)
  })
  it('сигнал по concurrency держит SLO почти весь день', () => {
    const r = simulateAutoscale(base)
    expect(r.violatedShare).toBeLessThan(0.05)
    expect(r.peakReplicas).toBeLessThan(32)
  })
  it('CPU почти не реагирует: реплик мало, очередь растёт', () => {
    const r = simulateAutoscale({ ...base, signal: 'cpu' })
    expect(r.violatedShare).toBeGreaterThan(0.5)
    expect(r.minutes.slice(120).every((m) => m.total <= 2)).toBe(true)
  })
  it('utilization GPU насыщается: реплики уходят в максимум, GPU-часов больше', () => {
    const c = simulateAutoscale(base)
    const g = simulateAutoscale({ ...base, signal: 'gpu' })
    expect(g.peakReplicas).toBe(32)
    expect(g.gpuHours).toBeGreaterThan(c.gpuHours * 1.3)
  })
  it('долгий cold start даёт больше нарушений на всплеске', () => {
    const fast = simulateAutoscale({ ...base, spike: true, coldStartMin: 1 })
    const slow = simulateAutoscale({ ...base, spike: true, coldStartMin: 15 })
    expect(slow.violationMinutes).toBeGreaterThan(fast.violationMinutes)
    expect(slow.maxWaitSec).toBeGreaterThan(fast.maxWaitSec)
  })
  it('scale to zero: ночью ноль реплик, первый утренний запрос ждёт холодный старт', () => {
    const r = simulateAutoscale({ ...base, preset: 'b2b', minReplicas: 0, coldStartMin: 6 })
    expect(r.minutes[3 * 60].total).toBe(0)
    const first = r.minutes.find((m) => m.t > 5 * 60 && m.lambda > 0)!
    expect(first.waitSec).toBeGreaterThan(4 * 60)
  })
  it('полосы нарушений', () => {
    const r = simulateAutoscale({ ...base, signal: 'cpu' })
    const bands = violationBands(r.minutes)
    expect(bands.length).toBeGreaterThan(0)
    for (const [a, b] of bands) expect(b).toBeGreaterThan(a)
  })
})
