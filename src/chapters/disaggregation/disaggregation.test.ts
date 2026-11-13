import { describe, expect, test } from 'vitest'
import { getGpu } from '@/engine/data/gpus'
import { buildRun } from '@/engine/flow'
import { SCENARIOS, scenarioPatch } from '@/engine/scenarios'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { DEFAULT_SETTINGS } from '@/state/settings'
import type { FlowSettings } from '@/engine/types'
import { simConfig } from '../batching/config'
import { COND_EXAMPLES, PRESETS, bestRun, decideConditional, linkFor, nvlinkFor, runComparison, runLanes, type RunSummary } from './logic'
import { readerPromptTokenCount } from './promptTokens'

describe('промпт читателя в DynamoFlow и Transfer', () => {
  test('столько же токенов, сколько в симуляторе, во всех сценариях (с tools тоже)', () => {
    const tk = loadQwen3ForTests()
    const counts: Record<string, number> = {}
    for (const sc of SCENARIOS) {
      const s: FlowSettings = { ...DEFAULT_SETTINGS, ...scenarioPatch(sc.id) }
      const n = readerPromptTokenCount(tk, s)
      expect(n, sc.id).toBe(buildRun(s, tk).promptTokens.length)
      counts[sc.id] = n
    }
    // описание инструмента добавляет к промпту больше сотни токенов
    const tool: FlowSettings = { ...DEFAULT_SETTINGS, ...scenarioPatch('tool') }
    expect(counts.tool - readerPromptTokenCount(tk, { ...tool, tools: false })).toBeGreaterThan(100)
  })
})

describe('conditional disaggregation (notes-dynamo §3.6)', () => {
  test('четыре запроса из разбора', () => {
    const d = COND_EXAMPLES.map((e) => decideConditional({ ...e, policy: 'isl_bounding', prefillBusy: false }).local)
    expect(d).toEqual([true, false, false, true])
  })
  test('короткий без кэша уходит локально только при перегрузке prefill в isl_or_load', () => {
    const base = { isl: 1800, cached: 0, prefillBusy: true }
    expect(decideConditional({ ...base, policy: 'isl_bounding' }).local).toBe(false)
    expect(decideConditional({ ...base, policy: 'isl_or_load' }).local).toBe(true)
    expect(decideConditional({ ...base, policy: 'prefill_load' }).local).toBe(true)
  })
  test('пороги строгие', () => {
    expect(decideConditional({ isl: 4096, cached: 2048, policy: 'isl_bounding', prefillBusy: false }).islOk).toBe(false)
    expect(decideConditional({ isl: 10000, cached: 3000, policy: 'isl_bounding', prefillBusy: false }).ratioOk).toBe(false)
  })
})

describe('каналы', () => {
  test('NVLink по поколению GPU', () => {
    expect(nvlinkFor(getGpu('h100-sxm')).id).toBe('nvlink4')
    expect(nvlinkFor(getGpu('b200')).id).toBe('nvlink5')
    expect(nvlinkFor(getGpu('l40s')).id).toBe('pcie4')
    expect(linkFor('rdma', getGpu('h100-sxm'), 4).GBs).toBe(200)
    expect(linkFor('tcp', getGpu('h100-sxm'), 4).GBs).toBeCloseTo(3.125)
  })
})

describe('прогоны', () => {
  const cfg = simConfig('qwen3-8b', 'h100-sxm')
  test('disagg выравнивает ITL на потоке фигуры 1', () => {
    const r = runLanes(cfg, 'nvlink')
    expect(r.disagg.summary.itlP99).toBeLessThan(r.agg.summary.itlP99)
    expect(r.disagg.transfers.length).toBe(12)
    // на TCP передача KV заметно длиннее, TTFT растёт
    const tcp = runLanes(cfg, 'tcp')
    expect(tcp.disagg.summary.ttftP50).toBeGreaterThan(r.disagg.summary.ttftP50)
  })
  test('«Рассуждения»: медианная цель по ITL отсекает схему с одним decode-воркером', () => {
    const p = PRESETS.find((q) => q.id === 'reason')!
    const r = runComparison(cfg, p, p.slo)
    expect(r.itlP50Ms).toBeGreaterThan(0)
    const by = Object.fromEntries(r.splits.map((s) => [s.key, s.slo]))
    expect(by['3P1D']).toBeLessThan(0.9)
    expect(by['1P3D']).toBeGreaterThan(0.99)
  })
  test('лучшая схема: доля в SLO, при равенстве выдано на GPU; если все в SLO, лучшей нет', () => {
    const run = (key: string, slo: number, perGpu: number, itlP99 = 10): RunSummary => ({ key, label: key, prefill: 0, decode: 0, gpus: 4, ttftP50: 0, ttftP99: 0, itlP50: 0, itlP99, perGpu, slo, transferP50: 0 })
    expect(bestRun([run('agg', 1, 100), run('1P3D', 1, 90)])).toBeNull()
    expect(bestRun([run('agg', 0.9, 100), run('1P3D', 0.9, 120), run('3P1D', 0.5, 200)])!.key).toBe('1P3D')
    expect(bestRun([run('agg', 0.5, 100), run('3P1D', 1, 50)])!.key).toBe('3P1D')
  })
  test('сравнение: все разбиения и agg', () => {
    const p = PRESETS[0]
    const { agg, splits } = runComparison(cfg, p, p.slo)
    expect(splits.map((s) => s.key)).toEqual(['1P3D', '2P2D', '3P1D'])
    expect(agg.gpus).toBe(4)
    for (const s of [agg, ...splits]) {
      expect(s.slo).toBeGreaterThanOrEqual(0)
      expect(s.slo).toBeLessThanOrEqual(1)
      expect(s.perGpu).toBeGreaterThan(0)
    }
  })
})
