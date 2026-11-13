import { describe, expect, it } from 'vitest'
import { getModel } from '@/engine/data/models'
import { getGpu } from '@/engine/data/gpus'
import { kvBytesPerToken } from '@/engine/perf'
import { enInt, kvPerTokenWithHeads, logGrid, memoryLayout, minGpus, shortGpuName } from './lib'

const q = getModel('qwen3-8b')
const h100 = getGpu('h100-sxm')

describe('помощники главы KV cache', () => {
  it('KV на токен Qwen3-8B и варианты MHA / MQA', () => {
    expect(kvBytesPerToken(q)).toBe(147456)
    expect(kvPerTokenWithHeads(q, 32)).toBe(589824)
    expect(kvPerTokenWithHeads(q, 1)).toBe(18432)
  })
  it('полоса памяти: части в сумме дают всю память, если всё влезло', () => {
    const l = memoryLayout(q, h100, { contextLen: 8192, seqs: 32, gpus: 1, dtype: 'bf16', kvDtype: 'bf16', memUtil: 0.92 })
    const sum = l.parts.reduce((a, p) => a + p.bytes, 0)
    expect(sum).toBeCloseTo(l.total, -3)
    expect(l.overflowKind).toBeNull()
    expect(l.fits.maxSeqs).toBe(49)
  })
  it('переполнение KV и невлезшие веса различаются', () => {
    const kv = memoryLayout(q, h100, { contextLen: 8192, seqs: 64, gpus: 1, dtype: 'bf16', kvDtype: 'bf16', memUtil: 0.92 })
    expect(kv.overflowKind).toBe('kv')
    const w = memoryLayout(getModel('llama-3.1-70b'), h100, { contextLen: 8192, seqs: 1, gpus: 1, dtype: 'bf16', kvDtype: 'bf16', memUtil: 0.92 })
    expect(w.overflowKind).toBe('weights')
    expect(w.overflowBytes).toBeGreaterThan(50e9)
  })
  it('сколько GPU нужно под веса', () => {
    expect(minGpus(q, h100)).toBe(1)
    expect(minGpus(getModel('llama-3.1-70b'), h100)).toBe(4)
    expect(minGpus(getModel('llama-3.1-70b'), h100, 'fp8')).toBe(2)
  })
  it('мелочи', () => {
    expect(shortGpuName(h100)).toBe('H100')
    expect(shortGpuName(getGpu('b200'))).toBe('B200')
    expect(enInt(407915)).toBe('407,915')
    const g = logGrid(512, 131072, 36)
    expect(g[0]).toBe(512)
    expect(g[g.length - 1]).toBe(131072)
    expect(new Set(g).size).toBe(g.length)
  })
})
