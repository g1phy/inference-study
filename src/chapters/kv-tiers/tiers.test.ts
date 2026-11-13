import { describe, expect, it } from 'vitest'
import { getModel } from '@/engine/data/models'
import { getGpu } from '@/engine/data/gpus'
import { findCrossover, gpuHostLink, tierBandwidth } from './tiers'

const q = getModel('qwen3-8b')
const h100 = getGpu('h100-sxm')

describe('уровни KV: полосы и паритет', () => {
  it('полосы уровней по умолчанию', () => {
    expect(tierBandwidth('g2', 1).GBs).toBe(50)
    expect(tierBandwidth('g2', 8).GBs).toBe(400)
    expect(tierBandwidth('g3', 1).GBs).toBe(14)
    expect(tierBandwidth('g4', 1, 'tcp').GBs).toBeCloseTo(3.125)
    expect(tierBandwidth('g4', 8, 's3')).toEqual({ GBs: 1, latencyMs: 100, capped: false })
  })
  it('хост-линк GPU: A100 на PCIe Gen4, Hopper и Blackwell на Gen5', () => {
    expect(gpuHostLink(h100)).toEqual({ id: 'pcie5', practicalGBs: 50, GBs: 63 })
    expect(gpuHostLink(getGpu('a100-80g'))).toEqual({ id: 'pcie4', practicalGBs: 25, GBs: 31.5 })
    // у B200 в gpus.ts строка «Gen6» (сам чип), но хост в DGX B200 даёт Gen5
    expect(gpuHostLink(getGpu('b200')).id).toBe('pcie5')
  })
  it('диски и сетевые карты не быстрее PCIe самих GPU', () => {
    // одна H100: 8 дисков или 8 карт RDMA упираются в её PCIe
    expect(tierBandwidth('g3', 8, 'rdma', { hostGBs: 50, gpus: 1 })).toEqual({ GBs: 50, latencyMs: 0, capped: true })
    expect(tierBandwidth('g4', 8, 'rdma', { hostGBs: 50, gpus: 1 })).toEqual({ GBs: 50, latencyMs: 0, capped: true })
    expect(tierBandwidth('g4', 8, 'tcp', { hostGBs: 50, gpus: 1 })).toEqual({ GBs: 25, latencyMs: 0, capped: false })
    expect(tierBandwidth('g3', 8, 'rdma', { hostGBs: 50, gpus: 4 }).GBs).toBe(112)
    // A100: даже одна карта RDMA быстрее её PCIe Gen4
    expect(tierBandwidth('g4', 1, 'rdma', { hostGBs: 25, gpus: 1 })).toEqual({ GBs: 25, latencyMs: 0, capped: true })
    // RAM: по линку на каждый GPU реплики, без упора
    expect(tierBandwidth('g2', 2, 'rdma', { hostGBs: 50, gpus: 2 })).toEqual({ GBs: 100, latencyMs: 0, capped: false })
    expect(tierBandwidth('g2', 1, 'rdma', { hostGBs: 25, gpus: 1 }).GBs).toBe(25)
    expect(tierBandwidth('g4', 8, 's3', { hostGBs: 25, gpus: 1 })).toEqual({ GBs: 1, latencyMs: 100, capped: false })
  })
  it('RAM по PCIe быстрее пересчёта на всех длинах', () => {
    const c = findCrossover(q, h100, 50, 0, 1)
    expect(c.tokens).toBeNull()
    expect(c.shortWinner).toBe('load')
  })
  it('TCP 25 Гбит/с: на коротком контексте выгоднее пересчитать, на длинном загрузить', () => {
    const c = findCrossover(q, h100, 3.125, 0, 1)
    expect(c.shortWinner).toBe('recompute')
    expect(c.tokens).toBeGreaterThan(16000)
    expect(c.tokens).toBeLessThan(64000)
  })
  it('S3 с задержкой 100 мс не обгоняет пересчёт до 128 тысяч токенов', () => {
    const c = findCrossover(q, h100, 1, 100, 1)
    expect(c.tokens).toBeNull()
    expect(c.shortWinner).toBe('recompute')
  })
})
