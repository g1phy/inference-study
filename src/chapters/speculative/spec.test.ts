import { describe, expect, it } from 'vitest'
import { getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'
import { promptLookupRounds } from './ngram'
import { acceptedDistribution, acceptedFraction, batchSpeedupSweep, expectedSpeedup, expectedTokens, optimalK, passMs, simulateRounds } from './spec'

describe('формулы Leviathan (notes-techniques §3.8)', () => {
  it('таблица E = (1 − α^(k+1)) / (1 − α)', () => {
    expect(expectedTokens(0.8, 4)).toBeCloseTo(3.3616, 4)
    expect(expectedTokens(0.7, 3)).toBeCloseTo(2.53, 2)
    expect(expectedTokens(0.9, 10)).toBeCloseTo(6.86, 2)
    expect(expectedTokens(0.5, 8)).toBeCloseTo(2.0, 2)
    expect(expectedTokens(0.6, 2)).toBeCloseTo(1.96, 2)
    expect(expectedTokens(1, 4)).toBe(5)
    expect(expectedTokens(0, 4)).toBe(1)
  })
  it('ускорение с ценой черновика: α = 0,8, k = 4, c = 0,05 → 2,80×', () => {
    expect(expectedSpeedup(0.8, 4, 0.05)).toBeCloseTo(2.8013, 3)
  })
  it('оптимальная длина черновика (таблица заметок)', () => {
    expect(optimalK(0.8, 0.05)).toMatchObject({ k: 8 })
    expect(optimalK(0.8, 0.05).speedup).toBeCloseTo(3.09, 2)
    expect(optimalK(0.9, 0.1)).toMatchObject({ k: 10 })
    expect(optimalK(0.9, 0.1).speedup).toBeCloseTo(3.43, 2)
    expect(optimalK(0.6, 0.2)).toMatchObject({ k: 2 })
    expect(optimalK(0.6, 0.2).speedup).toBeCloseTo(1.4, 2)
    expect(optimalK(0.7, 0.1)).toMatchObject({ k: 4 })
  })
  it('распределение принятых: α = 0,8, k = 4', () => {
    const d = acceptedDistribution(0.8, 4)
    expect(d.map((v) => Math.round(v * 1e4) / 1e4)).toEqual([0.2, 0.16, 0.128, 0.1024, 0.4096])
    expect(d.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(acceptedFraction(0.8, 4)).toBeCloseTo(0.5904, 4)
  })
})

describe('пошаговая симуляция', () => {
  it('покрывает ответ целиком, детерминирована', () => {
    const a = simulateRounds(40, 0.7, 4, 3)
    const b = simulateRounds(40, 0.7, 4, 3)
    expect(a).toEqual(b)
    const produced = a.reduce((s, r) => s + r.produced, 0)
    expect(produced).toBe(40)
    for (const r of a) {
      const rej = r.drafts.findIndex((d) => d.verdict === 'rejected')
      if (rej >= 0) {
        expect(r.drafts.slice(rej + 1).every((d) => d.verdict === 'discarded')).toBe(true)
        expect(r.targetKind).toBe('correction')
        expect(r.targetPos).toBe(r.drafts[rej].pos)
      } else expect(r.targetKind).toBe('bonus')
    }
  })
  it('среднее токенов за раунд близко к E на длинном ответе', () => {
    const rounds = simulateRounds(20000, 0.8, 4, 11)
    const mean = 20000 / rounds.length
    expect(mean).toBeGreaterThan(expectedTokens(0.8, 4) - 0.1)
    expect(mean).toBeLessThan(expectedTokens(0.8, 4) + 0.1)
  })
})

describe('зависимость от batch (engine/perf)', () => {
  const m = getModel('qwen3-8b')
  const g = getGpu('h100-sxm')
  it('при batch 1 проверка 5 токенов почти не дороже шага на 1 токен', () => {
    const one = passMs(m, g, 1, 2000).ms
    const five = passMs(m, g, 5, 2000).ms
    expect(five / one).toBeLessThan(1.05)
  })
  it('малый batch: ускорение близко к формуле; большой batch: выигрыш падает', () => {
    const pts = batchSpeedupSweep(m, g, { alpha: 0.8, k: 4, c: 0.05, avgContext: 512, batches: [1, 8, 64, 256, 1024] })
    expect(pts[0].speedup).toBeGreaterThan(expectedSpeedup(0.8, 4, 0.05) * 0.95)
    expect(pts[0].verifyBound).toBe('memory')
    expect(pts[4].verifyBound).toBe('compute')
    expect(pts[4].speedup).toBeLessThan(1)
    for (let i = 1; i < pts.length; i++) expect(pts[i].speedup).toBeLessThanOrEqual(pts[i - 1].speedup + 1e-9)
  })
})


describe('n-gram (prompt lookup)', () => {
  it('копирование из промпта принимается длинными сериями', () => {
    const prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    const output = [3, 4, 5, 6, 7, 8]
    const r = promptLookupRounds(prompt, output, { k: 4, nMin: 1, nMax: 2 })
    // первый раунд: суффикс [8] ... совпадения нет раньше, кроме самого себя → обычный шаг
    expect(r.reduce((s, x) => s + x.produced, 0)).toBe(output.length)
    expect(r.length).toBeLessThan(output.length)
    expect(r[1].accepted).toBeGreaterThan(0)
  })
  it('новый текст: совпадений нет, каждый проход даёт один токен', () => {
    const r = promptLookupRounds([1, 2, 3], [10, 11, 12, 13], { k: 4, nMin: 2, nMax: 3 })
    expect(r.every((x) => x.produced === 1 && x.drafts.length === 0)).toBe(true)
  })
  it('правка посередине: отказ на изменённом токене', () => {
    const prompt = [5, 6, 7, 8, 9, 10, 11]
    const output = [5, 6, 7, 99, 9, 10, 11]
    const r = promptLookupRounds(prompt, output, { k: 4, nMin: 1, nMax: 2 })
    expect(r.reduce((s, x) => s + x.produced, 0)).toBe(7)
    const withReject = r.find((x) => x.drafts.length > x.accepted)
    expect(withReject).toBeTruthy()
  })
})
