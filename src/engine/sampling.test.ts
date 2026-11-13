import { describe, expect, it } from 'vitest'
import { argmax, distribution, entropy, minPMask, sample, softmax, topKMask, topPMask } from './sampling'
import type { Candidate, Token } from './types'

const tk = (id: number, text: string): Token => ({ id, raw: text, bytes: [], text, special: false })
const cands = (logits: number[]): Candidate[] => logits.map((logit, i) => ({ token: tk(i, `t${i}`), logit }))
const close = (a: number[], b: number[], eps = 1e-3) => a.forEach((x, i) => expect(Math.abs(x - b[i])).toBeLessThan(eps))

describe('softmax и temperature', () => {
  it('сумма 1, температура делает распределение острее или площе', () => {
    const p1 = softmax([2, 1, 0])
    expect(p1.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const cold = softmax([2, 1, 0], 0.5)
    const hot = softmax([2, 1, 0], 2)
    expect(cold[0]).toBeGreaterThan(p1[0])
    expect(hot[0]).toBeLessThan(p1[0])
    expect(softmax([0, -Infinity])).toEqual([1, 0])
  })

  it('энтропия: равномерное на 4 = 2 бита, вырожденное = 0', () => {
    expect(entropy([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2, 12)
    expect(entropy([1, 0, 0])).toBe(0)
    expect(entropy([0.5, 0.5], 'e')).toBeCloseTo(Math.LN2, 12)
  })
})

describe('маска structured output (notes-engines §3.10)', () => {
  it('logits [1.2, 0.3, 2.5, −0.4, 0.9, 3.1, 0.2, 1.0], разрешены 2, 5, 6 → 0.342, 0.623, 0.034', () => {
    const c = cands([1.2, 0.3, 2.5, -0.4, 0.9, 3.1, 0.2, 1.0])
    const allowed = [false, false, true, false, false, true, true, false]
    const d = distribution(c, { temperature: 1, topK: 0, topP: 1 }, allowed)
    close(d.finalProbs, [0, 0, 0.342, 0, 0, 0.623, 0.034, 0])
    expect(d.dropReason[0]).toBe('mask')
    expect(d.kept).toEqual(allowed)
    // probs — до маски (видно, что модель «хотела» и запрещённые токены)
    expect(d.probs[0]).toBeGreaterThan(0)
  })

  it('перенормировка как в notes-tokens §3.12: 0.35 / (0.35 + 0.05) = 0.875', () => {
    // '"' 0.40 запрещён, '4' 0.35 разрешён, 'forty' 0.05 запрещён, остальные разрешённые вместе 0.05
    const probs = [0.4, 0.35, 0.05, 0.05, 0.15]
    const c = cands(probs.map(Math.log))
    const d = distribution(c, { temperature: 1, topK: 0, topP: 1 }, [false, true, false, true, false])
    expect(d.finalProbs[1]).toBeCloseTo(0.875, 9)
  })
})

describe('top-k, top-p, min-p', () => {
  const probs = [0.5, 0.3, 0.15, 0.05]

  it('top-k', () => {
    expect(topKMask(probs, 2)).toEqual([true, true, false, false])
    expect(topKMask(probs, 0)).toEqual([true, true, true, true])
    expect(topKMask([0.4, 0.3, 0.3], 2)).toEqual([true, true, true]) // ничья на границе остаётся
  })

  it('top-p как в vLLM: накопленная сумма снизу ≤ 1 − p отбрасывается', () => {
    expect(topPMask(probs, 0.8)).toEqual([true, true, false, false])
    expect(topPMask(probs, 0.81)).toEqual([true, true, true, false])
    expect(topPMask(probs, 1)).toEqual([true, true, true, true])
    expect(topPMask(probs, 0.01)).toEqual([true, false, false, false])
  })

  it('min-p: порог minP · p_max', () => {
    expect(minPMask([0.6, 0.3, 0.1], 0.2)).toEqual([true, true, false])
  })

  it('порядок: min-p → top-k → top-p по уже обрезанному top-k распределению', () => {
    const c = cands([0.5, 0.3, 0.15, 0.05].map(Math.log))
    const d = distribution(c, { temperature: 1, topK: 3, topP: 0.6 })
    // после top-k=3: [0.5, 0.3, 0.15]/0.95; снизу: 0.158 ≤ 0.4 → прочь, 0.158 + 0.316 > 0.4 → стоп
    expect(d.kept).toEqual([true, true, false, false])
    expect(d.dropReason).toEqual([null, null, 'top-p', 'top-k'])
    close(d.finalProbs, [0.625, 0.375, 0, 0])
  })
})

describe('sample', () => {
  const c = cands([2, 1, 0.5, -1])

  it('greedy при T = 0: argmax среди разрешённых', () => {
    const r = sample(c, { temperature: 0, topK: 0, topP: 1 }, () => 0.99)
    expect(r.chosen).toBe(0)
    expect(r.kept).toEqual([true, false, false, false])
    const masked = sample(c, { temperature: 0, topK: 0, topP: 1 }, () => 0.99, [false, true, true, true])
    expect(masked.chosen).toBe(1)
    expect(masked.probs[0]).toBe(1) // модель «хотела» 0, но он запрещён
    expect(masked.u).toBeUndefined()
  })

  it('детерминирован при одинаковом rng и покрывает распределение', () => {
    const p = { temperature: 1, topK: 0, topP: 1 }
    const fp = distribution(c, p).finalProbs
    expect(sample(c, p, () => 0).chosen).toBe(0)
    expect(sample(c, p, () => fp[0] + 1e-9).chosen).toBe(1)
    expect(sample(c, p, () => 0.999999).chosen).toBe(3)
    let seed = 7
    const rng = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    const counts = [0, 0, 0, 0]
    for (let i = 0; i < 4000; i++) counts[sample(c, p, rng).chosen]++
    counts.forEach((n, i) => expect(Math.abs(n / 4000 - fp[i])).toBeLessThan(0.03))
  })

  it('все кандидаты запрещены → chosen = -1', () => {
    expect(sample(c, { temperature: 1, topK: 0, topP: 1 }, () => 0.5, [false, false, false, false]).chosen).toBe(-1)
  })

  it('argmax: ничьи → меньший индекс', () => {
    expect(argmax([1, 3, 3])).toBe(1)
    expect(argmax([1, 3, 3], [true, false, true])).toBe(2)
  })

  it('пустой список кандидатов — ошибка', () => {
    expect(() => distribution([], { temperature: 1, topK: 0, topP: 1 })).toThrow()
  })
})
