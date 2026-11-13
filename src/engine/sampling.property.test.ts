/**
 * Независимая проверка sampling.ts против эталонной реализации порядка vLLM V1 (vllm/v1/sample/sampler.py +
 * ops/topk_topp_sampler.py apply_top_k_top_p + MinPLogitsProcessor), переписанной здесь отдельно, в float64:
 *   маска грамматики (−∞) → logits / T → min-p (p < min_p·p_max → −∞) → top-k (z < k-е по величине → −∞)
 *   → top-p (сортировка по возрастанию, softmax, cumsum ≤ 1 − p → −∞, последний остаётся) → softmax.
 * Плюс граничные случаи: точная граница top-p, ничьи top-k, T = 0, дробный top-k, частоты сэмплинга.
 */
import { describe, expect, it } from 'vitest'
import { mulberry32 } from './rng'
import { distribution, sample } from './sampling'
import type { Candidate, Token } from './types'

const tk = (id: number): Token => ({ id, raw: `t${id}`, bytes: [], text: `t${id}`, special: false })
const cands = (logits: number[]): Candidate[] => logits.map((logit, i) => ({ token: tk(i), logit }))

function softmaxRef(z: number[]): number[] {
  const m = Math.max(...z)
  if (m === -Infinity) return z.map(() => 0)
  const e = z.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** Эталон vLLM: возвращает финальное распределение (0 у выкинутых). */
function vllmReference(logits: number[], allowed: boolean[], T: number, topK: number, topP: number, minP: number): number[] {
  const n = logits.length
  let z = logits.map((l, i) => (allowed[i] ? l / T : -Infinity))
  if (minP > 0) {
    const p = softmaxRef(z)
    const pmax = Math.max(...p)
    z = z.map((v, i) => (p[i] < minP * pmax ? -Infinity : v))
  }
  if (topK > 0 && topK < n) {
    const sorted = [...z].sort((a, b) => a - b)
    const th = sorted[n - topK]
    z = z.map((v) => (v < th ? -Infinity : v))
  }
  if (topP < 1) {
    const idx = z.map((_, i) => i).sort((a, b) => z[a] - z[b])
    const ps = softmaxRef(idx.map((i) => z[i]))
    let cum = 0
    const drop: number[] = []
    for (let j = 0; j < n - 1; j++) {
      cum += ps[j]
      if (cum <= 1 - topP) drop.push(idx[j])
    }
    for (const i of drop) z[i] = -Infinity
  }
  return softmaxRef(z)
}

describe('distribution() = эталон порядка vLLM V1 на случайных логитах и параметрах', () => {
  it('2000 случайных случаев: kept и finalProbs совпадают', () => {
    const rng = mulberry32(31337)
    let compared = 0
    for (let t = 0; t < 2000; t++) {
      const n = 2 + Math.floor(rng() * 12)
      const logits = Array.from({ length: n }, () => (rng() - 0.5) * 10)
      const allowed = logits.map(() => rng() < 0.8)
      if (!allowed.some(Boolean)) allowed[0] = true
      const T = 0.05 + rng() * 2
      const topK = rng() < 0.4 ? 0 : 1 + Math.floor(rng() * n)
      const topP = rng() < 0.4 ? 1 : 0.05 + rng() * 0.95
      const minP = rng() < 0.6 ? 0 : rng() * 0.5
      const ref = vllmReference(logits, allowed, T, topK, topP, minP)
      const d = distribution(cands(logits), { temperature: T, topK, topP, minP }, allowed)
      d.finalProbs.forEach((p, i) => expect(Math.abs(p - ref[i])).toBeLessThan(1e-12))
      expect(d.kept).toEqual(ref.map((p) => p > 0))
      expect(d.finalProbs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      // причины отсечения согласованы с kept
      d.dropReason.forEach((r, i) => expect(r === null).toBe(d.kept[i]))
      compared++
    }
    expect(compared).toBe(2000)
  })
})

describe('граничные случаи', () => {
  // логиты = ln p: softmax восстанавливает p с точностью ~1e-16
  const P = [0.5, 0.3, 0.2]
  const L = P.map(Math.log)
  const keptFor = (topP: number) => distribution(cands(L), { temperature: 1, topK: 0, topP }).kept

  it('top-p на точной границе: накопленная масса ровно 1 − p отбрасывается (наименьшее множество с массой ≥ p)', () => {
    expect(keptFor(0.8)).toEqual([true, true, false]) // 0.2 ≤ 1 − 0.8 → прочь; осталось ровно 0.8
    expect(keptFor(0.8 + 1e-9)).toEqual([true, true, true])
    expect(keptFor(0.5)).toEqual([true, false, false]) // 0.2 + 0.3 ≤ 0.5
    expect(keptFor(0.5 + 1e-9)).toEqual([true, true, false])
    expect(keptFor(1e-9)).toEqual([true, false, false]) // самый вероятный остаётся всегда
  })

  it('top-k: ничьи на границе остаются; дробный k округляется вниз (раньше выкидывалось всё)', () => {
    const d = distribution(cands([2, 1, 1, 0]), { temperature: 1, topK: 2, topP: 1 })
    expect(d.kept).toEqual([true, true, true, false])
    const frac = distribution(cands([2, 1, 0.5, 0]), { temperature: 1, topK: 2.7, topP: 1 })
    expect(frac.kept).toEqual([true, true, false, false])
    expect(frac.chosen).toBe(0)
  })

  it('min-p > 1 не выкидывает самый вероятный токен (vLLM ограничивает min_p ∈ [0, 1])', () => {
    const d = distribution(cands([2, 1, 0]), { temperature: 1, topK: 0, topP: 1, minP: 3 })
    expect(d.kept).toEqual([true, false, false])
    expect(d.chosen).toBe(0)
  })

  it('T = 0: greedy — argmax среди разрешённых, ничья → меньший индекс, rng не используется, энтропия 0', () => {
    const c = cands([1, 3, 3, 2])
    const calls: number[] = []
    const rng = () => {
      calls.push(1)
      return 0.999
    }
    const r = sample(c, { temperature: 0, topK: 1, topP: 0.1 }, rng)
    expect(r.chosen).toBe(1)
    expect(r.finalProbs).toEqual([0, 1, 0, 0])
    expect(r.entropyBits).toBe(0)
    expect(calls).toHaveLength(0)
    expect(sample(c, { temperature: 0, topK: 0, topP: 1 }, rng, [true, false, true, true]).chosen).toBe(2)
    // T → 0+ сходится к greedy
    const tiny = distribution(c, { temperature: 1e-6, topK: 0, topP: 1 }, [true, false, true, true])
    expect(tiny.finalProbs[2]).toBeCloseTo(1, 12)
  })

  it('маска применяется до temperature/top-k/top-p: top-k=1 выбирает лучший РАЗРЕШЁННЫЙ', () => {
    const d = distribution(cands([5, 4, 3]), { temperature: 1, topK: 1, topP: 1 }, [false, true, true])
    expect(d.kept).toEqual([false, true, false])
    expect(d.dropReason).toEqual(['mask', null, 'top-k'])
  })

  it('частоты sample() сходятся к finalProbs (mulberry32, 20 000 выборок)', () => {
    const c = cands([1.5, 1, 0.2, -0.5, -2])
    const params = { temperature: 0.8, topK: 4, topP: 0.95, minP: 0.05 }
    const fp = distribution(c, params).finalProbs
    const rng = mulberry32(7)
    const counts = fp.map(() => 0)
    const N = 20000
    for (let i = 0; i < N; i++) counts[sample(c, params, rng).chosen]++
    counts.forEach((k, i) => {
      if (fp[i] === 0) expect(k).toBe(0)
      else expect(Math.abs(k / N - fp[i])).toBeLessThan(4 * Math.sqrt((fp[i] * (1 - fp[i])) / N) + 1e-3)
    })
  })

  it('обратная функция распределения: u на границах интервалов', () => {
    const c = cands([0, 0, 0, 0]) // по 0.25
    const p = { temperature: 1, topK: 0, topP: 1 }
    expect(sample(c, p, () => 0).chosen).toBe(0)
    expect(sample(c, p, () => 0.25 - 1e-12).chosen).toBe(0)
    expect(sample(c, p, () => 0.25).chosen).toBe(1)
    expect(sample(c, p, () => 0.9999999999).chosen).toBe(3)
  })
})
