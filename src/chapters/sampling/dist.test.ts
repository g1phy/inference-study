import { describe, expect, test } from 'vitest'
import { distribution, entropy } from '@/engine/sampling'
import type { Candidate, Token } from '@/engine/types'
import { computeDist, pickRow } from './dist'
import { HEAD, TAIL } from './data'

/**
 * Аналитический расчёт с «хвостом» должен совпадать с движком, которому дали все 151 936 логитов.
 * Логиты хвоста в развёрнутом виде слегка различаются (шаг 1e-9), чтобы у top-k не было ничьих:
 * так ведут себя настоящие логиты.
 */
const dummy: Token = { id: 0, raw: '', bytes: [], text: '', special: false }
const expanded: Candidate[] = [
  ...HEAD.map((c) => ({ token: dummy, logit: c.logit })),
  ...Array.from({ length: TAIL.count }, (_, j) => ({ token: dummy, logit: TAIL.logit - 1e-9 * j })),
]
const n = HEAD.length

const CASES = [
  { temperature: 1, topK: 0, topP: 1, minP: 0 },
  { temperature: 0.7, topK: 0, topP: 1, minP: 0 },
  { temperature: 1.5, topK: 0, topP: 1, minP: 0 },
  { temperature: 2, topK: 0, topP: 1, minP: 0 },
  { temperature: 1, topK: 3, topP: 1, minP: 0 },
  { temperature: 1.5, topK: 20, topP: 1, minP: 0 },
  { temperature: 1, topK: 0, topP: 0.9, minP: 0 },
  { temperature: 2, topK: 0, topP: 0.9, minP: 0 },
  { temperature: 2, topK: 0, topP: 0.99, minP: 0 },
  { temperature: 1.2, topK: 0, topP: 0.5, minP: 0 },
  { temperature: 2, topK: 0, topP: 1, minP: 0.05 },
  { temperature: 1, topK: 5, topP: 0.95, minP: 0.02 },
  { temperature: 0.3, topK: 50, topP: 0.8, minP: 0 },
]

describe('computeDist = engine.distribution на развёрнутом словаре', () => {
  for (const params of CASES) {
    test(JSON.stringify(params), () => {
      const mine = computeDist(HEAD, TAIL, params)
      const eng = distribution(expanded, { temperature: params.temperature, topK: params.topK, topP: params.topP, minP: params.minP })
      for (let i = 0; i < n; i++) {
        expect(mine.rows[i].before).toBeCloseTo(eng.probs[i], 4)
        expect(mine.rows[i].after).toBeCloseTo(eng.finalProbs[i], 4)
        expect(mine.rows[i].kept).toBe(eng.kept[i] ? 1 : 0)
      }
      let tailBefore = 0
      let tailAfter = 0
      let tailKept = 0
      for (let j = n; j < expanded.length; j++) {
        tailBefore += eng.probs[j]
        tailAfter += eng.finalProbs[j]
        if (eng.kept[j]) tailKept++
      }
      const tail = mine.rows[n]
      expect(Math.abs(tail.before - tailBefore)).toBeLessThan(1e-4 * Math.max(1, tailBefore))
      expect(Math.abs(tail.after - tailAfter)).toBeLessThan(1e-4)
      expect(Math.abs(tail.kept - tailKept)).toBeLessThanOrEqual(1)
      expect(mine.entropyAfter).toBeCloseTo(eng.entropyBits, 3)
      expect(mine.entropyBefore).toBeCloseTo(entropy(eng.probs), 2)
    })
  }
})

test('greedy при T = 0: всё у самого большого логита', () => {
  const d = computeDist(HEAD, TAIL, { temperature: 0, topK: 0, topP: 1, minP: 0 })
  expect(d.rows[0].after).toBe(1)
  expect(d.entropyAfter).toBe(0)
  expect(pickRow(d, 0.999).row).toBe(0)
})

test('выбор по обратной функции распределения', () => {
  const d = computeDist(HEAD, TAIL, { temperature: 1, topK: 2, topP: 1, minP: 0 })
  expect(pickRow(d, 0).row).toBe(0)
  expect(pickRow(d, d.rows[0].after - 1e-9).row).toBe(0)
  expect(pickRow(d, d.rows[0].after + 1e-9).row).toBe(1)
  expect(pickRow(d, 0.9999999).row).toBe(1)
})

test('высокая температура без отсечений отдаёт почти всё хвосту, top-p возвращает кандидатов', () => {
  const hot = computeDist(HEAD, TAIL, { temperature: 1.5, topK: 0, topP: 1, minP: 0 })
  expect(hot.rows[n].before).toBeGreaterThan(0.5)
  const cut = computeDist(HEAD, TAIL, { temperature: 1.5, topK: 40, topP: 1, minP: 0 })
  expect(cut.rows[n].after).toBeLessThan(1e-3)
})

test('top-p срезает снизу не больше 1 − p: тяжёлый хвост остаётся частично (как в тексте главы)', () => {
  const p = 0.9
  for (const T of [1.2, 1.5]) {
    const raw = computeDist(HEAD, TAIL, { temperature: T, topK: 0, topP: 1, minP: 0 })
    const cut = computeDist(HEAD, TAIL, { temperature: T, topK: 0, topP: p, minP: 0 })
    const W = raw.rows[n].before
    expect(W).toBeGreaterThan(1 - p)
    // кандидаты целы, от хвоста отрезано ровно 1 − p (с точностью до одного токена хвоста)
    expect(cut.rows.slice(0, n).every((r) => r.kept === 1)).toBe(true)
    expect(cut.rows[n].kept * cut.tailEachBefore).toBeCloseTo(W - (1 - p), 4)
  }
  // «из 151 924 токенов хвоста остаётся около 128 тысяч»
  expect(TAIL.count).toBe(151_924)
  expect(Math.round(computeDist(HEAD, TAIL, { temperature: 1.5, topK: 0, topP: p, minP: 0 }).rows[n].kept / 1000)).toBe(128)
})
