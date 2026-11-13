import { expect, test } from 'vitest'
import { mulberry32 } from '@/engine/rng'
import { END, GRAPH, MAX_STEPS, TOY_PREFIX, joinWords, runToy } from './toylm'

test('у каждого перехода есть состояние, у каждого состояния есть кандидаты', () => {
  for (const [state, edges] of Object.entries(GRAPH)) {
    expect(edges.length, state).toBeGreaterThan(0)
    for (const e of edges) expect(e.next === END || e.next in GRAPH, `${state} → ${e.next}`).toBe(true)
  }
})

test('greedy-путь даёт задуманную фразу', () => {
  const run = runToy([0.5], { temperature: 0, topK: 0, topP: 1 })
  expect(run.finished).toBe(true)
  expect(TOY_PREFIX + joinWords(run.words)).toBe('KV cache хранит ключи и значения всех прошлых токенов, чтобы не пересчитывать их заново.')
})

test('при любых seed генерация укладывается в лимит шагов и не ломается', () => {
  for (let seed = 1; seed <= 300; seed++) {
    const rng = mulberry32(seed)
    const us = Array.from({ length: MAX_STEPS }, () => rng())
    for (const temperature of [0.7, 1.5]) {
      const run = runToy(us, { temperature, topK: 0, topP: 1 })
      expect(run.steps.length).toBeLessThanOrEqual(MAX_STEPS)
      expect(run.words.length).toBe(run.steps.length)
    }
  }
})
