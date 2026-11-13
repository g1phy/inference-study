import { expect, test } from 'vitest'
import { getModel } from '@/engine/data/models'
import { moeExpertsTouched } from '@/engine/perf'
import { routeToken, routerOf, touchedExperts } from './moe'
import { lensAt } from './lens'
import { HEAD } from '../sampling/data'

test('роутер выбирает k разных экспертов, веса в сумме 1', () => {
  for (const id of [1, 42, 73239, 151644]) {
    const r = routeToken(id, 3, 128, 8)
    expect(new Set(r.chosen.map((c) => c.expert)).size).toBe(8)
    expect(r.chosen.reduce((s, c) => s + c.weight, 0)).toBeCloseTo(1, 12)
    for (let i = 1; i < r.chosen.length; i++) expect(r.chosen[i].weight).toBeLessThanOrEqual(r.chosen[i - 1].weight)
  }
})

test('число задетых экспертов на 64 токенах близко к ожиданию при равномерном выборе', () => {
  const ids = Array.from({ length: 64 }, (_, i) => 1000 + i * 37)
  const s = touchedExperts(ids, 63, 5, 128, 8)
  const expected = moeExpertsTouched(getModel('qwen3-30b-a3b'), 64)
  expect(Math.abs(s.size - expected)).toBeLessThan(8)
})

test('DeepSeek-V3: top-8 внутри 4 лучших групп из 8 по 32 эксперта, веса в сумме 1', () => {
  const r = routerOf('deepseek-v3')
  expect(r.firstMoeLayer).toBe(4)
  for (const id of [1, 42, 73239, 151644]) {
    const route = routeToken(id, 10, 256, 8, r)
    expect(route.groups).toHaveLength(4)
    expect(new Set(route.chosen.map((c) => c.expert)).size).toBe(8)
    for (const c of route.chosen) expect(route.groups).toContain(Math.floor(c.expert / 32))
    expect(route.chosen.reduce((s, c) => s + c.weight, 0)).toBeCloseTo(1, 12)
  }
  // по симметрии каждый эксперт выбирается с вероятностью k/E, поэтому ожидание задетых то же
  const ids = Array.from({ length: 64 }, (_, i) => 2000 + i * 53)
  const s = touchedExperts(ids, 63, 12, 256, 8, r)
  const expected = moeExpertsTouched(getModel('deepseek-v3'), 64)
  expect(Math.abs(s.size - expected)).toBeLessThan(16)
})

test('gpt-oss (top-k, потом softmax по выбранным) даёт те же веса, что softmax по всем и нормировка', () => {
  const a = routeToken(777, 5, 128, 4, routerOf('gpt-oss-120b'))
  const b = routeToken(777, 5, 128, 4, routerOf('qwen3-30b-a3b'))
  expect(a.chosen.map((c) => c.expert)).toEqual(b.chosen.map((c) => c.expert))
  a.chosen.forEach((c, i) => expect(c.weight).toBeCloseTo(b.chosen[i].weight, 12))
})

test('logit lens на последнем слое совпадает с логитами главы о сэмплинге', () => {
  const last = lensAt(1)
  for (const text of [' Париж', ' город', ' это', ' Лион']) {
    const lens = last.find((c) => c.text === text)!.logit
    const head = HEAD.find((c) => c.text === text)!.logit
    expect(Math.abs(lens - head)).toBeLessThan(0.05)
  }
})
