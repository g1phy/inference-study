import { expect, test } from 'vitest'
import { BARCODE_DIMS, embeddingRow, pseudoVector } from './vectors'

test('строка таблицы детерминирована: один ID даёт один и тот же вектор', () => {
  const a = embeddingRow(84648)
  const b = embeddingRow(84648)
  expect(a).toEqual(b)
  expect(a).toHaveLength(BARCODE_DIMS)
  expect(embeddingRow(6500)).not.toEqual(a)
})

test('разные семейства (salt) независимы, значения похожи на N(0, 1)', () => {
  expect(pseudoVector(7, 64, 0)).not.toEqual(pseudoVector(7, 64, 1))
  const xs = Array.from({ length: 200 }, (_, i) => embeddingRow(i, 64)).flat()
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length)
  expect(Math.abs(mean)).toBeLessThan(0.05)
  expect(Math.abs(sd - 1)).toBeLessThan(0.05)
})
