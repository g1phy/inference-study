import { describe, expect, it } from 'vitest'
import {
  exponential,
  fmix32,
  hash64Words,
  hashString,
  hex32,
  logNormal,
  makeRng,
  mulberry32,
  normal,
  pick,
  pickWeighted,
  randInt,
  shortHash,
  shuffle,
} from './rng'

describe('mulberry32', () => {
  it('детерминирован и в [0, 1)', () => {
    const a = mulberry32(42)
    const b = mulberry32(42)
    const xs = Array.from({ length: 1000 }, () => a())
    expect(Array.from({ length: 1000 }, () => b())).toEqual(xs)
    expect(xs.every((x) => x >= 0 && x < 1)).toBe(true)
    expect(mulberry32(43)()).not.toBe(mulberry32(42)())
    // эталон: первое значение mulberry32(1) из оригинальной реализации
    expect(mulberry32(1)()).toBeCloseTo(0.6270739405881613, 15)
  })
  it('примерно равномерен', () => {
    const r = mulberry32(7)
    const bins = new Array(10).fill(0)
    for (let i = 0; i < 20000; i++) bins[Math.floor(r() * 10)]++
    expect(Math.min(...bins)).toBeGreaterThan(1800)
    expect(Math.max(...bins)).toBeLessThan(2200)
  })
  it('makeRng принимает строку', () => {
    expect(makeRng('seed')()).toBe(makeRng('seed')())
    expect(makeRng('seed')()).toBe(mulberry32(hashString('seed'))())
  })
})

describe('хэши', () => {
  it('hashString стабилен и различает похожие строки', () => {
    expect(hashString('Кошка')).toBe(hashString('Кошка'))
    expect(hashString('abc')).not.toBe(hashString('abd'))
    expect(hashString('')).toBe(fmix32(0x811c9dc5))
    expect(Number.isInteger(hashString('x'))).toBe(true)
    expect(hashString('x')).toBeGreaterThanOrEqual(0)
  })
  it('shortHash: 8 hex', () => {
    expect(shortHash('hello')).toMatch(/^[0-9a-f]{8}$/)
    expect(shortHash([1, 2, 3])).toMatch(/^[0-9a-f]{8}$/)
    expect(shortHash(5)).toBe(shortHash([5]))
    expect(shortHash([1, 2, 3])).not.toBe(shortHash([3, 2, 1]))
    expect(hex32(255)).toBe('000000ff')
  })
  it('hash64Words: порядок, длина и seed важны', () => {
    const a = hash64Words([1, 2, 3])
    expect(hash64Words([1, 2, 3])).toEqual(a)
    expect(hash64Words([3, 2, 1])).not.toEqual(a)
    expect(hash64Words([1, 2, 3, 0])).not.toEqual(a)
    expect(hash64Words([])).not.toEqual(hash64Words([0]))
    expect(hash64Words([1, 2, 3], 1337)).not.toEqual(a)
    // нет коллизий на 20 000 последовательных входов
    const seen = new Set<string>()
    for (let i = 0; i < 20000; i++) {
      const [h, l] = hash64Words([i, i + 1])
      seen.add(`${h}:${l}`)
    }
    expect(seen.size).toBe(20000)
  })
})

describe('выборки', () => {
  const r = mulberry32(3)
  it('randInt, pick, pickWeighted', () => {
    for (let i = 0; i < 500; i++) {
      const x = randInt(r, 3, 5)
      expect(x).toBeGreaterThanOrEqual(3)
      expect(x).toBeLessThanOrEqual(5)
    }
    expect(['a', 'b']).toContain(pick(r, ['a', 'b']))
    expect(() => pick(r, [])).toThrow()
    const counts = [0, 0, 0]
    for (let i = 0; i < 6000; i++) counts[pickWeighted(r, [1, 0, 3])]++
    expect(counts[1]).toBe(0)
    expect(counts[2] / counts[0]).toBeGreaterThan(2.5)
    expect(() => pickWeighted(r, [0, 0])).toThrow()
  })
  it('shuffle: перестановка, исходник не меняется, детерминированно', () => {
    const src = [1, 2, 3, 4, 5, 6, 7, 8]
    const s1 = shuffle(mulberry32(9), src)
    expect(src).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
    expect(s1.slice().sort((a, b) => a - b)).toEqual(src)
    expect(shuffle(mulberry32(9), src)).toEqual(s1)
  })
  it('normal, logNormal, exponential: моменты', () => {
    const g = mulberry32(11)
    const n = Array.from({ length: 20000 }, () => normal(g, 5, 2))
    const m = n.reduce((a, b) => a + b, 0) / n.length
    const sd = Math.sqrt(n.reduce((a, b) => a + (b - m) ** 2, 0) / n.length)
    expect(m).toBeCloseTo(5, 1)
    expect(sd).toBeCloseTo(2, 1)
    const ln = Array.from({ length: 20001 }, () => logNormal(g, 100, 0.5)).sort((a, b) => a - b)
    expect(ln[10000]).toBeGreaterThan(95)
    expect(ln[10000]).toBeLessThan(105)
    const e = Array.from({ length: 20000 }, () => exponential(g, 0.5))
    expect(e.reduce((a, b) => a + b, 0) / e.length).toBeCloseTo(2, 1)
  })
})
