import { describe, expect, it } from 'vitest'
import { barPath, fontCss, linearTicks, logDomain, logTicks, makeScale, nearestIndex, niceDomain, niceScale, spreadLabels } from './chart-utils'

describe('chart-utils', () => {
  it('красивые тики и домен', () => {
    expect(linearTicks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100])
    expect(linearTicks(0, 1, 4)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1])
    expect(niceDomain(3, 97, 5)).toEqual([0, 100])
  })
  it('niceScale: края домена на тиках', () => {
    expect(niceScale(3, 1373, 4)).toEqual({ domain: [0, 1500], ticks: [0, 500, 1000, 1500], step: 500 })
    expect(niceScale(0, 1, 5).ticks).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1])
  })
  it('логарифмический домен без лишней декады', () => {
    expect(logDomain(0.5, 10000)).toEqual([0.5, 10000])
    expect(logDomain(1.675, 989)).toEqual([1, 1000])
    expect(logDomain(3, 70)).toEqual([2, 100])
  })
  it('логарифмические тики', () => {
    expect(logTicks(1, 1000)).toEqual([1, 10, 100, 1000])
    expect(logTicks(10, 50)).toEqual([10, 20, 50])
  })
  it('шкалы и обратное преобразование', () => {
    const s = makeScale('log', [1, 1000], [0, 300])
    expect(s(10)).toBeCloseTo(100)
    expect(s.invert(200)).toBeCloseTo(100)
    const l = makeScale('linear', [0, 10], [100, 0])
    expect(l(5)).toBe(50)
  })
  it('ближайший индекс и разнос подписей', () => {
    expect(nearestIndex([1, 4, 9], 5)).toBe(1)
    expect(nearestIndex([1, 4, 9], 7)).toBe(2)
    const ys = spreadLabels([10, 12, 50], 14, 0, 100)
    expect(ys[1] - ys[0]).toBeGreaterThanOrEqual(14)
    expect(ys[2]).toBe(50)
  })
  it('путь столбика: без экспоненты и одной структуры для анимации', () => {
    const shape = (d: string) => d.replace(/-?[\d.]+/g, '#')
    const tiny = barPath(20, 10, 2.4868995751603507e-14, 12, 'right')
    expect(tiny).not.toMatch(/e[-+]/)
    const full = barPath(20, 10, 300, 12, 'right')
    const left = barPath(20, 10, 300, 12, 'left')
    const up = barPath(20, 10, 12, 1.075e-12, 'up')
    const down = barPath(20, 10, 12, 80, 'down')
    for (const d of [full, left, up, down]) expect(shape(d)).toBe(shape(tiny))
    expect(full).toBe('M20,10L316,10A4,4 0 0 1 320,14L320,18A4,4 0 0 1 316,22L20,22Z')
  })
})

describe('fontCss', () => {
  it('раскладывает строку шрифта canvas на длинные свойства', () => {
    expect(fontCss('600 12px "Onest Variable", sans-serif')).toEqual({ fontWeight: '600', fontSize: '12px', fontFamily: '"Onest Variable", sans-serif' })
    expect(fontCss('11.5px "JetBrains Mono Variable", monospace')).toEqual({ fontSize: '11.5px', fontFamily: '"JetBrains Mono Variable", monospace' })
    expect(fontCss('italic 500 13px/18px Onest')).toEqual({ fontStyle: 'italic', fontWeight: '500', fontSize: '13px', lineHeight: '18px', fontFamily: 'Onest' })
  })
})
