import { describe, expect, it } from 'vitest'
import { LOG_RES, logArrowStep, makeSliderScale, niceCeil } from './slider-scale'

/** Пройти стрелкой n раз (как Slider: продолжая с прошлой позиции). */
function press(min: number, max: number, start: number, dir: 1 | -1, n: number, step = 1, parentRound?: (v: number) => number) {
  const s = makeSliderScale({ min, max, step, scale: 'log' })
  let value = start
  let from: { pos: number; value: number } | undefined
  const seen: number[] = []
  for (let i = 0; i < n; i++) {
    const r = logArrowStep(s, value, dir, from)
    if (!r) break
    from = r
    value = parentRound ? parentRound(r.value) : r.value
    seen.push(value)
  }
  return { value, seen }
}

describe('логарифмический слайдер', () => {
  it('niceCeil', () => {
    expect(niceCeil(44)).toBe(50)
    expect(niceCeil(0.7)).toBe(1)
    expect(niceCeil(100)).toBe(100)
    expect(niceCeil(101)).toBe(200)
  })

  it('стрелка на min = 1, step = 1 двигает значение на каждом нажатии', () => {
    const { seen } = press(1, 1024, 1, 1, 5)
    expect(seen).toEqual([2, 3, 4, 5, 6])
    const back = press(1, 1024, 6, -1, 5)
    expect(back.seen).toEqual([5, 4, 3, 2, 1])
  })

  it('каждое нажатие меняет значение и на больших числах', () => {
    const { seen } = press(64, 32768, 16000, 1, 10)
    expect(new Set(seen).size).toBe(10)
    for (let i = 1; i < seen.length; i++) expect(seen[i]).toBeGreaterThan(seen[i - 1])
  })

  it('родитель округляет значение обратно: нажатия накапливаются', () => {
    // глава округляет до 10 (как «Средняя длина ответа»): стрелка всё равно сдвигает значение
    const { value } = press(32, 4096, 30, 1, 8, 1, (v) => Math.round(v / 10) * 10)
    expect(value).toBeGreaterThan(30)
  })

  it('круглые значения и степени двойки достижимы', () => {
    const s = makeSliderScale({ min: 512, max: 131072, scale: 'log' })
    const values = new Set<number>()
    for (let p = 0; p <= LOG_RES; p++) values.add(s.fromPos(p))
    for (const v of [1000, 2000, 4000, 8000, 16000, 32000, 1024, 4096, 8192, 16384, 32768]) expect(values.has(v)).toBe(true)
    const b = makeSliderScale({ min: 1, max: 1024, scale: 'log' })
    const bv = new Set<number>()
    for (let p = 0; p <= LOG_RES; p++) bv.add(b.fromPos(p))
    for (const v of [1, 2, 8, 16, 64, 100, 128, 256, 500, 512, 1024]) expect(bv.has(v)).toBe(true)
  })

  it('края шкалы дают точные min и max, значение зажато в пределы', () => {
    const s = makeSliderScale({ min: 32, max: 4096, scale: 'log' })
    expect(s.fromPos(0)).toBe(32)
    expect(s.fromPos(LOG_RES)).toBe(4096)
    expect(s.clamp(4100)).toBe(4096)
    expect(s.toPos(5000)).toBeCloseTo(LOG_RES)
  })

  it('дробный шаг: значения кратны шагу', () => {
    const s = makeSliderScale({ min: 0.1, max: 50, step: 0.01, scale: 'log' })
    for (const p of [1, 137, 500, 999]) {
      const v = s.fromPos(p)
      expect(Math.abs(v * 100 - Math.round(v * 100))).toBeLessThan(1e-6)
    }
    const r = logArrowStep(s, 0.3, 1)
    expect(r && r.value).toBeGreaterThan(0.3)
  })

  it('линейная шкала не изменилась', () => {
    const s = makeSliderScale({ min: 0, max: 2, step: 0.05 })
    expect(s.fromPos(0.733)).toBe(0.75)
    expect(s.input).toEqual({ min: 0, max: 2, step: 0.05 })
  })
})
