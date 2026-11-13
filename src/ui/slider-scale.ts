/**
 * Шкала слайдера: линейная или логарифмическая. Чистые функции (тесты в slider-scale.test.ts).
 *
 * Логарифмическая шкала отображает значение на 0..LOG_RES позиций нативного <input type=range>.
 * Без доработки у неё две беды:
 * 1) стрелка двигает позицию на 1/1000 диапазона, округление по step возвращает то же значение,
 *    и бегунок не едет (batch 1 → позиция 1 → 1,007 → снова 1);
 * 2) позиции ложатся на «некруглые» числа (7 996, 15 991), и просьбу подписи «поставьте 16 000»
 *    выполнить нельзя.
 * Поэтому значения прилипают к «круглой» сетке (1, 2, 5 × 10^k не мельче расстояния между позициями)
 * и к степеням двойки (512, 4 096, 8 192), а стрелки идут по позициям, пока значение не сменится.
 */

export const LOG_RES = 1000

export interface SliderScaleOptions {
  min: number
  max: number
  /** Шаг значения (по умолчанию 1). */
  step?: number
  scale?: 'linear' | 'log'
}

export interface SliderScale {
  log: boolean
  /** Значение → позиция нативного ползунка. */
  toPos(v: number): number
  /** Позиция → значение (с прилипанием к сетке, в пределах [min, max]). */
  fromPos(p: number): number
  /** Значение в пределах [min, max]. */
  clamp(v: number): number
  /** Атрибуты нативного <input type=range>. */
  input: { min: number; max: number; step: number }
}

/** Наименьшее «круглое» число 1, 2, 5 × 10^k, не меньшее x. */
export function niceCeil(x: number): number {
  if (!(x > 0) || !Number.isFinite(x)) return 0
  const p = 10 ** Math.floor(Math.log10(x))
  for (const m of [1, 2, 5, 10]) if (m * p >= x * (1 - 1e-9)) return m * p
  return 10 * p
}

function isPow10(x: number): boolean {
  if (!(x > 0)) return false
  const l = Math.log10(x)
  return Math.abs(l - Math.round(l)) < 1e-9
}

function decimalsOf(x: number): number {
  if (!Number.isFinite(x)) return 0
  const s = String(x)
  if (s.includes('e-')) return Number(s.split('e-')[1]) || 0
  return (s.split('.')[1] ?? '').length
}

export function makeSliderScale({ min, max, step = 1, scale = 'linear' }: SliderScaleOptions): SliderScale {
  const lo = Math.min(min, max)
  const hi = Math.max(min, max)
  const clamp = (v: number) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : lo)
  const log = scale === 'log' && lo > 0
  const decimals = decimalsOf(step)
  const round = (v: number, g: number) => (g > 0 ? Number((Math.round(v / g) * g).toFixed(Math.max(decimals, decimalsOf(g)))) : v)

  if (!log) {
    return {
      log,
      clamp,
      toPos: (v) => clamp(v),
      fromPos: (p) => clamp(round(p, step)),
      input: { min: lo, max: hi, step },
    }
  }

  const lnMin = Math.log(lo)
  const lnMax = Math.log(Math.max(hi, lo * (1 + 1e-9)))
  const span = lnMax - lnMin
  /** Относительное изменение значения на одну позицию. */
  const rel = span / LOG_RES
  const pos = (v: number) => ((Math.log(clamp(v)) - lnMin) / span) * LOG_RES
  const roundGrid = isPow10(step)

  const fromPos = (p: number): number => {
    if (p <= 0) return lo
    if (p >= LOG_RES) return hi
    const raw = Math.exp(lnMin + (p / LOG_RES) * span)
    // степень двойки рядом с позицией (не дальше половины позиции): 512, 4 096, 8 192 достижимы
    if (raw >= 16) {
      const p2 = 2 ** Math.round(Math.log2(raw))
      const onStep = step <= 0 || Math.abs(p2 / step - Math.round(p2 / step)) < 1e-9
      if (onStep && p2 >= lo && p2 <= hi && Math.abs(pos(p2) - p) <= 0.5) return p2
    }
    // круглая сетка не мельче расстояния между соседними позициями: каждая её точка достижима
    const g = roundGrid ? Math.max(step, niceCeil(raw * rel)) : step
    return clamp(round(raw, g))
  }

  return { log, clamp, toPos: pos, fromPos, input: { min: 0, max: LOG_RES, step: 1 } }
}

/**
 * Следующее значение по стрелке на логарифмической шкале: идём по позициям в сторону dir,
 * пока значение не изменится. from: с какой позиции и значения начинать (продолжение после
 * прошлого нажатия, если родитель округлил значение обратно). Возвращает null, если упёрлись в край.
 */
export function logArrowStep(
  s: SliderScale,
  value: number,
  dir: 1 | -1,
  from?: { pos: number; value: number },
): { pos: number; value: number } | null {
  const cur = from ? from.value : s.clamp(value)
  const exact = s.toPos(s.clamp(value))
  let p = from ? from.pos : dir > 0 ? Math.floor(exact + 1e-9) : Math.ceil(exact - 1e-9)
  let next = cur
  while (next === cur) {
    const q = p + dir
    if (q < 0 || q > LOG_RES) return null
    p = q
    next = s.fromPos(p)
    // значение по стрелке не идёт назад (сетка грубее позиции): ищем дальше
    if ((dir > 0 && next < cur) || (dir < 0 && next > cur)) next = cur
  }
  return { pos: p, value: next }
}
