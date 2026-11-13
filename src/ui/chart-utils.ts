/** Шкалы, тики и геометрия для SVG-графиков ui-kit. Чистые функции. */

export type ScaleKind = 'linear' | 'log'

export interface Scale {
  (v: number): number
  invert(px: number): number
  domain: [number, number]
  range: [number, number]
  kind: ScaleKind
}

export function makeScale(kind: ScaleKind, domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain
  const [r0, r1] = range
  const f = kind === 'log' ? (v: number) => Math.log10(Math.max(v, Number.MIN_VALUE)) : (v: number) => v
  const fi = kind === 'log' ? (v: number) => 10 ** v : (v: number) => v
  const a = f(d0)
  const b = f(d1)
  const span = b - a || 1
  const s = ((v: number) => r0 + ((f(v) - a) / span) * (r1 - r0)) as Scale
  s.invert = (px: number) => fi(a + ((px - r0) / (r1 - r0 || 1)) * span)
  s.domain = domain
  s.range = range
  s.kind = kind
  return s
}

/** «Красивый» шаг: 1, 2, 5 × 10^k. */
export function niceStep(span: number, count: number): number {
  const raw = Math.abs(span) / Math.max(1, count)
  if (!Number.isFinite(raw) || raw === 0) return 1
  const pow = 10 ** Math.floor(Math.log10(raw))
  const err = raw / pow
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1
  return mult * pow
}

/** Расширить диапазон до «красивых» границ. */
export function niceDomain(min: number, max: number, count = 5): [number, number] {
  if (min === max) {
    const d = Math.abs(min) || 1
    return [min - d / 2, max + d / 2]
  }
  const step = niceStep(max - min, count)
  return [Math.floor(min / step) * step, Math.ceil(max / step) * step]
}

/** Тики внутри [min, max] с «красивым» шагом. */
export function linearTicks(min: number, max: number, count = 5): number[] {
  const step = niceStep(max - min, count)
  const out: number[] = []
  const start = Math.ceil(min / step - 1e-9) * step
  for (let v = start; v <= max + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toPrecision(12)))
  return out
}

/** Тики логарифмической шкалы: степени 10, при узком диапазоне ещё 2× и 5×. */
export function logTicks(min: number, max: number): number[] {
  const lo = Math.floor(Math.log10(min))
  const hi = Math.ceil(Math.log10(max))
  const decades: number[] = []
  for (let k = lo; k <= hi; k++) decades.push(10 ** k)
  const inRange = (v: number) => v >= min * 0.999 && v <= max * 1.001
  let ticks = decades.filter(inRange)
  if (ticks.length < 3) {
    const dense: number[] = []
    for (let k = lo; k <= hi; k++) for (const m of [1, 2, 5]) dense.push(m * 10 ** k)
    ticks = dense.filter(inRange)
  }
  return ticks
}

/** Расширить лог-диапазон до ближайших «красивых» значений 1, 2, 5 × 10^k (без лишней декады). */
export function logDomain(min: number, max: number): [number, number] {
  const lo = Math.max(min, Number.MIN_VALUE)
  const hi = Math.max(max, lo * 1.0001)
  const steps = [1, 2, 5, 10]
  const floorNice = (v: number) => {
    const p = 10 ** Math.floor(Math.log10(v))
    let best = p
    for (const m of steps) if (m * p <= v * (1 + 1e-9)) best = m * p
    return best
  }
  const ceilNice = (v: number) => {
    const p = 10 ** Math.floor(Math.log10(v))
    for (const m of steps) if (m * p >= v * (1 - 1e-9)) return m * p
    return 10 * p
  }
  return [floorNice(lo), ceilNice(hi)]
}

/**
 * Линейная шкала с «красивыми» тиками: границы домена совпадают с крайними тиками.
 * niceScale(3, 1373, 4) → { domain: [0, 1500], ticks: [0, 500, 1000, 1500] }.
 */
export function niceScale(min: number, max: number, count = 5): { domain: [number, number]; ticks: number[]; step: number } {
  if (min === max) {
    const d = Math.abs(min) || 1
    min -= d / 2
    max += d / 2
  }
  // наименьший «красивый» шаг, при котором интервалов не больше count
  const raw = (max - min) / Math.max(1, count)
  const pow = 10 ** Math.floor(Math.log10(raw))
  const step = ([1, 2, 5, 10].map((m) => m * pow).find((v) => v >= raw * (1 - 1e-9)) ?? 10 * pow)
  const d0 = Math.floor(min / step + 1e-9) * step
  const d1 = Math.ceil(max / step - 1e-9) * step
  const ticks: number[] = []
  for (let v = d0; v <= d1 + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toPrecision(12)))
  return { domain: [d0, d1], ticks, step }
}

/** Число для SVG-пути: два знака после запятой и никогда экспонента («2.4e-14» ломает разбор пути). */
function n2(v: number): string {
  const r = Math.round(v * 100) / 100
  return Object.is(r, -0) || Math.abs(r) < 0.005 ? '0' : String(r)
}

/**
 * Путь столбика со скруглением только на конце данных (dataviz: 4px на конце, квадрат у базовой линии).
 * dir: куда растёт столбик от базовой линии.
 *
 * Структура команд у всех путей одна и та же (M L A L A L Z, дуги с одинаковыми флагами), в том числе
 * у пустого столбика (радиус 0) и у столбиков разного направления: motion плавно переводит один путь
 * в другой, интерполируя числа, и при разной структуре получал бы невалидный путь (ошибки в консоли).
 */
export function barPath(x: number, y: number, w: number, h: number, dir: 'right' | 'left' | 'up' | 'down', radius = 4): string {
  const W = Math.max(0, w)
  const H = Math.max(0, h)
  // совсем крошечный столбик рисуем прямоугольником (радиус 0), но в той же форме пути
  const r = W < 0.5 || H < 0.5 ? 0 : Math.max(0, Math.min(radius, W / 2, H / 2))
  const R = n2(r)
  const pts = ((): Array<[number, number]> => {
    // обход всегда по часовой стрелке (флаг дуги 1): шесть опорных точек, скругление на конце данных
    switch (dir) {
      case 'right':
        return [[x, y], [x + W - r, y], [x + W, y + r], [x + W, y + H - r], [x + W - r, y + H], [x, y + H]]
      case 'left':
        return [[x + W, y + H], [x + r, y + H], [x, y + H - r], [x, y + r], [x + r, y], [x + W, y]]
      case 'up':
        return [[x, y + H], [x, y + r], [x + r, y], [x + W - r, y], [x + W, y + r], [x + W, y + H]]
      case 'down':
        return [[x + W, y], [x + W, y + H - r], [x + W - r, y + H], [x + r, y + H], [x, y + H - r], [x, y]]
    }
  })()
  const p = pts.map(([a, b]) => `${n2(a)},${n2(b)}`)
  return `M${p[0]}L${p[1]}A${R},${R} 0 0 1 ${p[2]}L${p[3]}A${R},${R} 0 0 1 ${p[4]}L${p[5]}Z`
}

/** Индекс ближайшего значения в отсортированном массиве. */
export function nearestIndex(sorted: number[], v: number): number {
  if (sorted.length === 0) return -1
  let lo = 0
  let hi = sorted.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] < v) lo = mid
    else hi = mid
  }
  return Math.abs(sorted[lo] - v) <= Math.abs(sorted[hi] - v) ? lo : hi
}

/** Разнести подписи по вертикали (минимальный зазор gap), сохраняя порядок. */
export function spreadLabels(ys: number[], gap: number, min: number, max: number): number[] {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y)
  const out = new Array<number>(ys.length)
  let prev = -Infinity
  for (const o of order) {
    const y = Math.max(o.y, prev + gap, min)
    out[o.i] = y
    prev = y
  }
  // если вылезли за низ, сдвигаем всё вверх
  const overflow = Math.max(0, ...out.map((y) => y - max))
  if (overflow > 0) {
    let next = Infinity
    for (const o of [...order].reverse()) {
      const y = Math.min(out[o.i] - overflow, next - gap)
      out[o.i] = y
      next = y
    }
  }
  return out
}

/**
 * Строка шрифта как у canvas ('600 12px "Onest Variable", sans-serif') → длинные CSS-свойства.
 * Короткое `font` в style вместе с `fontVariantNumeric` React в dev ругает, когда `font` меняется между
 * рендерами (узкий график переключил размер): «Updating a style property during rerender (font) when a
 * conflicting property is set (fontVariantNumeric)». Длинные свойства друг другу не мешают.
 */
export function fontCss(font: string): { fontStyle?: string; fontWeight?: string; fontSize: string; lineHeight?: string; fontFamily: string } {
  const m = /^\s*(?:(italic|oblique|normal)\s+)?(?:(\d{3}|bold|normal)\s+)?([\d.]+px)(?:\/([\d.]+(?:px)?))?\s+(.+?)\s*$/.exec(font)
  if (!m) return { fontSize: '12px', fontFamily: font }
  const [, style, weight, size, lh, family] = m
  return {
    ...(style ? { fontStyle: style } : {}),
    ...(weight ? { fontWeight: weight } : {}),
    fontSize: size,
    ...(lh ? { lineHeight: lh } : {}),
    fontFamily: family,
  }
}
