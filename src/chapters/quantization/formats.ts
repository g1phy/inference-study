/**
 * Форматы чисел для главы «Квантизация»: раскладка бит и точное округление.
 *
 * Плавающая точка: значение нормального числа = (−1)^S · 2^(E − bias) · (1 + M / 2^m),
 * субнормального (E = 0) = (−1)^S · 2^(1 − bias) · (M / 2^m). Округление к ближайшему,
 * при равенстве к чётной мантиссе (round-to-nearest-even), как в IEEE 754 и в конвертерах FP8/FP4.
 * Источник раскладок и диапазонов: research/notes-techniques.md §2.2 (FP8: arXiv 2209.05433, Table 1;
 * FP4 E2M1: NVIDIA NVFP4 blog).
 *
 * Переполнение:
 * - IEEE-подобные форматы (FP32, BF16, FP16, FP8 E5M2) уходят в ±∞;
 * - у FP8 E4M3 (вариант «fn») и FP4 E2M1 бесконечностей нет: значение насыщается до ±max,
 *   как в квантующих ядрах (satfinite). Без насыщения E4M3 дал бы NaN (notes-techniques §3.2).
 * Целые (INT8, INT4): q = round(x / scale) с насыщением в [min, max], дополнительный код.
 */

export type FormatId = 'fp32' | 'bf16' | 'fp16' | 'e4m3' | 'e5m2' | 'e2m1' | 'int8' | 'int4'

interface FormatBase {
  id: FormatId
  /** Короткое имя: «FP8 E4M3». */
  name: string
  /** Всего бит. */
  bits: number
  /** Одна строка по-русски: где используется. */
  note: string
}

export interface FloatFormat extends FormatBase {
  kind: 'float'
  exp: number
  man: number
  bias: number
  /**
   * 'ieee': поле экспоненты из одних единиц занято под ∞ и NaN;
   * 'fn': ∞ нет, NaN только S.1111.111 (FP8 E4M3);
   * 'none': ни ∞, ни NaN (FP4 E2M1).
   */
  special: 'ieee' | 'fn' | 'none'
}

export interface IntFormat extends FormatBase {
  kind: 'int'
  min: number
  max: number
}

export type NumberFormat = FloatFormat | IntFormat

export const FORMATS: readonly NumberFormat[] = [
  { id: 'fp32', name: 'FP32', bits: 32, kind: 'float', exp: 8, man: 23, bias: 127, special: 'ieee', note: 'обучение, аккумуляторы матричных умножений' },
  { id: 'bf16', name: 'BF16', bits: 16, kind: 'float', exp: 8, man: 7, bias: 127, special: 'ieee', note: 'стандарт весов и KV: диапазон FP32, мало мантиссы' },
  { id: 'fp16', name: 'FP16', bits: 16, kind: 'float', exp: 5, man: 10, bias: 15, special: 'ieee', note: 'точнее BF16, но максимум всего 65\u00a0504' },
  { id: 'e4m3', name: 'FP8 E4M3', bits: 8, kind: 'float', exp: 4, man: 3, bias: 7, special: 'fn', note: 'веса, активации и KV в FP8 (Hopper и новее)' },
  { id: 'e5m2', name: 'FP8 E5M2', bits: 8, kind: 'float', exp: 5, man: 2, bias: 15, special: 'ieee', note: 'шире диапазон, грубее шаг: чаще для градиентов' },
  { id: 'e2m1', name: 'FP4 E2M1', bits: 4, kind: 'float', exp: 2, man: 1, bias: 1, special: 'none', note: 'элементы MXFP4 и NVFP4 (Blackwell)' },
  { id: 'int8', name: 'INT8', bits: 8, kind: 'int', min: -128, max: 127, note: 'равномерная сетка, нужен масштаб' },
  { id: 'int4', name: 'INT4', bits: 4, kind: 'int', min: -8, max: 7, note: 'веса при weight-only (W4A16), нужен масштаб' },
]

export function getFormat(id: FormatId): NumberFormat {
  const f = FORMATS.find((x) => x.id === id)
  if (!f) throw new Error(`Неизвестный формат ${id}`)
  return f
}

/* ============================================================================
 * Диапазоны
 * ========================================================================== */

const pow2 = (e: number) => Math.pow(2, e)

/** Максимальное поле экспоненты нормального конечного числа. */
function maxNormalExpField(f: FloatFormat): number {
  const all = pow2(f.exp) - 1
  return f.special === 'ieee' ? all - 1 : all
}

/** Максимальная мантисса при максимальной экспоненте (у E4M3 код 111 занят под NaN). */
function maxManAtTop(f: FloatFormat): number {
  const all = pow2(f.man) - 1
  return f.special === 'fn' ? all - 1 : all
}

export interface FormatRange {
  /** Наибольшее конечное значение. */
  max: number
  /** Наименьшее положительное нормальное (у INT: шаг сетки, то есть scale). */
  minNormal: number
  /** Наименьшее положительное субнормальное (у INT: шаг сетки). */
  minSubnormal: number
  /** Число кодов: 2^bits. */
  codes: number
  /** Число различных конечных значений (+0 и −0 считаются одним). */
  finiteValues: number
}

export function formatRange(f: NumberFormat, scale = 1): FormatRange {
  if (f.kind === 'int') {
    return { max: f.max * scale, minNormal: scale, minSubnormal: scale, codes: pow2(f.bits), finiteValues: pow2(f.bits) }
  }
  const eMax = maxNormalExpField(f) - f.bias
  const max = pow2(eMax) * (1 + maxManAtTop(f) / pow2(f.man))
  const minNormal = pow2(1 - f.bias)
  const minSubnormal = pow2(1 - f.bias - f.man)
  const codes = pow2(f.bits)
  // коды под ∞/NaN: у ieee вся верхняя экспонента (2^man кодов на знак), у fn один код на знак
  const reserved = f.special === 'ieee' ? 2 * pow2(f.man) : f.special === 'fn' ? 2 : 0
  return { max: max * scale, minNormal: minNormal * scale, minSubnormal: minSubnormal * scale, codes, finiteValues: codes - reserved - 1 }
}

/* ============================================================================
 * Округление
 * ========================================================================== */

/** Округление к ближайшему целому, при равенстве к чётному (x ≥ 0, точное в double). */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x)
  const d = x - f
  if (d > 0.5) return f + 1
  if (d < 0.5) return f
  return f % 2 === 0 ? f : f + 1
}

/** floor(log2(a)) без ошибок округления Math.log2 у степеней двойки. */
function floorLog2(a: number): number {
  let e = Math.floor(Math.log2(a))
  if (pow2(e) > a) e -= 1
  else if (pow2(e + 1) <= a) e += 1
  return e
}

export type QuantKind = 'zero' | 'subnormal' | 'normal' | 'inf' | 'nan' | 'int'

export interface QuantResult {
  format: FormatId
  input: number
  /** Ближайшее представимое значение (с учётом масштаба), ±Infinity или NaN. */
  value: number
  /** Биты от старшего к младшему, длина = bits. */
  bits: number[]
  sign: 0 | 1
  /** Сырые поля: экспонента и мантисса у float, у int поле value = q (целое до масштаба). */
  expField: number
  manField: number
  /** Целое q для INT (до умножения на масштаб). */
  q?: number
  kind: QuantKind
  /** |x| больше максимума: насыщение (E4M3, E2M1, INT) или ∞ (IEEE). */
  overflow: boolean
  /** x ≠ 0, а результат 0. */
  underflow: boolean
  /** value − input (NaN/∞ при переполнении в ∞). */
  absError: number
  /** |value − input| / |input| (0 при input = 0). */
  relError: number
}

function toBits(code: number, n: number): number[] {
  const out: number[] = []
  for (let i = n - 1; i >= 0; i--) out.push(Math.floor(code / pow2(i)) % 2)
  return out
}

function floatCode(f: FloatFormat, sign: number, e: number, m: number): number {
  return sign * pow2(f.bits - 1) + e * pow2(f.man) + m
}

function finish(f: NumberFormat, input: number, value: number, partial: Omit<QuantResult, 'format' | 'input' | 'value' | 'absError' | 'relError' | 'underflow'> & { underflow?: boolean }): QuantResult {
  const absError = value - input
  const relError = input === 0 ? (value === 0 ? 0 : Infinity) : Math.abs(absError) / Math.abs(input)
  return {
    format: f.id,
    input,
    value,
    absError,
    relError,
    underflow: partial.underflow ?? (input !== 0 && value === 0),
    ...partial,
  }
}

/** Квантовать x в формат float: ближайшее представимое, RNE, субнормали, переполнение. */
function quantizeFloat(f: FloatFormat, x: number): QuantResult {
  const sign: 0 | 1 = x < 0 || Object.is(x, -0) ? 1 : 0
  const s = sign ? -1 : 1
  const allOnes = pow2(f.exp) - 1
  if (Number.isNaN(x)) {
    if (f.special === 'none') {
      // у FP4 NaN нет: показываем ноль с флагом переполнения
      return finish(f, x, 0, { bits: toBits(0, f.bits), sign: 0, expField: 0, manField: 0, kind: 'zero', overflow: true })
    }
    const man = f.special === 'fn' ? pow2(f.man) - 1 : pow2(f.man - 1)
    return finish(f, x, NaN, { bits: toBits(floatCode(f, 0, allOnes, man), f.bits), sign: 0, expField: allOnes, manField: man, kind: 'nan', overflow: false })
  }
  const a = Math.abs(x)
  const range = formatRange(f)
  const top = maxNormalExpField(f)

  const encodeMax = (): QuantResult => {
    const m = maxManAtTop(f)
    return finish(f, x, s * range.max, { bits: toBits(floatCode(f, sign, top, m), f.bits), sign, expField: top, manField: m, kind: 'normal', overflow: true })
  }
  const encodeInf = (): QuantResult =>
    finish(f, x, s * Infinity, { bits: toBits(floatCode(f, sign, allOnes, 0), f.bits), sign, expField: allOnes, manField: 0, kind: 'inf', overflow: true })

  if (a === Infinity) return f.special === 'ieee' ? encodeInf() : encodeMax()
  if (a === 0) return finish(f, x, sign ? -0 : 0, { bits: toBits(floatCode(f, sign, 0, 0), f.bits), sign, expField: 0, manField: 0, kind: 'zero', overflow: false })

  const eMin = 1 - f.bias
  let e = floorLog2(a)
  let magnitude: number
  let expField: number
  let manField: number
  let kind: QuantKind
  if (e < eMin) {
    // субнормальная сетка: шаг 2^(eMin − man)
    const step = pow2(eMin - f.man)
    const r = roundHalfEven(a / step)
    magnitude = r * step
    if (r === 0) {
      return finish(f, x, sign ? -0 : 0, { bits: toBits(floatCode(f, sign, 0, 0), f.bits), sign, expField: 0, manField: 0, kind: 'zero', overflow: false, underflow: true })
    }
    if (r >= pow2(f.man)) {
      // округлилось вверх до минимального нормального
      expField = 1
      manField = r - pow2(f.man)
      kind = 'normal'
    } else {
      expField = 0
      manField = r
      kind = 'subnormal'
    }
  } else {
    const step = pow2(e - f.man)
    let r = roundHalfEven(a / step)
    if (r >= pow2(f.man + 1)) {
      r /= 2
      e += 1
    }
    magnitude = r * pow2(e - f.man)
    expField = e + f.bias
    manField = r - pow2(f.man)
    kind = 'normal'
  }
  // переполнение: результат с неограниченной экспонентой больше максимума
  if (magnitude > range.max) return f.special === 'ieee' ? encodeInf() : encodeMax()
  return finish(f, x, s * magnitude, { bits: toBits(floatCode(f, sign, expField, manField), f.bits), sign, expField, manField, kind, overflow: false })
}

function quantizeInt(f: IntFormat, x: number, scale: number): QuantResult {
  const sc = scale > 0 ? scale : 1
  if (Number.isNaN(x)) {
    return finish(f, x, 0, { bits: toBits(0, f.bits), sign: 0, expField: 0, manField: 0, q: 0, kind: 'int', overflow: true })
  }
  const t = x / sc
  const r = t < 0 ? -roundHalfEven(-t) : roundHalfEven(t)
  const q = Math.max(f.min, Math.min(f.max, r))
  const overflow = r !== q
  const code = q < 0 ? q + pow2(f.bits) : q
  const value = q * sc
  return finish(f, x, Object.is(value, -0) ? 0 : value, {
    bits: toBits(code, f.bits),
    sign: q < 0 ? 1 : 0,
    expField: 0,
    manField: q,
    q,
    kind: 'int',
    overflow,
  })
}

/**
 * Ближайшее представимое значение формата. scale (по умолчанию 1) делит вход перед
 * квантизацией и умножает результат обратно: x̂ = Q(x / scale) · scale.
 */
export function quantize(x: number, f: NumberFormat, scale = 1): QuantResult {
  if (f.kind === 'int') return quantizeInt(f, x, scale)
  if (scale === 1) return quantizeFloat(f, x)
  const r = quantizeFloat(f, x / scale)
  const value = r.value * scale
  const absError = value - x
  return { ...r, input: x, value, absError, relError: x === 0 ? (value === 0 ? 0 : Infinity) : Math.abs(absError) / Math.abs(x) }
}

/** Значение по битам (обратное преобразование; для проверки и подписей). */
export function decodeBits(f: NumberFormat, bits: readonly number[], scale = 1): number {
  const code = bits.reduce((acc, b) => acc * 2 + b, 0)
  if (f.kind === 'int') {
    const q = code >= pow2(f.bits - 1) ? code - pow2(f.bits) : code
    return q * scale
  }
  const sign = bits[0] === 1 ? -1 : 1
  const e = Math.floor((code % pow2(f.bits - 1)) / pow2(f.man))
  const m = code % pow2(f.man)
  const allOnes = pow2(f.exp) - 1
  if (f.special === 'ieee' && e === allOnes) return m === 0 ? sign * Infinity : NaN
  if (f.special === 'fn' && e === allOnes && m === pow2(f.man) - 1) return NaN
  if (e === 0) return sign * pow2(1 - f.bias) * (m / pow2(f.man)) * scale
  return sign * pow2(e - f.bias) * (1 + m / pow2(f.man)) * scale
}

/**
 * Соседи по сетке: ближайшие представимые значения ниже и выше результата
 * (null за краем диапазона). Для float считаем по коду модуля, он монотонен.
 */
export function gridNeighbors(r: QuantResult, f: NumberFormat, scale = 1): { below: number | null; above: number | null } {
  if (!Number.isFinite(r.value)) return { below: null, above: null }
  if (f.kind === 'int') {
    const q = r.q ?? 0
    return { below: q > f.min ? (q - 1) * scale : null, above: q < f.max ? (q + 1) * scale : null }
  }
  const magCode = r.expField * pow2(f.man) + r.manField
  const maxCode = maxNormalExpField(f) * pow2(f.man) + maxManAtTop(f)
  const magOf = (c: number) => decodeBits(f, toBits(c, f.bits), scale)
  // соседние по модулю коды: модуль кода монотонен по значению
  const up = magCode < maxCode ? magOf(magCode + 1) : null
  const down = magCode > 0 ? magOf(magCode - 1) : null
  const neg = (v: number | null) => (v === null ? null : v === 0 ? 0 : -v)
  if (r.value === 0) return { below: neg(up), above: up }
  if (r.value > 0) return { below: down, above: up }
  // отрицательное: ниже по значению тот, у кого модуль больше
  return { below: neg(up), above: neg(down) }
}

/**
 * Все положительные конечные значения формата (для числовой оси). Только для форматов
 * до 8 бит: у FP16 их 31 743, у FP32 миллиарды.
 */
export function positiveValues(f: NumberFormat, scale = 1): number[] {
  if (f.bits > 8) return []
  if (f.kind === 'int') return Array.from({ length: f.max }, (_, i) => (i + 1) * scale)
  const out: number[] = []
  const maxCode = maxNormalExpField(f) * pow2(f.man) + maxManAtTop(f)
  for (let c = 1; c <= maxCode; c++) out.push(decodeBits(f, toBits(c, f.bits), scale))
  return out
}

/** Разбор числа из поля ввода: запятая или точка, пробелы, «e», минус «−». */
export function parseNumberInput(s: string): number | null {
  const t = s.trim().replace(/\s+/g, '').replace(/−/g, '-').replace(',', '.')
  if (t === '' || t === '-' || t === '.') return null
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) return null
  const v = Number(t)
  return Number.isFinite(v) ? v : null
}
