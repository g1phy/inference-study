/**
 * Форматирование чисел по-русски (DESIGN §7): Intl.NumberFormat('ru-RU').
 * Разделитель тысяч неразрывный пробел (151 936), десятичная запятая (3,35),
 * настоящий минус «−» вместо дефиса.
 *
 *   fmt.num(3.3456)            → "3,35"
 *   fmt.num(0.5, 2)            → "0,50"
 *   fmt.int(151936)            → "151 936"
 *   fmt.bytes(1.24e9)          → "1,24 ГБ"
 *   fmt.bytes(147456, { mode: 'binary' }) → "144 КиБ"
 *   fmt.ms(4.93)               → "4,9 мс"
 *   fmt.ms(1234)               → "1,2 с"
 *   fmt.pct(0.42)              → "42 %"
 *   fmt.rate(205.3)            → "205 ток/с"
 *   fmt.count(5, TOKEN_FORMS)  → "5 токенов"
 */

const LOCALE = 'ru-RU'
const MINUS = '−'
const NBSP = ' '

const cache = new Map<string, Intl.NumberFormat>()
function nf(opts: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = JSON.stringify(opts)
  let f = cache.get(key)
  if (!f) {
    f = new Intl.NumberFormat(LOCALE, opts)
    cache.set(key, f)
  }
  return f
}

/** Заменить ASCII-дефис минуса на типографский «−». */
function withMinus(s: string): string {
  return s.replace(/-/g, MINUS)
}

/** Ноль без знака: −0 → 0. */
function noNegZero(x: number): number {
  return Object.is(x, -0) ? 0 : x
}

function nonFinite(x: number): string | null {
  if (Number.isNaN(x)) return 'н/д'
  if (x === Infinity) return '∞'
  if (x === -Infinity) return `${MINUS}∞`
  return null
}

/** Автоматическое число знаков после запятой: чем больше число, тем меньше знаков. */
function autoDigits(abs: number): number {
  if (abs >= 100) return 0
  if (abs >= 10) return 1
  return 2
}

/**
 * Число. Без `digits`: разумная точность (3,35; 48,3; 151 936; 0,0123),
 * хвостовые нули убираются. С `digits`: ровно столько знаков после запятой (0,50).
 */
function num(x: number, digits?: number): string {
  const bad = nonFinite(x)
  if (bad) return bad
  const abs = Math.abs(x)
  let s: string
  if (digits !== undefined) {
    s = nf({ minimumFractionDigits: digits, maximumFractionDigits: digits }).format(x)
  } else if (abs > 0 && abs < 1) {
    s = nf({ maximumSignificantDigits: 3 }).format(x)
  } else {
    s = nf({ maximumFractionDigits: autoDigits(abs) }).format(x)
  }
  // «−0,00» после округления маленького отрицательного числа выглядит странно
  if (/^-0([,.]0+)?$/.test(s)) s = s.slice(1)
  return withMinus(s)
}

/** Целое с разделителем тысяч: 151 936. */
function int(x: number): string {
  const bad = nonFinite(x)
  if (bad) return bad
  return withMinus(nf({ maximumFractionDigits: 0 }).format(noNegZero(Math.round(x))))
}

/** Компактная запись для осей и тесных мест: 1,2 тыс., 622 млн, 1,5 млрд. */
function compact(x: number, digits = 1): string {
  const bad = nonFinite(x)
  if (bad) return bad
  if (Math.abs(x) < 1000) return num(x)
  return withMinus(nf({ notation: 'compact', maximumFractionDigits: digits }).format(x))
}

/** Со знаком: +1,2 / −0,5 / 0. */
function signed(x: number, digits?: number): string {
  const s = num(x, digits)
  return x > 0 ? `+${s}` : s
}

/** Множитель: ×2,5. */
function multiplier(x: number, digits?: number): string {
  return `×${num(x, digits)}`
}

/** Округлить до 3 значащих цифр (без экспоненты). */
function sig3(x: number): string {
  const abs = Math.abs(x)
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2
  return withMinus(nf({ maximumFractionDigits: digits }).format(x))
}

const SI_BYTES = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ', 'ПБ', 'ЭБ']
const BIN_BYTES = ['Б', 'КиБ', 'МиБ', 'ГиБ', 'ТиБ', 'ПиБ', 'ЭиБ']

export interface BytesOptions {
  /** 'si' (по умолчанию): ГБ = 10⁹, как в прозе и даташитах. 'binary': ГиБ = 2³⁰, для «Глубже». */
  mode?: 'si' | 'binary'
  /** Фиксированное число знаков после запятой (по умолчанию до 3 значащих цифр). */
  digits?: number
}

/** Байты: «1,24 ГБ» (SI) или «144 КиБ» (binary). */
function bytes(b: number, opts: BytesOptions = {}): string {
  const bad = nonFinite(b)
  if (bad) return bad
  const binary = opts.mode === 'binary'
  const base = binary ? 1024 : 1000
  const units = binary ? BIN_BYTES : SI_BYTES
  const last = units.length - 1
  const sign = b < 0 ? MINUS : ''
  let v = Math.abs(b)
  let u = 0
  while (v >= base && u < last) {
    v /= base
    u++
  }
  // 999,7 КБ после округления превратилось бы в «1000 КБ»: переносим в следующую единицу
  const rounded = u === 0 ? Math.round(v) : opts.digits !== undefined ? Number(v.toFixed(opts.digits)) : Number(v.toPrecision(3))
  if (rounded >= base && u < last) {
    v /= base
    u++
  }
  if (u === 0) return `${sign}${int(v)}${NBSP}${units[0]}`
  const text = opts.digits !== undefined ? num(v, opts.digits) : sig3(v)
  return `${sign}${text}${NBSP}${units[u]}`
}

/** Миллисекунды: «850 мкс», «4,9 мс», «48,3 мс», «245 мс», «1,2 с», «2 мин 5 с». */
function ms(x: number): string {
  const bad = nonFinite(x)
  if (bad) return bad
  const sign = x < 0 ? MINUS : ''
  const a = Math.abs(x)
  if (a === 0) return `0${NBSP}мс`
  if (a < 1) {
    const us = a * 1000
    return `${sign}${nf({ maximumFractionDigits: us < 10 ? 1 : 0 }).format(us)}${NBSP}мкс`
  }
  if (a < 100) return `${sign}${nf({ maximumFractionDigits: 1 }).format(a)}${NBSP}мс`
  if (a < 1000) return `${sign}${int(a)}${NBSP}мс`
  if (a < 60_000) {
    const s = a / 1000
    return `${sign}${nf({ maximumFractionDigits: 1 }).format(s)}${NBSP}с`
  }
  if (a < 3_600_000) {
    const m = Math.floor(a / 60_000)
    const s = Math.round((a - m * 60_000) / 1000)
    return s ? `${sign}${m}${NBSP}мин ${s}${NBSP}с` : `${sign}${m}${NBSP}мин`
  }
  const h = Math.floor(a / 3_600_000)
  const m = Math.round((a - h * 3_600_000) / 60_000)
  return m ? `${sign}${h}${NBSP}ч ${m}${NBSP}мин` : `${sign}${h}${NBSP}ч`
}

/** Доля → проценты: 0,42 → «42 %», 0,042 → «4,2 %». */
function pct(fraction: number, digits?: number): string {
  const bad = nonFinite(fraction)
  if (bad) return bad
  const a = Math.abs(fraction)
  if (digits === undefined && a > 0 && a < 0.001) return `<0,1${NBSP}%`
  const d = digits ?? (a < 0.1 && a > 0 ? 1 : 0)
  return withMinus(nf({ style: 'percent', minimumFractionDigits: 0, maximumFractionDigits: d }).format(fraction))
}

/** Русские формы множественного числа: plural(5, ['токен', 'токена', 'токенов']) → 'токенов'. */
function plural(n: number, forms: readonly [one: string, few: string, many: string]): string {
  if (!Number.isInteger(n)) return forms[1]
  const a = Math.abs(n) % 100
  const b = a % 10
  if (a > 10 && a < 20) return forms[2]
  if (b === 1) return forms[0]
  if (b >= 2 && b <= 4) return forms[1]
  return forms[2]
}

/** Число и слово в нужной форме: count(151669, TOKEN_FORMS) → «151 669 токенов». */
function count(n: number, forms: readonly [string, string, string]): string {
  const text = Number.isInteger(n) ? int(n) : num(n)
  return `${text}${NBSP}${plural(n, forms)}`
}

export const TOKEN_FORMS = ['токен', 'токена', 'токенов'] as const

/** Токены: «1 токен», «24 токена», «151 669 токенов». */
function tokens(n: number): string {
  return count(n, TOKEN_FORMS)
}

/** Формы слова «токен» в родительном падеже (после «из», «около», «всех», «для»). */
export const TOKEN_GEN_FORMS = ['токена', 'токенов', 'токенов'] as const

/** Токены в родительном падеже: «из 1 токена», «из 43 токенов», «около 21 токена», «всех 1 167 токенов». */
function tokensGen(n: number): string {
  return count(n, TOKEN_GEN_FORMS)
}

/** Скорость генерации: «205 ток/с», «4,5 ток/с»; с long: «205 токенов/с». */
function rate(tokensPerSec: number, opts: { long?: boolean; digits?: number } = {}): string {
  const bad = nonFinite(tokensPerSec)
  if (bad) return bad
  const a = Math.abs(tokensPerSec)
  const value =
    opts.digits !== undefined
      ? num(tokensPerSec, opts.digits)
      : withMinus(nf({ maximumFractionDigits: a >= 10 ? 0 : a >= 1 ? 1 : 2 }).format(tokensPerSec))
  const unit = opts.long ? `${plural(Math.round(a), TOKEN_FORMS)}/с` : 'ток/с'
  return `${value}${NBSP}${unit}`
}

/** Полоса пропускания из ГБ/с: «900 ГБ/с», «3,35 ТБ/с». */
function bandwidth(gbPerSec: number): string {
  const bad = nonFinite(gbPerSec)
  if (bad) return bad
  if (Math.abs(gbPerSec) >= 1000) return `${sig3(gbPerSec / 1000)}${NBSP}ТБ/с`
  return `${sig3(gbPerSec)}${NBSP}ГБ/с`
}

const FLOP_PREFIX = ['', 'K', 'M', 'G', 'T', 'P', 'E']

/** FLOP: «16,6 TFLOP»; с { perSecond: true }: «989 TFLOPS». */
function flops(x: number, opts: { perSecond?: boolean } = {}): string {
  const bad = nonFinite(x)
  if (bad) return bad
  let v = Math.abs(x)
  let u = 0
  while (v >= 1000 && u < FLOP_PREFIX.length - 1) {
    v /= 1000
    u++
  }
  const sign = x < 0 ? MINUS : ''
  return `${sign}${sig3(v)}${NBSP}${FLOP_PREFIX[u]}FLOP${opts.perSecond ? 'S' : ''}`
}

/** Байты в hex: [0xD0, 0x9F] → «D0 9F». */
function hex(data: ArrayLike<number>, sep = ' '): string {
  let out = ''
  for (let i = 0; i < data.length; i++) {
    out += (i ? sep : '') + (data[i] & 0xff).toString(16).toUpperCase().padStart(2, '0')
  }
  return out
}

export const fmt = {
  num,
  int,
  compact,
  signed,
  multiplier,
  bytes,
  ms,
  pct,
  rate,
  plural,
  count,
  tokens,
  tokensGen,
  bandwidth,
  flops,
  hex,
} as const

export type Fmt = typeof fmt
