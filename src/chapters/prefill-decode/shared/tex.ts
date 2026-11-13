import { fmt } from '@/ui/fmt'

/**
 * Числа для формул KaTeX по-русски: десятичная запятая без лишнего пробела ({,}),
 * разряды через тонкую шпацию (\,), большие числа в виде a·10^b.
 */

/** fmt.num → TeX: «3,35» → «3{,}35», «151 936» → «151\,936». */
export function texNum(x: number, digits?: number): string {
  return fmt
    .num(x, digits)
    .replace(/,/g, '{,}')
    .replace(/[\s  ]/g, '\\,')
    .replace(/−/g, '-')
}

/** Целое с разрядами: 147456 → «147\,456». */
export function texInt(x: number): string {
  return fmt.int(x).replace(/[\s  ]/g, '\\,')
}

/** Научная запись: 3.2e13 → «3{,}2 \cdot 10^{13}» (до 1000 обычное число). */
export function texSci(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return '\\infty'
  const a = Math.abs(x)
  if (a < 1000) return texNum(x)
  const exp = Math.floor(Math.log10(a))
  const mant = x / 10 ** exp
  const m = Number(mant.toFixed(digits))
  // 9,999 → 10,0: переносим в следующую степень
  if (Math.abs(m) >= 10) return `${texNum(m / 10, digits)} \\cdot 10^{${exp + 1}}`
  return `${texNum(m)} \\cdot 10^{${exp}}`
}

/** Байты в ГБ для формул: 1.638e10 → «16{,}4\ \text{ГБ}». */
export function texBytes(b: number): string {
  const s = fmt.bytes(b)
  const i = s.search(/[\s ]\S+$/)
  const num = s.slice(0, i).replace(/,/g, '{,}')
  const unit = s.slice(i + 1)
  return `${num}\\ \\text{${unit}}`
}

/** Миллисекунды для формул: «4{,}9\ \text{мс}». */
export function texMs(ms: number): string {
  const s = fmt.ms(ms)
  const i = s.search(/[\s ]\S+$/)
  const num = s.slice(0, i).replace(/,/g, '{,}').replace(/[\s ]/g, '\\,')
  const unit = s.slice(i + 1)
  return `${num}\\ \\text{${unit}}`
}
