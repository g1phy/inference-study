import { describe, expect, it } from 'vitest'
import { fmt, TOKEN_FORMS } from './fmt'

/** Intl ставит неразрывные пробелы: для читаемости тестов приводим их к обычным. */
const plain = (s: string) => s.replace(/[  ]/g, ' ')

describe('fmt.num / fmt.int', () => {
  it('ставит запятую и неразрывный пробел по-русски', () => {
    expect(plain(fmt.int(151936))).toBe('151 936')
    expect(fmt.num(3.3456)).toBe('3,35')
    expect(fmt.num(48.34)).toBe('48,3')
    expect(fmt.num(0.0123)).toBe('0,0123')
    expect(plain(fmt.num(151936.4))).toBe('151 936')
  })
  it('фиксированные знаки и настоящий минус', () => {
    expect(fmt.num(0.5, 2)).toBe('0,50')
    expect(fmt.num(-0.52, 2)).toBe('−0,52')
    expect(fmt.num(-0.001, 2)).toBe('0,00')
    expect(fmt.signed(1.2)).toBe('+1,2')
    expect(fmt.multiplier(2.5)).toBe('×2,5')
  })
  it('бесконечности и NaN', () => {
    expect(fmt.num(-Infinity)).toBe('−∞')
    expect(fmt.num(Infinity)).toBe('∞')
    expect(fmt.num(NaN)).toBe('н/д')
  })
  it('compact для осей', () => {
    expect(plain(fmt.compact(622_000_000))).toBe('622 млн')
    expect(plain(fmt.compact(1_500))).toBe('1,5 тыс.')
    expect(fmt.compact(950)).toBe('950')
  })
})

describe('fmt.bytes', () => {
  it('SI по умолчанию', () => {
    expect(plain(fmt.bytes(1.24e9))).toBe('1,24 ГБ')
    expect(plain(fmt.bytes(1_244_659_712))).toBe('1,24 ГБ')
    expect(plain(fmt.bytes(80e9))).toBe('80 ГБ')
    expect(plain(fmt.bytes(512))).toBe('512 Б')
    expect(plain(fmt.bytes(0))).toBe('0 Б')
  })
  it('двоичный режим (КиБ, ГиБ)', () => {
    // Qwen3-8B: 2 × 36 × 8 × 128 × 2 = 147 456 байт на токен
    expect(plain(fmt.bytes(147_456, { mode: 'binary' }))).toBe('144 КиБ')
    expect(plain(fmt.bytes(2 ** 30, { mode: 'binary' }))).toBe('1 ГиБ')
  })
  it('округление переносит в следующую единицу', () => {
    expect(plain(fmt.bytes(999_700))).toBe('1 МБ')
    expect(plain(fmt.bytes(1023.6, { mode: 'binary' }))).toBe('1 КиБ')
  })
})

describe('fmt.ms / pct / rate', () => {
  it('время', () => {
    expect(plain(fmt.ms(4.93))).toBe('4,9 мс')
    expect(plain(fmt.ms(48.34))).toBe('48,3 мс')
    expect(plain(fmt.ms(245.4))).toBe('245 мс')
    expect(plain(fmt.ms(1234))).toBe('1,2 с')
    expect(plain(fmt.ms(0.85))).toBe('850 мкс')
    expect(plain(fmt.ms(125_000))).toBe('2 мин 5 с')
    expect(plain(fmt.ms(0))).toBe('0 мс')
  })
  it('проценты из долей', () => {
    expect(plain(fmt.pct(0.42))).toBe('42 %')
    expect(plain(fmt.pct(0.042))).toBe('4,2 %')
    expect(plain(fmt.pct(1))).toBe('100 %')
    expect(plain(fmt.pct(0.0004))).toBe('<0,1 %')
  })
  it('скорость генерации', () => {
    expect(plain(fmt.rate(205.3))).toBe('205 ток/с')
    expect(plain(fmt.rate(4.54))).toBe('4,5 ток/с')
    expect(plain(fmt.rate(21, { long: true }))).toBe('21 токен/с')
    expect(plain(fmt.rate(205, { long: true }))).toBe('205 токенов/с')
  })
})

describe('fmt.plural / count / прочее', () => {
  it('русские формы множественного числа', () => {
    const f = TOKEN_FORMS
    expect(fmt.plural(1, f)).toBe('токен')
    expect(fmt.plural(3, f)).toBe('токена')
    expect(fmt.plural(5, f)).toBe('токенов')
    expect(fmt.plural(11, f)).toBe('токенов')
    expect(fmt.plural(21, f)).toBe('токен')
    expect(fmt.plural(112, f)).toBe('токенов')
    expect(fmt.plural(1.5, f)).toBe('токена')
    expect(plain(fmt.tokens(151_669))).toBe('151 669 токенов')
  })
  it('токены в родительном падеже', () => {
    expect(plain(fmt.tokensGen(1))).toBe('1 токена')
    expect(plain(fmt.tokensGen(21))).toBe('21 токена')
    expect(plain(fmt.tokensGen(2))).toBe('2 токенов')
    expect(plain(fmt.tokensGen(11))).toBe('11 токенов')
    expect(plain(fmt.tokensGen(43))).toBe('43 токенов')
    expect(plain(fmt.tokensGen(1167))).toBe('1 167 токенов')
  })
  it('полоса, FLOP, hex', () => {
    expect(plain(fmt.bandwidth(3350))).toBe('3,35 ТБ/с')
    expect(plain(fmt.bandwidth(900))).toBe('900 ГБ/с')
    expect(plain(fmt.flops(989e12, { perSecond: true }))).toBe('989 TFLOPS')
    expect(plain(fmt.flops(1.66e13))).toBe('16,6 TFLOP')
    expect(fmt.hex([0xd0, 0x9f, 0x0a])).toBe('D0 9F 0A')
  })
})
