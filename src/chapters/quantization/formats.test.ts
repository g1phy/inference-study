import { describe, expect, it } from 'vitest'
import { decodeBits, FORMATS, formatRange, getFormat, gridNeighbors, parseNumberInput, positiveValues, quantize, roundHalfEven } from './formats'

const hex = (bits: number[]) => '0x' + bits.reduce((a, b) => a * 2 + b, 0).toString(16).toUpperCase().padStart(Math.ceil(bits.length / 4), '0')
const q = (x: number, id: Parameters<typeof getFormat>[0]) => quantize(x, getFormat(id))

describe('диапазоны форматов (notes-techniques §2.2)', () => {
  it('максимумы и минимумы', () => {
    expect(formatRange(getFormat('e4m3')).max).toBe(448)
    expect(formatRange(getFormat('e4m3')).minNormal).toBe(2 ** -6)
    expect(formatRange(getFormat('e4m3')).minSubnormal).toBe(2 ** -9)
    expect(formatRange(getFormat('e5m2')).max).toBe(57344)
    expect(formatRange(getFormat('e5m2')).minSubnormal).toBe(2 ** -16)
    expect(formatRange(getFormat('fp16')).max).toBe(65504)
    expect(formatRange(getFormat('fp16')).minSubnormal).toBe(2 ** -24)
    expect(formatRange(getFormat('e2m1')).max).toBe(6)
    expect(formatRange(getFormat('e2m1')).minSubnormal).toBe(0.5)
    expect(formatRange(getFormat('fp32')).max).toBeCloseTo(3.4028234663852886e38, -30)
    expect(formatRange(getFormat('bf16')).max).toBeCloseTo(3.3895313892515355e38, -30)
  })
  it('число конечных значений: E4M3 253, E2M1 15, INT8 256', () => {
    expect(formatRange(getFormat('e4m3')).finiteValues).toBe(253)
    expect(formatRange(getFormat('e2m1')).finiteValues).toBe(15)
    expect(formatRange(getFormat('int8')).finiteValues).toBe(256)
    expect(formatRange(getFormat('e4m3')).codes).toBe(256)
  })
})

describe('примеры из заметок (notes-techniques §3.2)', () => {
  it('0,1 в E4M3 → 0,1015625, биты 0x1D, ошибка +1,56 %', () => {
    const r = q(0.1, 'e4m3')
    expect(r.value).toBe(0.1015625)
    expect(hex(r.bits)).toBe('0x1D')
    expect(r.relError).toBeCloseTo(0.015625, 6)
  })
  it('0,1 в E5M2 → 0,09375, биты 0x2E, ошибка −6,25 %', () => {
    const r = q(0.1, 'e5m2')
    expect(r.value).toBe(0.09375)
    expect(hex(r.bits)).toBe('0x2E')
    expect(r.relError).toBeCloseTo(0.0625, 6)
  })
  it('300: E4M3 → 288 (0x79), E5M2 → 320 (0x5D)', () => {
    expect(q(300, 'e4m3').value).toBe(288)
    expect(hex(q(300, 'e4m3').bits)).toBe('0x79')
    expect(q(300, 'e5m2').value).toBe(320)
    expect(hex(q(300, 'e5m2').bits)).toBe('0x5D')
  })
  it('500: E4M3 насыщается до 448, E5M2 → 512', () => {
    const r = q(500, 'e4m3')
    expect(r.value).toBe(448)
    expect(r.overflow).toBe(true)
    expect(r.relError).toBeCloseTo(0.104, 3)
    expect(q(500, 'e5m2').value).toBe(512)
    expect(q(500, 'e5m2').overflow).toBe(false)
  })
  it('0,001: E4M3 грубо искажается (ближе к 2^−9), E5M2 → 0,0009765625', () => {
    const r = q(0.001, 'e4m3')
    expect(r.value).toBe(2 ** -9)
    expect(r.kind).toBe('subnormal')
    expect(r.relError).toBeGreaterThan(0.9)
    expect(q(0.001, 'e5m2').value).toBe(0.0009765625)
  })
})

describe('округление к чётному и края', () => {
  it('roundHalfEven', () => {
    expect([0.5, 1.5, 2.5, 3.5, 2.4, 2.6].map(roundHalfEven)).toEqual([0, 2, 2, 4, 2, 3])
  })
  it('FP4 E2M1: сетка и ничьи', () => {
    expect(positiveValues(getFormat('e2m1'))).toEqual([0.5, 1, 1.5, 2, 3, 4, 6])
    expect(q(2.5, 'e2m1').value).toBe(2) // между 2 (M=0) и 3 (M=1) к чётной
    expect(q(5, 'e2m1').value).toBe(4)
    expect(q(0.25, 'e2m1').value).toBe(0)
    expect(q(0.25, 'e2m1').underflow).toBe(true)
    expect(q(0.3, 'e2m1').value).toBe(0.5)
    expect(q(7, 'e2m1').value).toBe(6)
    expect(q(7, 'e2m1').overflow).toBe(true)
    expect(q(-1.4, 'e2m1').value).toBe(-1.5)
  })
  it('FP16: 65 520 уходит в ∞, 65 519 остаётся 65 504', () => {
    expect(q(65519, 'fp16').value).toBe(65504)
    expect(q(65520, 'fp16').value).toBe(Infinity)
    expect(q(65520, 'fp16').kind).toBe('inf')
    expect(q(-1e6, 'fp16').value).toBe(-Infinity)
  })
  it('E4M3: 464 (середина между 448 и 480) к чётной даёт 448, 465 насыщается', () => {
    expect(q(464, 'e4m3').value).toBe(448)
    expect(q(464, 'e4m3').overflow).toBe(false)
    expect(q(465, 'e4m3').value).toBe(448)
    expect(q(465, 'e4m3').overflow).toBe(true)
  })
  it('субнормальные и переход в нормальные', () => {
    expect(q(2 ** -24, 'fp16').value).toBe(2 ** -24)
    expect(q(2 ** -24, 'fp16').kind).toBe('subnormal')
    expect(q(2 ** -26, 'fp16').value).toBe(0)
    expect(q(0.0156, 'e4m3').value).toBe(0.015625) // минимальное нормальное
    expect(q(1e-50, 'fp32').value).toBe(0)
    expect(q(1e-50, 'fp32').underflow).toBe(true)
  })
  it('BF16 и FP16 для 0,1', () => {
    expect(q(0.1, 'bf16').value).toBe(0.10009765625)
    expect(q(0.1, 'fp16').value).toBe(0.0999755859375)
  })
  it('FP32 совпадает с Math.fround', () => {
    let s = 12345
    const rnd = () => ((s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31)
    for (let i = 0; i < 2000; i++) {
      const x = (rnd() - 0.5) * Math.pow(10, Math.floor(rnd() * 80) - 40)
      expect(q(x, 'fp32').value).toBe(Math.fround(x))
    }
    expect(q(Math.PI, 'fp32').value).toBe(Math.fround(Math.PI))
  })
  it('BF16 совпадает с округлением битов FP32 (на числах, точных в FP32)', () => {
    const buf = new DataView(new ArrayBuffer(4))
    let s = 99
    const rnd = () => ((s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31)
    for (let i = 0; i < 2000; i++) {
      const x = Math.fround((rnd() - 0.5) * Math.pow(2, Math.floor(rnd() * 60) - 30))
      buf.setFloat32(0, x)
      const u = buf.getUint32(0)
      const lsb = (u >>> 16) & 1
      const rounded = ((u + 0x7fff + lsb) >>> 16) << 16
      buf.setUint32(0, rounded >>> 0)
      expect(q(x, 'bf16').value).toBe(buf.getFloat32(0))
    }
  })
})

describe('биты и обратное преобразование', () => {
  it('известные коды', () => {
    expect(q(448, 'e4m3').bits.join('')).toBe('01111110')
    expect(q(57344, 'e5m2').bits.join('')).toBe('01111011')
    expect(q(1, 'fp16').bits.join('')).toBe('0011110000000000')
    expect(q(-2, 'bf16').bits.join('')).toBe('1100000000000000')
    expect(q(6, 'e2m1').bits.join('')).toBe('0111')
  })
  it('каждый конечный код E4M3, E5M2, E2M1 переживает decode → quantize', () => {
    for (const id of ['e4m3', 'e5m2', 'e2m1'] as const) {
      const f = getFormat(id)
      for (let c = 0; c < 2 ** f.bits; c++) {
        const bits = Array.from({ length: f.bits }, (_, i) => Math.floor(c / 2 ** (f.bits - 1 - i)) % 2)
        const v = decodeBits(f, bits)
        if (!Number.isFinite(v)) continue
        const r = quantize(v, f)
        expect(r.value, `${id} code ${c}`).toBe(v)
        if (v !== 0) expect(r.bits.join(''), `${id} code ${c}`).toBe(bits.join(''))
      }
    }
  })
  it('у E4M3 код S.1111.111 это NaN, у E5M2 11111.00 это ∞', () => {
    expect(decodeBits(getFormat('e4m3'), [0, 1, 1, 1, 1, 1, 1, 1])).toBeNaN()
    expect(decodeBits(getFormat('e5m2'), [0, 1, 1, 1, 1, 1, 0, 0])).toBe(Infinity)
  })
  it('INT8 и INT4: дополнительный код, насыщение, к чётному', () => {
    expect(q(0.1, 'int8').value).toBe(0)
    expect(q(0.1, 'int8').underflow).toBe(true)
    expect(q(300, 'int8').value).toBe(127)
    expect(q(300, 'int8').overflow).toBe(true)
    expect(q(-200, 'int8').value).toBe(-128)
    expect(q(-1, 'int8').bits.join('')).toBe('11111111')
    expect(q(2.5, 'int8').value).toBe(2)
    expect(q(3.5, 'int8').value).toBe(4)
    expect(q(-2.5, 'int8').value).toBe(-2)
    expect(q(7.6, 'int4').value).toBe(7)
    expect(q(7.6, 'int4').overflow).toBe(true)
    expect(q(-8.4, 'int4').value).toBe(-8)
    expect(q(-8, 'int4').bits.join('')).toBe('1000')
  })
  it('масштаб: x̂ = Q(x / s) · s', () => {
    const r = quantize(0.37, getFormat('int8'), 0.5 / 127)
    expect(r.q).toBe(94)
    expect(r.value).toBeCloseTo((94 * 0.5) / 127, 12)
    expect(quantize(0.6, getFormat('e2m1'), 0.1).value).toBeCloseTo(0.6, 12)
  })
})

describe('соседи по сетке и прочее', () => {
  it('соседи E4M3 вокруг 1', () => {
    const f = getFormat('e4m3')
    expect(gridNeighbors(quantize(1, f), f)).toEqual({ below: 0.9375, above: 1.125 })
    expect(gridNeighbors(quantize(-1, f), f)).toEqual({ below: -1.125, above: -0.9375 })
    expect(gridNeighbors(quantize(0, f), f)).toEqual({ below: -(2 ** -9), above: 2 ** -9 })
    expect(gridNeighbors(quantize(448, f), f).above).toBeNull()
  })
  it('соседи INT4', () => {
    const f = getFormat('int4')
    expect(gridNeighbors(quantize(3, f), f)).toEqual({ below: 2, above: 4 })
    expect(gridNeighbors(quantize(-8, f), f).below).toBeNull()
  })
  it('positiveValues: E4M3 126 значений, у 16 бит пусто', () => {
    const v = positiveValues(getFormat('e4m3'))
    expect(v.length).toBe(126)
    expect(v[v.length - 1]).toBe(448)
    expect(positiveValues(getFormat('fp16'))).toEqual([])
  })
  it('parseNumberInput', () => {
    expect(parseNumberInput('0,1')).toBe(0.1)
    expect(parseNumberInput(' −3,5 ')).toBe(-3.5)
    expect(parseNumberInput('1e-3')).toBe(0.001)
    expect(parseNumberInput('65 504')).toBe(65504)
    expect(parseNumberInput('abc')).toBeNull()
    expect(parseNumberInput('')).toBeNull()
  })
  it('все форматы в списке уникальны', () => {
    expect(new Set(FORMATS.map((f) => f.id)).size).toBe(FORMATS.length)
  })
})
