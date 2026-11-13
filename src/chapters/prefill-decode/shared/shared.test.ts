import { describe, expect, it } from 'vitest'
import { getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'
import { prefillMs } from '@/engine/perf'
import { durationValueUnit, times, valueUnit } from './format'
import { autoTp, dtypeSupported, makeHardware, resolveGpu, resolveModel } from './hw-core'
import { texInt, texNum, texSci } from './tex'
import { requestTiming } from './timing'

describe('hw-core', () => {
  it('fallback для неизвестных id', () => {
    expect(resolveModel('nope').id).toBe('qwen3-8b')
    expect(resolveGpu('nope').id).toBe('h100-sxm')
  })

  it('autoTp: маленькая модель на одном GPU, 70B раскладывается на несколько', () => {
    const h100 = getGpu('h100-sxm')
    expect(autoTp(getModel('qwen3-8b'), h100, 'bf16')).toBe(1)
    // 141 ГБ весов BF16 не помещаются в 80 ГБ, на двух H100 под KV почти ничего не остаётся
    expect(autoTp(getModel('llama-3.1-70b'), h100, 'bf16')).toBe(4)
    // FP8: 70,6 ГБ весов, TP2 даёт достаточно места под KV (notes-numbers §3.15: 29 запросов по 8k)
    expect(autoTp(getModel('llama-3.1-70b'), h100, 'fp8')).toBe(2)
    expect(autoTp(getModel('llama-3.1-70b'), getGpu('h200'), 'bf16')).toBe(2)
  })

  it('dtypeSupported: нет FP8 на A100 и FP4 на Hopper', () => {
    expect(dtypeSupported(getGpu('a100-80g'), 'fp8')).toBe(false)
    expect(dtypeSupported(getGpu('h100-sxm'), 'fp4')).toBe(false)
    expect(dtypeSupported(getGpu('b200'), 'fp4')).toBe(true)
  })

  it('makeHardware передаёт TP в опции perf', () => {
    const hw = makeHardware(getModel('llama-3.1-70b'), getGpu('h100-sxm'), 'bf16')
    expect(hw.perf).toEqual({ dtype: 'bf16', gpus: 4 })
  })
})

describe('timing', () => {
  const hw = makeHardware(getModel('qwen3-8b'), getGpu('h100-sxm'), 'bf16')

  it('TTFT = очередь + prefill, E2E = TTFT + ITL × (N − 1)', () => {
    const t = requestTiming(hw, 2000, 400, 120)
    expect(t.ttftMs).toBeCloseTo(120 + prefillMs(hw.model, hw.gpu, 2000), 6)
    expect(t.e2eMs).toBeCloseTo(t.ttftMs + t.itlMs * 399, 6)
    expect(t.prefillBound).toBe('compute')
    // шаг decode растёт с контекстом: читается весь KV
    expect(t.lastItlMs).toBeGreaterThan(t.firstItlMs)
  })

  it('один токен ответа: E2E = TTFT', () => {
    const t = requestTiming(hw, 43, 1, 0)
    expect(t.e2eMs).toBeCloseTo(t.ttftMs, 9)
    // короткий промпт упирается в чтение весов
    expect(t.prefillBound).toBe('memory')
  })
})

describe('format', () => {
  it('times: русские формы', () => {
    expect(times(2)).toBe('2 раза')
    expect(times(5)).toBe('5 раз')
    expect(times(12)).toBe('12 раз')
    expect(times(21)).toBe('21 раз')
    expect(times(22.4)).toBe('22 раза')
    expect(times(295.2)).toBe('295 раз')
  })

  it('durationValueUnit: от минуты целые секунды, а не «2 мин 1» + «с»', () => {
    expect(durationValueUnit(121_000)).toEqual(['121', 'с'])
    expect(durationValueUnit(7.3)).toEqual(['7,3', 'мс'])
    expect(durationValueUnit(2900)).toEqual(['2,9', 'с'])
  })

  it('valueUnit режет по неразрывному пробелу', () => {
    expect(valueUnit('16,4 ГБ')).toEqual(['16,4', 'ГБ'])
    expect(valueUnit('7,3 мс')).toEqual(['7,3', 'мс'])
    expect(valueUnit('42')).toEqual(['42', ''])
  })
})

describe('tex', () => {
  it('десятичная запятая и разряды', () => {
    expect(texNum(3.35)).toBe('3{,}35')
    expect(texInt(147456)).toBe('147\\,456')
    expect(texNum(4096)).toBe('4\\,096')
  })

  it('научная запись', () => {
    expect(texSci(3.2e13)).toBe('3{,}2 \\cdot 10^{13}')
    expect(texSci(7.568e9)).toBe('7{,}57 \\cdot 10^{9}')
    expect(texSci(512)).toBe('512')
    // 9,999 при округлении становится 10: переносим в следующую степень
    expect(texSci(9.999e5)).toBe('1{,}00 \\cdot 10^{6}')
  })
})
