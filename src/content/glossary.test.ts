import { describe, expect, it } from 'vitest'
import { GLOSSARY, findTerm, normalizeTerm } from './glossary'

describe('глоссарий', () => {
  it('записи в нужной форме', () => {
    expect(GLOSSARY.length).toBeGreaterThan(20)
    for (const e of GLOSSARY) {
      expect(typeof e.term).toBe('string')
      expect(Array.isArray(e.aliases)).toBe(true)
      expect(e.ru.length).toBeGreaterThan(10)
      expect(typeof e.category).toBe('string')
    }
  })
  it('поиск по term и aliases без учёта регистра', () => {
    for (const t of ['KV cache', 'kv cache', 'prefill', 'decode', 'TTFT', 'attention']) expect(findTerm(t), t).toBeTruthy()
    expect(findTerm('такого термина нет')).toBeUndefined()
    expect(normalizeTerm('  Ёлка  ')).toBe('елка')
  })
  it('части имени «A / B», «A vs B» и «Имя (пояснение)» тоже находятся', () => {
    expect(findTerm('ISL')?.term).toBe('ISL / OSL')
    expect(findTerm('OSL')?.term).toBe('ISL / OSL')
    expect(findTerm('MHA')?.term).toBe('Attention head / MHA')
    expect(findTerm('GEMM')?.term).toBe('GEMM / matmul')
    expect(findTerm('prefill worker')?.term).toBe('Prefill worker / decode worker')
    expect(findTerm('воркер')?.term).toBe('Worker (воркер)')
    expect(findTerm('router')?.term).toBe('Router vs load balancer')
  })
  it('алиасы ведут на термин с тем же смыслом', () => {
    expect(findTerm('SM')?.term).toBe('GPU')
    expect(findTerm('L1')?.term).toContain('Tiered KV cache')
    expect(findTerm('L3')?.term).toContain('L3 KV cache')
    expect(findTerm('autoscaler')?.term).toBe('Autoscaling')
    expect(findTerm('replicas')?.term).toBe('Replica')
    expect(findTerm('engine core')?.term).toBe('EngineCore')
    expect(findTerm('detokenizer')?.term).toContain('Detokenization')
    expect(findTerm('хэш блока')?.term).toBe('Block hash')
    for (const e of GLOSSARY) {
      for (const a of e.aliases) expect(findTerm(a), `${e.term}: ${a}`).toBe(e)
    }
  })
  it('mentionOnly только у терминов с главой', () => {
    for (const e of GLOSSARY) if (e.mentionOnly) expect(e.chapter, e.term).toBeTruthy()
  })
})
