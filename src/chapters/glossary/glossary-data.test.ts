import { describe, expect, it } from 'vitest'
import { CHAPTER_IDS } from '@/content/chapters'
import { GLOSSARY, GLOSSARY_CATEGORIES, findTerm, normalizeTerm } from '@/content/glossary'
import { highlightParts, searchGlossary, slug, visibleAliases } from './search'

describe('данные глоссария', () => {
  it('у каждого термина есть глава из реестра', () => {
    for (const e of GLOSSARY) expect(CHAPTER_IDS as readonly string[], e.term).toContain(e.chapter)
  })
  it('12 категорий', () => {
    expect(GLOSSARY_CATEGORIES).toHaveLength(12)
  })
  it('без длинных тире и с «почему важно»', () => {
    for (const e of GLOSSARY) {
      expect(e.ru, e.term).not.toContain('—')
      expect(e.why, e.term).not.toContain('—')
      expect(e.term, e.term).not.toContain('—')
      expect(e.why.length, e.term).toBeGreaterThan(10)
    }
  })
  it('термины уникальны, алиасы не перехватывают чужие термины', () => {
    const terms = GLOSSARY.map((e) => normalizeTerm(e.term))
    expect(new Set(terms).size).toBe(terms.length)
    for (const e of GLOSSARY) {
      for (const a of e.aliases) {
        const owner = findTerm(a)
        expect(owner, `${e.term}: ${a}`).toBeTruthy()
      }
    }
  })
  it('находятся термины, нужные другим главам', () => {
    const need = [
      'NIXL', 'PrefillRouter', 'DaemonSet', 'LMCache', 'Mooncake', 'HiCache', 'XGrammar', 'jump-forward decoding', 'goodput',
      'MFU', 'MBU', 'CUDA graphs', 'FlashAttention', 'RoPE', 'SwiGLU', 'RMSNorm', 'residual stream', 'MoE', 'EAGLE', 'MTP',
      'MXFP4', 'NVFP4', 'SSE', 'reasoning_content', 'hermes', 'Harmony', 'minijinja', 'DynamoGraphDeployment', 'Planner',
      'xPyD', 'chunked prefill', 'preemption', 'copy-on-write', 'batch', 'KV cache', 'TTFT', 'ITL', 'prefill', 'decode',
      'BF16', 'FP4', 'INT8', 'Scale factor', 'Perplexity', 'Weight-only quantization', "Little's law", 'DCGM exporter',
      'Scale to zero', 'Rejection sampling', 'UTF-8', 'add_generation_prompt', 'Logprobs', 'Min-p', 'Sliding window attention',
    ]
    for (const t of need) expect(findTerm(t), t).toBeTruthy()
    expect(findTerm('SwiGLU')!.term).toBe('SwiGLU')
    expect(findTerm('RMSNorm')!.term).toBe('RMSNorm')
    expect(findTerm('hermes')!.term).toContain('Hermes')
    expect(findTerm('MFU')!.chapter).toBe('roofline')
  })
})

describe('поиск', () => {
  it('точное совпадение по алиасу первым', () => {
    const hits = searchGlossary(GLOSSARY, 'ttft', null)
    expect(hits[0].entry.term).toContain('TTFT')
  })
  it('ё и е, регистр', () => {
    expect(searchGlossary(GLOSSARY, 'ХОЛОДНЫЙ СТАРТ', null)[0].entry.term).toBe('Cold start')
    expect(searchGlossary(GLOSSARY, 'тёплый пул', null).length).toBeGreaterThan(0)
  })
  it('фильтр по категории', () => {
    const hits = searchGlossary(GLOSSARY, '', 'прод')
    expect(hits.length).toBeGreaterThan(5)
    expect(hits.every((h) => h.entry.category === 'прод')).toBe(true)
  })
  it('флаги как в командной строке: дефисы, подчёркивания и ведущие «--» не важны', () => {
    expect(searchGlossary(GLOSSARY, '--gpu-memory-utilization', null).length).toBeGreaterThan(0)
    expect(searchGlossary(GLOSSARY, 'kv-cache-dtype', null)[0].entry.term).toBe('KV cache quantization')
    expect(searchGlossary(GLOSSARY, 'prefix-caching', null)[0].entry.term).toBe('Prefix caching')
    expect(searchGlossary(GLOSSARY, 'structured outputs', null)[0].entry.term).toBe('Structured output')
    expect(searchGlossary(GLOSSARY, '--enable-prefix-caching', null)[0].entry.term).toBe('Prefix caching')
    expect(searchGlossary(GLOSSARY, '--speculative-config', null)[0].entry.term).toBe('Speculative decoding')
  })
  it('русские окончания и порядок слов', () => {
    const first = (q: string) => searchGlossary(GLOSSARY, q, null)[0]?.entry.term
    expect(first('квантизации')).toBe('Quantization')
    expect(first('квантование')).toBe('Quantization')
    expect(first('детокенизации')).toContain('Detokenization')
    expect(first('холодного старта')).toBe('Cold start')
    expect(first('дезагрегации')).toBe('Disaggregated serving')
    expect(first('эмбеддинги')).toBe('Embedding')
    expect(first('реплики')).toBe('Replica')
    expect(first('шаблона')).toBe('Chat template')
    expect(first('префикс кэш')).toBe('Prefix caching')
    expect(first('спекулятивный')).toBe('Speculative decoding')
    expect(first('tool calls')).toContain('Function calling')
  })
  it('основы подсвечиваются, если запроса целиком в тексте нет', () => {
    expect(highlightParts('Квантизация весов', 'квантизации').some((p) => p.hit && p.text === 'Квантизаци')).toBe(true)
    expect(highlightParts('kv_cache_dtype', 'kv-cache-dtype')).toEqual([{ text: 'kv_cache_dtype', hit: true }])
  })
  it('синонимы-повторы названия не показываются на карточке', () => {
    const kv = GLOSSARY.find((e) => e.term === 'KV cache')!
    expect(visibleAliases(kv).map((a) => a.toLowerCase())).not.toContain('kv cache')
    expect(visibleAliases(kv).length).toBeLessThan(kv.aliases.length)
  })
  it('подсветка и slug', () => {
    expect(highlightParts('KV cache хранит', 'cache')).toEqual([
      { text: 'KV ', hit: false },
      { text: 'cache', hit: true },
      { text: ' хранит', hit: false },
    ])
    expect(slug('Tiered KV cache (G1–G4 / L1–L3)')).toBe('tiered-kv-cache-g1-g4-l1-l3')
  })
})
