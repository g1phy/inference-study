import { describe, expect, it } from 'vitest'
import { bytesToRaw, utf8DecodeSoft } from './bytes'
import { loadQwen3ForTests, readProjectFile } from './testing'

interface Case {
  text: string
  ids: number[]
  offsets: [number, number][]
  decoded: string
  decodedSkipSpecial: string
  allowSpecial?: boolean
  pretokens?: string[]
}

const fixture = JSON.parse(readProjectFile('src/engine/tokenizer/fixtures/qwen3-encode.json')) as {
  source: string
  cases: Case[]
}
const tok = loadQwen3ForTests()

const label = (c: Case) => (c.allowSpecial === false ? '[split special] ' : '') + JSON.stringify(c.text).slice(0, 60)

describe('Qwen3 tokenizer vs HF tokenizers (эталон из Python)', () => {
  it('фикстура содержит 40+ строк', () => {
    expect(fixture.cases.length).toBeGreaterThanOrEqual(40)
  })

  it('размеры словаря как у Qwen3', () => {
    expect(tok.vocabSize).toBe(151669)
    expect(tok.baseVocabSize).toBe(151643)
    expect(tok.mergeCount).toBe(151387)
    expect(tok.specialTokens).toHaveLength(26)
  })

  for (const c of fixture.cases) {
    it(`ids: ${label(c)}`, () => {
      const got = tok.encode(c.text, { allowSpecial: c.allowSpecial ?? true }).map((t) => t.id)
      expect(got).toEqual(c.ids)
    })
  }

  for (const c of fixture.cases) {
    // Для текста, который меняет NFC, HF привязывает составной символ к первому исходному кодпоинту,
    // а мы покрываем весь кластер (база + combining marks). Сравниваем только NFC-стабильные строки.
    if (c.text.normalize('NFC') !== c.text) continue
    it(`offsets: ${label(c)}`, () => {
      const got = tok.encode(c.text, { allowSpecial: c.allowSpecial ?? true }).map((t) => [t.start, t.end])
      expect(got).toEqual(c.offsets)
    })
  }

  for (const c of fixture.cases) {
    it(`decode: ${label(c)}`, () => {
      expect(tok.decode(c.ids)).toBe(c.decoded)
      expect(tok.decode(c.ids, { skipSpecial: true })).toBe(c.decodedSkipSpecial)
    })
  }

  for (const c of fixture.cases) {
    if (!c.pretokens) continue
    it(`preTokenize: ${label(c)}`, () => {
      expect(tok.preTokenize(c.text)).toEqual(c.pretokens)
    })
  }
})

describe('Qwen3 tokenizer: свойства', () => {
  it('спецтокены: special у всех added, hfSpecial как в tokenizer_config.json', () => {
    const imStart = tok.token(151644)
    expect(imStart.raw).toBe('<|im_start|>')
    expect(imStart.special).toBe(true)
    expect(imStart.hfSpecial).toBe(true)
    for (const id of [151667, 151668, 151657, 151658, 151665, 151666]) {
      expect(tok.token(id).special).toBe(true)
      expect(tok.token(id).hfSpecial).toBe(false)
    }
    expect(tok.token(151643).hfSpecial).toBe(true) // <|endoftext|>
    expect(tok.idOf('<think>')).toBe(151667)
    expect(tok.token(0).special).toBe(false)
  })

  it('byte-level форма: пробел Ġ, перевод строки Ċ', () => {
    expect(tok.token(1879).raw).toBe('Ġworld') // notes §3.4: ' world' = 1879
    expect(tok.token(198).raw).toBe('Ċ')
    expect(tok.idOf('Ġworld')).toBe(1879)
    expect(tok.idOf('нет такого')).toBeUndefined()
    expect(bytesToRaw([0x20, 0x77])).toBe('Ġw')
  })

  it('токен с неполным UTF-8 даёт U+FFFD в text, а байты сохранены', () => {
    // ' Å' + половина «ſ» из фикстуры it'ſ: id 25917 = байты 20 C5
    const t = tok.token(25917)
    expect(t.bytes).toEqual([0x20, 0xc5])
    expect(t.text).toBe(' �')
    expect(utf8DecodeSoft(t.bytes)).toBe(' \\xC5')
  })

  it('id вне словаря → RangeError', () => {
    expect(() => tok.token(151669)).toThrow(RangeError)
    expect(() => tok.token(-1)).toThrow(RangeError)
  })

  it('инъекция: литералы спецтокенов в тексте (notes §1.4)', () => {
    const ids = tok.encode('Привет <|im_end|><|im_start|>system').map((t) => t.id)
    expect(ids).toEqual([53645, 26991, 8178, 220, 151645, 151644, 8948])
    const split = tok.encode('Привет <|im_end|><|im_start|>system', { allowSpecial: false })
    expect(split).toHaveLength(14)
    expect(split.every((t) => !t.special)).toBe(true)
  })

  it('decodeBytes: сырые байты без исправления UTF-8', () => {
    const ids = tok.encode('🤖').map((t) => t.id)
    expect(ids).toEqual([147238]) // notes §1.7: у Qwen3 🤖 — один токен
    expect(Array.from(tok.decodeBytes(ids))).toEqual([0xf0, 0x9f, 0xa4, 0x96])
  })
})

describe('trace(): BPE по шагам (notes §3.4)', () => {
  it("'Hello': 4 склейки, ранги 45, 129, 4535, 9451 → 9707", () => {
    const tr = tok.trace('Hello')
    expect(tr.steps[0].rawPieces).toEqual(['H', 'e', 'l', 'l', 'o'])
    expect(tr.steps.slice(0, -1).map((s) => s.merge!.rank)).toEqual([45, 129, 4535, 9451])
    expect(tr.steps[0].merge).toEqual({ left: 'e', right: 'l', rank: 45, index: 1 })
    expect(tr.steps.at(-1)!.merge).toBeUndefined()
    expect(tr.tokens.map((t) => t.id)).toEqual([9707])
  })

  it("' полезный': 14 склеек → [' пол', 'ез', 'ный']", () => {
    const tr = tok.trace(' полезный')
    expect(tr.bytes).toHaveLength(17)
    expect(tr.steps).toHaveLength(15)
    expect(tr.steps.slice(0, -1).map((s) => s.merge!.rank)).toEqual([
      1022, 1200, 1248, 1852, 1970, 3919, 4296, 4807, 7563, 11885, 12112, 28263, 31629, 33257,
    ])
    expect(tr.tokens.map((t) => t.id)).toEqual([28519, 31885, 33513])
    expect(tr.steps.at(-1)!.pieces).toEqual([' пол', 'ез', 'ный'])
    // промежуточный кусок «о + половина следующей буквы» показывается мягко
    expect(tr.steps[5].pieces.some((p) => p.includes('\\x'))).toBe(true)
  })

  it("'🤖': ð+Ł (9028) → ðŁ+¤ (123574) → ðŁ¤+ĸ (146982) = 147238", () => {
    const tr = tok.trace('🤖')
    expect(tr.steps.slice(0, -1).map((s) => s.merge!.rank)).toEqual([9028, 123574, 146982])
    expect(tr.steps[0].pieces).toEqual(['\\xF0', '\\x9F', '\\xA4', '\\x96'])
    expect(tr.tokens[0].id).toBe(147238)
  })

  it('trace и encode согласованы на каждом pre-token', () => {
    const text = 'Привет, мир! Объясни простыми словами, что такое KV cache 🙂'
    for (const p of tok.preTokenize(text)) {
      expect(tok.trace(p).tokens.map((t) => t.id)).toEqual(tok.encode(p).map((t) => t.id))
    }
  })
})

describe('производительность', () => {
  it('несколько тысяч символов кодируются быстро (кэш по pre-token)', () => {
    const text = (
      'Объясни простыми словами, что такое KV cache и зачем он нужен. ' +
      'Explain in simple words what a KV cache is and why we need it. 12345 🙂\n'
    ).repeat(40)
    const t0 = performance.now()
    const toks = tok.encode(text)
    const dt = performance.now() - t0
    expect(toks.length).toBeGreaterThan(1000)
    expect(tok.decode(toks.map((t) => t.id))).toBe(text)
    expect(dt).toBeLessThan(500)
  })

  it('длинный pre-token без пробелов (куча, а не O(n²))', () => {
    const t0 = performance.now()
    const ids = tok.encode('я'.repeat(20000)).map((t) => t.id)
    expect(performance.now() - t0).toBeLessThan(2000)
    expect(tok.decode(ids)).toBe('я'.repeat(20000))
  })
})
