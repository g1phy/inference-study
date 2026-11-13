import { describe, expect, it } from 'vitest'
import { loadO200kTokenizer } from './o200k'

describe('o200k через gpt-tokenizer (числа из notes-tokens §1.7, §2.8, §3.6, §3.15)', async () => {
  const h = await loadO200kTokenizer('o200k_harmony')
  const base = await loadO200kTokenizer('o200k_base')

  it('кэш модуля: повторная загрузка отдаёт тот же объект', async () => {
    expect(await loadO200kTokenizer()).toBe(h)
  })

  it('метаданные', () => {
    expect(h.name).toContain('gpt-oss')
    expect(h.vocabSize).toBe(200019)
    expect(h.specialTokens.find((s) => s.text === '<|start|>')!.id).toBe(200006)
  })

  it("'🤖' → [50378, 244]: первый токен — 3 из 4 байтов, text с U+FFFD", () => {
    const toks = h.encode('🤖')
    expect(toks.map((t) => t.id)).toEqual([50378, 244])
    expect(toks[0].bytes).toEqual([240, 159, 164])
    expect(toks[0].text).toBe('�')
    expect(toks[1].bytes).toEqual([150])
    expect(h.decode([50378, 244])).toBe('🤖')
    expect(Array.from(h.decodeBytes([50378]))).toEqual([240, 159, 164])
  })

  it("'👍🏽' → [82514, 52622, 121] (модификатор тона разрезан)", () => {
    expect(h.encode('👍🏽').map((t) => t.id)).toEqual([82514, 52622, 121])
  })

  it("'Привет, мир!' → 5 токенов, а у Qwen3 их 6", () => {
    expect(h.encode('Привет, мир!').map((t) => t.id)).toEqual([23881, 131903, 11, 37934, 0])
  })

  it("цифры группами по 3: '12345678' → '123' '456' '78'", () => {
    expect(h.encode('12345678').map((t) => t.text)).toEqual(['123', '456', '78'])
  })

  it('Harmony-разметка со спецтокенами', () => {
    const ids = h.encode('<|start|>assistant<|channel|>final<|message|>4<|return|>').map((t) => t.id)
    expect(ids).toEqual([200006, 173781, 200005, 17196, 200008, 19, 200002])
    expect(h.token(200006).special).toBe(true)
    expect(h.decode(ids, { skipSpecial: true })).toBe('assistantfinal4')
  })

  it('allowSpecial=false кодирует литерал спецтокена как текст', () => {
    expect(h.encode('a <|endoftext|> b').map((t) => t.id)).toEqual([64, 220, 199999, 287])
    expect(h.encode('a <|endoftext|> b', { allowSpecial: false }).map((t) => t.id)).toEqual([
      64, 464, 91, 419, 1440, 919, 91, 29, 287,
    ])
  })

  it('смещения UTF-16: токен с половиной эмодзи покрывает весь символ', () => {
    const toks = h.encode('Hi 🤖!')
    expect(toks.map((t) => t.id)).toEqual([12194, 93643, 244, 0])
    // 93643 = байты 20 F0 9F A4 (пробел + 3 из 4 байтов 🤖), 244 = 96
    expect(toks.map((t) => [t.start, t.end])).toEqual([
      [0, 2],
      [2, 5],
      [3, 5],
      [5, 6],
    ])
    expect(toks[1].raw.startsWith('Ġ')).toBe(true)
    expect(toks[1].text).toBe(' �')
    expect(h.idOf(toks[1].raw)).toBe(93643)
  })

  it('preTokenize: o200k режет camelCase и прилепляет сокращения (notes §3.5)', () => {
    expect(h.preTokenize('HelloWorld camelCase JSONParser')).toEqual(['Hello', 'World', ' camel', 'Case', ' JSONParser'])
    expect(h.preTokenize("I'm  learning")[0]).toBe("I'm")
  })

  it('trace совпадает с encode', () => {
    for (const w of ['🤖', ' Hello', 'Привет', ' tokenization']) {
      expect(h.trace(w).tokens.map((t) => t.id)).toEqual(h.encode(w).map((t) => t.id))
    }
    const tr = h.trace('🤖')
    expect(tr.steps[0].pieces).toEqual(['\\xF0', '\\x9F', '\\xA4', '\\x96'])
  })

  it('o200k_base: обычный текст кодируется так же', () => {
    expect(base.encode('Привет, мир!').map((t) => t.id)).toEqual([23881, 131903, 11, 37934, 0])
    expect(base.name).toContain('o200k_base')
  })
})

describe('o200k vs tiktoken 0.14 на состязательных строках (fixtures/qwen3-adversarial.json)', async () => {
  const { readProjectFile } = await import('./testing')
  const fx = JSON.parse(readProjectFile('src/engine/tokenizer/fixtures/qwen3-adversarial.json')) as {
    o200kSource: string | null
    encode: Array<{ text: string; o200kBase?: number[]; o200kHarmony?: number[] }>
  }
  const h = await loadO200kTokenizer('o200k_harmony')
  const base = await loadO200kTokenizer('o200k_base')

  it('o200k_base encode_ordinary и o200k_harmony encode(allowed_special=all) совпадают id в id', () => {
    expect(fx.o200kSource).toContain('tiktoken')
    const bad: string[] = []
    for (const c of fx.encode) {
      if (!c.o200kBase || !c.o200kHarmony) continue
      if (JSON.stringify(base.encode(c.text, { allowSpecial: false }).map((t) => t.id)) !== JSON.stringify(c.o200kBase)) bad.push('base ' + c.text)
      if (JSON.stringify(h.encode(c.text).map((t) => t.id)) !== JSON.stringify(c.o200kHarmony)) bad.push('harmony ' + c.text)
    }
    expect(bad).toEqual([])
  })

  it('смещения покрывают текст без дыр и нахлёстов на границах спецтокенов', () => {
    for (const c of fx.encode.slice(0, 200)) {
      const toks = h.encode(c.text)
      for (const t of toks) {
        expect(t.start).toBeDefined()
        expect(t.end! >= t.start!).toBe(true)
      }
      if (toks.length) expect(toks.at(-1)!.end).toBe(c.text.length)
    }
  })
})

describe('o200k_harmony: reserved-спецтокены (баг: RangeError на <|reserved_200000|>)', async () => {
  const h = await loadO200kTokenizer('o200k_harmony')

  it('все литералы <|reserved_N|>, которые gpt-tokenizer считает спецтокенами, кодируются без исключения', () => {
    for (const id of [200000, 200001, 200004, 200009, 200010, 200011, 200013, 200017, 201087]) {
      const toks = h.encode(`a<|reserved_${id}|>b`)
      expect(toks.map((t) => t.id)).toEqual([64, id, 65])
      expect(toks[1]).toMatchObject({ text: `<|reserved_${id}|>`, special: true, hfSpecial: true, start: 1, end: 20 })
      expect(toks[2]).toMatchObject({ start: 20, end: 21 })
      expect(h.preTokenize(`a<|reserved_${id}|>b`)).toEqual(['a', `<|reserved_${id}|>`, 'b'])
      expect(h.idOf(`<|reserved_${id}|>`)).toBe(id)
    }
  })

  it('<|reserved_200018|> — синоним <|endofprompt|>: id 200018, смещения по реальному литералу', () => {
    const toks = h.encode('a<|reserved_200018|>b')
    expect(toks.map((t) => t.id)).toEqual([64, 200018, 65])
    expect(toks[1].text).toBe('<|endofprompt|>')
    expect([toks[1].start, toks[1].end, toks[2].start]).toEqual([1, 20, 20])
  })

  it('номер вне списка или с ведущим нулём — обычный текст', () => {
    expect(h.encode('<|reserved_199|>').every((t) => !t.special)).toBe(true)
    expect(h.encode('<|reserved_0200000|>').every((t) => !t.special)).toBe(true)
    expect(h.preTokenize('<|reserved_199|>')).not.toContain('<|reserved_199|>')
  })
})

describe('o200k trace(): последний шаг = encode() и для строки из нескольких pre-tokens', async () => {
  const h = await loadO200kTokenizer('o200k_harmony')
  it('слова с пробелами, цифры, спецтокены', () => {
    for (const s of ['a  b', 'Hello world', '12345678', 'Привет, мир!', 'x<|start|>y', 'HelloWorld', "I'm  here\n\n"]) {
      expect(h.trace(s).tokens.map((t) => t.id), s).toEqual(h.encode(s).map((t) => t.id))
    }
    expect(h.trace('x<|start|>y').steps[0].rawPieces).toEqual(['x', '<|start|>', 'y'])
  })
})
