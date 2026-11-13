/**
 * Независимая проверка токенизатора Qwen3 на состязательных строках (эталон: HF tokenizers 0.23.3,
 * fixtures/gen_adversarial.py): случайные смеси Unicode, серии пробелов, цифры, ZWJ-эмодзи, \r\n,
 * литералы спецтокенов вплотную к тексту, NFC, буквы Unicode 17, управляющие символы, фузз по веткам регулярки.
 */
import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests, readProjectFile } from './testing'

interface EncodeCase {
  cat: string
  text: string
  ids: number[]
  splitIds?: number[]
  offsets?: number[]
  decoded: string
  decodedSkip: string
  pretokens?: string[]
}

const fx = JSON.parse(readProjectFile('src/engine/tokenizer/fixtures/qwen3-adversarial.json')) as {
  encode: EncodeCase[]
  decode: Array<{ ids: number[]; decoded: string; decodedSkip: string }>
}
const tok = loadQwen3ForTests()
const label = (c: EncodeCase, i: number) => `#${i} [${c.cat}] ${JSON.stringify(c.text).slice(0, 50)}`

describe('Qwen3 vs HF tokenizers: состязательные строки', () => {
  it('в фикстуре 250+ строк всех категорий', () => {
    expect(fx.encode.length).toBeGreaterThanOrEqual(250)
    const cats = new Set(fx.encode.map((c) => c.cat))
    for (const c of ['mix', 'fuzz', 'ws', 'digits', 'emoji', 'crlf', 'special', 'contraction', 'nfc', 'unicode17', 'control', 'code', 'long']) {
      expect(cats.has(c)).toBe(true)
    }
  })

  fx.encode.forEach((c, i) => {
    it(`ids ${label(c, i)}`, () => {
      expect(tok.encode(c.text).map((t) => t.id)).toEqual(c.ids)
      if (c.splitIds) expect(tok.encode(c.text, { allowSpecial: false }).map((t) => t.id)).toEqual(c.splitIds)
    })
  })

  it('смещения (UTF-16) совпадают с HF для всех NFC-стабильных строк', () => {
    const bad: string[] = []
    fx.encode.forEach((c, i) => {
      if (!c.offsets) return
      const got = tok.encode(c.text).flatMap((t) => [t.start, t.end])
      if (JSON.stringify(got) !== JSON.stringify(c.offsets)) bad.push(label(c, i))
    })
    expect(bad).toEqual([])
  })

  it('decode с skip_special и без совпадает с HF', () => {
    const bad: string[] = []
    fx.encode.forEach((c, i) => {
      if (tok.decode(c.ids) !== c.decoded || tok.decode(c.ids, { skipSpecial: true }) !== c.decodedSkip) bad.push(label(c, i))
    })
    expect(bad).toEqual([])
  })

  it('preTokenize совпадает с Split(Isolated) HF после NFC', () => {
    const bad: string[] = []
    fx.encode.forEach((c, i) => {
      if (c.pretokens && JSON.stringify(tok.preTokenize(c.text)) !== JSON.stringify(c.pretokens)) bad.push(label(c, i))
    })
    expect(bad).toEqual([])
  })

  it('decode случайных последовательностей id (неполный UTF-8, спецтокены) = from_utf8_lossy в HF', () => {
    const bad: number[][] = []
    for (const c of fx.decode) {
      if (tok.decode(c.ids) !== c.decoded || tok.decode(c.ids, { skipSpecial: true }) !== c.decodedSkip) bad.push(c.ids)
    }
    expect(bad).toEqual([])
  })
})

describe('trace(): пошаговое BPE согласовано с encode', () => {
  const words = new Set<string>()
  for (const c of fx.encode) for (const p of tok.preTokenize(c.text)) if (!tok.specialTokens.some((s) => s.text === p)) words.add(p)

  it(`последний шаг trace = bpeWord для ${words.size} pre-tokens, каждый шаг применяет ровно одну склейку`, () => {
    const bad: string[] = []
    for (const w of words) {
      const tr = tok.trace(w)
      const ids = tr.tokens.map((t) => t.id)
      if (JSON.stringify(ids) !== JSON.stringify(tok.bpeWord(w.normalize('NFC')))) bad.push(w)
      // steps[0] — отдельные байты
      if (tr.steps[0].rawPieces.length !== tr.bytes.length) bad.push('bytes:' + w)
      for (let s = 0; s + 1 < tr.steps.length; s++) {
        const { rawPieces } = tr.steps[s]
        const m = tr.steps[s].merge!
        // склейка — минимальный ранг среди соседних пар, при равенстве левее
        let best = Infinity
        let bestAt = -1
        for (let k = 0; k + 1 < rawPieces.length; k++) {
          const r = tok.mergeRank(tok.idOf(rawPieces[k])!, tok.idOf(rawPieces[k + 1])!)
          if (r !== undefined && r < best) {
            best = r
            bestAt = k
          }
        }
        if (m.rank !== best || m.index !== bestAt || m.left !== rawPieces[m.index] || m.right !== rawPieces[m.index + 1]) bad.push('rank:' + w)
        const expected = [...rawPieces.slice(0, m.index), m.left + m.right, ...rawPieces.slice(m.index + 2)]
        if (JSON.stringify(expected) !== JSON.stringify(tr.steps[s + 1].rawPieces)) bad.push('step:' + w)
      }
      const last = tr.steps.at(-1)!
      if (last.merge) bad.push('last:' + w)
      for (let k = 0; k + 1 < last.rawPieces.length; k++) {
        if (tok.mergeRank(tok.idOf(last.rawPieces[k])!, tok.idOf(last.rawPieces[k + 1])!) !== undefined) bad.push('unmerged:' + w)
      }
      if (last.rawPieces.join('') !== tr.tokens.map((t) => t.raw).join('')) bad.push('raw:' + w)
    }
    expect(bad).toEqual([])
  })

  it('trace() целой строки (несколько pre-tokens, спецтокены) заканчивается ровно encode() — все строки фикстуры', () => {
    const bad: string[] = []
    for (const c of fx.encode) {
      const tr = tok.trace(c.text)
      if (JSON.stringify(tr.tokens.map((t) => t.id)) !== JSON.stringify(c.ids)) bad.push(c.text)
      if (tr.pretoken !== tok.decode(c.ids)) bad.push('pretoken:' + c.text)
    }
    expect(bad).toEqual([])
    // склейка через границу pre-tokens запрещена: '  b' не склеивается в '  ' + 'b'
    expect(tok.trace('a  b').tokens.map((t) => t.text)).toEqual(['a', ' ', ' b'])
    expect(tok.trace('x<think>y').steps[0].rawPieces).toEqual(['x', '<think>', 'y'])
  })
})

describe('смещения для строк, которые меняет NFC', () => {
  it('определены у всех токенов, не убывают, лежат в строке и доходят до её конца (раньше хангыль NFD оставался без смещений)', () => {
    for (const c of fx.encode) {
      if (c.text.normalize('NFC') === c.text) continue
      const toks = tok.encode(c.text)
      let prev = 0
      for (const t of toks) {
        expect(t.start, c.text).toBeDefined()
        expect(t.start! >= prev && t.end! >= t.start! && t.end! <= c.text.length, c.text).toBe(true)
        prev = t.start!
      }
      expect(toks.at(-1)!.end, c.text).toBe(c.text.length)
    }
  })

  it('чамо хангыля склеиваются NFC через границы кластеров: каждый слог покрывает свои чамо', () => {
    const toks = tok.encode('\u1100\u1161\u11a8 \u1100\u1161')
    expect(toks.map((t) => [t.text, t.start, t.end])).toEqual([
      ['\uac01', 0, 3],
      [' \uac00', 3, 6],
    ])
  })
})
