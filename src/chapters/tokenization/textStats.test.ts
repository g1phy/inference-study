import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { charCount, classifyPiece, overlapping, pieceSpans, utf8Length, wordOfToken } from './textStats'
import { EN_PHRASE, RU_PHRASE } from './LiveTokenizer'

describe('textStats', () => {
  it('символы считаются графемами, байты по UTF-8', () => {
    expect(charCount('Привет')).toBe(6)
    expect(charCount('🧑‍💻')).toBe(1)
    expect(utf8Length('Пр')).toBe(4)
    expect(utf8Length('🙂')).toBe(4)
  })

  it('классы пре-токенов', () => {
    expect(classifyPiece(' learning')).toBe('word')
    expect(classifyPiece('2')).toBe('digit')
    expect(classifyPiece('123')).toBe('digit')
    expect(classifyPiece('!!\n\n')).toBe('punct')
    expect(classifyPiece('  ')).toBe('space')
    expect(classifyPiece("'m")).toBe('contraction')
    expect(classifyPiece('<|im_start|>', true)).toBe('special')
  })

  it('слово под токеном: все токены пре-токена « простыми»', () => {
    const tk = loadQwen3ForTests()
    const text = 'Объясни простыми словами'
    const tokens = tk.encode(text)
    const pieces = tk.preTokenize(text)
    expect(pieceSpans(text, pieces)).not.toBeNull()
    const i = tokens.findIndex((t) => t.text === ' прост')
    const w = wordOfToken(text, pieces, tokens, i)!
    expect(w.word).toBe(' простыми')
    expect(w.indices.map((k) => tokens[k].text).join('')).toBe(' простыми')
    // trace этого слова даёт те же токены, что encode
    expect(tk.trace(w.word).tokens.map((t) => t.id)).toEqual(w.indices.map((k) => tokens[k].id))
  })

  it('пересечение диапазонов токенов', () => {
    const tk = loadQwen3ForTests()
    const tokens = tk.encode('Привет, мир')
    const all = overlapping(tokens, 0, 6)
    expect(all.map((k) => tokens[k].text).join('')).toBe('Привет')
  })

  it('русская фраза длиннее английской в словаре Qwen3', () => {
    const tk = loadQwen3ForTests()
    const ru = tk.encode(RU_PHRASE).length
    const en = tk.encode(EN_PHRASE).length
    expect(ru / en).toBeGreaterThan(1.2)
  })
})
