import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { incrementalDetokenize, naiveDetokenize } from '@/engine/detok'
import { byteRole, codePointName, countReplacement, onlyModifiers, pendingChar, shownText, utf8SeqLen, visibleText } from './utf8'

describe('utf8 helpers', () => {
  it('длина символа по ведущему байту', () => {
    expect(utf8SeqLen(0x4b)).toBe(1)
    expect(utf8SeqLen(0xd1)).toBe(2)
    expect(utf8SeqLen(0xe6)).toBe(3)
    expect(utf8SeqLen(0xf0)).toBe(4)
    expect(utf8SeqLen(0x89)).toBe(0)
    expect(byteRole(0x9f)).toBe('cont')
    expect(byteRole(0xf0)).toBe('lead')
  })

  it('находит недописанный символ в конце', () => {
    expect(pendingChar([0x20, 0xd1])).toEqual({ bytes: [0xd1], need: 2, before: [0x20] })
    expect(pendingChar([0x20, 0xf0, 0x9f, 0x9a])).toEqual({ bytes: [0xf0, 0x9f, 0x9a], need: 4, before: [0x20] })
    expect(pendingChar([0xd1, 0x89])).toBeNull()
    expect(pendingChar([])).toBeNull()
  })

  it('видимые пробелы и счёт «�»', () => {
    expect(visibleText(' a\nb')).toBe('·a↵b')
    expect(countReplacement('a��b�')).toBe(3)
  })

  it('модификаторы эмодзи записываются кодами', () => {
    expect(onlyModifiers('🏽')).toEqual([{ cp: 0x1f3fd, kind: 'skin' }])
    expect(onlyModifiers('\u200d')).toEqual([{ cp: 0x200d, kind: 'zwj' }])
    expect(onlyModifiers('👍🏽')).toBeNull()
    expect(onlyModifiers('')).toBeNull()
    expect(codePointName(0x200d)).toBe('U+200D')
    expect(shownText(' 🏽')).toBe('·[U+1F3FD]')
  })
})

describe('пример фигуры на настоящем словаре Qwen3', () => {
  const tk = loadQwen3ForTests()
  const text = 'KV cache щадит GPU и экономит время 🚀'
  const tokens = tk.encode(text)

  it('буква «щ» и эмодзи разрезаны между токенами', () => {
    const partial = tokens.filter((t) => pendingChar(t.bytes) !== null || byteRole(t.bytes[0]) === 'cont').length
    expect(partial).toBeGreaterThanOrEqual(4)
    expect(countReplacement(naiveDetokenize(tokens, tk))).toBe(5)
  })

  it('инкрементальный вывод совпадает с decode всей строки и без «�»', () => {
    const steps = incrementalDetokenize(tokens, tk)
    const out = steps.map((s) => s.emitted).join('')
    expect(out).toBe(text)
    expect(countReplacement(out)).toBe(0)
    // на шагах с половиной символа ничего не отдаётся
    const held = steps.filter((s) => s.heldBytes.length > 0)
    expect(held.length).toBe(3)
    for (const s of held) expect(s.emitted).toBe('')
  })

  it('stop-строка </answer>: оба режима обрезают одинаково', () => {
    const answer = '<answer>KV cache хранит ключи и значения прошлых токенов.</answer>\nПользователь: а зачем?'
    const toks = tk.encode(answer, { allowSpecial: false })
    const jail = incrementalDetokenize(toks, tk, { stop: ['</answer>'], stopMode: 'jail' })
    const vllm = incrementalDetokenize(toks, tk, { stop: ['</answer>'], stopMode: 'vllm' })
    const last = (l: typeof jail) => l[l.length - 1]
    expect(last(jail).stopped).toBe(true)
    expect(last(vllm).stopped).toBe(true)
    expect(jail.length).toBe(vllm.length)
    expect(last(jail).output).toBe('<answer>KV cache хранит ключи и значения прошлых токенов.')
    expect(last(vllm).output).toBe(last(jail).output)
    // jail держит в среднем намного меньше, чем фиксированный хвост
    const avg = (l: typeof jail) => l.reduce((a, s) => a + s.stopHeld.length, 0) / l.length
    expect(avg(jail)).toBeLessThan(avg(vllm) / 4)
  })
})
