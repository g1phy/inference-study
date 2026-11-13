/**
 * Маски structured.ts против XGrammar 0.2.9 на ВСЁМ словаре Qwen3 (151 669 токенов), шаг за шагом:
 * whitespace: 'xgrammar' ↔ compile_json_schema(any_whitespace=False). Эталон: число разрешённых id и FNV-1a
 * хэш их списка (tokenizer/fixtures/gen_xgrammar_masks.py). Внутри строки маска байтовая: токены с битым
 * UTF-8 запрещены, недописанный символ (половина эмодзи) разрешает только продолжения.
 */
import { describe, expect, it } from 'vitest'
import { allowedMaskAfterTokens } from './structured'
import { loadQwen3ForTests, readProjectFile } from './tokenizer/testing'
import type { JsonSchema } from './types'

const fx = JSON.parse(readProjectFile('src/engine/structured.xgrammar.fixture.json')) as {
  source: string
  cases: Array<{ name: string; schema: JsonSchema; target: string; ids: number[]; steps: Array<{ count: number; hash: number }> }>
}
const tok = loadQwen3ForTests()
const all = Array.from({ length: tok.vocabSize }, (_, i) => tok.token(i))

function fnv(ids: number[]): number {
  let h = 0x811c9dc5
  for (const id of ids) {
    for (let k = 0; k < 4; k++) {
      h ^= (id >>> (8 * k)) & 0xff
      h = Math.imul(h, 0x01000193) >>> 0
    }
  }
  return h
}

describe(`маски = ${fx.source}`, () => {
  it('эталон notes-tokens §3.12: 2, 4, 2, 121, 147 061, …, 2, 3, 2, 2, 11, 11, 11, 1', () => {
    expect(fx.cases[0].steps.map((s) => s.count)).toEqual([2, 4, 2, 121, 147061, 147061, 147061, 2, 3, 2, 2, 11, 11, 11, 1])
  })
  for (const c of fx.cases) {
    it(c.name, () => {
      expect(tok.encode(c.target).map((t) => t.id)).toEqual(c.ids)
      const got = c.steps.map((_, k) => {
        const prefix = c.ids.slice(0, k).map((id) => tok.token(id))
        const mask = allowedMaskAfterTokens(c.schema, prefix, all, { whitespace: 'xgrammar', addedTokensAsText: true })
        const ids: number[] = []
        mask.forEach((m, i) => m && ids.push(i))
        return { count: ids.length, hash: fnv(ids) }
      })
      expect(got).toEqual(c.steps)
    }, 30000)
  }

  it('по умолчанию спецтокены запрещены: внутри строки ровно на 25 токенов меньше, чем у XGrammar', () => {
    const c = fx.cases[0]
    const prefix = c.ids.slice(0, 4).map((id) => tok.token(id)) // '{"name": "'
    const strict = allowedMaskAfterTokens(c.schema, prefix, all, { whitespace: 'xgrammar' }).filter(Boolean).length
    expect(strict).toBe(c.steps[4].count - 25)
  })
})
