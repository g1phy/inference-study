import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { pyJson } from '@/engine/templates'
import { grammarState } from '@/engine/structured'
import type { JsonSchema } from '@/engine/types'
import { PRESETS, chatterFor, exampleValue, parseSchema, serialize, tryParse, validate } from './schemas'
import { EOS_ID, WHITESPACE, buildVocabStore, freeTrajectory, fullMask, maskedTrajectory } from './trajectory'
import { forcedPieces, jumpForwardCost } from './jumpforward'

const tk = loadQwen3ForTests()

describe('схемы', () => {
  it('пресеты: пример ответа проходит свою схему, болтовня нет', () => {
    for (const p of PRESETS) {
      expect(validate(p.instance, p.schema)).toEqual([])
      expect(p.chatter.startsWith(p.opener)).toBe(true)
      const parsed = tryParse(p.chatter)
      expect(parsed.whole).toBe(false)
      expect(parsed.value === undefined || validate(parsed.value, p.schema).length > 0).toBe(true)
    }
  })

  it('serialize повторяет json.dumps (разделители ", " и ": ")', () => {
    for (const p of PRESETS) expect(serialize(p.instance, p.schema).text).toBe(pyJson(p.instance))
  })

  it('parseSchema принимает подмножество и отклоняет остальное', () => {
    expect(parseSchema('{"type":"object","properties":{"a":{"type":"string"}}}').ok).toBe(true)
    expect(parseSchema('{').ok).toBe(false)
    expect(parseSchema('{"type":"string"}').ok).toBe(false)
    expect(parseSchema('{"type":"object","properties":{"a":{"type":"date"}}}').ok).toBe(false)
    expect(parseSchema('{"type":"object","properties":{"a":{"type":"string"}},"required":["b"]}').ok).toBe(false)
  })

  it('parseSchema: enum без type и ключевые слова, которые маска не проверяет', () => {
    const e = parseSchema('{"type":"object","properties":{"level":{"enum":["low","high"]},"n":{"enum":[1,2]}}}')
    expect(e.ok).toBe(true)
    const es = (e as { schema: JsonSchema }).schema
    expect(es.properties?.level.type).toBe('string')
    expect(es.properties?.n.type).toBe('integer')
    expect(parseSchema('{"type":"object","properties":{"x":{"enum":["a",1]}}}').ok).toBe(false)
    const f = parseSchema('{"type":"object","description":"d","properties":{"email":{"type":"string","format":"email"},"n":{"type":"integer","minimum":5}}}')
    expect(f.ok).toBe(true)
    expect((f as { ignored: string[] }).ignored.sort()).toEqual(['format', 'minimum'])
  })

  it('additionalProperties: true или схема не идут в ignored, лишний ключ разрешён, его значение проверяется', () => {
    const r = parseSchema('{"type":"object","properties":{"answer":{"type":"string"}},"additionalProperties":true}')
    expect(r.ok).toBe(true)
    const { schema, ignored, extraKeys } = r as { schema: JsonSchema; ignored: string[]; extraKeys: boolean }
    expect(ignored).toEqual([])
    expect(extraKeys).toBe(true)
    expect(validate({ answer: 'да', extra: 1 }, schema)).toEqual([])
    const sub = parseSchema('{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":{"type":"integer"}}')
    expect(sub.ok).toBe(true)
    const subSchema = (sub as { schema: JsonSchema }).schema
    expect((sub as { extraKeys: boolean }).extraKeys).toBe(true)
    expect(validate({ a: 'x', b: 2 }, subSchema)).toEqual([])
    expect(validate({ a: 'x', b: 'два' }, subSchema)).toHaveLength(1)
    // без additionalProperties лишний ключ остаётся ошибкой (как strict_mode в XGrammar)
    const strict = parseSchema('{"type":"object","properties":{"a":{"type":"string"}}}')
    expect((strict as { extraKeys: boolean }).extraKeys).toBe(false)
    expect(validate({ a: 'x', b: 1 }, (strict as { schema: JsonSchema }).schema)).toEqual(['лишний ключ "b"'])
    expect(parseSchema('{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":"да"}').ok).toBe(false)
    const any = parseSchema('{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":{}}')
    expect(any.ok).toBe(true)
    expect(validate({ a: 'x', b: [1] }, (any as { schema: JsonSchema }).schema)).toEqual([])
  })

  it('своя схема из одних строк: объект, вырезанный из болтовни, проходит схему, хотя ответ целиком не JSON', () => {
    const r = parseSchema('{"type":"object","properties":{"answer":{"type":"string"}}}')
    const schema = (r as { schema: JsonSchema }).schema
    const parsed = tryParse(chatterFor(schema))
    expect(parsed.whole).toBe(false)
    expect(parsed.value).toBeDefined()
    expect(validate(parsed.value, schema)).toEqual([])
  })

  it('exampleValue и chatterFor для своей схемы', () => {
    const r = parseSchema('{"type":"object","properties":{"title":{"type":"string"},"n":{"type":"integer"},"ok":{"type":"boolean"},"tags":{"type":"array","items":{"type":"string"}}},"required":["title"]}')
    expect(r.ok).toBe(true)
    const schema = (r as { schema: JsonSchema }).schema
    const v = exampleValue(schema)
    expect(validate(v, schema)).toEqual([])
    const bad = tryParse(chatterFor(schema))
    expect(bad.whole).toBe(false)
  })
})

describe('траектории на настоящем словаре', () => {
  it('под маской: выбранный токен всегда разрешён и самый вероятный после перенормировки', () => {
    for (const p of PRESETS) {
      const { steps, target } = maskedTrajectory(tk, p.schema, p.instance, p.opener)
      expect(steps.at(-1)!.chosen.id).toBe(EOS_ID)
      for (const s of steps) {
        expect(s.valid).toBe(true)
        const i = s.cands.findIndex((c) => c.chosen)
        expect(s.cands[i].allowed).toBe(true)
        expect(Math.max(...s.finalProbs)).toBeCloseTo(s.finalProbs[i], 10)
        const sum = s.finalProbs.reduce((a, b) => a + b, 0)
        expect(sum).toBeCloseTo(1, 10)
        s.cands.forEach((c, k) => {
          if (!c.allowed) expect(s.finalProbs[k]).toBe(0)
        })
      }
      // итоговый текст полный и валидный
      expect(grammarState(p.schema, target, { whitespace: WHITESPACE }).complete).toBe(true)
      expect(validate(JSON.parse(target), p.schema)).toEqual([])
    }
  })

  it('первый шаг одинаков с маской и без: модель «хочет» начать со слова', () => {
    const p = PRESETS[0]
    const m = maskedTrajectory(tk, p.schema, p.instance, p.opener)
    const f = freeTrajectory(tk, p.schema, p.chatter, m.steps[0])
    expect(f[0].probs).toEqual(m.steps[0].probs)
    expect(f[0].chosen.id).toBe(m.steps[0].cands[0].token.id)
    expect(f[1].valid).toBe(false)
  })

  it('маска по всему словарю: числа как в заметках (XGrammar, any_whitespace=False)', () => {
    const p = PRESETS[0]
    const store = buildVocabStore(tk)
    const { steps } = maskedTrajectory(tk, p.schema, p.instance, p.opener)
    const count = (k: number) => fullMask(p.schema, steps[k].prefix, store).reduce((a, b) => a + b, 0)
    expect(count(0)).toBe(2) // '{' и '{"'
    expect(count(1)).toBe(4) // 'n' 'na' 'nam' 'name'
    expect(count(2)).toBe(2) // '"' и '":'
    expect(count(4)).toBe(147036) // внутри строки: почти весь словарь (XGrammar 147 061, он ещё пускает 25 added tokens)
    expect(count(12)).toBe(11) // '-' и 10 цифр
    expect(count(steps.length - 1)).toBe(1) // только <|im_end|>
  })
})

describe('jump-forward', () => {
  it('схема человека: принудительные куски как у XGrammar', () => {
    const p = PRESETS[0]
    const { text } = serialize(p.instance, p.schema)
    const forced = forcedPieces(p.schema, text)
      .filter((x) => x.forced)
      .map((x) => text.slice(x.start, x.end))
    expect(forced[0]).toBe('{"name": "')
    expect(forced).toContain(', "age": ')
    const cost = jumpForwardCost(tk.encode(text), forcedPieces(p.schema, text))
    expect(cost.plain).toBe(25)
    expect(cost.jf).toBeLessThan(cost.plain)
  })
})
