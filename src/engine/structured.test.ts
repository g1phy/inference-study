import { describe, expect, it } from 'vitest'
import { allowedMask, allowedMaskTokens, cursorAccepts, forcedContinuation, grammarCursor, grammarState } from './structured'
import { loadQwen3ForTests } from './tokenizer/testing'
import type { JsonSchema } from './types'

/** Схема из notes-tokens §3.12. */
const NA: JsonSchema = {
  type: 'object',
  properties: { name: { type: 'string' }, age: { type: 'integer' } },
  required: ['name', 'age'],
  additionalProperties: false,
}
const XG = { whitespace: 'xgrammar' as const }

describe('пошагово как в notes-tokens §3.12 (whitespace: xgrammar = any_whitespace=False)', () => {
  const tok = loadQwen3ForTests()
  const target = '{"name": "Иван", "age": 42}'
  const toks = tok.encode(target)

  it("токены Qwen3: '{\"' 'name' '\":' ' \"' 'И' 'ван' '\",' ' \"' 'age' '\":' ' ' '4' '2' '}'", () => {
    expect(toks.map((t) => t.text)).toEqual(['{"', 'name', '":', ' "', 'И', 'ван', '",', ' "', 'age', '":', ' ', '4', '2', '}'])
  })

  it('каждый целевой токен разрешён на своём шаге, EOS — только в конце', () => {
    let text = ''
    for (const t of toks) {
      expect(allowedMask(NA, text, [t.text, '<|im_end|>'], XG)).toEqual([true, false])
      text += t.text
    }
    expect(allowedMask(NA, text, ['<|im_end|>', '}', ' '], XG)).toEqual([true, false, false])
    expect(grammarState(NA, text, XG)).toEqual({ validPrefix: true, complete: true, expecting: 'конец ответа (EOS)' })
  })

  const step = (prefix: string, cands: string[]) => allowedMask(NA, prefix, cands, XG)

  it('шаг 0: только { и {"', () => {
    expect(step('', ['{', '{"', ' {', '"', '[', 'Конечно', ' '])).toEqual([true, true, false, false, false, false, false])
  })
  it('шаг 1: начало ключа name', () => {
    expect(step('{"', ['n', 'na', 'nam', 'name', 'name"', 'age', 'x'])).toEqual([true, true, true, true, true, false, false])
  })
  it('шаг 2: кавычка или \'":\'', () => {
    expect(step('{"name', ['"', '":', '":"', '": "', 's'])).toEqual([true, true, false, true, false])
  })
  it("шаг 3: ' ', ' \"', ' \"\"', ' \"\\\\', ' \",', ' \"<' (пробел обязателен)", () => {
    expect(step('{"name":', [' ', ' "', ' ""', ' "\\', ' ",', ' "<', '"', '  ', ' 4'])).toEqual([
      true, true, true, true, true, true, false, false, false,
    ])
  })
  it("шаг 4: внутри строки почти всё, но не '\\n'", () => {
    expect(step('{"name": "', ['И', 'ван', '"', '",', '\\', '\\n', '\n', '\t', '\u0001', '🙂', '{"'])).toEqual([
      true, true, true, true, true, true, false, false, false, true, true,
    ])
  })
  it("шаг 7: после '\",' только ' ' и ' \"'", () => {
    expect(step('{"name": "Иван",', [' ', ' "', '"', ' "age', ' "name'])).toEqual([true, true, false, true, false])
  })
  it("шаги 10–13: integer без ведущих нулей", () => {
    expect(step('{"name": "Иван", "age":', [' ', ' -', ' 4', '4', ' "'])).toEqual([true, true, true, false, false])
    expect(step('{"name": "Иван", "age": ', ['-', '0', '9', '}', '"', '-0'])).toEqual([true, true, true, false, false, false])
    expect(step('{"name": "Иван", "age": 4', ['0', '2', '}', ',', ' ', '.5'])).toEqual([true, true, true, false, false, false])
    expect(step('{"name": "Иван", "age": 0', ['1', '}'])).toEqual([false, true])
  })
  it('спецтокены запрещены, EOS — только у полного документа', () => {
    expect(step('{"name": "', ['<think>', '<|im_start|>'])).toEqual([false, false])
  })
})

describe('политика пробелов', () => {
  it("compact (по умолчанию): пробел после ':' и ',' необязателен, но не больше одного", () => {
    expect(grammarState(NA, '{"name":"Анна","age":29}').complete).toBe(true)
    expect(grammarState(NA, '{"name": "Анна", "age": 29}').complete).toBe(true)
    expect(grammarState(NA, '{"name": "Анна","age":29}').complete).toBe(true)
    expect(grammarState(NA, '{"name":  "Анна"').validPrefix).toBe(false)
    expect(grammarState(NA, '{ "name": "Анна"').validPrefix).toBe(false)
    expect(grammarState(NA, '{"name" : "Анна"').validPrefix).toBe(false)
    // токены через границу структуры в compact: '":"' (без пробела) и '": "' оба допустимы
    expect(allowedMask(NA, '{"name', ['":"', '": "', '":  "'])).toEqual([true, true, false])
    expect(allowedMask(NA, '{"name": "А', ['","', '", "', '",  "'])).toEqual([true, true, false])
  })
  it('xgrammar: пробел обязателен', () => {
    expect(grammarState(NA, '{"name":"Анна"', XG).validPrefix).toBe(false)
    expect(grammarState(NA, '{"name": "Анна","age": 1}', XG).validPrefix).toBe(false)
  })
  it('minified: пробелов нет совсем', () => {
    expect(grammarState(NA, '{"name":"Анна","age":29}', { whitespace: 'minified' }).complete).toBe(true)
    expect(grammarState(NA, '{"name": "Анна"', { whitespace: 'minified' }).validPrefix).toBe(false)
  })
})

describe('forcedContinuation (jump-forward, notes-tokens §3.13)', () => {
  it("xgrammar: '{\"name\": \"' в начале и ', \"age\": ' после имени", () => {
    expect(forcedContinuation(NA, '', XG)).toBe('{"name": "')
    expect(forcedContinuation(NA, '{"name": "Иван"', XG)).toBe(', "age": ')
    expect(forcedContinuation(NA, '{"name": "Ив', XG)).toBe('')
    expect(forcedContinuation(NA, '{"name": "Иван", "age": 42}', XG)).toBe('')
  })
  it('compact: необязательный пробел обрывает принудительный кусок', () => {
    expect(forcedContinuation(NA, '')).toBe('{"name":')
    expect(forcedContinuation(NA, '{"name": "Иван"')).toBe(',')
    expect(forcedContinuation(NA, '{"name":"Иван",')).toBe('')
  })
  it('minified', () => {
    expect(forcedContinuation(NA, '', { whitespace: 'minified' })).toBe('{"name":"')
    expect(forcedContinuation(NA, '{"name":"Иван"', { whitespace: 'minified' })).toBe(',"age":')
  })
  it('невалидный префикс → пустая строка', () => {
    expect(forcedContinuation(NA, 'Конечно, вот JSON:')).toBe('')
  })
})

describe('вложенные объекты, массивы, enum, необязательные поля', () => {
  const NESTED: JsonSchema = {
    type: 'object',
    properties: {
      a: { type: 'object', properties: { b: { type: 'string' } }, required: ['b'] },
      c: { type: 'integer' },
    },
    required: ['a', 'c'],
  }

  it("токен '\"},' закрывает строку, объект и ставит запятую", () => {
    expect(allowedMask(NESTED, '{"a": {"b": "x', ['"},', '"}}', '"},"'], XG)).toEqual([true, false, false])
    expect(allowedMask(NESTED, '{"a": {"b": "x', ['"},'])).toEqual([true])
    expect(grammarState(NESTED, '{"a": {"b": "x"}, "c": 1}', XG).complete).toBe(true)
  })

  it("токен '\"}}' закрывает два объекта в конце", () => {
    const S: JsonSchema = { type: 'object', properties: { a: NESTED.properties!.a }, required: ['a'] }
    expect(allowedMask(S, '{"a": {"b": "x', ['"}}'], XG)).toEqual([true])
    expect(grammarState(S, '{"a": {"b": "x"}}', XG).complete).toBe(true)
  })

  it('массивы примитивов', () => {
    const S: JsonSchema = { type: 'object', properties: { tags: { type: 'array', items: { type: 'string' } } }, required: ['tags'] }
    expect(grammarState(S, '{"tags": []}').complete).toBe(true)
    expect(grammarState(S, '{"tags": ["a", "b"]}').complete).toBe(true)
    expect(grammarState(S, '{"tags": ["a",]').validPrefix).toBe(false)
    expect(grammarState(S, '{"tags": ["a"').expecting).toBe('запятая или ]')
    expect(allowedMask(S, '{"tags": ["a', ['"]', '"]}', '", "'], XG)).toEqual([true, true, true])
    const ints: JsonSchema = { type: 'array', items: { type: 'integer' } }
    expect(grammarState(ints, '[1, 2, 30]').complete).toBe(true)
    expect(grammarState(ints, '[1, "2"]').validPrefix).toBe(false)
  })

  it('enum строк и чисел', () => {
    const S: JsonSchema = { type: 'object', properties: { mode: { type: 'string', enum: ['fast', 'slow'] } }, required: ['mode'] }
    expect(grammarState(S, '{"mode": "f').expecting).toBe('продолжение: "fast"')
    expect(allowedMask(S, '{"mode": "', ['f', 's', 'x', 'fast"}'])).toEqual([true, true, false, true])
    expect(forcedContinuation(S, '{"mode": "f', XG)).toBe('ast"}')
    const N: JsonSchema = { type: 'integer', enum: [1, 10] }
    expect(grammarState(N, '1').complete).toBe(true)
    expect(grammarState(N, '10').complete).toBe(true)
    expect(grammarState(N, '2').validPrefix).toBe(false)
  })

  it('необязательные поля: порядок фиксирован, пропуск разрешён', () => {
    const S: JsonSchema = {
      type: 'object',
      properties: { a: { type: 'string' }, b: { type: 'integer' }, c: { type: 'boolean' } },
      required: ['a', 'c'],
    }
    expect(grammarState(S, '{"a": "x", "c": true}').complete).toBe(true)
    expect(grammarState(S, '{"a": "x", "b": 1, "c": false}').complete).toBe(true)
    expect(grammarState(S, '{"a": "x", "c": true, "b": 1}').validPrefix).toBe(false)
    expect(grammarState(S, '{"a": "x", "').expecting).toBe('продолжение ключа "b" или "c"')
    expect(grammarState(S, '{"a": "x", "b": 1}').validPrefix).toBe(false)
    const opt: JsonSchema = { type: 'object', properties: { x: { type: 'string' } } }
    expect(grammarState(opt, '{}').complete).toBe(true)
    expect(grammarState(opt, '{').expecting).toBe('ключ "x" или }')
    expect(forcedContinuation(opt, '')).toBe('{')
  })

  it('числа и строки', () => {
    const num: JsonSchema = { type: 'number' }
    expect(grammarState(num, '-3.5e+2').complete).toBe(true)
    expect(grammarState(num, '-0.5').complete).toBe(true)
    expect(grammarState(num, '1.').complete).toBe(false)
    expect(grammarState(num, '1.').validPrefix).toBe(true)
    expect(grammarState(num, '01').validPrefix).toBe(false)
    expect(grammarState({ type: 'integer' }, '-0').validPrefix).toBe(false)
    expect(grammarState({ type: 'integer' }, '42 ').validPrefix).toBe(false)
    const str: JsonSchema = { type: 'string' }
    expect(grammarState(str, '"caf\\u00e9"').complete).toBe(true)
    expect(grammarState(str, '"\\x').validPrefix).toBe(false)
    expect(grammarState(str, '"a\nb"').validPrefix).toBe(false)
    expect(grammarState({ type: 'boolean' }, 'fal').expecting).toBe('продолжение: false')
  })
})

describe('grammarState: что ждём (по-русски)', () => {
  it('ключевые состояния', () => {
    expect(grammarState(NA, '').expecting).toBe('объект {')
    expect(grammarState(NA, '{', XG).expecting).toBe('ключ "name"')
    expect(grammarState(NA, '{"na').expecting).toBe('продолжение ключа "name"')
    expect(grammarState(NA, '{"name"').expecting).toBe('двоеточие :')
    expect(grammarState(NA, '{"name":', XG).expecting).toBe('пробел после двоеточия')
    expect(grammarState(NA, '{"name": ', XG).expecting).toBe('значение поля "name": строка в кавычках')
    expect(grammarState(NA, '{"name":').expecting).toBe('значение поля "name": строка в кавычках (можно с одним пробелом перед ним)')
    expect(grammarState(NA, '{"name": "Ив').expecting).toBe('символы строки "name" или закрывающая кавычка "')
    expect(grammarState(NA, '{"name": "Иван"').expecting).toBe('запятая')
    expect(grammarState(NA, '{"name": "Иван", "age": ').expecting).toBe('значение поля "age": целое число')
    expect(grammarState(NA, '{"name": "Иван", "age": 4').expecting).toBe('цифра или }')
  })

  it('невалидный префикс: позиция ошибки', () => {
    const s = grammarState(NA, '{"name": 5')
    expect(s.validPrefix).toBe(false)
    expect(s.complete).toBe(false)
    expect(s.failedAt).toBe(9)
    expect(grammarState(NA, 'Конечно, вот JSON:').failedAt).toBe(0)
  })
})

describe('маска по токенам Qwen3', () => {
  const tok = loadQwen3ForTests()
  it('«Конечно, вот JSON:» отсекается, {" разрешён, спецтокены запрещены', () => {
    // 'Конечно' у Qwen3 = 'К' 'он' 'ечно': первый же токен вне грамматики
    const cands = [tok.encode('Конечно')[0], ...tok.encode('{"'), tok.token(151667), tok.token(151645)]
    expect(allowedMaskTokens(NA, '', cands)).toEqual([false, true, false, false])
    const end = '{"name": "Анна", "age": 29}'
    expect(allowedMaskTokens(NA, end, [tok.token(151645), tok.token(151667)])).toEqual([true, false])
  })

  it('курсор переиспользуется для многих кандидатов', () => {
    const c = grammarCursor(NA, '{"name": "', XG)
    expect(cursorAccepts(c, 'Анна')).toBe(true)
    expect(cursorAccepts(c, '\n')).toBe(false)
    const dead = grammarCursor(NA, 'x')
    expect(dead.stack).toBeNull()
    expect(cursorAccepts(dead, '{')).toBe(false)
  })

  it('маска по нескольким тысячам токенов словаря считается быстро', () => {
    const texts: string[] = []
    for (let id = 0; id < 20000; id++) texts.push(tok.token(id).text)
    const t0 = performance.now()
    const inStr = allowedMask(NA, '{"name": "', texts, XG)
    const atInt = allowedMask(NA, '{"name": "Иван", "age": ', texts, XG)
    expect(performance.now() - t0).toBeLessThan(1500)
    // внутри строки разрешено почти всё, в начале integer — только '-' и цифры
    expect(inStr.filter(Boolean).length / texts.length).toBeGreaterThan(0.9)
    expect(atInt.filter(Boolean).length).toBe(11)
  })
})
