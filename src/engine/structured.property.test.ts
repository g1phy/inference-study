/**
 * Property-based проверка structured.ts на случайных схемах и экземплярах (детерминированный ГПСЧ):
 *  1) любой валидный экземпляр, сериализованный по политике пробелов, — валидный префикс на КАЖДОМ разрезе,
 *     полный документ complete, собственные префиксы объекта — нет;
 *  2) испорченные экземпляры (лишний/пропущенный ключ, порядок, тип, enum, ведущий ноль, управляющий символ,
 *     битое экранирование, висячая запятая, лишний пробел…) — невалидны;
 *  3) allowedMask согласован с grammarState по любым кандидатам, EOS только у полного документа;
 *  4) forcedContinuation: каждый символ — единственный допустимый, и строка максимальна.
 */
import { describe, expect, it } from 'vitest'
import { mulberry32 } from './rng'
import { allowedMask, forcedContinuation, grammarState, type WhitespaceMode } from './structured'
import type { JsonSchema } from './types'

type Rng = () => number
const KEY_POOL = ['name', 'age', 'a', 'ab', 'abc', 'имя', 'город', 'a b', 'q"uote', 'back\\slash', 'emoji🙂', 'x', 'y', 'nested', 'list', '']
const STR_POOL = ['', 'Анна', 'hello world', 'кавычка " внутри', 'слэш \\ и /', 'перевод\nстроки', 'таб\t', '\u0001', '🙂🫠', 'Ѣ', ' ', '{"json": [1]}', 'true', '42']
const ENUM_STR = [['c', 'f'], ['Москва', 'Казань', 'Мос'], ['a', 'ab'], ['x"y', 'x\\y'], ['only']]
const ENUM_NUM = [[1, 12, 123], [0, -1], [7]]
const pick = <T,>(rng: Rng, a: readonly T[]): T => a[Math.floor(rng() * a.length)]

function genSchema(rng: Rng, depth: number): JsonSchema {
  const r = rng()
  if (depth > 0 && r < 0.22) return genObject(rng, depth - 1)
  if (depth > 0 && r < 0.36) return { type: 'array', items: genSchema(rng, depth - 1) }
  const p = rng()
  if (p < 0.12) return { type: 'string', enum: [...pick(rng, ENUM_STR)] }
  if (p < 0.2) return { type: 'integer', enum: [...pick(rng, ENUM_NUM)] }
  if (p < 0.45) return { type: 'string' }
  if (p < 0.65) return { type: 'integer' }
  if (p < 0.85) return { type: 'number' }
  return { type: 'boolean' }
}

function genObject(rng: Rng, depth: number): JsonSchema {
  const n = 1 + Math.floor(rng() * 4)
  const keys: string[] = []
  while (keys.length < n) {
    const k = pick(rng, KEY_POOL)
    if (!keys.includes(k)) keys.push(k)
  }
  const properties: Record<string, JsonSchema> = {}
  for (const k of keys) properties[k] = genSchema(rng, depth)
  const required = keys.filter(() => rng() < 0.55)
  return { type: 'object', properties, required, additionalProperties: false }
}

/** Значение: уже сериализованный кусок JSON (строка) или структура. */
type Val =
  | { t: 'raw'; s: string; kind: 'string' | 'enum' | 'integer' | 'number' | 'boolean' }
  | { t: 'obj'; entries: Array<[string, Val]>; schema: JsonSchema }
  | { t: 'arr'; items: Val[] }

/** Строка JSON со случайными допустимыми формами экранирования (\uXXXX, \/). */
function jsonString(rng: Rng, s: string): string {
  let out = '"'
  for (const ch of s) {
    const cp = ch.codePointAt(0)!
    const r = rng()
    if (ch === '"' || ch === '\\') out += '\\' + ch
    else if (cp < 0x20) out += r < 0.5 ? JSON.stringify(ch).slice(1, -1) : '\\u' + cp.toString(16).padStart(4, '0').toUpperCase()
    else if (ch === '/' && r < 0.5) out += '\\/'
    else if (cp < 0x10000 && r < 0.1) out += '\\u' + cp.toString(16).padStart(4, '0')
    else out += ch
  }
  return out + '"'
}

function genValue(rng: Rng, s: JsonSchema): Val {
  if (s.enum && s.enum.length) return { t: 'raw', s: JSON.stringify(pick(rng, s.enum)), kind: 'enum' }
  switch (s.type) {
    case 'string':
      return { t: 'raw', s: jsonString(rng, pick(rng, STR_POOL) + (rng() < 0.3 ? pick(rng, STR_POOL) : '')), kind: 'string' }
    case 'integer':
      return { t: 'raw', s: pick(rng, ['0', '1', '-1', '7', '-42', '1234567', '-9007199254740991', '10']), kind: 'integer' }
    case 'number':
      return { t: 'raw', s: pick(rng, ['0', '-0', '3.14', '-0.5', '0.0', '1e10', '1E-5', '-2.5e+3', '12', '100.25E2']), kind: 'number' }
    case 'boolean':
      return { t: 'raw', s: rng() < 0.5 ? 'true' : 'false', kind: 'boolean' }
    case 'array': {
      const n = Math.floor(rng() * 4)
      return { t: 'arr', items: Array.from({ length: n }, () => genValue(rng, s.items!)) }
    }
    case 'object': {
      const req = new Set(s.required ?? [])
      const entries: Array<[string, Val]> = []
      for (const [k, ps] of Object.entries(s.properties ?? {})) if (req.has(k) || rng() < 0.6) entries.push([k, genValue(rng, ps)])
      return { t: 'obj', entries, schema: s }
    }
  }
}

type Corruption =
  | 'none'
  | 'unknown-key'
  | 'missing-required'
  | 'swap-order'
  | 'ctrl-in-string'
  | 'bad-escape'
  | 'leading-zero'
  | 'int-fraction'
  | 'bad-enum'
  | 'type-swap'
  | 'trailing-comma'
  | 'bad-space'
  | 'garbage-after-string'
  | 'bool-typo'
  | 'dangling-dot'
const CORRUPTIONS: Corruption[] = [
  'unknown-key',
  'missing-required',
  'swap-order',
  'ctrl-in-string',
  'bad-escape',
  'leading-zero',
  'int-fraction',
  'bad-enum',
  'type-swap',
  'trailing-comma',
  'bad-space',
  'garbage-after-string',
  'bool-typo',
  'dangling-dot',
]

/** Сериализация с политикой пробелов и (необязательно) одной порчей в первом подходящем месте. */
function serialize(rng: Rng, v: Val, ws: WhitespaceMode, corruption: Corruption): { text: string; applied: boolean } {
  let applied = corruption === 'none'
  const sep = (ch: ',' | ':') => {
    if (!applied && corruption === 'bad-space' && ch === ':') {
      applied = true
      return ws === 'compact' ? ':  ' : ws === 'xgrammar' ? ':' : ': '
    }
    if (ws === 'minified') return ch
    if (ws === 'xgrammar') return ch + ' '
    return rng() < 0.5 ? ch : ch + ' '
  }
  const go = (x: Val): string => {
    if (x.t === 'raw') {
      if (!applied) {
        if (corruption === 'ctrl-in-string' && x.kind === 'string') {
          applied = true
          return x.s.slice(0, -1) + '\u0001"'
        }
        if (corruption === 'bad-escape' && x.kind === 'string') {
          applied = true
          return x.s.slice(0, -1) + '\\x"'
        }
        if (corruption === 'garbage-after-string' && x.kind === 'string') {
          applied = true
          return x.s + 'q'
        }
        if (corruption === 'leading-zero' && x.kind === 'integer') {
          applied = true
          return x.s.startsWith('-') ? '-0' + x.s.slice(1) : '0' + x.s
        }
        if (corruption === 'int-fraction' && x.kind === 'integer') {
          applied = true
          return x.s + '.0'
        }
        if (corruption === 'bad-enum' && x.kind === 'enum') {
          applied = true
          return x.s.startsWith('"') ? '"zz~"' : '99999'
        }
        if (corruption === 'type-swap' && x.kind !== 'enum') {
          applied = true
          return x.kind === 'string' ? '123' : '"' + x.s + '"'
        }
        if (corruption === 'bool-typo' && x.kind === 'boolean') {
          applied = true
          return x.s.slice(0, 3) + 'x'
        }
        if (corruption === 'dangling-dot' && x.kind === 'number') {
          applied = true
          return '1.'
        }
      }
      return x.s
    }
    if (x.t === 'arr') return '[' + x.items.map(go).join(sep(',')) + ']'
    let entries = x.entries
    let prefix = ''
    if (!applied && corruption === 'unknown-key') {
      applied = true
      prefix = '"zz~"' + sep(':') + '1' + sep(',')
    }
    if (!applied && corruption === 'missing-required') {
      const req = new Set(x.schema.required ?? [])
      const i = entries.findIndex(([k]) => req.has(k))
      if (i >= 0) {
        applied = true
        entries = entries.filter((_, j) => j !== i)
      }
    }
    if (!applied && corruption === 'swap-order' && entries.length >= 2) {
      applied = true
      entries = [entries[1], entries[0], ...entries.slice(2)]
    }
    const body = entries.map(([k, val]) => JSON.stringify(k) + sep(':') + go(val)).join(sep(','))
    let tail = ''
    if (!applied && corruption === 'trailing-comma' && entries.length > 0) {
      applied = true
      tail = ','
    }
    return '{' + prefix + body + tail + '}'
  }
  return { text: go(v), applied }
}

const MODES: WhitespaceMode[] = ['compact', 'xgrammar', 'minified']
const cps = (s: string) => [...s]
const prefixes = (s: string) => {
  const a = cps(s)
  return a.map((_, i) => a.slice(0, i).join('')).concat([s])
}

function corpus(seed: number, n: number) {
  const rng = mulberry32(seed)
  const out: Array<{ schema: JsonSchema; value: Val; rng: Rng }> = []
  for (let i = 0; i < n; i++) {
    const schema = genObject(rng, 2)
    out.push({ schema, value: genValue(rng, schema), rng })
  }
  return out
}

describe('1) валидные экземпляры — валидный префикс на каждом разрезе', () => {
  for (const ws of MODES) {
    it(`whitespace=${ws}: 150 случайных схем × экземпляр`, () => {
      let checked = 0
      for (const { schema, value, rng } of corpus(1000 + MODES.indexOf(ws), 150)) {
        const { text } = serialize(rng, value, ws, 'none')
        expect(() => JSON.parse(text)).not.toThrow()
        const ps = prefixes(text)
        ps.forEach((p, i) => {
          const g = grammarState(schema, p, { whitespace: ws })
          if (!g.validPrefix) throw new Error(`невалиден префикс ${JSON.stringify(p)} из ${text} (${ws}) ${JSON.stringify(schema)}`)
          expect(g.complete, p).toBe(i === ps.length - 1)
          checked++
        })
      }
      expect(checked).toBeGreaterThan(3000)
    })
  }

  it('compact принимает и формы xgrammar, и minified; xgrammar и minified отвергают чужую форму', () => {
    for (const { schema, value, rng } of corpus(77, 150)) {
      const x = serialize(rng, value, 'xgrammar', 'none').text
      const m = serialize(rng, value, 'minified', 'none').text
      expect(grammarState(schema, x).complete).toBe(true)
      expect(grammarState(schema, m).complete).toBe(true)
      const hasSep = x !== m
      expect(grammarState(schema, m, { whitespace: 'xgrammar' }).validPrefix).toBe(!hasSep)
      expect(grammarState(schema, x, { whitespace: 'minified' }).validPrefix).toBe(!hasSep)
    }
  })
})

describe('2) испорченные экземпляры невалидны', () => {
  for (const corruption of CORRUPTIONS) {
    it(corruption, () => {
      let applied = 0
      for (const ws of MODES) {
        for (const { schema, value, rng } of corpus(5000 + CORRUPTIONS.indexOf(corruption) * 10 + MODES.indexOf(ws), 120)) {
          const r = serialize(rng, value, ws, corruption)
          if (!r.applied) continue
          applied++
          const g = grammarState(schema, r.text, { whitespace: ws })
          if (g.validPrefix) throw new Error(`порча ${corruption} не замечена: ${r.text} (${ws}) ${JSON.stringify(schema)}`)
        }
      }
      expect(applied).toBeGreaterThan(20)
    })
  }
})

describe('3) allowedMask = grammarState(prefix + кандидат); EOS только у полного документа', () => {
  it('случайные разрезы, кандидаты из настоящего продолжения и мусора', () => {
    const extra = ['"', '",', '":', ' "', '}', ']', ',', ', "', '0', '-', '.5', 'e', 'true', ' ', '  ', '\n', '\\', '\\u00', 'Я', '🙂', '{"', '<|im_end|>', '<think>']
    for (const ws of MODES) {
      for (const { schema, value, rng } of corpus(9000 + MODES.indexOf(ws), 80)) {
        const { text } = serialize(rng, value, ws, 'none')
        const a = cps(text)
        for (let n = 0; n < 6; n++) {
          const cut = Math.floor(rng() * (a.length + 1))
          const prefix = a.slice(0, cut).join('')
          const cont = [1, 2, 3, 5].map((l) => a.slice(cut, cut + l).join('')).filter(Boolean)
          const cands = [...cont, ...extra]
          const mask = allowedMask(schema, prefix, cands, { whitespace: ws })
          cands.forEach((c, i) => {
            const expected =
              c === '<|im_end|>' ? cut === a.length : c === '<think>' ? false : grammarState(schema, prefix + c, { whitespace: ws }).validPrefix
            if (mask[i] !== expected) throw new Error(`маска ${JSON.stringify(prefix)} + ${JSON.stringify(c)}: ${mask[i]} ≠ ${expected} (${ws})`)
          })
          for (const c of cont) expect(mask[cands.indexOf(c)]).toBe(true)
        }
      }
    }
  })
})

describe('4) forcedContinuation: каждый символ единственный допустимый, строка максимальна', () => {
  function allowedNext(schema: JsonSchema, p: string, ws: WhitespaceMode, alphabet: string[]): string[] {
    return alphabet.filter((ch) => grammarState(schema, p + ch, { whitespace: ws }).validPrefix)
  }
  it('на случайных валидных префиксах во всех режимах', () => {
    const base = [...'{}[],:" \\/-+.eE0123456789tfnrubaxyqzЯ🙂\n\t\u0001'].concat([...'abcdefghijklmnopqrstuvwxyzABCDEF'])
    let forcedChars = 0
    for (const ws of MODES) {
      for (const { schema, value, rng } of corpus(12000 + MODES.indexOf(ws), 60)) {
        const alphabet = [...new Set([...base, ...cps(JSON.stringify(schema))])]
        const { text } = serialize(rng, value, ws, 'none')
        const a = cps(text)
        for (let n = 0; n < 5; n++) {
          const p = a.slice(0, Math.floor(rng() * (a.length + 1))).join('')
          const f = forcedContinuation(schema, p, { whitespace: ws })
          expect(grammarState(schema, p + f, { whitespace: ws }).validPrefix).toBe(true)
          const fc = cps(f)
          for (let k = 0; k < fc.length; k++) {
            const next = allowedNext(schema, p + fc.slice(0, k).join(''), ws, alphabet)
            if (next.length !== 1 || next[0] !== fc[k]) {
              throw new Error(`не принудительный символ ${JSON.stringify(fc[k])} после ${JSON.stringify(p + fc.slice(0, k).join(''))}: ${JSON.stringify(next)} (${ws})`)
            }
            forcedChars++
          }
          const after = allowedNext(schema, p + f, ws, alphabet)
          if (after.length === 1) throw new Error(`не максимально: после ${JSON.stringify(p + f)} ещё однозначно ${JSON.stringify(after)} (${ws})`)
        }
      }
    }
    expect(forcedChars).toBeGreaterThan(200)
  })

  it("XGrammar jump-forward (notes-tokens §3.13): '{\"name\": \"' и ', \"age\": '", () => {
    const NA: JsonSchema = { type: 'object', properties: { name: { type: 'string' }, age: { type: 'integer' } }, required: ['name', 'age'], additionalProperties: false }
    expect(forcedContinuation(NA, '', { whitespace: 'xgrammar' })).toBe('{"name": "')
    expect(forcedContinuation(NA, '{"name": "Иван"', { whitespace: 'xgrammar' })).toBe(', "age": ')
  })

  it('массив без items допускает только [] — и это принудительно', () => {
    const S: JsonSchema = { type: 'object', properties: { l: { type: 'array' } }, required: ['l'] }
    expect(grammarState(S, '{"l":[1').validPrefix).toBe(false)
    expect(forcedContinuation(S, '{"l":[', { whitespace: 'minified' })).toBe(']}')
  })
})
