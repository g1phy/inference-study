/**
 * Пошаговая генерация JSON для главы «Structured output».
 *
 * Что настоящее: токены (словарь Qwen3), маски (structured.ts, побайтно как XGrammar, режим пробелов
 * 'xgrammar' = any_whitespace=False), состояние грамматики, перенормировка (sampling.ts).
 * Что условное: логиты модели. Для каждого шага подбираются кандидаты по контексту
 * (ключ, строка, число, enum…) с правдоподобными, но выдуманными логитами.
 */
import { distribution } from '@/engine/sampling'
import { allowedMaskAfterTokens, grammarState } from '@/engine/structured'
import type { JsonSchema, Token, Tokenizer } from '@/engine/types'
import { RU_KEYS, serialize, type Span } from './schemas'

export const WHITESPACE = 'xgrammar' as const
export const EOS_ID = 151645

export type Mode = 'mask' | 'free'

export interface Cand {
  token: Token
  /** Подсказка, если токен только начало слова: «начало «Конечно»». */
  hint?: string
  logit: number
  /** Разрешён ли маской (в режиме free: разрешила бы грамматика). */
  allowed: boolean
  chosen: boolean
}

export interface Step {
  index: number
  /** Токены до шага. */
  prefix: Token[]
  /** Текст до шага (целые символы). */
  prefixText: string
  chosen: Token
  /** Кандидаты по убыванию вероятности модели. */
  cands: Cand[]
  /** Вероятности модели (softmax по кандидатам, T = 1). */
  probs: number[]
  /** После маски: перенормировка среди разрешённых (в режиме free равны probs). */
  finalProbs: number[]
  /** Что ждёт грамматика перед шагом. */
  expecting: string
  /** Префикс ещё может стать валидным JSON по схеме. */
  valid: boolean
}

/* ---------------------------------------------------------------------------------------------
 * Контекст токена в целевом JSON
 * ------------------------------------------------------------------------------------------- */

type Ctx =
  | { kind: 'start' }
  | { kind: 'end' }
  | { kind: 'keyOpen' }
  | { kind: 'key'; key: string }
  | { kind: 'colon'; key: string }
  | { kind: 'valueOpen'; span: Span }
  | { kind: 'string'; span: Span; first: boolean }
  | { kind: 'enum'; span: Span; first: boolean }
  | { kind: 'number'; span: Span; first: boolean }
  | { kind: 'bool'; span: Span }
  | { kind: 'close' }

function spanAt(spans: Span[], pos: number): Span | undefined {
  return spans.find((s) => pos >= s.start && pos < s.end)
}

function contextOf(text: string, spans: Span[], start: number, end: number, isFirst: boolean, isEos: boolean): Ctx {
  if (isEos) return { kind: 'end' }
  if (isFirst) return { kind: 'start' }
  const a = spanAt(spans, start)
  const b = spanAt(spans, end - 1)
  if (!a || !b) return { kind: 'close' }
  if (b.kind === 'key') return end - 1 === b.start ? { kind: 'keyOpen' } : { kind: 'key', key: b.key ?? '' }
  if (a.kind === 'key') return { kind: 'colon', key: a.key ?? '' }
  if (b.kind === 'string' || b.kind === 'enum') {
    if (end - 1 === b.start) return { kind: 'valueOpen', span: b }
    return { kind: b.kind, span: b, first: a !== b || start <= b.start + 1 }
  }
  if (b.kind === 'number') return { kind: 'number', span: b, first: a !== b || start === b.start }
  if (b.kind === 'bool') return { kind: 'bool', span: b }
  // последний символ токена в разделителе
  if (a.kind !== 'punct') return { kind: 'close' }
  const ch = text[start]
  if (ch === ',' || ch === '}' || ch === ']') return { kind: 'close' }
  const next = spans.find((s) => s.start >= end && s.kind !== 'punct')
  if (next?.kind === 'key') return { kind: 'keyOpen' }
  if (next) return { kind: 'valueOpen', span: next }
  return { kind: 'close' }
}

/* ---------------------------------------------------------------------------------------------
 * Кандидаты по контексту
 * ------------------------------------------------------------------------------------------- */

/** Альтернативные значения, которые модель могла бы выбрать внутри строки (первый токен). */
const ALT_STRINGS: Record<string, string[]> = {
  name: ['Мария', 'Иван'],
  city: ['Москва', 'Самара'],
  model: ['Llama', 'qwen3'],
}

/** «Соблазны»: что модель хотела бы написать вместо разрешённого (иллюстрация). */
const INVENTED_ENUM: Record<string, string> = {
  category: 'двойное списание',
  priority: 'срочно',
  type: 'NVIDIA H100',
}

interface Proposal {
  text: string
  /** Логит относительно целевого токена. */
  logit: number
}

function proposals(ctx: Ctx, target: Token, opener: string): Proposal[] {
  switch (ctx.kind) {
    case 'start':
      return [
        { text: opener, logit: 1.0 },
        { text: '```', logit: 0.3 },
        { text: '{', logit: -0.8 },
        { text: opener === 'Вот' ? 'Конечно' : 'Вот', logit: -1.0 },
      ]
    case 'end':
      return [
        { text: '\n', logit: 0.4 },
        { text: '\n\n', logit: -0.4 },
        { text: '<|endoftext|>', logit: -1.2 },
      ]
    case 'keyOpen':
      return [
        { text: ' ', logit: -1.4 },
        { text: '\n', logit: -0.3 },
        { text: '}', logit: -1.0 },
      ]
    case 'key': {
      const out: Proposal[] = []
      const ru = RU_KEYS[ctx.key]
      if (ru) out.push({ text: ru, logit: 0.2 })
      out.push({ text: ctx.key.charAt(0).toUpperCase() + ctx.key.slice(1), logit: -0.7 })
      out.push({ text: ctx.key.charAt(0), logit: -2.2 })
      return out
    }
    case 'colon':
      return [
        { text: '"', logit: -1.6 },
        { text: '":"', logit: -0.5 },
        { text: '"=', logit: -1.8 },
      ]
    case 'valueOpen': {
      const sch = ctx.span.schema
      if (ctx.span.kind === 'number') return [{ text: ' "', logit: 0.4 }, { text: ' -', logit: -2.5 }, { text: ' null', logit: -1.4 }]
      if (ctx.span.kind === 'bool') return [{ text: ' "', logit: 0.3 }, { text: ' null', logit: -1.2 }]
      return [
        { text: ' null', logit: -0.9 },
        { text: ' ', logit: -2.0 },
        { text: sch?.type === 'object' ? ' "' : ' {', logit: -1.3 },
      ]
    }
    case 'string': {
      const out: Proposal[] = []
      if (ctx.first) for (const alt of ALT_STRINGS[ctx.span.key ?? ''] ?? ['Текст']) out.push({ text: alt, logit: -1.5 - out.length * 0.6 })
      else out.push({ text: '"', logit: -1.7 })
      out.push({ text: '\n', logit: -1.1 })
      out.push({ text: '<|im_end|>', logit: -1.9 })
      return out
    }
    case 'enum': {
      const out: Proposal[] = []
      const key = ctx.span.key ?? ''
      const options = (ctx.span.schema?.enum ?? []).map(String)
      if (ctx.first) {
        const invented = INVENTED_ENUM[key]
        if (invented) out.push({ text: invented, logit: 0.6 })
        for (const o of options.slice(1, 3)) out.push({ text: o, logit: -1.6 - out.length * 0.4 })
        if (options[0]) out.push({ text: options[0].charAt(0).toUpperCase() + options[0].slice(1), logit: -0.9 })
      } else {
        out.push({ text: '"', logit: -2.0 })
        out.push({ text: ' ', logit: -1.2 })
      }
      return out
    }
    case 'number': {
      if (ctx.first) return [{ text: '3', logit: -1.3 }, { text: '1', logit: -1.9 }, { text: '"', logit: 0.3 }]
      return [
        { text: ' лет', logit: -0.4 },
        { text: '.', logit: -1.3 },
        { text: '0', logit: -2.1 },
      ]
    }
    case 'bool':
      return [
        { text: target.text.trim() === 'true' ? 'false' : 'true', logit: -2.0 },
        { text: ' "', logit: 0.5 },
        { text: 'да', logit: -0.6 },
      ]
    case 'close':
      return [
        { text: '"\n', logit: -0.4 },
        { text: ',\n', logit: -0.6 },
        { text: '"}', logit: -1.3 },
        { text: '"', logit: -2.2 },
      ]
  }
}

/** Первый токен строки (у литерала спецтокена сам спецтокен). */
function firstToken(tk: Tokenizer, text: string): Token | null {
  const t = tk.encode(text)
  return t.length > 0 ? t[0] : null
}

/* ---------------------------------------------------------------------------------------------
 * Траектории
 * ------------------------------------------------------------------------------------------- */

function decodePrefix(tk: Tokenizer, prefix: Token[]): string {
  const text = tk.decode(prefix.filter((t) => !t.special).map((t) => t.id))
  return text.replace(/�+$/, '')
}

type RawCand = { token: Token; logit: number; hint?: string }

function finishStep(index: number, prefix: Token[], prefixText: string, chosen: Token, raw: RawCand[], schema: JsonSchema, mode: Mode): Step {
  // уникальные кандидаты, выбранный всегда первый в списке до сортировки
  const seen = new Set<number>()
  const list: RawCand[] = []
  for (const c of raw) {
    if (seen.has(c.token.id)) continue
    seen.add(c.token.id)
    list.push(c)
  }
  const mask = allowedMaskAfterTokens(schema, prefix, list.map((c) => c.token), { whitespace: WHITESPACE, eosIds: [EOS_ID] })
  // разрешённые альтернативы не должны обгонять выбранный токен (иначе сэмплер выбрал бы их)
  const chosenLogit = list.find((c) => c.token.id === chosen.id)!.logit
  const cands: Cand[] = list.map((c, i) => ({
    token: c.token,
    hint: c.hint,
    logit: mode === 'mask' && mask[i] && c.token.id !== chosen.id ? Math.min(c.logit, chosenLogit - 0.35) : c.logit,
    allowed: mask[i],
    chosen: c.token.id === chosen.id,
  }))
  if (mode === 'free') for (const c of cands) if (!c.chosen) c.logit = Math.min(c.logit, chosenLogit - 0.4)
  cands.sort((x, y) => y.logit - x.logit)
  const dist = distribution(
    cands.map((c) => ({ token: c.token, logit: c.logit })),
    { temperature: 1, topK: 0, topP: 1 },
    mode === 'mask' ? cands.map((c) => c.allowed) : undefined,
  )
  const gs = grammarState(schema, prefixText, { whitespace: WHITESPACE })
  return {
    index,
    prefix,
    prefixText,
    chosen,
    cands,
    probs: dist.probs,
    finalProbs: mode === 'mask' ? dist.maskedProbs : dist.probs,
    expecting: gs.expecting,
    valid: gs.validPrefix,
  }
}

/** Генерация под маской: модель пишет целевой JSON, на каждом шаге видно, что маска запретила. */
export function maskedTrajectory(tk: Tokenizer, schema: JsonSchema, instance: unknown, opener = 'Конечно'): { steps: Step[]; target: string } {
  const { text, spans } = serialize(instance, schema)
  const tokens = tk.encode(text, { allowSpecial: false })
  const eos = tk.token(EOS_ID)
  const all = [...tokens, eos]
  const steps: Step[] = []
  let pos = 0
  for (let k = 0; k < all.length; k++) {
    const t = all[k]
    const isEos = k === all.length - 1
    // смещения из токенизатора (UTF-16); токен с половиной символа покрывает символ целиком
    const start = isEos ? pos : (t.start ?? pos)
    const end = isEos ? pos : (t.end ?? pos + t.text.length)
    if (!isEos) pos = end
    const ctx = contextOf(text, spans, start, end, k === 0, isEos)
    const raw: RawCand[] = [{ token: t, logit: 0 }]
    for (const pr of proposals(ctx, t, opener)) {
      const tok = firstToken(tk, pr.text)
      if (tok) raw.push({ token: tok, logit: pr.logit, hint: tok.text !== pr.text && !tok.special ? `начало «${pr.text}»` : undefined })
    }
    const prefix = all.slice(0, k)
    steps.push(finishStep(k, prefix, decodePrefix(tk, prefix), t, raw, schema, 'mask'))
  }
  return { steps, target: text }
}

const FREE_ALTS = ['.', ',', ' и', '\n', ':', ' "']

/** Без маски: модель пишет свой самый вероятный текст («болтовня» из пресета). */
export function freeTrajectory(tk: Tokenizer, schema: JsonSchema, chatter: string, firstStep?: Step): Step[] {
  const tokens = tk.encode(chatter, { allowSpecial: false })
  const all = [...tokens, tk.token(EOS_ID)]
  const steps: Step[] = []
  for (let k = 0; k < all.length; k++) {
    const t = all[k]
    const prefix = all.slice(0, k)
    let raw: RawCand[]
    if (k === 0 && firstStep) {
      // первый шаг тот же, что под маской: модель «хочет» то же самое, просто маски нет
      raw = firstStep.cands.map((c) => ({ token: c.token, logit: c.logit, hint: c.hint }))
      if (!raw.some((c) => c.token.id === t.id)) raw.unshift({ token: t, logit: Math.max(...raw.map((c) => c.logit)) + 0.5 })
    } else {
      raw = [{ token: t, logit: 0 }]
      let n = 0
      for (const s of FREE_ALTS) {
        if (n >= 2) break
        const tok = firstToken(tk, s)
        if (tok && tok.id !== t.id) {
          raw.push({ token: tok, logit: -1.4 - n * 0.7 })
          n++
        }
      }
    }
    steps.push(finishStep(k, prefix, decodePrefix(tk, prefix), t, raw, schema, 'free'))
  }
  return steps
}

/**
 * Компактный словарь для маски по всем 151 669 токенам: байты всех токенов в одном буфере (≈ 1,5 МБ)
 * вместо 151 тысячи объектов Token (≈ 34 МБ, если брать tokenizer.token(id) для каждого id).
 */
export interface VocabStore {
  n: number
  data: Uint8Array
  off: Uint32Array
  special: Uint8Array
}

export function buildVocabStore(tk: Tokenizer): VocabStore {
  const n = tk.vocabSize
  const special = new Uint8Array(n)
  for (const s of tk.specialTokens) if (s.id < n) special[s.id] = 1
  const parts: Uint8Array[] = new Array(n)
  let total = 0
  for (let id = 0; id < n; id++) {
    const b = tk.decodeBytes([id])
    parts[id] = b
    total += b.length
  }
  const data = new Uint8Array(total)
  const off = new Uint32Array(n + 1)
  let pos = 0
  for (let id = 0; id < n; id++) {
    off[id] = pos
    data.set(parts[id], pos)
    pos += parts[id].length
  }
  off[n] = pos
  return { n, data, off, special }
}

/** Маска по всему словарю для шага (настоящая): 1 = токен разрешён. Считается пачками, объекты токенов временные. */
export function fullMask(schema: JsonSchema, prefix: Token[], store: VocabStore): Uint8Array {
  const out = new Uint8Array(store.n)
  const CHUNK = 8192
  for (let s = 0; s < store.n; s += CHUNK) {
    const e = Math.min(store.n, s + CHUNK)
    const toks: Token[] = []
    for (let id = s; id < e; id++) {
      toks.push({ id, raw: '', text: '', special: store.special[id] === 1, bytes: Array.from(store.data.subarray(store.off[id], store.off[id + 1])) })
    }
    const m = allowedMaskAfterTokens(schema, prefix, toks, { whitespace: WHITESPACE, eosIds: [EOS_ID] })
    for (let i = 0; i < m.length; i++) if (m[i]) out[s + i] = 1
  }
  return out
}
