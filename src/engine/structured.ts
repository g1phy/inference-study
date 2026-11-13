/**
 * Structured output: посимвольный автомат JSON со стеком (pushdown automaton), ограниченный схемой.
 * Отвечает на вопрос «может ли этот текст ещё стать JSON по схеме?» и строит маску по кандидатам:
 * токен разрешён, если textSoFar + token.text остаётся валидным префиксом (проверка по символам токена,
 * поэтому токены, захватывающие несколько терминалов сразу, вроде '":' или '"},', работают как в XGrammar).
 *
 * Поддерживаемое подмножество JSON Schema (types.ts JsonSchema): object (свойства строго в порядке
 * объявления, required обязательны, необязательные можно пропустить, лишних нет), string (+ enum),
 * integer, number, boolean, enum, вложенные объекты, массивы.
 *   integer: "0" | "-"?[1-9][0-9]*  (как basic_integer в XGrammar, notes-tokens §3.12: "-0" запрещён);
 *   number:  -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?;
 *   string:  без управляющих символов < 0x20, экранирование \" \\ \/ \b \f \n \r \t \uXXXX.
 *
 * Пробелы (опция whitespace):
 *   'compact' (по умолчанию): никаких пробелов, кроме НЕобязательного одного пробела после ':' и ',';
 *   'xgrammar': ровно один пробел после ':' и ',' — как XGrammar с any_whitespace=False (separators ', ' и ': ');
 *   'minified': пробелов нет совсем.
 * В XGrammar по умолчанию any_whitespace=True: между элементами разрешены любые пробелы и переводы строк,
 * поэтому в структурных местах допустимы сотни токенов-пробелов (notes-tokens §3.12, §5 п. 21);
 * этот режим здесь не моделируется.
 */
import type { GrammarState, JsonSchema, Token } from './types'
import { FAMILY_SPECIAL_TOKENS } from './templates'

/** Политика пробелов автомата. */
export type WhitespaceMode = 'compact' | 'xgrammar' | 'minified'

/** Опции грамматики. */
export interface GrammarOptions {
  whitespace?: WhitespaceMode
}

/** Опции маски по строкам кандидатов. */
export interface MaskOptions extends GrammarOptions {
  /** Тексты, означающие EOS: разрешены, только если документ уже полный. По умолчанию ['<|im_end|>'] (см. MASK_EOS_IDS). */
  eosTexts?: readonly string[]
  /** Тексты, которые запрещены всегда (спецтокены). По умолчанию остальные added tokens Qwen3. */
  forbiddenTexts?: readonly string[]
}

/** Опции маски по настоящим токенам (байтовая проверка, как в XGrammar). */
export interface TokenMaskOptions extends GrammarOptions {
  /** id стоп-токенов: разрешены только у полного документа. По умолчанию MASK_EOS_IDS = [151645]. */
  eosIds?: readonly number[]
  /**
   * Added tokens (кроме стоп-токенов) считать обычным текстом своего литерала. Так делает XGrammar 0.2.9
   * с токенизатором Qwen3: внутри строки JSON он разрешает даже <|im_start|> (замер: 25 added tokens в маске).
   * По умолчанию false: спецтокены запрещены всегда.
   */
  addedTokensAsText?: boolean
  /** Байты недописанного UTF-8 символа в конце префикса (textSoFar тогда — только целые символы). */
  pendingBytes?: readonly number[]
}

/** GrammarState с позицией ошибки. */
export interface GrammarStateEx extends GrammarState {
  /** Индекс символа (UTF-16), на котором префикс перестал быть валидным. */
  failedAt?: number
}

/** EOS-токены Qwen3 (generation_config.json, notes-tokens §1.3). */
export const QWEN3_EOS_TEXTS: readonly string[] = ['<|im_end|>', '<|endoftext|>']
/**
 * Стоп-токены маски по умолчанию: только <|im_end|> (151645). Так считает XGrammar: stop_token_ids из
 * TokenizerInfo.from_huggingface = [tokenizer.eos_token_id] (замер XGrammar 0.2.9; notes-tokens §3.12, шаг 14:
 * «только '<|im_end|>' (151645)»). <|endoftext|> в маске — обычный спецтокен (запрещён).
 */
export const MASK_EOS_IDS: readonly number[] = [151645]
const MASK_EOS_TEXTS: readonly string[] = ['<|im_end|>']
const QWEN3_FORBIDDEN: readonly string[] = FAMILY_SPECIAL_TOKENS.qwen3.filter((t) => !MASK_EOS_TEXTS.includes(t))

/* ---------------------------------------------------------------------------------------------
 * Кадры стека (неизменяемые: шаг автомата возвращает новый стек, старый остаётся валидным)
 * ------------------------------------------------------------------------------------------- */

interface ObjInfo {
  keys: string[]
  keyJson: string[]
  required: boolean[]
}

type ObjPhase = 'open' | 'key' | 'colon' | 'afterColon' | 'value' | 'afterValue' | 'afterComma'
type ArrPhase = 'open' | 'value' | 'afterValue' | 'afterComma'
type NumSub = 'sign' | 'zero' | 'int' | 'dot' | 'frac' | 'e' | 'esign' | 'exp'

type Frame =
  | { k: 'root'; schema: JsonSchema; phase: 'start' | 'value' | 'done' }
  | {
      k: 'obj'
      schema: JsonSchema
      info: ObjInfo
      phase: ObjPhase
      /** Индекс первого свойства, которое ещё может появиться. */
      next: number
      /** Набранная часть ключа (в JSON-виде, с открывающей кавычкой). */
      typed: string
      /** Ключи, совпадающие с набранным. */
      cands: number[]
      /** Выбранный ключ (после закрывающей кавычки). */
      chosen: number
      /** Уже был пробел после ':' или ','. */
      space: boolean
    }
  | { k: 'arr'; schema: JsonSchema; phase: ArrPhase; count: number; space: boolean }
  | { k: 'str'; schema: JsonSchema; esc: number }
  | { k: 'num'; schema: JsonSchema; integer: boolean; sub: NumSub }
  | { k: 'lit'; schema: JsonSchema; options: string[]; typed: string; boolean: boolean }

type Stack = readonly Frame[]

const infoCache = new WeakMap<JsonSchema, ObjInfo>()
function objInfo(schema: JsonSchema): ObjInfo {
  let info = infoCache.get(schema)
  if (!info) {
    const keys = Object.keys(schema.properties ?? {})
    const req = new Set(schema.required ?? [])
    info = { keys, keyJson: keys.map((k) => JSON.stringify(k)), required: keys.map((k) => req.has(k)) }
    infoCache.set(schema, info)
  }
  return info
}

/** Ключи, которые могут идти следующими: от next до первого обязательного включительно. */
function keyCandidates(info: ObjInfo, next: number): number[] {
  const out: number[] = []
  for (let i = next; i < info.keys.length; i++) {
    out.push(i)
    if (info.required[i]) break
  }
  return out
}

/** Можно ли закрыть объект: обязательных свойств впереди не осталось. */
function closable(info: ObjInfo, next: number): boolean {
  for (let i = next; i < info.keys.length; i++) if (info.required[i]) return false
  return true
}

function propSchema(f: Extract<Frame, { k: 'obj' }>): JsonSchema {
  return f.schema.properties![f.info.keys[f.chosen]]
}

const isDigit = (c: string) => c >= '0' && c <= '9'
const isHex = (c: string) => /^[0-9a-fA-F]$/.test(c)
const NUM_DONE: ReadonlySet<NumSub> = new Set(['zero', 'int', 'frac', 'exp'])

/** Начать значение по схеме первым символом ch; null, если ch не может начать такое значение. */
function startValue(schema: JsonSchema, ch: string): Frame | null {
  if (schema.enum && schema.enum.length > 0) {
    const options = schema.enum.map((v) => JSON.stringify(v))
    return options.some((o) => o.startsWith(ch)) ? { k: 'lit', schema, options, typed: ch, boolean: false } : null
  }
  switch (schema.type) {
    case 'object':
      return ch === '{'
        ? { k: 'obj', schema, info: objInfo(schema), phase: 'open', next: 0, typed: '', cands: [], chosen: -1, space: false }
        : null
    case 'array':
      return ch === '[' ? { k: 'arr', schema, phase: 'open', count: 0, space: false } : null
    case 'string':
      return ch === '"' ? { k: 'str', schema, esc: 0 } : null
    case 'integer':
    case 'number': {
      const integer = schema.type === 'integer'
      if (ch === '-') return { k: 'num', schema, integer, sub: 'sign' }
      if (ch === '0') return { k: 'num', schema, integer, sub: 'zero' }
      if (ch >= '1' && ch <= '9') return { k: 'num', schema, integer, sub: 'int' }
      return null
    }
    case 'boolean':
      return ch === 't' || ch === 'f' ? { k: 'lit', schema, options: ['true', 'false'], typed: ch, boolean: true } : null
    default:
      return null
  }
}

type StepResult =
  | { r: 'stay'; f: Frame }
  | { r: 'push'; f: Frame; child: Frame }
  /** Значение закончилось на этом символе (символ поглощён). */
  | { r: 'done' }
  /** Значение закончилось ДО этого символа: символ передаётся родителю. */
  | { r: 'delegate' }
  | null

function stepFrame(f: Frame, ch: string, ws: WhitespaceMode): StepResult {
  switch (f.k) {
    case 'root': {
      if (f.phase !== 'start') return null
      const child = startValue(f.schema, ch)
      return child ? { r: 'push', f: { ...f, phase: 'value' }, child } : null
    }
    case 'obj': {
      const info = f.info
      switch (f.phase) {
        case 'open':
        case 'afterComma': {
          if (f.phase === 'afterComma' && ch === ' ' && ws !== 'minified' && !f.space) return { r: 'stay', f: { ...f, space: true } }
          if (f.phase === 'afterComma' && ws === 'xgrammar' && !f.space) return null
          if (ch === '"') {
            const cands = keyCandidates(info, f.next)
            return cands.length ? { r: 'stay', f: { ...f, phase: 'key', typed: '"', cands } } : null
          }
          if (ch === '}' && f.phase === 'open' && closable(info, f.next)) return { r: 'done' }
          return null
        }
        case 'key': {
          const t = f.typed + ch
          const cands = f.cands.filter((i) => info.keyJson[i].startsWith(t))
          if (cands.length === 0) return null
          const exact = cands.find((i) => info.keyJson[i] === t)
          if (exact !== undefined) return { r: 'stay', f: { ...f, phase: 'colon', typed: t, cands: [exact], chosen: exact } }
          return { r: 'stay', f: { ...f, typed: t, cands } }
        }
        case 'colon':
          return ch === ':' ? { r: 'stay', f: { ...f, phase: 'afterColon', space: false } } : null
        case 'afterColon': {
          if (ch === ' ' && ws !== 'minified' && !f.space) return { r: 'stay', f: { ...f, space: true } }
          if (ws === 'xgrammar' && !f.space) return null
          const child = startValue(propSchema(f), ch)
          return child ? { r: 'push', f: { ...f, phase: 'value' }, child } : null
        }
        case 'afterValue':
          if (ch === ',' && f.next < info.keys.length) return { r: 'stay', f: { ...f, phase: 'afterComma', space: false } }
          if (ch === '}' && closable(info, f.next)) return { r: 'done' }
          return null
        default:
          return null
      }
    }
    case 'arr': {
      const items = f.schema.items
      switch (f.phase) {
        case 'open': {
          if (ch === ']') return { r: 'done' }
          const child = items ? startValue(items, ch) : null
          return child ? { r: 'push', f: { ...f, phase: 'value' }, child } : null
        }
        case 'afterValue':
          if (ch === ',') return { r: 'stay', f: { ...f, phase: 'afterComma', space: false } }
          if (ch === ']') return { r: 'done' }
          return null
        case 'afterComma': {
          if (ch === ' ' && ws !== 'minified' && !f.space) return { r: 'stay', f: { ...f, space: true } }
          if (ws === 'xgrammar' && !f.space) return null
          const child = items ? startValue(items, ch) : null
          return child ? { r: 'push', f: { ...f, phase: 'value' }, child } : null
        }
        default:
          return null
      }
    }
    case 'str': {
      if (f.esc === 0) {
        if (ch === '"') return { r: 'done' }
        if (ch === '\\') return { r: 'stay', f: { ...f, esc: 1 } }
        if (ch.codePointAt(0)! < 0x20) return null
        return { r: 'stay', f }
      }
      if (f.esc === 1) {
        if ('"\\/bfnrt'.includes(ch)) return { r: 'stay', f: { ...f, esc: 0 } }
        if (ch === 'u') return { r: 'stay', f: { ...f, esc: 2 } }
        return null
      }
      // esc 2..5: ждём 4 шестнадцатеричные цифры \uXXXX
      if (!isHex(ch)) return null
      return { r: 'stay', f: { ...f, esc: f.esc === 5 ? 0 : f.esc + 1 } }
    }
    case 'num': {
      const s = f.sub
      const go = (sub: NumSub): StepResult => ({ r: 'stay', f: { ...f, sub } })
      if (s === 'sign') {
        if (ch === '0') return f.integer ? null : go('zero')
        return ch >= '1' && ch <= '9' ? go('int') : null
      }
      if ((s === 'int' || s === 'frac' || s === 'exp') && isDigit(ch)) return go(s)
      if (!f.integer) {
        if ((s === 'zero' || s === 'int') && ch === '.') return go('dot')
        if (s === 'dot') return isDigit(ch) ? go('frac') : null
        if ((s === 'zero' || s === 'int' || s === 'frac') && (ch === 'e' || ch === 'E')) return go('e')
        if (s === 'e') return ch === '+' || ch === '-' ? go('esign') : isDigit(ch) ? go('exp') : null
        if (s === 'esign') return isDigit(ch) ? go('exp') : null
      }
      return NUM_DONE.has(s) ? { r: 'delegate' } : null
    }
    case 'lit': {
      const t = f.typed + ch
      if (f.options.some((o) => o.startsWith(t))) return { r: 'stay', f: { ...f, typed: t } }
      return f.options.includes(f.typed) ? { r: 'delegate' } : null
    }
  }
}

function childDone(parent: Frame): Frame {
  if (parent.k === 'root') return { ...parent, phase: 'done' }
  if (parent.k === 'obj') return { ...parent, phase: 'afterValue', next: parent.chosen + 1, typed: '', cands: [] }
  if (parent.k === 'arr') return { ...parent, phase: 'afterValue', count: parent.count + 1 }
  throw new Error('structured: значение внутри примитива')
}

function popDone(st: Stack): Stack {
  return [...st.slice(0, -2), childDone(st[st.length - 2])]
}

/** Один символ: новый стек или null, если префикс перестаёт быть валидным. */
function feedChar(st: Stack, ch: string, ws: WhitespaceMode): Stack | null {
  let cur = st
  for (;;) {
    const res = stepFrame(cur[cur.length - 1], ch, ws)
    if (!res) return null
    if (res.r === 'stay') return [...cur.slice(0, -1), res.f]
    if (res.r === 'push') return [...cur.slice(0, -1), res.f, res.child]
    if (res.r === 'done') return popDone(cur)
    cur = popDone(cur) // delegate: число/литерал закончились, символ получает родитель
  }
}

function feedText(st: Stack, text: string, ws: WhitespaceMode): { st: Stack | null; failedAt?: number } {
  let cur: Stack = st
  let i = 0
  for (const ch of text) {
    const next = feedChar(cur, ch, ws)
    if (!next) return { st: null, failedAt: i }
    cur = next
    i += ch.length
  }
  return { st: cur }
}

/** Может ли документ закончиться прямо сейчас (EOS). */
function canFinish(st: Stack): boolean {
  let cur = st
  for (;;) {
    const top = cur[cur.length - 1]
    if (top.k === 'root') return top.phase === 'done'
    const completable = (top.k === 'num' && NUM_DONE.has(top.sub)) || (top.k === 'lit' && top.options.includes(top.typed))
    if (!completable) return false
    cur = popDone(cur)
  }
}

/* ---------------------------------------------------------------------------------------------
 * «Что ждём дальше» по-русски
 * ------------------------------------------------------------------------------------------- */

function describeValue(schema: JsonSchema | undefined): string {
  if (!schema) return 'значение'
  if (schema.enum && schema.enum.length > 0) {
    const list = schema.enum.slice(0, 5).map((v) => JSON.stringify(v))
    return 'одно из: ' + list.join(', ') + (schema.enum.length > 5 ? ', …' : '')
  }
  switch (schema.type) {
    case 'object':
      return 'объект {'
    case 'array':
      return 'массив ['
    case 'string':
      return 'строка в кавычках'
    case 'integer':
      return 'целое число'
    case 'number':
      return 'число'
    case 'boolean':
      return 'true или false'
  }
}

const quoteKeys = (info: ObjInfo, idx: number[]) => idx.map((i) => `"${info.keys[i]}"`).join(' или ')

/** Что может идти после законченного значения: по родителю. */
function follow(parent: Frame): string {
  if (parent.k === 'root') return 'конец ответа (EOS)'
  if (parent.k === 'arr') return 'запятая или ]'
  if (parent.k === 'obj') {
    const next = parent.chosen + 1
    const opts = [next < parent.info.keys.length ? 'запятая' : '', closable(parent.info, next) ? '}' : ''].filter(Boolean)
    return opts.join(' или ')
  }
  return ''
}

function fieldName(st: Stack): string | null {
  const parent = st[st.length - 2]
  return parent && parent.k === 'obj' && parent.phase === 'value' ? parent.info.keys[parent.chosen] : null
}

function describe(st: Stack, ws: WhitespaceMode): string {
  const top = st[st.length - 1]
  const parent = st[st.length - 2]
  switch (top.k) {
    case 'root':
      return top.phase === 'done' ? 'конец ответа (EOS)' : describeValue(top.schema)
    case 'obj': {
      const info = top.info
      switch (top.phase) {
        case 'open': {
          const c = keyCandidates(info, top.next)
          if (c.length === 0) return '}'
          return `ключ ${quoteKeys(info, c)}` + (closable(info, top.next) ? ' или }' : '')
        }
        case 'key':
          return `продолжение ключа ${quoteKeys(info, top.cands)}`
        case 'colon':
          return 'двоеточие :'
        case 'afterColon': {
          if (ws === 'xgrammar' && !top.space) return 'пробел после двоеточия'
          const v = `значение поля "${info.keys[top.chosen]}": ${describeValue(propSchema(top))}`
          return ws === 'compact' && !top.space ? v + ' (можно с одним пробелом перед ним)' : v
        }
        case 'afterValue': {
          const opts = [top.next < info.keys.length ? 'запятая' : '', closable(info, top.next) ? '}' : ''].filter(Boolean)
          return opts.join(' или ')
        }
        case 'afterComma': {
          if (ws === 'xgrammar' && !top.space) return 'пробел после запятой'
          const k = `ключ ${quoteKeys(info, keyCandidates(info, top.next))}`
          return ws === 'compact' && !top.space ? k + ' (можно с одним пробелом перед ним)' : k
        }
        default:
          return 'значение'
      }
    }
    case 'arr':
      if (top.phase === 'afterValue') return 'запятая или ]'
      if (top.phase === 'open') return `элемент массива (${describeValue(top.schema.items)}) или ]`
      if (ws === 'xgrammar' && !top.space) return 'пробел после запятой'
      return `элемент массива (${describeValue(top.schema.items)})`
    case 'str': {
      if (top.esc === 1) return 'экранированный символ после \\'
      if (top.esc >= 2) return 'шестнадцатеричная цифра в \\uXXXX'
      const name = fieldName(st)
      return `символы строки${name ? ` "${name}"` : ''} или закрывающая кавычка "`
    }
    case 'num': {
      const after = follow(parent)
      const text: Record<NumSub, string> = {
        sign: 'цифра',
        zero: top.integer ? after : `точка, e или ${after}`,
        int: top.integer ? `цифра или ${after}` : `цифра, точка, e или ${after}`,
        dot: 'цифра после точки',
        frac: `цифра, e или ${after}`,
        e: 'цифра или знак экспоненты',
        esign: 'цифра экспоненты',
        exp: `цифра или ${after}`,
      }
      return text[top.sub]
    }
    case 'lit': {
      const rest = top.options.filter((o) => o.startsWith(top.typed) && o !== top.typed)
      const parts = rest.length ? [`продолжение: ${rest.join(' или ')}`] : []
      if (top.options.includes(top.typed)) parts.push(follow(parent))
      return parts.join(' или ')
    }
  }
}

/* ---------------------------------------------------------------------------------------------
 * Публичный API
 * ------------------------------------------------------------------------------------------- */

function initial(schema: JsonSchema): Stack {
  return [{ k: 'root', schema, phase: 'start' }]
}

/** Разобранный префикс: можно переиспользовать для многих кандидатов (null — префикс уже невалиден). */
export interface GrammarCursor {
  readonly schema: JsonSchema
  readonly whitespace: WhitespaceMode
  readonly stack: Stack | null
  readonly failedAt?: number
}

/** Разбирает textSoFar один раз (для масок по многим кандидатам). */
export function grammarCursor(schema: JsonSchema, textSoFar: string, opts: GrammarOptions = {}): GrammarCursor {
  const ws = opts.whitespace ?? 'compact'
  const { st, failedAt } = feedText(initial(schema), textSoFar, ws)
  return { schema, whitespace: ws, stack: st, failedAt }
}

/** Принимает ли курсор продолжение text (весь текст токена). */
export function cursorAccepts(cursor: GrammarCursor, text: string): boolean {
  if (!cursor.stack) return false
  return feedText(cursor.stack, text, cursor.whitespace).st !== null
}

/** Состояние грамматики: валидный ли это префикс, полный ли документ и что ждём дальше (по-русски). */
export function grammarState(schema: JsonSchema, text: string, opts: GrammarOptions = {}): GrammarStateEx {
  const c = grammarCursor(schema, text, opts)
  if (!c.stack) {
    return { validPrefix: false, complete: false, expecting: 'ничего: текст уже нарушает схему', failedAt: c.failedAt }
  }
  return { validPrefix: true, complete: canFinish(c.stack), expecting: describe(c.stack, c.whitespace) }
}

/**
 * Маска по строкам кандидатов: true, если textSoFar + candidate остаётся валидным префиксом.
 * Тексты EOS разрешены только у полного документа, спецтокены (forbiddenTexts) запрещены всегда.
 */
export function allowedMask(schema: JsonSchema, textSoFar: string, candidates: string[], opts: MaskOptions = {}): boolean[] {
  const c = grammarCursor(schema, textSoFar, opts)
  const eos = new Set(opts.eosTexts ?? MASK_EOS_TEXTS)
  const forbidden = new Set(opts.forbiddenTexts ?? QWEN3_FORBIDDEN)
  const done = c.stack ? canFinish(c.stack) : false
  return candidates.map((t) => {
    if (!c.stack) return false
    if (eos.has(t)) return done
    if (forbidden.has(t)) return false
    return cursorAccepts(c, t)
  })
}

/** Длина UTF-8 последовательности по первому байту (0 — байт не может начинать символ). */
function utf8SeqLen(b: number): number {
  if (b < 0x80) return 1
  if (b >= 0xc2 && b <= 0xdf) return 2
  if (b >= 0xe0 && b <= 0xef) return 3
  if (b >= 0xf0 && b <= 0xf4) return 4
  return 0
}

/** Допустим ли байт продолжения на позиции k (RFC 3629: без overlong, суррогатов и > U+10FFFF). */
function utf8ContOk(lead: number, k: number, b: number): boolean {
  if (b < 0x80 || b > 0xbf) return false
  if (k !== 1) return true
  if (lead === 0xe0) return b >= 0xa0
  if (lead === 0xed) return b <= 0x9f
  if (lead === 0xf0) return b >= 0x90
  if (lead === 0xf4) return b <= 0x8f
  return true
}

/**
 * Байты → целые символы + недописанный хвост. null, если внутри есть невалидная последовательность
 * (байт продолжения без начала, overlong и т. п.): такой токен не может быть частью JSON нигде.
 */
function splitUtf8(bytes: readonly number[]): { text: string; tail: number[] } | null {
  let text = ''
  let i = 0
  while (i < bytes.length) {
    const b = bytes[i]
    const n = utf8SeqLen(b)
    if (n === 0) return null
    const avail = Math.min(n, bytes.length - i)
    for (let k = 1; k < avail; k++) if (!utf8ContOk(b, k, bytes[i + k])) return null
    if (avail < n) return { text, tail: bytes.slice(i) }
    let cp = n === 1 ? b : b & (0xff >> (n + 1))
    for (let k = 1; k < n; k++) cp = (cp << 6) | (bytes[i + k] & 0x3f)
    text += String.fromCodePoint(cp)
    i += n
  }
  return { text, tail: [] }
}

const encoder = new TextEncoder()

/** Может ли следующий символ начинаться байтами tail (неполный UTF-8) в этом состоянии автомата. */
function acceptsPartialChar(st: Stack, tail: readonly number[], ws: WhitespaceMode): boolean {
  const top = st[st.length - 1]
  // Внутри строки разрешён любой символ ≥ U+0020, а валидный префикс UTF-8 всегда достраивается до такого.
  if (top.k === 'str') return top.esc === 0
  let options: string[] = []
  if (top.k === 'obj' && top.phase === 'key') options = top.cands.map((i) => top.info.keyJson[i].slice(top.typed.length))
  else if (top.k === 'lit') options = top.options.filter((o) => o.startsWith(top.typed)).map((o) => o.slice(top.typed.length))
  // Ключи и enum могут содержать не-ASCII: годится вариант, чей следующий символ начинается этими байтами.
  return options.some((rest) => {
    const ch = String.fromCodePoint(rest.codePointAt(0) ?? 0x20)
    const b = encoder.encode(ch)
    return rest.length > 0 && tail.every((x, k) => b[k] === x) && b.length > tail.length && feedChar(st, ch, ws) !== null
  })
}

/** Принимает ли курсор продолжение байтами (pending — недописанный символ из префикса). */
function cursorAcceptsBytes(cursor: GrammarCursor, pending: readonly number[], bytes: readonly number[]): boolean {
  if (!cursor.stack) return false
  const ws = cursor.whitespace
  const all = pending.length > 0 ? [...pending, ...bytes] : bytes
  let st: Stack = cursor.stack
  // Быстрый путь (маска по 150 тыс. токенов внутри строки): без кавычки, обратного слэша и управляющих
  // байтов токен остаётся внутри строки — достаточно, чтобы байты были валидным UTF-8 (хвост может быть неполным).
  const top = st[st.length - 1]
  const plainInString = top.k === 'str' && top.esc === 0 && all.every((b) => b >= 0x20 && b !== 0x22 && b !== 0x5c)
  let i = 0
  while (i < all.length) {
    const b = all[i]
    const n = utf8SeqLen(b)
    if (n === 0) return false
    const avail = Math.min(n, all.length - i)
    for (let k = 1; k < avail; k++) if (!utf8ContOk(b, k, all[i + k])) return false
    if (avail < n) return plainInString || acceptsPartialChar(st, all.slice(i), ws)
    if (!plainInString) {
      let cp = n === 1 ? b : b & (0xff >> (n + 1))
      for (let k = 1; k < n; k++) cp = (cp << 6) | (all[i + k] & 0x3f)
      // символ сразу в автомат: на структурных шагах почти все токены отваливаются на первом символе
      const next = feedChar(st, String.fromCodePoint(cp), ws)
      if (!next) return false
      st = next
    }
    i += n
  }
  return true
}

/**
 * Маска по токенам, побайтно как XGrammar: токен разрешён, если его байты продолжают валидный UTF-8
 * и префикс JSON по схеме. Токен с неполным UTF-8 разрешён там, где символ можно дописать (внутри строки,
 * в не-ASCII ключе или enum), а с невалидными байтами (например, байт продолжения после целого символа) —
 * нигде. Стоп-токены (eosIds, по умолчанию <|im_end|>) — только у полного документа; остальные added tokens
 * запрещены (или считаются своим литералом при addedTokensAsText, как XGrammar 0.2.9).
 * Сверено с XGrammar 0.2.9 (any_whitespace=False ↔ whitespace: 'xgrammar') на всём словаре Qwen3:
 * маски совпадают на каждом шаге (structured.xgrammar.test.ts).
 */
export function allowedMaskTokens(schema: JsonSchema, textSoFar: string, tokens: Token[], opts: TokenMaskOptions = {}): boolean[] {
  const c = grammarCursor(schema, textSoFar, opts)
  const eos = new Set(opts.eosIds ?? MASK_EOS_IDS)
  const pending = opts.pendingBytes ?? []
  const done = c.stack && pending.length === 0 ? canFinish(c.stack) : false
  return tokens.map((t) => {
    if (!c.stack) return false
    if (eos.has(t.id)) return done
    if (t.special && !opts.addedTokensAsText) return false
    return cursorAcceptsBytes(c, pending, t.bytes)
  })
}

/**
 * То же по префиксу из токенов: целые символы префикса идут в автомат, недописанный UTF-8 хвост
 * (например, половина эмодзи внутри строки) учитывается побайтно.
 */
export function allowedMaskAfterTokens(
  schema: JsonSchema,
  prefix: Token[],
  tokens: Token[],
  opts: Omit<TokenMaskOptions, 'pendingBytes'> = {},
): boolean[] {
  const bytes: number[] = []
  for (const t of prefix) {
    if (t.special && !opts.addedTokensAsText) continue
    for (const b of t.bytes) bytes.push(b)
  }
  const split = splitUtf8(bytes)
  if (!split) return tokens.map(() => false)
  return allowedMaskTokens(schema, split.text, tokens, { ...opts, pendingBytes: split.tail })
}

function commonNext(options: string[], pos: number): string | null {
  let ch: string | null = null
  for (const o of options) {
    if (o.length <= pos) return null
    const c = o[pos]
    if (ch === null) ch = c
    else if (c !== ch) return null
  }
  return ch
}

function forcedStart(schema: JsonSchema | undefined): string | null {
  if (!schema) return null
  if (schema.enum && schema.enum.length > 0) return commonNext(schema.enum.map((v) => JSON.stringify(v)), 0)
  if (schema.type === 'object') return '{'
  if (schema.type === 'array') return '['
  if (schema.type === 'string') return '"'
  return null
}

/**
 * Литерал, который уже точно закончился (совпал с вариантом, и длиннее вариантов нет), закрываем:
 * тогда следующий принудительный символ определяет родитель (после enum "fast" — '}' или ',').
 */
function settle(st: Stack): Stack {
  let cur = st
  for (;;) {
    const top = cur[cur.length - 1]
    if (top.k !== 'lit' || !top.options.includes(top.typed)) return cur
    if (top.options.some((o) => o !== top.typed && o.startsWith(top.typed))) return cur
    cur = popDone(cur)
  }
}

function forcedNextChar(st: Stack, ws: WhitespaceMode): string | null {
  const top = st[st.length - 1]
  switch (top.k) {
    case 'root':
      return top.phase === 'start' ? forcedStart(top.schema) : null
    case 'obj': {
      const info = top.info
      switch (top.phase) {
        case 'open': {
          const c = keyCandidates(info, top.next)
          const close = closable(info, top.next)
          if (c.length === 0) return close ? '}' : null
          return close ? null : '"'
        }
        case 'key':
          return commonNext(
            top.cands.map((i) => info.keyJson[i]),
            top.typed.length,
          )
        case 'colon':
          return ':'
        case 'afterColon':
          if (!top.space && ws === 'xgrammar') return ' '
          if (!top.space && ws === 'compact') return null
          return forcedStart(propSchema(top))
        case 'afterValue': {
          const comma = top.next < info.keys.length
          const close = closable(info, top.next)
          return comma && !close ? ',' : close && !comma ? '}' : null
        }
        case 'afterComma':
          if (!top.space && ws === 'xgrammar') return ' '
          if (!top.space && ws === 'compact') return null
          return keyCandidates(info, top.next).length ? '"' : null
        default:
          return null
      }
    }
    case 'arr':
      // массив без items допускает только [] — закрывающая скобка однозначна
      if (top.phase === 'open') return top.schema.items ? null : ']'
      if (top.phase !== 'afterComma') return null
      if (!top.space && ws === 'xgrammar') return ' '
      if (!top.space && ws === 'compact') return null
      return forcedStart(top.schema.items)
    case 'lit': {
      if (top.options.includes(top.typed)) return null
      return commonNext(
        top.options.filter((o) => o.startsWith(top.typed)),
        top.typed.length,
      )
    }
    default:
      return null
  }
}

/**
 * Jump-forward: строка, которая однозначно следует за textSoFar (пока у грамматики ровно один вариант).
 * В режиме 'xgrammar' для схемы {name, age} в начале это '{"name": "', после имени — ', "age": '
 * (как find_jump_forward_string в XGrammar, notes-tokens §3.13); в 'compact' необязательный пробел
 * обрывает принудительный кусок: '{"name":'.
 */
export function forcedContinuation(schema: JsonSchema, textSoFar: string, opts: GrammarOptions = {}): string {
  const c = grammarCursor(schema, textSoFar, opts)
  if (!c.stack) return ''
  let st: Stack = c.stack
  let out = ''
  for (let guard = 0; guard < 10000; guard++) {
    st = settle(st)
    const ch = forcedNextChar(st, c.whitespace)
    if (ch === null) break
    const next = feedChar(st, ch, c.whitespace)
    if (!next) break
    st = next
    out += ch
  }
  return out
}
