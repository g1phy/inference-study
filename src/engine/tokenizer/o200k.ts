/**
 * Сравнительный токенизатор o200k (gpt-oss / GPT-4o) через npm `gpt-tokenizer` 4.1.2 в интерфейсе Tokenizer.
 * Нужен главе «Токенизация», чтобы показать: у разных семейств разные словари и разные id.
 *
 * Модули грузятся динамически (отдельный чанк ~1 МБ gzip, источник размера: notes-tokens §2.8):
 *   gpt-tokenizer/encoding/o200k_harmony | gpt-tokenizer/encoding/o200k_base — encode;
 *   gpt-tokenizer/bpeRanks/o200k_base — массив id → строка (валидный UTF-8) или number[] байтов.
 *     Путь недокументирован (notes §2.8, §3.15), поэтому все обращения к нему собраны здесь.
 *   gpt-tokenizer/encodingParams/constants — регулярка пре-токенизации o200k.
 */
import type { BpeMergeStep, BpeTrace, Token, Tokenizer } from '../types'
import { bytesToRaw, byteToUtf16Spans, utf8Bytes, utf8DecodeLossy, utf8DecodeSoft } from './bytes'

/** Какая кодировка: o200k_harmony (gpt-oss, по умолчанию) или o200k_base (GPT-4o). */
export type O200kVariant = 'o200k_harmony' | 'o200k_base'

interface EncodeFn {
  (text: string, opts?: { allowedSpecial?: Set<string> | 'all'; disallowedSpecial?: Set<string> | 'all' }): number[]
}

/** Число BPE-рангов o200k: 199 998 (notes-tokens §2.1). */
const O200K_RANKS = 199998

/**
 * Спецтокены. Harmony: id из спецификации (notes-tokens §2.1 «Harmony id»; совпадают с gpt-tokenizer).
 * o200k_base: так их нумерует gpt-tokenizer (у tiktoken в o200k_base только <|endoftext|> 199999 и
 * <|endofprompt|> 200018; остальное — расширение библиотеки).
 */
const SPECIALS: Record<O200kVariant, Array<[string, number]>> = {
  o200k_harmony: [
    ['<|startoftext|>', 199998],
    ['<|endoftext|>', 199999],
    ['<|return|>', 200002],
    ['<|constrain|>', 200003],
    ['<|channel|>', 200005],
    ['<|start|>', 200006],
    ['<|end|>', 200007],
    ['<|message|>', 200008],
    ['<|call|>', 200012],
    ['<|endofprompt|>', 200018],
  ],
  o200k_base: [
    ['<|endoftext|>', 199999],
    ['<|fim_prefix|>', 200000],
    ['<|fim_middle|>', 200001],
    ['<|fim_suffix|>', 200002],
    ['<|im_start|>', 200003],
    ['<|im_end|>', 200004],
    ['<|im_sep|>', 200005],
    ['<|endofprompt|>', 200006],
  ],
}

/** Ширина логитов gpt-oss = n_vocab o200k_harmony с reserved: 201 088 (notes-tokens §1.9). */
export const O200K_HARMONY_LOGITS_WIDTH = 201088

/**
 * Reserved-токены Harmony: `<|reserved_N|>` для N = 200000, 200001, 200004, 200009…200011 и 200013…201087
 * (gpt-tokenizer encodingParams/o200k_harmony.js: STATIC_SPECIAL_TOKEN_ENTRIES + RESERVED_TOKEN_RANGE 200013…201088).
 * 200018 занят <|endofprompt|>, но литерал <|reserved_200018|> библиотека тоже кодирует в 200018.
 */
function isHarmonyReserved(id: number): boolean {
  return id === 200000 || id === 200001 || id === 200004 || (id >= 200009 && id <= 200011) || (id >= 200013 && id < O200K_HARMONY_LOGITS_WIDTH)
}

const RESERVED_RE = /^<\|reserved_(\d+)\|>$/

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Токенизатор o200k поверх gpt-tokenizer (собирается через loadO200kTokenizer). */
export class O200kTokenizer implements Tokenizer {
  readonly name: string
  /** 199 998 BPE + спецтокены: n_vocab 200 019 (notes-tokens §1.9, §2.1). */
  readonly vocabSize = 200019
  readonly specialTokens: ReadonlyArray<{ id: number; text: string; hfSpecial: boolean }>
  readonly variant: O200kVariant

  private readonly encodeFn: EncodeFn
  private readonly ranks: ReadonlyArray<string | number[]>
  private readonly splitRe: RegExp
  private readonly specialById = new Map<number, string>()
  private readonly specialByText = new Map<string, number>()
  private readonly specialRe: RegExp
  private readonly cache = new Map<number, Token>()
  private rawIndex: Map<string, number> | null = null
  private byteRank: Map<string, number> | null = null

  constructor(variant: O200kVariant, encodeFn: EncodeFn, ranks: ReadonlyArray<string | number[]>, splitRe: RegExp) {
    this.variant = variant
    this.name = variant === 'o200k_harmony' ? 'o200k_harmony (gpt-oss)' : 'o200k_base (GPT-4o)'
    this.encodeFn = encodeFn
    this.ranks = ranks
    this.splitRe = new RegExp(splitRe.source, splitRe.flags.includes('g') ? splitRe.flags : splitRe.flags + 'g')
    for (const [text, id] of SPECIALS[variant]) {
      this.specialById.set(id, text)
      this.specialByText.set(text, id)
    }
    // Все спецтокены o200k помечены special=True (у gpt-oss skip_special_tokens вырезает разметку Harmony, notes §1.4).
    this.specialTokens = SPECIALS[variant].map(([text, id]) => ({ id, text, hfSpecial: true }))
    // У Harmony спецтокенами считаются и литералы <|reserved_N|> (их ~1 000, поэтому одной общей альтернативой;
    // номер проверяет reservedId).
    this.specialRe = new RegExp(
      SPECIALS[variant]
        .map(([t]) => t)
        .sort((a, b) => b.length - a.length)
        .map(escapeRegExp)
        .join('|') + (variant === 'o200k_harmony' ? '|<\\|reserved_\\d+\\|>' : ''),
      'g',
    )
  }

  /** id для литерала <|reserved_N|>, если gpt-tokenizer считает его спецтокеном (только Harmony). */
  private reservedId(literal: string): number | undefined {
    if (this.variant !== 'o200k_harmony') return undefined
    const m = RESERVED_RE.exec(literal)
    if (!m) return undefined
    const id = Number(m[1])
    // только точный литерал (без ведущих нулей): так его хранит gpt-tokenizer
    if (literal !== `<|reserved_${id}|>`) return undefined
    return isHarmonyReserved(id) || id === 200018 ? id : undefined
  }

  /** id спецтокена по литералу (включая reserved у Harmony). */
  private specialIdOf(literal: string): number | undefined {
    return this.specialByText.get(literal) ?? this.reservedId(literal)
  }

  /** Токен по id; неполные UTF-8 последовательности дают U+FFFD в text. */
  token(id: number): Token {
    const hit = this.cache.get(id)
    if (hit) return hit
    let t: Token
    if (Number.isInteger(id) && id >= 0 && id < O200K_RANKS) {
      const v = this.ranks[id]
      const bytes = typeof v === 'string' ? Array.from(utf8Bytes(v)) : v.slice()
      t = { id, raw: bytesToRaw(bytes), bytes, text: utf8DecodeLossy(bytes), special: false, hfSpecial: false }
    } else {
      let text = this.specialById.get(id)
      // reserved-токены Harmony: 200000, 200001, 200004, 200009…200011, 200013…201087 (см. isHarmonyReserved)
      if (text === undefined && this.variant === 'o200k_harmony' && isHarmonyReserved(id)) {
        text = `<|reserved_${id}|>`
      }
      if (text === undefined) throw new RangeError(`Нет токена с id ${id} в ${this.name}`)
      t = { id, raw: text, bytes: Array.from(utf8Bytes(text)), text, special: true, hfSpecial: true }
    }
    this.cache.set(id, t)
    return t
  }

  /** Кодирование; allowSpecial=false кодирует литералы спецтокенов как обычный текст. */
  encode(text: string, opts?: { allowSpecial?: boolean }): Token[] {
    const ids =
      (opts?.allowSpecial ?? true)
        ? this.encodeFn(text, { allowedSpecial: 'all' })
        : this.encodeFn(text, { disallowedSpecial: new Set() })
    const spans = byteToUtf16Spans(text)
    const total = spans.start.length
    let b = 0
    return ids.map((id) => {
      const base = this.token(id)
      const tok: Token = { ...base }
      if (base.special) {
        // Литерал в тексте может отличаться от канонического (<|reserved_200018|> → 200018 = <|endofprompt|>),
        // поэтому сдвигаемся на длину того литерала, который реально стоит в тексте.
        const u = b < total ? spans.start[b] : text.length
        const literal = [base.text, `<|reserved_${id}|>`].find((l) => text.startsWith(l, u)) ?? base.text
        tok.start = u
        tok.end = u + literal.length
        b += utf8Bytes(literal).length
        return tok
      }
      const len = base.bytes.length
      if (len > 0 && b + len <= total) {
        tok.start = spans.start[b]
        tok.end = spans.end[b + len - 1]
      }
      b += len
      return tok
    })
  }

  /** Сырые байты последовательности. */
  decodeBytes(ids: number[]): Uint8Array {
    const parts = ids.map((id) => this.token(id).bytes)
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0))
    let o = 0
    for (const p of parts) {
      out.set(p, o)
      o += p.length
    }
    return out
  }

  /** Текст последовательности (неполный UTF-8 → U+FFFD). */
  decode(ids: number[], opts?: { skipSpecial?: boolean }): string {
    const list = opts?.skipSpecial ? ids.filter((id) => !this.token(id).hfSpecial) : ids
    return utf8DecodeLossy(this.decodeBytes(list))
  }

  /** id по byte-level форме ('Ġworld') или литералу спецтокена. Индекс строится при первом вызове. */
  idOf(raw: string): number | undefined {
    const s = this.specialIdOf(raw)
    if (s !== undefined) return s
    if (!this.rawIndex) {
      const idx = new Map<string, number>()
      for (let id = 0; id < O200K_RANKS; id++) idx.set(this.token(id).raw, id)
      this.rawIndex = idx
    }
    return this.rawIndex.get(raw)
  }

  /** Пре-токены регуляркой o200k (спецтокены остаются целыми). */
  preTokenize(text: string): string[] {
    const out: string[] = []
    const split = (seg: string) => {
      this.splitRe.lastIndex = 0
      let last = 0
      let m: RegExpExecArray | null
      while ((m = this.splitRe.exec(seg))) {
        if (m[0].length === 0) {
          this.splitRe.lastIndex++
          continue
        }
        if (m.index > last) out.push(seg.slice(last, m.index))
        out.push(m[0])
        last = m.index + m[0].length
      }
      if (last < seg.length) out.push(seg.slice(last))
    }
    this.specialRe.lastIndex = 0
    let pos = 0
    let m: RegExpExecArray | null
    while ((m = this.specialRe.exec(text))) {
      // <|reserved_N|> с номером вне списка спецтокенов — обычный текст
      if (this.specialIdOf(m[0]) === undefined) continue
      if (m.index > pos) split(text.slice(pos, m.index))
      out.push(m[0])
      pos = m.index + m[0].length
    }
    if (pos < text.length) split(text.slice(pos))
    return out
  }

  /**
   * Пошаговое BPE в стиле tiktoken: ранг куска = его id, на каждом шаге склеивается соседняя пара
   * с наименьшим рангом результата (при равенстве левее). Если в строке несколько pre-tokens, склейки
   * не переходят их границы, а спецтокены остаются целыми: последний шаг совпадает с encode().
   */
  trace(pretoken: string): BpeTrace {
    if (!this.byteRank) {
      const m = new Map<string, number>()
      for (let id = 0; id < O200K_RANKS; id++) m.set(String.fromCharCode(...this.token(id).bytes), id)
      this.byteRank = m
    }
    const ranks = this.byteRank
    const bytes = Array.from(utf8Bytes(pretoken))
    const toBytes = (p: string) => Array.from(p, (c) => c.charCodeAt(0))
    // Куски: строка из «байтовых» символов, номер группы (pre-token) и id спецтокена, если это он.
    const parts: string[] = []
    const group: number[] = []
    const special: Array<number | undefined> = []
    this.preTokenize(pretoken).forEach((piece, g) => {
      const sid = this.specialIdOf(piece)
      if (sid !== undefined) {
        parts.push(String.fromCharCode(...utf8Bytes(piece)))
        group.push(g)
        special.push(sid)
        return
      }
      for (const b of utf8Bytes(piece)) {
        parts.push(String.fromCharCode(b))
        group.push(g)
        special.push(undefined)
      }
    })
    const steps: BpeMergeStep[] = []
    for (;;) {
      let best = -1
      let bestRank = Infinity
      for (let i = 0; i + 1 < parts.length; i++) {
        if (group[i] !== group[i + 1] || special[i] !== undefined || special[i + 1] !== undefined) continue
        const r = ranks.get(parts[i] + parts[i + 1])
        if (r !== undefined && r < bestRank) {
          bestRank = r
          best = i
        }
      }
      const step: BpeMergeStep = {
        pieces: parts.map((p, i) => (special[i] !== undefined ? this.token(special[i]!).text : utf8DecodeSoft(toBytes(p)))),
        rawPieces: parts.map((p, i) => (special[i] !== undefined ? this.token(special[i]!).raw : bytesToRaw(toBytes(p)))),
      }
      if (best >= 0) {
        step.merge = { left: bytesToRaw(toBytes(parts[best])), right: bytesToRaw(toBytes(parts[best + 1])), rank: bestRank, index: best }
      }
      steps.push(step)
      if (best < 0) break
      parts.splice(best, 2, parts[best] + parts[best + 1])
      group.splice(best, 2, group[best])
      special.splice(best, 2, undefined)
    }
    const tokens = parts.map((p, i) => {
      if (special[i] !== undefined) return this.token(special[i]!)
      const id = ranks.get(p)
      if (id === undefined) throw new Error(`o200k: кусок без ранга после BPE (${bytesToRaw(toBytes(p))})`)
      return this.token(id)
    })
    return { pretoken, bytes, steps, tokens }
  }
}

const loaded = new Map<O200kVariant, Promise<O200kTokenizer>>()

/** Загружает o200k (по умолчанию o200k_harmony, словарь gpt-oss) динамическим импортом; кэш на модуль. */
export function loadO200kTokenizer(variant: O200kVariant = 'o200k_harmony'): Promise<O200kTokenizer> {
  const hit = loaded.get(variant)
  if (hit) return hit
  const p = Promise.all([
    variant === 'o200k_harmony' ? import('gpt-tokenizer/encoding/o200k_harmony') : import('gpt-tokenizer/encoding/o200k_base'),
    import('gpt-tokenizer/bpeRanks/o200k_base'),
    import('gpt-tokenizer/encodingParams/constants'),
  ]).then(([enc, bpe, consts]) => new O200kTokenizer(variant, enc.encode as EncodeFn, bpe.default, consts.O200K_TOKEN_SPLIT_REGEX))
  p.catch(() => loaded.delete(variant))
  loaded.set(variant, p)
  return p
}
