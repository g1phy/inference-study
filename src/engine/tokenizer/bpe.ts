/**
 * Byte-level BPE как в Hugging Face `tokenizers` для Qwen2/Qwen3:
 *   1) added tokens вырезаются из сырого текста (leftmost-longest, как AddedVocabulary в HF);
 *   2) остальные куски нормализуются NFC;
 *   3) регулярка пре-токенизации (Split, behavior=Isolated) режет текст на pre-tokens;
 *   4) каждый pre-token → UTF-8 байты → byte-level символы;
 *   5) склейки по рангу: на каждом шаге пара с наименьшим рангом, при равенстве левее
 *      (это ровно порядок кучи в `Word::merge_all` из HF tokenizers, src/models/bpe/word.rs).
 * Источник алгоритма: research/notes-tokens-templates-structured.md §1.5, §3.4.
 * Сверено один в один с `tokenizers 0.23.3` (фикстура fixtures/qwen3-encode.json).
 */
import type { BpeMergeStep, BpeTrace, Token, Tokenizer } from '../types'
import type { AddedTokenSpec, BpeAsset } from './asset'
import { byteToUtf16Spans, rawToBytes, utf8Bytes, utf8DecodeLossy, utf8DecodeSoft } from './bytes'

/** Опции encode: allowSpecial=false повторяет HF `split_special_tokens=True`. */
export interface EncodeOptions {
  /**
   * По умолчанию true: литерал added token в тексте становится одним токеном (поведение HF по умолчанию,
   * отсюда инъекции вроде «<|im_end|>» в пользовательском тексте).
   * false: как HF `split_special_tokens=True` — токены с hfSpecial=true кодируются как обычный текст,
   * а added tokens с special=false (<think>, <tool_call>) по-прежнему распознаются целиком.
   */
  allowSpecial?: boolean
}

/** Опции decode. */
export interface DecodeOptions {
  /** Выкинуть токены с hfSpecial=true (как skip_special_tokens=True в HF / vLLM). */
  skipSpecial?: boolean
}

/** 2^18: упаковка пары id в одно число (id < 262 144). */
const PAIR = 262144
/** 2^32: упаковка (ранг, позиция) в ключ кучи. */
const POS = 4294967296
/** Максимум закэшированных pre-tokens (кэш сбрасывается целиком при переполнении). */
const WORD_CACHE_LIMIT = 50000

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Минимальная двоичная куча чисел (ключ = ранг·2^32 + позиция). */
class MinHeap {
  private a: number[] = []
  get size(): number {
    return this.a.length
  }
  push(v: number): void {
    const a = this.a
    a.push(v)
    let i = a.length - 1
    while (i > 0) {
      const p = (i - 1) >> 1
      if (a[p] <= v) break
      a[i] = a[p]
      i = p
    }
    a[i] = v
  }
  pop(): number {
    const a = this.a
    const top = a[0]
    const last = a.pop()!
    if (a.length > 0) {
      let i = 0
      const n = a.length
      for (;;) {
        const l = 2 * i + 1
        if (l >= n) break
        const r = l + 1
        const c = r < n && a[r] < a[l] ? r : l
        if (a[c] >= last) break
        a[i] = a[c]
        i = c
      }
      a[i] = last
    }
    return top
  }
}

/**
 * Отображение позиций NFC-строки на исходную: для каждой UTF-16 позиции нормализованной строки —
 * диапазон в исходной. null, если отобразить не удалось (тогда смещения токенов не известны).
 */
function normalizationMap(seg: string, norm: string): { start: Int32Array; end: Int32Array } | null {
  // Режем исходник на «кластеры»: базовый символ + следующие combining marks.
  const chunks = seg.match(/\P{M}\p{M}*|\p{M}+/gu) ?? []
  const start = new Int32Array(norm.length)
  const end = new Int32Array(norm.length)
  let o = 0
  let n = 0
  for (let ci = 0; ci < chunks.length; ) {
    // Обычно кластер нормализуется сам по себе. Но NFC умеет склеивать и соседние кластеры
    // (чамо хангыля ᄀ + ᅡ + ᆨ → 각): тогда берём несколько кластеров подряд, пока следующий
    // кластер не перестанет менять уже нормализованное начало.
    let k = 1
    let merged = chunks[ci]
    let nc = merged.normalize('NFC')
    for (; ci + k < chunks.length && k < 16; k++) {
      const grown = merged + chunks[ci + k]
      const ng = grown.normalize('NFC')
      if (ng.startsWith(nc) && norm.startsWith(nc, n)) break
      merged = grown
      nc = ng
    }
    if (!norm.startsWith(nc, n)) return null
    for (let j = 0; j < nc.length; j++) {
      start[n + j] = o
      end[n + j] = o + merged.length
    }
    o += merged.length
    n += nc.length
    ci += k
  }
  return n === norm.length ? { start, end } : null
}

/** Byte-level BPE токенизатор (реализация интерфейса Tokenizer для Qwen3). */
export class ByteLevelBpeTokenizer implements Tokenizer {
  readonly name: string
  readonly vocabSize: number
  /** Все added tokens (id, литерал, флаг special из HF). */
  readonly specialTokens: ReadonlyArray<{ id: number; text: string; hfSpecial: boolean }>
  /** Размер BPE-базы без added tokens (Qwen3: 151 643). */
  readonly baseVocabSize: number
  /** Число склеек (Qwen3: 151 387). */
  readonly mergeCount: number

  private readonly raw: string[]
  private readonly byteId = new Int32Array(256)
  private readonly ranks = new Map<number, number>()
  private readonly mergeLeft: Int32Array
  private readonly mergeRight: Int32Array
  private readonly addedByContent = new Map<string, AddedTokenSpec>()
  private readonly addedById = new Map<number, AddedTokenSpec>()
  private readonly addedReAll: RegExp | null
  private readonly addedReNonSpecial: RegExp | null
  private readonly pattern: RegExp
  private readonly nfc: boolean
  private readonly tokens: Array<Token | undefined>
  private readonly wordCache = new Map<string, number[]>()
  private rawIndex: Map<string, number> | null = null

  constructor(asset: BpeAsset) {
    this.name = asset.name
    this.vocabSize = asset.vocabSize
    this.nfc = asset.normalizer === 'NFC'
    this.pattern = new RegExp(asset.pattern, asset.patternFlags.includes('g') ? asset.patternFlags : asset.patternFlags + 'g')

    const base = [...asset.base]
    const m = asset.merges
    const count = m.length / 2
    this.mergeCount = count
    this.baseVocabSize = 256 + count
    this.mergeLeft = new Int32Array(count)
    this.mergeRight = new Int32Array(count)

    const raw: string[] = new Array(this.baseVocabSize)
    for (let i = 0; i < 256; i++) {
      raw[i] = base[i]
      const b = rawToBytes(base[i])
      if (!b || b.length !== 1) throw new Error(`Словарь ${asset.name}: символ базы ${base[i]} не из byte-level алфавита`)
      this.byteId[b[0]] = i
    }
    for (let r = 0; r < count; r++) {
      const l = m[2 * r]
      const rr = m[2 * r + 1]
      this.mergeLeft[r] = l
      this.mergeRight[r] = rr
      raw[256 + r] = raw[l] + raw[rr]
      this.ranks.set(l * PAIR + rr, r)
    }
    this.raw = raw

    for (const t of asset.added) {
      this.addedByContent.set(t.content, t)
      this.addedById.set(t.id, t)
    }
    this.specialTokens = asset.added.map((t) => ({ id: t.id, text: t.content, hfSpecial: t.special }))
    const mk = (list: AddedTokenSpec[]): RegExp | null =>
      list.length === 0
        ? null
        : new RegExp(
            list
              .map((t) => t.content)
              .sort((a, b) => b.length - a.length)
              .map(escapeRegExp)
              .join('|'),
            'g',
          )
    this.addedReAll = mk(asset.added)
    this.addedReNonSpecial = mk(asset.added.filter((t) => !t.special))
    this.tokens = new Array(this.vocabSize)
  }

  /** Токен по id (кэшируется, объект заморожен). Бросает RangeError для id вне словаря. */
  token(id: number): Token {
    if (!Number.isInteger(id) || id < 0 || id >= this.vocabSize) {
      throw new RangeError(`Нет токена с id ${id}: словарь ${this.name} содержит id 0..${this.vocabSize - 1}`)
    }
    const cached = this.tokens[id]
    if (cached) return cached
    let t: Token
    if (id < this.baseVocabSize) {
      const raw = this.raw[id]
      const bytes = rawToBytes(raw)!
      t = { id, raw, bytes: Object.freeze(bytes) as number[], text: utf8DecodeLossy(bytes), special: false, hfSpecial: false }
    } else {
      const a = this.addedById.get(id)
      if (!a) throw new RangeError(`Нет токена с id ${id} в словаре ${this.name}`)
      const bytes = Array.from(utf8Bytes(a.content))
      t = { id, raw: a.content, bytes: Object.freeze(bytes) as number[], text: a.content, special: true, hfSpecial: a.special }
    }
    Object.freeze(t)
    this.tokens[id] = t
    return t
  }

  /** id по форме из словаря (byte-level строка вроде 'Ġworld' или литерал added token). */
  idOf(raw: string): number | undefined {
    const a = this.addedByContent.get(raw)
    if (a) return a.id
    if (!this.rawIndex) {
      const idx = new Map<string, number>()
      for (let i = 0; i < this.raw.length; i++) idx.set(this.raw[i], i)
      this.rawIndex = idx
    }
    return this.rawIndex.get(raw)
  }

  /** Ранг склейки пары id (или undefined, если такой склейки нет). */
  mergeRank(left: number, right: number): number | undefined {
    return this.ranks.get(left * PAIR + right)
  }

  /** Из каких двух токенов склеивается токен id (undefined для байтов и added tokens). */
  mergeOf(id: number): { left: number; right: number; rank: number } | undefined {
    if (id < 256 || id >= this.baseVocabSize) return undefined
    const r = id - 256
    return { left: this.mergeLeft[r], right: this.mergeRight[r], rank: r }
  }

  /** Кодирование текста в токены со смещениями (UTF-16) в исходной строке. */
  encode(text: string, opts?: EncodeOptions): Token[] {
    const out: Token[] = []
    const re = (opts?.allowSpecial ?? true) ? this.addedReAll : this.addedReNonSpecial
    let pos = 0
    if (re) {
      re.lastIndex = 0
      let m: RegExpExecArray | null
      while ((m = re.exec(text))) {
        if (m.index > pos) this.encodeSegment(text, pos, m.index, out)
        const spec = this.addedByContent.get(m[0])!
        out.push({ ...this.token(spec.id), start: m.index, end: m.index + m[0].length })
        pos = m.index + m[0].length
      }
    }
    if (pos < text.length) this.encodeSegment(text, pos, text.length, out)
    return out
  }

  /** Только id (быстрее, без объектов Token). */
  encodeIds(text: string, opts?: EncodeOptions): number[] {
    return this.encode(text, opts).map((t) => t.id)
  }

  /** Сырые байты последовательности (added tokens дают байты своего литерала). */
  decodeBytes(ids: number[]): Uint8Array {
    let n = 0
    for (const id of ids) n += this.token(id).bytes.length
    const out = new Uint8Array(n)
    let o = 0
    for (const id of ids) {
      const b = this.token(id).bytes
      out.set(b, o)
      o += b.length
    }
    return out
  }

  /** Текст последовательности: байты → UTF-8 с заменой неполных последовательностей на U+FFFD. */
  decode(ids: number[], opts?: DecodeOptions): string {
    const list = opts?.skipSpecial ? ids.filter((id) => !this.token(id).hfSpecial) : ids
    return utf8DecodeLossy(this.decodeBytes(list))
  }

  /** Разбиение на pre-tokens (после NFC); литералы added tokens остаются целыми. */
  preTokenize(text: string): string[] {
    const out: string[] = []
    const pushSeg = (seg: string) => {
      const norm = this.nfc ? seg.normalize('NFC') : seg
      this.forEachPretoken(norm, (p) => out.push(p))
    }
    const re = this.addedReAll
    let pos = 0
    if (re) {
      re.lastIndex = 0
      let m: RegExpExecArray | null
      while ((m = re.exec(text))) {
        if (m.index > pos) pushSeg(text.slice(pos, m.index))
        out.push(m[0])
        pos = m.index + m[0].length
      }
    }
    if (pos < text.length) pushSeg(text.slice(pos))
    return out
  }

  /**
   * Пошаговое BPE: steps[0] — отдельные байты, далее по одной склейке за шаг (наименьший ранг среди
   * соседних пар, при равенстве левее). merge.left/right в byte-level форме, index — позиция левого
   * куска в pieces. Обычно сюда передают один pre-token; если в строке их несколько (например, слово
   * с пробелами вводят руками), склейки не переходят границы pre-tokens, а литералы added tokens
   * остаются целыми — так последний шаг всегда совпадает с encode() (раньше «a  b» давало
   * ['a', '  ', 'b'] вместо ['a', ' ', ' b']). NFC применяется как в encode.
   */
  trace(pretoken: string): BpeTrace {
    // Символы: id и номер группы (pre-token); склеиваются только соседи из одной группы.
    const ids: number[] = []
    const group: number[] = []
    const atomic: boolean[] = []
    let g = 0
    let word = ''
    const pushSegment = (seg: string) => {
      const norm = this.nfc ? seg.normalize('NFC') : seg
      word += norm
      this.forEachPretoken(norm, (piece) => {
        for (const b of utf8Bytes(piece)) {
          ids.push(this.byteId[b])
          group.push(g)
          atomic.push(false)
        }
        g++
      })
    }
    const re = this.addedReAll
    let pos = 0
    if (re) {
      re.lastIndex = 0
      let m: RegExpExecArray | null
      while ((m = re.exec(pretoken))) {
        if (m.index > pos) pushSegment(pretoken.slice(pos, m.index))
        ids.push(this.addedByContent.get(m[0])!.id)
        group.push(g++)
        atomic.push(true)
        word += m[0]
        pos = m.index + m[0].length
      }
    }
    if (pos < pretoken.length) pushSegment(pretoken.slice(pos))

    const bytes = Array.from(utf8Bytes(word))
    const steps: BpeMergeStep[] = []
    for (;;) {
      let best = -1
      let bestRank = Infinity
      for (let i = 0; i + 1 < ids.length; i++) {
        if (group[i] !== group[i + 1] || atomic[i] || atomic[i + 1]) continue
        const r = this.ranks.get(ids[i] * PAIR + ids[i + 1])
        if (r !== undefined && r < bestRank) {
          bestRank = r
          best = i
        }
      }
      const step: BpeMergeStep = {
        pieces: ids.map((id) => utf8DecodeSoft(this.token(id).bytes)),
        rawPieces: ids.map((id) => this.token(id).raw),
      }
      if (best >= 0) {
        step.merge = { left: this.raw[ids[best]], right: this.raw[ids[best + 1]], rank: bestRank, index: best }
      }
      steps.push(step)
      if (best < 0) break
      ids.splice(best, 2, 256 + bestRank)
      group.splice(best, 2, group[best])
      atomic.splice(best, 2, false)
    }
    return { pretoken: word, bytes, steps, tokens: ids.map((id) => this.token(id)) }
  }

  /** BPE одного pre-token через кучу (O(n log n)), с кэшем по строке. */
  bpeWord(word: string): number[] {
    const hit = this.wordCache.get(word)
    if (hit) return hit
    const bytes = utf8Bytes(word)
    const n = bytes.length
    const ids = new Int32Array(n)
    const prev = new Int32Array(n)
    const next = new Int32Array(n)
    const alive = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      ids[i] = this.byteId[bytes[i]]
      prev[i] = i - 1
      next[i] = i + 1 < n ? i + 1 : -1
      alive[i] = 1
    }
    const heap = new MinHeap()
    for (let i = 0; i + 1 < n; i++) {
      const r = this.ranks.get(ids[i] * PAIR + ids[i + 1])
      if (r !== undefined) heap.push(r * POS + i)
    }
    while (heap.size > 0) {
      const key = heap.pop()
      const rank = Math.floor(key / POS)
      const pos = key - rank * POS
      if (!alive[pos]) continue
      const nx = next[pos]
      if (nx < 0) continue
      // Запись устарела, если пара на этой позиции уже другая.
      if (this.ranks.get(ids[pos] * PAIR + ids[nx]) !== rank) continue
      ids[pos] = 256 + rank
      alive[nx] = 0
      const nn = next[nx]
      next[pos] = nn
      if (nn >= 0) prev[nn] = pos
      const pv = prev[pos]
      if (pv >= 0) {
        const r = this.ranks.get(ids[pv] * PAIR + ids[pos])
        if (r !== undefined) heap.push(r * POS + pv)
      }
      if (nn >= 0) {
        const r = this.ranks.get(ids[pos] * PAIR + ids[nn])
        if (r !== undefined) heap.push(r * POS + pos)
      }
    }
    const out: number[] = []
    for (let i = 0; i < n; i++) if (alive[i]) out.push(ids[i])
    if (this.wordCache.size >= WORD_CACHE_LIMIT) this.wordCache.clear()
    this.wordCache.set(word, out)
    return out
  }

  /** Обходит pre-tokens нормализованной строки (behavior=Isolated: промежутки без совпадения тоже pre-tokens). */
  private forEachPretoken(norm: string, fn: (piece: string, index: number) => void): void {
    const re = this.pattern
    re.lastIndex = 0
    let last = 0
    let m: RegExpExecArray | null
    while ((m = re.exec(norm))) {
      if (m[0].length === 0) {
        re.lastIndex++
        continue
      }
      if (m.index > last) fn(norm.slice(last, m.index), last)
      fn(m[0], m.index)
      last = m.index + m[0].length
    }
    if (last < norm.length) fn(norm.slice(last), last)
  }

  private encodeSegment(text: string, from: number, to: number, out: Token[]): void {
    const seg = text.slice(from, to)
    const norm = this.nfc ? seg.normalize('NFC') : seg
    const map = norm === seg ? undefined : normalizationMap(seg, norm)
    this.forEachPretoken(norm, (piece, at) => {
      const ids = this.bpeWord(piece)
      const known = map !== null
      const spans = known ? byteToUtf16Spans(piece) : null
      let b = 0
      for (const id of ids) {
        const base = this.token(id)
        const len = base.bytes.length
        if (spans && len > 0) {
          let s = at + spans.start[b]
          let e = at + spans.end[b + len - 1]
          if (map) {
            s = map.start[s]
            e = map.end[e - 1]
          }
          out.push({ ...base, start: from + s, end: from + e })
        } else {
          out.push({ ...base })
        }
        b += len
      }
    })
  }
}

/** Собирает токенизатор из распарсенного словаря (см. parseBpeAsset). */
export function createBpeTokenizer(asset: BpeAsset): ByteLevelBpeTokenizer {
  return new ByteLevelBpeTokenizer(asset)
}
