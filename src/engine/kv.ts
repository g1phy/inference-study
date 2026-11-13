/**
 * KV cache: хэши блоков для prefix caching, симуляция PagedAttention (и наивного непрерывного
 * резервирования для сравнения), radix tree в стиле SGLang RadixAttention (агент A2).
 *
 * Источники: research/notes-dynamo.md §1.6, §3.1 (хэши блоков Dynamo);
 * research/notes-engines.md §1.1–1.2 (PagedAttention), §1.7 (APC vLLM), §1.14 (RadixAttention),
 * §3.1–3.2 (пример «Four score…» и copy-on-write), §3.12 (рост и вытеснение radix tree).
 */
import type { BlockInfo, PrefixMatch, RadixNodeView } from './types'
import { hash64Words, hex32 } from './rng'

/* ============================================================================
 * Хэши блоков
 * ========================================================================== */

/**
 * Размер блока по умолчанию: 16 токенов (vLLM, все релизы до v0.33; в main с 2026-11-05 стало 64,
 * notes-engines.md §1.2; SPEC §8.8).
 */
export const DEFAULT_BLOCK_SIZE = 16

/**
 * Seed цепочки. Dynamo хэширует блоки XXH3_64 с seed 1337 (notes-dynamo.md §1.6 [C-HASH]);
 * мы берём тот же seed для узнаваемости, но сама функция другая (см. blockHashes).
 */
export const CHAIN_SEED = 1337

/** BlockInfo + полные 16 hex хэша и хэш родителя (для «Глубже»). */
export interface BlockInfoExt extends BlockInfo {
  /** Полный 64-битный хэш, 16 hex; '' у неполного хвостового блока. */
  fullHash: string
  /** Полный хэш предыдущего блока цепочки ('' у первого блока без соли). */
  parentHash: string
  /** Индекс первого токена блока в последовательности. */
  start: number
}

/** Строка → 32-битные слова (UTF-16 code units) для хэширования соли. */
function stringWords(s: string): number[] {
  const out: number[] = []
  for (let i = 0; i < s.length; i++) out.push(s.charCodeAt(i))
  return out
}

/**
 * Разбивает токены на блоки по blockSize и считает цепочку хэшей: hash_i = H(hash_{i−1}, токены блока i),
 * поэтому правка раннего токена меняет хэши всех следующих блоков.
 *
 * Хэшируются только ПОЛНЫЕ блоки: неполный хвост (меньше blockSize токенов) возвращается с
 * full = false и hash = '' (так делают и vLLM, «we only cache full blocks», и Dynamo, хэширующий
 * floor(ISL / B) блоков: notes-engines.md §1.7, notes-dynamo.md §1.6).
 *
 * Функция хэша: быстрый детерминированный 64-битный хэш hash64Words (rng.ts) по словам
 * [parent_hi, parent_lo, token ids...]. Настоящие системы используют другое: Dynamo XXH3_64
 * с seed 1337 (local hash блока + sequence hash цепочки), vLLM sha256 по умолчанию
 * (`--prefix-caching-hash-algo`, с v0.11). Значения хэшей у нас для показа, принцип тот же.
 * `hash` — первые 8 hex полного хэша (BlockInfo: «короткий хэш для показа»).
 *
 * salt (cache_salt / тенант / LoRA) подмешивается в начало цепочки: при другой соли меняются все хэши
 * (в vLLM соль входит в хэш первого блока, в Dynamo в seed: notes-engines.md §3.6, notes-dynamo.md §1.6).
 */
export function blockHashes(tokenIds: readonly number[], blockSize = DEFAULT_BLOCK_SIZE, salt?: string): BlockInfoExt[] {
  if (!(blockSize >= 1)) throw new Error('blockHashes: blockSize должен быть ≥ 1')
  const B = Math.floor(blockSize)
  const out: BlockInfoExt[] = []
  let parent: [number, number] = salt ? hash64Words(stringWords(salt), CHAIN_SEED ^ 0x5a17) : [0, 0]
  let parentHex = salt ? hex32(parent[0]) + hex32(parent[1]) : ''
  for (let start = 0, index = 0; start < tokenIds.length; start += B, index++) {
    const ids = tokenIds.slice(start, start + B)
    if (ids.length < B) {
      out.push({ index, hash: '', tokenIds: ids.slice(), full: false, fullHash: '', parentHash: parentHex, start })
      break
    }
    const words = new Array<number>(ids.length + 2)
    words[0] = parent[0]
    words[1] = parent[1]
    for (let i = 0; i < ids.length; i++) words[i + 2] = ids[i]
    const h = hash64Words(words, CHAIN_SEED)
    const fullHash = hex32(h[0]) + hex32(h[1])
    out.push({ index, hash: fullHash.slice(0, 8), tokenIds: ids.slice(), full: true, fullHash, parentHash: parentHex, start })
    parent = h
    parentHex = fullHash
  }
  return out
}

/** Сколько первых ПОЛНЫХ блоков совпадает у двух цепочек (совпадение только точное и только с начала). */
export function commonPrefixBlocks(a: readonly BlockInfo[], b: readonly BlockInfo[]): number {
  let n = 0
  while (n < a.length && n < b.length && a[n].full && b[n].full && a[n].hash === b[n].hash) n++
  return n
}

/** Сколько первых полных блоков запроса лежит в наборе хэшей (prefix hit, в блоках). */
export function cachedPrefixBlocks(blocks: readonly BlockInfo[], cached: ReadonlySet<string>): number {
  let n = 0
  while (n < blocks.length && blocks[n].full && cached.has(blocks[n].hash)) n++
  return n
}

/* ============================================================================
 * Статистика аллокаторов
 * ========================================================================== */

/** Утилизация памяти KV (в слотах = токенах). */
export interface AllocatorStats {
  capacitySlots: number
  /** Слоты в выделенных блоках / зарезервированных регионах. */
  allocatedSlots: number
  /** Слоты с настоящими токенами (общий блок считается один раз). */
  usedSlots: number
  /** Сумма длин последовательностей (общий префикс считается у каждой). */
  logicalTokens: number
  freeSlots: number
  /** Выделено, но пусто: allocated − used (внутренняя фрагментация + резерв под будущие токены). */
  internalWasteSlots: number
  /** Свободно, но не годится под новый запрос: дыры меньше типичной брони (только у непрерывного). */
  externalWasteSlots: number
  /** used / allocated: какая доля выделенной памяти занята токенами (0, если ничего не выделено). */
  utilization: number
  /** allocated / capacity. */
  occupancy: number
  internalFragmentation: number
  externalFragmentation: number
  sequences: number
  /** Сколько слотов сэкономил шаринг блоков (logicalTokens − usedSlots), у непрерывного 0. */
  sharedSavedSlots: number
}

/** Событие аллокатора (для пошаговой анимации), note по-русски. */
export interface AllocEvent {
  kind: 'allocate' | 'append' | 'new-block' | 'cow' | 'free' | 'fork' | 'fail'
  seqId: string
  /** Затронутые физические блоки (или начало региона у непрерывного). */
  blocks: number[]
  note: string
}

function finishStats(s: Omit<AllocatorStats, 'utilization' | 'occupancy' | 'internalFragmentation' | 'externalFragmentation' | 'internalWasteSlots'>): AllocatorStats {
  const internal = s.allocatedSlots - s.usedSlots
  return {
    ...s,
    internalWasteSlots: internal,
    utilization: s.allocatedSlots > 0 ? s.usedSlots / s.allocatedSlots : 0,
    occupancy: s.capacitySlots > 0 ? s.allocatedSlots / s.capacitySlots : 0,
    internalFragmentation: s.capacitySlots > 0 ? internal / s.capacitySlots : 0,
    externalFragmentation: s.capacitySlots > 0 ? s.externalWasteSlots / s.capacitySlots : 0,
  }
}

/* ============================================================================
 * PagedAllocator: KV блоками фиксированного размера, block tables, ref counts, CoW
 * ========================================================================== */

/** Состояние физического блока. */
export interface PhysicalBlockView {
  id: number
  refCount: number
  /** Сколько слотов блока занято токенами. */
  filled: number
  /** Последовательности, в block table которых стоит блок. */
  owners: string[]
  free: boolean
}

/** Состояние последовательности. */
export interface PagedSeqView {
  id: string
  tokens: number
  /** Логический блок i → физический блок blockTable[i]. */
  blockTable: number[]
  /** Сколько слотов занято в последнем логическом блоке. */
  filledInLast: number
}

/**
 * Симуляция PagedAttention (vLLM): KV последовательности лежит в физических блоках по blockSize
 * токенов, не обязательно подряд; block table отображает логические блоки в физические; блок
 * выдаётся, только когда заполнен предыдущий (потери ≤ blockSize − 1 на последовательность).
 * fork() делит блоки между последовательностями (ref count), запись в общий неполный блок
 * вызывает copy-on-write (notes-engines.md §1.1, §3.1–3.2).
 */
export class PagedAllocator {
  readonly blockSize: number
  readonly numBlocks: number
  /** События последней операции (для анимации). */
  lastEvents: AllocEvent[] = []
  private refCount: number[]
  private filled: number[]
  private freeList: number[]
  private seqs = new Map<string, { tokens: number; table: number[] }>()

  /**
   * @param opts.numBlocks сколько физических блоков в пуле
   * @param opts.blockSize токенов в блоке (по умолчанию 16)
   * @param opts.freeOrder порядок выдачи свободных блоков: по возрастанию id или «вперемешку»
   *   (детерминированно), чтобы было видно, что блоки последовательности не обязаны лежать подряд
   */
  constructor(opts: { numBlocks: number; blockSize?: number; freeOrder?: 'ascending' | 'scattered' }) {
    this.numBlocks = Math.max(0, Math.floor(opts.numBlocks))
    this.blockSize = Math.max(1, Math.floor(opts.blockSize ?? DEFAULT_BLOCK_SIZE))
    this.refCount = new Array(this.numBlocks).fill(0)
    this.filled = new Array(this.numBlocks).fill(0)
    const ids = Array.from({ length: this.numBlocks }, (_, i) => i)
    if (opts.freeOrder === 'scattered') {
      // Детерминированная перестановка: шаг, взаимно простой с numBlocks.
      const n = this.numBlocks
      let step = Math.max(1, Math.round(n * 0.618))
      while (n > 1 && gcd(step, n) !== 1) step++
      this.freeList = ids.map((i) => (i * step + Math.floor(n / 3)) % Math.max(1, n))
    } else {
      this.freeList = ids
    }
  }

  /** Сколько свободных физических блоков. */
  get freeBlocks(): number {
    return this.freeList.length
  }

  private takeBlock(): number {
    const id = this.freeList.shift()
    if (id === undefined) throw new Error('PagedAllocator: нет свободных блоков')
    this.refCount[id] = 1
    this.filled[id] = 0
    return id
  }

  private releaseBlock(id: number): void {
    this.refCount[id] = 0
    this.filled[id] = 0
    this.freeList.push(id)
  }

  /** Новая последовательность с `tokens` токенами промпта: выделяет ceil(tokens / B) блоков. false, если не хватает. */
  allocate(seqId: string, tokens: number): boolean {
    this.lastEvents = []
    if (this.seqs.has(seqId)) throw new Error(`PagedAllocator: последовательность «${seqId}» уже есть`)
    const t = Math.max(0, Math.floor(tokens))
    const need = Math.ceil(t / this.blockSize)
    if (need > this.freeList.length) {
      this.lastEvents.push({ kind: 'fail', seqId, blocks: [], note: `Нужно блоков: ${need}, свободно: ${this.freeList.length}. Запрос ждёт.` })
      return false
    }
    const table: number[] = []
    for (let i = 0; i < need; i++) {
      const id = this.takeBlock()
      this.filled[id] = Math.min(this.blockSize, t - i * this.blockSize)
      table.push(id)
    }
    this.seqs.set(seqId, { tokens: t, table })
    this.lastEvents.push({ kind: 'allocate', seqId, blocks: table.slice(), note: `Промпт ${t} ток. занял ${need} бл.: ${table.join(', ')}.` })
    return true
  }

  /**
   * Дописывает n токенов (decode). Новый блок выдаётся, когда последний заполнен; запись в общий
   * неполный блок (refCount > 1) сначала копирует его (copy-on-write). Атомарно: при нехватке
   * блоков ничего не меняется и возвращается false.
   */
  append(seqId: string, n = 1): boolean {
    this.lastEvents = []
    const s = this.seqs.get(seqId)
    if (!s) throw new Error(`PagedAllocator: нет последовательности «${seqId}»`)
    const add = Math.max(0, Math.floor(n))
    if (add === 0) return true
    // Сколько новых блоков понадобится (включая копию для CoW).
    const B = this.blockSize
    const lastId = s.table.length ? s.table[s.table.length - 1] : -1
    const lastFilled = s.tokens - (s.table.length - 1) * B
    const roomInLast = s.table.length ? B - lastFilled : 0
    const cowNeeded = s.table.length > 0 && roomInLast > 0 && this.refCount[lastId] > 1
    const newBlocks = Math.ceil(Math.max(0, add - roomInLast) / B) + (cowNeeded ? 1 : 0)
    if (newBlocks > this.freeList.length) {
      this.lastEvents.push({ kind: 'fail', seqId, blocks: [], note: `Не хватает блоков (нужно ${newBlocks}, свободно ${this.freeList.length}): нужна preemption.` })
      return false
    }
    if (cowNeeded) {
      const copy = this.takeBlock()
      this.filled[copy] = this.filled[lastId]
      this.refCount[lastId] -= 1
      s.table[s.table.length - 1] = copy
      this.lastEvents.push({
        kind: 'cow',
        seqId,
        blocks: [lastId, copy],
        note: `Блок ${lastId} общий (ref = ${this.refCount[lastId] + 1}): copy-on-write в новый блок ${copy}.`,
      })
    }
    let left = add
    while (left > 0) {
      const tail = s.table.length ? s.table[s.table.length - 1] : -1
      const tailFilled = s.table.length ? s.tokens - (s.table.length - 1) * B : B
      if (s.table.length === 0 || tailFilled >= B) {
        const id = this.takeBlock()
        s.table.push(id)
        this.lastEvents.push({ kind: 'new-block', seqId, blocks: [id], note: `Последний блок полон: выдан новый блок ${id}.` })
        continue
      }
      const put = Math.min(left, B - tailFilled)
      this.filled[tail] = tailFilled + put
      s.tokens += put
      left -= put
    }
    this.lastEvents.push({ kind: 'append', seqId, blocks: [s.table[s.table.length - 1]], note: `+${add} ток., всего ${s.tokens}.` })
    return true
  }

  /** Освобождает последовательность: ref count её блоков уменьшается, блоки с нулём возвращаются в пул. */
  free(seqId: string): void {
    this.lastEvents = []
    const s = this.seqs.get(seqId)
    if (!s) return
    const released: number[] = []
    // В обратном порядке, как vLLM: хвостовые блоки первыми попадают в очередь свободных.
    for (let i = s.table.length - 1; i >= 0; i--) {
      const id = s.table[i]
      this.refCount[id] -= 1
      if (this.refCount[id] <= 0) {
        this.releaseBlock(id)
        released.push(id)
      }
    }
    this.seqs.delete(seqId)
    this.lastEvents.push({
      kind: 'free',
      seqId,
      blocks: released,
      note: released.length ? `Освобождены блоки: ${released.join(', ')}.` : 'Все блоки ещё нужны другим последовательностям.',
    })
  }

  /** Копия последовательности, делящая с ней все блоки (parallel sampling, beam search): ref count +1. */
  fork(parentId: string, childId: string): void {
    this.lastEvents = []
    const p = this.seqs.get(parentId)
    if (!p) throw new Error(`PagedAllocator: нет последовательности «${parentId}»`)
    if (this.seqs.has(childId)) throw new Error(`PagedAllocator: последовательность «${childId}» уже есть`)
    for (const id of p.table) this.refCount[id] += 1
    this.seqs.set(childId, { tokens: p.tokens, table: p.table.slice() })
    this.lastEvents.push({ kind: 'fork', seqId: childId, blocks: p.table.slice(), note: `«${childId}» делит с «${parentId}» блоки ${p.table.join(', ')}.` })
  }

  /** Есть ли последовательность. */
  has(seqId: string): boolean {
    return this.seqs.has(seqId)
  }

  /** Block table последовательности (копия). */
  blockTable(seqId: string): number[] {
    return this.seqs.get(seqId)?.table.slice() ?? []
  }

  /** Физический слот токена на позиции pos: block_table[pos / B]·B + pos mod B (slot_mapping, notes-engines.md §1.2). */
  slotOf(seqId: string, pos: number): number {
    const s = this.seqs.get(seqId)
    if (!s || pos < 0 || pos >= s.tokens) return -1
    return s.table[Math.floor(pos / this.blockSize)] * this.blockSize + (pos % this.blockSize)
  }

  /** Все последовательности. */
  sequences(): PagedSeqView[] {
    const out: PagedSeqView[] = []
    for (const [id, s] of this.seqs) {
      out.push({ id, tokens: s.tokens, blockTable: s.table.slice(), filledInLast: s.table.length ? s.tokens - (s.table.length - 1) * this.blockSize : 0 })
    }
    return out
  }

  /** Все физические блоки по id. */
  physicalBlocks(): PhysicalBlockView[] {
    const owners: string[][] = Array.from({ length: this.numBlocks }, () => [])
    for (const [id, s] of this.seqs) for (const b of s.table) owners[b].push(id)
    return Array.from({ length: this.numBlocks }, (_, id) => ({
      id,
      refCount: this.refCount[id],
      filled: this.filled[id],
      owners: owners[id],
      free: this.refCount[id] === 0,
    }))
  }

  /** Утилизация: у PagedAttention внешней фрагментации нет, внутренняя ≤ B − 1 на последовательность. */
  stats(): AllocatorStats {
    const allocatedBlocks = this.numBlocks - this.freeList.length
    let used = 0
    for (let id = 0; id < this.numBlocks; id++) if (this.refCount[id] > 0) used += this.filled[id]
    let logical = 0
    for (const s of this.seqs.values()) logical += s.tokens
    return finishStats({
      capacitySlots: this.numBlocks * this.blockSize,
      allocatedSlots: allocatedBlocks * this.blockSize,
      usedSlots: used,
      logicalTokens: logical,
      freeSlots: this.freeList.length * this.blockSize,
      externalWasteSlots: 0,
      sequences: this.seqs.size,
      sharedSavedSlots: logical - used,
    })
  }
}

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b]
  return a
}

/* ============================================================================
 * ContiguousAllocator: наивное непрерывное резервирование (до PagedAttention)
 * ========================================================================== */

/** Регион памяти непрерывного аллокатора: занятый (seqId) или дыра (seqId = null). */
export interface ContiguousRegion {
  start: number
  length: number
  seqId: string | null
  /** Сколько слотов региона заняты токенами. */
  used: number
}

/**
 * Наивный аллокатор для сравнения: каждой последовательности сразу резервируется непрерывный
 * кусок под максимальную длину (как Orca / FasterTransformer до vLLM), first-fit в линейной памяти.
 * Потери: зарезервировано, но пусто (внутренние) и дыры между кусками разного размера (внешние),
 * notes-engines.md §1.1, §3.1 (у Orca (Max) из брони 2048 слотов не используется 2038).
 */
export class ContiguousAllocator {
  readonly capacity: number
  /** Бронь по умолчанию (max_len). */
  readonly maxLen: number
  lastEvents: AllocEvent[] = []
  private regionsList: Array<{ start: number; length: number; seqId: string; used: number }> = []

  /** @param opts.capacity сколько слотов (токенов) в памяти; @param opts.maxLen бронь по умолчанию на запрос */
  constructor(opts: { capacity: number; maxLen: number }) {
    this.capacity = Math.max(0, Math.floor(opts.capacity))
    this.maxLen = Math.max(1, Math.floor(opts.maxLen))
  }

  /** Свободные дыры по возрастанию адреса (брони нулевой длины память не занимают и дыру не делят). */
  private holes(): Array<{ start: number; length: number }> {
    const sorted = this.regionsList.filter((r) => r.length > 0).sort((a, b) => a.start - b.start)
    const out: Array<{ start: number; length: number }> = []
    let cur = 0
    for (const r of sorted) {
      if (r.start > cur) out.push({ start: cur, length: r.start - cur })
      cur = r.start + r.length
    }
    if (cur < this.capacity) out.push({ start: cur, length: this.capacity - cur })
    return out
  }

  /** Новая последовательность: резервирует непрерывно `reserve` слотов (по умолчанию maxLen), first-fit. */
  allocate(seqId: string, tokens: number, reserve = this.maxLen): boolean {
    this.lastEvents = []
    if (this.regionsList.some((r) => r.seqId === seqId)) throw new Error(`ContiguousAllocator: «${seqId}» уже есть`)
    const t = Math.max(0, Math.floor(tokens))
    const size = Math.max(t, Math.floor(reserve), 0)
    const hole = size === 0 ? { start: 0, length: 0 } : this.holes().find((h) => h.length >= size)
    if (!hole) {
      const free = this.capacity - this.regionsList.reduce((a, r) => a + r.length, 0)
      this.lastEvents.push({
        kind: 'fail',
        seqId,
        blocks: [],
        note:
          free >= size
            ? `Свободно ${free} слотов, но нет непрерывного куска на ${size}: внешняя фрагментация.`
            : `Нужно ${size} слотов, свободно ${free}. Запрос ждёт.`,
      })
      return false
    }
    this.regionsList.push({ start: hole.start, length: size, seqId, used: t })
    this.lastEvents.push({ kind: 'allocate', seqId, blocks: [hole.start], note: `Забронировано ${size} слотов с адреса ${hole.start}, занято ${t}.` })
    return true
  }

  /** Дописывает n токенов внутри брони; false, если бронь кончилась (у наивной схемы запрос обрывается). */
  append(seqId: string, n = 1): boolean {
    this.lastEvents = []
    const r = this.regionsList.find((x) => x.seqId === seqId)
    if (!r) throw new Error(`ContiguousAllocator: нет последовательности «${seqId}»`)
    const add = Math.max(0, Math.floor(n))
    if (r.used + add > r.length) {
      this.lastEvents.push({ kind: 'fail', seqId, blocks: [r.start], note: `Бронь ${r.length} слотов кончилась.` })
      return false
    }
    r.used += add
    this.lastEvents.push({ kind: 'append', seqId, blocks: [r.start], note: `+${add} ток., занято ${r.used} из ${r.length}.` })
    return true
  }

  /** Освобождает бронь последовательности целиком. */
  free(seqId: string): void {
    this.lastEvents = []
    const i = this.regionsList.findIndex((x) => x.seqId === seqId)
    if (i < 0) return
    const [r] = this.regionsList.splice(i, 1)
    this.lastEvents.push({ kind: 'free', seqId, blocks: [r.start], note: `Освобождено ${r.length} слотов с адреса ${r.start}.` })
  }

  /** Есть ли последовательность. */
  has(seqId: string): boolean {
    return this.regionsList.some((r) => r.seqId === seqId)
  }

  /** Вся память по порядку адресов: занятые регионы и дыры (брони нулевой длины не показываются). */
  regions(): ContiguousRegion[] {
    const busy: ContiguousRegion[] = this.regionsList
      .filter((r) => r.length > 0)
      .map((r) => ({ start: r.start, length: r.length, seqId: r.seqId, used: r.used }))
    const free: ContiguousRegion[] = this.holes().map((h) => ({ start: h.start, length: h.length, seqId: null, used: 0 }))
    return busy.concat(free).sort((a, b) => a.start - b.start)
  }

  /** Утилизация: внешние потери = свободные слоты в дырах меньше maxLen (туда новую бронь не положить). */
  stats(): AllocatorStats {
    let allocated = 0
    let used = 0
    for (const r of this.regionsList) {
      allocated += r.length
      used += r.used
    }
    let externalWaste = 0
    for (const h of this.holes()) if (h.length < this.maxLen) externalWaste += h.length
    return finishStats({
      capacitySlots: this.capacity,
      allocatedSlots: allocated,
      usedSlots: used,
      logicalTokens: used,
      freeSlots: this.capacity - allocated,
      externalWasteSlots: externalWaste,
      sequences: this.regionsList.length,
      sharedSavedSlots: 0,
    })
  }
}

/* ============================================================================
 * RadixTree: префиксное дерево KV в стиле SGLang RadixAttention
 * ========================================================================== */

interface RNode {
  id: string
  seq: number
  tokens: number[]
  children: Map<number, RNode>
  parent: RNode | null
  refs: number
  lastAccess: number
  hit: boolean
}

/** Результат вставки в radix tree. */
export interface RadixInsertResult {
  /** Сколько токенов уже было в дереве (prefix hit). */
  matchedTokens: number
  /** Сколько токенов добавлено (их KV пришлось бы посчитать). */
  insertedTokens: number
  totalTokens: number
  /** id узлов на пути от корня (без корня), включая новый лист. */
  path: string[]
}

/** Результат вытеснения. */
export interface RadixEvictResult {
  evictedTokens: number
  evictedNodes: string[]
}

/**
 * Сжатое префиксное дерево (radix tree) последовательностей токенов, как в SGLang RadixAttention:
 * ребро помечено последовательностью токенов, общий префикс хранится один раз, совпадение
 * посреди ребра расщепляет узел, вытесняются LRU-листья без активных ссылок (lock_ref = 0),
 * общие предки переживают потомков (notes-engines.md §1.14, §3.12). Размер страницы 1 токен
 * (SGLang `--page-size` по умолчанию 1). Время доступа — логические часы (каждая операция +1).
 */
export class RadixTree {
  private root: RNode
  private nodes = new Map<string, RNode>()
  private clock = 0
  private seqCounter = 0
  private total = 0

  constructor() {
    this.root = { id: 'root', seq: 0, tokens: [], children: new Map(), parent: null, refs: 0, lastAccess: 0, hit: false }
  }

  /** Сколько токенов хранится в дереве (сумма длин рёбер). */
  get size(): number {
    return this.total
  }

  /** Текущее значение логических часов. */
  get now(): number {
    return this.clock
  }

  /** Число узлов без корня. */
  nodeCount(): number {
    return this.nodes.size
  }

  private newNode(tokens: number[], parent: RNode): RNode {
    const n: RNode = { id: `n${++this.seqCounter}`, seq: this.seqCounter, tokens, children: new Map(), parent, refs: 0, lastAccess: this.clock, hit: false }
    this.nodes.set(n.id, n)
    return n
  }

  /** Расщепляет ребро узла child после k токенов: появляется промежуточный узел с первыми k токенами. */
  private split(child: RNode, k: number): RNode {
    const parent = child.parent!
    const mid = this.newNode(child.tokens.slice(0, k), parent)
    mid.refs = child.refs // как SGLang _split_node: блокировка потомка держит и префикс
    mid.lastAccess = child.lastAccess
    mid.hit = child.hit
    parent.children.set(mid.tokens[0], mid)
    child.tokens = child.tokens.slice(k)
    child.parent = mid
    mid.children.set(child.tokens[0], child)
    return mid
  }

  private clearHits(): void {
    for (const n of this.nodes.values()) n.hit = false
  }

  /** Спуск по дереву: самый длинный префикс; на частичном совпадении ребро расщепляется. */
  private walk(tokens: readonly number[]): { node: RNode; matched: number; path: string[] } {
    this.clock += 1
    this.clearHits()
    let node = this.root
    let pos = 0
    const path: string[] = []
    while (pos < tokens.length) {
      const child = node.children.get(tokens[pos])
      if (!child) break
      let k = 0
      while (k < child.tokens.length && pos + k < tokens.length && child.tokens[k] === tokens[pos + k]) k++
      const target = k < child.tokens.length ? this.split(child, k) : child
      target.lastAccess = this.clock
      target.hit = true
      path.push(target.id)
      pos += k
      node = target
      if (target !== child) break // остановились посреди бывшего ребра
    }
    return { node, matched: pos, path }
  }

  /** Самый длинный закэшированный префикс последовательности (узлы пути помечаются hit и «трогаются» для LRU). */
  match(tokens: readonly number[]): PrefixMatch {
    const w = this.walk(tokens)
    return { matchedTokens: w.matched, totalTokens: tokens.length, path: w.path }
  }

  /** Вставляет последовательность: совпавший префикс переиспользуется, остаток становится новым листом. */
  insert(tokens: readonly number[]): RadixInsertResult {
    const w = this.walk(tokens)
    const rest = tokens.slice(w.matched)
    const path = w.path.slice()
    if (rest.length > 0) {
      const leaf = this.newNode(rest.slice(), w.node)
      leaf.hit = false // новый узел: промах, а не попадание
      w.node.children.set(rest[0], leaf)
      this.total += rest.length
      path.push(leaf.id)
    }
    return { matchedTokens: w.matched, insertedTokens: rest.length, totalTokens: tokens.length, path }
  }

  private resolve(target: PrefixMatch | string): RNode | null {
    if (typeof target === 'string') return target === 'root' ? null : this.nodes.get(target) ?? null
    const last = target.path[target.path.length - 1]
    return last ? this.nodes.get(last) ?? null : null
  }

  /** Блокирует путь до узла (ref +1 на узле и всех предках): активный запрос, вытеснять нельзя. */
  lock(target: PrefixMatch | string): void {
    for (let n = this.resolve(target); n && n !== this.root; n = n.parent) n.refs += 1
  }

  /** Снимает блокировку (ref −1 на узле и всех предках). */
  unlock(target: PrefixMatch | string): void {
    for (let n = this.resolve(target); n && n !== this.root; n = n.parent) n.refs = Math.max(0, n.refs - 1)
  }

  /** Токенов в узлах с активными ссылками (их вытеснить нельзя). */
  protectedTokens(): number {
    let s = 0
    for (const n of this.nodes.values()) if (n.refs > 0) s += n.tokens.length
    return s
  }

  /**
   * Вытесняет LRU-листья без ссылок, пока в дереве больше capacityTokens токенов. Лист, у которого
   * не осталось детей, сам становится кандидатом. Ничья по времени: раньше созданный узел первым.
   */
  evictLRU(capacityTokens: number): RadixEvictResult {
    const evicted: string[] = []
    let evictedTokens = 0
    while (this.total > Math.max(0, capacityTokens)) {
      let victim: RNode | null = null
      for (const n of this.nodes.values()) {
        if (n.children.size > 0 || n.refs > 0) continue
        if (!victim || n.lastAccess < victim.lastAccess || (n.lastAccess === victim.lastAccess && n.seq < victim.seq)) victim = n
      }
      if (!victim) break // всё оставшееся заблокировано активными запросами
      victim.parent!.children.delete(victim.tokens[0])
      this.nodes.delete(victim.id)
      this.total -= victim.tokens.length
      evictedTokens += victim.tokens.length
      evicted.push(victim.id)
    }
    return { evictedTokens, evictedNodes: evicted }
  }

  /** Снимок дерева для визуализации (дети по возрастанию первого токена ребра). */
  view(): RadixNodeView {
    const conv = (n: RNode): RadixNodeView => ({
      id: n.id,
      tokens: n.tokens.slice(),
      children: Array.from(n.children.values())
        .sort((a, b) => a.tokens[0] - b.tokens[0])
        .map(conv),
      refs: n.refs,
      lastAccess: n.lastAccess,
      hit: n.hit,
    })
    return conv(this.root)
  }
}
