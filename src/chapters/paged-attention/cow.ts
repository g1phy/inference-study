/**
 * Сценарий фигуры «Таблица блоков и copy-on-write»: один промпт, два варианта ответа (parallel sampling, n = 2).
 * Ведёт настоящий PagedAllocator движка: fork делит блоки (ref count), запись в общий неполный блок
 * вызывает copy-on-write, освобождение снимает ссылку. Тексты токенов приходят снаружи (настоящий токенизатор).
 */
import { PagedAllocator } from '@/engine/kv'

export const COW_BLOCK = 4
export const COW_BLOCKS = 12

export interface CowSeqView {
  id: string
  label: string
  table: number[]
  tokens: string[]
  /** Сколько первых токенов это промпт (дальше токены ответа). */
  promptLen: number
}

export type SlotKind = 'prompt' | 'answer'

export interface CowBlockView {
  id: number
  ref: number
  /** Текст токена в каждом слоте (null, если слот пуст). */
  slots: Array<string | null>
  /** Чей токен в слоте: промпта или ответа (null, если слот пуст). */
  kinds: Array<SlotKind | null>
  owners: string[]
  /** Блок затронут на этом шаге. */
  touched: boolean
  /** Этот блок копия (copy-on-write) блока from. */
  copyOf?: number
}

export interface CowStep {
  label: string
  /** Пояснение аллокатора по-русски (события последней операции). */
  notes: string[]
  seqs: CowSeqView[]
  blocks: CowBlockView[]
}

interface SeqState {
  label: string
  tokens: string[]
}

/** Сколько токенов промпта берёт сценарий: два полных блока и 2 токена в третьем. Подписи шагов написаны под это число. */
export const COW_PROMPT = 10

/**
 * prompt: тексты токенов промпта (нужно не меньше COW_PROMPT, берутся первые COW_PROMPT), ans1/ans2: продолжения
 * двух ответов (по 3 токена). Подписи шагов верны именно для COW_PROMPT токенов промпта: короче промпт фигура не передаёт.
 */
export function buildCow(prompt: string[], ans1: string[], ans2: string[]): CowStep[] {
  const P = prompt.slice(0, COW_PROMPT)
  const a1 = ans1.slice(0, 3)
  const a2 = ans2.slice(0, 3)
  const alloc = new PagedAllocator({ numBlocks: COW_BLOCKS, blockSize: COW_BLOCK, freeOrder: 'scattered' })
  const seqs = new Map<string, SeqState>()
  const steps: CowStep[] = []
  let cow: { from: number; to: number } | null = null

  const snap = (label: string, notes: string[], touched: number[]) => {
    const phys = alloc.physicalBlocks()
    const slots: Array<Array<string | null>> = phys.map(() => Array<string | null>(COW_BLOCK).fill(null))
    const kinds: Array<Array<SlotKind | null>> = phys.map(() => Array<SlotKind | null>(COW_BLOCK).fill(null))
    const views: CowSeqView[] = []
    for (const s of alloc.sequences()) {
      const st = seqs.get(s.id)!
      st.tokens.slice(0, s.tokens).forEach((t, pos) => {
        const b = s.blockTable[Math.floor(pos / COW_BLOCK)]
        slots[b][pos % COW_BLOCK] = t
        kinds[b][pos % COW_BLOCK] = pos < P.length ? 'prompt' : 'answer'
      })
      views.push({ id: s.id, label: st.label, table: s.blockTable, tokens: st.tokens.slice(0, s.tokens), promptLen: Math.min(P.length, s.tokens) })
    }
    steps.push({
      label,
      notes,
      seqs: views,
      blocks: phys.map((b) => ({
        id: b.id,
        ref: b.refCount,
        slots: slots[b.id],
        kinds: kinds[b.id],
        owners: b.owners.map((o) => seqs.get(o)?.label ?? o),
        touched: touched.includes(b.id),
        copyOf: cow && cow.to === b.id ? cow.from : undefined,
      })),
    })
  }

  const write = (id: string, token: string): { notes: string[]; touched: number[] } => {
    seqs.get(id)!.tokens.push(token)
    alloc.append(id, 1)
    const notes: string[] = []
    const touched: number[] = []
    for (const e of alloc.lastEvents) {
      if (e.kind === 'append') continue
      notes.push(e.note)
      touched.push(...e.blocks)
      if (e.kind === 'cow') cow = { from: e.blocks[0], to: e.blocks[1] }
    }
    const table = alloc.blockTable(id)
    touched.push(table[table.length - 1])
    return { notes, touched }
  }

  snap(`Пул из ${COW_BLOCKS} физических блоков по ${COW_BLOCK} слота`, ['Все блоки свободны.'], [])

  seqs.set('s1', { label: 'Ответ 1', tokens: P.slice() })
  alloc.allocate('s1', P.length)
  snap(`Prefill: промпт из ${P.length} токенов занял ${alloc.blockTable('s1').length} блока`, alloc.lastEvents.map((e) => e.note), alloc.blockTable('s1'))

  seqs.set('s2', { label: 'Ответ 2', tokens: P.slice() })
  alloc.fork('s1', 's2')
  snap('Нужно два ответа (n = 2): второй делит все блоки промпта', ['Блоки не копируются: у каждого ref count стал 2.'], alloc.blockTable('s1'))

  {
    const r = write('s1', a1[0])
    snap('Ответ 1 пишет первый токен: последний блок общий, копия при записи', r.notes, r.touched)
  }
  {
    const r = write('s2', a2[0])
    snap('Ответ 2 пишет первый токен прямо в свой блок', r.notes.length ? r.notes : ['Блок теперь только у ответа 2 (ref = 1), копировать нечего.'], r.touched)
  }
  cow = null
  {
    const r1 = write('s1', a1[1])
    const r2 = write('s2', a2[1])
    snap('Оба пишут ещё по токену: последние блоки заполнены', [...r1.notes, ...r2.notes, 'Блоки заполнены до конца.'], [...r1.touched, ...r2.touched])
  }
  {
    const r1 = write('s1', a1[2])
    const r2 = write('s2', a2[2])
    snap('Ещё по токену: каждому нужен новый блок', [...r1.notes, ...r2.notes], [...r1.touched, ...r2.touched])
  }
  {
    const table = alloc.blockTable('s1')
    alloc.free('s1')
    const notes = alloc.lastEvents.map((e) => e.note)
    seqs.delete('s1')
    snap('Ответ 1 закончен: его личные блоки свободны, общие остались у ответа 2', notes, table)
  }
  return steps
}

/** Физический слот позиции pos: table[pos / B] · B + pos mod B (slot_mapping). */
export function slotOf(table: number[], pos: number): number {
  return table[Math.floor(pos / COW_BLOCK)] * COW_BLOCK + (pos % COW_BLOCK)
}
