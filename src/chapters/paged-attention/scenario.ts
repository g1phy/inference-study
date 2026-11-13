/**
 * Сценарий фигуры «Непрерывная бронь против блоков по требованию»: одни и те же запросы приходят,
 * генерируют токены и уходят, а два аллокатора движка (src/engine/kv.ts) раскладывают их KV в одну и ту
 * же память из 512 слотов. Очередь у обоих FIFO, как в vLLM: если первый в очереди не помещается,
 * следующие ждут за ним.
 */
import { ContiguousAllocator, PagedAllocator, type AllocatorStats } from '@/engine/kv'

export const CAPACITY = 512
export const BLOCK = 16
export const NUM_BLOCKS = CAPACITY / BLOCK

export interface ReqSpec {
  id: string
  /** Токенов промпта. */
  prompt: number
  /** max_tokens из запроса: столько старая система бронирует под ответ. */
  maxTokens: number
  /** Сколько токенов запрос сгенерирует на самом деле, прежде чем закончить. */
  output: number
}

export const REQS: Record<string, ReqSpec> = {
  A: { id: 'A', prompt: 40, maxTokens: 120, output: 32 },
  B: { id: 'B', prompt: 64, maxTokens: 128, output: 48 },
  C: { id: 'C', prompt: 24, maxTokens: 96, output: 64 },
  D: { id: 'D', prompt: 100, maxTokens: 100, output: 32 },
  E: { id: 'E', prompt: 30, maxTokens: 60, output: 48 },
}

type Action = { kind: 'arrive'; id: string } | { kind: 'decode'; tokens: number } | { kind: 'finish' }

interface StepDef {
  label: string
  actions: Action[]
}

const STEP_DEFS: StepDef[] = [
  { label: 'Память пуста', actions: [] },
  { label: 'Приходит A: промпт 40, max_tokens 120', actions: [{ kind: 'arrive', id: 'A' }] },
  { label: 'Приходит B: промпт 64, max_tokens 128', actions: [{ kind: 'arrive', id: 'B' }] },
  { label: 'Приходит C: промпт 24, max_tokens 96', actions: [{ kind: 'arrive', id: 'C' }] },
  { label: 'Decode: каждый пишет по 16 токенов', actions: [{ kind: 'decode', tokens: 16 }] },
  { label: 'Decode: ещё по 16, A закончил ответ', actions: [{ kind: 'decode', tokens: 16 }] },
  { label: 'A уходит и освобождает память', actions: [{ kind: 'finish' }] },
  { label: 'Приходит D: промпт 100, max_tokens 100', actions: [{ kind: 'arrive', id: 'D' }] },
  { label: 'Приходит E: промпт 30, max_tokens 60', actions: [{ kind: 'arrive', id: 'E' }] },
  { label: 'Decode: по 16 токенов, B закончил', actions: [{ kind: 'decode', tokens: 16 }] },
  { label: 'B уходит, очередь двигается', actions: [{ kind: 'finish' }] },
  { label: 'Decode: по 16 токенов', actions: [{ kind: 'decode', tokens: 16 }] },
  { label: 'Закончившие уходят', actions: [{ kind: 'finish' }] },
]

export type CellKind = 'prompt' | 'gen' | 'reserved' | 'waste' | 'free'

export interface Cell {
  owner: string | null
  kind: CellKind
}

export interface PanelSnapshot {
  cells: Cell[]
  /** Где рисовать букву запроса: начало брони или каждого блока. */
  labels: Array<{ slot: number; id: string }>
  running: string[]
  waiting: Array<{ id: string; reason: string }>
  finished: string[]
  stats: AllocatorStats
  /** Важные события шага (по-русски). */
  notes: string[]
  /** Шаг, на котором запрос впервые получил память (для подсчёта ожидания). */
  admittedAt: Record<string, number>
}

export interface Snapshot {
  step: number
  label: string
  contiguous: PanelSnapshot
  paged: PanelSnapshot
}

type Kind = 'contiguous' | 'paged'

class PanelSim {
  readonly kind: Kind
  readonly alloc: ContiguousAllocator | PagedAllocator
  running = new Map<string, number>() // id → сгенерировано
  waiting: string[] = []
  done: string[] = []
  finished: string[] = []
  reasons = new Map<string, string>()
  admittedAt: Record<string, number> = {}
  notes: string[] = []
  step = 0

  constructor(kind: Kind) {
    this.kind = kind
    this.alloc =
      kind === 'contiguous'
        ? new ContiguousAllocator({ capacity: CAPACITY, maxLen: 128 })
        : new PagedAllocator({ numBlocks: NUM_BLOCKS, blockSize: BLOCK, freeOrder: 'scattered' })
  }

  private allocate(r: ReqSpec): boolean {
    if (this.alloc instanceof ContiguousAllocator) return this.alloc.allocate(r.id, r.prompt, r.prompt + r.maxTokens)
    return this.alloc.allocate(r.id, r.prompt)
  }

  /** FIFO: пускаем ожидающих по порядку, пока первый не упрётся. */
  tryAdmit(): void {
    while (this.waiting.length) {
      const id = this.waiting[0]
      const r = REQS[id]
      if (!this.allocate(r)) {
        const ev = this.alloc.lastEvents.find((e) => e.kind === 'fail')
        const reason = ev?.note ?? 'Нет места.'
        if (this.reasons.get(id) !== reason) this.notes.push(`${id} ждёт. ${reason}`)
        this.reasons.set(id, reason)
        return
      }
      this.waiting.shift()
      this.reasons.delete(id)
      this.running.set(id, 0)
      this.admittedAt[id] = this.step
      if (this.alloc instanceof ContiguousAllocator) {
        this.notes.push(`${id}: забронировано ${r.prompt + r.maxTokens} слотов подряд, занято ${r.prompt}.`)
      } else {
        const table = this.alloc.blockTable(id)
        this.notes.push(`${id}: ${table.length} ${table.length === 1 ? 'блок' : table.length < 5 ? 'блока' : 'блоков'} (${table.join(', ')}).`)
      }
    }
  }

  apply(a: Action): void {
    if (a.kind === 'arrive') {
      this.waiting.push(a.id)
      this.tryAdmit()
    } else if (a.kind === 'decode') {
      for (const [id, gen] of this.running) {
        if (this.done.includes(id)) continue
        const r = REQS[id]
        const add = Math.min(a.tokens, r.output - gen)
        if (add <= 0) continue
        const before = this.alloc instanceof PagedAllocator ? this.alloc.blockTable(id).length : 0
        const ok = this.alloc.append(id, add)
        if (!ok) {
          this.notes.push(`${id}: не хватило памяти на новые токены.`)
          continue
        }
        if (this.alloc instanceof PagedAllocator) {
          const after = this.alloc.blockTable(id).length
          if (after > before) this.notes.push(`${id}: последний блок заполнился, выдан новый (${this.alloc.blockTable(id).slice(before).join(', ')}).`)
        }
        this.running.set(id, gen + add)
        if (gen + add >= r.output) this.done.push(id)
      }
    } else {
      for (const id of this.done) {
        this.alloc.free(id)
        this.running.delete(id)
        this.finished.push(id)
        this.notes.push(`${id} закончил: память свободна.`)
      }
      this.done = []
      this.tryAdmit()
    }
  }

  snapshot(): PanelSnapshot {
    const cells: Cell[] = Array.from({ length: CAPACITY }, () => ({ owner: null, kind: 'free' as CellKind }))
    const labels: Array<{ slot: number; id: string }> = []
    if (this.alloc instanceof ContiguousAllocator) {
      for (const reg of this.alloc.regions()) {
        if (!reg.seqId) continue
        const r = REQS[reg.seqId]
        labels.push({ slot: reg.start, id: reg.seqId })
        for (let i = 0; i < reg.length; i++) {
          const kind: CellKind = i < reg.used ? (i < r.prompt ? 'prompt' : 'gen') : 'reserved'
          cells[reg.start + i] = { owner: reg.seqId, kind }
        }
      }
    } else {
      for (const s of this.alloc.sequences()) {
        const r = REQS[s.id]
        s.blockTable.forEach((b, i) => {
          labels.push({ slot: b * BLOCK, id: s.id })
          for (let j = 0; j < BLOCK; j++) {
            const t = i * BLOCK + j
            const kind: CellKind = t < s.tokens ? (t < r.prompt ? 'prompt' : 'gen') : 'waste'
            cells[b * BLOCK + j] = { owner: s.id, kind }
          }
        })
      }
    }
    return {
      cells,
      labels,
      running: [...this.running.keys()],
      waiting: this.waiting.map((id) => ({ id, reason: this.reasons.get(id) ?? '' })),
      finished: [...this.finished],
      stats: this.alloc.stats(),
      notes: this.notes.slice(),
      admittedAt: { ...this.admittedAt },
    }
  }
}

/** Все шаги сценария (считается один раз). */
export function buildScenario(): Snapshot[] {
  const c = new PanelSim('contiguous')
  const p = new PanelSim('paged')
  return STEP_DEFS.map((def, step) => {
    for (const sim of [c, p]) {
      sim.step = step
      sim.notes = []
      for (const a of def.actions) sim.apply(a)
    }
    return { step, label: def.label, contiguous: c.snapshot(), paged: p.snapshot() }
  })
}

/** Средняя утилизация выделенного (used / allocated) по шагам, где что-то выделено. */
export function meanUtilization(snaps: Snapshot[], panel: Kind): number {
  const vals = snaps.map((s) => s[panel].stats).filter((st) => st.allocatedSlots > 0)
  return vals.length ? vals.reduce((a, st) => a + st.utilization, 0) / vals.length : 0
}
