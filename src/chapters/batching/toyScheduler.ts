/**
 * Игрушечный планировщик continuous batching в стиле vLLM V1 (глава 15, фигура «Как собирается шаг»).
 *
 * Логика как в scheduler.py vLLM V1 (research/notes-engines.md §1.5, пример §3.7): у шага есть
 * бюджет токенов (max_num_batched_tokens). Сначала RUNNING по порядку: идущий decode берёт 1 токен,
 * недосчитанный prefill берёт остаток промпта, урезанный бюджетом (chunked prefill). Потом WAITING по
 * FCFS, пока есть бюджет и бегущих меньше max_num_seqs. Токен сэмплируется, только когда запрос
 * «догнал» свой промпт. Числа игрушечные: шаг измеряется не в миллисекундах, а в токенах.
 */

export interface ToyRequest {
  id: string
  /** Номер шага (с 1), на котором запрос появляется в очереди WAITING. */
  arrive: number
  prompt: number
  output: number
}

export type ToyKind = 'prefill' | 'decode'

export interface ToyAlloc {
  id: string
  kind: ToyKind
  tokens: number
  /** Prefill этого запроса закончился в этом шаге (будет первый токен). */
  completes: boolean
  /** Кусок промпта меньше остатка: сработал chunked prefill. */
  chunked: boolean
  /** Запрос допущен из WAITING в этом шаге. */
  admitted: boolean
}

export interface ToyEmit {
  id: string
  /** Номер выходного токена (с 1). */
  index: number
  first: boolean
}

export interface ToyStep {
  step: number
  budget: number
  used: number
  allocs: ToyAlloc[]
  arrived: string[]
  runningBefore: string[]
  waitingBefore: string[]
  emitted: ToyEmit[]
  finished: string[]
  runningAfter: string[]
  waitingAfter: string[]
  /** Ждут в очереди, хотя бюджет ещё есть: упёрлись в max_num_seqs. */
  blockedBySeqs: string[]
  /** Ждут, потому что бюджет шага кончился. */
  blockedByBudget: string[]
}

export interface ToyOptions {
  budget: number
  maxSeqs: number
  /** Предохранитель от бесконечного цикла. */
  maxSteps?: number
}

interface State {
  req: ToyRequest
  computed: number
  generated: number
}

/** Прогон игрушечного планировщика до конца всех запросов (или maxSteps шагов). */
export function toySchedule(reqs: readonly ToyRequest[], opts: ToyOptions): ToyStep[] {
  const budgetMax = Math.max(1, Math.floor(opts.budget))
  const maxSeqs = Math.max(1, Math.floor(opts.maxSeqs))
  const maxSteps = opts.maxSteps ?? 40
  const pending = reqs.map((r) => ({ req: { ...r, prompt: Math.max(1, r.prompt), output: Math.max(1, r.output) }, computed: 0, generated: 0 }))
  const waiting: State[] = []
  let running: State[] = []
  const steps: ToyStep[] = []
  let done = 0

  for (let s = 1; s <= maxSteps && done < pending.length; s++) {
    const arrived: string[] = []
    for (const st of pending) {
      if (st.req.arrive === s) {
        waiting.push(st)
        arrived.push(st.req.id)
      }
    }
    const runningBefore = running.map((r) => r.req.id)
    const waitingBefore = waiting.map((r) => r.req.id)
    let budget = budgetMax
    const allocs: ToyAlloc[] = []

    // 1. RUNNING по порядку: decode по 1 токену, хвосты prefill кусками.
    for (const st of running) {
      if (budget <= 0) break
      if (st.computed < st.req.prompt) {
        const need = st.req.prompt - st.computed
        const n = Math.min(need, budget)
        allocs.push({ id: st.req.id, kind: 'prefill', tokens: n, completes: n === need, chunked: n < need, admitted: false })
        budget -= n
      } else {
        allocs.push({ id: st.req.id, kind: 'decode', tokens: 1, completes: false, chunked: false, admitted: false })
        budget -= 1
      }
    }

    // 2. WAITING по FCFS, пока есть бюджет и место по max_num_seqs.
    while (waiting.length > 0 && running.length < maxSeqs && budget > 0) {
      const st = waiting.shift()!
      const need = st.req.prompt - st.computed
      const n = Math.min(need, budget)
      allocs.push({ id: st.req.id, kind: 'prefill', tokens: n, completes: n === need, chunked: n < need, admitted: true })
      budget -= n
      running.push(st)
    }
    const blockedBySeqs = running.length >= maxSeqs && budget > 0 ? waiting.map((r) => r.req.id) : []
    const blockedByBudget = budget <= 0 ? waiting.map((r) => r.req.id) : []

    // 3. Итог шага: кто догнал промпт, кто выдал токен, кто закончил.
    const emitted: ToyEmit[] = []
    const finished: string[] = []
    const byId = new Map(running.map((r) => [r.req.id, r]))
    for (const a of allocs) {
      const st = byId.get(a.id)!
      if (a.kind === 'prefill') {
        st.computed += a.tokens
        if (st.computed >= st.req.prompt) {
          st.generated += 1
          emitted.push({ id: a.id, index: st.generated, first: true })
        }
      } else {
        st.generated += 1
        emitted.push({ id: a.id, index: st.generated, first: false })
      }
      if (st.generated >= st.req.output && st.computed >= st.req.prompt) finished.push(a.id)
    }
    if (finished.length) {
      const gone = new Set(finished)
      running = running.filter((r) => !gone.has(r.req.id))
      done += finished.length
    }

    steps.push({
      step: s,
      budget: budgetMax,
      used: budgetMax - budget,
      allocs,
      arrived,
      runningBefore,
      waitingBefore,
      emitted,
      finished,
      runningAfter: running.map((r) => r.req.id),
      waitingAfter: waiting.map((r) => r.req.id),
      blockedBySeqs,
      blockedByBudget,
    })
  }
  return steps
}

/** Набор запросов фигуры: A и B с самого начала, C на втором шаге, D и E на третьем. */
export const TOY_REQUESTS: readonly ToyRequest[] = [
  { id: 'A', arrive: 1, prompt: 6, output: 4 },
  { id: 'B', arrive: 1, prompt: 9, output: 3 },
  { id: 'C', arrive: 2, prompt: 3, output: 2 },
  { id: 'D', arrive: 3, prompt: 7, output: 3 },
  { id: 'E', arrive: 3, prompt: 5, output: 2 },
]
