/**
 * Разбор результатов sched.ts для таймлайнов глав 15 и 16: отрезки жизни запроса
 * (ожидание, prefill, decode, простой слота), состояние в момент t, ITL во времени.
 * Чистые функции без React.
 */
import type { SimRequestStats, SimResultExt, SimStepRecord } from '@/engine/sched'

export type SegKind = 'wait' | 'prefill' | 'decode' | 'idle'

export interface Segment {
  kind: SegKind
  t0: number
  t1: number
  /** Для prefill: сколько токенов промпта посчитано в этом шаге. */
  tokens?: number
  /** Static batching: до скольких токенов дополнен промпт (длина самого длинного в batch), если больше tokens. */
  padTo?: number
  /** Индекс шага в res.steps. */
  step?: number
}

export interface RequestTrack {
  id: string
  stats: SimRequestStats
  segments: Segment[]
  /** Static batching: номер batch (с 0) и момент, когда batch закончился целиком. */
  batch?: number
  batchEnd?: number
}

const EPS = 1e-6

/** Конец шага. */
export function stepEnd(s: SimStepRecord): number {
  return s.tMs + s.durationMs
}

/**
 * Отрезки жизни каждого запроса. Для static batching (opts.static) добавляется простой слота:
 * запрос уже закончил, а batch ещё идёт, и слот нельзя отдать другому.
 */
export function requestTracks(res: SimResultExt, opts: { static?: boolean } = {}): RequestTrack[] {
  const byId = new Map<string, Segment[]>()
  const batchOf = new Map<string, number>()
  let batch = -1
  res.steps.forEach((s, i) => {
    const t1 = stepEnd(s)
    if (opts.static && s.prefill.length > 0 && s.decode.length === 0) batch += 1
    // static batching дополняет промпты до самого длинного в batch (simulateStatic, padPrompts по умолчанию)
    const longest = opts.static && s.prefill.length > 0 ? Math.max(...s.prefill.map((p) => p.tokens)) : 0
    for (const p of s.prefill) {
      if (!byId.has(p.reqId)) byId.set(p.reqId, [])
      byId.get(p.reqId)!.push({ kind: 'prefill', t0: s.tMs, t1, tokens: p.tokens, step: i, padTo: longest > p.tokens ? longest : undefined })
      if (opts.static && !batchOf.has(p.reqId)) batchOf.set(p.reqId, batch)
    }
    for (const id of s.decode) {
      if (!byId.has(id)) byId.set(id, [])
      byId.get(id)!.push({ kind: 'decode', t0: s.tMs, t1, step: i })
    }
  })

  const batchEnd = new Map<number, number>()
  if (opts.static) {
    for (const r of res.requests) {
      const b = batchOf.get(r.id)
      if (b === undefined) continue
      batchEnd.set(b, Math.max(batchEnd.get(b) ?? -Infinity, r.finishMs))
    }
  }

  return res.requests
    .slice()
    .sort((a, b) => a.arrivalMs - b.arrivalMs || a.id.localeCompare(b.id))
    .map((r) => {
      const work = (byId.get(r.id) ?? []).sort((a, b) => a.t0 - b.t0)
      const segments: Segment[] = []
      const start = work.length ? work[0].t0 : r.finishMs
      if (start > r.arrivalMs + EPS) segments.push({ kind: 'wait', t0: r.arrivalMs, t1: start })
      segments.push(...work)
      const track: RequestTrack = { id: r.id, stats: r, segments }
      if (opts.static) {
        const b = batchOf.get(r.id)
        if (b !== undefined) {
          const end = batchEnd.get(b)!
          track.batch = b
          track.batchEnd = end
          if (end > r.finishMs + EPS) segments.push({ kind: 'idle', t0: r.finishMs, t1: end })
        }
      }
      return track
    })
}

export type Phase = 'future' | 'wait' | 'prefill' | 'decode' | 'idle' | 'done'

/** Что делает запрос в момент t. */
export function phaseAt(track: RequestTrack, t: number): Phase {
  if (t < track.stats.arrivalMs) return 'future'
  for (const s of track.segments) if (t >= s.t0 && t < s.t1) return s.kind
  if (t >= track.stats.finishMs) return 'done'
  // между шагами (движок свободен, запрос в работе): считаем ожиданием следующего шага
  return 'wait'
}

export interface PhaseCounts {
  future: number
  wait: number
  work: number
  idle: number
  done: number
}

export function phaseCounts(tracks: readonly RequestTrack[], t: number): PhaseCounts {
  const c: PhaseCounts = { future: 0, wait: 0, work: 0, idle: 0, done: 0 }
  for (const tr of tracks) {
    const p = phaseAt(tr, t)
    if (p === 'prefill' || p === 'decode') c.work += 1
    else c[p] += 1
  }
  return c
}

/** Сколько токенов выдано всеми запросами к моменту t. */
export function tokensBy(tracks: readonly RequestTrack[], t: number): number {
  let n = 0
  for (const tr of tracks) {
    if (tr.stats.firstTokenMs <= t) n += 1
    for (const s of tr.segments) if (s.kind === 'decode' && s.t1 <= t && s.t1 > tr.stats.firstTokenMs + EPS) n += 1
  }
  return n
}

/** Доля слото-времени static batching, когда слот занят уже закончившим запросом. */
export function idleShare(tracks: readonly RequestTrack[]): number {
  const batches = new Map<number, { start: number; end: number; size: number }>()
  let idle = 0
  for (const tr of tracks) {
    if (tr.batch === undefined || tr.batchEnd === undefined) continue
    const first = tr.segments.find((s) => s.kind === 'prefill')
    const b = batches.get(tr.batch) ?? { start: Infinity, end: tr.batchEnd, size: 0 }
    b.start = Math.min(b.start, first ? first.t0 : tr.stats.firstTokenMs)
    b.size += 1
    batches.set(tr.batch, b)
    for (const s of tr.segments) if (s.kind === 'idle') idle += s.t1 - s.t0
  }
  let total = 0
  for (const b of batches.values()) total += b.size * (b.end - b.start)
  return total > 0 ? idle / total : 0
}

export interface ItlPoint {
  /** Момент выдачи токена (конец шага). */
  t: number
  /** Максимальный ITL среди запросов, получивших токен в этом шаге. */
  itl: number
  /** Сколько запросов получили токен. */
  count: number
  step: number
}

/**
 * ITL во времени: для каждого шага с decode-токенами берём разрыв между этим токеном и предыдущим
 * токеном того же запроса (максимум по запросам шага). filter оставляет только нужные запросы.
 */
export function itlTimeline(res: SimResultExt, filter?: (id: string) => boolean): ItlPoint[] {
  const last = new Map<string, number>()
  for (const r of res.requests) if (r.firstTokenMs >= 0) last.set(r.id, r.firstTokenMs)
  const order = res.steps.map((s, i) => ({ s, i, end: stepEnd(s) })).sort((a, b) => a.end - b.end || a.i - b.i)
  const out: ItlPoint[] = []
  for (const { s, i, end } of order) {
    let worst = -1
    let count = 0
    for (const id of s.decode) {
      if (filter && !filter(id)) continue
      const prev = last.get(id)
      if (prev === undefined) continue
      if (end <= prev + EPS) continue // шаг, в котором вышел первый токен (disagg: первый decode-шаг)
      worst = Math.max(worst, end - prev)
      count += 1
      last.set(id, end)
    }
    if (count > 0) out.push({ t: end, itl: worst, count, step: i })
  }
  return out
}

/** Среднее. */
export function avg(values: readonly number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0
}
