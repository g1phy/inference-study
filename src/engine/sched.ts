/**
 * Дискретно-событийные симуляции сервинга на основе perf.ts (агент A2, «engine-sim»):
 * static batching, continuous batching в стиле vLLM V1 (бюджет токенов, chunked prefill,
 * смешанные шаги), агрегированный и дезагрегированный serving с передачей KV, очередь для
 * хвостов латентности. Время каждого шага = perf.stepTime(prefill-куски + decode-последовательности).
 *
 * Источники: research/notes-engines.md §1.5–1.6 (scheduler V1, preemption), §3.7 (игрушка с
 * бюджетом 10); notes-foundations-production.md §1.11–1.12 (TTFT/ITL/TPOT, перцентили), §3.3, §3.5;
 * notes-numbers.md §3.20 (chunked prefill и всплески ITL); notes-dynamo.md §1.4, §3.8 (disagg, KV transfer);
 * SPEC §8.5 (в vLLM disagg первый токен сэмплирует decode-инстанс).
 */
import type { GpuSpec, InterconnectSpec, ModelSpec, RequestStats, SimRequest, SimResult, StepRecord } from './types'
import type { GpuSpecExt } from './data/gpus'
import { exponential, logNormal, mulberry32, normal, pickWeighted, type Rng } from './rng'
import { DEFAULT_BLOCK_SIZE } from './kv'
import { decodeStepMs, kvBytesForSeq, kvCapacityTokens, prefillMs, stepTime, transferMs, type PerfOptions } from './perf'

/* ============================================================================
 * Статистика: перцентили, гистограмма, goodput
 * ========================================================================== */

/**
 * Квантиль q ∈ [0, 1]. 'linear' — линейная интерполяция (numpy по умолчанию, h = (n − 1)·q),
 * 'nearest-rank' — k = ceil(q·n). Пустой массив даёт 0. Пример на 20 замерах:
 * notes-foundations-production.md §3.3 (p90 = 468 линейно, 420 по nearest-rank).
 */
export function quantile(values: readonly number[], q: number, method: 'linear' | 'nearest-rank' = 'linear'): number {
  if (values.length === 0) return 0
  return quantileSorted(sortedValues(values), q, method)
}

/** Копия значений, отсортированная по возрастанию (Float64Array: числовая сортировка без компаратора, заметно быстрее на длинных массивах ITL). */
export function sortedValues(values: readonly number[]): Float64Array {
  return Float64Array.from(values).sort()
}

/** Квантиль по уже отсортированному массиву (несколько перцентилей одной выборки без повторной сортировки). */
export function quantileSorted(s: ArrayLike<number>, q: number, method: 'linear' | 'nearest-rank' = 'linear'): number {
  if (s.length === 0) return 0
  const qq = Math.min(1, Math.max(0, q))
  if (method === 'nearest-rank') return s[Math.max(1, Math.ceil(qq * s.length)) - 1]
  const h = (s.length - 1) * qq
  const lo = Math.floor(h)
  const hi = Math.ceil(h)
  return s[lo] + (h - lo) * (s[hi] - s[lo])
}

/** Перцентиль p ∈ [0, 100] (p99 → 99). */
export function percentile(values: readonly number[], p: number, method: 'linear' | 'nearest-rank' = 'linear'): number {
  return quantile(values, p / 100, method)
}

/** Среднее (0 у пустого массива). */
export function mean(values: readonly number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0
}

/** Столбец гистограммы. */
export interface HistogramBin {
  x0: number
  x1: number
  count: number
}

/** Гистограмма с равными бинами (для графика TTFT под нагрузкой, SPEC §6.10). */
export function histogram(values: readonly number[], opts: { bins?: number; min?: number; max?: number } = {}): HistogramBin[] {
  const bins = Math.max(1, Math.floor(opts.bins ?? 20))
  if (values.length === 0) return []
  let vmin = Infinity
  let vmax = -Infinity
  for (const v of values) {
    if (v < vmin) vmin = v
    if (v > vmax) vmax = v
  }
  const lo = opts.min ?? vmin
  const hi = opts.max ?? vmax
  const width = hi > lo ? (hi - lo) / bins : 1
  const out: HistogramBin[] = Array.from({ length: bins }, (_, i) => ({ x0: lo + i * width, x1: lo + (i + 1) * width, count: 0 }))
  for (const v of values) {
    if (v < lo || v > hi) continue
    const i = Math.min(bins - 1, Math.floor((v - lo) / width))
    out[i].count += 1
  }
  return out
}

/** SLO для goodput. */
export interface Slo {
  /** Порог TTFT, мс. */
  ttftMs: number
  /** Порог TPOT (средний межтокенный интервал запроса), мс. */
  tpotMs: number
}

/**
 * Goodput: доля и интенсивность запросов, уложившихся в SLO по TTFT и TPOT
 * (DistServe; notes-foundations-production.md §2.4 F6).
 */
export function goodput(result: SimResult, slo: Slo): { fraction: number; reqPerSec: number; ok: number } {
  const reqs = result.requests
  if (reqs.length === 0) return { fraction: 0, reqPerSec: 0, ok: 0 }
  let ok = 0
  for (const r of reqs) {
    const tpot = r.itlMs.length ? mean(r.itlMs) : 0
    if (r.ttftMs <= slo.ttftMs && tpot <= slo.tpotMs) ok++
  }
  const span = result.summary.makespanMs
  return { fraction: ok / reqs.length, reqPerSec: span > 0 ? (ok * 1000) / span : 0, ok }
}

/* ============================================================================
 * Генератор нагрузки
 * ========================================================================== */

/** Распределение длины (токены). Число означает константу. */
export type Dist =
  | number
  | { kind: 'const'; value: number }
  | { kind: 'uniform'; min: number; max: number }
  | { kind: 'normal'; mean: number; sd: number; min?: number; max?: number }
  | { kind: 'lognormal'; median: number; sigma: number; min?: number; max?: number }
  | { kind: 'choice'; values: number[]; weights?: number[] }

/** Сэмпл из распределения длины: целое ≥ 1 (с учётом min/max). */
export function sampleDist(d: Dist, rng: Rng): number {
  let x: number
  let lo = 1
  let hi = Infinity
  if (typeof d === 'number') x = d
  else if (d.kind === 'const') x = d.value
  else if (d.kind === 'uniform') x = d.min + rng() * (d.max - d.min + 1)
  else if (d.kind === 'normal') {
    x = normal(rng, d.mean, d.sd)
    lo = d.min ?? 1
    hi = d.max ?? Infinity
  } else if (d.kind === 'lognormal') {
    x = logNormal(rng, d.median, d.sigma)
    lo = d.min ?? 1
    hi = d.max ?? Infinity
  } else {
    const i = d.weights ? pickWeighted(rng, d.weights) : Math.floor(rng() * d.values.length)
    x = d.values[i]
  }
  return Math.max(Math.max(1, lo), Math.min(hi, Math.floor(x)))
}

/** Параметры makeWorkload. */
export interface WorkloadOptions {
  /** Интенсивность пуассоновского потока, запросов в секунду. 0 или Infinity: все приходят в startMs. */
  rate: number
  promptDist: Dist
  outputDist: Dist
  seed?: number
  /** Сколько запросов (по умолчанию 64, если не задан durationMs). */
  count?: number
  /** Или генерировать, пока время прихода < durationMs. */
  durationMs?: number
  startMs?: number
  /**
   * Общие префиксы (системные промпты): каждый запрос начинается с одного из `count` префиксов длиной
   * `tokens`; если префикс уже встречался, его полные блоки считаются лежащими в prefix cache
   * (cachedTokens = floor(tokens / B)·B, B = blockSize, по умолчанию 16).
   */
  sharedPrefixes?: { count: number; tokens: number; blockSize?: number; weights?: number[] }
  idPrefix?: string
}

/**
 * Генератор нагрузки: пуассоновские приходы с заданной интенсивностью, длины промпта и ответа
 * из распределений, детерминированно по seed. Возвращает запросы по времени прихода.
 */
export function makeWorkload(opts: WorkloadOptions): SimRequest[] {
  const rng = mulberry32(opts.seed ?? 1)
  const instant = !(opts.rate > 0) || !Number.isFinite(opts.rate)
  // При мгновенном приходе время не растёт, поэтому durationMs поток не ограничит: берём count или 64.
  const count = opts.count ?? (opts.durationMs && !instant ? Infinity : 64)
  const limitMs = opts.durationMs ?? Infinity
  const prefix = opts.sharedPrefixes
  const seen = new Set<number>()
  const out: SimRequest[] = []
  let t = opts.startMs ?? 0
  for (let i = 0; i < count; i++) {
    if (!instant) t += exponential(rng, opts.rate / 1000)
    if (t >= (opts.startMs ?? 0) + limitMs) break
    let prompt = sampleDist(opts.promptDist, rng)
    const output = sampleDist(opts.outputDist, rng)
    let cached = 0
    if (prefix && prefix.count > 0 && prefix.tokens > 0) {
      const p = prefix.weights ? pickWeighted(rng, prefix.weights) : Math.floor(rng() * prefix.count)
      prompt += prefix.tokens
      const B = prefix.blockSize ?? DEFAULT_BLOCK_SIZE
      if (seen.has(p)) cached = Math.floor(prefix.tokens / B) * B
      seen.add(p)
    }
    const r: SimRequest = { id: `${opts.idPrefix ?? 'r'}${i}`, arrivalMs: t, promptTokens: prompt, outputTokens: output }
    if (cached > 0) r.cachedTokens = cached
    out.push(r)
  }
  return out
}

/* ============================================================================
 * Результаты
 * ========================================================================== */

/** RequestStats + подробности. */
export interface SimRequestStats extends RequestStats {
  promptTokens: number
  outputTokens: number
  cachedTokens: number
  /** Воркер, где шёл decode (или единственный). */
  worker?: string
  /** Prefill-воркер в дезагрегированном режиме. */
  prefillWorker?: string
  e2eMs: number
  /** Ожидание до первого шага с этим запросом, мс. */
  queueMs: number
  /** Сколько раз запрос вытесняли (preemption, recompute). */
  preemptions: number
}

/** Расширенная сводка. */
export interface SimSummaryExt {
  ttftP50: number
  ttftP99: number
  itlP50: number
  itlP99: number
  /** Выходные токены в секунду на весь сервис. */
  throughputTokS: number
  makespanMs: number
  ttftMean: number
  ttftP90: number
  itlMean: number
  itlP90: number
  e2eP50: number
  e2eP99: number
  /** Средний TPOT по запросам: (E2E − TTFT) / (N − 1). */
  tpotMean: number
  requests: number
  outputTokens: number
  gpuCount: number
  throughputPerGpuTokS: number
  preemptions: number
}

/** Передача KV prefill → decode. */
export interface KvTransferRecord {
  reqId: string
  from: string
  to: string
  startMs: number
  endMs: number
  bytes: number
}

/** StepRecord + сколько токенов KV держит движок к концу шага (для графика памяти и проверки ёмкости). */
export interface SimStepRecord extends StepRecord {
  kvTokens?: number
}

/** SimResult с подробностями. */
export interface SimResultExt extends SimResult {
  steps: SimStepRecord[]
  requests: SimRequestStats[]
  summary: SimSummaryExt
  transfers: KvTransferRecord[]
  /** Короткая подпись конфигурации по-русски. */
  label: string
}

/** Сводка по запросам (перцентили линейной интерполяцией; ITL всех запросов сливаются в один список, как vllm bench). */
export function summarize(requests: readonly SimRequestStats[], gpuCount = 1): SimSummaryExt {
  const ttft = requests.map((r) => r.ttftMs)
  const itl = requests.flatMap((r) => r.itlMs)
  const e2e = requests.map((r) => r.e2eMs)
  const tpot = requests.filter((r) => r.outputTokens > 1).map((r) => (r.finishMs - r.firstTokenMs) / (r.outputTokens - 1))
  const outTokens = requests.reduce((a, r) => a + r.outputTokens, 0)
  let t0 = requests.length ? Infinity : 0
  let t1 = requests.length ? -Infinity : 0
  for (const r of requests) {
    if (r.arrivalMs < t0) t0 = r.arrivalMs
    if (r.finishMs > t1) t1 = r.finishMs
  }
  const span = Math.max(0, t1 - t0)
  const thr = span > 0 ? (outTokens * 1000) / span : 0
  // каждую выборку сортируем один раз: ITL у длинных ответов это сотни тысяч значений
  const sTtft = sortedValues(ttft)
  const sItl = sortedValues(itl)
  const sE2e = sortedValues(e2e)
  return {
    ttftP50: quantileSorted(sTtft, 0.5),
    ttftP99: quantileSorted(sTtft, 0.99),
    itlP50: quantileSorted(sItl, 0.5),
    itlP99: quantileSorted(sItl, 0.99),
    throughputTokS: thr,
    makespanMs: span,
    ttftMean: mean(ttft),
    ttftP90: quantileSorted(sTtft, 0.9),
    itlMean: mean(itl),
    itlP90: quantileSorted(sItl, 0.9),
    e2eP50: quantileSorted(sE2e, 0.5),
    e2eP99: quantileSorted(sE2e, 0.99),
    tpotMean: mean(tpot),
    requests: requests.length,
    outputTokens: outTokens,
    gpuCount,
    throughputPerGpuTokS: gpuCount > 0 ? thr / gpuCount : 0,
    preemptions: requests.reduce((a, r) => a + r.preemptions, 0),
  }
}

/* ============================================================================
 * Лимиты планировщика vLLM по умолчанию
 * ========================================================================== */

/**
 * Дефолты vLLM (`vllm serve`, v0.32, notes-engines.md §1.5): GPU ≥ 160 ГиБ → бюджет 16384 и
 * max_num_seqs 1024; ≥ 70 ГиБ и не A100 → 8192 и 1024; прочие (A100, L40S) → 2048 и 256.
 */
export function vllmDefaultLimits(gpu: GpuSpec & Partial<GpuSpecExt>): { tokenBudget: number; maxSeqs: number } {
  const gib = gpu.usableMiB ? gpu.usableMiB / 1024 : gpu.memGB
  if (gib >= 160) return { tokenBudget: 16384, maxSeqs: 1024 }
  if (gib >= 70 && !/a100/i.test(gpu.id)) return { tokenBudget: 8192, maxSeqs: 1024 }
  return { tokenBudget: 2048, maxSeqs: 256 }
}

/* ============================================================================
 * Ядро: движок с шагами (mixed / prefill-only / decode-only)
 * ========================================================================== */

type Role = 'mixed' | 'prefill' | 'decode'

interface ReqState {
  req: SimRequest
  prompt: number
  output: number
  cached: number
  /** Цель prefill: промпт (после preemption: промпт + сгенерированное). */
  target: number
  /** Токены цели, KV которых уже есть (вместе с кэшем). */
  computed: number
  /** Длина KV последовательности. */
  kvTokens: number
  generated: number
  firstTokenMs: number
  lastTokenMs: number
  itl: number[]
  finishMs: number
  firstScheduledMs: number
  preemptions: number
  order: number
  worker: string
  prefillWorker?: string
}

interface EngineCfg {
  model: ModelSpec
  gpu: GpuSpec
  perf: PerfOptions
  tokenBudget: number
  maxSeqs: number
  chunked: boolean
  kvCap: number
  /** Где сэмплируется первый токен в disagg: 'decode' (vLLM / Dynamo) или 'prefill' (SGLang, книга). */
  firstTokenOn: 'prefill' | 'decode'
  /** Счётчик допуска (общий для движков одного прогона): порядок для выбора жертвы preemption. */
  counter: { n: number }
}

class Engine {
  readonly id: string
  readonly role: Role
  readonly cfg: EngineCfg
  waiting: ReqState[] = []
  running: ReqState[] = []
  busy = false
  busyUntil = 0
  /** KV, которые сейчас едут на этот движок (disagg). */
  incoming = 0
  private cur: { chunks: Array<{ rs: ReqState; tokens: number }>; decodes: ReqState[] } | null = null

  constructor(id: string, role: Role, cfg: EngineCfg) {
    this.id = id
    this.role = role
    this.cfg = cfg
  }

  /** Нагрузка для роутинга: оставшиеся prefill-токены + число активных запросов. */
  load(): number {
    let s = this.incoming
    for (const r of this.waiting) s += r.target - r.computed + 1
    for (const r of this.running) s += Math.max(0, r.target - r.computed) + 1
    return s
  }

  private kvReserved(): number {
    let s = 0
    for (const r of this.running) s += Math.max(r.kvTokens, r.target)
    return s
  }

  private emit(rs: ReqState, t: number): void {
    if (rs.generated === 0) rs.firstTokenMs = t
    else rs.itl.push(t - rs.lastTokenMs)
    rs.generated += 1
    rs.lastTokenMs = t
  }

  /** Recompute-preemption (vLLM V1): KV освобождается, запрос в начало очереди и пересчитает промпт + сгенерированное. */
  private preempt(rs: ReqState): void {
    this.running = this.running.filter((r) => r !== rs)
    rs.preemptions += 1
    rs.target = rs.prompt + rs.generated
    rs.computed = Math.min(rs.cached, rs.target)
    rs.kvTokens = rs.computed
    this.waiting.unshift(rs)
  }

  /** Собирает и запускает шаг в момент t; false, если делать нечего. */
  start(t: number, steps: SimStepRecord[]): boolean {
    if (this.busy || (this.running.length === 0 && this.waiting.length === 0)) return false
    const c = this.cfg
    let budget = this.role === 'decode' ? Infinity : c.tokenBudget
    const chunks: Array<{ rs: ReqState; tokens: number }> = []
    const decodes: ReqState[] = []
    let preempted = false

    // Рост KV на decode-шаге может не влезть: вытесняем последнего допущенного. Так делают и mixed-,
    // и decode-движки (decode-инстанс vLLM в disagg — обычный движок с recompute-preemption);
    // prefill-движок KV не растит.
    if (this.role !== 'prefill') {
      const decoding = () => this.running.filter((r) => r.computed >= r.target).length
      while (this.running.length > 1 && this.kvReserved() + decoding() > c.kvCap) {
        let victim = this.running[0]
        for (const r of this.running) if (r.order > victim.order) victim = r
        this.preempt(victim)
        preempted = true
      }
    }

    // 1. Сначала RUNNING по порядку: decode по 1 токену, незаконченный prefill кусками
    //    (у decode-движка prefill бывает только как пересчёт после preemption).
    let prefillScheduled = 0
    for (const rs of this.running) {
      if (rs.computed < rs.target) {
        const need = rs.target - rs.computed
        let n = c.chunked ? Math.min(need, budget) : need
        if (!c.chunked && need > budget && prefillScheduled > 0) n = 0
        if (n <= 0) {
          if (budget <= 0) break
          continue
        }
        chunks.push({ rs, tokens: n })
        budget -= n
        prefillScheduled += n
      } else {
        if (this.role === 'prefill') continue
        if (budget < 1) break
        decodes.push(rs)
        budget -= 1
      }
    }

    // 2. Потом WAITING (FCFS), если на этом шаге никого не вытеснили. Проверка памяти учитывает и
    //    токены, которые допишут decode-шаги этого же шага (decodes.length), иначе KV вылезает за ёмкость.
    if (!preempted) {
      while (this.waiting.length > 0 && this.running.length < c.maxSeqs && budget > 0) {
        const rs = this.waiting[0]
        if (this.role === 'decode' && rs.computed >= rs.target) {
          // KV приехал с prefill-воркера: сразу decode. Вытесненный ранее запрос идёт ниже, как prefill-пересчёт.
          if (this.running.length > 0 && this.kvReserved() + decodes.length + rs.kvTokens + 1 > c.kvCap) break
          this.waiting.shift()
          rs.order = ++c.counter.n
          this.running.push(rs)
          decodes.push(rs)
          budget -= 1
          continue
        }
        if (this.running.length > 0 && this.kvReserved() + decodes.length + rs.target > c.kvCap) break
        const need = rs.target - rs.computed
        let n: number
        if (c.chunked) n = Math.min(need, budget)
        else if (need <= budget || prefillScheduled === 0) n = need // без чанков длинный промпт идёт целиком
        else break
        this.waiting.shift()
        rs.order = ++c.counter.n
        if (rs.firstScheduledMs < 0) rs.firstScheduledMs = t
        this.running.push(rs)
        if (n > 0) {
          chunks.push({ rs, tokens: n })
          budget -= n
          prefillScheduled += n
        } else if (this.role === 'mixed') {
          decodes.push(rs) // весь промпт в кэше: сразу decode
          budget -= 1
        }
      }
    }
    if (chunks.length === 0 && decodes.length === 0) return false
    for (const d of decodes) if (d.firstScheduledMs < 0) d.firstScheduledMs = t

    const work = {
      prefill: chunks.map((ch) => ({ tokens: ch.tokens, context: ch.rs.computed, last: ch.rs.computed + ch.tokens >= ch.rs.target })),
      decode: decodes.map((d) => d.kvTokens + 1),
    }
    const dur = stepTime(c.model, c.gpu, work, c.perf).ms
    let kvTokens = chunks.reduce((a, ch) => a + ch.tokens, 0) + decodes.length
    for (const r of this.running) kvTokens += r.kvTokens
    steps.push({
      tMs: t,
      durationMs: dur,
      prefill: chunks.map((ch) => ({ reqId: ch.rs.req.id, tokens: ch.tokens })),
      decode: decodes.map((d) => d.req.id),
      worker: this.id,
      kvTokens,
    })
    this.busy = true
    this.busyUntil = t + dur
    this.cur = { chunks, decodes }
    return true
  }

  /** Завершает текущий шаг в момент t. */
  finish(t: number): { prefillDone: ReqState[]; finished: ReqState[] } {
    const prefillDone: ReqState[] = []
    const finished: ReqState[] = []
    const cur = this.cur
    this.busy = false
    this.cur = null
    if (!cur) return { prefillDone, finished }
    const emitted: ReqState[] = []
    for (const { rs, tokens } of cur.chunks) {
      rs.computed += tokens
      rs.kvTokens += tokens
      if (rs.computed >= rs.target) {
        if (this.role === 'prefill') {
          if (this.cfg.firstTokenOn === 'prefill' && rs.generated === 0) this.emit(rs, t)
          if (rs.generated >= rs.output) finished.push(rs)
          else prefillDone.push(rs)
        } else {
          this.emit(rs, t)
          emitted.push(rs)
        }
      }
    }
    for (const rs of cur.decodes) {
      rs.kvTokens += 1
      this.emit(rs, t)
      emitted.push(rs)
    }
    for (const rs of emitted) if (rs.generated >= rs.output) finished.push(rs)
    if (prefillDone.length || finished.length) {
      const gone = new Set<ReqState>([...prefillDone, ...finished])
      this.running = this.running.filter((r) => !gone.has(r))
      for (const rs of finished) rs.finishMs = t
    }
    return { prefillDone, finished }
  }
}

/** Новое состояние запроса. */
function newState(r: SimRequest): ReqState {
  const prompt = Math.max(1, Math.floor(r.promptTokens))
  const cached = Math.min(Math.max(0, Math.floor(r.cachedTokens ?? 0)), prompt - 1) // хотя бы 1 токен считается
  return {
    req: r,
    prompt,
    output: Math.max(1, Math.floor(r.outputTokens)),
    cached,
    target: prompt,
    computed: cached,
    kvTokens: cached,
    generated: 0,
    firstTokenMs: -1,
    lastTokenMs: -1,
    itl: [],
    finishMs: -1,
    firstScheduledMs: -1,
    preemptions: 0,
    order: 0,
    worker: '',
  }
}

/** Итоговая статистика запроса. */
function toStats(s: ReqState): SimRequestStats {
  return {
    id: s.req.id,
    arrivalMs: s.req.arrivalMs,
    firstTokenMs: s.firstTokenMs,
    finishMs: s.finishMs,
    ttftMs: s.firstTokenMs - s.req.arrivalMs,
    itlMs: s.itl,
    promptTokens: s.prompt,
    outputTokens: s.output,
    cachedTokens: s.cached,
    worker: s.worker,
    prefillWorker: s.prefillWorker,
    e2eMs: s.finishMs - s.req.arrivalMs,
    queueMs: Math.max(0, s.firstScheduledMs - s.req.arrivalMs),
    preemptions: s.preemptions,
  }
}

interface LoopCfg {
  /** Куда отправить новый запрос. */
  route: (rs: ReqState) => Engine
  /** Prefill закончен на prefill-движке: передать KV (возвращает время прибытия и decode-движок). */
  handoff?: (rs: ReqState, from: Engine, t: number) => { at: number; to: Engine }
}

/** Общий событийный цикл по нескольким движкам. */
function runLoop(reqs: readonly SimRequest[], engines: Engine[], loop: LoopCfg): { states: ReqState[]; steps: SimStepRecord[] } {
  const states = reqs
    .slice()
    .sort((a, b) => a.arrivalMs - b.arrivalMs)
    .map(newState)
  const steps: SimStepRecord[] = []
  const transfers: Array<{ at: number; rs: ReqState; to: Engine }> = []
  let ai = 0
  let t = 0
  let guard = 0
  for (;;) {
    if (++guard > 2_000_000) throw new Error('sched: слишком много событий (проверьте параметры)')
    let tNext = Infinity
    if (ai < states.length) tNext = Math.min(tNext, states[ai].req.arrivalMs)
    for (const e of engines) if (e.busy) tNext = Math.min(tNext, e.busyUntil)
    for (const tr of transfers) tNext = Math.min(tNext, tr.at)
    if (tNext === Infinity) break
    t = Math.max(t, tNext)
    for (const e of engines) {
      if (!e.busy || e.busyUntil > t) continue
      const res = e.finish(t)
      if (loop.handoff) for (const rs of res.prefillDone) transfers.push({ ...loop.handoff(rs, e, t), rs })
    }
    for (let i = transfers.length - 1; i >= 0; i--) {
      if (transfers[i].at <= t) {
        const tr = transfers[i]
        transfers.splice(i, 1)
        tr.rs.worker = tr.to.id
        tr.to.incoming = Math.max(0, tr.to.incoming - 1)
        tr.to.waiting.push(tr.rs)
      }
    }
    while (ai < states.length && states[ai].req.arrivalMs <= t) {
      const rs = states[ai++]
      const e = loop.route(rs)
      rs.worker = e.id
      e.waiting.push(rs)
    }
    for (const e of engines) if (!e.busy) e.start(t, steps)
  }
  return { states, steps }
}

/** Движок с наименьшей нагрузкой (ничья: первый). */
function leastLoaded(engines: readonly Engine[]): Engine {
  let best = engines[0]
  for (const e of engines) if (e.load() < best.load()) best = e
  return best
}

/** Общие параметры симуляций. */
export interface SimBaseOptions extends PerfOptions {
  model: ModelSpec
  gpu: GpuSpec
  /** Ёмкость KV одного движка в токенах (по умолчанию из памяти GPU: perf.kvCapacityTokens). */
  kvCapacityTokens?: number
  memUtil?: number
  reserveGiB?: number
}

function perfOf(o: SimBaseOptions): PerfOptions {
  return { dtype: o.dtype, kvDtype: o.kvDtype, mfu: o.mfu, mbu: o.mbu, gpus: o.gpus, overheadMs: o.overheadMs, dpAttention: o.dpAttention, swaAware: o.swaAware }
}

function kvCapOf(o: SimBaseOptions, gpus?: number): number {
  if (o.kvCapacityTokens !== undefined) return o.kvCapacityTokens
  const cap = kvCapacityTokens(o.model, o.gpu, { dtype: o.dtype, kvDtype: o.kvDtype, gpus: gpus ?? o.gpus, memUtil: o.memUtil, reserveGiB: o.reserveGiB, dpAttention: o.dpAttention })
  return Math.max(cap, 1)
}

/* ============================================================================
 * Continuous batching (vLLM V1) и агрегированный serving
 * ========================================================================== */

/** Параметры simulateContinuous. */
export interface ContinuousOptions extends SimBaseOptions {
  /** max_num_batched_tokens (по умолчанию как vLLM для этого GPU: 8192 на H100). */
  tokenBudget?: number
  /** max_num_seqs (по умолчанию как vLLM: 1024 на H100 / H200 / B200, 256 на A100 / L40S). */
  maxSeqs?: number
  /** Chunked prefill (в vLLM V1 включён по умолчанию). */
  chunkedPrefill?: boolean
  /** Сколько одинаковых реплик (агрегированный serving на N GPU-движках), по умолчанию 1. */
  workers?: number
}

/**
 * Continuous batching как в vLLM V1: на каждом шаге сначала RUNNING (decode по токену, хвосты
 * prefill), затем WAITING по FCFS в остаток бюджета токенов; длинный промпт режется на куски
 * (chunked prefill), prefill и decode идут в одном смешанном шаге; первый токен появляется, когда
 * запрос «догнал» свой промпт. При нехватке KV последний допущенный вытесняется (recompute).
 * С workers > 1: несколько реплик, новый запрос идёт на наименее загруженную (агрегированный serving).
 */
export function simulateContinuous(reqs: readonly SimRequest[], opts: ContinuousOptions): SimResultExt {
  const lim = vllmDefaultLimits(opts.gpu)
  const cfg: EngineCfg = {
    model: opts.model,
    gpu: opts.gpu,
    perf: perfOf(opts),
    tokenBudget: Math.max(1, opts.tokenBudget ?? lim.tokenBudget),
    maxSeqs: Math.max(1, opts.maxSeqs ?? lim.maxSeqs),
    chunked: opts.chunkedPrefill !== false,
    kvCap: kvCapOf(opts),
    firstTokenOn: 'decode',
    counter: { n: 0 },
  }
  const n = Math.max(1, Math.floor(opts.workers ?? 1))
  const engines = Array.from({ length: n }, (_, i) => new Engine(n === 1 ? 'gpu-0' : `agg-${i}`, 'mixed', cfg))
  const { states, steps } = runLoop(reqs, engines, { route: () => leastLoaded(engines) })
  const gpus = n * Math.max(1, opts.gpus ?? 1)
  const requests = states.map(toStats)
  return {
    steps,
    requests,
    summary: summarize(requests, gpus),
    transfers: [],
    label: `Continuous batching: ${n} × движок, бюджет ${cfg.tokenBudget} ток./шаг, ${cfg.chunked ? 'chunked prefill' : 'без chunked prefill'}`,
  }
}

/* ============================================================================
 * Static batching
 * ========================================================================== */

/** Параметры simulateStatic. */
export interface StaticOptions extends SimBaseOptions {
  /** Размер batch. */
  batchSize: number
  /** Dynamic batching: стартовать не позже чем через maxWaitMs после первого запроса в очереди. */
  maxWaitMs?: number
  /** Дополнять промпты до самого длинного в batch (как классический static batching), по умолчанию true. */
  padPrompts?: boolean
}

/**
 * Static batching: ждём batchSize запросов (или maxWaitMs в режиме dynamic), делаем общий prefill,
 * затем decode-шаги, пока не закончит самый длинный ответ; закончившие раньше слоты простаивают,
 * следующий batch стартует только после конца текущего (notes-foundations-production.md §3.5).
 */
export function simulateStatic(reqs: readonly SimRequest[], opts: StaticOptions): SimResultExt {
  const B = Math.max(1, Math.floor(opts.batchSize))
  const perf = perfOf(opts)
  const pad = opts.padPrompts !== false
  const states = reqs
    .slice()
    .sort((a, b) => a.arrivalMs - b.arrivalMs)
    .map(newState)
  const steps: SimStepRecord[] = []
  let t = 0
  let i = 0
  while (i < states.length) {
    const first = states[i]
    const fullIdx = Math.min(states.length, i + B) - 1
    let startT = Math.max(t, states[fullIdx].req.arrivalMs)
    if (opts.maxWaitMs !== undefined) startT = Math.max(t, first.req.arrivalMs, Math.min(states[fullIdx].req.arrivalMs, first.req.arrivalMs + opts.maxWaitMs))
    const batch: ReqState[] = []
    while (i < states.length && batch.length < B && states[i].req.arrivalMs <= startT) batch.push(states[i++])
    if (batch.length === 0) {
      t = states[i].req.arrivalMs
      continue
    }
    for (const s of batch) {
      s.firstScheduledMs = startT
      s.worker = 'gpu-0'
    }
    // Prefill всего batch одним шагом.
    const maxPrompt = Math.max(...batch.map((s) => s.prompt - s.cached))
    const prefillWork = batch.map((s) => ({ tokens: pad ? maxPrompt : s.prompt - s.cached, context: s.cached }))
    const pDur = stepTime(opts.model, opts.gpu, { prefill: prefillWork }, perf).ms
    steps.push({ tMs: startT, durationMs: pDur, prefill: batch.map((s) => ({ reqId: s.req.id, tokens: s.prompt - s.cached })), decode: [], worker: 'gpu-0' })
    let now = startT + pDur
    for (const s of batch) {
      s.kvTokens = s.prompt
      s.firstTokenMs = now
      s.lastTokenMs = now
      s.generated = 1
      if (s.generated >= s.output) s.finishMs = now
    }
    // Decode: batch занимает все слоты, пока не закончит самый длинный.
    const maxOut = Math.max(...batch.map((s) => s.output))
    for (let k = 1; k < maxOut; k++) {
      const ctx = mean(batch.map((s) => s.prompt + Math.min(k, s.output)))
      const dur = stepTime(opts.model, opts.gpu, { decodeUniform: { batch: batch.length, context: ctx } }, perf).ms
      const active = batch.filter((s) => s.generated < s.output)
      steps.push({ tMs: now, durationMs: dur, prefill: [], decode: active.map((s) => s.req.id), worker: 'gpu-0' })
      now += dur
      for (const s of active) {
        s.itl.push(now - s.lastTokenMs)
        s.lastTokenMs = now
        s.generated += 1
        if (s.generated >= s.output) s.finishMs = now
      }
    }
    t = now
  }
  const requests = states.map(toStats)
  return {
    steps,
    requests,
    summary: summarize(requests, Math.max(1, opts.gpus ?? 1)),
    transfers: [],
    label: opts.maxWaitMs !== undefined ? `Dynamic batching: batch ${B}, ожидание ≤ ${opts.maxWaitMs} мс` : `Static batching: batch ${B}`,
  }
}

/* ============================================================================
 * Дезагрегированный serving
 * ========================================================================== */

/** Параметры simulateDisagg. */
export interface DisaggOptions extends SimBaseOptions {
  prefillWorkers: number
  decodeWorkers: number
  /** Канал передачи KV: InterconnectSpec или ГБ/с в одну сторону. */
  link: InterconnectSpec | number
  /** Латентность одной передачи, мс (по умолчанию 0). */
  linkLatencyMs?: number
  /** Бюджет токенов на шаг prefill-воркера (по умолчанию как vLLM для GPU). */
  tokenBudget?: number
  /** max_num_seqs decode-воркера. */
  maxSeqs?: number
  /** Где сэмплируется первый токен: 'decode' (vLLM / Dynamo, по умолчанию) или 'prefill' (SGLang). */
  firstTokenOn?: 'prefill' | 'decode'
}

/**
 * Дезагрегация: prefill-воркеры только считают промпты (выбирается наименее загруженный),
 * готовый KV едет по каналу link (передачи с одного воркера идут по очереди), decode-воркер
 * (наименее загруженный: в disagg decode маршрутизируется только по нагрузке) делает только
 * decode-шаги, поэтому ITL не страдает от чужих prefill. По умолчанию первый токен сэмплирует
 * decode-инстанс, как vLLM в Dynamo (SPEC §8.5); 'prefill' — вариант SGLang.
 */
export function simulateDisagg(reqs: readonly SimRequest[], opts: DisaggOptions): SimResultExt {
  const lim = vllmDefaultLimits(opts.gpu)
  const base: EngineCfg = {
    model: opts.model,
    gpu: opts.gpu,
    perf: perfOf(opts),
    tokenBudget: Math.max(1, opts.tokenBudget ?? lim.tokenBudget),
    maxSeqs: Math.max(1, opts.maxSeqs ?? lim.maxSeqs),
    chunked: true,
    kvCap: kvCapOf(opts),
    firstTokenOn: opts.firstTokenOn ?? 'decode',
    counter: { n: 0 },
  }
  const P = Math.max(1, Math.floor(opts.prefillWorkers))
  const D = Math.max(1, Math.floor(opts.decodeWorkers))
  const prefill = Array.from({ length: P }, (_, i) => new Engine(`prefill-${i}`, 'prefill', base))
  const decode = Array.from({ length: D }, (_, i) => new Engine(`decode-${i}`, 'decode', base))
  const GBs = typeof opts.link === 'number' ? opts.link : opts.link.GBs
  const linkFree = new Map<string, number>()
  const transfers: KvTransferRecord[] = []
  const { states, steps } = runLoop(reqs, [...prefill, ...decode], {
    route: (rs) => {
      const e = leastLoaded(prefill)
      rs.prefillWorker = e.id
      return e
    },
    handoff: (rs, from, t) => {
      const to = leastLoaded(decode)
      const bytes = kvBytesForSeq(opts.model, rs.prompt, opts.kvDtype ?? 'bf16', { swaAware: opts.swaAware !== false })
      const start = Math.max(t, linkFree.get(from.id) ?? 0)
      const end = start + transferMs(bytes, GBs, opts.linkLatencyMs ?? 0)
      linkFree.set(from.id, end)
      transfers.push({ reqId: rs.req.id, from: from.id, to: to.id, startMs: start, endMs: end, bytes })
      to.incoming += 1 // следующий handoff увидит, что сюда уже едет KV
      return { at: end, to }
    },
  })
  const gpus = (P + D) * Math.max(1, opts.gpus ?? 1)
  const requests = states.map(toStats)
  const linkName = typeof opts.link === 'number' ? `${opts.link} ГБ/с` : opts.link.name
  return { steps, requests, summary: summarize(requests, gpus), transfers, label: `Disagg ${P}P${D}D, KV по ${linkName}` }
}

/* ============================================================================
 * Очередь для хвостов латентности
 * ========================================================================== */

/** Параметры simulateQueue. */
export interface QueueOptions extends SimBaseOptions {
  /** Сколько реплик обслуживают очередь (FCFS, общая очередь), по умолчанию 1. */
  servers?: number
  /** Своё время обслуживания запроса, мс (по умолчанию prefill из perf.ts). */
  serviceMs?: (r: SimRequest) => number
}

/**
 * Очередь для демонстрации хвостов: c реплик берут запросы из общей FCFS-очереди, обслуживание
 * (prefill) занимает реплику, decode идёт параллельно и не блокирует prefill (упрощение).
 * TTFT = ожидание + prefill; ITL = шаг decode при текущем числе декодирующих на реплике.
 * С ростом нагрузки к ρ → 1 хвост TTFT растёт «хоккейной клюшкой» (notes-foundations-production.md §3.8).
 */
export function simulateQueue(reqs: readonly SimRequest[], opts: QueueOptions): SimResultExt & { utilization: number } {
  const c = Math.max(1, Math.floor(opts.servers ?? 1))
  const perf = perfOf(opts)
  const freeAt = new Array<number>(c).fill(0)
  const decodingUntil: number[][] = Array.from({ length: c }, () => [])
  const sorted = reqs.slice().sort((a, b) => a.arrivalMs - b.arrivalMs)
  const steps: SimStepRecord[] = []
  const requests: SimRequestStats[] = []
  let busyTotal = 0
  for (const r of sorted) {
    let k = 0
    for (let j = 1; j < c; j++) if (freeAt[j] < freeAt[k]) k = j
    const start = Math.max(r.arrivalMs, freeAt[k])
    const cached = Math.min(r.cachedTokens ?? 0, r.promptTokens - 1)
    const svc = opts.serviceMs ? opts.serviceMs(r) : prefillMs(opts.model, opts.gpu, r.promptTokens, { ...perf, cachedTokens: cached })
    const first = start + svc
    freeAt[k] = first
    busyTotal += svc
    steps.push({ tMs: start, durationMs: svc, prefill: [{ reqId: r.id, tokens: r.promptTokens - cached }], decode: [], worker: `replica-${k}` })
    const active = decodingUntil[k].filter((x) => x > first).length + 1
    const itl = decodeStepMs(opts.model, opts.gpu, { ...perf, batch: active, avgContext: r.promptTokens + r.outputTokens / 2 })
    const out = Math.max(1, r.outputTokens)
    const finish = first + itl * (out - 1)
    decodingUntil[k] = decodingUntil[k].filter((x) => x > first)
    decodingUntil[k].push(finish)
    requests.push({
      id: r.id,
      arrivalMs: r.arrivalMs,
      firstTokenMs: first,
      finishMs: finish,
      ttftMs: first - r.arrivalMs,
      itlMs: new Array(out - 1).fill(itl),
      promptTokens: r.promptTokens,
      outputTokens: out,
      cachedTokens: cached,
      worker: `replica-${k}`,
      e2eMs: finish - r.arrivalMs,
      queueMs: start - r.arrivalMs,
      preemptions: 0,
    })
  }
  const summary = summarize(requests, c * Math.max(1, opts.gpus ?? 1))
  const t0 = sorted.length ? sorted[0].arrivalMs : 0
  const t1 = sorted.length ? Math.max(...freeAt) : 0
  const utilization = t1 > t0 ? busyTotal / (c * (t1 - t0)) : 0
  return { steps, requests, summary, transfers: [], label: `Очередь FCFS: ${c} реплик(и)`, utilization }
}
