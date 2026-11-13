import { prefillMs } from '@/engine/perf'
import { exponential, mulberry32 } from '@/engine/rng'
import { quantileSorted, sampleDist, type Dist } from '@/engine/sched'
import type { Hardware } from '@/chapters/prefill-decode/shared/hw-core'

/**
 * Очередь для хвостов TTFT (глава «Метрики», фигура 2): пуассоновские приходы, одна реплика берёт
 * запросы по очереди (FCFS), TTFT = ожидание + prefill из perf.ts.
 *
 * Устойчивый режим. На каждую нагрузку один длинный прогон: первые WARMUP запросов отбрасываем
 * (очередь стартует пустой и ещё не разогналась), из следующих HORIZON берём каждый STRIDE-й, это
 * POPULATION запросов, «весь трафик сервиса». Длинный прогон нужен высоким нагрузкам: при 95-98 % очередь
 * набегает и рассасывается за десятки тысяч запросов, и 30 000 запросов подряд накрывали один-два таких
 * всплеска, а не устойчивое распределение (при 98 % среднее почти равнялось медиане, p99 при 95 % был
 * занижен). Соседние запросы загруженной очереди ждут почти одинаково, поэтому каждый десятый описывает
 * распределение так же, как все подряд (сверено: p50, p90, p99 совпадают с точностью до процента), а
 * сортировать и хранить в десять раз меньше. Выборка из n запросов берётся из POPULATION случайно, без
 * повторов, по seed из состояния фигуры: p99 маленькой выборки прыгает от выборки к выборке, а большой
 * почти стоит на месте.
 *
 * Для одной реплики FCFS ожидание даёт рекурсия Линдли W(i) = max(0, W(i−1) + S(i−1) − (A(i) − A(i−1))).
 * Это ровно то, что считает engine/sched.simulateQueue при servers = 1 (тест сверяет), только без объекта
 * на каждый запрос: прогон на 350 000 запросов занимает пару миллисекунд. Приходы и длины промптов общие
 * для всех нагрузок (один seed, тот же поток, что у makeWorkload), меняется только масштаб интервалов
 * между приходами. Поэтому TTFT каждого запроса растёт с нагрузкой, и кривая «нагрузка → p50, p99»
 * монотонна и не шумит от seed.
 */

/** Самый длинный промпт потока, токенов. */
const PROMPT_MAX = 16000
/** Длины промптов: лог-нормальное распределение (медиана 1000, длинный хвост до 16k), как у типичного чата с контекстом. */
const PROMPT_DIST: Dist = { kind: 'lognormal', median: 1000, sigma: 0.8, min: 32, max: PROMPT_MAX }
/** Нагрузки для кривой p50 / p99. */
export const LOADS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
/** Сколько первых запросов отбрасываем, пока очередь выходит на устойчивый режим (при 98 % ей нужны десятки тысяч). */
export const WARMUP = 50_000
/** Сколько запросов устойчивого режима прогоняем после разгона. */
export const HORIZON = 300_000
/** Из прогона в «весь трафик» идёт каждый STRIDE-й запрос. */
export const STRIDE = 10
/** Сколько запросов устойчивого режима храним: из них берутся выборки. */
export const POPULATION = Math.ceil(HORIZON / STRIDE)
/** Seed потока запросов (один на все нагрузки). */
const SEED = 3
/** Интенсивность, с которой генерируем поток, запросов в секунду: средний интервал 1 мс, дальше масштабируем под нагрузку. */
const BASE_RATE = 1000

interface QueueBase {
  key: string
  /** Интервалы между приходами при BASE_RATE, мс (gaps[i] = A(i) − A(i−1), первый от нуля); общие для любого железа. */
  gaps: Float32Array
  /** Время prefill каждого запроса (время обслуживания), мс. */
  service: Float32Array
  /** Среднее время prefill по всему потоку, мс. */
  svcMs: number
}

function hwKey(hw: Hardware): string {
  return `${hw.model.id}|${hw.gpu.id}|${hw.dtype}|${hw.tp}`
}

/** Маленький LRU: Map помнит порядок вставки. */
function remember<V>(cache: Map<string, V>, key: string, limit: number, make: () => V): V {
  const hit = cache.get(key)
  if (hit !== undefined) {
    cache.delete(key)
    cache.set(key, hit)
    return hit
  }
  const v = make()
  cache.set(key, v)
  if (cache.size > limit) cache.delete(cache.keys().next().value as string)
  return v
}

const baseCache = new Map<string, QueueBase>()
const popCache = new Map<string, Float64Array>()

/**
 * Поток запросов: интервалы между приходами при BASE_RATE, мс, и длины промптов. Порядок вызовов rng тот же,
 * что у makeWorkload({ rate: BASE_RATE, promptDist: PROMPT_DIST, outputDist: 1, seed }), то есть поток тот же,
 * только без объекта на каждый запрос (тест сверяет).
 */
export function requestStream(count: number, seed = SEED): { gaps: Float32Array; prompts: Uint16Array } {
  const rng = mulberry32(seed)
  const gaps = new Float32Array(count)
  const prompts = new Uint16Array(count)
  for (let i = 0; i < count; i++) {
    gaps[i] = exponential(rng, BASE_RATE / 1000)
    prompts[i] = sampleDist(PROMPT_DIST, rng)
  }
  return { gaps, prompts }
}

/** Поток запросов один на страницу: от железа зависит только время prefill. */
let stream: ReturnType<typeof requestStream> | null = null

/** Поток запросов и время prefill каждого для выбранных модели и GPU (считается один раз на железо). */
function queueBase(hw: Hardware): QueueBase {
  const key = hwKey(hw)
  return remember(baseCache, key, 2, () => {
    stream ??= requestStream(WARMUP + HORIZON)
    const { gaps, prompts } = stream
    const service = new Float32Array(prompts.length)
    // время prefill по длине промпта: длин всего несколько тысяч разных
    const memo = new Float64Array(PROMPT_MAX + 1)
    let sum = 0
    for (let i = 0; i < prompts.length; i++) {
      const n = prompts[i]
      if (!(memo[n] > 0)) memo[n] = prefillMs(hw.model, hw.gpu, n, hw.perf)
      service[i] = memo[n]
      sum += service[i]
    }
    return { key, gaps, service, svcMs: sum / prompts.length }
  })
}

/** Среднее время обслуживания (prefill) одного запроса, мс: из него нагрузка пересчитывается в запросы в секунду. */
export function meanServiceMs(hw: Hardware): number {
  return queueBase(hw).svcMs
}

/** Сколько запросов в секунду дают загрузку реплики rho. */
export function rateForLoad(hw: Hardware, rho: number): number {
  return (rho / meanServiceMs(hw)) * 1000
}

/**
 * TTFT каждого запроса одной FCFS-реплики (рекурсия Линдли): arrivals по возрастанию, мс;
 * service — время обслуживания каждого, мс. Совпадает с simulateQueue(…, { servers: 1 }).
 */
export function lindleyTtft(arrivals: ArrayLike<number>, service: ArrayLike<number>, from = 0): Float64Array {
  const n = arrivals.length
  const out = new Float64Array(Math.max(0, n - from))
  let w = 0
  for (let i = 0; i < n; i++) {
    if (i > 0) w = Math.max(0, w + service[i - 1] - (arrivals[i] - arrivals[i - 1]))
    if (i >= from) out[i - from] = w + service[i]
  }
  return out
}

/**
 * То же по интервалам между приходами (gaps[i] = A(i) − A(i−1)), умноженным на scale: TTFT каждого stride-го
 * запроса начиная с from. При stride 1 и scale 1 совпадает с lindleyTtft по накопленным интервалам (тест сверяет).
 */
export function lindleyStrided(gaps: ArrayLike<number>, service: ArrayLike<number>, scale: number, from: number, stride: number): Float64Array {
  return lindleyStridedMulti(gaps, service, [scale], from, stride)[0]
}

/**
 * То же сразу для нескольких масштабов интервалов (нагрузок): поток читается один раз, а не по разу на каждую
 * нагрузку, поэтому кривая из 13 нагрузок считается примерно вдвое быстрее.
 */
export function lindleyStridedMulti(
  gaps: ArrayLike<number>,
  service: ArrayLike<number>,
  scales: readonly number[],
  from: number,
  stride: number,
): Float64Array[] {
  const n = Math.min(gaps.length, service.length)
  const k = Math.max(1, Math.floor(stride))
  const start = Math.max(0, from)
  const len = Math.max(0, Math.ceil((n - start) / k))
  const m = scales.length
  const sc = Float64Array.from(scales)
  const outs = Array.from({ length: m }, () => new Float64Array(len))
  const w = new Float64Array(m)
  let next = start
  let j = 0
  for (let i = 0; i < n; i++) {
    if (i > 0) {
      const sPrev = service[i - 1]
      const g = gaps[i]
      for (let l = 0; l < m; l++) {
        const v = w[l] + sPrev - g * sc[l]
        w[l] = v > 0 ? v : 0
      }
    }
    if (i === next) {
      const sCur = service[i]
      for (let l = 0; l < m; l++) outs[l][j] = w[l] + sCur
      j++
      next += k
    }
  }
  return outs
}

/** Во сколько раз растянуть интервалы потока, чтобы загрузка реплики стала rho. */
function scaleFor(b: QueueBase, rho: number): number {
  // средний интервал между приходами должен стать svc / rho мс (при BASE_RATE он 1000 / BASE_RATE мс)
  return b.svcMs / Math.max(1e-6, rho) / (1000 / BASE_RATE)
}

const popKey = (b: QueueBase, rho: number) => `${b.key}|${rho.toFixed(3)}`
/** Сколько нагрузок помним (кривая занимает 13, остальное для слайдера). */
const POP_CACHE = 24

/** TTFT устойчивого режима при загрузке rho: POPULATION значений по возрастанию. */
export function steadyTtft(hw: Hardware, rho: number): Float64Array {
  const b = queueBase(hw)
  return remember(popCache, popKey(b, rho), POP_CACHE, () => lindleyStrided(b.gaps, b.service, scaleFor(b, rho), WARMUP, STRIDE).sort())
}

export interface TtftSample {
  /** TTFT выборки по возрастанию, мс. */
  values: Float64Array
  p50: number
  p90: number
  p99: number
  mean: number
  /** p99 всего трафика устойчивого режима (из него и берётся выборка). */
  popP99: number
}

/** Случайная выборка n запросов устойчивого режима (без повторов); seed задаёт, каких именно. */
export function sampleTtft(hw: Hardware, rho: number, n: number, seed: number): TtftSample {
  const pop = steadyTtft(hw, rho)
  const m = Math.max(1, Math.min(n, pop.length))
  const rng = mulberry32(seed)
  const idx = new Uint32Array(pop.length)
  for (let i = 0; i < idx.length; i++) idx[i] = i
  const values = new Float64Array(m)
  let sum = 0
  // частичная перетасовка Фишера-Йетса: первые m индексов случайны и не повторяются
  for (let i = 0; i < m; i++) {
    const j = i + Math.floor(rng() * (idx.length - i))
    const t = idx[i]
    idx[i] = idx[j]
    idx[j] = t
    values[i] = pop[idx[i]]
    sum += values[i]
  }
  values.sort()
  return {
    values,
    p50: quantileSorted(values, 0.5),
    p90: quantileSorted(values, 0.9),
    p99: quantileSorted(values, 0.99),
    mean: sum / m,
    popP99: quantileSorted(pop, 0.99),
  }
}

export interface QueuePoint {
  rho: number
  p50: number
  p99: number
}

/** p50 и p99 TTFT устойчивого режима на сетке нагрузок (график «хвост растёт» и текст главы). */
export function queueCurve(hw: Hardware): QueuePoint[] {
  // нагрузки, которых ещё нет в кэше, считаем одним проходом по потоку
  const b = queueBase(hw)
  const missing = LOADS.filter((rho) => !popCache.has(popKey(b, rho)))
  if (missing.length > 0) {
    const runs = lindleyStridedMulti(
      b.gaps,
      b.service,
      missing.map((rho) => scaleFor(b, rho)),
      WARMUP,
      STRIDE,
    )
    missing.forEach((rho, i) => remember(popCache, popKey(b, rho), POP_CACHE, () => runs[i].sort()))
  }
  return LOADS.map((rho) => {
    const s = steadyTtft(hw, rho)
    return { rho, p50: quantileSorted(s, 0.5), p99: quantileSorted(s, 0.99) }
  })
}
