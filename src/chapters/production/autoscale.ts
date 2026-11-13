/**
 * Упрощённая симуляция автоскейлинга реплик LLM за сутки (глава 26, фигура 1).
 *
 * Модель (всё условно, шаг 1 минута):
 * - трафик λ(t), запросов/с: суточная кривая (чат или рабочие часы) плюс необязательный всплеск;
 * - каждый запрос занимает слот реплики W секунд (prefill + decode), реплика держит до C запросов
 *   одновременно (concurrency target = batch size, notes-foundations-production §1.16);
 * - очередь: жидкостная FIFO-модель: за минуту приходит λ·60 запросов, реплики успевают обслужить
 *   R·C/W·60; остаток ждёт. Ожидание запроса = через сколько обслуженных догонит его номер;
 * - автоскейлер раз в минуту считает желаемое число реплик по сигналу (как HPA:
 *   desired = ceil(текущие · метрика / цель), notes-foundations-production §1.15, F9) с целевой
 *   загрузкой 70 % (Knative), scale up сразу (реплика готова через cold start), scale down только
 *   если желаемое держится ниже текущего дольше scale-down delay.
 * Сигналы:
 * - 'concurrency': запросы в системе (обслуживаются + ждут) на реплику; правильный сигнал;
 * - 'gpu': utilization.gpu из nvidia-smi: доля времени, когда работает хоть одно ядро; уже при
 *   одном запросе на реплике это ≈ 100 %, поэтому сигнал не видит запаса (notes §1.24, §5.10);
 * - 'cpu': загрузка CPU сервера модели почти не зависит от трафика (работает GPU), автоскейлер
 *   её не замечает.
 */
import { mulberry32 } from '@/engine/rng'

export type TrafficPreset = 'chat' | 'b2b'
export type ScaleSignal = 'concurrency' | 'gpu' | 'cpu'

export interface AutoscaleOptions {
  preset: TrafficPreset
  spike: boolean
  signal: ScaleSignal
  /** Холодный старт реплики, минут. */
  coldStartMin: number
  /** Задержка перед уменьшением, минут. */
  scaleDownDelayMin: number
  minReplicas: number
  maxReplicas: number
  /** Concurrency target: запросов одновременно на реплику. */
  concurrency?: number
  /** Время запроса в системе при полной реплике, с. */
  serviceSec?: number
  /** Целевая загрузка (доля concurrency target), по умолчанию 0,7. */
  targetUtil?: number
  /** Порог SLO по ожиданию в очереди, с. */
  sloWaitSec?: number
  /** Окно усреднения сигнала, минут. */
  windowMin?: number
  seed?: number
}

export interface AutoscaleMinute {
  /** Минута суток 0..1439. */
  t: number
  /** Приход, запросов/с. */
  lambda: number
  /** Сколько запросов было бы в обслуживании при бесконечной ёмкости: λ·W. */
  demand: number
  /** Ёмкость готовых реплик: R·C. */
  capacity: number
  ready: number
  /** Готовые + стартующие (за них тоже платят). */
  total: number
  /** Ждут в очереди (в конце минуты). */
  queue: number
  /** Ожидание запроса, пришедшего в эту минуту, с (Infinity, если не обслужен до конца суток). */
  waitSec: number
  /** Значение сигнала в процентах от цели (для подписи). */
  signalPct: number
  violation: boolean
}

export interface AutoscaleResult {
  minutes: AutoscaleMinute[]
  gpuHours: number
  /** GPU-часы, если бы реплик было ровно сколько нужно в каждую минуту (идеал без запаса). */
  idealGpuHours: number
  violationMinutes: number
  /** Доля пришедших запросов, ждавших дольше SLO. */
  violatedShare: number
  maxWaitSec: number
  maxQueue: number
  peakReplicas: number
}

/** Суточная форма трафика, запросов/с. */
export function trafficAt(minute: number, preset: TrafficPreset, spike: boolean): number {
  const h = minute / 60
  let base: number
  if (preset === 'chat') {
    // минимум около 4:00, рост к обеду, пик вечером около 21:00
    const day = 0.5 - 0.5 * Math.cos(((h - 4) / 24) * 2 * Math.PI)
    const evening = Math.exp(-(((h - 21) / 2.2) ** 2))
    base = 14 + 70 * day + 45 * evening
  } else {
    // рабочие часы 9-18 с провалом на обед; ночью почти ноль
    const ramp = (x: number, w: number) => 1 / (1 + Math.exp(-x / w))
    const work = ramp(h - 9, 0.35) * (1 - ramp(h - 18.2, 0.45))
    const lunch = 1 - 0.25 * Math.exp(-(((h - 13) / 0.6) ** 2))
    base = 0.15 + 110 * work * lunch
    if (h < 6.5 || h > 21.5) base = 0
  }
  if (spike) {
    const s = 14
    if (h >= s && h < s + 0.75) {
      const rise = Math.min(1, (h - s) / 0.08)
      const fall = Math.min(1, (s + 0.75 - h) / 0.15)
      base *= 1 + 1.0 * Math.min(rise, fall)
    }
  }
  return base
}

const MIN_PER_DAY = 1440

export function simulateAutoscale(opts: AutoscaleOptions): AutoscaleResult {
  const C = opts.concurrency ?? 32
  const W = opts.serviceSec ?? 3
  const util = opts.targetUtil ?? 0.7
  const slo = opts.sloWaitSec ?? 2
  const win = Math.max(1, Math.round(opts.windowMin ?? 2))
  const cold = Math.max(0, Math.round(opts.coldStartMin))
  const downDelay = Math.max(0, Math.round(opts.scaleDownDelayMin))
  const minR = Math.max(0, Math.round(opts.minReplicas))
  const maxR = Math.max(minR, Math.round(opts.maxReplicas))
  const rng = mulberry32(opts.seed ?? 5)

  // шум трафика ±4 %, сглаженный
  const noise: number[] = []
  let nz = 0
  for (let t = 0; t < MIN_PER_DAY; t++) {
    nz = 0.85 * nz + 0.15 * (rng() * 2 - 1)
    noise.push(1 + 0.3 * nz)
  }

  // старт: столько реплик, сколько нужно на трафик в полночь (система уже работала)
  const lambda0 = trafficAt(0, opts.preset, opts.spike)
  let ready = clamp(Math.ceil((lambda0 * W) / (C * util)), minR, maxR)
  const pending: number[] = [] // минуты готовности стартующих реплик
  let backlog = 0
  let belowSince: number | null = null
  const hist: number[] = []
  const minutes: AutoscaleMinute[] = []
  const arrivalsCum: number[] = []
  const servedCum: number[] = []
  let aCum = 0
  let sCum = 0
  let gpuMinutes = 0
  let idealMinutes = 0

  for (let t = 0; t < MIN_PER_DAY; t++) {
    // готовые к этой минуте реплики
    for (let i = pending.length - 1; i >= 0; i--) {
      if (pending[i] <= t) {
        pending.splice(i, 1)
        ready++
      }
    }
    const lambda = trafficAt(t, opts.preset, opts.spike) * noise[t]
    const arrivals = lambda * 60
    const capPerMin = (ready * C * 60) / W
    const served = Math.min(backlog + arrivals, capPerMin)
    backlog = backlog + arrivals - served
    aCum += arrivals
    sCum += served
    arrivalsCum.push(aCum)
    servedCum.push(sCum)

    const inService = (served / 60) * W
    const inSystem = inService + backlog
    hist.push(inSystem)
    if (hist.length > win) hist.shift()
    const avgInSystem = hist.reduce((a, b) => a + b, 0) / hist.length

    // сигнал и желаемое число реплик
    const totalNow = ready + pending.length
    let desired: number
    let signalPct: number
    if (opts.signal === 'concurrency') {
      desired = Math.ceil(avgInSystem / (C * util) - 1e-9)
      signalPct = ready > 0 ? avgInSystem / ready / (C * util) : avgInSystem > 0 ? Infinity : 0
      // scale from zero: запрос в очереди будит систему
      if (backlog > 0 && desired === 0) desired = 1
    } else if (opts.signal === 'gpu') {
      // utilization.gpu: доля реплик, на которых хоть что-то выполняется
      const perReplica = ready > 0 ? avgInSystem / ready : 0
      const gpuUtil = ready > 0 ? Math.min(1, perReplica) : 0
      signalPct = gpuUtil / util
      desired = ready > 0 ? Math.ceil(totalNow * (gpuUtil / util) - 1e-9) : backlog > 0 ? 1 : 0
    } else {
      // CPU: токенизация и планировщик, почти постоянная загрузка
      const perReplica = ready > 0 ? avgInSystem / ready : 0
      const cpu = ready > 0 ? 0.06 + 0.08 * Math.min(1, perReplica / C) : 0
      signalPct = cpu / util
      desired = ready > 0 ? Math.ceil(totalNow * (cpu / util) - 1e-9) : backlog > 0 ? 1 : 0
    }
    desired = clamp(desired, minR, maxR)

    if (desired > totalNow) {
      for (let i = 0; i < desired - totalNow; i++) pending.push(t + cold)
      belowSince = null
    } else if (desired < ready) {
      if (belowSince === null) belowSince = t
      if (t - belowSince >= downDelay) {
        // сначала отменяем ещё не готовые, потом гасим готовые
        let excess = ready + pending.length - desired
        while (excess > 0 && pending.length > 0) {
          pending.pop()
          excess--
        }
        ready = Math.max(desired, ready - excess)
        belowSince = null
      }
    } else belowSince = null

    gpuMinutes += ready + pending.length
    idealMinutes += Math.ceil((lambda * W) / C - 1e-9)
    minutes.push({
      t,
      lambda,
      demand: lambda * W,
      capacity: ready * C,
      ready,
      total: ready + pending.length,
      queue: backlog,
      waitSec: 0,
      signalPct: Number.isFinite(signalPct) ? signalPct * 100 : 999,
      violation: false,
    })
  }

  // ожидание по FIFO: запрос, пришедший к концу минуты t (номер arrivalsCum[t]),
  // обслужен, когда servedCum догонит этот номер (линейная интерполяция внутри минуты)
  let u = 0
  let violatedArrivals = 0
  let violationMinutes = 0
  let maxWait = 0
  let maxQueue = 0
  for (let t = 0; t < MIN_PER_DAY; t++) {
    const target = arrivalsCum[t] - 1e-6
    if (u < t) u = t
    while (u < MIN_PER_DAY && servedCum[u] < target) u++
    let wait: number
    if (u >= MIN_PER_DAY) wait = Infinity
    else if (u === t) wait = 0
    else {
      const prev = servedCum[u - 1]
      const frac = (target - prev) / Math.max(1e-9, servedCum[u] - prev)
      wait = (u - 1 + frac - t) * 60
    }
    const m = minutes[t]
    m.waitSec = Math.max(0, wait)
    m.violation = m.waitSec > slo
    if (m.violation) {
      violationMinutes++
      violatedArrivals += m.lambda * 60
    }
    maxWait = Math.max(maxWait, m.waitSec)
    maxQueue = Math.max(maxQueue, m.queue)
  }

  return {
    minutes,
    gpuHours: gpuMinutes / 60,
    idealGpuHours: idealMinutes / 60,
    violationMinutes,
    violatedShare: aCum > 0 ? violatedArrivals / aCum : 0,
    maxWaitSec: maxWait,
    maxQueue,
    peakReplicas: minutes.reduce((m, x) => Math.max(m, x.total), 0),
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** Интервалы нарушения SLO (минуты начала и конца) для полос на графике. */
export function violationBands(minutes: readonly AutoscaleMinute[]): Array<[number, number]> {
  const out: Array<[number, number]> = []
  let start: number | null = null
  for (const m of minutes) {
    if (m.violation && start === null) start = m.t
    if (!m.violation && start !== null) {
      out.push([start, m.t])
      start = null
    }
  }
  if (start !== null) out.push([start, minutes.length])
  return out
}
