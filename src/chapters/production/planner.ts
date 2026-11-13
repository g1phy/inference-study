/**
 * Независимое масштабирование prefill и decode: расчёт реплик как в Dynamo Planner v1.0
 * (research/notes-dynamo.md §1.17, §3.9):
 *   prefill_replicas = ceil(next_requests · next_isl / interval · min(1, prefill_correction) / throughput_per_gpu / gpus_per_engine)
 *   decode_replicas  = ceil(next_requests · next_osl / interval / throughput_per_gpu / gpus_per_engine)
 * В Planner пропускную способность на GPU дают профили движка. Здесь её считает engine/perf
 * (roofline): prefill: сколько токенов промпта в секунду успевает реплика, не выходя за TTFT SLA;
 * decode: сколько токенов в секунду даёт самый большой batch, у которого шаг ≤ ITL SLA.
 */
import { VLLM_MEM_UTIL, decodeStepMs, fitsCheck, maxConcurrentSeqs, stepTime, type GpuLike, type ModelLike } from '@/engine/perf'
import type { DType } from '@/engine/types'

/** Формула Planner v1.0 для prefill. */
export function plannerPrefillReplicas(o: { requests: number; isl: number; intervalSec: number; correction?: number; throughputPerGpu: number; gpusPerEngine: number }): number {
  const load = ((o.requests * o.isl) / o.intervalSec) * Math.min(1, o.correction ?? 1)
  return Math.max(1, Math.ceil(load / o.throughputPerGpu / o.gpusPerEngine - 1e-9))
}

/** Формула Planner v1.0 для decode. */
export function plannerDecodeReplicas(o: { requests: number; osl: number; intervalSec: number; throughputPerGpu: number; gpusPerEngine: number }): number {
  return Math.max(1, Math.ceil((o.requests * o.osl) / o.intervalSec / o.throughputPerGpu / o.gpusPerEngine - 1e-9))
}

/**
 * TP реплики: наименьший из 1, 2, 4, 8, при котором веса занимают не больше половины памяти
 * процесса (правило книги: веса плюс не меньше 50 % сверху на KV cache, notes-foundations-production §2.2).
 */
export function gpusPerEngine(model: ModelLike, gpu: GpuLike, contextLen: number): number {
  for (const g of [1, 2, 4, 8]) {
    const f = fitsCheck(model, gpu, { contextLen, seqs: 8, gpus: g })
    if (f.fits && f.weightBytes <= 0.5 * f.processBytes) return g
  }
  return 8
}

/**
 * Сколько GPU нужно реплике в этом формате весов (для холодного старта): сначала 1, 2, 4, 8 по тому же
 * правилу «веса не больше половины памяти»; если не выходит, хватает ли места вообще на 8, 16 или 32 GPU
 * (целые узлы по 8). Если не помещается и в 32, fits = false.
 */
export function replicaGpus(model: ModelLike, gpu: GpuLike, contextLen: number, dtype: DType = 'bf16'): { gpus: number; fits: boolean } {
  for (const g of [1, 2, 4, 8]) {
    const f = fitsCheck(model, gpu, { contextLen, seqs: 8, gpus: g, dtype })
    if (f.fits && f.weightBytes <= 0.5 * f.processBytes) return { gpus: g, fits: true }
  }
  for (const g of [8, 16, 32]) {
    if (fitsCheck(model, gpu, { contextLen, seqs: 8, gpus: g, dtype }).fits) return { gpus: g, fits: true }
  }
  return { gpus: 32, fits: false }
}

export interface PoolPlan {
  /** Реплик (движков) в пуле. */
  replicas: number
  gpusPerEngine: number
  /** Токенов в секунду на одну реплику. */
  engineTokS: number
  /** Нагрузка на пул, токенов в секунду. */
  loadTokS: number
  /** Batch одной реплики. */
  batch: number
  /** Время шага (prefill) или ITL (decode) при этом batch, мс. */
  stepMs: number
  /** SLA недостижим даже при batch 1. */
  slaMissed: boolean
  /**
   * Что ограничило batch: prefill: цель TTFT ('sla') или потолок перебора ('cap');
   * decode: память под KV ('kv': больше последовательностей такой длины не влезает) или цель ITL ('sla').
   */
  limit: 'sla' | 'kv' | 'cap'
}

export interface PlannerPlan {
  prefill: PoolPlan
  decode: PoolPlan
}

export function planPools(
  model: ModelLike,
  gpu: GpuLike,
  o: { rps: number; isl: number; osl: number; ttftMs: number; itlMs: number },
): PlannerPlan {
  const ctx = o.isl + o.osl / 2
  const tp = gpusPerEngine(model, gpu, o.isl + o.osl)
  // prefill: больше промптов за шаг, пока шаг укладывается в TTFT SLA
  let pb = 1
  let pMs = stepTime(model, gpu, { prefill: [{ tokens: o.isl }] }, { gpus: tp }).ms
  const pMissed = pMs > o.ttftMs
  let pLimit: PoolPlan['limit'] = 'sla'
  if (!pMissed) {
    pLimit = 'cap'
    for (let b = 2; b <= 64; b *= 2) {
      const ms = stepTime(model, gpu, { prefill: Array.from({ length: b }, () => ({ tokens: o.isl })) }, { gpus: tp }).ms
      if (ms > o.ttftMs) {
        pLimit = 'sla'
        break
      }
      pb = b
      pMs = ms
    }
  }
  const pTokS = (pb * o.isl * 1000) / pMs
  const pLoad = o.rps * o.isl
  // decode: самый большой batch, у которого ITL в SLA и KV влезает
  const maxSeqs = Math.max(1, maxConcurrentSeqs(model, gpu, { contextLen: Math.ceil(o.isl + o.osl), gpus: tp, memUtil: VLLM_MEM_UTIL }))
  let lo = 1
  const itl1 = decodeStepMs(model, gpu, { batch: 1, avgContext: ctx, gpus: tp })
  const dMissed = itl1 > o.itlMs
  if (!dMissed) {
    let hi = maxSeqs
    if (decodeStepMs(model, gpu, { batch: hi, avgContext: ctx, gpus: tp }) <= o.itlMs) lo = hi
    else {
      while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2)
        if (decodeStepMs(model, gpu, { batch: mid, avgContext: ctx, gpus: tp }) <= o.itlMs) lo = mid
        else hi = mid
      }
    }
  }
  const dMs = decodeStepMs(model, gpu, { batch: lo, avgContext: ctx, gpus: tp })
  const dTokS = (lo * 1000) / dMs
  const dLoad = o.rps * o.osl
  return {
    prefill: {
      replicas: plannerPrefillReplicas({ requests: o.rps, isl: o.isl, intervalSec: 1, throughputPerGpu: pTokS / tp, gpusPerEngine: tp }),
      gpusPerEngine: tp,
      engineTokS: pTokS,
      loadTokS: pLoad,
      batch: pb,
      stepMs: pMs,
      slaMissed: pMissed,
      limit: pLimit,
    },
    decode: {
      replicas: plannerDecodeReplicas({ requests: o.rps, osl: o.osl, intervalSec: 1, throughputPerGpu: dTokS / tp, gpusPerEngine: tp }),
      gpusPerEngine: tp,
      engineTokS: dTokS,
      loadTokS: dLoad,
      batch: lo,
      stepMs: dMs,
      slaMissed: dMissed,
      limit: !dMissed && lo >= maxSeqs ? 'kv' : 'sla',
    },
  }
}
