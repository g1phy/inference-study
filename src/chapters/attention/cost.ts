/**
 * Расчёты главы attention на настоящих данных моделей (engine/data, engine/perf):
 * варианты MHA / GQA / MQA, стоимость attention в prefill и decode, игрушка «зачем √d_k»,
 * трафик HBM обычного attention и FlashAttention.
 */
import { gqaGroups } from '@/engine/attention'
import type { GpuSpecExt } from '@/engine/data/gpus'
import type { ModelSpecExt } from '@/engine/data/models'
import { attentionFlopsPerPair, attentionPairs, kvBudgetBytes, kvBytesForSeq, kvBytesPerToken, maxConcurrentSeqs, tpKvReplication } from '@/engine/perf'
import { mulberry32, normal } from '@/engine/rng'

/* ============================================================================
 * MHA / GQA / MQA
 * ========================================================================== */

export type AttnScheme = 'mha' | 'gqa' | 'mqa'

/** Длина контекста для сравнения схем: 32K токенов (нативный контекст Qwen3). */
export const COMPARE_CONTEXT = 32_768

/** Доля памяти GPU под движок: gpu_memory_utilization vLLM по умолчанию с v0.20 (SPEC §8.8), как в главах про KV cache. */
export const MEM_UTIL = 0.92

/** Сколько KV-голов у схемы для данной модели: MHA = все головы, GQA = как в config, MQA = одна. */
export function kvHeadsFor(model: ModelSpecExt, scheme: AttnScheme): number {
  if (scheme === 'mha') return model.heads
  if (scheme === 'mqa') return 1
  return model.kvHeads
}

export interface SchemeStats {
  scheme: AttnScheme
  heads: number
  kvHeads: number
  /** Q-голов на одну пару K/V. */
  group: number
  groups: ReturnType<typeof gqaGroups>
  bytesPerToken: number
  /** KV одного запроса длиной COMPARE_CONTEXT. */
  bytesPerSeq: number
  /** Во сколько раз меньше, чем у MHA. */
  vsMha: number
  /** Сколько GPU нужно, чтобы веса влезли (TP), и сколько запросов COMPARE_CONTEXT помещается в KV. */
  gpus: number
  maxSeqs: number
  kvBudget: number
  /** Во сколько раз KV копируется при TP: пар K/V меньше, чем GPU, и каждую держат несколько GPU. */
  replication: number
}

/** Модель с другим числом KV-голов (для «что было бы при MHA/MQA»). */
export function withKvHeads(model: ModelSpecExt, kvHeads: number): ModelSpecExt {
  return { ...model, kvHeads }
}

/**
 * Наименьшее число GPU из 1, 2, 4, 8, при котором веса модели помещаются и в KV остаётся место
 * хотя бы для одного запроса COMPARE_CONTEXT с настоящей конфигурацией модели (её GQA).
 */
export function gpusToFit(model: ModelSpecExt, gpu: GpuSpecExt): number {
  return [1, 2, 4, 8].find((n) => maxConcurrentSeqs(model, gpu, { contextLen: COMPARE_CONTEXT, gpus: n, memUtil: MEM_UTIL }) >= 1) ?? 8
}

export function schemeStats(model: ModelSpecExt, gpu: GpuSpecExt, scheme: AttnScheme): SchemeStats {
  const kvHeads = kvHeadsFor(model, scheme)
  const variant = withKvHeads(model, kvHeads)
  const bytesPerToken = kvBytesPerToken(variant)
  const mhaBytes = kvBytesPerToken(withKvHeads(model, model.heads))
  const gpus = gpusToFit(model, gpu)
  return {
    scheme,
    heads: model.heads,
    kvHeads,
    group: model.heads / kvHeads,
    groups: gqaGroups(model.heads, kvHeads),
    bytesPerToken,
    bytesPerSeq: kvBytesForSeq(variant, COMPARE_CONTEXT),
    vsMha: mhaBytes / bytesPerToken,
    gpus,
    maxSeqs: maxConcurrentSeqs(variant, gpu, { contextLen: COMPARE_CONTEXT, gpus, memUtil: MEM_UTIL }),
    kvBudget: kvBudgetBytes(variant, gpu, { gpus, memUtil: MEM_UTIL }),
    replication: tpKvReplication(variant, gpus),
  }
}

/* ============================================================================
 * Сколько стоит attention: prefill против одного шага decode
 * ========================================================================== */

export interface AttnCostRow {
  context: number
  /** Пары (запрос, ключ) на одну голову и один слой. */
  prefillPairs: number
  prefillFlops: number
  decodePairs: number
  decodeFlops: number
}

/** FLOPs attention (QKᵀ и взвешивание V) по всем слоям и головам: prefill промпта и один шаг decode. */
export function attnCost(model: ModelSpecExt, context: number): AttnCostRow {
  const perPair = attentionFlopsPerPair(model, 'prefill') * model.layers
  const prefillPairs = attentionPairs(model, context, 0).full
  const decodePairs = attentionPairs(model, 1, context).full
  return {
    context,
    prefillPairs,
    prefillFlops: prefillPairs * perPair,
    decodePairs,
    decodeFlops: decodePairs * perPair,
  }
}

/** FLOPs линейных слоёв на один токен decode: 2 × параметры, которые умножаются на каждый токен. */
export function linearFlopsPerToken(model: ModelSpecExt): number {
  return 2 * (model.activeMatmulParamsB ?? model.activeParamsB) * 1e9
}

/** С какой длины контекста attention одного шага decode дороже всех линейных слоёв вместе. */
export function attentionCrossover(model: ModelSpecExt): number {
  return linearFlopsPerToken(model) / (attentionFlopsPerPair(model, 'decode') * model.layers)
}

/* ============================================================================
 * Зачем делить на √d_k: случайные векторы
 * ========================================================================== */

export interface ScaleDemo {
  dK: number
  /** Оценки одного запроса против 8 ключей до деления. */
  scores: number[]
  /** Разброс (стандартное отклонение) оценок по многим случайным парам: ≈ √d_k. */
  sigma: number
}

const KEYS = 8

/**
 * Один запрос и 8 ключей со случайными компонентами N(0, 1) (детерминированно, seed от d_k),
 * плюс разброс скалярного произведения по 2000 случайным парам.
 */
const scaleCache = new Map<number, ScaleDemo>()

export function scaleDemo(dK: number): ScaleDemo {
  const hit = scaleCache.get(dK)
  if (hit) return hit
  const demo = computeScaleDemo(dK)
  scaleCache.set(dK, demo)
  return demo
}

function computeScaleDemo(dK: number): ScaleDemo {
  const rng = mulberry32(1000 + dK)
  const vec = () => Array.from({ length: dK }, () => normal(rng))
  const q = vec()
  const scores = Array.from({ length: KEYS }, () => {
    const k = vec()
    return q.reduce((s, x, i) => s + x * k[i], 0)
  })
  const sample = Array.from({ length: 2000 }, () => {
    const a = vec()
    const b = vec()
    return a.reduce((s, x, i) => s + x * b[i], 0)
  })
  const mean = sample.reduce((a, b) => a + b, 0) / sample.length
  const sigma = Math.sqrt(sample.reduce((s, x) => s + (x - mean) ** 2, 0) / sample.length)
  return { dK, scores, sigma }
}

/** Устойчивый softmax одного вектора. */
export function softmax(xs: readonly number[]): number[] {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

/* ============================================================================
 * FlashAttention: трафик HBM одной головы (FP16/BF16, 2 байта)
 * ========================================================================== */

export interface FlashTraffic {
  n: number
  d: number
  /** Матрица n × n одной головы в байтах (S или P). */
  scoreMatrixBytes: number
  /** Обычный attention: читаем Q, K, пишем S, читаем S, пишем P, читаем P и V, пишем O = 8n² + 8nd. */
  standardBytes: number
  /** Идеальный fused-вариант: только Q, K, V и O = 8nd. */
  flashBytes: number
}

export function flashTraffic(n: number, d: number): FlashTraffic {
  return {
    n,
    d,
    scoreMatrixBytes: 2 * n * n,
    standardBytes: 8 * n * n + 8 * n * d,
    flashBytes: 8 * n * d,
  }
}
