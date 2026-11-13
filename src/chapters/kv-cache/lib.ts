/**
 * Общие помощники глав части IV («Память: KV cache»): списки моделей и GPU для селекторов,
 * подбор числа GPU, разбивка памяти GPU для калькулятора, KV-варианты для сравнения MHA / GQA / MQA.
 * Всё считает движок (src/engine/perf.ts, src/engine/data/*), здесь только склейка для UI.
 */
import type { DType } from '@/engine/types'
import { findModel, getModel, type ModelSpecExt } from '@/engine/data/models'
import { findGpu, getGpu, type GpuSpecExt } from '@/engine/data/gpus'
import { fitsCheck, kvBytesPerToken, usableMemBytes, weightBytes, type FitsResult } from '@/engine/perf'

/** Модели для калькуляторов главы (обязательные из SPEC §4, A2). */
export const KV_MODEL_IDS = [
  'qwen3-8b',
  'qwen3-32b',
  'qwen3-30b-a3b',
  'qwen3-235b-a22b',
  'llama-3.1-8b',
  'llama-3.1-70b',
  'deepseek-v3',
  'gpt-oss-120b',
] as const

/** GPU для калькуляторов (от старых к новым). */
export const KV_GPU_IDS = ['l40s', 'a100-80g', 'h100-sxm', 'h200', 'b200', 'b300'] as const

export const DEFAULT_MODEL_ID = 'qwen3-8b'
export const DEFAULT_GPU_ID = 'h100-sxm'

/** Модель по id из настроек; если такой нет в данных, Qwen3-8B. */
export function modelOr(id: string | undefined): ModelSpecExt {
  return (id && findModel(id)) || getModel(DEFAULT_MODEL_ID)
}

/** GPU по id из настроек; если такого нет, H100 SXM. */
export function gpuOr(id: string | undefined): GpuSpecExt {
  return (id && findGpu(id)) || getGpu(DEFAULT_GPU_ID)
}

/** id модели, если она есть в списке селектора, иначе Qwen3-8B (селектор не должен показывать пустое значение). */
export function pickModelId(id: string | undefined, allowed: readonly string[] = KV_MODEL_IDS): string {
  return id && allowed.includes(id) ? id : DEFAULT_MODEL_ID
}

export function pickGpuId(id: string | undefined, allowed: readonly string[] = KV_GPU_IDS): string {
  return id && allowed.includes(id) ? id : DEFAULT_GPU_ID
}

export const modelOptions = (ids: readonly string[] = KV_MODEL_IDS) => ids.map((id) => ({ value: id, label: getModel(id).name }))

export const gpuOptions = (ids: readonly string[] = KV_GPU_IDS) =>
  ids.map((id) => {
    const g = getGpu(id)
    return { value: id, label: `${shortGpuName(g)} · ${g.memGB} ГБ` }
  })

/** «H100 SXM» → «H100», «B200 (HGX)» → «B200», «A100 80GB SXM» → «A100»: короткое имя для подписей. */
export function shortGpuName(g: { name: string }): string {
  return g.name.split(/[\s(]/)[0] || g.name
}

/**
 * Сколько GPU (tensor parallel) нужно, чтобы веса поместились и под KV осталось хотя бы
 * `minKvShare` памяти процесса. Перебираем 1, 2, 4, 8; если не хватает и восьми, возвращаем 8.
 */
export function minGpus(model: ModelSpecExt, gpu: GpuSpecExt, dtype: DType = 'bf16', minKvShare = 0.15): number {
  for (const n of [1, 2, 4, 8]) {
    const f = fitsCheck(model, gpu, { contextLen: 1, seqs: 0, gpus: n, dtype })
    if (f.kvBudgetBytes > f.processBytes * minKvShare) return n
  }
  return 8
}

/* ============================================================================
 * Разбивка памяти GPU для «высокой сегментированной полосы» калькулятора
 * ========================================================================== */

export type MemPartId = 'weights' | 'reserve' | 'kvUsed' | 'kvFree' | 'outside'

export interface MemPart {
  id: MemPartId
  bytes: number
  /** Доля всей памяти GPU (0..1). */
  share: number
}

export interface MemoryLayout {
  fits: FitsResult
  /** Вся память всех GPU (nvidia-smi), байт. */
  total: number
  parts: MemPart[]
  /** Сколько не помещается (0, если всё влезло): KV сверх пула или веса сверх памяти. */
  overflowBytes: number
  /** Что именно не влезло. */
  overflowKind: 'kv' | 'weights' | null
  /** Ёмкость пула KV в токенах (бюджет / байт на токен), 0 если веса не влезли. */
  poolTokens: number
}

/**
 * Полоса «веса / активации / KV занято / KV свободно / вне движка» для fitsCheck.
 * Если KV не влезает, «KV занято» заполняет весь пул, а остаток уходит в overflowBytes.
 * Если не влезают веса, веса режутся по доступной процессу памяти (полоса = 100 %).
 */
export function memoryLayout(
  model: ModelSpecExt,
  gpu: GpuSpecExt,
  opts: { contextLen: number; seqs: number; gpus: number; dtype: DType; kvDtype: DType; memUtil: number },
): MemoryLayout {
  const fits = fitsCheck(model, gpu, opts)
  const total = fits.totalBytes
  const outside = Math.max(0, total - fits.processBytes)
  const budget = Math.max(0, fits.kvBudgetBytes)
  const weightsShown = Math.min(fits.weightBytes, Math.max(0, fits.processBytes - fits.reserveBytes))
  const reserveShown = Math.min(fits.reserveBytes, Math.max(0, fits.processBytes - weightsShown))
  const kvUsed = Math.min(fits.kvNeededBytes, budget)
  const kvFree = Math.max(0, budget - fits.kvNeededBytes)
  const overflowBytes = fits.weightsFit ? Math.max(0, fits.kvNeededBytes - budget) : Math.max(0, fits.weightBytes - weightsShown)
  const overflowKind = overflowBytes <= 0 ? null : fits.weightsFit ? 'kv' : 'weights'
  const raw: Array<[MemPartId, number]> = [
    ['weights', weightsShown],
    ['reserve', reserveShown],
    ['kvUsed', kvUsed],
    ['kvFree', kvFree],
    ['outside', outside],
  ]
  const parts = raw.map(([id, bytes]) => ({ id, bytes, share: total > 0 ? bytes / total : 0 }))
  const perToken = kvBytesPerToken(model, opts.kvDtype)
  return { fits, total, parts, overflowBytes, overflowKind, poolTokens: budget > 0 && perToken > 0 ? Math.floor(budget / perToken) : 0 }
}

/** Память одного GPU в байтах (как видит nvidia-smi). */
export const gpuBytes = (g: GpuSpecExt) => usableMemBytes(g)

/* ============================================================================
 * MHA / GQA / MQA на одной архитектуре (для «Глубже»)
 * ========================================================================== */

/** Байт KV на токен, если бы у модели было другое число KV-голов (остальное как есть). */
export function kvPerTokenWithHeads(model: ModelSpecExt, kvHeads: number, kvDtype: DType = 'bf16'): number {
  return kvBytesPerToken({ ...model, attention: kvHeads === model.heads ? 'mha' : 'gqa', kvHeads, mlaDim: undefined }, kvDtype)
}

/** Веса модели, байт (для подписи «для сравнения: веса …»). */
export const modelWeights = (m: ModelSpecExt, dtype: DType = 'bf16') => weightBytes(m, dtype)

/* ============================================================================
 * Мелочи форматирования для логов движков (они пишут числа по-английски)
 * ========================================================================== */

/** 407935 → «407,935» (так печатает vLLM в логе старта). */
export function enInt(n: number): string {
  return Math.round(n).toLocaleString('en-US')
}

/** 49.795 → «49.80» (два знака, точка). */
export function enFixed(n: number, digits = 2): string {
  return n.toFixed(digits)
}

/** Логарифмическая сетка из count точек от lo до hi (включительно), целые значения без повторов. */
export function logGrid(lo: number, hi: number, count: number): number[] {
  const out: number[] = []
  const a = Math.log(lo)
  const b = Math.log(hi)
  for (let i = 0; i < count; i++) {
    const v = Math.round(Math.exp(a + ((b - a) * i) / (count - 1)))
    if (!out.length || v > out[out.length - 1]) out.push(v)
  }
  return out
}
