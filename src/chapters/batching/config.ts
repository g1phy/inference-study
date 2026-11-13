/**
 * Общая настройка симуляций для глав 15–17 (batching, disaggregation, parallelism).
 *
 * Модель и GPU берутся из настроек симулятора (settings.modelId / settings.gpuId). Если веса
 * не помещаются в один GPU или под KV остаётся слишком мало места, один «воркер» получает
 * несколько GPU (tensor parallel), как сделал бы инженер: Llama-3.1-70B на H100 → TP4.
 * Сценарии глав написаны во времени эталона (Qwen3-8B на одном H100); timeScale растягивает
 * моменты прихода запросов под выбранную конфигурацию, чтобы картинка сохраняла форму.
 */
import { findModel, getModel, type ModelSpecExt } from '@/engine/data/models'
import { findGpu, getGpu, type GpuSpecExt } from '@/engine/data/gpus'
import { kvCapacityTokens, prefillMs } from '@/engine/perf'
import type { DType } from '@/engine/types'

export interface SimConfig {
  model: ModelSpecExt
  gpu: GpuSpecExt
  /** GPU на одного воркера (tensor parallel). */
  tp: number
  /** Формат весов в симуляции. */
  dtype: DType
  /** Во сколько раз prefill медленнее эталона (Qwen3-8B, 1 × H100 SXM). */
  timeScale: number
  /** Модель из настроек не поместилась даже на 8 GPU: взята Qwen3-8B. */
  fallback: boolean
}

/** Сколько токенов KV должно помещаться на воркер, чтобы симуляция не упиралась в память. */
export const MIN_KV_TOKENS = 200_000
const TP_OPTIONS = [1, 2, 4, 8] as const
const REF_TOKENS = 4096

let refPrefill: number | null = null
function referencePrefillMs(): number {
  if (refPrefill === null) refPrefill = prefillMs(getModel('qwen3-8b'), getGpu('h100-sxm'), REF_TOKENS)
  return refPrefill
}

/** Минимальный TP (1, 2, 4, 8), при котором веса и MIN_KV_TOKENS токенов KV помещаются. */
export function minTp(model: ModelSpecExt, gpu: GpuSpecExt, dtype: DType = 'bf16'): number | null {
  for (const tp of TP_OPTIONS) {
    if (kvCapacityTokens(model, gpu, { gpus: tp, dtype }) >= MIN_KV_TOKENS) return tp
  }
  return null
}

const cache = new Map<string, SimConfig>()

/** Конфигурация симуляции для модели и GPU из настроек (с запасным вариантом). */
export function simConfig(modelId: string, gpuId: string): SimConfig {
  const key = `${modelId}|${gpuId}`
  const hit = cache.get(key)
  if (hit) return hit
  const gpu = findGpu(gpuId) ?? getGpu('h100-sxm')
  let model = findModel(modelId) ?? getModel('qwen3-8b')
  let dtype: DType = 'bf16'
  let tp = minTp(model, gpu, 'bf16')
  if (tp === null) {
    dtype = 'fp8'
    tp = minTp(model, gpu, 'fp8')
  }
  let fallback = false
  if (tp === null) {
    model = getModel('qwen3-8b')
    dtype = 'bf16'
    tp = minTp(model, gpu, 'bf16') ?? 1
    fallback = true
  }
  const timeScale = prefillMs(model, gpu, REF_TOKENS, { gpus: tp, dtype }) / referencePrefillMs()
  const cfg: SimConfig = { model, gpu, tp, dtype, timeScale, fallback }
  cache.set(key, cfg)
  return cfg
}

/** Параметры производительности для sched / perf. */
export function perfOf(cfg: SimConfig) {
  return { model: cfg.model, gpu: cfg.gpu, gpus: cfg.tp, dtype: cfg.dtype }
}

/** «Qwen3-8B · H100 SXM» или «Llama-3.1-70B · 4 × H100 SXM (TP4)». */
export function configLabel(cfg: SimConfig): string {
  const gpu = cfg.tp > 1 ? `${cfg.tp} × ${cfg.gpu.name} (TP${cfg.tp})` : cfg.gpu.name
  const dt = cfg.dtype === 'fp8' ? ', FP8' : ''
  return `${cfg.model.name} · ${gpu}${dt}`
}
