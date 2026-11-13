import { findGpu, getGpu, type GpuSpecExt } from '@/engine/data/gpus'
import { findModel, getModel, type ModelSpecExt } from '@/engine/data/models'
import { VLLM_MEM_UTIL, kvBudgetBytes, usableMemBytes, type PerfOptions } from '@/engine/perf'
import type { DType } from '@/engine/types'

/**
 * Чистое ядро выбора железа для глав S4 (без React): списки, автоматический TP, Hardware.
 */

/** Модели для локальных селекторов (плотные 8B, MoE, большие). */
export const MODEL_CHOICES: readonly string[] = ['qwen3-8b', 'llama-3.1-8b', 'qwen3-30b-a3b', 'qwen3-32b', 'llama-3.1-70b']
/** GPU для локальных селекторов. */
export const GPU_CHOICES: readonly string[] = ['h100-sxm', 'h200', 'b200', 'a100-80g', 'l40s']

const FALLBACK_MODEL = 'qwen3-8b'
const FALLBACK_GPU = 'h100-sxm'

export const DTYPE_LABEL: Record<DType, string> = { bf16: 'BF16', fp8: 'FP8', fp4: 'FP4' }

/**
 * Доля памяти GPU под процесс движка: gpu_memory_utilization vLLM по умолчанию с v0.20 (SPEC §8.8).
 * В engine/perf DEFAULT_MEM_UTIL остаётся 0,9 (сверка с numbers_calc.py), поэтому главы передают это значение явно.
 */
export { VLLM_MEM_UTIL }

export function resolveModel(id: string): ModelSpecExt {
  return findModel(id) ?? getModel(FALLBACK_MODEL)
}

export function resolveGpu(id: string): GpuSpecExt {
  return findGpu(id) ?? getGpu(FALLBACK_GPU)
}

/** Есть ли у GPU тензорные ядра под формат (FP8 на A100 и FP4 на Hopper нет). */
export function dtypeSupported(gpu: GpuSpecExt, dtype: DType): boolean {
  return gpu.tflops[dtype] !== undefined
}

/**
 * Наименьший tensor parallel из 1, 2, 4, 8, при котором веса помещаются и под KV остаётся
 * хотя бы 15 % памяти процесса. Идеализация (TP без all-reduce), как во всём perf.ts.
 */
export function autoTp(model: ModelSpecExt, gpu: GpuSpecExt, dtype: DType): number {
  for (const tp of [1, 2, 4, 8]) {
    const budget = kvBudgetBytes(model, gpu, { dtype, gpus: tp, memUtil: VLLM_MEM_UTIL })
    const process = tp * usableMemBytes(gpu) * VLLM_MEM_UTIL
    if (budget >= 0.15 * process) return tp
  }
  return 8
}

export interface Hardware {
  model: ModelSpecExt
  gpu: GpuSpecExt
  dtype: DType
  /** Сколько GPU нужно (TP), чтобы модель поместилась. */
  tp: number
  /** Опции для perf.ts: формат весов и число GPU. */
  perf: PerfOptions
}

export function makeHardware(model: ModelSpecExt, gpu: GpuSpecExt, dtype: DType): Hardware {
  const tp = autoTp(model, gpu, dtype)
  return { model, gpu, dtype, tp, perf: { dtype, gpus: tp } }
}

/** «Qwen3-8B, H100 SXM, BF16» (+ «TP 4»). */
export function hwLabel(hw: Hardware): string {
  const parts = [hw.model.name, hw.tp > 1 ? `${hw.tp} × ${hw.gpu.name}` : hw.gpu.name, DTYPE_LABEL[hw.dtype]]
  return parts.join(', ')
}

