import { useMemo } from 'react'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { findModel, getModel } from '@/engine/data/models'
import { fitsCheck } from '@/engine/perf'
import { useSettings } from '@/state/settings'

/** Почему пример взят не из симулятора: там MoE-модель или модель, которая не влезает в одну GPU. */
export type RunningFallback = 'moe' | 'fit' | null

/**
 * Модель и GPU для примеров глав speculative и quantization: плотная модель из настроек симулятора на его GPU.
 * MoE подменяем плотной Qwen3-8B: у MoE каждый токен в проходе зовёт своих экспертов, проход из нескольких токенов
 * читает больше весов, и правило «веса читаются один раз на проход» для неё неверно. Так же подменяем модель,
 * которая не влезает в одну GPU. `sim` это модель, выбранная в симуляторе (для пояснения подмены).
 */
export function useRunningExample() {
  const [settings] = useSettings()
  return useMemo(() => {
    const m = findModel(settings.modelId) ?? getModel('qwen3-8b')
    const g = findGpu(settings.gpuId) ?? getGpu('h100-sxm')
    const fits = fitsCheck(m, g, { contextLen: 2048, seqs: 1 }).fits
    const moe = Boolean(m.experts)
    const fallback: RunningFallback = fits && !moe ? null : moe ? 'moe' : 'fit'
    return { model: fallback ? getModel('qwen3-8b') : m, gpu: g, sim: m, simFits: fits, fallback }
  }, [settings.modelId, settings.gpuId])
}
