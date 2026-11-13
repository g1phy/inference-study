import { useEffect, useMemo, useState } from 'react'
import { findGpu } from '@/engine/data/gpus'
import { findModel } from '@/engine/data/models'
import type { DType } from '@/engine/types'
import { useSettings } from '@/state/settings'
import { Segmented, Select } from '@/ui'
import { DTYPE_LABEL, GPU_CHOICES, MODEL_CHOICES, dtypeSupported, makeHardware, resolveGpu, resolveModel, type Hardware } from './hw-core'

export * from './hw-core'
export { durationValueUnit, times, valueUnit } from './format'

/**
 * Общие для глав S4 (prefill-decode, roofline, metrics, cheatsheet) помощники:
 * выбор модели, GPU и формата весов, автоматический TP, если модель не помещается в один GPU.
 * По умолчанию всё берётся из настроек симулятора (useSettings), локальный выбор живёт в главе.
 */

export interface HardwareState {
  modelId: string
  gpuId: string
  dtype: DType
  setModelId: (id: string) => void
  setGpuId: (id: string) => void
  setDtype: (d: DType) => void
  hw: Hardware
}

/**
 * Локальный выбор модели, GPU и формата. Начальные значения и любые изменения настроек
 * симулятора подхватываются; локальный выбор остаётся в главе и настройки не меняет.
 */
export function useHardwareState(initialDtype: DType = 'bf16'): HardwareState {
  const [settings] = useSettings()
  const [modelId, setModelId] = useState(settings.modelId)
  const [gpuId, setGpuId] = useState(settings.gpuId)
  const [dtype, setDtype] = useState<DType>(initialDtype)
  useEffect(() => setModelId(settings.modelId), [settings.modelId])
  useEffect(() => setGpuId(settings.gpuId), [settings.gpuId])

  const hw = useMemo(() => {
    const model = resolveModel(modelId)
    const gpu = resolveGpu(gpuId)
    // формат без тензорных ядер на этом GPU: откатываемся к BF16, чтобы не путать читателя
    const d = dtypeSupported(gpu, dtype) ? dtype : 'bf16'
    return makeHardware(model, gpu, d)
  }, [modelId, gpuId, dtype])

  return { modelId, gpuId, dtype: hw.dtype, setModelId, setGpuId, setDtype, hw }
}

function withCurrent(ids: readonly string[], current: string, valid: (id: string) => boolean): string[] {
  return ids.includes(current) || !valid(current) ? [...ids] : [current, ...ids]
}

export function ModelSelect({ state, label = 'Модель' }: { state: HardwareState; label?: string }) {
  const ids = withCurrent(MODEL_CHOICES, state.modelId, (id) => Boolean(findModel(id)))
  return (
    <Select
      label={label}
      value={state.hw.model.id}
      onChange={state.setModelId}
      options={ids.map((id) => ({ value: id, label: resolveModel(id).name }))}
    />
  )
}

export function GpuSelect({ state, label = 'GPU' }: { state: HardwareState; label?: string }) {
  const ids = withCurrent(GPU_CHOICES, state.gpuId, (id) => Boolean(findGpu(id)))
  return (
    <Select
      label={label}
      value={state.hw.gpu.id}
      onChange={state.setGpuId}
      options={ids.map((id) => ({ value: id, label: resolveGpu(id).name }))}
    />
  )
}

export function DtypeSegmented({ state, label = 'Формат весов' }: { state: HardwareState; label?: string }) {
  const gpu = state.hw.gpu
  return (
    <Segmented<DType>
      label={label}
      value={state.hw.dtype}
      onChange={state.setDtype}
      options={(['bf16', 'fp8', 'fp4'] as const).map((d) => ({
        value: d,
        label: DTYPE_LABEL[d],
        disabled: !dtypeSupported(gpu, d),
        title: dtypeSupported(gpu, d) ? undefined : `У ${gpu.name} нет тензорных ядер ${DTYPE_LABEL[d]}`,
      }))}
    />
  )
}

/** Короткая пометка, если модель пришлось разложить на несколько GPU. */
export function TpNote({ hw }: { hw: Hardware }) {
  if (hw.tp <= 1) return null
  return (
    <p className="text-[13px] leading-snug text-ink-3">
      {hw.model.name} в {DTYPE_LABEL[hw.dtype]} не помещается в один {hw.gpu.name} с запасом под KV cache, поэтому считаем на {hw.tp} GPU
      (tensor parallel, идеализированно: без затрат на обмен между GPU).
    </p>
  )
}

