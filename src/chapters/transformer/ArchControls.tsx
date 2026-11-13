import { Select, Slider, fmt } from '@/ui'
import { getModel } from '@/engine/data/models'
import { ARCH_MODEL_IDS } from './arch'

const DENSE = ['qwen3-8b', 'qwen3-32b', 'llama-3.1-8b', 'llama-3.1-70b']

/** Выбор модели и длины контекста (общие для фигур главы). */
export function ArchControls({
  modelId,
  onModel,
  ctx,
  onCtx,
  maxCtx,
}: {
  modelId: string
  onModel: (id: string) => void
  ctx: number
  onCtx: (n: number) => void
  maxCtx: number
}) {
  return (
    <>
      <Select
        label="Модель"
        value={modelId}
        onChange={onModel}
        options={ARCH_MODEL_IDS.map((id) => ({
          value: id,
          label: getModel(id).name,
          group: DENSE.includes(id) ? 'Плотные' : 'MoE',
        }))}
      />
      <Slider
        label="Контекст: сколько токенов уже позади"
        value={Math.min(ctx, maxCtx)}
        min={16}
        max={maxCtx}
        step={16}
        scale="log"
        onChange={onCtx}
        format={(v) => fmt.int(v)}
      />
    </>
  )
}
