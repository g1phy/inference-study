import { useMemo, useState } from 'react'
import { InfoIcon, WarningIcon } from '@phosphor-icons/react'
import { getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'
import { fitsCheck } from '@/engine/perf'
import type { DType } from '@/engine/types'
import { Button, Figure, MEDIA, Segmented, Select, Stat, Toggle, cn, fmt, seqColor, seqInk, useMediaQuery } from '@/ui'
import { coldStartStages, coldStartTotal, type ColdStartStageId, type EngineKind, type WeightSource } from './coldstartModel'
import { replicaGpus } from './planner'

const MODELS = ['qwen3-8b', 'qwen3-32b', 'llama-3.1-70b', 'qwen3-235b-a22b', 'deepseek-v3'] as const
const STAGE_ORDER: ColdStartStageId[] = ['node', 'image', 'download', 'load', 'compile', 'warmup']
const H100 = getGpu('h100-sxm')

/**
 * Этапы идут по порядку, поэтому цвет из последовательной шкалы: от светлого к тёмному (в тёмной теме наоборот).
 * Шкала начинается с середины, чтобы даже первый этап был заметен на фоне фигуры.
 */
const stageT = (id: ColdStartStageId) => 0.45 + (0.55 * STAGE_ORDER.indexOf(id)) / (STAGE_ORDER.length - 1)
const stageFill = (id: ColdStartStageId) => seqColor(stageT(id))
const stageInk = (id: ColdStartStageId) => seqInk(stageT(id))

export function fmtDuration(s: number): string {
  if (s < 1) return '< 1 с'
  if (s < 90) return `${fmt.int(Math.round(s))} с`
  if (s < 3600) return `${fmt.num(s / 60, s < 600 ? 1 : 0)} мин`
  return `${fmt.num(s / 3600, 1)} ч`
}

/** «1 × H100», «8 × H100», «2 узла × 8 H100». */
function gpuLabel(n: number): string {
  if (n <= 8) return `${n} × H100`
  const nodes = Math.ceil(n / 8)
  return `${fmt.count(nodes, ['узел', 'узла', 'узлов'])} × 8 H100`
}

export function ColdStart() {
  const [modelId, setModelId] = useState<string>('qwen3-8b')
  const [dtype, setDtype] = useState<DType>('bf16')
  const [source, setSource] = useState<WeightSource>('hf')
  const [warmPool, setWarmPool] = useState(false)
  const [imageCached, setImageCached] = useState(false)
  const [engine, setEngine] = useState<EngineKind>('vllm')
  const [compileCache, setCompileCache] = useState(false)
  const [showHow, setShowHow] = useState(false)
  const desk = useMediaQuery(MEDIA.desk)
  const tablet = useMediaQuery(MEDIA.tablet)

  const model = getModel(modelId)
  // GPU на реплику считаем для выбранного формата весов: FP8 вдвое меньше и может уместиться в меньший узел
  const fit = useMemo(() => replicaGpus(model, H100, 4096, dtype), [model, dtype])
  const fitFp8 = useMemo(() => replicaGpus(model, H100, 4096, 'fp8'), [model])
  const node8 = useMemo(() => fitsCheck(model, H100, { contextLen: 4096, seqs: 8, gpus: 8, dtype }), [model, dtype])
  const stages = useMemo(
    () => coldStartStages({ model, dtype, source, warmPool, imageCached, engine, compileCache, gpus: fit.gpus }),
    [model, dtype, source, warmPool, imageCached, engine, compileCache, fit.gpus],
  )
  const total = coldStartTotal(stages)
  const best = useMemo(
    () => coldStartTotal(coldStartStages({ model, dtype: 'fp8', source: 'node', warmPool: true, imageCached: true, engine: 'vllm', compileCache: true, gpus: fitFp8.gpus })),
    [model, fitFp8.gpus],
  )
  const biggest = stages.reduce((a, b) => (b.seconds > a.seconds ? b : a))
  const shownStages = stages.filter((s) => s.seconds > 0)
  const howVisible = tablet || showHow

  return (
    <Figure
      id="production-figure-coldstart"
      title="Из чего складывается холодный старт реплики"
      controlsPosition={desk ? 'side' : 'top'}
      controls={
        <>
          <Select label="Модель" value={modelId} onChange={setModelId} options={MODELS.map((id) => ({ value: id, label: getModel(id).name }))} />
          <Segmented
            label="Веса"
            value={dtype}
            onChange={setDtype}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp8', label: 'FP8' },
            ]}
          />
          <Segmented
            label="Откуда веса"
            value={source}
            onChange={setSource}
            options={[
              { value: 'hf', label: 'HF' },
              { value: 's3', label: 'S3' },
              { value: 'node', label: 'Кэш узла' },
            ]}
          />
          <Segmented
            label="Движок"
            value={engine}
            onChange={setEngine}
            options={[
              { value: 'vllm', label: 'vLLM' },
              { value: 'trtllm', label: 'TensorRT-LLM' },
            ]}
          />
          <Toggle label="Тёплый пул узлов" checked={warmPool} onChange={setWarmPool} />
          <Toggle label="Образ уже на узле" checked={imageCached} onChange={setImageCached} />
          <Toggle label="Кэш компиляции" checked={compileCache} onChange={setCompileCache} hint="та же GPU, CUDA и версии" />
        </>
      }
      caption="Попробуйте: возьмите Llama-3.1-70B с Hugging Face, а потом включите FP8, кэш весов на узле и кэш компиляции."
      source="Время скачивания весов посчитано из их размера. Полосы источников и длительность остальных этапов условные: их задаёт ваша инфраструктура."
    >
      <div className="grid min-w-0 gap-5">
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-3">
          <Stat label="Холодный старт" value={fmtDuration(total)} size="lg" />
          <Stat label="Дольше всего" value={biggest.label} hint={fmtDuration(biggest.seconds)} />
          <Stat label="Если сократить всё" value={fmtDuration(best)} hint={`тёплый узел, FP8, веса на узле, кэш (${gpuLabel(fitFp8.gpus)})`} />
        </div>
        <div className="min-w-0">
          <div
            className="flex h-10 w-full gap-[2px]"
            role="img"
            aria-label={`Холодный старт ${fmtDuration(total)}: ${stages.map((s) => `${s.label} ${fmtDuration(s.seconds)}`).join(', ')}`}
          >
            {shownStages.map((s) => (
              <span
                key={s.id}
                className="flex h-full min-w-[3px] items-center justify-center overflow-hidden rounded-[4px] px-1 text-[12px] font-medium whitespace-nowrap tabular-nums transition-[flex-grow] duration-300"
                style={{
                  flexGrow: s.seconds,
                  flexBasis: 0,
                  background: stageFill(s.id),
                  color: stageInk(s.id),
                  ...(s.id === biggest.id ? { outline: '2px solid var(--ink)', outlineOffset: '1px' } : null),
                }}
                title={`${s.label}: ${fmtDuration(s.seconds)}`}
              >
                {s.seconds / total > 0.09 ? fmtDuration(s.seconds) : ''}
              </span>
            ))}
          </div>
          <p className="mt-2 text-[12.5px] text-ink-3">В рамке самый долгий этап; чем позже этап, тем сильнее его цвет выделяется на фоне.</p>
        </div>
        {(fit.gpus > 8 || !fit.fits) && (
          <p className="flex gap-2 rounded-control bg-surface-2 px-3 py-2 text-[14px] leading-snug text-ink-2">
            <WarningIcon size={18} weight="fill" className="mt-0.5 shrink-0 text-warn" aria-hidden />
            <span>
              {model.name} в {dtype === 'fp8' ? 'FP8' : 'BF16'} не помещается в один узел 8 × H100 (около {fmt.bytes(node8.processBytes)} памяти под движок), поэтому
              реплика занимает {fit.fits ? gpuLabel(fit.gpus) : 'больше четырёх узлов'}: узлы нужно получить, а веса загрузить на все GPU.
              {modelId === 'deepseek-v3' ? ' На практике DeepSeek-V3 в FP8 чаще ставят на один узел 8 × H200 или B200, а BF16 (апкаст из родного FP8) на четыре узла H100.' : ''}
            </span>
          </p>
        )}
        <div className="grid min-w-0 gap-2.5">
          <ol className="grid min-w-0 gap-2.5">
            {stages.map((s, i) => (
              <li key={s.id} className={cn('grid min-w-0 grid-cols-[18px_minmax(0,1fr)_auto] items-start gap-x-3 gap-y-0.5', s.seconds === 0 && 'opacity-60')}>
                <span className="mt-1 h-3.5 w-3.5 rounded-[4px] shadow-[inset_0_0_0_1px_var(--line)]" style={{ background: stageFill(s.id) }} aria-hidden />
                <span className="min-w-0">
                  <span className={cn('text-[14.5px] text-ink', s.id === biggest.id ? 'font-semibold' : 'font-medium')}>
                    {i + 1}. {s.label}
                    {s.id === biggest.id && s.seconds > 0 && <span className="ml-1.5 text-[12.5px] font-normal text-ink-2">дольше всего</span>}
                  </span>
                  <span className="block text-[13px] leading-snug text-ink-2">{s.detail}</span>
                  {howVisible && <span className="block text-[13px] leading-snug text-ink-3">Сокращают: {s.shorten}</span>}
                </span>
                <span className="font-mono text-[13px] text-ink tabular-nums">{s.seconds === 0 ? '0 с' : fmtDuration(s.seconds)}</span>
              </li>
            ))}
          </ol>
          {!tablet && (
            <div>
              <Button size="sm" variant="secondary" aria-expanded={showHow} onClick={() => setShowHow((v) => !v)}>
                {showHow ? 'Скрыть, как сокращают' : 'Как сокращают каждый этап'}
              </Button>
            </div>
          )}
        </div>
        {engine === 'trtllm' && (
          <p className="flex gap-2 text-[13.5px] leading-snug text-ink-2">
            <InfoIcon size={18} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
            <span>
              В TensorRT-LLM до версии 1.0 модель сначала собирали командой <code className="font-mono text-[12.5px]">trtllm-build</code> в engine под
              конкретную GPU, и это занимало минуты. С 1.0 по умолчанию работает PyTorch-рантайм без отдельной сборки, а в ветке 1.4 старый бэкенд удалён.
            </span>
          </p>
        )}
      </div>
    </Figure>
  )
}
