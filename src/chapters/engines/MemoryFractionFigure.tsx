import { useMemo, useState, type CSSProperties } from 'react'
import { XCircleIcon } from '@phosphor-icons/react'
import { useSettings } from '@/state/settings'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'
import type { DType } from '@/engine/types'
import { DEFAULT_RESERVE_GIB, GiB, kvBytesPerToken, usableMemBytes, weightBytes } from '@/engine/perf'
import { Figure, Select, Slider, cssVar, fmt } from '@/ui'

const CHOICES: Array<{ value: string; label: string; model: string; dtype: DType }> = [
  { value: 'qwen3-8b', label: 'Qwen3-8B, BF16', model: 'qwen3-8b', dtype: 'bf16' },
  { value: 'qwen3-32b', label: 'Qwen3-32B, BF16', model: 'qwen3-32b', dtype: 'bf16' },
  { value: 'llama-70b-fp8', label: 'Llama-3.1-70B, FP8', model: 'llama-3.1-70b', dtype: 'fp8' },
]

type Part = 'weights' | 'reserve' | 'kv' | 'outside'

const PART_LABEL: Record<Part, string> = {
  weights: 'веса',
  reserve: 'активации и CUDA graphs',
  kv: 'KV cache',
  outside: 'не используется движком',
}

const WEIGHTS_BG = 'color-mix(in oklab, var(--ink) 55%, var(--surface))'
const RESERVE_BG = 'repeating-linear-gradient(135deg, var(--ink-3) 0 1.5px, var(--surface-2) 1.5px 5px)'
/** Красная штриховка поверх части, которая вылезла за долю из конфига. */
const OVER_HATCH = 'repeating-linear-gradient(45deg, var(--bad) 0 2.5px, transparent 2.5px 6px)'

function partStyle(p: Part, over = false): CSSProperties {
  let style: CSSProperties
  if (p === 'weights') style = { background: WEIGHTS_BG }
  else if (p === 'reserve') style = { background: RESERVE_BG }
  else if (p === 'kv') style = { background: `linear-gradient(to bottom, ${cssVar('k')} 0 50%, ${cssVar('v')} 50% 100%)` }
  else style = { background: 'transparent', boxShadow: 'inset 0 0 0 1px var(--line-strong)' }
  if (over) style = { background: `${OVER_HATCH}, ${String(style.background)}` }
  return style
}

interface Seg {
  part: Part
  bytes: number
  /** Часть за долей из конфига: не влезает. */
  over?: boolean
}

interface Row {
  engine: string
  param: string
  meaning: string
  parts: Seg[]
  kv: number
  /** Сколько не влезает в долю (веса и резерв за отметкой). */
  overBytes: number
  /** Веса сами по себе больше доли. */
  weightsOver: boolean
}

/**
 * Одна и та же доля памяти в трёх движках (notes-engines.md §1.2, §1.16, §1.25, пример книги с. 139).
 * Активации и CUDA graphs условно 2 ГиБ (perf.DEFAULT_RESERVE_GIB); веса и память GPU из engine/data.
 * Полоса каждого движка во всю память GPU; вертикальная отметка на всех полосах стоит на «доле от всей памяти»,
 * поэтому видно, где число из конфига значит одно и то же, а где другое.
 */
export function MemoryFractionFigure() {
  const [settings] = useSettings()
  const gpu = findGpu(settings.gpuId) ?? getGpu('h100-sxm')
  const [choice, setChoice] = useState('llama-70b-fp8')
  const [frac, setFrac] = useState(0.9)
  const c = CHOICES.find((x) => x.value === choice) ?? CHOICES[0]
  const model = getModel(c.model)

  const { total, rows, perToken } = useMemo(() => {
    const total = usableMemBytes(gpu)
    const w = weightBytes(model, c.dtype)
    const r = DEFAULT_RESERVE_GIB * GiB
    const clamp = (x: number) => Math.max(0, x)
    const F = frac * total
    // vLLM: веса, активации и KV вместе должны уложиться в долю от всей памяти
    const vReserveIn = clamp(Math.min(r, F - w))
    const vllmKv = F - w - r
    // SGLang: в долю укладываются веса и пул KV, активации живут в остатке
    const sgKv = F - w
    // TensorRT-LLM: доля от того, что свободно после весов и активаций, и только под KV
    const free = clamp(total - w - r)
    const trtKv = frac * free
    const rows: Row[] = [
      {
        engine: 'vLLM',
        param: 'gpu_memory_utilization',
        meaning: 'доля всей памяти GPU на инстанс: в неё входят веса, активации и KV',
        kv: vllmKv,
        overBytes: clamp(w + r - F),
        weightsOver: w > F,
        parts: [
          { part: 'weights', bytes: Math.min(w, F) },
          { part: 'weights', bytes: clamp(w - F), over: true },
          { part: 'reserve', bytes: vReserveIn },
          { part: 'reserve', bytes: w + r > F ? r - vReserveIn : 0, over: true },
          { part: 'kv', bytes: clamp(vllmKv) },
          { part: 'outside', bytes: clamp(total - Math.max(F, w + r)) },
        ],
      },
      {
        engine: 'SGLang',
        param: 'mem_fraction_static',
        meaning: 'доля всей памяти под веса и пул KV; активации живут в остатке',
        kv: sgKv,
        overBytes: clamp(w - F),
        weightsOver: w > F,
        parts: [
          { part: 'weights', bytes: Math.min(w, F) },
          { part: 'weights', bytes: clamp(w - F), over: true },
          { part: 'kv', bytes: clamp(sgKv) },
          { part: 'reserve', bytes: clamp(total - Math.max(F, w)) },
        ],
      },
      {
        engine: 'TensorRT-LLM',
        param: 'free_gpu_memory_fraction',
        meaning: 'доля памяти, свободной после загрузки модели, и только под KV',
        kv: trtKv,
        overBytes: 0,
        weightsOver: false,
        parts: [
          { part: 'weights', bytes: Math.min(w, total) },
          { part: 'reserve', bytes: Math.min(r, clamp(total - w)) },
          { part: 'kv', bytes: trtKv },
          { part: 'outside', bytes: (1 - frac) * free },
        ],
      },
    ]
    return { total, rows, perToken: kvBytesPerToken(model, 'bf16') }
  }, [gpu, model, c.dtype, frac])
  const markLeft = `${frac * 100}%`

  return (
    <Figure
      id="engines-figure-memory"
      title={`Одна и та же доля памяти в трёх движках: ${model.name} на ${gpu.name}`}
      width="wide"
      controls={
        <>
          <Select label="Модель" value={choice} onChange={setChoice} options={CHOICES.map((x) => ({ value: x.value, label: x.label }))} />
          <Slider label="Доля в конфиге" value={frac} min={0.5} max={0.95} step={0.01} onChange={setFrac} format={(v) => fmt.num(v, 2)} />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> у Llama-3.1-70B в FP8 поставьте долю 0,8. Вертикальная отметка на всех полосах стоит на
          80 % всей памяти: в vLLM и SGLang веса вылезают за неё (красная штриховка), и под KV места нет, а в TensorRT-LLM то же число значит 80 % того,
          что осталось после весов, и KV уходит за отметку. Активации и CUDA graphs условно {DEFAULT_RESERVE_GIB} ГиБ, память GPU по nvidia-smi, KV в BF16.
          GPU меняется в симуляторе.
        </>
      }
    >
      <div className="grid min-w-0 gap-5">
        <div className="flex flex-wrap gap-x-4 gap-y-1" aria-hidden>
          {(['weights', 'reserve', 'kv', 'outside'] as Part[]).map((p) => (
            <span key={p} className="inline-flex items-center gap-1.5 text-[13px] text-ink-2">
              <span className="h-3 w-3 rounded-[3px]" style={partStyle(p)} />
              {PART_LABEL[p]}
            </span>
          ))}
          <span className="inline-flex items-center gap-1.5 text-[13px] text-ink-2">
            <span className="h-3 w-3 rounded-[3px]" style={partStyle('weights', true)} />
            не влезает в долю
          </span>
          <span className="inline-flex items-center gap-1.5 text-[13px] text-ink-2">
            <span className="h-4 w-0.5 rounded-full bg-ink" />
            доля {fmt.num(frac, 2)} от всей памяти GPU
          </span>
        </div>
        {rows.map((row) => (
          <div key={row.engine} className="grid min-w-0 gap-1.5">
            <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className="text-[15px] font-semibold text-ink">{row.engine}</span>
                <code className="font-mono text-[12.5px] text-ink-2">
                  {row.param} = {String(frac.toFixed(2))}
                </code>
              </div>
              <span className="text-[14px] text-ink tabular-nums">
                {row.kv > 0 ? (
                  <>
                    KV: <span className="font-semibold">{fmt.bytes(row.kv)}</span>
                    <span className="text-ink-3"> · примерно {fmt.tokens(Math.floor(row.kv / perToken))}</span>
                  </>
                ) : (
                  <span className="inline-flex items-center gap-1.5">
                    <XCircleIcon size={16} aria-hidden style={{ color: cssVar('bad') }} />
                    {row.weightsOver ? 'веса не помещаются в долю' : 'на KV ничего не осталось'}
                    {row.overBytes > 0 && <span className="text-ink-3">· лишние {fmt.bytes(row.overBytes)}</span>}
                  </span>
                )}
              </span>
            </div>
            <div className="relative min-w-0">
              <div
                className="flex h-6 min-w-0 overflow-hidden rounded-[6px]"
                role="img"
                aria-label={`${row.engine}: ${row.parts
                  .filter((p) => p.bytes > 0)
                  .map((p) => `${p.over ? `не влезает (${PART_LABEL[p.part]})` : PART_LABEL[p.part]} ${fmt.bytes(p.bytes)}`)
                  .join(', ')}`}
              >
                {row.parts
                  .filter((p) => p.bytes > 0)
                  .map((p, i) => (
                    <span
                      key={i}
                      className="h-full"
                      style={{ width: `${(p.bytes / total) * 100}%`, ...partStyle(p.part, p.over) }}
                      title={p.over ? `не влезает: ${fmt.bytes(p.bytes)} (${PART_LABEL[p.part]})` : `${PART_LABEL[p.part]}: ${fmt.bytes(p.bytes)}`}
                    />
                  ))}
              </div>
              <span aria-hidden className="absolute -top-1 -bottom-1 w-0.5 -translate-x-1/2 rounded-full bg-ink" style={{ left: markLeft }} />
            </div>
            <span className="text-[13px] text-ink-3">{row.meaning}</span>
          </div>
        ))}
      </div>
    </Figure>
  )
}
