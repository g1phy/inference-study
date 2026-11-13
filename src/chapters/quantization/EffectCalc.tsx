import { useMemo, useState } from 'react'
import { InfoIcon, XCircleIcon } from '@phosphor-icons/react'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { findModel, getModel } from '@/engine/data/models'
import { decodeStepMs, fitsCheck, maxConcurrentSeqs, prefillMs, weightBytes } from '@/engine/perf'
import type { DType } from '@/engine/types'
import { useSettings } from '@/state/settings'
import { Figure, Segmented, Select, Slider, Stat, cn, cssVar, fmt, tint } from '@/ui'

const MODEL_IDS = ['qwen3-8b', 'qwen3-32b', 'qwen3-30b-a3b', 'llama-3.1-8b', 'llama-3.1-70b', 'qwen3-235b-a22b', 'gpt-oss-120b'] as const
const GPU_IDS = ['h100-sxm', 'h200', 'b200', 'a100-80g', 'l40s'] as const
const WDT: Array<{ value: DType; label: string }> = [
  { value: 'bf16', label: 'BF16' },
  { value: 'fp8', label: 'FP8' },
  { value: 'fp4', label: 'FP4' },
]
const KDT: Array<{ value: DType; label: string }> = [
  { value: 'bf16', label: 'BF16' },
  { value: 'fp8', label: 'FP8' },
]
const DT_NAME: Record<DType, string> = { bf16: 'BF16', fp8: 'FP8', fp4: 'FP4 (NVFP4)' }
/** Доля памяти под движок как у vLLM по умолчанию с v0.20 (SPEC §8.8; так же в главах kv-cache и engines). */
const MEM_UTIL = 0.92

interface Row {
  w: DType
  kv: DType
  weightBytes: number
  fits: boolean
  itl: number
  maxSeqs: number
  throughput: number
  ttft: number
}

function evaluate(modelId: string, gpuId: string, gpus: number, ctx: number, w: DType, kv: DType): Row {
  const m = getModel(modelId)
  const g = getGpu(gpuId)
  const opts = { dtype: w, kvDtype: kv, gpus }
  const maxSeqs = maxConcurrentSeqs(m, g, { ...opts, contextLen: ctx, memUtil: MEM_UTIL })
  const fits = maxSeqs > 0
  const itl = decodeStepMs(m, g, { ...opts, batch: 1, avgContext: ctx })
  const big = Math.max(1, maxSeqs)
  const throughput = fits ? (big * 1000) / decodeStepMs(m, g, { ...opts, batch: big, avgContext: ctx }) : 0
  const ttft = prefillMs(m, g, Math.min(ctx, 4096), opts)
  return { w, kv, weightBytes: weightBytes(m, w), fits, itl, maxSeqs, throughput, ttft }
}

export function EffectCalc() {
  const [settings] = useSettings()
  const [modelId, setModelId] = useState<string>(() => (MODEL_IDS as readonly string[]).includes(settings.modelId) ? settings.modelId : 'qwen3-8b')
  const [gpuId, setGpuId] = useState<string>(() => (GPU_IDS as readonly string[]).includes(settings.gpuId) ? settings.gpuId : 'h100-sxm')
  const [gpus, setGpus] = useState<number>(1)
  const [w, setW] = useState<DType>('bf16')
  const [kv, setKv] = useState<DType>('bf16')
  const [ctx, setCtx] = useState(8192)

  const model = findModel(modelId) ?? getModel('qwen3-8b')
  const gpu = findGpu(gpuId) ?? getGpu('h100-sxm')
  const rows = useMemo(() => WDT.flatMap(({ value: ww }) => KDT.map(({ value: kk }) => evaluate(modelId, gpuId, gpus, ctx, ww, kk))), [modelId, gpuId, gpus, ctx])
  const cur = rows.find((r) => r.w === w && r.kv === kv)!
  const base = rows.find((r) => r.w === 'bf16' && r.kv === 'bf16')!
  const fit = useMemo(
    () => fitsCheck(model, gpu, { contextLen: ctx, seqs: Math.max(1, cur.maxSeqs), dtype: w, kvDtype: kv, gpus, memUtil: MEM_UTIL }),
    [model, gpu, ctx, cur.maxSeqs, w, kv, gpus],
  )
  const noFp4 = w === 'fp4' && gpu.tflops.fp4 === undefined
  const noFp8 = w === 'fp8' && gpu.tflops.fp8 === undefined

  return (
    <Figure
      id="quantization-figure-effect"
      title="Что даёт квантизация на конкретной модели и GPU"
      controlsPosition="side"
      controls={
        <>
          <Select label="Модель" value={modelId} onChange={setModelId} options={MODEL_IDS.map((id) => ({ value: id, label: getModel(id).name }))} />
          <Select label="GPU" value={gpuId} onChange={setGpuId} options={GPU_IDS.map((id) => ({ value: id, label: getGpu(id).name }))} />
          <Segmented label="Число GPU (TP)" value={gpus} onChange={setGpus} options={[1, 2, 4, 8].map((g) => ({ value: g, label: String(g) }))} />
          <Segmented label="Веса" value={w} onChange={setW} options={WDT} />
          <Segmented label="KV cache" value={kv} onChange={setKv} options={KDT} />
          <Slider label="Длина контекста" value={ctx} min={1024} max={131072} step={256} scale="log" onChange={setCtx} format={(v) => fmt.tokens(v)} />
        </>
      }
      caption="Попробуйте: выберите Qwen3-32B на одной H100. В BF16 веса занимают почти всю память, и запросов по 8K влезает всего пять; FP8 веса и FP8 KV дают 40 запросов и вдвое более быстрый decode. На практике шаг вниз по точности даёт +30-50 %, а не ×2: двукратный выигрыш здесь это потолок по полосе памяти."
      source="Числа из roofline-модели движка: MBU 0,7, MFU 0,5, резерв 2 ГиБ (2,15 ГБ) на GPU под активации и CUDA graphs, gpu_memory_utilization 0,92 (как в vLLM с v0.20). FP8 и FP4 здесь считаются как W8A8 и W4A4: если у GPU есть тензорные ядра этого формата, в нём идут и активации, поэтому колонка «Prefill 4K» ускоряется. Если таких ядер нет или сжаты только веса (weight-only), prefill остаётся как в BF16."
    >
      <div className="grid min-w-0 gap-5">
        <MemoryBar fit={fit} model={model.name} gpus={gpus} gpuName={gpu.name} memGB={gpu.memGB} />
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
          <Stat label="Веса" value={fmt.bytes(cur.weightBytes)} hint={DT_NAME[w]} />
          <Stat
            label="Decode, 1 запрос"
            value={cur.fits ? fmt.rate(1000 / cur.itl) : 'н/д'}
            swatch="var(--c-decode)"
            hint={cur.fits ? `${fmt.ms(cur.itl)} на токен, ${fmt.multiplier(base.itl / cur.itl, 1)} к BF16` : 'не влезает'}
          />
          <Stat label="Запросов влезает" value={cur.fits ? fmt.int(cur.maxSeqs) : '0'} hint={`по ${fmt.tokens(ctx)}`} />
          <Stat label="Throughput" value={cur.fits ? fmt.rate(cur.throughput) : 'н/д'} hint="при максимальном batch" />
        </div>
        {(noFp4 || noFp8) && (
          <p className="flex gap-2 rounded-control bg-surface-2 px-3 py-2 text-[14px] leading-snug text-ink-2">
            <InfoIcon size={18} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
            У {gpu.name} нет тензорных ядер {noFp4 ? 'FP4' : 'FP8'}: веса хранятся сжатыми, но перед умножением распаковываются в BF16 (weight-only).
            Память и decode выигрывают, prefill нет.
          </p>
        )}
        <div className="min-w-0 overflow-x-auto">
          <table className="w-full min-w-[260px] text-left text-[13.5px]">
            <thead className="text-[12px] text-ink-3">
              <tr className="border-b border-line">
                <th className="py-1.5 pr-3 font-medium whitespace-nowrap">Веса / KV</th>
                <th className="hidden py-1.5 pr-3 text-right font-medium @xl:table-cell">Веса</th>
                <th className="py-1.5 pr-3 text-right font-medium">Decode, ток/с</th>
                <th className="py-1.5 pr-3 text-right font-medium">Запросов</th>
                <th className="hidden py-1.5 pr-3 text-right font-medium @md:table-cell">Throughput, ток/с</th>
                <th className="hidden py-1.5 text-right font-medium @xl:table-cell">Prefill 4K</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {rows.map((r) => {
                const active = r.w === w && r.kv === kv
                return (
                  <tr
                    key={`${r.w}-${r.kv}`}
                    className={cn('cursor-pointer border-b border-line last:border-0 transition-colors', active ? 'bg-accent-soft' : 'hover:bg-surface-3/50')}
                    onClick={() => {
                      setW(r.w)
                      setKv(r.kv)
                    }}
                  >
                    <td className="py-1 pr-3 pl-1">
                      <button
                        type="button"
                        aria-pressed={active}
                        onClick={(e) => {
                          e.stopPropagation()
                          setW(r.w)
                          setKv(r.kv)
                        }}
                        className="rounded-[4px] px-0.5 font-mono text-[13px] whitespace-nowrap text-ink focus-visible:-outline-offset-1"
                      >
                        {DT_NAME[r.w].split(' ')[0]} / {DT_NAME[r.kv]}
                      </button>
                    </td>
                    <td className="hidden py-1.5 pr-3 text-right whitespace-nowrap text-ink-2 @xl:table-cell">{fmt.bytes(r.weightBytes)}</td>
                    <td className="py-1.5 pr-3 text-right whitespace-nowrap text-ink">{r.fits ? fmt.int(Math.round(1000 / r.itl)) : 'не влезает'}</td>
                    <td className="py-1.5 pr-3 text-right whitespace-nowrap text-ink">{fmt.int(r.maxSeqs)}</td>
                    <td className="hidden py-1.5 pr-3 text-right whitespace-nowrap text-ink @md:table-cell">{r.fits ? fmt.int(Math.round(r.throughput)) : 'н/д'}</td>
                    <td className="hidden py-1.5 text-right whitespace-nowrap text-ink-2 @xl:table-cell">{r.fits ? fmt.ms(r.ttft) : 'н/д'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Figure>
  )
}

/**
 * Полоса памяти в той же раскладке, что калькулятор в главе KV cache: веса, резерв штриховкой, KV двумя полосками
 * (K сверху, V снизу), невыданная движку доля с рамкой; если веса не влезают, справа красная плашка с нехваткой.
 */
function MemoryBar({ fit, model, gpus, gpuName, memGB }: { fit: ReturnType<typeof fitsCheck>; model: string; gpus: number; gpuName: string; memGB: number }) {
  const total = fit.totalBytes
  const kv = Math.max(0, fit.kvBudgetBytes)
  const over = Math.max(0, -fit.kvBudgetBytes)
  const kvFill = `linear-gradient(to bottom, ${cssVar('k')} 0 50%, ${cssVar('v')} 50% 100%)`
  const parts = [
    { id: 'w', label: 'веса', bytes: Math.min(fit.weightBytes, fit.processBytes), className: '', style: { background: 'var(--ink-2)' } },
    { id: 'r', label: 'активации, CUDA graphs', bytes: fit.reserveBytes, className: 'bg-hatch', style: { backgroundColor: 'var(--surface-3)' } },
    { id: 'kv', label: 'KV cache', bytes: kv, className: '', style: { background: kvFill } },
    {
      id: 'f',
      label: 'не отдано движку',
      bytes: Math.max(0, total - fit.processBytes),
      className: '',
      style: { background: 'var(--surface-2)', boxShadow: 'inset 0 0 0 1px var(--line-strong)' },
    },
  ]
  return (
    <div className="min-w-0">
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="text-sm font-medium text-ink-2">
          Память {gpus > 1 ? `${gpus} × ${gpuName}` : gpuName}: {fmt.bytes(total)} ({fmt.bytes(total, { mode: 'binary' })}, по паспорту{' '}
          {gpus > 1 ? `${gpus} × ` : ''}
          {fmt.int(memGB)}&nbsp;ГБ)
        </span>
        <span className="text-[13px] text-ink-3">{model}</span>
      </div>
      <div className="flex min-w-0 items-stretch gap-2">
        <div className="flex h-7 min-w-0 flex-1 gap-[2px]" role="img" aria-label={fit.message}>
          {parts.map((p) =>
            p.bytes > 0 ? (
              <span
                key={p.id}
                className={cn('h-full min-w-[3px] rounded-[4px] transition-[flex-grow] duration-300', p.className)}
                style={{ flexGrow: p.bytes, flexBasis: 0, ...p.style }}
                title={`${p.label}: ${fmt.bytes(p.bytes)}`}
              />
            ) : null,
          )}
        </div>
        {over > 0 && (
          <div
            className="flex h-7 shrink-0 items-center gap-1.5 rounded-[4px] px-2 text-xs font-medium whitespace-nowrap text-ink"
            style={{ background: tint('bad', 16), boxShadow: `inset 0 0 0 1px ${cssVar('bad')}` }}
          >
            <XCircleIcon size={16} aria-hidden style={{ color: cssVar('bad') }} />+{fmt.bytes(over)}
          </div>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-[3px] bg-ink-2" />
          веса {fmt.bytes(fit.weightBytes)}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-[3px] bg-hatch" style={{ backgroundColor: 'var(--surface-3)' }} />
          резерв {fmt.bytes(fit.reserveBytes)}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-[3px]" style={{ background: kvFill }} />
          KV cache, K и V {kv > 0 ? fmt.bytes(kv) : 'нет места'}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-[3px] bg-surface-2 shadow-[inset_0_0_0_1px_var(--line-strong)]" />
          не отдано движку
        </span>
      </div>
      {over > 0 && (
        <p className="mt-2 flex items-center gap-1.5 text-[13.5px] text-ink">
          <XCircleIcon size={16} weight="fill" className="shrink-0 text-bad" aria-hidden />
          Веса не влезают: не хватает {fmt.bytes(over)}. Возьмите формат меньше, больше GPU или GPU с большей памятью.
        </p>
      )}
    </div>
  )
}
