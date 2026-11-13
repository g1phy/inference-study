import { useMemo, useState } from 'react'
import { Badge, ChapterLink, Figure, Segmented, Select, Slider, Toggle, fmt } from '@/ui'
import { cn } from '@/ui/cn'
import { useSettings } from '@/state/settings'
import { findModel, getModel } from '@/engine/data/models'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { VLLM_MEM_UTIL, fitsCheck, tpKvReplication, weightBytes } from '@/engine/perf'
import type { DType } from '@/engine/types'
import { LegendItem, Swatch } from '../batching/chart'

const MODEL_IDS = ['qwen3-8b', 'llama-3.1-8b', 'qwen3-32b', 'qwen3-30b-a3b', 'gpt-oss-120b', 'llama-3.1-70b', 'qwen3-235b-a22b', 'llama-3.1-405b', 'deepseek-v3', 'kimi-k2']
const GPU_IDS = ['a100-80g', 'h100-sxm', 'h200', 'b200', 'b300', 'l40s', 'mi300x']
const COUNTS = [1, 2, 4, 8, 16, 32] as const
/** Штриховка «веса сверх предела»: красная по фону полосы. */
const OVER_HATCH = 'repeating-linear-gradient(45deg, var(--bad) 0 3px, color-mix(in oklab, var(--bad) 35%, var(--surface)) 3px 6px)'
/** Размер для моделей, у которых его нет в названии. */
const SIZE_NOTE: Record<string, string> = { 'deepseek-v3': ' (671B)', 'kimi-k2': ' (1T)' }

export function FitsFigure() {
  const [settings] = useSettings()
  const [modelId, setModelId] = useState<string>('llama-3.1-70b')
  const [gpuId, setGpuId] = useState<string>(GPU_IDS.includes(settings.gpuId) ? settings.gpuId : 'h100-sxm')
  const [count, setCount] = useState(1)
  const [dtype, setDtype] = useState<DType>('bf16')
  const [kvDtype, setKvDtype] = useState<DType>('bf16')
  const [ctxExp, setCtxExp] = useState(13) // 8192
  const [seqs, setSeqs] = useState(32)
  const [dpAttn, setDpAttn] = useState(false)
  const model = findModel(modelId) ?? getModel('llama-3.1-70b')
  const gpu = findGpu(gpuId) ?? getGpu('h100-sxm')
  const contextLen = 2 ** ctxExp
  const isMla = model.attention === 'mla'

  const r = useMemo(
    () => fitsCheck(model, gpu, { contextLen, seqs, gpus: count, dtype, kvDtype, dpAttention: isMla && dpAttn, memUtil: VLLM_MEM_UTIL }),
    [model, gpu, contextLen, seqs, count, dtype, kvDtype, isMla, dpAttn],
  )
  const rep = tpKvReplication(model, count, isMla && dpAttn)
  const overflow = Math.max(0, r.weightBytes + r.reserveBytes + r.kvNeededBytes - r.processBytes)
  // что именно не влезло: сначала веса сверх предела, остальное KV
  const weightsOver = Math.max(0, r.weightBytes + r.reserveBytes - r.processBytes)
  const kvOver = Math.max(0, overflow - weightsOver)
  const scale = Math.max(r.processBytes, r.weightBytes + r.reserveBytes + r.kvNeededBytes)
  const pct = (b: number) => `${Math.max(0, (b / scale) * 100)}%`
  const status = !r.weightsFit ? 'bad' : !r.fits ? 'warn' : 'good'
  const gib = gpu.usableMiB ? gpu.usableMiB / 1024 : gpu.memGB

  return (
    <Figure
      id="parallelism-figure-fits"
      title="Влезет ли модель на N GPU"
      actions={
        <Badge tone={status}>{status === 'bad' ? 'веса не влезают' : status === 'warn' ? 'KV не влезает' : 'влезает'}</Badge>
      }
      controls={
        <>
          <Select
            label="Модель"
            value={modelId}
            onChange={(v) => {
              setModelId(v)
              setDpAttn(false)
            }}
            options={MODEL_IDS.map((id) => ({ value: id, label: `${getModel(id).name}${SIZE_NOTE[id] ?? ''}` }))}
          />
          <Select label="GPU" value={gpuId} onChange={setGpuId} options={GPU_IDS.map((id) => ({ value: id, label: `${getGpu(id).name}` }))} />
          <Segmented<number> label="GPU в tensor parallel" value={count} onChange={setCount} options={COUNTS.map((c) => ({ value: c, label: String(c) }))} />
          <Segmented<DType>
            label="Веса"
            value={dtype}
            onChange={setDtype}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp8', label: 'FP8' },
              { value: 'fp4', label: 'FP4' },
            ]}
          />
          <Segmented<DType>
            label="KV cache"
            value={kvDtype}
            onChange={setKvDtype}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp8', label: 'FP8' },
            ]}
          />
          <Slider label="Длина контекста" value={ctxExp} min={10} max={17} onChange={setCtxExp} format={(v) => fmt.tokens(2 ** v)} />
          <Slider label="Последовательностей" value={seqs} min={1} max={512} scale="log" step={1} onChange={setSeqs} />
          {isMla && <Toggle label="DP-attention" checked={dpAttn} onChange={setDpAttn} hint="Каждый GPU хранит KV только своих запросов" />}
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> Llama-3.1-70B в BF16 на одном H100 не влезает даже весами. Поставьте 2 GPU: веса
          помещаются, но под KV почти ничего не остаётся; 4 GPU дают место для десятков запросов. У DeepSeek-V3 в FP8 на 8 × H200 включите
          DP-attention: KV перестаёт дублироваться, и последовательностей влезает в 8 раз больше.
          Расчёт упрощённый: движку отдаётся 92 % памяти GPU (<code>gpu_memory_utilization</code> в vLLM по умолчанию, см. главу{' '}
          <ChapterLink to="kv-cache" />), минус 2 ГиБ резерва на GPU, минус веса.
        </>
      }
    >
      <div className="grid gap-3">
        <div aria-hidden className={cn('relative', overflow > 0 && 'pt-5')}>
          <div className="flex h-9 w-full overflow-hidden rounded-control bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]">
            <span className="h-full" style={{ width: pct(Math.min(r.weightBytes, r.processBytes - r.reserveBytes)), background: 'var(--ink-2)' }} />
            <span className="bg-hatch h-full" style={{ width: pct(r.reserveBytes) }} />
            <span className="flex h-full flex-col" style={{ width: pct(Math.min(Math.max(0, r.kvNeededBytes), Math.max(0, r.kvBudgetBytes))) }}>
              <span className="flex-1" style={{ background: 'var(--c-k)' }} />
              <span className="flex-1" style={{ background: 'var(--c-v)' }} />
            </span>
            <span className="h-full" style={{ width: pct(Math.max(0, r.freeBytes)) }} />
            {weightsOver > 0 && <span className="h-full" style={{ width: pct(weightsOver), background: OVER_HATCH }} />}
            {kvOver > 0 && <span className="h-full" style={{ width: pct(kvOver), background: 'var(--bad)' }} />}
          </div>
          {overflow > 0 && (
            <>
              <span className="absolute bottom-0 w-0.5 bg-ink" style={{ left: `calc(${pct(r.processBytes)} - 1px)`, top: '1rem' }} />
              <span
                className="absolute top-0 text-xs whitespace-nowrap text-ink-2 tabular-nums"
                style={r.processBytes / scale > 0.5 ? { right: `calc(${100 - (r.processBytes / scale) * 100}% + 4px)` } : { left: `calc(${pct(r.processBytes)} + 4px)` }}
              >
                предел {fmt.bytes(r.processBytes)}
              </span>
            </>
          )}
        </div>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
          <LegendItem swatch={<Swatch color="var(--ink-2)" />}>веса {fmt.bytes(r.weightBytes)}</LegendItem>
          <LegendItem swatch={<Swatch hatch />}>резерв {fmt.bytes(r.reserveBytes)}</LegendItem>
          <LegendItem
            swatch={
              <span className="inline-flex h-3 w-4 flex-col overflow-hidden rounded-[3px]">
                <span className="flex-1" style={{ background: 'var(--c-k)' }} />
                <span className="flex-1" style={{ background: 'var(--c-v)' }} />
              </span>
            }
          >
            KV для {fmt.int(seqs)} × {fmt.tokens(contextLen)}: {fmt.bytes(r.kvNeededBytes)}
          </LegendItem>
          <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px] bg-surface-2 shadow-[inset_0_0_0_1px_var(--line-strong)]" />}>
            {r.freeBytes >= 0 ? `свободно ${fmt.bytes(r.freeBytes)}` : 'свободного места нет'}
          </LegendItem>
          {weightsOver > 0 && (
            <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px]" style={{ background: OVER_HATCH }} />}>веса сверх предела {fmt.bytes(weightsOver)}</LegendItem>
          )}
          {kvOver > 0 && <LegendItem swatch={<Swatch color="var(--bad)" />}>KV сверх предела {fmt.bytes(kvOver)}</LegendItem>}
        </ul>
        <div className="grid gap-x-6 gap-y-2 text-[15px] @md:grid-cols-3">
          <div>
            <div className="text-sm text-ink-2">{overflow > 0 ? 'Память движку (до отметки)' : 'Память движку (вся полоса)'}</div>
            <div className="font-semibold text-ink tabular-nums">
              {fmt.bytes(r.processBytes)} <span className="text-sm font-normal text-ink-3">из {fmt.bytes(r.totalBytes)}</span>
            </div>
            <div className="text-[13px] leading-snug text-ink-3">
              nvidia-smi показывает {count > 1 ? `${count} × ` : ''}
              {fmt.num(gib, 1)} ГиБ, в ГБ по 10⁹ байт это {fmt.bytes(r.totalBytes)}
            </div>
          </div>
          <div>
            <div className="text-sm text-ink-2">KV на одну последовательность</div>
            <div className="font-semibold text-ink tabular-nums">
              {fmt.bytes(r.kvPerSeqBytes)}
              {rep > 1 && <span className="ml-1.5 text-sm font-normal text-ink-3">с учётом дублей ×{fmt.num(rep)}</span>}
            </div>
          </div>
          <div>
            <div className="text-sm text-ink-2">Влезет последовательностей</div>
            <div className="font-semibold text-ink tabular-nums">{fmt.int(r.maxSeqs)}</div>
          </div>
        </div>
        <p className="text-[15px] text-ink" aria-live="polite">
          {r.message}
          {rep > 1 &&
            !isMla &&
            ` У модели ${model.kvHeads} ${fmt.plural(model.kvHeads, ['KV-голова', 'KV-головы', 'KV-голов'])}, а GPU ${count}: каждая голова хранится на ${fmt.num(rep)} GPU сразу, и KV занимает в ${fmt.num(rep)} ${fmt.plural(rep, ['раз', 'раза', 'раз'])} больше места.`}
          {rep > 1 && isMla && ` У MLA один общий вектор на токен, при TP${count} он целиком лежит на каждом GPU: KV ×${count}. DP-attention это убирает.`}
        </p>
        <p className="text-sm text-ink-3">
          Веса {model.name} в {dtype.toUpperCase()}: {fmt.bytes(weightBytes(model, dtype))}; на один GPU {fmt.bytes(weightBytes(model, dtype) / count)}.
        </p>
      </div>
    </Figure>
  )
}
