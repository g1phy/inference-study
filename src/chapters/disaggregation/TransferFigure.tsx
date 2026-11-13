import { useMemo, useState } from 'react'
import { Figure, Formula, Segmented, Select, Slider, fmt, measureText, useElementSize, useFontsReady } from '@/ui'
import { cn } from '@/ui/cn'
import { useSettings } from '@/state/settings'
import { getModel } from '@/engine/data/models'
import { getInterconnect, KV_TRANSFER_LINK_IDS } from '@/engine/data/interconnects'
import { kvBytesForSeq, kvLayout, prefillMs, transferMs, weightBytes } from '@/engine/perf'
import type { GpuSpecExt } from '@/engine/data/gpus'
import type { ModelSpecExt } from '@/engine/data/models'
import { findGpu, getGpu } from '@/engine/data/gpus'
import type { DType } from '@/engine/types'
import { minTp } from '../batching/config'
import { usablePerGpu } from '../parallelism/plan'
import { useReaderPromptTokens } from './promptTokens'

/**
 * На скольких GPU и в каком формате считать prefill для сравнения: минимальный TP (до 8),
 * при котором помещаются веса и KV; если не помещается и в FP8, столько GPU, сколько нужно весам с запасом.
 */
function prefillSetup(model: ModelSpecExt, gpu: GpuSpecExt): { gpus: number; dtype: DType } {
  const bf = minTp(model, gpu, 'bf16')
  if (bf !== null) return { gpus: bf, dtype: 'bf16' }
  const f8 = minTp(model, gpu, 'fp8')
  if (f8 !== null) return { gpus: f8, dtype: 'fp8' }
  const need = Math.ceil((weightBytes(model, 'fp8') * 1.3) / usablePerGpu(gpu))
  return { gpus: 2 ** Math.ceil(Math.log2(Math.max(1, need))), dtype: 'fp8' }
}

const MODEL_IDS = ['qwen3-8b', 'llama-3.1-8b', 'qwen3-32b', 'llama-3.1-70b', 'qwen3-235b-a22b', 'deepseek-v3', 'gpt-oss-120b'] as const

/** Короткие подписи каналов для строк калькулятора. */
const LINK_LABEL: Record<string, string> = {
  nvlink5: 'NVLink 5',
  nvlink4: 'NVLink 4',
  'ib-ndr-x8': '8 × InfiniBand NDR',
  'ib-xdr': 'InfiniBand XDR 800G',
  'ib-ndr': 'InfiniBand NDR 400G',
  'roce-400': 'RoCE 400GbE',
  pcie5: 'PCIe Gen5 x16',
  'tcp-25g': 'TCP 25 Гбит/с',
}

const LINK_NOTE: Record<string, string> = {
  nvlink5: 'внутри ноды B200',
  nvlink4: 'внутри ноды H100',
  'ib-ndr-x8': 'вся нода',
  'ib-xdr': 'одна сетевая карта',
  'ib-ndr': 'одна сетевая карта',
  'roce-400': 'RDMA поверх Ethernet',
  pcie5: 'через память хоста',
  'tcp-25g': 'без RDMA',
}

/** Ось времени: от 1 мкс до 100 с, логарифмическая, одна и та же при любых настройках. */
const T_MIN = 1e-3
const T_MAX = 1e5
const DECADES = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
const pos = (ms: number) => (Math.log10(Math.min(T_MAX, Math.max(T_MIN, ms))) - Math.log10(T_MIN)) / (Math.log10(T_MAX) - Math.log10(T_MIN))
const PRESET_TOKENS = [8192, 32768, 131072] as const
const VALUE_FONT = '600 13px "Onest Variable", sans-serif'

/** Число для KaTeX: десятичная запятая без лишнего пробела, тонкий пробел в тысячах. */
function texStr(s: string): string {
  return s.replace(/,/g, '{,}').replace(/\s/g, '\\,')
}
function texNum(v: number, digits?: number): string {
  return texStr(fmt.num(v, digits))
}

export function TransferFigure() {
  const [settings] = useSettings()
  const initial = (MODEL_IDS as readonly string[]).includes(settings.modelId) ? settings.modelId : 'qwen3-8b'
  const [modelId, setModelId] = useState<string>(initial)
  const [tokens, setTokens] = useState(8192)
  const [kvDtype, setKvDtype] = useState<DType>('bf16')
  const reader = useReaderPromptTokens()
  const model = getModel(modelId)
  const gpu = findGpu(settings.gpuId) ?? getGpu('h100-sxm')
  const setup = useMemo(() => prefillSetup(model, gpu), [model, gpu])

  const data = useMemo(() => {
    const bytes = kvBytesForSeq(model, tokens, kvDtype)
    const prefill = prefillMs(model, gpu, tokens, { gpus: setup.gpus, dtype: setup.dtype, overheadMs: 0 })
    const rows = KV_TRANSFER_LINK_IDS.map((id) => {
      const l = getInterconnect(id)
      return { id, name: LINK_LABEL[id] ?? l.name, note: LINK_NOTE[id] ?? '', GBs: l.GBs, ms: transferMs(bytes, l.GBs) }
    }).sort((a, b) => b.GBs - a.GBs)
    return { bytes, prefill, rows }
  }, [model, tokens, kvDtype, gpu, setup])

  const lay = kvLayout(model, kvDtype)
  const prefillPos = pos(data.prefill)
  const ib = data.rows.find((r) => r.id === 'ib-ndr')!
  const [tableRef, { width: tableW }] = useElementSize<HTMLDivElement>()
  useFontsReady() // ширины подписей меряются шрифтом сайта: перерисовать, когда он загрузится
  // ширина полосы с точками: на широком экране минус колонки названия и доли (11,5rem + 7,5rem + промежутки)
  const wideRows = tableW >= 448
  const plotW = Math.max(120, wideRows ? tableW - 184 - 120 - 24 : tableW)
  const isReader = reader !== null && tokens === reader

  const perTokenSym = model.attention === 'mla' ? `L \\cdot d_{\\text{latent}} \\cdot b` : `2 \\cdot L \\cdot h_{kv} \\cdot d_{head} \\cdot b`
  const perTokenNums =
    model.attention === 'mla'
      ? `${model.layers} \\cdot ${model.mlaDim} \\cdot ${texNum(lay.bytesPerElem, 0)}`
      : `2 \\cdot ${model.layers} \\cdot ${model.kvHeads} \\cdot ${model.headDim} \\cdot ${texNum(lay.bytesPerElem, 0)}`
  const perTokenRes = `${perTokenNums}\\,\\text{Б} = ${texNum(lay.bytesPerTokenNaive / 1024)}\\,\\text{КиБ}`

  return (
    <Figure
      id="disaggregation-figure-transfer"
      title="Сколько весит KV cache промпта и сколько он едет"
      controls={
        <>
          <Select label="Модель" value={modelId} onChange={setModelId} options={MODEL_IDS.map((id) => ({ value: id, label: getModel(id).name }))} />
          <Slider label="Токенов в промпте" value={tokens} min={16} max={131072} scale="log" step={1} onChange={setTokens} format={(v) => fmt.tokens(v)} />
          <Segmented<DType>
            label="Формат KV cache"
            value={kvDtype}
            onChange={setKvDtype}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp8', label: 'FP8' },
            ]}
          />
          <div className="col-span-full grid gap-1.5">
            <div className="text-sm font-medium text-ink">Длина промпта</div>
            <div className="flex flex-wrap gap-1.5">
              {reader !== null && <PresetButton on={isReader} onClick={() => setTokens(Math.max(16, reader))} label={`ваш запрос: ${fmt.tokens(reader)}`} />}
              {PRESET_TOKENS.map((t) => (
                <PresetButton key={t} on={tokens === t} onClick={() => setTokens(t)} label={fmt.tokens(t)} />
              ))}
            </div>
          </div>
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> у Llama-3.1-70B двигайте длину промпта от 8 до 32 тысяч токенов. KV растёт
          линейно, а prefill быстрее (attention квадратичен), поэтому доля передачи падает; дольше самого prefill остаётся только TCP. Потом
          переключите KV на FP8: объём и время делятся пополам. Время передачи идеальное (полоса канала, без задержек и протокольных накладных);
          prefill посчитан для {setup.gpus > 1 ? `${setup.gpus} × ` : ''}
          {gpu.name}
          {setup.dtype === 'fp8' ? ' в FP8' : ''} при MFU 50 %.
        </>
      }
    >
      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-4">
        <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-2 text-[15px]">
          <div className="hidden @md:block">
            <Formula block tex={`\\text{KV на токен} = ${perTokenSym} = ${perTokenRes}`} />
          </div>
          <div className="grid gap-1 @md:hidden">
            <Formula block tex={`\\text{KV на токен} = ${perTokenSym}`} />
            <Formula block tex={`= ${perTokenRes}`} />
          </div>
          <div className="hidden @md:block">
            <Formula
              block
              tex={`\\text{KV промпта} = ${texNum(tokens, 0)} \\times \\text{KV на токен} ${model.slidingWindow ? '\\approx' : '='} ${texStr(fmt.bytes(data.bytes))}`}
            />
          </div>
          <div className="grid gap-1 @md:hidden">
            <Formula block tex={`\\text{KV промпта} = ${texNum(tokens, 0)} \\times \\text{KV на токен}`} />
            <Formula block tex={`${model.slidingWindow ? '\\approx' : '='} ${texStr(fmt.bytes(data.bytes))}`} />
          </div>
          {model.slidingWindow && (
            <p className="text-sm text-ink-2">
              У {model.name} половина слоёв смотрит только на последние {model.slidingWindow.window} токенов, поэтому весь KV промпта меньше, чем
              токены × KV на токен.
            </p>
          )}
          {model.attention === 'mla' && (
            <p className="text-sm text-ink-2">
              У {model.name} MLA: на слой хранится один сжатый вектор из {model.mlaDim} чисел вместо отдельных K и V для каждой головы.
            </p>
          )}
          {isReader && (
            <p className="text-sm text-ink-2">
              Это ваш запрос из симулятора вместе с шаблоном чата{settings.tools ? ' и описанием инструмента' : ''}. Короткий чат передаётся за доли
              миллисекунды; дорого обходятся длинные документы и история диалога.
            </p>
          )}
        </div>

        <div ref={tableRef} className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-1" role="table" aria-label="Время передачи KV по каналам">
          <div role="row" className="grid grid-cols-[minmax(0,1fr)] gap-x-3 text-xs text-ink-3 @md:grid-cols-[11.5rem_minmax(0,1fr)_7.5rem]">
            <span role="columnheader" className="hidden @md:block">
              Канал
            </span>
            <span role="columnheader" className="hidden @md:block">
              время передачи, логарифмическая шкала
            </span>
            <span role="columnheader" className="hidden text-right @md:block">
              от времени prefill
            </span>
          </div>
          {data.rows.map((r) => {
            const slower = r.ms > data.prefill
            const p = pos(r.ms)
            const text = fmt.ms(r.ms)
            const side = labelSide(p * plotW, measureText(text, VALUE_FONT), prefillPos * plotW, plotW)
            return (
              <div key={r.id} role="row" className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 py-1 @md:grid-cols-[11.5rem_minmax(0,1fr)_7.5rem]">
                <span role="cell" className="min-w-0">
                  <span className="block truncate text-sm font-medium text-ink">{r.name}</span>
                  <span className="block truncate text-xs text-ink-3">
                    {fmt.bandwidth(r.GBs)}, {r.note}
                  </span>
                </span>
                <span role="cell" className="relative order-3 col-span-2 h-6 @md:order-none @md:col-span-1">
                  <DecadeGrid />
                  <span aria-hidden className="absolute top-1/2 left-0 h-px -translate-y-1/2 bg-line-strong" style={{ width: `${p * 100}%` }} />
                  <span
                    aria-hidden
                    className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full"
                    style={{ left: `${p * 100}%`, background: slower ? 'var(--ink)' : 'var(--ink-2)' }}
                  />
                  <span
                    className="absolute top-1/2 -translate-y-1/2 text-[13px] font-semibold whitespace-nowrap text-ink"
                    style={side === 'right' ? { left: `calc(${p * 100}% + 7px)` } : { right: `calc(${(1 - p) * 100}% + 7px)` }}
                  >
                    {text}
                  </span>
                  <span aria-hidden className="absolute -inset-y-0.5 w-0 border-l-2 border-dashed" style={{ left: `${prefillPos * 100}%`, borderColor: 'var(--c-prefill)' }} />
                </span>
                <span role="cell" className="text-right text-sm tabular-nums text-ink-2">
                  {slower ? <span className="font-semibold text-ink">{fmt.multiplier(r.ms / data.prefill, 1)}</span> : fmt.pct(r.ms / data.prefill)}
                  <span className="@md:hidden"> {slower ? 'к prefill' : 'от prefill'}</span>
                </span>
              </div>
            )
          })}
          <div aria-hidden className="grid grid-cols-[minmax(0,1fr)] gap-x-3 @md:grid-cols-[11.5rem_minmax(0,1fr)_7.5rem]">
            <span className="hidden @md:block" />
            <span className="relative h-5">
              {DECADES.map((d, k) => (
                <span
                  key={d}
                  className={cn(
                    'absolute top-0.5 text-xs whitespace-nowrap text-ink-3 tabular-nums',
                    k === 0 ? 'translate-x-0' : k === DECADES.length - 1 ? '-translate-x-full' : '-translate-x-1/2',
                    !wideRows && k % 2 === 1 && 'hidden',
                  )}
                  style={{ left: `${pos(d) * 100}%` }}
                >
                  {d >= 1000 ? `${fmt.int(d / 1000)} с` : fmt.ms(d)}
                </span>
              ))}
            </span>
          </div>
          <div className="mt-1 flex items-center gap-2 text-sm text-ink-2">
            <span aria-hidden className="inline-block h-4 w-0 border-l-2 border-dashed" style={{ borderColor: 'var(--c-prefill)' }} />
            prefill этого промпта: <span className="font-semibold text-ink tabular-nums">{fmt.ms(data.prefill)}</span>
          </div>
        </div>
        <p className="text-sm text-ink-2">
          Пример расчёта: {fmt.bytes(data.bytes)} по одной карте InfiniBand NDR ({fmt.bandwidth(ib.GBs)}) идут {fmt.ms(ib.ms)}, это{' '}
          {fmt.pct(ib.ms / data.prefill)} от времени prefill. При tensor parallel каждый GPU отправляет свою долю KV по своему каналу, и время
          делится на число GPU.
        </p>
      </div>
    </Figure>
  )
}

/** Вертикальные линии на каждой декаде шкалы. */
function DecadeGrid() {
  return (
    <>
      {DECADES.map((d) => (
        <span key={d} aria-hidden className="absolute inset-y-0 w-px bg-grid" style={{ left: `${pos(d) * 100}%` }} />
      ))}
    </>
  )
}

/** Подпись справа от точки, если влезает и не наезжает на отметку prefill, иначе слева. */
function labelSide(x: number, w: number, marker: number, plotW: number): 'left' | 'right' {
  // запас на неточный замер шрифта: подпись не должна касаться пунктира prefill
  const pad = 8
  const rightOk = x + 7 + w + pad <= plotW && !(marker > x - 2 && marker < x + 7 + w + pad)
  if (rightOk) return 'right'
  const leftOk = x - 7 - w - pad >= 0 && !(marker > x - 7 - w - pad && marker < x + 2)
  if (leftOk) return 'left'
  return x + 7 + w <= plotW ? 'right' : 'left'
}

function PresetButton({ on, onClick, label }: { on: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={cn('h-8 rounded-control px-2.5 text-sm transition-colors', on ? 'bg-accent-soft text-accent-hover' : 'bg-surface-2 text-ink-2 hover:bg-surface-3')}
    >
      {label}
    </button>
  )
}
