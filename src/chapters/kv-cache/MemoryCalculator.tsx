import { useDeferredValue, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { CheckCircleIcon, WarningIcon, XCircleIcon } from '@phosphor-icons/react'
import type { DType } from '@/engine/types'
import { kvBytesPerToken, maxConcurrentSeqs } from '@/engine/perf'
import { DEFAULT_BLOCK_SIZE } from '@/engine/kv'
import { useSettings } from '@/state/settings'
import { logDomain } from '@/ui/chart-utils'
import {
  ChapterLink,
  CodeBlock,
  Figure,
  LineChart,
  Segmented,
  Select,
  Slider,
  Stat,
  TipRow,
  cn,
  cssVar,
  fmt,
  measureText,
  tint,
  useElementSize,
  useTip,
} from '@/ui'
import {
  KV_GPU_IDS,
  KV_MODEL_IDS,
  enFixed,
  enInt,
  gpuOptions,
  gpuOr,
  logGrid,
  memoryLayout,
  minGpus,
  modelOptions,
  modelOr,
  pickGpuId,
  pickModelId,
  shortGpuName,
  type MemPartId,
} from './lib'
import { BackToSim, useFollowSim } from './followSim'

/** «1,21 ГБ» → { value: '1,21', unit: 'ГБ' } для Stat (fmt.bytes ставит неразрывный пробел). Ноль пишем «0 ГБ», а не «0 Б». */
function bytesParts(b: number): { value: string; unit: string } {
  if (b <= 0) return { value: '0', unit: 'ГБ' }
  const t = fmt.bytes(b)
  const i = t.lastIndexOf('\u00a0')
  return i > 0 ? { value: t.slice(0, i), unit: t.slice(i + 1) } : { value: t, unit: '' }
}

const PART_INFO: Record<MemPartId, { name: string; hint: string }> = {
  weights: { name: 'Веса модели', hint: 'загружаются один раз, размер не зависит от нагрузки' },
  reserve: { name: 'Активации и CUDA graphs', hint: 'рабочая память прохода модели; здесь упрощённо 2 ГиБ на GPU' },
  kvUsed: { name: 'KV cache: занято', hint: 'K и V всех одновременных последовательностей' },
  kvFree: { name: 'KV cache: свободно в пуле', hint: 'пул выделен при старте, но пока пуст' },
  outside: { name: 'Вне движка', hint: 'доля памяти, которую движок не берёт (1 − gpu_memory_utilization)' },
}

/** Заливка части полосы. KV рисуем двумя полосками: K сверху, V снизу. */
function partStyle(id: MemPartId): CSSProperties {
  switch (id) {
    case 'weights':
      return { background: 'var(--ink-2)' }
    case 'reserve':
      return { backgroundColor: 'var(--surface-3)' }
    case 'kvUsed':
      return { background: `linear-gradient(to bottom, ${cssVar('k')} 0 50%, ${cssVar('v')} 50% 100%)` }
    case 'kvFree':
      return {
        background: `linear-gradient(to bottom, ${tint('k', 22)} 0 50%, ${tint('v', 22)} 50% 100%)`,
        boxShadow: 'inset 0 0 0 1px var(--line-strong)',
      }
    case 'outside':
      return { background: 'var(--surface-2)', boxShadow: 'inset 0 0 0 1px var(--line)' }
  }
}

/** Цвет подписи внутри части (почти чёрный на насыщенных заливках в обеих темах). */
function partInk(id: MemPartId): string {
  if (id === 'weights') return 'var(--surface)'
  if (id === 'kvUsed') return '#121211'
  return 'var(--ink-2)'
}

function Swatch({ id }: { id: MemPartId }) {
  return <span aria-hidden className={cn('inline-block h-3 w-3 shrink-0 rounded-[3px]', id === 'reserve' && 'bg-hatch')} style={partStyle(id)} />
}

const LABEL_FONT = '500 12px "Onest Variable", sans-serif'

function MemoryBar({ layout }: { layout: ReturnType<typeof memoryLayout> }) {
  const [ref, size] = useElementSize<HTMLDivElement>()
  const rootRef = useRef<HTMLDivElement | null>(null)
  const tip = useTip({ scopeEl: rootRef.current, side: 'top' })
  const visible = layout.parts.filter((p) => p.share > 0.0004)
  const overflow = layout.overflowBytes > 0
  // полоса сжимается, если рядом нужно место под переполнение
  const showTip = (el: HTMLElement, content: ReactNode) => tip.show(el.getBoundingClientRect(), content)
  const gaps = (visible.length - 1) * 2

  return (
    <div ref={rootRef} className="grid min-w-0 gap-2">
      <div className="flex min-w-0 items-stretch gap-2">
        <div ref={ref} className="flex h-7 min-w-0 flex-1 gap-[2px]" role="list" aria-label="Память GPU по частям">
          {visible.map((p) => {
            const px = Math.max(0, (size.width - gaps) * p.share)
            const label = fmt.bytes(p.bytes)
            const fits = px >= measureText(label, LABEL_FONT) + 12
            const content = (
              <div className="grid gap-1">
                <div className="font-medium text-ink">{PART_INFO[p.id].name}</div>
                <TipRow value={fmt.bytes(p.bytes)} label={fmt.pct(p.share)} />
                <div className="text-xs text-ink-3">{PART_INFO[p.id].hint}</div>
              </div>
            )
            return (
              <div
                key={p.id}
                role="listitem"
                tabIndex={0}
                aria-label={`${PART_INFO[p.id].name}: ${fmt.bytes(p.bytes)}, ${fmt.pct(p.share)}`}
                className={cn(
                  'relative flex min-w-[3px] items-center overflow-hidden rounded-[4px] transition-[flex-grow] duration-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
                  p.id === 'reserve' && 'bg-hatch',
                )}
                style={{ ...partStyle(p.id), flexGrow: p.share, flexBasis: 0 }}
                onPointerEnter={(e) => showTip(e.currentTarget, content)}
                onPointerDown={(e) => showTip(e.currentTarget, content)}
                onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
                onFocus={(e) => showTip(e.currentTarget, content)}
                onBlur={() => tip.hide()}
              >
                {fits && (
                  <span className="pl-2 text-xs font-medium whitespace-nowrap tabular-nums" style={{ color: partInk(p.id) }}>
                    {label}
                  </span>
                )}
              </div>
            )
          })}
        </div>
        {overflow && (
          <div
            className="flex h-7 shrink-0 items-center gap-1.5 rounded-[4px] px-2 text-xs font-medium text-ink"
            style={{ background: tint('bad', 16), boxShadow: `inset 0 0 0 1px ${cssVar('bad')}` }}
          >
            <XCircleIcon size={16} aria-hidden style={{ color: cssVar('bad') }} />
            {layout.overflowKind === 'weights' ? 'не влезают веса' : 'не влезает KV'}: {fmt.bytes(layout.overflowBytes)}
          </div>
        )}
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        {layout.parts.map((p) => (
          <li key={p.id} className="inline-flex items-center gap-1.5">
            <Swatch id={p.id} />
            {PART_INFO[p.id].name}
          </li>
        ))}
      </ul>
      <table className="sr-only">
        <caption>Разбивка памяти GPU</caption>
        <tbody>
          {layout.parts.map((p) => (
            <tr key={p.id}>
              <th scope="row">{PART_INFO[p.id].name}</th>
              <td>{fmt.bytes(p.bytes)}</td>
              <td>{fmt.pct(p.share)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {tip.node}
    </div>
  )
}

/**
 * Сводка для узкой фигуры: на телефоне контролы стоят под всей фигурой, поэтому над ними
 * повторяем полосу памяти (без подписей) и итог одной строкой, чтобы причина и следствие были рядом.
 */
function CompactSummary({
  layout,
  tone,
  text,
}: {
  layout: ReturnType<typeof memoryLayout>
  tone: 'good' | 'warn' | 'bad'
  text: string
}) {
  const visible = layout.parts.filter((p) => p.share > 0.0004)
  const Icon = tone === 'good' ? CheckCircleIcon : tone === 'warn' ? WarningIcon : XCircleIcon
  return (
    <div className="grid gap-2 @min-[760px]/fig:hidden" aria-hidden>
      <div className="flex h-4 min-w-0 gap-[2px]">
        {visible.map((p) => (
          <div
            key={p.id}
            className={cn('min-w-[3px] rounded-[3px]', p.id === 'reserve' && 'bg-hatch')}
            style={{ ...partStyle(p.id), flexGrow: p.share, flexBasis: 0 }}
          />
        ))}
        {layout.overflowBytes > 0 && (
          <div className="w-4 shrink-0 rounded-[3px]" style={{ background: tint('bad', 30), boxShadow: `inset 0 0 0 1px ${cssVar('bad')}` }} />
        )}
      </div>
      <div className="flex items-start gap-1.5 text-[13px] leading-snug font-medium text-ink">
        <Icon size={16} className="mt-px shrink-0" style={{ color: cssVar(tone) }} />
        {text}
      </div>
    </div>
  )
}

export function MemoryCalculator() {
  const [settings] = useSettings()
  // модель и GPU следуют за симулятором, пока читатель не выбрал в калькуляторе свои
  const simModelId = pickModelId(settings.modelId, KV_MODEL_IDS)
  const simGpuId = pickGpuId(settings.gpuId, KV_GPU_IDS)
  const modelSel = useFollowSim(simModelId)
  const gpuSel = useFollowSim(simGpuId)
  const modelId = modelSel.id
  const gpuId = gpuSel.id
  const [dtype, setDtype] = useState<DType>('bf16')
  // Число GPU по умолчанию подбирается под модель, GPU и формат весов. Выбор читателя держится, пока они те же:
  // сменились (в калькуляторе или в симуляторе), и число снова подбирается само.
  const [gpusChoice, setGpusChoice] = useState<{ n: number; key: string } | null>(null)
  const [kvDtype, setKvDtype] = useState<DType>('bf16')
  const [ctx, setCtx] = useState(8192)
  const [seqs, setSeqs] = useState(32)
  const [memUtil, setMemUtil] = useState(0.92)

  const model = modelOr(modelId)
  const gpu = gpuOr(gpuId)
  const gpusKey = `${modelId}|${gpuId}|${dtype}`
  const autoGpus = useMemo(() => minGpus(model, gpu, dtype), [model, gpu, dtype])
  // выбор для прежней конфигурации забываем (setState во время рендера: React сразу перерисует компонент)
  if (gpusChoice && gpusChoice.key !== gpusKey) setGpusChoice(null)
  const gpus = gpusChoice && gpusChoice.key === gpusKey ? gpusChoice.n : autoGpus
  const dCtx = useDeferredValue(ctx)
  const layout = useMemo(
    () => memoryLayout(model, gpu, { contextLen: ctx, seqs, gpus, dtype, kvDtype, memUtil }),
    [model, gpu, ctx, seqs, gpus, dtype, kvDtype, memUtil],
  )
  const f = layout.fits
  const perToken = kvBytesPerToken(model, kvDtype)
  const tooLong = ctx > model.maxContext

  const curve = useMemo(() => {
    const pts: Array<[number, number]> = []
    for (const c of logGrid(512, 131072, 36)) {
      const n = maxConcurrentSeqs(model, gpu, { contextLen: c, gpus, dtype, kvDtype, memUtil })
      if (n > 0) pts.push([c, n])
    }
    return pts
  }, [model, gpu, gpus, dtype, kvDtype, memUtil])

  const status: { tone: 'good' | 'warn' | 'bad'; text: ReactNode; short: string } = !f.weightsFit
    ? {
        tone: 'bad',
        short: `Не влезают веса: не хватает ${fmt.bytes(layout.overflowBytes)}`,
        text: (
          <>
            Веса не помещаются: {fmt.bytes(f.weightBytes)} при {fmt.bytes(Math.max(0, f.processBytes - f.reserveBytes))} доступной
            памяти. Нужно больше GPU (<ChapterLink to="parallelism">tensor parallel</ChapterLink>) или{' '}
            <ChapterLink to="quantization">квантизация</ChapterLink>.
          </>
        ),
      }
    : tooLong
      ? {
          tone: 'bad',
          short: `Контекст длиннее, чем поддерживает ${model.name}`,
          text: (
            <>
              Такой max_model_len движок не примет: {model.name} поддерживает до {fmt.tokens(model.maxContext)}. Уменьшите длину
              контекста.
            </>
          ),
        }
    : !f.fits
      ? {
          tone: 'warn',
          short: `Не влезает KV: не хватает ${fmt.bytes(layout.overflowBytes)}, влезет ${fmt.int(f.maxSeqs)} из ${fmt.int(seqs)}`,
          text: (
            <>
              KV для {fmt.int(seqs)} {fmt.plural(seqs, ['последовательности', 'последовательностей', 'последовательностей'])} по{' '}
              {fmt.tokens(ctx)} не помещается: нужно {fmt.bytes(f.kvNeededBytes)}, в пуле{' '}
              {fmt.bytes(f.kvBudgetBytes)}. Сервер не упадёт: лишние запросы подождут в очереди, а если память кончится посреди генерации,
              движок вытеснит часть запросов (preemption) и пересчитает их позже.
            </>
          ),
        }
      : {
          tone: 'good',
          short: `Влезает: ${fmt.int(seqs)} из ${fmt.int(f.maxSeqs)} возможных`,
          text: (
            <>
              Влезает: KV {fmt.int(seqs)} {fmt.plural(seqs, ['последовательности', 'последовательностей', 'последовательностей'])} занимает {fmt.bytes(f.kvNeededBytes)} из {fmt.bytes(f.kvBudgetBytes)} пула.
            </>
          ),
        }

  const StatusIcon = status.tone === 'good' ? CheckCircleIcon : status.tone === 'warn' ? WarningIcon : XCircleIcon
  const gpuLabel = `${gpus > 1 ? `${gpus} × ` : ''}${shortGpuName(gpu)}`
  const totalGiB = Math.round(layout.total / 2 ** 30)
  const poolBlocks = Math.floor(layout.poolTokens / DEFAULT_BLOCK_SIZE)
  // vLLM V1 считает ёмкость целыми блоками: num_blocks / ceil(max_model_len / block_size) × max_model_len,
  // при длине, кратной блоку, это ровно число блоков × размер блока (kv_cache_utils.get_kv_cache_capacity)
  const logTokens = poolBlocks * DEFAULT_BLOCK_SIZE
  const concurrency = poolBlocks / Math.ceil(ctx / DEFAULT_BLOCK_SIZE)
  // Подпись точки «49 при 8,2 тыс.» стоит над точкой: если пунктир «нужно N» проходит прямо над ней,
  // линия перечёркивает текст. Тогда подпись точки убираем, а число переносим в подпись пунктира.
  const pointLabelHit = useMemo(() => {
    if (f.maxSeqs <= 0 || seqs <= f.maxSeqs || curve.length < 2) return false
    const ys = [...curve.map((p) => p[1]), seqs, f.maxSeqs]
    const [lo, hi] = logDomain(Math.min(...ys), Math.max(...ys))
    const px = (190 * Math.log(seqs / f.maxSeqs)) / Math.log(hi / lo)
    return px < 24
  }, [curve, seqs, f.maxSeqs])
  // с v0.28 vLLM пишет ёмкость пула и concurrency одной строкой (update_kv_cache_capacity)
  const log = `INFO [kv_cache_utils.py] GPU KV cache size: ${enInt(logTokens)} tokens, Maximum concurrency for ${enInt(ctx)} tokens per request: ${enFixed(concurrency)}x`

  return (
    <Figure
      id="kv-cache-figure-memory"
      title="Калькулятор памяти: что помещается в GPU"
      width="wide"
      controlsPosition="side"
      controls={
        <>
          <CompactSummary layout={layout} tone={status.tone} text={status.short} />
          <Slider
            label="Одновременных последовательностей"
            value={seqs}
            min={1}
            max={1024}
            scale="log"
            onChange={(v) => setSeqs(Math.max(1, Math.round(v)))}
            format={(v) => fmt.int(Math.round(v))}
          />
          <Segmented
            label="Формат KV cache"
            size="sm"
            value={kvDtype}
            onChange={setKvDtype}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp8', label: 'FP8' },
            ]}
          />
          <Slider
            label="Длина контекста"
            value={ctx}
            min={512}
            max={131072}
            step={256}
            scale="log"
            onChange={setCtx}
            format={(v) => fmt.tokens(v)}
            hint="промпт + ответ одной последовательности"
          />
          <Select
            label="Модель"
            value={modelId}
            onChange={modelSel.set}
            options={modelOptions()}
            hint={modelSel.overridden ? <BackToSim what="модель" name={modelOr(simModelId).name} onClick={modelSel.reset} /> : 'из симулятора'}
          />
          <Select
            label="GPU"
            value={gpuId}
            onChange={gpuSel.set}
            options={gpuOptions()}
            hint={gpuSel.overridden ? <BackToSim what="GPU" name={shortGpuName(gpuOr(simGpuId))} onClick={gpuSel.reset} /> : 'из симулятора'}
          />
          <Segmented
            label="Формат весов"
            size="sm"
            value={dtype}
            onChange={setDtype}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp8', label: 'FP8' },
              { value: 'fp4', label: 'FP4', title: 'NVFP4; у gpt-oss родной формат MXFP4' },
            ]}
          />
          <Segmented
            label="Число GPU"
            size="sm"
            value={gpus}
            onChange={(n) => setGpusChoice({ n, key: gpusKey })}
            options={[1, 2, 4, 8].map((n) => ({ value: n, label: String(n) }))}
          />
          <Slider
            label="gpu_memory_utilization"
            value={memUtil}
            min={0.5}
            max={0.98}
            step={0.01}
            onChange={setMemUtil}
            format={(v) => fmt.num(v, 2)}
            hint="доля памяти GPU под движок (vLLM)"
          />
        </>
      }
      caption="Попробуйте: поднимите число последовательностей до 64 и посмотрите на переполнение, затем переключите KV на FP8. Пул тот же, а последовательностей влезает вдвое больше."
    >
      <div className="grid min-w-0 gap-5">
        <div className="text-sm text-ink-2">
          {model.name} на {gpuLabel}: {fmt.bytes(layout.total)} памяти (≈ {fmt.int(totalGiB)} ГиБ: в паспорте GPU гигабайты двоичные), KV{' '}
          {fmt.bytes(perToken, { mode: 'binary' })} на токен
        </div>
        <MemoryBar layout={layout} />
        <div
          className="flex items-start gap-2 rounded-control px-3 py-2.5 text-sm leading-snug text-ink"
          style={{ background: tint(status.tone, 12) }}
          role="status"
        >
          <StatusIcon size={18} aria-hidden className="mt-px shrink-0" style={{ color: cssVar(status.tone) }} />
          <span>{status.text}</span>
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-5 @4xl:grid-cols-4">
          <Stat
            label="Влезет одновременно"
            value={fmt.int(f.maxSeqs)}
            hint={`${fmt.plural(f.maxSeqs, ['последовательность', 'последовательности', 'последовательностей'])} по ${fmt.tokens(ctx)}`}
          />
          <Stat label="KV на запрос" value={bytesParts(f.kvPerSeqBytes).value} unit={bytesParts(f.kvPerSeqBytes).unit} />
          <Stat
            label="Пул KV в токенах"
            value={fmt.compact(layout.poolTokens)}
            hint={`${fmt.int(poolBlocks)} ${fmt.plural(poolBlocks, ['блок', 'блока', 'блоков'])} по ${DEFAULT_BLOCK_SIZE}`}
          />
          <Stat label="Пул KV" value={bytesParts(Math.max(0, f.kvBudgetBytes)).value} unit={bytesParts(Math.max(0, f.kvBudgetBytes)).unit} />
        </div>
        <div className="grid min-w-0 gap-2">
          <div className="text-sm font-medium text-ink">Сколько последовательностей влезает при другой длине контекста</div>
          {curve.length > 1 ? (
            <LineChart
              xLabel="Длина контекста, токенов"
              yLabel="Последовательностей"
              xScale="log"
              yScale="log"
              height={190}
              markers={false}
              series={[{ id: 'seqs', label: 'Влезет последовательностей', color: 'var(--ink-2)', points: curve }]}
              annotations={[
                ...(f.maxSeqs > 0
                  ? [{ kind: 'point' as const, x: dCtx, y: f.maxSeqs, label: pointLabelHit ? undefined : `${fmt.int(f.maxSeqs)} при ${fmt.compact(dCtx)}` }]
                  : []),
                {
                  kind: 'y' as const,
                  y: seqs,
                  label: pointLabelHit ? `нужно ${fmt.int(seqs)}, влезет ${fmt.int(f.maxSeqs)}` : `нужно ${fmt.int(seqs)}`,
                  labelAlign: 'end' as const,
                },
              ]}
              xFormat={(v) => fmt.compact(v)}
              yFormat={(v) => fmt.compact(v)}
              aria-label="Максимум одновременных последовательностей от длины контекста"
            />
          ) : (
            <p className="text-sm text-ink-3">Веса не оставляют места под KV: кривой нет.</p>
          )}
        </div>
        {f.weightsFit && (
          <div className="grid min-w-0 gap-2">
            <CodeBlock lang="text" title="Лог старта vLLM V1" code={log} copy={false} wrap />
            {concurrency < 1 && (
              <p className="flex items-start gap-2 text-sm text-ink-2">
                <XCircleIcon size={16} aria-hidden className="mt-0.5 shrink-0" style={{ color: cssVar('bad') }} />
                Concurrency меньше 1: в пул не помещается даже одна последовательность такой длины. vLLM в таком случае не стартует и
                предлагает поднять gpu_memory_utilization или уменьшить max_model_len.
              </p>
            )}
          </div>
        )}
      </div>
    </Figure>
  )
}
