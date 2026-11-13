import { useDeferredValue, useMemo, useState, type ReactNode } from 'react'
import { CheckCircleIcon } from '@phosphor-icons/react'
import { Badge, Figure, Segmented, Select, Slider, fmt } from '@/ui'
import { cn } from '@/ui/cn'
import { useSettings } from '@/state/settings'
import { vllmDefaultLimits } from '@/engine/sched'
import { getModel } from '@/engine/data/models'
import { configLabel, simConfig } from '../batching/config'
import { COMPARE_COUNT, COMPARE_WORKERS, PRESETS, bestRun, runComparison, type PresetId, type RunSummary } from './logic'

type MetricKey = 'ttftP50' | 'ttftP99' | 'itlP50' | 'itlP99' | 'perGpu' | 'slo'

const METRICS: ReadonlyArray<{ key: MetricKey; label: string; hint: string; better: 'low' | 'high'; format: (v: number) => string }> = [
  { key: 'ttftP50', label: 'TTFT, медиана', hint: 'p50', better: 'low', format: fmt.ms },
  { key: 'ttftP99', label: 'TTFT, хвост', hint: 'p99', better: 'low', format: fmt.ms },
  { key: 'itlP50', label: 'ITL, медиана', hint: 'p50', better: 'low', format: fmt.ms },
  { key: 'itlP99', label: 'ITL, хвост', hint: 'p99', better: 'low', format: fmt.ms },
  { key: 'perGpu', label: 'Выдано токенов в секунду на GPU', hint: 'при этой нагрузке', better: 'high', format: (v) => fmt.int(v) },
  { key: 'slo', label: 'Запросов в SLO', hint: 'TTFT и все паузы', better: 'high', format: (v) => fmt.pct(v) },
]

const MODEL_IDS = ['qwen3-8b', 'qwen3-32b', 'llama-3.1-70b'] as const
/** Уменьшенный бюджет шага в agg (совет документации vLLM ради ITL). */
const SMALL_AGG_BUDGET = 2048

export function CompareFigure() {
  const [settings] = useSettings()
  const [modelId, setModelId] = useState<string>((MODEL_IDS as readonly string[]).includes(settings.modelId) ? settings.modelId : 'qwen3-8b')
  const cfg = simConfig(modelId, settings.gpuId)
  const [presetId, setPresetId] = useState<PresetId>('long')
  const preset = PRESETS.find((p) => p.id === presetId)!
  const [prompt, setPrompt] = useState(preset.prompt)
  const [output, setOutput] = useState(preset.output)
  const [rate, setRate] = useState(preset.rate)
  const [split, setSplit] = useState('3P1D')
  const defaultBudget = vllmDefaultLimits(cfg.gpu).tokenBudget
  const [aggBudget, setAggBudget] = useState<'default' | 'small'>('default')

  const choosePreset = (id: PresetId) => {
    const p = PRESETS.find((q) => q.id === id)!
    setPresetId(id)
    setPrompt(p.prompt)
    setOutput(p.output)
    setRate(p.rate)
  }

  // объект параметров стабилен между рендерами, иначе useDeferredValue и useMemo пересчитывали бы симуляцию на каждом рендере
  const params = useMemo(() => ({ prompt, output, rate, presetId, cfg, aggBudget }), [prompt, output, rate, presetId, cfg, aggBudget])
  const input = useDeferredValue(params)
  const stale =
    input.prompt !== prompt || input.output !== output || input.rate !== rate || input.presetId !== presetId || input.aggBudget !== aggBudget || input.cfg !== cfg
  const result = useMemo(() => {
    const slo = PRESETS.find((p) => p.id === input.presetId)!.slo
    return runComparison(input.cfg, { prompt: input.prompt, output: input.output, rate: input.rate }, slo, {
      aggBudget: input.aggBudget === 'small' ? SMALL_AGG_BUDGET : undefined,
    })
  }, [input])

  const chosen = result.splits.find((s) => s.key === split) ?? result.splits[result.splits.length - 1]
  const all = [result.agg, ...result.splits]
  const best = bestRun(all)
  const gpuNote = cfg.tp > 1 ? `${COMPARE_WORKERS} воркерах по ${cfg.tp} GPU (TP${cfg.tp}), всего ${COMPARE_WORKERS * cfg.tp} GPU` : `${COMPARE_WORKERS} GPU`
  const realRate = rate / cfg.timeScale

  return (
    <Figure
      id="disaggregation-figure-compare"
      title={`Агрегированный режим против xPyD на ${gpuNote}`}
      actions={<Badge>{configLabel(cfg)}</Badge>}
      controls={
        <>
          <Segmented<PresetId>
            className="col-span-full"
            size="sm"
            label="Профиль нагрузки"
            value={presetId}
            onChange={choosePreset}
            options={PRESETS.map((p) => ({ value: p.id, label: p.label }))}
          />
          <Slider label="Промпт, медиана" value={prompt} min={128} max={32768} scale="log" step={1} onChange={setPrompt} format={(v) => fmt.tokens(v)} />
          <Slider label="Ответ, медиана" value={output} min={16} max={4096} scale="log" step={1} onChange={setOutput} format={(v) => fmt.tokens(v)} />
          <Select label="Модель" value={modelId} onChange={setModelId} options={MODEL_IDS.map((id) => ({ value: id, label: getModel(id).name }))} />
          <Segmented<'default' | 'small'>
            label="Бюджет шага в agg"
            value={aggBudget}
            onChange={setAggBudget}
            options={[
              { value: 'default', label: fmt.int(defaultBudget) },
              { value: 'small', label: fmt.int(SMALL_AGG_BUDGET) },
            ]}
          />
          <Slider
            label="Поток запросов"
            value={rate}
            min={1}
            max={120}
            scale="log"
            step={0.5}
            onChange={setRate}
            format={() => `${fmt.num(realRate, 1)} запр/с`}
          />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> в «Длинном контексте» сравните хвост ITL у агрегированного режима и у 3P1D, а
          потом выберите 1P3D: один prefill-воркер не успевает, и TTFT уходит в секунды. Уменьшите бюджет шага в agg до {fmt.int(SMALL_AGG_BUDGET)}: chunked prefill режет
          промпты мельче, и хвост ITL падает в разы даже без дезагрегации. В «Чате» на Qwen3-8B (H100) все схемы укладываются в SLO; выберите
          Llama-3.1-70B: шаги prefill большой модели длиннее, и выигрыш дезагрегации растёт. В «Рассуждениях» ответы длинные, и SLO требует ещё
          скорости: медианная пауза запроса не дольше двух пауз одиночного запроса
          {input.presetId === 'reason' && result.itlP50Ms !== undefined ? ` (здесь ${fmt.ms(result.itlP50Ms)})` : ''}; выигрывают схемы, где decode
          делают больше воркеров. SLO условный: TTFT не больше {fmt.ms(preset.slo.ttftMs)} и ни одной паузы между токенами дольше{' '}
          {fmt.ms(preset.slo.itlMs)}, такое подвисание уже заметно глазу. Строка «Выдано токенов» показывает не предел GPU, а нагрузку: пока система
          успевает за потоком, выдано почти столько, сколько пришло, и разница только в том, кто дольше дорабатывает хвост. В каждом прогоне{' '}
          {COMPARE_COUNT} запросов, моменты прихода случайные (пуассоновский поток); KV передаётся по RDMA 400G.
        </>
      }
    >
      <div className={cn('grid gap-4 transition-opacity', stale && 'opacity-60')} aria-busy={stale}>
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink-2">
            Доля запросов в SLO при разном разбиении тех же {COMPARE_WORKERS} воркеров
            {best === null && <Badge tone="neutral">все схемы в SLO</Badge>}
          </div>
          <div className="grid grid-cols-2 gap-2 @md:grid-cols-4" role="group" aria-label="Разбиение воркеров">
            <SplitCard run={result.agg} title={`Агрегированный`} sub={`${COMPARE_WORKERS} × prefill и decode`} selected={false} best={best?.key === 'agg'} />
            {result.splits.map((s) => (
              <SplitCard
                key={s.key}
                run={s}
                title={s.key}
                sub={`${s.prefill} prefill + ${s.decode} decode`}
                selected={s.key === chosen.key}
                best={best?.key === s.key}
                onSelect={() => setSplit(s.key)}
              />
            ))}
          </div>
        </div>
        <div className="scrollbar-thin overflow-x-auto">
          <table className="w-full border-collapse text-[15px]">
            <caption className="sr-only">Метрики агрегированного режима и выбранного разбиения {chosen.key}</caption>
            <thead>
              <tr className="border-b border-line-strong text-left text-sm text-ink-2">
                <th scope="col" className="py-2 pr-3 font-normal">
                  Метрика
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-normal">
                  <span className="hidden @md:inline">Агрегированный</span>
                  <span className="@md:hidden">Agg</span>
                </th>
                <th scope="col" className="py-2 pr-3 text-right font-normal">
                  <span className="hidden @md:inline">Дезагрегированный </span>
                  {chosen.key}
                </th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map((m) => {
                const a = result.agg[m.key]
                const d = chosen[m.key]
                const winA = m.better === 'low' ? a < d * 0.95 : a > d * 1.05 + 1e-9
                const winD = m.better === 'low' ? d < a * 0.95 : d > a * 1.05 + 1e-9
                return (
                  <tr key={m.key} className="border-b border-line last:border-b-0">
                    <th scope="row" className="py-2 pr-3 text-left font-normal text-ink">
                      {m.label} <span className="text-sm whitespace-nowrap text-ink-3">{m.hint}</span>
                    </th>
                    <Cell win={winA}>{m.format(a)}</Cell>
                    <Cell win={winD}>{m.format(d)}</Cell>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-ink-2">
          Медианная передача KV в {chosen.key}: {fmt.ms(chosen.transferP50)}. Отметка <CheckCircleIcon size={14} weight="fill" aria-hidden className="inline align-[-2px] text-good" />{' '}
          «лучше» ставится, если разница больше 5 %.
        </p>
      </div>
    </Figure>
  )
}

function Cell({ win, children }: { win: boolean; children: ReactNode }) {
  return (
    <td className="py-2 pr-3 text-right tabular-nums">
      <span className={cn('inline-flex items-center gap-1.5', win ? 'font-semibold text-ink' : 'text-ink-2')}>
        {win && (
          <span className="inline-flex items-center gap-1 text-xs font-medium text-ink-2">
            <CheckCircleIcon size={15} weight="fill" aria-hidden className="text-good" />
            <span className="sr-only @md:not-sr-only">лучше</span>
          </span>
        )}
        {children}
      </span>
    </td>
  )
}

function SplitCard({
  run,
  title,
  sub,
  selected,
  best,
  onSelect,
}: {
  run: RunSummary
  title: string
  sub: string
  selected: boolean
  best: boolean
  onSelect?: () => void
}) {
  const body = (
    <>
      <span className="flex items-baseline justify-between gap-2">
        <span className="text-[15px] font-semibold text-ink">{title}</span>
        {best && (
          <span className="inline-flex items-center gap-1 text-xs text-ink-2">
            <CheckCircleIcon size={14} weight="fill" aria-hidden className="text-good" />
            лучший
          </span>
        )}
      </span>
      <span className="text-xs text-ink-2">{sub}</span>
      <span className="mt-1 text-[1.375rem] leading-tight font-semibold text-ink tabular-nums">{fmt.pct(run.slo)}</span>
      <span className="text-xs text-ink-2 tabular-nums">TTFT p99 {fmt.ms(run.ttftP99)}</span>
      <span className="text-xs text-ink-2 tabular-nums">ITL p99 {fmt.ms(run.itlP99)}</span>
    </>
  )
  const cls = cn(
    'grid content-start gap-0.5 rounded-control border px-3 py-2.5 text-left transition-colors',
    selected ? 'border-accent bg-accent-soft' : 'border-line bg-surface',
    onSelect && !selected && 'hover:bg-surface-3/60',
  )
  if (!onSelect) return <div className={cls}>{body}</div>
  return (
    <button type="button" aria-pressed={selected} onClick={onSelect} className={cls}>
      {body}
    </button>
  )
}
