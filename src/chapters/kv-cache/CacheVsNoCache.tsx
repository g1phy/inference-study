import { useDeferredValue, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { ChapterLink, Figure, LineChart, Segmented, Slider, Stat, cssVar, fmt, useTip } from '@/ui'
import { decodeStepMs, prefillFlops, prefillMs } from '@/engine/perf'
import { useSettings } from '@/state/settings'
import { gpuOr, minGpus, modelOr, shortGpuName } from './lib'
import { usePromptTokens } from './usePromptTokens'

/* ============================================================================
 * Мини-сетка «что считает модель на каждом шаге» (упрощённо: 6 токенов промпта, 6 шагов)
 * ========================================================================== */

/** «16,6 TFLOP» → { value: '16,6', unit: 'TFLOP' } для Stat. */
function flopsParts(x: number): { value: string; unit: string } {
  const s = fmt.flops(x)
  const i = s.lastIndexOf('\u00a0')
  return i > 0 ? { value: s.slice(0, i), unit: s.slice(i + 1) } : { value: s, unit: '' }
}

const MINI_P = 6
const MINI_G = 6
const MINI_COLS = MINI_P + MINI_G
const CELL = 17
const GAP = 3
const LABEL_W = 52
const HEAD_H = 22

type CellState = 'computed' | 'read' | 'none'

function cellState(row: number, col: number, cached: boolean): CellState {
  if (row === 0) return col < MINI_P ? 'computed' : 'none'
  const last = MINI_P + row - 1 // позиция токена, который обрабатывается на этом шаге
  if (col > last) return 'none'
  if (!cached) return 'computed'
  return col === last ? 'computed' : 'read'
}

function computedCount(cached: boolean): number {
  let n = 0
  for (let r = 0; r <= MINI_G; r++) for (let c = 0; c < MINI_COLS; c++) if (cellState(r, c, cached) === 'computed') n++
  return n
}

function rowText(row: number, cached: boolean): string {
  if (row === 0) return `Prefill: ${MINI_P} токенов промпта считаются за один проход, их K и V записываются в кэш.`
  const len = MINI_P + row
  if (!cached) return `Шаг ${row}: без кэша модель заново прогоняет все ${len} токенов через все слои, чтобы получить K и V прошлых токенов.`
  return `Шаг ${row}: считается один новый токен, K и V предыдущих ${len - 1} читаются из кэша.`
}

function MiniGrid({ cached }: { cached: boolean }) {
  const ref = useRef<SVGSVGElement | null>(null)
  const [hoverRow, setHoverRow] = useState<number | null>(null)
  const tip = useTip({ scopeEl: ref.current, side: 'top', wide: true, onOutside: () => setHoverRow(null) })
  const W = LABEL_W + MINI_COLS * (CELL + GAP)
  const H = HEAD_H + (MINI_G + 1) * (CELL + GAP)
  const total = computedCount(cached)
  const liveId = `kvc-mini-${cached ? 'kv' : 'no'}-row`

  const show = (row: number, e: PointerEvent<SVGGElement>) => {
    setHoverRow(row)
    tip.show(e.currentTarget.getBoundingClientRect(), <span>{rowText(row, cached)}</span>)
  }
  const clear = () => {
    setHoverRow(null)
    tip.hide()
  }
  // Вся сетка один Tab-стоп: стрелки вверх и вниз ходят по шагам и показывают ту же подсказку.
  const showRow = (row: number) => {
    setHoverRow(row)
    const el = ref.current?.querySelector(`[data-row="${row}"]`)
    if (el) tip.show(el.getBoundingClientRect(), <span>{rowText(row, cached)}</span>)
  }
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    const d = e.key === 'ArrowDown' || e.key === 'ArrowRight' ? 1 : e.key === 'ArrowUp' || e.key === 'ArrowLeft' ? -1 : 0
    if (!d) return
    e.preventDefault()
    showRow(Math.max(0, Math.min(MINI_G, (hoverRow ?? (d > 0 ? -1 : MINI_G + 1)) + d)))
  }

  return (
    <div className="grid min-w-0 content-start gap-2">
      <div className="text-sm font-medium text-ink">{cached ? 'С KV cache' : 'Без кэша'}</div>
      <div className="overflow-x-auto pb-1">
        <svg
          ref={ref}
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          role="list"
          tabIndex={0}
          aria-label={`${cached ? 'С KV cache' : 'Без кэша'}: за ${MINI_G} шагов посчитано ${total} ${fmt.plural(total, ['позиция', 'позиции', 'позиций'])}. Стрелки вверх и вниз переключают шаги.`}
          onKeyDown={onKey}
          onFocus={(e) => e.currentTarget.matches(':focus-visible') && showRow(hoverRow ?? 0)}
          onBlur={clear}
          className="block rounded-[4px] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          {/* шапка: промпт и ответ */}
          <g fontSize={11} fill="var(--ink-3)">
            <rect x={LABEL_W} y={4} width={8} height={8} rx={2} fill={cssVar('prefill')} />
            <text x={LABEL_W + 12} y={12}>
              промпт
            </text>
            <rect x={LABEL_W + MINI_P * (CELL + GAP)} y={4} width={8} height={8} rx={2} fill={cssVar('decode')} />
            <text x={LABEL_W + MINI_P * (CELL + GAP) + 12} y={12}>
              ответ
            </text>
          </g>
          {Array.from({ length: MINI_G + 1 }, (_, row) => {
            const y = HEAD_H + row * (CELL + GAP)
            const dim = hoverRow !== null && hoverRow !== row
            return (
              <g
                key={row}
                data-row={row}
                role="listitem"
                aria-label={rowText(row, cached)}
                onPointerEnter={(e) => show(row, e)}
                onPointerDown={(e) => show(row, e)}
                onPointerLeave={(e) => e.pointerType === 'mouse' && clear()}
                style={{ opacity: dim ? 0.45 : 1, transition: 'opacity 150ms' }}
              >
                <rect x={0} y={y - 1} width={W} height={CELL + 2} fill="transparent" />
                <text x={0} y={y + CELL - 5} fontSize={11.5} fill="var(--ink-2)">
                  {row === 0 ? 'prefill' : `шаг ${row}`}
                </text>
                {Array.from({ length: MINI_COLS }, (_, col) => {
                  const s = cellState(row, col, cached)
                  if (s === 'none') return null
                  const x = LABEL_W + col * (CELL + GAP)
                  if (s === 'computed') {
                    return <rect key={col} x={x} y={y} width={CELL} height={CELL} rx={3} fill={cssVar(col < MINI_P ? 'prefill' : 'decode')} />
                  }
                  return (
                    <g key={col}>
                      <rect x={x} y={y} width={CELL} height={CELL} rx={3} fill="var(--surface-2)" stroke="var(--line-strong)" strokeWidth={1} />
                      <rect x={x + 3} y={y + 3} width={CELL - 6} height={3} rx={1} fill={cssVar('k')} />
                      <rect x={x + 3} y={y + CELL - 6} width={CELL - 6} height={3} rx={1} fill={cssVar('v')} />
                    </g>
                  )
                })}
              </g>
            )
          })}
        </svg>
      </div>
      <div className="text-sm text-ink-2">
        Посчитано позиций: <span className="font-semibold text-ink">{total}</span>
      </div>
      <span id={liveId} className="sr-only" aria-live="polite">
        {hoverRow !== null ? rowText(hoverRow, cached) : ''}
      </span>
      {tip.node}
    </div>
  )
}

function MiniLegend() {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-1.5 text-[13px] text-ink-2">
      <li className="inline-flex items-center gap-2">
        <span aria-hidden className="inline-block h-3 w-3 rounded-[3px]" style={{ background: cssVar('prefill') }} />
        посчитано: токен промпта
      </li>
      <li className="inline-flex items-center gap-2">
        <span aria-hidden className="inline-block h-3 w-3 rounded-[3px]" style={{ background: cssVar('decode') }} />
        посчитано: сгенерированный токен
      </li>
      <li className="inline-flex items-center gap-2">
        <span aria-hidden className="relative inline-block h-3 w-3 rounded-[3px] border border-line-strong bg-surface-2">
          <span className="absolute inset-x-[2px] top-[2px] h-[2px] rounded-[1px]" style={{ background: cssVar('k') }} />
          <span className="absolute inset-x-[2px] bottom-[2px] h-[2px] rounded-[1px]" style={{ background: cssVar('v') }} />
        </span>
        K и V взяты из кэша
      </li>
    </ul>
  )
}

/* ============================================================================
 * Настоящие числа: работа и время генерации N токенов
 * ========================================================================== */

interface WorkResult {
  no: Array<[number, number]>
  cached: Array<[number, number]>
  flopsNo: number
  flopsCached: number
  msNo: number
  msCached: number
}

/**
 * Без кэша каждый шаг decode = заново prefill всей последовательности (P + k токенов);
 * с кэшем шаг считает один токен, а attention читает P + k − 1 закэшированных ключей.
 * FLOPs суммируются точно, время интегрируется по выборке шагов (оценка perf.ts).
 */
function workSeries(modelId: string, gpuId: string, P: number, N: number, gpus: number): WorkResult {
  const m = modelOr(modelId)
  const g = gpuOr(gpuId)
  let fNo = prefillFlops(m, P)
  let fC = fNo
  const no: Array<[number, number]> = [[1, fNo / 1e12]]
  const cached: Array<[number, number]> = [[1, fC / 1e12]]
  const every = Math.max(1, Math.floor(N / 160))
  for (let k = 1; k < N; k++) {
    const ctx = P + k
    fNo += prefillFlops(m, ctx)
    fC += prefillFlops(m, ctx, { cachedTokens: ctx - 1 })
    if (k % every === 0 || k === N - 1) {
      no.push([k + 1, fNo / 1e12])
      cached.push([k + 1, fC / 1e12])
    }
  }
  // время: prefill промпта + сумма шагов (по выборке из ~200 точек, шаг s ≈ N / 200)
  const t0 = prefillMs(m, g, P, { gpus })
  let msNo = t0
  let msCached = t0
  const s = Math.max(1, Math.ceil((N - 1) / 200))
  for (let k = 1; k < N; k += s) {
    const width = Math.min(s, N - k)
    const ctx = P + k + Math.floor(width / 2)
    msNo += prefillMs(m, g, ctx, { gpus }) * width
    msCached += decodeStepMs(m, g, { batch: 1, avgContext: ctx, gpus }) * width
  }
  return { no, cached, flopsNo: fNo, flopsCached: fC, msNo, msCached }
}

export function CacheVsNoCache() {
  const [settings] = useSettings()
  const { tokens } = usePromptTokens()
  const realP = tokens?.length ?? null
  const [promptOverride, setPromptOverride] = useState<number | null>(null)
  const [gen, setGen] = useState(1024)
  const [axes, setAxes] = useState<'linear' | 'log'>('linear')
  const P = promptOverride ?? realP ?? 40
  const dP = useDeferredValue(P)
  const dN = useDeferredValue(gen)

  const model = modelOr(settings.modelId)
  const gpu = gpuOr(settings.gpuId)
  const gpus = useMemo(() => minGpus(model, gpu), [model, gpu])
  const r = useMemo(() => workSeries(model.id, gpu.id, dP, dN, gpus), [model.id, gpu.id, dP, dN, gpus])
  const ratio = r.flopsNo / r.flopsCached
  const gpuLabel = `${gpus > 1 ? `${gpus} × ` : ''}${shortGpuName(gpu)}`

  return (
    <Figure
      id="kv-cache-figure-why"
      title="Сколько работы на генерацию: без кэша и с ним"
      width="wide"
      actions={
        <Segmented
          size="sm"
          aria-label="Оси графика"
          value={axes}
          onChange={setAxes}
          options={[
            { value: 'linear', label: 'Линейные оси' },
            { value: 'log', label: 'Логарифмические' },
          ]}
        />
      }
      controls={
        <>
          <Slider
            label="Длина промпта"
            value={P}
            min={8}
            max={16384}
            scale="log"
            onChange={(v) => setPromptOverride(Math.round(v))}
            format={(v) => fmt.tokens(Math.round(v))}
            hint={
              realP !== null ? (
                promptOverride === null ? (
                  `как в вашем запросе из симулятора`
                ) : (
                  <button type="button" className="text-accent underline-offset-2 hover:underline" onClick={() => setPromptOverride(null)}>
                    вернуть длину вашего запроса ({fmt.tokens(realP)})
                  </button>
                )
              ) : undefined
            }
          />
          <Slider
            label="Сгенерировать токенов"
            value={gen}
            min={16}
            max={4096}
            scale="log"
            onChange={(v) => setGen(Math.round(v))}
            format={(v) => fmt.int(Math.round(v))}
          />
        </>
      }
      caption={`Попробуйте: поставьте ${fmt.int(4096)} сгенерированных токенов и переключите оси на логарифмические. Справа, где ответ уже намного длиннее промпта, синяя линия идёт вдвое круче оранжевой: на логарифмических осях это наклон 2 против 1, «квадрат против линии». Потом поднимите длину промпта до ${fmt.int(16384)}. Пока ответ короче промпта, без кэша тоже почти прямая, а разрыв растёт с каждым токеном: после сотни токенов работы без кэша в сто раз больше, к концу генерации в тысячи раз.`}
    >
      <div className="grid min-w-0 gap-6">
        <div className="grid min-w-0 gap-3">
          <p className="max-w-[68ch] text-sm text-ink-2">
            Упрощённо: промпт из {MINI_P} токенов и {MINI_G} шагов decode. Строка это один шаг модели, клетка это позиция токена.
          </p>
          <div className="grid min-w-0 gap-6 @2xl:grid-cols-2">
            <MiniGrid cached={false} />
            <MiniGrid cached />
          </div>
          <MiniLegend />
        </div>

        <div className="grid min-w-0 gap-4 border-t border-line pt-5">
          <div className="max-w-[68ch] text-sm text-ink-2">
            Настоящие числа: {model.name} на {gpuLabel}, промпт {fmt.tokens(dP)}, генерация {fmt.tokens(dN)}. Вычисления и время
            считает движок сайта по формулам из главы <ChapterLink to="roofline">про compute- и memory-bound</ChapterLink>; время
            оценочное.
          </div>
          <LineChart
            xLabel="Сгенерировано токенов"
            yLabel="Вычисления с начала генерации"
            xScale={axes}
            yScale={axes}
            height={240}
            series={[
              { id: 'no', label: 'Без кэша: каждый шаг заново считает всю последовательность', color: 'prefill', points: r.no },
              { id: 'kv', label: 'С KV cache: каждый шаг считает один токен', color: 'decode', points: r.cached },
            ]}
            directLabels={false}
            xFormat={(v) => fmt.compact(v)}
            yFormat={(v) => (v === 0 ? '0' : fmt.flops(v * 1e12))}
            aria-label="Суммарные вычисления на генерацию без кэша и с кэшем"
          />
          <div className="grid grid-cols-2 gap-x-6 gap-y-5 @3xl:grid-cols-4">
            <Stat
              label="Без кэша"
              value={flopsParts(r.flopsNo).value}
              unit={flopsParts(r.flopsNo).unit}
              swatch={cssVar('prefill')}
              hint={`≈ ${fmt.ms(r.msNo)} на ${gpuLabel}`}
            />
            <Stat
              label="С KV cache"
              value={flopsParts(r.flopsCached).value}
              unit={flopsParts(r.flopsCached).unit}
              swatch={cssVar('decode')}
              hint={`≈ ${fmt.ms(r.msCached)} на ${gpuLabel}`}
            />
            <Stat label="Разница в FLOP" value={fmt.multiplier(ratio, ratio < 10 ? 1 : 0)} hint="во столько раз больше вычислений без кэша" />
            <Stat
              label="Разница во времени"
              value={fmt.multiplier(r.msNo / r.msCached, 1)}
              hint="меньше, чем по FLOP: шаг decode упирается в чтение весов"
            />
          </div>
        </div>
      </div>
    </Figure>
  )
}
