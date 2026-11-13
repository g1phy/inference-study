import { useEffect, useMemo, useState } from 'react'
import { useSettings } from '@/state/settings'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { findModel, getModel } from '@/engine/data/models'
import type { RouteMode, RoutingStrategyResult } from '@/engine/router'
import { Figure, LineChart, Segmented, Skeleton, Slider, cn, cssVar, fmt, useInView, type LineSeries } from '@/ui'
import { FLEET_MODES, FLEET_RATES, FLEET_REQUESTS, FLEET_WORKERS, MODE_LABEL, runFleet, type FleetConfig } from './fleet'

const CAPACITIES = [
  { value: '16384', label: '16 тыс.' },
  { value: '32768', label: '32 тыс.' },
  { value: '131072', label: '128 тыс.' },
] as const

/** Значение, которое догоняет исходное через delay мс тишины (тяжёлая симуляция не на каждый тик слайдера). */
function useDebounced<T>(value: T, delay = 220): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), delay)
    return () => window.clearTimeout(t)
  }, [value, delay])
  return v
}

/**
 * Высоты заглушек под высоты готовых блоков (замер в Chromium, 1440/1024/768 и 375): таблица стратегий на широкой
 * фигуре, три карточки на узкой (@md), график с легендой (легенда ≈ 29 + поля LineChart 72 + область 260).
 */
const RESULT_TABLE_H = 286
const RESULT_CARDS_H = 666
const CHART_H = 361

/**
 * Стратегии не сущности данных, поэтому без цветов prefill и decode и без пунктира (пунктир на сайте значит порог):
 * нейтральные тона, KV-aware самый тёмный и с заливкой, остальные различают подписи у концов линий и легенда.
 */
const SERIES_STYLE: Record<RouteMode, { color: string; area?: boolean }> = {
  kv: { color: 'var(--ink)', area: true },
  'round-robin': { color: 'var(--ink-2)' },
  random: { color: 'var(--ink-3)' },
}

/**
 * Фигура 2 главы 20: много запросов от нескольких тенантов. KV-aware против round-robin и случайного
 * выбора: доля токенов из кэша, TTFT, баланс нагрузки, плюс кривая «средний TTFT против потока».
 */
export function FleetFigure() {
  const [settings] = useSettings()
  const model = findModel(settings.modelId) ?? getModel('qwen3-8b')
  const gpu = findGpu(settings.gpuId) ?? getGpu('h100-sxm')
  const [tenants, setTenants] = useState(8)
  const [systemTokens, setSystemTokens] = useState(2048)
  const [capacity, setCapacity] = useState<(typeof CAPACITIES)[number]['value']>('16384')
  const [rate, setRate] = useState(100)

  const cfg = useMemo<FleetConfig>(
    () => ({ tenants, systemTokens, capacityTokens: Number(capacity), ratePerSec: rate }),
    [tenants, systemTokens, capacity, rate],
  )
  const settled = useDebounced(cfg)
  const pending = settled !== cfg
  // симуляция флота тяжёлая (сотни мс на слабом CPU): считаем, только когда фигура подъехала к экрану,
  // а не при фоновом монтировании главы
  const [viewRef, near] = useInView<HTMLDivElement>({ rootMargin: '100% 0px', once: true })
  const result = useMemo(() => (near ? runFleet(settled, model, gpu) : null), [near, settled, model, gpu])

  // Кривая TTFT против потока: считаем по точке за тик, чтобы не блокировать страницу.
  const sweepKey = `${settled.tenants}|${settled.systemTokens}|${settled.capacityTokens}|${model.id}|${gpu.id}`
  const [sweep, setSweep] = useState<{ key: string; points: Array<{ rate: number; byMode: Partial<Record<RouteMode, RoutingStrategyResult>> }> }>({
    key: '',
    points: [],
  })
  useEffect(() => {
    if (!near) return
    let cancelled = false
    let i = 0
    let m = 0
    const points: Array<{ rate: number; byMode: Partial<Record<RouteMode, RoutingStrategyResult>> }> = []
    let cur: Partial<Record<RouteMode, RoutingStrategyResult>> = {}
    // одна стратегия за тик (а не все три): каждая задача короче, страница отвечает и на слабом CPU
    const tick = () => {
      if (cancelled || i >= FLEET_RATES.length) return
      const r = FLEET_RATES[i]
      const mode = FLEET_MODES[m]
      cur = { ...cur, ...runFleet(settled, model, gpu, r, [mode]).byMode }
      if (++m === FLEET_MODES.length) {
        points.push({ rate: r, byMode: cur })
        setSweep({ key: sweepKey, points: points.slice() })
        cur = {}
        m = 0
        i++
      }
      timer = window.setTimeout(tick, 16)
    }
    let timer = window.setTimeout(tick, 60)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [sweepKey, near])

  const series: LineSeries[] = useMemo(() => {
    const pts = sweep.key === sweepKey ? sweep.points : []
    return FLEET_MODES.map((m) => ({
      id: m,
      label: MODE_LABEL[m],
      color: SERIES_STYLE[m].color,
      area: SERIES_STYLE[m].area,
      markers: true,
      points: pts.map((p) => [p.rate, p.byMode[m]?.meanTtftMs ?? NaN] as [number, number]),
    }))
  }, [sweep, sweepKey])
  const sweepDone = sweep.key === sweepKey && sweep.points.length === FLEET_RATES.length

  const rows = result ? FLEET_MODES.map((m) => result.byMode[m]).filter((x): x is RoutingStrategyResult => Boolean(x)) : []
  const best = (pick: (r: RoutingStrategyResult) => number, dir: 'min' | 'max') => {
    const vals = rows.map(pick)
    const target = dir === 'min' ? Math.min(...vals) : Math.max(...vals)
    return (r: RoutingStrategyResult) => pick(r) === target
  }

  const controls = (
    <>
      <Slider label="Тенантов" value={tenants} min={1} max={16} onChange={setTenants} hint="у каждого свой системный промпт" />
      <Slider
        label="Системный промпт"
        value={systemTokens}
        min={256}
        max={4096}
        step={256}
        onChange={setSystemTokens}
        format={(v) => fmt.tokens(v)}
        hint="общий префикс запросов тенанта"
      />
      <Segmented label="Кэш воркера, токенов" options={CAPACITIES} value={capacity} onChange={setCapacity} size="sm" />
      <Slider label="Поток запросов" value={rate} min={10} max={200} step={5} onChange={setRate} format={(v) => `${fmt.int(v)} запр./с`} />
    </>
  )

  return (
    <Figure
      id="routing-figure-fleet"
      title={`${fmt.int(FLEET_REQUESTS)} запросов, ${FLEET_WORKERS} воркера: KV-aware против round-robin`}
      width="wide"
      controls={controls}
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> поставьте один тенант, и стратегии почти сравняются: общий префикс со временем
          окажется в кэше у каждого воркера. Потом верните 8 тенантов и поднимите поток до 200 запр./с: round-robin заставляет каждый воркер держать
          префиксы всех тенантов, и в кэш на 16 тыс. токенов они не помещаются. Поставьте кэш 128 тыс.: доля из кэша у round-robin вырастет, а у KV-aware
          почти не изменится. На малом потоке среднее TTFT бывает больше p90: его тянут вверх редкие холодные промахи, когда префикса тенанта ещё нет в кэше. Токены и хэши
          настоящие, длины условные, время prefill по perf.ts для {model.name} на {gpu.name}, TTFT здесь только очередь и prefill.
        </>
      }
    >
      <div ref={viewRef} className={cn('grid min-w-0 gap-6 transition-opacity', pending && 'opacity-60')}>
        <div className="min-w-0">
          {!result ? (
            // заглушка той же высоты, что и результат (таблица на широкой фигуре, карточки на узкой): когда симуляция
            // досчитается, текст ниже фигуры не прыгает, в том числе при прокрутке вверх
            <>
              <Skeleton className="@md:hidden" height={RESULT_CARDS_H} radius={12} label="Считаю симуляцию" />
              <Skeleton className="hidden @md:block" height={RESULT_TABLE_H} radius={12} label="Считаю симуляцию" />
            </>
          ) : (
          <>
          {/* узкая фигура: карточка на стратегию вместо таблицы, где заголовки ломались посреди слова */}
          <div className="grid min-w-0 gap-3 @md:hidden">
            {rows.map((r) => (
              <StrategyCard
                key={r.mode}
                row={r}
                metrics={[
                  { label: 'Токенов промпта из кэша', value: fmt.pct(r.hitRate), best: best((x) => x.hitRate, 'max')(r) },
                  { label: 'Средний TTFT', value: fmt.ms(r.meanTtftMs), best: best((x) => x.meanTtftMs, 'min')(r) },
                  { label: 'TTFT p90', value: fmt.ms(r.p90TtftMs), best: best((x) => x.p90TtftMs, 'min')(r) },
                  { label: 'GPU-время на prefill', value: fmt.ms(r.prefillGpuMs), best: best((x) => x.prefillGpuMs, 'min')(r) },
                ]}
              />
            ))}
          </div>
          <table className="hidden w-full table-fixed border-collapse text-[14px] tabular-nums @md:table">
            <thead>
              <tr className="border-b border-line text-left text-[13px] text-ink-3">
                <th className="w-[34%] py-2 pr-2 font-normal" />
                {rows.map((r) => (
                  <th key={r.mode} className="px-1.5 py-2 text-right align-bottom font-medium break-words text-ink @md:px-3">
                    {MODE_LABEL[r.mode]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <MetricRow label="Токенов промпта из кэша" rows={rows} value={(r) => fmt.pct(r.hitRate)} isBest={best((r) => r.hitRate, 'max')} />
              <MetricRow label="Средний TTFT" rows={rows} value={(r) => fmt.ms(r.meanTtftMs)} isBest={best((r) => r.meanTtftMs, 'min')} />
              <MetricRow label="TTFT p90" rows={rows} value={(r) => fmt.ms(r.p90TtftMs)} isBest={best((r) => r.p90TtftMs, 'min')} />
              <MetricRow label="GPU-время на prefill" rows={rows} value={(r) => fmt.ms(r.prefillGpuMs)} isBest={best((r) => r.prefillGpuMs, 'min')} />
              <tr className="border-b border-line align-middle">
                <th scope="row" className="py-2.5 pr-2 text-left text-[13px] font-normal text-ink-2">
                  Запросов на воркер
                </th>
                {rows.map((r) => (
                  <td key={r.mode} className="px-1.5 py-2.5 @md:px-3">
                    <WorkerBars counts={r.perWorker.map((p) => p.requests)} />
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
          </>
          )}
        </div>
        <div className="grid min-w-0 gap-2">
          <div className="text-sm font-semibold text-ink">Средний TTFT при разном потоке (остальное как выбрано)</div>
          {!sweepDone ? (
            // кривая появляется целиком, когда посчитаны все точки: заглушка высотой с график и легенду,
            // счётчик точек внутри неё (раньше строка «Досчитываю…» под графиком появлялась и исчезала, сдвигая текст)
            <div className="relative">
              <Skeleton height={CHART_H} radius={12} label="Считаю точки кривой" />
              <p className="pointer-events-none absolute inset-0 grid place-items-center text-[13px] text-ink-2 tabular-nums" aria-hidden>
                Считаю точки кривой: {sweep.key === sweepKey ? sweep.points.length : 0} из {FLEET_RATES.length}
              </p>
            </div>
          ) : (
            <LineChart
              series={series}
              xLabel="Поток, запросов в секунду"
              yLabel="Средний TTFT, мс"
              xDomain={[FLEET_RATES[0], FLEET_RATES[FLEET_RATES.length - 1]]}
              yFormat={(v) => fmt.int(v)}
              annotations={[{ kind: 'x', x: rate, label: 'сейчас' }]}
              height={260}
              aria-label="Средний TTFT в зависимости от потока запросов для трёх стратегий"
            />
          )}
        </div>
      </div>
    </Figure>
  )
}

function MetricRow({
  label,
  rows,
  value,
  isBest,
}: {
  label: string
  rows: RoutingStrategyResult[]
  value: (r: RoutingStrategyResult) => string
  isBest: (r: RoutingStrategyResult) => boolean
}) {
  return (
    <tr className="border-b border-line">
      <th scope="row" className="py-2.5 pr-2 text-left text-[13px] font-normal text-ink-2">
        {label}
      </th>
      {rows.map((r) => (
        <td key={r.mode} className={cn('px-1.5 py-2.5 text-right @md:px-3', isBest(r) ? 'font-semibold text-ink' : 'text-ink-2')}>
          {value(r)}
        </td>
      ))}
    </tr>
  )
}

/** Карточка стратегии для узкой фигуры: четыре метрики и запросы по воркерам. */
function StrategyCard({ row, metrics }: { row: RoutingStrategyResult; metrics: Array<{ label: string; value: string; best: boolean }> }) {
  return (
    <div className="grid min-w-0 gap-2 rounded-panel border border-line p-3">
      <div className="text-[15px] font-semibold text-ink">{MODE_LABEL[row.mode]}</div>
      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 text-[13.5px] tabular-nums">
        {metrics.map((m) => (
          <div key={m.label} className="contents">
            <dt className="text-ink-2">{m.label}</dt>
            <dd className={cn('text-right', m.best ? 'font-semibold text-ink' : 'text-ink-2')}>{m.value}</dd>
          </div>
        ))}
        <dt className="self-end text-ink-2">Запросов на воркер</dt>
        <dd>
          <WorkerBars counts={row.perWorker.map((p) => p.requests)} />
        </dd>
      </dl>
    </div>
  )
}

/** Четыре тонких столбика: сколько запросов досталось каждому воркеру (без подложки-дорожки). */
function WorkerBars({ counts }: { counts: number[] }) {
  const max = Math.max(1, ...counts)
  return (
    <div className="flex items-end justify-end gap-2" role="img" aria-label={`запросов на воркер: ${counts.join(', ')}`}>
      {counts.map((c, i) => (
        <div key={i} className="grid w-5 justify-items-center gap-1" aria-hidden>
          <span className="text-[11px] text-ink-3">{c}</span>
          <span className="w-3 rounded-t-[2px]" style={{ height: `${Math.max(2, (c / max) * 28)}px`, background: cssVar('ink-3') }} />
        </div>
      ))}
    </div>
  )
}
