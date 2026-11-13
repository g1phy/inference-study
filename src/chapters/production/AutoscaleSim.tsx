import { useMemo, useState } from 'react'
import { Figure, LineChart, MEDIA, Segmented, Slider, Stat, Toggle, fmt, useMediaQuery, type LineAnnotation } from '@/ui'
import { ChipGroup } from './ChipGroup'
import { simulateAutoscale, violationBands, type ScaleSignal, type TrafficPreset } from './autoscale'

const SIGNAL_NOTE: Record<ScaleSignal, string> = {
  concurrency:
    'Сигнал: сколько запросов в системе (обрабатываются и ждут) на одну реплику, цель 70 % от concurrency target. Очередь сразу видна автоскейлеру.',
  gpu: 'Сигнал: utilization.gpu из nvidia-smi, цель 70 %. Это доля времени, когда на GPU работает хоть одно ядро (kernel, программа на GPU): уже один запрос на реплике даёт почти 100 %, поэтому сигнал не отличает «чуть занята» от «перегружена» и раздувает реплики до максимума.',
  cpu: 'Сигнал: загрузка CPU, цель 70 %. Работу делает GPU, а CPU сервера модели почти не загружен при любом трафике: автоскейлер думает, что всё спокойно, и держит минимум реплик.',
}

type Zoom = 'day' | 'spike' | 'morning'

/** Окна графиков: сутки целиком или два часа вокруг события, чтобы задержка cold start была видна. */
const ZOOM: Record<Zoom, { domain: [number, number]; ticks: number[]; step: number }> = {
  day: { domain: [0, 24], ticks: [0, 6, 12, 18, 24], step: 2 },
  spike: { domain: [13.5, 15.5], ticks: [13.5, 13.75, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5], step: 1 },
  morning: { domain: [6, 8], ticks: [6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8], step: 1 },
}

const fmtClock = (h: number) => {
  const total = Math.round(h * 60)
  const hh = Math.floor(total / 60)
  const mm = total % 60
  return `${hh}:${String(mm).padStart(2, '0')}`
}

const fmtWait = (s: number) => (!Number.isFinite(s) ? 'больше суток' : s < 1 ? '0 с' : s < 90 ? `${fmt.int(Math.round(s))} с` : s < 5400 ? `${fmt.num(s / 60, 1)} мин` : `${fmt.num(s / 3600, 1)} ч`)

export function AutoscaleSim() {
  const [signal, setSignal] = useState<ScaleSignal>('concurrency')
  const [preset, setPreset] = useState<TrafficPreset>('chat')
  const [spike, setSpike] = useState(true)
  const [cold, setCold] = useState(5)
  const [delay, setDelay] = useState(10)
  const [minR, setMinR] = useState(1)
  const [zoomPick, setZoom] = useState<Zoom>('spike')
  const desk = useMediaQuery(MEDIA.desk)
  // без всплеска окно «Всплеск» пустое: показываем сутки
  const zoom: Zoom = !spike && zoomPick === 'spike' ? 'day' : zoomPick
  const view = ZOOM[zoom]

  const res = useMemo(
    () => simulateAutoscale({ preset, spike, signal, coldStartMin: cold, scaleDownDelayMin: delay, minReplicas: minR, maxReplicas: 32 }),
    [preset, spike, signal, cold, delay, minR],
  )
  const ideal = useMemo(
    () => simulateAutoscale({ preset, spike, signal: 'concurrency', coldStartMin: 0, scaleDownDelayMin: 0, minReplicas: 0, maxReplicas: 32 }),
    [preset, spike],
  )

  const { demandSeries, replicaSeries, bands, yMax } = useMemo(() => {
    const [h0, h1] = view.domain
    const inView = (m: { t: number }) => m.t / 60 >= h0 - 1 / 60 && m.t / 60 <= h1 + 1 / 60
    const pts = res.minutes.filter((m) => m.t % view.step === 0 && inView(m))
    const visible = res.minutes.filter(inView)
    const cap = Math.max(1, ...visible.map((m) => Math.max(m.demand, m.capacity)))
    const demand: Array<[number, number]> = pts.map((m) => [m.t / 60, m.demand])
    const capacity: Array<[number, number]> = pts.map((m) => [m.t / 60, m.capacity])
    const ready: Array<[number, number]> = pts.map((m) => [m.t / 60, m.ready])
    const total: Array<[number, number]> = pts.map((m) => [m.t / 60, m.total])
    // полосы нарушений только внутри окна (аннотации графика не обрезаются по области построения)
    const b = violationBands(res.minutes)
      .map(([a, z]) => [Math.max(a / 60, h0), Math.min(z / 60, h1)] as [number, number])
      .filter(([a, z]) => z > a)
    const widest = b.reduce((w, x, i) => (x[1] - x[0] > b[w][1] - b[w][0] ? i : w), 0)
    const annotations: LineAnnotation[] = b.map(([a, z], i) => ({
      kind: 'band',
      x: a,
      x2: z,
      label: i === widest && (z - a) / (h1 - h0) >= 0.02 ? 'очередь дольше 2 с' : undefined,
    }))
    return {
      demandSeries: [
        { id: 'demand', label: 'нужно слотов', color: 'prefill', area: true, points: demand },
        { id: 'capacity', label: 'ёмкость готовых реплик', color: 'ink', dashed: true, points: capacity },
      ],
      replicaSeries: [
        { id: 'total', label: 'оплачено (с учётом стартующих)', color: 'ink-3', dashed: true, points: total },
        { id: 'ready', label: 'готовы', color: 'ink', points: ready },
      ],
      bands: annotations,
      yMax: cap,
    }
  }, [res, view])

  const overpay = ideal.gpuHours > 0 ? res.gpuHours / ideal.gpuHours - 1 : 0
  const yStep = yMax > 300 ? 50 : yMax > 60 ? 20 : 5

  const primary = (
    <>
      <ChipGroup
        label="Сигнал автоскейлера"
        value={signal}
        onChange={setSignal}
        options={[
          { value: 'concurrency', label: 'Concurrency' },
          { value: 'gpu', label: 'GPU util' },
          { value: 'cpu', label: 'CPU' },
        ]}
      />
      <ChipGroup
        label="Трафик"
        value={preset}
        onChange={setPreset}
        options={[
          { value: 'chat', label: 'Чат' },
          { value: 'b2b', label: 'Рабочие часы' },
        ]}
      />
      <Slider label="Cold start" value={cold} min={1} max={20} onChange={setCold} format={(v) => `${fmt.int(v)} мин`} hint="от заказа реплики до первого ответа" />
    </>
  )
  const secondary = (
    <>
      <Toggle label="Всплеск в 14:00" checked={spike} onChange={setSpike} hint="трафик вдвое выше на 45 минут" />
      <Slider label="Задержка уменьшения" value={delay} min={0} max={60} step={5} onChange={setDelay} format={(v) => `${fmt.int(v)} мин`} hint="scale-down delay" />
      <Slider label="Минимум реплик" value={minR} min={0} max={4} onChange={setMinR} format={(v) => fmt.int(v)} hint="0 означает scale to zero" />
    </>
  )

  return (
    <Figure
      id="production-figure-autoscale"
      title="Сутки трафика: реплики догоняют спрос"
      // на десктопе все ручки справа от графиков; на узком экране главные ручки над графиками, остальные под ними
      inputs={desk ? undefined : <div className="grid w-full gap-4 sm:grid-cols-2">{primary}</div>}
      controls={
        desk ? (
          <>
            {primary}
            {secondary}
          </>
        ) : (
          secondary
        )
      }
      controlsPosition={desk ? 'side' : 'bottom'}
      caption="Попробуйте: на масштабе «Всплеск» поднимите cold start с 5 до 15 минут: реплики заказаны вовремя, но очередь держится дольше. Затем переключите сигнал на CPU и на GPU util и посмотрите на сутки целиком."
      source="Модель условная: реплика держит 32 запроса одновременно, запрос занимает слот 3 секунды, автоскейлер раз в минуту целится в 70 % загрузки, как Knative."
    >
      <div className="grid min-w-0 gap-5">
        <p className="text-[14px] leading-snug text-ink-2">{SIGNAL_NOTE[signal]}</p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
          <Stat label="GPU-часов в сутки" value={fmt.num(res.gpuHours, 0)} hint={overpay > 0.02 ? `на ${fmt.pct(overpay, 0)} больше, чем при мгновенном старте` : overpay < -0.05 ? `на ${fmt.pct(-overpay, 0)} меньше нужного: реплик не хватает` : 'почти как при мгновенном старте'} />
          <Stat label="Ждали дольше 2 с" value={fmt.pct(res.violatedShare, res.violatedShare < 0.1 ? 1 : 0)} hint={`доля запросов; ${fmt.count(res.violationMinutes, ['минута', 'минуты', 'минут'])} с нарушением SLO`} />
          <Stat label="Худшее ожидание" value={fmtWait(res.maxWaitSec)} hint="в очереди до начала prefill" />
          <Stat label="Пик реплик" value={fmt.int(res.peakReplicas)} hint="максимум 32" />
        </div>
        <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
          <Segmented
            label="Масштаб графиков"
            size="sm"
            value={zoom}
            onChange={setZoom}
            options={[
              { value: 'day', label: 'Сутки' },
              { value: 'spike', label: 'Всплеск', disabled: !spike, title: spike ? '13:30 до 15:30' : 'включите всплеск' },
              { value: 'morning', label: 'Утро' },
            ]}
          />
          <span className="text-[13px] text-ink-3">
            {zoom === 'day' ? 'статистика выше за все сутки' : `окно ${fmtClock(view.domain[0])} до ${fmtClock(view.domain[1])}, статистика выше за все сутки`}
          </span>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-sm font-medium text-ink-2">Запросы в обработке: сколько нужно и сколько помещается</div>
          <LineChart
            series={demandSeries}
            xLabel="время суток"
            yLabel="одновременных запросов"
            height={210}
            xDomain={view.domain}
            xTicks={zoom === 'day' ? view.ticks : view.ticks.filter((_, i) => i % 2 === 0)}
            yDomain={[0, Math.ceil((yMax * 1.08) / yStep) * yStep]}
            xFormat={fmtClock}
            yFormat={(v) => fmt.int(v)}
            annotations={bands}
            directLabels={false}
            aria-label="Спрос и ёмкость реплик"
          />
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-sm font-medium text-ink-2">Реплики</div>
          <LineChart
            series={replicaSeries}
            xLabel="время суток"
            yLabel="реплик"
            height={110}
            xDomain={view.domain}
            xTicks={zoom === 'day' ? view.ticks : view.ticks.filter((_, i) => i % 2 === 0)}
            xFormat={fmtClock}
            yFormat={(v) => fmt.int(v)}
            directLabels={false}
            aria-label="Готовые и оплачиваемые реплики"
          />
        </div>
        <p className="text-[13px] text-ink-3">
          Серые полосы на верхнем графике: минуты, когда новый запрос ждал в очереди дольше 2 секунд. Зазор между линиями на нижнем: реплики,
          которые уже оплачены, но ещё стартуют.
        </p>
      </div>
    </Figure>
  )
}
