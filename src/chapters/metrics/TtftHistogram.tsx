import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { ShuffleIcon } from '@phosphor-icons/react'
import { histogram } from '@/engine/sched'
import { Button, Figure, LineChart, Segmented, Slider, Stat, TipRow, fmt, measureText, mix, useElementSize, useFontsReady, useTip } from '@/ui'
import { linearTicks, niceStep } from '@/ui/chart-utils'
import { hwLabel, valueUnit, type Hardware } from '@/chapters/prefill-decode/shared/hardware'
import { HORIZON, POPULATION, STRIDE, WARMUP, rateForLoad, sampleTtft, type QueuePoint } from './queue-sim'

export { meanServiceMs, queueCurve, type QueuePoint } from './queue-sim'

const SAMPLE_SIZES = [200, 2000, 10000] as const
type SampleSize = (typeof SAMPLE_SIZES)[number]
/** По какой нагрузке выбираем общую ось X: при ней p99 ещё на графике, при 98 % уже уходит в «ещё». */
const AXIS_LOAD = 0.95

export function TtftHistogramFigure({ hw, curve }: { hw: Hardware; curve: QueuePoint[] }) {
  const [load, setLoad] = useState(0.7)
  const [size, setSize] = useState<SampleSize>(2000)
  const [seed, setSeed] = useState(1)

  const run = useMemo(() => sampleTtft(hw, load, size, seed), [hw, load, size, seed])
  // Ось X одна на модель и GPU: при росте нагрузки распределение заметно расползается вправо.
  const xMax = useMemo(() => {
    const ref = curve.find((c) => Math.abs(c.rho - AXIS_LOAD) < 1e-9) ?? curve[curve.length - 1]
    const raw = Math.max(1, ref.p99 * 1.1)
    const step = niceStep(raw, 10)
    return Math.ceil(raw / step) * step
  }, [curve])
  const meanHint =
    run.mean > run.p50 * 1.05
      ? `больше медианы в ${fmt.num(run.mean / run.p50, 1)} раза`
      : run.mean < run.p50 * 0.95
        ? 'меньше медианы: хвост короткий'
        : 'почти равно медиане'
  const vu = (ms: number) => valueUnit(fmt.ms(ms))

  return (
    <Figure
      id="metrics-figure-histogram"
      title="TTFT под нагрузкой: распределение и хвост"
      width="wide"
      actions={<span className="block max-w-[18rem] text-[13px] sm:max-w-none leading-snug text-ink-3">{hwLabel(hw)}, одна реплика</span>}
      controls={
        <>
          <Slider
            label="Нагрузка (загрузка GPU prefill-ом)"
            value={load}
            min={0.1}
            max={0.98}
            step={0.01}
            onChange={setLoad}
            format={(v) => `${fmt.pct(v)}, ${fmt.num(rateForLoad(hw, v), 1)} запр/с`}
          />
          <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
            <Segmented<SampleSize>
              label="Запросов в выборке"
              value={size}
              onChange={setSize}
              options={SAMPLE_SIZES.map((s) => ({ value: s, label: fmt.int(s) }))}
            />
            <Button variant="secondary" size="sm" icon={ShuffleIcon} onClick={() => setSeed((s) => s + 1)}>
              Другая выборка
            </Button>
          </div>
        </>
      }
      caption="Попробуйте: поднимите нагрузку с 50 до 95 % и смотрите, как медиана сдвигается вправо, а p99 уезжает намного дальше. Потом выберите 200 запросов и несколько раз нажмите «Другая выборка»: p99 выборки прыгает вокруг p99 всего трафика (он подписан под числом), потому что на маленькой выборке это почти максимум. На 10 000 запросов он почти не двигается."
      source={`Модель упрощённая: одна реплика берёт запросы по очереди, TTFT = ожидание + prefill (время prefill посчитано по модели и GPU), длины промптов случайные. Первые ${fmt.int(WARMUP)} запросов прогона отброшены, пока очередь разгоняется. Из следующих ${fmt.int(HORIZON)} каждый ${STRIDE === 10 ? 'десятый' : `${STRIDE}-й`} (${fmt.int(POPULATION)} запросов) считается всем трафиком, из него и берётся выборка. Длинный прогон нужен при нагрузке 95-98 %: там очередь набегает и рассасывается за десятки тысяч запросов.`}
    >
      <div className="grid gap-5">
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
          <Stat label="p50 (медиана)" value={vu(run.p50)[0]} unit={vu(run.p50)[1]} hint="половина запросов быстрее" />
          <Stat label="p90" value={vu(run.p90)[0]} unit={vu(run.p90)[1]} hint="каждый 10-й медленнее" />
          <Stat label="p99" value={vu(run.p99)[0]} unit={vu(run.p99)[1]} hint={`на всём трафике ${fmt.ms(run.popP99)}`} />
          <Stat label="Среднее" value={vu(run.mean)[0]} unit={vu(run.mean)[1]} hint={meanHint} />
        </div>
        <div className="grid gap-6 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <div className="min-w-0">
            <div className="mb-1 text-[14px] font-medium text-ink">Распределение TTFT, {fmt.int(size)} запросов</div>
            <Histogram values={run.values} p50={run.p50} p90={run.p90} p99={run.p99} xMax={xMax} />
          </div>
          <div className="min-w-0">
            <div className="mb-1 text-[14px] font-medium text-ink">p50 и p99 от нагрузки</div>
            <LineChart
              series={[
                { id: 'p50', label: 'p50', color: 'ink-3', points: curve.map((c) => [c.rho * 100, c.p50]), markers: true },
                { id: 'p99', label: 'p99', color: 'prefill', points: curve.map((c) => [c.rho * 100, c.p99]), markers: true },
              ]}
              xLabel="Нагрузка, %"
              yLabel="TTFT, мс"
              yScale="log"
              xDomain={[0, 100]}
              annotations={[{ kind: 'x', x: load * 100, label: fmt.pct(load) }]}
              height={210}
              xFormat={(v) => fmt.int(v)}
              yFormat={(v) => fmt.compact(v)}
              aria-label="TTFT p50 и p99 в зависимости от нагрузки"
            />
          </div>
        </div>
      </div>
    </Figure>
  )
}

/* ------------------------------------------------------------------------- */

const LABEL_FONT = '500 12px "Onest Variable", sans-serif'
const TICK_FONT = '12px "Onest Variable", sans-serif'

function Histogram({ values, p50, p90, p99, xMax }: { values: Float64Array; p50: number; p90: number; p99: number; xMax: number }) {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement | null>(null)
  useFontsReady()
  const [hover, setHover] = useState<number | null>(null)
  const tip = useTip({ scopeEl: svgRef.current, side: 'top', onOutside: () => setHover(null) })
  const W = Math.max(280, width)
  // сверху три строки подписей перцентилей под подписью оси «Запросов»
  const m = { top: 62, right: 8, bottom: 38, left: 40 }
  const plotH = 190
  const plotW = W - m.left - m.right
  // столбец не уже 9 px: на телефоне меньше столбцов, на десктопе подробнее
  const BINS = Math.max(24, Math.min(60, Math.floor(plotW / 9)))
  const bins = useMemo(() => histogram(Array.from(values), { bins: BINS, min: 0, max: xMax }), [values, xMax, BINS])
  let overflow = 0
  for (let i = values.length - 1; i >= 0 && values[i] > xMax; i--) overflow++
  const counts = bins.map((b) => b.count)
  const binMax = Math.max(...counts, 1)
  // при 98 % в «ещё» уходит четверть запросов: такой столбец не задаёт масштаб, а обрезается сверху с подписью доли
  const overClipped = overflow > binMax * 1.5
  const yMaxRaw = overClipped ? binMax : Math.max(binMax, overflow)
  const yTicks = linearTicks(0, yMaxRaw, 4)
  const yMax = Math.max(yMaxRaw, yTicks[yTicks.length - 1])
  const slots = BINS + (overflow > 0 ? 1.4 : 0)
  const bw = plotW / slots
  const x = (v: number) => m.left + (v / xMax) * BINS * bw
  const y = (c: number) => m.top + plotH - (c / yMax) * plotH
  const overX = m.left + (BINS + 0.4) * bw
  const binsEnd = m.left + BINS * bw
  // последняя подпись оси не должна налезать на подпись «ещё»
  const overLabelLeft = overX + bw / 2 - measureText('ещё', TICK_FONT) / 2
  const xTicks = linearTicks(0, xMax, Math.max(3, Math.floor(plotW / 80))).filter(
    (v) => overflow === 0 || x(v) + measureText(fmt.ms(v), TICK_FONT) / 2 <= Math.min(binsEnd, overLabelLeft - 8),
  )
  const total = values.length

  // Перцентили: линия начинается со своей строки подписи. Подпись справа от линии, если влезает; иначе слева,
  // но не поверх линий выше по списку (иначе она читается как подпись соседней линии); иначе только имя.
  const raw = [
    { id: 'p50', v: p50 },
    { id: 'p90', v: p90 },
    { id: 'p99', v: p99 },
  ]
  const lineX = (v: number) => (v > xMax && overflow > 0 ? overX + bw / 2 - 1 : x(Math.min(v, xMax)))
  const xs = raw.map((l) => lineX(l.v))
  const lines = raw.map((l, k) => {
    const lx = xs[k]
    const leftBound = Math.max(m.left, ...xs.slice(0, k).map((o) => o + 4))
    const options = [`${l.id} ${fmt.ms(l.v)}`, l.id]
    let place = { text: l.id, x: lx - 5, anchor: 'end' as 'start' | 'end' }
    for (const text of options) {
      const w = measureText(text, LABEL_FONT)
      if (lx + 5 + w <= m.left + plotW) {
        place = { text, x: lx + 5, anchor: 'start' }
        break
      }
      if (lx - 5 - w >= leftBound) {
        place = { text, x: lx - 5, anchor: 'end' }
        break
      }
    }
    return { ...l, lx, row: k, ...place }
  })

  const lastIndex = overflow > 0 ? BINS : BINS - 1
  const binContent = (i: number) => {
    const isOver = i === BINS
    const count = isOver ? overflow : bins[i].count
    const label = isOver ? `дольше ${fmt.ms(xMax)}` : `от ${fmt.ms(bins[i].x0)} до ${fmt.ms(bins[i].x1)}`
    return <TipRow value={`${fmt.int(count)} (${fmt.pct(count / total)})`} label={label} />
  }
  const showBin = (e: PointerEvent<SVGRectElement>, i: number) => {
    setHover(i)
    tip.show(e.currentTarget.getBoundingClientRect(), binContent(i))
  }
  const clear = (e: PointerEvent<SVGRectElement>) => {
    if (e.pointerType !== 'mouse') return
    setHover(null)
    tip.hide()
  }
  // клавиатура: стрелки ходят по столбцам, тултип у вершины столбца
  const showAt = (i: number) => {
    const svg = svgRef.current
    if (!svg) return
    setHover(i)
    const box = svg.getBoundingClientRect()
    const count = i === BINS ? overflow : bins[i].count
    const bx = i === BINS ? overX : m.left + i * bw
    tip.show({ left: box.left + bx, top: box.top + y(count), width: bw, height: 1 }, binContent(i))
  }
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    let i = hover ?? -1
    if (e.key === 'ArrowRight') i = Math.min(lastIndex, i + 1)
    else if (e.key === 'ArrowLeft') i = i < 0 ? lastIndex : Math.max(0, i - 1)
    else if (e.key === 'Home') i = 0
    else if (e.key === 'End') i = lastIndex
    else if (e.key === 'Escape') {
      setHover(null)
      tip.hide()
      return
    } else return
    e.preventDefault()
    showAt(i)
  }

  return (
    <div
      ref={ref}
      tabIndex={0}
      role="group"
      aria-label="Гистограмма TTFT: стрелки влево и вправо переходят по столбцам"
      onKeyDown={onKeyDown}
      onBlur={() => {
        setHover(null)
        tip.hide()
      }}
      className="min-w-0 rounded-control focus-visible:outline-offset-4"
    >
      {width > 0 && (
        <svg
          ref={svgRef}
          width={W}
          height={m.top + plotH + m.bottom}
          role="img"
          aria-label={`Гистограмма TTFT: p50 ${fmt.ms(p50)}, p90 ${fmt.ms(p90)}, p99 ${fmt.ms(p99)}`}
          className="block select-none overflow-visible"
        >
          <text x={0} y={12} className="fill-ink-3 text-[12px]">
            Запросов
          </text>
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={m.left} x2={m.left + plotW} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
              <text x={m.left - 6} y={y(v)} textAnchor="end" dominantBaseline="central" className="fill-ink-3 text-[12px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {fmt.int(v)}
              </text>
            </g>
          ))}
          {bins.map((b, i) => (
            <rect
              key={i}
              x={m.left + i * bw + 0.5}
              y={y(b.count)}
              width={Math.max(1, bw - 1.5)}
              height={Math.max(0, m.top + plotH - y(b.count))}
              rx={1.5}
              fill={mix('prefill', hover === i ? 85 : 60)}
              onPointerMove={(e) => showBin(e, i)}
              onPointerDown={(e) => showBin(e, i)}
              onPointerLeave={clear}
            />
          ))}
          {overflow > 0 && (
            <g>
              <rect
                x={overX}
                y={y(Math.min(overflow, yMax))}
                width={Math.max(1, bw - 1.5)}
                height={Math.max(0, m.top + plotH - y(Math.min(overflow, yMax)))}
                rx={1.5}
                fill={mix('prefill', hover === BINS ? 85 : 60)}
                onPointerMove={(e) => showBin(e, BINS)}
                onPointerDown={(e) => showBin(e, BINS)}
                onPointerLeave={clear}
              />
              <text x={overX + bw / 2} y={m.top + plotH + 17} textAnchor="middle" className="fill-ink-3 text-[12px]">
                ещё
              </text>
              {overClipped && (
                <text
                  x={overX - 4}
                  y={m.top + 12}
                  textAnchor="end"
                  className="fill-ink-2 text-[12px] font-medium"
                  style={{ paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4, strokeLinejoin: 'round' }}
                >
                  {fmt.pct(overflow / total)} дольше {fmt.ms(xMax)}
                </text>
              )}
            </g>
          )}
          {/* перцентили */}
          {lines.map((l) => (
            <g key={l.id} pointerEvents="none">
              <line x1={l.lx} x2={l.lx} y1={m.top - 40 + l.row * 14} y2={m.top + plotH} stroke="var(--ink)" strokeWidth={1.25} strokeDasharray="4 3" />
              <text
                x={l.x}
                y={m.top - 31 + l.row * 14}
                textAnchor={l.anchor}
                className="fill-ink text-[12px] font-medium"
                style={{ paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4, strokeLinejoin: 'round' }}
              >
                {l.text}
              </text>
            </g>
          ))}
          <line x1={m.left} x2={m.left + plotW} y1={m.top + plotH} y2={m.top + plotH} stroke="var(--line-strong)" />
          {xTicks.map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={m.top + plotH} y2={m.top + plotH + 4} stroke="var(--line-strong)" />
              <text x={x(v)} y={m.top + plotH + 17} textAnchor={v === 0 ? 'start' : 'middle'} className="fill-ink-3 text-[12px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {fmt.ms(v)}
              </text>
            </g>
          ))}
          <text x={m.left + plotW} y={m.top + plotH + 34} textAnchor="end" className="fill-ink-3 text-[12px]">
            TTFT
          </text>
        </svg>
      )}
      {tip.node}
    </div>
  )
}
