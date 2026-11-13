import { useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { cn } from './cn'
import { resolveColor } from './color'
import { fontCss, linearTicks, logDomain, logTicks, makeScale, nearestIndex, niceScale, spreadLabels, type ScaleKind } from './chart-utils'
import { TipRow, useTip } from './floating'
import { fmt } from './fmt'
import { measureText, useElementSize, useFontsReady } from './hooks'

export interface LineSeries {
  id: string
  label: string
  /** Роль ('prefill', 'decode', 'q', …) или CSS-цвет. */
  color?: string
  /** Точки [x, y]; NaN в y разрывает линию. */
  points: Array<[number, number]>
  /** Пунктир (теория, оценка, предел). */
  dashed?: boolean
  /** Заливка под линией (10% цвета серии). */
  area?: boolean
  /** Маркеры точек для этой серии. */
  markers?: boolean
}

export interface LineAnnotation {
  /** x: вертикальная линия; y: горизонтальная (порог, SLO); point: точка; band: полоса x..x2. */
  kind: 'x' | 'y' | 'point' | 'band'
  x?: number
  y?: number
  x2?: number
  label?: string
  /** Цвет метки (по умолчанию нейтральный --ink-3). */
  color?: string
  /** Для kind 'y': подпись у левого (по умолчанию) или правого края. */
  labelAlign?: 'start' | 'end'
  /** Для kind 'point': подпись над точкой (по умолчанию) или под ней (на растущей кривой подпись над точкой ложится на линию). */
  labelPosition?: 'above' | 'below'
}

export interface LineChartProps {
  series: LineSeries[]
  xLabel: string
  yLabel: string
  xScale?: ScaleKind
  yScale?: ScaleKind
  /** Маркеры во всех точках. */
  markers?: boolean
  annotations?: LineAnnotation[]
  /** Высота области построения (без легенды), px. */
  height?: number
  xDomain?: [number, number]
  yDomain?: [number, number]
  xFormat?: (v: number) => string
  yFormat?: (v: number) => string
  xTicks?: number[]
  yTicks?: number[]
  /** Легенда: по умолчанию при 2+ сериях. */
  legend?: boolean
  /** Прямые подписи у концов линий: по умолчанию при 2-4 сериях и ширине от 420px. */
  directLabels?: boolean
  onHoverX?: (x: number | null) => void
  'aria-label'?: string
  className?: string
}

const TICK_FONT = '12px "Onest Variable", sans-serif'
/** Длинными свойствами: короткое font рядом с fontVariantNumeric React ругает при смене (см. fontCss). */
const TICK_CSS = fontCss(TICK_FONT)
const warnedColors = new Set<string>()
/** Ореол цвета фона под подписями поверх линий (фон графика задаётся --chart-bg, по умолчанию --surface). */
const HALO = { paintOrder: 'stroke', stroke: 'var(--chart-bg, var(--surface))', strokeWidth: 4, strokeLinejoin: 'round' } as const
const LABEL_FONT = '13px "Onest Variable", sans-serif'

/**
 * Линейный график (SVG): одна ось Y, линейные или логарифмические шкалы, едва видная сетка,
 * легенда и прямые подписи, crosshair с тултипом по всем сериям (мышь и стрелки), аннотации.
 *
 *   <LineChart xLabel="Арифметическая интенсивность, FLOP/байт" yLabel="TFLOPS" xScale="log" yScale="log"
 *     series={[{ id: 'h100', label: 'H100', color: 'prefill', points }]}
 *     annotations={[{ kind: 'x', x: 295, label: 'ridge point' }]} />
 */
export function LineChart({
  series,
  xLabel,
  yLabel,
  xScale = 'linear',
  yScale = 'linear',
  markers = false,
  annotations = [],
  height = 260,
  xDomain,
  yDomain,
  xFormat = (v) => fmt.compact(v),
  yFormat = (v) => fmt.compact(v),
  xTicks,
  yTicks,
  legend,
  directLabels,
  onHoverX,
  'aria-label': ariaLabel,
  className,
}: LineChartProps) {
  const uid = useId().replace(/:/g, '')
  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement | null>(null)
  useFontsReady()
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const tip = useTip({
    scopeEl: svgRef.current,
    side: 'right',
    onOutside: () => {
      setHoverIdx(null)
      onHoverX?.(null)
    },
  })

  if (import.meta.env.DEV && series.length >= 2 && series.filter((s) => !s.color).length >= 2 && !warnedColors.has(yLabel)) {
    warnedColors.add(yLabel)
    console.warn(`[LineChart] «${yLabel}»: у нескольких серий нет color, все они нарисуются цветом prefill. Задайте цвет сущности каждой серии.`)
  }
  const showLegend = legend ?? series.length >= 2
  const useDirect = (directLabels ?? (series.length >= 2 && series.length <= 4)) && width >= 420

  const model = useMemo(() => {
    const xs: number[] = []
    const ys: number[] = []
    for (const s of series)
      for (const [x, y] of s.points) {
        if (Number.isFinite(x) && (xScale === 'linear' || x > 0)) xs.push(x)
        if (Number.isFinite(y) && (yScale === 'linear' || y > 0)) ys.push(y)
      }
    for (const a of annotations) {
      if (a.x !== undefined && Number.isFinite(a.x)) xs.push(a.x)
      if (a.x2 !== undefined && Number.isFinite(a.x2)) xs.push(a.x2)
      if (a.y !== undefined && Number.isFinite(a.y)) ys.push(a.y)
    }
    const xmin = xs.length ? Math.min(...xs) : 0
    const xmax = xs.length ? Math.max(...xs) : 1
    const ymin = ys.length ? Math.min(...ys) : 0
    const ymax = ys.length ? Math.max(...ys) : 1
    // X: точно по данным (от нуля, если данные начинаются почти с нуля)
    const xStart = xmin >= 0 && xmin <= (xmax - xmin) * 0.15 ? 0 : xmin
    const xd: [number, number] = xDomain ?? (xScale === 'log' ? logDomain(xmin, xmax) : [xStart, xmax === xStart ? xStart + 1 : xmax])
    // Y: от нуля, верх домена совпадает с последним тиком
    const yCount = Math.max(3, Math.floor(height / 56))
    const yNice = yScale === 'linear' && !yDomain ? niceScale(Math.min(0, ymin), ymax, yCount) : null
    const yd: [number, number] = yDomain ?? (yScale === 'log' ? logDomain(ymin, ymax) : yNice!.domain)
    const uniqueX = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])).filter((x) => Number.isFinite(x)))].sort((a, b) => a - b)
    return { xd, yd, uniqueX, yNiceTicks: yNice?.ticks, yCount }
  }, [series, annotations, xScale, yScale, xDomain, yDomain, height])

  const plotH = height
  const yTickVals =
    yTicks ?? (yScale === 'log' ? logTicks(model.yd[0], model.yd[1]) : (model.yNiceTicks ?? linearTicks(model.yd[0], model.yd[1], model.yCount)))
  const yTickW = Math.max(0, ...yTickVals.map((v) => measureText(yFormat(v), TICK_FONT)))
  const labelW = useDirect ? Math.max(0, ...series.map((s) => measureText(s.label, LABEL_FONT))) : 0
  const m = { top: 30, right: useDirect ? Math.min(labelW + 18, width * 0.3) : 14, bottom: 42, left: Math.ceil(yTickW) + 12 }
  const plotW = Math.max(40, width - m.left - m.right)
  const x = makeScale(xScale, model.xd, [m.left, m.left + plotW])
  const y = makeScale(yScale, model.yd, [m.top + plotH, m.top])
  const xTickVals = xTicks ?? (xScale === 'log' ? logTicks(model.xd[0], model.xd[1]) : linearTicks(model.xd[0], model.xd[1], Math.max(2, Math.floor(plotW / 90))))
  const svgH = m.top + plotH + m.bottom

  const colorOf = (s: LineSeries) => resolveColor(s.color, 'prefill')

  const paths = series.map((s) => {
    let d = ''
    let pen = false
    for (const [px, py] of s.points) {
      const ok = Number.isFinite(px) && Number.isFinite(py) && (xScale === 'linear' || px > 0) && (yScale === 'linear' || py > 0)
      if (!ok) {
        pen = false
        continue
      }
      d += `${pen ? 'L' : 'M'}${x(px).toFixed(2)},${y(py).toFixed(2)}`
      pen = true
    }
    const valid = s.points.filter(([px, py]) => Number.isFinite(px) && Number.isFinite(py))
    const base = yScale === 'log' ? m.top + plotH : y(Math.max(model.yd[0], Math.min(0, model.yd[1])))
    const area =
      s.area && valid.length > 1
        ? `M${x(valid[0][0])},${base}` + valid.map(([px, py]) => `L${x(px)},${y(py)}`).join('') + `L${x(valid[valid.length - 1][0])},${base}Z`
        : ''
    return { d, area, last: valid[valid.length - 1] }
  })

  const endLabels = useDirect
    ? (() => {
        const raw = paths.map((p) => (p.last ? y(p.last[1]) : m.top))
        const spread = spreadLabels(raw, 16, m.top, m.top + plotH)
        return series.map((s, i) => ({ s, i, y0: raw[i], y: spread[i], x0: paths[i].last ? x(paths[i].last![0]) : m.left + plotW }))
      })()
    : []

  const hoverX = hoverIdx !== null ? model.uniqueX[hoverIdx] : null

  const showAt = (idx: number) => {
    const X = model.uniqueX[idx]
    if (X === undefined) return
    setHoverIdx(idx)
    onHoverX?.(X)
    const svg = svgRef.current
    if (!svg) return
    const box = svg.getBoundingClientRect()
    const rows = series
      .map((s) => {
        const pts = s.points.filter((p) => Number.isFinite(p[1]))
        const xsSorted = pts.map((p) => p[0])
        const k = nearestIndex(xsSorted, X)
        return k >= 0 ? { s, p: pts[k] } : null
      })
      .filter((r): r is { s: LineSeries; p: [number, number] } => r !== null)
      .sort((a, b) => b.p[1] - a.p[1])
    tip.show(
      { left: box.left + x(X), top: box.top + m.top + plotH / 2, width: 0, height: 0 },
      <div className="grid gap-1">
        <div className="text-xs text-ink-3">
          {xLabel}: <span className="font-medium text-ink-2 tabular-nums">{xFormat(X)}</span>
        </div>
        {rows.map(({ s, p }) => (
          <TipRow key={s.id} color={colorOf(s)} dashed={s.dashed} value={yFormat(p[1])} label={series.length > 1 ? s.label : yLabel} />
        ))}
      </div>,
    )
  }
  const clear = () => {
    setHoverIdx(null)
    onHoverX?.(null)
    tip.hide()
  }

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    const box = svgRef.current!.getBoundingClientRect()
    const px = e.clientX - box.left
    const pxs = model.uniqueX.map((v) => x(v))
    const idx = nearestIndex(pxs, px)
    if (idx >= 0 && idx !== hoverIdx) showAt(idx)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const n = model.uniqueX.length
    if (!n) return
    let i = hoverIdx ?? -1
    if (e.key === 'ArrowRight') i = Math.min(n - 1, i + 1)
    else if (e.key === 'ArrowLeft') i = i < 0 ? n - 1 : Math.max(0, i - 1)
    else if (e.key === 'Home') i = 0
    else if (e.key === 'End') i = n - 1
    else if (e.key === 'Escape') return clear()
    else return
    e.preventDefault()
    showAt(i)
  }

  const annColor = (a: LineAnnotation) => (a.color ? resolveColor(a.color) : 'var(--ink-3)')

  return (
    <div className={cn('min-w-0', className)}>
      {showLegend && (
        <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
          {series.map((s) => (
            <li key={s.id} className="inline-flex items-center gap-2">
              <span aria-hidden className="inline-block w-4 border-t-2" style={{ borderColor: colorOf(s), borderTopStyle: s.dashed ? 'dashed' : 'solid' }} />
              {s.label}
            </li>
          ))}
        </ul>
      )}
      <div
        ref={wrapRef}
        tabIndex={0}
        role="group"
        aria-label={ariaLabel ?? `График: ${yLabel} от ${xLabel}. Стрелки перемещают по точкам`}
        onKeyDown={onKeyDown}
        onBlur={clear}
        className="w-full rounded-control focus-visible:outline-offset-4"
      >
        {width > 0 && (
          <svg ref={svgRef} width={width} height={svgH} role="img" aria-hidden className="block select-none overflow-visible">
            <defs>
              <clipPath id={`clip-${uid}`}>
                <rect x={m.left} y={m.top - 4} width={plotW} height={plotH + 8} />
              </clipPath>
            </defs>

            {/* подпись оси Y над графиком, горизонтально */}
            <text x={0} y={14} style={{ font: TICK_FONT, fill: 'var(--ink-3)' }}>
              {yLabel}
            </text>

            {/* сетка и тики Y */}
            {yTickVals.map((v) => (
              <g key={`y${v}`}>
                <line x1={m.left} x2={m.left + plotW} y1={y(v)} y2={y(v)} style={{ stroke: 'var(--grid)', strokeWidth: 1 }} />
                <text x={m.left - 8} y={y(v)} textAnchor="end" dominantBaseline="central" style={{ ...TICK_CSS, fill: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>
                  {yFormat(v)}
                </text>
              </g>
            ))}

            {/* ось X и тики */}
            <line x1={m.left} x2={m.left + plotW} y1={m.top + plotH} y2={m.top + plotH} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
            {xTickVals.map((v) => (
              <g key={`x${v}`}>
                <line x1={x(v)} x2={x(v)} y1={m.top + plotH} y2={m.top + plotH + 4} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
                <text x={x(v)} y={m.top + plotH + 17} textAnchor="middle" style={{ ...TICK_CSS, fill: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>
                  {xFormat(v)}
                </text>
              </g>
            ))}
            <text x={m.left + plotW} y={svgH - 4} textAnchor="end" style={{ font: TICK_FONT, fill: 'var(--ink-3)' }}>
              {xLabel}
            </text>

            {/* аннотации под линиями */}
            {(() => {
              // подписи вертикальных отметок (x): где они стоят, чтобы подпись полосы их не перекрыла
              const xLabelBoxes = annotations
                .filter((a) => a.kind === 'x' && a.x !== undefined && a.label)
                .map((a) => {
                  const ax = x(a.x!)
                  const lw = measureText(a.label!, TICK_FONT)
                  const left = ax + 6 + lw > m.left + plotW - 2 && ax - 6 - lw >= m.left
                  return left ? [ax - 6 - lw, ax - 6] : [ax + 6, ax + 6 + lw]
                })
              return annotations.map((a, i) => {
                if (a.kind === 'band' && a.x !== undefined && a.x2 !== undefined) {
                  const x1 = x(a.x)
                  const x2 = x(a.x2)
                  let label: { x: number; y: number; anchor: 'start' | 'end' } | null = null
                  if (a.label) {
                    const lw = measureText(a.label, TICK_FONT)
                    // подпись у правого края полосы внутри области построения; если полоса уже подписи
                    // у левого края графика, подпись начинается от левого края полосы
                    const right = Math.min(Math.max(x1, x2), m.left + plotW) - 6
                    const startX = Math.max(m.left + 2, Math.min(Math.min(x1, x2) + 6, m.left + plotW - lw - 4))
                    const useRight = right - lw >= m.left + 2
                    const lx0 = useRight ? right - lw : startX
                    const lx1 = useRight ? right : startX + lw
                    // столкнулась с подписью отметки x: уходит на вторую строку
                    const hit = xLabelBoxes.some(([b0, b1]) => b0 < lx1 + 4 && b1 > lx0 - 4)
                    label = { x: useRight ? right : startX, y: hit ? m.top + 30 : m.top + 14, anchor: useRight ? 'end' : 'start' }
                  }
                  return (
                    <g key={`a${i}`}>
                      <rect x={Math.min(x1, x2)} y={m.top} width={Math.abs(x2 - x1)} height={plotH} style={{ fill: 'var(--ink)', opacity: 0.05 }} />
                      {label && (
                        <text x={label.x} y={label.y} textAnchor={label.anchor} style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}>
                          {a.label}
                        </text>
                      )}
                    </g>
                  )
                }
                if (a.kind === 'x' && a.x !== undefined) {
                  const ax = x(a.x)
                  // подпись справа от линии, если помещается в область построения (правое поле занято прямыми
                  // подписями серий), иначе слева; если не помещается и слева, остаётся справа
                  const lw = a.label ? measureText(a.label, TICK_FONT) : 0
                  const left = a.label ? ax + 6 + lw > m.left + plotW - 2 && ax - 6 - lw >= m.left : false
                  return (
                    <g key={`a${i}`}>
                      <line x1={ax} x2={ax} y1={m.top} y2={m.top + plotH} style={{ stroke: annColor(a), strokeWidth: 1, strokeDasharray: '4 3' }} />
                      {a.label && (
                        <text x={left ? ax - 6 : ax + 6} y={m.top + 12} textAnchor={left ? 'end' : 'start'} style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}>
                          {a.label}
                        </text>
                      )}
                    </g>
                  )
                }
                if (a.kind === 'y' && a.y !== undefined) {
                  const ay = y(a.y)
                  return (
                    <g key={`a${i}`}>
                      <line x1={m.left} x2={m.left + plotW} y1={ay} y2={ay} style={{ stroke: annColor(a), strokeWidth: 1, strokeDasharray: '4 3' }} />
                      {a.label && (
                        <text
                          x={a.labelAlign === 'end' ? m.left + plotW - 4 : m.left + 6}
                          y={ay - 6}
                          textAnchor={a.labelAlign === 'end' ? 'end' : 'start'}
                          style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}
                        >
                          {a.label}
                        </text>
                      )}
                    </g>
                  )
                }
                return null
              })
            })()}

            {/* серии */}
            <g clipPath={`url(#clip-${uid})`}>
              {series.map((s, i) =>
                paths[i].area ? <path key={`ar${s.id}`} d={paths[i].area} style={{ fill: colorOf(s), opacity: 0.1 }} /> : null,
              )}
              {series.map((s, i) => (
                <path
                  key={s.id}
                  d={paths[i].d}
                  style={{
                    fill: 'none',
                    stroke: colorOf(s),
                    strokeWidth: 2,
                    strokeLinejoin: 'round',
                    strokeLinecap: 'round',
                    strokeDasharray: s.dashed ? '6 4' : undefined,
                  }}
                />
              ))}
              {series.map((s) =>
                markers || s.markers
                  ? s.points
                      .filter(([px, py]) => Number.isFinite(px) && Number.isFinite(py))
                      .map(([px, py], k) => (
                        <circle key={`${s.id}-${k}`} cx={x(px)} cy={y(py)} r={4} style={{ fill: colorOf(s), stroke: 'var(--surface)', strokeWidth: 2 }} />
                      ))
                  : null,
              )}
            </g>

            {/* точки-аннотации поверх */}
            {annotations.map((a, i) => {
              if (a.kind !== 'point' || a.x === undefined || a.y === undefined) return null
              const px = x(a.x)
              // правое поле занято прямыми подписями серий: подпись точки не заходит туда
              const limit = useDirect ? m.left + plotW : width - 2
              const flip = a.label ? px + 9 + measureText(a.label, TICK_FONT) > limit : false
              return (
                <g key={`p${i}`}>
                  <circle cx={px} cy={y(a.y)} r={5} style={{ fill: a.color ? resolveColor(a.color) : 'var(--ink)', stroke: 'var(--surface)', strokeWidth: 2 }} />
                  {a.label && (
                    <text
                      x={flip ? px - 9 : px + 9}
                      y={a.labelPosition === 'below' ? y(a.y) + 16 : y(a.y) - 8}
                      textAnchor={flip ? 'end' : 'start'}
                      style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}
                    >
                      {a.label}
                    </text>
                  )}
                </g>
              )
            })}

            {/* прямые подписи у концов линий (с выносками при сдвиге) */}
            {endLabels.map(({ s, y: ly, y0, x0 }) => (
              <g key={`l${s.id}`}>
                {Math.abs(ly - y0) > 3 && (
                  <line x1={x0 + 3} y1={y0} x2={m.left + plotW + 8} y2={ly} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
                )}
                <text x={m.left + plotW + 10} y={ly} dominantBaseline="central" style={{ font: LABEL_FONT, fill: 'var(--ink)' }}>
                  {s.label}
                </text>
              </g>
            ))}

            {/* crosshair */}
            {hoverX !== null && (
              <g pointerEvents="none">
                <line x1={x(hoverX)} x2={x(hoverX)} y1={m.top} y2={m.top + plotH} style={{ stroke: 'var(--ink-3)', strokeWidth: 1 }} />
                {series.map((s) => {
                  const pts = s.points.filter((p) => Number.isFinite(p[1]))
                  const k = nearestIndex(
                    pts.map((p) => p[0]),
                    hoverX,
                  )
                  if (k < 0) return null
                  const [px, py] = pts[k]
                  if (yScale === 'log' && py <= 0) return null
                  return <circle key={`h${s.id}`} cx={x(px)} cy={y(py)} r={4.5} style={{ fill: colorOf(s), stroke: 'var(--surface)', strokeWidth: 2 }} />
                })}
              </g>
            )}

            {/* слой для мыши */}
            <rect
              x={m.left}
              y={m.top}
              width={plotW}
              height={plotH}
              fill="transparent"
              onPointerMove={onPointerMove}
              // тап не порождает pointermove: показываем по pointerdown; после тапа pointerleave тултип не закрывает
              onPointerDown={onPointerMove}
              onPointerLeave={(e) => e.pointerType === 'mouse' && clear()}
            />
          </svg>
        )}
      </div>
      <div className="sr-only">
        <table>
          <caption>{ariaLabel ?? `${yLabel} от ${xLabel}`}</caption>
          <thead>
            <tr>
              <th scope="col">Серия</th>
              <th scope="col">{xLabel}</th>
              <th scope="col">{yLabel}</th>
            </tr>
          </thead>
          <tbody>
            {series.flatMap((s) =>
              s.points.map(([px, py], k) => (
                <tr key={`${s.id}-${k}`}>
                  <td>{s.label}</td>
                  <td>{xFormat(px)}</td>
                  <td>{yFormat(py)}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      {tip.node}
    </div>
  )
}
