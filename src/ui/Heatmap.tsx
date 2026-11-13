import { useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import { cn } from './cn'
import { divColor, divInk, legendGradient, seqColor, seqInk } from './color'
import { TipRow, useTip } from './floating'
import { fmt } from './fmt'
import { measureText, truncateText, useElementSize, useFontsReady, useScrollEdges } from './hooks'
import { ScrollHint } from './ScrollX'

export interface HeatmapCell {
  row: number
  col: number
  value: number
}

export interface HeatmapProps {
  data: number[][]
  rowLabels: string[]
  colLabels: string[]
  /** Формат значения в тултипе, легенде и ячейках. */
  valueFormat?: (value: number) => string
  onHover?: (cell: HeatmapCell | null) => void
  onClick?: (cell: HeatmapCell) => void
  /** Диапазон шкалы (по умолчанию min..max; для весов attention передайте [0, 1]). */
  domain?: [number, number]
  /** 'seq' (по умолчанию): одна синяя гамма; 'diverging': знак, ноль серый. */
  scale?: 'seq' | 'diverging'
  highlight?: { rows?: number[]; cols?: number[]; cells?: Array<[number, number]> }
  maxCellSize?: number
  minCellSize?: number
  /** Числа в ячейках: true, false или 'auto' (если ячейка ≥ 30px). */
  showValues?: boolean | 'auto'
  legend?: boolean
  legendLabel?: string
  /** Подпись строк (например, «кто смотрит»). */
  rowAxisLabel?: string
  /** Подпись столбцов (например, «на кого смотрит»). */
  colAxisLabel?: string
  /** Свой тултип ячейки. */
  tooltip?: (cell: HeatmapCell, rowLabel: string, colLabel: string) => ReactNode
  /** Какие ячейки замаскированы (по умолчанию значения −∞): рисуются штриховкой. */
  isMasked?: (row: number, col: number, value: number) => boolean
  /** Шрифт подписей: mono для токенов (по умолчанию), sans для слов. */
  labelFont?: 'mono' | 'sans'
  'aria-label'?: string
  className?: string
}

const LABEL_FONT = { mono: '11.5px "JetBrains Mono Variable", monospace', sans: '12px "Onest Variable", sans-serif' }

/**
 * Тепловая карта на SVG: подписи строк и столбцов (длинные столбцы вертикально),
 * последовательная шкала DESIGN §3, легенда, тултип при наведении и с клавиатуры (стрелки).
 * Для скринридеров рядом лежит скрытая таблица значений.
 *
 *   <Heatmap data={weights} rowLabels={toks} colLabels={toks} domain={[0, 1]}
 *     rowAxisLabel="кто смотрит" colAxisLabel="на кого смотрит" valueFormat={(v) => fmt.pct(v)} />
 */
export function Heatmap({
  data,
  rowLabels,
  colLabels,
  valueFormat = (v) => fmt.num(v, 2),
  onHover,
  onClick,
  domain,
  scale = 'seq',
  highlight,
  maxCellSize = 40,
  minCellSize = 14,
  showValues = 'auto',
  legend = true,
  legendLabel,
  rowAxisLabel,
  colAxisLabel,
  tooltip,
  isMasked = (_r, _c, v) => v === -Infinity,
  labelFont = 'mono',
  'aria-label': ariaLabel,
  className,
}: HeatmapProps) {
  const uid = useId().replace(/:/g, '')
  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  const [scrollRef, edges] = useScrollEdges<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement | null>(null)
  const fontsReady = useFontsReady()
  const [hover, setHover] = useState<[number, number] | null>(null)
  const tip = useTip({
    scopeEl: svgRef.current,
    onOutside: () => {
      setHover(null)
      onHover?.(null)
    },
  })
  const rows = data.length
  const cols = rows ? Math.max(...data.map((r) => r.length)) : 0
  const font = LABEL_FONT[labelFont]

  const { lo, hi, absMax } = useMemo(() => {
    let mn = Infinity
    let mx = -Infinity
    for (const r of data) for (const v of r) if (Number.isFinite(v)) (mn = Math.min(mn, v)), (mx = Math.max(mx, v))
    if (!Number.isFinite(mn)) (mn = 0), (mx = 1)
    const [l, h] = domain ?? [mn, mx]
    return { lo: l, hi: h, absMax: Math.max(Math.abs(l), Math.abs(h)) || 1 }
  }, [data, domain])

  const layout = useMemo(() => {
    const W = Math.max(width, 0)
    const rowLabelMax = Math.max(0, ...rowLabels.map((l) => measureText(l, font)))
    const rowLabelW = Math.ceil(Math.min(rowLabelMax + 10, Math.max(60, W * 0.32), 150))
    const avail = W - rowLabelW - 4
    const cs = Math.max(minCellSize, Math.min(maxCellSize, Math.floor(avail / Math.max(1, cols))))
    const colLabelMax = Math.max(0, ...colLabels.map((l) => measureText(l, font)))
    const vertical = colLabelMax > cs - 4
    const colLabelH = vertical ? Math.min(colLabelMax, 110) + 8 : 20
    const axisH = colAxisLabel ? 20 : 0
    const top = axisH + colLabelH
    return { rowLabelW, cs, vertical, colLabelH, axisH, top, svgW: rowLabelW + cs * cols + 2, svgH: top + cs * rows + 2 }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, rowLabels, colLabels, rows, cols, minCellSize, maxCellSize, colAxisLabel, font, fontsReady])

  const hl = useMemo(() => {
    const r = new Set(highlight?.rows ?? [])
    const c = new Set(highlight?.cols ?? [])
    const cells = new Set((highlight?.cells ?? []).map(([a, b]) => `${a},${b}`))
    return { r, c, cells, any: r.size + c.size + cells.size > 0 }
  }, [highlight])

  const tOf = (v: number) => (scale === 'seq' ? (hi > lo ? (v - lo) / (hi - lo) : 0) : v / absMax)
  const fill = (v: number) => (scale === 'seq' ? seqColor(tOf(v)) : divColor(tOf(v)))
  const ink = (v: number) => (scale === 'seq' ? seqInk(tOf(v)) : divInk(tOf(v)))
  const valuesOn = showValues === true || (showValues === 'auto' && layout.cs >= 30)

  const { rowLabelW, cs, top } = layout

  const showCell = (r: number, c: number) => {
    setHover([r, c])
    const v = data[r]?.[c]
    if (v === undefined) return
    const cell = { row: r, col: c, value: v }
    onHover?.(cell)
    const svg = svgRef.current
    if (!svg) return
    const box = svg.getBoundingClientRect()
    const anchor = { left: box.left + rowLabelW + c * cs, top: box.top + top + r * cs, width: cs, height: cs }
    const masked = isMasked(r, c, v)
    tip.show(
      anchor,
      tooltip ? (
        tooltip(cell, rowLabels[r] ?? '', colLabels[c] ?? '')
      ) : (
        <div className="grid gap-1">
          <TipRow value={masked ? 'замаскировано' : valueFormat(v)} />
          <div className="text-xs text-ink-2">
            <span className="font-mono">{rowLabels[r]}</span>
            <span className="text-ink-3"> → </span>
            <span className="font-mono">{colLabels[c]}</span>
          </div>
        </div>
      ),
    )
  }
  const clear = () => {
    setHover(null)
    onHover?.(null)
    tip.hide()
  }

  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - box.left - rowLabelW
    const y = e.clientY - box.top - top
    const c = Math.floor(x / cs)
    const r = Math.floor(y / cs)
    if (r < 0 || c < 0 || r >= rows || c >= cols) {
      if (hover) clear()
      return
    }
    if (!hover || hover[0] !== r || hover[1] !== c) showCell(r, c)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const [r, c] = hover ?? [0, 0]
    let nr = r
    let nc = c
    if (e.key === 'ArrowRight') nc = Math.min(cols - 1, c + 1)
    else if (e.key === 'ArrowLeft') nc = Math.max(0, c - 1)
    else if (e.key === 'ArrowDown') nr = Math.min(rows - 1, r + 1)
    else if (e.key === 'ArrowUp') nr = Math.max(0, r - 1)
    else if (e.key === 'Escape') return clear()
    else if ((e.key === 'Enter' || e.key === ' ') && onClick && hover) {
      e.preventDefault()
      onClick({ row: r, col: c, value: data[r][c] })
      return
    } else return
    e.preventDefault()
    showCell(nr, nc)
  }

  return (
    <div className={cn('min-w-0', className)}>
      <div ref={wrapRef} className="w-full">
        {width > 0 && (
          <div
            ref={scrollRef}
            className="scroll-fade max-w-full overflow-x-auto scrollbar-thin focus-visible:outline-offset-4"
            tabIndex={0}
            role="group"
            aria-label={ariaLabel ?? 'Тепловая карта: стрелки перемещают по ячейкам'}
            onKeyDown={onKeyDown}
            onFocus={(e) => e.currentTarget.matches(':focus-visible') && !hover && showCell(0, 0)}
            onBlur={clear}
          >
            <svg
              ref={svgRef}
              width={layout.svgW}
              height={layout.svgH}
              role="img"
              aria-hidden
              className="block select-none"
              onPointerMove={onPointerMove}
              // тап не порождает pointermove: показываем по pointerdown; после тапа pointerleave тултип не закрывает
              onPointerDown={onPointerMove}
              onPointerLeave={(e) => e.pointerType === 'mouse' && clear()}
              onClick={() => hover && onClick?.({ row: hover[0], col: hover[1], value: data[hover[0]][hover[1]] })}
              style={{ cursor: onClick ? 'pointer' : undefined }}
            >
              <defs>
                <pattern id={`hatch-${uid}`} patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
                  <rect width="6" height="6" style={{ fill: 'var(--surface-2)' }} />
                  <line x1="0" y1="0" x2="0" y2="6" style={{ stroke: 'var(--ink-3)', strokeOpacity: 0.35, strokeWidth: 1.5 }} />
                </pattern>
              </defs>

              {colAxisLabel && (
                <text x={rowLabelW} y={13} className="text-axis" style={{ fill: 'var(--ink-3)', fontSize: 12 }}>
                  {colAxisLabel} →
                </text>
              )}
              {rowAxisLabel && (
                <text x={rowLabelW - 8} y={top - 6} textAnchor="end" style={{ fill: 'var(--ink-3)', fontSize: 12 }}>
                  {rowAxisLabel} ↓
                </text>
              )}

              {colLabels.map((l, c) => {
                const x = rowLabelW + c * cs + cs / 2
                const active = hl.c.has(c) || hover?.[1] === c
                const style = { fill: active ? 'var(--ink)' : 'var(--ink-2)', font, fontWeight: active ? 600 : 400 }
                return layout.vertical ? (
                  <text key={c} transform={`translate(${x + 4}, ${top - 6}) rotate(-90)`} style={style}>
                    {truncateText(l, 110, font)}
                  </text>
                ) : (
                  <text key={c} x={x} y={top - 6} textAnchor="middle" style={style}>
                    {l}
                  </text>
                )
              })}

              {rowLabels.map((l, r) => {
                const active = hl.r.has(r) || hover?.[0] === r
                return (
                  <text
                    key={r}
                    x={rowLabelW - 8}
                    y={top + r * cs + cs / 2}
                    dominantBaseline="central"
                    textAnchor="end"
                    style={{ fill: active ? 'var(--ink)' : 'var(--ink-2)', font, fontWeight: active ? 600 : 400 }}
                  >
                    {truncateText(l, rowLabelW - 12, font)}
                  </text>
                )
              })}

              {data.map((row, r) =>
                row.map((v, c) => {
                  const masked = isMasked(r, c, v)
                  const dim = hl.any && !(hl.r.has(r) || hl.c.has(c) || hl.cells.has(`${r},${c}`))
                  const x = rowLabelW + c * cs + 1
                  const y = top + r * cs + 1
                  const s = cs - 2
                  const base = masked ? `url(#hatch-${uid})` : Number.isFinite(v) ? fill(v) : 'var(--surface-2)'
                  // приглушение без opacity у текста: заливка уходит к поверхности, число остаётся читаемым (--ink-2)
                  return (
                    <g key={`${r}-${c}`} opacity={dim && masked ? 0.4 : 1}>
                      <rect
                        x={x}
                        y={y}
                        width={s}
                        height={s}
                        rx={Math.min(3, s / 6)}
                        style={{ fill: dim && !masked ? `color-mix(in oklab, ${base} 20%, var(--surface))` : base }}
                      />
                      {valuesOn && !masked && Number.isFinite(v) && (
                        <text
                          x={x + s / 2}
                          y={y + s / 2}
                          textAnchor="middle"
                          dominantBaseline="central"
                          style={{ fill: dim ? 'var(--ink-2)' : ink(v), font: '10.5px "JetBrains Mono Variable", monospace' }}
                        >
                          {valueFormat(v)}
                        </text>
                      )}
                    </g>
                  )
                }),
              )}

              {(highlight?.cells ?? []).map(([r, c]) => (
                <rect
                  key={`h${r}-${c}`}
                  x={rowLabelW + c * cs}
                  y={top + r * cs}
                  width={cs}
                  height={cs}
                  rx={Math.min(4, cs / 5)}
                  style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 2 }}
                />
              ))}

              {hover && (
                <rect
                  x={rowLabelW + hover[1] * cs + 0.5}
                  y={top + hover[0] * cs + 0.5}
                  width={cs - 1}
                  height={cs - 1}
                  rx={Math.min(3, cs / 6)}
                  style={{ fill: 'none', stroke: 'var(--ink)', strokeWidth: 1.5 }}
                  pointerEvents="none"
                />
              )}
            </svg>
          </div>
        )}
        {width > 0 && edges !== 'none' && <ScrollHint />}
      </div>

      {legend && (
        <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-3">
          {legendLabel && <span className="text-ink-2">{legendLabel}</span>}
          <span className="tabular-nums">{valueFormat(scale === 'seq' ? lo : -absMax)}</span>
          <span aria-hidden className="h-2 w-28 rounded-full" style={{ background: legendGradient(scale === 'seq' ? 'seq' : 'diverging') }} />
          <span className="tabular-nums">{valueFormat(scale === 'seq' ? hi : absMax)}</span>
          {data.some((r, ri) => r.some((v, ci) => isMasked(ri, ci, v))) && (
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden className="h-3 w-3 rounded-[3px] bg-surface-2 bg-hatch" />
              замаскировано
            </span>
          )}
        </div>
      )}

      {rows * cols <= 1024 && (
        <div className="sr-only">
          <table>
            {ariaLabel && <caption>{ariaLabel}</caption>}
            <thead>
              <tr>
                <th scope="col">{rowAxisLabel ?? ''}</th>
                {colLabels.map((l, c) => (
                  <th key={c} scope="col">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, r) => (
                <tr key={r}>
                  <th scope="row">{rowLabels[r]}</th>
                  {row.map((v, c) => (
                    <td key={c}>{isMasked(r, c, v) ? 'замаскировано' : valueFormat(v)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {tip.node}
    </div>
  )
}
