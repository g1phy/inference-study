import { useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react'
import { cn } from './cn'
import { cssVar, divColor, divInk, mix, seqColor, seqInk, tint } from './color'
import { useTip } from './floating'
import { fmt } from './fmt'
import { useScrollEdges } from './hooks'
import { ScrollHint } from './ScrollX'

export interface MatrixCell {
  row: number
  col: number
  value: number
}

export interface MatrixHighlight {
  rows?: number[]
  cols?: number[]
  cells?: Array<[number, number]>
}

export interface MatrixViewProps {
  data: number[][]
  rowLabels?: ReactNode[]
  colLabels?: ReactNode[]
  /** Знаков после запятой (по умолчанию 2). */
  digits?: number
  /** Заливка по значению: 'seq' (одна синяя гамма), 'diverging' (знак), false (без заливки). */
  heat?: 'seq' | 'diverging' | false
  /** Диапазон для заливки (по умолчанию min..max конечных значений). */
  domain?: [number, number]
  highlight?: MatrixHighlight
  /** Приглушать невыделенные ячейки, когда есть выделение (по умолчанию true). */
  dimOthers?: boolean
  onCellHover?: (cell: MatrixCell | null) => void
  onCellClick?: (cell: MatrixCell) => void
  /** −∞ рисуется как «−∞» на штриховке (маска causal attention). */
  maskInfinity?: boolean
  /**
   * Какие ячейки замаскированы, независимо от значения (например, нули после softmax над диагональю):
   * штриховка вместо заливки, текст maskedText, не приглушаются. Маска не теряется на шаге, где −∞ стали нулями.
   */
  isMasked?: (row: number, col: number) => boolean
  /** Текст в замаскированной ячейке (по умолчанию «0»). */
  maskedText?: string
  /** Подпись ячейки для скринридера (по умолчанию «строка · столбец: значение», если подписи строками). */
  cellLabel?: (cell: MatrixCell) => string
  cellSize?: 'sm' | 'md' | 'lg' | number
  /** Матрица Q, K или V: цвет сущности (подложка и интенсивность по модулю значения). */
  colorRole?: 'q' | 'k' | 'v'
  /** Подпись над матрицей: «Q = X·W_Q». */
  label?: ReactNode
  /** Своё форматирование значения ячейки. */
  format?: (value: number) => string
  /** Тултип ячейки (например, слагаемые скалярного произведения). */
  cellTooltip?: (cell: MatrixCell) => ReactNode
  'aria-label'?: string
  className?: string
}

/**
 * Приглушённая ячейка (есть выделение, а эта не выделена). Без opacity: она гасила и заливку, и текст,
 * и тёмная цифра на тёмной заливке пропадала совсем. Заливка уходит к поверхности, число остаётся
 * читаемым цветом --ink-2 на 20 % заливки (у --ink-3 на 30 % насыщенной заливки было ниже AA, 3,7:1).
 * Ячейки без своей заливки (−∞ на штриховке) приглушаются классом opacity.
 */
function dimStyle(style: CSSProperties): CSSProperties {
  const bg = typeof style.background === 'string' ? style.background : undefined
  return bg ? { ...style, background: `color-mix(in oklab, ${bg} 20%, var(--surface))`, color: 'var(--ink-2)' } : style
}

const SIZES = {
  sm: { w: 44, h: 28, text: 'text-[12px]' },
  md: { w: 56, h: 32, text: 'text-[13px]' },
  lg: { w: 68, h: 40, text: 'text-[14px]' },
} as const

/**
 * Матрица чисел (HTML-сетка): подписи строк и столбцов, подсветка строк/столбцов/ячеек,
 * заливка по значению, −∞ на штриховке. Широкие матрицы прокручиваются внутри.
 * С onCellClick/onCellHover ячейки доступны с клавиатуры (стрелки, Enter).
 *
 *   <MatrixView data={scores} rowLabels={words} colLabels={words} heat="seq" domain={[0, 1]}
 *     highlight={{ rows: [8] }} onCellClick={setCell} />
 */
export function MatrixView({
  data,
  rowLabels,
  colLabels,
  digits = 2,
  heat = false,
  domain,
  highlight,
  dimOthers = true,
  onCellHover,
  onCellClick,
  maskInfinity = true,
  isMasked,
  maskedText = '0',
  cellLabel,
  cellSize = 'md',
  colorRole,
  label,
  format,
  cellTooltip,
  'aria-label': ariaLabel,
  className,
}: MatrixViewProps) {
  const rows = data.length
  const cols = rows ? Math.max(...data.map((r) => r.length)) : 0
  const size =
    typeof cellSize === 'number'
      ? { w: cellSize, h: Math.max(24, Math.round(cellSize * 0.6)), text: cellSize < 48 ? 'text-[12px]' : 'text-[13px]' }
      : SIZES[cellSize]
  const interactive = Boolean(onCellClick || onCellHover || cellTooltip)
  const gridRef = useRef<HTMLDivElement | null>(null)
  const [scrollRef, edges] = useScrollEdges<HTMLDivElement>()
  const tip = useTip({ scopeEl: gridRef.current, onOutside: () => onCellHover?.(null) })
  const [active, setActive] = useState<[number, number]>([0, 0])
  const fmtValue = format ?? ((v: number) => fmt.num(v, digits))

  const { lo, hi, absMax } = useMemo(() => {
    let mn = Infinity
    let mx = -Infinity
    for (const r of data) for (const v of r) if (Number.isFinite(v)) (mn = Math.min(mn, v)), (mx = Math.max(mx, v))
    if (!Number.isFinite(mn)) (mn = 0), (mx = 1)
    const [l, h] = domain ?? [mn, mx]
    return { lo: l, hi: h, absMax: Math.max(Math.abs(l), Math.abs(h)) || 1 }
  }, [data, domain])

  const hl = useMemo(() => {
    const r = new Set(highlight?.rows ?? [])
    const c = new Set(highlight?.cols ?? [])
    const cells = new Set((highlight?.cells ?? []).map(([a, b]) => `${a},${b}`))
    return { r, c, cells, any: r.size + c.size + cells.size > 0 }
  }, [highlight])

  const hasRowLabels = Boolean(rowLabels?.length)
  const hasColLabels = Boolean(colLabels?.length)
  const rowOffset = hasColLabels ? 2 : 1
  const colOffset = hasRowLabels ? 2 : 1
  const verticalCols = useMemo(
    () => (colLabels ?? []).some((l) => typeof l === 'string' && l.length * 7.4 > size.w - 6),
    [colLabels, size.w],
  )

  const cellLook = (v: number): { style: CSSProperties; className?: string; text: string } => {
    if (!Number.isFinite(v)) {
      if (maskInfinity) {
        return { style: { color: 'var(--ink-3)' }, className: 'bg-surface-2 bg-hatch', text: v > 0 ? '∞' : '−∞' }
      }
      return { style: { background: 'var(--surface-2)' }, text: fmt.num(v) }
    }
    const text = fmtValue(v)
    if (colorRole) {
      if (heat) return { style: { background: mix(colorRole, 6 + 50 * Math.min(1, Math.abs(v) / absMax)), color: 'var(--ink)' }, text }
      return { style: { background: tint(colorRole, 13), color: 'var(--ink)' }, text }
    }
    if (heat === 'seq') {
      const t = hi > lo ? (v - lo) / (hi - lo) : 0
      return { style: { background: seqColor(t), color: seqInk(t) }, text }
    }
    if (heat === 'diverging') {
      const t = v / absMax
      return { style: { background: divColor(t), color: divInk(t) }, text }
    }
    return { style: { background: 'var(--surface-2)', color: 'var(--ink)' }, text }
  }

  const cellAt = (r: number, c: number): MatrixCell => ({ row: r, col: c, value: data[r]?.[c] ?? NaN })

  const maskedLook = { style: { color: 'var(--ink-3)' } as CSSProperties, className: 'bg-surface-2 bg-hatch', text: maskedText }

  /** Подпись ячейки для скринридера: «она · забор: 2 %» вместо одного «2 %». */
  const ariaFor = (r: number, c: number, text: string): string => {
    if (cellLabel) return cellLabel(cellAt(r, c))
    const rl = rowLabels?.[r]
    const cl = colLabels?.[c]
    const parts = [typeof rl === 'string' || typeof rl === 'number' ? String(rl) : null, typeof cl === 'string' || typeof cl === 'number' ? String(cl) : null].filter(Boolean)
    return parts.length ? `${parts.join(' · ')}: ${text}` : text
  }

  const enterCell = (el: HTMLElement, r: number, c: number) => {
    const cell = cellAt(r, c)
    onCellHover?.(cell)
    if (cellTooltip) tip.show(el.getBoundingClientRect(), cellTooltip(cell))
  }
  const leaveGrid = () => {
    onCellHover?.(null)
    tip.hide()
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    // клавиши обрабатываем только на ячейках: кнопки в подписях строк и столбцов получают свои Enter/пробел
    if (!(e.target as HTMLElement).closest?.('[data-cell]')) return
    const [r, c] = active
    let nr = r
    let nc = c
    if (e.key === 'ArrowRight') nc = Math.min(cols - 1, c + 1)
    else if (e.key === 'ArrowLeft') nc = Math.max(0, c - 1)
    else if (e.key === 'ArrowDown') nr = Math.min(rows - 1, r + 1)
    else if (e.key === 'ArrowUp') nr = Math.max(0, r - 1)
    else if (e.key === 'Home') nc = 0
    else if (e.key === 'End') nc = cols - 1
    else if ((e.key === 'Enter' || e.key === ' ') && onCellClick) {
      e.preventDefault()
      onCellClick(cellAt(r, c))
      return
    } else if (e.key === 'Escape') {
      leaveGrid()
      return
    } else return
    e.preventDefault()
    setActive([nr, nc])
    const el = gridRef.current?.querySelector<HTMLElement>(`[data-cell="${nr},${nc}"]`)
    if (el) {
      el.focus()
      enterCell(el, nr, nc)
    }
  }

  return (
    <div className={cn('min-w-0', className)}>
      {label && (
        <div className="mb-2 flex items-center gap-2 text-sm font-medium text-ink-2">
          {colorRole && <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: cssVar(colorRole) }} />}
          <span className="min-w-0">{label}</span>
        </div>
      )}
      <div ref={scrollRef} className="scroll-fade max-w-full overflow-x-auto p-[3px] scrollbar-thin">
        <div
          ref={gridRef}
          role={interactive ? 'grid' : 'table'}
          aria-label={ariaLabel}
          aria-rowcount={rows}
          aria-colcount={cols}
          onKeyDown={interactive ? onKeyDown : undefined}
          onPointerLeave={interactive ? (e) => e.pointerType === 'mouse' && leaveGrid() : undefined}
          className="relative inline-grid gap-[2px] align-top"
          style={{
            gridTemplateColumns: `${hasRowLabels ? 'auto ' : ''}repeat(${cols}, ${size.w}px)`,
            gridTemplateRows: `${hasColLabels ? 'auto ' : ''}repeat(${rows}, ${size.h}px)`,
          }}
        >
          {hasColLabels && (
            <div role="row" className="contents">
              {hasRowLabels && <div role="presentation" style={{ gridRow: 1, gridColumn: 1 }} />}
              {Array.from({ length: cols }, (_, c) => (
                <div
                  key={c}
                  role="columnheader"
                  style={{ gridRow: 1, gridColumn: c + colOffset }}
                  className={cn(
                    'flex min-w-0 justify-center self-end pb-1 text-[12px] leading-tight whitespace-nowrap text-ink-2',
                    hl.c.has(c) && 'font-semibold text-ink',
                  )}
                >
                  <span className={cn(verticalCols ? 'max-h-[7.5rem] truncate [writing-mode:vertical-rl] rotate-180' : 'truncate')}>
                    {colLabels?.[c]}
                  </span>
                </div>
              ))}
            </div>
          )}
          {data.map((row, r) => (
            <div role="row" key={r} className="contents">
              {hasRowLabels && (
                <div
                  role="rowheader"
                  style={{ gridRow: r + rowOffset, gridColumn: 1 }}
                  className={cn(
                    'flex items-center justify-end pr-2 text-[12px] leading-tight whitespace-nowrap text-ink-2',
                    hl.r.has(r) && 'font-semibold text-ink',
                  )}
                >
                  {rowLabels?.[r]}
                </div>
              )}
              {Array.from({ length: cols }, (_, c) => {
                const v = row[c]
                const masked = isMasked?.(r, c) ?? false
                const look = masked ? maskedLook : cellLook(v)
                const isHl = hl.r.has(r) || hl.c.has(c) || hl.cells.has(`${r},${c}`)
                const dim = hl.any && dimOthers && !isHl && !masked
                const isActive = active[0] === r && active[1] === c
                return (
                  <div
                    key={c}
                    role={interactive ? 'gridcell' : 'cell'}
                    data-cell={`${r},${c}`}
                    tabIndex={interactive ? (isActive ? 0 : -1) : undefined}
                    aria-label={interactive ? ariaFor(r, c, masked ? `${look.text}, замаскировано` : look.text) : undefined}
                    onPointerEnter={interactive ? (e) => enterCell(e.currentTarget, r, c) : undefined}
                    onClick={
                      onCellClick
                        ? () => {
                            setActive([r, c])
                            onCellClick(cellAt(r, c))
                          }
                        : undefined
                    }
                    className={cn(
                      'flex items-center justify-center rounded-[4px] font-mono tabular-nums transition-[opacity,box-shadow,background-color,color] duration-150',
                      size.text,
                      look.className,
                      interactive && 'cursor-pointer focus-visible:outline-offset-1',
                      hl.cells.has(`${r},${c}`) && 'shadow-[0_0_0_2px_var(--accent)]',
                      dim && typeof look.style.background !== 'string' && 'opacity-45',
                    )}
                    style={{ ...(dim ? dimStyle(look.style) : look.style), gridRow: r + rowOffset, gridColumn: c + colOffset }}
                  >
                    {look.text}
                  </div>
                )
              })}
            </div>
          ))}
          {[...hl.r].map((r) => (
            <div
              key={`hr${r}`}
              aria-hidden
              className="pointer-events-none rounded-[5px] shadow-[0_0_0_2px_var(--accent)]"
              style={{ gridRow: r + rowOffset, gridColumn: `${colOffset} / -1` }}
            />
          ))}
          {[...hl.c].map((c) => (
            <div
              key={`hc${c}`}
              aria-hidden
              className="pointer-events-none rounded-[5px] shadow-[0_0_0_2px_var(--accent)]"
              style={{ gridRow: `${rowOffset} / -1`, gridColumn: c + colOffset }}
            />
          ))}
        </div>
      </div>
      {edges !== 'none' && <ScrollHint />}
      {tip.node}
    </div>
  )
}
