import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { ProhibitIcon } from '@phosphor-icons/react'
import { cn } from './cn'
import { resolveColor } from './color'
import { barPath, fontCss, makeScale } from './chart-utils'
import { TipRow, useTip } from './floating'
import { fmt } from './fmt'
import { measureText, truncateText, useElementSize, useFontsReady } from './hooks'

export interface BarDatum {
  label: string
  value: number
  /** Цвет: роль ('prefill', 'decode', …) или CSS-цвет. По умолчанию цвет графика. */
  color?: string
  /** Приглушённый столбик (отсечён top-k / top-p). */
  muted?: boolean
  /** Зачёркнутый (замаскирован грамматикой): иконка и зачёркнутая подпись. */
  struck?: boolean
  /** Дополнительная строка в тултипе. */
  note?: ReactNode
  /** Стабильный ключ (для анимации при пересортировке). */
  id?: string
}

export interface BarChartProps {
  data: BarDatum[]
  valueFormat?: (value: number) => string
  /** Максимум шкалы (например 1 для вероятностей). */
  max?: number
  /** Минимум шкалы (по умолчанию min(0, данные)). */
  min?: number
  /** Горизонтальные столбики (по умолчанию) или вертикальные. */
  horizontal?: boolean
  /** Цвет по умолчанию: роль или CSS-цвет (по умолчанию 'prefill', первый цвет категориальной палитры). */
  color?: string
  /** Высота области построения для вертикального варианта. */
  height?: number
  /** Толщина столбика (не больше 24px). */
  barSize?: number
  highlightIndex?: number | null
  onHover?: (index: number | null) => void
  onClick?: (index: number) => void
  /** Моноширинные подписи (для токенов). */
  labelFont?: 'sans' | 'mono'
  showValues?: boolean
  'aria-label'?: string
  className?: string
}

const FONTS = {
  sans: '13px "Onest Variable", sans-serif',
  mono: '12.5px "JetBrains Mono Variable", monospace',
}
const VALUE_FONT = '600 13px "Onest Variable", sans-serif'
const VALUE_CSS = fontCss(VALUE_FONT)
const VALUE_CSS_V = fontCss('600 12px "Onest Variable", sans-serif')

/**
 * Столбчатая диаграмма (SVG). Тонкие столбики со скруглением на конце данных, значения у конца,
 * подписи цветом текста, тултип на наведении и с клавиатуры (стрелки).
 * muted: приглушён (отсечён сэмплером); struck: зачёркнут с иконкой (маска structured output).
 *
 *   <BarChart data={cands.map((c) => ({ label: c.text, value: c.p, muted: !c.kept }))}
 *     max={1} valueFormat={(v) => fmt.pct(v)} labelFont="mono" />
 */
export function BarChart({
  data,
  valueFormat = (v) => fmt.num(v),
  max,
  min,
  horizontal = true,
  color = 'prefill',
  height = 200,
  barSize,
  highlightIndex,
  onHover,
  onClick,
  labelFont = 'sans',
  showValues = true,
  'aria-label': ariaLabel,
  className,
}: BarChartProps) {
  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement | null>(null)
  const reduce = useReducedMotion()
  useFontsReady() // перерисовать, когда шрифты загрузятся (точные ширины подписей)
  const [hover, setHover] = useState<number | null>(null)
  const tip = useTip({
    scopeEl: svgRef.current,
    onOutside: () => {
      setHover(null)
      onHover?.(null)
    },
  })
  const font = FONTS[labelFont]
  const transition = { duration: reduce ? 0 : 0.35, ease: [0.2, 0, 0, 1] as const }

  const [d0, d1] = useMemo(() => {
    const vals = data.map((d) => d.value).filter(Number.isFinite)
    const lo = min ?? Math.min(0, ...vals)
    const hi = max ?? Math.max(0, ...vals)
    return [lo, hi > lo ? hi : lo + 1]
  }, [data, min, max])

  const active = hover ?? highlightIndex ?? null

  const show = (i: number, anchor: DOMRect | { left: number; top: number; width: number; height: number }) => {
    setHover(i)
    onHover?.(i)
    const d = data[i]
    if (!d) return
    tip.show(
      anchor,
      <div className="grid gap-1">
        <TipRow color={resolveColor(d.color ?? color)} value={valueFormat(d.value)} label={labelFont === 'mono' ? <span className="font-mono">{d.label}</span> : d.label} />
        {d.struck && <div className="text-xs text-ink-2">Замаскирован: этот токен запрещён</div>}
        {d.muted && !d.struck && <div className="text-xs text-ink-2">Отсечён</div>}
        {d.note && <div className="text-xs text-ink-2">{d.note}</div>}
      </div>,
      horizontal ? 'top' : 'top',
    )
  }
  const clear = () => {
    setHover(null)
    onHover?.(null)
    tip.hide()
  }
  // тач: после тапа приходит pointerleave, но тултип должен остаться до тапа в другом месте
  const leave = (e: PointerEvent<SVGSVGElement>) => e.pointerType === 'mouse' && clear()

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const fwd = horizontal ? 'ArrowDown' : 'ArrowRight'
    const back = horizontal ? 'ArrowUp' : 'ArrowLeft'
    let i = hover ?? -1
    if (e.key === fwd) i = Math.min(data.length - 1, i + 1)
    else if (e.key === back) i = Math.max(0, i - 1)
    else if (e.key === 'Escape') return clear()
    else if ((e.key === 'Enter' || e.key === ' ') && onClick && hover !== null) {
      e.preventDefault()
      onClick(hover)
      return
    } else return
    e.preventDefault()
    const el = svgRef.current?.querySelector<SVGElement>(`[data-bar="${i}"]`)
    if (el) show(i, el.getBoundingClientRect())
  }

  const content = width > 0 && (horizontal ? renderHorizontal() : renderVertical())

  function fillFor(d: BarDatum) {
    const c = resolveColor(d.color ?? color)
    return d.muted || d.struck ? `color-mix(in oklab, ${c} 32%, var(--surface))` : c
  }

  function renderHorizontal() {
    const thick = Math.min(24, barSize ?? 20)
    const rowH = Math.max(thick + 10, 30)
    const labelMax = Math.max(0, ...data.map((d) => measureText(d.label, font) + (d.struck ? 20 : 0)))
    const labelW = Math.ceil(Math.min(labelMax + 2, width * 0.42, 240))
    const valueW = showValues ? Math.ceil(Math.max(0, ...data.map((d) => measureText(valueFormat(d.value), VALUE_FONT))) + 10) : 6
    const negW = showValues ? Math.ceil(Math.max(0, ...data.filter((d) => d.value < 0).map((d) => measureText(valueFormat(d.value), VALUE_FONT) + 10))) : 0
    const x = makeScale('linear', [d0, d1], [labelW + 10 + negW, Math.max(labelW + 40 + negW, width - valueW)])
    const x0 = x(Math.max(d0, Math.min(0, d1)))
    const h = rowH * data.length
    return (
      <svg ref={svgRef} width={width} height={h} role="img" aria-hidden className="block select-none" onPointerLeave={leave}>
        {data.map((d, i) => {
          const y = i * rowH
          const xv = x(d.value)
          const neg = xv < x0
          const bx = neg ? xv : x0
          const bw = Math.abs(xv - x0)
          const labelText = truncateText(d.label, labelW - (d.struck ? 20 : 0), font)
          const tw = measureText(labelText, font)
          const dim = active !== null && active !== i
          return (
            // приглушаем только столбик: подпись и значение остаются читаемыми (AA), иначе выделение строки роняло их до 2,4:1
            <g key={d.id ?? `${i}-${d.label}`}>
              {active === i && <rect x={0} y={y + 1} width={width} height={rowH - 2} rx={6} style={{ fill: 'var(--surface-3)', opacity: 0.55 }} />}
              <text
                x={labelW}
                y={y + rowH / 2}
                dominantBaseline="central"
                textAnchor="end"
                textDecoration={d.struck ? 'line-through' : undefined}
                style={{ font, fill: d.muted || d.struck ? 'var(--ink-3)' : 'var(--ink-2)' }}
              >
                {labelText}
              </text>
              {d.struck && <ProhibitIcon x={labelW - tw - 18} y={y + rowH / 2 - 7} size={14} style={{ color: 'var(--bad)' }} aria-hidden />}
              <motion.path
                data-bar={i}
                initial={false}
                animate={{ d: barPath(bx, y + (rowH - thick) / 2, bw, thick, neg ? 'left' : 'right') }}
                transition={transition}
                style={{ fill: fillFor(d), opacity: dim ? 0.4 : 1, transition: 'opacity 150ms' }}
              />
              {showValues && (
                <motion.text
                  initial={false}
                  animate={{ x: neg ? xv - 6 : xv + 6 }}
                  transition={transition}
                  y={y + rowH / 2}
                  dominantBaseline="central"
                  textAnchor={neg ? 'end' : 'start'}
                  textDecoration={d.struck ? 'line-through' : undefined}
                  style={{ ...VALUE_CSS, fill: d.muted || d.struck ? 'var(--ink-3)' : 'var(--ink)', fontVariantNumeric: 'tabular-nums' }}
                >
                  {valueFormat(d.value)}
                </motion.text>
              )}
              <rect
                x={0}
                y={y}
                width={width}
                height={rowH}
                fill="transparent"
                onPointerEnter={(e) => show(i, (e.currentTarget.parentNode as SVGGElement).querySelector('[data-bar]')!.getBoundingClientRect())}
                onClick={onClick ? () => onClick(i) : undefined}
                style={{ cursor: onClick ? 'pointer' : undefined }}
              />
            </g>
          )
        })}
        <line x1={x0} x2={x0} y1={0} y2={h} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
      </svg>
    )
  }

  function renderVertical() {
    const top = showValues ? 22 : 8
    const hasNeg = data.some((d) => d.value < 0)
    const labelGap = showValues && hasNeg ? 34 : 17
    const bottom = labelGap + 9
    const plotH = height
    const n = Math.max(1, data.length)
    const band = width / n
    const thick = Math.min(24, barSize ?? 24, band * 0.7)
    const y = makeScale('linear', [d0, d1], [top + plotH, top])
    const y0 = y(Math.max(d0, Math.min(0, d1)))
    const h = top + plotH + bottom
    const vFont = labelFont === 'mono' ? '11.5px "JetBrains Mono Variable", monospace' : '12px "Onest Variable", sans-serif'
    return (
      <svg ref={svgRef} width={width} height={h} role="img" aria-hidden className="block select-none" onPointerLeave={leave}>
        {data.map((d, i) => {
          const cx = band * i + band / 2
          const yv = y(d.value)
          const neg = yv > y0
          const by = neg ? y0 : yv
          const bh = Math.abs(yv - y0)
          const dim = active !== null && active !== i
          const iconW = d.struck ? 18 : 0
          const labelText = truncateText(d.label, band - 4 - iconW, vFont)
          const tw = measureText(labelText, vFont)
          const labelX = cx + iconW / 2
          return (
            // приглушаем только столбик: подпись и значение остаются читаемыми (AA), иначе выделение строки роняло их до 2,4:1
            <g key={d.id ?? `${i}-${d.label}`}>
              {active === i && <rect x={band * i + 2} y={0} width={band - 4} height={h} rx={6} style={{ fill: 'var(--surface-3)', opacity: 0.55 }} />}
              <motion.path
                data-bar={i}
                initial={false}
                animate={{ d: barPath(cx - thick / 2, by, thick, bh, neg ? 'down' : 'up') }}
                transition={transition}
                style={{ fill: fillFor(d), opacity: dim ? 0.4 : 1, transition: 'opacity 150ms' }}
              />
              {showValues && (
                <motion.text
                  initial={false}
                  animate={{ y: neg ? yv + 14 : yv - 7 }}
                  transition={transition}
                  x={cx}
                  textAnchor="middle"
                  textDecoration={d.struck ? 'line-through' : undefined}
                  style={{ ...VALUE_CSS_V, fill: d.muted || d.struck ? 'var(--ink-3)' : 'var(--ink)', fontVariantNumeric: 'tabular-nums' }}
                >
                  {valueFormat(d.value)}
                </motion.text>
              )}
              <text
                x={labelX}
                y={top + plotH + labelGap}
                textAnchor="middle"
                textDecoration={d.struck ? 'line-through' : undefined}
                style={{ font: vFont, fill: d.muted || d.struck ? 'var(--ink-3)' : 'var(--ink-2)' }}
              >
                {labelText}
              </text>
              {/* замаскированный: иконка рядом с подписью, а не только зачёркивание (DESIGN §8) */}
              {d.struck && (
                <ProhibitIcon x={labelX - tw / 2 - 17} y={top + plotH + labelGap - 11} size={14} style={{ color: 'var(--bad)' }} aria-hidden />
              )}
              <rect
                x={band * i}
                y={0}
                width={band}
                height={h}
                fill="transparent"
                onPointerEnter={(e) => show(i, (e.currentTarget.parentNode as SVGGElement).querySelector('[data-bar]')!.getBoundingClientRect())}
                onClick={onClick ? () => onClick(i) : undefined}
                style={{ cursor: onClick ? 'pointer' : undefined }}
              />
            </g>
          )
        })}
        <line x1={0} x2={width} y1={y0} y2={y0} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
      </svg>
    )
  }

  return (
    <div className={cn('min-w-0', className)}>
      <div
        ref={wrapRef}
        tabIndex={0}
        role="group"
        aria-label={ariaLabel ?? 'Столбчатая диаграмма: стрелки перемещают по столбикам'}
        onKeyDown={onKeyDown}
        onBlur={clear}
        className="w-full rounded-control focus-visible:outline-offset-4"
      >
        {content}
      </div>
      <div className="sr-only">
        <table>
          {ariaLabel && <caption>{ariaLabel}</caption>}
          <tbody>
            {data.map((d, i) => (
              <tr key={i}>
                <th scope="row">{d.label}</th>
                <td>
                  {valueFormat(d.value)}
                  {d.struck ? ' (замаскирован)' : d.muted ? ' (отсечён)' : ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {tip.node}
    </div>
  )
}
