/**
 * Мелкие SVG-примитивы для таймлайнов глав 15–16: ось времени, штриховка, легенда.
 * Цвета только из токенов DESIGN.md.
 */
import { useId, type ReactNode } from 'react'
import { ScrollX, fmt, fontCss } from '@/ui'
import type { Scale } from '@/ui/chart-utils'

export const AXIS_FONT = '12px "Onest Variable", sans-serif'
export const LABEL_FONT = '13px "Onest Variable", sans-serif'
export const MONO_FONT = '600 12px "JetBrains Mono Variable", monospace'

/** id без двоеточий (useId даёт «:r1:», а в url(#…) двоеточия ломаются). */
export function useSvgId(prefix: string): string {
  return `${prefix}-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
}

/** Штриховка «простой / пусто» для SVG. */
export function HatchPattern({ id, tone = 'ink-3' }: { id: string; tone?: string }) {
  return (
    <pattern id={id} width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width={6} height={6} style={{ fill: 'var(--surface-2)' }} />
      <line x1={0} y1={0} x2={0} y2={6} style={{ stroke: `var(--${tone})`, strokeWidth: 1.6, opacity: 0.55 }} />
    </pattern>
  )
}

/** Подписи времени для оси: «0 мс», «250 мс», «1,5 с». */
export function msTick(v: number): string {
  if (v === 0) return '0'
  return fmt.ms(v)
}

/** Горизонтальная ось времени: линия, засечки и подписи под ней. */
export function TimeAxis({
  x,
  y,
  ticks,
  format = msTick,
  label,
  labelX,
}: {
  x: Scale
  y: number
  ticks: number[]
  format?: (v: number) => string
  label?: string
  labelX?: number
}) {
  const [r0, r1] = x.range
  return (
    <g aria-hidden>
      <line x1={r0} x2={r1} y1={y} y2={y} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
      {ticks.map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={y} y2={y + 4} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
          <text x={x(v)} y={y + 16} textAnchor="middle" style={{ ...fontCss(AXIS_FONT), fill: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>
            {format(v)}
          </text>
        </g>
      ))}
      {label && (
        <text x={labelX ?? r1} y={y + 32} textAnchor="end" style={{ font: AXIS_FONT, fill: 'var(--ink-3)' }}>
          {label}
        </text>
      )}
    </g>
  )
}

/** Вертикальные линии сетки по тикам. */
export function GridLines({ x, ticks, y0, y1 }: { x: Scale; ticks: number[]; y0: number; y1: number }) {
  return (
    <g aria-hidden>
      {ticks.map((v) => (
        <line key={v} x1={x(v)} x2={x(v)} y1={y0} y2={y1} style={{ stroke: 'var(--grid)', strokeWidth: 1 }} />
      ))}
    </g>
  )
}

/** Элемент легенды: метка и подпись цветом текста. */
export function LegendItem({ swatch, children }: { swatch: ReactNode; children: ReactNode }) {
  return (
    <li className="inline-flex items-center gap-2">
      <span aria-hidden className="inline-flex shrink-0 items-center">
        {swatch}
      </span>
      <span>{children}</span>
    </li>
  )
}

/** Квадратная метка цвета сущности. */
export function Swatch({ color, hatch = false, dashed = false, line = false }: { color?: string; hatch?: boolean; dashed?: boolean; line?: boolean }) {
  if (line) return <span className="inline-block w-4 border-t-2" style={{ borderColor: color ?? 'var(--ink-3)' }} />
  if (hatch) return <span className="bg-hatch inline-block h-3 w-4 rounded-[3px] shadow-[inset_0_0_0_1px_var(--line)]" />
  if (dashed) return <span className="inline-block w-4 border-t-2 border-dashed" style={{ borderColor: color ?? 'var(--ink-3)' }} />
  return <span className="inline-block h-3 w-4 rounded-[3px]" style={{ background: color }} />
}

/** Компактная пара «подпись: значение» для сводок под таймлайнами. */
export function Metric({ label, value, swatch }: { label: ReactNode; value: ReactNode; swatch?: string }) {
  return (
    <div className="flex min-w-0 items-baseline gap-1.5 text-sm">
      {swatch && <span aria-hidden className="h-2.5 w-2.5 shrink-0 translate-y-[1px] rounded-[3px]" style={{ background: swatch }} />}
      <span className="text-ink-3">{label}</span>
      <span className="font-semibold text-ink tabular-nums">{value}</span>
    </div>
  )
}

/** Компактная таблица для блоков «Глубже» и сравнений: на узком экране прокручивается, обрезанный край затухает, под ней подсказка. */
export function SpecTable({ head, rows, minWidth = 520, caption }: { head: ReactNode[]; rows: ReactNode[][]; minWidth?: number; caption?: string }) {
  return (
    <ScrollX label={caption ?? 'Таблица'}>
      <table className="w-full border-collapse text-[14.5px] leading-snug" style={{ minWidth }}>
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-line-strong text-left">
            {head.map((h, i) => (
              <th key={i} scope="col" className="py-2 pr-3 align-bottom text-[13px] font-medium text-ink-2">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line align-top last:border-b-0">
              {r.map((c, k) => (
                <td key={k} className={k === 0 ? 'py-2 pr-3 font-medium text-ink' : 'py-2 pr-3 text-ink'}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollX>
  )
}
