import { memo } from 'react'
import { divColor } from '@/ui'
import { cn } from '@/ui/cn'

/**
 * «Штрих-код» вектора: каждая координата это узкая полоска, цвет по расходящейся шкале
 * (минус красный, ноль серый, плюс синий). Полоски растягиваются на ширину контейнера.
 */
export const Barcode = memo(function Barcode({
  values,
  height = 20,
  range = 3,
  label,
  className,
  rounded = true,
}: {
  values: readonly number[]
  /** Высота, px. */
  height?: number
  /** |значение|, при котором цвет насыщается. */
  range?: number
  /** Подпись для скринридеров. */
  label: string
  className?: string
  rounded?: boolean
}) {
  const n = Math.max(1, values.length)
  return (
    <svg
      role="img"
      aria-label={label}
      viewBox={`0 0 ${n} 1`}
      preserveAspectRatio="none"
      className={cn('block w-full', rounded && 'rounded-[4px]', className)}
      style={{ height }}
      shapeRendering="crispEdges"
    >
      {values.map((v, i) => (
        <rect key={i} x={i} y={0} width={1.02} height={1} style={{ fill: divColor(v / range) }} />
      ))}
    </svg>
  )
})

/** Легенда шкалы штрих-кода. */
export function BarcodeLegend({ className }: { className?: string }) {
  return (
    <div className={cn('flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-3', className)}>
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="h-2.5 w-2.5 rounded-[2px] shadow-[inset_0_0_0_1px_var(--line-strong)]" style={{ background: divColor(-0.85) }} />
        минус
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="h-2.5 w-2.5 rounded-[2px] shadow-[inset_0_0_0_1px_var(--line-strong)]" style={{ background: divColor(0) }} />
        около нуля
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="h-2.5 w-2.5 rounded-[2px] shadow-[inset_0_0_0_1px_var(--line-strong)]" style={{ background: divColor(0.85) }} />
        плюс
      </span>
    </div>
  )
}
