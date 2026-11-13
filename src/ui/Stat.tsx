import type { ReactNode } from 'react'
import { cn } from './cn'
import { fmt } from './fmt'

export interface StatProps {
  label: ReactNode
  /** Число (будет отформатировано fmt.num) или готовая строка. */
  value: ReactNode
  unit?: ReactNode
  hint?: ReactNode
  size?: 'md' | 'lg'
  /** Цветная метка сущности слева от подписи (например, 'var(--c-decode)'). */
  swatch?: string
  className?: string
}

/**
 * Большое число с подписью (dataviz stat tile): подпись обычным регистром без двоеточия,
 * значение пропорциональными цифрами, единица мельче.
 *
 *   <Stat label="TTFT" value="182" unit="мс" hint="очередь + prefill" />
 */
export function Stat({ label, value, unit, hint, size = 'md', swatch, className }: StatProps) {
  const shown = typeof value === 'number' ? fmt.num(value) : value
  return (
    <div className={cn('grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-0.5', className)}>
      <div className="flex min-w-0 items-center gap-1.5 text-sm text-ink-2">
        {swatch && <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: swatch }} />}
        <span className="line-clamp-2 min-w-0 break-words">{label}</span>
      </div>
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-1 text-ink">
        <span className={cn('font-semibold tracking-tight', size === 'lg' ? 'text-[2.25rem] leading-[1.1]' : 'text-[1.625rem] leading-[1.15]')}>
          {shown}
        </span>
        {unit && <span className={cn('text-ink-2', size === 'lg' ? 'text-lg' : 'text-[15px]')}>{unit}</span>}
      </div>
      {hint && <div className="text-[13px] leading-snug text-ink-3">{hint}</div>}
    </div>
  )
}
