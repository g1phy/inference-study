import type { ReactNode } from 'react'
import { cn } from './cn'

/** Подпись поля над контролом (DESIGN §8: видимая подпись, не placeholder). Справа можно показать значение. */
export function FieldLabel({
  htmlFor,
  id,
  children,
  aside,
  className,
}: {
  htmlFor?: string
  id?: string
  children: ReactNode
  aside?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex min-w-0 items-baseline justify-between gap-3', className)}>
      {htmlFor ? (
        <label htmlFor={htmlFor} id={id} className="min-w-0 text-sm font-medium text-ink-2">
          {children}
        </label>
      ) : (
        <span id={id} className="min-w-0 text-sm font-medium text-ink-2">
          {children}
        </span>
      )}
      {aside !== undefined && <span className="shrink-0 text-sm font-medium text-ink">{aside}</span>}
    </div>
  )
}

/** Пояснение под полем. */
export function FieldHint({ id, children }: { id?: string; children: ReactNode }) {
  return (
    <p id={id} className="text-[13px] leading-snug text-ink-3">
      {children}
    </p>
  )
}
