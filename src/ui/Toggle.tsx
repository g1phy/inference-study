import type { ReactNode } from 'react'
import { cn } from './cn'

export interface ToggleProps {
  label: ReactNode
  checked: boolean
  onChange: (checked: boolean) => void
  hint?: ReactNode
  disabled?: boolean
  id?: string
  className?: string
}

/**
 * Переключатель (role="switch"). Подпись справа и кликабельна целиком.
 *
 *   <Toggle label="Thinking" checked={on} onChange={setOn} />
 */
export function Toggle({ label, checked, onChange, hint, disabled, id, className }: ToggleProps) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        'group inline-flex min-h-8 items-start gap-2.5 rounded-control py-1 text-left disabled:opacity-45',
        className,
      )}
    >
      <span
        aria-hidden
        className={cn(
          'relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-colors duration-150',
          checked ? 'bg-accent' : 'bg-ink-3',
        )}
      >
        <span
          className="absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-surface transition-transform duration-150"
          style={{ transform: checked ? 'translateX(16px)' : 'none' }}
        />
      </span>
      <span className="grid gap-0.5">
        <span className="text-[15px] leading-6 text-ink">{label}</span>
        {hint && <span className="text-[13px] leading-snug text-ink-3">{hint}</span>}
      </span>
    </button>
  )
}
