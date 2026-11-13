import { useId, useRef, type KeyboardEvent } from 'react'
import { FieldLabel, cn } from '@/ui'

/**
 * Выбор одного варианта кнопками, которые переносятся на новую строку (примеры и пресеты; для режимов
 * из 2-4 вариантов есть Segmented). radiogroup, стрелки двигают выбор. Вид как у пресетов живого
 * токенизатора: невыбранные с рамкой, выбранный на мягком акцентном фоне.
 */
export function ChipGroup<T extends string>({
  label,
  options,
  value,
  onChange,
  className,
}: {
  label: string
  options: ReadonlyArray<{ value: T; label: string }>
  value: T
  onChange: (v: T) => void
  className?: string
}) {
  const id = useId()
  const refs = useRef<Array<HTMLButtonElement | null>>([])
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const keys = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End']
    if (!keys.includes(e.key)) return
    e.preventDefault()
    const cur = options.findIndex((o) => o.value === value)
    let n = cur
    if (e.key === 'Home') n = 0
    else if (e.key === 'End') n = options.length - 1
    else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') n = (cur + 1) % options.length
    else n = (cur - 1 + options.length) % options.length
    onChange(options[n].value)
    refs.current[n]?.focus()
  }
  return (
    <div className={cn('grid content-start gap-1.5', className)}>
      <FieldLabel id={id}>{label}</FieldLabel>
      <div role="radiogroup" aria-labelledby={id} onKeyDown={onKey} className="flex flex-wrap gap-1.5">
        {options.map((o, i) => {
          const checked = o.value === value
          return (
            <button
              key={o.value}
              ref={(el) => {
                refs.current[i] = el
              }}
              type="button"
              role="radio"
              aria-checked={checked}
              tabIndex={checked ? 0 : -1}
              onClick={() => onChange(o.value)}
              className={cn(
                'h-8 rounded-control px-3 text-[13.5px] font-medium whitespace-nowrap transition-colors',
                checked
                  ? 'bg-accent-soft text-accent-hover shadow-[inset_0_0_0_1px_var(--accent)]'
                  : 'bg-surface text-ink-2 shadow-[inset_0_0_0_1px_var(--line-strong)] hover:bg-surface-3 hover:text-ink',
              )}
            >
              {o.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
