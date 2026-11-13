import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react'
import { cn } from './cn'
import { FieldLabel } from './Field'
import { renderIcon, type IconLike } from './icon'

export interface SegmentedOption<T extends string | number> {
  value: T
  label: ReactNode
  icon?: IconLike
  disabled?: boolean
  /** Подсказка (title), полезна для вариантов только с иконкой. */
  title?: string
}

export interface SegmentedProps<T extends string | number> {
  label?: ReactNode
  options: ReadonlyArray<SegmentedOption<T>>
  value: T
  onChange: (value: T) => void
  size?: 'sm' | 'md'
  fullWidth?: boolean
  /** Переносить варианты на следующую строку вместо горизонтальной прокрутки (узкие колонки контролов). */
  wrap?: boolean
  className?: string
  /** Если нет видимой подписи. */
  'aria-label'?: string
}

/**
 * Сегментированный выбор одного варианта (radiogroup, стрелки двигают выбор).
 *
 *   <Segmented label="Деплой" value={mode} onChange={setMode}
 *     options={[{ value: 'agg', label: 'Агрегированный' }, { value: 'disagg', label: 'Дезагрегированный' }]} />
 */
export function Segmented<T extends string | number>({
  label,
  options,
  value,
  onChange,
  size = 'md',
  fullWidth = false,
  wrap = false,
  className,
  'aria-label': ariaLabel,
}: SegmentedProps<T>) {
  const labelId = useId()
  const refs = useRef<Array<HTMLButtonElement | null>>([])
  const enabled = options.map((o, i) => (o.disabled ? -1 : i)).filter((i) => i >= 0)

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const keys = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End']
    if (!keys.includes(e.key) || enabled.length === 0) return
    e.preventDefault()
    const cur = enabled.indexOf(options.findIndex((o) => o.value === value))
    let next = cur
    if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = enabled.length - 1
    else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (cur + 1) % enabled.length
    else next = (cur - 1 + enabled.length) % enabled.length
    const idx = enabled[next]
    onChange(options[idx].value)
    refs.current[idx]?.focus()
  }

  return (
    // content-start и self-start: в строке контролов с более высоким соседом (чипы, слайдер с подсказкой)
    // сетка тянула дорожку на всю высоту строки, и под кнопками оставалась пустая серая полоса
    <div className={cn('grid content-start gap-1.5', fullWidth ? 'w-full' : 'justify-items-start', className)}>
      {label && <FieldLabel id={labelId}>{label}</FieldLabel>}
      <div
        role="radiogroup"
        aria-labelledby={label ? labelId : undefined}
        aria-label={label ? undefined : ariaLabel}
        onKeyDown={onKeyDown}
        className={cn(
          'inline-flex max-w-full self-start gap-0.5 rounded-control bg-surface-2 p-0.5',
          wrap ? 'flex-wrap' : 'overflow-x-auto no-scrollbar',
          'shadow-[inset_0_0_0_1px_var(--line)]',
          fullWidth && 'flex w-full',
        )}
      >
        {options.map((o, i) => {
          const checked = o.value === value
          return (
            <button
              key={String(o.value)}
              ref={(el) => {
                refs.current[i] = el
              }}
              type="button"
              role="radio"
              aria-checked={checked}
              disabled={o.disabled}
              title={o.title}
              tabIndex={checked ? 0 : -1}
              onClick={() => onChange(o.value)}
              className={cn(
                'inline-flex shrink-0 items-center justify-center gap-1.5 rounded-[6px] font-medium whitespace-nowrap transition-colors duration-150',
                // контейнер прокручивается (overflow-x-auto) и обрезал бы внешнее кольцо фокуса: рисуем его внутри
                'disabled:opacity-40 focus-visible:-outline-offset-2',
                // вместе с полем 2px: sm = 32px, md = 40px, как Button и Select тех же размеров
                size === 'sm' ? 'h-7 px-2.5 text-[13px]' : 'h-9 px-3 text-sm',
                fullWidth && 'flex-1',
                checked
                  ? 'bg-surface text-ink shadow-[0_0_0_1px_var(--line),0_1px_1px_rgba(0,0,0,0.04)]'
                  : 'text-ink-2 hover:text-ink',
              )}
            >
              {renderIcon(o.icon, 16)}
              {o.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
