import { useId, type ReactNode } from 'react'
import { CaretDownIcon } from '@phosphor-icons/react'
import { cn } from './cn'
import { FieldHint, FieldLabel } from './Field'

export interface SelectOption<T extends string> {
  value: T
  label: string
  disabled?: boolean
  /** Группа (optgroup), например «Qwen3». */
  group?: string
}

export interface SelectProps<T extends string> {
  label: ReactNode
  value: T
  onChange: (value: T) => void
  options: ReadonlyArray<SelectOption<T>>
  hint?: ReactNode
  disabled?: boolean
  id?: string
  size?: 'sm' | 'md'
  className?: string
}

/**
 * Выпадающий список на нативном <select> (лучшая доступность и мобильный UX).
 *
 *   <Select label="GPU" value={gpu} onChange={setGpu} options={[{ value: 'h100-sxm', label: 'H100 SXM' }]} />
 */
export function Select<T extends string>({
  label,
  value,
  onChange,
  options,
  hint,
  disabled,
  id,
  size = 'md',
  className,
}: SelectProps<T>) {
  const autoId = useId()
  const selectId = id ?? `select-${autoId}`
  const hintId = hint ? `${selectId}-hint` : undefined
  const groups = new Map<string, Array<SelectOption<T>>>()
  for (const o of options) {
    const g = o.group ?? ''
    if (!groups.has(g)) groups.set(g, [])
    groups.get(g)!.push(o)
  }
  const renderOption = (o: SelectOption<T>) => (
    <option key={o.value} value={o.value} disabled={o.disabled}>
      {o.label}
    </option>
  )

  return (
    <div className={cn('grid gap-1.5', className)}>
      <FieldLabel htmlFor={selectId}>{label}</FieldLabel>
      <div className="relative">
        <select
          id={selectId}
          value={value}
          disabled={disabled}
          aria-describedby={hintId}
          onChange={(e) => onChange(e.currentTarget.value as T)}
          className={cn(
            'w-full appearance-none rounded-control bg-surface-2 pr-9 pl-3 text-ink',
            'shadow-[inset_0_0_0_1px_var(--line)] transition-shadow hover:shadow-[inset_0_0_0_1px_var(--line-strong)]',
            'disabled:opacity-45',
            size === 'sm' ? 'h-8 text-sm' : 'h-10 text-[15px]',
          )}
        >
          {[...groups.entries()].map(([g, opts]) =>
            g ? (
              <optgroup key={g} label={g}>
                {opts.map(renderOption)}
              </optgroup>
            ) : (
              opts.map(renderOption)
            ),
          )}
        </select>
        <CaretDownIcon
          size={16}
          aria-hidden
          className="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2 text-ink-3"
        />
      </div>
      {hint && <FieldHint id={hintId}>{hint}</FieldHint>}
    </div>
  )
}
