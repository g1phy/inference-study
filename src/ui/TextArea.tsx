import { useId, useLayoutEffect, useRef, type KeyboardEventHandler, type ReactNode } from 'react'
import { cn } from './cn'
import { FieldHint, FieldLabel } from './Field'
import { fmt } from './fmt'

export interface TextAreaProps {
  label: ReactNode
  value: string
  onChange: (value: string) => void
  rows?: number
  hint?: ReactNode
  placeholder?: string
  maxLength?: number
  /**
   * Счётчик символов справа от подписи. 'auto' (по умолчанию): показывается, когда задан maxLength
   * и текст занял больше половины лимита («3 416 / 4 000»).
   */
  counter?: boolean | 'auto'
  /** Подстраивать высоту под текст (не меньше rows и не больше maxRows, дальше прокрутка внутри). */
  autoGrow?: boolean
  /**
   * Предел роста при autoGrow, строк (по умолчанию 10, около 240 px). Длинный промпт иначе
   * растягивал поле на несколько экранов и уводил кнопку «Отправить» и всё под ней вниз.
   */
  maxRows?: number
  /** Моноширинный шрифт (для JSON, шаблонов). */
  mono?: boolean
  disabled?: boolean
  readOnly?: boolean
  /** Симулятор ставит id="prompt-input": чип запроса в навигации фокусирует это поле. */
  id?: string
  className?: string
  /** Классы строки подписи (подпись и счётчик), например отступ справа под свою кнопку поверх этой строки. */
  labelClassName?: string
  onKeyDown?: KeyboardEventHandler<HTMLTextAreaElement>
}

/**
 * Многострочное поле с подписью над ним.
 *
 *   <TextArea id="prompt-input" label="Сообщение пользователя" value={text} onChange={setText} rows={3} autoGrow />
 */
export function TextArea({
  label,
  value,
  onChange,
  rows = 3,
  hint,
  placeholder,
  maxLength,
  counter = 'auto',
  autoGrow = false,
  maxRows = 10,
  mono = false,
  disabled,
  readOnly,
  id,
  className,
  labelClassName,
  onKeyDown,
}: TextAreaProps) {
  const autoId = useId()
  const fieldId = id ?? `textarea-${autoId}`
  const hintId = hint ? `${fieldId}-hint` : undefined
  const ref = useRef<HTMLTextAreaElement>(null)

  // autoGrow: высота по содержимому, но не выше maxRows строк (дальше прокрутка внутри поля);
  // пересчёт и при изменении ширины поля
  useLayoutEffect(() => {
    const el = ref.current
    if (!autoGrow || !el) return
    const fit = () => {
      const cs = window.getComputedStyle(el)
      const line = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.625 || 24
      const pad = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0)
      const maxPx = Math.max(line * Math.max(rows, maxRows) + pad + 2, 0)
      el.style.height = 'auto'
      const want = el.scrollHeight + 2
      el.style.height = `${Math.min(want, maxPx)}px`
      el.style.overflowY = want > maxPx ? 'auto' : 'hidden'
    }
    fit()
    if (typeof ResizeObserver === 'undefined') return
    let lastWidth = el.clientWidth
    const ro = new ResizeObserver(() => {
      if (Math.abs(el.clientWidth - lastWidth) < 1) return
      lastWidth = el.clientWidth
      fit()
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [value, autoGrow, rows, maxRows])

  const showCounter = counter === 'auto' ? Boolean(maxLength) && value.length >= maxLength! * 0.5 : counter
  const aside = showCounter ? `${fmt.int(value.length)}${maxLength ? ` / ${fmt.int(maxLength)}` : ''}` : undefined

  return (
    <div className={cn('grid gap-1.5', className)}>
      <FieldLabel htmlFor={fieldId} aside={aside} className={labelClassName}>
        {label}
      </FieldLabel>
      <textarea
        ref={ref}
        id={fieldId}
        value={value}
        rows={rows}
        placeholder={placeholder}
        maxLength={maxLength}
        disabled={disabled}
        readOnly={readOnly}
        aria-describedby={hintId}
        onChange={(e) => onChange(e.currentTarget.value)}
        onKeyDown={onKeyDown}
        className={cn(
          'block w-full resize-y rounded-control bg-surface-2 px-3 py-2.5 text-ink placeholder:text-ink-3',
          'shadow-[inset_0_0_0_1px_var(--line)] transition-shadow hover:shadow-[inset_0_0_0_1px_var(--line-strong)]',
          'focus-visible:outline-2 focus-visible:outline-offset-0 disabled:opacity-45',
          mono ? 'font-mono text-[13.5px] leading-relaxed' : 'text-[15px] leading-relaxed',
          autoGrow && 'resize-none overflow-hidden scrollbar-thin',
        )}
      />
      {hint && <FieldHint id={hintId}>{hint}</FieldHint>}
    </div>
  )
}
