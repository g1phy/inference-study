import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react'
import { cn } from './cn'
import { useScrollEdges } from './hooks'
import { renderIcon, type IconLike } from './icon'

export interface TabItem<T extends string> {
  id: T
  label: ReactNode
  icon?: IconLike
  disabled?: boolean
}

export interface TabsProps<T extends string> {
  tabs: ReadonlyArray<TabItem<T>>
  value: T
  onChange: (id: T) => void
  /** Содержимое активной вкладки (рендерится в role="tabpanel" с правильными aria-связями). */
  children?: (active: T) => ReactNode
  /** Подпись списка вкладок для скринридеров. */
  'aria-label'?: string
  size?: 'sm' | 'md'
  /**
   * Переносить вкладки на вторую строку, если не влезают (узкая фигура, телефон). Без wrap ряд прокручивается вбок,
   * а обрезанный край затухает: так видно, что вкладки есть и дальше.
   */
  wrap?: boolean
  className?: string
  panelClassName?: string
}

/**
 * Вкладки (WAI-ARIA tabs, стрелки и Home/End). Подчёркивание активной вкладки акцентом.
 *
 *   <Tabs tabs={[{ id: 'qwen3', label: 'Qwen3' }, { id: 'llama3', label: 'Llama 3.1' }]} value={f} onChange={setF}>
 *     {(active) => <TemplateView family={active} />}
 *   </Tabs>
 */
export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  children,
  'aria-label': ariaLabel,
  size = 'md',
  wrap = false,
  className,
  panelClassName,
}: TabsProps<T>) {
  const base = useId()
  const [scrollRef] = useScrollEdges<HTMLDivElement>()
  const refs = useRef<Array<HTMLButtonElement | null>>([])
  const tabId = (id: string) => `${base}-tab-${id}`
  const panelId = (id: string) => `${base}-panel-${id}`
  const enabled = tabs.map((t, i) => (t.disabled ? -1 : i)).filter((i) => i >= 0)

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const keys = ['ArrowRight', 'ArrowLeft', 'Home', 'End']
    if (!keys.includes(e.key) || enabled.length === 0) return
    e.preventDefault()
    const cur = enabled.indexOf(tabs.findIndex((t) => t.id === value))
    let next = cur
    if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = enabled.length - 1
    else if (e.key === 'ArrowRight') next = (cur + 1) % enabled.length
    else next = (cur - 1 + enabled.length) % enabled.length
    const idx = enabled[next]
    onChange(tabs[idx].id)
    refs.current[idx]?.focus()
  }

  return (
    <div className={className}>
      <div
        ref={wrap ? undefined : scrollRef}
        role="tablist"
        aria-label={ariaLabel}
        onKeyDown={onKeyDown}
        className={cn('flex max-w-full gap-1 border-b border-line', wrap ? 'flex-wrap' : 'scroll-fade overflow-x-auto no-scrollbar')}
      >
        {tabs.map((t, i) => {
          const selected = t.id === value
          return (
            <button
              key={t.id}
              ref={(el) => {
                refs.current[i] = el
              }}
              id={tabId(t.id)}
              type="button"
              role="tab"
              aria-selected={selected}
              // панель рендерится только у выбранной вкладки: ссылаться на несуществующие id нельзя
              aria-controls={children && selected ? panelId(t.id) : undefined}
              tabIndex={selected ? 0 : -1}
              disabled={t.disabled}
              onClick={() => onChange(t.id)}
              className={cn(
                'relative -mb-px inline-flex shrink-0 items-center gap-1.5 border-b-2 font-medium whitespace-nowrap transition-colors duration-150',
                // список вкладок прокручивается и обрезал бы внешнее кольцо фокуса: рисуем его внутри
                'rounded-t-[6px] disabled:opacity-40 focus-visible:-outline-offset-2',
                size === 'sm' ? 'h-9 px-2.5 text-[13px]' : 'h-10 px-3 text-sm',
                selected ? 'border-accent text-ink' : 'border-transparent text-ink-2 hover:text-ink',
              )}
            >
              {renderIcon(t.icon, 16)}
              {t.label}
            </button>
          )
        })}
      </div>
      {children && (
        <div
          role="tabpanel"
          id={panelId(value)}
          aria-labelledby={tabId(value)}
          tabIndex={0}
          className={cn('pt-4 focus-visible:outline-offset-4', panelClassName)}
        >
          {children(value)}
        </div>
      )}
    </div>
  )
}
