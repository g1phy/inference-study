import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type FocusEvent,
  type PointerEvent,
  type ReactNode,
} from 'react'
import { cn } from './cn'
import { FloatingLayer, TipCard, type Align, type RectLike, type Side } from './floating'

/* ===========================================================================
 * Tooltip: подсказка при наведении и фокусе (не интерактивная).
 * ========================================================================= */

export interface TooltipProps {
  content: ReactNode
  children: ReactNode
  side?: Side
  align?: Align
  /** Задержка показа при наведении мышью, мс. */
  delay?: number
  disabled?: boolean
  /** Шире стандартных 280px. */
  wide?: boolean
  /** Класс обёртки (по умолчанию inline-flex). */
  className?: string
}

/**
 * Обёртка, показывающая подсказку над содержимым. На тач-устройствах открывается тапом.
 *
 *   <Tooltip content="Time to first token"><span>TTFT</span></Tooltip>
 */
export function Tooltip({ content, children, side = 'top', align = 'center', delay = 150, disabled, wide, className }: TooltipProps) {
  const id = useId()
  const ref = useRef<HTMLSpanElement>(null)
  const timer = useRef<number | undefined>(undefined)
  const described = useRef<HTMLElement | null>(null)
  const [anchor, setAnchor] = useState<RectLike | null>(null)

  const open = useCallback(() => {
    const el = ref.current
    if (el) setAnchor(el.getBoundingClientRect())
  }, [])
  const close = useCallback(() => {
    window.clearTimeout(timer.current)
    setAnchor(null)
  }, [])

  useEffect(() => {
    if (!anchor) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close()
    const onDown = (e: globalThis.PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) close()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown, true)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown, true)
    }
  }, [anchor, close])

  useEffect(() => () => window.clearTimeout(timer.current), [])

  const onPointerEnter = (e: PointerEvent) => {
    if (e.pointerType !== 'mouse') return
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(open, delay)
  }
  const onPointerLeave = (e: PointerEvent) => e.pointerType === 'mouse' && close()
  const onPointerUp = (e: PointerEvent) => {
    if (e.pointerType === 'mouse') return
    if (anchor) close()
    else open()
  }
  const onFocus = (e: FocusEvent) => {
    const t = e.target as HTMLElement
    if (!t.matches(':focus-visible')) return
    described.current = t
    t.setAttribute('aria-describedby', id)
    open()
  }
  const onBlur = () => {
    described.current?.removeAttribute('aria-describedby')
    described.current = null
    close()
  }

  return (
    <span
      ref={ref}
      className={cn('inline-flex', className)}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
      onPointerUp={onPointerUp}
      onFocus={onFocus}
      onBlur={onBlur}
    >
      {children}
      {anchor && !disabled && (
        <FloatingLayer anchor={anchor} id={id} role="tooltip" side={side} align={align} scopeEl={ref.current}>
          <TipCard wide={wide}>{content}</TipCard>
        </FloatingLayer>
      )}
    </span>
  )
}

/* ===========================================================================
 * Popover: панель по клику с интерактивным содержимым (role="dialog", не модальная).
 * ========================================================================= */

export interface PopoverTriggerProps {
  ref: (el: HTMLElement | null) => void
  onClick: () => void
  'aria-expanded': boolean
  'aria-controls': string
  'aria-haspopup': 'dialog'
}

export interface PopoverProps {
  /** Содержимое панели; функция получает close(). */
  content: ReactNode | ((close: () => void) => ReactNode)
  /** Рендер-проп триггера: раздайте props своей кнопке. */
  children: (trigger: PopoverTriggerProps) => ReactNode
  /** Подпись панели для скринридеров. */
  label: string
  side?: Side
  align?: Align
  open?: boolean
  onOpenChange?: (open: boolean) => void
  className?: string
}

/**
 * Поповер по клику. Закрывается по Escape (фокус возвращается на триггер) и кликом снаружи.
 *
 *   <Popover label="Параметры" content={<div className="p-4">…</div>}>
 *     {(p) => <Button {...p} icon={SlidersHorizontalIcon}>Параметры</Button>}
 *   </Popover>
 */
export function Popover({ content, children, label, side = 'bottom', align = 'start', open: openProp, onOpenChange, className }: PopoverProps) {
  const id = useId()
  const [openState, setOpenState] = useState(false)
  const open = openProp ?? openState
  const trigger = useRef<HTMLElement | null>(null)
  const panel = useRef<HTMLDivElement | null>(null)
  const [anchor, setAnchor] = useState<RectLike | null>(null)

  const setOpen = useCallback(
    (v: boolean) => {
      if (openProp === undefined) setOpenState(v)
      onOpenChange?.(v)
    },
    [openProp, onOpenChange],
  )
  const close = useCallback(() => setOpen(false), [setOpen])

  useEffect(() => {
    if (!open) {
      setAnchor(null)
      return
    }
    if (trigger.current) setAnchor(trigger.current.getBoundingClientRect())
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        close()
        trigger.current?.focus()
      }
    }
    const onDown = (e: globalThis.PointerEvent) => {
      const t = e.target as Node
      if (panel.current?.contains(t) || trigger.current?.contains(t)) return
      close()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown, true)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown, true)
    }
  }, [open, close])

  // фокус внутрь панели, когда слой уже позиционирован и видим (после синхронного перерендера слоя)
  useEffect(() => {
    if (!anchor) return
    const timer = window.setTimeout(() => {
      const root = panel.current
      if (!root) return
      const first = root.querySelector<HTMLElement>(
        'button:not(:disabled), [href], input:not(:disabled), select, textarea, [tabindex]:not([tabindex="-1"])',
      )
      ;(first ?? root.querySelector<HTMLElement>('[role="dialog"]') ?? root).focus({ preventScroll: true })
    }, 0)
    return () => window.clearTimeout(timer)
  }, [anchor])

  return (
    <>
      {children({
        ref: (el) => {
          trigger.current = el
        },
        onClick: () => setOpen(!open),
        'aria-expanded': open,
        'aria-controls': id,
        'aria-haspopup': 'dialog',
      })}
      {open && anchor && (
        <FloatingLayer
          anchor={anchor}
          side={side}
          align={align}
          interactive
          scopeEl={trigger.current}
          layerRef={(el) => {
            panel.current = el
          }}
        >
          <div
            id={id}
            role="dialog"
            aria-label={label}
            tabIndex={-1}
            className={cn(
              'animate-fade-in rounded-panel border border-line bg-surface text-ink shadow-pop outline-none',
              className,
            )}
          >
            {typeof content === 'function' ? content(close) : content}
          </div>
        </FloatingLayer>
      )}
    </>
  )
}
