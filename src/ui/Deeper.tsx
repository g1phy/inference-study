import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { CaretRightIcon } from '@phosphor-icons/react'
import { cn } from './cn'

export interface DeeperProps {
  /** Заголовок. По умолчанию «Глубже»; свой показывается как «Глубже: <title>». */
  title?: ReactNode
  children: ReactNode
  defaultOpen?: boolean
  /** Монтировать содержимое только после первого открытия (для тяжёлых фигур внутри). */
  lazy?: boolean
  className?: string
}

/**
 * Сворачиваемый блок «Глубже» для деталей, которые не нужны с первого прочтения.
 * Закрытое содержимое находится поиском по странице (hidden="until-found") и само раскрывается.
 *
 *   <Deeper title="зачем делить на √d_k">…</Deeper>
 */
export function Deeper({ title = 'Глубже', children, defaultOpen = false, lazy = false, className }: DeeperProps) {
  const [open, setOpen] = useState(defaultOpen)
  const [everOpened, setEverOpened] = useState(defaultOpen)
  const id = useId()
  const panel = useRef<HTMLDivElement>(null)

  // hidden="until-found": поиск по странице находит текст в закрытом блоке (Chrome), блок раскрывается сам
  useLayoutEffect(() => {
    const el = panel.current
    if (!el) return
    if (open) el.removeAttribute('hidden')
    else el.setAttribute('hidden', 'until-found')
    const onMatch = () => {
      setOpen(true)
      setEverOpened(true)
    }
    el.addEventListener('beforematch', onMatch)
    return () => el.removeEventListener('beforematch', onMatch)
  }, [open])

  const toggle = () => {
    setOpen((o) => !o)
    setEverOpened(true)
  }
  const custom = title !== 'Глубже'

  return (
    <div className={cn('deeper rounded-panel border border-line', className)}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={toggle}
        className="flex w-full items-start gap-2.5 rounded-panel px-4 py-3 text-left transition-colors hover:bg-surface-3/60 sm:px-5"
      >
        <CaretRightIcon
          size={16}
          aria-hidden
          className={cn('mt-[5px] shrink-0 text-ink-3 transition-transform duration-200', open && 'rotate-90')}
        />
        <span className="text-[15px] leading-6 font-semibold text-ink">
          {custom ? (
            <>
              <span className="text-ink-3">Глубже: </span>
              {title}
            </>
          ) : (
            'Глубже'
          )}
        </span>
      </button>
      <div ref={panel} id={id} hidden={!open}>
        <div className="px-4 pb-5 sm:px-5">
          <div className={cn('prose border-t border-line pt-4 text-[16px] leading-[1.6]', open && 'animate-fade-in')}>
            {lazy && !everOpened ? null : children}
          </div>
        </div>
      </div>
    </div>
  )
}
