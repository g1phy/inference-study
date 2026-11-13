import type { ReactNode } from 'react'
import { ArrowsHorizontalIcon } from '@phosphor-icons/react'
import { cn } from './cn'
import { useScrollEdges } from './hooks'

/** Строка-подсказка под широким блоком: на телефоне полосы прокрутки не видно. */
export function ScrollHint({ children = 'шире экрана: прокрутите вбок', className }: { children?: ReactNode; className?: string }) {
  return (
    <div className={cn('mt-1.5 flex items-center gap-1.5 text-[12.5px] leading-snug text-ink-3', className)}>
      <ArrowsHorizontalIcon size={16} aria-hidden className="shrink-0" />
      {children}
    </div>
  )
}

export interface ScrollXProps {
  children: ReactNode
  /** Подсказка «шире экрана: прокрутите вбок», когда содержимое не влезает (по умолчанию есть). */
  hint?: boolean | ReactNode
  /** Подпись прокручиваемой области для скринридеров (когда она прокручивается, её можно выбрать Tab). */
  label?: string
  className?: string
  /** Классы самого прокручиваемого блока (поля, чтобы не обрезалось кольцо фокуса, и т.п.). */
  scrollerClassName?: string
}

/**
 * Горизонтальная прокрутка для широкого содержимого (таблица, треугольник attention, схема):
 * если не влезает, обрезанный край затухает, а под блоком подсказка. Страница вбок не едет.
 *
 *   <ScrollX label="Таблица стоимости"><table className="data-table min-w-[560px]">…</table></ScrollX>
 */
export function ScrollX({ children, hint = true, label, className, scrollerClassName }: ScrollXProps) {
  const [ref, edges] = useScrollEdges<HTMLDivElement>()
  const scrolls = edges !== 'none'
  return (
    <div className={cn('min-w-0', className)}>
      <div
        ref={ref}
        // прокручиваемую область можно выбрать с клавиатуры и листать стрелками (только когда есть что листать)
        tabIndex={scrolls ? 0 : undefined}
        role={scrolls ? 'region' : undefined}
        aria-label={scrolls ? (label ?? 'Прокручиваемая область') : undefined}
        className={cn('scroll-fade max-w-full overflow-x-auto scrollbar-thin focus-visible:-outline-offset-2', scrollerClassName)}
      >
        {children}
      </div>
      {scrolls && hint !== false && <ScrollHint>{hint === true ? undefined : hint}</ScrollHint>}
    </div>
  )
}
