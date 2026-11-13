import type { ReactNode } from 'react'
import { BookmarkSimpleIcon } from '@phosphor-icons/react'
import { useId } from 'react'
import type { StageId } from '@/engine/types'
import { SimulatorLink } from './ChapterLink'
import { cn } from './cn'

/**
 * «Запомнить»: 2-4 пункта в конце главы. simulator: ссылка «Где это в симуляторе» справа в шапке
 * (одно место во всех главах вместо одинокой строки между блоками).
 *
 *   <Takeaways items={['Prefill считает весь промпт за один проход.', <>Decode упирается в <b>память</b>.</>]} simulator="prefill" />
 */
export function Takeaways({
  items,
  title = 'Запомнить',
  simulator,
  className,
}: {
  items: ReactNode[]
  title?: ReactNode
  simulator?: StageId
  className?: string
}) {
  const id = useId()
  return (
    <aside aria-labelledby={id} className={cn('takeaways rounded-panel border border-line bg-surface px-4 py-5 sm:px-6', className)}>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <h3 id={id} className="flex items-center gap-2 text-[1.0625rem] leading-6 font-semibold text-ink">
          <BookmarkSimpleIcon size={18} aria-hidden className="text-ink-2" />
          {title}
        </h3>
        {simulator && <SimulatorLink stage={simulator} variant="chip" className="-my-1" />}
      </div>
      <ol className="mt-3 grid gap-2.5">
        {items.map((item, i) => (
          <li key={i} className="grid grid-cols-[1.75rem_minmax(0,1fr)] items-baseline text-[16px] leading-[1.6] text-ink">
            <span className="text-[15px] font-medium text-ink-3 tabular-nums">{i + 1}</span>
            <div className="min-w-0">{item}</div>
          </li>
        ))}
      </ol>
    </aside>
  )
}
