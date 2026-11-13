import type { MouseEvent, ReactNode } from 'react'
import { ArrowRightIcon, FlowArrowIcon } from '@phosphor-icons/react'
import { getChapter, type ChapterId } from '@/content/chapters'
import { navigateTo } from '@/state/nav'
import { selectStage } from '@/state/settings'
import type { StageId } from '@/engine/types'
import { cn } from './cn'

/** Классы текстовой ссылки (для своих <a>/<button>, оформленных как ссылка). */
export const linkClass =
  'text-accent underline decoration-1 underline-offset-[0.2em] decoration-[color-mix(in_oklab,var(--accent)_45%,transparent)] transition-colors hover:text-accent-hover hover:decoration-current'

function onNavigate(to: ChapterId) {
  return (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return
    e.preventDefault()
    navigateTo(to)
  }
}

export interface ChapterLinkProps {
  to: ChapterId
  /** Текст ссылки (по умолчанию короткое имя главы из реестра). */
  children?: ReactNode
  /** inline: ссылка в тексте; card: карточка с номером, заголовком и лидом. */
  variant?: 'inline' | 'card'
  className?: string
}

/**
 * Ссылка на главу по id. Переход монтирует главу, если она ещё не загружена.
 *
 *   Подробнее в главе <ChapterLink to="kv-cache" />.
 *   <ChapterLink to="routing" variant="card" />
 */
export function ChapterLink({ to, children, variant = 'inline', className }: ChapterLinkProps) {
  const info = getChapter(to)
  if (variant === 'card') {
    return (
      <a
        href={`#${to}`}
        onClick={onNavigate(to)}
        className={cn(
          'group grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4 rounded-panel border border-line bg-surface px-4 py-3.5 transition-colors hover:bg-surface-3/50',
          className,
        )}
      >
        <span className="grid min-w-0 gap-0.5">
          <span className="text-[15px] leading-snug font-semibold text-ink group-hover:text-accent">
            <span className="mr-1.5 font-normal text-ink-3">{info.number}</span>
            {children ?? info.title}
          </span>
          <span className="line-clamp-2 text-sm leading-snug text-ink-2">{info.lead}</span>
        </span>
        <ArrowRightIcon size={18} aria-hidden className="text-ink-3 transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />
      </a>
    )
  }
  return (
    <a href={`#${to}`} onClick={onNavigate(to)} title={`${info.number}. ${info.title}`} className={cn(linkClass, className)}>
      {children ?? info.nav}
    </a>
  )
}

/**
 * Кнопка-ссылка «Где это в симуляторе»: выбирает стадию в инспекторе и скроллит к симулятору.
 * Отдельной строкой между блоками главы рисуется чипом с рамкой (index.css: .flow > .sim-link),
 * внутри абзаца остаётся текстовой ссылкой. Лучшее место: проп simulator у <Takeaways>.
 *
 *   <SimulatorLink stage="route" />
 */
export function SimulatorLink({
  stage,
  children = 'Где это в симуляторе',
  variant = 'inline',
  prepare,
  className,
}: {
  stage: StageId
  children?: ReactNode
  /** inline: текстовая ссылка с иконкой; chip: чип с рамкой (так же выглядит отдельной строкой в главе). */
  variant?: 'inline' | 'chip'
  /** Вызывается до перехода: например, включить сценарий, без которого на стадии нечего показать (patchSettings). */
  prepare?: () => void
  className?: string
}) {
  return (
    <button
      type="button"
      onClick={() => {
        prepare?.()
        selectStage(stage)
      }}
      className={cn(
        'sim-link inline-flex h-8 items-center gap-1.5 justify-self-start rounded-control text-sm font-medium text-accent transition-colors hover:bg-accent-soft hover:text-accent-hover',
        variant === 'chip' ? 'px-3 shadow-[inset_0_0_0_1px_var(--line-strong)]' : '-ml-2.5 px-2.5',
        className,
      )}
    >
      <FlowArrowIcon size={16} aria-hidden />
      {children}
    </button>
  )
}
