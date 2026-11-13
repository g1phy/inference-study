import type { MouseEvent, ReactNode } from 'react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { getChapter, isFirstInPart, nextChapter, type ChapterId, type ChapterInfo } from '@/content/chapters'
import { navigateTo } from '@/state/nav'
import { cn } from './cn'

export interface ChapterProps {
  id: ChapterId
  /** Лид 1-3 предложения простыми словами. Если не передан, берётся черновой из реестра. */
  lead?: ReactNode
  /** Заголовок вместо реестрового (редко). */
  title?: ReactNode
  children?: ReactNode
  /** 'default': номер и h2; 'hero': h1 крупно (глава hero); 'none': без шапки (своя вёрстка). */
  header?: 'default' | 'hero' | 'none'
  /**
   * Ссылка «Дальше: …» в конце главы (по умолчанию есть у всех, кроме последней). Перед новой частью
   * её нет: там стоит заставка части со списком глав, и название следующей главы не повторяется трижды.
   */
  next?: boolean
  className?: string
}

/**
 * Обёртка главы. Рендерит <section id>, шапку из реестра, детей в сетке прозы и ссылку «Дальше».
 *
 * Прямые потомки встают в колонку прозы (60ch). <Figure width="wide|full"> и <Block width>
 * выходят шире. Абзацы можно писать прямо внутри <Chapter> или в <Prose>.
 *
 *   export default function AttentionChapter() {
 *     return (
 *       <Chapter id="attention" lead={<>Attention даёт каждому токену…</>}>
 *         <p>…</p>
 *         <Figure title="…" width="wide">…</Figure>
 *         <Takeaways items={[…]} />
 *       </Chapter>
 *     )
 *   }
 */
export function Chapter({ id, lead, title, children, header, next = true, className }: ChapterProps) {
  const info = getChapter(id)
  const mode = header ?? (id === 'hero' ? 'hero' : 'default')
  const upcoming = next ? nextChapter(id) : undefined
  const nextInfo = upcoming && !isFirstInPart(upcoming.id) ? upcoming : undefined
  return (
    <section
      id={id}
      data-chapter={id}
      aria-labelledby={mode === 'none' ? undefined : `${id}-title`}
      aria-label={mode === 'none' ? info.title : undefined}
      className={cn(mode === 'hero' ? 'pt-10 pb-16 desk:pt-16 desk:pb-24' : 'py-14 desk:py-24', className)}
    >
      <div className="page-container">
        <div className="flow">
          {mode !== 'none' && <ChapterHeader info={info} lead={lead} title={title} hero={mode === 'hero'} />}
          {children}
          {nextInfo && <NextChapterLink info={nextInfo} />}
        </div>
      </div>
    </section>
  )
}

/** Шапка главы: номер обычным начертанием рядом с заголовком, лид. Используется и заглушкой до монтирования. */
export function ChapterHeader({
  info,
  lead,
  title,
  hero = false,
}: {
  info: ChapterInfo
  lead?: ReactNode
  title?: ReactNode
  hero?: boolean
}) {
  const shownLead = lead ?? info.lead
  if (hero) {
    return (
      <header className="grid gap-5">
        <h1
          id={`${info.id}-title`}
          data-chapter-heading
          className="max-w-[18ch] text-[clamp(2.25rem,4.5vw,3.75rem)] leading-[1.05] font-[650] tracking-tight text-ink outline-none"
        >
          {title ?? info.title}
        </h1>
        {shownLead && <p className="max-w-[46ch] text-[1.1875rem] leading-[1.55] text-ink-2 desk:text-[1.3125rem]">{shownLead}</p>}
      </header>
    )
  }
  return (
    <header className="grid gap-5 pb-2">
      <h2
        id={`${info.id}-title`}
        data-chapter-heading
        className="text-[2rem] leading-[1.1] font-[650] tracking-[-0.02em] text-ink outline-none desk:text-[2.5rem]"
      >
        {/* пропорциональные цифры: табличная «1» в Onest широкая, «11» выглядело как «1 1» */}
        <span className="mr-[0.4em] font-normal text-ink-3">{info.number}</span>
        {title ?? info.title}
      </h2>
      {/* 54ch при 19px совпадает по правому краю с колонкой прозы 60ch при 17px */}
      {shownLead && <p className="max-w-[54ch] text-[1.1875rem] leading-[1.6] text-ink-2">{shownLead}</p>}
    </header>
  )
}

/** Первое предложение лида: зачем читать следующую главу. */
function firstSentence(text: string): string {
  const m = /^(.+?[.!?…])(?=\s|$)/.exec(text.trim())
  return (m ? m[1] : text).trim()
}

function NextChapterLink({ info }: { info: ChapterInfo }) {
  const onClick = (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    navigateTo(info.id)
  }
  const teaser = info.lead ? firstSentence(info.lead) : ''
  return (
    <nav aria-label="Следующая глава" className="block-gap mt-16 desk:mt-20">
      <a
        href={`#${info.id}`}
        onClick={onClick}
        className="group flex items-center justify-between gap-6 border-t border-line pt-5 text-ink"
      >
        <span className="grid min-w-0 gap-1">
          <span className="text-sm text-ink-3">Дальше</span>
          <span className="text-[1.25rem] leading-snug font-semibold transition-colors group-hover:text-accent">
            <span className="mr-2 font-normal text-ink-3">{info.number}</span>
            {info.title}
          </span>
          {teaser && <span className="line-clamp-2 text-[15px] leading-snug text-ink-2">{teaser}</span>}
        </span>
        <ArrowRightIcon
          size={20}
          aria-hidden
          className="shrink-0 text-ink-3 transition-[transform,color] group-hover:translate-x-0.5 group-hover:text-accent"
        />
      </a>
    </nav>
  )
}
