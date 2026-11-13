import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type MouseEvent } from 'react'
import { CaretDownIcon, ChatTextIcon, DesktopIcon, ListIcon, MoonIcon, SunIcon } from '@phosphor-icons/react'
import { CHAPTERS, PARTS, chapterIndex, chaptersOfPart, getChapter, type ChapterId } from '@/content/chapters'
import { navigateTo, scrollToSimulator, useActiveChapter } from '@/state/nav'
import { useCommittedSetting } from '@/state/settings'
import { useTheme, type ThemePreference } from '@/state/theme'
import { cn } from './cn'
import { Segmented } from './Segmented'

/* ---------- helpers ---------- */

/** Первые слова текста: «Объясни простыми словами, что такое…». */
export function firstWords(text: string, n = 7): string {
  const words = text.trim().split(/\s+/).filter(Boolean)
  if (words.length === 0) return ''
  const head = words.slice(0, n).join(' ')
  return words.length > n ? `${head.replace(/[,.;:!?]+$/, '')}…` : head
}

function onChapterClick(id: ChapterId, after?: () => void) {
  return (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return
    e.preventDefault()
    navigateTo(id)
    after?.()
  }
}

/* ---------- переключатель темы ---------- */

export function ThemeSwitch({ className }: { className?: string }) {
  const { theme, setTheme } = useTheme()
  return (
    <Segmented<ThemePreference>
      size="sm"
      aria-label="Тема оформления"
      value={theme}
      onChange={setTheme}
      className={className}
      options={[
        { value: 'light', icon: SunIcon, label: <span className="sr-only">Светлая</span>, title: 'Светлая тема' },
        { value: 'dark', icon: MoonIcon, label: <span className="sr-only">Тёмная</span>, title: 'Тёмная тема' },
        { value: 'system', icon: DesktopIcon, label: <span className="sr-only">Как в системе</span>, title: 'Как в системе' },
      ]}
    />
  )
}

/* ---------- чип текущего запроса ---------- */

export function PromptChip({ onDone, className }: { onDone?: () => void; className?: string }) {
  // принятый текст (с задержкой после набора): чип не перерисовывается на каждую букву
  const text = useCommittedSetting('userText')
  const preview = firstWords(text, 7) || 'Напишите свой запрос'
  return (
    <button
      type="button"
      onClick={() => {
        onDone?.()
        scrollToSimulator({ focusPrompt: true })
      }}
      className={cn(
        'group grid w-full gap-0.5 rounded-control border border-line bg-surface px-3 py-2 text-left transition-colors hover:bg-surface-3/60',
        className,
      )}
    >
      <span className="flex items-center gap-1.5 text-xs text-ink-3">
        <ChatTextIcon size={16} aria-hidden />
        Ваш запрос в симуляторе
      </span>
      <span className="line-clamp-2 text-[13.5px] leading-snug text-ink group-hover:text-accent">«{preview}»</span>
    </button>
  )
}

/* ---------- полоса прогресса чтения ---------- */

/** Прогресс чтения: CSS scroll-driven animation, где поддерживается; иначе по активной главе. */
export function ReadingProgress({ className }: { className?: string }) {
  const active = useActiveChapter()
  const fallback = (chapterIndex(active) + 1) / CHAPTERS.length
  return (
    <div aria-hidden className={cn('pointer-events-none h-[2px] overflow-hidden', className)}>
      <div className="read-progress h-full w-full bg-accent" style={{ '--progress': fallback } as CSSProperties} />
    </div>
  )
}

/* ---------- оглавление ---------- */

export function TocList({ onNavigate, className, columns = false }: { onNavigate?: () => void; className?: string; columns?: boolean }) {
  const active = useActiveChapter()
  const activeIdx = chapterIndex(active)

  const item = (id: ChapterId) => {
    const ch = getChapter(id)
    const idx = chapterIndex(id)
    const isActive = id === active
    const passed = idx < activeIdx
    return (
      <li key={id}>
        <a
          href={`#${id}`}
          onClick={onChapterClick(id, onNavigate)}
          aria-current={isActive ? 'location' : undefined}
          className={cn(
            'relative flex items-baseline gap-2.5 rounded-control py-[5px] pr-2 pl-4 text-[13.5px] leading-[1.35] transition-colors',
            isActive ? 'bg-accent-soft font-medium text-ink' : 'text-ink-2 hover:bg-surface-3/70 hover:text-ink',
          )}
        >
          <span
            aria-hidden
            className={cn(
              'absolute top-0 bottom-0 left-1 w-[2px] rounded-full',
              isActive ? 'bg-accent' : passed ? 'bg-[color-mix(in_oklab,var(--accent)_45%,transparent)]' : 'bg-line',
            )}
          />
          {/* пропорциональные цифры: табличная «1» в Onest широкая, «11» читалось как «1 1»;
              на подложке accent-soft цвет --accent даёт 4,3:1, поэтому номер темнее (--accent-hover) */}
          <span className={cn('w-5 shrink-0 text-right text-[12.5px]', isActive ? 'text-accent-hover' : 'text-ink-3')}>
            {ch.number}
          </span>
          <span className="min-w-0">{ch.nav}</span>
        </a>
      </li>
    )
  }

  return (
    <nav aria-label="Оглавление" className={className}>
      {/* columns: в выпадающем оглавлении на планшете две колонки вместо одной длинной в 800 px */}
      <ul className={cn('grid', columns && 'md:block md:columns-2 md:gap-x-8')}>
        {item('hero')}
        {PARTS.map((part) => (
          <li key={part.id} className="mt-3 break-inside-avoid">
            <div className="px-4 pb-1 text-[12px] leading-snug font-medium text-ink-3">
              <span className="tabular-nums">{part.roman}.</span> {part.title}
            </div>
            <ul className="grid">{chaptersOfPart(part.id).map((c) => item(c.id))}</ul>
          </li>
        ))}
      </ul>
    </nav>
  )
}

function ProgressText() {
  const active = useActiveChapter()
  const ch = getChapter(active)
  return (
    <span className="text-xs text-ink-3">
      {ch.number === 0 ? 'Начало' : `Глава ${ch.number} из ${CHAPTERS.length - 1}`}
    </span>
  )
}

/* ---------- левая колонка (≥1200px) ---------- */

export function NavSidebar() {
  const scroller = useRef<HTMLDivElement>(null)
  const active = useActiveChapter()

  // держим активный пункт в видимой части списка (скролл только внутри колонки)
  useEffect(() => {
    const box = scroller.current
    const el = box?.querySelector<HTMLElement>('[aria-current="location"]')
    if (!box || !el) return
    const top = el.offsetTop - box.offsetTop
    if (top < box.scrollTop + 40 || top > box.scrollTop + box.clientHeight - 80) {
      const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
      box.scrollTo({ top: Math.max(0, top - box.clientHeight / 3), behavior: reduce ? 'instant' : 'smooth' })
    }
  }, [active])

  return (
    <aside className="fixed inset-y-0 left-0 z-40 hidden w-[264px] flex-col border-r border-line bg-bg desk:flex">
      <ReadingProgress className="absolute inset-x-0 top-0" />
      <div className="px-5 pt-6 pb-4">
        <a href="#hero" onClick={onChapterClick('hero')} className="block text-[16px] leading-tight font-semibold tracking-[-0.01em] text-ink">
          Инференс LLM наглядно
        </a>
        <p className="mt-1 text-[12.5px] leading-snug text-ink-3">Путь запроса от Enter до последнего токена</p>
      </div>
      <div className="px-4">
        <PromptChip />
      </div>
      <div ref={scroller} className="relative mt-3 min-h-0 flex-1 overflow-y-auto px-2 pb-6 scrollbar-thin">
        <TocList />
      </div>
      <div className="flex items-center justify-between gap-2 border-t border-line px-4 py-3">
        <ProgressText />
        <ThemeSwitch />
      </div>
    </aside>
  )
}

/* ---------- верхняя панель (<1200px) ---------- */

export function NavTopBar() {
  const [open, setOpen] = useState(false)
  const active = useActiveChapter()
  const ch = getChapter(active)
  const panelId = useId()
  const root = useRef<HTMLElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  const toggleRef = useRef<HTMLButtonElement>(null)

  // открыли оглавление: текущая глава сразу на виду (иначе к главе 25 пришлось бы листать список)
  useLayoutEffect(() => {
    if (!open) return
    const box = panel.current
    const el = box?.querySelector<HTMLElement>('[aria-current="location"]')
    if (box && el) box.scrollTop = Math.max(0, el.offsetTop - box.clientHeight / 3)
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // фокус был в закрывающемся списке: возвращаем его на кнопку, а не роняем в <body>
      const inside = panel.current?.contains(document.activeElement) ?? false
      setOpen(false)
      if (inside) toggleRef.current?.focus()
    }
    const onDown = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown, true)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown, true)
    }
  }, [open])

  return (
    <header ref={root} className="sticky top-0 z-40 border-b border-line bg-[color-mix(in_oklab,var(--bg)_92%,transparent)] backdrop-blur-md desk:hidden">
      <div className="flex h-13 items-center gap-2 px-2 md:px-6">
        <button
          ref={toggleRef}
          type="button"
          aria-expanded={open}
          aria-controls={open ? panelId : undefined}
          onClick={() => setOpen((o) => !o)}
          className="flex h-10 min-w-0 flex-1 items-center gap-2 rounded-control px-2 text-left transition-colors hover:bg-surface-3/70"
        >
          <ListIcon size={20} aria-hidden className="shrink-0 text-ink-2" />
          {/* без flex-1: стрелка идёт сразу за названием главы, а не у названия сайта справа */}
          <span className="min-w-0 truncate text-[15px] font-medium text-ink">
            <span className="sr-only">Оглавление. Сейчас: </span>
            {ch.number > 0 && <span className="mr-1.5 font-normal text-ink-3">{ch.number}</span>}
            {ch.nav}
          </span>
          <CaretDownIcon size={16} aria-hidden className={cn('shrink-0 text-ink-3 transition-transform', open && 'rotate-180')} />
        </button>
        <span className="hidden shrink-0 text-sm text-ink-3 md:block">Инференс LLM наглядно</span>
        <ThemeSwitch className="shrink-0" />
      </div>
      <ReadingProgress className="absolute inset-x-0 -bottom-px" />
      {open && (
        <div
          ref={panel}
          id={panelId}
          className="absolute inset-x-0 top-full max-h-[calc(100dvh-3.25rem)] overflow-y-auto overscroll-contain border-b border-line bg-bg px-2 pt-3 pb-6 shadow-pop md:px-6"
        >
          <div className="px-2 pb-3">
            <a href="#hero" onClick={onChapterClick('hero', () => setOpen(false))} className="block text-[15px] leading-tight font-semibold text-ink">
              Инференс LLM наглядно
            </a>
            <p className="mt-0.5 text-xs text-ink-3">Путь запроса от Enter до последнего токена</p>
          </div>
          <div className="px-2 pb-2 md:max-w-sm">
            <PromptChip onDone={() => setOpen(false)} />
          </div>
          <TocList columns onNavigate={() => setOpen(false)} />
        </div>
      )}
    </header>
  )
}
