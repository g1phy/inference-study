import { useCallback, useEffect, useId, useRef, useState, useSyncExternalStore, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import { findChapter } from '@/content/chapters'
import type { GlossaryEntry } from '@/content/glossary'
import { categoryLabel } from '@/content/glossary-categories'
import { cn } from './cn'
import { FloatingLayer, TipCard, type RectLike } from './floating'

/* ---------- глоссарий грузится отдельным чанком ----------
 * Определения нужны только при наведении или тапе, а JSON на 237 терминов (~46 КБ gzip) иначе ехал
 * в критическом чанке ui вместе с симулятором. Заранее его подгружает App в простое, а первое
 * наведение на термин загружает его сразу. */

type GlossaryModule = typeof import('@/content/glossary')
let glossary: GlossaryModule | null = null
let glossaryLoading: Promise<GlossaryModule> | null = null
const glossaryListeners = new Set<() => void>()

/** Загрузить глоссарий (повторные вызовы возвращают тот же промис; после сбоя можно снова). */
export function loadGlossary(): Promise<GlossaryModule> {
  if (!glossaryLoading) {
    glossaryLoading = import('@/content/glossary').then(
      (m) => {
        glossary = m
        glossaryListeners.forEach((l) => l())
        return m
      },
      (e: unknown) => {
        glossaryLoading = null
        throw e
      },
    )
  }
  return glossaryLoading
}

function useGlossary(): GlossaryModule | null {
  return useSyncExternalStore(
    (cb) => {
      glossaryListeners.add(cb)
      return () => glossaryListeners.delete(cb)
    },
    () => glossary,
    () => null,
  )
}


const warned = new Set<string>()

export interface TermProps {
  /** Термин для поиска в глоссарии (term или alias, без учёта регистра). */
  t: string
  /** Видимый текст (по умолчанию t): <Term t="KV cache">кэша KV</Term>. */
  children?: ReactNode
  className?: string
}

/**
 * Термин с всплывающим определением из глоссария (src/content/glossary.json).
 * Оборачивайте термин при первом появлении в главе (DESIGN §7).
 * Открывается наведением, фокусом с клавиатуры или тапом; Escape закрывает.
 * Если термина нет в глоссарии: просто текст и предупреждение в консоли (только dev).
 *
 *   Здесь работает <Term t="prefill" />, а дальше <Term t="decode">декодирование</Term>.
 */
export function Term({ t, children, className }: TermProps) {
  const g = useGlossary()
  const entry: GlossaryEntry | undefined = g?.findTerm(t)
  const id = useId()
  const ref = useRef<HTMLSpanElement>(null)
  const openTimer = useRef<number | undefined>(undefined)
  const [anchor, setAnchor] = useState<RectLike | null>(null)
  const [pinned, setPinned] = useState(false)
  // открыть, когда глоссарий догрузится (первое наведение до загрузки)
  const [wantOpen, setWantOpen] = useState(false)

  const open = useCallback(() => {
    if (!glossary) {
      setWantOpen(true)
      loadGlossary().catch(() => setWantOpen(false))
      return
    }
    if (ref.current) setAnchor(ref.current.getBoundingClientRect())
  }, [])
  const close = useCallback(() => {
    window.clearTimeout(openTimer.current)
    setWantOpen(false)
    setAnchor(null)
    setPinned(false)
  }, [])

  useEffect(() => {
    if (!wantOpen || !g) return
    setWantOpen(false)
    if (ref.current) setAnchor(ref.current.getBoundingClientRect())
  }, [wantOpen, g])

  useEffect(() => {
    if (!anchor && !wantOpen) return
    const onKey = (e: globalThis.KeyboardEvent) => e.key === 'Escape' && close()
    const onDown = (e: globalThis.PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) close()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown, true)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown, true)
    }
  }, [anchor, wantOpen, close])

  useEffect(() => () => window.clearTimeout(openTimer.current), [])

  if (g && !entry) {
    if (import.meta.env.DEV && !warned.has(t)) {
      warned.add(t)
      console.warn(`[Term] нет в глоссарии: «${t}»`)
    }
    return <>{children ?? t}</>
  }

  const chapter = entry?.chapter ? findChapter(entry.chapter) : undefined
  // термин раскрыт в той главе, где его читают: ссылка «Подробно в главе …» не нужна
  const here = Boolean(chapter && ref.current?.closest('section[data-chapter]')?.id === chapter.id)

  const onPointerEnter = (e: PointerEvent) => {
    if (e.pointerType !== 'mouse' || pinned) return
    if (!glossary) loadGlossary().catch(() => {})
    window.clearTimeout(openTimer.current)
    openTimer.current = window.setTimeout(open, 180)
  }
  const onPointerLeave = (e: PointerEvent) => {
    if (e.pointerType !== 'mouse' || pinned) return
    close()
  }
  const toggle = () => {
    if (pinned) close()
    else {
      setPinned(true)
      open()
    }
  }
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      toggle()
    }
  }

  return (
    <span
      ref={ref}
      role="button"
      tabIndex={0}
      aria-expanded={anchor !== null}
      aria-describedby={anchor ? id : undefined}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
      onClick={toggle}
      onKeyDown={onKeyDown}
      onFocus={(e) => e.currentTarget.matches(':focus-visible') && open()}
      onBlur={() => !pinned && close()}
      className={cn(
        'cursor-help rounded-[3px] underline decoration-ink-3 decoration-dotted decoration-[1.5px] underline-offset-[0.22em]',
        'transition-colors hover:decoration-ink-2',
        anchor && 'bg-accent-soft decoration-accent',
        className,
      )}
    >
      {children ?? t}
      {anchor && entry && (
        <FloatingLayer anchor={anchor} id={id} role="tooltip" side="top" scopeEl={ref.current}>
          <TipCard wide className="grid gap-1.5 px-3.5 py-3 text-[14px] leading-[1.5]">
            <span className="flex items-baseline justify-between gap-3">
              <span className="font-semibold text-ink">{entry.term}</span>
              <span className="shrink-0 text-xs text-ink-3">{categoryLabel(entry.category)}</span>
            </span>
            <span className="block text-ink">{entry.ru}</span>
            {entry.why && (
              <span className="block text-ink-2">
                <span className="font-medium text-ink">Почему важно. </span>
                {entry.why}
              </span>
            )}
            {chapter && !here && (
              <span className="block text-ink-3">
                {/* mentionOnly: ни одна глава термин не разбирает, глава только ближайшая по теме */}
                {entry.mentionOnly ? 'Ближе всего к главе' : 'Подробно в главе'} {chapter.number} «{chapter.title}»
              </span>
            )}
          </TipCard>
        </FloatingLayer>
      )}
    </span>
  )
}
