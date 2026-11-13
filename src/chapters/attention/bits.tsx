import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { cn, cssVar, tint, type DataRole } from '@/ui'

/** Нижний индекс в подписях без роста межстрочного интервала: W<Sub>Q</Sub>. */
export function Sub({ children }: { children: ReactNode }) {
  return <span className="relative top-[0.28em] text-[0.74em] leading-none">{children}</span>
}

/** Квадратик цвета сущности рядом с текстом (сам текст цветом --ink). */
export function Swatch({ role, className, dashed }: { role: DataRole; className?: string; dashed?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn('inline-block h-2.5 w-2.5 shrink-0 rounded-[3px] align-[-0.05em]', dashed && 'border border-dashed', className)}
      style={dashed ? { borderColor: cssVar(role) } : { background: cssVar(role) }}
    />
  )
}

/** Высота шапки столбцов у матриц фигуры: все матрицы в одном ряду выравниваются по строкам. */
export const HEAD_H = 112

/** Вертикальная подпись столбца фиксированной высоты (читается снизу вверх). */
export function VLabel({ children, height = HEAD_H, title }: { children: ReactNode; height?: number; title?: string }) {
  return (
    <span className="flex items-end justify-center" style={{ height }} title={title}>
      <span className="rotate-180 whitespace-nowrap [writing-mode:vertical-rl]">{children}</span>
    </span>
  )
}

export type ChipTone = 'neutral' | 'prefill' | 'decode' | 'ghost'

/**
 * Чип слова (игрушечный пример идёт по словам, поэтому шрифт обычный, а не моноширинный).
 * Выбранное слово (focus) показано заливкой акцента, а не кольцом: кольцо остаётся только у фокуса клавиатуры.
 */
export function WordChip({
  children,
  focus = false,
  tone = 'neutral',
  marker,
  size = 'md',
  style,
}: {
  children: ReactNode
  focus?: boolean
  tone?: ChipTone
  /** Метка роли слева: запрос (q) или ключ (k). */
  marker?: DataRole
  size?: 'sm' | 'md'
  style?: CSSProperties
}) {
  const bg = focus
    ? 'var(--accent-soft)'
    : tone === 'prefill'
      ? tint('prefill', 16)
      : tone === 'decode'
        ? tint('decode', 18)
        : tone === 'ghost'
          ? 'transparent'
          : 'var(--surface-2)'
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-chip whitespace-nowrap transition-[background-color,color] duration-150',
        size === 'sm' ? 'h-6 px-1.5 text-[13px]' : 'h-7 px-2 text-[14px]',
        focus ? 'font-medium text-accent-hover shadow-[inset_0_0_0_1px_var(--accent)]' : 'text-ink',
        tone === 'ghost' && !focus && 'text-ink-3 shadow-[inset_0_0_0_1px_var(--line)] [border-style:dashed]',
      )}
      style={{ background: bg, ...style }}
    >
      {marker && <Swatch role={marker} />}
      {children}
    </span>
  )
}

/** Знак операции между матрицами (·, =, ÷√2, +). */
export function OpGlyph({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span aria-hidden className={cn('flex shrink-0 items-center justify-center px-1 text-[20px] leading-none text-ink-2 select-none', className)}>
      {children}
    </span>
  )
}

/**
 * Roving tabindex для ряда кнопок (слова, куски формулы, токены): в ряд попадают одним Tab,
 * дальше ← и → (Home, End) двигают фокус, Enter и пробел нажимают кнопку.
 * Клавиши ловит нативный слушатель на самом ряду: он срабатывает раньше слушателя плеера на фигуре
 * (React вешает свои обработчики на корень приложения, то есть позже), и preventDefault
 * не даёт плееру листать шаги стрелками, пока фокус в ряду.
 *
 *   const r = useRoving(words.length, selected)
 *   <div role="toolbar" ref={r.container}>{words.map((w, i) => <button {...r.item(i)}>…</button>)}</div>
 */
export function useRoving(count: number, selected: number) {
  const refs = useRef<Array<HTMLElement | null>>([])
  const [active, setActive] = useState(selected)
  useEffect(() => {
    setActive(selected)
  }, [selected])
  const cur = Math.max(0, Math.min(count - 1, active))
  const latest = useRef({ cur, count })
  latest.current = { cur, count }
  const detach = useRef<(() => void) | null>(null)

  const container = useCallback((el: HTMLElement | null) => {
    detach.current?.()
    detach.current = null
    if (!el) return
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.altKey || e.ctrlKey || e.metaKey) return
      const { cur: c, count: n } = latest.current
      const at = refs.current.indexOf(e.target as HTMLElement)
      const from = at >= 0 ? at : c
      let next = -1
      if (e.key === 'ArrowRight') next = Math.min(n - 1, from + 1)
      else if (e.key === 'ArrowLeft') next = Math.max(0, from - 1)
      else if (e.key === 'Home') next = 0
      else if (e.key === 'End') next = n - 1
      if (next < 0) return
      e.preventDefault()
      setActive(next)
      refs.current[next]?.focus()
    }
    el.addEventListener('keydown', onKey)
    detach.current = () => el.removeEventListener('keydown', onKey)
  }, [])

  const item = (i: number) => ({
    ref: (el: HTMLElement | null) => {
      refs.current[i] = el
    },
    tabIndex: i === cur ? 0 : -1,
    onFocus: () => setActive(i),
  })

  return { container, item }
}
