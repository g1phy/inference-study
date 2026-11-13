import { useCallback, useEffect, useRef } from 'react'
import { useReducedMotion } from 'motion/react'
import { cn, Formula } from '@/ui'
import { useRoving } from './bits'

type PieceId = 'x' | 'softOpen' | 'q' | 'dot1' | 'k' | 'div' | 'mask' | 'softClose' | 'dot2' | 'v' | 'arcs'

interface Piece {
  id: PieceId
  tex?: string
  text?: string
  /** На какой шаг ведёт клик. */
  step: number
  /** Подпись для скринридера. */
  label: string
}

const PIECES: Array<Piece | 'arrow'> = [
  { id: 'x', tex: 'X', step: 0, label: 'Признаки слов X' },
  'arrow',
  { id: 'softOpen', tex: '\\mathrm{softmax}\\big(', step: 7, label: 'Softmax по строкам' },
  { id: 'q', tex: '\\cq{Q}', step: 1, label: 'Запросы Q' },
  { id: 'dot1', tex: '\\cdot', step: 4, label: 'Оценки: Q умножить на K транспонированное' },
  { id: 'k', tex: '\\ck{K}^{\\top}', step: 2, label: 'Ключи K' },
  { id: 'div', tex: '/\\sqrt{d_k}', step: 5, label: 'Деление на корень из d_k' },
  { id: 'mask', tex: '+\\,M', step: 6, label: 'Causal mask M' },
  { id: 'softClose', tex: '\\big)', step: 7, label: 'Конец softmax' },
  { id: 'dot2', tex: '\\cdot', step: 8, label: 'Выход: веса умножить на V' },
  { id: 'v', tex: '\\cv{V}', step: 3, label: 'Значения V' },
  'arrow',
  { id: 'arcs', text: 'дуги', step: 9, label: 'Дуги внимания' },
]

const BUTTONS = PIECES.filter((p): p is Piece => p !== 'arrow')
/** Номер куска среди кнопок (для roving tabindex: стрелки пропускают значки «→»). */
const BUTTON_INDEX = new Map(BUTTONS.map((p, i) => [p.id, i] as const))

/** Что подсвечено на каждом шаге фигуры. */
const CURRENT: Record<number, PieceId[]> = {
  0: ['x'],
  1: ['q'],
  2: ['k'],
  3: ['v'],
  4: ['q', 'dot1', 'k'],
  5: ['div'],
  6: ['mask'],
  7: ['softOpen', 'softClose'],
  8: ['dot2', 'v'],
  9: ['arcs'],
}

/** С какого шага кусок формулы считается пройденным. */
const FIRST: Record<PieceId, number> = { x: 0, q: 1, k: 2, v: 3, dot1: 4, div: 5, mask: 6, softOpen: 7, softClose: 7, dot2: 8, arcs: 9 }

/**
 * Полоска «где мы»: формула attention, в которой подсвечена часть текущего шага.
 * Пройденные куски обычным цветом, будущие приглушены. Клик по куску ведёт на его шаг.
 * Текущий кусок залит акцентом (без кольца: кольцо остаётся у фокуса клавиатуры).
 * В полоску попадают одним Tab, дальше стрелки. На узком экране она переносится на вторую строку.
 */
export function FormulaStrip({ step, onStep }: { step: number; onStep: (step: number) => void }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const reduce = useReducedMotion()
  const current = new Set(CURRENT[step] ?? [])
  const firstCurrent = Math.max(
    0,
    BUTTONS.findIndex((p) => current.has(p.id)),
  )
  const roving = useRoving(BUTTONS.length, firstCurrent)
  const { container } = roving
  const setBox = useCallback(
    (el: HTMLDivElement | null) => {
      ref.current = el
      container(el)
    },
    [container],
  )

  // если полоска всё же прокручивается (очень узкий экран): держим текущий кусок в поле зрения
  useEffect(() => {
    const box = ref.current
    const el = box?.querySelector<HTMLElement>('[data-current="true"]')
    if (!box || !el || box.scrollWidth <= box.clientWidth) return
    const left = el.offsetLeft - box.clientWidth / 2 + el.offsetWidth / 2
    box.scrollTo({ left: Math.max(0, left), behavior: reduce ? 'auto' : 'smooth' })
  }, [step, reduce])

  return (
    <div
      ref={setBox}
      role="toolbar"
      aria-label="Где мы в формуле attention"
      className="-mx-1 flex max-w-full items-center gap-0.5 overflow-x-auto px-1 py-1 scrollbar-thin @max-[520px]:flex-wrap @max-[520px]:gap-y-1"
    >
      {PIECES.map((p, i) => {
        if (p === 'arrow') {
          return (
            <span key={`a${i}`} aria-hidden className="px-1 text-ink-3">
              →
            </span>
          )
        }
        const r = roving.item(BUTTON_INDEX.get(p.id) ?? 0)
        const isCurrent = current.has(p.id)
        const done = !isCurrent && FIRST[p.id] < step
        return (
          <button
            key={p.id}
            ref={r.ref}
            tabIndex={r.tabIndex}
            onFocus={r.onFocus}
            type="button"
            data-current={isCurrent}
            aria-current={isCurrent ? 'step' : undefined}
            aria-label={p.label}
            onClick={() => onStep(p.step)}
            className={cn(
              'shrink-0 rounded-[6px] px-1 py-0.5 text-[15px] leading-6 transition-[background-color,color] duration-200 focus-visible:outline-offset-1',
              isCurrent ? 'bg-accent-soft text-accent-hover' : done ? 'text-ink hover:bg-surface-3' : 'text-ink-3 hover:bg-surface-3',
            )}
          >
            {p.tex ? <Formula tex={p.tex} /> : <span className="px-0.5 text-[14px] font-medium">{p.text}</span>}
          </button>
        )
      })}
    </div>
  )
}
