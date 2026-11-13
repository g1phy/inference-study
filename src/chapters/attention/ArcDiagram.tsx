import { useCallback, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { cn, fmt, ScrollHint, seqColor, useScrollEdges } from '@/ui'
import { arcGeom, labelWidth, LOOP_DX, loopLabelOnRight, pickLabels } from './arcLayout'
import { useRoving } from './bits'

/** Ширина затухания у края прокрутки (класс scroll-fade в index.css), px. */
const EDGE_FADE = 28

export interface ArcDiagramProps {
  count: number
  /** Содержимое чипа (слово или токен). */
  renderItem: (i: number, focus: boolean) => ReactNode
  /** Подпись элемента для aria. */
  itemLabel: (i: number) => string
  /** Веса внимания от focus к каждому элементу (0 у невидимых). */
  weights: readonly number[]
  focus: number
  onFocus?: (i: number) => void
  onHover?: (i: number | null) => void
  /** Сколько самых толстых дуг подписать процентами (подпись, которая налезла бы на другую, пропускается). */
  labelTop?: number
  /** Высота области дуг, px. */
  height?: number
  /** Строка, которая меняется вместе с содержимым чипов (перемерить центры). */
  layoutKey?: string
  /** Расстояние между чипами, px. */
  gap?: number
  /** Выровнять строку по центру, если она уже контейнера. */
  center?: boolean
  /**
   * Какой элемент держать в поле зрения, когда строка шире контейнера: прокрутка подводит его
   * к правому краю (внимание смотрит влево, так видно, на кого он смотрит). Наведение не прокручивает.
   */
  scrollTo?: number
  /** Сколько места нужно строке целиком, px: родитель решает, рисовать ли дуги. */
  onMeasure?: (needWidth: number) => void
  /** Подпись ряда кнопок для скринридера. */
  groupLabel?: string
  className?: string
}

/**
 * Дуги внимания над строкой слов или токенов: от выбранного элемента к каждому предыдущему,
 * толщина и насыщенность дуги равны весу. Центры чипов меряются после раскладки, поэтому
 * чипы могут быть любой ширины. Если строка шире контейнера, она прокручивается вместе с дугами.
 * В ряд чипов попадают одним Tab, дальше стрелки (roving tabindex).
 */
export function ArcDiagram({
  count,
  renderItem,
  itemLabel,
  weights,
  focus,
  onFocus,
  onHover,
  labelTop = 3,
  height = 150,
  layoutKey = '',
  gap = 6,
  center = false,
  scrollTo,
  onMeasure,
  groupLabel = 'Выберите элемент: чьё внимание показать',
  className,
}: ArcDiagramProps) {
  const boxRef = useRef<HTMLDivElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const rowRef = useRef<HTMLDivElement | null>(null)
  const chipRefs = useRef<Array<HTMLElement | null>>([])
  const [geom, setGeom] = useState<{ xs: number[]; width: number }>({ xs: [], width: 0 })
  const reduce = useReducedMotion() ?? false
  const roving = useRoving(count, scrollTo ?? focus)
  const measureCb = useRef(onMeasure)
  measureCb.current = onMeasure
  const lastNeed = useRef(-1)
  const scrolled = useRef(false)
  const [edgeRef, edges] = useScrollEdges<HTMLDivElement>(layoutKey)
  const interactive = Boolean(onFocus)
  const { container } = roving
  const setRow = useCallback(
    (el: HTMLDivElement | null) => {
      rowRef.current = el
      container(interactive ? el : null)
    },
    [container, interactive],
  )
  const setBox = useCallback(
    (el: HTMLDivElement | null) => {
      boxRef.current = el
      edgeRef(el)
    },
    [edgeRef],
  )

  useLayoutEffect(() => {
    const wrap = wrapRef.current
    const row = rowRef.current
    if (!wrap || !row) return
    const measure = () => {
      const xs = chipRefs.current.slice(0, count).map((el) => (el ? row.offsetLeft + el.offsetLeft + el.offsetWidth / 2 : 0))
      const width = wrap.clientWidth
      setGeom((g) => (g.width === width && g.xs.length === xs.length && g.xs.every((x, i) => Math.abs(x - xs[i]) < 0.5) ? g : { xs, width }))
      // строка целиком плюс поля прокрутки (px-1 с обеих сторон)
      const need = Math.ceil(row.offsetWidth + 8)
      if (need !== lastNeed.current) {
        lastNeed.current = need
        measureCb.current?.(need)
      }
    }
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(wrap)
    ro.observe(row)
    return () => ro.disconnect()
  }, [count, layoutKey])

  const ready = geom.xs.length === count && geom.width > 0

  // строка шире контейнера: подвести элемент scrollTo к правому краю (без наведения, только по выбору)
  useLayoutEffect(() => {
    const box = boxRef.current
    const row = rowRef.current
    const el = scrollTo === undefined ? null : chipRefs.current[scrollTo]
    if (!box || !row || !el || !ready) return
    const max = box.scrollWidth - box.clientWidth
    if (max <= 1) return
    const right = row.offsetLeft + el.offsetLeft + el.offsetWidth
    // справа от чипа стоит подпись петли на себя: окно должно вместить и её, мимо затухания у края
    // (иначе «16 %» срезалось до «1» и читалось как «1 %»)
    const xf = right - el.offsetWidth / 2
    const loopRight = loopLabelOnRight(xf, geom.width) ? xf + LOOP_DX + labelWidth(fmt.pct(1)) / 2 + EDGE_FADE : 0
    const left = Math.max(0, Math.min(max, Math.max(right + 16, loopRight) - box.clientWidth))
    box.scrollTo({ left, behavior: reduce || !scrolled.current ? 'auto' : 'smooth' })
    scrolled.current = true
  }, [scrollTo, ready, geom.width, layoutKey, reduce])

  const H = height
  const max = Math.max(1e-9, ...weights.slice(0, focus + 1))
  const visible = weights.map((w, j) => ({ w, j })).filter(({ w, j }) => j <= focus && w > 0.004)
  const ranked = [...visible].sort((a, b) => b.w - a.w)
  const arcs = ready ? visible.map(({ w, j }) => arcGeom(geom.xs, focus, j, w, H, geom.width)) : []
  const labelled = pickLabels(arcs, labelTop, (w) => fmt.pct(w))
  const describe = ranked
    .slice(0, 4)
    .map(({ w, j }) => `${itemLabel(j)} ${fmt.pct(w)}`)
    .join(', ')

  return (
    <div className={cn('min-w-0', className)}>
      <div ref={setBox} className="scroll-fade max-w-full overflow-x-auto px-1 scrollbar-thin">
        <div ref={wrapRef} className="relative inline-block min-w-full align-top">
          <svg
            width={geom.width || 1}
            height={H}
            role="img"
            aria-label={`Дуги внимания от «${itemLabel(focus)}»: ${describe}`}
            className="block overflow-visible"
          >
            {arcs.map(({ w, j, d, lx, ly }) => {
              const strength = w / max
              const stroke = seqColor(0.45 + 0.55 * strength)
              const sw = 1.25 + 7 * w
              return (
                <g key={`${focus}-${j}-${layoutKey}`}>
                  <motion.path
                    d={d}
                    fill="none"
                    strokeLinecap="round"
                    style={{ stroke, strokeWidth: sw }}
                    initial={reduce ? false : { pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.55 + 0.45 * strength }}
                    transition={reduce ? { duration: 0 } : { duration: 0.5, ease: 'easeOut', delay: Math.min(0.3, 0.03 * Math.abs(focus - j)) }}
                  />
                  {labelled.has(j) && (
                    <motion.text
                      x={lx}
                      y={ly}
                      textAnchor="middle"
                      className="font-mono tabular-nums"
                      style={{
                        fill: 'var(--ink)',
                        fontSize: 11.5,
                        fontWeight: 600,
                        paintOrder: 'stroke',
                        stroke: 'var(--surface)',
                        strokeWidth: 4,
                        strokeLinejoin: 'round',
                      }}
                      initial={reduce ? false : { opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={reduce ? { duration: 0 } : { duration: 0.25, delay: 0.3 }}
                    >
                      {fmt.pct(w)}
                    </motion.text>
                  )}
                </g>
              )
            })}
          </svg>
          <div
            ref={setRow}
            role={onFocus ? 'toolbar' : undefined}
            aria-label={onFocus ? groupLabel : undefined}
            className={cn('relative flex w-max items-start pb-1', center && 'mx-auto')}
            style={{ gap }}
            onPointerLeave={(e) => e.pointerType === 'mouse' && onHover?.(null)}
          >
            {Array.from({ length: count }, (_, i) => {
              const isFocus = i === focus
              const content = renderItem(i, isFocus)
              if (!onFocus) {
                return (
                  <span
                    key={i}
                    ref={(el) => {
                      chipRefs.current[i] = el
                    }}
                    className="shrink-0"
                  >
                    {content}
                  </span>
                )
              }
              const r = roving.item(i)
              return (
                <button
                  key={i}
                  ref={(el) => {
                    chipRefs.current[i] = el
                    r.ref(el)
                  }}
                  type="button"
                  tabIndex={r.tabIndex}
                  aria-pressed={isFocus}
                  aria-label={itemLabel(i)}
                  onClick={() => onFocus(i)}
                  onPointerEnter={(e) => e.pointerType === 'mouse' && onHover?.(i)}
                  onFocus={(e) => {
                    r.onFocus()
                    if (e.currentTarget.matches(':focus-visible')) onHover?.(i)
                  }}
                  onBlur={() => onHover?.(null)}
                  className="shrink-0 cursor-pointer rounded-chip focus-visible:outline-offset-1"
                >
                  {content}
                </button>
              )
            })}
          </div>
        </div>
      </div>
      {edges !== 'none' && <ScrollHint>дуги шире экрана: прокрутите их вбок</ScrollHint>}
    </div>
  )
}
