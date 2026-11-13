import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { cn } from './cn'
import { Z } from './z'

/**
 * Плавающий слой для тултипов и поповеров: портал в <body>, позиция в координатах
 * документа (поэтому слой едет вместе со страницей без scroll-слушателей),
 * переворот на другую сторону и прижим к краям экрана.
 */

export type Side = 'top' | 'bottom' | 'left' | 'right'
export type Align = 'center' | 'start' | 'end'

export interface RectLike {
  left: number
  top: number
  width: number
  height: number
}

export interface PositionOptions {
  side?: Side
  align?: Align
  /** Отступ от якоря, px. */
  offset?: number
  /** Минимальный отступ от краёв окна, px. */
  padding?: number
}

const OPPOSITE: Record<Side, Side> = { top: 'bottom', bottom: 'top', left: 'right', right: 'left' }

/** Чистая функция: где поставить слой размера size возле якоря (координаты окна). */
export function computePosition(
  anchor: RectLike,
  size: { width: number; height: number },
  viewport: { width: number; height: number },
  { side = 'top', align = 'center', offset = 8, padding = 8 }: PositionOptions = {},
): { x: number; y: number; side: Side } {
  const fits = (s: Side) => {
    if (s === 'top') return anchor.top - offset - size.height >= padding
    if (s === 'bottom') return anchor.top + anchor.height + offset + size.height <= viewport.height - padding
    if (s === 'left') return anchor.left - offset - size.width >= padding
    return anchor.left + anchor.width + offset + size.width <= viewport.width - padding
  }
  let s = side
  if (!fits(s) && fits(OPPOSITE[s])) s = OPPOSITE[s]

  let x: number
  let y: number
  if (s === 'top' || s === 'bottom') {
    y = s === 'top' ? anchor.top - offset - size.height : anchor.top + anchor.height + offset
    x =
      align === 'start'
        ? anchor.left
        : align === 'end'
          ? anchor.left + anchor.width - size.width
          : anchor.left + anchor.width / 2 - size.width / 2
  } else {
    x = s === 'left' ? anchor.left - offset - size.width : anchor.left + anchor.width + offset
    y =
      align === 'start'
        ? anchor.top
        : align === 'end'
          ? anchor.top + anchor.height - size.height
          : anchor.top + anchor.height / 2 - size.height / 2
  }
  x = Math.min(Math.max(x, padding), Math.max(padding, viewport.width - size.width - padding))
  y = Math.min(Math.max(y, padding), Math.max(padding, viewport.height - size.height - padding))
  return { x, y, side: s }
}

/** Прямоугольник вокруг точки (например, курсора на графике). */
export function pointRect(x: number, y: number, pad = 0): RectLike {
  return { left: x - pad, top: y - pad, width: pad * 2, height: pad * 2 }
}

export interface FloatingLayerProps extends PositionOptions {
  /** Якорь в координатах окна (getBoundingClientRect) на момент показа. */
  anchor: RectLike | null
  children: ReactNode
  id?: string
  role?: string
  className?: string
  style?: CSSProperties
  /** Элемент, от которого наследуется тема (data-theme) для витрины. */
  scopeEl?: Element | null
  /** Пропускать ли указатель (интерактивный поповер). */
  interactive?: boolean
  onPointerEnter?: () => void
  onPointerLeave?: () => void
  /** Ref на DOM-узел слоя (для «клика снаружи»). */
  layerRef?: (el: HTMLDivElement | null) => void
}

export function FloatingLayer({
  anchor,
  children,
  id,
  role,
  className,
  style,
  scopeEl,
  interactive = false,
  side = 'top',
  align = 'center',
  offset = 8,
  padding = 8,
  onPointerEnter,
  onPointerLeave,
  layerRef,
}: FloatingLayerProps) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  const place = useCallback(() => {
    const el = ref.current
    if (!el || !anchor) return
    const vw = document.documentElement.clientWidth
    const vh = window.innerHeight
    const p = computePosition(anchor, { width: el.offsetWidth, height: el.offsetHeight }, { width: vw, height: vh }, {
      side,
      align,
      offset,
      padding,
    })
    setPos({ left: p.x + window.scrollX, top: p.y + window.scrollY })
  }, [anchor, side, align, offset, padding])

  useLayoutEffect(() => {
    place()
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(place)
    ro.observe(el)
    return () => ro.disconnect()
  }, [place])

  if (!anchor || typeof document === 'undefined') return null
  const theme = scopeEl?.closest('[data-theme]')?.getAttribute('data-theme') ?? undefined

  return createPortal(
    <div
      ref={(el) => {
        ref.current = el
        layerRef?.(el)
      }}
      id={id}
      role={role}
      data-theme={theme}
      className={cn('absolute', !interactive && 'pointer-events-none', className)}
      style={{
        left: pos?.left ?? 0,
        top: pos?.top ?? 0,
        zIndex: Z.popover,
        visibility: pos ? 'visible' : 'hidden',
        ...style,
      }}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
    >
      {children}
    </div>,
    document.body,
  )
}

/** Карточка тултипа: поверхность, рамка, тень поповера (DESIGN §4). */
export function TipCard({ children, className, wide = false }: { children: ReactNode; className?: string; wide?: boolean }) {
  return (
    <div
      className={cn(
        'animate-fade-in rounded-control border border-line bg-surface px-3 py-2 text-[13px] leading-snug text-ink shadow-pop',
        wide ? 'max-w-[360px]' : 'max-w-[280px]',
        className,
      )}
    >
      {children}
    </div>
  )
}

/**
 * Строка тултипа графика: значение крупно, подпись следом, слева короткий
 * штрих цвета серии (dataviz: «values lead, labels follow», «line keys, not boxes»).
 */
export function TipRow({ color, value, label, dashed }: { color?: string; value: ReactNode; label?: ReactNode; dashed?: boolean }) {
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      {color && (
        <span
          aria-hidden
          className="inline-block h-0 w-3 shrink-0 border-t-2"
          style={{ borderColor: color, borderTopStyle: dashed ? 'dashed' : 'solid' }}
        />
      )}
      <span className="font-semibold tabular-nums text-ink">{value}</span>
      {label && <span className="text-ink-2">{label}</span>}
    </div>
  )
}

export interface UseTipOptions extends PositionOptions {
  /**
   * Корневой элемент компонента: от него наследуется тема (data-theme), и тап внутри него
   * не закрывает тултип (компонент сам покажет новый).
   */
  scopeEl?: Element | null
  wide?: boolean
  /** Тултип закрыт тапом или кликом вне scopeEl: сбросьте здесь своё состояние наведения. */
  onOutside?: () => void
}

/**
 * Хук одного тултипа на компонент (графики, токены, матрицы).
 *
 *   const tip = useTip({ scopeEl: rootRef.current, onOutside: () => setHover(null) })
 *   onPointerEnter={(e) => tip.show(e.currentTarget.getBoundingClientRect(), <TipRow … />)}
 *   onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
 *   {tip.node}
 *
 * Тач: тап показывает тултип, а pointerleave после тапа его не закрывает (проверяйте pointerType).
 * Закрывает тап вне scopeEl (нужен scopeEl), Escape и уход фокуса компонент обрабатывает сам.
 */
export function useTip(opts: UseTipOptions = {}) {
  const [state, setState] = useState<{ anchor: RectLike; content: ReactNode; side?: Side } | null>(null)
  const optsRef = useRef(opts)
  optsRef.current = opts
  const show = useCallback((anchor: RectLike, content: ReactNode, side?: Side) => {
    // копия прямоугольника: DOMRect нельзя мутировать, а якорь должен быть стабилен
    setState({ anchor: { left: anchor.left, top: anchor.top, width: anchor.width, height: anchor.height }, content, side })
  }, [])
  const hide = useCallback(() => setState(null), [])
  const visible = state !== null

  useEffect(() => {
    if (!visible || !optsRef.current.scopeEl) return
    const onDown = (e: PointerEvent) => {
      const scope = optsRef.current.scopeEl
      if (scope && e.target instanceof Node && scope.contains(e.target)) return
      setState(null)
      optsRef.current.onOutside?.()
    }
    window.addEventListener('pointerdown', onDown, true)
    return () => window.removeEventListener('pointerdown', onDown, true)
  }, [visible])
  const node = state ? (
    <FloatingLayer
      anchor={state.anchor}
      role="tooltip"
      side={state.side ?? opts.side ?? 'top'}
      align={opts.align}
      offset={opts.offset}
      scopeEl={opts.scopeEl}
    >
      <TipCard wide={opts.wide}>{state.content}</TipCard>
    </FloatingLayer>
  ) : null
  return { show, hide, node, visible }
}
