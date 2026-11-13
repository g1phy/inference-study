import { useCallback, useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react'

export { useReducedMotion } from 'motion/react'
export { useTheme } from '@/state/theme'

/* ---------------------------------------------------------------------------
 * useInView: виден ли элемент (IntersectionObserver, без scroll-слушателей).
 *
 *   const [ref, inView] = useInView<HTMLDivElement>({ rootMargin: '200px' })
 *   <div ref={ref}>…</div>
 *
 * once: после первого появления больше не следить (для ленивого монтирования).
 * ------------------------------------------------------------------------- */
export interface InViewOptions {
  rootMargin?: string
  threshold?: number | number[]
  once?: boolean
  /** Значение до первого измерения (по умолчанию false). */
  initial?: boolean
}

export function useInView<T extends Element = HTMLElement>(
  opts: InViewOptions = {},
): [ref: (el: T | null) => void, inView: boolean] {
  const { rootMargin = '0px', threshold = 0, once = false, initial = false } = opts
  const [inView, setInView] = useState(initial)
  const [el, setEl] = useState<T | null>(null)
  const thresholdKey = Array.isArray(threshold) ? threshold.join(',') : String(threshold)
  const done = useRef(false)

  useEffect(() => {
    if (!el || (once && done.current)) return
    if (typeof IntersectionObserver === 'undefined') {
      setInView(true)
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries.some((e) => e.isIntersecting)
        setInView(visible)
        if (visible && once) {
          done.current = true
          io.disconnect()
        }
      },
      { rootMargin, threshold: thresholdKey.split(',').map(Number) },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [el, rootMargin, thresholdKey, once])

  return [setEl, inView]
}

/* ---------------------------------------------------------------------------
 * useMediaQuery('(min-width: 75rem)') → boolean
 * ------------------------------------------------------------------------- */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (cb: () => void) => {
      if (typeof window === 'undefined' || !window.matchMedia) return () => {}
      const mql = window.matchMedia(query)
      mql.addEventListener('change', cb)
      return () => mql.removeEventListener('change', cb)
    },
    [query],
  )
  const get = () => (typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(query).matches : false)
  return useSyncExternalStore(subscribe, get, () => false)
}

/** Готовые брейкпоинты DESIGN §5. */
export const MEDIA = {
  /** ≥ 768px: планшет и шире. */
  tablet: '(min-width: 48rem)',
  /** ≥ 1200px: десктоп с левой навигацией. */
  desk: '(min-width: 75rem)',
  /** Устройство с мышью (есть hover). */
  hover: '(hover: hover) and (pointer: fine)',
} as const

/* ---------------------------------------------------------------------------
 * useElementSize: размер элемента (ResizeObserver). Первое измерение
 * синхронное (до отрисовки), поэтому графики не мигают. Исключение: элемент в главе за экраном
 * (content-visibility пропускает её раскладку) или в закрытом «Глубже»: там размер приходит
 * от ResizeObserver, когда глава подъезжает к экрану, а до того width = 0.
 *
 *   const [ref, { width }] = useElementSize<HTMLDivElement>()
 * ------------------------------------------------------------------------- */
export interface Size {
  width: number
  height: number
}

export function useElementSize<T extends Element = HTMLDivElement>(): [ref: (el: T | null) => void, size: Size] {
  const [el, setEl] = useState<T | null>(null)
  const [size, setSize] = useState<Size>({ width: 0, height: 0 })

  useLayoutEffect(() => {
    if (!el) return
    // нулевую ширину после настоящей не принимаем: это скрытый на миг элемент (снимок страницы, display:none у предка),
    // а графики с проверкой width > 0 иначе на этот миг пустеют
    const measure = (w: number, h: number) =>
      setSize((s) =>
        (w < 1 && s.width >= 1) || (Math.abs(s.width - w) < 0.5 && Math.abs(s.height - h) < 0.5) ? s : { width: w, height: h },
      )
    // глава за экраном (content-visibility: auto пропускает её раскладку) или в закрытом «Глубже»:
    // синхронный замер заставил бы браузер разложить всю главу прямо при фоновом монтировании.
    // Ждём ResizeObserver: он сообщит размер, когда глава подъедет к экрану (или блок откроют).
    const check = (el as Element & { checkVisibility?: (o?: Record<string, boolean>) => boolean }).checkVisibility
    const skipped = typeof ResizeObserver !== 'undefined' && typeof check === 'function' && !check.call(el, { contentVisibilityAuto: true })
    if (!skipped) {
      const r = el.getBoundingClientRect()
      measure(r.width, r.height)
    }
    if (typeof ResizeObserver === 'undefined') return
    // та же рамка, что у getBoundingClientRect (border-box): иначе на элементе с padding или рамкой
    // первый замер и все следующие расходились, и график рисовался дважды с разной шириной
    const ro = new ResizeObserver((entries) => {
      const e = entries[0]
      if (!e) return
      const b = e.borderBoxSize?.[0]
      if (b) measure(b.inlineSize, b.blockSize)
      else measure(e.contentRect.width, e.contentRect.height)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [el])

  return [setEl, size]
}

/* ---------------------------------------------------------------------------
 * useScrollEdges: есть ли у горизонтально прокручиваемого блока скрытое содержимое
 * и с какой стороны. На мобильном полосы прокрутки не видны, поэтому широкой матрице
 * или формуле нужна подсказка: затухание у обрезанного края и строка «прокрутите вбок».
 *
 *   const [ref, edges] = useScrollEdges<HTMLDivElement>()
 *   <div ref={ref} className="scroll-fade overflow-x-auto">…</div>
 *   {edges !== 'none' && <ScrollHint />}
 *
 * edges: 'none' (всё видно), 'end' (скрыто справа), 'start' (слева), 'both'.
 * Хук сам ставит data-overflow на элемент: класс scroll-fade рисует по нему затухание.
 * contentKey: передайте, если содержимое меняется через innerHTML (KaTeX), чтобы следить за новым.
 * ------------------------------------------------------------------------- */
export type ScrollEdges = 'none' | 'start' | 'end' | 'both'

export function useScrollEdges<T extends HTMLElement = HTMLDivElement>(contentKey?: unknown): [ref: (el: T | null) => void, edges: ScrollEdges] {
  const [el, setEl] = useState<T | null>(null)
  const [edges, setEdges] = useState<ScrollEdges>('none')

  useLayoutEffect(() => {
    if (!el) return
    const update = () => {
      const max = el.scrollWidth - el.clientWidth
      let e: ScrollEdges = 'none'
      if (max > 1) {
        const left = el.scrollLeft > 1
        const right = el.scrollLeft < max - 1
        e = left && right ? 'both' : left ? 'start' : 'end'
      }
      if (e === 'none') el.removeAttribute('data-overflow')
      else el.setAttribute('data-overflow', e)
      setEdges((prev) => (prev === e ? prev : e))
    }
    update()
    el.addEventListener('scroll', update, { passive: true })
    if (typeof ResizeObserver === 'undefined') return () => el.removeEventListener('scroll', update)
    // размер самого блока и его содержимого (матрица выросла, формула сменилась)
    const ro = new ResizeObserver(update)
    ro.observe(el)
    for (const child of Array.from(el.children)) ro.observe(child)
    return () => {
      ro.disconnect()
      el.removeEventListener('scroll', update)
    }
  }, [el, contentKey])

  return [setEl, edges]
}

/* ---------------------------------------------------------------------------
 * useFontsReady: true, когда веб-шрифты загрузились (для точных измерений текста).
 * ------------------------------------------------------------------------- */
let fontsReady = typeof document === 'undefined' || !('fonts' in document) || document.fonts.status === 'loaded'
const fontListeners = new Set<() => void>()
if (!fontsReady && typeof document !== 'undefined') {
  document.fonts.ready.then(() => {
    fontsReady = true
    fontListeners.forEach((l) => l())
  })
}

export function useFontsReady(): boolean {
  return useSyncExternalStore(
    (cb) => {
      fontListeners.add(cb)
      return () => fontListeners.delete(cb)
    },
    () => fontsReady,
    () => true,
  )
}

/* ---------------------------------------------------------------------------
 * useLatest: ref с последним значением (для колбэков в эффектах без переподписки).
 * ------------------------------------------------------------------------- */
export function useLatest<T>(value: T) {
  const ref = useRef(value)
  useLayoutEffect(() => {
    ref.current = value
  })
  return ref
}

/* ---------------------------------------------------------------------------
 * measureText: ширина текста в px (canvas, с кэшем). Для подписей в SVG-графиках.
 * ------------------------------------------------------------------------- */
let ctx: CanvasRenderingContext2D | null = null
const widthCache = new Map<string, number>()

export function measureText(text: string, font = '12px "Onest Variable", sans-serif'): number {
  const key = `${font}|${text}`
  const hit = widthCache.get(key)
  if (hit !== undefined) return hit
  if (typeof document === 'undefined') return text.length * 7
  if (!ctx) ctx = document.createElement('canvas').getContext('2d')
  if (!ctx) return text.length * 7
  ctx.font = font
  const w = ctx.measureText(text).width
  if (fontsReady) widthCache.set(key, w)
  return w
}

/** Обрезать текст с многоточием под ширину. */
export function truncateText(text: string, maxWidth: number, font?: string): string {
  if (measureText(text, font) <= maxWidth) return text
  let lo = 0
  let hi = text.length
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (measureText(text.slice(0, mid) + '…', font) <= maxWidth) lo = mid
    else hi = mid - 1
  }
  return lo > 0 ? text.slice(0, lo) + '…' : '…'
}
