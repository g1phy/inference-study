import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react'
import type { KatexOptions } from 'katex'
import { cn } from './cn'
import { useElementSize, useFontsReady, useScrollEdges } from './hooks'

/**
 * Макросы для подсветки сущностей в формулах (подложка цвета сущности, текст остаётся --ink):
 *   \cq{Q}  \ck{K}  \cv{V}  \cprefill{…}  \cdecode{…}  \ctool{…}  \cmuted{…}
 */
export const TEX_MACROS: Record<string, string> = {
  '\\cq': '\\htmlClass{tex-q}{#1}',
  '\\ck': '\\htmlClass{tex-k}{#1}',
  '\\cv': '\\htmlClass{tex-v}{#1}',
  '\\cprefill': '\\htmlClass{tex-prefill}{#1}',
  '\\cdecode': '\\htmlClass{tex-decode}{#1}',
  '\\ctool': '\\htmlClass{tex-tool}{#1}',
  '\\cmuted': '\\htmlClass{tex-muted}{#1}',
}

/* ---------- KaTeX грузится отдельным чанком (тяжёлый) ----------
 * Заранее его подгружает App в простое, когда готов словарь токенизатора (симулятору сеть нужнее).
 * Формула на экране сама запускает загрузку, если KaTeX ещё нет. */

type Katex = { renderToString(tex: string, options?: KatexOptions): string }
let katex: Katex | null = null
let failed = false
let loading: Promise<Katex> | null = null
const listeners = new Set<() => void>()
const FAILED = 'failed' as const

/** Загрузить KaTeX (повторные вызовы возвращают тот же промис; после сбоя можно вызвать снова). */
export function loadKatex(): Promise<Katex> {
  if (!loading) {
    loading = import('katex').then(
      (m) => {
        katex = ((m as { default?: Katex }).default ?? (m as unknown as Katex)) as Katex
        failed = false
        listeners.forEach((l) => l())
        return katex
      },
      (e: unknown) => {
        // не держим отклонённый промис вечно: следующая попытка (после восстановления сети) пойдёт заново
        loading = null
        failed = true
        listeners.forEach((l) => l())
        throw e
      },
    )
  }
  return loading
}

function useKatex(): Katex | typeof FAILED | null {
  const k = useSyncExternalStore(
    (cb) => {
      listeners.add(cb)
      return () => listeners.delete(cb)
    },
    () => katex ?? (failed ? FAILED : null),
    () => null,
  )
  useEffect(() => {
    if (k === null) loadKatex().catch(() => {})
  }, [k])
  return k
}

/** Кэш отрисованных формул (LRU по порядку вставки): слайдеры порождают много разных формул, держим не больше CACHE_MAX. */
const cache = new Map<string, string>()
const CACHE_MAX = 500

/**
 * TeX → HTML-строка KaTeX (с кэшем). Пока KaTeX не загружен, возвращает null
 * (компонент Formula сам дождётся загрузки). Ошибка в формуле не роняет страницу.
 */
export function renderTex(tex: string, block = false, macros?: Record<string, string>): string | null {
  if (!katex) return null
  const key = `${block ? 'B' : 'I'}|${macros ? JSON.stringify(macros) : ''}|${tex}`
  const hit = cache.get(key)
  if (hit !== undefined) {
    cache.delete(key)
    cache.set(key, hit)
    return hit
  }
  const html = katex.renderToString(tex, {
    displayMode: block,
    throwOnError: false,
    errorColor: '#d03b3b',
    strict: 'ignore',
    output: 'htmlAndMathml',
    macros: { ...TEX_MACROS, ...macros },
    trust: (ctx) => ctx.command === '\\htmlClass' && /^tex-[a-z]+$/.test(String((ctx as { class?: string }).class ?? '')),
  })
  cache.set(key, html)
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value as string)
  return html
}

export interface FormulaProps {
  /** TeX без $…$: 'E = TTFT + ITL \\times (N - 1)'. */
  tex: string
  /** Блочная (отдельной строкой, по центру) или в строке текста. */
  block?: boolean
  /** Подпись под блочной формулой. */
  label?: ReactNode
  macros?: Record<string, string>
  className?: string
  /**
   * Запасной вариант блочной формулы для узкой колонки (телефон): та же формула в несколько строк
   * (\\begin{aligned}…). Показывается, только когда основная строка не влезает по ширине, иначе
   * читатель видел бы «… = 4,9 мс ⇒» и не видел итога за краем.
   */
  narrowTex?: string
}

/**
 * Формула через KaTeX (CSS подключён в main.tsx, JS догружается сам).
 *
 *   <Formula tex="\\mathrm{softmax}\\!\\left(\\frac{\\cq{Q}\\ck{K}^\\top}{\\sqrt{d_k}}\\right)\\cv{V}" block />
 *   Размер KV на токен: <Formula tex="2 \\cdot L \\cdot h_{kv} \\cdot d_{head} \\cdot b" />
 */
export function Formula({ tex, block = false, label, macros, className, narrowTex }: FormulaProps) {
  const k = useKatex()
  const html = useMemo(() => (k && k !== FAILED ? renderTex(tex, block, macros) : null), [k, tex, block, macros])
  const narrowHtml = useMemo(
    () => (block && narrowTex && k && k !== FAILED ? renderTex(narrowTex, true, macros) : null),
    [k, narrowTex, block, macros],
  )
  if (k === FAILED) return <FormulaFallback tex={tex} block={block} className={className} />
  if (!block) {
    if (html === null) {
      return <span aria-busy="true" className={cn('inline-block h-[1em] w-[3em] rounded-[3px] bg-surface-3 align-[-0.1em]', className)} />
    }
    return <span className={cn('formula-inline', className)} dangerouslySetInnerHTML={{ __html: html }} />
  }
  return <BlockFormula html={html} narrowHtml={narrowHtml} label={label} className={className} />
}

/**
 * Блочная формула. Длинная строка прокручивается внутри, а не распирает колонку; если она не влезает
 * (телефон), обрезанный край затухает: иначе читатель видит «… = 4,9 мс ⇒» и не знает, что дальше есть итог.
 */
function BlockFormula({
  html,
  narrowHtml = null,
  label,
  className,
}: {
  html: string | null
  narrowHtml?: string | null
  label?: ReactNode
  className?: string
}) {
  // ширина колонки и естественная ширина основной строки: узкий вариант, только если строка не влезает
  const [boxRef, box] = useElementSize<HTMLDivElement>()
  const measureRef = useRef<HTMLDivElement>(null)
  const [wideW, setWideW] = useState(0)
  const fonts = useFontsReady()
  useLayoutEffect(() => {
    // меряем только после настоящего замера колонки: глава за экраном не раскладывается ради формулы
    if (!narrowHtml || !measureRef.current || box.width < 1) return
    setWideW(measureRef.current.getBoundingClientRect().width)
  }, [html, narrowHtml, fonts, box.width])
  const narrow = narrowHtml !== null && box.width >= 1 && wideW > box.width - 2
  const shown = narrow ? narrowHtml : html
  const [ref] = useScrollEdges<HTMLDivElement>(shown)
  return (
    <div ref={boxRef} className={cn('formula block-gap relative min-w-0 max-w-full', className)}>
      {shown === null ? (
        <div aria-busy="true" className="formula-block" style={{ minHeight: '2.75em' }} />
      ) : (
        <div ref={ref} className="formula-block scroll-fade" dangerouslySetInnerHTML={{ __html: shown }} />
      )}
      {narrowHtml !== null && html !== null && (
        // невидимая копия основной строки: по ней узнаём, влезает ли она. Внешняя рамка 0 × 0 с overflow-hidden,
        // иначе копия шире экрана раздвигала бы страницу вбок на телефоне
        <div aria-hidden className="pointer-events-none invisible absolute top-0 left-0 h-0 w-0 overflow-hidden">
          <div ref={measureRef} className="w-max" dangerouslySetInnerHTML={{ __html: html }} />
        </div>
      )}
      {label && <div className="mt-2 text-center text-sm text-ink-2">{label}</div>}
    </div>
  )
}

/** KaTeX не загрузился (сеть): показываем исходник формулы, чтобы смысл не пропал. */
function FormulaFallback({ tex, block, className }: { tex: string; block: boolean; className?: string }) {
  if (!block) return <code className={cn(className) || undefined}>{tex}</code>
  return (
    <div className={cn('formula block-gap min-w-0 max-w-full', className)}>
      <code className="block overflow-x-auto font-mono text-[13px] whitespace-pre-wrap text-ink-2">{tex}</code>
      <div className="mt-1 text-xs text-ink-3">Формула не загрузилась, обновите страницу.</div>
    </div>
  )
}
