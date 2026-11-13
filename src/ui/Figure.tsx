import { createContext, createElement, Fragment, isValidElement, useContext, useId, useState, type FocusEvent, type HTMLAttributes, type ReactNode } from 'react'
import { Badge } from './Badge'
import { cn } from './cn'
import { ErrorBoundary } from './ErrorBoundary'
import type { BlockWidth } from './Prose'

export interface FigureProps {
  title?: ReactNode
  /** narrow: как проза; wide (по умолчанию): до 1040px; full: вся ширина контента. */
  width?: BlockWidth
  /**
   * Выбор сценария или пресета (что показываем): строка сразу под заголовком, до тела фигуры.
   * Так у всех фигур одна привычка: «сценарий сверху, проигрывание снизу».
   */
  inputs?: ReactNode
  /** Плеер пошагового демо (<Player {...p.bind} />): всегда первая строка нижней панели на всю ширину. */
  player?: ReactNode
  /** Панель контролов (слайдеры, переключатели): настройка демо. С player встаёт под ним. */
  controls?: ReactNode
  /** Где панель контролов: снизу (по умолчанию), сверху или сбоку (на десктопе справа, на узких экранах снизу). */
  controlsPosition?: 'bottom' | 'top' | 'side'
  /**
   * Липкая нижняя панель: держится у нижнего края экрана, пока фигура на экране.
   * По умолчанию на телефоне липнет сама, если в нижней панели только Player (высокие пошаговые фигуры);
   * true: липнет на любой ширине; false: никогда.
   */
  stickyControls?: boolean
  /** Подпись под фигурой. Там, где есть интерактив, начинается с «Попробуйте: …». */
  caption?: ReactNode
  /** Откуда данные: одна строка мелким шрифтом под подписью («Модель условная: …»). Подпись остаётся про эксперимент. */
  source?: ReactNode
  /** Пометка о точности в шапке: «упрощённо», «иллюстрация», «условные числа» (нейтральный бейдж справа). */
  note?: ReactNode
  /** Элементы справа в шапке (переключатель вида, бейдж). */
  actions?: ReactNode
  id?: string
  children: ReactNode
  /** Внутренние поля тела (по умолчанию true). */
  padded?: boolean
  className?: string
  bodyClassName?: string
  /** Подпись для скринридеров, если нет title. */
  'aria-label'?: string
}

/** Фокус внутри фигуры. Вне фигуры true: живой регион вне фигуры ведёт себя как обычно. */
const FigureFocusContext = createContext(true)

/**
 * Живые регионы фигур озвучиваются, только пока читатель работает с фигурой (фокус внутри, в том числе после клика):
 * иначе скринридер читал бы тексты далёких фигур, пока читатель печатает запрос в симуляторе.
 *
 *   <p aria-live={useFigureLive()}>…</p>
 */
export function useFigureLive(): 'polite' | 'off' {
  return useContext(FigureFocusContext) ? 'polite' : 'off'
}

/** true, когда фокус внутри ближайшей фигуры (или компонент стоит не в фигуре). */
export function useFigureFocusWithin(): boolean {
  return useContext(FigureFocusContext)
}

/**
 * Живой регион внутри фигуры (тот же useFigureLive, но компонентом: хук нужно звать внутри <Figure>,
 * а текст часто рисует сам компонент фигуры).
 *
 *   <FigureLive as="p" className="text-sm">{status}</FigureLive>
 */
export function FigureLive({ as = 'div', children, ...rest }: { as?: 'div' | 'p' | 'span' | 'ol' | 'ul'; children?: ReactNode } & HTMLAttributes<HTMLElement>) {
  const live = useFigureLive()
  return createElement(as, { ...rest, 'aria-live': live }, children)
}

/**
 * Рамка интерактивной фигуры: шапка, выбор сценария, тело, плеер и панель контролов, подпись.
 * Тело фигуры является container-query контейнером: внутри работают @sm: @md: @lg: варианты Tailwind.
 * Фигура получает фокус по клику (tabIndex=-1), поэтому Player слушает клавиши, когда фокус внутри.
 * Ошибка рендера внутри фигуры не роняет главу: на месте фигуры появится сообщение.
 *
 *   <Figure title="Attention по шагам" width="wide" player={<Player {...p.bind} />}
 *     caption="Попробуйте: нажмите на ячейку, чтобы увидеть слагаемые скалярного произведения.">
 *     …
 *   </Figure>
 */
export function Figure({
  title,
  width = 'wide',
  inputs,
  player,
  controls,
  controlsPosition = 'bottom',
  stickyControls,
  caption,
  source,
  note,
  actions,
  id,
  children,
  padded = true,
  className,
  bodyClassName,
  'aria-label': ariaLabel,
}: FigureProps) {
  const titleId = useId()
  const [focusInside, setFocusInside] = useState(false)
  const onBlur = (e: FocusEvent<HTMLElement>) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFocusInside(false)
  }
  const hasHead = Boolean(title || actions || note)
  const side = controlsPosition === 'side' && Boolean(controls)
  const bottomControls = Boolean(controls) && controlsPosition === 'bottom'
  const hasBottom = Boolean(player) || bottomControls

  const controlsBox = (pos: 'top' | 'bottom' | 'side', content: ReactNode) => (
    <div
      className={cn(
        'fig-controls grid content-start gap-x-6 gap-y-4 px-4 py-4 sm:px-5',
        pos === 'side'
          ? 'border-t border-line @min-[760px]/fig:border-t-0 @min-[760px]/fig:border-l'
          : 'grid-cols-[repeat(auto-fit,minmax(min(200px,100%),1fr))]',
        pos === 'top' && 'border-b border-line',
        pos === 'bottom' && 'border-t border-line',
      )}
      data-sticky={pos === 'bottom' ? (stickyControls === true ? 'on' : stickyControls === false ? undefined : 'auto') : undefined}
    >
      {content}
    </div>
  )

  return (
    <figure
      id={id}
      data-figure
      tabIndex={-1}
      aria-labelledby={title ? titleId : undefined}
      aria-label={title ? undefined : ariaLabel}
      onFocus={() => setFocusInside(true)}
      onBlur={onBlur}
      className={cn(
        'figure min-w-0 outline-none',
        width === 'wide' && 'bleed-wide',
        width === 'full' && 'bleed-full',
        className,
      )}
    >
      <FigureFocusContext.Provider value={focusInside}>
        <ErrorBoundary label={typeof title === 'string' ? `фигура «${title}»` : 'фигура'} fallback={(error, reset) => <FigureFailed error={error} reset={reset} />}>
          <div className="@container/fig min-w-0 overflow-clip rounded-panel border border-line bg-surface">
            {hasHead && (
              <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-4 pt-4 sm:px-5">
                {title && (
                  <div id={titleId} className="min-w-0 text-[17px] leading-snug font-semibold text-ink">
                    {title}
                  </div>
                )}
                {(actions || note) && (
                  // длинный текст в actions (модель, GPU, формат) переносится, а не обрезается рамкой на телефоне
                  <div className="flex max-w-full min-w-0 flex-wrap items-center gap-2 sm:shrink-0">
                    {note && <Badge tone="neutral">{note}</Badge>}
                    {actions}
                  </div>
                )}
              </div>
            )}
            {inputs && <div className="flex flex-wrap items-end gap-x-5 gap-y-3 px-4 pt-3 sm:px-5">{inputs}</div>}
            {controls && controlsPosition === 'top' && controlsBox('top', controls)}
            <div className={cn(side && 'grid @min-[760px]/fig:grid-cols-[minmax(0,1fr)_minmax(220px,280px)]')}>
              <div
                className={cn(
                  '@container min-w-0',
                  padded && (hasHead || inputs ? 'px-4 pt-3 pb-4 sm:px-5 sm:pb-5' : 'p-4 sm:p-5'),
                  bodyClassName,
                )}
              >
                {children}
              </div>
              {side && controlsBox('side', controls)}
            </div>
            {hasBottom &&
              controlsBox(
                'bottom',
                <>
                  {player}
                  {bottomControls && controls}
                </>,
              )}
          </div>
        </ErrorBoundary>
      </FigureFocusContext.Provider>
      {caption && <FigureCaption>{caption}</FigureCaption>}
      {source && <p className="mt-1.5 max-w-[68ch] px-1 text-xs leading-normal text-ink-3">{source}</p>}
    </figure>
  )
}

function FigureFailed({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <div role="alert" className="rounded-panel border border-line bg-surface-2 px-5 py-4 text-[15px] text-ink-2">
      <p className="font-semibold text-ink">Эта фигура не загрузилась</p>
      <p className="mt-1">Остальная глава работает. Можно попробовать ещё раз или обновить страницу.</p>
      {import.meta.env.DEV && <pre className="mt-3 overflow-x-auto font-mono text-xs whitespace-pre-wrap text-bad">{String(error.stack ?? error)}</pre>}
      <button type="button" onClick={reset} className="mt-3 text-sm font-medium text-accent underline underline-offset-2">
        Попробовать ещё раз
      </button>
    </div>
  )
}

/** «Попробуйте:» в начале подписи выделяется весом автоматически (строка или фрагмент, который начинается с текста). */
function FigureCaption({ children }: { children: ReactNode }) {
  return <figcaption className="mt-3 max-w-[68ch] px-1 text-sm leading-normal text-ink-2">{emphasizeTry(children)}</figcaption>
}

function emphasizeTry(node: ReactNode): ReactNode {
  if (typeof node === 'string') {
    const m = /^(\s*)(Попробуйте:)([\s\S]*)$/.exec(node)
    if (!m) return node
    return (
      <>
        {m[1]}
        <span className="font-medium text-ink">{m[2]}</span>
        {m[3]}
      </>
    )
  }
  if (isValidElement(node) && node.type === Fragment) {
    const kids = (node.props as { children?: ReactNode }).children
    if (Array.isArray(kids) && kids.length > 0) {
      const first = emphasizeTry(kids[0] as ReactNode)
      // статические дети фрагмента: раскладываем аргументами, чтобы React не требовал key
      return first === kids[0] ? node : createElement(Fragment, null, first, ...(kids.slice(1) as ReactNode[]))
    }
    const one = emphasizeTry(kids)
    return one === kids ? node : one
  }
  return node
}
