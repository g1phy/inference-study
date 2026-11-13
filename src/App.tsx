import {
  Fragment,
  lazy,
  memo,
  startTransition,
  Suspense,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ComponentType,
  type CSSProperties,
  type LazyExoticComponent,
  type MouseEvent,
} from 'react'
import { flushSync } from 'react-dom'
import { MotionConfig } from 'motion/react'
import {
  CHAPTERS,
  chapterIndex,
  chaptersOfPart,
  getChapter,
  getPart,
  isChapterId,
  isFirstInPart,
  type ChapterId,
  type ChapterInfo,
  type ChapterModule,
  type PartInfo,
} from '@/content/chapters'
import { getTokenizerState, loadTokenizer, subscribeTokenizer } from '@/engine/tokenizer/loader'
import {
  activeChapterStore,
  handlePopState,
  initHistory,
  isPinning,
  isRestoredEntry,
  navigateTo,
  pinScrollTo,
  registerMountController,
  safeDecode,
  scrollToAnchor,
} from '@/state/nav'
import { resetSettings, SettingsScope } from '@/state/settings'
import { ChapterHeader } from '@/ui/Chapter'
import { ErrorBoundary } from '@/ui/ErrorBoundary'
import { loadKatex } from '@/ui/Formula'
import { NavSidebar, NavTopBar } from '@/ui/Nav'
import { loadGlossary } from '@/ui/Term'

const KitPage = lazy(() => import('@/ui/kit/KitPage'))

// словарь токенизатора нужен симулятору первым: забираем его сразу (index.html уже начал загрузку
// через <link rel=preload>), а не после загрузки чанка hero. hero смонтирован всегда, так что словарь
// понадобится в любом случае, даже при входе по ссылке на дальнюю главу.
if (typeof window !== 'undefined') loadTokenizer().catch(() => {})

/* ===========================================================================
 * Маршрут: '#kit' показывает витрину ui-kit вместо сайта.
 * ========================================================================= */

function subscribeHash(cb: () => void) {
  window.addEventListener('hashchange', cb)
  return () => window.removeEventListener('hashchange', cb)
}

function useHashRoute(): 'kit' | 'site' {
  return useSyncExternalStore(
    subscribeHash,
    () => (window.location.hash === '#kit' ? 'kit' : 'site'),
    () => 'site',
  )
}

export default function App() {
  const route = useHashRoute()
  return (
    <MotionConfig reducedMotion="user">
      {route === 'kit' ? (
        <Suspense fallback={null}>
          <KitPage />
        </Suspense>
      ) : (
        // граница на всю страницу: ошибка рендера вне глав не оставляет белый экран
        <ErrorBoundary label="страница" fallback={() => <PageFailed />}>
          <Site />
        </ErrorBoundary>
      )}
    </MotionConfig>
  )
}

function PageFailed() {
  return (
    <main className="page-container py-16">
      <div className="flow">
        <div role="alert" className="rounded-panel border border-line bg-surface-2 px-5 py-4 text-[15px] text-ink-2">
          <p className="font-semibold text-ink">Страница не загрузилась</p>
          <p className="mt-1">Что-то сломалось при отрисовке. Обновите страницу. Если не помогло, сбросьте свой запрос к примеру: он хранится в браузере.</p>
          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
            <button type="button" onClick={() => window.location.reload()} className="text-sm font-medium text-accent underline underline-offset-2">
              Обновить страницу
            </button>
            <button
              type="button"
              onClick={() => {
                resetSettings()
                window.location.reload()
              }}
              className="text-sm font-medium text-accent underline underline-offset-2"
            >
              Сбросить запрос и обновить
            </button>
          </div>
        </div>
      </div>
    </main>
  )
}

/* ===========================================================================
 * Ленивая загрузка глав.
 * Компонент главы после загрузки кэшируется; слот держит один и тот же тип
 * компонента, поэтому смонтированная глава не пересоздаётся.
 * ========================================================================= */

const loadedChapters = new Map<ChapterId, ComponentType>()
const lazyChapters = new Map<ChapterId, LazyExoticComponent<ComponentType>>()

/**
 * Загрузка чанка главы. Повторять import() бесполезно: Chromium запоминает неудачный динамический
 * импорт до перезагрузки документа и новый запрос не отправляет. Поэтому при сбое глава показывает
 * «Обновить страницу», а после выкладки новой версии (старые чанки отвечают 404) main.tsx
 * один раз перезагружает страницу сам (событие vite:preloadError).
 */
function loadChapter(ch: ChapterInfo): Promise<ChapterModule> {
  return ch.load().then((m) => {
    loadedChapters.set(ch.id, m.default)
    return m
  })
}

/** Ошибка загрузки чанка (сеть, устаревшая выкладка), а не ошибка рендера. */
function isChunkError(e: Error): boolean {
  return /dynamically imported module|importing a module script failed|error loading dynamically imported|chunkloaderror|loading chunk/i.test(`${e.name} ${e.message}`)
}

function chapterComponent(ch: ChapterInfo): ComponentType {
  const ready = loadedChapters.get(ch.id)
  if (ready) return ready
  let l = lazyChapters.get(ch.id)
  if (!l) {
    l = lazy(() => loadChapter(ch))
    lazyChapters.set(ch.id, l)
  }
  return l
}

/* ---------- кэш высот глав: резерв места до монтирования ---------- */

const HEIGHTS_KEY = 'inference-study:slot-heights:v1'
const DEFAULT_HEIGHT = 1800
/** estimatedHeight в реестре измерен на десктопе (1440 px); в узкой колонке главы выше (замер QA: мобильный ≈ 1,7×). */
const ESTIMATE_SCALE = { m: 1.7, t: 1.25, d: 1 } as const

function widthBucket(): 'm' | 't' | 'd' {
  const w = window.innerWidth
  return w < 768 ? 'm' : w < 1200 ? 't' : 'd'
}

const heightCache: Record<string, number> = (() => {
  try {
    const raw = window.localStorage.getItem(HEIGHTS_KEY)
    const parsed = raw ? (JSON.parse(raw) as unknown) : null
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, number>) : {}
  } catch {
    return {}
  }
})()
let heightTimer: number | undefined

function cachedHeight(id: ChapterId): number | undefined {
  const v = heightCache[`${id}@${widthBucket()}`]
  return typeof v === 'number' && v > 200 && v < 40_000 ? v : undefined
}

/** Ожидаемая высота главы: замер с прошлого визита или оценка из реестра с поправкой на ширину. */
function estimateHeight(chapter: ChapterInfo): number {
  return (
    cachedHeight(chapter.id) ?? (chapter.estimatedHeight ? Math.round(chapter.estimatedHeight * ESTIMATE_SCALE[widthBucket()]) : DEFAULT_HEIGHT)
  )
}

function rememberHeight(id: ChapterId, h: number) {
  const key = `${id}@${widthBucket()}`
  if (Math.abs((heightCache[key] ?? 0) - h) < 24) return
  heightCache[key] = Math.round(h)
  window.clearTimeout(heightTimer)
  heightTimer = window.setTimeout(() => {
    try {
      window.localStorage.setItem(HEIGHTS_KEY, JSON.stringify(heightCache))
    } catch {
      /* не критично */
    }
  }, 1500)
}

/* ===========================================================================
 * Сайт: навигация + лента глав с прогрессивным монтированием.
 * ========================================================================= */

/** Глава по hash: '#attention' или вложенный якорь вида '#attention-figure-2'. */
function chapterForHash(raw: string): ChapterId | null {
  if (!raw) return null
  if (isChapterId(raw)) return raw
  let best: ChapterId | null = null
  for (const c of CHAPTERS) if (raw.startsWith(`${c.id}-`) && (!best || c.id.length > best.length)) best = c.id
  return best
}

function neighbors(id: ChapterId): ChapterId[] {
  const i = chapterIndex(id)
  return [CHAPTERS[i - 1]?.id, id, CHAPTERS[i + 1]?.id].filter(Boolean) as ChapterId[]
}

function Site() {
  const initialHash = useMemo(() => safeDecode(window.location.hash.slice(1)), [])
  const [mounted, setMounted] = useState<ReadonlySet<ChapterId>>(() => {
    const set = new Set<ChapterId>(['hero', 'request'])
    const target = chapterForHash(initialHash)
    if (target) neighbors(target).forEach((id) => set.add(id))
    return set
  })
  const mountedRef = useRef(mounted)

  const mount = useCallback((ids: ChapterId[], sync = false) => {
    const missing = ids.filter((id) => !mountedRef.current.has(id))
    if (missing.length === 0) return
    const next = new Set(mountedRef.current)
    missing.forEach((id) => next.add(id))
    mountedRef.current = next
    if (sync) flushSync(() => setMounted(next))
    // фоновое монтирование в переходе: React рендерит главу кусками по ~5 мс и уступает вводу
    // (набор текста, тап, прокрутка), а не замораживает страницу одним коммитом на полсекунды
    else startTransition(() => setMounted(next))
  }, [])

  /**
   * Сначала загрузить чанк главы, потом смонтировать уже готовый компонент: без Suspense-заглушки.
   * Иначе React придерживает показ загруженных глав, пока рядом появляются новые заглушки
   * (троттлинг fallback ~300 мс), и при прогрессивном монтировании главы «застревают».
   */
  const mountWhenLoaded = useCallback(
    (ids: ChapterId[]) => {
      for (const id of ids) {
        if (mountedRef.current.has(id)) continue
        const done = () => mount([id])
        loadChapter(getChapter(id)).then(done, done) // при ошибке монтируем: глава покажет ErrorBoundary
      }
    },
    [mount],
  )

  // navigateTo() из любого места монтирует цель синхронно перед прыжком
  useEffect(() => registerMountController((ids) => mount(ids, true)), [mount])

  // приземление по якорю при загрузке
  useLayoutEffect(() => {
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual'
    initHistory()
    if (!initialHash || initialHash === 'hero') {
      window.scrollTo({ top: 0, behavior: 'instant' }) // например, вернулись с витрины #kit
      return
    }
    const target = chapterForHash(initialHash)
    const el = document.getElementById(initialHash) ?? (target ? document.getElementById(target) : null)
    if (el) scrollToAnchor(el) // фигура в закрытом «Глубже»: блок раскроется, pinScrollTo докрутит
    if (target) activeChapterStore.set(target)
    pinScrollTo(initialHash, 2500, target && target !== initialHash ? { fallback: target } : {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // переходы по обычным ссылкам и кнопкам «назад/вперёд»
  useEffect(() => {
    // «Назад»/«Вперёд»: popstate приходит раньше hashchange и возвращает читателя к абзацу,
    // с которого он ушёл (место запомнено при уходе, nav.ts)
    const onPop = () => {
      handlePopState()
    }
    const onHash = () => {
      if (isRestoredEntry()) return // popstate уже вернул на место чтения, к началу главы не прыгаем
      const raw = safeDecode(window.location.hash.slice(1))
      // «назад» к началу страницы: адрес без якоря (pushState для hero пишет чистый путь)
      if (!raw || raw === 'hero') {
        navigateTo('hero', { history: 'none', focus: false })
        return
      }
      const target = chapterForHash(raw)
      if (!target) return
      if (raw === target) navigateTo(target, { history: 'none', focus: false })
      else {
        mount(neighbors(target), true)
        const el = document.getElementById(raw) ?? document.getElementById(target)
        if (el) scrollToAnchor(el)
        pinScrollTo(raw, undefined, { fallback: target })
      }
    }
    window.addEventListener('popstate', onPop)
    window.addEventListener('hashchange', onHash)
    return () => {
      window.removeEventListener('popstate', onPop)
      window.removeEventListener('hashchange', onHash)
    }
  }, [mount])

  // монтирование при приближении к экрану
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        const ids = entries.filter((e) => e.isIntersecting).map((e) => (e.target as HTMLElement).dataset.slot as ChapterId)
        if (ids.length) mountWhenLoaded(ids)
      },
      { rootMargin: '150% 0px 150% 0px' },
    )
    document.querySelectorAll<HTMLElement>('[data-slot]').forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [mountWhenLoaded])

  // остальное монтируем в простое: ближайшие к текущей главе сначала, по одной.
  // Старт только после загрузки словаря токенизатора (симулятору он нужен первым, а чанки глав,
  // KaTeX и шрифты иначе делят с ним сеть), но не позже чем через 8 с. Пока читатель печатает,
  // тапает или крутит страницу, следующий шаг откладывается.
  useEffect(() => {
    type Idle = (cb: () => void, opts?: { timeout: number }) => number
    const ric: Idle = window.requestIdleCallback ? (cb, o) => window.requestIdleCallback(cb, o) : (cb) => window.setTimeout(cb, 200)
    const cic = window.cancelIdleCallback ? (h: number) => window.cancelIdleCallback(h) : (h: number) => window.clearTimeout(h)
    let idleId = 0
    let timeoutId = 0
    let capId = 0
    let cancelled = false
    let started = false
    let unsubTokenizer = () => {}

    let lastInput = 0
    const onInput = () => {
      lastInput = performance.now()
    }
    const inputOpts: AddEventListenerOptions = { passive: true, capture: true }
    const inputEvents = ['keydown', 'pointerdown', 'wheel', 'touchstart'] as const
    inputEvents.forEach((e) => window.addEventListener(e, onInput, inputOpts))

    const pickNext = (): ChapterId | null => {
      const a = chapterIndex(activeChapterStore.get())
      let best: ChapterId | null = null
      let bestD = Infinity
      CHAPTERS.forEach((c, i) => {
        if (mountedRef.current.has(c.id)) return
        const d = Math.abs(i - a) - (i > a ? 0.5 : 0)
        if (d < bestD) (bestD = d), (best = c.id)
      })
      return best
    }
    const schedule = (delay: number) => {
      window.clearTimeout(timeoutId)
      timeoutId = window.setTimeout(() => (idleId = ric(step, { timeout: 15_000 })), delay)
    }
    const step = async () => {
      if (cancelled) return
      const quiet = performance.now() - lastInput
      if (quiet < 2000) {
        schedule(2000 - quiet + 50)
        return
      }
      const next = pickNext()
      if (!next) return
      try {
        await loadChapter(getChapter(next))
      } catch {
        /* смонтируем всё равно: ErrorBoundary покажет ошибку */
      }
      if (cancelled) return
      mount([next])
      schedule(150)
    }
    const start = () => {
      if (started || cancelled) return
      started = true
      unsubTokenizer()
      window.clearTimeout(capId)
      schedule(600)
      // формулы и глоссарий тоже в простое и после словаря
      ric(() => {
        if (cancelled) return
        loadKatex().catch(() => {})
        loadGlossary().catch(() => {})
      }, { timeout: 15_000 })
    }
    const settled = () => {
      const s = getTokenizerState().status
      return s === 'ready' || s === 'error'
    }
    if (settled()) start()
    else {
      unsubTokenizer = subscribeTokenizer(() => settled() && start())
      capId = window.setTimeout(start, 8000)
    }
    return () => {
      cancelled = true
      unsubTokenizer()
      window.clearTimeout(timeoutId)
      window.clearTimeout(capId)
      cic(idleId)
      inputEvents.forEach((e) => window.removeEventListener(e, onInput, inputOpts))
    }
  }, [mount])

  // scrollspy: глава, пересекающая линию на 30% высоты окна
  useEffect(() => {
    const visible = new Set<string>()
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          const id = (e.target as HTMLElement).dataset.slot!
          if (e.isIntersecting) visible.add(id)
          else visible.delete(id)
        }
        const first = CHAPTERS.find((c) => visible.has(c.id))
        if (first) activeChapterStore.set(first.id)
      },
      { rootMargin: '-30% 0px -69% 0px' },
    )
    document.querySelectorAll<HTMLElement>('[data-slot]').forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [])

  // адресная строка следует за активной главой (без новых записей в истории)
  useEffect(() => {
    let timer: number | undefined
    const unsub = activeChapterStore.subscribe(() => {
      window.clearTimeout(timer)
      timer = window.setTimeout(() => {
        if (isPinning()) return
        const id = activeChapterStore.get()
        const current = safeDecode(window.location.hash.slice(1))
        if (chapterForHash(current) === id) return
        const url = id === 'hero' ? window.location.pathname + window.location.search : `#${id}`
        // state записи не трогаем: в нём ключ и место чтения для «Назад» (nav.ts)
        window.history.replaceState(window.history.state, '', url)
      }, 400)
    })
    return () => {
      unsub()
      window.clearTimeout(timer)
    }
  }, [])

  // удержание позиции там, где браузер не умеет scroll anchoring (Safari)
  useEffect(() => {
    if (typeof CSS !== 'undefined' && CSS.supports?.('overflow-anchor: auto')) return
    const heights = new Map<Element, number>()
    const ro = new ResizeObserver((entries) => {
      let delta = 0
      for (const e of entries) {
        const rect = e.target.getBoundingClientRect()
        const prev = heights.get(e.target)
        heights.set(e.target, rect.height)
        if (prev === undefined || isPinning()) continue
        if (rect.top + prev <= 0) delta += rect.height - prev
      }
      if (delta) window.scrollBy(0, delta)
    })
    document.querySelectorAll<HTMLElement>('[data-slot]').forEach((el) => ro.observe(el))
    return () => ro.disconnect()
  }, [])

  return (
    <>
      <a
        href="#main"
        className="sr-only z-50 rounded-control bg-accent px-4 py-2 text-accent-ink focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
      >
        Перейти к содержанию
      </a>
      <NavSidebar />
      <NavTopBar />
      <main id="main" tabIndex={-1} className="outline-none desk:pl-[264px]">
        {CHAPTERS.map((ch) => (
          <Fragment key={ch.id}>
            {isFirstInPart(ch.id) && ch.part && <PartOpener part={getPart(ch.part)} />}
            <ChapterSlot chapter={ch} mounted={mounted.has(ch.id)} />
          </Fragment>
        ))}
        <SiteFooter />
      </main>
    </>
  )
}

/* ---------- слот главы ---------- */

const ChapterSlot = memo(function ChapterSlot({ chapter, mounted }: { chapter: ChapterInfo; mounted: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const hero = chapter.id === 'hero'
  const Comp = useMemo(() => (mounted ? chapterComponent(chapter) : null), [mounted, chapter])

  useEffect(() => {
    const el = ref.current
    if (!mounted || !el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => {
      if (!el.querySelector('[aria-busy="true"]')) rememberHeight(chapter.id, el.offsetHeight)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [mounted, chapter.id])

  // content-visibility: браузер не считает стили и раскладку глав за экраном (смена темы, набор текста,
  // фоновое монтирование), а резерв высоты держит прокрутку. hero всегда на виду, его не трогаем.
  const style = useMemo<CSSProperties | undefined>(
    () => (hero ? undefined : { contentVisibility: 'auto', containIntrinsicSize: `none auto ${estimateHeight(chapter)}px` }),
    [hero, chapter],
  )

  const placeholder = <ChapterPlaceholder chapter={chapter} />
  return (
    <div ref={ref} data-slot={chapter.id} style={style}>
      {Comp ? (
        <ErrorBoundary label={`глава ${chapter.id}`} fallback={(error, reset) => <ChapterFailed chapter={chapter} error={error} reset={reset} />}>
          {/* главы вне симулятора читают текст запроса с задержкой: печать не перерисовывает весь сайт на каждую букву */}
          <SettingsScope mode={hero ? 'live' : 'committed'}>
            <Suspense fallback={placeholder}>
              <Comp />
            </Suspense>
          </SettingsScope>
        </ErrorBoundary>
      ) : (
        placeholder
      )}
    </div>
  )
})

/** До монтирования: настоящая шапка главы из реестра и резерв высоты (якоря работают сразу). */
function ChapterPlaceholder({ chapter }: { chapter: ChapterInfo }) {
  const hero = chapter.id === 'hero'
  return (
    <section
      id={chapter.id}
      data-chapter={chapter.id}
      aria-busy="true"
      aria-labelledby={`${chapter.id}-title`}
      style={{ minHeight: estimateHeight(chapter) }}
      className={hero ? 'pt-10 pb-16 desk:pt-16 desk:pb-24' : 'py-14 desk:py-24'}
    >
      <div className="page-container">
        <div className="flow">
          <ChapterHeader info={chapter} hero={hero} />
        </div>
      </div>
    </section>
  )
}

function ChapterFailed({ chapter, error, reset }: { chapter: ChapterInfo; error: Error; reset: () => void }) {
  const chunk = isChunkError(error)
  const hero = chapter.id === 'hero'
  const linkBtn = 'text-sm font-medium text-accent underline underline-offset-2'
  return (
    <section id={chapter.id} data-chapter={chapter.id} aria-labelledby={`${chapter.id}-title`} className="py-14 desk:py-24">
      <div className="page-container">
        <div className="flow">
          <ChapterHeader info={chapter} hero={hero} />
          <div role="alert" className="rounded-panel border border-line bg-surface-2 px-5 py-4 text-[15px] text-ink-2">
            <p className="font-semibold text-ink">Глава не загрузилась</p>
            <p className="mt-1">
              {chunk
                ? 'Похоже, пропала сеть или сайт обновился. Остальные главы работают, а эта появится после обновления страницы.'
                : hero
                  ? 'Остальные главы работают. Если симулятор сломал какой-то запрос, сбросьте его к примеру; иначе обновите страницу.'
                  : 'Остальные главы работают. Обновите страницу.'}
            </p>
            {import.meta.env.DEV && <pre className="mt-3 overflow-x-auto font-mono text-xs whitespace-pre-wrap text-bad">{String(error.stack ?? error)}</pre>}
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
              {hero && !chunk && (
                <button
                  type="button"
                  onClick={() => {
                    resetSettings()
                    reset()
                  }}
                  className={linkBtn}
                >
                  Сбросить запрос к примеру
                </button>
              )}
              <button type="button" onClick={() => window.location.reload()} className={linkBtn}>
                Обновить страницу
              </button>
              {/* в разработке модуль мог исправиться горячей заменой: можно перерисовать без перезагрузки */}
              {import.meta.env.DEV && !chunk && (
                <button type="button" onClick={reset} className={linkBtn}>
                  Попробовать ещё раз
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

/* ---------- разделитель частей ---------- */

function PartOpener({ part }: { part: PartInfo }) {
  const chapters = chaptersOfPart(part.id)
  const go = (id: ChapterId) => (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    navigateTo(id)
  }
  return (
    <div id={`part-${part.number}`} className="page-container pt-10 desk:pt-16">
      <div className="flow">
        <div className="border-t border-line pt-6">
          <p className="text-[1.375rem] leading-tight font-[650] tracking-[-0.012em] text-ink desk:text-[1.5rem]">
            <span className="mr-2 font-normal text-ink-3">Часть {part.roman}.</span>
            {part.title}
          </p>
          <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-[14px]">
            {chapters.map((c) => (
              <li key={c.id}>
                <a href={`#${c.id}`} onClick={go(c.id)} className="text-ink-2 transition-colors hover:text-accent">
                  <span className="mr-1.5 text-ink-3">{c.number}</span>
                  {c.nav}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}

/* ---------- подвал ---------- */

function SiteFooter() {
  return (
    <footer className="page-container pt-6 pb-16">
      <div className="flow">
        <div className="grid gap-2 border-t border-line pt-6 text-sm text-ink-3">
          <p>
            По мотивам книги «Inference Engineering» (Philip Kiely, Baseten Books, 2026) и документации vLLM, SGLang и NVIDIA Dynamo.
            Ответы модели в симуляторе симулированы; токенизация, шаблоны и расчёты настоящие.
          </p>
          {/* витрина ui-kit нужна только разработчикам */}
          {import.meta.env.DEV && (
            <p>
              <a href="#kit" className="text-ink-2 underline decoration-line-strong underline-offset-2 hover:text-accent">
                Витрина компонентов
              </a>
            </p>
          )}
        </div>
      </div>
    </footer>
  )
}
