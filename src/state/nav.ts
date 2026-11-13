import { isChapterId, nextChapter, prevChapter, type ChapterId } from '@/content/chapters'
import { createStore, useStore } from './store'

/**
 * Навигация по длинной странице: активная глава (scrollspy), переход к главе
 * с гарантией монтирования, скролл к симулятору, возврат «Назад» к месту чтения.
 *
 * Главы монтируются прогрессивно (App.tsx). Чтобы переход по якорю приземлялся
 * точно, navigateTo сначала просит App смонтировать цель, потом скроллит и
 * «держит» позицию, пока выше догружаются главы (до первого действия пользователя).
 *
 * История: перед каждым прыжком (navigateTo, scrollToSimulator) место чтения запоминается
 * в текущей записи, а прыжок добавляет новую. «Назад» возвращает к абзацу, с которого ушли,
 * а не к началу главы (App слушает popstate и зовёт handlePopState).
 */

/* ---------- активная глава ---------- */

export const activeChapterStore = createStore<ChapterId>('hero')

export function useActiveChapter(): ChapterId {
  return useStore(activeChapterStore)
}

/* ---------- контроллер монтирования (регистрирует App) ---------- */

type MountController = (ids: ChapterId[]) => void
let mountController: MountController | null = null

export function registerMountController(fn: MountController): () => void {
  mountController = fn
  return () => {
    if (mountController === fn) mountController = null
  }
}

/** Синхронно смонтировать главы (если App зарегистрировал контроллер). */
export function ensureMounted(ids: ChapterId[]): void {
  mountController?.(ids)
}

/** Глава и её соседи: их монтируем перед прыжком, чтобы цель не сдвигалась. */
function withNeighbors(id: ChapterId): ChapterId[] {
  const ids: ChapterId[] = [id]
  const prev = prevChapter(id)
  const next = nextChapter(id)
  if (prev) ids.push(prev.id)
  if (next) ids.push(next.id)
  return ids
}

/* ---------- закрытые «Глубже» ---------- */

/**
 * Раскрыть закрытые блоки «Глубже» вокруг элемента. Закрытое содержимое лежит под hidden="until-found",
 * и scrollIntoView до него не докручивает. Браузер раскрывает такие блоки сам только при поиске по странице
 * и при переходе по якорю, который обрабатывает сам; у нас якорь появляется позже (глава грузится лениво),
 * поэтому шлём то же событие beforematch. Deeper открывается по нему, страница становится выше,
 * и следующий realign (ResizeObserver в pinScrollTo) докручивает до цели.
 * Возвращает самый внешний закрытый блок (его родитель виден) или null, если раскрывать было нечего.
 */
export function revealHidden(el: Element): Element | null {
  let outer: Element | null = null
  for (let n = el.closest('[hidden="until-found"]'); n; n = n.parentElement?.closest('[hidden="until-found"]') ?? null) {
    n.dispatchEvent(new Event('beforematch', { bubbles: true }))
    outer = n
  }
  return outer
}

/**
 * Прокрутить к якорю (верх окна, с учётом scroll-margin). Якорь в закрытом «Глубже»: блок раскрывается,
 * а пока React его открывает, встаём на сам блок. Иначе глава вдали осталась бы за экраном, а там
 * content-visibility не пересчитывает её высоту, ResizeObserver молчит и до фигуры никто не докрутит.
 */
export function scrollToAnchor(el: Element): void {
  const hidden = revealHidden(el)
  ;(hidden?.parentElement ?? el).scrollIntoView({ block: 'start', behavior: 'instant' })
}

/* ---------- удержание позиции после прыжка ---------- */

let unpin: (() => void) | null = null

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

/**
 * Повторять realign при каждом изменении высоты страницы (догружаются главы, шрифты, раскрылся «Глубже»).
 * Отпускает при любом действии пользователя или по таймауту.
 */
function hold(realign: () => void, ms: number): void {
  unpin?.()
  const ro = new ResizeObserver(realign)
  ro.observe(document.body)
  const opts: AddEventListenerOptions = { passive: true, capture: true }
  const events = ['wheel', 'touchstart', 'keydown', 'pointerdown'] as const
  let timer: number | undefined
  const arm = () => {
    window.clearTimeout(timer)
    timer = window.setTimeout(stop, ms)
  }
  // ссылку открыли в фоновой вкладке: там нет кадров, ResizeObserver молчит, а таймер истёк бы раньше,
  // чем вкладку покажут. Держим позицию до показа вкладки и только потом запускаем отсчёт.
  const onVisibility = () => {
    if (document.visibilityState !== 'visible') return
    realign()
    arm()
  }
  const stop = () => {
    ro.disconnect()
    events.forEach((e) => window.removeEventListener(e, stop, opts))
    document.removeEventListener('visibilitychange', onVisibility)
    window.clearTimeout(timer)
    if (unpin === stop) unpin = null
  }
  events.forEach((e) => window.addEventListener(e, stop, opts))
  document.addEventListener('visibilitychange', onVisibility)
  if (document.visibilityState !== 'hidden') arm()
  unpin = stop
}

/**
 * Держать элемент с данным id у верхнего края, пока меняется высота страницы
 * (догружаются главы, шрифты). Отпускает при любом действии пользователя или по таймауту.
 * Цель внутри закрытого «Глубже» сначала раскрывается.
 */
export function pinScrollTo(id: string, ms = 1800, pin: { refocus?: boolean; fallback?: string } = {}): void {
  if (typeof window === 'undefined') return
  hold(() => {
    // fallback: якорь внутри главы (#attention-figure-heads) появится только после загрузки главы,
    // а до того держим позицию на самой главе
    const el = document.getElementById(id) ?? (pin.fallback ? document.getElementById(pin.fallback) : null)
    if (el) scrollToAnchor(el)
    // глава догрузилась и заменила заглушку вместе с заголовком, на котором стоял фокус: фокус ушёл
    // в <body>, возвращаем его на новый заголовок (клавиатура и скринридер не теряют место)
    if (pin.refocus && (document.activeElement === null || document.activeElement === document.body)) focusTarget(id)
  }, ms)
}

export function isPinning(): boolean {
  return unpin !== null
}

/* ---------- история: «Назад» возвращает к месту чтения ---------- */

/** Место чтения: глава под верхним краем окна и смещение от её начала (y, если главы нет). */
export interface ReadingPosition {
  y: number
  chapter: ChapterId | null
  offset: number
}

interface EntryMemo {
  pos: ReadingPosition
  /** Элемент в фокусе при уходе (обычно ссылка, по которой ушли): Tab после возврата продолжит с него. */
  focus: Element | null
  /** Элемент на линии чтения и его отступ от верха окна: точнее главы, если в ней выше догрузились формулы. */
  anchor: { el: Element; top: number } | null
}

/** Место и фокус на каждой покинутой записи истории. В памяти: ссылку на элемент в history.state не положить. */
const memos = new Map<string, EntryMemo>()
/** Ключ текущей записи истории (history.state.key). */
let currentKey: string | null = null
/** Запись, на которой popstate уже вернул читателя на место: hashchange следом её не трогает. */
let restoredKey: string | null = null

function historyState(): Record<string, unknown> {
  const s: unknown = window.history.state
  return s && typeof s === 'object' ? (s as Record<string, unknown>) : {}
}

function newKey(): string {
  return Math.random().toString(36).slice(2, 10)
}

function isReadingPosition(v: unknown): v is ReadingPosition {
  if (!v || typeof v !== 'object') return false
  const p = v as Record<string, unknown>
  return (
    Number.isFinite(p.y) && Number.isFinite(p.offset) && (p.chapter === null || (typeof p.chapter === 'string' && isChapterId(p.chapter)))
  )
}

/** Где сейчас читатель. Слоты глав (data-slot, App) идут по порядку: берём последний, чей верх выше края окна. */
function readingPosition(): ReadingPosition {
  const y = Math.round(window.scrollY)
  let chapter: ChapterId | null = null
  let offset = y
  for (const slot of document.querySelectorAll<HTMLElement>('[data-slot]')) {
    const top = slot.getBoundingClientRect().top
    if (top > 0.5) break
    const id = slot.dataset.slot
    if (id && isChapterId(id)) {
      chapter = id
      offset = Math.round(-top)
    }
  }
  return { y, chapter, offset }
}

/**
 * Элемент главы под линией чтения (30% высоты окна, как у scrollspy) посередине колонки текста.
 * elementsFromPoint, а не elementFromPoint: сверху может лежать открытое меню навигации (телефон).
 */
function readingAnchor(): EntryMemo['anchor'] {
  const main = document.getElementById('main')
  if (!main) return null
  const r = main.getBoundingClientRect()
  const pad = parseFloat(getComputedStyle(main).paddingLeft) || 0 // слева на десктопе оглавление
  const el = document.elementsFromPoint(r.left + pad + (r.width - pad) / 2, window.innerHeight * 0.3).find((e) => e.closest('[data-slot]'))
  return el ? { el, top: el.getBoundingClientRect().top } : null
}

function snapshot(): EntryMemo {
  return { pos: readingPosition(), focus: document.activeElement, anchor: readingAnchor() }
}

/** Ключ текущей записи; запись без ключа (первая, переход по обычному якорю) получает его здесь. */
function ensureEntryKey(): string {
  const s = historyState()
  if (typeof s.key === 'string') return s.key
  const key = newKey()
  window.history.replaceState({ ...s, key }, '', window.location.href)
  return key
}

/** Запомнить место чтения в текущей записи перед уходом с неё. Копия в history.state переживает перезагрузку. */
function rememberPosition(): void {
  const memo = snapshot()
  const s = historyState()
  const key = typeof s.key === 'string' ? s.key : newKey()
  memos.set(key, memo)
  currentKey = key
  window.history.replaceState({ ...s, key, pos: memo.pos }, '', window.location.href)
}

function pushEntry(url: string, extra: Record<string, unknown> = {}): void {
  const key = newKey()
  window.history.pushState({ ...extra, key }, '', url)
  currentKey = key
}

/** Вернуть читателя на запомненное место и держать его, пока догружаются главы выше. */
function restorePosition(pos: ReadingPosition, memo: EntryMemo | undefined): void {
  if (pos.chapter) ensureMounted(withNeighbors(pos.chapter))
  const anchor = memo?.anchor
  const top = () => {
    const slot = pos.chapter ? document.querySelector<HTMLElement>(`[data-slot="${pos.chapter}"]`) : null
    const byChapter = slot ? slot.getBoundingClientRect().top + window.scrollY + pos.offset : pos.y
    // элемент с линии чтения точнее, если он на месте и не уехал далеко от расчёта по главе
    // (далеко: например, это была прилипшая панель фигуры, её место зависит от прокрутки)
    if (anchor && anchor.el.isConnected && !anchor.el.closest('[hidden]')) {
      const byAnchor = anchor.el.getBoundingClientRect().top + window.scrollY - anchor.top
      if (Math.abs(byAnchor - byChapter) < window.innerHeight / 2) return byAnchor
    }
    return byChapter
  }
  const apply = () => window.scrollTo({ top: top(), behavior: 'instant' })
  apply()
  hold(apply, 1800)
  // фокус туда, где он был (ссылка, по которой ушли): иначе он остался бы на заголовке далёкой главы,
  // и первый же Tab унёс бы читателя обратно
  const focus = memo?.focus
  if (focus instanceof HTMLElement && focus !== document.body && focus.isConnected) focus.focus({ preventScroll: true })
}

/** Глава, из которой прыгнули к симулятору (текущая запись истории создана scrollToSimulator), иначе null. */
const simulatorReturnStore = createStore<ChapterId | null>(null)

function simulatorReturnFrom(s: Record<string, unknown>): ChapterId | null {
  return s.sim === true && typeof s.from === 'string' && isChapterId(s.from) && s.from !== 'hero' ? s.from : null
}

/** Начало работы с историей (App при монтировании): у текущей записи появляется ключ. */
export function initHistory(): void {
  if (typeof window === 'undefined') return
  currentKey = ensureEntryKey()
  simulatorReturnStore.set(simulatorReturnFrom(historyState()))
}

/**
 * popstate (App): «Назад» и «Вперёд» внутри сайта, а также переход по обычному якорю.
 * Сначала запоминает место на покинутой записи (окно ещё не сдвинулось: прокрутку браузер сам
 * не восстанавливает, scrollRestoration = 'manual'), потом возвращает читателя туда, где он был
 * на новой записи. true: место восстановлено, переход к началу главы по якорю не нужен.
 */
export function handlePopState(): boolean {
  if (typeof window === 'undefined') return false
  const left = snapshot()
  const key = ensureEntryKey()
  // ключ совпал с новой записью: currentKey устарел (в dev после горячей замены модуля), чужое место не пишем
  if (currentKey && currentKey !== key) memos.set(currentKey, left)
  const s = historyState()
  currentKey = key
  simulatorReturnStore.set(simulatorReturnFrom(s))
  const memo = memos.get(key)
  const pos = memo?.pos ?? (isReadingPosition(s.pos) ? s.pos : null)
  let restored = true
  if (pos) restorePosition(pos, memo)
  else if (s.sim === true) scrollToSimulator({ history: 'none' })
  else restored = false
  restoredKey = restored ? key : null
  return restored
}

/** Вернул ли popstate читателя на место на текущей записи (тогда hashchange следом ничего не делает). */
export function isRestoredEntry(): boolean {
  if (typeof window === 'undefined' || restoredKey === null) return false
  return historyState().key === restoredKey
}

/**
 * Пришли к симулятору из главы (ссылка «Где это в симуляторе», чип «Ваш запрос»): id этой главы, иначе null.
 * Для кнопки «Вернуться к главе» рядом с симулятором; сам возврат: returnFromSimulator().
 */
export function useSimulatorReturn(): ChapterId | null {
  return useStore(simulatorReturnStore)
}

/** Вернуться из симулятора к месту чтения: то же, что кнопка «Назад» браузера. */
export function returnFromSimulator(): void {
  if (typeof window === 'undefined' || simulatorReturnStore.get() === null) return
  window.history.back()
}

/* ---------- переходы ---------- */

export interface NavigateOptions {
  /** Обновить адресную строку (по умолчанию pushState: место чтения запоминается, «Назад» вернёт к нему). */
  history?: 'push' | 'replace' | 'none'
  /** Перевести фокус на заголовок главы (по умолчанию true, для клавиатуры и скринридеров). */
  focus?: boolean
}

/**
 * Перейти к главе по id (или к любому элементу с id внутри смонтированной главы).
 * Сначала монтирует цель и соседей, затем мгновенно скроллит и удерживает позицию.
 */
export function navigateTo(id: string, opts: NavigateOptions = {}): void {
  if (typeof window === 'undefined') return
  const { history: mode = 'push', focus = true } = opts
  // место чтения до прыжка: «Назад» вернёт к этому абзацу, а не к началу главы
  if (mode === 'push') rememberPosition()
  if (isChapterId(id)) ensureMounted(withNeighbors(id))
  const el = document.getElementById(id)
  if (!el) {
    window.location.hash = id
    return
  }
  scrollToAnchor(el)
  if (id !== 'hero') pinScrollTo(id, undefined, { refocus: focus })
  if (isChapterId(id)) activeChapterStore.set(id)

  const url = id === 'hero' ? window.location.pathname + window.location.search : `#${id}`
  if (mode === 'push') {
    pushEntry(url)
    simulatorReturnStore.set(null)
  } else if (mode === 'replace') window.history.replaceState(window.history.state, '', url)

  if (focus) {
    focusTarget(id)
    // цель ещё в закрытом «Глубже», и фокус на неё не встал: снимаем его со ссылки, по которой пришли,
    // а pinScrollTo (refocus) поставит его на цель, когда блок раскроется
    const active = document.activeElement
    if (el.closest('[hidden]') && active instanceof HTMLElement && !el.contains(active)) active.blur()
  }
}

/** Фокус на заголовок главы (или на сам элемент с этим id), без прокрутки. */
function focusTarget(id: string): void {
  const el = document.getElementById(id)
  if (!el) return
  const heading = el.querySelector<HTMLElement>('[data-chapter-heading]') ?? el
  if (!heading.hasAttribute('tabindex')) heading.setAttribute('tabindex', '-1')
  heading.focus({ preventScroll: true })
}

/**
 * Скролл к симулятору (элемент id="simulator", если S1 его поставил, иначе к началу hero).
 * focusPrompt: перевести фокус в поле запроса (элемент id="prompt-input").
 * history: 'push' (по умолчанию) добавляет запись в историю, если симулятор дальше экрана от читателя:
 * «Назад» вернёт к месту чтения. 'none' только прокручивает.
 */
export function scrollToSimulator(opts: { focusPrompt?: boolean; history?: 'push' | 'none' } = {}): void {
  if (typeof window === 'undefined') return
  ensureMounted(['hero'])
  const target = document.getElementById('simulator') ?? document.getElementById('hero')
  if (!target) return
  const distance = Math.abs(target.getBoundingClientRect().top)
  const far = distance > window.innerHeight * 2.5
  if ((opts.history ?? 'push') === 'push' && distance > window.innerHeight) {
    const from = activeChapterStore.get()
    rememberPosition()
    pushEntry(window.location.pathname + window.location.search, { sim: true, from })
    simulatorReturnStore.set(from === 'hero' ? null : from)
  }
  target.scrollIntoView({ block: 'start', behavior: far || prefersReducedMotion() ? 'instant' : 'smooth' })
  if (opts.focusPrompt) {
    const input = document.getElementById('prompt-input')
    if (input instanceof HTMLElement) input.focus({ preventScroll: true })
  }
}

/**
 * decodeURIComponent без исключений: обрезанная ссылка вида «#attention%» (мессенджер отрезал хвост)
 * иначе бросает URIError прямо в рендере и оставляет пустую страницу.
 */
export function safeDecode(s: string): string {
  try {
    return decodeURIComponent(s)
  } catch {
    return s
  }
}

/** Разобрать hash адресной строки в id главы (или null). */
export function chapterFromHash(hash: string = typeof window !== 'undefined' ? window.location.hash : ''): ChapterId | null {
  const id = safeDecode(hash.replace(/^#/, ''))
  return id && isChapterId(id) ? id : null
}
