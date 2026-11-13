import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { useReducedMotion } from 'motion/react'
import { ArrowCounterClockwiseIcon, CaretLeftIcon, CaretRightIcon, PauseIcon, PlayIcon } from '@phosphor-icons/react'
import { cn } from './cn'
import { useFigureFocusWithin } from './Figure'
import { fmt } from './fmt'
import { Segmented } from './Segmented'

/* ===========================================================================
 * usePlayer: состояние пошагового демо.
 * ========================================================================= */

export interface UsePlayerOptions {
  /** Запустить само, когда плеер появится на экране (не при reduced motion). */
  autoplay?: boolean
  /** Длительность шага при скорости 1×, мс. Можно функцией от номера шага. */
  intervalMs?: number | ((step: number) => number)
  /** После последнего шага начать сначала. */
  loop?: boolean
  initialStep?: number
}

/** Пропсы для <Player {...player.bind} />. */
export interface PlayerBindProps {
  steps: number
  step: number
  onStep: (step: number) => void
  playing: boolean
  onPlayingChange: (playing: boolean) => void
  speed: number
  onSpeedChange: (speed: number) => void
  viewRef: (el: Element | null) => void
}

export interface PlayerState {
  steps: number
  step: number
  setStep: (step: number | ((prev: number) => number)) => void
  playing: boolean
  setPlaying: (playing: boolean) => void
  toggle: () => void
  next: () => void
  prev: () => void
  /** На первый шаг и стоп. */
  reset: () => void
  speed: number
  setSpeed: (speed: number) => void
  atStart: boolean
  atEnd: boolean
  bind: PlayerBindProps
}

/**
 * Состояние плеера: шаг, проигрывание, скорость. Таймер идёт только пока демо видно на экране
 * (Player сообщает о видимости через bind.viewRef): ушли со страницы, демо встало на паузу,
 * вернулись, продолжилось. При prefers-reduced-motion автоплей выключен.
 *
 *   const p = usePlayer(STEPS.length, { autoplay: true, intervalMs: 1400 })
 *   <Player {...p.bind} labels={STEPS.map((s) => s.label)} />
 *   {render(STEPS[p.step])}
 */
export function usePlayer(steps: number, opts: UsePlayerOptions = {}): PlayerState {
  const { autoplay = false, intervalMs = 1200, loop = false, initialStep = 0 } = opts
  const reduce = useReducedMotion() ?? false
  const total = Math.max(1, Math.floor(steps))
  const clamp = useCallback((s: number) => Math.max(0, Math.min(total - 1, Math.round(s))), [total])

  const [step, setStepState] = useState(() => clamp(initialStep))
  const [playing, setPlayingState] = useState(false)
  const [speed, setSpeed] = useState(1)

  const playingRef = useRef(playing)
  playingRef.current = playing
  const intervalRef = useRef(intervalMs)
  intervalRef.current = intervalMs
  const wantAutoplay = useRef(autoplay && !reduce)
  wantAutoplay.current = autoplay && !reduce
  const autoplayed = useRef(false)
  const autoPaused = useRef(false)
  const io = useRef<IntersectionObserver | null>(null)

  // число шагов уменьшилось: не выходим за край
  useEffect(() => {
    setStepState((s) => clamp(s))
  }, [clamp])

  const setStep = useCallback(
    (s: number | ((prev: number) => number)) => setStepState((prev) => clamp(typeof s === 'function' ? s(prev) : s)),
    [clamp],
  )

  const setPlaying = useCallback(
    (p: boolean) => {
      autoPaused.current = false
      if (p) setStepState((prev) => (prev >= total - 1 && !loop ? 0 : prev))
      setPlayingState(p)
    },
    [total, loop],
  )

  // тик
  useEffect(() => {
    if (!playing) return
    const iv = intervalRef.current
    const base = typeof iv === 'function' ? iv(step) : iv
    const t = window.setTimeout(() => {
      if (step >= total - 1) {
        if (loop) setStepState(0)
        else setPlayingState(false)
      } else setStepState(step + 1)
    }, Math.max(16, base / speed))
    return () => window.clearTimeout(t)
  }, [playing, step, speed, total, loop])

  // видимость: автоплей при первом появлении, пауза за экраном. Player передаёт сюда всю фигуру
  // (а не только свою панель): в высокой фигуре читатель смотрит на схему над плеером, и проигрывание
  // не должно вставать, пока фигура на экране.
  const viewRef = useCallback((el: Element | null) => {
    io.current?.disconnect()
    io.current = null
    if (!el || typeof IntersectionObserver === 'undefined') return
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (!entry) return
        // «видно»: пятая часть фигуры или половина экрана (для фигур выше экрана)
        const enough =
          entry.isIntersecting &&
          (entry.intersectionRatio >= 0.2 || (entry.rootBounds !== null && entry.intersectionRect.height >= entry.rootBounds.height * 0.5))
        if (enough) {
          if (wantAutoplay.current && !autoplayed.current) {
            autoplayed.current = true
            setPlayingState(true)
          } else if (autoPaused.current) {
            autoPaused.current = false
            setPlayingState(true)
          }
        } else if (!entry.isIntersecting && playingRef.current) {
          // пауза, только когда фигура ушла с экрана совсем
          autoPaused.current = true
          setPlayingState(false)
        }
      },
      { threshold: [0, 0.1, 0.2] },
    )
    obs.observe(el)
    io.current = obs
  }, [])

  // плеер без viewRef: автоплей сразу
  useEffect(() => {
    if (!io.current && wantAutoplay.current && !autoplayed.current) {
      autoplayed.current = true
      setPlayingState(true)
    }
    return () => io.current?.disconnect()
  }, [])

  const toggle = useCallback(() => setPlaying(!playingRef.current), [setPlaying])
  const next = useCallback(() => setStep((s) => s + 1), [setStep])
  const prev = useCallback(() => setStep((s) => s - 1), [setStep])
  const reset = useCallback(() => {
    setPlaying(false)
    setStepState(0)
  }, [setPlaying])

  // шагов стало меньше: эффект выше поправит состояние только после рендера, а фигура, которая
  // берёт views[p.step], упала бы на этот один рендер. Наружу отдаём уже ограниченный шаг.
  const shown = clamp(step)

  return {
    steps: total,
    step: shown,
    setStep,
    playing,
    setPlaying,
    toggle,
    next,
    prev,
    reset,
    speed,
    setSpeed,
    atStart: shown === 0,
    atEnd: shown === total - 1,
    bind: {
      steps: total,
      step: shown,
      onStep: setStep,
      playing,
      onPlayingChange: setPlaying,
      speed,
      onSpeedChange: setSpeed,
      viewRef,
    },
  }
}

/* ===========================================================================
 * Player: панель управления.
 * ========================================================================= */

export interface PlayerProps {
  steps: number
  step: number
  onStep: (step: number) => void
  playing: boolean
  onPlayingChange: (playing: boolean) => void
  speed?: number
  onSpeedChange?: (speed: number) => void
  /** Подписи шагов (показывается подпись текущего). */
  labels?: string[]
  /** Варианты скорости (по умолчанию 0,5×, 1×, 2×). */
  speeds?: number[]
  /** Для автопаузы за экраном (приходит из usePlayer().bind). */
  viewRef?: (el: Element | null) => void
  /**
   * Свой счётчик вместо «Шаг N из M» (step с нуля). Например, когда кадры плеера означают время,
   * а «шаг» в главе значит проход модели: counter={(s, n) => `${fmt.ms(t)} из ${fmt.ms(total)}`}.
   */
  counter?: (step: number, total: number) => ReactNode
  className?: string
  'aria-label'?: string
}

const iconBtn =
  'inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-control text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink disabled:pointer-events-none disabled:opacity-35'

/**
 * Панель пошагового демо: в начало, назад, пуск/пауза, вперёд, ползунок шагов, скорость.
 * Клавиши (когда фокус внутри фигуры): пробел пуск/пауза, ← и → шаг.
 */
export function Player({
  steps,
  step,
  onStep,
  playing,
  onPlayingChange,
  speed = 1,
  onSpeedChange,
  labels,
  speeds = [0.5, 1, 2],
  viewRef,
  counter,
  className,
  'aria-label': ariaLabel = 'Пошаговое демо',
}: PlayerProps) {
  const root = useRef<HTMLDivElement | null>(null)
  // подпись шага озвучивается, только когда читатель работает с этой фигурой и проигрывание стоит
  const inside = useFigureFocusWithin()
  // стабильный ref: инлайновая функция вызывалась бы на каждом шаге и пересоздавала IntersectionObserver
  const setRoot = useCallback(
    (el: HTMLDivElement | null) => {
      root.current = el
      viewRef?.(el ? (el.closest('[data-figure]') ?? el) : null)
    },
    [viewRef],
  )
  const total = Math.max(1, steps)
  const last = total - 1
  const label = labels?.[step]

  const go = useCallback(
    (s: number) => {
      onPlayingChange(false)
      onStep(Math.max(0, Math.min(last, s)))
    },
    [last, onPlayingChange, onStep],
  )

  // клавиши на всей фигуре
  const keyState = useRef({ step, playing, go, onPlayingChange })
  keyState.current = { step, playing, go, onPlayingChange }
  useEffect(() => {
    const el = root.current
    if (!el) return
    const host = (el.closest('[data-figure]') as HTMLElement | null) ?? el
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return
      const t = e.target as HTMLElement
      if (t.closest('input, textarea, select, [contenteditable="true"], [role="slider"], [role="tablist"], [role="radiogroup"], [role="grid"], [role="listbox"], [role="menu"]')) return
      const s = keyState.current
      if (e.key === ' ') {
        if (t.closest('button, a, [role="button"], summary')) return
        e.preventDefault()
        s.onPlayingChange(!s.playing)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        s.go(s.step + 1)
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        s.go(s.step - 1)
      }
    }
    host.addEventListener('keydown', onKey)
    return () => host.removeEventListener('keydown', onKey)
  }, [])

  const fill = last > 0 ? (step / last) * 100 : 100
  const custom = counter ? counter(step, total) : null
  const countText = typeof custom === 'string' ? custom : `Шаг ${step + 1} из ${total}`

  return (
    <div
      ref={setRoot}
      role="group"
      aria-label={ariaLabel}
      data-player
      // col-span-full: в сетке контролов фигуры плеер всегда занимает свою строку целиком
      // (иначе ему доставалась одна колонка, скорость уезжала под кнопки, а дорожка сжималась)
      className={cn('col-span-full flex flex-wrap items-center gap-x-3 gap-y-2', className)}
    >
      <div className="flex items-center gap-0.5">
        <button type="button" className={iconBtn} onClick={() => go(0)} disabled={step === 0 && !playing} aria-label="В начало">
          <ArrowCounterClockwiseIcon size={18} aria-hidden />
        </button>
        <button type="button" className={iconBtn} onClick={() => go(step - 1)} disabled={step === 0} aria-label="Шаг назад" aria-keyshortcuts="ArrowLeft">
          <CaretLeftIcon size={18} aria-hidden />
        </button>
        <button
          type="button"
          onClick={() => onPlayingChange(!playing)}
          aria-label={playing ? 'Пауза' : step === last ? 'Проиграть сначала' : 'Пуск'}
          aria-keyshortcuts="Space"
          className="mx-0.5 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent text-accent-ink transition-colors hover:bg-accent-hover"
        >
          {playing ? <PauseIcon size={18} weight="fill" aria-hidden /> : <PlayIcon size={18} weight="fill" aria-hidden className="translate-x-px" />}
        </button>
        <button type="button" className={iconBtn} onClick={() => go(step + 1)} disabled={step === last} aria-label="Шаг вперёд" aria-keyshortcuts="ArrowRight">
          <CaretRightIcon size={18} aria-hidden />
        </button>
      </div>

      <div className="grid min-w-[180px] flex-1 basis-[240px] gap-1">
        <div className="flex min-w-0 items-baseline gap-2 text-sm leading-5" aria-live={playing || !inside ? 'off' : 'polite'}>
          <span className="shrink-0 text-ink-3 tabular-nums">{custom ?? countText}</span>
          {label && <span className="min-w-0 truncate font-medium text-ink">{label}</span>}
        </div>
        <input
          type="range"
          className="range"
          min={0}
          max={last}
          step={1}
          value={step}
          disabled={total < 2}
          aria-label="Шаг"
          aria-valuetext={`${countText}${label ? `: ${label}` : ''}`}
          onChange={(e) => go(Number(e.currentTarget.value))}
          style={{ '--fill': `${fill}%` } as CSSProperties}
        />
      </div>

      {onSpeedChange && (
        <div data-player-speed>
          <Segmented
            size="sm"
            aria-label="Скорость"
            value={speed}
            onChange={onSpeedChange}
            options={speeds.map((s) => ({ value: s, label: `${fmt.num(s)}×` }))}
          />
        </div>
      )}
    </div>
  )
}
