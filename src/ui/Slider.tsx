import { useId, useMemo, useRef, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react'
import { cn } from './cn'
import { FieldHint, FieldLabel } from './Field'
import { fmt } from './fmt'
import { LOG_RES, logArrowStep, makeSliderScale } from './slider-scale'

export interface SliderProps {
  label: ReactNode
  value: number
  min: number
  max: number
  /**
   * Шаг значения. Округляйте здесь, а не в onChange: слайдер знает шаг и двигается на него стрелками.
   * На log-шкале при шаге 1, 0,1, 0,01… значения ещё и прилипают к круглым числам и степеням двойки.
   */
  step?: number
  onChange: (value: number) => void
  /** Как показать значение справа от подписи (и в aria-valuetext). */
  format?: (value: number) => string
  hint?: ReactNode
  disabled?: boolean
  id?: string
  /** 'log' удобна для длин промпта, batch и т.п. (min > 0). */
  scale?: 'linear' | 'log'
  className?: string
}

/**
 * Слайдер: подпись над полем, значение справа, заливка дорожки акцентом.
 * Стрелки двигают значение на каждом нажатии (и на log-шкале), PageUp/PageDown прыгают на десятую часть шкалы.
 *
 *   <Slider label="Temperature" value={t} min={0} max={2} step={0.05} onChange={setT} format={(v) => fmt.num(v, 2)} />
 */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v) => fmt.num(v),
  hint,
  disabled,
  id,
  scale = 'linear',
  className,
}: SliderProps) {
  const autoId = useId()
  const inputId = id ?? `slider-${autoId}`
  const hintId = hint ? `${inputId}-hint` : undefined
  const s = useMemo(() => makeSliderScale({ min, max, step, scale }), [min, max, step, scale])
  // значение вне [min, max] (глава округлила его в onChange) не показываем и не рисуем за краем
  const shown = s.clamp(value)
  const pos = s.toPos(shown)
  const fill = s.log ? (pos / LOG_RES) * 100 : ((shown - min) / (max - min || 1)) * 100

  // последнее нажатие стрелки: если глава округлила значение обратно, следующее нажатие идёт дальше
  const arrow = useRef<{ pos: number; value: number; before: number } | null>(null)

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (!s.log || disabled) return
    const dir = e.key === 'ArrowRight' || e.key === 'ArrowUp' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? -1 : 0
    if (!dir) return
    e.preventDefault()
    const a = arrow.current
    const resume = a && (value === a.value || value === a.before) ? { pos: a.pos, value: a.value } : undefined
    const r = logArrowStep(s, value, dir, resume)
    if (!r) return
    arrow.current = { pos: r.pos, value: r.value, before: value }
    onChange(r.value)
  }

  return (
    <div className={cn('grid gap-1.5', className)}>
      <FieldLabel htmlFor={inputId} aside={format(shown)}>
        {label}
      </FieldLabel>
      <input
        id={inputId}
        type="range"
        className="range"
        min={s.input.min}
        max={s.input.max}
        step={s.input.step}
        value={pos}
        disabled={disabled}
        aria-valuetext={format(shown)}
        aria-describedby={hintId}
        onKeyDown={onKeyDown}
        onChange={(e) => {
          arrow.current = null
          onChange(s.fromPos(Number(e.currentTarget.value)))
        }}
        style={{ '--fill': `${Math.min(100, Math.max(0, fill))}%` } as CSSProperties}
      />
      {hint && <FieldHint id={hintId}>{hint}</FieldHint>}
    </div>
  )
}
