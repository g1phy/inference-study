import { useState } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { ArrowsClockwiseIcon } from '@phosphor-icons/react'
import { Badge, Button, Figure, Segmented, Slider, fmt, useElementSize } from '@/ui'

/**
 * RoPE в игрушечном виде: одна пара координат = точка на плоскости. Вектор q токена на позиции m
 * поворачивается на угол m·θ, вектор k на позиции n на n·θ. Скалярное произведение после поворота
 * зависит только от разности позиций (n − m). Числа условные, углы в градусах для наглядности
 * (в настоящем RoPE θᵢ = base^(−2i/d) радиан на позицию).
 */
const Q0 = { r: 1, deg: 20 }
const K0 = { r: 0.8, deg: 65 }
const MAX_POS = 12

type Speed = 'fast' | 'slow'
const THETA_DEG: Record<Speed, number> = { fast: 30, slow: 3 }
const RAD = Math.PI / 180

function dot(m: number, n: number, thetaDeg: number): number {
  return Q0.r * K0.r * Math.cos((K0.deg + n * thetaDeg - (Q0.deg + m * thetaDeg)) * RAD)
}

/** Угол в (−180°, 180°]. */
function wrapDeg(a: number): number {
  const x = ((((a + 180) % 360) + 360) % 360) - 180
  return x === -180 ? 180 : x
}

export function RopeFigure() {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  const reduce = useReducedMotion()
  const [m, setM] = useState(7)
  const [n, setN] = useState(4)
  const [speed, setSpeed] = useState<Speed>('fast')
  const [shifted, setShifted] = useState<{ before: number; m: number; n: number; speed: Speed } | null>(null)
  const theta = THETA_DEG[speed]
  const value = dot(m, n, theta)

  // сдвиг обеих позиций на одно и то же число: +3, если влезает, иначе −3, иначе сколько возможно
  const shift = (() => {
    if (Math.max(m, n) + 3 <= MAX_POS) return 3
    if (Math.min(m, n) - 3 >= 0) return -3
    const up = MAX_POS - Math.max(m, n)
    return up > 0 ? up : -Math.min(m, n)
  })()
  const shiftBoth = () => {
    if (shift === 0) return
    setM(m + shift)
    setN(n + shift)
    setShifted({ before: value, m: m + shift, n: n + shift, speed })
  }
  const showShift = shifted !== null && shifted.m === m && shifted.n === n && shifted.speed === speed

  const S = Math.min(Math.max(240, width), 340)
  const c = S / 2
  const R = S * 0.36
  const angQ = Q0.deg + m * theta
  const angK = K0.deg + n * theta
  const pt = (r: number, deg: number) => ({ x: c + R * r * Math.cos(deg * RAD), y: c - R * r * Math.sin(deg * RAD) })
  const q = pt(Q0.r, angQ)
  const k = pt(K0.r, angK)
  const q0 = pt(Q0.r, Q0.deg)
  const k0 = pt(K0.r, K0.deg)
  const between = wrapDeg(angK - angQ)
  const spring = reduce ? { duration: 0 } : { type: 'spring' as const, stiffness: 170, damping: 22 }

  // дуга угла между q и k (по короткой стороне)
  const arcR = R * 0.32
  const a1 = angQ + between
  const p0 = { x: c + arcR * Math.cos(angQ * RAD), y: c - arcR * Math.sin(angQ * RAD) }
  const p1 = { x: c + arcR * Math.cos(a1 * RAD), y: c - arcR * Math.sin(a1 * RAD) }
  const arc = `M ${p0.x} ${p0.y} A ${arcR} ${arcR} 0 0 ${between > 0 ? 0 : 1} ${p1.x} ${p1.y}`
  const labelAt = (r: number, deg: number) => ({ x: c + (R * r + 18) * Math.cos(deg * RAD), y: c - (R * r + 18) * Math.sin(deg * RAD) })
  const ql = labelAt(Q0.r, angQ)
  const kl = labelAt(K0.r, angK)

  return (
    <Figure
      id="embeddings-figure-rope"
      title="RoPE на игрушке: поворот на угол, который зависит от позиции"
      controls={
        <>
          <Slider label="Позиция q (m)" value={m} min={0} max={MAX_POS} onChange={(v) => setM(v)} format={(v) => fmt.int(v)} />
          <Slider label="Позиция k (n)" value={n} min={0} max={MAX_POS} onChange={(v) => setN(v)} format={(v) => fmt.int(v)} />
          <Segmented<Speed>
            label="Пара координат"
            value={speed}
            onChange={setSpeed}
            options={[
              { value: 'fast', label: 'быстрая' },
              { value: 'slow', label: 'медленная' },
            ]}
          />
        </>
      }
      caption="Попробуйте: нажмите «Сдвинуть оба». Оба вектора повернутся, но угол между ними и скалярное произведение останутся прежними: важна только разность позиций. Числа условные, показана одна пара координат из 64."
    >
      <div className="grid min-w-0 items-center gap-6 @xl:grid-cols-[minmax(0,300px)_minmax(0,1fr)]">
        <div ref={ref} className="min-w-0">
          {width > 0 && (
            <svg
              width={S}
              height={S}
              role="img"
              aria-label={`Плоскость одной пары координат: q повёрнут на ${fmt.int(m * theta)}°, k на ${fmt.int(n * theta)}°, угол между ними ${fmt.int(between)}°`}
              className="mx-auto block"
            >
              <defs>
                <marker id="rope-arrow-q" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--c-q)' }} />
                </marker>
                <marker id="rope-arrow-k" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--c-k)' }} />
                </marker>
              </defs>
              <line x1={8} x2={S - 8} y1={c} y2={c} style={{ stroke: 'var(--grid)', strokeWidth: 1 }} />
              <line x1={c} x2={c} y1={8} y2={S - 8} style={{ stroke: 'var(--grid)', strokeWidth: 1 }} />
              <circle cx={c} cy={c} r={R} style={{ fill: 'none', stroke: 'var(--line-strong)', strokeWidth: 1 }} />
              {/* исходные векторы на позиции 0 */}
              <line x1={c} y1={c} x2={q0.x} y2={q0.y} style={{ stroke: 'var(--c-q)', strokeWidth: 1.25, strokeDasharray: '3 3', opacity: 0.6 }} />
              <line x1={c} y1={c} x2={k0.x} y2={k0.y} style={{ stroke: 'var(--c-k)', strokeWidth: 1.25, strokeDasharray: '3 3', opacity: 0.6 }} />
              <path d={arc} style={{ fill: 'none', stroke: 'var(--ink-3)', strokeWidth: 1.25 }} />
              <motion.line
                x1={c}
                y1={c}
                initial={false}
                animate={{ x2: q.x, y2: q.y }}
                transition={spring}
                markerEnd="url(#rope-arrow-q)"
                style={{ stroke: 'var(--c-q)', strokeWidth: 2.5 }}
              />
              <motion.line
                x1={c}
                y1={c}
                initial={false}
                animate={{ x2: k.x, y2: k.y }}
                transition={spring}
                markerEnd="url(#rope-arrow-k)"
                style={{ stroke: 'var(--c-k)', strokeWidth: 2.5 }}
              />
              <motion.text
                initial={false}
                animate={{ x: ql.x, y: ql.y }}
                transition={spring}
                textAnchor="middle"
                dominantBaseline="central"
                style={{ font: '600 13px "Onest Variable", sans-serif', fill: 'var(--ink)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}
              >
                q
              </motion.text>
              <motion.text
                initial={false}
                animate={{ x: kl.x, y: kl.y }}
                transition={spring}
                textAnchor="middle"
                dominantBaseline="central"
                style={{ font: '600 13px "Onest Variable", sans-serif', fill: 'var(--ink)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}
              >
                k
              </motion.text>
            </svg>
          )}
          <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1 text-xs text-ink-3">
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-0 w-3 border-t-2" style={{ borderColor: 'var(--c-q)' }} />
              Query
            </span>
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-0 w-3 border-t-2" style={{ borderColor: 'var(--c-k)' }} />
              Key
            </span>
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-0 w-3 border-t border-dashed border-ink-3" />
              на позиции 0
            </span>
          </div>
        </div>

        <div className="grid min-w-0 content-start gap-4">
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] items-baseline gap-x-4 gap-y-2 text-[15px]">
            <dt className="text-ink-2">Шаг поворота θ</dt>
            <dd>{fmt.int(theta)}° на позицию</dd>
            <dt className="text-ink-2">Поворот q, m · θ</dt>
            <dd>
              {m} · {theta}° = {fmt.int(m * theta)}°
            </dd>
            <dt className="text-ink-2">Поворот k, n · θ</dt>
            <dd>
              {n} · {theta}° = {fmt.int(n * theta)}°
            </dd>
            <dt className="text-ink-2">Разность позиций</dt>
            <dd>n − m = {fmt.signed(n - m)}</dd>
            <dt className="text-ink-2">Угол от q до k</dt>
            <dd>{fmt.signed(Math.round(between))}°</dd>
            <dt className="font-medium text-ink">q · k</dt>
            <dd className="text-[1.25rem] leading-tight font-semibold">{fmt.num(value, 3)}</dd>
          </dl>
          <div className="grid gap-1.5">
            <div className="flex flex-wrap items-center gap-3">
              <Button icon={ArrowsClockwiseIcon} onClick={shiftBoth} disabled={shift === 0}>
                {shift === 0 ? 'Сдвинуть оба' : `Сдвинуть оба на ${fmt.signed(shift)}`}
              </Button>
              {showShift && Math.abs(shifted.before - value) < 1e-9 && <Badge tone="good">q · k не изменилось</Badge>}
            </div>
            {shift === 0 && (
              <p className="text-[13px] leading-snug text-ink-3">Нет места для сдвига: обе позиции должны остаться от 0 до {MAX_POS}.</p>
            )}
          </div>
          <p className="text-sm leading-snug text-ink-2">
            Быстрая пара заметно поворачивается уже между соседними токенами: она различает близкий порядок. Медленная почти не
            поворачивается на коротком отрезке и «чувствует» только большие расстояния.
          </p>
        </div>
      </div>
    </Figure>
  )
}
