import { useRef, useState, type FocusEvent, type PointerEvent, type ReactNode } from 'react'
import { attainableTflops, peakTflops, ridgePoint } from '@/engine/perf'
import type { DType, GpuSpec } from '@/engine/types'
import { cssVar, fmt, fontCss, measureText, tint, useElementSize, useFontsReady, useTip } from '@/ui'
import { logDomain, logTicks, makeScale } from '@/ui/chart-utils'

export interface RooflinePoint {
  id: string
  /** Короткая подпись у точки. */
  label: string
  /** Ещё короче, для узкого графика (на телефоне длинные подписи не помещаются рядом с точкой). */
  shortLabel?: string
  /** Арифметическая интенсивность, FLOP/байт. */
  ai: number
  role: 'prefill' | 'decode'
  /** Полая точка (attention в decode). */
  hollow?: boolean
  /** Такой конфигурации не бывает (KV не влезает в память): полая точка с пунктирной обводкой. */
  impossible?: boolean
  /** Содержимое тултипа с расчётом. */
  tip: ReactNode
  /** Текст для скринридера. */
  ariaLabel: string
}

interface Props {
  gpu: GpuSpec
  dtype: DType
  points: RooflinePoint[]
  height?: number
}

const X_DOMAIN: [number, number] = [0.2, 1e5]
const TICK_FONT = '12px "Onest Variable", sans-serif'
const LABEL_FONT = '13px "Onest Variable", sans-serif'
const HALO = { paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4, strokeLinejoin: 'round' } as const

/**
 * Roofline в лог-лог осях: наклон = полоса памяти, полка = пиковые FLOPS, излом = ridge point.
 * Слева от излома операция ждёт память, справа считает на пределе. Точки фокусируемые,
 * тултип показывает расчёт.
 */
export function RooflineChart({ gpu, dtype, points, height = 300 }: Props) {
  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement | null>(null)
  useFontsReady()
  const [active, setActive] = useState<string | null>(null)
  const tip = useTip({ scopeEl: svgRef.current, side: 'top', wide: true, onOutside: () => setActive(null) })

  const peak = peakTflops(gpu, dtype)
  const bw = gpu.bwTBs
  const ridge = ridgePoint(gpu, dtype)
  const yd = logDomain(bw * X_DOMAIN[0], peak * 1.9)
  const yTicks = logTicks(yd[0], yd[1])
  const xTicksAll = logTicks(X_DOMAIN[0], X_DOMAIN[1])
  const yTickW = Math.max(...yTicks.map((v) => measureText(fmt.compact(v), TICK_FONT)))
  const m = { top: 30, right: 14, bottom: 44, left: Math.ceil(yTickW) + 12 }
  const W = Math.max(300, width)
  const plotW = W - m.left - m.right
  const x = makeScale('log', X_DOMAIN, [m.left, m.left + plotW])
  const y = makeScale('log', yd, [m.top + height, m.top])
  const svgH = m.top + height + m.bottom
  // на узком экране подписи «10 тыс.» и «100 тыс.» слипаются: оставляем каждую вторую декаду
  const xTicks = plotW / xTicksAll.length < 56 ? xTicksAll.filter((_, i) => i % 2 === 0) : xTicksAll

  const roofAt = (ai: number) => attainableTflops(gpu, dtype, ai)
  const xr = x(ridge)
  const x0 = x(X_DOMAIN[0])
  const x1 = x(X_DOMAIN[1])
  const roof = `M${x0},${y(roofAt(X_DOMAIN[0]))}L${xr},${y(peak)}L${x1},${y(peak)}`
  // центры точек (на крыше)
  const centers = points.map((p) => ({ id: p.id, cx: x(clampX(p.ai)), cy: y(Math.max(yd[0], roofAt(clampX(p.ai)))) }))
  // подпись полосы памяти вдоль наклона: ищем место на наклоне, где её не перекрывает ни одна точка
  const slopeA = { x: x(Math.max(X_DOMAIN[0] * 2, ridge / 60)), y: y(roofAt(Math.max(X_DOMAIN[0] * 2, ridge / 60))) }
  const slopeB = { x: xr, y: y(peak) }
  const angle = (Math.atan2(slopeB.y - slopeA.y, slopeB.x - slopeA.x) * 180) / Math.PI
  const slopeLabel = `полоса памяти ${fmt.bandwidth(bw * 1000)}`
  const slopeW = measureText(slopeLabel, TICK_FONT)
  const slopeHit = (f: number) => {
    const c = { x: slopeA.x + (slopeB.x - slopeA.x) * f, y: slopeA.y + (slopeB.y - slopeA.y) * f - 8 }
    const rad = (-angle * Math.PI) / 180
    return centers.some((p) => {
      const dx = p.cx - c.x
      const dy = p.cy - c.y
      const lx = dx * Math.cos(rad) - dy * Math.sin(rad)
      const ly = dx * Math.sin(rad) + dy * Math.cos(rad)
      return Math.abs(lx) < slopeW / 2 + 10 && Math.abs(ly) < 7 + 10
    })
  }
  const slopeF = plotW > 420 ? [0.38, 0.22, 0.6, 0.75].find((f) => !slopeHit(f)) : undefined
  const slopeMid = slopeF === undefined ? null : { x: slopeA.x + (slopeB.x - slopeA.x) * slopeF, y: slopeA.y + (slopeB.y - slopeA.y) * slopeF }
  // подпись пика над полкой с зазором: точки лежат на крыше, выше полки их не бывает, поэтому подпись их не задевает
  const peakLabel = `пик ${fmt.int(peak)} TFLOPS`
  const narrow = plotW < 420
  const labelOf = (p: RooflinePoint) => (narrow && p.shortLabel ? p.shortLabel : p.label)

  // x наклона крыши на высоте yy (в пикселях): левее него подпись пересекла бы линию крыши
  const slopeXAt = (yy: number) => {
    const y0 = y(roofAt(X_DOMAIN[0]))
    if (yy >= y0) return -Infinity
    return x0 + ((yy - y0) * (xr - x0)) / (y(peak) - y0)
  }
  // подписи точек: под крышей, без наложений друг на друга и без пересечения наклона крыши
  const placed: Array<{ id: string; x: number; y: number; x0: number; x1: number; anchor: 'start' | 'end' }> = []
  const sorted = [...points].sort((a, b) => a.ai - b.ai)
  for (const p of sorted) {
    const { cx, cy } = centers.find((c) => c.id === p.id)!
    const w = measureText(labelOf(p), LABEL_FONT)
    const right = m.left + plotW
    let best: (typeof placed)[number] | null = null
    for (const anchor of ['start', 'end'] as const) {
      const lx = anchor === 'start' ? cx + 9 : cx - 9
      const px0 = anchor === 'start' ? lx : lx - w
      const px1 = px0 + w
      if (px1 > right || px0 < m.left) continue
      for (let ly = cy + 20; ly <= m.top + height - 6; ly += 17) {
        const crossesRoof = px0 < slopeXAt(ly - 10)
        const clash = placed.some((q) => Math.abs(q.y - ly) < 16 && px0 < q.x1 + 6 && q.x0 < px1 + 6)
        if (!crossesRoof && !clash) {
          best = { id: p.id, x: lx, y: ly, x0: px0, x1: px1, anchor }
          break
        }
      }
      if (best) break
    }
    if (!best) {
      const anchor: 'start' | 'end' = cx + 10 + w > right ? 'end' : 'start'
      const lx = anchor === 'start' ? cx + 9 : cx - 9
      const px0 = anchor === 'start' ? lx : lx - w
      best = { id: p.id, x: lx, y: Math.min(cy + 20, m.top + height - 6), x0: px0, x1: px0 + w, anchor }
    }
    placed.push(best)
  }

  const showTip = (p: RooflinePoint, el: Element) => {
    setActive(p.id)
    tip.show(el.getBoundingClientRect(), p.tip)
  }
  const hideTip = () => {
    setActive(null)
    tip.hide()
  }

  return (
    <div ref={wrapRef} className="min-w-0">
      {width > 0 && (
        <svg
          ref={svgRef}
          width={W}
          height={svgH}
          role="group"
          aria-label={`Roofline ${gpu.name}: наклон ${fmt.bandwidth(bw * 1000)}, пик ${fmt.int(peak)} TFLOPS, ridge point ${fmt.num(ridge)} FLOP на байт`}
          className="block select-none overflow-visible"
        >
          {/* зоны: слева упор в память, справа упор в вычисления */}
          <rect x={x0} y={m.top} width={Math.max(0, xr - x0)} height={height} style={{ fill: tint('decode', 7) }} />
          <rect x={xr} y={m.top} width={Math.max(0, x1 - xr)} height={height} style={{ fill: tint('prefill', 7) }} />
          <text x={x0 + 8} y={m.top + 16} style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}>
            упор в память
          </text>
          {x1 - xr > 280 && (
            <text x={xr + 8} y={m.top + 16} style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}>
              упор в вычисления
            </text>
          )}

          {/* подпись оси Y */}
          <text x={0} y={14} style={{ font: TICK_FONT, fill: 'var(--ink-3)' }}>
            Достижимые TFLOPS
          </text>
          {yTicks.map((v) => (
            <g key={`y${v}`}>
              <line x1={m.left} x2={m.left + plotW} y1={y(v)} y2={y(v)} style={{ stroke: 'var(--grid)', strokeWidth: 1 }} />
              <text x={m.left - 8} y={y(v)} textAnchor="end" dominantBaseline="central" style={{ ...fontCss(TICK_FONT), fill: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>
                {fmt.compact(v)}
              </text>
            </g>
          ))}
          <line x1={m.left} x2={m.left + plotW} y1={m.top + height} y2={m.top + height} style={{ stroke: 'var(--line-strong)' }} />
          {xTicks.map((v) => (
            <g key={`x${v}`}>
              <line x1={x(v)} x2={x(v)} y1={m.top} y2={m.top + height} style={{ stroke: 'var(--grid)', strokeWidth: 1 }} />
              <line x1={x(v)} x2={x(v)} y1={m.top + height} y2={m.top + height + 4} style={{ stroke: 'var(--line-strong)' }} />
              <text x={x(v)} y={m.top + height + 17} textAnchor="middle" style={{ ...fontCss(TICK_FONT), fill: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>
                {fmt.compact(v)}
              </text>
            </g>
          ))}
          <text x={m.left + plotW} y={svgH - 4} textAnchor="end" style={{ font: TICK_FONT, fill: 'var(--ink-3)' }}>
            Арифметическая интенсивность, FLOP/байт
          </text>

          {/* ridge point */}
          <line x1={xr} x2={xr} y1={y(peak)} y2={m.top + height} style={{ stroke: 'var(--ink-3)', strokeWidth: 1, strokeDasharray: '4 3' }} />
          {(() => {
            const full = `ridge point ≈ ${fmt.int(ridge)} FLOP/байт`
            const short = `ridge ≈ ${fmt.int(ridge)}`
            const rightRoom = m.left + plotW - xr - 8
            const leftRoom = xr - m.left - 8
            const wf = measureText(full, TICK_FONT)
            const ws = measureText(short, TICK_FONT)
            const pick =
              rightRoom >= wf ? { text: full, side: 1 } : leftRoom >= wf ? { text: full, side: -1 } : rightRoom >= ws ? { text: short, side: 1 } : { text: short, side: -1 }
            return (
              <text
                x={xr + pick.side * 6}
                y={m.top + height - 8}
                textAnchor={pick.side > 0 ? 'start' : 'end'}
                style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}
              >
                {pick.text}
              </text>
            )
          })()}

          {/* крыша */}
          <path d={roof} style={{ fill: 'none', stroke: 'var(--ink)', strokeWidth: 2, strokeLinejoin: 'round' }} />
          {slopeMid && (
            <text
              x={slopeMid.x}
              y={slopeMid.y - 8}
              transform={`rotate(${angle} ${slopeMid.x} ${slopeMid.y - 8})`}
              textAnchor="middle"
              style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}
            >
              {slopeLabel}
            </text>
          )}
          <text x={x1 - 6} y={y(peak) - 12} textAnchor="end" style={{ font: TICK_FONT, fill: 'var(--ink-2)', ...HALO }}>
            {peakLabel}
          </text>

          {/* точки */}
          {sorted.map((p) => {
            const ax = clampX(p.ai)
            const cx = x(ax)
            const cy = y(Math.max(yd[0], roofAt(ax)))
            const lab = placed.find((q) => q.id === p.id)!
            const color = cssVar(p.role)
            const isActive = active === p.id
            return (
              <g key={p.id}>
                <text x={lab.x} y={lab.y} textAnchor={lab.anchor} style={{ font: LABEL_FONT, fill: 'var(--ink)', ...HALO }}>
                  {labelOf(p)}
                </text>
                <g
                  tabIndex={0}
                  role="img"
                  aria-label={p.ariaLabel}
                  className="cursor-pointer outline-none"
                  onPointerEnter={(e: PointerEvent<SVGGElement>) => showTip(p, e.currentTarget)}
                  onPointerDown={(e: PointerEvent<SVGGElement>) => showTip(p, e.currentTarget)}
                  onPointerLeave={(e: PointerEvent<SVGGElement>) => e.pointerType === 'mouse' && hideTip()}
                  onFocus={(e: FocusEvent<SVGGElement>) => showTip(p, e.currentTarget)}
                  onBlur={hideTip}
                >
                  <circle cx={cx} cy={cy} r={16} fill="transparent" />
                  {isActive && <circle cx={cx} cy={cy} r={11} style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 2 }} />}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={6.5}
                    style={
                      p.impossible
                        ? { fill: 'var(--surface)', stroke: color, strokeWidth: 2, strokeDasharray: '3 2' }
                        : p.hollow
                          ? { fill: 'var(--surface)', stroke: color, strokeWidth: 2.5 }
                          : { fill: color, stroke: 'var(--surface)', strokeWidth: 2 }
                    }
                  />
                </g>
              </g>
            )
          })}
        </svg>
      )}
      {tip.node}
    </div>
  )
}

function clampX(ai: number): number {
  return Math.min(X_DOMAIN[1], Math.max(X_DOMAIN[0], ai))
}
