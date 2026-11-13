import { useMemo, useRef, useState, type PointerEvent, type ReactNode } from 'react'
import { Figure, Slider, Stat, TipRow, fmt, useElementSize, useTip, cssVar, tint } from '@/ui'
import { linearTicks } from '@/ui/chart-utils'
import { DtypeSegmented, GpuSelect, ModelSelect, TpNote, durationValueUnit, hwLabel, times, type HardwareState } from './shared/hardware'
import { requestTiming, type RequestTiming } from './shared/timing'

interface Props {
  state: HardwareState
  /** Длина промпта читателя в токенах (null, пока грузится словарь). */
  promptCount: number | null
}

export function RequestTimelineFigure({ state, promptCount }: Props) {
  const [promptLenRaw, setPromptLen] = useState<number | null>(null)
  const userLen = promptCount ?? 43
  const promptLen = promptLenRaw ?? userLen
  const [outLen, setOutLen] = useState(400)
  const [queueMs, setQueueMs] = useState(0)
  const hw = state.hw

  const t = useMemo(() => requestTiming(hw, promptLen, outLen, queueMs), [hw, promptLen, outLen, queueMs])
  // точка отсчёта: ваш запрос, ответ 400 токенов, без очереди (призрачная полоса над «Весь запрос»)
  const base = useMemo(() => requestTiming(hw, userLen, 400, 0), [hw, userLen])
  const perPromptToken = t.prefillMs / promptLen
  const ratio = t.itlMs / perPromptToken

  return (
    <Figure
      id="prefill-decode-figure-timeline"
      title="Таймлайн запроса: TTFT и ITL"
      width="wide"
      actions={<span className="block max-w-[18rem] text-[13px] sm:max-w-none leading-snug text-ink-3">{hwLabel(hw)}, один запрос</span>}
      controls={
        <>
          <div className="col-span-full grid gap-x-6 gap-y-4 sm:grid-cols-3">
            <Slider
              label="Длина промпта"
              value={promptLen}
              min={16}
              max={32768}
              scale="log"
              onChange={setPromptLen}
              format={(v) => fmt.tokens(v)}
              hint={
                promptLenRaw !== null && promptCount !== null ? (
                  <button type="button" className="text-accent underline decoration-dotted underline-offset-2" onClick={() => setPromptLen(null)}>
                    вернуть ваш запрос ({fmt.tokens(promptCount)})
                  </button>
                ) : (
                  'сейчас: ваш запрос после шаблона'
                )
              }
            />
            <Slider label="Длина ответа" value={outLen} min={1} max={4096} scale="log" onChange={setOutLen} format={(v) => fmt.tokens(v)} />
            <Slider
              label="Очередь перед prefill"
              value={queueMs}
              min={0}
              max={2000}
              step={10}
              onChange={setQueueMs}
              format={(v) => fmt.ms(v)}
              hint="условно: зависит от нагрузки"
            />
          </div>
          <div className="col-span-full grid gap-x-6 gap-y-4 sm:grid-cols-3">
            <ModelSelect state={state} />
            <GpuSelect state={state} />
            <DtypeSegmented state={state} />
          </div>
        </>
      }
      caption="Попробуйте: поставьте промпт 8 000 токенов (длинный документ в RAG), а потом ответ 4 000 токенов (рассуждение) и сравните, насколько полоса уходит дальше пунктирной отметки вашего запроса."
      source="Время посчитано по характеристикам модели и GPU (подробно в главе «Compute-bound и memory-bound»), без сети и без соседей по batch."
    >
      <div className="grid gap-5">
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
          <Stat label="TTFT" value={durationValueUnit(t.ttftMs)[0]} unit={durationValueUnit(t.ttftMs)[1]} hint="очередь + prefill" swatch="var(--c-prefill)" />
          <Stat label="ITL" value={durationValueUnit(t.itlMs)[0]} unit={durationValueUnit(t.itlMs)[1]} hint="один шаг decode" swatch="var(--c-decode)" />
          <Stat label="E2E" value={durationValueUnit(t.e2eMs)[0]} unit={durationValueUnit(t.e2eMs)[1]} hint="TTFT + ITL × (N − 1)" />
          <Stat label="Скорость печати" value={fmt.int(1000 / t.itlMs)} unit="ток/с" hint="1000 / ITL" />
        </div>
        <TimelineSvg t={t} base={base} />
        <div className="grid gap-2 rounded-control bg-surface-2 px-3 py-2.5 text-[14px] leading-snug text-ink-2 @xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] @xl:items-center @xl:gap-6">
          <div>
            <span className="text-ink-3">Prefill на токен промпта: </span>
            <span className="font-medium text-ink">{fmt.ms(perPromptToken)}</span>
          </div>
          <div>
            <span className="text-ink-3">Decode на токен ответа: </span>
            <span className="font-medium text-ink">{fmt.ms(t.itlMs)}</span>
          </div>
          <div className="font-medium text-ink">
            в {times(ratio)} дороже
          </div>
        </div>
        <TpNote hw={hw} />
      </div>
    </Figure>
  )
}

/* ------------------------------------------------------------------------- */

const BAR_H = 26

/**
 * Две полосы: весь запрос (над ним пунктиром точка отсчёта: ваш запрос с ответом в 400 токенов, на той же оси)
 * и его начало крупным планом после очереди (видно, что шаг decode = один токен).
 */
function TimelineSvg({ t, base }: { t: RequestTiming; base: RequestTiming }) {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement | null>(null)
  const tip = useTip({ scopeEl: svgRef.current, side: 'top' })
  const W = Math.max(280, width)
  const decodeTokens = Math.max(0, t.outLen - 1)
  const zoomSteps = Math.min(decodeTokens, 6)
  const zoomSpan = t.prefillMs + Math.max(1, zoomSteps) * t.itlMs
  // окно крупного плана начинается почти у конца очереди: шаги decode не сжимаются в волоски
  const zoomFrom = t.queueMs > 0 ? Math.max(0, t.queueMs - 0.1 * zoomSpan) : 0
  const zoomT = t.ttftMs + Math.max(1, zoomSteps) * t.itlMs
  const showZoom = decodeTokens > 0 && t.e2eMs > zoomT * 1.6
  const totalMs = Math.max(t.e2eMs, base.e2eMs)
  const laneH = 98
  const ghostH = 30
  const svgH = ghostH + (showZoom ? laneH * 2 + 6 : laneH)

  const show = (e: PointerEvent<SVGRectElement>, content: ReactNode) => tip.show(e.currentTarget.getBoundingClientRect(), content)
  const hide = (e: PointerEvent<SVGRectElement>) => e.pointerType === 'mouse' && tip.hide()

  return (
    <div ref={ref} className="min-w-0">
      {width > 0 && (
        <svg
          ref={svgRef}
          width={W}
          height={svgH}
          role="img"
          aria-label={`Таймлайн: очередь ${fmt.ms(t.queueMs)}, prefill ${fmt.ms(t.prefillMs)}, TTFT ${fmt.ms(t.ttftMs)}, дальше ${fmt.count(decodeTokens, ['шаг', 'шага', 'шагов'])} decode по ${fmt.ms(t.itlMs)}, E2E ${fmt.ms(t.e2eMs)}`}
          className="block select-none overflow-visible"
        >
          <defs>
            <pattern id="pd-queue-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" fill="var(--surface-2)" />
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--line-strong)" strokeWidth="2" />
            </pattern>
          </defs>
          <Ghost base={base} width={W} totalMs={totalMs} />
          <Lane t={t} y={ghostH} width={W} totalMs={totalMs} title="Весь запрос" mode="whole" onShow={show} onHide={hide} />
          {showZoom && (
            <Lane
              t={t}
              y={ghostH + laneH + 6}
              width={W}
              fromMs={zoomFrom}
              totalMs={zoomT}
              title={t.queueMs > 0 ? `Крупно после очереди (${fmt.ms(t.queueMs)}): prefill и первые шаги decode` : `Начало крупно: первые ${fmt.ms(zoomT)}`}
              mode="zoom"
              steps={zoomSteps}
              onShow={show}
              onHide={hide}
            />
          )}
        </svg>
      )}
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className="h-3 w-4 rounded-[3px] border border-line-strong"
            style={{ background: 'repeating-linear-gradient(45deg, var(--surface-2) 0 3px, var(--line-strong) 3px 5px)' }}
          />
          очередь
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="h-3 w-4 rounded-[3px]" style={{ background: cssVar('prefill') }} />
          prefill
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="h-3 w-4 rounded-[3px] border" style={{ background: tint('decode', 26), borderColor: cssVar('decode') }} />
          decode: один шаг на токен
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="w-5 border-t-2 border-dashed border-ink-3" />
          ваш запрос с ответом в 400 токенов
        </span>
      </div>
      {tip.node}
    </div>
  )
}

function Lane({
  t,
  y,
  width,
  fromMs = 0,
  totalMs,
  title,
  mode,
  steps = 0,
  onShow,
  onHide,
}: {
  t: RequestTiming
  y: number
  width: number
  /** Начало окна, мс (крупный план после очереди). */
  fromMs?: number
  totalMs: number
  title: string
  mode: 'whole' | 'zoom'
  steps?: number
  onShow: (e: PointerEvent<SVGRectElement>, content: ReactNode) => void
  onHide: (e: PointerEvent<SVGRectElement>) => void
}) {
  const left = 4
  const plotW = width - 8
  const T = Math.max(totalMs - fromMs, 1e-6)
  const x = (ms: number) => left + (Math.max(0, Math.min(ms - fromMs, T)) / T) * plotW
  const barY = y + 40
  const axisY = barY + BAR_H + 8
  const decodeTokens = Math.max(0, t.outLen - 1)
  const pX0 = x(t.queueMs)
  const pX1 = Math.max(x(t.ttftMs), pX0 + 2)
  const dEnd = mode === 'whole' ? x(t.e2eMs) : x(t.ttftMs + steps * t.itlMs)
  // крупный план после очереди: время на оси считаем от конца очереди («+5 мс»), иначе все подписи округлятся до «2 с»
  const origin = mode === 'zoom' && fromMs > 0 ? t.queueMs : 0
  const axisTicks = linearTicks(fromMs - origin, fromMs + T - origin, Math.max(2, Math.floor(plotW / 110)))
    .filter((r) => r >= (origin > 0 ? 0 : -Infinity))
    .map((r) => ({ at: origin + r, label: origin > 0 ? (r === 0 ? fmt.ms(origin) : `+${fmt.ms(r)}`) : fmt.ms(r) }))
  // подпись TTFT слева от линии, если есть место, иначе справа
  const ttftLeft = pX1 > 96

  // «весь запрос»: черта на токен, если помещается, иначе на каждые k токенов
  const gapPx = decodeTokens > 0 ? (dEnd - pX1) / decodeTokens : 0
  const every = gapPx >= 4 ? 1 : Math.max(1, Math.ceil(4 / Math.max(gapPx, 1e-6)))
  const ticks: number[] = []
  if (mode === 'whole' && decodeTokens > 0) for (let i = every; i < decodeTokens; i += every) ticks.push(pX1 + i * gapPx)

  const queueTip = <TipRow value={fmt.ms(t.queueMs)} label="очередь: ждём места в batch" />
  const prefillTip = <PrefillTip t={t} />
  const decodeTip = <DecodeTip t={t} every={every} />

  return (
    <g>
      <text x={left} y={y + 12} className="fill-ink-2 text-[12px] font-medium">
        {title}
      </text>
      {t.queueMs > 0 && (
        <rect
          x={x(fromMs)}
          y={barY}
          width={Math.max(1, pX0 - x(fromMs))}
          height={BAR_H}
          rx={4}
          fill="url(#pd-queue-hatch)"
          stroke="var(--line-strong)"
          onPointerMove={(e) => onShow(e, queueTip)}
          onPointerDown={(e) => onShow(e, queueTip)}
          onPointerLeave={onHide}
        />
      )}

      <rect
        x={pX0}
        y={barY}
        width={Math.max(2, pX1 - pX0)}
        height={BAR_H}
        rx={3}
        fill={cssVar('prefill')}
        onPointerMove={(e) => onShow(e, prefillTip)}
        onPointerDown={(e) => onShow(e, prefillTip)}
        onPointerLeave={onHide}
      />
      {mode === 'whole' && decodeTokens > 0 && (
        <g>
          <rect
            x={pX1}
            y={barY}
            width={Math.max(1, dEnd - pX1)}
            height={BAR_H}
            rx={3}
            fill={tint('decode', 26)}
            stroke={cssVar('decode')}
            onPointerMove={(e) => onShow(e, decodeTip)}
            onPointerDown={(e) => onShow(e, decodeTip)}
            onPointerLeave={onHide}
          />
          <g pointerEvents="none">
            {ticks.map((tx, i) => (
              <line key={i} x1={tx} x2={tx} y1={barY + 6} y2={barY + BAR_H - 6} stroke={cssVar('decode')} strokeWidth={1.25} />
            ))}
          </g>
        </g>
      )}
      {mode === 'zoom' &&
        Array.from({ length: steps }, (_, i) => {
          const s0 = x(t.ttftMs + i * t.itlMs)
          const s1 = x(t.ttftMs + (i + 1) * t.itlMs)
          const stepTip = <TipRow color={cssVar('decode')} value={fmt.ms(t.itlMs)} label={`шаг decode ${i + 1}: токен ответа № ${i + 2}`} />
          return (
            <g key={i}>
              <rect
                x={s0 + 1}
                y={barY}
                width={Math.max(1, s1 - s0 - 2)}
                height={BAR_H}
                rx={3}
                fill={tint('decode', 26)}
                stroke={cssVar('decode')}
                onPointerMove={(e) => onShow(e, stepTip)}
                onPointerDown={(e) => onShow(e, stepTip)}
                onPointerLeave={onHide}
              />
              {s1 - s0 > 34 && (
                <text x={(s0 + s1) / 2} y={barY + BAR_H / 2} textAnchor="middle" dominantBaseline="central" className="pointer-events-none fill-ink-2 text-[11px]">
                  шаг {i + 1}
                </text>
              )}
            </g>
          )
        })}

      {/* TTFT: первый токен ответа появился */}
      <line x1={pX1} x2={pX1} y1={barY - 9} y2={barY + BAR_H + 3} stroke="var(--ink)" strokeWidth={1.5} pointerEvents="none" />
      <text x={ttftLeft ? pX1 - 5 : pX1 + 5} y={barY - 9} textAnchor={ttftLeft ? 'end' : 'start'} className="fill-ink text-[12px] font-medium">
        TTFT {fmt.ms(t.ttftMs)}
      </text>
      {mode === 'whole' && decodeTokens > 0 && dEnd - pX1 > 190 && (
        <text x={dEnd} y={barY - 9} textAnchor="end" className="fill-ink-2 text-[12px]">
          E2E {fmt.ms(t.e2eMs)}
        </text>
      )}
      {mode === 'zoom' && steps > 0 && dEnd - pX1 > 190 && (
        <text x={dEnd} y={barY - 9} textAnchor="end" className="fill-ink-2 text-[12px]">
          ITL {fmt.ms(t.itlMs)} на каждый шаг
        </text>
      )}

      <line x1={left} x2={left + plotW} y1={axisY} y2={axisY} stroke="var(--line-strong)" />
      {axisTicks.map(({ at, label }) => (
        <g key={at}>
          <line x1={x(at)} x2={x(at)} y1={axisY} y2={axisY + 4} stroke="var(--line-strong)" />
          <text
            x={x(at)}
            y={axisY + 16}
            textAnchor={x(at) < left + 20 ? 'start' : x(at) > width - 30 ? 'end' : 'middle'}
            className="fill-ink-3 text-[12px]"
            style={{ fontVariantNumeric: 'tabular-nums' }}
          >
            {label}
          </text>
        </g>
      ))}
    </g>
  )
}

/** Точка отсчёта: тонкая пунктирная полоса «ваш запрос, 400 токенов ответа» на той же оси, что и «Весь запрос». */
function Ghost({ base, width, totalMs }: { base: RequestTiming; width: number; totalMs: number }) {
  const left = 4
  const plotW = width - 8
  const T = Math.max(totalMs, 1e-6)
  const x = (ms: number) => left + (Math.min(ms, T) / T) * plotW
  const xEnd = x(base.e2eMs)
  const label = `ваш запрос: ${fmt.tokens(base.promptLen)} → ${fmt.tokens(base.outLen)}, ${fmt.ms(base.e2eMs)}`
  const labelRight = xEnd < width * 0.55
  return (
    <g pointerEvents="none">
      <rect x={x(0)} y={16} width={Math.max(2, x(base.ttftMs) - x(0))} height={6} rx={2} fill={tint('prefill', 45)} />
      <rect x={x(base.ttftMs)} y={16} width={Math.max(1, xEnd - x(base.ttftMs))} height={6} rx={2} fill={tint('decode', 22)} stroke={cssVar('decode')} strokeDasharray="3 2" strokeWidth={1} />
      <line x1={xEnd} x2={xEnd} y1={12} y2={26} stroke="var(--ink-3)" strokeWidth={1.25} />
      <text x={labelRight ? xEnd + 6 : xEnd - 6} y={12} textAnchor={labelRight ? 'start' : 'end'} className="fill-ink-2 text-[12px]">
        {label}
      </text>
    </g>
  )
}

function PrefillTip({ t }: { t: RequestTiming }) {
  return (
    <div className="grid gap-1">
      <TipRow color={cssVar('prefill')} value={fmt.ms(t.prefillMs)} label={`prefill ${fmt.tokens(t.promptLen)}`} />
      <div className="text-xs text-ink-3">
        {t.prefillBound === 'compute' ? 'упор в вычисления: промпт длинный' : 'упор в чтение весов: промпт короткий'}
      </div>
    </div>
  )
}

function DecodeTip({ t, every }: { t: RequestTiming; every: number }) {
  return (
    <div className="grid gap-1">
      <TipRow color={cssVar('decode')} value={fmt.ms(t.e2eMs - t.ttftMs)} label={`decode, ${fmt.count(t.outLen - 1, ['шаг', 'шага', 'шагов'])}`} />
      <div className="text-xs text-ink-3">
        шаг {fmt.ms(t.firstItlMs)} в начале и {fmt.ms(t.lastItlMs)} в конце: KV растёт{every > 1 ? `; черта = ${fmt.tokens(every)}` : ''}
      </div>
    </div>
  )
}
