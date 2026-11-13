import { useMemo, useRef, useState, type PointerEvent } from 'react'
import { Badge, Figure, Segmented, Slider, TipRow, fmt, fontCss, measureText, useElementSize, useFontsReady, useTip } from '@/ui'
import { linearTicks, logTicks, makeScale, niceScale, type Scale } from '@/ui/chart-utils'
import { useSettings } from '@/state/settings'
import { simulateContinuous, vllmDefaultLimits, type SimResultExt } from '@/engine/sched'
import type { SimRequest } from '@/engine/types'
import { configLabel, perfOf, simConfig, type SimConfig } from './config'
import { itlTimeline, stepEnd, type ItlPoint } from './timeline'
import { AXIS_FONT, LegendItem, Metric, Swatch, TimeAxis, msTick } from './chart'

const DECODERS = 16
const LONG_ID = 'long'
const SLO_MS = 50

interface Mode {
  key: 'plain' | 'chunked'
  label: string
  color: string
  res: SimResultExt
  points: ItlPoint[]
  ttft: number
  maxItl: number
  prefillSteps: number
}

function buildRequests(promptLen: number, scale: number): SimRequest[] {
  const reqs: SimRequest[] = []
  for (let i = 0; i < DECODERS; i++) reqs.push({ id: `u${i}`, arrivalMs: i * 2 * scale, promptTokens: 600, outputTokens: 700 })
  reqs.push({ id: LONG_ID, arrivalMs: 1000 * scale, promptTokens: promptLen, outputTokens: 40 })
  return reqs
}

function runMode(key: Mode['key'], reqs: SimRequest[], cfg: SimConfig, budget: number): Mode {
  // без чанков бюджет почти ни на что не влияет (промпт идёт целиком), поэтому берём умолчание vLLM: линия не дёргается от ползунка
  const tokenBudget = key === 'chunked' ? budget : vllmDefaultLimits(cfg.gpu).tokenBudget
  const res = simulateContinuous(reqs, { ...perfOf(cfg), chunkedPrefill: key === 'chunked', tokenBudget })
  const long = res.requests.find((r) => r.id === LONG_ID)!
  const points = itlTimeline(res, (id) => id !== LONG_ID)
  const prefillSteps = res.steps.filter((s) => s.prefill.some((p) => p.reqId === LONG_ID)).length
  const inWindow = points.filter((p) => p.t >= long.arrivalMs && p.t <= long.firstTokenMs + 1)
  return {
    key,
    label: key === 'chunked' ? `С chunked prefill, бюджет ${fmt.int(budget)}` : 'Без chunked prefill',
    color: key === 'chunked' ? 'var(--c-decode)' : 'var(--ink-3)',
    res,
    points,
    ttft: long.ttftMs,
    maxItl: inWindow.length ? Math.max(...inWindow.map((p) => p.itl)) : 0,
    prefillSteps,
  }
}

const M = { top: 42, right: 16, bottom: 0, left: 66 }
const PLOT_H = 190
const STRIP_H = 16
const STRIP_GAP = 8
const BUDGETS = [8, 9, 10, 11, 12, 13, 14].map((e) => 2 ** e)

/** Самый большой бюджет, при котором самая длинная пауза у остальных не длиннее SLO (null, если такого нет). */
function sloBudgetFor(promptLen: number, cfg: SimConfig): number | null {
  const reqs = buildRequests(promptLen, cfg.timeScale)
  let best: number | null = null
  for (const b of BUDGETS) if (runMode('chunked', reqs, cfg, b).maxItl <= SLO_MS) best = b
  return best
}

export function ChunkedFigure() {
  const [settings] = useSettings()
  const cfg = simConfig(settings.modelId, settings.gpuId)
  const [promptExp, setPromptExp] = useState(14) // 16 384 токена
  const [budgetExp, setBudgetExp] = useState(11) // 2048
  const [yKind, setYKind] = useState<'linear' | 'log'>('linear')
  const promptLen = 2 ** promptExp
  const budget = 2 ** budgetExp
  const defaultBudget = vllmDefaultLimits(cfg.gpu).tokenBudget

  const data = useMemo(() => {
    const reqs = buildRequests(promptLen, cfg.timeScale)
    const plain = runMode('plain', reqs, cfg, budget)
    const chunked = runMode('chunked', reqs, cfg, budget)
    const arrival = reqs[reqs.length - 1].arrivalMs
    const span = Math.max(plain.ttft, chunked.ttft)
    const t0 = arrival - Math.max(60, span * 0.14)
    const t1 = arrival + span + Math.max(60, span * 0.16)
    return { plain, chunked, arrival, t0, t1 }
  }, [promptLen, budget, cfg])
  const sloBudget = useMemo(() => sloBudgetFor(promptLen, cfg), [promptLen, cfg])

  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  useFontsReady() // подписи на графике раскладываются по ширине текста
  const svgW = Math.max(300, width || 960)
  const { plain, chunked, arrival, t0, t1 } = data
  const modes = [plain, chunked]
  const x = makeScale('linear', [t0 - arrival, t1 - arrival], [M.left, svgW - M.right])
  const visible = (p: ItlPoint) => p.t >= t0 && p.t <= t1
  const yMax = Math.max(SLO_MS * 1.2, ...modes.flatMap((m) => m.points.filter(visible).map((p) => p.itl)))
  const yNice = niceScale(0, yMax, 4)
  const yMin = Math.max(1, Math.min(...modes.flatMap((m) => m.points.filter(visible).map((p) => p.itl))) * 0.7)
  const y: Scale =
    yKind === 'log'
      ? makeScale('log', [Math.pow(10, Math.floor(Math.log10(yMin))), Math.pow(10, Math.ceil(Math.log10(yMax)))], [M.top + PLOT_H, M.top])
      : makeScale('linear', yNice.domain, [M.top + PLOT_H, M.top])
  const yTicks = yKind === 'log' ? logTicks(y.domain[0], y.domain[1]) : yNice.ticks
  const narrowSvg = svgW < 560
  const xTicks = linearTicks(t0 - arrival, t1 - arrival, Math.max(2, Math.floor((svgW - M.left - M.right) / 90)))
  const stripTop = M.top + PLOT_H + 46
  const svgH = stripTop + 2 * (STRIP_H + STRIP_GAP) + 34

  const svgRef = useRef<SVGSVGElement | null>(null)
  const [hoverT, setHoverT] = useState<number | null>(null)
  const tip = useTip({ scopeEl: svgRef.current, side: 'right', onOutside: () => setHoverT(null) })

  const nearest = (m: Mode, t: number) => {
    let best: ItlPoint | null = null
    for (const p of m.points) if (visible(p) && (!best || Math.abs(p.t - t) < Math.abs(best.t - t))) best = p
    return best
  }
  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const box = svgRef.current!.getBoundingClientRect()
    const t = x.invert(e.clientX - box.left) + arrival
    setHoverT(t)
    const rows = modes.map((m) => ({ m, p: nearest(m, t) })).filter((r): r is { m: Mode; p: ItlPoint } => r.p !== null)
    tip.show(
      { left: e.clientX, top: box.top + M.top + PLOT_H / 2, width: 0, height: 0 },
      <div className="grid gap-1.5">
        <div className="text-xs text-ink-3">
          {fmt.ms(Math.abs(t - arrival))} {t >= arrival ? 'после прихода промпта' : 'до прихода промпта'}
        </div>
        {rows.map(({ m, p }) => {
          const s = m.res.steps[p.step]
          const pre = s.prefill.reduce((a, c) => a + c.tokens, 0)
          return (
            <div key={m.key} className="grid gap-0.5">
              <TipRow color={m.color} value={`ITL ${fmt.ms(p.itl)}`} label={m.key === 'chunked' ? 'с чанками' : 'без чанков'} />
              <div className="pl-5 text-xs text-ink-2">
                шаг {fmt.ms(s.durationMs)}: {pre ? `prefill ${fmt.tokens(pre)} + ` : ''}decode {s.decode.length}
              </div>
            </div>
          )
        })}
      </div>,
    )
  }
  const clear = () => {
    setHoverT(null)
    tip.hide()
  }

  // Ступеньки: каждый токен рисуется горизонталью от предыдущего токена до своего на высоте своего ITL,
  // поэтому пауза выглядит как плато на всю свою длину.
  const path = (m: Mode) => {
    let d = ''
    let prevEnd: number | null = null
    for (const p of m.points.filter(visible)) {
      const a = x(Math.max(p.t - p.itl, t0) - arrival)
      const b = x(p.t - arrival)
      const yy = y(Math.max(p.itl, y.domain[0]))
      d += prevEnd !== null && Math.abs(prevEnd - a) < 1.5 ? `V${yy.toFixed(1)}H${b.toFixed(1)}` : `M${a.toFixed(1)},${yy.toFixed(1)}H${b.toFixed(1)}`
      prevEnd = b
    }
    return d
  }

  const sloY = y(SLO_MS)
  const arrivalX = x(0)
  const hx = hoverT !== null ? x(hoverT - arrival) : null
  // decode-токены 16 пользователей занимают бюджет первыми: промпт входит целиком, только если влезает вместе с ними
  const sameAsPlain = budget >= promptLen + DECODERS
  const naiveSteps = Math.ceil(promptLen / budget)

  return (
    <Figure
      id="batching-figure-chunked"
      title="Длинный промпт посреди decode: ITL без chunked prefill и с ним"
      actions={<Badge>{configLabel(cfg)}</Badge>}
      controls={
        <>
          <Slider
            label="Длина нового промпта"
            value={promptExp}
            min={10}
            max={15}
            onChange={setPromptExp}
            format={(v) => fmt.tokens(2 ** v)}
          />
          <Slider
            label="Бюджет токенов на шаг"
            hint={`max_num_batched_tokens; у vllm serve на ${cfg.gpu.name} по умолчанию ${fmt.int(defaultBudget)}`}
            value={budgetExp}
            min={8}
            max={14}
            onChange={setBudgetExp}
            format={(v) => fmt.int(2 ** v)}
          />
          <Segmented<'linear' | 'log'>
            label="Шкала ITL"
            value={yKind}
            onChange={setYKind}
            options={[
              { value: 'linear', label: 'Линейная' },
              { value: 'log', label: 'Логарифм.' },
            ]}
          />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> уменьшайте бюджет, пока оранжевая линия не опустится под SLO
          {sloBudget !== null ? ` (при этом промпте до ${fmt.int(sloBudget)})` : ''}: длинных пауз у остальных больше нет. Сравните отметки «первый
          токен» на полосах внизу: TTFT длинного запроса сначала почти не меняется, а на самых маленьких бюджетах растёт, потому что шагов всё больше
          и каждый заново читает все веса. Поставьте бюджет {fmt.int(defaultBudget)} (значение по умолчанию) и промпт {fmt.tokens(defaultBudget / 2)}:
          промпт входит в шаг целиком, и линии совпадут. При промпте {fmt.tokens(defaultBudget)} от него всё же отрежется хвост в {DECODERS} токенов:
          эти места бюджета уже заняли decode-токены {DECODERS} пользователей. Время по roofline-модели, SLO {SLO_MS} мс условный.
        </>
      }
    >
      <div className="mb-3 grid gap-x-6 gap-y-2 @md:grid-cols-3">
        <div className="grid gap-0.5">
          <div className="text-sm text-ink-2">Самая длинная пауза у остальных</div>
          <Metric swatch="var(--ink-3)" label="без чанков" value={fmt.ms(plain.maxItl)} />
          <Metric swatch="var(--c-decode)" label="с чанками" value={fmt.ms(chunked.maxItl)} />
        </div>
        <div className="grid gap-0.5">
          <div className="text-sm text-ink-2">TTFT длинного запроса</div>
          <Metric swatch="var(--ink-3)" label="без чанков" value={fmt.ms(plain.ttft)} />
          <Metric swatch="var(--c-decode)" label="с чанками" value={fmt.ms(chunked.ttft)} />
        </div>
        <div className="grid gap-0.5">
          <div className="text-sm text-ink-2">Шагов с его prefill</div>
          <Metric swatch="var(--ink-3)" label="без чанков" value={fmt.int(plain.prefillSteps)} />
          <Metric swatch="var(--c-decode)" label="с чанками" value={fmt.int(chunked.prefillSteps)} />
        </div>
      </div>
      {chunked.prefillSteps > 1 && (
        <p className="mb-2 text-sm text-ink-2">
          Бюджет делят {DECODERS} decode-токенов и кусок промпта: за шаг входит {fmt.int(budget)} − {DECODERS} = {fmt.int(budget - DECODERS)}{' '}
          {fmt.plural(budget - DECODERS, ['токен', 'токена', 'токенов'])} промпта, поэтому шагов {fmt.int(chunked.prefillSteps)}
          {naiveSteps !== chunked.prefillSteps ? `, а не ${fmt.int(naiveSteps)}` : ''}.
        </p>
      )}
      <ul className="mb-1 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        <LegendItem swatch={<Swatch line color="var(--ink-3)" />}>без chunked prefill</LegendItem>
        <LegendItem swatch={<Swatch line color="var(--c-decode)" />}>с chunked prefill</LegendItem>
        <LegendItem swatch={<Swatch dashed color="var(--ink-2)" />}>SLO ITL {SLO_MS} мс</LegendItem>
        <LegendItem swatch={<Swatch color="var(--c-prefill)" />}>шаг с prefill</LegendItem>
        <LegendItem swatch={<Swatch color="color-mix(in oklab, var(--c-decode) 55%, var(--surface-2))" />}>шаг только с decode</LegendItem>
        <LegendItem swatch={<span className="inline-block h-3.5 w-0.5 rounded-full bg-ink" />}>первый токен длинного запроса</LegendItem>
      </ul>
      {sameAsPlain && (
        <p className="mb-1 text-sm text-ink-2">Промпт вместе с decode-токенами помещается в бюджет одного шага: chunked prefill ничего не режет, обе линии одинаковые.</p>
      )}
      <div ref={wrapRef} className="scrollbar-thin -mx-1 overflow-x-auto px-1">
          <svg
            ref={svgRef}
            width={svgW}
            height={svgH}
            role="img"
            aria-label={`ITL во времени: без chunked prefill пауза до ${fmt.ms(plain.maxItl)}, с бюджетом ${fmt.int(budget)} до ${fmt.ms(chunked.maxItl)}`}
            className="block select-none"
          >
            <text x={0} y={14} style={{ font: AXIS_FONT, fill: 'var(--ink-3)' }}>
              {svgW < 760 ? 'ITL, мс' : 'ITL остальных запросов, мс'}
            </text>
            {yTicks.map((v) => (
              <g key={v}>
                <line x1={M.left} x2={svgW - M.right} y1={y(v)} y2={y(v)} style={{ stroke: 'var(--grid)', strokeWidth: 1 }} />
                <text x={M.left - 8} y={y(v)} textAnchor="end" dominantBaseline="central" style={{ ...fontCss(AXIS_FONT), fill: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>
                  {fmt.int(v)}
                </text>
              </g>
            ))}
            <line x1={M.left} x2={svgW - M.right} y1={sloY} y2={sloY} style={{ stroke: 'var(--ink-2)', strokeWidth: 1, strokeDasharray: '4 3' }} />
            {!narrowSvg && (
              <text x={svgW - M.right - 4} y={sloY - 6} textAnchor="end" style={{ font: AXIS_FONT, fill: 'var(--ink-2)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4 }}>
                SLO ITL {SLO_MS} мс
              </text>
            )}
            {/* линия прихода промпта прерывается на подписях оси, чтобы не перечёркивать «0» */}
            <line x1={arrivalX} x2={arrivalX} y1={M.top} y2={M.top + PLOT_H} style={{ stroke: 'var(--ink-2)', strokeWidth: 1, strokeDasharray: '4 3' }} />
            <line x1={arrivalX} x2={arrivalX} y1={M.top + PLOT_H + 22} y2={stripTop + 2 * (STRIP_H + STRIP_GAP)} style={{ stroke: 'var(--ink-2)', strokeWidth: 1, strokeDasharray: '4 3' }} />
            {(() => {
              const label = svgW < 460 ? 'новый промпт' : `пришёл промпт ${fmt.tokens(promptLen)}`
              const left = arrivalX - M.left > measureText(label, AXIS_FONT) + 12
              // над областью графика: там не проходят линии (подпись оси Y на узком экране короче)
              const above = !left
              return (
                <text
                  x={left ? arrivalX - 6 : arrivalX + 6}
                  y={above ? M.top - 8 : M.top + 12}
                  textAnchor={left ? 'end' : 'start'}
                  style={{ font: AXIS_FONT, fill: 'var(--ink-2)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4 }}
                >
                  {label}
                </text>
              )
            })()}
            {modes.map((m) => (
              <path key={m.key} d={path(m)} style={{ fill: 'none', stroke: m.color, strokeWidth: 2, strokeLinejoin: 'round' }} />
            ))}
            <TimeAxis x={x} y={M.top + PLOT_H} ticks={xTicks} format={(v) => (v === 0 ? '0' : msTick(v))} />
            {modes.map((m, i) => (
              <StepStrip key={m.key} mode={m} x={x} arrival={arrival} t0={t0} t1={t1} y={stripTop + i * (STRIP_H + STRIP_GAP)} narrow={narrowSvg} />
            ))}
            <text x={svgW - M.right} y={svgH - 4} textAnchor="end" style={{ font: AXIS_FONT, fill: 'var(--ink-3)' }}>
              {svgW < 460 ? 'от прихода промпта' : 'время от прихода длинного промпта'}
            </text>
            {hx !== null && <line x1={hx} x2={hx} y1={M.top} y2={stripTop + 2 * (STRIP_H + STRIP_GAP) - STRIP_GAP} style={{ stroke: 'var(--ink-3)', strokeWidth: 1 }} />}
            <rect
              x={M.left}
              y={M.top}
              width={Math.max(0, svgW - M.left - M.right)}
              height={stripTop + 2 * (STRIP_H + STRIP_GAP) - M.top}
              fill="transparent"
              onPointerMove={onMove}
              onPointerDown={onMove}
              onPointerLeave={(e) => e.pointerType === 'mouse' && clear()}
            />
          </svg>
      </div>
      {tip.node}
    </Figure>
  )
}

function StepStrip({ mode, x, arrival, t0, t1, y, narrow }: { mode: Mode; x: Scale; arrival: number; t0: number; t1: number; y: number; narrow: boolean }) {
  const [r0, r1] = x.range
  // отметка первого токена длинного запроса: штрих через полосу и подпись рядом (справа, а у края слева)
  const ft = x(mode.ttft)
  const label = narrow ? fmt.ms(mode.ttft) : `первый токен ${fmt.ms(mode.ttft)}`
  const lw = measureText(label, AXIS_FONT)
  const right = ft + 6 + lw <= r1
  return (
    <g>
      <text x={r0 - 8} y={y + STRIP_H / 2} textAnchor="end" dominantBaseline="central" style={{ font: AXIS_FONT, fill: 'var(--ink-2)' }}>
        {mode.key === 'chunked' ? 'чанки' : 'целиком'}
      </text>
      <rect x={r0} y={y} width={r1 - r0} height={STRIP_H} rx={3} style={{ fill: 'var(--surface-2)' }} />
      {mode.res.steps.map((s, i) => {
        const a = Math.max(s.tMs, t0)
        const b = Math.min(stepEnd(s), t1)
        if (b <= a) return null
        const xa = x(a - arrival)
        const xb = x(b - arrival)
        const hasPrefill = s.prefill.length > 0
        return (
          <rect
            key={i}
            x={xa}
            y={y}
            width={Math.max(0.8, xb - xa - (xb - xa > 3 ? 1 : 0))}
            height={STRIP_H}
            rx={1.5}
            style={{ fill: hasPrefill ? 'var(--c-prefill)' : 'var(--c-decode)', opacity: hasPrefill ? 1 : 0.55 }}
          />
        )
      })}
      {ft >= r0 && ft <= r1 && (
        <>
          <line x1={ft} x2={ft} y1={y - 3} y2={y + STRIP_H + 3} style={{ stroke: 'var(--ink)', strokeWidth: 2 }} />
          <text
            x={right ? ft + 6 : ft - 6}
            y={y + STRIP_H / 2}
            textAnchor={right ? 'start' : 'end'}
            dominantBaseline="central"
            style={{ font: AXIS_FONT, fill: 'var(--ink)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 4, strokeLinejoin: 'round' }}
          >
            {label}
          </text>
        </>
      )}
    </g>
  )
}
