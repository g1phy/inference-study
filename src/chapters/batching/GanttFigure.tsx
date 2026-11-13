import { memo, useMemo, useRef, useState, type PointerEvent } from 'react'
import { Badge, Figure, Player, Segmented, Toggle, TipRow, fmt, fontCss, useElementSize, usePlayer, useTip } from '@/ui'
import { linearTicks, makeScale, type Scale } from '@/ui/chart-utils'
import { useSettings } from '@/state/settings'
import { simulateContinuous, simulateStatic } from '@/engine/sched'
import type { SimRequest } from '@/engine/types'
import { configLabel, perfOf, simConfig } from './config'
import { avg, idleShare, phaseCounts, requestTracks, type RequestTrack } from './timeline'
import { AXIS_FONT, GridLines, HatchPattern, LegendItem, Metric, MONO_FONT, Swatch, TimeAxis, useSvgId } from './chart'

type Scenario = 'mixed' | 'offline'

/** Поток из восьми запросов (время эталона: Qwen3-8B на H100). Длины ответов нарочно разные. */
const IDS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
const ARRIVALS = [0, 30, 60, 90, 170, 230, 280, 340]
const PROMPTS = [420, 760, 300, 610, 520, 380, 700, 450]
const OUTPUTS = [34, 8, 26, 12, 10, 38, 16, 22]

function buildRequests(scn: Scenario, timeScale: number): SimRequest[] {
  return IDS.map((id, i) => ({
    id,
    arrivalMs: scn === 'offline' ? 0 : ARRIVALS[i] * timeScale,
    promptTokens: PROMPTS[i],
    outputTokens: scn === 'offline' ? 20 : OUTPUTS[i],
  }))
}

const FRAMES = 64
/** Ширина колонки подписей и правое поле под E2E (компактнее на узком экране). */
function layout(width: number) {
  const narrow = width > 0 && width < 520
  return { narrow, labelW: narrow ? 64 : 72, rightPad: narrow ? 50 : 76 }
}

/** Синяя штриховка: пустые токены, которыми static дополняет короткие промпты. */
const PAD_HATCH_CSS = 'repeating-linear-gradient(45deg, var(--c-prefill) 0 1.5px, var(--surface-2) 1.5px 4.5px)'
const ROW_H = 18
const ROW_PITCH = 25
const TOP = 6

interface PanelModel {
  key: 'static' | 'continuous'
  title: string
  tracks: RequestTrack[]
  ttft: number
  e2e: number
  makespan: number
  idle: number
}

export function GanttFigure() {
  const [settings] = useSettings()
  const [scenario, setScenario] = useState<Scenario>('mixed')
  const [batchSize, setBatchSize] = useState(4)
  const [dynamic, setDynamic] = useState(false)
  const cfg = simConfig(settings.modelId, settings.gpuId)

  const model = useMemo(() => {
    const reqs = buildRequests(scenario, cfg.timeScale)
    const base = perfOf(cfg)
    const st = simulateStatic(reqs, { ...base, batchSize, maxWaitMs: dynamic ? 100 * cfg.timeScale : undefined })
    const ct = simulateContinuous(reqs, base)
    const mk = (key: PanelModel['key'], title: string, res: typeof st): PanelModel => {
      const tracks = requestTracks(res, { static: key === 'static' })
      return {
        key,
        title,
        tracks,
        ttft: avg(res.requests.map((r) => r.ttftMs)),
        e2e: avg(res.requests.map((r) => r.e2eMs)),
        makespan: Math.max(...res.requests.map((r) => r.finishMs)),
        idle: key === 'static' ? idleShare(tracks) : 0,
      }
    }
    const panels = [
      mk('static', dynamic ? `Dynamic batching: batch ${batchSize}, ожидание не дольше ${fmt.ms(100 * cfg.timeScale)}` : `Static batching: batch ${batchSize}`, st),
      mk('continuous', 'Continuous batching', ct),
    ]
    const tMax = Math.max(...panels.map((p) => p.makespan)) * 1.02
    return { panels, tMax }
  }, [scenario, batchSize, dynamic, cfg])

  const player = usePlayer(FRAMES, { intervalMs: 190, initialStep: FRAMES - 1 })
  const playT = (model.tMax * player.step) / (FRAMES - 1)
  const atEnd = player.step === FRAMES - 1

  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  const lay = layout(width)
  const svgW = Math.max(300, width || 960)
  // шкала стабильна между кадрами плеера, иначе memo-слои перерисовывались бы на каждом кадре
  const x = useMemo(() => makeScale('linear', [0, model.tMax], [lay.labelW, svgW - lay.rightPad]), [model.tMax, lay.labelW, lay.rightPad, svgW])
  const ticks = useMemo(
    () => linearTicks(0, model.tMax, Math.max(2, Math.floor((svgW - lay.labelW - lay.rightPad) / 100))),
    [model.tMax, lay.labelW, lay.rightPad, svgW],
  )

  const [st, ct] = model.panels
  const better = st.e2e > 0 ? st.e2e / Math.max(1, ct.e2e) : 1

  return (
    <Figure
      id="batching-figure-gantt"
      title="Static и continuous batching на одном потоке запросов"
      actions={<Badge>{configLabel(cfg)}</Badge>}
      player={
        // кадры здесь это время, а «шаг» в главе значит проход модели: счётчик показывает время, а не «Шаг N из 64»
        <Player {...player.bind} aria-label="Время на таймлайне" counter={(s) => timeLabel(s, model.tMax, model.tMax / 1.02)} />
      }
      controls={
        <>
          <Segmented<Scenario>
            label="Поток запросов"
            value={scenario}
            onChange={setScenario}
            options={[
              { value: 'mixed', label: 'Онлайн' },
              { value: 'offline', label: 'Офлайн' },
            ]}
          />
          <Segmented<number>
            label="Размер batch для static"
            value={batchSize}
            onChange={setBatchSize}
            options={[2, 4, 8].map((v) => ({ value: v, label: String(v) }))}
          />
          <Toggle
            label="Dynamic batching"
            checked={dynamic}
            onChange={setDynamic}
            hint="Static, но не ждать полный batch дольше таймера"
          />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> нажмите пуск и посмотрите, как static ждёт четвёртый запрос, пока continuous уже
          отвечает. При batch 8 static ждёт все восемь; включите dynamic batching, и он стартует по таймеру, но пустые слоты никуда не деваются.
          Static к тому же дополняет все промпты batch до самого длинного (padding): лишние токены считаются впустую, поэтому его prefill дольше даже
          на тех же запросах. В режиме «Офлайн» с batch 8 (все пришли сразу, ответы одной длины) static отстаёт только из-за этого выравнивания. Время
          посчитано по roofline-модели для {configLabel(cfg)}; длины запросов условные.
        </>
      }
    >
      <ul className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        <LegendItem swatch={<Swatch dashed />}>ожидание в очереди</LegendItem>
        <LegendItem swatch={<Swatch color="var(--c-prefill)" />}>prefill</LegendItem>
        <LegendItem swatch={<Swatch color="var(--c-decode)" />}>decode, токен за шаг</LegendItem>
        <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px] shadow-[inset_0_0_0_1px_var(--line)]" style={{ background: PAD_HATCH_CSS }} />}>
          padding: пустые токены до длины самого длинного промпта
        </LegendItem>
        <LegendItem swatch={<Swatch hatch />}>слот занят, но ответ уже готов</LegendItem>
        <LegendItem swatch={<span className="font-mono text-[11px] leading-none text-ink-2">▾</span>}>приход запроса</LegendItem>
        <LegendItem swatch={<span className="text-xs leading-none text-ink-2 tabular-nums">99 мс</span>}>в конце строки: E2E, от прихода до последнего токена</LegendItem>
      </ul>
      <div ref={wrapRef} className="grid min-w-0 gap-4">
        {model.panels.map((p) => (
          <GanttPanel key={p.key} panel={p} x={x} ticks={ticks} width={svgW} playT={playT} reveal={!atEnd} narrow={lay.narrow} />
        ))}
      </div>
      <p className="mt-3 text-sm text-ink-2" aria-live={player.playing ? 'off' : 'polite'}>
        {atEnd ? (
          <>
            Continuous batching закончил весь поток за {fmt.ms(ct.makespan)} против {fmt.ms(st.makespan)}, средний E2E{' '}
            {better > 1.05 ? `в ${fmt.num(better, 1)} раза меньше` : 'почти такой же'}.
          </>
        ) : (
          <>
            <span className="font-medium text-ink tabular-nums">{fmt.ms(playT)}.</span> {narrate(st.tracks, playT, 'static', batchSize)}{' '}
            {narrate(ct.tracks, playT, 'continuous', batchSize)}
          </>
        )}
      </p>
    </Figure>
  )
}

/** Отрезки всех запросов панели: memo, чтобы наведение и плеер не перерисовывали сотни прямоугольников. */
const GanttRows = memo(function GanttRows({ tracks, x, hatchId, padId, faded = false }: { tracks: RequestTrack[]; x: Scale; hatchId: string; padId: string; faded?: boolean }) {
  return (
    <>
      {tracks.map((tr, i) => {
        const yMid = TOP + i * ROW_PITCH + ROW_H / 2
        const y0 = TOP + i * ROW_PITCH
        return (
          <g key={tr.id} opacity={faded ? 0.16 : 1}>
            {tr.segments.map((s, k) => {
              const a = x(s.t0)
              const b = x(s.t1)
              if (s.kind === 'wait')
                return <line key={k} x1={a} x2={b} y1={yMid} y2={yMid} style={{ stroke: 'var(--ink-3)', strokeWidth: 1.5, strokeDasharray: '3 3' }} />
              if (s.kind === 'idle')
                return (
                  <rect key={k} x={a} y={y0 + 2} width={Math.max(0, b - a)} height={ROW_H - 4} rx={2} fill={`url(#${hatchId})`} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
                )
              const w = Math.max(1, b - a - (s.kind === 'decode' && b - a > 3 ? 1 : 0))
              if (s.kind === 'prefill' && s.padTo && s.tokens) {
                // доля настоящих токенов промпта, остальное дополнение до самого длинного в batch
                const real = Math.max(1, w * (s.tokens / s.padTo))
                return (
                  <g key={k}>
                    <rect x={a} y={y0} width={real} height={ROW_H} rx={3} style={{ fill: 'var(--c-prefill)' }} />
                    <rect x={a + real} y={y0 + 1} width={Math.max(0, w - real)} height={ROW_H - 2} rx={2} fill={`url(#${padId})`} style={{ stroke: 'var(--c-prefill)', strokeWidth: 1 }} />
                  </g>
                )
              }
              return (
                <rect
                  key={k}
                  x={a}
                  y={y0}
                  width={w}
                  height={ROW_H}
                  rx={s.kind === 'prefill' ? 3 : 1}
                  style={{ fill: s.kind === 'prefill' ? 'var(--c-prefill)' : 'var(--c-decode)' }}
                />
              )
            })}
            <path d={`M${x(tr.stats.arrivalMs)},${y0 - 1} l-3.5,-4 h7 z`} style={{ fill: 'var(--ink-2)' }} />
          </g>
        )
      })}
    </>
  )
})

/** Счётчик плеера временем: «120 мс из 777 мс»; total: когда готов весь поток (ось длиннее на 2 %). */
function timeLabel(k: number, tMax: number, total: number): string {
  return k === FRAMES - 1 ? 'весь поток' : `${fmt.ms(Math.min(total, (tMax * k) / (FRAMES - 1)))} из ${fmt.ms(total)}`
}

function narrate(tracks: RequestTrack[], t: number, mode: 'static' | 'continuous', batch: number): string {
  const c = phaseCounts(tracks, t)
  if (mode === 'static') {
    if (c.work === 0 && c.idle === 0 && c.wait > 0) return `Static: ${c.wait} в очереди, batch из ${batch} ещё не набран.`
    const parts = [`в работе ${c.work}`]
    if (c.idle) parts.push(`пустых слотов ${c.idle}`)
    if (c.wait) parts.push(`в очереди ${c.wait}`)
    parts.push(`готово ${c.done}`)
    return `Static: ${parts.join(', ')}.`
  }
  const parts = [`в работе ${c.work}`]
  if (c.wait) parts.push(`в очереди ${c.wait}`)
  parts.push(`готово ${c.done}`)
  return `Continuous: ${parts.join(', ')}.`
}

function GanttPanel({
  panel,
  x,
  ticks,
  width,
  playT,
  reveal,
  narrow,
}: {
  panel: PanelModel
  x: Scale
  ticks: number[]
  width: number
  playT: number
  reveal: boolean
  narrow: boolean
}) {
  const hatchId = useSvgId('gantt-hatch')
  const padId = useSvgId('gantt-pad')
  const clipId = useSvgId('gantt-clip')
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const tip = useTip({ scopeEl: svgRef.current, side: 'top', onOutside: () => setHover(null) })
  const rows = panel.tracks.length
  const plotBottom = TOP + rows * ROW_PITCH
  const h = plotBottom + 40
  const px = x(Math.min(playT, x.domain[1]))

  const showRow = (i: number, e: PointerEvent<SVGRectElement>) => {
    const tr = panel.tracks[i]
    setHover(i)
    const idle = tr.segments.filter((s) => s.kind === 'idle').reduce((a, s) => a + s.t1 - s.t0, 0)
    const padded = tr.segments.find((s) => s.kind === 'prefill' && s.padTo)
    const box = e.currentTarget.getBoundingClientRect()
    const anchorX = Math.min(Math.max(e.clientX, box.left), box.right)
    tip.show(
      { left: anchorX, top: box.top, width: 0, height: box.height },
      <div className="grid gap-1">
        <div className="font-semibold text-ink">
          Запрос {tr.id}: промпт {fmt.int(tr.stats.promptTokens)}, ответ {fmt.tokens(tr.stats.outputTokens)}
        </div>
        <TipRow color="var(--c-prefill)" value={fmt.ms(tr.stats.ttftMs)} label="TTFT, от прихода до первого токена" />
        <TipRow color="var(--c-decode)" value={fmt.ms(tr.stats.e2eMs)} label="E2E, до последнего токена" />
        {tr.stats.queueMs > 0.5 && <TipRow value={fmt.ms(tr.stats.queueMs)} label="ждал в очереди" />}
        {padded?.padTo && <TipRow value={fmt.tokens(padded.padTo)} label={`промпт дополнен до длины самого длинного в batch (свой ${fmt.int(padded.tokens ?? 0)})`} />}
        {idle > 0.5 && <TipRow value={fmt.ms(idle)} label="слот простаивал после ответа" />}
      </div>,
    )
  }
  const clear = () => {
    setHover(null)
    tip.hide()
  }

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
        <div className="text-[15px] font-semibold text-ink">{panel.title}</div>
        <Metric label="Средний TTFT" value={fmt.ms(panel.ttft)} />
        <Metric label="Средний E2E" value={fmt.ms(panel.e2e)} />
        <Metric label="Весь поток готов" value={fmt.ms(panel.makespan)} />
        {panel.key === 'static' && <Metric label="Простой слотов" value={fmt.pct(panel.idle)} />}
      </div>
      <svg
        ref={svgRef}
        width={width}
        height={h}
        role="img"
        aria-label={`${panel.title}: средний TTFT ${fmt.ms(panel.ttft)}, средний E2E ${fmt.ms(panel.e2e)}, весь поток готов за ${fmt.ms(panel.makespan)}`}
        className="block select-none"
      >
        <defs>
          <HatchPattern id={hatchId} />
          <HatchPattern id={padId} tone="c-prefill" />
          <clipPath id={clipId}>
            <rect x={0} y={0} width={px} height={h} />
          </clipPath>
        </defs>
        <GridLines x={x} ticks={ticks} y0={TOP - 6} y1={plotBottom} />
        {panel.tracks.map((tr, i) => (
          <g key={tr.id}>
            {hover === i && <rect x={0} y={TOP + i * ROW_PITCH - 3} width={width} height={ROW_H + 6} rx={4} style={{ fill: 'var(--surface-3)', opacity: 0.6 }} />}
            <text x={4} y={TOP + i * ROW_PITCH + ROW_H / 2} dominantBaseline="central" style={{ font: MONO_FONT, fill: 'var(--ink)' }}>
              {tr.id}
            </text>
            <text
              key={narrow ? 'n' : 'w'}
              x={18}
              y={TOP + i * ROW_PITCH + ROW_H / 2}
              dominantBaseline="central"
              style={{ ...fontCss(narrow ? '11px "Onest Variable", sans-serif' : AXIS_FONT), fill: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}
            >
              {tr.stats.outputTokens} ток.
            </text>
          </g>
        ))}
        {reveal ? (
          <>
            <GanttRows tracks={panel.tracks} x={x} hatchId={hatchId} padId={padId} faded />
            <g clipPath={`url(#${clipId})`}>
              <GanttRows tracks={panel.tracks} x={x} hatchId={hatchId} padId={padId} />
            </g>
          </>
        ) : (
          <GanttRows tracks={panel.tracks} x={x} hatchId={hatchId} padId={padId} />
        )}
        {!reveal &&
          panel.tracks.map((tr, i) => {
            const end = tr.batchEnd !== undefined ? Math.max(tr.batchEnd, tr.stats.finishMs) : tr.stats.finishMs
            const lx = x(end) + 6
            return (
              <text
                key={tr.id}
                x={lx}
                y={TOP + i * ROW_PITCH + ROW_H / 2}
                dominantBaseline="central"
                style={{ ...fontCss(AXIS_FONT), fill: 'var(--ink-2)', fontVariantNumeric: 'tabular-nums' }}
              >
                {fmt.ms(tr.stats.e2eMs)}
              </text>
            )
          })}
        {reveal && <line x1={px} x2={px} y1={TOP - 6} y2={plotBottom + 2} style={{ stroke: 'var(--accent)', strokeWidth: 1.5 }} />}
        <TimeAxis x={x} y={plotBottom + 4} ticks={ticks} label={narrow ? 'время' : 'время от первого запроса'} labelX={width - 4} />
        {panel.tracks.map((tr, i) => (
          <rect
            key={tr.id}
            x={0}
            y={TOP + i * ROW_PITCH - 3}
            width={width}
            height={ROW_PITCH}
            fill="transparent"
            onPointerMove={(e) => showRow(i, e)}
            onPointerDown={(e) => showRow(i, e)}
            onPointerLeave={(e) => e.pointerType === 'mouse' && clear()}
          />
        ))}
      </svg>
      {tip.node}
    </div>
  )
}
