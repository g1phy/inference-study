import { memo, useMemo, useRef, useState, type PointerEvent, type ReactNode } from 'react'
import { Badge, Figure, Segmented, TipRow, fmt, measureText, useElementSize, useFontsReady, useTip } from '@/ui'
import { cn } from '@/ui/cn'
import { linearTicks, makeScale, type Scale } from '@/ui/chart-utils'
import { useSettings } from '@/state/settings'
import type { SimResultExt, SimStepRecord } from '@/engine/sched'
import { configLabel, simConfig, type SimConfig } from '../batching/config'
import { itlTimeline, stepEnd, type ItlPoint } from '../batching/timeline'
import { AXIS_FONT, GridLines, LegendItem, Metric, Swatch, TimeAxis } from '../batching/chart'
import { STALL_MS, linkFor, nvlinkFor, runLanes, type LinkChoice } from './logic'

const RIGHT = 12
const LANE_H = 20
const ITL_H = 26
const KV_H = 14
const SLO = 50

interface Lane {
  key: string
  label: string
  kind: 'gpu' | 'itl' | 'kv'
  worker: string
  y: number
  h: number
}

interface Section {
  key: 'agg' | 'disagg'
  title: string
  res: SimResultExt
  lanes: Lane[]
  height: number
  itlByWorker: Map<string, ItlPoint[]>
}

function buildSection(key: Section['key'], title: string, res: SimResultExt, cfg: SimConfig): Section {
  const lanes: Lane[] = []
  let y = 0
  const gpuName = (i: number) => (cfg.tp > 1 ? `Воркер ${i + 1}` : `GPU ${i + 1}`)
  if (key === 'agg') {
    for (let i = 0; i < 2; i++) {
      lanes.push({ key: `g${i}`, label: gpuName(i), kind: 'gpu', worker: `agg-${i}`, y, h: LANE_H })
      y += LANE_H + 4
      lanes.push({ key: `i${i}`, label: 'ITL', kind: 'itl', worker: `agg-${i}`, y, h: ITL_H })
      y += ITL_H + 14
    }
  } else {
    lanes.push({ key: 'p', label: 'Prefill', kind: 'gpu', worker: 'prefill-0', y, h: LANE_H })
    y += LANE_H + 6
    lanes.push({ key: 'kv', label: 'Канал KV', kind: 'kv', worker: 'link', y, h: KV_H })
    y += KV_H + 6
    lanes.push({ key: 'd', label: 'Decode', kind: 'gpu', worker: 'decode-0', y, h: LANE_H })
    y += LANE_H + 4
    lanes.push({ key: 'id', label: 'ITL', kind: 'itl', worker: 'decode-0', y, h: ITL_H })
    y += ITL_H + 14
  }
  const itlByWorker = new Map<string, ItlPoint[]>()
  for (const p of itlTimeline(res)) {
    const w = res.steps[p.step].worker ?? ''
    if (!itlByWorker.has(w)) itlByWorker.set(w, [])
    itlByWorker.get(w)!.push(p)
  }
  return { key, title, res, lanes, height: y, itlByWorker }
}

export function LanesFigure() {
  const [settings] = useSettings()
  const cfg = simConfig(settings.modelId, settings.gpuId)
  // по умолчанию RDMA: в Kubernetes KV так едет даже между подами одной ноды (NVLink между подами закрыт)
  const [link, setLink] = useState<LinkChoice>('rdma')
  const data = useMemo(() => {
    const runs = runLanes(cfg, link)
    const agg = buildSection(
      'agg',
      cfg.tp > 1 ? `Агрегированный: 2 воркера (TP${cfg.tp}), каждый делает и prefill, и decode` : 'Агрегированный: 2 GPU, каждый делает и prefill, и decode',
      runs.agg,
      cfg,
    )
    const disagg = buildSection(
      'disagg',
      cfg.tp > 1 ? `Дезагрегированный: prefill-воркер и decode-воркер (TP${cfg.tp})` : 'Дезагрегированный: 1 prefill GPU и 1 decode GPU',
      runs.disagg,
      cfg,
    )
    const tMax = Math.max(...[runs.agg, runs.disagg].flatMap((r) => r.requests.map((q) => q.finishMs))) * 1.01
    let itlMax = SLO * 1.3
    for (const s of [agg, disagg]) for (const pts of s.itlByWorker.values()) for (const p of pts) itlMax = Math.max(itlMax, p.itl)
    // по умолчанию выделен запрос, который сильнее всего «подвис» в агрегированном режиме
    let worst = runs.agg.requests[0]
    for (const r of runs.agg.requests) if (Math.max(0, ...r.itlMs) > Math.max(0, ...worst.itlMs)) worst = r
    return { runs, agg, disagg, tMax, itlMax, worstId: worst.id }
  }, [cfg, link])
  const [picked, setPicked] = useState<string | null>(null)
  const sel = picked ?? data.worstId

  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  useFontsReady() // подпись самой длинной паузы ставится по ширине текста
  const narrow = width > 0 && width < 560
  const labelW = narrow ? 62 : 96
  const svgW = Math.max(300, width || 960)
  const x = makeScale('linear', [0, data.tMax], [labelW, svgW - RIGHT])
  const ticks = linearTicks(0, data.tMax, Math.max(2, Math.floor((svgW - labelW) / 100)))
  const yItl = (v: number, lane: Lane) => lane.y + lane.h - (Math.min(v, data.itlMax) / data.itlMax) * (lane.h - 2)

  const reqAgg = data.runs.agg.requests.find((r) => r.id === sel)!
  const reqDis = data.runs.disagg.requests.find((r) => r.id === sel)!
  const trSel = data.runs.disagg.transfers.find((t) => t.reqId === sel)
  const maxItl = (itl: number[]) => (itl.length ? Math.max(...itl) : 0)
  const nv = nvlinkFor(cfg.gpu)
  const linkInfo = linkFor(link, cfg.gpu, cfg.tp)

  return (
    <Figure
      id="disaggregation-figure-lanes"
      title="Один поток запросов: агрегированный и дезагрегированный режим"
      actions={<Badge>{configLabel(cfg)}</Badge>}
      controls={
        <>
          <Segmented<LinkChoice>
            label="Канал для KV cache"
            value={link}
            onChange={setLink}
            options={[
              { value: 'rdma', label: 'RDMA 400G' },
              { value: 'nvlink', label: nv.id.startsWith('nvlink') ? nv.name.split(' (')[0] : 'PCIe', title: 'если воркеры видят GPU друг друга' },
              { value: 'tcp', label: 'TCP 25G' },
            ]}
          />
          <div className="grid gap-1.5">
            <div className="text-sm font-medium text-ink">Запрос</div>
            <div className="flex flex-wrap gap-1" role="group" aria-label="Выделить запрос">
              {data.runs.agg.requests.map((r) => (
                <button
                  key={r.id}
                  type="button"
                  aria-pressed={r.id === sel}
                  onClick={() => setPicked(r.id)}
                  title={`${r.id}: промпт ${fmt.tokens(r.promptTokens)}, ответ ${fmt.tokens(r.outputTokens)}`}
                  className={cn(
                    'h-7 rounded-chip px-1.5 font-mono text-[12px] tabular-nums transition-colors',
                    r.id === sel ? 'bg-accent text-accent-ink' : 'bg-surface-2 text-ink-2 shadow-[inset_0_0_0_1px_var(--line)] hover:bg-surface-3',
                  )}
                >
                  {r.id}
                  <span className={cn('ml-1', r.id === sel ? 'opacity-90' : 'text-ink-3')}>{fmt.compact(r.promptTokens)}</span>
                </button>
              ))}
            </div>
          </div>
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> выберите r5 (промпт 12 тысяч токенов) и посмотрите, кто ждёт, пока считается
          его prefill: в агрегированном режиме пауза достаётся всем, кто генерирует на том же {cfg.tp > 1 ? 'воркере' : 'GPU'}. Канал по умолчанию
          RDMA: так KV едет в Kubernetes
          даже внутри одной ноды. Переключите канал на TCP: передача KV растягивается на сотни миллисекунд, и TTFT каждого запроса вырастает в разы.
          ITL остаётся ровным, но такой ценой первого токена дезагрегация уже не окупается. Время по roofline-модели, поток условный; порог ITL здесь{' '}
          {SLO} мс, в сравнении ниже {STALL_MS} мс.
        </>
      }
    >
      <ul className="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        <LegendItem swatch={<Swatch color="var(--c-prefill)" />}>шаг с prefill</LegendItem>
        <LegendItem swatch={<Swatch color="var(--c-decode)" />}>шаг только с decode</LegendItem>
        <LegendItem swatch={<Swatch line color="var(--c-decode)" />}>ITL: пауза перед токеном</LegendItem>
        <LegendItem swatch={<Swatch color="var(--ink-2)" />}>передача KV</LegendItem>
        <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px] border-2 border-accent" />}>выбранный запрос</LegendItem>
        <LegendItem swatch={<span className="inline-block w-4 border-t-[2.5px] border-accent" />}>под дорожкой: decode выбранного запроса</LegendItem>
        <LegendItem swatch={<Swatch dashed />}>пунктир: SLO ITL {SLO} мс (условный)</LegendItem>
      </ul>
      <div ref={wrapRef} className="grid min-w-0 gap-5">
        {[data.agg, data.disagg].map((sec) => (
          <SectionView
            key={sec.key}
            sec={sec}
            x={x}
            ticks={ticks}
            width={svgW}
            labelW={labelW}
            narrow={narrow}
            sel={sel}
            yItl={yItl}
            itlMax={data.itlMax}
            linkName={linkInfo.spec.name}
            tp={cfg.tp}
          />
        ))}
      </div>
      <div className="mt-4 grid gap-3 rounded-panel bg-surface-2 p-3.5 text-sm @lg:grid-cols-[auto_1fr_1fr] @lg:items-start">
        <div className="font-medium text-ink">
          {sel}: промпт {fmt.tokens(reqAgg.promptTokens)}
        </div>
        <SelBlock title="Агрегированный">
          <Metric label="TTFT" value={fmt.ms(reqAgg.ttftMs)} />
          <Metric label="самая длинная пауза" value={fmt.ms(maxItl(reqAgg.itlMs))} />
        </SelBlock>
        <SelBlock title="Дезагрегированный">
          <Metric label="TTFT" value={fmt.ms(reqDis.ttftMs)} />
          {trSel && <Metric label="из них передача KV" value={`${fmt.ms(trSel.endMs - trSel.startMs)} (${fmt.bytes(trSel.bytes)})`} />}
          <Metric label="самая длинная пауза" value={fmt.ms(maxItl(reqDis.itlMs))} />
        </SelBlock>
      </div>
    </Figure>
  )
}

function SelBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="grid gap-0.5">
      <div className="text-ink-2">{title}</div>
      {children}
    </div>
  )
}

function stepLabel(s: SimStepRecord): string {
  const pre = s.prefill.reduce((a, p) => a + p.tokens, 0)
  const parts: string[] = []
  if (pre) parts.push(`prefill ${s.prefill.map((p) => `${p.reqId} ${fmt.int(p.tokens)}`).join(', ')}`)
  if (s.decode.length) parts.push(`decode ${s.decode.length} запр.`)
  return parts.join(' + ')
}

function SectionView({
  sec,
  x,
  ticks,
  width,
  labelW: LABEL_W,
  narrow,
  sel,
  yItl,
  itlMax,
  linkName,
  tp,
}: {
  sec: Section
  x: Scale
  ticks: number[]
  width: number
  labelW: number
  narrow: boolean
  sel: string
  yItl: (v: number, lane: Lane) => number
  itlMax: number
  linkName: string
  tp: number
}) {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const tip = useTip({ scopeEl: svgRef.current, side: 'top' })
  const s = sec.res.summary
  const TOP = 4
  const axisY = TOP + sec.height
  const h = axisY + 24

  const stepsBy = useMemo(() => {
    const m = new Map<string, SimStepRecord[]>()
    for (const st of sec.res.steps) {
      const w = st.worker ?? ''
      if (!m.has(w)) m.set(w, [])
      m.get(w)!.push(st)
    }
    return m
  }, [sec])

  const onMove = (e: PointerEvent<SVGRectElement>, lane: Lane) => {
    const box = svgRef.current!.getBoundingClientRect()
    const t = x.invert(e.clientX - box.left)
    const anchor = { left: e.clientX, top: box.top + TOP + lane.y, width: 0, height: lane.h }
    if (lane.kind === 'kv') {
      const tr = sec.res.transfers.find((q) => t >= q.startMs - 4 && t <= q.endMs + 4)
      if (!tr) return tip.hide()
      tip.show(
        anchor,
        <div className="grid gap-1">
          <div className="font-semibold text-ink">KV cache запроса {tr.reqId}</div>
          <TipRow value={fmt.bytes(tr.bytes)} label={`по ${linkName.split(' (')[0]}${tp > 1 ? `, ${tp} канала параллельно` : ''}`} />
          <TipRow value={fmt.ms(tr.endMs - tr.startMs)} label="длится передача" />
        </div>,
      )
      return
    }
    const steps = stepsBy.get(lane.worker) ?? []
    const st = steps.find((q) => t >= q.tMs && t < stepEnd(q))
    if (!st) return tip.hide()
    if (lane.kind === 'itl') {
      const pts = sec.itlByWorker.get(lane.worker) ?? []
      const p = pts.find((q) => t >= q.t - q.itl && t <= q.t)
      if (!p) return tip.hide()
      tip.show(anchor, <TipRow color="var(--c-decode)" value={fmt.ms(p.itl)} label={`ждали следующий токен ${p.count} запр.`} />)
      return
    }
    tip.show(
      anchor,
      <div className="grid gap-1">
        <div className="font-semibold text-ink">
          {lane.label}: шаг {fmt.ms(st.durationMs)}
        </div>
        <div className="text-ink-2">{stepLabel(st)}</div>
      </div>,
    )
  }

  const selPrefillSteps = sec.res.steps.filter((st) => st.prefill.some((p) => p.reqId === sel))
  const selReq = sec.res.requests.find((r) => r.id === sel)!
  const selTr = sec.res.transfers.find((t) => t.reqId === sel)

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
        <div className="text-[15px] font-semibold text-ink">{sec.title}</div>
        <Metric label="TTFT p50 / p99" value={`${fmt.ms(s.ttftP50)} / ${fmt.ms(s.ttftP99)}`} />
        <Metric label="ITL p50 / p99" value={`${fmt.ms(s.itlP50)} / ${fmt.ms(s.itlP99)}`} />
      </div>
      <svg ref={svgRef} width={width} height={h} role="img" aria-label={`${sec.title}. TTFT p50 ${fmt.ms(s.ttftP50)}, ITL p99 ${fmt.ms(s.itlP99)}`} className="block select-none">
        <g transform={`translate(0, ${TOP})`}>
          <GridLines x={x} ticks={ticks} y0={-2} y1={sec.height - 10} />
          {sec.lanes.map((lane) => (
            <g key={lane.key}>
              <text
                x={lane.kind === 'gpu' ? 0 : 12}
                y={lane.y + lane.h / 2}
                dominantBaseline="central"
                style={{ font: lane.kind === 'gpu' ? '500 13px "Onest Variable", sans-serif' : AXIS_FONT, fill: lane.kind === 'gpu' ? 'var(--ink)' : 'var(--ink-3)' }}
              >
                {narrow && lane.kind === 'kv' ? 'KV' : lane.label}
              </text>
              {lane.kind === 'gpu' && (
                <>
                  <rect x={LABEL_W} y={lane.y} width={width - LABEL_W - RIGHT} height={lane.h} rx={3} style={{ fill: 'var(--surface-2)' }} />
                  <StepRects steps={stepsBy.get(lane.worker) ?? EMPTY_STEPS} x={x} y={lane.y} h={lane.h} />
                  {selPrefillSteps
                    .filter((st) => st.worker === lane.worker)
                    .map((st, i) => (
                      <rect
                        key={`sel${i}`}
                        x={x(st.tMs) - 1}
                        y={lane.y - 2}
                        width={Math.max(3, x(stepEnd(st)) - x(st.tMs) + 2)}
                        height={lane.h + 4}
                        rx={3}
                        style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 2 }}
                      />
                    ))}
                  {selReq.worker === lane.worker && selReq.finishMs > selReq.firstTokenMs && (
                    <line
                      x1={x(selReq.firstTokenMs)}
                      x2={x(selReq.finishMs)}
                      y1={lane.y + lane.h + 2}
                      y2={lane.y + lane.h + 2}
                      style={{ stroke: 'var(--accent)', strokeWidth: 2.5, strokeLinecap: 'round' }}
                    />
                  )}
                </>
              )}
              {lane.kind === 'itl' && (
                <>
                  <line x1={LABEL_W} x2={width - RIGHT} y1={lane.y + lane.h} y2={lane.y + lane.h} style={{ stroke: 'var(--line-strong)', strokeWidth: 1 }} />
                  <line x1={LABEL_W} x2={width - RIGHT} y1={yItl(SLO, lane)} y2={yItl(SLO, lane)} style={{ stroke: 'var(--ink-3)', strokeWidth: 1, strokeDasharray: '3 3' }} />
                  {narrow ? (
                    <text
                      x={width - RIGHT - 2}
                      y={yItl(SLO, lane) - 3}
                      textAnchor="end"
                      style={{ font: '11px "Onest Variable", sans-serif', fill: 'var(--ink-3)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}
                    >
                      {SLO} мс
                    </text>
                  ) : (
                    <text x={LABEL_W - 6} y={yItl(SLO, lane)} textAnchor="end" dominantBaseline="central" style={{ font: '11px "Onest Variable", sans-serif', fill: 'var(--ink-3)' }}>
                      {SLO} мс
                    </text>
                  )}
                  <path d={itlPath(sec.itlByWorker.get(lane.worker) ?? [], x, (v) => yItl(v, lane))} style={{ fill: 'none', stroke: 'var(--c-decode)', strokeWidth: 1.6, strokeLinejoin: 'round' }} />
                  {maxLabel(sec.itlByWorker.get(lane.worker) ?? [], itlMax, x, lane, yItl, width - RIGHT)}
                </>
              )}
              {lane.kind === 'kv' && (
                <>
                  <line x1={LABEL_W} x2={width - RIGHT} y1={lane.y + lane.h / 2} y2={lane.y + lane.h / 2} style={{ stroke: 'var(--line-strong)', strokeWidth: 1, strokeDasharray: '2 3' }} />
                  {sec.res.transfers.map((tr) => {
                    const a = x(tr.startMs)
                    const b = Math.max(a + 2.5, x(tr.endMs))
                    const isSel = tr.reqId === sel
                    return (
                      <g key={tr.reqId}>
                        <line x1={a} x2={a} y1={lane.y - 6} y2={lane.y + 1} style={{ stroke: 'var(--ink-3)', strokeWidth: 1 }} />
                        <line x1={b} x2={b} y1={lane.y + lane.h - 1} y2={lane.y + lane.h + 6} style={{ stroke: 'var(--ink-3)', strokeWidth: 1 }} />
                        <rect
                          x={a}
                          y={lane.y + 2}
                          width={b - a}
                          height={lane.h - 4}
                          rx={2}
                          style={{ fill: 'var(--ink-2)', stroke: isSel ? 'var(--accent)' : 'none', strokeWidth: isSel ? 2 : 0 }}
                        />
                      </g>
                    )
                  })}
                  {selTr && (
                    <rect
                      x={x(selTr.startMs) - 3}
                      y={lane.y - 1}
                      width={Math.max(6, x(selTr.endMs) - x(selTr.startMs) + 6)}
                      height={lane.h + 2}
                      rx={4}
                      style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 2 }}
                    />
                  )}
                </>
              )}
              <rect
                x={LABEL_W}
                y={lane.y - 3}
                width={width - LABEL_W - RIGHT}
                height={lane.h + 6}
                fill="transparent"
                onPointerMove={(e) => onMove(e, lane)}
                onPointerDown={(e) => onMove(e, lane)}
                onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
              />
            </g>
          ))}
          <TimeAxis x={x} y={sec.height - 6} ticks={ticks} />
        </g>
      </svg>
      {tip.node}
    </div>
  )
}

const EMPTY_STEPS: SimStepRecord[] = []

/** Шаги одной дорожки (сотни прямоугольников): memo, чтобы наведение мыши их не перерисовывало. */
const StepRects = memo(function StepRects({ steps, x, y, h }: { steps: SimStepRecord[]; x: Scale; y: number; h: number }) {
  return (
    <>
      {steps.map((st, i) => {
        const a = x(st.tMs)
        const b = x(stepEnd(st))
        const pre = st.prefill.length > 0
        return (
          <rect
            key={i}
            x={a}
            y={y}
            width={Math.max(0.6, b - a - (b - a > 3 ? 0.8 : 0))}
            height={h}
            style={{ fill: pre ? 'var(--c-prefill)' : 'var(--c-decode)', opacity: pre ? 1 : 0.75 }}
          />
        )
      })}
    </>
  )
})

/** ITL ступеньками: каждый токен рисуется плато от предыдущего токена до своего на высоте своей паузы. */
function itlPath(points: ItlPoint[], x: Scale, y: (v: number) => number): string {
  let d = ''
  let prev: number | null = null
  for (const p of points) {
    const a = x(p.t - p.itl)
    const b = x(p.t)
    const yy = y(p.itl)
    d += prev !== null && Math.abs(prev - a) < 1.5 ? `V${yy.toFixed(1)}H${b.toFixed(1)}` : `M${a.toFixed(1)},${yy.toFixed(1)}H${b.toFixed(1)}`
    prev = b
  }
  return d
}

/** Подпись самой длинной паузы на дорожке ITL: справа от её плато (там линия уже упала), а у правого края слева от плато. */
function maxLabel(points: ItlPoint[], itlMax: number, x: Scale, lane: Lane, yItl: (v: number, l: Lane) => number, right: number) {
  if (!points.length) return null
  let worst = points[0]
  for (const p of points) if (p.itl > worst.itl) worst = p
  if (worst.itl < SLO) return null
  const text = fmt.ms(worst.itl)
  const font = '600 11px "Onest Variable", sans-serif'
  const after = x(worst.t) + 4
  const fitsRight = after + measureText(text, font) <= right
  const cy = yItl(Math.min(worst.itl, itlMax), lane) + 9
  return (
    <text
      x={fitsRight ? after : x(worst.t - worst.itl) - 4}
      y={cy}
      textAnchor={fitsRight ? 'start' : 'end'}
      style={{ font, fill: 'var(--ink)', paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3 }}
    >
      {text}
    </text>
  )
}
