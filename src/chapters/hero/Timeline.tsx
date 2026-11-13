/**
 * Таймлайн прогона: стадии полосами по реальному времени на разрывной оси. Слева от разрыва всё до первого
 * токена (TTFT), справа генерация до последнего чанка (E2E). Тики прихода чанков, маркер TTFT, плейхед.
 * На узкой ширине две части встают друг под другом.
 */
import { memo, useId, useMemo, useRef, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { cssVar, fmt, TipRow, useElementSize, useTip } from '@/ui'
import type { FlowRunEx } from '@/engine/flow'
import type { StageId, TimelineItem } from '@/engine/types'
import { stepAtTime, ticks } from './model'

const ROW = 22
const BAR = 12
const LABEL_W = 112
const TOP = 30
const AXIS = 24
const GAP_AB = 20
const STACK_BELOW = 620

const ROW_LABEL: Record<StageId, string> = {
  client: 'Сеть: запрос',
  frontend: 'Frontend',
  preprocess: 'Preprocessor',
  route: 'Router',
  prefill: 'Prefill',
  transfer: 'NIXL',
  decode: 'Decode',
  postprocess: 'Postprocessor',
  stream: 'Сеть: чанки',
}

interface Panel {
  key: 'a' | 'b'
  title: string
  t0: number
  t1: number
  x0: number
  w: number
  y0: number
  rows: StageId[]
  labels: boolean
}

function fill(stage: StageId): string {
  if (stage === 'prefill') return cssVar('prefill')
  if (stage === 'decode') return cssVar('decode')
  if (stage === 'transfer') return 'color-mix(in oklab, var(--ink-2) 70%, transparent)'
  return 'color-mix(in oklab, var(--ink-3) 38%, transparent)'
}

export interface TimelineProps {
  run: FlowRunEx
  step: number
  stepMs: number
  speed: number
  playing: boolean
  onSeek: (step: number) => void
}

export function Timeline({ run, step, stepMs, speed, playing, onSeek }: TimelineProps) {
  const [sizeRef, { width }] = useElementSize<HTMLDivElement>()
  const W = Math.max(280, width)
  const stacked = W < STACK_BELOW
  const T = run.metrics.ttftMs
  const E = Math.max(run.metrics.e2eMs, T + 1)
  const rows = useMemo(() => {
    const present = new Set(run.timeline.map((t) => t.stage))
    return (['client', 'frontend', 'preprocess', 'route', 'prefill', 'transfer', 'decode', 'postprocess', 'stream'] as StageId[]).filter((s) => present.has(s))
  }, [run.timeline])

  const panels = useMemo<Panel[]>(() => {
    if (!stacked) {
      const plot = W - LABEL_W - 8
      const wA = Math.round(plot * 0.42)
      return [
        { key: 'a', title: 'До первого токена', t0: 0, t1: T, x0: LABEL_W, w: wA, y0: TOP, rows, labels: true },
        { key: 'b', title: 'Генерация', t0: T, t1: E, x0: LABEL_W + wA + GAP_AB, w: plot - wA - GAP_AB, y0: TOP, rows, labels: false },
      ]
    }
    const lw = 96
    const rowsB: StageId[] = rows.filter((r) => r === 'decode' || r === 'postprocess' || r === 'stream')
    const hA = rows.length * ROW + AXIS
    return [
      { key: 'a', title: 'До первого токена', t0: 0, t1: T, x0: lw, w: W - lw - 6, y0: TOP, rows, labels: true },
      { key: 'b', title: 'Генерация', t0: T, t1: E, x0: lw, w: W - lw - 6, y0: TOP + hA + TOP, rows: rowsB, labels: true },
    ]
  }, [stacked, W, T, E, rows])

  const height = stacked ? panels[1].y0 + panels[1].rows.length * ROW + AXIS : TOP + rows.length * ROW + AXIS
  const cur = run.playback?.[step]
  const tNow = cur?.tMs ?? 0

  const svgRef = useRef<SVGSVGElement>(null)
  const seekAt = (clientX: number, clientY: number) => {
    const svg = svgRef.current
    if (!svg || !run.playback) return
    const r = svg.getBoundingClientRect()
    const px = clientX - r.left
    const py = clientY - r.top
    const p = panels.find((pp) => px >= pp.x0 - 4 && px <= pp.x0 + pp.w + 4 && py >= pp.y0 - TOP && py <= pp.y0 + pp.rows.length * ROW + AXIS)
    if (!p) return
    const t = p.t0 + ((Math.min(Math.max(px, p.x0), p.x0 + p.w) - p.x0) / p.w) * (p.t1 - p.t0)
    onSeek(stepAtTime(run.playback, t))
  }

  return (
    <div ref={sizeRef} className="min-w-0">
      <div className="relative" style={{ height }}>
        {width > 0 && (
          <>
            <svg
              ref={svgRef}
              width={W}
              height={height}
              className="absolute inset-0 cursor-pointer overflow-visible select-none"
              role="img"
              aria-label={`Таймлайн запроса: TTFT ${fmt.ms(T)}, весь ответ ${fmt.ms(E)}, prefill ${fmt.ms(run.prefill.ms)}, ${run.decode.length} токенов. Щелчок переводит плеер в этот момент.`}
              onPointerDown={(e) => seekAt(e.clientX, e.clientY)}
            >
              <StaticLayer run={run} panels={panels} rows={rows} />
              {panels.map((p) => (
                <Playhead key={p.key} panel={p} t={tNow} durationS={playing ? stepMs / Math.max(0.25, speed) / 1000 : 0.15} />
              ))}
            </svg>
            <BarHover run={run} panels={panels} width={W} height={height} onSeekAt={seekAt} />
          </>
        )}
      </div>
      <Legend transfer={rows.includes('transfer')} neighbor={Boolean(run.neighbor?.applied)} />
    </div>
  )
}

/** Через сколько тиков рисовать тик прихода чанка, чтобы между соседними было не меньше 4 px. */
function streamTickStep(width: number, tokens: number): number {
  const per = width / Math.max(1, tokens)
  return per >= 4 ? 1 : Math.ceil(4 / Math.max(0.01, per))
}

/** Положение времени на панели или null, если момент вне панели. */
function px(p: Panel, t: number): number {
  const c = Math.min(Math.max(t, p.t0), p.t1)
  return p.x0 + ((c - p.t0) / Math.max(1e-9, p.t1 - p.t0)) * p.w
}

const StaticLayer = memo(function StaticLayer({ run, panels, rows }: { run: FlowRunEx; panels: Panel[]; rows: StageId[] }) {
  const uid = useId().replace(/:/g, '')
  const hatch = `${uid}-hatch`
  const T = run.metrics.ttftMs
  const E = run.metrics.e2eMs
  const coarse = run.timeline.filter((t) => t.step === undefined)
  const decodeItems = run.timeline.filter((t) => t.stage === 'decode')
  const decodeBand = decodeItems.length ? { start: decodeItems[0].startMs, end: decodeItems[decodeItems.length - 1].endMs } : null
  const firstPost = run.timeline.find((t) => t.stage === 'postprocess')
  const firstStream = run.timeline.find((t) => t.stage === 'stream')
  return (
    <g>
      <defs>
        <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill="color-mix(in oklab, var(--ink-3) 22%, transparent)" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="color-mix(in oklab, var(--ink-3) 75%, transparent)" strokeWidth="2.5" />
        </pattern>
        {panels.map((p) => (
          <clipPath key={p.key} id={`${uid}-clip-${p.key}`}>
            <rect x={p.x0} y={p.y0 - 2} width={p.w} height={p.rows.length * ROW + 4} />
          </clipPath>
        ))}
      </defs>

      {panels.map((p) => {
        const tks = p.key === 'a' ? ticks(0, p.t1, p.w < 200 ? 3 : 4) : ticks(p.t0, p.t1, p.w < 160 ? 3 : 4).filter((v) => v > p.t0 + (p.t1 - p.t0) * 0.06)
        const rowY = (r: StageId) => p.y0 + p.rows.indexOf(r) * ROW
        const inRows = (r: StageId) => p.rows.includes(r)
        return (
          <g key={p.key}>
            <text x={p.x0} y={p.y0 - 12} className="fill-[var(--ink-2)] text-[12px] font-medium">
              {p.title}
            </text>
            {/* сетка и подписи осей */}
            {tks.map((v) => (
              <g key={v}>
                <line x1={px(p, v)} x2={px(p, v)} y1={p.y0} y2={p.y0 + p.rows.length * ROW} stroke="var(--grid)" />
                <text x={px(p, v)} y={p.y0 + p.rows.length * ROW + 15} textAnchor="middle" className="fill-[var(--ink-3)] text-[11.5px] tabular-nums">
                  {fmt.ms(v)}
                </text>
              </g>
            ))}
            <line x1={p.x0} x2={p.x0 + p.w} y1={p.y0 + p.rows.length * ROW} y2={p.y0 + p.rows.length * ROW} stroke="var(--line-strong)" />
            {/* подписи строк */}
            {p.labels &&
              p.rows.map((r, i) => (
                <text key={r} x={p.x0 - 10} y={p.y0 + i * ROW + ROW / 2 + 4} textAnchor="end" className="fill-[var(--ink-2)] text-[12px]">
                  {ROW_LABEL[r]}
                </text>
              ))}
            {/* полосы */}
            <g clipPath={`url(#${uid}-clip-${p.key})`}>
              {p.rows.map((r, i) => (
                <rect key={`bg${r}`} x={p.x0} y={p.y0 + i * ROW + (ROW - BAR) / 2} width={p.w} height={BAR} fill="var(--surface-2)" opacity={0.6} />
              ))}
              {coarse
                .filter((it) => inRows(it.stage))
                .map((it, i) => {
                  const x0 = px(p, it.startMs)
                  const x1 = px(p, it.endMs)
                  if (it.endMs < p.t0 || it.startMs > p.t1) return null
                  const w = Math.max(2, x1 - x0)
                  const y = rowY(it.stage) + (ROW - BAR) / 2
                  return (
                    <g key={`c${i}`}>
                      <rect x={x0} y={y} width={w} height={BAR} rx={2} fill={it.simulated ? `url(#${hatch})` : fill(it.stage)} />
                    </g>
                  )
                })}
              {decodeBand && inRows('decode') && decodeBand.end >= p.t0 && decodeBand.start <= p.t1 && (
                <rect
                  x={px(p, decodeBand.start)}
                  y={rowY('decode') + (ROW - BAR) / 2}
                  width={Math.max(2, px(p, decodeBand.end) - px(p, decodeBand.start))}
                  height={BAR}
                  rx={2}
                  fill={fill('decode')}
                />
              )}
              {/* agg под нагрузкой: шаг decode, в который chunked prefill поставил промпт соседа (X-sim-7), цветом prefill */}
              {inRows('decode') &&
                run.decode.map((d) => {
                  if (!d.neighborPrefill) return null
                  const s = d.startMs
                  const e = d.startMs + d.durationMs
                  if (e < p.t0 || s > p.t1) return null
                  return (
                    <rect
                      key={`nb${d.step}`}
                      x={px(p, s)}
                      y={rowY('decode') + (ROW - BAR) / 2}
                      width={Math.max(2, px(p, e) - px(p, s))}
                      height={BAR}
                      rx={2}
                      fill={fill('prefill')}
                      stroke="var(--surface)"
                      strokeWidth={1}
                    />
                  )
                })}
              {/* первый токен: детокенизация и путь чанка к клиенту (в панели до TTFT видны как полосы) */}
              {p.key === 'a' && firstPost && inRows('postprocess') && (
                <rect x={px(p, firstPost.startMs)} y={rowY('postprocess') + (ROW - BAR) / 2} width={Math.max(2, px(p, firstPost.endMs) - px(p, firstPost.startMs))} height={BAR} rx={2} fill={`url(#${hatch})`} />
              )}
              {p.key === 'a' && firstStream && inRows('stream') && (
                <rect x={px(p, firstStream.startMs)} y={rowY('stream') + (ROW - BAR) / 2} width={Math.max(2, px(p, firstStream.endMs) - px(p, firstStream.startMs))} height={BAR} rx={2} fill={`url(#${hatch})`} />
              )}
              {/* тики: детокенизация каждого токена и приход каждого чанка к клиенту */}
              {p.key === 'b' &&
                inRows('postprocess') &&
                (p.w / Math.max(1, run.decode.length) < 3 ? (
                  // тики гуще 3 px сливаются: показываем тонкую полосу (детокенизация идёт на каждом токене короткими вспышками)
                  <rect
                    x={px(p, run.decode[1]?.emitMs ?? T)}
                    y={rowY('postprocess') + (ROW - 4) / 2}
                    width={Math.max(2, px(p, run.decode[run.decode.length - 1]?.emitMs ?? E) - px(p, run.decode[1]?.emitMs ?? T))}
                    height={4}
                    rx={2}
                    fill="color-mix(in oklab, var(--ink-3) 45%, transparent)"
                  />
                ) : (
                  run.decode.slice(1).map((d) => {
                    const x = px(p, d.emitMs ?? 0)
                    return <line key={`pp${d.step}`} x1={x} x2={x} y1={rowY('postprocess') + 5} y2={rowY('postprocess') + ROW - 5} stroke="var(--ink-3)" strokeWidth={1} />
                  })
                ))}
              {p.key === 'b' &&
                inRows('stream') &&
                (() => {
                  // гуще 4 px тики сливаются в «штрихкод»: рисуем каждый k-й, тише
                  const k = streamTickStep(p.w, run.decode.length)
                  return run.decode.slice(1).map((d, j) => {
                    if (j % k !== 0) return null
                    const x = px(p, d.clientMs ?? 0)
                    return (
                      <line
                        key={`st${d.step}`}
                        x1={x}
                        x2={x}
                        y1={rowY('stream') + 5}
                        y2={rowY('stream') + ROW - 5}
                        stroke={k > 1 ? 'var(--ink-3)' : 'var(--ink-2)'}
                        strokeOpacity={k > 1 ? 0.7 : 1}
                        strokeWidth={1}
                      />
                    )
                  })
                })()}
            </g>
            {/* маркеры TTFT и E2E */}
            {p.key === 'a' && (
              <g>
                <line x1={px(p, T)} x2={px(p, T)} y1={p.y0 - 6} y2={p.y0 + p.rows.length * ROW} stroke="var(--ink)" strokeDasharray="3 3" />
                <text x={px(p, T) - 4} y={p.y0 - 12} textAnchor="end" className="fill-[var(--ink)] text-[12px] font-semibold tabular-nums">
                  TTFT {fmt.ms(T)}
                </text>
              </g>
            )}
            {p.key === 'b' && (
              <g>
                <line x1={p.x0 + p.w} x2={p.x0 + p.w} y1={p.y0 - 6} y2={p.y0 + p.rows.length * ROW} stroke="var(--ink)" strokeDasharray="3 3" />
                <text x={p.x0 + p.w} y={p.y0 - 12} textAnchor="end" className="fill-[var(--ink)] text-[12px] font-semibold tabular-nums">
                  E2E {fmt.ms(E)}
                </text>
              </g>
            )}
          </g>
        )
      })}
      {/* разрыв оси между панелями */}
      {panels[0].y0 === panels[1].y0 && (
        <g stroke="var(--ink-3)" strokeWidth={1.25}>
          {(() => {
            const a = panels[0]
            const y = a.y0 + a.rows.length * ROW
            const x = a.x0 + a.w + GAP_AB / 2
            return (
              <>
                <line x1={x - 6} y1={y + 5} x2={x - 2} y2={y - 5} />
                <line x1={x + 2} y1={y + 5} x2={x + 6} y2={y - 5} />
              </>
            )
          })()}
        </g>
      )}
    </g>
  )
})

function Playhead({ panel, t, durationS }: { panel: Panel; t: number; durationS: number }) {
  const reduce = useReducedMotion() ?? false
  const inside = panel.key === 'a' ? t <= panel.t1 + 1e-9 : t > panel.t0 + 1e-9
  if (!inside) return null
  const x = px(panel, t)
  const h = panel.rows.length * ROW
  return (
    <motion.g
      initial={false}
      animate={{ x }}
      transition={reduce ? { duration: 0 } : { duration: Math.min(1.2, durationS), ease: 'linear' }}
      style={{ pointerEvents: 'none' }}
    >
      <line x1={0} x2={0} y1={panel.y0 - 4} y2={panel.y0 + h + 2} stroke="var(--accent)" strokeWidth={2} />
      <circle cx={0} cy={panel.y0 - 4} r={3.5} fill="var(--accent)" />
    </motion.g>
  )
}

/** Невидимые зоны строк поверх SVG: тултип с реальным временем стадии. */
function BarHover({
  run,
  panels,
  width,
  height,
  onSeekAt,
}: {
  run: FlowRunEx
  panels: Panel[]
  width: number
  height: number
  onSeekAt: (clientX: number, clientY: number) => void
}) {
  const rootRef = useRef<HTMLDivElement>(null)
  const tip = useTip({ scopeEl: rootRef.current })
  const items = useMemo(() => {
    const out: Array<{ key: string; panel: Panel; stage: StageId; x: number; y: number; w: number; content: ReactNode }> = []
    const coarse = run.timeline.filter((t) => t.step === undefined)
    const dec = run.stages?.find((s) => s.stage === 'decode')
    const post = run.stages?.find((s) => s.stage === 'postprocess')
    const str = run.stages?.find((s) => s.stage === 'stream')
    for (const p of panels) {
      for (const r of p.rows) {
        const y = p.y0 + p.rows.indexOf(r) * ROW
        let content: ReactNode = null
        const its: TimelineItem[] = coarse.filter((c) => c.stage === r)
        if (its.length) {
          content = (
            <div className="grid grid-cols-[minmax(0,1fr)] gap-1">
              {its.map((it, i) => (
                <div key={i} className="grid grid-cols-[minmax(0,1fr)] gap-0.5">
                  <TipRow value={fmt.ms(it.endMs - it.startMs)} label={it.label} />
                  {it.detail && <div className="text-xs text-ink-2">{it.detail}</div>}
                  <div className="text-xs text-ink-3">
                    {fmt.ms(it.startMs)} → {fmt.ms(it.endMs)}
                    {it.simulated ? ', условная константа' : ', посчитано по модели и GPU'}
                  </div>
                </div>
              ))}
            </div>
          )
        } else if (r === 'decode' && dec) {
          content = (
            <div className="grid grid-cols-[minmax(0,1fr)] gap-0.5">
              <TipRow value={fmt.ms(dec.ms)} label={`${dec.count} ${fmt.plural(dec.count, ['шаг', 'шага', 'шагов'])} decode`} />
              <div className="text-xs text-ink-2">
                {/* (X-sim-r2-4) шаг с prefill соседа тянет среднее вверх: обычный шаг отдельно */}
                {run.neighbor?.applied ? 'Обычный шаг' : 'Шаг ≈'} {fmt.ms(run.neighbor?.applied ? run.neighbor.plainMs : dec.ms / Math.max(1, dec.count))}: один токен на шаг,
                посчитано по модели и GPU.
              </div>
              {run.neighbor?.applied && (
                <div className="text-xs text-ink-2">
                  Синий кусок: шаг токена {fmt.int(run.neighbor.atStep + 1)}, в который chunked prefill поставил промпт соседа ({fmt.tokens(run.neighbor.promptTokens)},
                  условно): {fmt.ms(run.neighbor.stepMs)} вместо {fmt.ms(run.neighbor.plainMs)}.
                </div>
              )}
              {p.key === 'a' && (
                <div className="text-xs text-ink-2">
                  Decode начинается сразу после prefill: пока первый токен едет к клиенту по сети, GPU уже считает следующий. Поэтому полоса decode
                  начинается до TTFT.
                </div>
              )}
            </div>
          )
        } else if (r === 'postprocess' && post) {
          content = (
            <div className="grid grid-cols-[minmax(0,1fr)] gap-0.5">
              <TipRow value={fmt.ms(post.ms / Math.max(1, post.count))} label="на токен" />
              <div className="text-xs text-ink-2">Детокенизация, парсеры и сборка чанка, условная константа.</div>
            </div>
          )
        } else if (r === 'stream' && str) {
          const k = streamTickStep(p.w, run.decode.length)
          content = (
            <div className="grid grid-cols-[minmax(0,1fr)] gap-0.5">
              <TipRow value={fmt.ms(str.ms / Math.max(1, str.count))} label="сеть на чанк" />
              <div className="text-xs text-ink-2">
                Тики: моменты прихода чанков к клиенту. Расстояние между ними и есть ITL.
                {p.key === 'b' && k > 1 && <> Тики прорежены: на этой ширине показан каждый {k}-й.</>}
              </div>
            </div>
          )
        }
        if (content) out.push({ key: `${p.key}-${r}`, panel: p, stage: r, x: p.x0, y, w: p.w, content })
      }
    }
    return out
  }, [run, panels])

  return (
    <div ref={rootRef} className="pointer-events-none absolute inset-0" style={{ width, height }}>
      {items.map((it) => (
        <div
          key={it.key}
          data-row={it.stage}
          data-panel={it.panel.key}
          className="pointer-events-auto absolute"
          style={{ left: it.x, top: it.y, width: it.w, height: ROW }}
          onPointerEnter={(e) => tip.show(e.currentTarget.getBoundingClientRect(), it.content)}
          onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
          onPointerDown={(e) => {
            tip.show(e.currentTarget.getBoundingClientRect(), it.content)
            onSeekAt(e.clientX, e.clientY)
          }}
        />
      ))}
      {tip.node}
    </div>
  )
}

function Legend({ transfer, neighbor }: { transfer: boolean; neighbor: boolean }) {
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1.5 text-[12.5px] leading-5 text-ink-2">
      <LegendItem swatch={<span className="h-2.5 w-4 rounded-[2px]" style={{ background: cssVar('prefill') }} />}>prefill</LegendItem>
      <LegendItem swatch={<span className="h-2.5 w-4 rounded-[2px]" style={{ background: cssVar('decode') }} />}>decode</LegendItem>
      {neighbor && (
        <LegendItem swatch={<span className="h-2.5 w-4 rounded-[2px]" style={{ background: cssVar('prefill') }} />}>
          в полосе decode: prefill соседа в том же шаге
        </LegendItem>
      )}
      {transfer && (
        <LegendItem swatch={<span className="h-2.5 w-4 rounded-[2px]" style={{ background: 'color-mix(in oklab, var(--ink-2) 70%, transparent)' }} />}>
          передача KV
        </LegendItem>
      )}
      <LegendItem
        swatch={
          <span
            className="h-2.5 w-4 rounded-[2px]"
            style={{ background: 'repeating-linear-gradient(135deg, color-mix(in oklab, var(--ink-3) 75%, transparent) 0 2px, color-mix(in oklab, var(--ink-3) 22%, transparent) 2px 4px)' }}
          />
        }
      >
        сеть и CPU-стадии (условные константы)
      </LegendItem>
      <LegendItem swatch={<span className="h-3 w-0.5 rounded-full" style={{ background: 'var(--accent)' }} />}>текущий шаг плеера</LegendItem>
    </div>
  )
}

function LegendItem({ swatch, children }: { swatch: ReactNode; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden className="inline-flex items-center">
        {swatch}
      </span>
      {children}
    </span>
  )
}
