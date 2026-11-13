import { useMemo, useRef, useState, type PointerEvent, type ReactNode } from 'react'
import { Figure, Formula, Segmented, Slider, Stat, TipRow, cssVar, fmt, tint, useElementSize, useTip } from '@/ui'
import { linearTicks } from '@/ui/chart-utils'
import { hwLabel, valueUnit, type Hardware } from '@/chapters/prefill-decode/shared/hardware'
import { requestTiming } from '@/chapters/prefill-decode/shared/timing'
import { texInt, texMs } from '@/chapters/prefill-decode/shared/tex'

const STEP_FORMS = ['шаг', 'шага', 'шагов'] as const

type Chunk = 1 | 2 | 4 | 8

interface Model {
  half: number
  queue: number
  prefill: number
  itl: number
  out: number
  chunk: number
  serverArrive: number
  serverFirst: number
  serverLast: number
  tokens: number[]
  chunks: Array<{ sent: number; arrive: number; n: number }>
  clientTtft: number
  clientE2e: number
  serverTtft: number
  tpot: number
  chunkGap: number
}

function build(hw: Hardware, promptLen: number, out: number, rtt: number, queue: number, chunk: number): Model {
  const t = requestTiming(hw, promptLen, out, queue)
  const half = rtt / 2
  const serverArrive = half
  const serverFirst = serverArrive + t.ttftMs
  const tokens = Array.from({ length: out }, (_, i) => serverFirst + i * t.itlMs)
  // Как --stream-interval в vLLM и SGLang: первый токен уходит сразу отдельным чанком, дальше по `chunk`
  // токенов, а последний чанк уходит вместе с последним токеном, даже неполный.
  const chunks: Model['chunks'] = [{ sent: tokens[0], arrive: tokens[0] + half, n: 1 }]
  for (let i = 1; i < out; i += chunk) {
    const last = Math.min(out, i + chunk) - 1
    chunks.push({ sent: tokens[last], arrive: tokens[last] + half, n: last - i + 1 })
  }
  const clientTtft = chunks[0].arrive
  const clientE2e = chunks[chunks.length - 1].arrive
  return {
    half,
    queue,
    prefill: t.prefillMs,
    itl: t.itlMs,
    out,
    chunk,
    serverArrive,
    serverFirst,
    serverLast: tokens[tokens.length - 1],
    tokens,
    chunks,
    clientTtft,
    clientE2e,
    serverTtft: t.ttftMs,
    tpot: out > 1 ? (clientE2e - clientTtft) / (out - 1) : 0,
    chunkGap: chunks.length > 1 ? chunks[1].arrive - chunks[0].arrive : 0,
  }
}

export function MeasureTimelineFigure({ hw, promptLen }: { hw: Hardware; promptLen: number }) {
  const [out, setOut] = useState(40)
  const [rtt, setRtt] = useState(60)
  const [queue, setQueue] = useState(120)
  const [chunk, setChunk] = useState<Chunk>(1)
  const m = useMemo(() => build(hw, promptLen, out, rtt, queue, chunk), [hw, promptLen, out, rtt, queue, chunk])

  return (
    <Figure
      id="metrics-figure-stopwatch"
      title="Секундомер запроса: что меряет сервер и что видит клиент"
      width="wide"
      actions={<span className="block max-w-[18rem] text-[13px] sm:max-w-none leading-snug text-ink-3">{hwLabel(hw)}, промпт {fmt.tokens(promptLen)}</span>}
      controls={
        <>
          <Slider label="Токенов в ответе" value={out} min={2} max={400} scale="log" onChange={setOut} format={(v) => fmt.int(v)} />
          <Slider label="Сеть, RTT" value={rtt} min={0} max={300} step={5} onChange={setRtt} format={(v) => fmt.ms(v)} hint="туда и обратно, условно" />
          <Slider label="Очередь на сервере" value={queue} min={0} max={1000} step={10} onChange={setQueue} format={(v) => fmt.ms(v)} hint="условно" />
          <Segmented<Chunk>
            label="Токенов в SSE-чанке"
            value={chunk}
            onChange={setChunk}
            options={[
              { value: 1, label: '1' },
              { value: 2, label: '2' },
              { value: 4, label: '4' },
              { value: 8, label: '8' },
            ]}
          />
        </>
      }
      caption="Попробуйте: поставьте 4 токена в чанке. TTFT и TPOT не изменятся: первый токен движок отправляет сразу, а последний уходит, как только готов. Зато интервалы между чанками вырастут вчетверо: клиент, который считает ITL по чанкам, увидит рваную печать. Потом добавьте RTT: серверный TTFT стоит на месте, клиентский растёт."
      source="Время prefill и шага decode посчитано для выбранных модели и GPU, сеть и очередь условные. Чанки нарезаны как в vLLM и SGLang (--stream-interval): первый токен отдельно, дальше по N токенов."
    >
      <div className="grid gap-5">
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-3 @3xl:grid-cols-6">
          <Stat label="TTFT клиента" value={valueUnit(fmt.ms(m.clientTtft))[0]} unit={valueUnit(fmt.ms(m.clientTtft))[1]} hint="сеть + очередь + prefill" swatch="var(--c-prefill)" />
          <Stat label="TTFT сервера" value={valueUnit(fmt.ms(m.serverTtft))[0]} unit={valueUnit(fmt.ms(m.serverTtft))[1]} hint="очередь + prefill" />
          <Stat label="ITL на сервере" value={valueUnit(fmt.ms(m.itl))[0]} unit={valueUnit(fmt.ms(m.itl))[1]} hint="шаг decode" swatch="var(--c-decode)" />
          <Stat
            label="Между чанками"
            value={m.chunks.length > 1 ? valueUnit(fmt.ms(m.chunkGap))[0] : 'н/д'}
            unit={m.chunks.length > 1 ? valueUnit(fmt.ms(m.chunkGap))[1] : undefined}
            hint={m.chunks.length > 1 ? `чанк = ${fmt.tokens(m.chunk)}` : 'весь ответ одним чанком'}
          />
          <Stat
            label="TPOT клиента"
            value={m.tpot > 0 ? valueUnit(fmt.ms(m.tpot))[0] : 'н/д'}
            unit={m.tpot > 0 ? valueUnit(fmt.ms(m.tpot))[1] : undefined}
            hint="(E2E − TTFT) / (N − 1)"
          />
          <Stat
            label="E2E клиента"
            value={valueUnit(fmt.ms(m.clientE2e))[0]}
            unit={valueUnit(fmt.ms(m.clientE2e))[1]}
            hint={m.tpot > 0 ? `${fmt.int(1000 / m.tpot)} ток/с после первого` : 'весь ответ пришёл одним чанком'}
          />
        </div>
        <Lanes m={m} />
        <div className="overflow-x-auto">
          <Formula
            block
            tex={`\\text{E2E} = \\text{TTFT} + \\text{TPOT} \\cdot (N - 1) = ${texMs(m.clientTtft)} + ${texMs(m.tpot)} \\cdot ${texInt(m.out - 1)} = ${texMs(m.clientE2e)}`}
            narrowTex={`\\begin{aligned} \\text{E2E} &= \\text{TTFT} + \\text{TPOT} \\cdot (N - 1) \\\\ &= ${texMs(m.clientTtft)} + ${texMs(m.tpot)} \\cdot ${texInt(m.out - 1)} \\\\ &= ${texMs(m.clientE2e)} \\end{aligned}`}
            label="всё глазами клиента; TPOT это средний интервал на токен после первого"
          />
        </div>
      </div>
    </Figure>
  )
}

/* ------------------------------------------------------------------------- */

function Lanes({ m }: { m: Model }) {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement | null>(null)
  const tip = useTip({ scopeEl: svgRef.current, side: 'top' })
  const W = Math.max(300, width)
  const labelW = 64
  const left = labelW
  const right = 8
  const plotW = W - left - right
  const T = m.clientE2e * 1.02
  const x = (ms: number) => left + (ms / T) * plotW
  const serverY = 34
  const clientY = 112
  const barH = 22
  const axisY = clientY + barH + 36
  const svgH = axisY + 24
  const show = (e: PointerEvent<SVGElement>, c: ReactNode) => tip.show(e.currentTarget.getBoundingClientRect(), c)
  const hide = (e: PointerEvent<SVGElement>) => e.pointerType === 'mouse' && tip.hide()
  const ticks = linearTicks(0, T, Math.max(2, Math.floor(plotW / 110)))
  const tokenGap = m.out > 1 ? x(m.tokens[1]) - x(m.tokens[0]) : 99
  const showTokens = tokenGap >= 3

  return (
    <div ref={ref} className="min-w-0">
      {width > 0 && (
        <svg
          ref={svgRef}
          width={W}
          height={svgH}
          role="img"
          aria-label={`Сервер получил запрос через ${fmt.ms(m.half)}, первый токен через ${fmt.ms(m.serverTtft)} после прихода; клиент увидел первый чанк на ${fmt.ms(m.clientTtft)}, последний на ${fmt.ms(m.clientE2e)}`}
          className="block select-none overflow-visible"
        >
          <defs>
            <pattern id="mt-queue-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="6" fill="var(--surface-2)" />
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--line-strong)" strokeWidth="2" />
            </pattern>
            <marker id="mt-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--ink-3)" />
            </marker>
          </defs>

          <text x={0} y={serverY + barH / 2} dominantBaseline="central" className="fill-ink-2 text-[13px] font-medium">
            Сервер
          </text>
          <text x={0} y={clientY + barH / 2} dominantBaseline="central" className="fill-ink-2 text-[13px] font-medium">
            Клиент
          </text>

          {/* сервер: дорожка */}
          <line x1={left} x2={left + plotW} y1={serverY + barH / 2} y2={serverY + barH / 2} stroke="var(--line)" />
          {m.queue > 0 && (
            <rect
              x={x(m.serverArrive)}
              y={serverY}
              width={Math.max(1, x(m.serverArrive + m.queue) - x(m.serverArrive))}
              height={barH}
              rx={3}
              fill="url(#mt-queue-hatch)"
              stroke="var(--line-strong)"
              onPointerMove={(e) => show(e, <TipRow value={fmt.ms(m.queue)} label="очередь: ждём места в batch" />)}
              onPointerDown={(e) => show(e, <TipRow value={fmt.ms(m.queue)} label="очередь: ждём места в batch" />)}
              onPointerLeave={hide}
            />
          )}
          <rect
            x={x(m.serverArrive + m.queue)}
            y={serverY}
            width={Math.max(2, x(m.serverFirst) - x(m.serverArrive + m.queue))}
            height={barH}
            rx={3}
            fill={cssVar('prefill')}
            onPointerMove={(e) => show(e, <TipRow color={cssVar('prefill')} value={fmt.ms(m.prefill)} label="prefill" />)}
            onPointerDown={(e) => show(e, <TipRow color={cssVar('prefill')} value={fmt.ms(m.prefill)} label="prefill" />)}
            onPointerLeave={hide}
          />
          <rect
            x={x(m.serverFirst)}
            y={serverY}
            width={Math.max(1, x(m.serverLast) - x(m.serverFirst))}
            height={barH}
            rx={3}
            fill={tint('decode', 24)}
            stroke={cssVar('decode')}
            onPointerMove={(e) => show(e, <TipRow color={cssVar('decode')} value={fmt.ms(m.itl)} label={`на шаг, ${fmt.count(m.out - 1, STEP_FORMS)} decode`} />)}
            onPointerDown={(e) => show(e, <TipRow color={cssVar('decode')} value={fmt.ms(m.itl)} label={`на шаг, ${fmt.count(m.out - 1, STEP_FORMS)} decode`} />)}
            onPointerLeave={hide}
          />
          {showTokens && (
            <g pointerEvents="none">
              {m.tokens.map((tt, i) => (
                <line key={i} x1={x(tt)} x2={x(tt)} y1={serverY + 5} y2={serverY + barH - 5} stroke={cssVar('decode')} strokeWidth={1.25} />
              ))}
            </g>
          )}

          {/* сеть: запрос и первый ответ */}
          <line x1={x(0)} y1={clientY} x2={x(m.serverArrive)} y2={serverY + barH} stroke="var(--ink-3)" strokeDasharray="3 3" markerEnd="url(#mt-arrow)" />
          <line x1={x(m.chunks[0].sent)} y1={serverY + barH} x2={x(m.chunks[0].arrive)} y2={clientY} stroke="var(--ink-3)" strokeDasharray="3 3" markerEnd="url(#mt-arrow)" />

          {/* клиент: отправка и чанки */}
          <line x1={left} x2={left + plotW} y1={clientY + barH / 2} y2={clientY + barH / 2} stroke="var(--line)" />
          <circle cx={x(0)} cy={clientY + barH / 2} r={5} fill="var(--ink)" />
          {m.chunks.map((c, i) => (
            <rect
              key={i}
              x={x(c.arrive) - 2}
              y={clientY + 3}
              width={4}
              height={barH - 6}
              rx={1.5}
              fill={cssVar('decode')}
              onPointerMove={(e) => show(e, <ChunkTip i={i} m={m} />)}
              onPointerDown={(e) => show(e, <ChunkTip i={i} m={m} />)}
              onPointerLeave={hide}
            />
          ))}

          {/* скобки TTFT */}
          <Bracket x0={x(m.serverArrive)} x1={x(m.serverFirst)} y={serverY - 8} up label={`TTFT сервера ${fmt.ms(m.serverTtft)}`} width={W} />
          <Bracket x0={x(0)} x1={x(m.clientTtft)} y={clientY + barH + 8} label={`TTFT клиента ${fmt.ms(m.clientTtft)}`} width={W} />

          {/* ось */}
          <line x1={left} x2={left + plotW} y1={axisY} y2={axisY} stroke="var(--line-strong)" />
          {ticks.map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={axisY} y2={axisY + 4} stroke="var(--line-strong)" />
              <text x={x(v)} y={axisY + 17} textAnchor={v === 0 ? 'start' : 'middle'} className="fill-ink-3 text-[12px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {fmt.ms(v)}
              </text>
            </g>
          ))}
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
          <span aria-hidden className="h-3 w-4 rounded-[3px] border" style={{ background: tint('decode', 24), borderColor: cssVar('decode') }} />
          decode, черта = токен
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="h-3 w-1 rounded-[1px]" style={{ background: cssVar('decode') }} />
          SSE-чанк у клиента
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="w-5 border-t border-dashed border-ink-3" />
          сеть
        </span>
      </div>
      {tip.node}
    </div>
  )
}

function ChunkTip({ i, m }: { i: number; m: Model }) {
  const c = m.chunks[i]
  const prev = i > 0 ? m.chunks[i - 1].arrive : 0
  return (
    <div className="grid gap-1">
      <TipRow color={cssVar('decode')} value={fmt.ms(c.arrive)} label={`чанк ${i + 1}, ${fmt.tokens(c.n)}`} />
      <div className="text-xs text-ink-3">{i === 0 ? 'от отправки запроса: это TTFT клиента' : `через ${fmt.ms(c.arrive - prev)} после предыдущего`}</div>
    </div>
  )
}

function Bracket({ x0, x1, y, up = false, label, width }: { x0: number; x1: number; y: number; up?: boolean; label: string; width: number }) {
  const d = up ? -5 : 5
  const mid = (x0 + x1) / 2
  const lw = label.length * 6.6
  const anchor = mid - lw / 2 < 4 ? 'start' : mid + lw / 2 > width - 4 ? 'end' : 'middle'
  const tx = anchor === 'start' ? Math.max(4, x0) : anchor === 'end' ? Math.min(width - 4, x1) : mid
  return (
    <g pointerEvents="none">
      <path d={`M${x0},${y - d} L${x0},${y} L${x1},${y} L${x1},${y - d}`} fill="none" stroke="var(--ink-2)" strokeWidth={1.25} />
      <text x={tx} y={up ? y - 6 : y + 15} textAnchor={anchor} className="fill-ink text-[12px] font-medium">
        {label}
      </text>
    </g>
  )
}
