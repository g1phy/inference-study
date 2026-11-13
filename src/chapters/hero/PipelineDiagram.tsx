/**
 * Схема пайплайна симулятора: узлы-стадии в ряд, под ними «рельс» со станциями, по которому едет запрос,
 * а на шагах-токенах обратно к клиенту бегут точки токенов. Клик по узлу закрепляет его в инспекторе.
 * На узком экране ряд прокручивается внутри своего блока, активный узел держится в поле зрения.
 */
import { memo, useCallback, useEffect, useMemo, useRef } from 'react'
import { useReducedMotion } from 'motion/react'
import { PushPinIcon } from '@phosphor-icons/react'
import { cn, cssVar, fmt, mix, ScrollHint, tint, useElementSize, useScrollEdges } from '@/ui'
import { pipelineStages, type FlowRunEx } from '@/engine/flow'
import type { StageId } from '@/engine/types'
import { STAGES } from './copy'
import { nodeStatus, packetMove, tokenAtStep, type NodeStatus, type PacketMove } from './model'

const WEIGHT: Record<StageId, number> = {
  client: 0.98,
  frontend: 0.94,
  preprocess: 1.14,
  route: 0.98,
  prefill: 0.98,
  transfer: 0.74,
  decode: 0.96,
  postprocess: 1.16,
  stream: 1.04,
}
const UNIT_MIN = 100
const GAP = 12
const NODE_TOP = 26
const NODE_H = 94
const RAIL_Y = NODE_TOP + NODE_H + 20
const HEIGHT = RAIL_Y + 16

interface Layout {
  stages: StageId[]
  x: number[]
  w: number[]
  cx: number[]
  total: number
}

function layoutOf(stages: StageId[], width: number): Layout {
  const sum = stages.reduce((s, st) => s + WEIGHT[st], 0)
  const gaps = GAP * (stages.length - 1)
  const total = Math.max(width, sum * UNIT_MIN + gaps)
  const unit = (total - gaps) / sum
  const x: number[] = []
  const w: number[] = []
  let at = 0
  for (const st of stages) {
    x.push(at)
    w.push(WEIGHT[st] * unit)
    at += WEIGHT[st] * unit + GAP
  }
  return { stages, x, w, cx: x.map((v, i) => v + w[i] / 2), total }
}

const PACKET_LABELS = {
  json: 'JSON',
  tokens: (n: number) => fmt.tokens(n),
  kv: (b: number) => `KV ${fmt.bytes(b)}`,
  meta: 'адрес KV',
}

/** Время стадии для угла узла (реальное или условное, подробности в инспекторе). У поточных стадий единица видна сразу. */
function nodeMs(run: FlowRunEx, stage: StageId): { text: string; title: string } | null {
  const s = run.stages?.find((x) => x.stage === stage)
  if (!s) return null
  if (stage === 'postprocess') return { text: `${fmt.ms(s.ms / Math.max(1, s.count))}/ток`, title: 'на каждый токен, условная константа' }
  if (stage === 'stream') return { text: `${fmt.ms(s.ms / Math.max(1, s.count))}/чанк`, title: 'сеть до клиента на каждый чанк, условная константа' }
  if (stage === 'decode') return { text: fmt.ms(s.ms), title: `всего за ${s.count} ${fmt.plural(s.count, ['шаг', 'шага', 'шагов'])} decode` }
  return { text: fmt.ms(s.ms), title: s.simulated ? 'условная константа' : 'посчитано по модели и GPU' }
}

function nodeValue(run: FlowRunEx, stage: StageId, k: number, sseCount: number): string {
  const N = run.decode.length
  switch (stage) {
    case 'client': {
      const n = (run.request.messages as unknown[]).length
      return fmt.count(n, ['сообщение', 'сообщения', 'сообщений'])
    }
    case 'frontend':
      return 'stream: true'
    case 'preprocess':
      return fmt.tokens(run.promptTokens.length)
    case 'route':
      return `→ ${run.prefill.workerName ?? run.prefill.worker}`
    case 'prefill':
      return `кэш ${fmt.int(run.prefill.cachedTokens)} из ${fmt.int(run.prefill.totalTokens)}`
    case 'transfer':
      return run.transfer ? fmt.bytes(run.transfer.bytes) : ''
    case 'decode':
      return k >= 0 ? `${fmt.int(k + 1)} / ${fmt.int(N)}` : fmt.tokens(N)
    case 'postprocess':
      return k >= 0 ? (run.decode[k].channel === 'tool_call' ? 'tool_calls' : run.decode[k].channel) : 'парсер qwen3'
    case 'stream':
      return fmt.count(sseCount, ['строка', 'строки', 'строк'])
  }
}

/** Цветная метка фазы у узлов prefill и decode (те же цвета, что на таймлайне). */
const PHASE_COLOR: Partial<Record<StageId, string>> = { prefill: cssVar('prefill'), decode: cssVar('decode') }

interface NodeProps {
  stage: StageId
  x: number
  w: number
  status: NodeStatus
  pinned: boolean
  value: string | null
  ms: { text: string; title: string } | null
  onSelect: (s: StageId) => void
}

const Node = memo(function Node({ stage, x, w, status, pinned, value, ms, onSelect }: NodeProps) {
  const c = STAGES[stage]
  const Icon = c.icon
  const active = status === 'active'
  const pending = status === 'pending'
  return (
    <button
      type="button"
      data-stage={stage}
      onClick={() => onSelect(stage)}
      aria-pressed={pinned}
      aria-label={`${c.title}: ${c.sub}${value ? `, ${value}` : ''}${ms ? `, ${ms.text} (${ms.title})` : ''}. ${pinned ? 'Закреплено в инспекторе' : 'Открыть в инспекторе'}`}
      className={cn(
        'absolute grid content-start gap-px rounded-control border px-2 py-2 text-left transition-[background-color,border-color,box-shadow] duration-200',
        active ? 'border-accent bg-accent-soft shadow-[0_0_0_1px_var(--accent)]' : 'bg-surface hover:bg-surface-2',
        !active && (pinned ? 'border-accent border-dashed' : pending ? 'border-line' : 'border-line-strong'),
      )}
      style={{ left: x, top: NODE_TOP, width: w, height: NODE_H }}
    >
      {PHASE_COLOR[stage] && (
        <span aria-hidden className="absolute top-0 right-2 left-2 h-[3px] rounded-b-[2px]" style={{ background: PHASE_COLOR[stage] }} />
      )}
      <span className="flex h-5 items-center justify-between gap-1">
        <Icon size={16} aria-hidden className={pending ? 'text-ink-3' : 'text-ink-2'} />
        <span className="flex min-w-0 items-center gap-1">
          {pinned && <PushPinIcon size={16} aria-hidden className="shrink-0 text-accent-hover" />}
          {ms && (
            <span title={ms.title} className={cn('truncate text-[11.5px] leading-4 tabular-nums', active ? 'text-ink-2' : 'text-ink-3')}>
              {ms.text}
            </span>
          )}
        </span>
      </span>
      <span className={cn('truncate text-[13px] leading-5 font-semibold tracking-[-0.01em]', pending ? 'text-ink-2' : 'text-ink')}>{c.title}</span>
      <span className={cn('truncate text-[11.5px] leading-4', active ? 'text-ink-2' : 'text-ink-3')}>{c.sub}</span>
      <span className="mt-0.5 truncate text-[12px] leading-5 text-ink tabular-nums">{value ?? '\u00a0'}</span>
    </button>
  )
})

function Packet({ move, fromX, toX, durationS, reduce }: { move: PacketMove; fromX: number; toX: number; durationS: number; reduce: boolean }) {
  const style =
    move.kind === 'kv'
      ? { background: 'var(--surface)', boxShadow: 'inset 0 0 0 1.5px var(--ink-2)' }
      : move.kind === 'meta'
        ? { background: 'var(--surface)', boxShadow: 'inset 0 0 0 1px var(--line-strong)' }
        : move.label === 'JSON'
          ? { background: 'var(--surface)', boxShadow: 'inset 0 0 0 1px var(--line-strong)' }
          : { background: mix('prefill', 22), boxShadow: `inset 0 0 0 1px ${tint('prefill', 60)}` }
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || reduce || typeof el.animate !== 'function') return
    const a = el.animate([{ transform: `translateX(${fromX}px)` }, { transform: `translateX(${toX}px)` }], {
      duration: durationS * 1000,
      easing: 'cubic-bezier(0.45, 0, 0.2, 1)',
      fill: 'forwards',
    })
    return () => a.cancel()
    // пакет едет один раз на шаг (ключ компонента меняется с шагом)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return (
    <div ref={ref} aria-hidden className="pointer-events-none absolute left-0" style={{ top: RAIL_Y - 11, transform: `translateX(${toX}px)` }}>
      <div className="-translate-x-1/2 rounded-full px-2 font-mono text-[11.5px] leading-[22px] whitespace-nowrap text-ink" style={style}>
        {move.label}
      </div>
    </div>
  )
}

/**
 * Точка токена: летит от decode к клиенту и гаснет. Web Animations API, а не React-состояние:
 * анимация живёт на compositor и не перезапускается, пока плеер перерисовывает схему каждые 50 мс.
 */
function TokenDot({ from, to, durationMs }: { from: number; to: number; durationMs: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof el.animate !== 'function') return
    const a = el.animate(
      [
        { transform: `translateX(${from}px)`, opacity: 1 },
        { transform: `translateX(${from + (to - from) * 0.85}px)`, opacity: 1, offset: 0.85 },
        { transform: `translateX(${to}px)`, opacity: 0 },
      ],
      { duration: durationMs, easing: 'linear', fill: 'forwards' },
    )
    return () => a.cancel()
    // точка запускается один раз при появлении; новые координаты (ресайз) ей не нужны
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return (
    <span
      ref={ref}
      aria-hidden
      className="pointer-events-none absolute left-0 h-2 w-2 rounded-full"
      style={{ top: RAIL_Y - 4, marginLeft: -4, background: cssVar('decode'), transform: `translateX(${from}px)` }}
    />
  )
}

export interface PipelineDiagramProps {
  run: FlowRunEx | null
  deployment: 'agg' | 'disagg'
  step: number
  speed: number
  /** Длительность текущего шага при 1×, мс. */
  stepMs: number
  pinned: StageId | null
  onSelect: (stage: StageId) => void
}

export function PipelineDiagram({ run, deployment, step, speed, stepMs, pinned, onSelect }: PipelineDiagramProps) {
  const reduce = useReducedMotion() ?? false
  const [sizeRef, { width }] = useElementSize<HTMLDivElement>()
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const stages = useMemo(() => pipelineStages(run?.settings.deployment ?? deployment), [run, deployment])
  const L = useMemo(() => layoutOf(stages, Math.max(0, width)), [stages, width])
  // ряд шире блока: обрезанный край затухает, под схемой подсказка (на телефоне полосы прокрутки не видно)
  const [edgeRef, edges] = useScrollEdges<HTMLDivElement>(L.total)
  const setScrollEl = useCallback(
    (el: HTMLDivElement | null) => {
      scrollRef.current = el
      edgeRef(el)
    },
    [edgeRef],
  )
  const pb = run?.playback ?? []
  const cur = pb[step]
  const N = run?.decode.length ?? 0
  const k = run ? tokenAtStep(pb, step, N) : -1
  const sseCount = cur?.sseCount ?? 0
  const statuses = stages.map((s) => (run ? nodeStatus(pb, step, s) : 'pending'))
  const activeIdx = Math.max(...statuses.map((s, i) => (s === 'active' ? i : -1)))
  const reachedIdx = Math.max(...statuses.map((s, i) => (s === 'pending' ? -1 : i)))
  const idx = (s: StageId) => stages.indexOf(s)
  const runKey = run ? `${run.seed}-${run.promptTokens.length}-${run.settings.deployment}` : 'none'

  // активный узел в поле зрения, если ряд не влез и прокручивается
  const focusIdx = cur?.kind === 'token' ? idx('decode') : activeIdx
  useEffect(() => {
    const el = scrollRef.current
    if (!el || focusIdx < 0 || el.scrollWidth <= el.clientWidth + 1) return
    const target = L.x[focusIdx] + L.w[focusIdx] / 2 - el.clientWidth / 2
    el.scrollTo({ left: Math.max(0, target), behavior: reduce ? 'auto' : 'smooth' })
  }, [focusIdx, L, reduce])

  const move = run ? packetMove(run, step, PACKET_LABELS) : null
  const stepS = stepMs / Math.max(0.25, speed) / 1000
  const travel = Math.max(0.18, Math.min(0.7, stepS * 0.6))
  const flight = Math.max(0.3, 0.75 / Math.sqrt(Math.max(0.25, speed)))

  // точки токенов: каждый токен летит от decode к клиенту; держим столько последних, сколько успевает долететь
  const dots: number[] = []
  const inFlight = Math.min(24, Math.ceil((flight * 1000) / Math.max(16, stepMs / Math.max(0.25, speed))) + 1)
  if (run && cur?.kind === 'token' && !reduce) for (let j = Math.max(0, k - inFlight); j <= k; j++) dots.push(j)
  const decodeX = L.cx[idx('decode')] ?? 0
  const endX = L.cx[L.cx.length - 1] ?? 0
  const railStart = L.cx[0] ?? 0
  const railReached = reachedIdx >= 0 ? L.cx[reachedIdx] : railStart

  const tp = run?.gpus ?? 1
  const agg = (run?.settings.deployment ?? deployment) === 'agg'
  const pIdx = idx('prefill')
  const dIdx = idx('decode')
  const feIdx = idx('frontend')
  const rtIdx = idx('route')
  const ppIdx = idx('postprocess')

  return (
    <div ref={sizeRef} className="min-w-0">
      <div ref={setScrollEl} className="scroll-fade -m-1.5 overflow-x-auto p-1.5 scrollbar-thin">
        <div role="group" aria-label="Схема пайплайна: узлы стадий запроса" className="relative" style={{ width: L.total, height: HEIGHT }}>
          <svg aria-hidden className="absolute inset-0 overflow-visible" width={L.total} height={HEIGHT}>
            {/* рамка воркера в agg: prefill и decode на одних GPU */}
            {agg && pIdx >= 0 && dIdx >= 0 && (
              <rect
                x={L.x[pIdx] - 7}
                y={NODE_TOP - 22}
                width={L.x[dIdx] + L.w[dIdx] - L.x[pIdx] + 14}
                height={NODE_H + 30}
                rx={12}
                fill="none"
                stroke="var(--line-strong)"
                strokeDasharray="4 4"
              />
            )}
            {/* рельс */}
            <line x1={railStart} x2={endX} y1={RAIL_Y} y2={RAIL_Y} stroke="var(--line-strong)" strokeWidth={2} strokeLinecap="round" />
            <line x1={railStart} x2={railReached} y1={RAIL_Y} y2={RAIL_Y} stroke="var(--ink-2)" strokeWidth={2} strokeLinecap="round" />
            {L.cx.slice(0, -1).map((x, i) => {
              const mid = (x + L.cx[i + 1]) / 2
              return <path key={`a${i}`} d={`M${mid - 3},${RAIL_Y - 4} L${mid + 1},${RAIL_Y} L${mid - 3},${RAIL_Y + 4}`} fill="none" stroke={i < reachedIdx ? 'var(--ink-2)' : 'var(--line-strong)'} strokeWidth={1.5} />
            })}
            {L.cx.map((x, i) => (
              <g key={`s${i}`}>
                <line x1={x} x2={x} y1={NODE_TOP + NODE_H} y2={RAIL_Y} stroke={statuses[i] === 'pending' ? 'var(--line-strong)' : 'var(--ink-3)'} strokeWidth={1} />
                <circle
                  cx={x}
                  cy={RAIL_Y}
                  r={statuses[i] === 'active' ? 5 : 4}
                  fill={statuses[i] === 'active' ? 'var(--accent)' : statuses[i] === 'done' ? 'var(--ink-2)' : 'var(--surface)'}
                  stroke={statuses[i] === 'active' ? 'var(--accent)' : statuses[i] === 'done' ? 'var(--ink-2)' : 'var(--line-strong)'}
                  strokeWidth={1.5}
                />
              </g>
            ))}
          </svg>

          {/* где работают стадии: frontend Dynamo на CPU (приём, шаблон, токены, роутер и на обратном пути postprocessor) */}
          {feIdx >= 0 && rtIdx >= 0 && (
            <div className="absolute truncate text-[12px] leading-4 text-ink-3" style={{ left: L.x[feIdx] + 2, top: NODE_TOP - 19, width: L.x[rtIdx] + L.w[rtIdx] - L.x[feIdx] - 4 }}>
              frontend Dynamo, CPU
            </div>
          )}
          {ppIdx >= 0 && (
            <div className="absolute truncate text-[12px] leading-4 text-ink-3" style={{ left: L.x[ppIdx] + 2, top: NODE_TOP - 19, width: L.w[ppIdx] - 4 }}>
              frontend, CPU
            </div>
          )}
          {/* подписи воркеров над узлами */}
          {run && agg && pIdx >= 0 && (
            <div className="absolute truncate text-[12px] leading-4 text-ink-2" style={{ left: L.x[pIdx] + 4, top: NODE_TOP - 19, width: L.x[dIdx] + L.w[dIdx] - L.x[pIdx] - 8 }}>
              {run.prefill.workerName} (vLLM): prefill и decode
            </div>
          )}
          {run && !agg && (
            <>
              <div className="absolute truncate text-[12px] leading-4 text-ink-3" style={{ left: L.x[pIdx] + 2, top: NODE_TOP - 19, width: L.w[pIdx] }}>
                {run.prefill.workerName}, {tp}× GPU
              </div>
              <div className="absolute truncate text-[12px] leading-4 text-ink-3" style={{ left: L.x[dIdx] + 2, top: NODE_TOP - 19, width: L.w[dIdx] }}>
                {run.transfer?.toName ?? ''}, {tp}× GPU
              </div>
            </>
          )}

          {stages.map((s, i) => {
            const reached = statuses[i] !== 'pending'
            return (
              <Node
                key={s}
                stage={s}
                x={L.x[i]}
                w={L.w[i]}
                status={statuses[i]}
                pinned={pinned === s}
                value={run && reached ? nodeValue(run, s, k, sseCount) : null}
                ms={run && reached ? nodeMs(run, s) : null}
                onSelect={onSelect}
              />
            )
          })}

          {move && idx(move.to) >= 0 && (
            <Packet key={`${runKey}-${step}`} move={move} fromX={L.cx[Math.max(0, idx(move.from))]} toX={L.cx[idx(move.to)]} durationS={travel} reduce={reduce} />
          )}

          {dots.map((j) => (
            <TokenDot key={`${runKey}-t${j}`} from={decodeX} to={endX} durationMs={flight * 1000} />
          ))}
        </div>
      </div>
      {edges !== 'none' && (
        <ScrollHint>
          {edges === 'start' ? 'Схема шире экрана: прокрутите вбок, слева начало пути' : 'Схема шире экрана: прокрутите вбок, справа обратный путь к клиенту'}
        </ScrollHint>
      )}
    </div>
  )
}
