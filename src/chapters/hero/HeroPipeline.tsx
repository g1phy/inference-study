/**
 * Живая схема в hero: один демо-запрос (ваш текст из настроек) один раз проходит путь туда и обратно.
 * Туда по верхнему ряду: клиент → frontend (HTTP, шаблон, токены) → router, вниз на GPU. Обратно по нижнему:
 * токены → postprocessor (текст, парсеры, SSE) → клиент. Значения узлов настоящие (buildRun), ответ печатается внизу.
 * При reduced motion сразу финальное состояние. На узкой ширине схема становится списком стадий.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useReducedMotion } from 'motion/react'
import type { Icon } from '@phosphor-icons/react'
import { ArrowCounterClockwiseIcon, ArrowsSplitIcon, DesktopIcon, GlobeSimpleIcon, GraphicsCardIcon, TreeStructureIcon } from '@phosphor-icons/react'
import { Button, cn, cssVar, fmt, Skeleton, useElementSize, useInView } from '@/ui'
import { resolveSimGpu, type FlowRunEx } from '@/engine/flow'
import type { TokenizerStatus } from '@/engine/types'
import { clientView, excerpt } from './model'

type NodeId = 'client' | 'frontend' | 'route' | 'gpu' | 'postprocess'

/** Фазы демо и их длительность, мс. */
const PHASES = [
  { id: 'client', ms: 700 },
  { id: 'frontend', ms: 800 },
  { id: 'preprocess', ms: 850 },
  { id: 'route', ms: 850 },
  { id: 'prefill', ms: 1000 },
  { id: 'decode', ms: 2500 },
] as const
type PhaseId = (typeof PHASES)[number]['id']
const TOTAL = PHASES.reduce((s, p) => s + p.ms, 0)
const TRAVEL = 450

function phaseAt(t: number): { i: number; local: number } {
  if (t < 0) return { i: -1, local: 0 }
  let acc = 0
  for (let i = 0; i < PHASES.length; i++) {
    if (t < acc + PHASES[i].ms) return { i, local: t - acc }
    acc += PHASES[i].ms
  }
  return { i: PHASES.length, local: 0 }
}

/** Часы демо: стартуют один раз, когда схема видна и прогон готов. */
function useDemoClock(ready: boolean, reduce: boolean): { t: number; replay: () => void } {
  const [start, setStart] = useState<number | null>(null)
  const [t, setT] = useState(-1)
  const started = useRef(false)
  useEffect(() => {
    if (!ready || started.current) return
    started.current = true
    if (reduce) setT(TOTAL)
    else setStart(performance.now())
  }, [ready, reduce])
  useEffect(() => {
    if (start === null) return
    let raf = 0
    const tick = () => {
      const now = performance.now() - start
      if (now >= TOTAL) {
        setT(TOTAL)
        return
      }
      // шаг 40 мс: React перерисовывает схему примерно 25 раз в секунду, не на каждый кадр
      setT(Math.floor(now / 40) * 40)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [start])
  const replay = () => {
    if (reduce) return
    setT(0)
    setStart(performance.now())
  }
  return { t, replay }
}

const NODES: Record<NodeId, { title: string; sub: string; icon: Icon }> = {
  client: { title: 'Клиент', sub: 'ваше приложение', icon: DesktopIcon },
  frontend: { title: 'Frontend', sub: 'HTTP, шаблон, токены', icon: GlobeSimpleIcon },
  route: { title: 'Router', sub: 'выбор воркера', icon: ArrowsSplitIcon },
  gpu: { title: 'GPU-воркер', sub: 'prefill, потом decode', icon: GraphicsCardIcon },
  postprocess: { title: 'Postprocessor', sub: 'текст, парсеры, SSE', icon: TreeStructureIcon },
}

/** В каких фазах узел активен: [первая, последняя]. Frontend делает и HTTP, и шаблон с токенизацией. */
const NODE_PHASE: Record<NodeId, [number, number]> = {
  client: [0, 0],
  frontend: [1, 2],
  route: [3, 3],
  gpu: [4, 5],
  postprocess: [5, 5],
}

type Status = 'pending' | 'active' | 'done'
function statusOf(node: NodeId, phase: number): Status {
  const [a, b] = NODE_PHASE[node]
  if (phase < a) return 'pending'
  if (phase <= b) return 'active'
  if (node === 'client' && phase === 5) return 'active'
  return 'done'
}

interface Box {
  x: number
  y: number
  w: number
  h: number
}

const ROW_H = 66
const BAND = 50

function loopLayout(W: number): Record<NodeId, Box> & { height: number; gap: number } {
  const gap = W < 520 ? 22 : 30
  const c0 = Math.round(Math.min(124, Math.max(100, W * 0.21)))
  const c = (W - c0 - 2 * gap) / 2
  const x1 = c0 + gap
  const x2 = x1 + c + gap
  const yb = ROW_H + BAND
  return {
    client: { x: 0, y: 0, w: c0, h: ROW_H * 2 + BAND },
    frontend: { x: x1, y: 0, w: c, h: ROW_H },
    route: { x: x2, y: 0, w: c, h: ROW_H },
    gpu: { x: x2, y: yb, w: c, h: ROW_H },
    postprocess: { x: x1, y: yb, w: c, h: ROW_H },
    height: ROW_H * 2 + BAND,
    gap,
  }
}

interface DemoData {
  values: Record<NodeId, string>
  status: (phase: PhaseId | 'done' | 'idle', local: number) => string
  textAt: (frac: number) => { reasoning: string; content: string }
  gpuValue: (phase: number, frac: number) => string
}

function demoData(run: FlowRunEx): DemoData {
  const N = run.decode.length
  const gpu = resolveSimGpu(run.gpuId ?? '').gpu.name
  const tokenSteps = (run.playback ?? []).filter((p) => p.kind === 'token')
  const final = clientView(run.sse, run.sse.length)
  // (X-sim-r2-4) обычный шаг decode: шаг с prefill соседа в среднее не берём
  const decodeAvg = run.neighbor?.applied
    ? run.neighbor.plainMs
    : run.metrics.decodeMs && N > 1
      ? run.metrics.decodeMs / (N - 1)
      : run.metrics.itlMs
  const values: Record<NodeId, string> = {
    client: `первый чанк через ${fmt.ms(run.metrics.ttftMs)}`,
    frontend: `${fmt.tokens(run.promptTokens.length)} промпта`,
    route: `→ ${run.prefill.workerName ?? run.prefill.worker}`,
    gpu: `${fmt.tokens(N)} ответа`,
    postprocess: `${final.tools.length ? 'tool_calls, ' : ''}${fmt.count(run.sse.filter((c) => c.json).length, ['чанк', 'чанка', 'чанков'])} SSE`,
  }
  return {
    values,
    gpuValue: (phase, frac) =>
      phase === 4 ? `prefill ${fmt.ms(run.prefill.ms)}` : phase === 5 ? `decode ${fmt.int(Math.max(1, Math.round(frac * N)))} / ${fmt.int(N)}` : values.gpu,
    textAt: (frac) => {
      const k = Math.max(0, Math.min(N - 1, Math.floor(frac * N)))
      const v = clientView(run.sse, tokenSteps[k]?.sseCount ?? 0)
      return { reasoning: v.reasoning, content: v.content }
    },
    status: (phase) => {
      switch (phase) {
        case 'idle':
          return 'Демо-запрос сейчас отправится.'
        case 'client':
          return 'Клиент шлёт POST /v1/chat/completions: JSON с сообщениями, stream: true.'
        case 'frontend':
          return 'Frontend проверяет запрос и собирает параметры генерации.'
        case 'preprocess':
          return `Шаблон Qwen3 склеил сообщения в текст, токенизатор нарезал ${fmt.tokens(run.promptTokens.length)}.`
        case 'route':
          return `Роутер выбрал ${run.prefill.workerName}: ${run.prefill.cachedTokens > 0 ? 'у него в KV cache уже есть начало промпта' : 'он сейчас свободнее остальных'}.`
        case 'prefill':
          return `Prefill: весь промпт за один проход через модель, ${fmt.ms(run.prefill.ms)} на ${gpu}.`
        case 'decode':
          return `Decode: по токену за шаг, около ${fmt.ms(decodeAvg)} на токен; текст сразу уходит клиенту чанками SSE.`
        case 'done':
          return `Готово: первый чанк через ${fmt.ms(run.metrics.ttftMs)}, весь ответ за ${fmt.ms(run.metrics.e2eMs)}. Ниже тот же путь по шагам.`
      }
    },
  }
}

export function HeroPipeline({ run, status }: { run: FlowRunEx | null; status: TokenizerStatus }) {
  const reduce = useReducedMotion() ?? false
  const [sizeRef, { width }] = useElementSize<HTMLDivElement>()
  const [viewRef, inView] = useInView<HTMLDivElement>({ threshold: 0.35 })
  const { t, replay } = useDemoClock(!!run && inView, reduce)
  const data = useMemo(() => (run ? demoData(run) : null), [run])
  const { i: phase, local } = phaseAt(t)
  const phaseId: PhaseId | 'done' | 'idle' = phase < 0 ? 'idle' : phase >= PHASES.length ? 'done' : PHASES[phase].id
  const decodeFrac = phase === 5 ? Math.min(1, local / PHASES[5].ms) : phase > 5 ? 1 : 0
  const text = data ? (phase >= 5 ? data.textAt(decodeFrac) : { reasoning: '', content: '' }) : null
  // список вместо петли только на совсем узкой колонке: при 400 px и шире блоки петли вмещают «Postprocessor»
  const narrow = width > 0 && width < 400
  const done = phaseId === 'done'

  const valueOf = (n: NodeId, st: Status): string | null => {
    if (!data || st === 'pending') return null
    if (n === 'gpu') return data.gpuValue(phase, decodeFrac)
    if (n === 'client') return phase >= 6 ? data.values.client : phase === 5 ? 'получает ответ' : 'шлёт запрос'
    if (n === 'frontend' && phase === 1) return 'stream: true'
    if (n === 'postprocess' && phase === 5) return `${fmt.count(Math.max(1, Math.round(decodeFrac * (run!.sse.length - 3))), ['чанк', 'чанка', 'чанков'])} клиенту`
    return data.values[n]
  }

  return (
    <div
      ref={(el) => {
        sizeRef(el)
        viewRef(el)
      }}
      className="grid grid-cols-[minmax(0,1fr)] min-w-0 gap-4"
    >
      {width > 0 &&
        (narrow ? (
          <ListLayout phase={phase} valueOf={valueOf} />
        ) : (
          <LoopLayout width={width} phase={phase} local={local} reduce={reduce} valueOf={valueOf} />
        ))}

      {/* высота карточки и строки статуса постоянная: ответ печатается внутри, страница под hero не прыгает */}
      <div className="grid grid-cols-[minmax(0,1fr)] gap-2 rounded-panel border border-line bg-surface px-4 py-3 text-[13.5px] leading-[1.45]">
        {!run ? (
          <>
            <div className="flex h-[1.45em] items-center">
              <Skeleton height={12} width="70%" label="Загружаем токенизатор" />
            </div>
            <div className="grid h-[2.9em] content-center gap-[0.6em]">
              <Skeleton height={12} width="90%" label="" />
              <Skeleton height={12} width="55%" label="" />
            </div>
          </>
        ) : (
          <>
            <p className="h-[1.45em] truncate text-ink">
              <span className="text-ink-3">Вы: </span>
              {excerpt(run.settings.userText || 'пустое сообщение', 96)}
            </p>
            <AnswerLine run={run} text={done || reduce ? null : text} started={phase >= 5} />
          </>
        )}
      </div>

      <div className="flex min-h-[4.2em] items-start justify-between gap-3 text-[13.5px] leading-[1.4] sm:min-h-[2.8em]">
        <p className="text-ink-2">
          {status === 'error' ? 'Не удалось загрузить токенизатор: схема без живых значений.' : data ? data.status(phaseId, local) : 'Загружаем словарь Qwen3, чтобы посчитать ваш запрос.'}
        </p>
        {done && !reduce && (
          <Button size="sm" variant="ghost" icon={ArrowCounterClockwiseIcon} onClick={replay} aria-label="Повторить демо-запрос" className="shrink-0">
            Ещё раз
          </Button>
        )}
      </div>
    </div>
  )
}

/** Хвост текста для печатающейся строки: последние max символов по границе слова (курсор всегда виден). */
function tailExcerpt(text: string, max: number): string {
  const s = text.replace(/\s+/g, ' ').trim()
  if (s.length <= max) return s
  const cut = s.slice(s.length - max)
  const sp = cut.indexOf(' ')
  return `…${sp >= 0 && sp < max * 0.4 ? cut.slice(sp + 1) : cut}`
}

function AnswerLine({ run, text, started }: { run: FlowRunEx; text: { reasoning: string; content: string } | null; started: boolean }) {
  const final = run.output
  const reasoning = text ? text.reasoning : ''
  const content = text ? text.content : final?.content ?? ''
  const tool = final?.toolCalls[0]
  const showContent = content.replace(/^\n+/, '').trim()
  // пока печатается, показываем хвост (там курсор), в конце начало ответа
  const streaming = !!text && started
  return (
    <div className="line-clamp-2 h-[2.9em] text-ink">
      <span className="text-ink-3">Модель: </span>
      {text && !started ? (
        <span className="text-ink-3">…</span>
      ) : showContent ? (
        <span>{streaming ? tailExcerpt(showContent, 110) : excerpt(showContent, 150)}</span>
      ) : reasoning ? (
        <>
          <span className="mr-1 inline-flex items-center gap-1 align-baseline font-mono text-[12px] text-ink-3">
            <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-[3px] border border-dashed border-ink-3" />
            reasoning
          </span>
          <span className="text-ink-2 italic">{streaming ? tailExcerpt(reasoning, 90) : excerpt(reasoning.trim(), 90)}</span>
        </>
      ) : tool && !text ? (
        <span className="font-mono text-[12.5px]">
          {tool.name}({JSON.stringify(tool.arguments)})
        </span>
      ) : (
        <span className="text-ink-3">…</span>
      )}
      {streaming && <span aria-hidden className="ml-0.5 inline-block h-[1em] w-[3px] translate-y-[0.15em] rounded-[1px]" style={{ background: cssVar('decode') }} />}
    </div>
  )
}

/* ============================================================================
 * Петля: верхний ряд туда, нижний обратно
 * ========================================================================== */

function NodeBox({ id, box, status, value, swatch }: { id: NodeId; box: Box; status: Status; value: string | null; swatch?: string }) {
  const n = NODES[id]
  const I = n.icon
  const tall = id === 'client'
  return (
    <div
      className={cn(
        'absolute grid rounded-control border px-2.5 py-2 transition-[background-color,border-color,box-shadow] duration-300',
        tall ? 'content-between' : 'content-start gap-0.5',
        status === 'active' ? 'border-accent bg-accent-soft shadow-[0_0_0_1px_var(--accent)]' : status === 'done' ? 'border-line-strong bg-surface' : 'border-line bg-surface',
      )}
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
    >
      <div className="grid grid-cols-[minmax(0,1fr)] gap-0.5">
        <div className="flex min-w-0 items-center gap-1.5">
          <I size={16} aria-hidden className={status === 'pending' ? 'shrink-0 text-ink-3' : 'shrink-0 text-ink-2'} />
          <span className={cn('truncate text-[13px] leading-5 font-semibold', status === 'pending' ? 'text-ink-2' : 'text-ink')}>{n.title}</span>
        </div>
        {(tall || !value) && (
          <span className={cn('text-[12px] leading-4', status === 'active' ? 'text-ink-2' : 'text-ink-3', tall ? 'whitespace-normal' : 'truncate')}>{n.sub}</span>
        )}
      </div>
      {value && (
        <span className={cn('truncate text-[12.5px] leading-5 text-ink tabular-nums', tall && 'whitespace-normal')}>
          {swatch && <span aria-hidden className="mr-1.5 inline-block h-2.5 w-2.5 rounded-[3px] align-[-1px]" style={{ background: swatch }} />}
          {value}
        </span>
      )}
    </div>
  )
}

interface EdgeDef {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
  /** Фаза, в которой по ребру едет запрос (−1: ребро обратного пути). */
  phase: number
  back?: boolean
}

function LoopLayout({
  width,
  phase,
  local,
  reduce,
  valueOf,
}: {
  width: number
  phase: number
  local: number
  reduce: boolean
  valueOf: (n: NodeId, st: Status) => string | null
}) {
  const L = loopLayout(width)
  const midTop = ROW_H / 2
  const midBot = ROW_H + BAND + ROW_H / 2
  const edges: EdgeDef[] = [
    { id: 'e1', x1: L.client.w, y1: midTop, x2: L.frontend.x, y2: midTop, phase: 1 },
    { id: 'e2', x1: L.frontend.x + L.frontend.w, y1: midTop, x2: L.route.x, y2: midTop, phase: 3 },
    { id: 'e3', x1: L.route.x + L.route.w / 2, y1: ROW_H, x2: L.route.x + L.route.w / 2, y2: ROW_H + BAND, phase: 4 },
    { id: 'e4', x1: L.gpu.x, y1: midBot, x2: L.postprocess.x + L.postprocess.w, y2: midBot, phase: 5, back: true },
    { id: 'e5', x1: L.postprocess.x, y1: midBot, x2: L.client.w, y2: midBot, phase: 5, back: true },
  ]
  const ids: NodeId[] = ['client', 'frontend', 'route', 'gpu', 'postprocess']
  return (
    <div
      role="img"
      aria-label="Схема пути запроса: клиент, frontend (шаблон и токены), router, GPU-воркер, затем обратно через postprocessor и стрим SSE к клиенту."
      className="relative"
      style={{ height: L.height }}
    >
      <svg aria-hidden className="absolute inset-0 overflow-visible" width={width} height={L.height}>
        {edges.map((e) => (
          <Edge key={e.id} e={e} phase={phase} local={local} reduce={reduce} />
        ))}
      </svg>
      <span className="absolute text-[12px] leading-4 text-ink-3" style={{ left: L.frontend.x + 2, top: ROW_H + 8 }}>
        туда: JSON, потом номера токенов
      </span>
      <span className="absolute text-[12px] leading-4 text-ink-3" style={{ left: L.postprocess.x + 2, top: ROW_H + BAND - 22 }}>
        обратно: токены, текст, чанки SSE
      </span>
      {ids.map((id) => {
        const st = statusOf(id, phase)
        return <NodeBox key={id} id={id} box={L[id]} status={st} value={valueOf(id, st)} swatch={id === 'gpu' ? gpuSwatch(phase) : undefined} />
      })}
    </div>
  )
}

/** Цвет фазы у значения GPU-узла: prefill синий, decode оранжевый (как на таймлайне симулятора). */
function gpuSwatch(phase: number): string | undefined {
  return phase === 4 ? cssVar('prefill') : phase === 5 ? cssVar('decode') : undefined
}

function Edge({ e, phase, local, reduce }: { e: EdgeDef; phase: number; local: number; reduce: boolean }) {
  const len = Math.hypot(e.x2 - e.x1, e.y2 - e.y1)
  const traversed = phase > e.phase || (phase === e.phase && (e.back || local >= TRAVEL))
  const travelling = !reduce && phase === e.phase && !e.back && local < TRAVEL
  const flowing = !reduce && e.back && phase === e.phase
  const dx = (e.x2 - e.x1) / (len || 1)
  const dy = (e.y2 - e.y1) / (len || 1)
  const ax = e.x2 - dx * 1
  const ay = e.y2 - dy * 1
  const head = `M${ax - dx * 6 - dy * 4},${ay - dy * 6 + dx * 4} L${ax},${ay} L${ax - dx * 6 + dy * 4},${ay - dy * 6 - dx * 4}`
  const base = traversed ? 'var(--ink-3)' : 'var(--line-strong)'
  return (
    <g>
      <line x1={e.x1} y1={e.y1} x2={e.x2 - dx * 2} y2={e.y2 - dy * 2} stroke={base} strokeWidth={1.5} />
      <path d={head} fill="none" stroke={base} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      {travelling && (
        <DashLine
          e={e}
          stroke="var(--accent)"
          dash={`${Math.max(8, len * 0.45)} ${len * 2}`}
          from={Math.max(8, len * 0.45)}
          to={-len}
          durationMs={TRAVEL}
          easing="ease-in-out"
          iterations={1}
        />
      )}
      {flowing && <DashLine e={e} stroke={cssVar('decode')} dash="3 7" from={0} to={-40} durationMs={900} easing="linear" iterations={Infinity} />}
    </g>
  )
}

/** Линия с бегущим пунктиром (Web Animations API: не перезапускается от перерисовок часов демо). */
function DashLine({
  e,
  stroke,
  dash,
  from,
  to,
  durationMs,
  easing,
  iterations,
}: {
  e: EdgeDef
  stroke: string
  dash: string
  from: number
  to: number
  durationMs: number
  easing: string
  iterations: number
}) {
  const ref = useRef<SVGLineElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof el.animate !== 'function') return
    const a = el.animate([{ strokeDashoffset: from }, { strokeDashoffset: to }], { duration: durationMs, easing, iterations, fill: 'forwards' })
    return () => a.cancel()
  }, [from, to, durationMs, easing, iterations])
  return <line ref={ref} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={stroke} strokeWidth={3} strokeLinecap="round" strokeDasharray={dash} strokeDashoffset={from} />
}

/* ============================================================================
 * Список: узкий экран
 * ========================================================================== */

function ListLayout({ phase, valueOf }: { phase: number; valueOf: (n: NodeId, st: Status) => string | null }) {
  const rows: Array<{ id: NodeId; note?: ReactNode }> = [{ id: 'client' }, { id: 'frontend' }, { id: 'route' }, { id: 'gpu' }, { id: 'postprocess' }]
  return (
    <ol aria-label="Путь запроса по стадиям" className="relative grid gap-1">
      <span aria-hidden className="absolute top-3 bottom-3 left-[11px] w-px bg-line-strong" />
      {rows.map(({ id }) => {
        const st = statusOf(id, phase)
        const n = NODES[id]
        const I = n.icon
        const v = valueOf(id, st)
        return (
          <li
            key={id}
            className={cn(
              'relative grid grid-cols-[24px_18px_minmax(0,1fr)_auto] items-center gap-2 rounded-control py-1.5 pr-2 transition-colors duration-300',
              st === 'active' && 'bg-accent-soft',
            )}
          >
            <span
              aria-hidden
              className="z-[1] mx-auto h-2.5 w-2.5 rounded-full border"
              style={{
                background: st === 'active' ? 'var(--accent)' : st === 'done' ? 'var(--ink-2)' : 'var(--surface)',
                borderColor: st === 'active' ? 'var(--accent)' : st === 'done' ? 'var(--ink-2)' : 'var(--line-strong)',
              }}
            />
            <I size={16} aria-hidden className="text-ink-2" />
            <span className="min-w-0 truncate text-[13.5px] font-semibold text-ink">{n.title}</span>
            <span className="text-[12.5px] text-ink tabular-nums">
              {v && id === 'gpu' && gpuSwatch(phase) && (
                <span aria-hidden className="mr-1.5 inline-block h-2.5 w-2.5 rounded-[3px] align-[-1px]" style={{ background: gpuSwatch(phase) }} />
              )}
              {v ?? ''}
            </span>
          </li>
        )
      })}
    </ol>
  )
}
