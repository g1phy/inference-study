import { useEffect, useMemo, useRef } from 'react'
import { HourglassMediumIcon } from '@phosphor-icons/react'
import { CodeBlock, Figure, Player, Skeleton, Stat, Toggle, cn, fmt, usePlayer } from '@/ui'
import type { ConversationSim } from './responseModel'
import { compactChunk, completionJson } from './responseModel'

export interface ResponseCompareProps {
  sim: ConversationSim | null
  warm: boolean
  onWarm: (v: boolean) => void
}

/**
 * «Ответ: целиком или потоком». Время идёт по кадрам: кадр 0 — запрос отправлен, кадр i — пришёл i-й токен.
 * Сетка 2×2 на любой ширине: слева stream: false, справа stream: true; сверху «экран» пользователя, снизу «провод».
 * Экраны всегда стоят рядом (на телефоне тоже): ради этого сравнения фигура и существует.
 * Плеер в нижней панели один, поэтому на телефоне он липнет к низу экрана (Figure, data-sticky).
 */
export function ResponseCompare({ sim, warm, onWarm }: ResponseCompareProps) {
  const frames = sim ? sim.frameTimes.length : 1
  const p = usePlayer(frames, { intervalMs: (s) => (s === 0 ? 900 : 260) })
  const step = Math.min(p.step, frames - 1)
  const last = frames - 1

  const labels = useMemo(() => {
    if (!sim) return []
    return sim.frameTimes.map((t, i) => {
      if (i === 0) return 'запрос отправлен'
      const tag = i === 1 ? ', первый токен' : i === last ? ', последний токен' : ''
      return `t = ${fmt.ms(t)}${tag}`
    })
  }, [sim, last])

  return (
    <Figure
      id="request-figure-response"
      title="Ответ: целиком или потоком"
      inputs={<Toggle label="Повторный запрос" checked={warm} onChange={onWarm} hint="начало промпта уже лежит в prefix cache" />}
      player={<Player {...p.bind} labels={labels} aria-label="Время ответа" />}
      caption="Попробуйте: нажмите пуск. В левой колонке клиент ждёт весь ответ, в правой текст появляется с первым токеном. Включите повторный запрос: в usage вырастет cached_tokens, а TTFT на таком коротком промпте почти не изменится."
      source={
        sim ? (
          <>
            Время посчитано по perf.ts: {sim.modelName} на {sim.gpuName}
            {sim.gpus > 1 ? ` (TP${sim.gpus})` : ''}, один запрос на свободной GPU, без сети, очереди и CPU-стадий. Текст ответа
            условный, токены, чанки и usage настоящие.
          </>
        ) : undefined
      }
    >
      {!sim ? (
        <div className="grid grid-cols-2 gap-3 @min-[680px]:gap-4">
          <Skeleton lines={6} label="Загружаем токенизатор" />
          <Skeleton lines={6} label="Загружаем токенизатор" />
        </div>
      ) : (
        <>
          <Timeline sim={sim} step={step} />
          <div className="mt-4 grid grid-cols-2 gap-x-3 gap-y-2 @min-[680px]:gap-x-4">
            <PanelHead title='"stream": false' note="один JSON в конце" />
            <PanelHead title='"stream": true' note="чанки по мере генерации" />
            <Screen sim={sim} step={step} last={last} stream={false} />
            <Screen sim={sim} step={step} last={last} stream />
            {/* провод: на узком экране во всю ширину (сначала поток SSE), от 680px под своими экранами */}
            <SseWire sim={sim} step={step} last={last} className="col-span-2 @min-[680px]:col-span-1 @min-[680px]:col-start-2 @min-[680px]:row-start-3" />
            <JsonWire sim={sim} step={step} last={last} className="col-span-2 @min-[680px]:col-span-1 @min-[680px]:col-start-1 @min-[680px]:row-start-3" />
          </div>
          <div className="mt-5 grid grid-cols-2 gap-x-6 gap-y-4 @min-[680px]:grid-cols-4">
            <Stat label="Текст со стримом" value={fmt.ms(sim.ttftMs)} hint="первые слова через TTFT" swatch="var(--c-prefill)" />
            <Stat label="Текст без стрима" value={fmt.ms(sim.e2eMs)} hint="весь ответ сразу" />
            <Stat label="Токенов в ответе" value={fmt.int(sim.answerTokens.length)} hint="по словарю Qwen3" swatch="var(--c-decode)" />
            <Stat
              label="Из prefix cache"
              value={fmt.int(sim.cachedTokens)}
              hint={`из ${fmt.int(sim.promptTokens)} ${fmt.plural(sim.promptTokens, ['токена', 'токенов', 'токенов'])} промпта`}
            />
          </div>
        </>
      )}
    </Figure>
  )
}

function Timeline({ sim, step }: { sim: ConversationSim; step: number }) {
  const total = Math.max(sim.e2eMs, 1)
  const t = sim.frameTimes[step] ?? 0
  const ttftPct = (sim.ttftMs / total) * 100
  const headPct = (t / total) * 100
  // prefill на коротком промпте занимает полтора процента шкалы: не тоньше 3px, и отметка TTFT под шкалой
  const edge = `max(3px, ${ttftPct}%)`
  return (
    <div className="grid gap-1.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="h-2.5 w-2.5 rounded-[3px] bg-prefill" />
          prefill: до первого токена
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="h-2.5 w-2.5 rounded-[3px] bg-decode" />
          decode: по токену за шаг ({fmt.ms(sim.itlMs)})
        </span>
        <span className="ml-auto font-mono text-[12px] text-ink tabular-nums">t = {fmt.ms(t)}</span>
      </div>
      <div
        className="relative h-3 overflow-hidden rounded-full bg-surface-2"
        role="img"
        aria-label={`Шкала времени: prefill ${fmt.ms(sim.ttftMs)}, весь ответ ${fmt.ms(sim.e2eMs)}`}
      >
        <div className="absolute inset-y-0 left-0 bg-prefill/70" style={{ width: edge }} />
        <div className="absolute inset-y-0 bg-decode/70" style={{ left: edge, right: 0 }} />
        <div className="absolute inset-y-0 left-0 bg-surface/55" style={{ left: `${headPct}%`, right: 0 }} />
        <div className="absolute inset-y-[-2px] w-0.5 bg-ink" style={{ left: `calc(${headPct}% - 1px)` }} />
      </div>
      <div aria-hidden className="relative h-5 text-[12px] text-ink-3 tabular-nums">
        <span className="absolute top-0 left-0">0</span>
        <span className="absolute -top-1.5 h-1.5 w-px bg-ink-2" style={{ left: edge }} />
        <span
          className="absolute top-0 font-medium whitespace-nowrap text-ink-2"
          style={{ left: `clamp(1.25rem, ${ttftPct}%, calc(100% - 10rem))` }}
        >
          TTFT {fmt.ms(sim.ttftMs)}
        </span>
        <span className="absolute top-0 right-0">{fmt.ms(sim.e2eMs)}</span>
      </div>
    </div>
  )
}

function PanelHead({ title, note }: { title: string; note: string }) {
  return (
    <div className="flex min-w-0 flex-wrap items-baseline justify-between gap-x-3">
      <span className="font-mono text-[13px] font-semibold text-ink">{title}</span>
      <span className="text-[12.5px] text-ink-3">{note}</span>
    </div>
  )
}

/** «Экран» пользователя: без стрима пусто до последнего токена, со стримом текст растёт с каждым чанком. */
function Screen({ sim, step, last, stream }: { sim: ConversationSim; step: number; last: number; stream: boolean }) {
  const waiting = !stream && step < last
  const text = sim.textAt[stream ? step : last] ?? { reasoning: '', content: '' }
  const empty = !text.reasoning && !text.content
  return (
    <div className="min-w-0 rounded-panel border border-line bg-surface px-2.5 py-2 @min-[680px]:px-3 @min-[680px]:py-2.5">
      <div className="mb-1 text-[12px] text-ink-3">
        На экране<span className="hidden @min-[680px]:inline"> у пользователя</span>
      </div>
      <div className="min-h-[112px] text-[13px] leading-snug @min-[680px]:min-h-[104px] @min-[680px]:text-[14.5px] @min-[680px]:leading-relaxed">
        {waiting ? (
          <div className="grid justify-items-center gap-1.5 py-2 text-center">
            <HourglassMediumIcon size={20} aria-hidden className="text-ink-3" />
            <span className="text-ink-2">Клиент ждёт весь ответ</span>
            <span className="font-mono text-[12px] text-ink-3 tabular-nums">
              {step === 0 ? 'запрос отправлен' : `${fmt.int(step)} из ${fmt.int(last)} токенов готово, на экране пусто`}
            </span>
          </div>
        ) : empty ? (
          <span className="text-ink-3">{step === 0 ? 'пока пусто' : '…'}</span>
        ) : (
          <>
            {text.reasoning && (
              <p className="mb-1.5 border-l-2 border-dashed border-reasoning pl-2 text-[0.93em] text-ink-2 italic">{text.reasoning.trim()}</p>
            )}
            {text.content && <p className="text-ink">{text.content.replace(/^\n+/, '')}</p>}
          </>
        )}
      </div>
    </div>
  )
}

function WireLabel({ children }: { children: string }) {
  return <div className="mb-1 text-[12px] text-ink-3">{children}</div>
}

/** stream: true на проводе: строки text/event-stream по мере генерации. */
function SseWire({ sim, step, last, className }: { sim: ConversationSim; step: number; last: number; className?: string }) {
  const shown = sim.chunks.filter((c) => c.frame <= step)
  const logRef = useRef<HTMLDivElement>(null)
  // лента прокручивается к последнему чанку внутри себя (страницу не двигаем)
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [shown.length])
  const newest = shown.length ? shown[shown.length - 1].frame : -1
  return (
    <div className={cn('min-w-0', className)}>
      <WireLabel>На проводе: text/event-stream</WireLabel>
      <div
        ref={logRef}
        className="h-[140px] overflow-y-auto rounded-panel bg-surface-2 px-2 py-2 font-mono text-[11.5px] leading-[1.55] shadow-[inset_0_0_0_1px_var(--line)] scrollbar-thin @min-[680px]:h-[240px]"
        tabIndex={0}
        aria-label="Лента SSE-чанков"
      >
        {shown.length === 0 ? (
          <span className="px-1 text-ink-3">{step >= last ? '' : 'чанков ещё нет'}</span>
        ) : (
          shown.map((c, i) => (
            <div
              key={i}
              className={cn('rounded-[4px] px-1 [overflow-wrap:anywhere]', c.frame === newest ? 'bg-accent-soft text-ink' : 'text-ink-2')}
            >
              {/* первый чанк полностью: видно, что «шапка» повторяется в каждом */}
              {i === 0 ? c.chunk.line : compactChunk(c.chunk)}
            </div>
          ))
        )}
      </div>
    </div>
  )
}

/** stream: false на проводе: до конца генерации ни байта, потом один chat.completion. */
function JsonWire({ sim, step, last, className }: { sim: ConversationSim; step: number; last: number; className?: string }) {
  const done = step >= last
  const json = useMemo(() => completionJson(sim), [sim])
  const ref = useRef<HTMLDivElement>(null)
  // когда ответ пришёл, показываем его конец: finish_reason и usage (с cached_tokens), на которые смотрит подпись
  useEffect(() => {
    if (!done) return
    const pre = ref.current?.querySelector('pre')
    if (pre) pre.scrollTop = pre.scrollHeight
  }, [done, json])
  return (
    <div ref={ref} className={cn('min-w-0', className)}>
      <WireLabel>На проводе: application/json</WireLabel>
      {done ? (
        <CodeBlock lang="json" code={json} wrap maxHeight={203} title="chat.completion" />
      ) : (
        <div className="grid place-content-center rounded-panel border border-dashed border-line-strong bg-surface-2 px-3 py-3 text-center font-mono text-[12px] text-ink-3 @min-[680px]:h-[240px]">
          {step === 0 ? 'запрос отправлен, ответа пока нет' : 'ни байта: сервер копит ответ до конца'}
        </div>
      )}
    </div>
  )
}
