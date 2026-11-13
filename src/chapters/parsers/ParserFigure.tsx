import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { Figure, Player, Segmented, TokenChip, TokenStripSkeleton, cn, tint, usePlayer } from '@/ui'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { runStreamingParser, toSseChunks, type ParserEventEx, type SseMeta } from '@/engine/parsers'
import type { Channel, SseChunk } from '@/engine/types'
import { StateDiagram } from './StateDiagram'
import { SCENARIOS, buildStream, compactLine, lanesAt, type ChunkMode, type LaneState, type ScenarioId, type StreamInput } from './scenarios'

const META: SseMeta = { id: 'chatcmpl-7f3a', created: 1762950000, model: 'Qwen/Qwen3-8B' }

interface FigureData {
  stream: StreamInput
  events: ParserEventEx[]
  finish: SseChunk[]
}

/** Строки SSE, которые порождает одно событие парсера (без чанка с ролью и финальных строк). */
function sseOf(ev: ParserEventEx): SseChunk[] {
  const all = toSseChunks([ev], META)
  return all.slice(1, -2)
}

export function ParserFigure() {
  const { tokenizer, status } = useTokenizer()
  // по умолчанию сценарий с вызовом: в нём видны все три поля, о них текст после фигуры
  const [scenario, setScenario] = useState<ScenarioId>('tool')
  const [mode, setMode] = useState<ChunkMode>('tokens')

  const data = useMemo<FigureData | null>(() => {
    if (!tokenizer) return null
    const sc = SCENARIOS.find((s) => s.id === scenario)!
    const stream = buildStream(tokenizer, sc.text, mode)
    const events = runStreamingParser(stream.chunks, { reasoning: 'qwen3', tools: 'hermes', idSeed: scenario })
    const finish = toSseChunks(events, META).slice(-2)
    return { stream, events, finish }
  }, [tokenizer, scenario, mode])

  const total = data ? data.events.length + 1 : 1
  const p = usePlayer(total, { intervalMs: 900 })
  const { setStep } = p
  useEffect(() => {
    setStep(0)
  }, [scenario, mode, setStep])

  const labels = useMemo(() => {
    if (!data) return undefined
    const l = data.events.map((e) => (e.input ? `кусок «${e.input.replace(/\n/g, '↵').replace(/ /g, '·')}»` : 'конец потока: сброс буфера'))
    l.push('<|im_end|>: конец генерации')
    return l
  }, [data])

  const step = Math.min(p.step, total - 1)
  const isEnd = data ? step === data.events.length : false
  const ev = data && !isEnd ? data.events[step] : null
  const lanesNow = data ? lanesAt(data.events, Math.min(step, data.events.length - 1)) : null
  // на шаге <|im_end|> ничего нового не пришло: подсветку «добавлено на этом шаге» снимаем
  const lanes = lanesNow && isEnd ? { ...lanesNow, added: { reasoning: '', content: '', toolName: false, toolArgs: '' } } : lanesNow
  // итоговые поля: по ним резервируется высота дорожек, чтобы плеер не уезжал вниз по ходу проигрывания
  const finalLanes = useMemo(() => (data ? lanesAt(data.events, data.events.length - 1) : null), [data])
  // самые длинные пояснение, буфер и тело вызова прогона: по ним держится высота панели «Что сделал парсер»
  const stepReserve = useMemo(() => (data ? reserveOf(data.events) : null), [data])
  // на шаге <|im_end|> панель шага невидима, но остаётся на месте (с последним событием), чтобы высота не менялась
  const infoStep = data ? (isEnd ? data.events.length - 1 : step) : 0
  const infoEv = data ? data.events[infoStep] : null
  const body = data && infoEv ? callBody(data.events, infoStep) : null
  const prevChannel: Channel = data && step > 0 ? data.events[Math.min(step, data.events.length) - 1].channelAfter : 'content'
  const curChannel: Channel = ev ? ev.channelAfter : prevChannel

  return (
    <Figure
      id="parsers-figure-state-machine"
      title="Парсер как машина состояний над потоком"
      width="full"
      controls={<Player {...p.bind} step={step} labels={labels} aria-label="Шаги парсера" />}
      caption="Попробуйте: включите «Кусками по 3 символа» и сделайте первые три шага. Парсер увидит «<th», не поймёт, тег это или текст, и придержит его в буфере. Тексты условные; в режиме токенов токены настоящие, словарь Qwen3."
    >
      <div className="grid gap-5">
        <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
          <Segmented<ScenarioId> label="Что пишет модель" value={scenario} onChange={setScenario} wrap options={SCENARIOS.map((s) => ({ value: s.id, label: s.label }))} />
          <Segmented<ChunkMode>
            label="Как поток режется на куски"
            value={mode}
            onChange={setMode}
            wrap
            options={[
              { value: 'tokens', label: 'По токенам Qwen3' },
              { value: 'chars', label: 'Кусками по 3 символа' },
            ]}
          />
        </div>

        {status === 'error' && <p className="text-sm text-ink-2">Не удалось загрузить словарь Qwen3. Обновите страницу.</p>}
        {!data && status !== 'error' && <TokenStripSkeleton count={30} />}

        {data && lanes && finalLanes && stepReserve && infoEv && (
          <>
            <div className="grid gap-4 @4xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] @4xl:items-start">
              <StateDiagram current={curChannel} previous={prevChannel} buffer={ev?.buffer ?? ''} tools={scenario === 'tool'} />
              <StepInfo ev={infoEv} isEnd={isEnd} body={body} tools={scenario === 'tool'} reserve={stepReserve} />
            </div>

            <InputStrip data={data} step={step} isEnd={isEnd} mode={mode} />

            <div className="grid gap-4 @3xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
              <Lanes lanes={lanes} final={finalLanes} />
              <StepSse lines={isEnd ? data.finish : ev ? sseOf(ev) : []} isEnd={isEnd} />
            </div>
          </>
        )}
      </div>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Пояснение шага
 * ------------------------------------------------------------------------------------------- */

/** Текст вызова, который парсер уже накопил в состоянии tool_call (без придержанного хвоста), или null вне вызова. */
function callBody(events: ParserEventEx[], step: number): string | null {
  const ev = events[step]
  if (ev.channelAfter !== 'tool_call') return null
  const seen = events
    .slice(0, step + 1)
    .map((e) => e.input)
    .join('')
  const consumed = ev.buffer ? seen.slice(0, seen.length - ev.buffer.length) : seen
  const at = consumed.lastIndexOf('<tool_call>')
  return at < 0 ? null : consumed.slice(at + '<tool_call>'.length).replace(/^\n/, '')
}

/** Как буфер и тело вызова показаны в панели шага. */
const bufferText = (buffer: string) => (buffer ? `«${buffer.replace(/\n/g, '↵')}»` : 'пусто')
const bodyText = (body: string | null) => (body !== null ? (body ? `«${body.replace(/\n/g, '↵')}»` : 'пока пусто') : 'нет: парсер не внутри <tool_call>')

interface StepReserve {
  /** Все разные пояснения прогона. */
  notes: string[]
  /** Заходит ли парсер в maybe_tag хоть раз (по токенам теги приходят целиком, и строка про maybe_tag не нужна). */
  maybe: boolean
  /** Самые длинные буфер и тело вызова, как они показаны в панели. */
  buffer: string
  body: string
}

const isMaybe = (ev: ParserEventEx) => ev.stateBefore === 'maybe_tag' || ev.stateAfter === 'maybe_tag'

function reserveOf(events: ParserEventEx[]): StepReserve {
  const longest = (xs: string[]) => xs.reduce((m, x) => (x.length > m.length ? x : m), '')
  return {
    notes: [...new Set(events.map(noteText))],
    maybe: events.some(isMaybe),
    buffer: longest(events.map((e) => bufferText(e.buffer))),
    body: longest(events.map((_, i) => bodyText(callBody(events, i)))),
  }
}

/** Текст и невидимые варианты в одной ячейке сетки (родитель с display: grid): высота равна самому высокому варианту. */
function Stack({ children, reserve }: { children: ReactNode; reserve: string[] }) {
  return (
    <>
      <span className="col-start-1 row-start-1 min-w-0">{children}</span>
      {reserve.map((r, i) => (
        <span key={i} aria-hidden className="invisible col-start-1 row-start-1 min-w-0">
          {r}
        </span>
      ))}
    </>
  )
}

/**
 * Панель «Что сделал парсер». Пояснение, буфер и тело вызова меняют длину от шага к шагу, строка про maybe_tag то есть, то нет.
 * Поэтому рядом с каждым полем лежит невидимый самый длинный вариант прогона, строка maybe_tag есть на всех шагах прогона,
 * который в maybe_tag заходит (невидимая вне него), а итог на <|im_end|> лежит в той же ячейке, что и панель шага:
 * высота панели постоянна, и плеер под фигурой не уезжает.
 */
function StepInfo({ ev, isEnd, body, tools, reserve }: { ev: ParserEventEx; isEnd: boolean; body: string | null; tools: boolean; reserve: StepReserve }) {
  const maybe = isMaybe(ev)
  return (
    <div className="grid content-start gap-3 rounded-panel bg-surface-2/60 px-4 py-3.5">
      <div className="text-sm font-medium text-ink-2">Что сделал парсер</div>
      <div className="grid">
        <div className={cn('col-start-1 row-start-1 grid min-w-0 content-start gap-3', isEnd && 'invisible')} aria-hidden={isEnd || undefined}>
          <p className="grid text-[15px] leading-relaxed text-ink">
            <Stack reserve={reserve.notes}>{noteText(ev)}</Stack>
          </p>
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[13px]">
            <dt className="text-ink-3">Состояние</dt>
            <dd className="font-mono text-ink">
              {ev.stateBefore} → {ev.stateAfter}
              {reserve.maybe && (
                <span className={cn('block font-sans text-[12.5px] leading-snug text-ink-3', !maybe && 'invisible')} aria-hidden={maybe ? undefined : true}>
                  maybe_tag: промежуточное состояние, парсер ждёт, тег это или текст
                </span>
              )}
            </dd>
            <dt className="text-ink-3">Буфер</dt>
            <dd className="grid font-mono text-ink">
              <Stack reserve={[reserve.buffer]}>{bufferText(ev.buffer)}</Stack>
            </dd>
            {tools && (
              <>
                <dt className="text-ink-3">Тело вызова</dt>
                <dd className="grid font-mono break-words text-ink">
                  <Stack reserve={[reserve.body]}>{bodyText(body)}</Stack>
                </dd>
              </>
            )}
          </dl>
        </div>
        <p className={cn('col-start-1 row-start-1 text-[15px] leading-relaxed text-ink', !isEnd && 'invisible')} aria-hidden={isEnd ? undefined : true}>
          Пришёл <span className="font-mono text-[13px]">&lt;|im_end|&gt;</span>. Это стоп-токен: в текст он не попадает, движок завершает запрос, а
          сервер отправляет финальный чанк с finish_reason и строку <span className="font-mono text-[13px]">data: [DONE]</span>.
        </p>
      </div>
    </div>
  )
}

/** Пояснение движка с заглавной буквы и точкой. */
function noteText(ev: ParserEventEx): string {
  const raw = (ev.note ?? '').trim()
  if (!raw || raw === 'без изменений') {
    return ev.channelAfter === 'tool_call'
      ? 'Кусок JSON вызова копится в теле: имя функции ещё не закрыто кавычкой или ключ "arguments" ещё не встретился, отправлять нечего.'
      : 'Ничего нового.'
  }
  const s = raw.charAt(0).toUpperCase() + raw.slice(1)
  return /[.!?»]$/.test(s) ? s : s + '.'
}

/* ---------------------------------------------------------------------------------------------
 * Лента входных кусков
 * ------------------------------------------------------------------------------------------- */

function InputStrip({ data, step, isEnd, mode }: { data: FigureData; step: number; isEnd: boolean; mode: ChunkMode }) {
  const { stream } = data
  return (
    <div className="grid gap-2">
      <div className="text-sm font-medium text-ink-2">{mode === 'tokens' ? 'Вход парсера: текст каждого токена после детокенизатора' : 'Вход парсера: куски по 3 символа'}</div>
      <div className="flex flex-wrap items-start gap-x-[3px] gap-y-1.5" role="list" aria-label="Куски потока">
        {stream.chunks.map((c, i) => {
          const future = i > step && !isEnd
          const current = i === step && !isEnd
          return (
            <span key={i} role="listitem" className={cn('inline-flex rounded-chip', current && CURRENT_RING)}>
              {stream.tokens ? (
                <TokenChip token={stream.tokens[i]} variant={future ? 'muted' : 'generated'} alt={i % 2 === 1} size="sm" />
              ) : (
                <ChunkChip text={c} future={future} alt={i % 2 === 1} />
              )}
            </span>
          )
        })}
        {stream.eos && (
          <span role="listitem" className={cn('inline-flex rounded-chip', isEnd && CURRENT_RING)}>
            <TokenChip token={stream.eos} size="sm" />
          </span>
        )}
      </div>
    </div>
  )
}

/**
 * Текущий кусок обведён цветом текста, а не акцентом: в этой фигуре зелёный уже занят каналом tool_calls,
 * и акцентное кольцо читалось бы как «это вызов инструмента».
 */
const CURRENT_RING = 'shadow-[0_0_0_2px_var(--ink)]'

function ChunkChip({ text, future, alt }: { text: string; future: boolean; alt: boolean }) {
  const shown = text.replace(/ /g, '·').replace(/\n/g, '↵')
  return (
    <span
      className={cn(
        'inline-block rounded-chip px-1 font-mono text-[12px] leading-[1.55] whitespace-pre',
        future ? 'text-ink-3 shadow-[inset_0_0_0_1px_var(--line)]' : 'text-ink',
      )}
      style={future ? undefined : { background: tint('decode', alt ? 24 : 13) }}
    >
      {shown}
    </span>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Поля ответа
 * ------------------------------------------------------------------------------------------- */

/**
 * Дорожка поля. reserve: итоговый текст поля, невидимый, в той же ячейке сетки: высота дорожки с первого шага
 * равна итоговой, и панель не растёт по ходу проигрывания.
 */
function Lane({ title, swatch, children, reasoning, reserve }: { title: string; swatch: ReactNode; children: ReactNode; reasoning?: boolean; reserve?: ReactNode }) {
  return (
    <div className="grid gap-1.5 @md:grid-cols-[172px_minmax(0,1fr)] @md:items-start">
      <div className="flex items-center gap-2 pt-1.5 font-mono text-[13px] text-ink-2">
        {swatch}
        {title}
      </div>
      <div
        className={cn(
          'grid min-h-9 rounded-control bg-surface px-3 py-1.5 text-[14px] leading-relaxed break-words whitespace-pre-wrap text-ink shadow-[inset_0_0_0_1px_var(--line)]',
          reasoning && 'border-l-2 border-dashed border-[var(--c-reasoning)] italic text-ink-2',
        )}
      >
        <div className="col-start-1 row-start-1 min-w-0">{children}</div>
        {reserve && (
          <div aria-hidden className="invisible col-start-1 row-start-1 min-w-0">
            {reserve}
          </div>
        )}
      </div>
    </div>
  )
}

function Added({ text }: { text: string }) {
  if (!text) return null
  return (
    <motion.span key={text} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.25 }} className="rounded-[3px]" style={{ background: tint('decode', 20) }}>
      {text}
    </motion.span>
  )
}

function Swatch({ color, dashed }: { color: string; dashed?: boolean }) {
  return <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-[3px]" style={dashed ? { border: `1.5px dashed ${color}` } : { background: color }} />
}

const visibleNl = (s: string) => s.replace(/\n/g, '↵\n')

function ToolText({ name, args }: { name: ReactNode; args: ReactNode }) {
  return (
    <span className="font-mono text-[13px]">
      <span className="text-ink-3">name: </span>
      {name}
      {'\n'}
      <span className="text-ink-3">arguments: </span>
      {args}
    </span>
  )
}

function Lanes({ lanes, final }: { lanes: LaneState; final: LaneState }) {
  const r = lanes.reasoning.slice(0, lanes.reasoning.length - lanes.added.reasoning.length)
  const c = lanes.content.slice(0, lanes.content.length - lanes.added.content.length)
  const a = lanes.toolArgs.slice(0, lanes.toolArgs.length - lanes.added.toolArgs.length)
  return (
    <div className="grid content-start gap-2.5">
      <div className="text-sm font-medium text-ink-2">Что уже ушло клиенту, по полям</div>
      <Lane title="reasoning_content" swatch={<Swatch color="var(--c-reasoning)" dashed />} reasoning reserve={visibleNl(final.reasoning)}>
        {visibleNl(r)}
        <Added text={visibleNl(lanes.added.reasoning)} />
        {!lanes.reasoning && <span className="text-ink-3 not-italic">пусто</span>}
      </Lane>
      <Lane title="content" swatch={<Swatch color="var(--c-decode)" />} reserve={visibleNl(final.content)}>
        {visibleNl(c)}
        <Added text={visibleNl(lanes.added.content)} />
        {!lanes.content && <span className="text-ink-3">пусто</span>}
      </Lane>
      <Lane
        title="tool_calls"
        swatch={<Swatch color="var(--c-tool)" />}
        reserve={final.toolName !== null ? <ToolText name={final.toolName} args={final.toolArgs} /> : undefined}
      >
        {lanes.toolName === null ? (
          <span className="text-ink-3">пусто</span>
        ) : (
          <ToolText
            name={lanes.added.toolName ? <Added text={lanes.toolName} /> : lanes.toolName}
            args={
              <>
                {a}
                <Added text={lanes.added.toolArgs} />
              </>
            }
          />
        )}
      </Lane>
    </div>
  )
}

/* ---------------------------------------------------------------------------------------------
 * SSE этого шага
 * ------------------------------------------------------------------------------------------- */

function StepSse({ lines, isEnd }: { lines: SseChunk[]; isEnd: boolean }) {
  return (
    <div className="grid min-w-0 content-start gap-2">
      <div className="text-sm font-medium text-ink-2">{isEnd ? 'Последние строки стрима' : 'Строки SSE на этом шаге'}</div>
      <div className="min-h-[5.5rem] rounded-control bg-surface-2 px-3 py-2">
        {lines.length === 0 ? (
          <p className="text-[13px] text-ink-3">Ничего: клиенту на этом шаге отправлять нечего.</p>
        ) : (
          <ul className="grid gap-1">
            {lines.map((l, i) => (
              <li key={i} className="font-mono text-[12px] leading-relaxed [overflow-wrap:anywhere] whitespace-pre-wrap text-ink">
                {compactLine(l.json)}
              </li>
            ))}
          </ul>
        )}
      </div>
      <p className="text-[12.5px] text-ink-3">Служебные поля id, object, created и model свёрнуты в «…»: они одинаковы во всех чанках.</p>
    </div>
  )
}
