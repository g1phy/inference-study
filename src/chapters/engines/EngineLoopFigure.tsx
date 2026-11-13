import { useId, useMemo, useState, type CSSProperties, type ReactNode } from 'react'
import { CaretRightIcon } from '@phosphor-icons/react'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { findModel, getModel } from '@/engine/data/models'
import { decodeStepMs, prefillMs } from '@/engine/perf'
import { vllmDefaultLimits } from '@/engine/sched'
import { ChapterLink, CodeBlock, Figure, Player, cn, cssVar, fmt, usePlayer } from '@/ui'
import { Diagram } from '../dynamo/diagram'
import { useFocusWithin } from '../dynamo/useFocusWithin'
import { buildPrompt, readerParts } from '../routing/prompt'
import { LOOP_EDGES, LOOP_H, LOOP_NODES, LOOP_W } from './loopLayout'
import { PARTS } from './engineData'

interface Ctx {
  tokens: number
  blocks: number
  budget: number
  prefillMs: number
  itlMs: number
  model: string
  gpu: string
}

interface QueueState {
  waiting: string[]
  running: string[]
  /** Сколько блоков KV у вашего запроса (prompt), сколько добавил decode, освобождены ли. */
  blocks: number
  decodeBlocks: number
  freed: boolean
}

interface LoopStep {
  label: string
  title: string
  nodes: string[]
  edges?: string[]
  packet?: { edgeId: string; color?: string }
  state: QueueState
  text: (c: Ctx) => ReactNode
}

const YOU = 'ваш'

function steps(c: Ctx): LoopStep[] {
  const pre: QueueState = { waiting: [], running: ['R1', 'R2', 'R3'], blocks: 0, decodeBlocks: 0, freed: false }
  const run: QueueState = { waiting: [], running: ['R1', 'R2', 'R3', YOU], blocks: c.blocks, decodeBlocks: 0, freed: false }
  return [
    {
      label: 'HTTP',
      title: 'Запрос приходит в API server',
      nodes: ['http', 'client'],
      packet: { edgeId: 'client-http' },
      state: pre,
      text: () => (
        <>
          Отдельный процесс на CPU принимает <code>POST /v1/chat/completions</code>. Разбор JSON, токенизация и стриминг живут здесь, чтобы медленный
          Python-код не задерживал цикл, который кормит GPU: шаг decode маленькой модели на H100 длится всего несколько миллисекунд.
        </>
      ),
    },
    {
      label: 'Токены',
      title: 'Шаблон и токенизация',
      nodes: ['input'],
      packet: { edgeId: 'http-input' },
      state: pre,
      text: (x) => (
        <>
          Chat template превращает сообщения в один текст, токенизатор режет его на {fmt.tokens(x.tokens)} (это ваш запрос из симулятора). Дальше уходит{' '}
          <code>EngineCoreRequest</code>: token IDs и параметры сэмплинга, текста в нём уже нет.
        </>
      ),
    },
    {
      label: 'Очередь',
      title: 'Запрос встаёт в очередь waiting',
      nodes: ['sched'],
      packet: { edgeId: 'input-sched' },
      state: { ...pre, waiting: [YOU] },
      text: () => (
        <>
          Запрос приходит в процесс EngineCore по ZMQ и ждёт в очереди. EngineCore крутится в бесконечном цикле: запланировать шаг, запустить его на GPU,
          разобрать результат. Сейчас в работе три чужих запроса, они в очереди running.
        </>
      ),
    },
    {
      label: 'Планирование',
      title: 'Scheduler допускает запрос и выделяет KV',
      nodes: ['sched', 'kvm'],
      edges: ['sched-kvm'],
      state: run,
      text: (x) => (
        <>
          Бюджет шага {fmt.tokens(x.budget)} (значение <code>vllm serve</code> для {x.gpu}). Сначала по токену получают три идущих запроса, остаток достаётся
          новому: ваш промпт ({fmt.tokens(x.tokens)}) {x.tokens + 3 <= x.budget ? 'влезает целиком' : 'не влезает и режется на куски (chunked prefill)'}. KV
          cache manager ищет начало промпта в prefix cache и выделяет {fmt.count(x.blocks, ['блок', 'блока', 'блоков'])} по 16 токенов. Запрос переходит
          в running.
        </>
      ),
    },
    {
      label: 'Forward',
      title: 'GPU-воркеры считают шаг',
      nodes: ['runner', 'kernels', 'som'],
      edges: ['sched-runner', 'runner-kernels'],
      packet: { edgeId: 'sched-runner' },
      state: run,
      text: (x) => (
        <>
          Model runner склеивает токены всех запросов шага в одну последовательность без padding: {fmt.tokens(Math.min(x.tokens, x.budget - 3))} prefill
          вашего промпта{x.tokens + 3 > x.budget ? ' (первый кусок)' : ''} и по одному токену трёх других. Слои считаются на GPU, attention читает KV по
          таблицам блоков. Пока GPU занят, CPU уже готовит следующий шаг и маски structured output, если они нужны.
        </>
      ),
    },
    {
      label: 'Sampler',
      title: 'Первый токен ответа',
      nodes: ['sampler', 'kernels'],
      packet: { edgeId: 'kernels-sampler' },
      state: run,
      text: (x) => (
        <>
          Логиты последней позиции идут в sampler: маска, temperature, top-p, выбор токена. Это первый токен: TTFT складывается из ожидания в очереди и
          prefill, а prefill вашего промпта у {x.model} на {x.gpu} занимает около {fmt.ms(x.prefillMs)} (оценка perf.ts).
        </>
      ),
    },
    {
      label: 'Обновление',
      title: 'EngineCore дописывает токен и проверяет стоп',
      nodes: ['sched', 'sampler'],
      edges: ['sampler-sched', 'core-output'],
      packet: { edgeId: 'sampler-sched' },
      state: run,
      text: () => (
        <>
          Токен возвращается в EngineCore: тот дописывает его в запрос, проверяет стоп-токены и <code>max_tokens</code> и отправляет новые token IDs в API
          server.
        </>
      ),
    },
    {
      label: 'SSE',
      title: 'API server отдаёт текст клиенту',
      nodes: ['output', 'client'],
      edges: ['core-output', 'output-client'],
      packet: { edgeId: 'output-client', color: 'decode' },
      state: run,
      text: () => (
        <>
          Детокенизатор превращает token ID в текст и придерживает неполные символы и возможное начало стоп-строки, парсеры отделяют рассуждение от ответа,
          и клиент получает SSE-чанк.
        </>
      ),
    },
    {
      label: 'Decode',
      title: 'Шаги decode вместе с другими запросами',
      nodes: ['sched', 'runner', 'kernels', 'sampler'],
      edges: ['sched-runner', 'runner-kernels', 'kernels-sampler', 'sampler-sched'],
      packet: { edgeId: 'sched-runner', color: 'decode' },
      state: { waiting: [], running: ['R1', 'R2', YOU, 'R5'], blocks: c.blocks, decodeBlocks: 1, freed: false },
      text: (x) => (
        <>
          Дальше каждый шаг даёт вашему запросу по одному токену. Batch меняется на лету: R3 закончил и ушёл, R5 только что пришёл (подробно в главе{' '}
          <ChapterLink to="batching" />). Шаг из одних decode-токенов запускается одним CUDA graph целиком и при batch из четырёх запросов длится около{' '}
          {fmt.ms(x.itlMs)}; пока в шаге есть prefill нового запроса, как сейчас у R5, граф кусочный.
          {Math.abs(x.prefillMs - x.itlMs) <= 0.2 * Math.max(x.prefillMs, x.itlMs) && (
            <>
              {' '}
              Почти столько же шёл prefill вашего промпта: на коротком промпте GPU почти всё время читает веса, а вычислители простаивают (глава{' '}
              <ChapterLink to="roofline" />
              ).
            </>
          )}{' '}
          Когда ответ заполнит блок, KV cache manager добавит новый.
        </>
      ),
    },
    {
      label: 'Конец',
      title: 'EOS: запрос завершён, блоки остаются в кэше',
      nodes: ['sched', 'kvm', 'output'],
      edges: ['core-output'],
      state: { waiting: [], running: ['R1', 'R2', 'R5'], blocks: c.blocks, decodeBlocks: 1, freed: true },
      text: () => (
        <>
          Модель выдала EOS. EngineCore завершает запрос и освобождает его блоки, но полные блоки не стираются: они остаются в prefix cache и пригодятся
          следующему ходу диалога, пока их не займут. Клиент получает финальный чанк с <code>finish_reason</code> и <code>[DONE]</code>.
        </>
      ),
    },
  ]
}

/** Фигура 1 главы 18: что происходит внутри vLLM V1, по шагам, на вашем запросе. */
export function EngineLoopFigure() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const model = findModel(settings.modelId) ?? getModel('qwen3-8b')
  const gpu = findGpu(settings.gpuId) ?? getGpu('h100-sxm')
  const tokens = useMemo(
    () => (tokenizer ? buildPrompt(readerParts(settings), tokenizer).ids.length : 0),
    [tokenizer, settings.systemText, settings.userText, settings.thinking, settings.tools],
  )
  const ctx = useMemo<Ctx>(() => {
    const n = Math.max(1, tokens)
    return {
      tokens: n,
      blocks: Math.ceil(n / 16),
      budget: vllmDefaultLimits(gpu).tokenBudget,
      prefillMs: prefillMs(model, gpu, n),
      itlMs: decodeStepMs(model, gpu, { batch: 4, avgContext: n + 200 }),
      model: model.name,
      gpu: gpu.name,
    }
  }, [tokens, model, gpu])
  const list = useMemo(() => steps(ctx), [ctx])
  const p = usePlayer(list.length, { intervalMs: 2400 })
  const step = list[p.step]
  const [picked, setPicked] = useState<{ step: number; id: string } | null>(null)
  // на узких экранах карточка блока свёрнута, чтобы текст шага стоял сразу под схемой; клик по блоку её раскрывает
  const [cardOpen, setCardOpen] = useState(false)
  const cardId = useId()
  const selected = picked && picked.step === p.step ? picked.id : step.nodes[0]
  const part = PARTS[selected] ?? PARTS.sched
  const inside = useFocusWithin(FIGURE_ID)

  // число ваших блоков пишем в подзаголовке KV cache manager, а легенда пула стоит под схемой постоянно
  const st = step.state
  const kvmSub = st.freed
    ? `в prefix cache: ${fmt.count(st.blocks, ['ваш блок', 'ваших блока', 'ваших блоков'])}`
    : st.blocks > 0
      ? `блоки по 16; ваших: ${st.blocks + st.decodeBlocks}${st.blocks > 10 ? ', видно 10' : ''}`
      : 'блоки по 16, prefix cache'
  const nodes = useMemo(() => LOOP_NODES.map((n) => (n.id === 'kvm' ? { ...n, sub: kvmSub } : n)), [kvmSub])

  return (
    <Figure
      id={FIGURE_ID}
      title="Внутри движка: путь запроса через vLLM V1"
      width="full"
      player={<Player {...p.bind} labels={list.map((s) => s.label)} aria-label="Шаги внутри движка" />}
      stickyControls
      caption="Попробуйте: пройдите по шагам и следите за очередями scheduler и блоками KV; затем нажмите на любой блок схемы, чтобы прочитать, что он делает и какие у него флаги. Число токенов ваше, чужие запросы и раскладка блоков условные."
    >
      <div className="grid min-w-0 gap-5">
        <div className="grid min-w-0 gap-2">
          <Diagram
            width={LOOP_W}
            height={LOOP_H}
            nodes={nodes}
            edges={LOOP_EDGES}
            activeNodes={new Set(step.nodes)}
            activeEdges={new Set(step.edges ?? (step.packet ? [step.packet.edgeId] : []))}
            packet={step.packet ? { ...step.packet, key: p.step } : null}
            selected={selected}
            onSelect={(id) => {
              setPicked({ step: p.step, id })
              setCardOpen(true)
            }}
            minWidth={860}
            aria-label={`Схема vLLM V1: API server, EngineCore, GPU-воркеры. Шаг ${p.step + 1}: ${step.title}`}
          >
            <QueueViz state={st} />
            <BlocksViz state={st} />
          </Diagram>
          <BlocksLegend />
        </div>
        <div className="grid min-w-0 gap-5 border-t border-line pt-5 @4xl:min-h-[27rem] @4xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="grid min-w-0 content-start gap-2">
            <div className="flex flex-wrap items-baseline gap-x-3" aria-live={inside ? 'polite' : 'off'}>
              <span className="text-[13px] text-ink-3 tabular-nums">
                шаг {p.step + 1} из {list.length}
              </span>
              <span className="text-[15px] font-semibold text-ink">{step.title}</span>
            </div>
            <p className="text-[15px] leading-relaxed text-ink">{step.text(ctx)}</p>
          </div>
          <div className="grid min-w-0 content-start gap-2">
            <button
              type="button"
              aria-expanded={cardOpen}
              aria-controls={cardId}
              onClick={() => setCardOpen((o) => !o)}
              className="inline-flex min-h-9 items-center gap-1.5 justify-self-start rounded-control px-1 text-left text-[14px] font-medium text-ink-2 hover:text-ink @4xl:hidden"
            >
              <CaretRightIcon size={16} aria-hidden className={cn('shrink-0 text-ink-3 transition-transform', cardOpen && 'rotate-90')} />
              Что это за блок: {part.title}
            </button>
            <div
              id={cardId}
              className={cn(
                'min-w-0 content-start gap-2 rounded-panel bg-surface-2 p-4 @4xl:grid @4xl:max-h-[27rem] @4xl:overflow-y-auto',
                cardOpen ? 'grid' : 'hidden',
              )}
            >
              <div>
                <div className="text-[15px] font-semibold text-ink">{part.title}</div>
                <div className="text-[13px] text-ink-3">{part.where}</div>
              </div>
              <p className="text-[14px] leading-relaxed text-ink [&_code]:whitespace-nowrap">{part.what}</p>
              <ul className="grid list-disc gap-1 pl-5 text-[13.5px] leading-snug text-ink-2 marker:text-ink-3 [&_code]:whitespace-nowrap">
                {part.facts.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
              {part.config && <CodeBlock lang={part.config.lang} title={part.config.title} code={part.config.code} wrap />}
            </div>
          </div>
        </div>
      </div>
    </Figure>
  )
}

const FIGURE_ID = 'engines-figure-loop'

/** Постоянная легенда пула KV под схемой (раньше подпись внутри схемы пропадала со второго шага). */
function BlocksLegend() {
  const sw = (style: CSSProperties, label: string) => (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden className="h-3 w-3 shrink-0 rounded-[3px]" style={style} />
      {label}
    </span>
  )
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-ink-2" aria-hidden>
      <span className="text-ink-3">Блоки KV в пуле:</span>
      {sw({ background: 'var(--surface-3)', boxShadow: 'inset 0 0 0 1px var(--line-strong)' }, 'чужие запросы')}
      {sw({ background: cssVar('prefill') }, 'ваш промпт')}
      {sw({ background: cssVar('decode') }, 'ваш ответ')}
      {sw({ border: `1.5px dashed ${cssVar('prefill')}` }, 'в prefix cache')}
      {sw({ border: '1.5px dashed var(--line-strong)' }, 'свободно')}
    </div>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Мини-визуализации внутри узлов схемы
 * ------------------------------------------------------------------------------------------- */

function Chip({ x, y, label, you }: { x: number; y: number; label: string; you: boolean }) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={36}
        height={22}
        rx={6}
        fill={you ? `color-mix(in oklab, ${cssVar('prefill')} 22%, var(--surface))` : 'var(--surface-2)'}
        stroke={you ? 'var(--accent)' : 'var(--line-strong)'}
        strokeWidth={you ? 1.6 : 1}
      />
      <text x={x + 18} y={y + 15} fontSize={11.5} textAnchor="middle" fill="var(--ink)" fontWeight={you ? 600 : 400}>
        {label}
      </text>
    </g>
  )
}

function QueueViz({ state }: { state: QueueState }) {
  const row = (label: string, y: number, items: string[]) => (
    <g>
      <text x={448} y={y + 15} fontSize={11.5} fill="var(--ink-3)">
        {label}
      </text>
      {items.length === 0 && (
        <text x={512} y={y + 15} fontSize={11.5} fill="var(--ink-3)">
          пусто
        </text>
      )}
      {items.map((it, i) => (
        <Chip key={it} x={512 + i * 42} y={y} label={it} you={it === YOU} />
      ))}
    </g>
  )
  return (
    <g aria-hidden>
      {row('waiting', 120, state.waiting)}
      {row('running', 156, state.running)}
    </g>
  )
}

/** 24 блока пула: чужие заняты, ваши синие (промпт) и оранжевый (ответ), после конца пунктир: «в кэше». Легенда под схемой. */
function BlocksViz({ state }: { state: QueueState }) {
  const others = new Set([0, 1, 2, 7, 8, 12, 13, 19, 20])
  const free = Array.from({ length: 24 }, (_, i) => i).filter((i) => !others.has(i))
  const shown = Math.min(state.blocks, 10)
  const mine = new Map<number, 'prefill' | 'decode'>()
  free.slice(0, shown).forEach((i) => mine.set(i, 'prefill'))
  if (state.decodeBlocks > 0 && free[shown] !== undefined) mine.set(free[shown], 'decode')
  return (
    <g aria-hidden>
      {Array.from({ length: 24 }, (_, i) => {
        const x = 448 + (i % 12) * 19.5
        const y = 280 + Math.floor(i / 12) * 21
        const m = mine.get(i)
        // после конца: полные блоки остаются в prefix cache (пунктир), последний неполный блок ответа просто свободен
        if (m === 'prefill' && state.freed) {
          return <rect key={i} x={x} y={y} width={16} height={16} rx={3} fill="none" stroke={cssVar('prefill')} strokeWidth={1.6} strokeDasharray="3 2" />
        }
        if (m === 'decode' && state.freed) {
          return <rect key={i} x={x} y={y} width={16} height={16} rx={3} fill="none" stroke="var(--line-strong)" strokeDasharray="2 2" />
        }
        if (m) return <rect key={i} x={x} y={y} width={16} height={16} rx={3} fill={cssVar(m)} />
        if (others.has(i)) return <rect key={i} x={x} y={y} width={16} height={16} rx={3} fill="var(--surface-3)" stroke="var(--line-strong)" />
        return <rect key={i} x={x} y={y} width={16} height={16} rx={3} fill="none" stroke="var(--line-strong)" strokeDasharray="2 2" />
      })}
    </g>
  )
}
