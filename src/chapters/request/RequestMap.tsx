import { Fragment, useId, useMemo, useState, type PointerEvent as RPointerEvent, type ReactNode } from 'react'
import { Figure, Toggle, cn, fmt, highlight, useFigureLive, type CodeToken } from '@/ui'
import { requestBlocks, type FieldBlock, type FieldId, type RequestOptions } from './requestModel'

/* ===========================================================================
 * Стадии мини-схемы и поля запроса: кто что читает.
 * Названия стадий как в Dynamo (notes-dynamo §1.3); в vLLM frontend, preprocessor и postprocessor
 * живут в одном процессе API-сервера (notes-engines §1.0).
 * ========================================================================= */

export type StageKey = 'client' | 'frontend' | 'preprocess' | 'router' | 'prefill' | 'decode' | 'postprocess' | 'sse'

interface StageInfo {
  name: string
  sub: string
  where?: 'CPU' | 'GPU'
  desc: string
}

const STAGES: Record<StageKey, StageInfo> = {
  client: {
    name: 'Клиент',
    sub: 'OpenAI SDK, браузер, агент',
    desc: 'Собирает JSON и отправляет POST /v1/chat/completions. Обычно это OpenAI SDK, которому указали адрес вашего сервера.',
  },
  frontend: {
    name: 'Frontend',
    sub: 'HTTP, проверка запроса',
    where: 'CPU',
    desc: 'Принимает HTTP, разбирает и проверяет JSON, находит модель. В Dynamo это отдельный процесс на Rust (порт 8000), в vLLM процесс API-сервера.',
  },
  preprocess: {
    name: 'Preprocessor',
    sub: 'шаблон и токенизация',
    where: 'CPU',
    desc: 'Склеивает messages и tools шаблоном чата в одну строку и превращает её в token IDs. Здесь же проверка длины: промпт и max_tokens должны поместиться в контекстное окно.',
  },
  router: {
    name: 'Router',
    sub: 'выбор воркера',
    where: 'CPU',
    desc: 'Решает, на какой воркер отправить запрос. В Dynamo по умолчанию по кругу (round-robin), а в режиме kv туда, где уже лежит KV cache такого же начала промпта, с учётом нагрузки.',
  },
  prefill: {
    name: 'prefill',
    sub: 'весь промпт за проход',
    where: 'GPU',
    desc: 'Весь промпт за один проход по модели: K и V каждого токена ложатся в KV cache, в конце появляется первый токен ответа. От длины промпта зависит время до первого токена.',
  },
  decode: {
    name: 'decode',
    sub: 'по токену за шаг',
    where: 'GPU',
    desc: 'По одному токену за шаг: логиты, маска structured output, сэмплер, проверка стоп-токенов и max_tokens.',
  },
  postprocess: {
    name: 'Postprocessor',
    sub: 'токены в текст, парсеры',
    where: 'CPU',
    desc: 'Превращает токены обратно в текст, ищет стоп-строки, отделяет рассуждение и вызовы инструментов от ответа.',
  },
  sse: {
    name: 'Ответ',
    sub: 'SSE или один JSON',
    desc: 'Поток чанков chat.completion.chunk или один JSON chat.completion в самом конце.',
  },
}

const ORDER: StageKey[] = ['client', 'frontend', 'preprocess', 'router', 'prefill', 'decode', 'postprocess', 'sse']

interface Consumer {
  stage: StageKey
  role: string
  /** false: стадия зависит от поля косвенно (например, prefill от длины messages). */
  primary: boolean
}

/** Числа для объяснений (считаются в главе по настоящему токенизатору и perf.ts). */
export interface MapContext {
  modelName: string
  promptTokens: number | null
  toolTokens: number | null
  /** Включён ли переключатель tools (тогда часть промпта занимает описание функций). */
  toolsOn: boolean
  kvPerToken: number
  maxContext: number
  maxTokens: number
}

interface FieldInfo {
  consumers: Consumer[]
  explain: (c: MapContext) => ReactNode
}

const num = (n: number | null) => (n === null ? '…' : fmt.int(n))

const FIELDS: Record<FieldId, FieldInfo> = {
  model: {
    consumers: [
      { stage: 'frontend', role: 'находит модель среди зарегистрированных', primary: true },
      { stage: 'router', role: 'выбирает воркер из пула этой модели', primary: true },
    ],
    explain: () => (
      <>
        Один сервер может обслуживать несколько моделей. Frontend проверяет, что модель зарегистрирована (в Dynamo воркеры сами
        публикуют её карточку при старте), а роутер выбирает воркер только среди тех, кто её держит. Опечатка в имени вернёт
        ошибку ещё до GPU.
      </>
    ),
  },
  messages: {
    consumers: [
      { stage: 'preprocess', role: 'шаблон чата и токенизация', primary: true },
      { stage: 'router', role: 'KV-aware: хэши блоков промпта', primary: false },
      { stage: 'prefill', role: 'длина промпта: работа и KV cache', primary: false },
    ],
    explain: (c) => (
      <>
        Реплики с ролями system, user, assistant, tool. Preprocessor склеивает их шаблоном чата в одну строку и режет на токены:
        у вас вышло <b className="tabular-nums">{num(c.promptTokens)}</b>{' '}
        {c.promptTokens === null ? 'токенов' : fmt.plural(c.promptTokens, ['токен', 'токена', 'токенов'])}
        {c.toolsOn && c.toolTokens !== null && (
          <>
            , из них <b className="tabular-nums">{fmt.int(c.toolTokens)}</b> приходится на описание tools
          </>
        )}
        . Все они пройдут через prefill и лягут в KV cache, а KV-aware роутер по хэшам их блоков ищет воркер, у которого такое
        начало промпта уже лежит в кэше. Каждый новый ход диалога присылает всю историю заново, поэтому длинный чат дорожает с
        каждой репликой.
      </>
    ),
  },
  tools: {
    consumers: [
      { stage: 'preprocess', role: 'описание функций в системный промпт', primary: true },
      { stage: 'router', role: 'KV-aware: хэши блоков промпта', primary: false },
      { stage: 'prefill', role: 'лишние токены в каждом запросе', primary: false },
      { stage: 'postprocess', role: 'tool parser собирает tool_calls', primary: true },
    ],
    explain: (c) => (
      <>
        Описания функций. Шаблон вписывает их в системный промпт, поэтому один get_weather добавляет{' '}
        <b className="tabular-nums">{num(c.toolTokens)}</b> токенов к каждому запросу. Модель отвечает текстом вида{' '}
        <code>{'<tool_call>{…}</tool_call>'}</code>, а tool-call parser на выходе превращает его в поле <code>tool_calls</code>.
        Саму функцию сервер не вызывает: это делает клиент и присылает результат сообщением с ролью tool.
      </>
    ),
  },
  chat_template_kwargs: {
    consumers: [{ stage: 'preprocess', role: 'аргумент для шаблона чата', primary: true }],
    explain: () => (
      <>
        Аргументы для шаблона, а не для модели. <code>enable_thinking: false</code> заставляет шаблон Qwen3 дописать в конец
        промпта пустой блок <code>{'<think></think>'}</code>, и модель отвечает сразу, без рассуждения. Поле появилось в
        запросе, потому что в настройках симулятора рассуждение выключено.
      </>
    ),
  },
  temperature: {
    consumers: [{ stage: 'decode', role: 'сэмплер: острее или ровнее', primary: true }],
    explain: () => (
      <>
        Температура делит логиты перед softmax: меньше 1 делает распределение острее, больше 1 ровнее, 0 означает всегда самый
        вероятный токен. Frontend только кладёт число в запрос к воркеру, а работает оно на GPU, на каждом шаге decode.
      </>
    ),
  },
  top_p: {
    consumers: [{ stage: 'decode', role: 'сэмплер: отрезает хвост', primary: true }],
    explain: () => (
      <>
        Nucleus sampling: сэмплер оставляет самые вероятные токены, пока их суммарная вероятность не дойдёт до top_p, остальные
        выбрасывает. Применяется вместе с temperature на каждом шаге decode.
      </>
    ),
  },
  max_tokens: {
    consumers: [
      { stage: 'preprocess', role: 'проверка: влезет ли в контекст', primary: true },
      { stage: 'decode', role: 'стоп по длине, потолок KV', primary: true },
    ],
    explain: (c) => {
      const total = c.promptTokens === null ? null : c.promptTokens + c.maxTokens
      return (
        <>
          Потолок длины ответа. До GPU препроцессор проверяет, что промпт и max_tokens вместе помещаются в контекстное окно (
          {fmt.int(c.maxContext)} токенов у {c.modelName}). Движок на каждом шаге сверяет счётчик и останавливает генерацию с{' '}
          <code>finish_reason: "length"</code>. Это и потолок памяти: KV запроса может вырасти до {num(total)} токенов ×{' '}
          {fmt.bytes(c.kvPerToken, { mode: 'binary' })} = <b>{total === null ? '…' : fmt.bytes(total * c.kvPerToken, { mode: 'binary' })}</b>.
        </>
      )
    },
  },
  stop: {
    consumers: [{ stage: 'postprocess', role: 'ищет строку в готовом тексте', primary: true }],
    explain: () => (
      <>
        Стоп-строки ищутся не в токенах, а в тексте после детокенизации: в vLLM это делает API-сервер, в Dynamo postprocessor.
        Строка может разрезаться между токенами, поэтому похожий на её начало хвост сервер придерживает. Часто stop вообще не
        нужен: конец реплики модель отмечает токеном <code>{'<|im_end|>'}</code>, сервер сам добавляет его в стоп-токены, и их
        проверяет движок.
      </>
    ),
  },
  response_format: {
    consumers: [{ stage: 'decode', role: 'маска грамматики перед сэмплером', primary: true }],
    explain: () => (
      <>
        Structured output. Схема компилируется в автомат, и на каждом шаге decode перед сэмплингом запрещённые токены получают
        −∞ в логитах. Модель выбирает только из того, что не ломает JSON: синтаксис гарантирован, смысл нет.
      </>
    ),
  },
  stream: {
    consumers: [
      { stage: 'postprocess', role: 'режет ответ на кусочки', primary: false },
      { stage: 'sse', role: 'поток чанков или один JSON', primary: true },
    ],
    explain: () => (
      <>
        true: ответ уходит потоком Server-Sent Events, по чанку на каждый новый кусок текста, а{' '}
        <code>stream_options.include_usage</code> добавляет в конце чанк с usage. false: сервер копит ответ и присылает один
        JSON, когда генерация закончится. Для GPU разницы нет, меняется только момент, когда пользователь увидит первые слова.
      </>
    ),
  },
}

/** Какие поля читает стадия (для подсветки в обратную сторону). */
function fieldsOfStage(stage: StageKey, present: FieldId[]): Array<{ id: FieldId; primary: boolean }> {
  if (stage === 'client') return present.map((id) => ({ id, primary: true }))
  const out: Array<{ id: FieldId; primary: boolean }> = []
  for (const id of present) {
    const c = FIELDS[id].consumers.find((x) => x.stage === stage)
    if (c) out.push({ id, primary: c.primary })
  }
  return out
}

/* ===========================================================================
 * Подсветка JSON (те же цвета, что у CodeBlock).
 * ========================================================================= */

const TOK_CLASS: Partial<Record<CodeToken['t'], string>> = {
  key: 'text-ink',
  str: 'text-[var(--code-string)]',
  num: 'text-[var(--code-number)]',
  kw: 'text-[var(--code-keyword)]',
  // пунктуация чуть темнее, чем в CodeBlock: на подсвеченном поле (accent-soft) иначе ниже AA
  punct: 'text-ink-2',
}

function JsonLine({ line }: { line: string }) {
  const indent = line.length - line.trimStart().length
  const toks = useMemo(() => highlight(line.trimStart(), 'json')[0] ?? [], [line])
  // висячий отступ: перенесённое продолжение строки встаёт под её началом
  return (
    <span className="block" style={{ paddingLeft: `${indent + 2}ch`, textIndent: '-2ch' }}>
      {toks.map((t, i) => (
        <span key={i} className={TOK_CLASS[t.t]}>
          {t.v}
        </span>
      ))}
    </span>
  )
}

/* ===========================================================================
 * Фигура
 * ========================================================================= */

type Target = { kind: 'field'; id: FieldId } | { kind: 'stage'; id: StageKey }

export interface RequestMapProps {
  opts: RequestOptions
  ctx: MapContext
  onToggle: (patch: Partial<Pick<RequestOptions, 'tools' | 'json' | 'stream'>>) => void
}

export function RequestMap({ opts, ctx, onToggle }: RequestMapProps) {
  const blocks = useMemo(() => requestBlocks(opts), [opts])
  const present = useMemo(() => blocks.map((b) => b.id), [blocks])
  const [pinned, setPinned] = useState<Target>({ kind: 'field', id: 'messages' })
  const [hover, setHover] = useState<Target | null>(null)
  const explId = useId()

  // выключили поле, которое было выбрано: возвращаемся к messages
  const pinnedOk = pinned.kind === 'stage' || present.includes(pinned.id)
  const active: Target = hover && (hover.kind === 'stage' || present.includes(hover.id)) ? hover : pinnedOk ? pinned : { kind: 'field', id: 'messages' }

  // что подсвечено
  const stageMark = new Map<StageKey, 'primary' | 'secondary'>()
  const fieldMark = new Map<FieldId, 'primary' | 'secondary'>()
  if (active.kind === 'field') {
    // клиент поля заполняет, а не читает: его не подсвечиваем (пунктир в легенде значит «читает косвенно»)
    fieldMark.set(active.id, 'primary')
    for (const c of FIELDS[active.id].consumers) stageMark.set(c.stage, c.primary ? 'primary' : 'secondary')
  } else {
    stageMark.set(active.id, 'primary')
    for (const f of fieldsOfStage(active.id, present)) fieldMark.set(f.id, f.primary ? 'primary' : 'secondary')
  }

  const bindTarget = (t: Target) => ({
    onPointerEnter: (e: RPointerEvent) => e.pointerType === 'mouse' && setHover(t),
    onPointerLeave: (e: RPointerEvent) => e.pointerType === 'mouse' && setHover(null),
    onFocus: () => setPinned(t),
    onClick: () => setPinned(t),
    'aria-describedby': explId,
  })
  const isPinned = (t: Target) => pinnedOk && pinned.kind === t.kind && pinned.id === t.id

  return (
    <Figure
      id="request-figure-map"
      title="Запрос как карта системы"
      controls={
        <>
          {/* включённое поле сразу выбираем: объяснение и цена в токенах видны без поиска нового блока */}
          <Toggle
            label="Инструменты (tools)"
            checked={opts.tools}
            onChange={(v) => {
              onToggle({ tools: v })
              if (v) setPinned({ kind: 'field', id: 'tools' })
            }}
          />
          <Toggle
            label="JSON по схеме (response_format)"
            checked={opts.json}
            onChange={(v) => {
              onToggle({ json: v })
              if (v) setPinned({ kind: 'field', id: 'response_format' })
            }}
          />
          <Toggle
            label="Стриминг (stream)"
            checked={opts.stream}
            onChange={(v) => {
              onToggle({ stream: v })
              setPinned({ kind: 'field', id: 'stream' })
            }}
          />
        </>
      }
      caption="Попробуйте: наведите на поле запроса или нажмите на него, и схема покажет стадию, которая его читает. Наведите на стадию, и подсветятся её поля. Включите tools и посмотрите, сколько токенов стоит одно описание функции."
    >
      {/* от 720px: слева JSON и под ним объяснение, справа путь запроса на две строки сетки; уже: всё в столбик */}
      <div className="grid items-start gap-4 @min-[720px]:grid-cols-[minmax(0,1fr)_236px] @min-[720px]:grid-rows-[auto_1fr]">
        <div className="min-w-0 rounded-panel bg-surface-2 px-1.5 py-3 font-mono text-[12.5px] leading-[1.6] text-ink shadow-[inset_0_0_0_1px_var(--line)]">
          <div className="px-2 pb-2 text-[12px] text-ink-3">POST /v1/chat/completions</div>
          <div className="px-2 text-ink-2">{'{'}</div>
          {blocks.map((b) => (
            <FieldButton
              key={b.id}
              block={b}
              mark={fieldMark.get(b.id)}
              pressed={isPinned({ kind: 'field', id: b.id })}
              bind={bindTarget({ kind: 'field', id: b.id })}
            />
          ))}
          <div className="px-2 text-ink-2">{'}'}</div>
        </div>

        <Pipeline
          className="@min-[720px]:col-start-2 @min-[720px]:row-span-2 @min-[720px]:row-start-1"
          marks={stageMark}
          bindTarget={bindTarget}
          isPinned={isPinned}
          active={active}
        />

        <Explanation
          className="@min-[720px]:col-start-1 @min-[720px]:row-start-2"
          id={explId}
          active={active}
          ctx={ctx}
          present={present}
        />
      </div>
    </Figure>
  )
}

function FieldButton({
  block,
  mark,
  pressed,
  bind,
}: {
  block: FieldBlock
  mark?: 'primary' | 'secondary'
  pressed: boolean
  bind: Record<string, unknown>
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      aria-label={`Поле ${block.id}`}
      {...bind}
      className={cn(
        // на узком экране (тач) строка поля не ниже 24px: соседние однострочные поля легко перепутать пальцем
        'my-px block w-full rounded-[6px] border-l-2 px-2 py-[3px] text-left transition-colors duration-150 focus-visible:outline-offset-0 @min-[720px]:py-0',
        mark === 'primary'
          ? 'border-accent bg-accent-soft'
          : mark === 'secondary'
            ? 'border-dashed border-accent bg-surface-3/50'
            : 'border-transparent hover:bg-surface-3/70',
      )}
    >
      {block.lines.map((l, i) => (
        <JsonLine key={i} line={l} />
      ))}
    </button>
  )
}

/** Номера стадий для узкой раскладки (там вместо стрелок порядок задают номера). */
const STAGE_NUM: Record<StageKey, string> = {
  client: '1',
  frontend: '2',
  preprocess: '3',
  router: '4',
  prefill: '5a',
  decode: '5b',
  postprocess: '6',
  sse: '7',
}

function StepNum({ n }: { n: string }) {
  return (
    <span
      aria-hidden
      className="inline-grid h-4 min-w-4 shrink-0 place-items-center rounded-[4px] bg-surface-3 px-0.5 font-mono text-[10.5px] leading-none text-ink-2 @min-[720px]:hidden"
    >
      {n}
    </span>
  )
}

function Pipeline({
  className,
  marks,
  bindTarget,
  isPinned,
  active,
}: {
  className?: string
  marks: Map<StageKey, 'primary' | 'secondary'>
  bindTarget: (t: Target) => Record<string, unknown>
  isPinned: (t: Target) => boolean
  active: Target
}) {
  const roleOf = (s: StageKey): string | undefined => {
    if (active.kind !== 'field') return undefined
    return FIELDS[active.id].consumers.find((c) => c.stage === s)?.role
  }
  const node = (s: StageKey) => {
    const info = STAGES[s]
    const mark = marks.get(s)
    const t: Target = { kind: 'stage', id: s }
    const role = roleOf(s)
    const isData = s === 'prefill' || s === 'decode'
    // неподсвеченные стадии приглушены цветом текста, а не прозрачностью: подписи остаются выше AA
    return (
      <button
        type="button"
        aria-pressed={isPinned(t)}
        {...bindTarget(t)}
        className={cn(
          'grid w-full min-w-0 gap-0.5 rounded-control border px-2.5 py-1.5 text-left transition-[background-color,border-color] duration-150',
          mark === 'primary'
            ? 'border-accent bg-accent-soft'
            : mark === 'secondary'
              ? 'border-dashed border-accent bg-surface'
              : 'border-line bg-surface hover:bg-surface-3/60',
        )}
      >
        <span className="flex items-center justify-between gap-2">
          <span className={cn('flex min-w-0 items-center gap-1.5 text-[13.5px] leading-5 font-medium', mark ? 'text-ink' : 'text-ink-2')}>
            <StepNum n={STAGE_NUM[s]} />
            {isData && (
              <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: `var(--c-${s})` }} />
            )}
            <span className="truncate">{info.name}</span>
          </span>
          {info.where && !isData && <span className={cn('shrink-0 font-mono text-[11px]', mark ? 'text-ink-2' : 'text-ink-3')}>{info.where}</span>}
        </span>
        <span className={cn('hidden text-[12px] leading-4 @min-[720px]:block', role ? 'text-ink' : mark ? 'text-ink-2' : 'text-ink-3')}>{role ?? info.sub}</span>
      </button>
    )
  }

  // от 720px стадии идут столбиком, между ними тонкие соединители; уже: сетка в две колонки, порядок задают номера
  const Connector = () => (
    <li aria-hidden className="hidden items-center justify-center @min-[720px]:flex">
      <span className="h-2.5 w-px bg-line-strong" />
    </li>
  )

  const items: Array<StageKey | 'worker'> = ['client', 'frontend', 'preprocess', 'router', 'worker', 'postprocess', 'sse']
  return (
    <div className={cn('min-w-0', className)}>
      <div className="mb-2 text-xs text-ink-3">Путь запроса (названия как в Dynamo)</div>
      <ol className="grid grid-cols-2 gap-1.5 @min-[720px]:flex @min-[720px]:flex-col @min-[720px]:gap-0">
        {items.map((it, i) => (
          <Fragment key={it}>
            {i > 0 && <Connector />}
            {it === 'worker' ? (
              <li className="col-span-2 min-w-0 rounded-panel border border-line bg-surface-2 p-1.5 @min-[720px]:w-full">
                <div className="mb-1 flex items-center justify-between gap-2 px-1 text-[12px] text-ink-2">
                  <span className="flex items-center gap-1.5">
                    <StepNum n="5" />
                    Worker (движок)
                  </span>
                  <span className="font-mono text-[11px] text-ink-2">GPU</span>
                </div>
                <div className="grid grid-cols-2 gap-1 @min-[720px]:grid-cols-1">
                  {node('prefill')}
                  {node('decode')}
                </div>
              </li>
            ) : (
              <li className="min-w-0 @min-[720px]:w-full">{node(it)}</li>
            )}
          </Fragment>
        ))}
      </ol>
    </div>
  )
}

function Explanation({
  className,
  id,
  active,
  ctx,
  present,
}: {
  className?: string
  id: string
  active: Target
  ctx: MapContext
  present: FieldId[]
}) {
  const box = cn('grid min-h-[10.5rem] min-w-0 content-start gap-2 rounded-panel border border-line px-4 py-3', className)
  // озвучка только пока читатель в этой фигуре: текст пересчитывается и при наборе запроса в симуляторе
  const live = useFigureLive()
  if (active.kind === 'field') {
    const f = FIELDS[active.id]
    return (
      <div id={id} aria-live={live} className={box}>
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[15px]">
          <code className="rounded-[4px] bg-surface-2 px-1.5 font-mono text-[13.5px] text-ink">{active.id}</code>
          <span className="text-ink-2">читают:</span>
        </div>
        <ul className="grid gap-1 text-[14px] leading-5">
          {f.consumers.map((c) => (
            <li key={c.stage} className="flex items-baseline gap-2">
              <span
                aria-hidden
                className={cn('relative top-[1px] h-2.5 w-2.5 shrink-0 rounded-full', c.primary ? 'bg-accent' : 'border border-dashed border-accent')}
              />
              <span>
                <span className="font-medium text-ink">{STAGE_TITLE[c.stage]}</span>
                <span className="text-ink-2">: {c.role}</span>
                {!c.primary && <span className="text-ink-3"> (косвенно)</span>}
              </span>
            </li>
          ))}
        </ul>
        <p className="text-[15px] leading-relaxed text-ink">{f.explain(ctx)}</p>
      </div>
    )
  }
  const s = STAGES[active.id]
  const fields = fieldsOfStage(active.id, present)
  const fieldsLabel = !fields.length ? 'Полей запроса не читает.' : active.id === 'client' ? 'Заполняет поля:' : 'Читает поля:'
  return (
    <div id={id} aria-live={live} className={box}>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[15px]">
        <span className="font-medium text-ink">{STAGE_TITLE[active.id]}</span>
        {s.where && <span className="font-mono text-[12px] text-ink-3">{s.where}</span>}
      </div>
      <p className="text-[15px] leading-relaxed text-ink">{s.desc}</p>
      <div className="flex flex-wrap items-center gap-1.5 text-[14px] text-ink-2">
        <span>{fieldsLabel}</span>
        {fields.map((f) => (
          <code
            key={f.id}
            className={cn(
              'rounded-[4px] px-1.5 font-mono text-[12.5px] text-ink',
              f.primary ? 'bg-accent-soft' : 'border border-dashed border-accent',
            )}
          >
            {f.id}
          </code>
        ))}
      </div>
    </div>
  )
}

const STAGE_TITLE: Record<StageKey, string> = {
  client: 'Клиент',
  frontend: 'Frontend',
  preprocess: 'Preprocessor',
  router: 'Router',
  prefill: 'Worker: prefill',
  decode: 'Worker: decode',
  postprocess: 'Postprocessor',
  sse: 'Ответ клиенту',
}

export { ORDER as STAGE_ORDER }
