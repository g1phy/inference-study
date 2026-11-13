import { Fragment, useMemo, useState, type ReactNode } from 'react'
import { ArrowDownIcon, ArrowRightIcon } from '@phosphor-icons/react'
import { CodeBlock, Figure, Player, Segmented, Term, TokenStrip, TokenStripSkeleton, cn, fmt, usePlayer } from '@/ui'
import { blockHashes } from '@/engine/kv'
import { renderTemplate } from '@/engine/templates'
import { QWEN3_STOP_TOKEN_IDS } from '@/engine/tokenizer'
import type { Token, Tokenizer } from '@/engine/types'
import { STOP_EXAMPLE, WEATHER_TOOL, requestBlocks, requestMessages, requestText, type RequestOptions } from '../request/requestModel'
import { PromptLegend, PromptView, Special } from './PromptView'

type System = 'dynamo' | 'vllm' | 'sglang'
type StepKey = 'http' | 'template' | 'tokenize' | 'hash' | 'worker'

const STEP_KEYS: StepKey[] = ['http', 'template', 'tokenize', 'hash', 'worker']

interface Box {
  key: StepKey
  title: string
  sub: string
}
interface Group {
  label: string
  boxes: Box[]
}

/** Процессы и компоненты трёх систем (notes-dynamo §1.3, §1.5; notes-engines §1.0, §1.3; notes-tokens, карта). */
const LAYOUT: Record<System, Group[]> = {
  dynamo: [
    {
      label: 'Frontend Dynamo (Rust)',
      boxes: [
        { key: 'http', title: 'HTTP', sub: 'OpenAI API, порт 8000' },
        { key: 'template', title: 'Шаблон', sub: 'minijinja' },
        { key: 'tokenize', title: 'Токенизатор', sub: 'HF tokenizers' },
        { key: 'hash', title: 'KV router', sub: 'хэши блоков' },
      ],
    },
    { label: 'Воркер: vLLM, SGLang или TRT-LLM', boxes: [{ key: 'worker', title: 'Движок', sub: 'токены на входе' }] },
  ],
  vllm: [
    {
      label: 'API-сервер vLLM (Python)',
      boxes: [
        { key: 'http', title: 'HTTP', sub: 'FastAPI, Uvicorn' },
        { key: 'template', title: 'Шаблон', sub: 'transformers, Jinja2' },
        { key: 'tokenize', title: 'Токенизатор', sub: 'HF tokenizers' },
      ],
    },
    { label: 'EngineCore', boxes: [{ key: 'hash', title: 'Планировщик', sub: 'хэши prefix cache' }] },
    { label: 'GPU-воркеры', boxes: [{ key: 'worker', title: 'Модель', sub: 'forward на GPU' }] },
  ],
  sglang: [
    {
      label: 'Главный процесс SGLang',
      boxes: [
        { key: 'http', title: 'HTTP', sub: 'OpenAI API' },
        { key: 'template', title: 'Шаблон', sub: 'TokenizerManager' },
        { key: 'tokenize', title: 'Токенизатор', sub: 'TokenizerManager' },
      ],
    },
    {
      label: 'Scheduler',
      boxes: [
        { key: 'hash', title: 'Radix cache', sub: 'дерево по токенам' },
        { key: 'worker', title: 'Модель', sub: 'forward на GPU' },
      ],
    },
  ],
}

/** У vLLM и SGLang поиск в кэше идёт уже внутри движка, после передачи запроса: шаги 4 и 5 подписаны так же. */
const STEP_LABEL: Record<System, string[]> = {
  dynamo: ['Запрос пришёл', 'Шаблон чата', 'Токенизация', 'Блоки и хэши для роутера', 'Запрос к воркеру'],
  vllm: ['Запрос пришёл', 'Шаблон чата', 'Токенизация', 'В движке: поиск в prefix cache', 'Batch и forward на GPU'],
  sglang: ['Запрос пришёл', 'Шаблон чата', 'Токенизация', 'В планировщике: поиск в radix cache', 'Batch и forward на GPU'],
}

interface StepText {
  where: string
  text: ReactNode
}

function stepText(sys: System, step: StepKey, n: number, defaultMax: number): StepText {
  const T: Record<System, Record<StepKey, StepText>> = {
    dynamo: {
      http: {
        where: 'Frontend Dynamo: процесс на Rust, python -m dynamo.frontend, порт 8000',
        text: 'Frontend проверяет JSON и находит модель. Веса ему не нужны: он берёт у модели config.json, tokenizer.json и tokenizer_config.json, в котором лежит шаблон чата.',
      },
      template: {
        where: 'Preprocessor во frontend, crate dynamo-renderer: Jinja-шаблон из tokenizer_config.json рендерит minijinja (Jinja на Rust)',
        text: 'Реплики склеиваются в одну строку со служебными токенами. Для моделей без Jinja-шаблона в репозитории Dynamo держит форматтеры на Rust. Свой шаблон задаётся воркеру флагом --custom-jinja-template.',
      },
      tokenize: {
        where: 'Preprocessor, crate dynamo-tokenizers: по умолчанию HF tokenizers на Rust (флаг --tokenizer)',
        text: (
          <>
            Строка становится списком из {fmt.tokensGen(n)}. Это работа CPU, и она стоит на пути к первому токену: пока она не
            закончилась, GPU не начнёт prefill. У Dynamo для неё есть свои метрики, dynamo_frontend_template_seconds и
            dynamo_frontend_tokenize_seconds.
          </>
        ),
      },
      hash: {
        where: 'KV router во frontend',
        text: 'В режиме KV-aware (--router-mode kv; по умолчанию Dynamo раздаёт запросы по кругу) роутер режет token IDs на блоки по 16 и считает хэш каждого блока цепочкой: хэш зависит от всех токенов до него. Воркеры сообщают в KV events, какие блоки у них лежат, и роутер видит, у кого уже есть такое же начало промпта. Хвост неполного блока не хэшируется. Поэтому Dynamo и токенизирует до роутера: без token IDs сравнивать нечего.',
      },
      worker: {
        where: 'Request plane (по умолчанию TCP и msgpack) до воркера',
        text: (
          <>
            Воркер получает готовые token IDs: движок работает в режиме «токены на входе, токены на выходе», свой шаблон и
            токенизатор не запускает. Ещё препроцессор собрал правила генерации. EOS-токены модели (&lt;|im_end|&gt; = 151645 и
            &lt;|endoftext|&gt; = 151643) стали скрытыми стоп-токенами, а max_tokens, которого в запросе нет, равен остатку
            контекстного окна: {fmt.int(defaultMax)}. Параметры сэмплинга, которых в запросе нет (например, top_k), препроцессор
            оставил пустыми: их выберет воркер. vLLM по умолчанию берёт их из <Term t="generation_config.json" /> модели (у Qwen3
            top_k = 20), у других движков умолчания бывают свои.
          </>
        ),
      },
    },
    vllm: {
      http: {
        where: 'API-сервер vLLM: FastAPI и Uvicorn, OpenAIServingChat',
        text: 'Этот же процесс дальше сделает шаблон и токенизацию, а на обратном пути детокенизацию, парсеры и SSE. По умолчанию API-сервер один на весь движок, при data parallel их больше (--api-server-count).',
      },
      template: {
        where: 'API-сервер: apply_chat_template из transformers (Jinja2 на Python)',
        text: 'Шаблон берётся из tokenizer_config.json модели. Подменить его можно флагом --chat-template, аргументы шаблона (enable_thinking) приходят в запросе в chat_template_kwargs.',
      },
      tokenize: {
        where: 'API-сервер: быстрый HF tokenizer, та же библиотека tokenizers на Rust',
        text: <>Строка становится списком из {fmt.tokensGen(n)}. Готовый EngineCoreRequest уходит в EngineCore по ZMQ.</>,
      },
      hash: {
        where: 'EngineCore, менеджер KV cache',
        text: 'EngineCoreRequest с token IDs пришёл из API-сервера по ZMQ. Менеджер KV cache считает цепочку хэшей блоков по 16 токенов и ищет их среди своих закэшированных блоков: совпавшие блоки prefill пропустит. Роутера между репликами у одиночного vLLM нет.',
      },
      worker: {
        where: 'EngineCore: планировщик, затем GPU-воркеры',
        text: (
          <>
            Планировщик решает, в какой шаг запрос попадёт в batch, а GPU-воркеры считают модель. В этом запросе нет max_tokens,
            поэтому лимит равен остатку контекстного окна: {fmt.int(defaultMax)}.
          </>
        ),
      },
    },
    sglang: {
      http: {
        where: 'HTTP-сервер в главном процессе SGLang',
        text: 'HTTP-сервер и TokenizerManager живут в одном главном процессе. Планировщик и детокенизатор работают в отдельных процессах и общаются через ZMQ.',
      },
      template: {
        where: 'TokenizerManager: HF-шаблон модели',
        text: 'Шаблон берётся из репозитория модели, свой задаётся флагом --chat-template.',
      },
      tokenize: {
        where: 'TokenizerManager: HF tokenizers',
        text: (
          <>
            Строка становится списком из {fmt.tokensGen(n)}. Если токенизация не успевает, процессов для неё можно добавить флагом
            --tokenizer-worker-num.
          </>
        ),
      },
      hash: {
        where: 'Scheduler, radix cache',
        text: 'TokenizedGenerateReqInput с input_ids пришёл из TokenizerManager по ZMQ. SGLang хранит KV cache деревом по token IDs и ищет совпадение с точностью до токена, хэши блоков ему не нужны. Роутер SGLang перед репликами сравнивает префиксы по символам текста, поэтому токенизировать до роутера не обязательно.',
      },
      worker: {
        where: 'Scheduler: batch и forward в том же процессе',
        text: 'Планировщик ставит запрос в batch, модель считает воркер внутри процесса планировщика.',
      },
    },
  }
  return T[sys][step]
}

export interface PreprocessorFlowProps {
  tokenizer: Tokenizer | null
  /** Запрос читателя как в главе «Запрос» (из настроек симулятора); max_tokens здесь нарочно не задан. */
  opts: RequestOptions
  maxContext: number
}

export function PreprocessorFlow({ tokenizer, opts, maxContext }: PreprocessorFlowProps) {
  const [sys, setSys] = useState<System>('dynamo')
  const p = usePlayer(STEP_KEYS.length, { intervalMs: 3200 })
  const step = STEP_KEYS[p.step]

  const { systemText, userText, tools, thinking } = opts
  // шаблон с теми же tools и enable_thinking, что в симуляторе и в главе «Запрос»: число токенов совпадает
  const rendered = useMemo(
    () =>
      renderTemplate({
        family: 'qwen3',
        messages: requestMessages({ systemText, userText }),
        tools: tools ? [WEATHER_TOOL] : undefined,
        enableThinking: thinking,
      }),
    [systemText, userText, tools, thinking],
  )
  const tokens = useMemo(() => (tokenizer ? tokenizer.encode(rendered.text) : null), [tokenizer, rendered.text])
  const blocks = useMemo(() => (tokens ? blockHashes(tokens.map((t) => t.id), 16) : []), [tokens])
  const n = tokens?.length ?? 0
  const defaultMax = Math.max(0, maxContext - n)
  const st = stepText(sys, step, n, defaultMax)

  return (
    <Figure
      id="template-figure-preprocessor"
      title="Что делает сервер с вашим запросом до GPU"
      actions={
        <Segmented<System>
          size="sm"
          aria-label="Система"
          value={sys}
          onChange={setSys}
          options={[
            { value: 'dynamo', label: 'Dynamo' },
            { value: 'vllm', label: 'vLLM' },
            { value: 'sglang', label: 'SGLang' },
          ]}
        />
      }
      player={<Player {...p.bind} labels={STEP_LABEL[sys]} aria-label="Шаги препроцессора" />}
      caption="Попробуйте: пройдите шаги плеером, а потом переключите систему наверху. Данные те же, меняется только то, какой процесс делает каждый шаг. Текст запроса берётся из симулятора в начале страницы: поменяйте его там, и все шаги пересчитаются."
    >
      <Diagram sys={sys} active={step} onPick={(k) => p.setStep(STEP_KEYS.indexOf(k))} />

      <div className="mt-4 rounded-panel border border-line px-4 py-3.5">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-[15px] font-semibold text-ink">
            {p.step + 1}. {STEP_LABEL[sys][p.step]}
          </span>
          <span className="text-[13px] text-ink-2">{st.where}</span>
        </div>
        {/* место под текст и данные шага зарезервировано: плеер под ними не прыгает при смене шага и системы */}
        <p className="mt-1.5 max-w-[78ch] text-[15px] leading-relaxed text-ink @min-[760px]:min-h-[6.1rem]">{st.text}</p>
        <div className="mt-3 min-h-[340px] min-w-0">
          <StepData
            step={step}
            sys={sys}
            opts={opts}
            rendered={rendered}
            tokens={tokens}
            blocks={blocks}
            defaultMax={defaultMax}
          />
        </div>
      </div>
    </Figure>
  )
}

function Diagram({ sys, active, onPick }: { sys: System; active: StepKey; onPick: (k: StepKey) => void }) {
  const groups = LAYOUT[sys]
  return (
    <div className="flex flex-col items-stretch gap-1.5 @min-[760px]:flex-row @min-[760px]:items-stretch" role="group" aria-label="Процессы системы">
      {groups.map((g, gi) => (
        <Fragment key={g.label}>
          {gi > 0 && <Arrow />}
          <div
            className="flex min-w-0 flex-col rounded-panel border border-dashed border-line-strong px-2 pt-1.5 pb-2 @min-[760px]:flex-[var(--grow)]"
            style={{ ['--grow' as string]: String(g.boxes.length) }}
          >
            <div className="mb-1.5 px-0.5 text-[12px] leading-4 text-ink-2">{g.label}</div>
            <div className="grid flex-1 grid-cols-[repeat(auto-fit,minmax(128px,1fr))] gap-1.5 @min-[760px]:flex @min-[760px]:flex-row">
              {g.boxes.map((b, bi) => (
                <Fragment key={b.key}>
                  {bi > 0 && <Arrow inner />}
                  <button
                    type="button"
                    onClick={() => onPick(b.key)}
                    aria-pressed={active === b.key}
                    className={cn(
                      'grid min-w-0 flex-1 content-center gap-0.5 rounded-control border px-2.5 py-2 text-left transition-colors duration-150',
                      active === b.key ? 'border-accent bg-accent-soft' : 'border-line bg-surface hover:bg-surface-3/60',
                    )}
                  >
                    <span className="flex items-center gap-1.5 text-[13.5px] leading-5 font-medium text-ink">
                      {/* номер шага: на узком экране стрелки внутри группы скрыты, порядок задают номера */}
                      <span
                        aria-hidden
                        className="inline-grid h-4 w-4 shrink-0 place-items-center rounded-full bg-surface-3 font-mono text-[11px] leading-none text-ink-2"
                      >
                        {STEP_KEYS.indexOf(b.key) + 1}
                      </span>
                      {b.title}
                    </span>
                    <span className="text-[12px] leading-4 text-ink-2">{b.sub}</span>
                  </button>
                </Fragment>
              ))}
            </div>
          </div>
        </Fragment>
      ))}
    </div>
  )
}

function Arrow({ inner = false }: { inner?: boolean }) {
  return (
    <span aria-hidden className={cn('shrink-0 items-center justify-center text-ink-3', inner ? 'hidden @min-[760px]:flex' : 'flex')}>
      <ArrowDownIcon size={14} className="@min-[760px]:hidden" />
      <ArrowRightIcon size={14} className="hidden @min-[760px]:block" />
    </span>
  )
}

/** Python-строка списка для показа: ['</answer>']. */
const pyList = (xs: string[]) => `[${xs.map((x) => `'${x}'`).join(', ')}]`

function StepData({
  step,
  sys,
  opts,
  rendered,
  tokens,
  blocks,
  defaultMax,
}: {
  step: StepKey
  sys: System
  opts: RequestOptions
  rendered: ReturnType<typeof renderTemplate>
  tokens: Token[] | null
  blocks: ReturnType<typeof blockHashes>
  defaultMax: number
}) {
  if (step === 'http') {
    // тот же JSON, что в главе «Запрос», только без max_tokens: его подставит сервер (шаг 5)
    const body = requestText(requestBlocks(opts).filter((b) => b.id !== 'max_tokens'))
    return (
      <div className="grid gap-2">
        <CodeBlock lang="json" code={body} wrap maxHeight={300} title="POST /v1/chat/completions" />
        <p className="text-[13px] text-ink-3">
          Тот же запрос, что в главе «Запрос», только max_tokens здесь нарочно не задан: его подставит сервер (шаг 5).
        </p>
      </div>
    )
  }
  if (step === 'template') {
    const chars = rendered.text.length
    return (
      <div className="grid gap-2.5">
        <div
          className="max-h-[260px] overflow-auto rounded-panel bg-surface-2 px-3 py-2.5 shadow-[inset_0_0_0_1px_var(--line)] scrollbar-thin"
          tabIndex={0}
          aria-label="Строка после шаблона"
        >
          <PromptView segments={rendered.segments} family="qwen3" />
        </div>
        <PromptLegend segments={rendered.segments} />
        <p className="text-[13px] text-ink-3">
          Шаблон Qwen3, {fmt.int(chars)} {fmt.plural(chars, ['символ', 'символа', 'символов'])}. Модель обучали ровно на таком
          формате.
        </p>
      </div>
    )
  }
  if (!tokens) return <TokenStripSkeleton count={36} />
  if (step === 'tokenize') {
    return (
      <div className="grid gap-2">
        <div className="max-h-[300px] overflow-y-auto rounded-panel px-0.5 py-1 scrollbar-thin" tabIndex={0} aria-label="Токены промпта">
          <TokenStrip tokens={tokens} showIds size="sm" variant="prompt" />
        </div>
        <p className="text-[13px] text-ink-3">
          {fmt.tokens(tokens.length)} по словарю Qwen3. Спецтокены (в пунктирной рамке) это один ID каждый, а не набор символов.
        </p>
      </div>
    )
  }
  if (step === 'hash') {
    if (sys === 'sglang') {
      return (
        <div className="grid gap-2">
          <div className="max-h-[300px] overflow-y-auto rounded-panel px-0.5 py-1 scrollbar-thin" tabIndex={0} aria-label="Токены промпта">
            <TokenStrip tokens={tokens} size="sm" variant="prompt" />
          </div>
          <p className="text-[13px] text-ink-3">
            Radix tree сравнивает эту последовательность с деревом токен за токеном: совпадение может оборваться на любом токене,
            а не на границе блока.
          </p>
        </div>
      )
    }
    return <BlockList tokens={tokens} blocks={blocks} sys={sys} />
  }
  // worker: что получил движок
  const ids = tokens.map((t) => t.id)
  const shown = `[${ids.slice(0, 8).join(', ')}, …]  (${fmt.int(ids.length)})`
  const { temperature, topP, json } = opts
  const code =
    sys === 'dynamo'
      ? [
          'PreprocessedRequest {',
          `  model: ${JSON.stringify(opts.model)},`,
          `  token_ids: ${shown},`,
          `  stop_conditions: { max_tokens: ${defaultMax}, stop: ${JSON.stringify(STOP_EXAMPLE)}, stop_token_ids_hidden: [${QWEN3_STOP_TOKEN_IDS.join(', ')}] },`,
          // top_k в запросе нет: препроцессор оставляет поле пустым, значение выберет воркер (текст шага 5)
          `  sampling_options: { temperature: ${temperature}, top_p: ${topP}, top_k: None${json ? ', guided_decoding: { json: {…} }' : ''} },`,
          '  …',
          '}',
        ].join('\n')
      : sys === 'vllm'
        ? [
            'EngineCoreRequest(',
            `  prompt_token_ids=${shown},`,
            `  sampling_params=SamplingParams(temperature=${temperature}, top_p=${topP}, max_tokens=${defaultMax}, stop=${pyList(STOP_EXAMPLE)}, …),`,
            '  …',
            ')',
          ].join('\n')
        : [
            'TokenizedGenerateReqInput(',
            `  input_ids=${shown},`,
            `  sampling_params=SamplingParams(temperature=${temperature}, top_p=${topP}, stop=${pyList(STOP_EXAMPLE)}, …),`,
            '  …',
            ')',
          ].join('\n')
  // python-подсветка: строки и числа цветом, а стоп-строка </answer> не выглядит спецтокеном (в text она обводится)
  return <CodeBlock lang="python" code={code} wrap title={sys === 'dynamo' ? 'запрос к воркеру (упрощённо)' : 'что получил движок (упрощённо)'} />
}

function BlockList({ tokens, blocks, sys }: { tokens: Token[]; blocks: ReturnType<typeof blockHashes>; sys: System }) {
  return (
    <div className="grid gap-2">
      {/* потолок высоты, как у лент токенов в других шагах: длинный промпт (десятки блоков) иначе растягивал фигуру на
          тысячи пикселей только на этом шаге, и при автоплее страница под ней прыгала */}
      <div className="max-h-[300px] overflow-y-auto rounded-panel p-0.5 scrollbar-thin" tabIndex={0} aria-label="Блоки промпта и их хэши">
        <ol className="grid gap-1.5 @min-[560px]:grid-cols-2">
          {blocks.map((b) => {
            const toks = tokens.slice(b.start, b.start + b.tokenIds.length)
            return (
              <li key={b.index} className={cn('min-w-0 rounded-control border px-2.5 py-1.5', b.full ? 'border-line bg-surface' : 'border-dashed border-line-strong bg-surface-2')}>
                <div className="flex items-baseline justify-between gap-2 text-[12px]">
                  <span className="text-ink-2">
                    Блок {b.index} · {fmt.tokens(b.tokenIds.length)}
                  </span>
                  <span className="font-mono text-ink tabular-nums">{b.full ? b.hash : 'не хэшируется'}</span>
                </div>
                <div className="mt-1 truncate font-mono text-[12px] leading-5 text-ink-2" title={toks.map((t) => t.text).join('')}>
                  {toks.map((t, i) => (t.special ? <Special key={i} text={t.raw} /> : <Fragment key={i}>{t.text.replace(/\n/g, '↵')}</Fragment>))}
                </div>
              </li>
            )
          })}
        </ol>
      </div>
      <p className="text-[13px] text-ink-3">
        {sys === 'dynamo'
          ? 'Хэши условные (наша функция для показа), принцип как в Dynamo: XXH3 с seed 1337, цепочкой от блока к блоку. Стоит поменять один токен в блоке 0, и изменятся хэши блока 0 и всех следующих.'
          : 'Хэши условные (наша функция для показа). vLLM по умолчанию считает SHA-256 от хэша предыдущего блока, токенов блока и дополнительных ключей (LoRA, картинки, cache_salt): поменяйте один токен в блоке 0, и изменятся хэши блока 0 и всех следующих.'}
      </p>
    </div>
  )
}
