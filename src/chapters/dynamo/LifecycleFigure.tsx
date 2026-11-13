import { useMemo, useRef, useState, type FocusEvent, type PointerEvent, type ReactNode } from 'react'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { KvRouteDecision } from '@/engine/router'
import { Badge, CodeBlock, Figure, Player, Segmented, Skeleton, TokenStrip, TokenStripSkeleton, cn, cssVar, fmt, useElementSize, usePlayer, useTip } from '@/ui'
import { Diagram } from './diagram'
import { ARCH_V_BREAK, archLayout } from './archLayout'
import { blockStored, chosenNode, computeRun, type LifecycleRun, type Mode } from './lifecycle'
import { useFocusWithin } from './useFocusWithin'

interface Step {
  id: string
  label: string
  title: string
  nodes: string[]
  edges?: string[]
  packet?: { edgeId: string; color?: string }
  body: (run: LifecycleRun) => ReactNode
}

const BLOCK_FORMS = ['блок', 'блока', 'блоков'] as const
const FIGURE_ID = 'dynamo-figure-lifecycle'

/** Фигура 2 главы 19: путь вашего запроса через Dynamo по шагам, на ваших токенах. */
export function LifecycleFigure() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const [mode, setMode] = useState<Mode>('disagg')
  const run = useMemo(
    () => (tokenizer ? computeRun(settings, tokenizer) : null),
    [
      tokenizer,
      settings.systemText,
      settings.userText,
      settings.thinking,
      settings.tools,
      settings.json,
      settings.scenarioId,
      settings.modelId,
      settings.gpuId,
      settings.temperature,
    ],
  )
  const decode = useMemo(() => (tokenizer ? (ids: number[]) => tokenizer.decode(ids) : null), [tokenizer])
  const steps = useMemo(() => (run && decode ? (mode === 'disagg' ? disaggSteps(run, decode) : aggSteps(run, decode)) : []), [run, mode, decode])
  const p = usePlayer(Math.max(1, steps.length), { intervalMs: 2600 })
  const step = steps[Math.min(p.step, steps.length - 1)]
  // на узком экране схема встаёт вертикально, а живой регион говорит только заголовок шага и только когда фокус в фигуре
  const [boxRef, box] = useElementSize<HTMLDivElement>()
  const narrow = box.width > 0 && box.width < ARCH_V_BREAK
  const layout = archLayout(mode, narrow)
  const inside = useFocusWithin(FIGURE_ID)

  return (
    <Figure
      id={FIGURE_ID}
      title="Ваш запрос проходит через Dynamo"
      width="full"
      actions={
        <Segmented
          aria-label="Режим"
          size="sm"
          value={mode}
          onChange={(m) => {
            setMode(m)
            p.reset()
          }}
          options={[
            { value: 'disagg', label: 'Дезагрегированный' },
            { value: 'agg', label: 'Агрегированный' },
          ]}
        />
      }
      player={<Player {...p.bind} labels={steps.map((s) => s.label)} aria-label="Шаги запроса" />}
      stickyControls
      caption="Попробуйте: идите по шагам стрелками или пробелом и посмотрите на шаг «Хэши»: роутеру нужны готовые token IDs, поэтому шаблон и токенизация стоят до него. Токены, хэши, время и SSE считаются по вашему запросу из симулятора, ответ модели условный и тот же, что в симуляторе: включите там tools или JSON, и запрос, ответ и парсер здесь тоже поменяются."
    >
      <div ref={boxRef} className="min-w-0">
        {!run || !step ? (
          <div className="grid gap-4" aria-busy="true">
            <Skeleton height={300} radius={12} />
            <TokenStripSkeleton count={40} />
          </div>
        ) : (
          <div className="grid min-w-0 gap-5">
            <Diagram
              width={layout.width}
              height={layout.height}
              nodes={layout.nodes}
              edges={layout.edges}
              activeNodes={new Set(step.nodes)}
              activeEdges={new Set(step.edges ?? (step.packet ? [step.packet.edgeId] : []))}
              packet={step.packet ? { ...step.packet, key: `${mode}-${p.step}` } : null}
              minWidth={layout.minWidth}
              className={cn(narrow && 'mx-auto w-full max-w-[420px]')}
              aria-label={`Схема Dynamo, шаг ${p.step + 1}: ${step.title}`}
            />
            {/* высота почти постоянная: плеер под текстом шага не прыгает от шага к шагу */}
            <div className="grid min-w-0 content-start gap-3 border-t border-line pt-5 @4xl:min-h-[27rem]">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1" aria-live={inside ? 'polite' : 'off'}>
                <span className="text-[13px] text-ink-3 tabular-nums">
                  шаг {p.step + 1} из {steps.length}
                </span>
                <span className="text-[15px] font-semibold text-ink">{step.title}</span>
              </div>
              {step.body(run)}
            </div>
          </div>
        )}
      </div>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Шаги
 * ------------------------------------------------------------------------------------------- */

function P({ children }: { children: ReactNode }) {
  return <p className="max-w-[80ch] text-[15px] leading-relaxed text-ink">{children}</p>
}

function Grid({ children }: { children: ReactNode }) {
  return <div className="grid min-w-0 gap-4 @4xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">{children}</div>
}

function Num({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="grid gap-0.5">
      <span className="text-[13px] text-ink-3">{label}</span>
      <span className="text-[19px] font-semibold text-ink tabular-nums">{value}</span>
      {hint && <span className="text-[12.5px] text-ink-3">{hint}</span>}
    </div>
  )
}

function Nums({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap gap-x-8 gap-y-3">{children}</div>
}

/** JSON с отступами, но массивы чисел и строк в одну строку: так короче и читается как в логах. */
function json(v: unknown, indent = ''): string {
  if (Array.isArray(v)) {
    if (v.every((x) => x === null || typeof x !== 'object')) return '[' + v.map((x) => JSON.stringify(x)).join(', ') + ']'
    const inner = indent + '  '
    return '[\n' + v.map((x) => inner + json(x, inner)).join(',\n') + '\n' + indent + ']'
  }
  if (v && typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>)
    if (entries.length === 0) return '{}'
    const inner = indent + '  '
    return '{\n' + entries.map(([k, x]) => `${inner}${JSON.stringify(k)}: ${json(x, inner)}`).join(',\n') + '\n' + indent + '}'
  }
  return JSON.stringify(v)
}

/** JSON в одну строку, но с пробелами после «,» и «:» между полями: длинная строка переносится между полями, а не посреди слова. */
function spaced(v: unknown): string {
  if (Array.isArray(v)) return '[' + v.map(spaced).join(', ') + ']'
  if (v && typeof v === 'object') {
    return '{' + Object.entries(v as Record<string, unknown>).map(([k, x]) => `${JSON.stringify(k)}: ${spaced(x)}`).join(', ') + '}'
  }
  return JSON.stringify(v)
}

/** Строка SSE; у всех, кроме первой, повторяющиеся поля id, object, created, model свёрнуты. */
function shortSse(c: { line: string; json: Record<string, unknown> | null }, full: boolean): string {
  if (!c.json) return c.line
  if (full) return 'data: ' + spaced(c.json)
  const { id: _id, object: _o, created: _c, model: _m, ...rest } = c.json
  return 'data: {…, ' + spaced(rest).slice(1)
}

/** Строка SSE с дельтой content, в которой есть не только пробелы и переводы строк. */
const CONTENT_TEXT_RE = /"content": "(?!(?:\\n|\s)*")/

/** Какие поля ответа встречаются в потоке после первой строки с ролью. */
function sseFields(chunks: readonly { json: Record<string, unknown> | null }[]) {
  let reasoning = false
  let content = false
  let tool = false
  for (const c of chunks.slice(1)) {
    const choices = (c.json?.choices ?? []) as { delta?: Record<string, unknown> }[]
    const delta = choices[0]?.delta
    if (!delta) continue
    if (typeof delta.reasoning_content === 'string' && delta.reasoning_content !== '') reasoning = true
    if (typeof delta.content === 'string' && delta.content.trim() !== '') content = true
    if (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0) tool = true
  }
  return { reasoning, content, tool }
}

const commonHead = (decode: (ids: number[]) => string): Step[] => [
  {
    id: 'request',
    label: 'Запрос',
    title: 'Клиент отправляет JSON на frontend',
    nodes: ['client', 'http'],
    packet: { edgeId: 'client-http' },
    body: (run) => (
      <Grid>
        <div className="grid content-start gap-3">
          <P>
            Обычный запрос OpenAI API на порт 8000. Frontend проверяет, что такая модель зарегистрирована, и передаёт запрос в preprocessor. Ни одного GPU
            пока не задействовано.
          </P>
          {'tools' in run.request && (
            <P>
              Описание инструмента <code>get_weather</code> едет в поле <code>tools</code>: на следующем шаге шаблон положит его в системную часть промпта.
            </P>
          )}
          {run.jsonSchemaName && (
            <P>
              Поле <code>response_format</code> с JSON-схемой frontend передаст дальше как правило для маски structured output: применять её будет движок
              воркера, который сгенерирует ответ (в дезагрегированном режиме это decode-воркер).
            </P>
          )}
        </div>
        <CodeBlock lang="json" title="POST /v1/chat/completions" code={json(run.request)} wrap maxHeight={260} />
      </Grid>
    ),
  },
  {
    id: 'template',
    label: 'Шаблон',
    title: 'Preprocessor рендерит chat template',
    nodes: ['pre'],
    packet: { edgeId: 'http-pre' },
    body: (run) => (
      <Grid>
        <P>
          Модель не понимает JSON: ей нужен один текст со служебными маркерами. Шаблон Qwen3 из <code>tokenizer_config.json</code> рендерит minijinja прямо во
          frontend, на Rust. Получилось {fmt.count(run.prompt.template.text.length, ['символ', 'символа', 'символов'])}.
        </P>
        <CodeBlock lang="text" title="промпт после шаблона" code={run.prompt.template.text} wrap maxHeight={260} />
      </Grid>
    ),
  },
  {
    id: 'tokens',
    label: 'Токены',
    title: 'Токенизация: текст становится token IDs',
    nodes: ['pre'],
    body: (run) => (
      <Grid>
        <div className="grid min-w-0 content-start gap-3">
          <P>
            HF tokenizers режет промпт на {fmt.tokens(run.prompt.ids.length)} словаря Qwen3. Preprocessor проверяет, что промпт плюс <code>max_tokens</code>{' '}
            ({fmt.int(run.maxTokens)}) помещаются в контекст, и собирает <code>PreprocessedRequest</code>: дальше по системе едут числа, а не текст.
            {run.jsonSchemaName && (
              <>
                {' '}
                Схема из <code>response_format</code> уходит в параметры сэмплинга: маску по ней на каждом шаге
                {run.answer.thinking && (
                  <>
                    {' '}
                    после <code>&lt;/think&gt;</code>
                  </>
                )}{' '}
                применит движок.
              </>
            )}
          </P>
          <TokenStrip tokens={run.prompt.tokens} variant="prompt" showIds limit={40} size="sm" />
        </div>
        <CodeBlock
          lang="json"
          title="PreprocessedRequest (упрощённо)"
          wrap
          maxHeight={180}
          code={json({
            model: run.request.model,
            token_ids: [...run.prompt.ids.slice(0, 10), '…'],
            stop_conditions: { max_tokens: run.maxTokens, stop_token_ids_hidden: [151645, 151643] },
            sampling_options: {
              temperature: run.request.temperature,
              ...(run.jsonSchemaName ? { guided_decoding: { json: `{…схема ${run.jsonSchemaName}…}` } } : {}),
            },
            output_options: { skip_special_tokens: true },
          })}
        />
      </Grid>
    ),
  },
  {
    id: 'hashes',
    label: 'Хэши',
    title: 'Роутер режет token IDs на блоки и считает хэши',
    nodes: ['router'],
    packet: { edgeId: 'pre-router' },
    body: (run) => (
      <div className="grid min-w-0 gap-4">
        <P>
          Промпт длиной {fmt.tokens(run.prompt.ids.length)} даёт {fmt.count(run.prompt.fullBlocks, BLOCK_FORMS)} по 16 токенов
          {run.prompt.tailTokens > 0 && <> и хвост из {fmt.tokensGen(run.prompt.tailTokens)}, который не хэшируется</>}. Хэш каждого блока зависит от его
          токенов и от хэша предыдущего. По этим хэшам роутер спрашивает свой индекс, у какого воркера лежит самое длинное начало запроса.
        </P>
        <BlockChips run={run} decode={decode} />
        <div className="rounded-panel bg-surface-2 px-4 py-3 text-[14px] leading-relaxed text-ink">
          <span className="font-semibold">Вот почему preprocessing стоит до роутера.</span> Без шаблона и токенизатора нет token IDs, без token IDs нет хэшей
          блоков, а без хэшей роутер не может сравнить запрос с кэшами воркеров. Поэтому в Dynamo токенизирует frontend, а не движок.
        </div>
      </div>
    ),
  },
]

function routeTable(d: KvRouteDecision, kind: 'prefill' | 'decode' | 'agg') {
  const th = 'px-3 py-1.5 text-right font-normal'
  const td = 'px-3 py-1.5 text-right'
  return (
    <div className="min-w-0 overflow-x-auto">
      <table className={cn('w-full border-collapse text-[14px] tabular-nums', kind === 'agg' ? 'min-w-[560px]' : 'min-w-[380px]')}>
        <thead>
          <tr className="border-b border-line text-left text-[13px] text-ink-3">
            <th className="py-1.5 pr-3 font-normal">Воркер</th>
            {kind !== 'decode' && <th className={th}>overlap</th>}
            {kind !== 'decode' && <th className={th}>очередь prefill</th>}
            {kind !== 'decode' && <th className={th}>prefill-часть, блоков</th>}
            {kind !== 'prefill' && <th className={th}>{kind === 'decode' ? 'активные блоки' : 'decode-часть, блоков'}</th>}
            {kind === 'agg' && <th className={th}>стоимость, блоков</th>}
            <th className="py-1.5 pl-3 font-normal" />
          </tr>
        </thead>
        <tbody>
          {d.scores.map((s) => (
            <tr key={s.workerId} className="border-b border-line">
              <td className="py-1.5 pr-3 text-ink">{s.name}</td>
              {kind !== 'decode' && <td className={cn(td, 'text-ink-2')}>{fmt.count(s.overlapBlocks, BLOCK_FORMS)}</td>}
              {kind !== 'decode' && <td className={cn(td, 'text-ink-2')}>{fmt.tokens(s.queuedPrefillTokens)}</td>}
              {kind !== 'decode' && <td className={cn(td, kind === 'prefill' ? 'text-ink' : 'text-ink-2')}>{blocksNum(s.prefillTerm ?? 0)}</td>}
              {kind === 'decode' && <td className={cn(td, 'text-ink')}>{fmt.int(s.activeBlocks)}</td>}
              {kind === 'agg' && <td className={cn(td, 'text-ink-2')}>{blocksNum(s.decodeTerm ?? 0)}</td>}
              {kind === 'agg' && <td className={cn(td, 'font-semibold text-ink')}>{blocksNum(s.cost)}</td>}
              <td className="py-1.5 pl-3">{s.workerId === d.chosen && <Badge tone="accent">выбран</Badge>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const name = (d: KvRouteDecision) => d.scores.find((s) => s.workerId === d.chosen)?.name ?? d.chosen
/** Блоки: целые без «,0», дробные с одним знаком. */
const blocksNum = (x: number) => fmt.num(x, Number.isInteger(Math.round(x * 10) / 10) ? 0 : 1)
const genitive = (n: string) => n.replace(/^Воркер /, 'Воркера ')

/** Почему агрегированный роутер выбрал этот воркер: кэш против нагрузки, с числами. */
function aggWhy(d: KvRouteDecision): string {
  const chosen = d.scores.find((s) => s.workerId === d.chosen)!
  const warm = d.scores.reduce((a, b) => (b.overlapBlocks > a.overlapBlocks ? b : a))
  if (warm.overlapBlocks === 0) return 'Начала промпта нет в кэше ни у одного воркера: решает нагрузка.'
  if (warm.workerId === chosen.workerId) {
    return `${chosen.name} помнит начало промпта (${fmt.count(warm.overlapBlocks, BLOCK_FORMS)}), и скидка за кэш перевешивает разницу в нагрузке.`
  }
  return (
    `${warm.name} помнит начало промпта (${fmt.count(warm.overlapBlocks, BLOCK_FORMS)}), но держит ${fmt.int(warm.activeBlocks)} активных блоков ` +
    `против ${fmt.int(chosen.activeBlocks)} у ${genitive(chosen.name)}. Скидка за кэш меньше разницы в нагрузке, поэтому выбран ${chosen.name}.`
  )
}

/** Событие BlockStored для воркера, который только что посчитал prefill (только новые полные блоки). */
function EventsBody({ run, overlap, delayNote }: { run: LifecycleRun; overlap: number; delayNote: boolean }) {
  const ev = blockStored(run.prompt, overlap)
  return (
    <Grid>
      <P>
        {ev ? (
          <>
            Движок публикует <code>BlockStored</code> с хэшами новых полных блоков ({fmt.count(ev.newBlocks, BLOCK_FORMS)}; то, что уже лежало в prefix
            cache, повторно не сообщается), Dynamo-воркер пересылает событие в event plane, и индекс роутера обновляется.
          </>
        ) : (
          <>
            Новых полных блоков нет: всё, что можно закэшировать, у воркера уже было, поэтому событие <code>BlockStored</code> не публикуется.
          </>
        )}{' '}
        Следующий запрос с тем же началом получит у этого воркера скидку за кэш.
        {delayNote && <> Событие приходит с задержкой: если похожий запрос прилетит раньше, роутер этих блоков ещё не увидит.</>}
      </P>
      {ev && <CodeBlock lang="json" title="BlockStored (упрощённо)" wrap maxHeight={220} code={json(ev.event)} />}
    </Grid>
  )
}

function disaggSteps(run: LifecycleRun, decode: (ids: number[]) => string): Step[] {
  const pNode = chosenNode('p', run.prefillRoute)
  const dNode = chosenNode('d', run.decodeRoute)
  const pOverlap = run.prefillRoute.scores.find((s) => s.workerId === run.prefillRoute.chosen)?.overlapBlocks ?? 0
  return [
    ...commonHead(decode),
    {
      id: 'route-prefill',
      label: 'Prefill-роутер',
      title: `Роутер выбирает prefill-воркер: ${name(run.prefillRoute)}`,
      nodes: ['router', pNode],
      packet: { edgeId: 'router-prefill' },
      body: (r) => (
        <Grid>
          <div className="grid content-start gap-3">
            <P>
              Сначала роутер ищет, где посчитать промпт. Учитываются блоки, которые уже лежат в кэше воркера, и prefill-работа в его очереди. Prefill-воркеру
              уходит копия запроса с <code>max_tokens = 1</code>: ему нужно только посчитать KV.
            </P>
            <P>
              {r.prefillRoute.scores[0].overlapBlocks === 0
                ? 'Prefill 1 недавно видел тот же системный промпт, но общая часть короче одного блока в 16 токенов, совпадать нечему. Выигрывает Prefill 1: у Prefill 2 в очереди ещё 512 токенов чужого prefill.'
                : 'Prefill 1 недавно обслуживал другого пользователя с тем же системным промптом: общие блоки начала лежат у него в кэше. Кэш Prefill 2 пуст, и в его очереди ещё 512 токенов чужого prefill.'}
            </P>
          </div>
          {routeTable(r.prefillRoute, 'prefill')}
        </Grid>
      ),
    },
    {
      id: 'prefill',
      label: 'Prefill',
      title: `${name(run.prefillRoute)} считает KV промпта`,
      nodes: [pNode],
      body: (r) => (
        <div className="grid min-w-0 gap-4">
          <Nums>
            <Num label="Токенов в промпте" value={fmt.int(r.prompt.ids.length)} />
            <Num label="Из prefix cache" value={fmt.int(r.cachedTokens)} />
            <Num label="Время prefill" value={fmt.ms(r.prefillMs)} hint={`${r.model.name}, ${r.gpu.name}, perf.ts`} />
            <Num label="KV этого запроса" value={fmt.bytes(r.kvBytes)} hint={`${fmt.count(r.kvBlocks, BLOCK_FORMS)} по 16 токенов`} />
          </Nums>
          <Grid>
            <P>
              Движок прогоняет промпт через все слои и сэмплирует ровно один токен, но в пути vLLM этот токен выбрасывается: первый видимый токен сделает
              decode-воркер. Блоки KV остаются у prefill-воркера, пока их не заберут, а роутеру возвращаются <code>disaggregated_params</code>: где лежит KV.
            </P>
            <CodeBlock
              lang="json"
              title="disaggregated_params (номера и адрес условные)"
              wrap
              maxHeight={220}
              code={json({
                kv_transfer_params: {
                  do_remote_prefill: true,
                  remote_block_ids: [r.kvBlocks <= 4 ? Array.from({ length: r.kvBlocks }, (_, i) => 17 + i) : [17, 18, '…', 16 + r.kvBlocks]],
                  remote_engine_id: 'e3f1…',
                  remote_host: '10.0.3.17',
                  remote_port: 5600,
                  tp_size: 1,
                },
              })}
            />
          </Grid>
        </div>
      ),
    },
    {
      id: 'events',
      label: 'KV events',
      title: 'Воркер сообщает роутеру о новых блоках',
      nodes: [pNode, 'events', 'router'],
      edges: ['prefill-events', 'events-router'],
      packet: { edgeId: 'prefill-events', color: 'var(--ink-3)' },
      body: (r) => <EventsBody run={r} overlap={pOverlap} delayNote />,
    },
    {
      id: 'route-decode',
      label: 'Decode-роутер',
      title: `Роутер выбирает decode-воркер по нагрузке: ${name(run.decodeRoute)}`,
      nodes: ['router', dNode],
      packet: { edgeId: 'router-decode' },
      body: (r) => (
        <Grid>
          <P>
            Decode-воркер выбирается только по нагрузке: кредит за кэш здесь ноль, ведь KV всё равно приедет от prefill-воркера. Роутер восстанавливает
            исходный <code>max_tokens</code> и кладёт в запрос метаданные prefill.
          </P>
          {routeTable(r.decodeRoute, 'decode')}
        </Grid>
      ),
    },
    {
      id: 'transfer',
      label: 'NIXL',
      title: 'NIXL переносит KV из GPU в GPU',
      nodes: ['nixl', pNode, dNode],
      edges: ['prefill-nixl', 'nixl-decode'],
      packet: { edgeId: 'nixl-decode' },
      body: (r) => (
        <div className="grid min-w-0 gap-4">
          <Nums>
            <Num label="Объём KV" value={fmt.bytes(r.kvBytes)} hint={`${fmt.tokens(r.prompt.ids.length)} × KV на токен`} />
            {r.transfer.map((t) => (
              <Num key={t.link} label={t.link} value={fmt.ms(t.ms)} hint="только полоса, без задержек" />
            ))}
          </Nums>
          <P>
            Decode-воркер vLLM по адресу из метаданных сам читает блоки из памяти prefill-воркера (NIXL READ). Данные идут по RDMA или NVLink мимо CPU, а по
            сети управления едут только номера блоков. Для короткого промпта это доли миллисекунды, для промпта на десятки тысяч токенов уже десятки
            миллисекунд.
          </P>
        </div>
      ),
    },
    {
      id: 'decode',
      label: 'Decode',
      title: `${name(run.decodeRoute)} генерирует ответ`,
      nodes: [dNode],
      packet: { edgeId: 'decode-post', color: 'decode' },
      body: (r) => decodeBody(r),
    },
    ...commonTail('disagg'),
  ]
}

function aggSteps(run: LifecycleRun, decode: (ids: number[]) => string): Step[] {
  const wNode = chosenNode('w', run.aggRoute)
  const wOverlap = run.aggRoute.scores.find((s) => s.workerId === run.aggRoute.chosen)?.overlapBlocks ?? 0
  return [
    ...commonHead(decode),
    {
      id: 'route-agg',
      label: 'Роутер',
      title: `Роутер выбирает воркер: ${name(run.aggRoute)}`,
      nodes: ['router', wNode],
      packet: { edgeId: 'router-workers' },
      body: (r) => (
        <div className="grid min-w-0 gap-4">
          <P>
            В агрегированном режиме роутер выбирает один воркер на весь запрос: стоимость складывает prefill-работу за вычетом кэша и decode-нагрузку, обе
            части в KV-блоках. Подробно формула разобрана в следующей главе. {aggWhy(r.aggRoute)}
          </P>
          {routeTable(r.aggRoute, 'agg')}
        </div>
      ),
    },
    {
      id: 'worker',
      label: 'Prefill + decode',
      title: `${name(run.aggRoute)} считает prefill и сразу генерирует`,
      nodes: [wNode],
      body: (r) => (
        <div className="grid min-w-0 gap-4">
          <Nums>
            <Num label="Токенов в промпте" value={fmt.int(r.prompt.ids.length)} />
            <Num label="Из prefix cache" value={fmt.int(r.aggCachedTokens)} />
            <Num label="Время prefill" value={fmt.ms(r.aggPrefillMs)} hint={`${r.model.name}, ${r.gpu.name}, perf.ts`} />
            <Num label="KV этого запроса" value={fmt.bytes(r.kvBytes)} />
          </Nums>
          <P>
            KV никуда не едет: prefill и decode идут на одних и тех же GPU, а scheduler движка перемешивает чанки prefill с шагами decode других запросов.
            Переноса нет, зато длинный prefill чужого запроса может задержать ваши токены.
          </P>
        </div>
      ),
    },
    {
      id: 'events',
      label: 'KV events',
      title: 'Воркер сообщает роутеру о новых блоках',
      nodes: [wNode, 'events', 'router'],
      edges: ['workers-events', 'events-router'],
      packet: { edgeId: 'workers-events', color: 'var(--ink-3)' },
      body: (r) => <EventsBody run={r} overlap={wOverlap} delayNote={false} />,
    },
    {
      id: 'decode',
      label: 'Decode',
      title: `${name(run.aggRoute)} стримит token IDs`,
      nodes: [wNode],
      packet: { edgeId: 'workers-post', color: 'decode' },
      body: (r) => decodeBody(r),
    },
    ...commonTail('agg'),
  ]
}

function decodeBody(r: LifecycleRun) {
  const first = r.answer.tokens.slice(0, 24)
  return (
    <div className="grid min-w-0 gap-4">
      <P>
        Каждый шаг decode добавляет один токен. Воркер отдаёт наружу только номера токенов, без текста: превратить их в буквы и разобрать на мысли, ответ и
        вызовы инструментов должен frontend. Ответ ниже условный (модель в браузере не запускается), токены настоящие. Один шаг decode при batch 1 занимает
        около {fmt.ms(r.itlMs)}.
      </P>
      <TokenStrip tokens={first} variant="generated" showIds size="sm" />
      <CodeBlock
        lang="json"
        title="LLMEngineOutput, первые чанки (упрощённо)"
        wrap
        maxHeight={160}
        code={[0, 1, 2].map((i) => JSON.stringify({ token_ids: [r.answer.tokens[i]?.id], finish_reason: null })).join('\n') + '\n…'}
      />
    </div>
  )
}

function commonTail(mode: Mode): Step[] {
  return [
    {
      id: 'post',
      label: 'Postprocessor',
      title: 'Postprocessor: детокенизация и парсеры',
      nodes: ['post'],
      body: (r) => <PostBody run={r} />,
    },
    {
      id: 'sse',
      label: 'SSE',
      title: 'Клиент получает поток SSE',
      nodes: ['post', 'client'],
      packet: { edgeId: 'post-client', color: 'decode' },
      body: (r) => {
        // usage считает тот воркер, который считал prefill: в disagg prefill-воркер, в agg единственный воркер запроса
        const all = mode === 'agg' ? r.answer.sseAgg : r.answer.sse
        const lines = all.map((c, i) => shortSse(c, i === 0))
        const shown = lines.length > 9 ? [...lines.slice(0, 5), '…', ...lines.slice(-4)] : lines
        const hl: number[] = []
        const mark = (pred: (l: string, i: number) => boolean) => {
          const i = shown.findIndex(pred)
          if (i >= 0) hl.push(2 * i + 1)
        }
        mark((l) => l.includes('"reasoning_content": "') && !l.includes('"reasoning_content": "\\n"'))
        mark((l, i) => i > 0 && CONTENT_TEXT_RE.test(l))
        mark((l) => l.includes('"tool_calls"'))
        // какие поля на самом деле есть в этом потоке: без thinking нет reasoning_content, у вызова инструмента нет текста
        // (одиночный «\n\n» между </think> и вызовом текстом не считается)
        const { reasoning, content, tool } = sseFields(all)
        const deltas = [reasoning && 'reasoning_content', content && 'content'].filter((f): f is string => Boolean(f))
        return (
          <div className="grid min-w-0 gap-3">
            <P>
              Каждый кусок ответа уходит клиенту отдельным событием <code>data: …</code>: сначала роль
              {deltas.length > 0 && (
                <>
                  , затем дельты <code>{deltas[0]}</code>
                  {deltas[1] && (
                    <>
                      {' '}
                      и <code>{deltas[1]}</code>
                    </>
                  )}
                </>
              )}
              {tool && (
                <>
                  {deltas.length === 0 ? ', затем ' : deltas.length > 1 ? ', потом ' : ' и '}вызов инструмента одним куском <code>tool_calls</code>
                </>
              )}
              , в конце <code>finish_reason</code>, usage с числом закэшированных токенов и <code>[DONE]</code>. Всего{' '}
              {fmt.count(lines.length, ['строка', 'строки', 'строк'])}; в строках после первой повторяющиеся поля <code>id</code>, <code>object</code>,{' '}
              <code>created</code> и <code>model</code> свёрнуты в «…».
            </P>
            <CodeBlock lang="text" title="text/event-stream" code={shown.join('\n\n')} wrap maxHeight={240} highlightLines={hl} />
          </div>
        )
      },
    },
  ]
}

function PostBody({ run }: { run: LifecycleRun }) {
  let reasoning = ''
  let content = ''
  const calls: string[] = []
  for (const e of run.answer.parser) {
    for (const o of e.out) {
      if (o.kind === 'reasoning') reasoning += o.text
      else if (o.kind === 'content') content += o.text
      else if (o.kind === 'tool') calls.push(`${o.delta.name ?? ''}(${o.delta.arguments ?? ''})`)
    }
  }
  const held = run.answer.detok.filter((d) => d.heldBytes.length > 0).length
  const tool = run.answer.bodyKind === 'tool_call' || calls.length > 0
  return (
    <div className="grid min-w-0 gap-4">
      <P>
        DecodeStream превращает номера в текст и придерживает байты, пока символ не собран целиком.{' '}
        {held > 0
          ? `В этом ответе так было на ${fmt.count(held, ['шаге', 'шагах', 'шагах'])} из ${fmt.int(run.answer.detok.length)}.`
          : 'В этом ответе каждый токен сразу дал целые символы; с эмодзи и редкими буквами так бывает не всегда.'}{' '}
        {run.answer.thinking ? (
          <>
            Затем reasoning-парсер Qwen3 видит теги <code>&lt;think&gt;</code> и раскладывает поток по полям,
          </>
        ) : (
          <>Thinking выключен, поэтому рассуждения нет,</>
        )}{' '}
        а EOS <code>&lt;|im_end|&gt;</code> заканчивает генерацию и в текст не попадает.
        {tool && (
          <>
            {' '}
            Парсер hermes находит <code>&lt;tool_call&gt;…&lt;/tool_call&gt;</code>, держит вызов в jail и отдаёт его одним чанком{' '}
            <code>tool_calls</code>; <code>finish_reason</code> становится <code>tool_calls</code>.
          </>
        )}
        {run.answer.bodyKind === 'json' && (
          <>
            {' '}
            Ответ
            {run.answer.thinking && (
              <>
                {' '}
                после <code>&lt;/think&gt;</code>
              </>
            )}{' '}
            шёл под маской JSON-схемы: её на каждом шаге накладывал движок воркера, который генерировал ответ, а postprocessor просто отдаёт текст в{' '}
            <code>content</code>.
          </>
        )}
      </P>
      <div className={cn('grid min-w-0 gap-3', tool ? '@3xl:grid-cols-3' : '@3xl:grid-cols-2')}>
        <Lane label="reasoning_content" role="reasoning" text={reasoning} empty="рассуждения нет: thinking выключен" italic />
        <Lane label="content" role="decode" text={content} empty={tool ? 'пусто: вместо текста вызов инструмента' : 'пусто'} />
        {tool && <Lane label="tool_calls" role="tool" text={calls.join('\n')} empty="вызовов нет" mono />}
      </div>
    </div>
  )
}

function Lane({ label, role, text, empty, italic, mono }: { label: string; role: string; text: string; empty?: string; italic?: boolean; mono?: boolean }) {
  return (
    <div className="grid min-w-0 content-start gap-1.5">
      <span className="inline-flex items-center gap-1.5 font-mono text-[12.5px] text-ink-2">
        <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: cssVar(role as never) }} />
        {label}
      </span>
      <div
        className={cn(
          'rounded-control bg-surface-2 px-3 py-2 text-[14px] break-words',
          italic ? 'border-l-2 border-dashed text-ink-2 italic' : 'text-ink',
          mono && 'font-mono text-[13px] whitespace-pre-wrap',
        )}
        style={italic ? { borderColor: cssVar('reasoning') } : undefined}
      >
        {text.trim() ? text.trim() : <span className="text-ink-3 not-italic">{empty}</span>}
      </div>
    </div>
  )
}

/** Блоки промпта цепочкой: стрелка между блоками, номера токенов, по наведению или фокусу текст блока. */
function BlockChips({ run, decode }: { run: LifecycleRun; decode: (ids: number[]) => string }) {
  const scope = useRef<HTMLOListElement>(null)
  const tip = useTip({ scopeEl: scope.current })
  const blocks = run.prompt.blocks
  const shown = blocks.length > 10 ? [...blocks.slice(0, 8), null, ...blocks.slice(-2)] : blocks
  const show = (el: Element) => {
    const i = Number(el.getAttribute('data-block'))
    const b = blocks[i]
    if (!b) return
    tip.show(
      el.getBoundingClientRect(),
      <div className="grid gap-1">
        <div className="font-medium text-ink">
          Блок {b.index}
          {b.full ? (
            <>
              {' '}
              · хэш <span className="font-mono">{b.hash}</span>
            </>
          ) : (
            ' · хвост, не хэшируется'
          )}
        </div>
        <div className="font-mono text-[12px] break-all whitespace-pre-wrap text-ink-2">{decode(b.tokenIds).replace(/ /g, '·').replace(/\n/g, '↵\n')}</div>
      </div>,
    )
  }
  const onMove = (e: PointerEvent<HTMLOListElement>) => {
    const el = (e.target as HTMLElement).closest('[data-block]')
    if (el) show(el)
  }
  const onFocus = (e: FocusEvent<HTMLOListElement>) => {
    const el = (e.target as HTMLElement).closest('[data-block]')
    if (el) show(el)
  }
  const second = blocks[1]
  return (
    <div className="grid min-w-0 gap-2">
      <ol
        ref={scope}
        className="flex flex-wrap items-stretch gap-y-1.5"
        aria-label="Блоки промпта и их хэши"
        onPointerMove={onMove}
        onPointerDown={onMove}
        onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
        onFocus={onFocus}
        onBlur={tip.hide}
      >
        {shown.map((b, i) =>
          b === null ? (
            <li key="gap" className="flex items-center px-1.5 text-ink-3">
              …
            </li>
          ) : (
            <li key={b.index} className="flex items-center">
              {i > 0 && (
                <span aria-hidden className="px-1 text-ink-3">
                  →
                </span>
              )}
              <span
                data-block={b.index}
                tabIndex={0}
                aria-label={
                  b.full
                    ? `блок ${b.index}, токены ${b.start}–${b.start + b.tokenIds.length - 1}, хэш ${b.hash}`
                    : `хвост из ${fmt.tokensGen(b.tokenIds.length)}, не хэшируется`
                }
                className={cn(
                  'grid gap-0.5 rounded-chip px-2 py-1 text-[12px] outline-offset-2',
                  !b.full && 'border border-dashed border-ink-3',
                )}
                style={b.full ? { background: `color-mix(in oklab, ${cssVar('prefill')} 14%, transparent)` } : undefined}
              >
                <span className="text-ink-3">блок {b.index}</span>
                <span className="font-mono text-ink">{b.full ? b.hash : `хвост ${b.tokenIds.length} ток.`}</span>
                <span className="text-[11.5px] text-ink-3 tabular-nums">
                  токены {b.start}–{b.start + b.tokenIds.length - 1}
                </span>
              </span>
            </li>
          ),
        )}
      </ol>
      {second?.full && (
        <p className="text-[13px] leading-snug text-ink-3">
          хэш блока 1 = H(хэш блока 0, токены 16–31): изменится хоть один токен в начале, и поменяются все хэши после него. Наведите на блок, чтобы увидеть
          его текст.
        </p>
      )}
      {tip.node}
    </div>
  )
}
