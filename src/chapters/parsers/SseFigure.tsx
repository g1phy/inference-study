import { useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { CodeBlock, Figure, Segmented, Toggle, TokenStripSkeleton, cn, fmt } from '@/ui'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { runStreamingParser, toSseChunks } from '@/engine/parsers'
import type { SseChunk } from '@/engine/types'
import { SCENARIOS, buildStream, compactLine, promptTokens, sseKind, type ScenarioId, type SseKind } from './scenarios'

type Field = 'reasoning_content' | 'reasoning'

const KIND_TEXT: Record<SseKind, { title: string; body: string }> = {
  role: {
    title: 'Первый чанк: роль',
    body: 'role: "assistant" и пустой content. Клиент заводит новое сообщение и дальше только дописывает в него.',
  },
  reasoning: {
    title: 'Кусок рассуждения',
    body: 'Поле нестандартное: в Chat Completions у OpenAI его нет. Dynamo, SGLang и DeepSeek API называют его reasoning_content, новые версии vLLM называют reasoning. Клиенту безопаснее читать оба.',
  },
  content: {
    title: 'Кусок ответа',
    body: 'Обычный текст ответа. Клиент дописывает его в content. Будет ли в начале перевод строки после </think>, зависит от версии парсера.',
  },
  'tool-head': {
    title: 'Начало вызова инструмента',
    body: 'index, id, type и имя функции. Приходит один раз, как только парсер увидел закрытое значение "name".',
  },
  'tool-args': {
    title: 'Кусок аргументов',
    body: 'Фрагмент строки function.arguments. Сам по себе он не JSON: клиент склеивает куски по index и разбирает строку целиком только в конце.',
  },
  finish: {
    title: 'Финальный чанк',
    // текст зависит от finish_reason чанка: см. finishText
    body: '',
  },
  usage: {
    title: 'Счётчики токенов',
    // текст зависит от сценария: см. usageText
    body: '',
  },
  done: {
    title: 'Конец потока',
    body: 'Строка data: [DONE]. Это не JSON: клиент должен проверить её до json.loads и закрыть поток.',
  },
}

/** Пояснение к финальному чанку по его finish_reason. */
function finishText(reason: unknown): string {
  if (reason === 'tool_calls') {
    return 'Пустой delta и finish_reason "tool_calls": модель вызвала инструмент. Клиент выполняет вызов и присылает результат следующим запросом. Без вызова здесь было бы "stop", а при упоре в max_tokens "length".'
  }
  if (reason === 'length') {
    return 'Пустой delta и finish_reason "length": генерация упёрлась в max_tokens, ответ оборван.'
  }
  return 'Пустой delta и finish_reason "stop": модель сама закончила ответ стоп-токеном <|im_end|>. Будь в ответе вызов инструмента, здесь стояло бы "tool_calls", а при упоре в max_tokens "length".'
}

/** Пояснение к чанку usage: откуда взялись числа (оба посчитаны, не условные). */
function usageText(id: ScenarioId, prompt: { total: number; tools: number }): string {
  const sc = SCENARIOS.find((s) => s.id === id)!
  return (
    'Приходит, только если в запросе stream_options.include_usage = true. choices у этого чанка пустой. Если поток оборвался, его может не быть. ' +
    `Число prompt_tokens посчитано настоящим шаблоном и словарём Qwen3 для вопроса «${sc.question}», completion_tokens равно числу сгенерированных токенов вместе с <|im_end|>.` +
    (prompt.tools > 0
      ? ` ${fmt.tokens(prompt.tools)} из ${fmt.int(prompt.total)} добавляет блок «# Tools»: когда в запросе есть tools, шаблон вписывает в системную часть промпта инструкцию и описание kv_cache_size, и так в каждом запросе.`
      : '')
  )
}

/** finish_reason чанка (или undefined, если чанк не финальный). */
function finishOf(json: Record<string, unknown> | null): unknown {
  const choices = json?.choices as Array<Record<string, unknown>> | undefined
  return choices && choices.length > 0 ? choices[0].finish_reason : undefined
}

/**
 * Чанк для показа целиком: верхний уровень и choices[0] по строкам, delta и usage одной строкой,
 * чтобы то, чем чанки отличаются друг от друга, было видно без прокрутки. Возвращает текст и номер строки с delta/usage.
 */
function prettyChunk(json: Record<string, unknown>): { code: string; focus: number[] } {
  const lines: string[] = ['{']
  const focus: number[] = []
  const entries = Object.entries(json)
  entries.forEach(([k, v], i) => {
    const comma = i < entries.length - 1 ? ',' : ''
    if (k === 'choices' && Array.isArray(v) && v.length > 0) {
      lines.push('  "choices": [{')
      const ch = Object.entries(v[0] as Record<string, unknown>)
      ch.forEach(([ck, cv], j) => {
        const c2 = j < ch.length - 1 ? ',' : ''
        lines.push(`    ${JSON.stringify(ck)}: ${spaced(cv)}${c2}`)
        if (ck === 'delta' || (ck === 'finish_reason' && cv !== null) || ck === 'stop_reason') focus.push(lines.length)
      })
      lines.push(`  }]${comma}`)
    } else {
      lines.push(`  ${JSON.stringify(k)}: ${spaced(v)}${comma}`)
      if (k === 'usage') focus.push(lines.length)
    }
  })
  lines.push('}')
  return { code: lines.join('\n'), focus }
}

/** JSON в одну строку с пробелами после двоеточий и запятых (как обычно печатают в доках). */
function spaced(v: unknown): string {
  if (Array.isArray(v)) return '[' + v.map(spaced).join(', ') + ']'
  if (v !== null && typeof v === 'object') {
    return '{' + Object.entries(v as Record<string, unknown>).map(([k, x]) => `${JSON.stringify(k)}: ${spaced(x)}`).join(', ') + '}'
  }
  return JSON.stringify(v)
}

export function SseFigure() {
  const { tokenizer, status } = useTokenizer()
  const [scenario, setScenario] = useState<ScenarioId>('tool')
  const [field, setField] = useState<Field>('reasoning_content')
  const [usage, setUsage] = useState(true)
  const [full, setFull] = useState(false)
  const [picked, setPicked] = useState(0)
  const btns = useRef<Array<HTMLButtonElement | null>>([])

  // длина промпта по настоящему шаблону Qwen3: в сценарии с вызовом в промпт входит и описание инструмента
  const prompt = useMemo(() => (tokenizer ? promptTokens(tokenizer, scenario) : null), [tokenizer, scenario])

  const lines = useMemo<SseChunk[] | null>(() => {
    if (!tokenizer || !prompt) return null
    const sc = SCENARIOS.find((s) => s.id === scenario)!
    const stream = buildStream(tokenizer, sc.text, 'tokens')
    const events = runStreamingParser(stream.chunks, { reasoning: 'qwen3', tools: 'hermes', idSeed: scenario })
    const completion = stream.chunks.length + (stream.eos ? 1 : 0)
    return toSseChunks(events, {
      id: 'chatcmpl-7f3a',
      created: 1762950000,
      model: 'Qwen/Qwen3-8B',
      reasoningField: field,
      includeUsage: usage,
      usage: { promptTokens: prompt.total, completionTokens: completion },
      // vLLM добавляет в финальный чанк своё поле stop_reason (null: остановка по EOS)
      ...(field === 'reasoning' ? { stopReason: null } : {}),
    })
  }, [tokenizer, prompt, scenario, field, usage])

  const sel = lines ? Math.min(picked, lines.length - 1) : 0
  const kind = lines ? sseKind(lines[sel].json) : 'role'
  const assembled = useMemo(() => (lines ? assemble(lines, sel) : null), [lines, sel])
  const pretty = useMemo(() => (lines && lines[sel].json ? prettyChunk(lines[sel].json) : null), [lines, sel])
  const body =
    kind === 'finish' && lines ? finishText(finishOf(lines[sel].json)) : kind === 'usage' && prompt ? usageText(scenario, prompt) : KIND_TEXT[kind].body

  // Список строк: одна остановка Tab, дальше стрелки, Home и End (roving tabindex)
  const onListKey = (e: KeyboardEvent<HTMLOListElement>) => {
    if (!lines) return
    const last = lines.length - 1
    const to =
      e.key === 'ArrowDown' ? Math.min(last, sel + 1) : e.key === 'ArrowUp' ? Math.max(0, sel - 1) : e.key === 'Home' ? 0 : e.key === 'End' ? last : null
    if (to === null) return
    e.preventDefault()
    setPicked(to)
    btns.current[to]?.focus()
  }

  return (
    <Figure
      id="parsers-figure-sse"
      title="Анатомия SSE-ответа: от первого чанка до [DONE]"
      width="full"
      caption="Попробуйте: нажимайте на строки потока. В пояснении видно, что значит строка и что клиент успел собрать к этому моменту. Переключите имя поля рассуждений и выключите include_usage: исчезнет предпоследняя строка. Чанки собраны движком сайта по коду vLLM, это не запись живого сервера."
    >
      <div className="grid gap-5">
        <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
          <Segmented<ScenarioId> label="Сценарий" value={scenario} onChange={setScenario} wrap options={SCENARIOS.map((s) => ({ value: s.id, label: s.label }))} />
          <Segmented<Field>
            label="Имя поля рассуждений"
            value={field}
            onChange={setField}
            wrap
            options={[
              { value: 'reasoning_content', label: 'reasoning_content: Dynamo, SGLang' },
              { value: 'reasoning', label: 'reasoning: vLLM' },
            ]}
          />
          <Toggle label="include_usage" checked={usage} onChange={setUsage} />
          <Toggle label="Служебные поля целиком" checked={full} onChange={setFull} />
        </div>

        {!lines && status !== 'error' && <TokenStripSkeleton count={24} />}
        {lines && assembled && (
          <div className="grid gap-4 @4xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
            <div className="grid min-w-0 content-start gap-2">
              <div className="flex items-baseline justify-between gap-3 text-sm text-ink-2">
                <span className="font-medium">Так поток выглядит на проводе</span>
                <span className="text-ink-3 tabular-nums">{fmt.count(lines.length, ['строка', 'строки', 'строк'])}</span>
              </div>
              <ol
                className="max-h-[460px] overflow-auto rounded-control bg-surface-2 p-1.5 scrollbar-thin"
                aria-label="Строки SSE-потока: стрелки вверх и вниз выбирают строку"
                onKeyDown={onListKey}
              >
                {lines.map((l, i) => {
                  const k = sseKind(l.json)
                  return (
                    <li key={i}>
                      <button
                        type="button"
                        ref={(el) => {
                          btns.current[i] = el
                        }}
                        tabIndex={i === sel ? 0 : -1}
                        onClick={() => setPicked(i)}
                        aria-pressed={i === sel}
                        className={cn(
                          'grid w-full grid-cols-[2.25rem_minmax(0,1fr)] items-start gap-2 rounded-[6px] px-1.5 py-1 text-left focus-visible:-outline-offset-2',
                          i === sel ? 'bg-accent-soft' : 'hover:bg-surface-3/70',
                        )}
                      >
                        <span className={cn('pt-px text-right font-mono text-[11px] tabular-nums', i === sel ? 'text-ink' : 'text-ink-3')}>{i + 1}</span>
                        <span className={cn('font-mono text-[12px] leading-relaxed text-ink', full ? 'break-all' : 'break-words')}>
                          <KindMark kind={k} />
                          {full ? l.line : compactLine(l.json)}
                          <span className="block h-[0.9em] text-[10px] leading-none text-ink-2" aria-hidden>
                            ↵
                          </span>
                        </span>
                      </button>
                    </li>
                  )
                })}
              </ol>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-ink-2" aria-label="Обозначения строк">
                <span className="inline-flex items-center">
                  <KindMark kind="reasoning" />
                  рассуждение
                </span>
                <span className="inline-flex items-center">
                  <KindMark kind="content" />
                  content
                </span>
                <span className="inline-flex items-center">
                  <KindMark kind="tool-args" />
                  tool_calls
                </span>
                <span className="inline-flex items-center">
                  <KindMark kind="role" />
                  служебные
                </span>
              </div>
              <p className="text-[12.5px] text-ink-3">После каждой строки идёт пустая строка (↵): так в SSE разделяются события.</p>
            </div>

            <div className="grid min-w-0 content-start gap-4">
              <div className="grid gap-2 rounded-panel bg-surface-2/60 px-4 py-3.5">
                <div className="text-sm text-ink-3 tabular-nums">Строка {sel + 1}</div>
                <div className="text-[15px] font-semibold text-ink">{KIND_TEXT[kind].title}</div>
                <p className="text-[14px] leading-relaxed text-ink-2">{body}</p>
              </div>
              {pretty && <CodeBlock lang="json" title="этот чанк целиком" code={pretty.code} highlightLines={pretty.focus} wrap maxHeight={320} />}
              <AssembledView a={assembled} field={field} />
            </div>
          </div>
        )}
      </div>
    </Figure>
  )
}

function KindMark({ kind }: { kind: SseKind }) {
  const color =
    kind === 'reasoning' ? 'var(--c-reasoning)' : kind === 'tool-head' || kind === 'tool-args' ? 'var(--c-tool)' : kind === 'content' ? 'var(--c-decode)' : null
  if (!color) return <span aria-hidden className="mr-1.5 inline-block h-2 w-2 rounded-full border border-line-strong align-middle" />
  return (
    <span
      aria-hidden
      className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
      style={kind === 'reasoning' ? { border: `1.5px dashed ${color}` } : { background: color }}
    />
  )
}

interface Assembled {
  reasoning: string
  content: string
  calls: Array<{ id: string; name: string; arguments: string }>
  finish: string | null
  usage: Record<string, unknown> | null
  done: boolean
}

/** Что клиент собрал из строк 0..upto (так же, как псевдокод в тексте главы). */
function assemble(lines: SseChunk[], upto: number): Assembled {
  const a: Assembled = { reasoning: '', content: '', calls: [], finish: null, usage: null, done: false }
  for (const l of lines.slice(0, upto + 1)) {
    if (!l.json) {
      a.done = true
      continue
    }
    const choices = l.json.choices as Array<Record<string, unknown>>
    if (!choices.length) {
      a.usage = (l.json.usage as Record<string, unknown>) ?? null
      continue
    }
    const c = choices[0]
    const d = (c.delta ?? {}) as Record<string, unknown>
    a.reasoning += ((d.reasoning ?? d.reasoning_content ?? '') as string) || ''
    a.content += ((d.content ?? '') as string) || ''
    for (const tc of (d.tool_calls as Array<Record<string, unknown>> | undefined) ?? []) {
      const i = tc.index as number
      const fn = (tc.function ?? {}) as Record<string, unknown>
      const slot = (a.calls[i] ??= { id: '', name: '', arguments: '' })
      if (typeof tc.id === 'string') slot.id = tc.id
      if (typeof fn.name === 'string') slot.name += fn.name
      if (typeof fn.arguments === 'string') slot.arguments += fn.arguments
    }
    if (c.finish_reason) a.finish = c.finish_reason as string
  }
  return a
}

function AssembledView({ a, field }: { a: Assembled; field: Field }) {
  let argsValid: boolean | null = null
  if (a.calls[0]) {
    try {
      JSON.parse(a.calls[0].arguments)
      argsValid = true
    } catch {
      argsValid = false
    }
  }
  const rows: Array<[string, string]> = [
    [field, a.reasoning ? JSON.stringify(a.reasoning) : '""'],
    ['content', JSON.stringify(a.content)],
  ]
  if (a.calls.length) {
    rows.push(['tool_calls[0].name', JSON.stringify(a.calls[0].name)])
    rows.push(['tool_calls[0].arguments', JSON.stringify(a.calls[0].arguments)])
  }
  rows.push(['finish_reason', a.finish ? JSON.stringify(a.finish) : 'ещё нет'])
  if (a.usage) rows.push(['usage', JSON.stringify(a.usage)])
  return (
    <div className="grid gap-2">
      <div className="text-sm font-medium text-ink-2">Клиент собрал к этой строке</div>
      <dl className="grid gap-1.5 rounded-control bg-surface px-3 py-2.5 shadow-[inset_0_0_0_1px_var(--line)]">
        {rows.map(([k, v]) => (
          <div key={k} className="grid gap-x-3 @md:grid-cols-[minmax(0,11rem)_minmax(0,1fr)]">
            <dt className="font-mono text-[12px] text-ink-3">{k}</dt>
            <dd className="font-mono text-[12.5px] break-words text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      {argsValid !== null && (
        <p className="text-[13px] text-ink-2">
          {argsValid
            ? 'Строка arguments уже разбирается как JSON: вызов можно выполнять.'
            : 'Строка arguments пока не JSON: разбирать её рано, ждём следующих кусков.'}
          {a.done ? ' Поток закрыт.' : ''}
        </p>
      )}
    </div>
  )
}
