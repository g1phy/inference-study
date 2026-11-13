/**
 * Инспектор симулятора, стадии «туда»: клиент, frontend, preprocessor, router.
 */
import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { ChapterLink, cn, CodeBlock, cssVar, fmt, Term, tint, TokenChip, TokenStrip, Tooltip, type TokenVariant } from '@/ui'
import { cachedBlockFlags, type FlowRunEx } from '@/engine/flow'
import type { SegmentKind, StageId, TemplateSegment } from '@/engine/types'
import { GroupLabel, KeyVals, Status } from './bits'
import { prettyJson } from './model'
import { STAGES } from './copy'

type SelectStage = (stage: StageId) => void

/** Кнопка-ссылка на стадию (подсвечивает узел и открывает его в инспекторе). */
function StageButton({ stage, onSelect }: { stage: StageId; onSelect: SelectStage }) {
  const c = STAGES[stage]
  const Icon = c.icon
  return (
    <button
      type="button"
      onClick={() => onSelect(stage)}
      className="inline-flex h-7 shrink-0 items-center gap-1.5 rounded-control px-2 text-[12.5px] font-medium text-ink shadow-[inset_0_0_0_1px_var(--line-strong)] transition-colors hover:bg-surface-3"
    >
      <Icon size={16} aria-hidden className="text-ink-2" />
      {c.title}
    </button>
  )
}

/* ============================================================================
 * client
 * ========================================================================== */

export const ClientStage = memo(function ClientStage({ run, onSelect }: { run: FlowRunEx; onSelect: SelectStage }) {
  const json = useMemo(() => prettyJson(run.request), [run.request])
  const r = run.request
  const rows: Array<{ field: string; stage: StageId; text: ReactNode }> = [
    { field: 'model', stage: 'route', text: 'запрос уйдёт в пул воркеров этой модели' },
    { field: 'messages', stage: 'preprocess', text: 'склеятся по шаблону чата в один текст' },
  ]
  if (r.tools) rows.push({ field: 'tools', stage: 'preprocess', text: 'описание попадёт в промпт, вызов разберёт парсер' })
  if (r.response_format) rows.push({ field: 'response_format', stage: 'decode', text: 'маска разрешённых токенов на каждом шаге' })
  rows.push(
    {
      field: 'temperature',
      stage: 'decode',
      text: (
        <>
          сэмплер делит <Term t="Logits">логиты</Term> на T перед <Term t="Softmax">softmax</Term>
        </>
      ),
    },
    { field: 'max_tokens', stage: 'frontend', text: 'стоп-условие и проверка бюджета контекста' },
    { field: 'stream', stage: 'stream', text: 'ответ придёт SSE-чанками, в конце usage' },
    {
      field: 'chat_template_kwargs',
      stage: 'preprocess',
      text: 'при enable_thinking: false шаблон сразу дописывает пустой блок <think></think>, и модель отвечает без рассуждения; при true шаблон ничего не добавляет, модель сама начинает с <think>',
    },
  )
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-4 @2xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
      <CodeBlock lang="json" code={json} title="POST /v1/chat/completions" maxHeight={360} wrap />
      <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
        <GroupLabel>Куда поедут поля</GroupLabel>
        <ul className="grid grid-cols-[minmax(0,1fr)] gap-2">
          {rows.map((row) => (
            <li key={row.field} className="grid grid-cols-[minmax(0,1fr)] gap-1">
              <div className="flex flex-wrap items-center gap-2">
                <code className="font-mono text-[12.5px] text-ink">{row.field}</code>
                <ArrowRightIcon size={16} aria-hidden className="text-ink-3" />
                <StageButton stage={row.stage} onSelect={onSelect} />
              </div>
              <p className="text-[13px] leading-snug text-ink-2">{row.text}</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
})

/* ============================================================================
 * frontend
 * ========================================================================== */

export const FrontendStage = memo(function FrontendStage({ run }: { run: FlowRunEx }) {
  const fe = run.frontend!
  const stopTokens = fe.stop.stopTokenIds.map((id, i) => `${id} ${fe.stop.stopTokenTexts[i] ?? ''}`.trim()).join(', ')
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5 @2xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
      <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
        <GroupLabel>Что проверено</GroupLabel>
        <ul className="grid grid-cols-[minmax(0,1fr)] gap-2">
          {fe.checks.map((c) => (
            <li key={c.label} className="grid grid-cols-[minmax(0,1fr)] gap-0.5">
              <Status tone={c.ok ? 'good' : 'bad'}>
                <span className="font-medium">{c.label}</span>
                <span className="text-ink-3">{c.ok ? 'в порядке' : 'ошибка'}</span>
              </Status>
              <p className="pl-5 text-[12.5px] leading-snug text-ink-2">{c.detail}</p>
            </li>
          ))}
        </ul>
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-4">
        <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
          <GroupLabel>Параметры сэмплера</GroupLabel>
          <KeyVals
            rows={[
              ['temperature', <SamplingValue value={fmt.num(fe.sampling.temperature, 2)} source="из запроса" />],
              [
                'top_p',
                <SamplingValue value={fmt.num(fe.sampling.topP, 2)} source={fe.samplingFromModel.includes('top_p') ? 'в запросе нет: из generation_config.json' : 'из запроса'} />,
              ],
              [
                'top_k',
                <SamplingValue value={fmt.int(fe.sampling.topK)} source={fe.samplingFromModel.includes('top_k') ? 'тоже из generation_config.json' : 'из запроса'} />,
              ],
              ['seed', fe.sampling.seed === null ? 'не задан' : String(fe.sampling.seed)],
            ]}
          />
          <p className="text-[12.5px] leading-snug text-ink-3">
            Чего нет в запросе, frontend передаёт пустым, а воркер vLLM подставляет из <Term t="generation_config.json" /> модели (по умолчанию --generation-config
            auto; у Qwen3 там T {fmt.num(fe.generationConfig.temperature, 1)}, top-p {fmt.num(fe.generationConfig.topP, 2)}, top-k{' '}
            {fmt.int(fe.generationConfig.topK)}). temperature задана в запросе, поэтому действует она. Симулятор применяет к кандидатам только
            temperature: top-p и top-k здесь не моделируются.
          </p>
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
          <GroupLabel>Когда остановиться</GroupLabel>
          <KeyVals
            rows={[
              ['max_tokens', fmt.int(fe.stop.maxTokens)],
              ['стоп-токены', stopTokens],
              ['стоп-строки', fe.stop.stopStrings.length ? fe.stop.stopStrings.join(', ') : 'нет'],
            ]}
          />
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
          <GroupLabel>Парсеры ответа</GroupLabel>
          <KeyVals
            rows={[
              ['reasoning', fe.parsers.reasoning],
              ['tool calls', run.settings.tools ? fe.parsers.tool : `${fe.parsers.tool} (инструментов нет, не нужен)`],
              ['enable_thinking', String(fe.chatTemplateKwargs.enable_thinking)],
            ]}
          />
        </div>
      </div>
      {fe.guidedJson && (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-2 @2xl:col-span-2">
          <GroupLabel aside="уйдёт в движок как sampling_options.guided_decoding.json">JSON-схема ответа</GroupLabel>
          <CodeBlock lang="json" code={prettyJson(fe.guidedJson)} maxHeight={220} copy={false} title="response_format.json_schema.schema" />
        </div>
      )}
    </div>
  )
})

/** Значение параметра сэмплера и откуда оно взялось. */
function SamplingValue({ value, source }: { value: string; source: string }) {
  return (
    <span>
      {value} <span className="font-sans text-[12px] text-ink-3">{source}</span>
    </span>
  )
}

/* ============================================================================
 * preprocess
 * ========================================================================== */

/** Кусок шаблона: спецтокены (по настоящей токенизации) в пунктирной рамке, переводы строк видимы. */
function SegText({
  seg,
  offset,
  specials,
  injected,
  active,
}: {
  seg: TemplateSegment
  offset: number
  specials: ReadonlyArray<[number, number]>
  /** Начала спецтокенов, которые пришли из текста сообщения (инъекция). */
  injected: ReadonlySet<number>
  active: boolean
}) {
  const base: Record<SegmentKind, string> = {
    special: '',
    role: 'font-semibold text-ink',
    content: 'text-ink',
    tools: 'text-ink',
    'gen-prompt': 'text-ink',
    format: 'text-ink-3',
  }
  const bg = seg.kind === 'content' ? tint('prefill', 13) : seg.kind === 'tools' ? tint('tool', 13) : undefined
  // режем кусок по границам спецтокенов, попавших в него
  const end = offset + seg.text.length
  const parts: Array<{ text: string; special: boolean; injected?: boolean }> = []
  let at = offset
  for (const [a, b] of specials) {
    if (b <= at || a >= end) continue
    if (a > at) parts.push({ text: seg.text.slice(at - offset, a - offset), special: false })
    parts.push({ text: seg.text.slice(Math.max(a, at) - offset, Math.min(b, end) - offset), special: true, injected: injected.has(a) })
    at = Math.min(b, end)
  }
  if (at < end) parts.push({ text: seg.text.slice(at - offset), special: false })
  return (
    <span
      className={cn(base[seg.kind], 'rounded-[3px] transition-shadow', active && 'shadow-[0_0_0_2px_var(--accent)]')}
      style={bg ? { background: bg } : undefined}
    >
      {parts.map((p, i) =>
        p.special ? (
          <span
            key={i}
            title={p.injected ? 'Инъекция: этот спецтокен пришёл из текста сообщения' : undefined}
            className={cn('rounded-[4px] border border-dashed px-0.5', p.injected ? 'text-ink' : 'border-special text-ink-2')}
            style={p.injected ? { borderColor: cssVar('warn'), background: tint('warn', 16) } : undefined}
          >
            {p.text}
          </span>
        ) : (
          // на подложке текста сообщений и tools ink-3 не дотягивает до AA: глиф «↵» там цветом ink-2
          <NewlineText key={i} text={p.text} glyphClass={bg ? 'text-ink-2' : 'text-ink-3'} />
        ),
      )}
    </span>
  )
}

function NewlineText({ text, glyphClass = 'text-ink-3' }: { text: string; glyphClass?: string }) {
  const lines = text.split('\n')
  return (
    <>
      {lines.map((l, i) => (
        <span key={i}>
          {l}
          {i < lines.length - 1 && (
            <>
              <span className={glyphClass}>↵</span>
              {'\n'}
            </>
          )}
        </span>
      ))}
    </>
  )
}

function LegendSwatch({ children, style, className }: { children: ReactNode; style?: React.CSSProperties; className?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden className={cn('h-3 w-4 shrink-0 rounded-[3px]', className)} style={style} />
      {children}
    </span>
  )
}

/** Сколько токенов промпта рисовать в ленте (длинный документ и длинный свой текст помещаются целиком). */
const TOKEN_LIMIT = 2000

/** Лента токенов в окне с прокруткой: внизу затухание и подсказка, сколько токенов ниже края. */
function TokenWindow({ children, contentKey, complete }: { children: ReactNode; contentKey: unknown; complete: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const [below, setBelow] = useState(0)
  const measure = useCallback(() => {
    const el = ref.current
    if (!el) return
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 4) {
      setBelow(0)
      return
    }
    const bottom = el.getBoundingClientRect().bottom - 2
    // чипы идут сверху вниз: первый чип ниже края ищем двоичным поиском (лента бывает на тысячи токенов)
    const chips = el.querySelectorAll<HTMLElement>('[data-ti]')
    let lo = 0
    let hi = chips.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (chips[mid].getBoundingClientRect().top >= bottom) hi = mid
      else lo = mid + 1
    }
    setBelow(chips.length - lo)
  }, [])
  useEffect(() => {
    const el = ref.current
    if (!el) return
    let raf = 0
    const onScroll = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(measure)
    }
    measure()
    el.addEventListener('scroll', onScroll, { passive: true })
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(onScroll) : null
    ro?.observe(el)
    return () => {
      cancelAnimationFrame(raf)
      el.removeEventListener('scroll', onScroll)
      ro?.disconnect()
    }
  }, [measure, contentKey])
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-1">
      <div
        ref={ref}
        className="max-h-[340px] overflow-auto rounded-control p-0.5 scrollbar-thin"
        style={below > 0 ? { maskImage: 'linear-gradient(#000 82%, transparent)', WebkitMaskImage: 'linear-gradient(#000 82%, transparent)' } : undefined}
      >
        {children}
      </div>
      {below > 0 && (
        <p className="text-[12.5px] leading-snug text-ink-3">
          Ещё {fmt.tokens(below)} ниже: прокрутите ленту.
          {complete && <> Там и {'<|im_start|>assistant'}, после которого модель начинает ответ.</>}
        </p>
      )}
    </div>
  )
}

export const PreprocessStage = memo(function PreprocessStage({ run }: { run: FlowRunEx }) {
  const [hoverTok, setHoverTok] = useState<number | null>(null)
  const pre = run.preprocess!
  const segOf = pre.segmentOfToken
  const hoverSeg = hoverTok !== null ? segOf[hoverTok] : null
  const kinds = run.template.segments.map((s) => s.kind)
  const variantOf = useMemo(
    () =>
      (_t: unknown, i: number): TokenVariant => {
        const k = kinds[segOf[i]]
        return k === 'content' ? 'prompt' : k === 'tools' || k === 'special' ? 'default' : 'muted'
      },
    [kinds, segOf],
  )
  const segStarts = useMemo(() => {
    const out: number[] = []
    let at = 0
    for (const sg of run.template.segments) {
      out.push(at)
      at += sg.text.length
    }
    return out
  }, [run.template.segments])
  const specials = useMemo(
    () => run.promptTokens.filter((t) => t.special && t.start !== undefined && t.end !== undefined).map((t) => [t.start!, t.end!] as [number, number]),
    [run.promptTokens],
  )
  const injected = run.injection
  const injectedStarts = useMemo(
    () => new Set(injected.map((x) => run.promptTokens[x.index]?.start).filter((v): v is number => v !== undefined)),
    [injected, run.promptTokens],
  )
  const injectedIdx = useMemo(() => injected.map((x) => x.index), [injected])
  const full = run.blocks.filter((b) => b.full).length
  const tail = run.blocks.find((b) => !b.full)?.tokenIds.length ?? 0
  const n = run.promptTokens.length
  const prepared = useMemo(() => {
    const p = { ...pre.preprocessedRequest } as Record<string, unknown>
    const ids = (p.token_ids as number[]) ?? []
    p.token_ids = '__IDS__'
    const shown = ids.length > 12 ? `[${ids.slice(0, 12).join(', ')}, …]` : `[${ids.join(', ')}]`
    return prettyJson(p).replace('"__IDS__"', shown)
  }, [pre.preprocessedRequest])

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5">
      {injected.length > 0 && (
        <div role="note" className="grid grid-cols-[minmax(0,1fr)] gap-2 rounded-control px-3 py-2.5" style={{ background: tint('warn', 10), boxShadow: `inset 0 0 0 1px ${tint('warn', 55)}` }}>
          <Status tone="warn" className="text-[13px] font-semibold">
            <Term t="инъекция спецтокенов">Инъекция спецтокенов</Term>
          </Status>
          {/* чипы на обычной подложке: номер под чипом цветом ink-3 на жёлтом фоне панели не дотягивает до AA */}
          <div className="flex flex-wrap items-center gap-1.5 justify-self-start rounded-control bg-surface px-2 py-1.5">
            {injected.map((x) => (
              <TokenChip key={x.index} token={run.promptTokens[x.index]} showId size="sm" />
            ))}
          </div>
          <p className="text-[13px] leading-snug text-ink-2">
            Эти литералы стоят в тексте сообщения, а токенизатор с настройками HF по умолчанию превратил их в настоящие служебные токены: модель
            увидит конец реплики, чужую роль или маркер рассуждения там, где их не было. split_special_tokens=True обезвреживает только special-токены
            вроде {'<|im_start|>'} и {'<|im_end|>'}; {'<think>'} и {'<tool_call>'} у Qwen3 не special и распознаются всё равно, поэтому текст
            пользователя нужно чистить (<ChapterLink to="tokenization" />).
          </p>
        </div>
      )}

      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside={`${fmt.int(pre.chars)} ${fmt.plural(pre.chars, ['символ', 'символа', 'символов'])}`}>
          Промпт по шаблону Qwen3
        </GroupLabel>
        <pre className="max-h-[220px] overflow-auto rounded-control bg-surface-2 px-3 py-2.5 font-mono text-[12.5px] leading-[1.75] break-words whitespace-pre-wrap text-ink shadow-[inset_0_0_0_1px_var(--line)] scrollbar-thin focus-visible:-outline-offset-2" tabIndex={0}>
          {run.template.segments.map((s, i) => (
            <SegText key={i} seg={s} offset={segStarts[i]} specials={specials} injected={injectedStarts} active={hoverSeg === i} />
          ))}
        </pre>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-ink-2">
          <LegendSwatch className="border border-dashed border-special">спецтокен</LegendSwatch>
          <LegendSwatch style={{ background: tint('prefill', 22) }}>текст сообщений</LegendSwatch>
          {run.settings.tools && <LegendSwatch style={{ background: tint('tool', 22) }}>описание tools</LegendSwatch>}
          {injected.length > 0 && (
            <LegendSwatch className="border border-dashed" style={{ borderColor: cssVar('warn'), background: tint('warn', 16) }}>
              спецтокен из текста (инъекция)
            </LegendSwatch>
          )}
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="font-mono text-ink-3">↵</span>перевод строки
          </span>
        </div>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside={`${fmt.tokens(n)}: ${full} ${fmt.plural(full, ['полный блок', 'полных блока', 'полных блоков'])} по ${run.blockSize}${tail ? ` и хвост ${tail}` : ''}`}>
          Токены (номера в словаре Qwen3)
        </GroupLabel>
        <TokenWindow contentKey={run.promptTokens} complete={run.promptTokens.length <= TOKEN_LIMIT}>
          <TokenStrip
            tokens={run.promptTokens}
            showIds
            variantOf={variantOf}
            onHoverIndex={setHoverTok}
            highlightIndex={injectedIdx.length ? injectedIdx : undefined}
            limit={TOKEN_LIMIT}
            size="sm"
          />
        </TokenWindow>
        <p className="text-[12.5px] leading-snug text-ink-3">
          Наведите на токен: в промпте подсветится кусок шаблона, из которого он получился.
          {pre.toolsTokens > 0 && <> Описание инструмента добавило {fmt.tokens(pre.toolsTokens)}.</>}
        </p>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside={`упрощённо; token_ids: ${fmt.int(n)} ${fmt.plural(n, ['число', 'числа', 'чисел'])}`}>Что уходит роутеру и воркеру</GroupLabel>
        <CodeBlock lang="json" code={prepared} title="PreprocessedRequest" maxHeight={200} copy={false} wrap />
      </div>
    </div>
  )
})

/* ============================================================================
 * route
 * ========================================================================== */

function BlocksRow({ run }: { run: FlowRunEx }) {
  const flags = cachedBlockFlags(run)
  const shown = run.blocks.slice(0, 24)
  const rest = run.blocks.length - shown.length
  const chosen = run.prefill.workerName ?? run.prefill.worker
  return (
    <div className="flex flex-wrap gap-1.5">
      {shown.map((b, i) => {
        const toks = run.promptTokens.slice(b.index * run.blockSize, b.index * run.blockSize + b.tokenIds.length)
        const status = !b.full ? 'tail' : flags[i] ? 'hit' : 'miss'
        return (
          <Tooltip
            key={b.index}
            wide
            content={
              <div className="grid grid-cols-[minmax(0,1fr)] gap-1.5">
                <div className="text-xs text-ink-3">
                  Блок {b.index + 1}: токены с {b.index * run.blockSize + 1} по {b.index * run.blockSize + b.tokenIds.length}
                </div>
                <TokenStrip tokens={toks} size="sm" tooltip={false} />
              </div>
            }
          >
            <span
              tabIndex={0}
              className={cn(
                'grid grid-cols-[minmax(0,1fr)] min-w-[92px] gap-0.5 rounded-control px-2 py-1.5 text-left',
                status === 'tail' ? 'border border-dashed border-line-strong' : 'shadow-[inset_0_0_0_1px_var(--line-strong)]',
              )}
              style={status === 'hit' ? { background: tint('good', 12) } : undefined}
            >
              <span className="font-mono text-[12px] leading-4 text-ink">{b.full ? b.hash : 'без хэша'}</span>
              {/* на зелёной подложке попадания ink-3 даёт 4,45:1: подпись цветом ink-2 */}
              <span className={cn('text-[11.5px] leading-4', status === 'hit' ? 'text-ink-2' : 'text-ink-3')}>
                №{b.index + 1}, {b.tokenIds.length} ток.
              </span>
              {status === 'hit' ? (
                <Status tone="good" className="text-[11.5px]">
                  в кэше
                </Status>
              ) : status === 'miss' ? (
                <Status tone="bad" className="text-[11.5px]">
                  нет в кэше
                </Status>
              ) : (
                <span className="text-[11.5px] leading-5 text-ink-2">хвост</span>
              )}
            </span>
          </Tooltip>
        )
      })}
      {rest > 0 && <span className="self-center px-1 text-[12.5px] text-ink-3">ещё {rest}</span>}
      <span className="sr-only">Кэш проверен у выбранного воркера {chosen}.</span>
    </div>
  )
}

function WorkerTable({ run, pool }: { run: FlowRunEx; pool: 'main' | 'decode' }) {
  const decision = pool === 'main' ? run.routing : run.decodeRouting!
  const infos = run.cluster?.workers ?? []
  const decodeOnly = pool === 'decode'
  const prefillOnly = pool === 'main' && run.settings.deployment === 'disagg'
  return (
    <div className="overflow-x-auto scrollbar-thin">
      <table className="w-full border-collapse text-[13px] leading-5 @lg:min-w-[520px]">
        <thead>
          <tr className="border-b border-line text-left text-[12px] text-ink-3">
            <th scope="col" className="py-1.5 pr-3 font-medium">Воркер</th>
            {!decodeOnly && <th scope="col" className="hidden px-2 py-1.5 text-right font-medium @lg:table-cell">Совпало блоков</th>}
            {!decodeOnly && <th scope="col" className="hidden px-2 py-1.5 text-right font-medium @lg:table-cell">Prefill</th>}
            {!prefillOnly && <th scope="col" className="hidden px-2 py-1.5 text-right font-medium @lg:table-cell">Decode</th>}
            <th scope="col" className="px-2 py-1.5 text-right font-medium">Стоимость</th>
            <th scope="col" className="hidden py-1.5 pl-2 text-right font-medium @lg:table-cell">Выбор</th>
          </tr>
        </thead>
        <tbody>
          {decision.scores.map((s) => {
            const info = infos.find((w) => w.id === s.workerId)
            const chosen = s.workerId === decision.chosen
            const prefillV = fmt.num(s.prefillTerm ?? s.prefillBlocks, 1)
            const decodeV = fmt.num(s.decodeTerm ?? s.decodeBlocks, 0)
            const compact = [
              !decodeOnly ? `совпало ${s.overlapBlocks}` : null,
              !decodeOnly ? `prefill ${prefillV}` : null,
              !prefillOnly ? `decode ${decodeV}` : null,
            ].filter(Boolean).join(', ')
            const cached = info && !decodeOnly ? info.cached : []
            return (
              <tr key={s.workerId} className="border-b border-line last:border-b-0" style={chosen ? { background: 'var(--accent-soft)' } : undefined}>
                <th scope="row" className="py-1.5 pr-3 text-left align-top font-normal">
                  <div className="font-medium text-ink">{s.name}</div>
                  {!decodeOnly && (
                    <div className={cn('text-[12px] leading-4', chosen ? 'text-ink-2' : 'text-ink-3')}>
                      {s.queuedPrefillTokens > 0 ? `в очереди prefill ${fmt.tokens(s.queuedPrefillTokens)}` : 'очередь prefill пуста'}
                    </div>
                  )}
                  {cached.length > 0 && (
                    <div className={cn('text-[12px] leading-4', chosen ? 'text-ink-2' : 'text-ink-3')}>
                      в кэше:{' '}
                      {cached.map((c, i) => (
                        <span key={i} className={cn(c.matchesPrompt && 'font-medium text-ink')}>
                          {i > 0 && '; '}
                          {c.label}, {fmt.int(c.blocks)}{'\u00a0'}бл.
                        </span>
                      ))}
                    </div>
                  )}
                  {decodeOnly && (
                    <div className={cn('text-[12px] leading-4', chosen ? 'text-ink-2' : 'text-ink-3')}>
                      {fmt.int(s.activeRequests)} {fmt.plural(s.activeRequests, ['запрос', 'запроса', 'запросов'])} в работе
                    </div>
                  )}
                  <div className="text-[12px] leading-4 text-ink-2 @lg:hidden">{compact}</div>
                  {chosen && (
                    <div className="mt-0.5 @lg:hidden">
                      <Status tone="good">выбран</Status>
                    </div>
                  )}
                </th>
                {!decodeOnly && <td className="hidden px-2 py-1.5 text-right align-top tabular-nums @lg:table-cell">{s.overlapBlocks}</td>}
                {!decodeOnly && <td className="hidden px-2 py-1.5 text-right align-top tabular-nums @lg:table-cell">{prefillV}</td>}
                {!prefillOnly && <td className="hidden px-2 py-1.5 text-right align-top tabular-nums @lg:table-cell">{decodeV}</td>}
                <td className="px-2 py-1.5 text-right align-top font-medium tabular-nums">{fmt.num(s.cost, 1)}</td>
                <td className="hidden py-1.5 pl-2 text-right align-top whitespace-nowrap @lg:table-cell">
                  {chosen ? <Status tone="good">выбран</Status> : <span className="text-ink-3">нет</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export const RouteStage = memo(function RouteStage({ run }: { run: FlowRunEx }) {
  const disagg = run.settings.deployment === 'disagg'
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5">
      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside={`по ${run.blockSize} токенов, хэш цепочкой`}>Блоки промпта</GroupLabel>
        <BlocksRow run={run} />
        <p className="text-[12.5px] leading-snug text-ink-3">
          Хэш блока зависит от его токенов и от хэша предыдущего блока, поэтому совпасть может только начало промпта. Неполный хвост не
          хэшируется. Отметки «в кэше» для выбранного воркера. Хэши здесь для показа: Dynamo считает XXH3, принцип тот же.
        </p>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside="температура роутера 0: всегда берётся воркер с минимальной стоимостью">
          {disagg ? 'Пул prefill-воркеров (PrefillRouter)' : 'Воркеры и их стоимость'}
        </GroupLabel>
        <WorkerTable run={run} pool="main" />
        <p className="text-[12.5px] leading-snug text-ink-2">
          Стоимость в блоках по 16 токенов: prefill = (очередь + промпт) / 16 − совпавшие блоки
          {disagg ? '; нагрузку decode prefill-роутер не считает.' : '; decode = активные блоки воркера + новые блоки запроса.'} Так выбирает
          KV-aware режим (в Dynamo флаг --router-mode kv; о содержимом кэшей он узнаёт из <Term t="KV events" /> воркеров). По умолчанию роутер
          Dynamo раздаёт запросы по кругу (round-robin) и на кэш не смотрит.
        </p>
        <p className="text-[13.5px] leading-relaxed text-ink">{run.routing.explanation}</p>
      </div>

      {disagg && run.decodeRouting && (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
          <GroupLabel aside="второй шаг роутера, после prefill">Пул decode-воркеров</GroupLabel>
          <WorkerTable run={run} pool="decode" />
          <p className="text-[13.5px] leading-relaxed text-ink">{run.decodeRouting.explanation}</p>
        </div>
      )}
    </div>
  )
})
