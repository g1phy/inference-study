/**
 * Инспектор симулятора, стадии «обратно»: postprocessor (детокенизация и парсеры) и стрим SSE.
 */
import { useEffect, useRef } from 'react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { ChapterLink, CodeBlock, cn, cssVar, fmt, Term, TokenChip } from '@/ui'
import type { FlowRunEx } from '@/engine/flow'
import type { ParserState } from '@/engine/types'
import { ChannelTag, GroupLabel, ReasoningText, ToolBox, WsText } from './bits'
import { lanesUpTo, sseChannel, sseSummary } from './model'

/* ============================================================================
 * postprocess
 * ========================================================================== */

const STATE_LABEL: Record<ParserState, string> = {
  content: 'content',
  reasoning: 'reasoning',
  tool_call: 'tool_call',
  maybe_tag: 'ждёт: тег или текст?',
}

function StateBadge({ state }: { state: ParserState }) {
  const swatch =
    state === 'reasoning' ? (
      <span aria-hidden className="h-2.5 w-2.5 rounded-[3px] border border-dashed border-ink-3" />
    ) : state === 'maybe_tag' ? (
      <span aria-hidden className="h-2.5 w-2.5 rounded-[3px] border border-ink-3" />
    ) : (
      <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: cssVar(state === 'tool_call' ? 'tool' : 'decode') }} />
    )
  return (
    <span className="inline-flex h-7 items-center gap-1.5 rounded-chip px-2 font-mono text-[12px] text-ink shadow-[inset_0_0_0_1px_var(--line-strong)]">
      {swatch}
      {STATE_LABEL[state]}
    </span>
  )
}

function tail(s: string, max = 520): string {
  return s.length > max ? `…${s.slice(s.length - max)}` : s
}

export function PostprocessStage({ run, k, reached }: { run: FlowRunEx; k: number; reached: boolean }) {
  const N = run.decode.length
  const i = Math.max(0, Math.min(N - 1, k))
  const det = run.detok[i]
  const ev = run.parser.find((e) => e.step === i)
  const flush = i === N - 1 ? run.parser.find((e) => e.step === N && e.deltas.length > 0) : undefined
  const lanes = lanesUpTo(run.parser, i === N - 1 ? N : i)
  const stop = det.stopped || run.decode[i].stop
  const held = det.heldBytes.length > 0
  const anyHeld = run.detok.some((x) => x.heldBytes.length > 0)
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5">
      {!reached && (
        <p className="rounded-control bg-surface-2 px-3 py-2 text-[13px] leading-snug text-ink-2 shadow-[inset_0_0_0_1px_var(--line)]">
          Плеер ещё не дошёл до генерации: показан первый токен.
        </p>
      )}
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 @2xl:grid-cols-2">
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
          <GroupLabel aside={`токен ${fmt.int(i + 1)} из ${fmt.int(N)}`}>Детокенизатор</GroupLabel>
          <div className="flex flex-wrap items-center gap-2">
            <TokenChip token={det.token} variant="generated" showId size="sm" />
            <ArrowRightIcon size={16} aria-hidden className="text-ink-3" />
            {stop ? (
              <span className="text-[13px] text-ink-2">стоп-токен: в текст не попадает, генерация закончена</span>
            ) : det.emitted ? (
              <span className="rounded-chip bg-surface-2 px-1.5 text-[13px] text-ink shadow-[inset_0_0_0_1px_var(--line)]">
                <WsText text={det.emitted} />
              </span>
            ) : (
              <span className="text-[13px] text-ink-2">ничего не отдано</span>
            )}
          </div>
          {held && (
            <p className="text-[12.5px] leading-snug text-ink-2">
              Придержаны байты <span className="font-mono text-ink">{fmt.hex(det.heldBytes)}</span>: последний символ ещё не дописан (в <Term t="UTF-8" /> он
              занимает несколько байтов), поэтому детокенизатор ждёт следующий токен
              {det.emitted ? ' и отдаёт только готовую часть.' : ' и пока ничего не отдаёт.'}
            </p>
          )}
          <p className="text-[12.5px] leading-snug text-ink-3">
            Текст отдаётся, только когда байты сложились в целые символы. Спецтокены вроде &lt;|im_end|&gt; скрыты (skip_special_tokens), а
            &lt;think&gt; и &lt;tool_call&gt; остаются: их ждут парсеры.
          </p>
          {!anyHeld && (
            <p className="text-[12.5px] leading-snug text-ink-3">
              В этом ответе каждый токен содержит целые символы, поэтому придержки не видно. Она появляется, когда символ режется на несколько токенов.
              Частые эмодзи вроде 🙂 лежат в словаре Qwen3 целым токеном, а редкие режутся на байты: во вкладке «Свой текст» спросите «Что значит 🫠?»,
              общий ответ повторит эмодзи, и детокенизатор будет ждать недостающие байты.
            </p>
          )}
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
          <GroupLabel>Парсер</GroupLabel>
          {ev ? (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <StateBadge state={ev.stateBefore} />
                <ArrowRightIcon size={16} aria-hidden className="text-ink-3" />
                <StateBadge state={ev.stateAfter} />
              </div>
              {ev.buffer && (
                <p className="text-[12.5px] leading-snug text-ink-2">
                  В буфере <span className="font-mono text-ink">«{ev.buffer}»</span>: пока неясно, начало это тега или обычный текст.
                </p>
              )}
              {ev.note && <p className="text-[13px] leading-snug text-ink">{ev.note}</p>}
              {ev.deltas.length > 0 && (
                <ul className="grid grid-cols-[minmax(0,1fr)] gap-1">
                  {ev.deltas.map((d, j) => (
                    <li key={j} className="flex flex-wrap items-center gap-2 text-[13px]">
                      <ChannelTag channel={d.channel} />
                      <WsText text={d.text} className="text-[12.5px]" />
                    </li>
                  ))}
                </ul>
              )}
              {flush && <p className="text-[12.5px] leading-snug text-ink-2">В конце потока парсер сбросил остаток буфера.</p>}
            </>
          ) : (
            <p className="text-[13px] text-ink-3">Событий парсера на этом шаге нет.</p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside="после этого шага">Дорожки ответа</GroupLabel>
        <div className="grid grid-cols-[minmax(0,1fr)] gap-3 @2xl:grid-cols-2">
          {(run.output?.reasoning || run.settings.thinking) && (
            <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-1">
              <ChannelTag channel="reasoning" />
              {lanes.reasoning.trim() ? <ReasoningText>{tail(lanes.reasoning.replace(/^\n+/, ''))}</ReasoningText> : <span className="text-[13px] text-ink-3">пусто</span>}
            </div>
          )}
          <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-1">
            <ChannelTag channel="content" />
            {lanes.content.trim() ? (
              <div className={cn('text-[14px] leading-relaxed whitespace-pre-wrap text-ink', run.settings.json && 'font-mono text-[12.5px]')}>{tail(lanes.content.replace(/^\n+/, ''))}</div>
            ) : (
              <span className="text-[13px] text-ink-3">пусто</span>
            )}
          </div>
          {(run.settings.tools || lanes.tool) && (
            <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-1">
              <ChannelTag channel="tool_call" />
              {lanes.tool.trim() ? <ToolBox>{tail(lanes.tool.trim())}</ToolBox> : <span className="text-[13px] text-ink-3">пусто</span>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* ============================================================================
 * stream (SSE)
 * ========================================================================== */

const CH_COLOR = { reasoning: 'reasoning', content: 'decode', tool: 'tool', meta: 'special' } as const

export function StreamStage({ run, sseCount }: { run: FlowRunEx; sseCount: number }) {
  const n = Math.max(0, Math.min(run.sse.length, sseCount))
  const start = Math.max(0, n - 40)
  const shown = run.sse.slice(start, n)
  const listRef = useRef<HTMLOListElement>(null)
  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [n])
  const last = n > 0 ? run.sse[n - 1] : null
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5 @2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
        <GroupLabel aside={`${fmt.int(n)} из ${fmt.int(run.sse.length)} ${fmt.plural(run.sse.length, ['строки', 'строк', 'строк'])}`}>
          Строки SSE у клиента
        </GroupLabel>
        {n === 0 ? (
          <p className="text-[13px] leading-snug text-ink-2">
            Пока пусто. Первый чанк придёт через TTFT = {fmt.ms(run.metrics.ttftMs)} с момента отправки запроса, включая сеть.
          </p>
        ) : (
          <ol
            ref={listRef}
            tabIndex={0}
            aria-label="Лента SSE-чанков"
            className="grid grid-cols-[minmax(0,1fr)] max-h-[300px] content-start gap-0.5 overflow-auto rounded-control bg-surface-2 px-2 py-1.5 font-mono text-[12px] leading-[1.6] shadow-[inset_0_0_0_1px_var(--line)] scrollbar-thin focus-visible:-outline-offset-2"
          >
            {start > 0 && <li className="text-ink-3">… ещё {fmt.int(start)} выше</li>}
            {shown.map((ch, j) => {
              const kind = sseChannel(ch)
              const idx = start + j
              return (
                <li key={idx} className="grid grid-cols-[2.5rem_10px_minmax(0,1fr)] items-baseline gap-1.5">
                  <span className="text-right text-ink-3 tabular-nums">{idx + 1}</span>
                  <span
                    aria-hidden
                    className={cn('h-2 w-2 self-center rounded-[2px]', (kind === 'reasoning' || kind === 'meta') && 'border border-dashed border-ink-3')}
                    style={{ background: kind === 'reasoning' || kind === 'meta' ? 'transparent' : cssVar(CH_COLOR[kind]) }}
                  />
                  <span className="break-words text-ink [overflow-wrap:anywhere]">{sseSummary(ch)}</span>
                </li>
              )
            })}
          </ol>
        )}
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
        <GroupLabel>Последний чанк целиком</GroupLabel>
        {last ? (
          <CodeBlock lang="text" code={last.line} wrap copy={false} title="text/event-stream" maxHeight={300} />
        ) : (
          <p className="text-[13px] text-ink-3">Ещё ничего не пришло.</p>
        )}
        <p className="text-[12.5px] leading-snug text-ink-3">
          Каждый чанк приходит строкой «data: …», за ней идёт пустая строка. Поле рассуждений называется reasoning_content, как в Dynamo и SGLang; в новых
          версиях vLLM оно переименовано в reasoning.
        </p>
        {run.output && run.output.toolCalls.length > 0 && (
          <p className="text-[12.5px] leading-snug text-ink-3">
            Аргументы вызова здесь приходят кусочками, как у hermes-парсера vLLM. Dynamo по умолчанию копит вызов в <Term t="Jail">jail</Term> и отдаёт его
            одним чанком с готовыми arguments (<ChapterLink to="parsers" />).
          </p>
        )}
      </div>
    </div>
  )
}
