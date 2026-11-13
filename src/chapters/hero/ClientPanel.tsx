/**
 * «Что видит клиент»: ответ, собранный из SSE-чанков, которые уже дошли до клиента на текущем шаге плеера.
 * Рядом честная подпись: ответ модели заготовлен, всё остальное посчитано.
 */
import { useEffect, useMemo, useRef } from 'react'
import { InfoIcon } from '@phosphor-icons/react'
import { cn, cssVar, fmt, Skeleton } from '@/ui'
import type { FlowRunEx } from '@/engine/flow'
import { ChannelTag, ReasoningText, ToolBox } from './bits'
import { clientView } from './model'

export function ClientPanel({ run, step }: { run: FlowRunEx | null; step: number }) {
  const cur = run?.playback?.[step]
  const sseCount = cur?.sseCount ?? 0
  const view = useMemo(() => (run ? clientView(run.sse, sseCount) : null), [run, sseCount])
  const scrollRef = useRef<HTMLDivElement>(null)
  const len = (view?.reasoning.length ?? 0) + (view?.content.length ?? 0) + (view?.tools.reduce((s, t) => s + t.args.length, 0) ?? 0)
  const chunks = view?.chunks ?? 0
  const done = view?.done ?? false
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    // держим конец текста в поле зрения, пока читатель сам не прокрутил вверх; финальные finish_reason, usage и [DONE]
    // длину текста не меняют, поэтому следим и за числом чанков
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 120) el.scrollTop = el.scrollHeight
  }, [len, chunks, done])

  const streaming = cur?.kind === 'token'
  const content = view?.content.replace(/^\n+/, '') ?? ''
  const isJson = !!run?.settings.json

  return (
    <aside aria-label="Что видит клиент" className="grid grid-cols-[minmax(0,1fr)] min-w-0 content-start gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-[15px] leading-6 font-semibold text-ink">Что видит клиент</h3>
        {run && (
          <span className="text-[12.5px] text-ink-3 tabular-nums">
            t = {fmt.ms(cur?.tMs ?? 0)}, {fmt.count(chunks, ['чанк', 'чанка', 'чанков'])}
            {done && ' и [DONE]'}
          </span>
        )}
      </div>

      {!run ? (
        <Skeleton lines={5} height={12} label="Загружаем токенизатор" />
      ) : (
        <div ref={scrollRef} className="max-h-[360px] min-h-[120px] overflow-auto pr-1 scrollbar-thin">
          {sseCount === 0 ? (
            <p className="text-[13.5px] leading-relaxed text-ink-2">
              Пока ничего: запрос ещё в пути. Первый чанк придёт через {fmt.ms(run.metrics.ttftMs)} после отправки.
            </p>
          ) : (
            <div className="grid grid-cols-[minmax(0,1fr)] gap-3">
              {(view!.reasoning || run.settings.thinking) && view!.reasoning && (
                <div className="grid grid-cols-[minmax(0,1fr)] gap-1">
                  <ChannelTag channel="reasoning" />
                  <ReasoningText>{view!.reasoning.trim()}</ReasoningText>
                </div>
              )}
              {content.trim() && (
                <div className="grid grid-cols-[minmax(0,1fr)] gap-1">
                  <ChannelTag channel="content" />
                  <div className={cn('text-[14.5px] leading-relaxed whitespace-pre-wrap text-ink', isJson && 'font-mono text-[13px]')}>
                    {content}
                    {streaming && <Caret />}
                  </div>
                </div>
              )}
              {view!.tools.map((t) => (
                <div key={t.index} className="grid grid-cols-[minmax(0,1fr)] gap-1">
                  <ChannelTag channel="tool" />
                  <ToolBox>
                    <div className="font-semibold">{t.name ?? '…'}</div>
                    <div className="text-ink-2">{t.args || '{'}</div>
                  </ToolBox>
                </div>
              ))}
              {view!.finishReason === 'tool_calls' && (
                <p className="text-[13px] leading-relaxed text-ink-2">
                  Модель не выполняет функцию: сервер вернул только её имя и аргументы. Дальше ваше приложение само вызывает{' '}
                  {view!.tools[0]?.name ?? 'функцию'}, добавляет в историю сообщение с <code className="font-mono text-[12.5px]">role: "tool"</code> и
                  результатом и отправляет второй запрос. Ответ модель напишет уже в нём, а начало второго промпта совпадёт с первым, так что prefix cache
                  сэкономит его prefill.
                </p>
              )}
              {!content.trim() && !view!.reasoning && view!.tools.length === 0 && (
                <p className="text-[13.5px] leading-relaxed text-ink-2">
                  Пришёл первый чанк с ролью assistant. Текста в нём ещё нет: {run.settings.thinking ? 'токен <think> парсер клиенту не отдаёт.' : 'текст будет в следующих чанках.'}
                </p>
              )}
              {view!.finishReason && (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-2 text-[12.5px] text-ink-2">
                  <span>
                    finish_reason: <span className="font-mono text-ink">{view!.finishReason}</span>
                  </span>
                  {view!.usage && (
                    <span className="tabular-nums">
                      usage: {fmt.int(view!.usage.prompt_tokens)} + {fmt.int(view!.usage.completion_tokens)} токенов
                    </span>
                  )}
                  {view!.done && <span className="font-mono text-ink-3">[DONE]</span>}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {run && (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-2 border-t border-line pt-3">
          <p className="flex gap-2 text-[12.5px] leading-snug text-ink-2">
            <InfoIcon size={16} aria-hidden className="mt-px shrink-0 text-ink-3" />
            <span>{run.simulatedNote}</span>
          </p>
          {(run.notes ?? []).slice(0, 4).map((n, i) => (
            <p key={i} className="pl-6 text-[12px] leading-snug text-ink-3">
              {n}
            </p>
          ))}
        </div>
      )}
    </aside>
  )
}

/** Курсор конца потока: статичный, без мигания (DESIGN: анимация только когда объясняет). */
function Caret() {
  return <span aria-hidden className="ml-0.5 inline-block h-[1.05em] w-[3px] translate-y-[0.15em] rounded-[1px]" style={{ background: cssVar('decode') }} />
}
