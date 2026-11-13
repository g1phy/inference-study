/**
 * Ряд метрик прогона: TTFT, ITL, E2E, скорость, токены, finish_reason. Значения посчитаны движком
 * (perf.ts + условные константы сети и CPU-стадий), обновляются сразу при смене настроек.
 */
import { memo } from 'react'
import { fmt, Skeleton, Stat, Term } from '@/ui'
import type { FlowRunEx } from '@/engine/flow'
import { neighborSteps, neighborStepsText } from './model'

/** «18 мс» → ['18', 'мс'] для Stat (значение и единица отдельно). */
function split(s: string): [string, string] {
  const i = s.lastIndexOf(' ')
  return i > 0 ? [s.slice(0, i), s.slice(i + 1)] : [s, '']
}

const FINISH_HINT: Record<string, string> = {
  stop: 'модель выдала стоп-токен',
  tool_calls: 'модель вызвала инструмент',
  length: 'генерация упёрлась в max_tokens',
}

export const Metrics = memo(function Metrics({ run }: { run: FlowRunEx | null }) {
  if (!run) {
    return (
      <div role="status" aria-label="Метрики считаются" className="grid grid-cols-2 gap-x-6 gap-y-5 @xl:grid-cols-4">
        {Array.from({ length: 8 }, (_, i) => (
          <div key={i} className="grid grid-cols-[minmax(0,1fr)] gap-1.5">
            <Skeleton width={72} height={12} label="" />
            <Skeleton width={96} height={26} label="" />
          </div>
        ))}
      </div>
    )
  }
  const m = run.metrics
  const [ttft, ttftU] = split(fmt.ms(m.ttftMs))
  const [itl, itlU] = split(fmt.ms(m.itlMs))
  const [e2e, e2eU] = split(fmt.ms(m.e2eMs))
  const [tps, tpsU] = split(fmt.rate(m.tokensPerSec))
  const visibleLater = m.ttftVisibleMs !== undefined && m.ttftVisibleMs - m.ttftMs > 0.5
  // с рассуждением видимый текст сначала идёт в reasoning_content, а сам ответ (content или tool_calls) приходит заметно позже
  const answerLater = m.firstAnswerMs !== undefined && m.ttftVisibleMs !== undefined && m.firstAnswerMs - m.ttftVisibleMs > 0.5
  const ttftHint = answerLater
    ? `рассуждение через ${fmt.ms(m.ttftVisibleMs!)}, ответ через ${fmt.ms(m.firstAnswerMs!)}`
    : visibleLater
      ? `видимый текст через ${fmt.ms(m.ttftVisibleMs!)}`
      : 'первый чанк у клиента'
  // (X-sim-r2-4) шаг с prefill соседа тянет среднее вверх: рядом показываем обычный шаг
  const nb = neighborSteps(run)
  const itlHint = nb ? `в среднем; обычный шаг ${fmt.ms(nb.plainMs)}, ${neighborStepsText(nb)}` : 'между чанками, в среднем'
  const tpsHint = nb
    ? `токенов в секунду: 1000 / ITL в мс; без шага соседа около ${fmt.rate(1000 / nb.plainMs)}`
    : 'токенов в секунду у этого запроса: 1000 / ITL в мс'
  // стоп-токена нет, если ответ оборван на max_tokens
  const lastStop = run.decode[run.decode.length - 1]?.stop === true
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-5 @xl:grid-cols-4">
      <Stat
        label={<Term t="TTFT">TTFT</Term>}
        value={<span className="tabular-nums">{ttft}</span>}
        unit={ttftU}
        hint={ttftHint}
      />
      <Stat label={<Term t="ITL">ITL</Term>} value={<span className="tabular-nums">{itl}</span>} unit={itlU} hint={itlHint} />
      <Stat label={<Term t="E2E latency">E2E</Term>} value={<span className="tabular-nums">{e2e}</span>} unit={e2eU} hint="до последнего чанка" />
      <Stat label={<Term t="TPS">Скорость</Term>} value={<span className="tabular-nums">{tps}</span>} unit={tpsU} hint={tpsHint} />
      <Stat
        label="prompt_tokens"
        value={<span className="tabular-nums">{fmt.int(m.promptTokens)}</span>}
        hint={(run.preprocess?.toolsTokens ?? 0) > 0 ? `из них tools ${fmt.int(run.preprocess!.toolsTokens)}` : 'шаблон и сообщения'}
      />
      <Stat
        label="completion_tokens"
        value={<span className="tabular-nums">{fmt.int(m.completionTokens)}</span>}
        hint={lastStop ? 'включая стоп-токен' : 'ответ оборван на max_tokens, стоп-токена нет'}
      />
      <Stat label="cached_tokens" value={<span className="tabular-nums">{fmt.int(m.cachedTokens)}</span>} hint={m.cachedTokens > 0 ? 'взято из prefix cache' : 'префикс не найден в кэше'} />
      <Stat label={<Term t="finish_reason">finish_reason</Term>} value={<span className="font-mono text-[1.25rem]">{m.finishReason}</span>} hint={FINISH_HINT[m.finishReason]} />
    </div>
  )
})
