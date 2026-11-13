import { useEffect, useId, useMemo, useState } from 'react'
import { motion } from 'motion/react'
import { HourglassMediumIcon, StopCircleIcon } from '@phosphor-icons/react'
import { Badge, Figure, FieldHint, FieldLabel, Player, Segmented, Stat, TokenChip, TokenStripSkeleton, cn, fmt, tint, tokenSegments, usePlayer } from '@/ui'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { incrementalDetokenize, type DetokStepEx } from '@/engine/detok'
import type { Token } from '@/engine/types'
import { visibleText } from './utf8'

/** Ответ модели (иллюстрация): стоп-строки обрежут его в разных местах. */
const MODEL_TEXT = '<answer>KV cache хранит ключи и значения прошлых токенов.</answer>\nПользователь: а зачем?'

const STOPS = [
  { id: 'answer', label: '</answer>', stop: '</answer>' },
  { id: 'user', label: '↵Пользователь:', stop: '\nПользователь:' },
  { id: 'word', label: 'ключи', stop: 'ключи' },
] as const

type StopId = (typeof STOPS)[number]['id'] | 'custom'

interface Run {
  tokens: Token[]
  jail: DetokStepEx[]
  vllm: DetokStepEx[]
  /** Индекс токена, на котором сработал стоп (−1: не сработал). */
  stopAt: number
}

export function StopFigure() {
  const { tokenizer, status } = useTokenizer()
  const [choice, setChoice] = useState<StopId>('answer')
  const [custom, setCustom] = useState('токенов')
  const stop = choice === 'custom' ? custom : STOPS.find((s) => s.id === choice)!.stop

  const run = useMemo<Run | null>(() => {
    if (!tokenizer) return null
    const tokens = tokenizer.encode(MODEL_TEXT, { allowSpecial: false })
    const stops = stop ? [stop] : []
    const jail = incrementalDetokenize(tokens, tokenizer, { stop: stops, stopMode: 'jail' })
    const vllm = incrementalDetokenize(tokens, tokenizer, { stop: stops, stopMode: 'vllm' })
    const last = jail[jail.length - 1]
    return { tokens, jail, vllm, stopAt: last?.stopped ? jail.length - 1 : -1 }
  }, [tokenizer, stop])

  const steps = run ? run.jail.length : 1
  const p = usePlayer(steps, { intervalMs: 1000 })
  const { setStep } = p
  useEffect(() => {
    setStep(0)
  }, [stop, setStep])

  const labels = useMemo(() => (run ? run.jail.map((s) => `токен «${tokenSegments(s.token).map((x) => x.value).join('')}»`) : undefined), [run])
  const step = run ? Math.min(p.step, run.jail.length - 1) : 0

  const avgHeld = (list: DetokStepEx[]) => (list.length ? list.reduce((a, s) => a + [...s.stopHeld].length, 0) / list.length : 0)
  const maxStopLen = stop ? [...stop].length : 0

  return (
    <Figure
      id="detokenization-figure-stop"
      title="Стоп-строка поперёк токенов: два способа придержать текст"
      width="wide"
      controls={<Player {...p.bind} step={step} labels={labels} aria-label="Шаги стрима со стоп-строкой" />}
      caption="Попробуйте: выберите стоп «↵Пользователь:» и сравните колонки. Jail отдаёт ответ почти без задержки, а фиксированный хвост vLLM всё время держит последние 13 символов. Текст ответа условный, токены настоящие."
    >
      <div className="grid gap-5">
        <div className="grid items-start gap-4 @md:grid-cols-[minmax(0,auto)_minmax(0,1fr)]">
          <Segmented<StopId>
            label="Стоп-строка в запросе"
            value={choice}
            onChange={setChoice}
            wrap
            options={[...STOPS.map((s) => ({ value: s.id as StopId, label: <span className="font-mono text-[13px]">{s.label}</span> })), { value: 'custom', label: 'Своя' }]}
          />
          {choice === 'custom' && <StopInput value={custom} onChange={setCustom} />}
        </div>

        {status === 'error' && <p className="text-sm text-ink-2">Не удалось загрузить словарь Qwen3. Обновите страницу.</p>}
        {!run && status !== 'error' && <TokenStripSkeleton count={24} />}

        {run && (
          <>
            <div className="grid gap-2">
              <div className="text-sm font-medium text-ink-2">Токены, которые выдаёт модель</div>
              <div className="flex flex-wrap items-start gap-x-[3px] gap-y-1.5" role="list" aria-label="Токены ответа">
                {run.tokens.map((t, i) => {
                  const cut = run.stopAt >= 0 && i > run.stopAt
                  const future = i > step
                  return (
                    <span
                      key={i}
                      role="listitem"
                      className={cn('inline-flex', cut && step === run.stopAt && 'line-through decoration-ink-3 decoration-[1.5px]')}
                    >
                      <TokenChip token={t} variant={future || cut ? 'muted' : 'generated'} highlight={i === step} alt={i % 2 === 1} />
                    </span>
                  )
                })}
              </div>
              {/* строка зарезервирована: сообщение о стопе не сдвигает колонки и плеер */}
              <p className="min-h-[1lh] text-[13px] text-ink-3">{tokensNote(run, step, stop)}</p>
            </div>

            <div className="grid gap-3 @lg:grid-cols-2">
              <StopColumn
                title="Dynamo: jail"
                rule="Держим только хвост, который может оказаться началом стоп-строки."
                steps={run.jail}
                step={step}
                reasonField="nvext.stop_reason"
              />
              <StopColumn
                title="vLLM: фиксированный хвост"
                rule={`Всегда держим последние max(len(stop)) − 1 = ${fmt.int(Math.max(0, maxStopLen - 1))} символов.`}
                steps={run.vllm}
                step={step}
                reasonField="stop_reason"
              />
            </div>

            <div className="grid grid-cols-2 gap-4 @md:grid-cols-4">
              <Stat label="Jail держит" value={fmt.num(avgHeld(run.jail), 1)} unit="симв." hint="в среднем на шаге" />
              <Stat label="vLLM держит" value={fmt.num(avgHeld(run.vllm), 1)} unit="симв." hint="в среднем на шаге" />
              <Stat label="Токенов до стопа" value={fmt.int(run.jail.length)} hint={run.stopAt >= 0 ? 'дальше генерация не идёт' : 'стоп не сработал'} />
              <Stat label="Длина стоп-строки" value={fmt.int(maxStopLen)} unit="симв." />
            </div>
          </>
        )}
      </div>
    </Figure>
  )
}

/** Строка под лентой: стоп сработал, строки нет в ответе или стоп-строка не задана. */
function tokensNote(run: Run, step: number, stop: string): string {
  if (!stop) return 'Стоп-строка не задана: придерживать нечего, текст уходит клиенту сразу.'
  if (run.stopAt < 0) return 'Такой строки в ответе нет: стоп не сработает, и в конце генерации придержанный хвост уйдёт клиенту.'
  if (step === run.stopAt && run.stopAt < run.tokens.length - 1) return 'Зачёркнутые токены модель уже не сгенерирует: запрос остановлен.'
  return ''
}

/**
 * Колонка одного способа. На широком экране колонки делят строки сетки (subgrid),
 * поэтому поля «Клиент уже получил» стоят на одной высоте, даже если правило в одной из колонок длиннее.
 */
function StopColumn({ title, rule, steps, step, reasonField }: { title: string; rule: string; steps: DetokStepEx[]; step: number; reasonField: string }) {
  const s = steps[step]
  const sent = s.output
  const delta = s.emitted
  const before = sent.slice(0, sent.length - delta.length)
  const reason = s.stopped && typeof s.stopReason === 'string' ? s.stopReason : null
  // Dynamo отдаёт stop_reason не в choices, а только по запросу в nvext
  const optional = reasonField.startsWith('nvext')
  return (
    <section className="grid min-w-0 content-start gap-2 rounded-panel border border-line bg-surface-2/40 p-3.5 @lg:row-span-4 @lg:grid-rows-subgrid">
      <div>
        <h4 className="text-[15px] font-semibold text-ink">{title}</h4>
        <p className="text-[13px] leading-snug text-ink-3">{rule}</p>
      </div>
      <div className="text-xs font-medium text-ink-3">Клиент уже получил</div>
      <div className="min-h-[4.5rem] rounded-control bg-surface px-3 py-2 font-mono text-[14px] leading-relaxed break-words whitespace-pre-wrap text-ink shadow-[inset_0_0_0_1px_var(--line)]">
        {before}
        {delta && (
          <motion.span key={`${step}-${delta}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.25 }} className="rounded-[3px]" style={{ background: tint('decode', 18) }}>
            {delta}
          </motion.span>
        )}
        {s.stopHeld && (
          <span className="ml-0.5 rounded-chip border border-dashed border-ink-3 px-1 text-ink-2" aria-label={`придержано: ${s.stopHeld}`}>
            {visibleText(s.stopHeld)}
          </span>
        )}
      </div>
      <div className="flex min-h-7 flex-wrap items-center gap-2 text-[13px] text-ink-2">
        {s.stopped ? (
          <>
            <Badge tone="neutral" icon={StopCircleIcon}>
              стоп
            </Badge>
            <span className="font-mono text-[12.5px] text-ink">finish_reason: "stop"</span>
            {reason !== null && (
              <span className="font-mono text-[12.5px] text-ink">
                {reasonField}: {JSON.stringify(reason)}
                {optional && <span className="font-sans text-ink-3"> (по запросу)</span>}
              </span>
            )}
          </>
        ) : s.stopHeld ? (
          <span className="inline-flex items-center gap-1.5">
            <HourglassMediumIcon size={16} aria-hidden style={{ color: 'var(--warn)' }} />
            Придержано {fmt.count([...s.stopHeld].length, ['символ', 'символа', 'символов'])}
          </span>
        ) : (
          <span>Ничего не придержано</span>
        )}
      </div>
    </section>
  )
}

function StopInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const id = useId()
  return (
    <div className="grid min-w-0 gap-1.5">
      <FieldLabel htmlFor={id}>Своя стоп-строка</FieldLabel>
      <input
        id={id}
        value={value}
        maxLength={24}
        onChange={(e) => onChange(e.currentTarget.value)}
        spellCheck={false}
        className="block h-9 w-full min-w-0 rounded-control bg-surface-2 px-3 font-mono text-[14px] text-ink shadow-[inset_0_0_0_1px_var(--line)] transition-shadow hover:shadow-[inset_0_0_0_1px_var(--line-strong)] focus-visible:outline-2 focus-visible:outline-offset-0"
      />
      <FieldHint>Например, «прошлых» или «.».</FieldHint>
    </div>
  )
}

