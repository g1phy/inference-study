import { useMemo, useState } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { Figure, Player, Segmented, Slider, fmt, mix, usePlayer } from '@/ui'
import { cn } from '@/ui/cn'
import { TOY_REQUESTS, toySchedule, type ToyStep } from './toyScheduler'
import { LegendItem } from './chart'

/**
 * Игрушечный планировщик vLLM V1: бюджет токенов на шаг как «места в автобусе».
 * Сначала садятся идущие запросы (decode по 1 токену, хвосты prefill), потом новые из очереди.
 */
export function BudgetFigure() {
  const [budget, setBudget] = useState(10)
  const [maxSeqs, setMaxSeqs] = useState(4)
  const steps = useMemo(() => toySchedule(TOY_REQUESTS, { budget, maxSeqs }), [budget, maxSeqs])
  const player = usePlayer(steps.length, { intervalMs: 2600 })
  const cur = steps[Math.min(player.step, steps.length - 1)]

  return (
    <Figure
      id="batching-figure-budget"
      title="Как планировщик собирает один шаг"
      player={<Player {...player.bind} aria-label="Шаги планировщика" labels={steps.map((s) => stepTitle(s))} />}
      controls={
        <>
          <Slider
            label="Бюджет шага"
            hint="max_num_batched_tokens"
            value={budget}
            min={4}
            max={16}
            onChange={setBudget}
            format={(v) => fmt.tokens(v)}
          />
          <Segmented<number>
            label="Запросов в работе не больше"
            value={maxSeqs}
            onChange={setMaxSeqs}
            options={[2, 3, 4, 8].map((v) => ({ value: v, label: String(v) }))}
          />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> поставьте бюджет 16. Промпт B войдёт целиком, и его первый токен выйдет уже на
          первом шаге. Зато этот шаг тяжелее (15 токенов вместо 10), значит длиннее по времени, и свой первый токен A получит позже по часам, хотя
          номер шага тот же и самому A ничего не добавилось. Потом ограничьте запросы двумя. Числа игрушечные, логика как у планировщика vLLM V1.
        </>
      }
    >
      <ul className="mb-4 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px]" style={{ background: mix('prefill', 32) }} />}>токен промпта (prefill)</LegendItem>
        <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px]" style={{ background: mix('prefill', 62) }} />}>промпт досчитан, будет первый токен</LegendItem>
        <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px]" style={{ background: mix('decode', 32) }} />}>новый токен (decode)</LegendItem>
        <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px] border border-dashed border-line-strong" />}>свободное место в бюджете</LegendItem>
        <LegendItem swatch={<EmitChip text="a1" />}>справа от шага: какие токены он выдал (a1 это первый выходной токен запроса A)</LegendItem>
      </ul>
      <div className="grid gap-5 @lg:grid-cols-[minmax(0,1fr)_240px]">
        <ol className="grid min-w-0 content-start gap-2" aria-label="Шаги">
          {steps.map((s, i) => (
            <StepRow key={s.step} s={s} state={i < player.step ? 'past' : i === player.step ? 'current' : 'future'} onSelect={() => player.setStep(i)} />
          ))}
        </ol>
        <QueuePanel s={cur} maxSeqs={maxSeqs} />
      </div>
      <p className="mt-4 min-h-[3.2em] text-[15px] leading-relaxed text-ink" aria-live="polite">
        {explain(cur, maxSeqs)}
      </p>
    </Figure>
  )
}

function stepTitle(s: ToyStep): string {
  return `${s.used} из ${s.budget} мест занято`
}

function StepRow({ s, state, onSelect }: { s: ToyStep; state: 'past' | 'current' | 'future'; onSelect: () => void }) {
  const reduce = useReducedMotion()
  const cells: Array<{ id: string; kind: 'prefill' | 'decode' | 'free'; last?: boolean }> = []
  for (const a of s.allocs) {
    for (let k = 0; k < a.tokens; k++) cells.push({ id: a.id, kind: a.kind, last: a.kind === 'prefill' && a.completes && k === a.tokens - 1 })
  }
  while (cells.length < s.budget) cells.push({ id: '', kind: 'free' })
  const hidden = state === 'future'
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-current={state === 'current' ? 'step' : undefined}
        aria-label={`Шаг ${s.step}: ${s.used} из ${s.budget} мест`}
        className={cn(
          'grid w-full grid-cols-[3.75rem_minmax(0,1fr)] items-center gap-3 rounded-control px-2 py-1.5 text-left transition-colors',
          state === 'current' ? 'bg-accent-soft' : 'hover:bg-surface-3/60',
        )}
      >
        <span className={cn('text-sm tabular-nums', state === 'current' ? 'font-semibold text-ink' : 'text-ink-3')}>Шаг {s.step}</span>
        <span className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1.5">
          <span className="flex flex-wrap gap-[3px]">
            {cells.map((c, k) => (
              <motion.span
                key={k}
                initial={false}
                animate={{ opacity: hidden ? 0.12 : 1 }}
                transition={{ duration: reduce ? 0 : 0.25, delay: reduce || hidden ? 0 : k * 0.02 }}
                className={cn(
                  'inline-flex h-7 w-6 items-center justify-center rounded-[5px] font-mono text-[12px] font-semibold text-ink',
                  c.kind === 'free' && 'border border-dashed border-line-strong',
                )}
                style={c.kind === 'free' ? undefined : { background: mix(c.kind === 'prefill' ? 'prefill' : 'decode', c.last ? 62 : 32) }}
              >
                {c.id}
              </motion.span>
            ))}
          </span>
          {!hidden && s.emitted.length > 0 && (
            <span className="flex flex-wrap gap-1">
              {s.emitted.map((e) => (
                <EmitChip key={e.id} text={`${e.id.toLowerCase()}${e.index}`} />
              ))}
            </span>
          )}
        </span>
      </button>
    </li>
  )
}

/** Выходной токен шага: «a1» = первый токен запроса A. */
function EmitChip({ text }: { text: string }) {
  return <span className="rounded-chip bg-surface-2 px-1.5 font-mono text-[12px] leading-5 text-ink-2 shadow-[inset_0_0_0_1px_var(--line)]">{text}</span>
}

function QueuePanel({ s, maxSeqs }: { s: ToyStep; maxSeqs: number }) {
  const running = s.runningBefore.concat(s.allocs.filter((a) => a.admitted).map((a) => a.id))
  return (
    <div className="grid content-start gap-3 rounded-panel bg-surface-2 p-3.5 text-sm">
      <div>
        <div className="text-ink-3">В работе (RUNNING), не больше {maxSeqs}</div>
        <div className="mt-1 flex min-h-7 flex-wrap gap-1.5">
          {running.length ? running.map((id) => <Chip key={id} id={id} done={s.finished.includes(id)} />) : <span className="text-ink-3">пусто</span>}
        </div>
      </div>
      <div>
        <div className="text-ink-3">Ждут (WAITING)</div>
        <div className="mt-1 flex min-h-7 flex-wrap gap-1.5">
          {s.waitingAfter.length ? s.waitingAfter.map((id) => <Chip key={id} id={id} />) : <span className="text-ink-3">пусто</span>}
        </div>
      </div>
      <div className="border-t border-line pt-2.5 text-ink-2">
        Занято <span className="font-semibold text-ink tabular-nums">{s.used}</span> из {s.budget} мест; токены вышли у{' '}
        {s.emitted.length ? s.emitted.map((e) => e.id).join(', ') : 'никого'}.
      </div>
    </div>
  )
}

function Chip({ id, done = false }: { id: string; done?: boolean }) {
  return (
    <span
      className={cn(
        'inline-flex h-7 min-w-7 items-center justify-center rounded-chip px-1.5 font-mono text-[13px] font-semibold',
        done ? 'bg-surface text-ink-3 line-through shadow-[inset_0_0_0_1px_var(--line)]' : 'bg-surface text-ink shadow-[inset_0_0_0_1px_var(--line-strong)]',
      )}
      title={done ? 'закончил на этом шаге' : undefined}
    >
      {id}
    </span>
  )
}

/** Объяснение шага по-русски (генерируется из данных шага). */
function explain(s: ToyStep, maxSeqs: number): string {
  const out: string[] = []
  if (s.arrived.length) out.push(`${listRu(s.arrived)} ${s.arrived.length > 1 ? 'пришли' : 'пришёл'} в очередь.`)
  const dec = s.allocs.filter((a) => a.kind === 'decode').map((a) => a.id)
  if (dec.length)
    out.push(
      dec.length > 1
        ? `${listRu(dec)} получают по одному месту: это decode, по новому токену.`
        : `${dec[0]} получает одно место: это decode, один новый токен.`,
    )
  for (const a of s.allocs.filter((x) => x.kind === 'prefill')) {
    if (a.chunked) out.push(`${a.id} получает только ${fmt.tokens(a.tokens)} промпта: бюджет кончился, остаток пойдёт на следующем шаге (chunked prefill), токена у ${a.id} пока нет.`)
    else if (a.admitted) out.push(`${a.id} входит в batch с промптом целиком (${fmt.tokens(a.tokens)}) и в конце шага выдаёт первый токен.`)
    else out.push(`${a.id} досчитывает хвост промпта (${fmt.tokens(a.tokens)}) и выдаёт первый токен.`)
  }
  const waits = (ids: string[]) => (ids.length > 1 ? 'ждут' : 'ждёт')
  if (s.blockedBySeqs.length)
    out.push(`${listRu(s.blockedBySeqs)} ${waits(s.blockedBySeqs)}, хотя места в бюджете есть: в работе уже ${fmt.count(maxSeqs, REQ_FORMS)}, это предел max_num_seqs.`)
  else if (s.blockedByBudget.length) out.push(`${listRu(s.blockedByBudget)} ${waits(s.blockedByBudget)}: бюджет шага исчерпан.`)
  if (s.finished.length)
    out.push(`${listRu(s.finished)} ${s.finished.length > 1 ? 'закончили и освобождают свои места' : 'закончил и освобождает место'} для следующего шага.`)
  return out.join(' ')
}

const REQ_FORMS = ['запрос', 'запроса', 'запросов'] as const

function listRu(ids: string[]): string {
  if (ids.length <= 1) return ids.join('')
  return `${ids.slice(0, -1).join(', ')} и ${ids[ids.length - 1]}`
}
