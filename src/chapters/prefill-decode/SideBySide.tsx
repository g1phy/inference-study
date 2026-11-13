import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { ArrowDownIcon, PencilSimpleIcon } from '@phosphor-icons/react'
import { decodeStepBreakdown, kvBytesPerToken, prefillBreakdown, ridgePoint, type StepBreakdown } from '@/engine/perf'
import type { Token } from '@/engine/types'
import { scrollToSimulator } from '@/state/nav'
import { Badge, Button, Figure, Player, Segmented, Stat, TokenChip, TokenStrip, TokenStripSkeleton, cn, cssVar, fmt, tint, usePlayer } from '@/ui'
import { BoundTag } from './shared/Bound'
import { hwLabel, times, valueUnit, type Hardware } from './shared/hardware'
import { illustrativeAnswer, type PromptTokens } from './shared/prompt'

/** Сколько клеток KV рисуем на промпт максимум (дальше одна клетка = несколько токенов промпта). */
const MAX_PROMPT_CELLS = 120
/** Условный документ, который добавляем к промпту в режиме «С документом» (RAG, длинная история). */
export const DOC_TOKENS = 2000
/** Сколько столбцов-токенов рисуем в блоке весов у prefill. */
const MAX_COLUMNS = 48
const LAYER_FORMS = ['слой', 'слоя', 'слоёв'] as const

type PromptMode = 'user' | 'doc'

interface Props {
  prompt: PromptTokens
  hw: Hardware
}

interface Pass {
  kind: 'prefill' | 'decode'
  /** Номер шага decode (1…), у prefill 0. */
  index: number
  input: Token[]
  output: Token
  b: StepBreakdown
}

/** Длительность шага плеера при скорости 1×, мс: пауза на промпте, долгий prefill, быстрые шаги decode. */
const intervalOf = (s: number) => (s === 0 ? 1300 : s === 1 ? 2600 : 800)

export function SideBySideFigure({ prompt, hw }: Props) {
  const [mode, setMode] = useState<PromptMode>('user')
  const answer = useMemo(
    () => (prompt.tokenizer ? prompt.tokenizer.encode(illustrativeAnswer(prompt.thinking, prompt.aboutKv)) : null),
    [prompt.tokenizer, prompt.thinking, prompt.aboutKv],
  )
  const promptTokens = prompt.tokens
  const userN = promptTokens?.length ?? 0
  const doc = mode === 'doc' ? DOC_TOKENS : 0
  const N = userN + doc
  const D = answer?.length ?? 0

  // Проходы модели: prefill (весь промпт → первый токен) и D − 1 шагов decode (последний выдаёт <|im_end|>).
  const passes = useMemo<Pass[]>(() => {
    if (!promptTokens || !answer || userN === 0 || D === 0) return []
    const { model, gpu, perf } = hw
    const out: Pass[] = [{ kind: 'prefill', index: 0, input: promptTokens, output: answer[0], b: prefillBreakdown(model, gpu, N, perf) }]
    for (let j = 1; j < D; j++) {
      out.push({
        kind: 'decode',
        index: j,
        input: [answer[j - 1]],
        output: answer[j],
        // длина последовательности с текущим токеном: N + j (его K и V уже пишутся в кэш)
        b: decodeStepBreakdown(model, gpu, { ...perf, batch: 1, avgContext: N + j }),
      })
    }
    return out
  }, [promptTokens, answer, userN, N, D, hw])

  const kvPerToken = useMemo(() => kvBytesPerToken(hw.model, 'bf16'), [hw.model])
  const ridge = ridgePoint(hw.gpu, hw.dtype)
  const steps = passes.length + 1
  const p = usePlayer(Math.max(2, steps), { autoplay: true, intervalMs: intervalOf })
  const labels = useMemo(() => ['Промпт готов', 'Prefill', ...passes.slice(1).map((ps) => `Decode, шаг ${ps.index}`)], [passes])

  const reduce = useReducedMotion() ?? false
  const loading = passes.length === 0
  const s = Math.min(p.step, steps - 1)
  // Проход длится столько, сколько бежит полоса чтения весов; итоги обновляются, когда проход закончился.
  const passMs = reduce ? 0 : Math.min(750, (0.85 * intervalOf(s)) / p.speed)
  const shown = useShownStep(s, passMs)
  const current = s >= 1 ? passes[s - 1] : null
  const done = passes.slice(0, shown)
  const weightsRead = done.reduce((a, ps) => a + ps.b.weightBytes, 0)
  const elapsed = done.reduce((a, ps) => a + ps.b.ms, 0)
  const kvTokens = shown === 0 ? 0 : N + shown - 1
  const outputs = done.map((ps) => ps.output)
  const prefillPass = passes[0]
  const decodeActive = current?.kind === 'decode' ? current : null
  const lastDecode = passes.length > 1 ? passes[passes.length - 1] : null
  // до начала decode показываем цифры его первого шага
  const decodeRef = decodeActive ?? passes[1] ?? lastDecode ?? prefillPass
  const inFlight = current !== null && shown < s

  const promptStrip = (limit: number) => (
    <div className="flex min-w-0 flex-wrap items-center gap-1.5">
      <TokenStrip tokens={promptTokens!} variant="prompt" size="sm" limit={limit} label="Токены промпта" />
      {doc > 0 && (
        <span className="inline-flex items-center rounded-chip border border-dashed border-line-strong px-1.5 py-0.5 text-[12px] text-ink-2">
          + {fmt.tokens(doc)} документа, условно
        </span>
      )}
    </div>
  )

  return (
    <Figure
      id="prefill-decode-figure-side"
      title="Prefill и decode на вашем запросе"
      width="wide"
      actions={<span className="block max-w-[18rem] text-[13px] sm:max-w-none leading-snug text-ink-3">{hwLabel(hw)}</span>}
      inputs={
        <>
          <Segmented<PromptMode>
            label="Промпт"
            value={mode}
            onChange={setMode}
            wrap
            options={[
              { value: 'user', label: 'Ваш запрос' },
              { value: 'doc', label: `С документом, +${fmt.int(DOC_TOKENS)} токенов` },
            ]}
          />
          <Button variant="ghost" size="sm" icon={PencilSimpleIcon} onClick={() => scrollToSimulator({ focusPrompt: true })}>
            Изменить запрос
          </Button>
        </>
      }
      player={<Player {...p.bind} labels={labels} aria-label="Шаги генерации" />}
      caption={
        <>
          Попробуйте: нажмите пуск и следите за счётчиком «Весов прочитано»: prefill читает веса один раз на все токены промпта, а decode читает их
          заново ради каждого нового токена. Потом включите «С документом»: у плотной модели (не MoE) prefill станет заметно дольше и упрётся в
          вычисления, а шаг decode почти не изменится. Свой запрос можно написать в симуляторе вверху страницы, фигура пересчитается.
        </>
      }
      source={`Промпт настоящий (шаблон и словарь Qwen3), документ и ответ условные: модель в браузере не запускается. Время посчитано по характеристикам модели и GPU (${hwLabel(hw)}), как в главе «Compute-bound и memory-bound».`}
    >
      {loading ? (
        <div className="grid gap-4">
          <TokenStripSkeleton count={36} />
          <div className="h-40 rounded-panel bg-surface-2" />
        </div>
      ) : (
        <div className="grid gap-5">
          <div className="grid gap-4 @3xl:grid-cols-2">
            <PhasePanel
              kind="prefill"
              state={s === 0 ? 'idle' : s === 1 ? 'active' : 'done'}
              narrowHidden={s >= 2}
              title="один проход по всему промпту"
              input={
                <>
                  <div className="@3xl:hidden">{promptStrip(12)}</div>
                  <div className="hidden @3xl:block">{promptStrip(22)}</div>
                </>
              }
              inputCaption={`На входе ${fmt.tokens(N)} сразу`}
              output={shown >= 1 ? prefillPass.output : null}
              runKey={`prefill-${mode}`}
              running={s === 1}
              passMs={passMs}
              weightsLabel={fmt.bytes(prefillPass.b.weightBytes)}
              layers={hw.model.layers}
              columns={N}
              facts={[
                ['Токенов за проход', fmt.int(N)],
                ['Весов прочитано', fmt.bytes(prefillPass.b.weightBytes)],
                ['Время прохода', fmt.ms(prefillPass.b.ms)],
              ]}
              verdict={<Verdict b={prefillPass.b} ridge={ridge} />}
            />
            <PhasePanel
              kind="decode"
              state={s < 2 ? 'idle' : 'active'}
              narrowHidden={s < 2}
              title={decodeActive ? `шаг ${decodeActive.index} из ${passes.length - 1}` : 'по одному токену за проход'}
              input={
                decodeActive ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <motion.span key={`in-${s}`} initial={reduce ? false : { opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
                      <TokenChip token={decodeActive.input[0]} variant="generated" size="sm" />
                    </motion.span>
                    <span className="text-[13px] text-ink-3">только что выбранный токен</span>
                  </div>
                ) : (
                  <span className="text-[13px] text-ink-2">ждёт первый токен от prefill</span>
                )
              }
              inputCaption="На входе 1 токен"
              output={decodeActive && shown >= s ? decodeActive.output : null}
              runKey={`decode-${s}-${mode}`}
              running={Boolean(decodeActive)}
              passMs={passMs}
              weightsLabel={fmt.bytes(decodeRef.b.weightBytes)}
              layers={hw.model.layers}
              columns={1}
              facts={[
                ['Токенов за проход', '1'],
                ['Весов прочитано', fmt.bytes(decodeRef.b.weightBytes)],
                ['Время шага', fmt.ms(decodeRef.b.ms)],
              ]}
              verdict={<Verdict b={decodeRef.b} ridge={ridge} />}
            />
          </div>

          <PerTokenBars prefill={prefillPass.b.weightBytes / N} decode={decodeRef.b.weightBytes} n={N} />

          <KvStrip prompt={N} total={kvTokens} step={shown} kvPerToken={kvPerToken} readKey={decodeActive ? s : null} passMs={passMs} />

          <div className="grid gap-1.5">
            <div className="text-sm text-ink-2">Ответ (условный), {fmt.tokens(outputs.length)}</div>
            <div className="min-h-[30px]">
              {outputs.length > 0 ? (
                <TokenStrip tokens={outputs} variant="generated" size="sm" highlightIndex={outputs.length - 1} label="Сгенерированные токены" />
              ) : (
                <span className="text-[13px] text-ink-2">пока пусто</span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-x-6 gap-y-4 border-t border-line pt-4 @xl:grid-cols-4">
            <Stat label="Проходов модели" value={fmt.int(shown)} hint={shown === 0 ? 'ещё ни одного' : `1 prefill + ${fmt.int(Math.max(0, shown - 1))} decode`} />
            <Stat label="Токенов обработано" value={fmt.int(shown === 0 ? 0 : N + shown - 1)} hint={`промпт ${fmt.int(N)} + ответ ${fmt.int(Math.max(0, shown - 1))}`} />
            <Stat
              label="Весов прочитано"
              value={weightsRead > 0 ? valueUnit(fmt.bytes(weightsRead))[0] : '0'}
              unit={weightsRead > 0 ? valueUnit(fmt.bytes(weightsRead))[1] : 'ГБ'}
              hint="суммарно за все проходы"
            />
            <Stat
              label="KV cache"
              value={kvTokens > 0 ? valueUnit(fmt.bytes(kvTokens * kvPerToken))[0] : '0'}
              unit={kvTokens > 0 ? valueUnit(fmt.bytes(kvTokens * kvPerToken))[1] : 'МБ'}
              hint={`${fmt.tokens(kvTokens)} × ${fmt.bytes(kvPerToken, { mode: 'binary' })}`}
            />
          </div>
          <p className="-mt-2 text-[13px] text-ink-3">
            Прошло времени: {fmt.ms(elapsed)}
            {inFlight && current ? `, идёт ${current.kind === 'prefill' ? 'prefill' : `шаг decode ${current.index}`} (${fmt.ms(current.b.ms)})` : ''}.
          </p>
        </div>
      )}
    </Figure>
  )
}

/** Шаг, до которого «досчитано»: догоняет шаг плеера, когда закончилась анимация прохода (назад сразу). */
function useShownStep(step: number, passMs: number): number {
  const [shown, setShown] = useState(step)
  useEffect(() => {
    if (passMs <= 0) {
      setShown(step)
      return
    }
    setShown((prev) => (step < prev ? step : prev))
    const t = window.setTimeout(() => setShown(step), passMs)
    return () => window.clearTimeout(t)
  }, [step, passMs])
  return Math.min(shown, step)
}

/* ------------------------------------------------------------------------- */

function Arrow() {
  return (
    <div aria-hidden className="-my-1.5 flex justify-center text-ink-3">
      <ArrowDownIcon size={16} />
    </div>
  )
}

function Verdict({ b, ridge }: { b: StepBreakdown; ridge: number }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[13px] leading-snug">
      <BoundTag bound={b.bound} memoryText="упор в чтение памяти" className="font-medium text-ink" />
      <span className="text-ink-3">
        {fmt.num(b.intensity)} FLOP на байт, порог ≈ {fmt.int(ridge)}
      </span>
    </div>
  )
}

function PhasePanel({
  kind,
  state,
  narrowHidden,
  title,
  input,
  inputCaption,
  output,
  runKey,
  running,
  passMs,
  weightsLabel,
  layers,
  columns,
  facts,
  verdict,
}: {
  kind: 'prefill' | 'decode'
  state: 'idle' | 'active' | 'done'
  /** На узком экране показываем только панель текущей фазы. */
  narrowHidden: boolean
  title: string
  input: ReactNode
  inputCaption: string
  output: Token | null
  runKey: string
  running: boolean
  passMs: number
  weightsLabel: string
  layers: number
  columns: number
  facts: Array<[string, string]>
  verdict: ReactNode
}) {
  const reduce = useReducedMotion() ?? false
  return (
    <section
      aria-label={kind === 'prefill' ? 'Prefill' : 'Decode'}
      className={cn(
        'min-w-0 content-start gap-3 rounded-panel border p-3 transition-colors sm:p-4',
        state === 'active' ? 'border-accent' : state === 'idle' ? 'border-dashed border-line-strong' : 'border-line',
        narrowHidden ? 'hidden @3xl:grid' : 'grid',
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={kind}>{kind === 'prefill' ? 'Prefill' : 'Decode'}</Badge>
        <span className="text-[13px] text-ink-2">{title}</span>
      </div>

      <div className="grid gap-1.5">
        <div className="text-[13px] text-ink-2">{inputCaption}</div>
        <div className="min-h-[30px] min-w-0">{input}</div>
      </div>

      <Arrow />
      <WeightsBlock kind={kind} layers={layers} columns={columns} running={running} runKey={runKey} reduce={reduce} idle={state === 'idle'} passMs={passMs} label={weightsLabel} />
      <Arrow />

      <div className="flex min-h-[30px] flex-wrap items-center gap-2">
        <span className="text-[13px] text-ink-2">{kind === 'prefill' ? 'На выходе первый токен ответа' : 'На выходе следующий токен'}</span>
        {output && (
          <motion.span key={`${runKey}-out`} initial={reduce ? false : { opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.25 }}>
            <TokenChip token={output} variant="generated" size="sm" />
          </motion.span>
        )}
      </div>

      <div className="grid gap-2 border-t border-line pt-3">
        <dl className="grid grid-cols-3 gap-x-3 text-[13px]">
          {facts.map(([k, v]) => (
            <div key={k} className="grid min-w-0 content-start gap-0.5">
              <dt className="text-ink-2">{k}</dt>
              <dd className="font-medium text-ink">{v}</dd>
            </div>
          ))}
        </dl>
        {verdict}
      </div>
    </section>
  )
}

/**
 * Веса модели: один и тот же блок в обеих фазах (полосы = слои). Поверх него столбцы токенов этого прохода:
 * в prefill столбец на каждый токен промпта (все идут параллельно), в decode один. Во время прохода сверху
 * вниз бежит полоса «чтение весов»: плотная модель читает веса целиком в обоих случаях, у MoE шаг decode читает
 * только выбранных экспертов (подпись показывает, сколько прочитано за проход).
 */
function WeightsBlock({
  kind,
  layers,
  columns,
  running,
  runKey,
  reduce,
  idle,
  passMs,
  label,
}: {
  kind: 'prefill' | 'decode'
  layers: number
  columns: number
  running: boolean
  runKey: string
  reduce: boolean
  idle: boolean
  passMs: number
  label: string
}) {
  const cols = Math.max(1, Math.min(columns, MAX_COLUMNS))
  const perCol = columns > MAX_COLUMNS ? Math.ceil(columns / MAX_COLUMNS) : 1
  return (
    <div className="grid gap-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 text-[12px] leading-tight">
        <span className="font-medium text-ink-2">
          Веса: {label}, {fmt.count(layers, LAYER_FORMS)}
        </span>
        <span className="text-ink-3">
          {kind === 'prefill' ? (perCol > 1 ? `столбец = ${fmt.tokens(perCol)}` : 'столбец на токен') : 'один столбец: один токен'}
        </span>
      </div>
      <div
        className={cn('relative h-16 overflow-hidden rounded-control border border-line-strong bg-surface-2', idle && 'opacity-60')}
        role="img"
        aria-label={`Весов прочитано за проход: ${label}, ${fmt.count(layers, LAYER_FORMS)}; ${kind === 'prefill' ? `через слои одновременно проходят все ${fmt.tokens(columns)} промпта` : 'через слои проходит один токен'}`}
        style={{ backgroundImage: 'repeating-linear-gradient(to bottom, transparent 0 7px, var(--line) 7px 8px)' }}
      >
        {kind === 'prefill' ? (
          <div className="absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            {Array.from({ length: cols }, (_, i) => (
              <div key={i} className={cn(i < cols - 1 && 'border-r border-line')} style={{ background: tint('prefill', running ? 16 : 9) }} />
            ))}
          </div>
        ) : (
          <div className="absolute inset-y-0 right-0 w-3 border-l border-line-strong" style={{ background: tint('decode', running ? 40 : 22) }} />
        )}
        {running && !reduce && passMs > 0 && (
          <motion.div
            key={runKey}
            aria-hidden
            className="absolute inset-x-0 top-0 h-1/4"
            style={{ background: tint(kind, 45) }}
            initial={{ y: '-100%' }}
            animate={{ y: '400%' }}
            transition={{ duration: passMs / 1000, ease: 'easeInOut' }}
          />
        )}
        {running && (reduce || passMs <= 0) && <div aria-hidden className="absolute inset-0" style={{ background: tint(kind, 18) }} />}
      </div>
    </div>
  )
}

/** Сколько весов приходится на один токен: две полосы на одной шкале (без серой дорожки). */
function PerTokenBars({ prefill, decode, n }: { prefill: number; decode: number; n: number }) {
  const max = Math.max(prefill, decode, 1)
  const row = (name: string, role: 'prefill' | 'decode', v: number) => (
    <div className="grid grid-cols-[4.25rem_minmax(0,1fr)] items-center gap-2">
      <span className="text-[13px] text-ink-2">{name}</span>
      <div className="flex min-w-0 items-center gap-2">
        <span aria-hidden className="h-3 shrink-0 rounded-[3px]" style={{ width: `calc((100% - 4.5rem) * ${v / max})`, minWidth: 2, background: cssVar(role) }} />
        <span className="text-[13px] font-medium whitespace-nowrap text-ink">{fmt.bytes(v)}</span>
      </div>
    </div>
  )
  return (
    <div className="grid gap-1.5 rounded-control bg-surface-2 px-3 py-2.5" role="img" aria-label={`Весов на один токен: prefill ${fmt.bytes(prefill)}, decode ${fmt.bytes(decode)}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 text-[13px]">
        <span className="font-medium text-ink">Весов прочитано на один токен</span>
        <span className="text-ink-2">у decode в {times(decode / Math.max(prefill, 1e-9))} больше</span>
      </div>
      {row('Prefill', 'prefill', prefill)}
      {row('Decode', 'decode', decode)}
      {n > 1 && <span className="sr-only">Prefill делит одно чтение весов на {fmt.tokens(n)}.</span>}
    </div>
  )
}

/**
 * KV cache: клетки промпта цветом prefill (при длинном промпте одна клетка = несколько токенов),
 * клетки ответа цветом decode, по одной на токен. Шаг decode читает весь кэш и дописывает одну клетку.
 */
function KvStrip({
  prompt,
  total,
  step,
  kvPerToken,
  readKey,
  passMs,
}: {
  prompt: number
  total: number
  step: number
  kvPerToken: number
  /** Ключ текущего шага decode, пока он читает кэш (null вне decode). */
  readKey: number | null
  passMs: number
}) {
  const reduce = useReducedMotion() ?? false
  const per = Math.max(1, Math.ceil(prompt / MAX_PROMPT_CELLS))
  const promptCells = total > 0 ? Math.ceil(prompt / per) : 0
  const answerCells = Math.max(0, total - prompt)
  const cells = promptCells + answerCells
  return (
    <div className="grid gap-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="text-sm text-ink-2">
          KV cache: {fmt.tokens(total)}
          {total > 0 && <span className="text-ink-3"> ({fmt.bytes(total * kvPerToken)})</span>}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-ink-3">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: tint('prefill', 55) }} />K и V токенов промпта
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: tint('decode', 55) }} />K и V токенов ответа
          </span>
          {per > 1 && <span>клетка промпта = {fmt.tokens(per)}, клетка ответа = 1 токен</span>}
        </div>
      </div>
      <div className="relative min-h-[22px] overflow-hidden rounded-control border border-line bg-surface-2 p-1" role="img" aria-label={`KV cache: ${fmt.tokens(total)}`}>
        <div className="flex flex-wrap gap-[2px]">
          {Array.from({ length: cells }, (_, i) => {
            const isPrompt = i < promptCells
            const isNew = step >= 2 && i === cells - 1
            const isPrefillWrite = step === 1 && isPrompt
            return (
              <motion.span
                key={i}
                aria-hidden
                className="block h-3.5 w-2 rounded-[2px]"
                style={{ background: tint(isPrompt ? 'prefill' : 'decode', 55) }}
                initial={reduce || (!isNew && !isPrefillWrite) ? false : { opacity: 0, scale: 0.4 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.2, delay: reduce || !isPrefillWrite ? 0 : i * 0.003 }}
              />
            )
          })}
        </div>
        {readKey !== null && !reduce && passMs > 0 && (
          <motion.div
            key={`read-${readKey}`}
            aria-hidden
            className="pointer-events-none absolute inset-y-0 left-0 w-1/5"
            style={{ background: `linear-gradient(90deg, transparent, ${tint('decode', 35)}, transparent)` }}
            initial={{ x: '-100%' }}
            animate={{ x: '500%' }}
            transition={{ duration: (passMs / 1000) * 0.9, ease: 'easeInOut' }}
          />
        )}
      </div>
      <div className="text-[12px] text-ink-3">
        {step === 0 && 'Пока пусто: кэш заполнит prefill.'}
        {step === 1 && `Prefill записал K и V всех ${fmt.tokens(prompt)} промпта во всех слоях за один проход.`}
        {step >= 2 && 'Шаг decode читает весь кэш (attention нового токена смотрит на все предыдущие) и дописывает одну клетку.'}
      </div>
    </div>
  )
}
