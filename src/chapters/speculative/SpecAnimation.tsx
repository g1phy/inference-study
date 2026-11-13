import { useEffect, useMemo, useRef, useState } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { CheckCircleIcon, MinusCircleIcon, XCircleIcon } from '@phosphor-icons/react'
import { piecesText, scriptFor } from '@/engine/scenarios'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { Token } from '@/engine/types'
import { useSettings } from '@/state/settings'
import { Figure, Player, Slider, Stat, TokenStripSkeleton, cn, fmt, mix, tint, usePlayer } from '@/ui'
import { pickAnswer } from './answer'
import { expectedTokens, simulateRounds, type SpecRound } from './spec'
import { Chip, LegendItem } from './TokenBits'

/** Слова, которыми «ошибается» черновик (иллюстрация). */
const DISTRACTORS = ' это так же и тоже потом их один все снова уже только'

const PHASES = ['черновик', 'проверка', 'итог'] as const

interface View {
  round: SpecRound | null
  phase: number
  committed: number
  fresh: [number, number] | null
  passes: number
}

function viewAt(rounds: SpecRound[], step: number): View {
  if (rounds.length === 0) return { round: null, phase: 0, committed: 0, fresh: null, passes: 0 }
  const r = Math.min(rounds.length - 1, Math.floor(step / 3))
  const phase = step % 3
  const round = rounds[r]
  const committed = phase === 2 ? round.targetPos + 1 : round.start
  return {
    round,
    phase,
    committed,
    fresh: phase === 2 ? [round.start, round.targetPos + 1] : null,
    passes: r + (phase >= 1 ? 1 : 0),
  }
}

export function SpecAnimation() {
  const { tokenizer, status } = useTokenizer()
  const [settings] = useSettings()
  const script = useMemo(() => scriptFor(settings), [settings])
  const body = script.bodyKind === 'tool_call' ? null : piecesText(script.body)
  const picked = useMemo(() => (tokenizer ? pickAnswer(tokenizer, body) : null), [tokenizer, body])
  const distractors = useMemo<Token[]>(
    () => (tokenizer ? tokenizer.encode(DISTRACTORS, { allowSpecial: false }).filter((t) => t.text.trim() !== '') : []),
    [tokenizer],
  )
  if (status !== 'ready' || !picked || picked.tokens.length === 0) {
    return (
      <Figure id="speculative-figure-rounds" title="Черновик предлагает, target проверяет за один проход">
        <div className="grid gap-4" aria-busy="true">
          <TokenStripSkeleton count={34} />
          <TokenStripSkeleton count={8} />
        </div>
      </Figure>
    )
  }
  return <SpecRounds answer={picked.tokens} fromRun={picked.fromRun} distractors={distractors} />
}

function SpecRounds({ answer, fromRun, distractors }: { answer: Token[]; fromRun: boolean; distractors: Token[] }) {
  const [alpha, setAlpha] = useState(0.7)
  const [k, setK] = useState(4)
  const reduce = useReducedMotion() ?? false
  const rounds = useMemo(() => simulateRounds(answer.length, alpha, k, 7), [answer.length, alpha, k])
  const steps = Math.max(1, rounds.length * 3)
  const labels = useMemo(() => Array.from({ length: steps }, (_, i) => `Проход ${Math.floor(i / 3) + 1}: ${PHASES[i % 3]}`), [steps])
  const p = usePlayer(steps, { autoplay: true, intervalMs: (s) => (s % 3 === 2 ? 1500 : 1100) })
  const { reset } = p
  const answerKey = answer.map((t) => t.id).join(',')
  const first = useRef(true)
  useEffect(() => {
    if (first.current) {
      first.current = false
      return
    }
    reset()
  }, [alpha, k, answerKey, reset])

  const v = viewAt(rounds, p.step)
  const wrong = (pos: number, i: number) => (distractors.length ? distractors[(pos * 7 + i * 3) % distractors.length] : answer[pos])
  const done = v.committed
  // среднее только по завершённым проходам: текущий ещё не выдал токены
  const completed = v.round ? Math.min(rounds.length, Math.floor(p.step / 3)) + (v.phase === 2 ? 1 : 0) : 0
  const perPass = completed > 0 ? done / completed : 0

  const caption = (() => {
    const r = v.round
    if (!r) return ''
    const n = r.drafts.length
    if (v.phase === 0)
      return n === 0
        ? 'Последний токен ответа: черновик ничего не предлагает, target делает обычный шаг.'
        : `Черновик дёшево угадывает ${fmt.count(n, ['токен', 'токена', 'токенов'])} вперёд, один за другим. Пока это только догадки.`
    if (v.phase === 1)
      return n === 0
        ? 'Target делает обычный шаг decode: проверять нечего.'
        : `Target один раз прогоняет все ${fmt.count(n + 1, ['позицию', 'позиции', 'позиций'])} сразу, как маленький prefill, и сравнивает свои предсказания с черновиком.`
    if (n === 0) return 'Последний токен ответа: черновика не было, target сделал обычный шаг decode: 1 токен за проход.'
    return r.targetKind === 'bonus'
      ? `Приняты все ${fmt.count(r.accepted, ['черновик', 'черновика', 'черновиков'])}, и target бесплатно добавляет свой следующий токен: ${fmt.count(r.produced, ['токен', 'токена', 'токенов'])} за один проход.`
      : `Принято ${fmt.count(r.accepted, ['черновик', 'черновика', 'черновиков'])} до первого расхождения, на месте ошибки встаёт токен target, остальное выброшено: ${fmt.count(r.produced, ['токен', 'токена', 'токенов'])} за проход.`
  })()
  const targetLabel = !v.round ? '' : v.phase === 1 ? 'токен target' : v.round.drafts.length === 0 ? 'токен target' : v.round.targetKind === 'bonus' ? 'бонус от target' : 'исправление target'
  const passesWord = fmt.count(done, ['проход', 'прохода', 'проходов']).replace(/^\S+\s/, '')

  return (
    <Figure
      id="speculative-figure-rounds"
      title="Черновик предлагает, target проверяет за один проход"
      player={<Player {...p.bind} labels={labels} aria-label="Шаги спекуляции" />}
      controls={
        <>
          <Slider label="Acceptance rate α" value={alpha} min={0.3} max={0.95} step={0.05} onChange={setAlpha} format={(x) => fmt.num(x, 2)} hint="вероятность, что очередной черновой токен совпадёт" />
          <Slider label="Длина черновика k" value={k} min={1} max={8} step={1} onChange={setK} format={(x) => fmt.int(x)} hint="сколько токенов угадывать за раз" />
        </>
      }
      caption="Попробуйте: поставьте α = 0,95 и k = 8, затем α = 0,4. Чем предсказуемее текст, тем длиннее принятые серии; при плохом черновике серии короткие: в среднем меньше двух токенов за проход, хотя черновик каждый раз предлагает восемь."
      source={
        fromRun
          ? 'Ответ взят из симулятора: начало заготовленного ответа на ваш запрос (текст условный), токены настоящие (словарь Qwen3). Принятие случайное с вероятностью α.'
          : 'Ответ модели условный, токены настоящие (словарь Qwen3). Принятие случайное с вероятностью α.'
      }
    >
      <div className="grid min-w-0 gap-5">
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @lg:grid-cols-4">
          <Stat label="Проходов target" value={fmt.int(v.passes)} swatch="var(--c-decode)" />
          <Stat label="Токенов ответа готово" value={fmt.int(done)} unit={`из ${fmt.int(answer.length)}`} />
          <Stat label="Токенов за проход" value={completed ? fmt.num(perPass, 2) : '0'} hint={`в среднем по формуле ${fmt.num(expectedTokens(alpha, k), 2)}`} />
          <Stat label="Без спекуляции" value={fmt.int(done)} unit={passesWord} hint="один токен за проход" />
        </div>

        <div className="min-w-0">
          <div className="mb-1.5 text-sm font-medium text-ink-2">{fromRun ? 'Ответ (начало ответа на ваш запрос в симуляторе)' : 'Ответ'}</div>
          <div className="flex min-h-[3.5rem] flex-wrap gap-x-[3px] gap-y-1.5 rounded-control bg-surface-2/60 p-2.5" aria-live="off">
            {answer.slice(0, done).map((t, i) => {
              const isFresh = v.fresh !== null && i >= v.fresh[0] && i < v.fresh[1]
              return (
                <motion.span key={i} initial={reduce ? false : { opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
                  <Chip token={t} look={isFresh ? 'fresh' : 'committed'} />
                </motion.span>
              )
            })}
            {done === 0 && <span className="self-center px-1 text-sm text-ink-3">пока пусто</span>}
          </div>
        </div>

        <div className="min-w-0">
          <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <span className="text-sm font-medium text-ink-2">
              {v.round ? `Проход ${Math.min(rounds.length, Math.floor(p.step / 3) + 1)} из ${rounds.length}: ${PHASES[v.phase]}` : ''}
            </span>
            <span className="text-[13px] text-ink-3">{v.phase === 0 ? `${fmt.count(v.round?.drafts.length ?? 0, ['шаг', 'шага', 'шагов'])} маленькой модели` : '1 проход большой модели'}</span>
          </div>
          <div
            className={cn('flex min-h-[4.75rem] flex-wrap items-start gap-x-1.5 gap-y-2 rounded-control p-2.5 transition-colors', v.phase === 1 ? 'shadow-[inset_0_0_0_1.5px_var(--accent)]' : 'shadow-[inset_0_0_0_1px_var(--line)]')}
          >
            {v.round?.drafts.map((d, i) => {
              const verdictShown = v.phase >= 1
              const truth = answer[d.pos]
              const tok = d.verdict === 'accepted' ? truth : wrong(d.pos, i)
              const look = !verdictShown ? 'draft' : d.verdict === 'accepted' ? (v.phase === 2 ? 'committed' : 'draft') : d.verdict === 'rejected' ? 'rejected' : 'discarded'
              return (
                <motion.span
                  key={`${v.round!.start}-${i}`}
                  className="inline-flex flex-col items-center gap-1"
                  initial={reduce ? false : { opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.22, delay: reduce || v.phase > 0 ? 0 : i * 0.12 }}
                >
                  <Chip token={tok} look={look} />
                  <VerdictMark verdict={verdictShown ? d.verdict : null} />
                </motion.span>
              )
            })}
            {v.round && v.phase >= 1 && (
              <motion.span
                key={`${v.round.start}-target`}
                className="inline-flex flex-col items-center gap-1"
                initial={reduce ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.25 }}
              >
                {v.phase === 1 ? (
                  <span className="inline-block rounded-chip px-2 font-mono text-[13px] leading-[1.6] text-ink-3 shadow-[inset_0_0_0_1px_var(--line-strong)]">?</span>
                ) : (
                  <Chip token={answer[v.round.targetPos]} look="committed" className="shadow-[inset_0_0_0_1.5px_var(--c-decode)]" />
                )}
                <span className="text-[11.5px] leading-none whitespace-nowrap text-ink-2">{targetLabel}</span>
              </motion.span>
            )}
          </div>
          <p className="mt-2 min-h-[2.75rem] text-[14px] leading-snug text-ink-2">{caption}</p>
        </div>

        <PassBars rounds={rounds} upto={v.passes} completed={completed} />

        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
          <LegendItem swatch={<span className="h-3 w-4 rounded-[3px] [outline:1.5px_dashed_var(--ink-3)] -outline-offset-[1.5px]" />}>черновик, не проверен</LegendItem>
          <LegendItem swatch={<CheckCircleIcon size={16} weight="fill" className="text-good" aria-hidden />}>принят</LegendItem>
          <LegendItem swatch={<XCircleIcon size={16} weight="fill" className="text-bad" aria-hidden />}>отклонён</LegendItem>
          <LegendItem swatch={<MinusCircleIcon size={16} className="text-ink-3" aria-hidden />}>выброшен без проверки</LegendItem>
          <LegendItem swatch={<span className="h-3 w-4 rounded-[3px]" style={{ background: tint('decode', 26) }} />}>токены ответа</LegendItem>
        </div>
      </div>
    </Figure>
  )
}

function VerdictMark({ verdict }: { verdict: SpecRound['drafts'][number]['verdict'] | null }) {
  if (!verdict) return <span className="h-[14px] text-[11.5px] leading-none text-ink-3">?</span>
  if (verdict === 'accepted')
    return (
      <span className="inline-flex items-center gap-0.5 text-[11.5px] leading-none text-ink-2">
        <CheckCircleIcon size={14} weight="fill" className="text-good" aria-hidden />
        принят
      </span>
    )
  if (verdict === 'rejected')
    return (
      <span className="inline-flex items-center gap-0.5 text-[11.5px] leading-none text-ink-2">
        <XCircleIcon size={14} weight="fill" className="text-bad" aria-hidden />
        отклонён
      </span>
    )
  return (
    <span className="inline-flex items-center gap-0.5 text-[11.5px] leading-none text-ink-3">
      <MinusCircleIcon size={14} aria-hidden />
      выброшен
    </span>
  )
}

/** Квадрат токена target: насыщенный цвет decode и рамка цветом текста (видна в обеих темах). */
const TARGET_SQUARE = { background: 'var(--c-decode)', boxShadow: 'inset 0 0 0 1.5px var(--ink)' }
/** Квадрат принятого черновика: непрозрачная смесь decode с фоном, без рамки. */
const DRAFT_SQUARE = { background: mix('decode', 55) }

/** Столбики: сколько токенов дал каждый проход target (без спекуляции каждый был бы высотой 1). */
function PassBars({ rounds, upto, completed }: { rounds: SpecRound[]; upto: number; completed: number }) {
  const maxH = rounds.reduce((m, r) => Math.max(m, r.produced), 1)
  const sq = Math.max(6, Math.min(10, 60 / maxH))
  return (
    <div className="min-w-0">
      <div className="mb-1.5 text-sm font-medium text-ink-2">Токенов за каждый проход target</div>
      <div
        className="relative flex h-[76px] items-end gap-[2px] border-b border-line-strong pb-1"
        role="img"
        aria-label={`Проходов ${completed}, токенов за проход: ${rounds.slice(0, completed).map((r) => r.produced).join(', ')}`}
      >
        {upto === 0 && <span className="absolute inset-x-0 bottom-2 text-center text-[13px] text-ink-3">проходов пока не было</span>}
        {rounds.map((r, i) => {
          const settled = i < completed
          const inProgress = !settled && i < upto
          return (
            <div key={i} className="flex h-full max-w-[28px] min-w-[4px] flex-1 flex-col justify-end gap-[2px]" title={settled ? `Проход ${i + 1}: ${r.produced}` : undefined}>
              {inProgress ? (
                // идущий проход: рамка без заливки, итог ещё не известен
                <span className="block w-full rounded-[2px] [outline:1.5px_dashed_var(--ink-3)] -outline-offset-[1.5px]" style={{ height: `${sq * 2}px` }} />
              ) : (
                Array.from({ length: r.produced }, (_, j) => (
                  <span
                    key={j}
                    className="block w-full rounded-[2px] transition-opacity duration-300"
                    style={{ ...(j === r.produced - 1 ? TARGET_SQUARE : DRAFT_SQUARE), opacity: settled ? 1 : 0, height: `${sq}px` }}
                  />
                ))
              )}
            </div>
          )
        })}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
        <LegendItem swatch={<span className="h-2.5 w-3.5 rounded-[2px]" style={TARGET_SQUARE} />}>квадрат с рамкой внизу столбика: токен, который target посчитал сам</LegendItem>
        <LegendItem swatch={<span className="h-2.5 w-3.5 rounded-[2px]" style={DRAFT_SQUARE} />}>без рамки над ним: принятые черновики</LegendItem>
        <LegendItem swatch={<span className="h-2.5 w-3.5 rounded-[2px] [outline:1.5px_dashed_var(--ink-3)] -outline-offset-[1.5px]" />}>идёт проверка</LegendItem>
      </div>
    </div>
  )
}
