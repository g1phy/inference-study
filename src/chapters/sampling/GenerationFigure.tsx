import { useMemo, useState } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { ShuffleIcon, WarningIcon } from '@phosphor-icons/react'
import { BarChart, Button, Figure, Player, Segmented, Toggle, fmt, mix, usePlayer } from '@/ui'
import { cn } from '@/ui/cn'
import { useSettings } from '@/state/settings'
import { mulberry32 } from '@/engine/rng'
import { MAX_STEPS, ODD_WORDS, TOY_PREFIX, runToy, type ToyRun } from './toylm'

/** Слова фразы группами: токен без пробела впереди (знак препинания) приклеен к предыдущему слову. */
function wordGroups(words: readonly string[]): Array<{ start: number; words: string[] }> {
  const groups: Array<{ start: number; words: string[] }> = []
  words.forEach((w, i) => {
    if (groups.length && !w.startsWith(' ')) groups[groups.length - 1].words.push(w)
    else groups.push({ start: i, words: [w] })
  })
  return groups
}

interface Lane {
  id: string
  T: number
  label: string
}

export function GenerationFigure() {
  const [settings] = useSettings()
  const reduce = useReducedMotion()
  const [seed, setSeed] = useState(7)
  const [nucleus, setNucleus] = useState(false)
  const userT = settings.temperature
  // средняя дорожка берёт T из симулятора, но не повторяет соседние (0 и 1,5): тогда сравнение было бы из двух дорожек
  const midT = userT === 0 || userT === 1.5 ? 0.7 : userT
  const lanes: Lane[] = useMemo(
    () => [
      { id: 'greedy', T: 0, label: 'T = 0 (greedy)' },
      { id: 'user', T: midT, label: midT === userT ? `T = ${fmt.num(midT)}, как в симуляторе` : `T = ${fmt.num(midT)}` },
      { id: 'hot', T: 1.5, label: 'T = 1,5' },
    ],
    [midT, userT],
  )
  const us = useMemo(() => {
    const rng = mulberry32(seed)
    return Array.from({ length: MAX_STEPS }, () => rng())
  }, [seed])
  const runs: ToyRun[] = useMemo(() => lanes.map((l) => runToy(us, { temperature: l.T, topK: 0, topP: nucleus ? 0.9 : 1 })), [lanes, us, nucleus])
  const maxLen = Math.max(...runs.map((r) => r.words.length))
  // сначала показываем готовые фразы целиком, проигрывание начинается с нуля
  const p = usePlayer(maxLen + 1, { intervalMs: 1100, initialStep: maxLen })
  // Новый seed, top-p или температура из симулятора: снова показываем фразы целиком, как при открытии. Иначе плеер
  // оставался на старом шаге, и новая фраза длиннее прежних обрывалась на полуслове. Поправка прямо в рендере, а не
  // в эффекте: React сразу перерисует фигуру, и кадр с оборванной фразой не попадёт на экран.
  const draw = `${seed}|${nucleus}|${midT}`
  const [shownDraw, setShownDraw] = useState(draw)
  if (shownDraw !== draw) {
    setShownDraw(draw)
    p.setPlaying(false) // и снимает автопаузу: фигура, ушедшая с экрана во время проигрывания, не запустится сама
    p.setStep(maxLen)
  }
  const step = Math.min(p.step, maxLen)
  const [laneId, setLaneId] = useState('user')
  const laneIdx = Math.max(0, lanes.findIndex((l) => l.id === laneId))
  const run = runs[laneIdx]
  const detailStep = Math.min(step, run.steps.length) - 1
  const ds = detailStep >= 0 ? run.steps[detailStep] : null
  const labels = useMemo(() => Array.from({ length: maxLen + 1 }, (_, i) => (i === 0 ? 'только начало фразы' : `сгенерировано слов: ${i}`)), [maxLen])

  return (
    <Figure
      id="sampling-figure-generation"
      title="Одна фраза, три температуры: генерация по шагам"
      controls={<Player {...p.bind} labels={labels} aria-label="Генерация по шагам" />}
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> пройдите шаги до конца, затем нажмите «Другой seed» несколько раз. Дорожки
          получают одни и те же случайные числа u, отличается только температура. Включите top-p 0,9 и сравните дорожку T = 1,5. Игрушечная
          модель: распределения заскриптованы по словам, а выбор делает настоящий код сэмплинга сайта, тот же, что в симуляторе.
        </>
      }
    >
      <div className="grid min-w-0 gap-5">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
          <Button icon={ShuffleIcon} onClick={() => setSeed((s) => s + 1)}>
            Другой seed
          </Button>
          <span className="text-sm text-ink-2 tabular-nums">seed {seed}</span>
          <Toggle label="top-p 0,9 для всех" checked={nucleus} onChange={setNucleus} />
        </div>

        <ol className="grid min-w-0 gap-3">
          {lanes.map((lane, li) => {
            const r = runs[li]
            const shown = r.words.slice(0, step)
            const done = step >= r.words.length
            return (
              <li
                key={lane.id}
                className={cn(
                  'grid min-w-0 gap-1.5 rounded-panel px-3 py-2.5 transition-colors @2xl:grid-cols-[190px_minmax(0,1fr)] @2xl:gap-4',
                  li === laneIdx ? 'bg-surface-2 shadow-[inset_0_0_0_1px_var(--line-strong)]' : 'bg-transparent shadow-[inset_0_0_0_1px_var(--line)]',
                )}
              >
                <button type="button" onClick={() => setLaneId(lane.id)} className="justify-self-start text-left text-sm font-semibold text-ink">
                  {lane.label}
                  <span className="block text-xs font-normal text-ink-3">
                    {done ? (r.finished ? `фраза закончена за ${fmt.int(r.words.length)} ${fmt.plural(r.words.length, ['шаг', 'шага', 'шагов'])}` : 'достигнут лимит шагов') : `шаг ${Math.min(step, r.words.length)}`}
                  </span>
                </button>
                <p className="min-w-0 text-[16px] leading-[1.9]">
                  <span className="text-ink-2">{TOY_PREFIX}</span>
                  {/* токен без пробела впереди (запятая, точка) держится за предыдущее слово: группа не переносится,
                      и строка на телефоне не начинается с запятой */}
                  {wordGroups(shown).map((g) => (
                    <span key={`${seed}-g${g.start}`} className={cn('whitespace-nowrap', g.words[0].startsWith(' ') && 'ml-[0.3em]')}>
                      {g.words.map((w, j) => {
                        const i = g.start + j
                        const odd = ODD_WORDS.has(w)
                        const isNew = i === step - 1
                        return (
                          <motion.span
                            key={`${seed}-${i}-${w}`}
                            initial={isNew && !reduce ? { opacity: 0, y: 3 } : false}
                            animate={{ opacity: 1, y: 0 }}
                            className={cn('inline-block rounded-chip px-1 text-ink', isNew && 'shadow-[inset_0_0_0_1.5px_var(--accent)]')}
                            style={{ background: mix('decode', 20) }}
                          >
                            {w.trim()}
                            {odd && <WarningIcon size={14} aria-label="сомнительное слово" className="ml-0.5 inline align-[-2px] text-warn" />}
                          </motion.span>
                        )
                      })}
                    </span>
                  ))}
                  {!done && <span aria-hidden className="ml-1 inline-block h-[1.1em] w-[2px] translate-y-[3px] bg-ink-3" />}
                </p>
              </li>
            )
          })}
        </ol>

        <div className="grid min-w-0 gap-3 rounded-panel bg-surface-2 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Segmented
              size="sm"
              aria-label="Дорожка для подробностей"
              value={laneId}
              onChange={setLaneId}
              options={lanes.map((l) => ({ value: l.id, label: l.id === 'greedy' ? 'T = 0' : `T = ${fmt.num(l.T)}` }))}
            />
            <span className="text-sm text-ink-2">
              {ds ? (
                <>
                  Шаг {detailStep + 1}:{' '}
                  {ds.u !== undefined ? <>u = {fmt.num(ds.u, 3)}, выбрано</> : <>greedy, выбрано</>}{' '}
                  <b className="font-mono font-semibold text-ink">«{ds.edges[ds.chosen].w.trim()}»</b> с вероятностью {fmt.pct(ds.finalProbs[ds.chosen])}
                </>
              ) : (
                'Сделайте шаг, чтобы увидеть распределение'
              )}
            </span>
          </div>
          {ds ? (
            <BarChart
              data={ds.edges.map((e, i) => ({
                label: i === ds.chosen ? `${e.w.trim()} (выбрано)` : e.w.trim(),
                value: ds.finalProbs[i],
                muted: !ds.kept[i],
                id: `${detailStep}-${e.w}`,
              }))}
              max={1}
              valueFormat={(v) => fmt.pct(v)}
              labelFont="mono"
              color="decode"
              aria-label="Кандидаты на текущем шаге выбранной дорожки"
            />
          ) : (
            <div className="h-[120px]" />
          )}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
            <span className="flex items-center gap-1.5">
              <WarningIcon size={14} aria-hidden className="text-warn" />
              фактическая ошибка или бессмыслица
            </span>
            {nucleus && <span>приглушённые столбики отсечены top-p</span>}
          </div>
        </div>
      </div>
    </Figure>
  )
}
