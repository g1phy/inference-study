import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { ArrowCounterClockwiseIcon, DiceFiveIcon, DiceSixIcon, ShuffleIcon } from '@phosphor-icons/react'
import { Button, FieldLabel, Figure, Slider, Stat, TokenChip, fmt, mix } from '@/ui'
import { cn } from '@/ui/cn'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { mulberry32, type Rng } from '@/engine/rng'
import { computeDist, pickRow, type DistParams, type DistResult } from './dist'
import { CONTEXT, HEAD, TAIL, TAIL_EXAMPLES } from './data'
import { DistBars, rowLabel } from './DistBars'

const DEFAULT_SEED = 42
const FIGURE_ID = 'sampling-figure-playground'
/** С этой ширины фигуры панель контролов стоит справа (как у Figure controlsPosition="side"). */
const SIDE_MIN_WIDTH = 762

/** «бит» / «бита» по значению, как оно показано (2 знака). */
const bits = (h: number) => fmt.plural(Math.round(h * 100) / 100, ['бит', 'бита', 'бит'])
/** Число «равновероятных вариантов» 2^H: до тысячи с одним знаком, дальше компактно. */
const variants = (h: number) => {
  const x = 2 ** h
  return x < 1000 ? fmt.num(x, 1) : fmt.compact(x)
}

interface Draw {
  row: number
  tailIndex?: number
  u: number
  n: number
  /** Настройки, при которых сделан бросок. */
  sig: string
}

const PRESETS: Array<{ id: string; label: string; params: DistParams }> = [
  { id: 'greedy', label: 'Greedy', params: { temperature: 0, topK: 0, topP: 1, minP: 0 } },
  { id: 'raw', label: 'T = 1 без отсечений', params: { temperature: 1, topK: 0, topP: 1, minP: 0 } },
  { id: 'qwen', label: 'Qwen3: T 0,6, top-p 0,95, top-k 20', params: { temperature: 0.6, topK: 20, topP: 0.95, minP: 0 } },
  { id: 'hot', label: 'T = 1,5 без отсечений', params: { temperature: 1.5, topK: 0, topP: 1, minP: 0 } },
]

/** Широкая ли фигура: на узкой контролы встают сразу под столбиками, а не в самый низ. */
function useWideFigure(id: string): boolean {
  const [wide, setWide] = useState(() => typeof window !== 'undefined' && window.innerWidth >= 1000)
  useLayoutEffect(() => {
    const el = document.getElementById(id)
    if (!el) return
    const update = () => setWide(el.getBoundingClientRect().width >= SIDE_MIN_WIDTH)
    update()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [id])
  return wide
}

/** Подпись выбранной строки: кандидат или пример токена из хвоста. */
function pickText(d: DistResult, pick: { row: number; tailIndex?: number }): string {
  const r = d.rows[pick.row]
  if (!r) return ''
  return r.tail ? TAIL_EXAMPLES[(pick.tailIndex ?? 0) % TAIL_EXAMPLES.length] : r.text
}

export function SamplingFigure() {
  const [settings] = useSettings()
  const reduce = useReducedMotion()
  const { tokenizer } = useTokenizer()
  const wide = useWideFigure(FIGURE_ID)
  const [params, setParams] = useState<DistParams>(() => ({ temperature: settings.temperature, topK: 0, topP: 1, minP: 0 }))
  const set = (patch: Partial<DistParams>) => setParams((p) => ({ ...p, ...patch }))
  const d = useMemo(() => computeDist(HEAD, TAIL, params), [params])
  const sig = `${params.temperature}|${params.topK}|${params.topP}|${params.minP}`

  const [seed, setSeed] = useState(DEFAULT_SEED)
  const rng = useRef<{ seed: number; next: Rng; n: number }>({ seed: DEFAULT_SEED, next: mulberry32(DEFAULT_SEED), n: 0 })
  const [draws, setDraws] = useState<Draw[]>([])
  const [batch, setBatch] = useState<{ sig: string; counts: number[]; total: number } | null>(null)

  const drawOnce = (): Draw => {
    const r = rng.current
    const u = r.next()
    r.n++
    const pick = pickRow(d, u)
    return { row: pick.row, tailIndex: pick.tailIndex, u, n: r.n, sig }
  }
  // бросок вне функции обновления состояния: в StrictMode она вызывается дважды и сдвинула бы генератор
  const sampleOne = () => {
    const dr = drawOnce()
    setDraws((prev) => [dr, ...prev].slice(0, 10))
  }
  const sampleMany = () => {
    const counts = d.rows.map(() => 0)
    let last: Draw | null = null
    for (let i = 0; i < 100; i++) {
      last = drawOnce()
      counts[last.row]++
    }
    setBatch((b) => (b && b.sig === sig ? { sig, counts: b.counts.map((c, i) => c + counts[i]), total: b.total + 100 } : { sig, counts, total: 100 }))
    if (last) setDraws((prev) => [last!, ...prev].slice(0, 10))
  }
  const restart = (s = seed) => {
    rng.current = { seed: s, next: mulberry32(s), n: 0 }
    setDraws([])
    setBatch(null)
  }

  const lastDraw = draws[0] ?? null
  // настройки поменялись после броска: то же u заново проходит по новому распределению, чтобы полоса, строка и подпись не спорили
  const stale = lastDraw !== null && lastDraw.sig !== sig
  const pick = lastDraw ? (stale ? pickRow(d, lastDraw.u) : { row: lastDraw.row, tailIndex: lastDraw.tailIndex }) : null
  const chosenRow = pick ? d.rows[pick.row] : null
  const chosenText = pick ? pickText(d, pick) : null
  const chosenShown = chosenRow ? (chosenRow.tail ? `${(chosenText ?? '').replace(/^ /, '·')} (из хвоста)` : rowLabel(chosenRow)) : ''
  const counts = batch && batch.sig === sig ? batch : null
  const parisTokens = useMemo(() => (tokenizer ? tokenizer.encode(' Париж') : null), [tokenizer])

  const controls: ReactNode = (
    <>
      <Slider
        label="Temperature"
        value={params.temperature}
        min={0}
        max={2}
        step={0.05}
        onChange={(v) => set({ temperature: v })}
        format={(v) => (v === 0 ? '0, greedy' : fmt.num(v, 2))}
        hint="0: всегда самый вероятный"
      />
      <Slider label="top-k" value={params.topK} min={0} max={50} onChange={(v) => set({ topK: v })} format={(v) => (v === 0 ? 'выкл' : fmt.int(v))} hint="сколько лучших оставить" />
      <Slider
        label="top-p"
        value={params.topP}
        min={0.05}
        max={1}
        step={0.01}
        onChange={(v) => set({ topP: v })}
        format={(v) => (v >= 1 ? 'выкл' : fmt.num(v, 2))}
        hint="сколько вероятности набрать"
      />
      <Slider
        label="min-p"
        value={params.minP}
        min={0}
        max={0.3}
        step={0.01}
        onChange={(v) => set({ minP: v })}
        format={(v) => (v === 0 ? 'выкл' : fmt.num(v, 2))}
        hint="порог от лидера"
      />
      <div className="grid gap-1.5">
        <FieldLabel>Готовые настройки</FieldLabel>
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((p) => {
            const on = p.params.temperature === params.temperature && p.params.topK === params.topK && p.params.topP === params.topP && p.params.minP === params.minP
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => setParams(p.params)}
                aria-pressed={on}
                className={cn(
                  'min-h-8 rounded-chip px-2.5 py-1 text-left text-[13px] leading-snug transition-colors',
                  on ? 'bg-accent-soft text-accent-hover shadow-[inset_0_0_0_1px_var(--accent)]' : 'bg-surface-2 text-ink-2 shadow-[inset_0_0_0_1px_var(--line)] hover:text-ink',
                )}
              >
                {p.label}
              </button>
            )
          })}
        </div>
      </div>
    </>
  )

  return (
    <Figure
      id={FIGURE_ID}
      title="Логиты → вероятности → выбор одного токена"
      controls={wide ? controls : undefined}
      controlsPosition="side"
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> поднимите temperature до 1,5 и посмотрите на строку «ещё 151 924»: каждый
          токен хвоста маловероятен, но их так много, что в этом примере вместе они забирают больше половины вероятности. Включите top-p 0,9:
          хвост почти не уменьшится, потому что top-p отбрасывает только последние 10 % вероятности, а хвост весит больше. Верните top-p на
          «выкл» и включите top-k 20 или min-p 0,05: хвост сожмётся почти до нуля. Логиты условные; кандидаты для простоты показаны целыми
          словами.
        </>
      }
    >
      <div className="grid min-w-0 gap-5">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[17px] leading-snug">
          <span className="text-ink-2">Контекст:</span>
          <span className="font-medium text-ink">{CONTEXT}</span>
          {chosenText !== null && (
            <motion.span
              key={`${lastDraw?.n}-${seed}-${pick?.row}-${pick?.tailIndex}`}
              initial={reduce ? false : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-chip px-1.5 font-mono text-[15px] text-ink"
              style={{ background: mix('decode', 22) }}
            >
              {chosenText.replace(/^ /, '·')}
            </motion.span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button variant="primary" icon={DiceFiveIcon} onClick={sampleOne}>
            Сэмплировать
          </Button>
          <Button icon={ShuffleIcon} onClick={sampleMany}>
            100 раз
          </Button>
          <Button variant="ghost" icon={ArrowCounterClockwiseIcon} onClick={() => restart()}>
            Сначала, seed {seed}
          </Button>
          <Button
            variant="ghost"
            icon={DiceSixIcon}
            onClick={() => {
              setSeed(seed + 1)
              restart(seed + 1)
            }}
          >
            Новый seed
          </Button>
        </div>

        <DistBars
          rows={d.rows}
          chosen={pick ? pick.row : null}
          greedy={d.greedy}
          temperature={params.temperature}
          counts={counts?.counts}
          total={counts?.total}
        />

        {!wide && <div className="grid gap-4 rounded-panel bg-surface-2 p-3">{controls}</div>}

        <CumulativeBar rows={d.rows} u={lastDraw?.u ?? null} greedy={d.greedy} />

        <div className="grid gap-1 text-sm text-ink-2" aria-live="polite">
          {lastDraw ? (
            <p>
              Бросок №{lastDraw.n} с seed {seed}: {d.greedy ? 'при T = 0 случайность не нужна, берём самый вероятный' : <>u = {fmt.num(lastDraw.u, 3)}, попало в отрезок</>}{' '}
              <b className="font-mono font-semibold text-ink">{chosenShown}</b>.
              {stale && <> Настройки изменились после броска{d.greedy ? '' : ', и тот же u пересчитан по новому распределению'}.</>}
              {draws.length > 1 && (
                <span className="text-ink-3"> Предыдущие: {draws.slice(1, 8).map((x) => (d.rows[x.row]?.tail ? 'хвост' : (d.rows[x.row]?.text ?? '').trim())).join(', ')}.</span>
              )}
            </p>
          ) : (
            <p>Нажмите «Сэмплировать»: генератор с seed {seed} выдаст число u от 0 до 1, и токен выберется по отрезкам полосы выше.</p>
          )}
          {counts && (
            <p>
              Из {fmt.int(counts.total)} бросков:{' '}
              {d.rows
                .map((r, i) => ({ r, c: counts.counts[i] }))
                .filter((x) => x.c > 0)
                .sort((a, b) => b.c - a.c)
                .map((x) => `${x.r.tail ? 'хвост' : x.r.text.trim()} ${fmt.int(x.c)}`)
                .join(', ')}
              . Чем больше бросков, тем ближе штрихи на столбиках к вероятностям.
            </p>
          )}
        </div>

        <div className="grid grid-cols-1 gap-x-6 gap-y-4 border-t border-line pt-4 @xl:grid-cols-3">
          <Stat
            label="Энтропия до отсечений"
            value={fmt.num(d.entropyBefore, 2)}
            unit={bits(d.entropyBefore)}
            hint={d.greedy ? 'greedy: выбор предрешён' : `как выбор из ${variants(d.entropyBefore)} равновероятных вариантов`}
          />
          <Stat
            label="Энтропия после"
            value={fmt.num(d.entropyAfter, 2)}
            unit={bits(d.entropyAfter)}
            hint={d.entropyAfter === 0 ? 'выбор предрешён' : `как выбор из ${variants(d.entropyAfter)} равновероятных вариантов`}
          />
          <Stat label="Осталось токенов" value={fmt.int(d.keptTokens)} hint={`из ${fmt.int(HEAD.length + TAIL.count)} строк логитов`} />
        </div>

        <p className="text-sm leading-snug text-ink-3">
          Упрощение. В словаре Qwen3 «·Париж» это не один токен, а три:{' '}
          {parisTokens ? (
            <span className="inline-flex flex-wrap gap-[3px] align-middle">
              {parisTokens.map((t, i) => (
                <TokenChip key={i} token={t} variant="generated" size="sm" showId />
              ))}
            </span>
          ) : (
            '«·П», «ари», «ж»'
          )}
          . Настоящая модель сначала выбирает «·П», а следующие шаги почти без вариантов дописывают остальное.
        </p>
      </div>
    </Figure>
  )
}

/** Полоса итогового распределения: отрезки оставшихся токенов и отметка u. */
function CumulativeBar({ rows, u, greedy }: { rows: ReturnType<typeof computeDist>['rows']; u: number | null; greedy: boolean }) {
  const reduce = useReducedMotion()
  const segs = rows.map((r, i) => ({ r, i })).filter((s) => s.r.after > 0)
  return (
    <div className="grid gap-1.5">
      <div className="text-xs text-ink-3">Итоговое распределение одной полосой: u выбирает отрезок</div>
      <div className="relative h-8 min-w-0">
        <div className="absolute inset-0 flex overflow-hidden rounded-control shadow-[inset_0_0_0_1px_var(--line)]" role="img" aria-label="Итоговые вероятности отрезками">
          {segs.map((s, k) => {
            const w = s.r.after * 100
            return (
              <div
                key={s.r.key}
                title={`${s.r.tail ? 'хвост' : s.r.text.trim()}: ${fmt.pct(s.r.after)}`}
                className={cn('relative h-full min-w-0 border-r border-surface last:border-r-0', s.r.tail && 'bg-hatch')}
                style={{ width: `${w}%`, backgroundColor: mix('decode', s.r.tail ? 30 : k % 2 === 0 ? 62 : 40) }}
              >
                {w > 9 && (
                  <span className="absolute inset-0 flex items-center justify-center truncate px-1 font-mono text-[12px] text-ink" style={{ textShadow: '0 0 3px var(--surface)' }}>
                    {s.r.tail ? 'хвост' : s.r.text.trim()}
                  </span>
                )}
              </div>
            )
          })}
        </div>
        {u !== null && !greedy && (
          // обёртка шириной с полосу: сдвиг на u·100% собственной ширины = u·ширина полосы (анимируется transform)
          <motion.div
            className="pointer-events-none absolute -top-1.5 bottom-[-6px] left-0 w-full"
            initial={false}
            animate={{ x: `${u * 100}%` }}
            transition={{ duration: reduce ? 0 : 0.35 }}
            aria-hidden
          >
            <div className="absolute top-0 bottom-0 left-0 -ml-px w-[2px] bg-ink" />
            <div className="absolute -top-[2px] left-0 -ml-[5px] h-0 w-0 border-x-[5px] border-t-[6px] border-x-transparent border-t-ink" />
          </motion.div>
        )}
      </div>
      <div className="flex justify-between text-xs text-ink-3 tabular-nums" aria-hidden>
        <span>0</span>
        <span>1</span>
      </div>
    </div>
  )
}
