import { Fragment, type CSSProperties, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { CheckIcon } from '@phosphor-icons/react'
import { cn, cssVar, Figure, fmt, mix, Player, ScrollX, Stat, tint, useElementSize, usePlayer } from '@/ui'
import { Swatch, VLabel } from './bits'
import { decodeRow, N, PROMPT_LEN, TRACE, trianglePairs, WORDS } from './toy'

/** Шаги: 0 prefill промпта, 1 и 2 шаги decode. */
const STEPS = [
  { label: 'Prefill: весь промпт за один проход', last: PROMPT_LEN - 1 },
  { label: 'Decode, шаг 1: токен «она»', last: PROMPT_LEN },
  { label: 'Decode, шаг 2: токен «устала»', last: PROMPT_LEN + 1 },
]

const CELL_W = 38
/** Узкий экран: треугольник из ячеек 21 px с числами без «%» (подпись «веса, %» в шапке), влезает в 375 px без прокрутки. */
const CELL_W_NARROW = 21
const CELL_H = 28
const HEAD = 84

const vec = (v: readonly number[]) => v.map((x) => fmt.num(x)).join('; ')
/** Подпись слова в вертикальной шапке: голая запятая там читается как соринка. */
const colWord = (w: string) => (w === ',' ? 'запятая' : w)

/** Пометка «прочитано из кэша на этом шаге»: пунктир акцентом внутри ячейки. */
const READ_MARK = 'outline-[1.5px] outline-dashed outline-accent -outline-offset-[3px]'

/**
 * Строки decode считаются по кэшу: q нового токена против K из кэша, softmax, смесь V из кэша (toy.decodeRow).
 * Строки промпта берутся из полного прохода prefill. Тест проверяет, что строки совпадают.
 */
const DECODE = new Map(Array.from({ length: N - PROMPT_LEN }, (_, k) => [PROMPT_LEN + k, decodeRow(PROMPT_LEN + k)] as const))
const rowWeights = (i: number) => DECODE.get(i)?.weights ?? TRACE.weights[i]
const rowQ = (i: number) => DECODE.get(i)?.q ?? TRACE.Q[i]
const rowK = (i: number) => DECODE.get(i)?.k ?? TRACE.K[i]
const rowV = (i: number) => DECODE.get(i)?.v ?? TRACE.V[i]
/** Строка decode по кэшу совпадает со строкой полного треугольника (до ошибки округления float). */
const sameAsFull = (i: number) => rowWeights(i).every((w, j) => Math.abs(w - TRACE.weights[i][j]) < 1e-12)

export function BridgeFigure({ id }: { id: string }) {
  const p = usePlayer(STEPS.length, { intervalMs: 4200 })
  const [bodyRef, { width: bodyW }] = useElementSize<HTMLDivElement>()
  const step = p.step
  const last = STEPS[step].last
  const first = step === 0 ? 0 : last
  const isNew = (i: number) => i >= first && i <= last
  const exists = (i: number) => i <= last
  const phase = (i: number) => (i < PROMPT_LEN ? 'prefill' : 'decode')

  const newTokens = last - first + 1
  const pairs = step === 0 ? trianglePairs(PROMPT_LEN) : last + 1
  const readFromCache = step === 0 ? 0 : last
  const noCachePairs = trianglePairs(last + 1)

  const narration: ReactNode[] = [
    <>
      Все {PROMPT_LEN} слов промпта проходят через модель одним проходом: считается весь нижний треугольник {PROMPT_LEN} × {PROMPT_LEN}, а K и V каждого слова
      сразу ложатся в кэш. В конце prefill выдаёт первый новый токен: «она».
    </>,
    <>
      Через модель идёт только «она». Для неё считаются q, k и v; k и v дописываются в кэш, а q сравнивается с {last + 1} ключами: {last} из кэша и своим.
      Получается одна новая строка, ровно та же, что в первой фигуре: «Кошке» {fmt.pct(rowWeights(last)[0])}.
    </>,
    <>
      То же для «устала»: одна строка из {last + 1} ячеек, {last} пар K и V читаются из кэша. Старые строки никто не пересчитывает: маска гарантирует, что новые
      токены их не меняют.
    </>,
  ]

  // широкий экран: одна сетка «Q | треугольник | K V»; узкий: треугольник отдельно, под ним кэш (иначе кэш уезжает за край)
  const narrow = bodyW > 0 && bodyW < 640
  const gridProps = { step, first, last, isNew, exists, phase }

  return (
    <Figure
      id={id}
      title="Prefill и decode: что считается заново, а что берётся из кэша"
      width="wide"
      controlsPosition="top"
      controls={<Player {...p.bind} labels={STEPS.map((s) => s.label)} aria-label="Шаги prefill и decode" />}
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> дойдите до шага «она»: её строка посчитана по кэшу, без пересчёта промпта, и веса совпадают
          с той же строкой полного прохода из первой фигуры точно (проверка под сеткой). Одна голова одного слоя, слова вместо токенов (упрощённо).
        </>
      }
    >
      <div ref={bodyRef} className="grid min-w-0 gap-5">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <Swatch role="prefill" /> посчитано в prefill
          </span>
          <span className="inline-flex items-center gap-1.5">
            <Swatch role="decode" /> посчитано в decode
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px] shadow-[0_0_0_2px_var(--accent)]" /> новое на этом шаге
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-3 w-3 rounded-[3px] border-[1.5px] border-dashed border-accent" /> пунктир: прочитано из кэша на этом шаге
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-3 w-3 rounded-[3px] bg-surface-2 bg-hatch" /> штриховка: закрыто маской (будущее)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-3 w-3 rounded-[3px] shadow-[inset_0_0_0_1px_var(--line-strong)]" /> пустая рамка: ещё нет или не хранится
          </span>
        </div>

        {narrow ? (
          <>
            <BridgeGrid {...gridProps} parts="tri" label={`${STEPS[step].label}: ${narrationText(step)}`} />
            <BridgeGrid {...gridProps} parts="cache" label="KV cache: K и V каждого обработанного токена" />
          </>
        ) : (
          <BridgeGrid {...gridProps} parts="all" label={`${STEPS[step].label}: ${narrationText(step)}`} />
        )}

        {step > 0 && (
          <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm leading-snug text-ink-2">
            <span>
              Та же строка «{WORDS[last]}» в полном проходе (первая фигура):{' '}
              <span className="tabular-nums">
                {TRACE.weights[last]
                  .slice(0, last + 1)
                  .map((w) => fmt.pct(w, 0))
                  .join(', ')}
              </span>
              .
            </span>
            {sameAsFull(last) && (
              <span className="inline-flex items-center gap-1 font-medium text-ink">
                <CheckIcon size={16} aria-hidden className="text-good" />
                совпадает
              </span>
            )}
          </p>
        )}

        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @[720px]:grid-cols-4">
          <Stat
            label="Q, K, V посчитано"
            value={String(newTokens)}
            unit={fmt.plural(newTokens, ['токен', 'токена', 'токенов'])}
            hint={step === 0 ? 'весь промпт сразу' : `без кэша: ${last + 1}`}
          />
          <Stat label="Произведений q·k" value={String(pairs)} hint={step === 0 ? `${PROMPT_LEN} · ${PROMPT_LEN + 1} / 2` : `без кэша: ${noCachePairs}`} />
          <Stat
            label="Из кэша"
            value={String(readFromCache)}
            hint={fmt.plural(readFromCache, ['пара K и V прочитана', 'пары K и V прочитаны', 'пар K и V прочитано'])}
          />
          <Stat label="В кэш" value={String(newTokens)} hint={fmt.plural(newTokens, ['пара K и V записана', 'пары K и V записаны', 'пар K и V записано'])} />
        </div>

        <p className="max-w-[72ch] text-[15px] leading-relaxed text-ink" aria-live="polite">
          <span className="font-semibold">{step === 0 ? 'Prefill. ' : `Decode, шаг ${step}. `}</span>
          {narration[step]}
        </p>
      </div>
    </Figure>
  )
}

interface GridProps {
  step: number
  first: number
  last: number
  isNew: (i: number) => boolean
  exists: (i: number) => boolean
  phase: (i: number) => 'prefill' | 'decode'
  parts: 'all' | 'tri' | 'cache'
  label: string
}

/** Сетка фигуры: подписи строк, Q нового шага, треугольник весов, KV cache (в любом сочетании частей). */
function BridgeGrid({ step, first, last, isNew, exists, phase, parts, label }: GridProps) {
  const tri = parts === 'tri'
  const cw = parts === 'all' ? CELL_W : CELL_W_NARROW
  const vw = Math.round(CELL_W * 1.6)
  const showQ = parts !== 'tri'
  const showM = parts !== 'cache'
  const showKV = parts !== 'tri'
  // номера столбцов сетки (1: подписи строк)
  const COL_Q = 2
  const COL_M = parts === 'all' ? 4 : 2
  const COL_K = parts === 'all' ? COL_M + N + 1 : 3
  const COL_V = COL_K + 1
  const template =
    parts === 'all' ? `auto ${vw}px 14px repeat(${N}, ${cw}px) 14px ${vw}px ${vw}px` : tri ? `auto repeat(${N}, ${cw}px)` : `auto ${vw}px ${vw}px ${vw}px`
  const head = showM ? HEAD : 62
  const grid: CSSProperties = { gridTemplateColumns: template, gridTemplateRows: `${head}px repeat(${N}, ${CELL_H}px)` }
  const headCell = 'flex items-end justify-center pb-1 text-[12px] text-ink-2'
  const kvNote = step === 0 ? `пишем ${newCount(first, last)}` : `читаем ${last}, пишем 1`

  return (
    <ScrollX label="Треугольник весов и KV cache" scrollerClassName="p-[3px]">
      <div className="relative inline-grid gap-[2px] align-top" style={grid} role="img" aria-label={label}>
        {/* шапка */}
        {tri && (
          <div style={{ gridRow: 1, gridColumn: 1 }} className="flex items-end justify-end pr-1.5 pb-1 text-[12px] text-ink-2">
            веса, %
          </div>
        )}
        {showQ && (
          <div style={{ gridRow: 1, gridColumn: COL_Q }} className={headCell}>
            <span className="grid justify-items-center leading-tight">
              <span className="flex items-center gap-1">
                <Swatch role="q" />Q
              </span>
              <span className="text-[11px] text-ink-3">не в кэше</span>
            </span>
          </div>
        )}
        {showM &&
          WORDS.map((w, j) => (
            <div key={`h${j}`} style={{ gridRow: 1, gridColumn: COL_M + j }} className={headCell}>
              {exists(j) ? <VLabel height={HEAD - 8}>{colWord(w)}</VLabel> : null}
            </div>
          ))}
        {showKV && (
          <>
            <div style={{ gridRow: 1, gridColumn: COL_K }} className={headCell}>
              <span className="flex items-center gap-1">
                <Swatch role="k" />K
              </span>
            </div>
            <div style={{ gridRow: 1, gridColumn: COL_V }} className={headCell}>
              <span className="flex items-center gap-1">
                <Swatch role="v" />V
              </span>
            </div>
            <div
              aria-hidden
              style={{ gridRow: 1, gridColumn: `${COL_K} / ${COL_V + 1}` }}
              className="mb-6 grid self-end border-b border-line-strong pb-0.5 text-center text-[12px] leading-tight text-ink-2"
            >
              <span>KV cache</span>
              <span className="text-[11.5px] text-ink-3">{kvNote}</span>
            </div>
          </>
        )}

        {/* строки */}
        {WORDS.map((w, i) => {
          const ex = exists(i)
          const nw = isNew(i)
          const row = i + 2
          // на шаге decode K и V старых токенов читаются из кэша
          const read = step > 0 && ex && !nw
          return (
            <Fragment key={i}>
              <div
                style={{ gridRow: row, gridColumn: 1, ...(tri && ex ? { borderLeft: `3px solid ${cssVar(phase(i))}` } : undefined) }}
                className={cn(
                  'flex items-center justify-end whitespace-nowrap',
                  tri ? 'pr-1.5 pl-1 text-[12px]' : 'gap-1.5 pr-2 text-[13px]',
                  ex ? 'text-ink' : 'text-ink-3',
                )}
              >
                {ex ? (
                  <>
                    {!tri && <Swatch role={phase(i)} />}
                    <span className={cn(nw && !tri && 'font-semibold')}>{w}</span>
                  </>
                ) : (
                  <span className="italic">ещё нет</span>
                )}
              </div>
              {/* Q живёт только на шаге, где посчитан: потом он не нужен и не хранится */}
              {showQ && (
                <Cell row={row} col={COL_Q} ghost={!nw} fresh={nw ? step : undefined} style={nw ? { background: mix('q', 22) } : undefined}>
                  {nw ? vec(rowQ(i)) : ''}
                </Cell>
              )}
              {showM &&
                WORDS.map((_, j) => {
                  if (j > i) return <Cell key={j} row={row} col={COL_M + j} masked />
                  if (!ex) return <Cell key={j} row={row} col={COL_M + j} ghost />
                  const wgt = rowWeights(i)[j]
                  return (
                    <Cell
                      key={j}
                      row={row}
                      col={COL_M + j}
                      small={tri}
                      style={{ background: mix(phase(i), 8 + 52 * wgt) }}
                      fresh={nw ? step : undefined}
                      delay={nw ? 0.12 + 0.025 * (step === 0 ? i + j : j) : 0}
                    >
                      {tri ? Math.round(wgt * 100) : fmt.pct(wgt, 0)}
                    </Cell>
                  )
                })}
              {showKV && (
                <>
                  <Cell
                    row={row}
                    col={COL_K}
                    ghost={!ex}
                    className={read ? READ_MARK : undefined}
                    style={ex ? { background: tint('k', nw ? 34 : 18) } : undefined}
                    fresh={nw ? step : undefined}
                    delay={nw ? 0.05 : 0}
                  >
                    {ex ? vec(rowK(i)) : ''}
                  </Cell>
                  <Cell
                    row={row}
                    col={COL_V}
                    ghost={!ex}
                    className={read ? READ_MARK : undefined}
                    style={ex ? { background: tint('v', nw ? 34 : 18) } : undefined}
                    fresh={nw ? step : undefined}
                    delay={nw ? 0.08 : 0}
                  >
                    {ex ? vec(rowV(i)) : ''}
                  </Cell>
                </>
              )}
            </Fragment>
          )
        })}
        {/* рамка новых строк: в треугольнике и в кэше (то, что записано на этом шаге) */}
        {showM && <NewFrame rows={`${first + 2} / ${last + 3}`} cols={`${COL_M} / ${COL_M + last + 1}`} />}
        {showKV && <NewFrame rows={`${first + 2} / ${last + 3}`} cols={`${COL_K} / ${COL_V + 1}`} />}
      </div>
    </ScrollX>
  )
}

const newCount = (first: number, last: number) => last - first + 1

function NewFrame({ rows, cols }: { rows: string; cols: string }) {
  const reduce = useReducedMotion() ?? false
  return (
    <motion.div
      aria-hidden
      layout={!reduce}
      transition={{ duration: 0.35, ease: [0.2, 0, 0, 1] }}
      className="pointer-events-none rounded-[6px] shadow-[0_0_0_2px_var(--accent)]"
      style={{ gridRow: rows, gridColumn: cols }}
    />
  )
}

function narrationText(step: number): string {
  return step === 0 ? 'посчитан весь треугольник промпта, K и V семи слов записаны в кэш' : 'посчитана одна новая строка, старые K и V прочитаны из кэша'
}

function Cell({
  row,
  col,
  children,
  ghost,
  masked,
  fresh,
  delay = 0,
  small,
  className,
  style,
}: {
  row: number
  col: number
  children?: ReactNode
  ghost?: boolean
  masked?: boolean
  /** Узкая ячейка: шрифт мельче. */
  small?: boolean
  /** Номер шага, на котором ячейка появилась (новая ячейка въезжает с анимацией). */
  fresh?: number
  delay?: number
  className?: string
  style?: CSSProperties
}) {
  const reduce = useReducedMotion() ?? false
  const base = 'flex items-center justify-center rounded-[4px] font-mono text-[11.5px] whitespace-nowrap tabular-nums text-ink'
  if (masked) return <div className={cn(base, 'bg-surface-2 bg-hatch opacity-60')} style={{ gridRow: row, gridColumn: col }} />
  if (ghost) return <div className={cn(base, 'shadow-[inset_0_0_0_1px_var(--line)]')} style={{ gridRow: row, gridColumn: col }} />
  return (
    <motion.div
      key={fresh !== undefined ? `f${fresh}` : 'old'}
      initial={fresh !== undefined && !reduce ? { opacity: 0, scale: 0.8 } : false}
      animate={{ opacity: 1, scale: 1 }}
      transition={reduce ? { duration: 0 } : { duration: 0.3, delay, ease: [0.2, 0, 0, 1] }}
      className={cn(base, small && 'text-[10px]', className)}
      style={{ gridRow: row, gridColumn: col, ...style }}
    >
      {children}
    </motion.div>
  )
}
