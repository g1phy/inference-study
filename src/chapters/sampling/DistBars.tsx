import { motion, useReducedMotion } from 'motion/react'
import { fmt, mix } from '@/ui'
import { cn } from '@/ui/cn'
import type { DistRow, DropReason } from './dist'

const REASON: Record<Exclude<DropReason, null>, string> = {
  greedy: 'не максимум',
  'min-p': 'отсечён min-p',
  'top-k': 'отсечён top-k',
  'top-p': 'отсечён top-p',
}

export function rowLabel(r: DistRow): string {
  return r.tail ? `ещё ${fmt.int(r.count)}` : r.text.replace(/^ /, '·')
}

/**
 * Колонки общие для всех строк (subgrid): подпись, логиты (с @md), столбики, вероятности.
 * Так столбики всех строк начинаются и кончаются на одной линии, и их длины сравнимы.
 */
const LIST_GRID = 'grid-cols-[max-content_minmax(0,1fr)_max-content] @md:grid-cols-[max-content_max-content_minmax(0,1fr)_max-content]'
const ROW_GRID = 'col-span-full grid-cols-subgrid'

/**
 * Для каждого кандидата: логит z и z / T, столбики «до» (тонкий светлый: после temperature, до отсечений)
 * и «после» (плотный: итог после отсечений и перенормировки). Отсечённые приглушены и зачёркнуты, причина словом.
 * counts: сколько раз строка выпала в серии бросков, рисуется штрихом поверх столбика «после».
 */
export function DistBars({
  rows,
  chosen,
  greedy,
  temperature,
  counts,
  total,
}: {
  rows: DistRow[]
  chosen: number | null
  greedy: boolean
  temperature: number
  counts?: readonly number[] | null
  total?: number
}) {
  const reduce = useReducedMotion()
  const t = { duration: reduce ? 0 : 0.3, ease: [0.2, 0, 0, 1] as const }
  const showCounts = Boolean(counts && total && total > 0)
  const logitText = (z: number) => (greedy ? fmt.num(z, 2) : `${fmt.num(z, 2)} → ${fmt.num(z / temperature, 2)}`)
  return (
    <div className="grid min-w-0 gap-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-1.5 w-4 rounded-[2px]" style={{ background: mix('decode', 38) }} />
          до отсечений (после temperature)
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-2.5 w-4 rounded-[2px] bg-decode" />
          после отсечений и перенормировки
        </span>
        {showCounts && (
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="h-3.5 w-[2px] bg-ink" />
            доля среди бросков
          </span>
        )}
      </div>
      <ul className={cn('grid min-w-0 gap-x-3', LIST_GRID)} aria-label="Логиты и вероятности кандидатов до и после отсечений">
        <li aria-hidden className={cn('grid items-end px-1.5 pb-1 text-xs leading-tight text-ink-3', ROW_GRID)}>
          <span>кандидат</span>
          <span className="hidden text-right @md:block">{greedy ? 'логит z' : 'логит z → z / T'}</span>
          <span />
          <span className="text-right">
            <span className="@md:hidden">вероятность</span>
            <span className="hidden @md:inline">вероятность до → после</span>
          </span>
        </li>
        {rows.map((r, i) => {
          const cut = r.after <= 0 && !(greedy && r.before > 0)
          const isChosen = chosen === i
          const reason = r.after <= 0 && r.reason ? REASON[r.reason] : null
          const share = showCounts && counts && total ? (counts[i] ?? 0) / total : null
          return (
            <li
              key={r.key}
              aria-label={`${r.tail ? `остальные ${fmt.tokens(r.count)}, логит каждого ${fmt.num(r.logit, 2)}` : `«${r.text.trim()}», логит ${fmt.num(r.logit, 2)}`}: до ${fmt.pct(r.before)}, ${reason ?? `после ${fmt.pct(r.after)}`}${share !== null ? `, выпал в ${fmt.pct(share)} бросков` : ''}`}
              className={cn(
                'grid items-center rounded-control px-1.5 py-[3px] transition-colors',
                ROW_GRID,
                isChosen && 'bg-accent-soft shadow-[inset_0_0_0_1.5px_var(--accent)]',
                r.tail && 'mt-1 border-t border-dashed border-line pt-[6px]',
              )}
            >
              <span
                className={cn(
                  'truncate text-[13px]',
                  r.tail ? 'font-sans text-ink-2' : 'font-mono',
                  cut ? 'text-ink-3 line-through decoration-ink-3' : 'text-ink',
                )}
              >
                {rowLabel(r)}
              </span>
              <span className={cn('hidden text-right font-mono text-[12px] whitespace-nowrap tabular-nums @md:block', cut ? 'text-ink-3' : 'text-ink-2')}>
                {logitText(r.logit)}
              </span>
              <span className="relative grid h-[22px] min-w-0 content-center gap-[3px]" aria-hidden>
                <span className="block h-[6px] w-full overflow-hidden">
                  <motion.span
                    className={cn('block h-full w-full origin-left', r.tail && 'bg-hatch')}
                    initial={false}
                    animate={{ scaleX: Math.min(1, r.before) }}
                    transition={t}
                    style={{ backgroundColor: mix('decode', 38) }}
                  />
                </span>
                <span className="block h-[10px] w-full overflow-hidden">
                  <motion.span
                    className={cn('block h-full w-full origin-left bg-decode', r.tail && 'bg-hatch')}
                    initial={false}
                    animate={{ scaleX: Math.min(1, r.after) }}
                    transition={t}
                  />
                </span>
                {share !== null && share > 0 && (
                  <span className="absolute top-[2px] bottom-[2px] w-[2px] -translate-x-1/2 bg-ink" style={{ left: `${Math.min(100, share * 100)}%` }} />
                )}
              </span>
              <span className={cn('text-right text-[13px] whitespace-nowrap tabular-nums', cut ? 'text-ink-3' : 'text-ink')}>
                {/* на узкой фигуре только итог: «до» видно по тонкому столбику */}
                <span className="hidden @md:inline">{fmt.pct(r.before)} → </span>
                {reason ? <span className="text-ink-3">{reason}</span> : <span className="font-semibold">{fmt.pct(r.after)}</span>}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
