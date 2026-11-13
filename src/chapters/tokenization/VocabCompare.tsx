import { useMemo, useState } from 'react'
import { Figure, Stat, TokenStrip, TokenStripSkeleton, cn, fmt, useInView } from '@/ui'
import type { Token, Tokenizer } from '@/engine/types'
import { useO200k } from '../template/useO200k'
import { overlapping } from './textStats'

type Side = 'q' | 'o'
type Pick = { side: Side; index: number } | null

export function VocabCompare({ tokenizer, text }: { tokenizer: Tokenizer | null; text: string }) {
  const [ref, near] = useInView<HTMLDivElement>({ rootMargin: '600px', once: true })
  const o200k = useO200k(near)
  const [hover, setHover] = useState<Pick>(null)
  // клик (или Enter на чипе) закрепляет подсветку: так сравнение доступно с клавиатуры и на тач-экране
  const [pinned, setPinned] = useState<Pick>(null)

  const q = useMemo(() => (tokenizer ? tokenizer.encode(text) : null), [tokenizer, text])
  const o = useMemo(() => (o200k.tokenizer ? o200k.tokenizer.encode(text) : null), [o200k.tokenizer, text])

  const active = hover ?? pinned
  const span = (t: Token | undefined) => (t && t.start !== undefined && t.end !== undefined ? ([t.start, t.end] as const) : null)
  let hlQ: number[] | null = null
  let hlO: number[] | null = null
  if (active && q && o) {
    const sp = span(active.side === 'q' ? q[active.index] : o[active.index])
    if (sp) {
      if (active.side === 'q') {
        hlQ = [active.index]
        hlO = overlapping(o, sp[0], sp[1])
      } else {
        hlO = [active.index]
        hlQ = overlapping(q, sp[0], sp[1])
      }
    }
  }
  const pick = (side: Side) => (i: number) => setPinned((p) => (p && p.side === side && p.index === i ? null : { side, index: i }))

  return (
    <Figure
      id="tokenization-figure-vocab"
      title="Один текст, два словаря: Qwen3 и o200k (gpt-oss)"
      caption="Попробуйте: наведите на токен в одной строке или нажмите на него, и в другой подсветятся токены с теми же символами. Текст берётся из живого токенизатора выше, поменяйте его там."
    >
      <div ref={ref} className="grid gap-4">
        {q && o && o200k.status === 'ready' && <BoundaryRuler text={text} q={q} o={o} />}
        <Row
          title="Qwen3"
          note={`${fmt.int(151669)} токенов в словаре`}
          tokens={q}
          highlight={hlQ}
          onHover={(i) => setHover(i === null ? null : { side: 'q', index: i })}
          onPick={pick('q')}
        />
        <Row
          title="o200k_harmony (gpt-oss)"
          note={`${fmt.int(200019)} токенов в словаре`}
          tokens={o}
          highlight={hlO}
          onHover={(i) => setHover(i === null ? null : { side: 'o', index: i })}
          onPick={pick('o')}
          loading={o200k.status !== 'ready'}
          error={o200k.status === 'error'}
        />
        <div className="grid grid-cols-3 gap-x-6">
          <Stat label="Qwen3" value={q ? fmt.int(q.length) : '…'} unit="ток." />
          <Stat label="o200k" value={o ? fmt.int(o.length) : '…'} unit="ток." />
          <Stat label="Разница" value={q && o && o.length ? fmt.multiplier(q.length / o.length, 2) : '…'} hint="Qwen3 к o200k" />
        </div>
      </div>
    </Figure>
  )
}

/**
 * Линейка границ: текст один раз, на месте каждой границы токена черта. Во всю высоту: граница есть в обоих словарях;
 * верхняя половина: только у Qwen3; нижняя: только у o200k. Видно сразу, где словари режут одинаково, а где расходятся.
 */
function BoundaryRuler({ text, q, o }: { text: string; q: Token[]; o: Token[] }) {
  const data = useMemo(() => {
    const starts = (ts: Token[]) => new Set(ts.flatMap((t) => (t.start !== undefined && t.start > 0 && t.start < text.length ? [t.start] : [])))
    const qb = starts(q)
    const ob = starts(o)
    const cuts = [...new Set([...qb, ...ob])].sort((a, b) => a - b)
    const segs: Array<{ text: string; mark: 'both' | 'q' | 'o' | null }> = []
    let prev = 0
    for (const c of [...cuts, text.length]) {
      if (c <= prev) continue
      segs.push({ text: text.slice(prev, c), mark: prev === 0 ? null : qb.has(prev) && ob.has(prev) ? 'both' : qb.has(prev) ? 'q' : 'o' })
      prev = c
    }
    const shared = cuts.filter((c) => qb.has(c) && ob.has(c)).length
    return { segs, shared, onlyQ: qb.size - shared, onlyO: ob.size - shared }
  }, [text, q, o])

  if (!text) return null
  return (
    <div className="grid min-w-0 gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="text-[14px] font-semibold text-ink">Где режут словари</span>
        <span className="text-[12.5px] text-ink-3 tabular-nums">
          общих границ {fmt.int(data.shared)}, только у Qwen3 {fmt.int(data.onlyQ)}, только у o200k {fmt.int(data.onlyO)}
        </span>
      </div>
      <div
        className="flex max-h-[180px] flex-wrap overflow-y-auto rounded-panel bg-surface-2/60 px-2.5 py-1.5 font-mono text-[14px] text-ink shadow-[inset_0_0_0_1px_var(--line)] scrollbar-thin"
        role="img"
        aria-label={`Границы токенов: общих ${data.shared}, только у Qwen3 ${data.onlyQ}, только у o200k ${data.onlyO}`}
      >
        {data.segs.map((s, i) => (
          <span key={i} className="relative inline-block px-[3px] py-2 leading-5 whitespace-pre">
            {s.mark && <Mark kind={s.mark} />}
            {s.text.replace(/ /g, '·').replace(/\n/g, '↵')}
          </span>
        ))}
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-ink-2">
        <li className="inline-flex items-center gap-1.5">
          <LegendMark kind="both" /> граница в обоих словарях
        </li>
        <li className="inline-flex items-center gap-1.5">
          <LegendMark kind="q" /> только Qwen3 (черта сверху)
        </li>
        <li className="inline-flex items-center gap-1.5">
          <LegendMark kind="o" /> только o200k (черта снизу)
        </li>
      </ul>
    </div>
  )
}

const MARK_POS = { both: 'top-0.5 bottom-0.5', q: 'top-0.5 h-[45%]', o: 'bottom-0.5 h-[45%]' } as const

function Mark({ kind }: { kind: 'both' | 'q' | 'o' }) {
  return <span aria-hidden className={cn('absolute left-0 w-[2px] -translate-x-1/2 rounded-full', kind === 'both' ? 'bg-ink-2' : 'bg-ink', MARK_POS[kind])} />
}

function LegendMark({ kind }: { kind: 'both' | 'q' | 'o' }) {
  return (
    <span aria-hidden className="relative inline-block h-5 w-2">
      <span className={cn('absolute left-1/2 w-[2px] -translate-x-1/2 rounded-full', kind === 'both' ? 'bg-ink-2' : 'bg-ink', MARK_POS[kind])} />
    </span>
  )
}

function Row({
  title,
  note,
  tokens,
  highlight,
  onHover,
  onPick,
  loading,
  error,
}: {
  title: string
  note: string
  tokens: Token[] | null
  highlight: number[] | null
  onHover: (i: number | null) => void
  onPick: (i: number) => void
  loading?: boolean
  error?: boolean
}) {
  return (
    <div className="grid min-w-0 gap-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <span className="text-[14px] font-semibold text-ink">{title}</span>
        <span className="text-[12.5px] text-ink-3">{note}</span>
      </div>
      <div
        className="min-h-[52px] rounded-panel bg-surface-2/60 px-3 py-2.5 shadow-[inset_0_0_0_1px_var(--line)]"
        // фокус на чипе с клавиатуры подсвечивает те же символы в другой строке, как наведение мышью
        onFocus={(e) => {
          const chip = (e.target as HTMLElement).closest<HTMLElement>('[data-ti]')
          if (chip) onHover(Number(chip.dataset.ti))
        }}
        onBlur={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) onHover(null)
        }}
      >
        {error ? (
          <p className="text-[14px] text-ink-2">Не удалось загрузить словарь gpt-oss.</p>
        ) : !tokens || loading ? (
          <TokenStripSkeleton count={16} label="Загружаем словарь" />
        ) : (
          <TokenStrip
            tokens={tokens}
            showIds
            size="sm"
            highlightIndex={highlight}
            onHoverIndex={onHover}
            onClickIndex={onPick}
            label={`${title}: ${fmt.tokens(tokens.length)}`}
          />
        )}
      </div>
    </div>
  )
}
