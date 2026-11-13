import type { ReactNode } from 'react'
import type { Token } from '@/engine/types'
import { cn, tint, tokenSegments } from '@/ui'

export type ChipLook = 'committed' | 'fresh' | 'draft' | 'rejected' | 'discarded' | 'prompt' | 'match' | 'source' | 'plain'

/**
 * Чип токена для фигур главы: те же видимые пробелы, что у TokenChip, но со своими состояниями
 * (черновик пунктиром, отклонённый, выброшенный). Цвет текста всегда --ink/--ink-2/--ink-3.
 */
export function Chip({ token, look, className, title }: { token: Token; look: ChipLook; className?: string; title?: string }) {
  const segs = tokenSegments(token)
  const style =
    look === 'committed' || look === 'fresh'
      ? { background: tint('decode', look === 'fresh' ? 26 : 13) }
      : look === 'prompt'
        ? { background: tint('prefill', 13) }
        : look === 'match'
          ? { background: tint('prefill', 34) }
          : look === 'source'
            ? { background: tint('prefill', 13) }
            : undefined
  return (
    <span
      title={title}
      className={cn(
        'inline-block rounded-chip px-[5px] font-mono text-[13px] leading-[1.6] whitespace-pre text-ink',
        look === 'draft' && 'bg-surface shadow-[inset_0_0_0_1px_var(--line-strong)] [outline:1.5px_dashed_var(--ink-3)] -outline-offset-[1.5px]',
        look === 'rejected' && 'bg-surface text-ink-2 line-through decoration-[1.5px] [outline:1.5px_dashed_var(--bad)] -outline-offset-[1.5px]',
        look === 'discarded' && 'bg-transparent text-ink-3 line-through shadow-[inset_0_0_0_1px_var(--line)]',
        look === 'plain' && 'bg-surface-2',
        look === 'match' && 'shadow-[inset_0_0_0_1.5px_var(--c-prefill)]',
        look === 'source' && 'underline decoration-[1.5px] decoration-dashed decoration-ink-3 underline-offset-[3px]',
        look === 'fresh' && 'shadow-[0_0_0_2px_var(--accent)]',
        className,
      )}
      style={style}
    >
      {segs.map((s, i) => (
        <span key={i} className={s.kind === 'ws' ? 'text-ink-2' : s.kind === 'hex' ? 'text-ink-2 italic' : undefined}>
          {s.value}
        </span>
      ))}
    </span>
  )
}

/** Маленькая подпись-легенда со свотчем. */
export function LegendItem({ swatch, children }: { swatch: ReactNode; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[13px] text-ink-2">
      {swatch}
      {children}
    </span>
  )
}
