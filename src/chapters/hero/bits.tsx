/**
 * Мелкие общие кусочки интерфейса симулятора: подписи групп, таблички «ключ: значение», статусы, каналы.
 */
import type { ReactNode } from 'react'
import { CheckCircleIcon, ProhibitIcon, WarningIcon, XCircleIcon } from '@phosphor-icons/react'
import { cn, cssVar, tint } from '@/ui'
import type { Channel } from '@/engine/types'

/** Подпись группы внутри инспектора (обычный регистр, без капса). */
export function GroupLabel({ children, aside, className }: { children: ReactNode; aside?: ReactNode; className?: string }) {
  return (
    <div className={cn('flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5', className)}>
      <h4 className="text-[13px] leading-5 font-semibold text-ink">{children}</h4>
      {aside && <span className="text-xs leading-5 text-ink-3 tabular-nums">{aside}</span>}
    </div>
  )
}

/** Табличка «ключ → значение» в две колонки. */
export function KeyVals({ rows, className, mono = true }: { rows: Array<[ReactNode, ReactNode]>; className?: string; mono?: boolean }) {
  return (
    <dl className={cn('grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[13px] leading-5', className)}>
      {rows.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-ink-2">{k}</dt>
          <dd className={cn('min-w-0 break-words text-ink', mono && 'font-mono text-[12.5px]')}>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Статус: иконка и слово (цвет никогда не единственный носитель смысла). */
export function Status({ tone, children, className }: { tone: 'good' | 'bad' | 'warn' | 'masked'; children: ReactNode; className?: string }) {
  const Icon = tone === 'good' ? CheckCircleIcon : tone === 'bad' ? XCircleIcon : tone === 'warn' ? WarningIcon : ProhibitIcon
  const color = tone === 'good' ? 'good' : tone === 'warn' ? 'warn' : 'bad'
  return (
    <span className={cn('inline-flex items-center gap-1 text-[12.5px] leading-5 whitespace-nowrap text-ink', className)}>
      <Icon size={16} aria-hidden style={{ color: cssVar(color) }} className="shrink-0" />
      {children}
    </span>
  )
}

const CHANNEL_ROLE: Record<Channel | 'tool', 'reasoning' | 'decode' | 'tool'> = {
  reasoning: 'reasoning',
  content: 'decode',
  tool_call: 'tool',
  tool: 'tool',
}
const CHANNEL_NAME: Record<Channel | 'tool', string> = {
  reasoning: 'reasoning',
  content: 'content',
  tool_call: 'tool_calls',
  tool: 'tool_calls',
}

/** Метка канала ответа: цветной квадрат и имя поля. */
export function ChannelTag({ channel, className }: { channel: Channel | 'tool'; className?: string }) {
  const role = CHANNEL_ROLE[channel]
  return (
    <span className={cn('inline-flex items-center gap-1.5 font-mono text-[12px] leading-5 text-ink-2', className)}>
      <span
        aria-hidden
        className={cn('h-2.5 w-2.5 shrink-0 rounded-[3px]', role === 'reasoning' && 'border border-dashed border-ink-3')}
        style={{ background: role === 'reasoning' ? 'transparent' : cssVar(role) }}
      />
      {CHANNEL_NAME[channel]}
    </span>
  )
}

/** Блок текста рассуждения: серый курсив с пунктирной левой границей (DESIGN §3.3). */
export function ReasoningText({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('border-l-2 border-dashed border-reasoning pl-3 text-[14px] leading-relaxed whitespace-pre-wrap text-ink-2 italic', className)}>
      {children}
    </div>
  )
}

/** Подложка вызова инструмента: метка цвета tool, текст цветом ink. */
export function ToolBox({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn('rounded-control border-l-2 px-3 py-2 font-mono text-[12.5px] leading-relaxed break-words text-ink', className)}
      style={{ borderColor: cssVar('tool'), background: tint('tool', 9) }}
    >
      {children}
    </div>
  )
}

/** Строка с видимыми пробелами для коротких кусков текста (детокенизация, эмит). */
export function WsText({ text, className }: { text: string; className?: string }) {
  if (!text) return <span className={cn('text-ink-3', className)}>пусто</span>
  const parts: ReactNode[] = []
  let buf = ''
  const flush = () => {
    if (buf) parts.push(buf)
    buf = ''
  }
  for (const ch of text) {
    if (ch === ' ' || ch === '\n' || ch === '\t') {
      flush()
      parts.push(
        <span key={parts.length} className="text-ink-3">
          {ch === ' ' ? '·' : ch === '\n' ? '↵' : '→'}
        </span>,
      )
    } else buf += ch
  }
  flush()
  return <span className={cn('font-mono whitespace-pre-wrap break-all', className)}>{parts}</span>
}

/** Маленькая подпись «условно» у симулированных величин. */
export function Approx({ children = 'условно' }: { children?: ReactNode }) {
  return <span className="text-xs text-ink-3">({children})</span>
}

/** Горизонтальная полоса из сегментов (доли), с подписями под ней. */
export function SplitBar({
  parts,
  className,
  label,
}: {
  parts: Array<{ value: number; color: string; label: ReactNode; hatched?: boolean }>
  className?: string
  label: string
}) {
  const total = parts.reduce((s, p) => s + p.value, 0) || 1
  return (
    <div className={cn('grid grid-cols-[minmax(0,1fr)] gap-1.5', className)}>
      <div role="img" aria-label={label} className="flex h-3 w-full overflow-hidden rounded-[4px] bg-surface-2">
        {parts.map((p, i) =>
          p.value > 0 ? (
            <div
              key={i}
              className="h-full first:rounded-l-[4px] last:rounded-r-[4px]"
              style={{
                width: `${(p.value / total) * 100}%`,
                background: p.hatched
                  ? `repeating-linear-gradient(135deg, ${p.color} 0 3px, transparent 3px 6px)`
                  : p.color,
                boxShadow: i > 0 ? 'inset 1px 0 0 var(--surface)' : undefined,
              }}
            />
          ) : null,
        )}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] leading-5 text-ink-2">
        {parts.map((p, i) => (
          <span key={i} className="inline-flex items-center gap-1.5">
            <span
              aria-hidden
              // прозрачная часть (пустая дорожка) показывается контуром, как сама дорожка
              className={cn('h-2.5 w-2.5 shrink-0 rounded-[3px]', p.color === 'transparent' && 'border border-line-strong bg-surface-2')}
              style={
                p.color === 'transparent' ? undefined : { background: p.hatched ? `repeating-linear-gradient(135deg, ${p.color} 0 2px, transparent 2px 4px)` : p.color }
              }
            />
            {p.label}
          </span>
        ))}
      </div>
    </div>
  )
}
