import type { CSSProperties, ReactNode } from 'react'
import { CheckCircleIcon, InfoIcon, WarningIcon, XCircleIcon } from '@phosphor-icons/react'
import { cn } from './cn'
import { cssVar, tint, type DataRole } from './color'
import { renderIcon, type IconLike } from './icon'

export type BadgeTone = 'neutral' | 'accent' | 'good' | 'warn' | 'bad' | DataRole

export interface BadgeProps {
  children: ReactNode
  tone?: BadgeTone
  /** Своя иконка; для статусов иконка ставится автоматически (null убирает). */
  icon?: IconLike | null
  size?: 'sm' | 'md'
  className?: string
  title?: string
}

const STATUS_ICON = { good: CheckCircleIcon, warn: WarningIcon, bad: XCircleIcon } as const

/**
 * Бейдж (радиус 6px). Статусы всегда с иконкой (цвет не единственный носитель смысла),
 * сущности с цветной меткой; текст всегда цветом --ink.
 *
 *   <Badge tone="good">cache hit</Badge>   <Badge tone="prefill">prefill</Badge>   <Badge tone="special">спецтокен</Badge>
 */
export function Badge({ children, tone = 'neutral', icon, size = 'sm', className, title }: BadgeProps) {
  const isStatus = tone === 'good' || tone === 'warn' || tone === 'bad'
  const isData = !isStatus && tone !== 'neutral' && tone !== 'accent'
  const autoIcon = isStatus ? STATUS_ICON[tone] : undefined
  const shownIcon = icon === null ? null : (icon ?? autoIcon)

  let style: CSSProperties | undefined
  let toneClass = ''
  if (tone === 'neutral') toneClass = 'bg-surface-2 text-ink-2 shadow-[inset_0_0_0_1px_var(--line)]'
  // на подложке accent-soft сам --accent даёт 4,3:1 (ниже AA для 12px), --accent-hover темнее в светлой теме и светлее в тёмной
  else if (tone === 'accent') toneClass = 'bg-accent-soft text-accent-hover'
  else if (tone === 'special') toneClass = 'border border-dashed border-special text-ink-2'
  else style = { background: tint(tone, 14) }

  return (
    <span
      title={title}
      className={cn(
        'inline-flex max-w-full items-center gap-1 rounded-chip align-middle font-medium whitespace-nowrap text-ink',
        size === 'sm' ? 'h-6 px-2 text-xs' : 'h-7 px-2.5 text-[13px]',
        toneClass,
        className,
      )}
      style={style}
    >
      {isData && tone !== 'special' && !shownIcon && (
        <span aria-hidden className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: cssVar(tone as DataRole) }} />
      )}
      {shownIcon && (
        <span aria-hidden className="inline-flex shrink-0" style={isStatus ? { color: cssVar(tone as 'good') } : undefined}>
          {renderIcon(shownIcon, 16)}
        </span>
      )}
      <span className="truncate">{children}</span>
    </span>
  )
}

export type StatusKind = 'good' | 'warn' | 'bad' | 'neutral'

const STATUS_ICONS = { good: CheckCircleIcon, warn: WarningIcon, bad: XCircleIcon, neutral: InfoIcon } as const

/**
 * Иконка статуса для своих таблиц и списков: одна форма на весь сайт (Phosphor regular, 16 px),
 * цвет от токена статуса. Смысл всегда дублируйте словом рядом (DESIGN §8): иконка aria-hidden.
 *
 *   <StatusIcon status="good" /> помещается
 */
export function StatusIcon({ status, size = 16, className }: { status: StatusKind; size?: 16 | 18 | 20; className?: string }) {
  const Icon = STATUS_ICONS[status]
  return (
    <Icon
      size={size}
      aria-hidden
      className={cn('inline-block shrink-0 align-[-0.15em]', className)}
      style={{ color: status === 'neutral' ? 'var(--ink-3)' : cssVar(status) }}
    />
  )
}
