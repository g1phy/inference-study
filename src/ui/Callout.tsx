import type { ReactNode } from 'react'
import { FactoryIcon, LightbulbIcon, RulerIcon, WarningIcon } from '@phosphor-icons/react'
import type { Icon } from '@phosphor-icons/react'
import { cn } from './cn'

export type CalloutKind = 'intuition' | 'production' | 'pitfall' | 'precise'

const KINDS: Record<CalloutKind, { label: string; icon: Icon; iconClass: string }> = {
  intuition: { label: 'Интуиция', icon: LightbulbIcon, iconClass: 'text-ink-2' },
  production: { label: 'В проде', icon: FactoryIcon, iconClass: 'text-ink-2' },
  pitfall: { label: 'Подвох', icon: WarningIcon, iconClass: 'text-warn' },
  precise: { label: 'Точнее', icon: RulerIcon, iconClass: 'text-ink-2' },
}

export interface CalloutProps {
  kind: CalloutKind
  /** Заголовок после метки вида: «В проде: таймауты и ретраи». */
  title?: ReactNode
  children: ReactNode
  className?: string
}

/**
 * Врезка. Виды: intuition (аналогия), production («В проде»: метрики, флаги, конфиги),
 * pitfall («Подвох»: частые заблуждения), precise («Точнее»: строгая формулировка).
 *
 *   <Callout kind="production" title="таймауты">Стриминг улучшает воспринимаемую задержку…</Callout>
 */
export function Callout({ kind, title, children, className }: CalloutProps) {
  const k = KINDS[kind]
  const IconCmp = k.icon
  return (
    <aside className={cn('callout rounded-panel bg-surface-2 px-4 py-4 sm:px-5', className)} aria-label={k.label}>
      <div className="flex items-start gap-2.5">
        <IconCmp size={18} aria-hidden className={cn('mt-[3px] shrink-0', k.iconClass)} />
        <div className="min-w-0 flex-1">
          <p className="text-[15px] leading-6 font-semibold text-ink">
            {k.label}
            {title && (
              <>
                <span className="text-ink-3">: </span>
                <span>{title}</span>
              </>
            )}
          </p>
          <div className="prose mt-1.5 text-[16px] leading-[1.6] text-ink">{children}</div>
        </div>
      </div>
    </aside>
  )
}
