import type { ReactNode } from 'react'
import { cn } from './cn'

/** Клавиша: <Kbd>Пробел</Kbd>, <Kbd>←</Kbd>. */
export function Kbd({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        'inline-flex h-[1.4rem] min-w-[1.4rem] items-center justify-center rounded-chip bg-surface px-1.5 align-[0.1em]',
        'font-mono text-[0.75rem] leading-none font-medium text-ink-2',
        'shadow-[inset_0_-1px_0_var(--line-strong),inset_0_0_0_1px_var(--line)]',
        className,
      )}
    >
      {children}
    </kbd>
  )
}
