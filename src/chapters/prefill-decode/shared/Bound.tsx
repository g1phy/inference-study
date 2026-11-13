import type { ReactNode } from 'react'
import { CpuIcon, MemoryIcon } from '@phosphor-icons/react'
import { cn } from '@/ui'

/**
 * Вердикт «во что упирается» иконкой и словом (цвет не единственный носитель смысла, DESIGN §8):
 * микросхема памяти для memory-bound, процессор для compute-bound.
 */
export function BoundTag({
  bound,
  memoryText = 'упор в память',
  computeText = 'упор в вычисления',
  className,
}: {
  bound: 'memory' | 'compute'
  memoryText?: ReactNode
  computeText?: ReactNode
  className?: string
}) {
  const Icon = bound === 'memory' ? MemoryIcon : CpuIcon
  return (
    <span className={cn('inline-flex items-center gap-1', className)}>
      <Icon size={14} aria-hidden className="shrink-0" />
      {bound === 'memory' ? memoryText : computeText}
    </span>
  )
}
