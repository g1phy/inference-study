import type { ReactNode } from 'react'
import { cn } from './cn'

/**
 * Типографика текстового блока: абзацы, списки, h3/h4, `code`, ссылки (DESIGN §2).
 * Внутри <Chapter> абзацы можно писать и без Prose: правила те же.
 *
 *   <Prose>
 *     <p>Каждый токен делает три вектора…</p>
 *     <ul><li>…</li></ul>
 *   </Prose>
 */
export function Prose({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('prose', className)}>{children}</div>
}

export type BlockWidth = 'narrow' | 'wide' | 'full'

/**
 * Блок произвольного содержимого нужной ширины внутри главы (то же, что Figure, но без рамки).
 * narrow: как проза (60ch); wide: до 1040px; full: вся ширина контента (до 1280px; до 1600px экрана равна wide).
 */
export function Block({
  width = 'narrow',
  children,
  className,
  id,
}: {
  width?: BlockWidth
  children: ReactNode
  className?: string
  id?: string
}) {
  return (
    <div id={id} className={cn('block-gap', width === 'wide' && 'bleed-wide', width === 'full' && 'bleed-full', className)}>
      {children}
    </div>
  )
}

/**
 * Панель пояснения или показаний внутри фигуры. Два тона на весь сайт:
 * inset: вдавленная серая (пояснение к шагу, итог расчёта); outline: рамка без заливки (пример, справка).
 *
 *   <Panel>На этом шаге softmax превращает оценки в веса…</Panel>
 *   <Panel tone="outline" as="section">…</Panel>
 */
export function Panel({
  tone = 'inset',
  as: Tag = 'div',
  children,
  className,
}: {
  tone?: 'inset' | 'outline'
  as?: 'div' | 'section' | 'aside'
  children: ReactNode
  className?: string
}) {
  return (
    <Tag className={cn('rounded-panel px-4 py-3', tone === 'inset' ? 'bg-surface-2' : 'border border-line', className)}>{children}</Tag>
  )
}
