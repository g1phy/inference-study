import { createElement, isValidElement, type ReactElement } from 'react'
import type { Icon } from '@phosphor-icons/react'

/** Иконка: компонент Phosphor (ArrowRightIcon) или уже готовый элемент (<ArrowRightIcon size={20} />). */
export type IconLike = Icon | ReactElement

/** Отрисовать IconLike нужного размера (DESIGN §1: 16/18/20, weight regular). */
export function renderIcon(icon: IconLike | undefined | null, size: 16 | 18 | 20 = 18, className?: string) {
  if (!icon) return null
  if (isValidElement(icon)) return icon
  return createElement(icon as Icon, { size, weight: 'regular', 'aria-hidden': true, className, focusable: false })
}
