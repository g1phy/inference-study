import type { ComponentPropsWithRef, ReactNode } from 'react'
import { cn } from './cn'
import { renderIcon, type IconLike } from './icon'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost'
export type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonOwnProps {
  variant?: ButtonVariant
  size?: ButtonSize
  /** Иконка слева: компонент Phosphor (PlayIcon) или элемент. */
  icon?: IconLike
  /** Иконка справа. */
  iconRight?: IconLike
  /** Квадратная кнопка только с иконкой: обязательно передайте aria-label. */
  iconOnly?: boolean
  fullWidth?: boolean
  children?: ReactNode
}

const BASE =
  'inline-flex select-none items-center justify-center whitespace-nowrap rounded-control font-medium transition-[background-color,color,box-shadow,transform] duration-150 active:translate-y-px disabled:pointer-events-none disabled:opacity-45 aria-disabled:pointer-events-none aria-disabled:opacity-45'

const VARIANTS: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-accent-ink hover:bg-accent-hover',
  secondary: 'bg-surface text-ink shadow-[inset_0_0_0_1px_var(--line-strong)] hover:bg-surface-3',
  ghost: 'bg-transparent text-ink-2 hover:bg-surface-3 hover:text-ink',
}

const SIZES: Record<ButtonSize, { box: string; square: string; icon: 16 | 18 | 20 }> = {
  sm: { box: 'h-8 gap-1.5 px-3 text-sm', square: 'h-8 w-8', icon: 16 },
  md: { box: 'h-10 gap-2 px-4 text-[15px]', square: 'h-10 w-10', icon: 18 },
  lg: { box: 'h-12 gap-2 px-5 text-base', square: 'h-12 w-12', icon: 20 },
}

/** Классы кнопки (если нужно оформить кнопкой свой элемент). */
export function buttonClass({
  variant = 'secondary',
  size = 'md',
  iconOnly = false,
  fullWidth = false,
}: Pick<ButtonOwnProps, 'variant' | 'size' | 'iconOnly' | 'fullWidth'> = {}): string {
  const s = SIZES[size]
  return cn(BASE, VARIANTS[variant], iconOnly ? s.square : s.box, fullWidth && 'w-full')
}

export type ButtonProps = ButtonOwnProps & Omit<ComponentPropsWithRef<'button'>, 'children'>

/**
 * Кнопка. Варианты: primary (акцент, одна главная кнопка на блок), secondary, ghost.
 *
 *   <Button variant="primary" icon={PaperPlaneRightIcon}>Отправить</Button>
 *   <Button iconOnly icon={ArrowCounterClockwiseIcon} aria-label="Сначала" />
 */
export function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  iconRight,
  iconOnly = false,
  fullWidth = false,
  className,
  children,
  type = 'button',
  ...rest
}: ButtonProps) {
  const s = SIZES[size]
  return (
    <button type={type} className={cn(buttonClass({ variant, size, iconOnly, fullWidth }), className)} {...rest}>
      {renderIcon(icon, s.icon)}
      {!iconOnly && children}
      {!iconOnly && renderIcon(iconRight, s.icon)}
    </button>
  )
}

export type ButtonLinkProps = ButtonOwnProps & Omit<ComponentPropsWithRef<'a'>, 'children'>

/** То же оформление для ссылки <a>. */
export function ButtonLink({
  variant = 'secondary',
  size = 'md',
  icon,
  iconRight,
  iconOnly = false,
  fullWidth = false,
  className,
  children,
  ...rest
}: ButtonLinkProps) {
  const s = SIZES[size]
  return (
    <a className={cn(buttonClass({ variant, size, iconOnly, fullWidth }), className)} {...rest}>
      {renderIcon(icon, s.icon)}
      {!iconOnly && children}
      {!iconOnly && renderIcon(iconRight, s.icon)}
    </a>
  )
}
