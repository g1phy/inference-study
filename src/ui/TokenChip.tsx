import { Fragment, memo, useCallback, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import type { Token } from '@/engine/types'
import { cn } from './cn'
import { tint } from './color'
import { useTip } from './floating'
import { fmt } from './fmt'

export type TokenVariant = 'default' | 'prompt' | 'generated' | 'muted'

export interface TokenSegment {
  kind: 'text' | 'ws' | 'hex'
  value: string
}

/**
 * Как показать текст токена: пробел «·», перевод строки «↵», таб «→»,
 * неполная UTF-8 последовательность байтами «\xD0». Спецтокен показывается литералом.
 */
export function tokenSegments(token: Token): TokenSegment[] {
  if (token.special) return [{ kind: 'text', value: token.text || token.raw }]
  const out: TokenSegment[] = []
  const push = (kind: TokenSegment['kind'], value: string) => {
    const last = out[out.length - 1]
    if (last && last.kind === kind && kind !== 'hex') last.value += value
    else out.push({ kind, value })
  }
  const partial = token.text.includes('�') && !containsReplacementChar(token.bytes)
  if (partial) {
    for (const b of token.bytes) {
      if (b === 0x20) push('ws', '·')
      else if (b === 0x0a) push('ws', '↵')
      else if (b === 0x09) push('ws', '→')
      else if (b > 0x20 && b < 0x7f) push('text', String.fromCharCode(b))
      else push('hex', `\\x${b.toString(16).toUpperCase().padStart(2, '0')}`)
    }
    return out
  }
  for (const ch of token.text) {
    if (ch === ' ') push('ws', '·')
    else if (ch === '\n') push('ws', '↵')
    else if (ch === '\t') push('ws', '→')
    else if (ch === '\r') push('ws', '␍')
    else push('text', ch)
  }
  if (out.length === 0) out.push({ kind: 'ws', value: '∅' })
  return out
}

function containsReplacementChar(bytes: number[]): boolean {
  for (let i = 0; i + 2 < bytes.length; i++) if (bytes[i] === 0xef && bytes[i + 1] === 0xbf && bytes[i + 2] === 0xbd) return true
  return false
}

function chipBackground(variant: TokenVariant, alt: boolean): string | undefined {
  if (variant === 'prompt') return tint('prefill', alt ? 24 : 13)
  if (variant === 'generated') return tint('decode', alt ? 24 : 13)
  if (variant === 'muted') return 'transparent'
  return alt ? 'var(--surface-3)' : 'var(--surface-2)'
}

/** Содержимое тултипа токена: ID, форма в словаре, байты. */
export function TokenTip({ token, index }: { token: Token; index?: number }) {
  return (
    <div className="grid gap-1.5">
      {index !== undefined && <div className="text-xs text-ink-3">Токен {fmt.int(index + 1)}</div>}
      <div className="grid grid-cols-[auto_minmax(0,1fr)] items-baseline gap-x-3 gap-y-0.5">
        <span className="text-xs text-ink-3">ID</span>
        <span className="font-mono text-[13px] font-semibold text-ink tabular-nums">{token.id}</span>
        <span className="text-xs text-ink-3">в словаре</span>
        <span className="font-mono text-[12px] break-all text-ink">{token.raw}</span>
        <span className="text-xs text-ink-3">байты</span>
        <span className="font-mono text-[12px] break-all text-ink-2">{fmt.hex(token.bytes)}</span>
      </div>
      {token.special && <div className="text-xs text-ink-2">Спецтокен: цельная запись словаря, BPE не собирает её из букв.</div>}
    </div>
  )
}

export interface TokenChipProps {
  token: Token
  variant?: TokenVariant
  showId?: boolean
  showBytes?: boolean
  /** Выделить (текущий шаг, выбранный токен): кольцо акцентом. */
  highlight?: boolean
  /** Второй оттенок фона (чередование соседей в ленте). */
  alt?: boolean
  onHover?: (hovering: boolean) => void
  onClick?: () => void
  /** Собственный тултип (в TokenStrip выключен: там один общий). */
  tooltip?: boolean
  size?: 'sm' | 'md'
  className?: string
  /** Служебное: индекс в ленте (для делегирования событий). */
  index?: number
  tabIndex?: number
  onKeyDown?: (e: KeyboardEvent<HTMLElement>) => void
}

/**
 * Один токен: моноширинный текст, радиус 6px, видимые пробелы.
 * Спецтокены с пунктирной рамкой без заливки. При наведении тултип с ID и байтами.
 *
 *   <TokenChip token={t} variant="generated" showId />
 */
export function TokenChip({
  token,
  variant = 'default',
  showId = false,
  showBytes = false,
  highlight = false,
  alt = false,
  onHover,
  onClick,
  tooltip = true,
  size = 'md',
  className,
  index,
  tabIndex,
  onKeyDown,
}: TokenChipProps) {
  const ref = useRef<HTMLElement | null>(null)
  const tip = useTip({ scopeEl: ref.current, onOutside: () => onHover?.(false) })
  const segments = useMemo(() => tokenSegments(token), [token])
  const interactive = Boolean(onClick)
  const Tag = interactive ? 'button' : 'span'

  const enter = (e: PointerEvent<HTMLElement>) => {
    onHover?.(true)
    if (tooltip) tip.show(e.currentTarget.getBoundingClientRect(), <TokenTip token={token} />)
  }
  const leave = (e: PointerEvent<HTMLElement>) => {
    if (e.pointerType !== 'mouse') return // тач: тултип остаётся до тапа в другом месте
    onHover?.(false)
    tip.hide()
  }

  return (
    <Tag
      ref={(el: HTMLElement | null) => {
        ref.current = el
      }}
      type={interactive ? 'button' : undefined}
      data-ti={index}
      tabIndex={tabIndex}
      onClick={onClick}
      onKeyDown={onKeyDown}
      onPointerEnter={tooltip || onHover ? enter : undefined}
      onPointerLeave={tooltip || onHover ? leave : undefined}
      onFocus={tooltip ? (e) => tip.show(e.currentTarget.getBoundingClientRect(), <TokenTip token={token} />) : undefined}
      onBlur={tooltip ? tip.hide : undefined}
      aria-label={interactive ? `Токен ${token.id}: ${token.special ? token.raw : JSON.stringify(token.text)}` : undefined}
      className={cn('inline-flex flex-col items-start align-top', interactive && 'cursor-pointer rounded-chip text-left', className)}
    >
      <span
        className={cn(
          'rounded-chip font-mono whitespace-pre text-ink transition-shadow',
          size === 'sm' ? 'px-1 text-[12px] leading-[1.55]' : 'px-[5px] text-[13px] leading-[1.6]',
          token.special && 'border border-dashed border-special px-1 text-ink-2',
          variant === 'muted' && !token.special && 'text-ink-3 shadow-[inset_0_0_0_1px_var(--line)]',
          highlight && 'shadow-[0_0_0_2px_var(--accent)]',
        )}
        style={{ background: token.special ? 'transparent' : chipBackground(variant, alt) }}
      >
        {segments.map((s, i) => (
          <span key={i} className={s.kind === 'ws' ? 'text-ink-2' : s.kind === 'hex' ? 'text-ink-2 italic' : undefined}>
            {s.value}
          </span>
        ))}
      </span>
      {showId && <span className="mt-0.5 px-[5px] font-mono text-[10.5px] leading-[1.3] text-ink-3 tabular-nums">{token.id}</span>}
      {showBytes && <span className="mt-0.5 px-[5px] font-mono text-[10px] leading-[1.3] whitespace-nowrap text-ink-3">{fmt.hex(token.bytes)}</span>}
      {tip.node}
    </Tag>
  )
}

/* ---------- лёгкий чип для ленты: без своего тултипа и хуков, перерисовывается только при смене токена ---------- */

interface TokenChipViewProps {
  token: Token
  index: number
  variant: TokenVariant
  alt: boolean
  showId: boolean
  showBytes: boolean
  highlight: boolean
  size: 'sm' | 'md'
  interactive: boolean
  tabIndex?: number
  onChipClick?: (index: number) => void
  onChipKey?: (e: KeyboardEvent<HTMLElement>) => void
  className?: string
}

function TokenChipViewImpl({
  token,
  index,
  variant,
  alt,
  showId,
  showBytes,
  highlight,
  size,
  interactive,
  tabIndex,
  onChipClick,
  onChipKey,
  className,
}: TokenChipViewProps) {
  const segments = tokenSegments(token)
  const Tag = interactive ? 'button' : 'span'
  return (
    <Tag
      type={interactive ? 'button' : undefined}
      data-ti={index}
      tabIndex={tabIndex}
      onClick={interactive && onChipClick ? () => onChipClick(index) : undefined}
      onKeyDown={interactive ? onChipKey : undefined}
      aria-label={interactive ? `Токен ${token.id}: ${token.special ? token.raw : JSON.stringify(token.text)}` : undefined}
      className={cn('inline-flex flex-col items-start align-top', interactive && 'cursor-pointer rounded-chip text-left', className)}
    >
      <span
        className={cn(
          'rounded-chip font-mono whitespace-pre text-ink transition-shadow',
          size === 'sm' ? 'px-1 text-[12px] leading-[1.55]' : 'px-[5px] text-[13px] leading-[1.6]',
          token.special && 'border border-dashed border-special px-1 text-ink-2',
          variant === 'muted' && !token.special && 'text-ink-3 shadow-[inset_0_0_0_1px_var(--line)]',
          highlight && 'shadow-[0_0_0_2px_var(--accent)]',
        )}
        style={{ background: token.special ? 'transparent' : chipBackground(variant, alt) }}
      >
        {segments.map((s, i) => (
          <span key={i} className={s.kind === 'ws' ? 'text-ink-2' : s.kind === 'hex' ? 'text-ink-2 italic' : undefined}>
            {s.value}
          </span>
        ))}
      </span>
      {showId && <span className="mt-0.5 px-[5px] font-mono text-[10.5px] leading-[1.3] text-ink-3 tabular-nums">{token.id}</span>}
      {showBytes && <span className="mt-0.5 px-[5px] font-mono text-[10px] leading-[1.3] whitespace-nowrap text-ink-3">{fmt.hex(token.bytes)}</span>}
    </Tag>
  )
}

/** Токен сравнивается по содержимому: после перетокенизации объекты новые, но чип на том же месте тот же. */
function sameToken(a: Token, b: Token): boolean {
  return a === b || (a.id === b.id && a.raw === b.raw && a.text === b.text && Boolean(a.special) === Boolean(b.special))
}

const TokenChipView = memo(TokenChipViewImpl, (prev, next) => {
  for (const k of Object.keys(next) as Array<keyof TokenChipViewProps>) {
    if (k === 'token') continue
    if (!Object.is(prev[k], next[k])) return false
  }
  return sameToken(prev.token, next.token)
})

export interface TokenStripProps {
  tokens: Token[]
  showIds?: boolean
  showBytes?: boolean
  /** Переносить строки (по умолчанию). false: одна строка с горизонтальной прокруткой. */
  wrap?: boolean
  highlightIndex?: number | number[] | null
  onHoverIndex?: (index: number | null) => void
  /** Клик по токену: чипы становятся кнопками (стрелки двигают фокус). */
  onClickIndex?: (index: number) => void
  /** Вариант для каждого токена (например, промпт и сгенерированные). */
  variantOf?: (token: Token, index: number) => TokenVariant
  variant?: TokenVariant
  /** Показать только первые N токенов и «ещё K токенов». */
  limit?: number
  tooltip?: boolean
  size?: 'sm' | 'md'
  className?: string
  /** Подпись для скринридеров (по умолчанию «N токенов»). */
  label?: string
  /** Что показать, если токенов нет. */
  empty?: ReactNode
}

/**
 * Лента токенов. Соседние токены чередуют два фона, перевод строки показывается «↵» и реальным переносом.
 * Один общий тултип на ленту (ID, форма в словаре, байты).
 *
 *   <TokenStrip tokens={tokens} showIds variantOf={(t, i) => (i < promptLen ? 'prompt' : 'generated')} />
 */
export function TokenStrip({
  tokens,
  showIds = false,
  showBytes = false,
  wrap = true,
  highlightIndex,
  onHoverIndex,
  onClickIndex,
  variantOf,
  variant = 'default',
  limit,
  tooltip = true,
  size = 'md',
  className,
  label,
  empty,
}: TokenStripProps) {
  const root = useRef<HTMLDivElement | null>(null)
  const hover = useRef<number | null>(null)
  const tip = useTip({
    scopeEl: root.current,
    onOutside: () => {
      hover.current = null
      onHoverIndex?.(null)
    },
  })
  const [focusIdx, setFocusIdx] = useState(0)
  const highlighted = useMemo(() => {
    if (highlightIndex === null || highlightIndex === undefined) return new Set<number>()
    return new Set(Array.isArray(highlightIndex) ? highlightIndex : [highlightIndex])
  }, [highlightIndex])
  const shown = limit !== undefined && tokens.length > limit ? tokens.slice(0, limit) : tokens
  const interactive = Boolean(onClickIndex)
  // стабильные обработчики для мемоизированных чипов: глава может передавать новую функцию на каждый рендер
  const clickRef = useRef(onClickIndex)
  clickRef.current = onClickIndex
  const onChipClick = useCallback((i: number) => clickRef.current?.(i), [])

  const showFor = useCallback(
    (el: Element, i: number) => {
      if (tooltip && tokens[i]) tip.show(el.getBoundingClientRect(), <TokenTip token={tokens[i]} index={i} />)
    },
    [tooltip, tokens, tip],
  )

  const onPointerOver = (e: PointerEvent<HTMLDivElement>) => {
    const el = (e.target as Element).closest('[data-ti]')
    if (!el || !root.current?.contains(el)) return
    const i = Number((el as HTMLElement).dataset.ti)
    if (i === hover.current) return
    hover.current = i
    onHoverIndex?.(i)
    showFor(el, i)
  }
  const onPointerLeave = (e: PointerEvent<HTMLDivElement>) => {
    if (e.pointerType !== 'mouse') return // тач: тултип остаётся до тапа в другом месте
    hover.current = null
    onHoverIndex?.(null)
    tip.hide()
  }

  const countRef = useRef(shown.length)
  countRef.current = shown.length
  const onChipKey = useCallback((e: KeyboardEvent<HTMLElement>) => {
    const i = Number((e.currentTarget as HTMLElement).dataset.ti)
    const n = countRef.current
    let next = i
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = Math.min(n - 1, i + 1)
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = Math.max(0, i - 1)
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = n - 1
    else return
    e.preventDefault()
    setFocusIdx(next)
    root.current?.querySelector<HTMLElement>(`[data-ti="${next}"]`)?.focus()
  }, [])

  const chipClass = wrap ? undefined : 'shrink-0 focus-visible:outline-offset-0'

  if (tokens.length === 0) {
    return <div className={cn('text-sm text-ink-3', className)}>{empty ?? 'Нет токенов'}</div>
  }

  return (
    <div
      ref={root}
      role="group"
      aria-label={label ?? fmt.tokens(tokens.length)}
      onPointerOver={onPointerOver}
      onPointerLeave={onPointerLeave}
      onFocus={(e) => {
        const chip = (e.target as Element).closest('[data-ti]')
        if (!interactive || !chip || !(e.target as Element).matches(':focus-visible')) return
        const i = Number((chip as HTMLElement).dataset.ti)
        setFocusIdx(i)
        showFor(chip, i)
      }}
      onBlur={(e) => {
        if (!root.current?.contains(e.relatedTarget as Node | null)) tip.hide()
      }}
      className={cn(
        'flex items-start gap-x-[3px]',
        showIds || showBytes ? 'gap-y-2' : 'gap-y-1.5',
        // без переноса: лента берёт ширину родителя и прокручивается сама, не распирая сетку вокруг;
        // поле 2px (с компенсацией по горизонтали), чтобы прокрутка не обрезала кольцо фокуса у чипов
        wrap ? 'flex-wrap' : '-mx-0.5 w-[calc(100%+4px)] overflow-x-auto px-0.5 pt-0.5 pb-1 scrollbar-thin [contain:inline-size]',
        className,
      )}
    >
      {shown.map((t, i) => (
        <Fragment key={i}>
          <TokenChipView
            token={t}
            index={i}
            variant={variantOf ? variantOf(t, i) : variant}
            alt={i % 2 === 1}
            showId={showIds}
            showBytes={showBytes}
            highlight={highlighted.has(i)}
            size={size}
            interactive={interactive}
            onChipClick={interactive ? onChipClick : undefined}
            tabIndex={interactive ? (i === focusIdx ? 0 : -1) : undefined}
            onChipKey={interactive ? onChipKey : undefined}
            className={chipClass}
          />
          {wrap && !t.special && t.text.includes('\n') && <span aria-hidden className="h-0 basis-full" />}
        </Fragment>
      ))}
      {shown.length < tokens.length && (
        <span className="self-center px-1 text-sm text-ink-3">ещё {fmt.tokens(tokens.length - shown.length)}</span>
      )}
      {tip.node}
    </div>
  )
}
