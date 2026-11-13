import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { CheckIcon, CopyIcon } from '@phosphor-icons/react'
import { cn } from './cn'
import { highlight, LANG_LABELS, type CodeLang, type TokType } from './highlight'

export interface CodeBlockProps {
  code: string
  lang?: CodeLang
  /** Номера строк (с 1), которые нужно подсветить. */
  highlightLines?: number[]
  /** Переносить длинные строки вместо горизонтальной прокрутки. */
  wrap?: boolean
  /** Заголовок в шапке (имя файла, команда). По умолчанию название языка. */
  title?: ReactNode
  lineNumbers?: boolean
  /** Максимальная высота, дальше вертикальная прокрутка (например 360). */
  maxHeight?: number | string
  /** Кнопка «Копировать» (по умолчанию есть). */
  copy?: boolean
  className?: string
}

const TOKEN_CLASS: Record<TokType, string> = {
  plain: '',
  key: 'text-ink',
  str: 'text-[var(--code-string)]',
  num: 'text-[var(--code-number)]',
  kw: 'text-[var(--code-keyword)]',
  com: 'text-[var(--code-comment)] italic',
  punct: 'text-[var(--code-punct)]',
  fn: 'text-ink font-semibold',
  var: 'text-[var(--code-number)]',
  flag: 'text-[var(--code-keyword)]',
  // mx-px: пунктирные рамки соседних спецтокенов (</tool_call><|im_end|>) не сливаются
  special: 'mx-px rounded-[4px] text-ink-2 outline outline-1 -outline-offset-1 outline-dashed outline-[var(--c-special)]',
  tag: 'text-[var(--code-punct)] font-semibold',
}

/** Отступ строки в знаках (таб = 2, как tabSize). */
function leadingIndent(line: string): number {
  let n = 0
  for (const ch of line) {
    if (ch === ' ') n += 1
    else if (ch === '\t') n += 2
    else break
  }
  return n
}

/**
 * Блок кода с подсветкой (json, yaml, bash, jinja, python, text) и кнопкой «Копировать».
 * В режиме text и jinja спецтокены (<|im_start|>, <think>) обводятся пунктиром, как в TokenStrip.
 *
 *   <CodeBlock lang="bash" code={'vllm serve Qwen/Qwen3-8B \\\n  --enable-prefix-caching'} highlightLines={[2]} />
 */
export function CodeBlock({
  code,
  lang = 'text',
  highlightLines,
  wrap = false,
  title,
  lineNumbers = false,
  maxHeight,
  copy = true,
  className,
}: CodeBlockProps) {
  const lines = useMemo(() => highlight(code, lang), [code, lang])
  const hl = useMemo(() => new Set(highlightLines ?? []), [highlightLines])
  const [copied, setCopied] = useState(false)
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(timer.current), [])

  const doCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      window.clearTimeout(timer.current)
      timer.current = window.setTimeout(() => setCopied(false), 1600)
    } catch {
      /* буфер обмена недоступен (не https): ничего не делаем */
    }
  }

  const label = title ?? LANG_LABELS[lang]
  const numWidth = String(lines.length).length
  // при переносе продолжение строки встаёт под её текст (+2 знака), а не в нулевую колонку:
  // иначе на телефоне «"content": "Ты полезный…» оказывалось левее своего объекта
  const indents = useMemo(() => (wrap ? code.split('\n').map(leadingIndent) : null), [code, wrap])
  const gutter = lineNumbers ? `${numWidth}ch + 1rem` : '0px'

  return (
    <div className={cn('codeblock block-gap min-w-0 overflow-clip rounded-panel border border-line bg-surface-2', className)}>
      <div className="flex h-9 items-center justify-between gap-3 border-b border-line pr-1.5 pl-4">
        <span className="min-w-0 truncate font-mono text-xs text-ink-3">{label}</span>
        {copy && (
          <button
            type="button"
            onClick={doCopy}
            className="inline-flex h-7 shrink-0 items-center gap-1.5 rounded-control px-2 text-xs font-medium text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink"
          >
            {copied ? <CheckIcon size={16} aria-hidden /> : <CopyIcon size={16} aria-hidden />}
            <span>{copied ? 'Скопировано' : 'Копировать'}</span>
          </button>
        )}
        <span className="sr-only" aria-live="polite">
          {copied ? 'Код скопирован' : ''}
        </span>
      </div>
      <pre
        className={cn(
          // прокручиваемый <pre> получает фокус с клавиатуры; рамка блока (overflow-clip) обрезала бы внешнее кольцо
          'overflow-auto py-3 font-mono text-[13px] leading-[1.65] text-ink scrollbar-thin focus-visible:-outline-offset-2',
          wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre',
        )}
        style={{ maxHeight, tabSize: 2 } as CSSProperties}
      >
        <code className={cn('block', !wrap && 'min-w-max')}>
          {lines.map((line, i) => (
            <span
              key={i}
              className={cn(
                'block border-l-2 pr-4 pl-[14px]',
                hl.has(i + 1) ? 'border-accent bg-[var(--code-line-hl)]' : 'border-transparent',
                wrap && 'min-w-0',
              )}
              style={
                indents
                  ? {
                      paddingLeft: `calc(14px + ${gutter} + ${(indents[i] ?? 0) + 2}ch)`,
                      textIndent: `calc(-1 * (${gutter} + ${(indents[i] ?? 0) + 2}ch))`,
                    }
                  : undefined
              }
            >
              {lineNumbers && (
                <span aria-hidden className="mr-4 inline-block text-right text-ink-3 select-none" style={{ width: `${numWidth}ch` }}>
                  {i + 1}
                </span>
              )}
              {line.length === 0 ? '\u200b' : line.map((tok, j) => (
                <span key={j} className={TOKEN_CLASS[tok.t] || undefined}>
                  {tok.v}
                </span>
              ))}
            </span>
          ))}
        </code>
      </pre>
    </div>
  )
}
