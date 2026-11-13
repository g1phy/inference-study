import { Fragment, type CSSProperties, type ReactNode } from 'react'
import type { ModelFamily, SegmentKind, TemplateSegment } from '@/engine/types'
import { cn, tint } from '@/ui'
import { splitByLiterals } from './promptTokens'

/**
 * Отрендеренный промпт с раскраской сегментов шаблона.
 * Спецтокены в пунктирной рамке (DESIGN §3.3), роль полужирным, текст сообщений обычным цветом,
 * обвязка шаблона приглушённо, описание инструментов на подложке цвета tool, рассуждение курсивом
 * с пунктирной левой границей, приглашение ассистента подчёркнуто цветом decode: отсюда модель начнёт писать.
 * Переводы строк видны значком «↵» и настоящим переносом.
 */
export function PromptView({
  segments,
  family,
  className,
  caret = true,
}: {
  segments: TemplateSegment[]
  family: ModelFamily
  className?: string
  /** Показать после приглашения ассистента метку «ответ модели». */
  caret?: boolean
}) {
  // рассуждение: content между <think> и </think>
  let inThink = false
  const parts: ReactNode[] = []
  segments.forEach((s, i) => {
    if (s.kind === 'special') {
      if (s.text === '<think>') inThink = true
      if (s.text === '</think>') inThink = false
      parts.push(<Special key={i} text={s.text} />)
      return
    }
    if (s.kind === 'gen-prompt') {
      parts.push(
        <span key={i} className="border-b-2 border-decode pb-px">
          {splitByLiterals(s.text, family).map((p, j) =>
            p.special ? <Special key={j} text={p.text} /> : <Text key={j} text={p.text} cls="font-semibold text-ink" />,
          )}
        </span>,
      )
      return
    }
    parts.push(<Text key={i} text={s.text} cls={classOf(s.kind, inThink)} style={s.kind === 'tools' ? { background: tint('tool', 13) } : undefined} />)
  })
  const hasGen = segments.some((s) => s.kind === 'gen-prompt')
  return (
    <div className={cn('font-mono text-[12.5px] leading-[1.75] break-words whitespace-pre-wrap text-ink', className)}>
      {parts}
      {caret && hasGen && (
        <span className="ml-1 inline-flex items-center gap-1 align-baseline font-sans text-[12px] whitespace-nowrap text-ink-3">
          <span aria-hidden className="inline-block h-3.5 w-1.5 translate-y-[2px] rounded-[2px] bg-decode" />
          ответ модели
        </span>
      )}
    </div>
  )
}

function classOf(kind: SegmentKind, inThink: boolean): string {
  if (kind === 'content' && inThink) return 'italic text-ink-2 border-l-2 border-dashed border-reasoning pl-1 [box-decoration-break:clone]'
  switch (kind) {
    case 'role':
      return 'font-semibold text-ink'
    case 'content':
      return 'text-ink'
    case 'tools':
      return 'text-ink rounded-[3px] [box-decoration-break:clone]'
    case 'format':
      return 'text-ink-3'
    default:
      return ''
  }
}

/** Спецтокен в пунктирной рамке (DESIGN §3.3). Не переносится посередине: это один токен, а не набор символов. */
export function Special({ text }: { text: string }) {
  return (
    <span className="rounded-[4px] px-[2px] whitespace-nowrap text-ink-2 outline outline-1 -outline-offset-1 outline-dashed outline-[var(--c-special)]">
      {text}
    </span>
  )
}

/** Текст с видимыми переводами строк: «↵» и настоящий перенос. */
function Text({ text, cls, style }: { text: string; cls?: string; style?: CSSProperties }) {
  const lines = text.split('\n')
  return (
    <span className={cls} style={style}>
      {lines.map((l, i) => (
        <Fragment key={i}>
          {i > 0 && (
            <>
              <span aria-hidden className="text-[11px] text-ink-3 not-italic select-none">
                ↵
              </span>
              {'\n'}
            </>
          )}
          {l}
        </Fragment>
      ))}
    </span>
  )
}

/** Легенда раскраски (только виды, которые есть в текущем промпте). */
export function PromptLegend({ segments, className }: { segments: TemplateSegment[]; className?: string }) {
  let think = false
  let hasReasoning = false
  const kinds = new Set<SegmentKind>()
  for (const s of segments) {
    kinds.add(s.kind)
    if (s.kind === 'special') {
      if (s.text === '<think>') think = true
      if (s.text === '</think>') think = false
    }
    if (s.kind === 'content' && think && s.text.trim()) hasReasoning = true
  }
  const items: Array<{ key: string; sample: ReactNode; label: string }> = []
  if (kinds.has('special'))
    items.push({ key: 's', sample: <span className="rounded-[3px] px-1 font-mono text-[11px] outline outline-1 -outline-offset-1 outline-dashed outline-[var(--c-special)]">&lt;|…|&gt;</span>, label: 'спецтокен' })
  if (kinds.has('role')) items.push({ key: 'r', sample: <span className="font-mono text-[12px] font-semibold">user</span>, label: 'роль' })
  if (kinds.has('content')) items.push({ key: 'c', sample: <span className="font-mono text-[12px]">Аа</span>, label: 'текст сообщений' })
  if (hasReasoning)
    items.push({ key: 'h', sample: <span className="border-l-2 border-dashed border-reasoning pl-1 font-mono text-[12px] text-ink-2 italic">Аа</span>, label: 'рассуждение' })
  if (kinds.has('tools'))
    items.push({ key: 't', sample: <span className="rounded-[3px] px-1 font-mono text-[12px]" style={{ background: tint('tool', 13) }}>{'{…}'}</span>, label: 'описание инструментов' })
  if (kinds.has('format')) items.push({ key: 'f', sample: <span className="font-mono text-[12px] text-ink-3">↵ Аа</span>, label: 'служебный текст шаблона' })
  if (kinds.has('gen-prompt'))
    items.push({ key: 'g', sample: <span className="border-b-2 border-decode font-mono text-[12px] font-semibold">assistant</span>, label: 'приглашение ассистента' })
  return (
    <ul className={cn('flex flex-wrap gap-x-4 gap-y-1.5 text-[13px] text-ink-2', className)} aria-label="Легенда раскраски">
      {items.map((it) => (
        <li key={it.key} className="inline-flex items-center gap-1.5">
          {it.sample}
          <span>{it.label}</span>
        </li>
      ))}
    </ul>
  )
}
