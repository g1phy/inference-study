import { useRef, useState, type CSSProperties, type PointerEvent } from 'react'
import { ArrowDownIcon, ArrowLeftIcon, ArrowRightIcon, ArrowUpIcon } from '@phosphor-icons/react'
import { ChapterLink, Figure, SimulatorLink, cn, cssVar } from '@/ui'
import { STAGES, type CheatStage } from './stages'

/**
 * Весь путь запроса одной схемой: запрос идёт слева направо по верхнему ряду, ответ возвращается
 * справа налево по нижнему. На широком экране наведение или фокус показывает стадию в карточке под схемой,
 * клик закрепляет её; на узком стадии идут списком, и карточка раскрывается прямо под нажатой.
 */
export function PathMapFigure() {
  const [selected, setSelected] = useState('prefill')
  const [preview, setPreview] = useState<string | null>(null)
  // узкая схема: какая карточка раскрыта (можно свернуть все)
  const [open, setOpen] = useState<string | null>('prefill')
  const cardRef = useRef<HTMLDivElement>(null)
  const byId = (id: string) => STAGES.find((s) => s.id === id) ?? STAGES[0]
  const shown = byId(preview ?? selected)
  // Tab с узла ведёт в карточку под схемой (ссылки на главы): предпросмотр остаётся, иначе карточка сменится на закреплённую
  // стадию, ссылка с фокусом исчезнет и фокус потеряется. Сбрасываем его, когда фокус уходит из карточки наружу.
  const insideCard = (next: EventTarget | null) => next instanceof Node && Boolean(cardRef.current?.contains(next))

  // широкая схема: U-образный конвейер, ячейки сетки заданы явно (порядок в DOM = порядок пути, для Tab)
  const wideNode = (id: string, col: number, row: number) => (
    <PathNode
      key={id}
      s={byId(id)}
      selected={selected === id}
      onSelect={() => setSelected(id)}
      onPreview={(on) => setPreview(on ? id : null)}
      onBlurTo={(next) => {
        if (!insideCard(next)) setPreview(null)
      }}
      style={{ gridColumn: col, gridRow: row }}
    />
  )

  return (
    <Figure
      id="cheatsheet-figure-path"
      title="Путь запроса: от Enter до последнего токена"
      width="wide"
      caption="Попробуйте: наведите на стадию или пройдитесь по схеме клавишей Tab: карточка под схемой покажет, что на ней происходит, кто это делает и какие главы её разбирают; нажмите на стадию, чтобы закрепить карточку (на телефоне она раскроется под стадией). Стадии 1-4 готовят запрос и выбирают воркер, 5-7 идут на GPU, 8-9 превращают токены в текст и отдают клиенту. Передача KV (стадия 6) идёт между GPU и есть только при дезагрегации."
    >
      <div className="grid gap-5">
        <div className="hidden @2xl:grid grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)_20px_minmax(0,1fr)_20px_minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-y-2">
          {wideNode('client', 1, 1)}
          <Arrow dir="right" col={2} row={1} />
          {wideNode('frontend', 3, 1)}
          <Arrow dir="right" col={4} row={1} />
          {wideNode('preprocess', 5, 1)}
          <Arrow dir="right" col={6} row={1} />
          {wideNode('route', 7, 1)}
          <Arrow dir="right" col={8} row={1} />
          {wideNode('prefill', 9, 1)}

          <div aria-hidden className="flex items-center justify-center text-ink-3" style={{ gridColumn: 9, gridRow: 2 }}>
            <ArrowDownIcon size={16} />
          </div>

          {wideNode('transfer', 9, 3)}
          <Arrow dir="left" col={8} row={3} />
          {wideNode('decode', 7, 3)}
          <Arrow dir="left" col={6} row={3} />
          {wideNode('postprocess', 5, 3)}
          <Arrow dir="left" col={4} row={3} />
          {wideNode('stream', 3, 3)}
          <Arrow dir="left" col={2} row={3} />
          <div className="flex flex-col items-center justify-center gap-1 text-center text-[12px] leading-tight text-ink-2" style={{ gridColumn: 1, gridRow: 3 }}>
            <ArrowUpIcon size={16} aria-hidden className="text-ink-3" />
            ответ возвращается клиенту
          </div>
        </div>

        {/* узкая схема: столбиком по порядку, карточка раскрывается под нажатой стадией */}
        <ol className="grid gap-1.5 @2xl:hidden">
          {STAGES.map((s, i) => (
            <li key={s.id} className="grid gap-1.5">
              <PathNode s={s} selected={open === s.id} expandable onSelect={() => setOpen(open === s.id ? null : s.id)} />
              {open === s.id && <StageCard s={s} />}
              {i < STAGES.length - 1 && (
                <div className="flex justify-center text-ink-3">
                  <ArrowDownIcon size={16} aria-hidden />
                </div>
              )}
            </li>
          ))}
        </ol>

        <div className="grid gap-2 text-[13px] leading-snug text-ink-2 @xl:grid-cols-2 @xl:gap-6">
          <div className="flex items-start gap-2">
            <span aria-hidden className="mt-1 h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: cssVar('prefill') }} />
            <span>
              <span className="font-medium text-ink">TTFT</span> складывается из всего, что до первого токена: сеть, очередь, шаблон и токенизация, роутинг,
              prefill (и передача KV при дезагрегации).
            </span>
          </div>
          <div className="flex items-start gap-2">
            <span aria-hidden className="mt-1 h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: cssVar('decode') }} />
            <span>
              <span className="font-medium text-ink">ITL</span> это почти целиком шаг decode: детокенизация и отправка чанка идут параллельно со следующим
              шагом.
            </span>
          </div>
        </div>

        <div
          ref={cardRef}
          className="hidden @2xl:block"
          aria-live="polite"
          onBlur={(e) => {
            if (!(e.relatedTarget instanceof Node && e.currentTarget.contains(e.relatedTarget))) setPreview(null)
          }}
        >
          <StageCard s={shown} />
        </div>
      </div>
    </Figure>
  )
}

function StageCard({ s }: { s: CheatStage }) {
  return (
    <section aria-label={`Стадия ${s.n}: ${s.name}`} className="grid gap-3 rounded-panel border border-line bg-surface-2 p-4 @xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="grid content-start gap-2 @xl:col-span-2">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-[13px] text-ink-3">Стадия {s.n}</span>
          <span className="text-[15px] font-semibold text-ink">{s.name}</span>
          <span className="text-[13px] text-ink-3">{s.where}</span>
        </div>
        <p className="max-w-[72ch] text-[15px] leading-relaxed text-ink">{s.what}</p>
      </div>
      {s.stack && (
        <div className="@xl:col-span-2">
          <Detail label="Кто делает" text={s.stack} />
        </div>
      )}
      <Detail label="Что тюнят" text={s.tune} />
      <Detail label="Главная метрика" text={s.metric} />
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[14px] @xl:col-span-2">
        <span className="text-ink-3">Главы:</span>
        {s.chapters.map((c) => (
          <ChapterLink key={c} to={c} />
        ))}
        {s.sim && <SimulatorLink stage={s.sim} />}
      </div>
    </section>
  )
}

function PathNode({
  s,
  selected,
  expandable = false,
  onSelect,
  onPreview,
  onBlurTo,
  style,
}: {
  s: CheatStage
  selected: boolean
  /** Узкая схема: кнопка раскрывает карточку под собой. */
  expandable?: boolean
  onSelect: () => void
  onPreview?: (on: boolean) => void
  /** Фокус ушёл с узла; next: куда (relatedTarget). Без обработчика предпросмотр просто сбрасывается. */
  onBlurTo?: (next: EventTarget | null) => void
  style?: CSSProperties
}) {
  return (
    <button
      type="button"
      aria-pressed={expandable ? undefined : selected}
      aria-expanded={expandable ? selected : undefined}
      onClick={onSelect}
      onPointerEnter={(e: PointerEvent<HTMLButtonElement>) => e.pointerType === 'mouse' && onPreview?.(true)}
      onPointerLeave={(e: PointerEvent<HTMLButtonElement>) => e.pointerType === 'mouse' && onPreview?.(false)}
      onFocus={() => onPreview?.(true)}
      onBlur={(e) => (onBlurTo ? onBlurTo(e.relatedTarget) : onPreview?.(false))}
      style={style}
      className={cn(
        'grid min-w-0 content-start gap-1 rounded-panel border bg-surface px-3 py-2.5 text-left transition-colors hover:bg-surface-3',
        selected ? 'border-accent' : 'border-line',
      )}
    >
      <span className="flex min-w-0 items-baseline gap-1.5">
        <span className="text-[12px] text-ink-3">{s.n}</span>
        {s.role && <span aria-hidden className="h-2 w-2 shrink-0 self-center rounded-full" style={{ background: cssVar(s.role) }} />}
        <span className="min-w-0 text-[14px] leading-tight font-semibold text-ink">{s.name}</span>
      </span>
      <span className="text-[12px] leading-snug text-ink-2">{s.short}</span>
      <span className="text-[12px] leading-snug text-ink-3">{s.metric}</span>
    </button>
  )
}

function Arrow({ dir, col, row }: { dir: 'left' | 'right'; col: number; row: number }) {
  return (
    <div aria-hidden className="flex items-center justify-center text-ink-3" style={{ gridColumn: col, gridRow: row }}>
      {dir === 'right' ? <ArrowRightIcon size={16} /> : <ArrowLeftIcon size={16} />}
    </div>
  )
}

function Detail({ label, text }: { label: string; text: string }) {
  return (
    <div className="grid content-start gap-0.5">
      <div className="text-[13px] text-ink-3">{label}</div>
      <div className="text-[14px] leading-snug text-ink">{text}</div>
    </div>
  )
}
