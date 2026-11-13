import { useMemo, useState, type PointerEvent } from 'react'
import { ArrowRightIcon, CopyIcon } from '@phosphor-icons/react'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { useSettings, DEFAULT_SETTINGS } from '@/state/settings'
import { Figure, FigureLive, Player, Skeleton, Tooltip, cn, fmt, mix, usePlayer } from '@/ui'
import { COW_BLOCK, COW_PROMPT, buildCow, slotOf, type CowSeqView, type CowStep, type SlotKind } from './cow'

/** Видимые пробелы в тексте токена. */
const vis = (t: string) => t.replace(/ /g, '·').replace(/\n/g, '↵') || '∅'

const ANSWER_1 = ' Это память модели'
const ANSWER_2 = ' Представьте конспект'

/** Сколько логических блоков бывает у последовательности в сценарии: 10 токенов промпта + 3 ответа = 4 блока. */
const MAX_LOGICAL = Math.ceil((COW_PROMPT + 3) / COW_BLOCK)

const SEQ_SLOTS: Array<{ id: string; label: string; before: string; after: string }> = [
  { id: 's1', label: 'Ответ 1', before: 'таблица появится после prefill', after: 'закончен, его таблица удалена' },
  { id: 's2', label: 'Ответ 2', before: 'появится, когда запрос попросит n = 2', after: 'закончен' },
]

/** Подложка чипа: промпт цветом prefill, ответ цветом decode (DESIGN §3.3), текст остаётся --ink. */
const chipBg = (kind: SlotKind) => mix(kind === 'prompt' ? 'prefill' : 'decode', 24)

/** Наведение только мышью: на таче enter и leave приходят подряд вокруг тапа. */
const mouseOnly = (fn: () => void) => (e: PointerEvent) => {
  if (e.pointerType === 'mouse') fn()
}

function LogicalTable({
  slot,
  seq,
  hover,
  onHover,
  phase,
}: {
  slot: (typeof SEQ_SLOTS)[number]
  seq: CowSeqView | undefined
  hover: number | null
  onHover: (b: number | null) => void
  phase: 'before' | 'after'
}) {
  const blocks = seq ? Math.ceil(seq.tokens.length / COW_BLOCK) : 0
  return (
    <div className="grid min-w-0 content-start gap-2">
      <div className="min-h-10 text-sm font-medium text-ink @3xl:min-h-0">
        {slot.label}
        {seq ? (
          <>
            : таблица блоков <span className="font-normal text-ink-3">(логический → физический)</span>
          </>
        ) : (
          <span className="font-normal text-ink-3">: {phase === 'before' ? slot.before : slot.after}</span>
        )}
      </div>
      {/* Место под все логические блоки сценария резервируется всегда: высота фигуры не прыгает между шагами */}
      <div className="relative">
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: MAX_LOGICAL }, (_, i) => {
            if (!seq || i >= blocks) return <LogicalBlockPlaceholder key={i} />
            const phys = seq.table[i]
            const active = hover === phys
            return (
              <div
                key={i}
                onPointerEnter={mouseOnly(() => onHover(phys))}
                onPointerLeave={mouseOnly(() => onHover(null))}
                className={cn(
                  'grid gap-1.5 rounded-control border p-1.5 transition-colors @md:p-2',
                  active ? 'border-accent bg-accent-soft' : 'border-line bg-surface-2',
                )}
              >
                <div className="flex items-center gap-1 text-xs text-ink-2">
                  <span>лог. {i}</span>
                  <ArrowRightIcon size={12} aria-hidden />
                  <span className="font-semibold text-ink">блок {phys}</span>
                </div>
                <div className="grid grid-cols-2 gap-1 @md:grid-cols-4">
                  {Array.from({ length: COW_BLOCK }, (_, j) => {
                    const pos = i * COW_BLOCK + j
                    const t = seq.tokens[pos]
                    if (t === undefined) return <span key={j} className="h-6 w-12 rounded-[4px] border border-dashed border-line" aria-hidden />
                    const slotNo = slotOf(seq.table, pos)
                    const kind: SlotKind = pos < seq.promptLen ? 'prompt' : 'answer'
                    return (
                      <Tooltip
                        key={j}
                        content={
                          <span>
                            {kind === 'prompt' ? 'Токен промпта' : 'Токен ответа'}, позиция {pos}: слот = таблица[⌊{pos}/{COW_BLOCK}⌋] · {COW_BLOCK} + ({pos} mod{' '}
                            {COW_BLOCK}) = {phys} · {COW_BLOCK} + {pos % COW_BLOCK} = <b>{slotNo}</b>
                          </span>
                        }
                      >
                        <span
                          tabIndex={0}
                          className="block h-6 w-12 truncate rounded-[4px] px-1 text-center font-mono text-[11px] leading-6 whitespace-nowrap text-ink focus-visible:outline-2 focus-visible:outline-accent"
                          style={{ background: chipBg(kind) }}
                        >
                          {vis(t)}
                        </span>
                      </Tooltip>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
        {!seq && <div aria-hidden className="absolute inset-0 rounded-control border border-dashed border-line-strong" />}
      </div>
    </div>
  )
}

/** Невидимая заглушка того же размера, что блок таблицы: держит место. */
function LogicalBlockPlaceholder() {
  return (
    <div aria-hidden className="invisible grid gap-1.5 rounded-control border border-line p-1.5 @md:p-2">
      <div className="flex items-center gap-1 text-xs">
        <span>лог. 0</span>
        <ArrowRightIcon size={12} />
        <span className="font-semibold">блок 0</span>
      </div>
      <div className="grid grid-cols-2 gap-1 @md:grid-cols-4">
        {Array.from({ length: COW_BLOCK }, (_, j) => (
          <span key={j} className="h-6 w-12" />
        ))}
      </div>
    </div>
  )
}

function PhysicalPool({ step, hover, onHover }: { step: CowStep; hover: number | null; onHover: (b: number | null) => void }) {
  return (
    <div className="grid min-w-0 gap-2">
      <div className="text-sm font-medium text-ink">
        Физические блоки в памяти GPU <span className="font-normal text-ink-3">(пул, общий для всех запросов)</span>
      </div>
      <ul className="grid grid-cols-2 gap-2 @lg:grid-cols-[repeat(auto-fill,minmax(176px,1fr))]">
        {step.blocks.map((b) => {
          const free = b.ref === 0
          const active = hover === b.id
          return (
            <li
              key={b.id}
              onPointerEnter={mouseOnly(() => !free && onHover(b.id))}
              onPointerLeave={mouseOnly(() => onHover(null))}
              aria-label={free ? `Блок ${b.id}: свободен` : `Блок ${b.id}: ref ${b.ref}, владельцы ${b.owners.join(', ')}`}
              className={cn(
                'grid gap-1.5 rounded-control border p-2 transition-[border-color,background-color]',
                free ? 'border-dashed border-line' : 'border-line bg-surface',
                (b.touched || active) && !free && 'border-accent',
                active && 'bg-accent-soft',
              )}
              style={b.touched && !free ? { boxShadow: '0 0 0 1px var(--accent)' } : undefined}
            >
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className={cn('font-semibold', free ? 'text-ink-2' : 'text-ink')}>блок {b.id}</span>
                {free ? (
                  <span className="text-ink-3">свободен</span>
                ) : (
                  <span className={cn('rounded-chip px-1.5 py-px', b.ref > 1 ? 'bg-surface-3 font-semibold text-ink' : 'text-ink-2')}>ref {b.ref}</span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-1 @lg:grid-cols-4">
                {b.slots.map((t, j) => {
                  const kind = b.kinds[j]
                  return t === null || kind === null ? (
                    <span key={j} className="h-6 rounded-[4px] border border-dashed border-line" />
                  ) : (
                    <span
                      key={j}
                      title={vis(t)}
                      className="block h-6 truncate rounded-[4px] px-px text-center font-mono text-[10.5px] leading-6 whitespace-nowrap text-ink"
                      style={{ background: chipBg(kind) }}
                    >
                      {vis(t)}
                    </span>
                  )
                })}
              </div>
              {/* строка владельцев есть и у свободного блока (пустая): высота пула не меняется между шагами */}
              <div className="flex min-h-4 items-center gap-1 truncate text-[11px] text-ink-3">
                {free ? null : b.copyOf !== undefined ? (
                  <>
                    <CopyIcon size={12} aria-hidden />
                    копия блока {b.copyOf}
                  </>
                ) : (
                  b.owners.join(' · ')
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export function BlockTableFigure() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const built = useMemo(() => {
    if (!tokenizer) return null
    const own = tokenizer.encode(settings.userText, { allowSpecial: false }).map((t) => t.text)
    // Сценарий и подписи шагов рассчитаны на 10 токенов промпта (два полных блока и 2 токена в третьем).
    // Короче вопрос: честно берём пример вместо него, а не показываем шаги, которых не происходит.
    const fallback = own.length < COW_PROMPT
    const prompt = fallback ? tokenizer.encode(DEFAULT_SETTINGS.userText, { allowSpecial: false }).map((t) => t.text) : own
    const a1 = tokenizer.encode(ANSWER_1).map((t) => t.text)
    const a2 = tokenizer.encode(ANSWER_2).map((t) => t.text)
    return { steps: buildCow(prompt, a1, a2), fallback }
  }, [tokenizer, settings.userText])
  const steps = built?.steps ?? null
  // Открываемся на шаге 3 (оба ответа делят блоки, ref 2): с него начинается эксперимент из подписи.
  const p = usePlayer(steps?.length ?? 8, { intervalMs: 2000, initialStep: 2 })
  const [hover, setHover] = useState<number | null>(null)
  const step = steps?.[p.step]
  const sample = DEFAULT_SETTINGS.userText.split(' ').slice(0, 3).join(' ').replace(/[,.;:!?]+$/, '')

  return (
    <Figure
      id="paged-attention-figure-block-table"
      title="Таблица блоков и copy-on-write: два ответа на один промпт"
      width="wide"
      controls={<Player {...p.bind} labels={steps?.map((s) => s.label)} aria-label="Шаги copy-on-write" />}
      caption="Попробуйте: пройдите шаги с 3 по 5 и сравните таблицы. Пока ответы совпадают, блоки общие (ref 2); первая же запись в общий неполный блок делает копию. Наведите на токен, чтобы увидеть, как позиция превращается в адрес слота."
    >
      {!step ? (
        <div className="grid gap-3">
          <Skeleton height={64} />
          <Skeleton height={160} />
        </div>
      ) : (
        <div className="grid min-w-0 gap-5">
          <p className="max-w-[68ch] text-sm text-ink-2">
            Упрощённо: блок на {COW_BLOCK} токена (в vLLM 16).{' '}
            {built?.fallback ? (
              <>
                Ваш вопрос из симулятора короче {COW_PROMPT} токенов, поэтому промптом здесь взят пример «{sample}…».
              </>
            ) : (
              <>Промпт это первые {COW_PROMPT} токенов вашего вопроса из симулятора.</>
            )}{' '}
            Ответы условные. Запрос просит n = 2 варианта ответа, как при parallel sampling в статье vLLM (нынешний vLLM делает это иначе,
            см. ниже).
          </p>
          <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-[13px] text-ink-2">
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden className="inline-block h-3 w-3 rounded-[3px]" style={{ background: chipBg('prompt'), boxShadow: 'inset 0 0 0 1px var(--line)' }} />
              токен промпта
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden className="inline-block h-3 w-3 rounded-[3px]" style={{ background: chipBg('answer'), boxShadow: 'inset 0 0 0 1px var(--line)' }} />
              токен ответа
            </span>
          </div>
          <div className="grid min-w-0 gap-4 @3xl:grid-cols-2">
            {SEQ_SLOTS.map((sl, i) => (
              <LogicalTable
                key={sl.id}
                slot={sl}
                seq={step.seqs.find((s) => s.id === sl.id)}
                hover={hover}
                onHover={setHover}
                phase={p.step >= steps!.length - 1 && i === 0 ? 'after' : 'before'}
              />
            ))}
          </div>
          <FigureLive className="min-h-[6.25rem] rounded-control bg-surface-2 px-3 py-2.5 text-sm text-ink @lg:min-h-[3.75rem]" role="status">
            {step.notes.map((n, i) => (
              <p key={i}>{n}</p>
            ))}
          </FigureLive>
          <PhysicalPool step={step} hover={hover} onHover={setHover} />
          <p className="min-h-[2.4rem] text-[13px] text-ink-3 @lg:min-h-0">
            Занято блоков: {step.blocks.filter((b) => b.ref > 0).length} из {step.blocks.length}.
            {step.seqs.length >= 2 && (
              <>
                {' '}
                Без общих блоков двум ответам понадобилось бы{' '}
                {fmt.int(step.seqs.reduce((a, s) => a + Math.ceil(s.tokens.length / COW_BLOCK), 0))}.
              </>
            )}
          </p>
        </div>
      )}
    </Figure>
  )
}
