import { useDeferredValue, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type RefObject } from 'react'
import { ArrowRightIcon, ArrowUpIcon, MagnifyingGlassIcon, XIcon } from '@phosphor-icons/react'
import { findChapter, type ChapterId } from '@/content/chapters'
import { GLOSSARY, GLOSSARY_CATEGORIES, categoryLabel, type GlossaryEntry } from '@/content/glossary'
import { Block, Button, Chapter, ChapterLink, MEDIA, Takeaways, Term, cn, fmt, useMediaQuery, useScrollEdges } from '@/ui'
import { highlightParts, searchGlossary, slug, visibleAliases, type Hit } from './search'

const label = categoryLabel

const COUNTS: Record<string, number> = GLOSSARY.reduce<Record<string, number>>((acc, e) => {
  acc[e.category] = (acc[e.category] ?? 0) + 1
  return acc
}, {})

/** Вид чипов выбора как у пресетов в других главах: невыбранные с рамкой, выбранный на мягком акценте. */
const CHIP_ON = 'bg-accent-soft text-accent-hover shadow-[inset_0_0_0_1px_var(--accent)]'
const CHIP_OFF = 'bg-surface text-ink-2 shadow-[inset_0_0_0_1px_var(--line-strong)] hover:bg-surface-3 hover:text-ink'

function Marked({ text, query }: { text: string; query: string }) {
  return (
    <>
      {highlightParts(text, query).map((p, i) =>
        p.hit ? (
          <mark key={i} className="rounded-[3px] bg-accent-soft px-px text-ink">
            {p.text}
          </mark>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </>
  )
}

function EntryCard({ e, query, showCategory }: { e: GlossaryEntry; query: string; showCategory: boolean }) {
  const ch = e.chapter ? findChapter(e.chapter) : undefined
  const aliases = visibleAliases(e)
  return (
    <article
      id={`glossary-${slug(e.term)}`}
      tabIndex={-1}
      className="flex min-w-0 flex-col gap-1.5 rounded-panel border border-line bg-surface p-4 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h4 className="min-w-0 text-[16px] leading-snug font-semibold text-ink">
          <Marked text={e.term} query={query} />
        </h4>
        {showCategory && <span className="shrink-0 text-[12px] text-ink-3">{label(e.category)}</span>}
      </header>
      {aliases.length > 0 && (
        <p className="text-[13px] leading-snug text-ink-3">
          Также: <Marked text={aliases.join(', ')} query={query} />
        </p>
      )}
      <p className="text-[14.5px] leading-[1.55] text-ink">
        <Marked text={e.ru} query={query} />
      </p>
      <p className="text-[13.5px] leading-[1.5] text-ink-2">
        <span className="font-medium text-ink">Почему важно. </span>
        <Marked text={e.why} query={query} />
      </p>
      {ch && (
        <p className="mt-auto pt-0.5 text-[14px]">
          <ChapterLink to={ch.id as ChapterId} className="inline-flex items-center gap-1">
            <span>
              {ch.number === 0 ? `В начале сайта: ${ch.nav}` : e.mentionOnly ? `Ближе всего: глава ${ch.number}, ${ch.nav}` : `Глава ${ch.number}: ${ch.nav}`}
            </span>
            <ArrowRightIcon size={14} aria-hidden />
          </ChapterLink>
        </p>
      )}
    </article>
  )
}

function CategoryFilter({ value, onChange }: { value: string | null; onChange: (v: string | null) => void }) {
  const id = useId()
  // на телефоне категории в одну строку с прокруткой (обрезанный край затухает), а не шесть строк чипов
  const [scrollRef] = useScrollEdges<HTMLDivElement>()
  const options: Array<{ value: string | null; text: string; count: number }> = [
    { value: null, text: 'Все', count: GLOSSARY.length },
    ...GLOSSARY_CATEGORIES.map((c) => ({ value: c, text: label(c), count: COUNTS[c] ?? 0 })),
  ]
  const onKey = (ev: KeyboardEvent<HTMLDivElement>) => {
    const keys = ['ArrowRight', 'ArrowLeft', 'Home', 'End']
    if (!keys.includes(ev.key)) return
    ev.preventDefault()
    const cur = options.findIndex((o) => o.value === value)
    let n = cur
    if (ev.key === 'ArrowRight') n = (cur + 1) % options.length
    else if (ev.key === 'ArrowLeft') n = (cur - 1 + options.length) % options.length
    else if (ev.key === 'Home') n = 0
    else n = options.length - 1
    onChange(options[n].value)
    const btn = ev.currentTarget.querySelectorAll<HTMLButtonElement>('button')[n]
    btn?.focus()
  }
  return (
    <div className="grid min-w-0 gap-1.5">
      <span id={id} className="text-sm font-medium text-ink-2">
        Категория
      </span>
      <div
        ref={scrollRef}
        role="radiogroup"
        aria-labelledby={id}
        onKeyDown={onKey}
        className="scroll-fade no-scrollbar -m-1 flex gap-1.5 overflow-x-auto p-1 md:flex-wrap md:overflow-visible"
      >
        {options.map((o) => {
          const on = o.value === value
          return (
            <button
              key={o.text}
              type="button"
              role="radio"
              aria-checked={on}
              tabIndex={on ? 0 : -1}
              onClick={() => onChange(o.value)}
              className={cn(
                'inline-flex h-8 shrink-0 items-center gap-1.5 rounded-control px-3 text-[13.5px] font-medium whitespace-nowrap transition-colors',
                on ? CHIP_ON : CHIP_OFF,
              )}
            >
              {o.text}
              <span className={cn('tabular-nums', on ? 'text-accent-hover' : 'text-ink-3')}>{o.count}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function Grouped({ hits, expandAll, searchRef }: { hits: Hit[]; expandAll: boolean; searchRef: RefObject<HTMLInputElement | null> }) {
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [focusId, setFocusId] = useState<string | null>(null)
  // на телефоне карточки в одну колонку: показываем по две, на широком экране по шесть (две строки по три)
  const tablet = useMediaQuery(MEDIA.tablet)
  const PREVIEW = tablet ? 6 : 2
  const groups = useMemo(() => {
    const m = new Map<string, Hit[]>()
    for (const h of hits) {
      const list = m.get(h.entry.category) ?? []
      list.push(h)
      m.set(h.entry.category, list)
    }
    // первые карточки раздела подобраны вручную (главное в разделе), остальные по алфавиту
    return GLOSSARY_CATEGORIES.filter((c) => m.has(c)).map((c) => {
      const list = m.get(c)!
      const rest = list.slice(PREVIEW).sort((a, b) => a.entry.term.localeCompare(b.entry.term, 'en', { sensitivity: 'base' }))
      return { c, items: [...list.slice(0, PREVIEW), ...rest] }
    })
  }, [hits, PREVIEW])
  // после «Показать ещё» фокус на первой новой карточке: клавиатура не теряет место на странице
  useEffect(() => {
    if (!focusId) return
    document.getElementById(focusId)?.focus()
    setFocusId(null)
  }, [focusId])
  const toSearch = () => {
    const el = searchRef.current
    if (!el) return
    el.scrollIntoView({ block: 'center' })
    el.focus({ preventScroll: true })
  }
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-10">
      {groups.map(({ c, items }) => {
        const short = items.length <= PREVIEW + 1
        const all = expandAll || open[c] || short
        const shown = all ? items : items.slice(0, PREVIEW)
        return (
          <section key={c} aria-labelledby={`glossary-cat-${slug(c)}`} className="grid grid-cols-[minmax(0,1fr)] gap-3">
            <h3 id={`glossary-cat-${slug(c)}`} className="text-[1.25rem] leading-tight font-semibold text-ink">
              {label(c)} <span className="font-normal text-ink-3 tabular-nums">{items.length}</span>
            </h3>
            <div className="grid gap-3 @2xl:grid-cols-2 @5xl:grid-cols-3">
              {shown.map((h) => (
                <EntryCard key={h.entry.term} e={h.entry} query="" showCategory={false} />
              ))}
            </div>
            {(!expandAll && !short) || !tablet ? (
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                {!expandAll && !short && (
                  <Button
                    size="sm"
                    aria-expanded={all}
                    onClick={() => {
                      if (all) {
                        setOpen((o) => ({ ...o, [c]: false }))
                        return
                      }
                      setOpen((o) => ({ ...o, [c]: true }))
                      setFocusId(`glossary-${slug(items[PREVIEW].entry.term)}`)
                    }}
                  >
                    {all ? 'Свернуть' : `Показать ещё ${fmt.count(items.length - PREVIEW, ['термин', 'термина', 'терминов'])}`}
                  </Button>
                )}
                {!tablet && (
                  <Button size="sm" variant="ghost" icon={ArrowUpIcon} onClick={toSearch}>
                    К поиску
                  </Button>
                )}
              </div>
            ) : null}
          </section>
        )
      })}
    </div>
  )
}

function GlossaryBrowser() {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState<string | null>(null)
  const deferred = useDeferredValue(query)
  const inputId = useId()
  const searchRef = useRef<HTMLInputElement | null>(null)
  const hits = useMemo(() => searchGlossary(GLOSSARY, deferred, category), [deferred, category])
  const searching = deferred.trim().length > 0
  const echo = deferred.trim().length > 40 ? `${deferred.trim().slice(0, 40)}…` : deferred.trim()

  return (
    <Block width="wide" id="glossary-browser" className="@container grid grid-cols-[minmax(0,1fr)] gap-6">
      <div className="grid min-w-0 gap-4 rounded-panel border border-line bg-surface p-4 sm:p-5">
        <div className="grid gap-1.5">
          <label htmlFor={inputId} className="text-sm font-medium text-ink-2">
            Поиск по терминам
          </label>
          <div className="relative">
            <MagnifyingGlassIcon size={18} className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-ink-3" aria-hidden />
            <input
              ref={searchRef}
              id={inputId}
              type="search"
              value={query}
              onChange={(e) => setQuery(e.currentTarget.value)}
              placeholder="например, prefill, кэш, TTFT, квантизация"
              autoComplete="off"
              spellCheck={false}
              className="block h-11 w-full rounded-control bg-surface-2 pr-10 pl-10 text-[15px] text-ink shadow-[inset_0_0_0_1px_var(--line)] transition-shadow placeholder:text-ink-3 hover:shadow-[inset_0_0_0_1px_var(--line-strong)] focus-visible:outline-2 focus-visible:outline-offset-0 [&::-webkit-search-cancel-button]:hidden"
            />
            {query && (
              <button
                type="button"
                aria-label="Очистить поиск"
                onClick={() => setQuery('')}
                className="absolute top-1/2 right-2 inline-flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-control text-ink-3 hover:bg-surface-3 hover:text-ink"
              >
                <XIcon size={16} aria-hidden />
              </button>
            )}
          </div>
          <p className="text-[13px] text-ink-3">
            Ищет по названию, синонимам и тексту, по-русски и по-английски. Регистр, ё, дефисы и окончания не важны, флаги можно писать как в
            командной строке.
          </p>
        </div>
        <CategoryFilter value={category} onChange={setCategory} />
        <p className="text-[14px] text-ink-2" aria-live="polite">
          {hits.length === GLOSSARY.length
            ? `Все ${fmt.count(GLOSSARY.length, ['термин', 'термина', 'терминов'])}.`
            : `Найдено ${fmt.count(hits.length, ['термин', 'термина', 'терминов'])} из ${fmt.int(GLOSSARY.length)}.`}
        </p>
      </div>

      {hits.length === 0 ? (
        <div className="grid grid-cols-[minmax(0,1fr)] justify-items-start gap-3 rounded-panel border border-dashed border-line-strong p-5">
          <p className="max-w-full text-[15px] break-words text-ink [overflow-wrap:anywhere]">
            По запросу «{echo}» {category ? `в категории «${label(category)}» ` : ''}ничего не нашлось.
          </p>
          <p className="max-w-full text-[14px] break-words text-ink-2 [overflow-wrap:anywhere]">
            Попробуйте английское название или начало слова (например, prefix caching, эмбеддинг) или сбросьте фильтр.
          </p>
          <Button
            size="sm"
            onClick={() => {
              setQuery('')
              setCategory(null)
            }}
          >
            Сбросить поиск и фильтр
          </Button>
        </div>
      ) : searching ? (
        <section aria-labelledby="glossary-results" className="grid grid-cols-[minmax(0,1fr)] gap-3">
          <h3 id="glossary-results" className="sr-only">
            Результаты поиска
          </h3>
          <div className="grid gap-3 @2xl:grid-cols-2 @5xl:grid-cols-3">
            {hits.map((h) => (
              <EntryCard key={h.entry.term} e={h.entry} query={deferred} showCategory />
            ))}
          </div>
        </section>
      ) : (
        <Grouped hits={hits} expandAll={category !== null} searchRef={searchRef} />
      )}
    </Block>
  )
}

export default function GlossaryChapter() {
  return (
    <Chapter
      id="glossary"
      lead={
        <>
          Все термины сайта в одном месте: короткое объяснение по-русски, почему это важно для инференса, и глава, где термин разобран на
          примерах.
        </>
      }
    >
      <p>
        Английские названия оставлены как есть, так их пишут в документации, флагах и метриках: <Term t="prefill" />, <Term t="KV cache" />,{' '}
        <Term t="TTFT" />. Ищите по любому варианту: по-русски, по-английски или по имени флага и метрики, если оно встречается в определении (
        <code>gpu_memory_utilization</code>, <code>num_requests_waiting</code>). Те же определения всплывают по наведению на подчёркнутые
        термины в тексте глав.
      </p>
      <GlossaryBrowser />
      <Takeaways
        title="Если запомнить только четыре"
        items={[
          <>
            <b>Prefill и decode.</b> Промпт считается одним большим проходом и, если в нём сотни токенов и больше, упирается в вычисления;
            ответ пишется по токену и упирается в память.
          </>,
          <>
            <b>KV cache.</b> Ключи и значения прошлых токенов, которые экономят пересчёт и съедают память GPU; вокруг него устроены
            PagedAttention, prefix caching и роутинг.
          </>,
          <>
            <b>TTFT и ITL.</b> Время до первого токена и между токенами. О скорости сервиса судят по их перцентилям (p90, p99), а не по
            среднему: пользователи запоминают медленный хвост.
          </>,
          <>
            <b>Batch.</b> Чем больше batch, тем дешевле токен и тем медленнее ответ каждому пользователю: это главный компромисс инференса.
          </>,
        ]}
      />
    </Chapter>
  )
}
