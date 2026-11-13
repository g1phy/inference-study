import { useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { headPatterns } from '@/engine/attention'
import { getModel, findModel } from '@/engine/data/models'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { DEFAULT_SETTINGS, useSettings } from '@/state/settings'
import { BarChart, Badge, cn, Figure, FigureLive, fmt, Heatmap, Segmented, Select, Skeleton, TokenChip, TokenStripSkeleton, useElementSize } from '@/ui'
import { ArcDiagram } from './ArcDiagram'
import { useRoving } from './bits'
import { buildPromptTokens, REPEAT_EXAMPLE, tokenLabel } from './promptTokens'

const LIMIT = 24

type Source = 'user' | 'repeat'

export function HeadsFigure({ id }: { id: string }) {
  const [settings] = useSettings()
  const { tokenizer, status, error } = useTokenizer()
  const [source, setSource] = useState<Source>('user')
  const [headId, setHeadId] = useState('previous-token')
  const [hover, setHover] = useState<number | null>(null)
  const [pinned, setPinned] = useState<number | null>(null)
  const [bodyRef, { width }] = useElementSize<HTMLDivElement>()
  const model = findModel(settings.modelId) ?? getModel('qwen3-8b')

  const userEmpty = !settings.userText.trim()
  const text = source === 'repeat' ? REPEAT_EXAMPLE : userEmpty ? DEFAULT_SETTINGS.userText : settings.userText
  const prompt = useMemo(() => (tokenizer ? buildPromptTokens(tokenizer, text, LIMIT) : null), [tokenizer, text])
  const patterns = useMemo(() => (prompt ? headPatterns(prompt.tokens) : []), [prompt])
  const labels = useMemo(() => (prompt ? prompt.tokens.map(tokenLabel) : []), [prompt])
  const head = patterns.find((h) => h.id === headId) ?? patterns[0]
  const n = prompt?.tokens.length ?? 0
  const clamp = (i: number | null) => (i === null ? null : Math.max(0, Math.min(n - 1, i)))
  const focus = clamp(hover) ?? clamp(pinned) ?? n - 1
  const row = head?.weights[focus] ?? []
  const layoutKey = `${source}|${text}|${head?.id ?? ''}`
  // сколько места нужно ряду токенов с дугами (меряет ArcDiagram); не влезает: ряд с переносами и столбики
  const [need, setNeed] = useState<{ key: string; w: number } | null>(null)
  const needW = need && need.key === `${source}|${text}` ? need.w : 0
  const arcsFit = width >= 560 && needW <= width
  const pickedOrLast = clamp(pinned) ?? n - 1
  const roving = useRoving(n, pickedOrLast)
  const touched = hover !== null || pinned !== null

  const best = useMemo(() => {
    let b = 0
    row.forEach((w, j) => {
      if (w > (row[b] ?? 0)) b = j
    })
    return b
  }, [row])

  const ready = status === 'ready' && prompt && head

  return (
    <Figure
      id={id}
      title="Ваш промпт глазами разных голов"
      width="wide"
      actions={
        <Segmented<Source>
          size="sm"
          aria-label="Какой текст показать"
          value={source}
          onChange={(v) => {
            setSource(v)
            setPinned(null)
          }}
          options={[
            { value: 'user', label: 'Ваш запрос' },
            { value: 'repeat', label: 'С повтором' },
          ]}
        />
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> выберите голову «Induction», переключитесь на текст с повтором и наведите курсор на
          последний токен «·KV» (на телефоне нажмите на него): голова смотрит на «·cache», который шёл после прошлого «·KV». В настоящей модели таких голов{' '}
          {model.heads} в каждом из {model.layers} слоёв ({model.name}), и каждая выучивает своё; паттерны здесь иллюстративные, а не веса настоящей модели.
        </>
      }
    >
      <div ref={bodyRef} className="grid min-w-0 gap-5">
        {status === 'error' ? (
          <p className="text-sm text-ink-2">Не удалось загрузить токенизатор: {error ?? 'ошибка сети'}. Обновите страницу, чтобы попробовать ещё раз.</p>
        ) : !ready ? (
          <div className="grid gap-4">
            <TokenStripSkeleton count={20} />
            <Skeleton height={320} radius={12} label="Готовим паттерны голов" />
          </div>
        ) : (
          <>
            <div className="grid gap-1.5 @[760px]:hidden">
              <Select label="Голова" value={head.id} onChange={setHeadId} options={patterns.map((h) => ({ value: h.id, label: h.name }))} />
              <p className="text-[13px] leading-snug text-ink-2">{head.description}</p>
            </div>
            <div className="grid min-w-0 gap-2">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink-2">
                <span>
                  {source === 'repeat' ? 'Пример с повтором' : userEmpty ? 'Запрос пуст, показан пример' : 'Ваша реплика'} в шаблоне Qwen3:{' '}
                  {prompt.truncated ? `первые ${fmt.int(n)} из ${fmt.tokens(prompt.total)}` : fmt.tokens(n)}
                </span>
                <Badge>иллюстрация</Badge>
              </div>
              <p className="text-[13px] leading-snug text-ink-3">
                Первые три токена добавил шаблон чата: <span className="font-mono">{'<|im_start|>'}</span>, <span className="font-mono">user</span> и перевод
                строки. Наведите на токен или нажмите на него, чтобы увидеть, куда он смотрит.
              </p>
              {arcsFit ? (
                <ArcDiagram
                  count={n}
                  focus={focus}
                  scrollTo={pickedOrLast}
                  weights={row}
                  onFocus={(i) => setPinned(i)}
                  onHover={setHover}
                  onMeasure={(w) => setNeed({ key: `${source}|${text}`, w })}
                  groupLabel="Выберите токен: чьё внимание показать"
                  itemLabel={(i) => labels[i]}
                  renderItem={(i, f) => <TokenChip token={prompt.tokens[i]} size="sm" alt={i % 2 === 1} highlight={f} tooltip={false} />}
                  height={130}
                  gap={3}
                  layoutKey={layoutKey}
                />
              ) : (
                <div className="grid min-w-0 gap-2">
                  <div className="flex flex-wrap gap-1" role="toolbar" aria-label="Выберите токен: чьё внимание показать" ref={roving.container}>
                    {prompt.tokens.map((t, i) => {
                      const r = roving.item(i)
                      return (
                        <button
                          key={i}
                          ref={r.ref}
                          tabIndex={r.tabIndex}
                          onFocus={r.onFocus}
                          type="button"
                          aria-pressed={i === focus}
                          aria-label={labels[i]}
                          onClick={() => setPinned(i)}
                          className="rounded-chip focus-visible:outline-offset-1"
                        >
                          <TokenChip token={t} size="sm" alt={i % 2 === 1} highlight={i === focus} tooltip={false} />
                        </button>
                      )
                    })}
                  </div>
                  <BarChart
                    data={row
                      .map((w, j) => ({ id: String(j), label: labels[j], value: w }))
                      .filter((d, j) => j <= focus && d.value >= 0.005)
                      .sort((a, b) => b.value - a.value)
                      .slice(0, 6)}
                    max={1}
                    valueFormat={(v) => fmt.pct(v)}
                    labelFont="mono"
                    aria-label={`Куда смотрит токен ${labels[focus]}`}
                  />
                </div>
              )}
              <FigureLive as="p" className="text-sm text-ink-2">
                Токен <span className="font-mono text-ink">{labels[focus]}</span> (№ {focus + 1}) больше всего смотрит на{' '}
                <span className="font-mono text-ink">{labels[best]}</span>
                {best === focus ? ' (на себя)' : ''}: {fmt.pct(row[best] ?? 0)}.
              </FigureLive>
            </div>

            <div className="grid min-w-0 items-start gap-5 @[760px]:grid-cols-[minmax(0,1fr)_260px]">
              <Heatmap
                data={head.weights}
                rowLabels={labels}
                colLabels={labels}
                domain={[0, 1]}
                valueFormat={(v) => fmt.pct(v)}
                isMasked={(r, c) => c > r}
                highlight={touched ? { rows: [focus] } : undefined}
                maxCellSize={20}
                minCellSize={14}
                showValues={false}
                rowAxisLabel="кто смотрит"
                colAxisLabel="на кого смотрит"
                legendLabel="Вес"
                onHover={(c) => setHover(c ? c.row : null)}
                onClick={(c) => setPinned(c.row)}
                aria-label={`Веса головы «${head.name}»: строки кто смотрит, столбцы на кого`}
              />
              <div className="hidden min-w-0 @[760px]:grid">
                <HeadList heads={patterns} value={head.id} onChange={setHeadId} />
              </div>
            </div>
          </>
        )}
      </div>
    </Figure>
  )
}

/** Вертикальный выбор головы (radiogroup, стрелки вверх и вниз). Описание видно у выбранной. */
function HeadList({ heads, value, onChange }: { heads: ReturnType<typeof headPatterns>; value: string; onChange: (id: string) => void }) {
  const refs = useRef<Array<HTMLButtonElement | null>>([])
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = heads.findIndex((h) => h.id === value)
    let next = i
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') next = (i + 1) % heads.length
    else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') next = (i - 1 + heads.length) % heads.length
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = heads.length - 1
    else return
    e.preventDefault()
    onChange(heads[next].id)
    refs.current[next]?.focus()
  }
  return (
    <div className="grid min-w-0 gap-1.5">
      <div id="attention-heads-label" className="text-sm font-medium text-ink-2">
        Голова
      </div>
      <div role="radiogroup" aria-labelledby="attention-heads-label" onKeyDown={onKey} className="grid gap-1">
        {heads.map((h, i) => {
          const checked = h.id === value
          return (
            <button
              key={h.id}
              ref={(el) => {
                refs.current[i] = el
              }}
              type="button"
              role="radio"
              aria-checked={checked}
              tabIndex={checked ? 0 : -1}
              onClick={() => onChange(h.id)}
              className={cn(
                'grid gap-1 rounded-control px-3 py-2 text-left transition-colors focus-visible:-outline-offset-2',
                checked ? 'bg-accent-soft shadow-[inset_0_0_0_1.5px_var(--accent)]' : 'hover:bg-surface-3',
              )}
            >
              <span className={cn('text-[14px] leading-snug', checked ? 'font-semibold text-ink' : 'text-ink-2')}>{h.name}</span>
              {checked && <span className="text-[13px] leading-snug text-ink-2">{h.description}</span>}
            </button>
          )
        })}
      </div>
    </div>
  )
}
