import { useId, useMemo, useRef, type CSSProperties, type PointerEvent } from 'react'
import { HourglassMediumIcon } from '@phosphor-icons/react'
import { Figure, Player, cssVar, fmt, tint, useElementSize, usePlayer, useTip } from '@/ui'
import { BLOCK, CAPACITY, REQS, buildScenario, meanUtilization, type Cell, type PanelSnapshot } from './scenario'

const ROWS = CAPACITY / BLOCK

/** Геометрия сетки: обычная (12 px клетка) и компактная для узкой фигуры (6 px, две сетки рядом). */
interface Geo {
  cell: number
  gap: number
  gutter: number
  head: number
  w: number
  h: number
}

function geo(compact: boolean): Geo {
  const cell = compact ? 6 : 12
  const gap = compact ? 1 : 2
  const gutter = compact ? 20 : 46
  const head = compact ? 0 : 14
  return { cell, gap, gutter, head, w: gutter + BLOCK * (cell + gap) - gap, h: head + ROWS * (cell + gap) - gap }
}

/** Ширина тела фигуры, ниже которой две сетки встают рядом в компактном виде. */
const COMPACT_BELOW = 640

function cellText(c: Cell, slot: number, paged: boolean, snap: PanelSnapshot): string {
  const where = paged ? `блок ${Math.floor(slot / BLOCK)}, слот ${slot % BLOCK}` : `адрес ${slot}`
  if (!c.owner) return `${where}: свободно`
  const r = REQS[c.owner]
  if (c.kind === 'prompt') return `${where}: ${c.owner}, токен промпта (промпт ${r.prompt})`
  if (c.kind === 'gen') return `${where}: ${c.owner}, сгенерированный токен`
  if (c.kind === 'reserved') return `${where}: ${c.owner}, забронировано под будущий ответ, пока пусто`
  const seq = snap.running.includes(c.owner) ? 'последний блок заполнится следующими токенами' : 'пусто'
  return `${where}: ${c.owner}, хвост последнего блока (${seq})`
}

function MemoryGrid({ snap, paged, label, compact }: { snap: PanelSnapshot; paged: boolean; label: string; compact: boolean }) {
  const ref = useRef<SVGSVGElement | null>(null)
  const tip = useTip({ scopeEl: ref.current, side: 'top' })
  const hatch = useId().replace(/:/g, '')
  const g = geo(compact)
  const rowLetters = useMemo(() => {
    const m = new Map<number, string[]>()
    for (const l of snap.labels) {
      const row = Math.floor(l.slot / BLOCK)
      const arr = m.get(row) ?? []
      if (!arr.includes(l.id)) arr.push(l.id)
      m.set(row, arr)
    }
    if (!compact) return m
    // компактная сетка: строка 7 px ниже буквы, поэтому соседние подписи пропускаем (кто где, скажет подсказка при наведении)
    const out = new Map<number, string[]>()
    let lastRow = -99
    let lastText = ''
    for (const row of [...m.keys()].sort((a, b) => a - b)) {
      const text = m.get(row)!.join(' ')
      if ((row - lastRow) * (geo(true).cell + geo(true).gap) < 11) continue
      if (text === lastText && row - lastRow <= 2) continue
      out.set(row, m.get(row)!)
      lastRow = row
      lastText = text
    }
    return out
  }, [snap.labels, compact])

  const at = (e: PointerEvent<SVGSVGElement>) => {
    const r = ref.current!.getBoundingClientRect()
    const x = e.clientX - r.left - g.gutter
    const y = e.clientY - r.top - g.head
    if (x < 0 || y < 0) return null
    const col = Math.floor(x / (g.cell + g.gap))
    const row = Math.floor(y / (g.cell + g.gap))
    if (col < 0 || col >= BLOCK || row < 0 || row >= ROWS) return null
    const slot = row * BLOCK + col
    return {
      slot,
      rect: { left: r.left + g.gutter + col * (g.cell + g.gap), top: r.top + g.head + row * (g.cell + g.gap), width: g.cell, height: g.cell },
    }
  }
  const lastSlot = useRef(-1)
  const move = (e: PointerEvent<SVGSVGElement>) => {
    const p = at(e)
    if (!p) {
      lastSlot.current = -1
      return tip.hide()
    }
    if (p.slot === lastSlot.current && tip.visible) return
    lastSlot.current = p.slot
    tip.show(p.rect, <span>{cellText(snap.cells[p.slot], p.slot, paged, snap)}</span>)
  }
  // 512 клеток: пересобираем только при смене шага, а не на каждое движение мыши
  const rows = useMemo(
    () =>
      Array.from({ length: ROWS }, (_, row) => {
        const y = g.head + row * (g.cell + g.gap)
        const letters = rowLetters.get(row)
        const showAddr = !compact && (paged || row % 4 === 0)
        return (
          <g key={row}>
            {showAddr && (
              <text x={paged ? 16 : 26} y={y + g.cell - 1.5} fontSize={11} textAnchor="end" fill="var(--ink-3)" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {paged ? row : row * BLOCK}
              </text>
            )}
            {letters && (
              <text
                x={compact ? g.gutter - 4 : paged ? 22 : 30}
                y={y + g.cell - (compact ? 0 : 1.5)}
                fontSize={compact ? 10 : 11}
                fontWeight={650}
                textAnchor={compact ? 'end' : 'start'}
                fill="var(--ink)"
              >
                {letters.join(' ')}
              </text>
            )}
            {Array.from({ length: BLOCK }, (_, col) => {
              const c = snap.cells[row * BLOCK + col]
              const x = g.gutter + col * (g.cell + g.gap)
              const empty = c.kind === 'reserved' || c.kind === 'waste'
              const fill = c.kind === 'prompt' ? cssVar('prefill') : c.kind === 'gen' ? cssVar('decode') : empty ? `url(#${hatch})` : 'var(--surface-2)'
              return (
                <rect
                  key={col}
                  x={x}
                  y={y}
                  width={g.cell}
                  height={g.cell}
                  rx={compact ? 1 : 2.5}
                  style={{ fill, transition: 'fill 220ms ease-out' }}
                  stroke={c.kind === 'free' ? 'var(--line)' : empty ? 'var(--line-strong)' : 'none'}
                  strokeWidth={compact ? 0.75 : 1}
                />
              )
            })}
          </g>
        )
      }),
    [snap.cells, rowLetters, paged, hatch, compact, g.head, g.cell, g.gap, g.gutter],
  )

  return (
    <div className={compact ? 'pb-1' : 'overflow-x-auto pb-1'}>
      <svg
        ref={ref}
        width={g.w}
        height={g.h}
        viewBox={`0 0 ${g.w} ${g.h}`}
        role="img"
        aria-label={label}
        onPointerMove={move}
        onPointerDown={move}
        onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
        className="block"
      >
        <defs>
          <pattern id={hatch} width={4} height={4} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width={4} height={4} fill="var(--surface-2)" />
            <line x1={0} y1={0} x2={0} y2={4} stroke="var(--ink-3)" strokeWidth={compact ? 1.1 : 1.4} strokeOpacity={0.55} />
          </pattern>
        </defs>
        {!compact && (
          <text x={0} y={9} fontSize={10} fill="var(--ink-3)">
            {paged ? 'блок' : 'адрес'}
          </text>
        )}
        {rows}
      </svg>
      {tip.node}
    </div>
  )
}

/** Сколько слотов занято токенами промпта и сгенерированными токенами. */
function usedSplit(snap: PanelSnapshot): { prompt: number; gen: number } {
  let prompt = 0
  let gen = 0
  for (const c of snap.cells) {
    if (c.kind === 'prompt') prompt++
    else if (c.kind === 'gen') gen++
  }
  return { prompt, gen }
}

function UsageBar({ snap, legend = true }: { snap: PanelSnapshot; legend?: boolean }) {
  const st = snap.stats
  const split = usedSplit(snap)
  const parts: Array<{ id: string; v: number; label: string; hatch?: boolean; style: CSSProperties }> = [
    { id: 'prompt', v: split.prompt, label: 'токены промпта', style: { background: cssVar('prefill') } },
    { id: 'gen', v: split.gen, label: 'сгенерированные токены', style: { background: cssVar('decode') } },
    { id: 'empty', v: st.allocatedSlots - st.usedSlots, label: 'выделено, но пусто', hatch: true, style: {} },
    { id: 'free', v: st.freeSlots, label: 'свободно', style: { background: 'var(--surface-3)' } },
  ]
  return (
    <div className="grid gap-1.5">
      <div className="flex h-2 gap-[2px] overflow-hidden rounded-full" aria-hidden>
        {parts
          .filter((p) => p.v > 0)
          .map((p) => (
            <div key={p.id} className={p.hatch ? 'bg-hatch bg-surface-2' : ''} style={{ ...p.style, flexGrow: p.v, flexBasis: 0, transition: 'flex-grow 250ms' }} />
          ))}
      </div>
      {legend && (
        <div className="grid gap-0.5 text-xs text-ink-2">
          {parts.map((p) => (
            <div key={p.id} className="flex items-center justify-between gap-3">
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden className={`inline-block h-2.5 w-2.5 rounded-[2px] ${p.hatch ? 'bg-hatch bg-surface-2' : ''}`} style={p.style} />
                {p.label}
              </span>
              <span className="text-ink tabular-nums">{fmt.int(p.v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** Большое число «доля выделенного, занятая токенами»; пока ничего не выделено, 0 % с пояснением. */
function UtilValue({ snap, size = 'lg' }: { snap: PanelSnapshot; size?: 'lg' | 'md' }) {
  const st = snap.stats
  return (
    <div className="flex flex-wrap items-baseline gap-x-2">
      <span className={size === 'lg' ? 'text-[1.625rem] leading-tight font-semibold text-ink' : 'text-xl leading-tight font-semibold text-ink'}>
        {fmt.pct(st.allocatedSlots ? st.utilization : 0)}
      </span>
      {!st.allocatedSlots && <span className="text-xs text-ink-3">пока ничего не выделено</span>}
    </div>
  )
}

function QueueAndNotes({ snap }: { snap: PanelSnapshot }) {
  return (
    <>
      <div className="grid gap-1 text-[13px]">
        <div className="text-ink-2">Очередь{snap.waiting.length ? '' : <span className="text-ink-3">: пусто</span>}</div>
        {snap.waiting.length ? (
          snap.waiting.map((w, i) => (
            <div key={w.id} className="flex items-start gap-1.5 rounded-control px-2 py-1.5 text-ink" style={{ background: tint('warn', 14) }}>
              <HourglassMediumIcon size={16} aria-hidden className="mt-px shrink-0" style={{ color: cssVar('warn') }} />
              <span>
                <b>{w.id}</b> {i === 0 ? `ждёт. ${w.reason}` : `ждёт за ${snap.waiting[0].id}: очередь FIFO`}
              </span>
            </div>
          ))
        ) : null}
      </div>
      {snap.notes.length > 0 && (
        <ul className="grid gap-1 text-[13px] leading-snug text-ink-2">
          {snap.notes.slice(0, 4).map((n, i) => (
            <li key={i} className="border-l-2 border-line-strong pl-2">
              {n}
            </li>
          ))}
        </ul>
      )}
    </>
  )
}

function panelLabel(title: string, snap: PanelSnapshot): string {
  const st = snap.stats
  return `${title}: занято токенами ${st.usedSlots} из ${st.allocatedSlots} выделенных слотов, свободно ${st.freeSlots}`
}

function Panel({ title, sub, snap, paged }: { title: string; sub: string; snap: PanelSnapshot; paged: boolean }) {
  return (
    <section className="@container grid min-w-0 content-start gap-3 rounded-panel border border-line p-3 sm:p-4" aria-label={title}>
      <div>
        <div className="text-[15px] font-semibold text-ink">{title}</div>
        <div className="text-[13px] text-ink-2">{sub}</div>
      </div>
      <div className="grid min-w-0 gap-4 @md:grid-cols-[auto_minmax(0,1fr)]">
        <MemoryGrid snap={snap} paged={paged} label={panelLabel(title, snap)} compact={false} />
        <div className="grid min-w-0 content-start gap-4">
          <div>
            <div className="text-sm text-ink-2">Выделенная память занята токенами</div>
            <UtilValue snap={snap} />
          </div>
          <UsageBar snap={snap} />
          <QueueAndNotes snap={snap} />
        </div>
      </div>
    </section>
  )
}

/** Узкая фигура: две сетки рядом, чтобы сравнение было видно на одном экране. */
function CompactPair({ s }: { s: ReturnType<typeof buildScenario>[number] }) {
  const panels = [
    { title: 'Непрерывная бронь', snap: s.contiguous, paged: false },
    { title: 'Блоки по требованию', snap: s.paged, paged: true },
  ]
  return (
    <div className="grid min-w-0 gap-4">
      <div className="grid grid-cols-2 gap-2">
        {panels.map((p) => (
          <section key={p.title} className="grid min-w-0 content-start justify-items-start gap-1.5" aria-label={p.title}>
            <div className="text-[13px] leading-tight font-semibold text-ink">{p.title}</div>
            <MemoryGrid snap={p.snap} paged={p.paged} label={panelLabel(p.title, p.snap)} compact />
            <div className="grid w-full gap-1">
              <div className="text-xs leading-tight text-ink-2">занято токенами</div>
              <UtilValue snap={p.snap} size="md" />
              <UsageBar snap={p.snap} legend={false} />
            </div>
          </section>
        ))}
      </div>
      <div className="grid gap-4 border-t border-line pt-3">
        {panels.map((p) => (
          <div key={p.title} className="grid gap-2">
            <div className="text-sm font-medium text-ink">{p.title}</div>
            <QueueAndNotes snap={p.snap} />
          </div>
        ))}
      </div>
    </div>
  )
}

export function AllocatorsFigure() {
  const snaps = useMemo(() => buildScenario(), [])
  // Открываемся на «Приходит C»: слева уже видны брони со штриховкой, справа разбросанные блоки.
  const p = usePlayer(snaps.length, { intervalMs: 1700, initialStep: 3 })
  const s = snaps[p.step]
  const meanC = meanUtilization(snaps, 'contiguous')
  const meanP = meanUtilization(snaps, 'paged')
  const last = snaps[snaps.length - 1]
  const waitC = (last.contiguous.admittedAt.D ?? 0) - 7
  const [boxRef, box] = useElementSize<HTMLDivElement>()
  const compact = box.width > 0 && box.width < COMPACT_BELOW
  return (
    <Figure
      id="paged-attention-figure-allocators"
      title="Одни и те же запросы в двух схемах памяти"
      width="wide"
      controls={<Player {...p.bind} labels={snaps.map((x) => x.label)} aria-label="Шаги сценария" />}
      caption="Попробуйте: дойдите до шага 9. Слева D стоит в очереди, хотя свободных слотов в сумме ему хватает, а E ждёт за ним; справа оба запроса уже получили блоки и со следующего шага генерируют. Наведите на клетку, чтобы увидеть, чья она."
    >
      <div ref={boxRef} className="grid min-w-0 gap-4">
        <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-[13px] text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-3 w-3 rounded-[3px]" style={{ background: cssVar('prefill') }} />
            токен промпта
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-3 w-3 rounded-[3px]" style={{ background: cssVar('decode') }} />
            сгенерированный токен
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="bg-hatch inline-block h-3 w-3 rounded-[3px] border border-line-strong bg-surface-2" />
            выделено, но пусто
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-3 w-3 rounded-[3px] border border-line bg-surface-2" />
            свободно
          </span>
          <span className="text-ink-3">
            1 клетка = 1 токен, строка = {BLOCK} слотов, всего {CAPACITY}
            {compact ? '; буква: чей запрос, точнее при наведении на клетку' : ''}
          </span>
        </div>
        {compact ? (
          <CompactPair s={s} />
        ) : (
          <div className="grid min-w-0 gap-4 @3xl:grid-cols-2">
            <Panel title="Непрерывная бронь" sub="под запрос сразу резервируется промпт + max_tokens подряд" snap={s.contiguous} paged={false} />
            <Panel title="Блоки по требованию (PagedAttention)" sub={`блок ${BLOCK} токенов, новый выдаётся, когда заполнен предыдущий`} snap={s.paged} paged />
          </div>
        )}
        <p className="text-sm text-ink-2">
          За весь сценарий в среднем: непрерывная бронь держит токенами{' '}
          <b className="font-semibold text-ink">{fmt.pct(meanC)}</b> выделенной памяти, блоки{' '}
          <b className="font-semibold text-ink">{fmt.pct(meanP)}</b>. Запрос D в непрерывной схеме прождал {waitC}{' '}
          {fmt.plural(waitC, ['шаг', 'шага', 'шагов'])}, а с блоками начал сразу.
        </p>
      </div>
    </Figure>
  )
}
