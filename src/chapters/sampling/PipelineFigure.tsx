import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react'
import { useReducedMotion } from 'motion/react'
import { ArrowsHorizontalIcon } from '@phosphor-icons/react'
import { Figure, Player, fmt, usePlayer } from '@/ui'
import { cn } from '@/ui/cn'
import { softmax, topPMask } from '@/engine/sampling'

/**
 * Порядок обработки логитов в сэмплере (как в vLLM V1): маска грамматики → logit_bias → штрафы →
 * temperature → отсечения (здесь top-p) → выбор. Условный пример: запрос просит ответ из списка городов,
 * поднимает « Лион» через logit_bias, а presence_penalty штрафует « Париж», который уже был в ответе.
 */
const CANDS = [
  { text: ' Париж', z: 6.2 },
  { text: ' город', z: 4.1 },
  { text: ' это', z: 3.6 },
  { text: ' Марсель', z: 2.3 },
  { text: ' конечно', z: 1.9 },
  { text: ' Лион', z: 1.6 },
]
const ALLOWED = new Set([' Париж', ' Марсель', ' Лион'])
const BIAS: Record<string, number> = { ' Лион': 1.5 }
const PRESENCE = 1.5
const SEEN = new Set([' Париж'])
const T = 0.7
const TOP_P = 0.9
const U = 0.93

interface Column {
  title: string
  values: Array<number | null>
  /** Формат значения: логит или вероятность. */
  kind: 'logit' | 'prob'
  changed: boolean[]
  note: ReactNode
}

function buildColumns() {
  const z0 = CANDS.map((c) => c.z)
  const masked = z0.map((z, i) => (ALLOWED.has(CANDS[i].text) ? z : -Infinity))
  const biased = masked.map((z, i) => (z === -Infinity ? z : z + (BIAS[CANDS[i].text] ?? 0)))
  const penal = biased.map((z, i) => (z === -Infinity ? z : z - (SEEN.has(CANDS[i].text) ? PRESENCE : 0)))
  const scaled = penal.map((z) => (z === -Infinity ? z : z / T))
  const probs = softmax(penal, T)
  const keep = topPMask(probs, TOP_P, probs.map((p) => p > 0))
  const mass = probs.reduce((s, p, i) => (keep[i] ? s + p : s), 0)
  const finalP = probs.map((p, i) => (keep[i] ? p / mass : 0))
  let acc = 0
  let chosen = -1
  finalP.forEach((p, i) => {
    if (chosen >= 0 || p <= 0) return
    acc += p
    if (U < acc) chosen = i
  })
  const diff = (a: number[], b: number[]) => a.map((x, i) => x !== b[i])
  const cols: Column[] = [
    { title: 'Логиты', values: z0, kind: 'logit', changed: z0.map(() => false), note: 'Из LM head приходит по логиту на каждую строку словаря (здесь 6 из 151 936).' },
    {
      title: 'Маска',
      values: masked,
      kind: 'logit',
      changed: diff(masked, z0),
      note: 'Structured output: запрос разрешает только названия городов, остальным ставится −∞. Их вероятность станет ровно 0.',
    },
    { title: 'logit_bias', values: biased, kind: 'logit', changed: diff(biased, masked), note: 'Клиент прибавил +1,5 к «·Лион» через logit_bias.' },
    {
      title: 'Штрафы',
      values: penal,
      kind: 'logit',
      changed: diff(penal, biased),
      note: `presence_penalty ${fmt.num(PRESENCE)}: «·Париж» уже встречался в ответе, из его логита вычитается ${fmt.num(PRESENCE)}.`,
    },
    { title: `÷ T (${fmt.num(T)})`, values: scaled, kind: 'logit', changed: diff(scaled, penal), note: `Temperature ${fmt.num(T)}: логиты делятся на T, разрыв между ними растёт.` },
    { title: 'softmax', values: probs, kind: 'prob', changed: probs.map(() => true), note: 'Softmax превращает логиты в вероятности; у замаскированных ровно 0.' },
    {
      title: `top-p ${fmt.num(TOP_P)}`,
      values: finalP,
      kind: 'prob',
      changed: finalP.map((p, i) => p !== probs[i]),
      note: `top-p ${fmt.num(TOP_P)}: самый маловероятный город отсечён, остальные перенормированы до суммы 1.`,
    },
  ]
  return { cols, chosen, finalP }
}

type Edges = 'none' | 'start' | 'end' | 'both'

/** Есть ли у прокручиваемой таблицы скрытая часть и с какой стороны (слушаем прокрутку самого блока, не окна). */
function useScrollEdges(ref: RefObject<HTMLDivElement | null>): Edges {
  const [edges, setEdges] = useState<Edges>('none')
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => {
      const max = el.scrollWidth - el.clientWidth
      let e: Edges = 'none'
      if (max > 1) {
        const left = el.scrollLeft > 1
        const right = el.scrollLeft < max - 1
        e = left && right ? 'both' : left ? 'start' : 'end'
      }
      setEdges((prev) => (prev === e ? prev : e))
    }
    update()
    el.addEventListener('scroll', update, { passive: true })
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(update) : null
    ro?.observe(el)
    return () => {
      el.removeEventListener('scroll', update)
      ro?.disconnect()
    }
  }, [ref])
  return edges
}

export function PipelineFigure() {
  const { cols, chosen } = useMemo(buildColumns, [])
  const steps = cols.length + 1
  const p = usePlayer(steps, { intervalMs: 1700 })
  const shownCols = Math.min(p.step + 1, cols.length)
  const atPick = p.step >= cols.length
  const labels = [...cols.map((c) => c.title), `Выбор: u = ${fmt.num(U, 2)}`]
  const scroller = useRef<HTMLDivElement | null>(null)
  const edges = useScrollEdges(scroller)
  // справа затухание, пока есть что прокручивать; слева столбец «Токен» закреплён, под ним уезжают числа
  const fadeRight = edges === 'end' || edges === 'both'
  const slid = edges === 'start' || edges === 'both'
  const reduce = useReducedMotion()
  // на узком экране таблица прокручивается: держим текущий столбец в поле зрения
  useEffect(() => {
    const box = scroller.current
    const th = box?.querySelector<HTMLElement>(`th[data-col="${shownCols - 1}"]`)
    if (!box || !th) return
    const right = th.offsetLeft + th.offsetWidth + 8
    const target = shownCols <= 1 ? 0 : Math.max(0, right - box.clientWidth)
    if (Math.abs(target - box.scrollLeft) > 2 && (target > box.scrollLeft || shownCols <= 1)) {
      box.scrollTo({ left: target, behavior: reduce ? 'instant' : 'smooth' })
    }
  }, [shownCols, reduce])
  const note = atPick ? (
    <>
      Выбор: случайное число u = {fmt.num(U, 2)} из генератора запроса (его задаёт seed) попадает в отрезок «{CANDS[chosen].text.trim()}». Без штрафа
      и logit_bias почти наверняка был бы «Париж»: обработчики логитов меняют ответ, а не только его стиль.
    </>
  ) : (
    cols[shownCols - 1].note
  )

  return (
    <Figure
      id="sampling-figure-pipeline"
      title="Что происходит с логитами между LM head и выбором"
      controls={<Player {...p.bind} labels={labels} aria-label="Этапы обработки логитов" />}
      caption="Попробуйте: пройдите этапы по одному. Пример условный, порядок этапов как в сэмплере vLLM V1."
    >
      <div className="grid min-w-0 gap-4">
        <div
          ref={scroller}
          className="min-w-0 overflow-x-auto rounded-control p-0.5 scrollbar-thin"
          style={fadeRight ? { maskImage: 'linear-gradient(to right, #000 calc(100% - 28px), transparent)', WebkitMaskImage: 'linear-gradient(to right, #000 calc(100% - 28px), transparent)' } : undefined}
        >
          <table className="w-full min-w-[560px] border-separate border-spacing-0 text-left text-[13px] tabular-nums">
            <thead>
              <tr>
                <th
                  className={cn(
                    'sticky left-0 bg-surface py-1.5 pr-3 font-medium text-ink-3',
                    // тень влево закрывает поле прокрутки (2px), иначе в нём видны уехавшие числа
                    slid ? 'shadow-[-2px_0_0_var(--surface),inset_-1px_0_0_var(--line-strong)]' : 'shadow-[-2px_0_0_var(--surface)]',
                  )}
                >
                  Токен
                </th>
                {cols.map((c, ci) => (
                  <th
                    key={c.title}
                    data-col={ci}
                    className={cn('px-2 py-1.5 text-right font-medium whitespace-nowrap', ci < shownCols ? 'text-ink-2' : 'text-ink-3', ci === shownCols - 1 && !atPick && 'text-ink')}
                  >
                    {c.title}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {CANDS.map((cand, ri) => {
                const isChosen = atPick && ri === chosen
                return (
                  <tr key={cand.text} className={cn(isChosen && 'bg-accent-soft')}>
                    <td
                      className={cn(
                        'sticky left-0 border-t border-line py-1.5 pr-3 font-mono whitespace-nowrap text-ink',
                        isChosen ? 'font-semibold' : 'bg-surface',
                        slid ? 'shadow-[-2px_0_0_var(--surface),inset_-1px_0_0_var(--line-strong)]' : 'shadow-[-2px_0_0_var(--surface)]',
                      )}
                      // та же подложка, что у строки (accent-soft поверх surface), но непрозрачная: под ячейкой уезжают числа
                      style={isChosen ? { background: 'linear-gradient(var(--accent-soft), var(--accent-soft)), var(--surface)' } : undefined}
                    >
                      {cand.text.replace(/^ /, '·')}
                    </td>
                    {cols.map((c, ci) => {
                      const v = c.values[ri]
                      const visible = ci < shownCols
                      const current = ci === shownCols - 1 && !atPick
                      const dead = v === -Infinity || (c.kind === 'prob' && v === 0)
                      return (
                        <td
                          key={c.title}
                          className={cn(
                            'border-t border-line px-2 py-1.5 text-right whitespace-nowrap',
                            current && 'bg-surface-2',
                            visible && c.changed[ri] && current && 'font-semibold',
                            visible && dead && 'text-ink-3',
                          )}
                        >
                          {!visible || v === null ? '' : v === -Infinity ? '−∞' : c.kind === 'prob' ? fmt.pct(v) : fmt.num(v, 2)}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {edges !== 'none' && (
          <div className="-mt-2 flex items-center gap-1.5 text-[12.5px] leading-snug text-ink-3">
            <ArrowsHorizontalIcon size={16} aria-hidden className="shrink-0" />
            таблица прокручивается вбок
          </div>
        )}
        <p className="min-h-[3em] text-[15px] leading-snug text-ink" aria-live="polite">
          {note}
        </p>
      </div>
    </Figure>
  )
}
