import { useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { CheckCircleIcon, WarningIcon, XCircleIcon } from '@phosphor-icons/react'
import { FieldHint, FieldLabel, Figure, Formula, cn, fmt, measureText, seqColor, seqInk, useElementSize } from '@/ui'
import { FORMATS, formatRange, gridNeighbors, parseNumberInput, positiveValues, quantize, type NumberFormat, type QuantResult } from './formats'

const PRESETS: Array<{ label: string; value: string }> = [
  { label: '0,1', value: '0,1' },
  { label: 'π', value: '3,14159265' },
  { label: '300', value: '300' },
  { label: '500', value: '500' },
  { label: '0,001', value: '0,001' },
  { label: '70\u00a0000', value: '70000' },
  { label: '−2,5', value: '-2,5' },
]

/** Цвета полей: знак нейтральный, экспонента и мантисса двумя оттенками последовательной шкалы. */
const EXP_T = 5 / 7
const MAN_T = 1 / 7
export const FIELD_STYLE = {
  sign: { background: 'var(--surface-3)', color: 'var(--ink)', boxShadow: 'inset 0 0 0 1px var(--line-strong)' },
  exp: { background: seqColor(EXP_T), color: seqInk(EXP_T) },
  man: { background: seqColor(MAN_T), color: seqInk(MAN_T) },
} as const

type FieldKind = keyof typeof FIELD_STYLE

function fieldOf(f: NumberFormat, i: number): FieldKind {
  if (i === 0) return 'sign'
  if (f.kind === 'int') return 'man'
  return i <= f.exp ? 'exp' : 'man'
}

const SUP: Record<string, string> = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' }

/** Число для таблицы: до 7 значащих цифр, очень большие и маленькие как 1,4·10⁻⁴⁵. */
export function fmtValue(v: number): string {
  if (Number.isNaN(v)) return 'NaN'
  if (!Number.isFinite(v)) return v > 0 ? '+∞' : '−∞'
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e7 || a < 1e-4) {
    const [m, e] = a.toExponential(3).split('e')
    const mant = fmt.num(Number(m), 3).replace(/0+$/, '').replace(/,$/, '')
    const exp = String(Number(e)).split('').map((ch) => SUP[ch] ?? ch).join('')
    return `${v < 0 ? '−' : ''}${mant}·10${exp}`
  }
  const digits = Math.max(0, Math.min(10, 7 - Math.floor(Math.log10(a)) - 1))
  const s = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: digits, useGrouping: true }).format(v)
  return s.replace('-', '−')
}

/** То же число для KaTeX. */
function texNum(v: number): string {
  if (!Number.isFinite(v)) return v > 0 ? '+\\infty' : '-\\infty'
  const a = Math.abs(v)
  if (a !== 0 && (a >= 1e7 || a < 1e-4)) {
    const [m, e] = a.toExponential(3).split('e')
    const mant = fmt.num(Number(m), 3).replace(/0+$/, '').replace(/,$/, '').replace(',', '{,}')
    return `${v < 0 ? '-' : ''}${mant}\\cdot 10^{${Number(e)}}`
  }
  return fmtValue(v).replace(/\s/g, '\\,').replace(',', '{,}').replace('−', '-')
}

function fmtErr(r: QuantResult): string {
  if (!Number.isFinite(r.value) || !Number.isFinite(r.relError)) return '∞'
  if (r.relError === 0) return '0'
  if (r.relError < 1e-6) return '< 0,0001 %'
  const sign = r.absError > 0 ? '+' : '−'
  return sign + fmt.pct(r.relError, r.relError < 0.01 ? 3 : r.relError < 0.1 ? 2 : 1)
}

function Status({ r }: { r: QuantResult }) {
  if (r.overflow)
    return (
      <span className="inline-flex items-center gap-1 text-[12.5px] text-ink-2">
        <XCircleIcon size={15} weight="fill" className="shrink-0 text-bad" aria-hidden />
        {Number.isFinite(r.value) ? 'насыщение' : 'переполнение'}
      </span>
    )
  if (r.underflow)
    return (
      <span className="inline-flex items-center gap-1 text-[12.5px] text-ink-2">
        <XCircleIcon size={15} weight="fill" className="shrink-0 text-bad" aria-hidden />
        стало нулём
      </span>
    )
  if (r.relError > 0.05)
    return (
      <span className="inline-flex items-center gap-1 text-[12.5px] text-ink-2">
        <WarningIcon size={15} weight="fill" className="shrink-0 text-warn" aria-hidden />
        грубо
      </span>
    )
  return (
    <span className="inline-flex items-center gap-1 text-[12.5px] text-ink-2">
      <CheckCircleIcon size={15} weight="fill" className="shrink-0 text-good" aria-hidden />
      {r.relError === 0 ? 'точно' : 'близко'}
    </span>
  )
}

/** Номера бит по полям: знак, экспонента, мантисса (у INT: знак и биты числа). */
function bitGroups(f: NumberFormat): number[][] {
  const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i)
  return f.kind === 'float' ? [[0], range(1, 1 + f.exp), range(1 + f.exp, f.bits)] : [[0], range(1, f.bits)]
}

/**
 * Биты числа, раскрашенные по полям. Поля стоят отдельными группами: на узком экране длинная мантисса FP32
 * переносится на следующую строку целиком, а клетки остаются читаемыми (11 px и больше).
 */
function Bits({ f, r, size = 'sm' }: { f: NumberFormat; r: QuantResult; size?: 'sm' | 'md' }) {
  return (
    <span className={cn('inline-flex flex-wrap gap-y-1', size === 'md' ? 'gap-x-[5px]' : 'gap-x-[3px] @md:gap-x-1')} aria-label={`биты ${r.bits.join('')}`}>
      {bitGroups(f).map((group, gi) => (
        <span key={gi} className={cn('inline-flex flex-wrap gap-y-1', size === 'md' ? 'gap-x-[2px]' : 'gap-x-px @md:gap-x-[2px]')}>
          {group.map((i) => (
            <span
              key={i}
              className={cn(
                'inline-flex items-center justify-center rounded-[3px] font-mono tabular-nums',
                size === 'md' ? 'h-7 w-[18px] text-[13px]' : 'h-[22px] w-[11px] text-[10.5px] @3xl:w-[14px] @3xl:text-[11.5px]',
              )}
              style={FIELD_STYLE[fieldOf(f, i)]}
            >
              {r.bits[i]}
            </span>
          ))}
        </span>
      ))}
    </span>
  )
}

export function FormatsExplorer() {
  const [text, setText] = useState('0,1')
  // таблица держит последнее корректное число: пока в поле «abc», она не притворяется, что ввели 0
  const [x, setX] = useState(0.1)
  const [sel, setSel] = useState<NumberFormat['id']>('e4m3')
  const inputId = useId()
  const hintId = useId()
  const parsed = parseNumberInput(text)
  const invalid = parsed === null
  const update = (t: string) => {
    setText(t)
    const v = parseNumberInput(t)
    if (v !== null) setX(v)
  }
  const results = useMemo(() => FORMATS.map((f) => ({ f, r: quantize(x, f) })), [x])
  const refs = useRef<Array<HTMLButtonElement | null>>([])
  const selIdx = FORMATS.findIndex((f) => f.id === sel)
  const cur = results[selIdx]

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) return
    e.preventDefault()
    let n = selIdx
    if (e.key === 'ArrowDown') n = (selIdx + 1) % FORMATS.length
    else if (e.key === 'ArrowUp') n = (selIdx - 1 + FORMATS.length) % FORMATS.length
    else if (e.key === 'Home') n = 0
    else n = FORMATS.length - 1
    setSel(FORMATS[n].id)
    refs.current[n]?.focus()
  }

  return (
    <Figure
      id="quantization-figure-formats"
      title="Одно число в восьми форматах"
      controlsPosition="top"
      controls={
        <div className="col-span-full grid gap-3 @2xl:grid-cols-[minmax(0,260px)_minmax(0,1fr)] @2xl:items-end">
          <div className="grid gap-1.5">
            <FieldLabel htmlFor={inputId}>Число</FieldLabel>
            <input
              id={inputId}
              value={text}
              inputMode="decimal"
              autoComplete="off"
              spellCheck={false}
              aria-describedby={hintId}
              aria-invalid={invalid}
              onChange={(e) => update(e.currentTarget.value)}
              className="block h-10 w-full rounded-control bg-surface-2 px-3 font-mono text-[15px] text-ink shadow-[inset_0_0_0_1px_var(--line)] transition-shadow hover:shadow-[inset_0_0_0_1px_var(--line-strong)] focus-visible:outline-2 focus-visible:outline-offset-0 aria-invalid:shadow-[inset_0_0_0_1.5px_var(--bad)] aria-invalid:outline-bad"
            />
          </div>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Примеры чисел">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                aria-pressed={text === p.value}
                onClick={() => update(p.value)}
                className={cn(
                  'h-8 rounded-control px-2.5 font-mono text-[13px] transition-colors',
                  text === p.value
                    ? 'bg-accent-soft text-accent-hover shadow-[inset_0_0_0_1px_var(--accent)]'
                    : 'bg-surface text-ink-2 shadow-[inset_0_0_0_1px_var(--line-strong)] hover:bg-surface-3 hover:text-ink',
                )}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="col-span-full">
            <FieldHint id={hintId}>
              {invalid ? (
                <span className="inline-flex flex-wrap items-center gap-1">
                  <WarningIcon size={14} weight="fill" className="text-warn" aria-hidden />
                  Не похоже на число. Примеры: 0,1, −3,5, 1e-3. Ниже пока последнее число: {fmtValue(x)}.
                </span>
              ) : (
                'Запятая или точка, можно 1e-3. Каждая строка показывает ближайшее число, которое формат умеет хранить, и его биты.'
              )}
            </FieldHint>
          </div>
        </div>
      }
      caption="Попробуйте: введите 0,1 и сравните ошибку FP8 E4M3 (+1,56 %) и E5M2 (−6,25 %); затем 500: E4M3 упирается в свой максимум 448, а INT8 без масштаба хранит только целые от −128 до 127. Округление к ближайшему, при равенстве к чётному; E4M3 и FP4 насыщаются, у них нет бесконечности."
    >
      <div className={cn('grid min-w-0 gap-5 transition-opacity', invalid && 'opacity-50')}>
        <div className="min-w-0">
          <div role="radiogroup" aria-label="Форматы" onKeyDown={onKey} className="grid gap-1">
            <div className="hidden grid-cols-[112px_minmax(0,1fr)_118px_96px_118px] items-end gap-3 px-2.5 pb-1 text-[12px] text-ink-3 @3xl:grid">
              <span>Формат</span>
              <span>Биты ближайшего значения</span>
              <span className="text-right">Хранится</span>
              <span className="text-right">Ошибка</span>
              <span />
            </div>
            {results.map(({ f, r }, i) => (
              <button
                key={f.id}
                ref={(el) => {
                  refs.current[i] = el
                }}
                type="button"
                role="radio"
                aria-checked={f.id === sel}
                tabIndex={f.id === sel ? 0 : -1}
                onClick={() => setSel(f.id)}
                className={cn(
                  'grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1.5 rounded-control px-1.5 py-1.5 text-left transition-colors focus-visible:-outline-offset-2 @md:px-2.5',
                  '@3xl:grid-cols-[112px_minmax(0,1fr)_118px_96px_118px]',
                  f.id === sel ? 'bg-accent-soft shadow-[inset_0_0_0_1px_var(--accent)]' : 'hover:bg-surface-3/60',
                )}
              >
                <span className="col-start-1 row-start-1 grid leading-tight">
                  <span className="text-[14px] font-medium text-ink">{f.name}</span>
                  <span className="text-[12px] text-ink-2">{fmt.count(f.bits, ['бит', 'бита', 'бит'])}</span>
                </span>
                <span className="col-span-2 col-start-1 row-start-2 min-w-0 @3xl:col-span-1 @3xl:col-start-2 @3xl:row-start-1">
                  <Bits f={f} r={r} />
                </span>
                <span className="col-start-2 row-start-1 grid justify-items-end gap-0.5 @3xl:contents">
                  <span className="text-right font-mono text-[13px] text-ink tabular-nums @3xl:col-start-3 @3xl:row-start-1">{fmtValue(r.value)}</span>
                  <span className="flex items-center gap-2 @3xl:contents">
                    <span className="text-right font-mono text-[12.5px] text-ink-2 tabular-nums @3xl:col-start-4 @3xl:row-start-1 @3xl:text-[13px]">{fmtErr(r)}</span>
                    <span className="justify-self-end @3xl:col-start-5 @3xl:row-start-1">
                      <Status r={r} />
                    </span>
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[13px] text-ink-2">
          {(['sign', 'exp', 'man'] as const).map((k) => (
            <span key={k} className="inline-flex items-center gap-1.5">
              <span className="h-3 w-3 rounded-[3px]" style={{ background: FIELD_STYLE[k].background, boxShadow: k === 'sign' ? FIELD_STYLE.sign.boxShadow : undefined }} />
              {k === 'sign' ? 'знак' : k === 'exp' ? 'экспонента (порядок)' : 'мантисса (у INT: биты числа)'}
            </span>
          ))}
        </div>
        {cur && <Details f={cur.f} r={cur.r} x={x} />}
      </div>
    </Figure>
  )
}

function Details({ f, r, x }: { f: NumberFormat; r: QuantResult; x: number }) {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  // на узкой панели формула и «= значение» на двух строках, иначе итог уезжает за край
  const narrow = width > 0 && width < 460
  const range = formatRange(f)
  const nb = gridNeighbors(r, f)
  const step = nb.above !== null && nb.below !== null ? Math.min(Math.abs(nb.above - r.value), Math.abs(r.value - nb.below)) : null
  const twoLines = (lhs: string, rhs: string) => (narrow ? `\\begin{aligned}&${lhs}\\\\&${rhs}\\end{aligned}` : `${lhs}${rhs.startsWith('=') || rhs.startsWith('\\Rightarrow') ? '' : '\\quad'}${rhs}`)
  let tex: string
  if (f.kind === 'int') {
    // минус в фигурных скобках: унарный, без пробела бинарной операции («от −128», а не «от − 128»)
    tex = twoLines(`q = ${r.q ?? 0}`, `(\\text{дополнительный код, от }{${f.min}}\\text{ до }${f.max})`)
  } else if (r.kind === 'inf') {
    tex = twoLines(`E = ${r.expField}\\ (\\text{все единицы}),\\ M = 0`, `\\Rightarrow ${r.sign ? '-' : '+'}\\infty`)
  } else if (r.kind === 'zero') {
    tex = `E = 0,\\ M = 0 \\Rightarrow ${r.sign ? '-' : ''}0`
  } else if (r.kind === 'subnormal') {
    tex = twoLines(`(-1)^{${r.sign}}\\cdot 2^{1-${f.bias}}\\cdot\\frac{${r.manField}}{2^{${f.man}}}`, `= ${texNum(r.value)}`)
  } else {
    tex = twoLines(`(-1)^{${r.sign}}\\cdot 2^{${r.expField}-${f.bias}}\\cdot\\left(1+\\frac{${r.manField}}{2^{${f.man}}}\\right)`, `= ${texNum(r.value)}`)
  }
  return (
    <div ref={ref} className="grid min-w-0 gap-4 rounded-panel bg-surface-2/60 p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="text-[15px] font-semibold text-ink">
          {f.name}
          {f.kind === 'float' ? (
            <span className="ml-2 font-normal text-ink-2">
              1 бит знака, {fmt.count(f.exp, ['бит', 'бита', 'бит'])} экспоненты, {fmt.count(f.man, ['бит', 'бита', 'бит'])} мантиссы, смещение {f.bias}
            </span>
          ) : (
            <span className="ml-2 font-normal text-ink-2">целое со знаком, без масштаба</span>
          )}
        </div>
        <div className="text-[13px] text-ink-3">{f.note}</div>
      </div>
      <div className="min-w-0">
        <Bits f={f} r={r} size="md" />
      </div>
      <div className="min-w-0">
        <Formula block tex={tex} />
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-[14px] @2xl:grid-cols-4">
        <Fact term="Максимум" value={fmtValue(range.max)} />
        <Fact term={f.kind === 'int' ? 'Шаг сетки' : 'Мин. нормальное'} value={fmtValue(range.minNormal)} />
        <Fact term="Соседи по сетке" value={`${nb.below !== null ? fmtValue(nb.below) : 'нет'} и ${nb.above !== null ? fmtValue(nb.above) : 'нет'}`} />
        <Fact term="Кодов всего" value={fmt.int(range.codes)} hint={step !== null ? `шаг здесь ${fmtValue(step)}` : undefined} />
      </dl>
      <NumberLine f={f} r={r} x={x} />
    </div>
  )
}

function Fact({ term, value, hint }: { term: string; value: string; hint?: string }) {
  return (
    <div className="grid content-start gap-0.5">
      <dt className="text-[12.5px] text-ink-3">{term}</dt>
      <dd className="font-mono text-[13px] text-ink tabular-nums">{value}</dd>
      {hint && <dd className="text-[12px] text-ink-3">{hint}</dd>}
    </div>
  )
}

const TICK_FONT = '11px "Onest Variable", sans-serif'
/** 2 значащие цифры: «0,0078», «32». */
const sig2 = (v: number) => fmtValue(Number(v.toPrecision(2)))

/** Шаг сетки у значения v: расстояние до следующего представимого (у максимума до предыдущего). */
function stepAt(values: readonly number[], v: number): number {
  let i = 0
  while (i + 1 < values.length && values[i + 1] <= v) i++
  return i + 1 < values.length ? values[i + 1] - values[i] : values[i] - values[i - 1]
}

/** Положительная полуось в логарифмическом масштабе: все значения формата (до 8 бит) и ваше число. */
function NumberLine({ f, r, x }: { f: NumberFormat; r: QuantResult; x: number }) {
  const values = useMemo(() => positiveValues(f), [f])
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  // ширина SVG равна ширине панели: подписи остаются 11 px и на телефоне
  const W = Math.max(280, Math.round(width || 640))
  const H = 64
  const pad = 16
  if (values.length === 0) {
    const pos = f.kind === 'float' ? formatRange(f).finiteValues : 0
    return (
      <p className="text-[13.5px] leading-snug text-ink-2">
        У {f.name} около {fmt.compact((pos - 1) / 2)} положительных значений, на оси их не показать. Чем больше бит мантиссы, тем гуще сетка:
        относительный шаг у {f.name} около {fmtValue(Number((Math.pow(2, -(f.kind === 'float' ? f.man : 1)) * 100).toPrecision(3)))}&nbsp;% от значения.
      </p>
    )
  }
  const lo = Math.log2(values[0] / 2)
  const hi = Math.log2(values[values.length - 1] * 1.6)
  const sx = (v: number) => pad + ((Math.log2(v) - lo) / (hi - lo)) * (W - 2 * pad)
  const ax = Math.abs(x)
  const av = Math.abs(r.value)
  const inside = (v: number) => v > 0 && Number.isFinite(v) && Math.log2(v) >= lo && Math.log2(v) <= hi
  const first = values[0]
  const last = values[values.length - 1]
  const few = values.length <= 16
  // подписи: у FP4 и INT4 каждое значение; у остальных края и степени десяти (на узкой оси через одну), без наездов
  const candidates = few
    ? values
    : [first, last, ...[1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 1e4].filter((t) => t > first && t < last && (W >= 480 || Math.round(Math.log10(t)) % 2 === 0))]
  const kept: Array<{ t: number; x0: number; x1: number }> = []
  for (const t of candidates) {
    const w = measureText(fmtValue(t), TICK_FONT)
    const cx = Math.min(W - w / 2 - 1, Math.max(w / 2 + 1, sx(t)))
    const box = { t, x0: cx - w / 2 - 5, x1: cx + w / 2 + 5 }
    if (kept.every((k) => box.x1 <= k.x0 || box.x0 >= k.x1)) kept.push(box)
  }
  let lineNote: string
  if (f.kind === 'int') lineNote = ': сетка равномерная, поэтому к краю она сгущается'
  else if (few) lineNote = `: шаг растёт вместе с числом, у единицы он ${sig2(stepAt(values, 1))}, у максимума уже ${sig2(stepAt(values, last))}`
  else {
    const ref0 = 0.1 >= formatRange(f).minNormal && 0.1 < last ? 0.1 : 1
    lineNote = `: штрихи стоят почти равномерно, потому что шаг растёт вместе с числом (около ${fmtValue(ref0)} он ${sig2(stepAt(values, ref0))}, у максимума ${fmtValue(last)} уже ${sig2(stepAt(values, last))}); редкие штрихи слева это субнормальные числа у самого нуля`
  }
  return (
    <div ref={ref} className="min-w-0">
      <div className="mb-1 text-[13px] leading-snug text-ink-2">
        Все {fmt.int(values.length)} положительных значений {f.name} на логарифмической оси{lineNote}
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block max-w-full" role="img" aria-label={`Значения ${f.name} на оси и ваше число ${fmtValue(x)}`}>
        <line x1={pad} x2={W - pad} y1={30} y2={30} stroke="var(--line-strong)" />
        {values.map((v, i) => (
          <line key={i} x1={sx(v)} x2={sx(v)} y1={22} y2={38} stroke="var(--ink-3)" strokeWidth={1} opacity={0.7} />
        ))}
        {kept.map(({ t, x0, x1 }) => (
          <text key={t} x={(x0 + x1) / 2} y={56} textAnchor="middle" fontSize={11} fill="var(--ink-3)" fontFamily="Onest Variable, sans-serif">
            {fmtValue(t)}
          </text>
        ))}
        {inside(ax) && <path d={`M ${sx(ax)} 18 l -5 -8 h 10 z`} fill="var(--ink)" />}
        {inside(av) && <circle cx={sx(av)} cy={30} r={5} fill="var(--surface)" stroke="var(--ink)" strokeWidth={2} />}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <svg width="10" height="9" aria-hidden>
            <path d="M5 9 L0 1 H10 Z" fill="var(--ink)" />
          </svg>
          ваше число (по модулю){!inside(ax) && ax !== 0 ? ', за пределами оси' : ''}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="12" height="12" aria-hidden>
            <circle cx="6" cy="6" r="4" fill="var(--surface)" stroke="var(--ink)" strokeWidth="2" />
          </svg>
          ближайшее представимое
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="4" height="12" aria-hidden>
            <line x1="2" x2="2" y1="0" y2="12" stroke="var(--ink-3)" />
          </svg>
          значения формата (отрицательные зеркально)
        </span>
      </div>
    </div>
  )
}
