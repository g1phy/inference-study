import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode, type Ref } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CaretLeftIcon, CaretRightIcon } from '@phosphor-icons/react'
import { transpose } from '@/engine/attention'
import { getModel } from '@/engine/data/models'
import {
  Badge,
  BarChart,
  Button,
  ChapterLink,
  cn,
  Figure,
  fmt,
  Formula,
  MatrixView,
  measureText,
  Player,
  Segmented,
  Term as GlossaryTerm,
  useElementSize,
  useFontsReady,
  usePlayer,
  type MatrixCell,
  type MatrixHighlight,
  type PlayerState,
  ScrollX,
} from '@/ui'
import { ArcDiagram } from './ArcDiagram'
import { HEAD_H, OpGlyph, Sub, Swatch, useRoving, VLabel, WordChip } from './bits'
import { FormulaStrip } from './FormulaStrip'
import {
  activeFeatures,
  argmaxWeight,
  CASES,
  FEATURES,
  FOCUS,
  isUniformRow,
  N,
  outTerms,
  PROJ_DIMS,
  PROJ_W,
  projectionTerms,
  projOut,
  rankedHead,
  rankedRow,
  scoreTerms,
  softmaxParts,
  SQRT_DK,
  TIE,
  TOY,
  topIndices,
  TRACE,
  type Proj,
  type ToyCase,
  type Variant,
} from './toy'

/* ============================================================================
 * Шаги
 * ========================================================================== */

const STEP_LABELS = [
  'Слова как векторы X',
  'Запросы Q',
  'Ключи K',
  'Значения V',
  'Оценки Q·Kᵀ',
  'Делим на √2',
  'Causal mask',
  'Softmax по строкам',
  'Выход: веса · V',
  'Дуги внимания',
]
const ARCS_STEP = 9
const isQK = (step: number) => step >= 4 && step <= 7

const PROJ_OF_STEP: Record<number, Proj> = { 1: 'q', 2: 'k', 3: 'v' }
const PROJ_NAME: Record<Proj, { noun: string; gen: string; eq: string; zero: string; letter: string }> = {
  q: { noun: 'запрос', gen: 'запроса', eq: 'равен', zero: 'нулевой', letter: 'Q' },
  k: { noun: 'ключ', gen: 'ключа', eq: 'равен', zero: 'нулевой', letter: 'K' },
  v: { noun: 'значение', gen: 'значения', eq: 'равно', zero: 'нулевое', letter: 'V' },
}

/** Настоящая голова для сравнения масштабов (шаг «Делим на √2»): Qwen3-8B, d_k = 128. */
const REAL_DK = getModel('qwen3-8b').headDim

/* ============================================================================
 * Мелочи форматирования
 * ========================================================================== */

/** «Кошка» в кавычках; запятая как «,». */
const qw = (c: ToyCase, i: number) => `«${c.words[i]}»`

/** Слово с нужным падежом: «слово «она»», «слова «она»»; запятая отдельно. */
function nm(c: ToyCase, i: number, form: 'nom' | 'gen' | 'dat'): string {
  if (c.words[i] === ',') return form === 'nom' ? 'запятая' : 'запятой'
  return `${form === 'nom' ? 'слово' : form === 'gen' ? 'слова' : 'слову'} ${qw(c, i)}`
}

/** Подпись слова в вертикальной шапке столбца: голая запятая там читается как соринка. */
const colWord = (w: string) => (w === ',' ? 'запятая' : w)

/** «A», «B» и «C». */
function listWords(c: ToyCase, idx: readonly number[]): string {
  const ws = idx.map((i) => qw(c, i))
  return ws.length <= 1 ? (ws[0] ?? '') : `${ws.slice(0, -1).join(', ')} и ${ws[ws.length - 1]}`
}

/** Вектор по-русски: [0,73; 0,85]. */
const vec = (v: readonly number[], digits?: number) => `[${v.map((x) => fmt.num(x, digits)).join('; ')}]`

/** Число для TeX: десятичная запятая, минус ASCII (KaTeX сам наберёт знак). */
function tex(v: number, digits?: number): string {
  if (!Number.isFinite(v)) return v > 0 ? '\\infty' : '-\\infty'
  const s = fmt.num(Math.abs(v), digits).replace(/,/g, '{,}').replace(/ /g, '\\,')
  return v < 0 && s !== '0' ? `-${s}` : s
}
/** В произведении отрицательное число в скобках. */
const texF = (v: number, digits?: number) => (v < 0 ? `(${tex(v, digits)})` : tex(v, digits))

const pct = (v: number) => fmt.pct(v)
const pct0 = (v: number) => (v === 0 ? '0' : fmt.pct(v, 0))
const num2 = (v: number) => (v === 0 ? '0' : fmt.num(v, 2))

/** Слагаемые оценки строкой: «2·1 + 3·0». */
const prodText = (c: ToyCase, row: number, col: number) => c.trace.Q[row].map((q, i) => `${fmt.num(q)}·${fmt.num(c.trace.K[col][i])}`).join(' + ')

/* ============================================================================
 * Геометрия: размер ячеек под ширину сцены
 * ========================================================================== */

interface Geo {
  w: number
  h: number
  size: 'sm' | number
}
const GEO_WIDE: Geo = { w: 44, h: 28, size: 'sm' }
const GEO_MID: Geo = { w: 40, h: 24, size: 40 }
const GEO_NARROW: Geo = { w: 36, h: 24, size: 36 }
const GEOS: readonly Geo[] = [GEO_WIDE, GEO_MID, GEO_NARROW]
/** Уже этого сцена считается телефонной: ячейки 36 px и маленькие чипы. */
const PHONE_W = 420

const GAP = 2
/** Поле прокрутки MatrixView. */
const PAD = 3
const TITLE_H = 29
/** Верх первой строки ячеек от верха блока матрицы: подпись + поле + шапка (HEAD_H + 4) + зазор. */
const ROW_TOP = TITLE_H + PAD + HEAD_H + 4 + GAP
const rowsH = (n: number, g: Geo) => n * g.h + (n - 1) * GAP
const LABEL_FONT = '12px "Onest Variable", sans-serif'
const labelsW = (labels: readonly string[]) => Math.max(...labels.map((l) => measureText(l, LABEL_FONT))) + 8 + GAP
const cellsW = (cols: number, g: Geo) => cols * g.w + (cols - 1) * GAP + 2 * PAD

/* Схема Фалька для Q·Kᵀ: Q слева, Kᵀ сверху, S на пересечении. */
const KT_HEAD_H = 84
const Q_HEAD_H = 104
const QK_GAP_X = 10
const QK_GAP_Y = 4
const QK_SIDE_W = 132
const qkTopH = (g: Geo) => PAD + KT_HEAD_H + 4 + GAP + rowsH(2, g) + PAD
const qkHeight = (g: Geo) => qkTopH(g) + QK_GAP_Y + PAD + rowsH(N, g) + PAD
const qkWidth = (g: Geo, words: readonly string[]) => labelsW(words) + cellsW(2, g) + QK_GAP_X + cellsW(N, g) + QK_GAP_X + QK_SIDE_W

/** Высота сцены в широкой раскладке: самая высокая из раскладок шагов, чтобы текст под ней не прыгал. */
const STAGE_MIN_H = Math.max(ROW_TOP + rowsH(N, GEO_WIDE) + 6, qkHeight(GEO_WIDE))

/* ============================================================================
 * Разметка ряда: матрицы и знаки
 * ========================================================================== */

interface MatTerm {
  kind: 'mat'
  key: string
  title: ReactNode
  data: number[][]
  colLabels: readonly string[]
  /** words: строки = слова (можно не подписывать, если матрица идёт следом за другой такой же); features: строки = признаки. */
  labels: 'words' | 'features'
  rowNames: readonly string[]
  role?: 'q' | 'k' | 'v'
  heat?: 'seq' | 'diverging' | false
  domain?: [number, number]
  format: (v: number) => string
  /** Формат для узких ячеек (36 px): «100 %» и «−0,35» туда не помещаются. */
  formatNarrow?: (v: number) => string
  /** Подпись для узких ячеек (например, «Веса, %», когда в ячейках числа без знака процента). */
  titleNarrow?: ReactNode
  highlight?: MatrixHighlight
  dimOthers?: boolean
  onCellClick?: (c: MatrixCell) => void
  cellTooltip?: (c: MatrixCell) => ReactNode
  /** Замаскированные ячейки (causal mask после softmax): штриховка и maskedText вместо заливки. */
  isMasked?: (row: number, col: number) => boolean
  maskedText?: string
  ariaLabel: string
  center?: boolean
}
/** Ячейки уже 40 px: числа короче (без «%», один знак после запятой). */
const isNarrowCell = (g: Geo) => g.w < 40
interface OpTerm {
  kind: 'op'
  key: string
  glyph: ReactNode
  /** Подпись знака в столбике: «умножаем на», «получаем». */
  label: string
  width: number
}
interface NodeTerm {
  kind: 'node'
  key: string
  node: ReactNode
  width: number
}
/** Схема Q·Kᵀ = S целиком (шаги 4-7). */
interface QKTerm {
  kind: 'qk'
  key: string
}
type Term = MatTerm | OpTerm | NodeTerm | QKTerm

function termWidth(t: Term, withLabels: boolean, g: Geo, words: readonly string[]): number {
  if (t.kind === 'mat') {
    const cols = t.data[0]?.length ?? 0
    return (withLabels ? labelsW(t.rowNames) : 0) + cellsW(cols, g)
  }
  if (t.kind === 'qk') return qkWidth(g, words)
  return t.width
}

/** Подписи строк показываем у матриц признаков всегда, у матриц «по словам» только у первой в ряду. */
function labelFlags(terms: Term[]): boolean[] {
  let seenWords = false
  return terms.map((t) => {
    if (t.kind !== 'mat') return false
    if (t.labels === 'features') return true
    if (seenWords) return false
    seenWords = true
    return true
  })
}

function rowWidth(terms: Term[], g: Geo, words: readonly string[]): number {
  const flags = labelFlags(terms)
  return terms.reduce((s, t, i) => s + termWidth(t, flags[i], g, words), 0) + 8 * (terms.length - 1)
}

/** В режиме столбика строка, которая не помещается по ширине, распадается на отдельные строки. */
function fitLines(lines: Term[][], width: number, g: Geo, words: readonly string[]): Term[][] {
  return lines.flatMap((line) => (line.length > 1 && rowWidth(line, g, words) > width ? line.map((t) => [t]) : [line]))
}

/**
 * Раскладка шага: всё в один ряд, если ряд влезает (ячейки 44, 40 или 36 px), иначе столбиком.
 * В столбике ячейки максимальные, при которых каждая матрица влезает целиком; на телефоне 36 px.
 */
function chooseLayout(lines: Term[][], stageW: number, words: readonly string[]): { geo: Geo; rowMode: boolean } {
  const flat = lines.flat()
  for (const g of GEOS) if (rowWidth(flat, g, words) <= stageW) return { geo: g, rowMode: true }
  if (stageW < PHONE_W) return { geo: GEO_NARROW, rowMode: false }
  for (const g of GEOS) if (flat.every((t) => termWidth(t, true, g, words) <= stageW)) return { geo: g, rowMode: false }
  return { geo: GEO_NARROW, rowMode: false }
}

function MatBlock({ t, showLabels, geo }: { t: MatTerm; showLabels: boolean; geo: Geo }) {
  const narrow = isNarrowCell(geo)
  return (
    <MatrixView
      data={t.data}
      rowLabels={showLabels ? [...t.rowNames] : undefined}
      colLabels={t.colLabels.map((l, i) => (
        <VLabel key={i} title={l}>
          {l}
        </VLabel>
      ))}
      label={<span className="block truncate leading-[21px]">{narrow && t.titleNarrow ? t.titleNarrow : t.title}</span>}
      colorRole={t.role}
      heat={t.heat}
      domain={t.domain}
      format={narrow && t.formatNarrow ? t.formatNarrow : t.format}
      cellSize={geo.size}
      highlight={t.highlight}
      dimOthers={t.dimOthers}
      onCellClick={t.onCellClick}
      cellTooltip={t.cellTooltip}
      isMasked={t.isMasked}
      maskedText={t.maskedText}
      aria-label={t.ariaLabel}
    />
  )
}

/* ============================================================================
 * Фигура
 * ========================================================================== */

interface Sel {
  row: number
  col: number
  dim: number
}
type SetSel = (f: (s: Sel) => Sel) => void
/** Выбор для пробной раскладки: ширина матриц от выбора не зависит. */
const NAV_PROBE_SEL: Sel = { row: 0, col: 0, dim: 0 }

interface Ctx {
  step: number
  sel: Sel
  setSel: SetSel
  c: ToyCase
  geo: Geo
  /** Сцена не влезает в один ряд: схема Q·Kᵀ прокручивается вбок, подписи над ней. */
  compact: boolean
}

export function StepsFigure({ id }: { id: string }) {
  const p = usePlayer(STEP_LABELS.length, { intervalMs: 4800 })
  const step = p.step
  const [variant, setVariant] = useState<Variant>('zabor')
  const c = CASES[variant]
  const [sel, setSel] = useState<Sel>({ row: FOCUS.query, col: FOCUS.target, dim: 0 })
  const [stageRef, { width: stageW }] = useElementSize<HTMLDivElement>()
  const fontsReady = useFontsReady()
  const roving = useRoving(N, sel.row)

  const focusWord = (i: number) => setSel((s) => ({ ...s, row: i, col: s.col <= i ? s.col : argmaxWeight(i, c) }))

  const lines = useMemo(() => buildLines(step, sel, setSel, c), [step, sel, c])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const { geo, rowMode } = useMemo(() => chooseLayout(lines, stageW, c.words), [lines, stageW, c, fontsReady])
  const flat = useMemo(() => lines.flat(), [lines])
  const flatLabels = useMemo(() => labelFlags(flat), [flat])
  const phone = stageW > 0 && stageW < PHONE_W
  // мини-переключатель шагов у текста нужен, если хоть один шаг на этой ширине идёт столбиком (фигура длинная)
  const navNeeded = useMemo(
    () => stageW > 0 && STEP_LABELS.some((_, s) => !chooseLayout(buildLines(s, NAV_PROBE_SEL, () => {}, c), stageW, c.words).rowMode),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [stageW, c, fontsReady],
  )
  const ctx: Ctx = { step, sel, setSel, c, geo, compact: !rowMode }
  const showQ = isQK(step)

  // мини-переключатель у текста (узкий экран): после шага он остаётся на том же месте экрана.
  // Сцена дорисовывается и после шага (столбики, подсказка прокрутки), поэтому поправку держим ещё 800 мс.
  const rootRef = useRef<HTMLDivElement | null>(null)
  const navRef = useRef<HTMLDivElement | null>(null)
  const anchor = useRef<{ top: number; until: number } | null>(null)
  const keepNav = () => {
    const a = anchor.current
    const el = navRef.current
    if (!a || !el) return
    if (performance.now() > a.until) {
      anchor.current = null
      return
    }
    const dy = el.getBoundingClientRect().top - a.top
    if (Math.abs(dy) > 1) window.scrollBy(0, dy)
  }
  const keepNavRef = useRef(keepNav)
  keepNavRef.current = keepNav
  const stepBy = (d: number) => {
    const top = navRef.current?.getBoundingClientRect().top
    anchor.current = top === undefined ? null : { top, until: performance.now() + 800 }
    p.setPlaying(false)
    p.setStep((s) => s + d)
  }
  useLayoutEffect(() => {
    keepNavRef.current()
  }, [step])
  useEffect(() => {
    const el = rootRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => keepNavRef.current())
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const D = FOCUS.distractor
  return (
    <Figure
      id={id}
      title="Attention по шагам: на кого смотрит «она»"
      width="full"
      controlsPosition="top"
      controls={<Player {...p.bind} labels={STEP_LABELS} aria-label="Шаги attention" />}
      actions={<Badge>упрощённо: по словам, а не по токенам</Badge>}
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> на шаге «Оценки» нажмите на ячейку «она» → «забор» и сравните слагаемые с ячейкой «она» →
          «Кошка». Первые слагаемые одинаковые ({fmt.num(TRACE.Q[FOCUS.query][0])}·{fmt.num(TRACE.K[D][0])}), а всю разницу даёт второе: запрос «ищу женский
          род» ({fmt.num(TRACE.Q[FOCUS.query][1])}) умножается на {fmt.num(TRACE.K[FOCUS.target][1])} у «Кошки» и на {fmt.num(TRACE.K[D][1])} у «забора». Потом
          переключите «забор» на «ограду»: вес «она» разделится между «Кошкой» и «оградой» поровну. Нажатие на слово в предложении выбирает строку, которую
          разбираем.
        </>
      }
    >
      <div ref={rootRef} className="grid min-w-0 gap-4 [overflow-anchor:none]">
        {/* предложение: выбор слова; рядом опыт «забор» → «ограду» */}
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
          <div className="flex flex-wrap items-center gap-1.5" role="toolbar" aria-label="Выберите слово: чью строку разбираем" ref={roving.container}>
            {c.words.map((w, i) => {
              const isRow = i === sel.row
              const isCol = showQ && i === sel.col && !isRow
              const r = roving.item(i)
              return (
                <button
                  key={i}
                  ref={r.ref}
                  tabIndex={r.tabIndex}
                  onFocus={r.onFocus}
                  type="button"
                  aria-pressed={isRow}
                  onClick={() => focusWord(i)}
                  className="rounded-chip focus-visible:outline-offset-1"
                  aria-label={w === ',' ? 'запятая' : undefined}
                >
                  <WordChip focus={isRow} marker={showQ ? (isRow ? 'q' : isCol ? 'k' : undefined) : undefined}>
                    {w}
                  </WordChip>
                </button>
              )
            })}
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-ink-2">
            <span>Четвёртое слово:</span>
            <Segmented<Variant>
              size="sm"
              aria-label="Четвёртое слово: забор (мужской род) или ограду (женский род)"
              value={variant}
              onChange={setVariant}
              options={[
                { value: 'zabor', label: 'забор (м. р.)' },
                { value: 'ograda', label: 'ограду (ж. р.)' },
              ]}
            />
          </div>
        </div>

        <FormulaStrip step={step} onStep={(s) => p.setStep(s)} />

        {/* сцена */}
        <div ref={stageRef} className="min-w-0" style={{ minHeight: rowMode && step !== ARCS_STEP ? STAGE_MIN_H : undefined }}>
          {stageW > 0 &&
            (step === ARCS_STEP ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
                <ArcsStage row={sel.row} onFocus={focusWord} c={c} tall={rowMode} phone={phone} />
              </motion.div>
            ) : rowMode ? (
              <div className="relative flex items-start gap-2">
                <AnimatePresence mode="popLayout" initial={false}>
                  {flat.map((t, i) => (
                    <TermBox key={t.key} t={t} showLabels={flatLabels[i]} row ctx={ctx} />
                  ))}
                </AnimatePresence>
              </div>
            ) : (
              <div className="grid min-w-0 gap-2">
                {fitLines(lines, stageW, geo, c.words).map((line) => {
                  const flags = labelFlags(line)
                  return (
                    <div key={line.map((t) => t.key).join('+')} className="flex min-w-0 items-start gap-2">
                      {line.map((t, i) => (
                        <TermBox key={t.key} t={t} showLabels={flags[i]} row={line.length > 1} ctx={ctx} />
                      ))}
                    </div>
                  )
                })}
              </div>
            ))}
        </div>

        {navNeeded && <StepNav p={p} onStep={stepBy} navRef={navRef} />}

        {/* что происходит и как посчитано */}
        <div className="grid min-w-0 gap-3 @[720px]:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <div className="grid content-start gap-2 text-[15px] leading-relaxed text-ink" aria-live="polite">
            <p>
              <span className="font-semibold">Что происходит. </span>
              {WHAT[step]}
            </p>
            <p className="text-ink-2">
              <span className="font-semibold text-ink">{step === 0 ? 'На что смотреть. ' : 'Что изменилось. '}</span>
              {changed(step, sel.row, c)}
            </p>
          </div>
          <Detail step={step} sel={sel} c={c} />
        </div>
      </div>
    </Figure>
  )
}

/** Шаг назад и вперёд прямо над текстом шага: на телефоне плеер вверху фигуры уезжает за экран. */
function StepNav({ p, onStep, navRef }: { p: PlayerState; onStep: (d: number) => void; navRef: Ref<HTMLDivElement> }) {
  return (
    <div ref={navRef} className="flex items-center justify-between gap-2 rounded-panel bg-surface-2 px-1.5 py-1">
      <Button variant="ghost" size="sm" iconOnly icon={CaretLeftIcon} aria-label="Предыдущий шаг" disabled={p.atStart} onClick={() => onStep(-1)} />
      <span className="min-w-0 text-center text-[13px] leading-snug text-ink-2">
        Шаг {p.step + 1} из {p.steps}: <span className="font-medium text-ink">{STEP_LABELS[p.step]}</span>
      </span>
      <Button variant="ghost" size="sm" iconOnly icon={CaretRightIcon} aria-label="Следующий шаг" disabled={p.atEnd} onClick={() => onStep(1)} />
    </div>
  )
}

function TermBox({ t, showLabels, row, ctx, ref }: { t: Term; showLabels: boolean; row: boolean; ctx: Ctx; ref?: Ref<HTMLDivElement> }) {
  let inner: ReactNode
  if (t.kind === 'mat') inner = <MatBlock t={t} showLabels={showLabels} geo={ctx.geo} />
  else if (t.kind === 'qk') inner = <QKGrid ctx={ctx} />
  else if (t.kind === 'op')
    inner = row ? (
      <div className="flex items-center" style={{ marginTop: ROW_TOP, height: rowsH(N, ctx.geo) }}>
        <OpGlyph>{t.glyph}</OpGlyph>
      </div>
    ) : (
      // в столбике знак подписан словами: точка между матрицами в столбик теряется
      <OpGlyph className="justify-start gap-2 py-1 pl-2">
        {t.glyph === '·' ? '×' : t.glyph}
        <span className="text-[13px] text-ink-3">{t.label}</span>
      </OpGlyph>
    )
  else inner = t.node
  return (
    <motion.div
      ref={ref}
      layout="position"
      className={cn('min-w-0', row && 'shrink-0', row && t.kind === 'mat' && t.center && 'self-center')}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, transition: { duration: 0.12 } }}
      transition={{ duration: 0.35, ease: [0.2, 0, 0, 1] }}
    >
      {inner}
    </motion.div>
  )
}

/* ============================================================================
 * Что на сцене на каждом шаге
 * ========================================================================== */

function buildLines(step: number, sel: Sel, setSel: SetSel, c: ToyCase): Term[][] {
  const pickRow = (cell: MatrixCell) => setSel((s) => ({ ...s, row: cell.row, col: s.col <= cell.row ? s.col : argmaxWeight(cell.row, c) }))
  const op = (key: string, glyph: ReactNode, label: string, width = 28): OpTerm => ({ kind: 'op', key, glyph, label, width })

  const X: MatTerm = {
    kind: 'mat',
    key: 'X',
    title: 'X: признаки слов',
    data: c.X,
    colLabels: FEATURES,
    labels: 'words',
    rowNames: c.words,
    heat: 'diverging',
    domain: [-4 / 3, 4 / 3],
    format: (v) => fmt.int(v),
    highlight: { rows: [sel.row] },
    dimOthers: step > 0,
    onCellClick: pickRow,
    cellTooltip: (cell) => (
      <span>
        {qw(c, cell.row)}, {FEATURES[cell.col]}: <b className="tabular-nums">{cell.value ? '1, да' : '0, нет'}</b>
      </span>
    ),
    ariaLabel: 'Матрица X: строки слова, столбцы признаки',
  }

  if (step === 0) {
    return [[X], [{ kind: 'node', key: 'legend', node: <FeatureLegend row={sel.row} c={c} />, width: 300 }]]
  }

  const proj = PROJ_OF_STEP[step]
  if (proj) {
    const names = PROJ_NAME[proj]
    const active = activeFeatures(sel.row, c)
    const W: MatTerm = {
      kind: 'mat',
      key: 'W',
      title: (
        <>
          W<Sub>{names.letter}</Sub>
        </>
      ),
      data: PROJ_W[proj],
      colLabels: PROJ_DIMS[proj],
      labels: 'features',
      rowNames: FEATURES,
      role: proj,
      heat: 'seq',
      format: (v) => fmt.num(v),
      // столбец, на который умножаются признаки слова, и в нём ячейки включённых признаков (они и дают сумму)
      highlight: { cols: [sel.dim], cells: active.map((r) => [r, sel.dim] as [number, number]) },
      dimOthers: true,
      onCellClick: (cell) => setSel((s) => ({ ...s, dim: cell.col })),
      cellTooltip: (cell) => (
        <span>
          {FEATURES[cell.row]} → {PROJ_DIMS[proj][cell.col]}: <b className="tabular-nums">{fmt.num(cell.value)}</b>
        </span>
      ),
      ariaLabel: `Матрица весов W_${names.letter}: строки признаки, столбцы измерения ${names.gen}`,
      center: true,
    }
    const R: MatTerm = {
      kind: 'mat',
      key: 'R',
      title: (
        <>
          {names.letter} = X·W<Sub>{names.letter}</Sub>
        </>
      ),
      data: projOut(proj, c),
      colLabels: PROJ_DIMS[proj],
      labels: 'words',
      rowNames: c.words,
      role: proj,
      heat: 'seq',
      format: (v) => fmt.num(v),
      highlight: { rows: [sel.row], cells: [[sel.row, sel.dim]] },
      dimOthers: false,
      onCellClick: (cell) => setSel((s) => ({ ...s, row: cell.row, dim: cell.col, col: s.col <= cell.row ? s.col : argmaxWeight(cell.row, c) })),
      cellTooltip: (cell) => (
        <span>
          {qw(c, cell.row)}, {PROJ_DIMS[proj][cell.col]}: <b className="tabular-nums">{fmt.num(cell.value)}</b>
        </span>
      ),
      ariaLabel: `Матрица ${names.letter}: строки слова`,
    }
    return [[X], [op('dot', '·', 'умножаем на', 24)], [W], [op('eq', '=', 'получаем')], [R]]
  }

  if (isQK(step)) return [[{ kind: 'qk', key: 'qk' }]]

  // шаг 8 (и 9: по нему считается ширина): веса · V = выход
  const S = scoresTerm(7, sel, setSel, c)
  const V: MatTerm = {
    kind: 'mat',
    key: 'V8',
    title: 'V',
    data: c.trace.V,
    colLabels: TOY.valueDimNames,
    labels: 'words',
    rowNames: c.words,
    role: 'v',
    heat: 'seq',
    format: (v) => fmt.num(v),
    highlight: { cols: [sel.dim] },
    dimOthers: false,
    onCellClick: (cell) => setSel((s) => ({ ...s, dim: cell.col })),
    cellTooltip: (cell) => (
      <span>
        Значение {qw(c, cell.row)}: <b className="tabular-nums">{vec(c.trace.V[cell.row])}</b>
      </span>
    ),
    ariaLabel: 'Матрица значений V',
  }
  const O: MatTerm = {
    kind: 'mat',
    key: 'O',
    title: 'Выход',
    data: c.trace.out,
    colLabels: TOY.valueDimNames,
    labels: 'words',
    rowNames: c.words,
    role: 'v',
    heat: 'seq',
    format: num2,
    highlight: { rows: [sel.row], cells: [[sel.row, sel.dim]] },
    dimOthers: false,
    onCellClick: (cell) => setSel((s) => ({ ...s, row: cell.row, dim: cell.col, col: s.col <= cell.row ? s.col : argmaxWeight(cell.row, c) })),
    cellTooltip: (cell) => (
      <span>
        Выход {qw(c, cell.row)}, {TOY.valueDimNames[cell.col]}: <b className="tabular-nums">{fmt.num(cell.value, 2)}</b>
      </span>
    ),
    ariaLabel: 'Выход attention: смесь значений V',
  }
  return [[S], [op('dot', '·', 'умножаем на', 24)], [V, op('eq', '=', 'получаем'), O]]
}

/** Матрица оценок на шагах 4-7 (S, S ÷ √2, + M, веса) и веса на шаге 8. */
function scoresTerm(step: number, sel: Sel, setSel: SetSel, c: ToyCase): MatTerm {
  const T = c.trace
  const rawMax = Math.max(...T.rawScores.flat().map(Math.abs))
  const base = {
    kind: 'mat' as const,
    key: 'S',
    colLabels: c.words.map(colWord),
    labels: 'words' as const,
    rowNames: c.words,
    highlight: { rows: [sel.row], cells: [[sel.row, sel.col]] as Array<[number, number]> },
    dimOthers: false,
    onCellClick: (cell: MatrixCell) => setSel((s) => ({ ...s, row: cell.row, col: cell.col })),
  }
  if (step === 4) {
    return {
      ...base,
      title: 'S = Q·Kᵀ: оценки',
      data: T.rawScores,
      heat: 'diverging',
      domain: [-rawMax, rawMax],
      format: (v) => fmt.num(v),
      cellTooltip: (cell) => (
        <span>
          {qw(c, cell.row)} · {qw(c, cell.col)} = <b className="tabular-nums">{fmt.num(cell.value)}</b>
        </span>
      ),
      ariaLabel: 'Оценки: строки слова-запросы, столбцы слова-ключи, в ячейке запрос строки, умноженный на ключ столбца',
    }
  }
  if (step === 5 || step === 6) {
    return {
      ...base,
      title: step === 5 ? 'S ÷ √2' : 'S ÷ √2 + M',
      data: step === 5 ? T.scaled : T.masked,
      heat: 'diverging',
      domain: [-rawMax / SQRT_DK, rawMax / SQRT_DK],
      format: num2,
      formatNarrow: (v) => (v === 0 ? '0' : fmt.num(v, 1)),
      cellTooltip: (cell) =>
        cell.value === -Infinity ? (
          <span>
            {qw(c, cell.col)} правее {qw(c, cell.row)}: <b>−∞</b>, смотреть нельзя
          </span>
        ) : (
          <span>
            {fmt.num(T.rawScores[cell.row][cell.col])} ÷ {fmt.num(SQRT_DK, 2)} = <b className="tabular-nums">{fmt.num(cell.value, 2)}</b>
          </span>
        ),
      ariaLabel: step === 5 ? 'Оценки после деления на корень из d_k' : 'Оценки после causal mask: над диагональю минус бесконечность',
    }
  }
  return {
    ...base,
    title: (
      <>
        Веса = softmax(…)<span className="text-ink-3"> по строкам</span>
      </>
    ),
    titleNarrow: (
      <>
        Веса, % = softmax(…)<span className="text-ink-3"> по строкам</span>
      </>
    ),
    data: T.weights,
    heat: 'seq',
    domain: [0, 1],
    // после softmax −∞ стали нулями: штриховка оставляет треугольник маски видимым (светлый ноль шкалы похож на веса 2-4 %)
    isMasked: (r, col) => col > r,
    maskedText: '0',
    format: pct0,
    formatNarrow: (v) => (v === 0 ? '0' : fmt.int(Math.round(v * 100))),
    cellTooltip: (cell) => (
      <span>
        {qw(c, cell.row)} → {qw(c, cell.col)}: <b className="tabular-nums">{cell.col > cell.row ? '0, замаскировано' : pct(cell.value)}</b>
      </span>
    ),
    ariaLabel: 'Веса внимания: softmax по строкам',
  }
}

/** Подпись над матрицей в схеме Q·Kᵀ: метка цвета сущности и текст. */
function MatTitle({ role, children }: { role?: 'q' | 'k'; children: ReactNode }) {
  return (
    <div className="flex items-center gap-2 text-sm leading-[21px] font-medium text-ink-2">
      {role && <Swatch role={role} />}
      <span className="min-w-0">{children}</span>
    </div>
  )
}

/** Что лежит на пересечении на шагах 4-7. */
const S_NAME: Record<number, { title: string; note: string; inline: string }> = {
  4: { title: 'S = Q·Kᵀ', note: 'оценки: запрос строки на ключ столбца', inline: 'оценки S = Q·Kᵀ' },
  5: { title: 'S ÷ √2', note: 'те же оценки в другом масштабе', inline: 'оценки, делённые на √2' },
  6: { title: 'S ÷ √2 + M', note: 'над диагональю −∞: будущее закрыто', inline: 'оценки после маски' },
  7: { title: 'Веса', note: 'softmax по строкам, сумма строки 100 %', inline: 'веса после softmax' },
}

/**
 * Схема Фалька для Q·Kᵀ: Q слева, Kᵀ сверху (ключи столбцами), матрица S на пересечении.
 * Строка Q и столбец Kᵀ сходятся в одной ячейке S: так видно, что каждая оценка это строка на столбец.
 * Столбцы Kᵀ стоят ровно над столбцами S (одинаковые ячейки и зазоры), строки Q ровно напротив строк S.
 */
function QKGrid({ ctx }: { ctx: Ctx }) {
  const { step, sel, setSel, c, geo, compact } = ctx
  const S = scoresTerm(step, sel, setSel, c)
  const KT = useMemo(() => transpose(c.trace.K), [c])
  const kRows = TOY.keyDimNames
  const base = S_NAME[step] ?? S_NAME[4]
  // в узких ячейках веса без знака процента: говорим об этом в подписи
  const name = step === 7 && isNarrowCell(geo) ? { ...base, note: 'softmax по строкам, числа в процентах', inline: 'веса после softmax, в процентах' } : base
  const pickRow = (row: number) => setSel((s) => ({ ...s, row, col: s.col <= row ? s.col : argmaxWeight(row, c) }))

  const grid = (
    <div className="inline-grid items-start align-top" style={{ gridTemplateColumns: `auto auto ${QK_SIDE_W}px`, columnGap: QK_GAP_X, rowGap: QK_GAP_Y }}>
      {/* угол: подпись Q и шапка его столбцов (стоят прямо над столбцами Q) */}
      <div className="flex flex-col justify-between self-stretch" style={{ gridRow: 1, gridColumn: 1 }}>
        <MatTitle role="q">Q: запросы</MatTitle>
        <div className="flex justify-end" style={{ gap: GAP, paddingRight: PAD }}>
          {TOY.queryDimNames.map((d) => (
            <div key={d} className="flex justify-center pb-1 text-[12px] leading-tight text-ink-2" style={{ width: geo.w }}>
              <VLabel height={Q_HEAD_H} title={d}>
                {d}
              </VLabel>
            </div>
          ))}
        </div>
      </div>

      {/* Kᵀ: строки измерения ключа, столбцы слова */}
      <div style={{ gridRow: 1, gridColumn: 2 }}>
        <MatrixView
          data={KT}
          colLabels={c.words.map((w, i) => (
            <VLabel key={i} height={KT_HEAD_H} title={w}>
              {colWord(w)}
            </VLabel>
          ))}
          colorRole="k"
          heat="seq"
          format={(v) => fmt.num(v)}
          cellSize={geo.size}
          highlight={{ cols: [sel.col] }}
          onCellClick={(cell) => setSel((s) => ({ ...s, col: cell.col }))}
          cellTooltip={(cell) => (
            <span>
              Ключ {qw(c, cell.col)}, «{kRows[cell.row]}»: <b className="tabular-nums">{fmt.num(cell.value)}</b>
            </span>
          )}
          aria-label="Матрица Kᵀ: строки измерения ключа, столбцы слова"
        />
      </div>

      {/* справа от Kᵀ: название и подписи её строк */}
      <div className="relative self-stretch" style={{ gridRow: 1, gridColumn: 3 }}>
        {!compact && (
          <div className="absolute inset-x-0 top-0 grid gap-0.5">
            <MatTitle role="k">Kᵀ</MatTitle>
            <span className="text-[13px] leading-snug text-ink-3">ключи столбцами</span>
          </div>
        )}
        <div className="grid text-[12px] leading-tight text-ink-2" style={{ paddingTop: PAD + KT_HEAD_H + 4 + GAP, rowGap: GAP }}>
          {kRows.map((d) => (
            <div key={d} className="flex items-center whitespace-nowrap" style={{ height: geo.h }}>
              {d}
            </div>
          ))}
        </div>
      </div>

      {/* Q: строки слова, столбцы стоят под своей шапкой в углу */}
      <div style={{ gridRow: 2, gridColumn: 1 }}>
        <MatrixView
          data={c.trace.Q}
          rowLabels={[...c.words]}
          colorRole="q"
          heat="seq"
          format={(v) => fmt.num(v)}
          cellSize={geo.size}
          highlight={{ rows: [sel.row] }}
          onCellClick={(cell) => pickRow(cell.row)}
          cellTooltip={(cell) => (
            <span>
              Запрос {qw(c, cell.row)}: <b className="tabular-nums">{vec(c.trace.Q[cell.row])}</b>
            </span>
          )}
          aria-label="Матрица запросов Q: строки слова, столбцы «ищу предмет» и «ищу женский род»"
        />
      </div>

      {/* S: строки как у Q, столбцы как у Kᵀ */}
      <div style={{ gridRow: 2, gridColumn: 2 }}>
        <MatrixView
          data={S.data}
          heat={S.heat}
          domain={S.domain}
          format={isNarrowCell(geo) && S.formatNarrow ? S.formatNarrow : S.format}
          cellSize={geo.size}
          highlight={S.highlight}
          dimOthers={false}
          onCellClick={S.onCellClick}
          cellTooltip={S.cellTooltip}
          isMasked={S.isMasked}
          maskedText={S.maskedText}
          aria-label={S.ariaLabel}
        />
      </div>

      {!compact && (
        <div className="grid content-start gap-0.5 pt-[3px]" style={{ gridRow: 2, gridColumn: 3 }}>
          <MatTitle>{name.title}</MatTitle>
          <span className="text-[13px] leading-snug text-ink-3">{name.note}</span>
        </div>
      )}
    </div>
  )

  if (!compact) return grid
  return (
    <div className="grid min-w-0 gap-2">
      <p className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[13px] leading-snug text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <Swatch role="q" /> Q слева,
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Swatch role="k" /> Kᵀ сверху (ключи столбцами),
        </span>
        <span>на пересечении {name.inline}.</span>
      </p>
      <ScrollX label="Схема Q·Kᵀ" hint="схема шире экрана: прокрутите её вбок">
        {grid}
      </ScrollX>
    </div>
  )
}

/* ============================================================================
 * Легенда признаков (шаг 0) и шаг с дугами
 * ========================================================================== */

function FeatureLegend({ row, c }: { row: number; c: ToyCase }) {
  const active = new Set(activeFeatures(row, c))
  return (
    <div className="grid max-w-[340px] content-start gap-2 rounded-panel bg-surface-2 px-4 py-3 text-sm leading-snug @[700px]:mt-[29px]">
      <div className="font-semibold text-ink">У кого какие признаки</div>
      <ul className="grid gap-1.5">
        {FEATURES.map((f, fi) => {
          const who = c.words.filter((_, i) => c.X[i][fi] !== 0)
          return (
            <li key={f} className={cn('grid gap-0.5', active.has(fi) ? 'text-ink' : 'text-ink-2')}>
              <span className={cn(active.has(fi) && 'font-semibold')}>
                {f}
                {active.has(fi) && <span className="font-normal text-ink-2">: есть у {qw(c, row)}</span>}
              </span>
              <span className="text-[13px] text-ink-3">{who.map((w) => `«${w}»`).join(', ')}</span>
            </li>
          )
        })}
      </ul>
      <p className="text-[13px] text-ink-3">«Не», «потому», «что» и запятая без признаков: в игрушке это пустые слова.</p>
    </div>
  )
}

function ArcsStage({ row, onFocus, c, tall, phone }: { row: number; onFocus: (i: number) => void; c: ToyCase; tall: boolean; phone: boolean }) {
  const ranked = rankedRow(row, c)
  return (
    <div className="grid min-w-0 items-start gap-6 @[880px]:grid-cols-[minmax(0,1.45fr)_minmax(260px,1fr)]">
      <div className="grid min-w-0 gap-2">
        <div className="text-sm font-medium text-ink-2">Внимание {nm(c, row, 'gen')}: толщина дуги равна весу</div>
        <ArcDiagram
          count={N}
          focus={row}
          scrollTo={row}
          groupLabel="Выберите слово: чьё внимание показать"
          weights={c.trace.weights[row]}
          onFocus={onFocus}
          itemLabel={(i) => (c.words[i] === ',' ? 'запятая' : c.words[i])}
          renderItem={(i, focus) => (
            <WordChip focus={focus} size={phone ? 'sm' : 'md'}>
              {c.words[i]}
            </WordChip>
          )}
          height={tall ? 240 : phone ? 160 : 190}
          gap={phone ? 4 : 6}
          layoutKey={`${c.variant}|${phone ? 's' : 'm'}`}
          center
        />
      </div>
      <div className="grid min-w-0 gap-2">
        <div className="text-sm font-medium text-ink-2">Строка {qw(c, row)} по убыванию веса</div>
        <BarChart
          data={ranked.map((r) => ({ id: String(r.index), label: r.word === ',' ? '«,»' : r.word, value: r.weight }))}
          max={1}
          valueFormat={pct}
          color="prefill"
          aria-label={`Веса внимания слова ${c.words[row]}`}
        />
      </div>
    </div>
  )
}

/* ============================================================================
 * Тексты шагов
 * ========================================================================== */

const WHAT: ReactNode[] = [
  <>Модель видит не слова, а векторы. Здесь у каждого слова 4 признака с понятными именами: 1 значит «да», 0 значит «нет».</>,
  <>
    Матрица W<Sub>Q</Sub> превращает признаки слова в запрос Q из двух чисел: насколько слово ищет предмет и насколько ищет женский род.
  </>,
  <>
    Та же операция с матрицей W<Sub>K</Sub> даёт ключ K: по каким признакам слово можно найти. Ключ отвечает запросу по тем же осям: «я предмет» на «ищу
    предмет», «я женского рода» на «ищу женский род».
  </>,
  <>
    Матрица W<Sub>V</Sub> даёт значение V: что слово передаст тем, кто на него посмотрит.
  </>,
  <>
    Каждый запрос скалярно умножается на каждый ключ: числа попарно перемножаются и складываются. Получается таблица {N} × {N}: строка показывает, кто ищет,
    столбец показывает, на кого он смотрит. Значок ᵀ у K означает транспонирование: ключи записаны столбцами, и одно умножение матриц даёт сразу все пары.
    Поэтому Kᵀ стоит над таблицей, а Q слева: каждая оценка лежит на пересечении строки запроса и столбца ключа.
  </>,
  <>
    Все оценки делятся на √d<Sub>k</Sub> = √{TOY.dK} ≈ {fmt.num(SQRT_DK, 2)}. Порядок оценок от этого не меняется, но softmax чувствителен к масштабу: чем
    больше разброс оценок, тем резче он отдаёт весь вес одному слову. У настоящей головы векторы из {REAL_DK} чисел, и без деления оценки разлетались бы
    примерно на ±{fmt.num(Math.sqrt(REAL_DK), 0)}: softmax почти всегда выбирал бы одно слово. Деление на √d<Sub>k</Sub> возвращает разброс примерно к единице.
  </>,
  <>Над диагональю ставим −∞: слово видит только себя и слова слева. При генерации правых слов ещё нет, и модель с самого обучения работает так же.</>,
  <>Softmax возводит e в степень каждой оценки и делит на сумму по строке. Получаются веса от 0 до 1, которые в сумме дают 1, а из −∞ получается ровно 0.</>,
  <>
    Выход слова собирается как смесь значений V всех видимых слов, взятых с весами из softmax. Этот выход не заменяет вектор слова, а прибавляется к нему: в
    настоящей модели он ещё проходит матрицу W<Sub>O</Sub> и добавляется в <GlossaryTerm t="residual stream">residual stream</GlossaryTerm> (глава{' '}
    <ChapterLink to="transformer" />
    ).
  </>,
  <>Те же веса одной строки в виде дуг над предложением: чем толще дуга, тем больше вес. Столбики показывают их же по убыванию.</>,
]

/** Строка весов из одних равных долей: у служебных слов запрос нулевой, все оценки 0. */
function uniformText(c: ToyCase, row: number): { why: string; share: string } {
  const zeroQuery = c.trace.Q[row].every((x) => x === 0)
  return { why: zeroQuery ? 'запрос нулевой' : 'все оценки равны', share: pct(1 / (row + 1)) }
}

function changed(step: number, row: number, c: ToyCase): ReactNode {
  const w = qw(c, row)
  const T = c.trace
  const D = FOCUS.distractor
  const ograda = c.variant === 'ograda'
  if (step === 0) {
    const f = activeFeatures(row, c)
    return f.length ? (
      <>
        У {nm(c, row, 'gen')} {f.length === 1 ? 'включён признак' : 'включены признаки'} {f.map((i) => `«${FEATURES[i]}»`).join(' и ')}: вектор {vec(c.X[row])}.
      </>
    ) : (
      <>У {nm(c, row, 'gen')} все признаки нулевые: служебные слова и запятая в игрушке ничего не несут.</>
    )
  }
  const proj = PROJ_OF_STEP[step]
  if (proj) {
    const n = PROJ_NAME[proj]
    const f = activeFeatures(row, c)
    const out = projOut(proj, c)[row]
    const W = PROJ_W[proj]
    const lead =
      f.length === 0 ? (
        <>
          У {nm(c, row, 'gen')} признаков нет, поэтому и {n.noun} {n.zero}: {vec(out)}.
        </>
      ) : f.length === 1 ? (
        <>
          У {nm(c, row, 'gen')} один признак, «{FEATURES[f[0]]}», поэтому {n.noun} {n.eq} строке W<Sub>{n.letter}</Sub> этого признака: {vec(out)}.
        </>
      ) : (
        <>
          У {nm(c, row, 'gen')} признаки {f.map((i) => `«${FEATURES[i]}»`).join(' и ')}, поэтому {n.noun} {n.eq} сумме их строк W<Sub>{n.letter}</Sub>:{' '}
          {f.map((i) => vec(W[i])).join(' + ')} = {vec(out)}.
        </>
      )
    if (proj === 'k')
      return ograda ? (
        <>
          {lead} Сравните ключи: у «Кошки» {vec(T.K[FOCUS.target])}, и у «{c.other.gen}» теперь тоже {vec(T.K[D])}: предмет и женский род.
        </>
      ) : (
        <>
          {lead} Сравните ключи: у «Кошки» {vec(T.K[FOCUS.target])}, то есть предмет и женский род, у «{c.other.gen}» {vec(T.K[D])}, только предмет.
        </>
      )
    if (proj === 'v') return <>{lead} Здесь V просто копирует два признака, «существительное» и «женский род».</>
    return lead
  }
  if (step === 4) {
    const raw = T.rawScores[row]
    if (raw.every((x) => x === 0)) return <>Запрос {nm(c, row, 'gen')} нулевой, поэтому вся его строка оценок из нулей.</>
    if (row === FOCUS.query) {
      const t = scoreTerms(row, FOCUS.target, c)
      const d = scoreTerms(row, D, c)
      return ograda ? (
        <>
          В строке {w} у «Кошки» и у «{c.other.gen}» оценки равны: {prodText(c, row, FOCUS.target)} = {fmt.num(t.sum)} в обоих случаях. Ключи у них одинаковые,
          и запросу {w} не за что предпочесть одно слово другому.
        </>
      ) : (
        <>
          В строке {w} самая большая оценка у «Кошки»: {prodText(c, row, FOCUS.target)} = {fmt.num(t.sum)}. У «{c.other.gen}» {prodText(c, row, D)} ={' '}
          {fmt.num(d.sum)}: ключу «{c.other.gen}» нечем ответить на «ищу женский род».
        </>
      )
    }
    const max = Math.max(...raw)
    const ties = raw.flatMap((x, j) => (max - x < TIE ? [j] : []))
    const right = ties.filter((j) => j > row)
    const head =
      ties.length === 1 ? (
        <>
          В строке {w} самая большая оценка у {nm(c, ties[0], 'gen')}: {fmt.num(max)}.
        </>
      ) : (
        <>
          В строке {w} самая большая оценка, {fmt.num(max)}, у слов {listWords(c, ties)}.
        </>
      )
    const tail =
      right.length === 0
        ? null
        : right.length === ties.length
          ? ties.length === 1
            ? ' Но это слово правее, и его скоро закроет маска.'
            : ` Но ${ties.length === 2 ? 'оба' : 'все'} правее, и их скоро закроет маска.`
          : ` Но ${listWords(c, right)} правее, ${right.length === 1 ? 'его' : 'их'} скоро закроет маска.`
    return (
      <>
        {head}
        {tail}
      </>
    )
  }
  if (step === 5) {
    const raw = T.rawScores[row]
    if (raw.every((x) => x === 0)) return <>Нули остаются нулями: строке {w} всё равно, на что делить.</>
    let best = 0
    raw.forEach((x, j) => {
      if (j <= row && x > raw[best]) best = j
    })
    return (
      <>
        Оценка {w} → {qw(c, best)} стала {fmt.num(T.scaled[row][best], 2)} вместо {fmt.num(raw[best])}. Как это выглядит при d<Sub>k</Sub> = {REAL_DK},
        показывает опыт в блоке «Глубже» после фигур.
      </>
    )
  }
  if (step === 6) {
    const k = N - 1 - row
    if (k === 0) return <>В последней строке закрывать нечего: {w} видит всё предложение.</>
    if (row === 0) return <>Строка «Кошка» видит только себя: закрыты все {k} слов правее.</>
    if (k === 1)
      return (
        <>
          В строке {w} закрыто одно слово, {qw(c, row + 1)}.
        </>
      )
    return (
      <>
        В строке {w} закрыто {fmt.count(k, ['слово', 'слова', 'слов'])}: всё, что правее.
      </>
    )
  }
  if (step === 7) {
    if (row === 0) return <>«Кошка» видит только себя, поэтому весь вес строки, 100 %, достаётся ей самой.</>
    if (isUniformRow(row, c)) {
      const u = uniformText(c, row)
      return u.why === 'запрос нулевой' ? (
        <>
          Запрос {nm(c, row, 'gen')} нулевой, все оценки равны, и вес делится поровну: по {u.share} на каждое из {row + 1} слов.
        </>
      ) : (
        <>
          Все оценки строки {w} равны, и вес делится поровну: по {u.share} на каждое из {row + 1} слов.
        </>
      )
    }
    const { head, more } = rankedHead(row, c)
    return (
      <>
        Строка {w}: {head.map((r) => `${qw(c, r.index)} ${pct(r.weight)}`).join(', ')}
        {more ? ', остальным меньше' : ''}. Сумма строки 100 %.
      </>
    )
  }
  if (step === 8) {
    const out = vec(T.out[row], 2)
    if (row === 0) return <>Выход слова «Кошка» = {out}: она видит только себя, и выход совпадает с её значением V.</>
    if (isUniformRow(row, c))
      return (
        <>
          Выход {nm(c, row, 'gen')} = {out}: веса у всех видимых слов одинаковые, поэтому выход равен среднему их значений V.
        </>
      )
    const top = topIndices(row, c)
    const lead =
      top.length === 1 ? (
        <>
          Выход {nm(c, row, 'gen')} = {out}. Больше всего в него вложило {nm(c, top[0], 'nom')} ({pct(T.weights[row][top[0]])} веса).
        </>
      ) : (
        <>
          Выход {nm(c, row, 'gen')} = {out}. Больше всего в него вложили слова {listWords(c, top)}, по {pct(T.weights[row][top[0]])} веса.
        </>
      )
    if (row !== FOCUS.query) return lead
    return ograda ? (
      <>
        {lead} Признак «существительное» пришёл поровну от «Кошки» и от «{c.other.gen}». Эта поправка прибавится к вектору «она», но не скажет, на кого из двух
        указывает местоимение: одной голове, которая различает слова только по признакам «предмет» и «женский род», здесь не на что опереться.
      </>
    ) : (
      <>
        {lead} Признак «существительное» пришёл почти целиком от «Кошки». Эта поправка прибавится к вектору «она»: слово останется местоимением, но теперь будет
        нести и то, на кого указывает.
      </>
    )
  }
  // шаг 9: дуги
  if (row === 0) return <>«Кошка» видит только себя: одна дуга, петля на себя, и в ней весь вес строки, 100 %.</>
  if (isUniformRow(row, c)) {
    const u = uniformText(c, row)
    return (
      <>
        Все дуги одинаковой толщины: {u.why}, и вес делится поровну, по {u.share}.
      </>
    )
  }
  if (row === FOCUS.query) {
    if (ograda) {
      return (
        <>
          Дуги к «Кошке» и к «{c.other.dat}» одинаковой толщины, по {pct(T.weights[row][FOCUS.target])}: по роду их не различить. Нажмите на «перепрыгнула»:
          этот глагол смотрит на «Кошку» ({pct(T.weights[2][0])}), он ищет, кто действует.
        </>
      )
    }
    return (
      <>
        Нажмите на «перепрыгнула»: этот глагол тоже смотрит на «Кошку» ({pct(T.weights[2][0])}), он ищет, кто действует. А у «{c.other.gen}» запрос почти
        нулевой, и его внимание размазано.
      </>
    )
  }
  const top = topIndices(row, c)
  if (top.length > 1) {
    return (
      <>
        Самые толстые дуги, по {pct(T.weights[row][top[0]])}, ведут к словам {listWords(c, top)}.
      </>
    )
  }
  return (
    <>
      Самая толстая дуга ведёт к {nm(c, top[0], 'dat')}: {pct(T.weights[row][top[0]])}.
    </>
  )
}

/* ============================================================================
 * «Как посчитано»: выбранная ячейка на текущем шаге
 * ========================================================================== */

function Detail({ step, sel, c }: { step: number; sel: Sel; c: ToyCase }) {
  const { row, col, dim } = sel
  const T = c.trace
  let head: ReactNode
  let body: ReactNode

  if (step === 0) {
    const f = activeFeatures(row, c)
    head = <>Вектор {nm(c, row, 'gen')}</>
    body = (
      <>
        <Formula tex={`x_{\\text{${c.words[row] === ',' ? 'запятая' : c.words[row]}}} = [${c.X[row].join(',\\ ')}]`} />
        <p className="text-[13px] text-ink-2">
          {f.length ? `Единицы стоят у признаков: ${f.map((i) => FEATURES[i]).join(', ')}.` : 'Все нули: признаков нет.'}
        </p>
      </>
    )
  } else if (PROJ_OF_STEP[step]) {
    const proj = PROJ_OF_STEP[step]
    const names = PROJ_NAME[proj]
    const t = projectionTerms(proj, row, dim, c)
    const Wc = PROJ_W[proj].map((r) => r[dim])
    const macro = proj === 'q' ? '\\cq' : proj === 'k' ? '\\ck' : '\\cv'
    head = (
      <>
        {names.letter}, {nm(c, row, 'nom')}, «{PROJ_DIMS[proj][dim]}»
      </>
    )
    body = (
      <>
        <Formula tex={`${c.X[row].map((x, i) => `${x}\\cdot ${macro}{${texF(Wc[i])}}`).join(' + ')} = ${tex(t.sum)}`} />
        <p className="text-[13px] text-ink-2">
          Признаки слова (0 или 1) умножаются на столбец «{PROJ_DIMS[proj][dim]}» матрицы W<Sub>{names.letter}</Sub> (он обведён), и в сумму попадают только
          ячейки включённых признаков. Нажмите на другую ячейку, чтобы разобрать её.
        </p>
      </>
    )
  } else if (isQK(step)) {
    const t = scoreTerms(row, col, c)
    const dotTex = `${T.Q[row].map((qv, i) => `\\cq{${texF(qv)}}\\cdot\\ck{${texF(T.K[col][i])}}`).join(' + ')} = ${tex(t.sum)}`
    const scaled = T.scaled[row][col]
    const masked = col > row
    head = (
      <>
        {qw(c, row)} → {qw(c, col)}
      </>
    )
    const lines: ReactNode[] = [<Formula key="dot" tex={`q_{\\text{${c.words[row]}}}\\cdot k_{\\text{${c.words[col]}}} = ${dotTex}`} />]
    if (step >= 5) lines.push(<Formula key="scale" tex={`${tex(t.sum)} \\,/\\, \\sqrt{2} = ${tex(scaled, 2)}`} />)
    if (step >= 6)
      lines.push(
        <p key="mask" className="text-[13px] text-ink-2">
          {masked ? (
            <>
              {qw(c, col)} правее {qw(c, row)}, поэтому маска ставит −∞.
            </>
          ) : (
            <>Маска не трогает: {col === row ? 'слово смотрит само на себя' : `${qw(c, col)} левее`}.</>
          )}
        </p>,
      )
    if (step >= 7) {
      const { exps, sum } = softmaxParts(row, c)
      lines.push(
        masked ? (
          <Formula key="sm" tex={`e^{-\\infty} = 0`} />
        ) : (
          <Formula
            key="sm"
            tex={`\\frac{e^{${tex(scaled, 2)}}}{\\sum_j e^{s_j}} = \\frac{${tex(exps[col], 1)}}{${tex(sum, 1)}} = ${tex(exps[col] / sum, 3)} \\approx ${fmt
              .pct(exps[col] / sum)
              .replace(/[\s ]/g, '')
              .replace('%', '')
              .replace(/,/g, '{,}')}\\,\\%`}
          />
        ),
      )
    }
    if (step === 4)
      lines.push(
        <p key="hint" className="text-[13px] text-ink-2">
          Строка Q и столбец Kᵀ сходятся в этой ячейке. Слагаемые попарно перемножают числа запроса и ключа: первое про предмет, второе про женский род.
        </p>,
      )
    body = <>{lines}</>
  } else if (step === 8) {
    const t = outTerms(row, dim, c)
    const parts = t.terms.map((term, j) => ({ term, j })).filter(({ j }) => T.weights[row][j] > 0 && T.V[j][dim] !== 0)
    head = (
      <>
        Выход {nm(c, row, 'gen')}, «{TOY.valueDimNames[dim]}»
      </>
    )
    body = (
      <>
        <Formula
          tex={`${parts.length ? parts.map(({ j }) => `${tex(T.weights[row][j], 3)}\\cdot\\cv{${tex(T.V[j][dim])}}`).join(' + ') : '0'} = ${tex(t.sum, 2)}`}
        />
        <p className="text-[13px] text-ink-2">
          Веса строки умножаются на столбец «{TOY.valueDimNames[dim]}» матрицы V. Показаны ненулевые слагаемые
          {parts.length ? `: ${parts.map(({ j }) => c.words[j]).join(', ')}` : ''}.
        </p>
      </>
    )
  } else {
    head = <>Строка {qw(c, row)}</>
    const top = topIndices(row, c)
    body = (
      <p className="text-[13px] text-ink-2">
        {row === 0 ? (
          <>В строке одно видимое слово, сама «Кошка», и весь вес, 100 %, у неё.</>
        ) : isUniformRow(row, c) ? (
          <>
            Все веса строки одинаковые: {uniformText(c, row).why}, и вес делится поровну, по {uniformText(c, row).share}.
          </>
        ) : top.length > 1 ? (
          <>
            Веса строки в сумме дают 100 %. Самый большой вес, по {pct(T.weights[row][top[0]])}, у слов {listWords(c, top)}.
          </>
        ) : (
          <>
            Веса строки в сумме дают 100 %. Самый большой вес у {nm(c, top[0], 'gen')}: {pct(T.weights[row][top[0]])}.
          </>
        )}{' '}
        Попробуйте другое слово в предложении сверху.
      </p>
    )
  }

  return (
    <div className="grid min-w-0 content-start gap-2 rounded-panel bg-surface-2 px-4 py-3">
      <div className="flex items-center gap-2 text-sm font-semibold text-ink">
        {step >= 1 && step <= 8 && <Swatch role={step <= 3 ? PROJ_OF_STEP[step] : step <= 7 ? 'q' : 'v'} />}
        <span className="min-w-0">Как посчитано: {head}</span>
      </div>
      <div className="grid min-w-0 gap-1.5 overflow-x-auto text-ink scrollbar-thin">{body}</div>
    </div>
  )
}
