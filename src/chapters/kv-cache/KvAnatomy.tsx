import { useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import type { DType } from '@/engine/types'
import { kvBytesPerToken, kvLayout, KV_BYTES_PER_ELEM } from '@/engine/perf'
import { mulberry32 } from '@/engine/rng'
import { useSettings } from '@/state/settings'
import {
  Figure,
  Formula,
  Segmented,
  Select,
  TokenStrip,
  TokenStripSkeleton,
  cn,
  cssVar,
  divColor,
  fmt,
  mix,
  useElementSize,
  useTip,
} from '@/ui'
import { BackToSim, useFollowSim } from './followSim'
import { modelOptions, modelOr, modelWeights, pickModelId } from './lib'
import { usePromptTokens } from './usePromptTokens'

// все модели симулятора здесь есть: фигура следует за его выбором
const ANATOMY_IDS = ['qwen3-8b', 'llama-3.1-8b', 'qwen3-32b', 'qwen3-30b-a3b', 'llama-3.1-70b', 'qwen3-235b-a22b', 'deepseek-v3'] as const

type Factor = 'kv' | 'layers' | 'heads' | 'dim' | 'bytes'

/** 147456 → «147\,456» для KaTeX. */
const texNum = (n: number) => Math.round(n).toLocaleString('en-US').replace(/,/g, '\\,')

/** «байт», «байта» или «байт» по числу: 147 456 байт, 70 272 байта, 1 байт. */
const byteWord = (n: number) => fmt.plural(Math.round(n), ['байт', 'байта', 'байт'])

const LABEL_W = 44
const HEAD_H = 34
const KV_GAP = 10

interface Geometry {
  mla: boolean
  layers: number
  kvHeads: number
  cols: number
  cw: number
  rh: number
  rg: number
  width: number
  height: number
}

function geometry(layers: number, kvHeads: number, mla: boolean, avail: number): Geometry {
  const cols = mla ? 1 : 2 * kvHeads
  const usable = Math.max(160, avail - LABEL_W - (mla ? 0 : KV_GAP))
  const cw = mla ? Math.min(150, usable) : Math.max(10, Math.min(30, Math.floor((usable - (cols - 1) * 2) / cols)))
  const rh = Math.max(3, Math.min(9, Math.floor(340 / layers)))
  const rg = rh >= 6 ? 1.5 : 0.75
  const width = LABEL_W + (mla ? cw : cols * cw + (cols - 1) * 2 + KV_GAP)
  const height = HEAD_H + layers * (rh + rg)
  return { mla, layers, kvHeads, cols, cw, rh, rg, width, height }
}

function colX(g: Geometry, col: number): number {
  if (g.mla) return LABEL_W
  const gap = col >= g.kvHeads ? KV_GAP : 0
  return LABEL_W + col * (g.cw + 2) + gap
}

const rowY = (g: Geometry, layer: number) => HEAD_H + layer * (g.rh + g.rg)

/** Пробелы и переводы строк видимыми: « кэш» → «·кэш», «\n» → «↵». */
function visibleToken(text: string): string {
  return text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→') || '∅'
}

/**
 * Дальше этой позиции лента токенов не растёт: промпт с описанием инструмента (~190 токенов) показывается до вопроса,
 * а длинный документ на тысячу токенов не превращается в стену чипов.
 */
const STRIP_MAX = 200

/**
 * Какой токен показать по умолчанию: «KV», иначе первый обычный токен пользовательской части, иначе (вопрос дальше
 * STRIP_MAX, например после длинного system prompt) первый обычный токен промпта. Всегда в пределах ленты.
 */
function defaultTokenIndex(tokens: ReadonlyArray<{ text: string; special: boolean }> | null): number {
  if (!tokens || !tokens.length) return 0
  const end = Math.min(tokens.length, STRIP_MAX)
  const plain = (i: number) => !tokens[i].special && tokens[i].text.trim().length > 1
  const kv = tokens.findIndex((t) => t.text.trim() === 'KV')
  if (kv >= 0 && kv < end) return kv
  const lastUser = tokens.map((t) => t.text).lastIndexOf('user')
  for (let i = Math.max(0, lastUser + 2); i < end; i++) if (plain(i)) return i
  // после роли и перевода строки первой реплики (обычно system)
  const firstRole = tokens.findIndex((t) => t.text === 'system' || t.text === 'user')
  for (let i = Math.max(0, firstRole + 2); i < end; i++) if (plain(i)) return i
  return 0
}

/** Иллюстративные значения вектора (детерминированно от токена, слоя и столбца). */
function vectorValues(n: number, seed: number): number[] {
  const rng = mulberry32(seed >>> 0)
  return Array.from({ length: n }, () => {
    const v = (rng() + rng() + rng() - 1.5) / 1.5
    return Math.max(-1, Math.min(1, v))
  })
}

export function KvAnatomy() {
  const [settings] = useSettings()
  const simModelId = pickModelId(settings.modelId, ANATOMY_IDS)
  const modelSel = useFollowSim(simModelId)
  const modelId = modelSel.id
  const [kvDtype, setKvDtype] = useState<DType>('bf16')
  const [focus, setFocus] = useState<Factor | null>(null)
  const [sel, setSel] = useState({ layer: 11, col: 2 })
  const [tokIdx, setTokIdx] = useState<number | null>(null)
  const { tokens, status } = usePromptTokens()
  const model = modelOr(modelId)
  const layout = kvLayout(model, kvDtype)
  const perToken = kvBytesPerToken(model, kvDtype)
  const mla = model.attention === 'mla'
  const b = KV_BYTES_PER_ELEM[kvDtype]
  const latentDim = model.mlaDim ?? 576
  const vecLen = mla ? latentDim : model.headDim
  const vecBytes = vecLen * b
  // «8 голов», «4 головы»: подпись над столбцами K и V и aria-label решётки
  const headsText = `${model.kvHeads} ${fmt.plural(model.kvHeads, ['голова', 'головы', 'голов'])}`

  const [boxRef, box] = useElementSize<HTMLDivElement>()
  const g = useMemo(() => geometry(model.layers, model.kvHeads, mla, Math.min(box.width || 420, 560)), [model.layers, model.kvHeads, mla, box.width])
  const layer = Math.min(sel.layer, model.layers - 1)
  const col = Math.min(sel.col, g.cols - 1)
  const kind: 'K' | 'V' | 'latent' = mla ? 'latent' : col < model.kvHeads ? 'K' : 'V'
  const head = mla ? 0 : col % model.kvHeads
  const defaultIdx = useMemo(() => defaultTokenIndex(tokens), [tokens])
  const shownIdx = tokens && tokens.length ? Math.min(tokIdx ?? defaultIdx, tokens.length - 1) : 0
  const token = tokens && tokens.length ? tokens[shownIdx] : null
  const values = useMemo(
    () => vectorValues(vecLen, (token?.id ?? 7) * 1_000_003 + layer * 131 + col * 17 + (kvDtype === 'fp8' ? 5 : 0)),
    [vecLen, token?.id, layer, col, kvDtype],
  )

  const svgRef = useRef<SVGSVGElement | null>(null)
  const tip = useTip({ scopeEl: svgRef.current, side: 'top' })
  const gradId = useId().replace(/:/g, '')

  const cellLabel = (l: number, c: number) => {
    if (mla) return `Слой ${l + 1}, latent: ${latentDim} чисел = ${fmt.int(latentDim * b)} ${byteWord(latentDim * b)}`
    const k = c < model.kvHeads ? 'K' : 'V'
    return `Слой ${l + 1}, ${k}, KV-голова ${(c % model.kvHeads) + 1}: ${model.headDim} чисел = ${fmt.int(model.headDim * b)} ${byteWord(model.headDim * b)}`
  }

  // Подсветка множителя: мышь наводит, тач и клавиатура переключают нажатием (на таче enter и leave
  // приходят подряд перед click, поэтому для них наведение не используем).
  const lastPointer = useRef('')
  const clickFactor = (id: Factor) => {
    if (lastPointer.current === 'mouse') setFocus(id)
    else setFocus((v) => (v === id ? null : id))
    lastPointer.current = ''
  }

  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current
    if (!svg) return null
    const r = svg.getBoundingClientRect()
    const x = e.clientX - r.left
    const y = e.clientY - r.top
    if (y < HEAD_H || x < LABEL_W) return null
    const l = Math.floor((y - HEAD_H) / (g.rh + g.rg))
    if (l < 0 || l >= model.layers) return null
    let c = 0
    if (!mla) {
      const kx = x - LABEL_W
      const vStart = model.kvHeads * (g.cw + 2) + KV_GAP
      c = kx >= vStart ? model.kvHeads + Math.floor((kx - vStart) / (g.cw + 2)) : Math.floor(kx / (g.cw + 2))
      if (c < 0 || c >= g.cols) return null
    }
    return { l, c, rect: { left: r.left + colX(g, c), top: r.top + rowY(g, l), width: g.cw, height: g.rh } }
  }

  const lastTip = useRef('')
  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const p = pick(e)
    if (!p) {
      lastTip.current = ''
      return tip.hide()
    }
    const key = `${p.l}:${p.c}`
    if (key === lastTip.current && tip.visible) return
    lastTip.current = key
    tip.show(p.rect, <span>{cellLabel(p.l, p.c)}</span>)
  }
  const onDown = (e: PointerEvent<SVGSVGElement>) => {
    const p = pick(e)
    if (!p) return
    setSel({ layer: p.l, col: p.c })
    tip.show(p.rect, <span>{cellLabel(p.l, p.c)}</span>)
  }
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    const moves: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }
    const m = moves[e.key]
    if (!m) return
    e.preventDefault()
    const nl = Math.max(0, Math.min(model.layers - 1, layer + m[0]))
    const nc = Math.max(0, Math.min(g.cols - 1, col + m[1]))
    setSel({ layer: nl, col: nc })
    const svg = svgRef.current
    if (svg) {
      const r = svg.getBoundingClientRect()
      tip.show({ left: r.left + colX(g, nc), top: r.top + rowY(g, nl), width: g.cw, height: g.rh }, <span>{cellLabel(nl, nc)}</span>)
    }
  }

  // клеток до 752: собираем один раз на набор параметров, а не на каждое движение мыши
  const cells = useMemo(() => {
    const dimCell = (l: number, c: number) => {
      if (focus === 'heads') return l !== layer
      if (focus === 'dim' || focus === 'bytes') return !(l === layer && c === col)
      return false
    }
    return Array.from({ length: model.layers }, (_, l) =>
      Array.from({ length: g.cols }, (_, c) => (
        <rect
          key={`${l}-${c}`}
          x={colX(g, c)}
          y={rowY(g, l)}
          width={g.cw}
          height={g.rh}
          rx={Math.min(2, g.rh / 2)}
          fill={mla ? `url(#${gradId}-lat)` : mix(c < model.kvHeads ? 'k' : 'v', 58)}
          opacity={dimCell(l, c) ? 0.22 : 1}
        />
      )),
    )
  }, [g, model.layers, model.kvHeads, mla, gradId, focus, layer, col])

  const factors: Array<{ id: Factor; value: string; label: string }> = mla
    ? [
        { id: 'layers', value: String(model.layers), label: fmt.plural(model.layers, ['слой', 'слоя', 'слоёв']) },
        { id: 'dim', value: String(latentDim), label: 'чисел latent на слой' },
        { id: 'bytes', value: fmt.num(b), label: kvDtype === 'fp8' ? 'байт на число (FP8)' : 'байта на число (BF16)' },
      ]
    : [
        { id: 'kv', value: '2', label: 'K и V' },
        { id: 'layers', value: String(model.layers), label: fmt.plural(model.layers, ['слой', 'слоя', 'слоёв']) },
        { id: 'heads', value: String(model.kvHeads), label: fmt.plural(model.kvHeads, ['KV-голова', 'KV-головы', 'KV-голов']) },
        { id: 'dim', value: String(model.headDim), label: 'чисел в голове' },
        { id: 'bytes', value: fmt.num(b), label: kvDtype === 'fp8' ? 'байт на число (FP8)' : 'байта на число (BF16)' },
      ]

  // две строки: на телефоне итог (147 456 байт) не уезжает за край
  const tex = mla
    ? `\\begin{aligned}\\text{KV на токен} &= L \\cdot (d_c + d_r) \\cdot b\\\\ &= ${model.layers} \\cdot (${model.kvLoraRank ?? 512} + ${model.qkRopeHeadDim ?? 64}) \\cdot ${b} = ${texNum(perToken)}\\ \\text{${byteWord(perToken)}}\\end{aligned}`
    : `\\begin{aligned}\\text{KV на токен} &= 2 \\cdot L \\cdot h_{kv} \\cdot d_{head} \\cdot b\\\\ &= 2 \\cdot ${model.layers} \\cdot ${model.kvHeads} \\cdot ${model.headDim} \\cdot ${b} = ${texNum(perToken)}\\ \\text{${byteWord(perToken)}}\\end{aligned}`

  const vectorsPerToken = mla ? model.layers : model.layers * 2 * model.kvHeads
  const lensCols = mla ? 24 : 16
  const lensCell = mla ? 7 : 11
  const lensRows = Math.ceil(vecLen / lensCols)
  const promptBytes = tokens ? tokens.length * perToken : 0
  const ctx32k = 32768 * perToken

  return (
    <Figure
      id="kv-cache-figure-anatomy"
      title="Из чего состоит KV одного токена"
      width="wide"
      controls={
        <>
          <Select
            label="Модель"
            value={modelId}
            onChange={modelSel.set}
            options={modelOptions(ANATOMY_IDS)}
            hint={modelSel.overridden ? <BackToSim what="модель" name={modelOr(simModelId).name} onClick={modelSel.reset} /> : 'из симулятора'}
          />
          <Segmented
            label="Формат KV"
            value={kvDtype}
            onChange={setKvDtype}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp8', label: 'FP8' },
            ]}
          />
        </>
      }
      caption="Попробуйте: наведите на множители над решёткой или нажмите на них, чтобы увидеть, за что отвечает каждый, а потом выберите DeepSeek-V3: вместо K и V по головам там один сжатый вектор на слой."
    >
      <div className="grid min-w-0 gap-5">
        <Formula block tex={tex} className="max-sm:text-[14px]" label={`= ${fmt.bytes(perToken, { mode: 'binary' })} на каждый токен контекста`} />

        <div role="group" aria-label="Множители формулы" className="flex flex-wrap gap-2">
          {factors.map((f, i) => (
            <span key={f.id} className="inline-flex items-center gap-2">
              {i > 0 && (
                <span aria-hidden className="text-ink-3">
                  ×
                </span>
              )}
              <button
                type="button"
                onPointerDown={(e) => {
                  lastPointer.current = e.pointerType
                }}
                onPointerEnter={(e) => e.pointerType === 'mouse' && setFocus(f.id)}
                onPointerLeave={(e) => e.pointerType === 'mouse' && setFocus(null)}
                onFocus={(e) => e.currentTarget.matches(':focus-visible') && setFocus(f.id)}
                onBlur={() => setFocus(null)}
                onClick={() => clickFactor(f.id)}
                aria-pressed={focus === f.id}
                className={cn(
                  'grid min-w-[72px] justify-items-start rounded-control border px-3 py-1.5 text-left transition-colors',
                  focus === f.id ? 'border-accent bg-accent-soft' : 'border-line bg-surface-2 hover:bg-surface-3',
                )}
              >
                <span className="text-[17px] leading-tight font-semibold text-ink">{f.value}</span>
                <span className="text-xs text-ink-2">{f.label}</span>
              </button>
            </span>
          ))}
        </div>

        <div className="grid min-w-0 gap-6 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,300px)]">
          <div ref={boxRef} className="min-w-0 overflow-x-auto pb-1">
            <svg
              ref={svgRef}
              width={g.width + 4}
              height={g.height}
              viewBox={`0 0 ${g.width + 4} ${g.height}`}
              tabIndex={0}
              role="img"
              aria-label={`KV одного токена ${model.name}: ${model.layers} ${fmt.plural(model.layers, ['слой', 'слоя', 'слоёв'])} × ${mla ? `один latent-вектор из ${latentDim} чисел` : `${headsText} K и ${headsText} V по ${model.headDim} чисел`}. Выбрано: ${cellLabel(layer, col)}`}
              onPointerMove={onMove}
              onPointerDown={onDown}
              onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
              onKeyDown={onKey}
              onBlur={() => tip.hide()}
              className="block cursor-crosshair rounded-[4px] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <defs>
                <linearGradient id={`${gradId}-lat`} x1="0" x2="1" y1="0" y2="0">
                  <stop offset="0" stopColor={mix('k', 60)} />
                  <stop offset="1" stopColor={mix('v', 60)} />
                </linearGradient>
              </defs>
              {/* шапка */}
              {mla ? (
                <g fontSize={11} fill="var(--ink-2)">
                  <text x={LABEL_W} y={14}>
                    latent = сжатые K и V вместе
                  </text>
                  <text x={LABEL_W} y={28} fill="var(--ink-3)">
                    {model.kvLoraRank ?? 512} + {model.qkRopeHeadDim ?? 64} RoPE
                  </text>
                </g>
              ) : (
                <g fontSize={11}>
                  {(['K', 'V'] as const).map((k, gi) => {
                    const x0 = colX(g, gi * model.kvHeads)
                    const w = model.kvHeads * (g.cw + 2) - 2
                    return (
                      <g key={k}>
                        <rect x={x0} y={3} width={8} height={8} rx={2} fill={cssVar(gi === 0 ? 'k' : 'v')} />
                        <text x={x0 + 12} y={11} fill="var(--ink)" fontWeight={600}>
                          {k}
                        </text>
                        <text x={x0 + 26} y={11} fill="var(--ink-3)">
                          {g.cw >= 18 ? headsText : ''}
                        </text>
                        {g.cw >= 14 &&
                          Array.from({ length: model.kvHeads }, (_, h) => (
                            <text key={h} x={colX(g, gi * model.kvHeads + h) + g.cw / 2} y={28} textAnchor="middle" fill="var(--ink-3)" fontSize={10}>
                              {h + 1}
                            </text>
                          ))}
                        {focus === 'kv' && (
                          <rect
                            x={x0 - 3}
                            y={HEAD_H - 3}
                            width={w + 6}
                            height={model.layers * (g.rh + g.rg) + 4}
                            rx={4}
                            fill="none"
                            stroke="var(--accent)"
                            strokeWidth={2}
                          />
                        )}
                      </g>
                    )
                  })}
                </g>
              )}
              {/* подписи слоёв */}
              <g fontSize={10} fill="var(--ink-3)" textAnchor="end">
                {Array.from({ length: model.layers }, (_, l) => l)
                  .filter((l) => l === 0 || l === model.layers - 1 || ((l + 1) % (model.layers > 60 ? 20 : 6) === 0 && l < model.layers - 4))
                  .map((l) => (
                    <text key={l} x={LABEL_W - 8} y={rowY(g, l) + g.rh / 2 + 3.5}>
                      {l + 1}
                    </text>
                  ))}
              </g>
              {focus === 'layers' && (
                <g>
                  <line x1={6} x2={6} y1={HEAD_H} y2={rowY(g, model.layers - 1) + g.rh} stroke="var(--accent)" strokeWidth={2} />
                  <line x1={6} x2={12} y1={HEAD_H} y2={HEAD_H} stroke="var(--accent)" strokeWidth={2} />
                  <line x1={6} x2={12} y1={rowY(g, model.layers - 1) + g.rh} y2={rowY(g, model.layers - 1) + g.rh} stroke="var(--accent)" strokeWidth={2} />
                </g>
              )}
              {/* клетки */}
              {cells}
              {/* выбранная клетка */}
              <rect
                x={colX(g, col) - 1.5}
                y={rowY(g, layer) - 1.5}
                width={g.cw + 3}
                height={g.rh + 3}
                rx={2.5}
                fill="none"
                stroke="var(--accent)"
                strokeWidth={2}
                pointerEvents="none"
              />
            </svg>
            <p className="mt-2 text-sm text-ink-2">
              Один токен: {model.layers} {fmt.plural(model.layers, ['слой', 'слоя', 'слоёв'])} ×{' '}
              {mla ? '1 вектор' : `${2 * model.kvHeads} ${fmt.plural(2 * model.kvHeads, ['вектор', 'вектора', 'векторов'])}`} × {fmt.int(vecBytes)}{' '}
              {byteWord(vecBytes)} ={' '}
              <span className="font-semibold text-ink">
                {fmt.int(vectorsPerToken * vecBytes)} {byteWord(vectorsPerToken * vecBytes)} ({fmt.bytes(perToken, { mode: 'binary' })})
              </span>
            </p>
          </div>

          <div className="grid min-w-0 content-start gap-3 self-start rounded-panel bg-surface-2 p-4">
            <div className="text-sm font-medium text-ink">Под лупой</div>
            <div className="text-[13px] leading-snug text-ink-2">
              {mla ? (
                <>
                  Слой {layer + 1}: latent-вектор из {latentDim} чисел
                </>
              ) : (
                <>
                  Слой {layer + 1}, {kind}, KV-голова {head + 1}: вектор из {model.headDim} чисел
                </>
              )}
              {token && (
                <>
                  {' '}
                  для токена <span className="rounded-chip bg-surface px-1 font-mono text-ink">{visibleToken(token.text)}</span>
                </>
              )}
            </div>
            <svg
              width={lensCols * lensCell}
              height={lensRows * lensCell}
              viewBox={`0 0 ${lensCols * lensCell} ${lensRows * lensCell}`}
              role="img"
              aria-label={`${vecLen} условных чисел вектора`}
              className={cn('block transition-opacity', focus === 'dim' || focus === 'bytes' || focus === null ? 'opacity-100' : 'opacity-60')}
            >
              {values.map((v, i) => (
                <rect
                  key={i}
                  x={(i % lensCols) * lensCell}
                  y={Math.floor(i / lensCols) * lensCell}
                  width={lensCell - 1}
                  height={lensCell - 1}
                  rx={1.5}
                  fill={divColor(v)}
                  stroke={focus === 'bytes' && i === 0 ? 'var(--accent)' : undefined}
                  strokeWidth={focus === 'bytes' && i === 0 ? 2 : undefined}
                />
              ))}
            </svg>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-2" aria-hidden>
              {([-1, 0, 1] as const).map((v) => (
                <span key={v} className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-2.5 rounded-[2px] border border-line" style={{ background: divColor(v) }} />
                  {v < 0 ? '−1' : v > 0 ? '+1' : '0'}
                </span>
              ))}
            </div>
            <div className="text-[13px] leading-snug text-ink-2">
              {vecLen} чисел × {fmt.num(b)} {byteWord(b)} ={' '}
              <span className="font-semibold text-ink">
                {fmt.int(vecBytes)} {byteWord(vecBytes)}
              </span>
              . Цвет показывает знак и величину числа (шкала выше); сами значения условные.
            </div>
            {focus === 'bytes' && (
              <div className="text-[13px] leading-snug text-ink-2">
                Одно число в {kvDtype === 'fp8' ? 'FP8 занимает 1 байт (8 бит)' : 'BF16 занимает 2 байта (16 бит)'}. Формат KV можно
                выбрать отдельно от формата весов.
              </div>
            )}
          </div>
        </div>

        <div className="grid min-w-0 gap-3 border-t border-line pt-5">
          <div className="text-sm text-ink-2">
            Такая решётка есть у каждого токена вашего запроса из симулятора{tokens ? `, их ${fmt.tokens(tokens.length)}` : ''}. Нажмите
            на токен, чтобы рассмотреть его.
          </div>
          {tokens ? (
            // первые 80 токенов; если выбранный токен дальше (с tools описание инструмента занимает ~150 токенов
            // перед вопросом), лента доходит до него (но не дальше STRIP_MAX), чтобы подсвеченный токен был виден
            <TokenStrip tokens={tokens} size="sm" limit={Math.max(80, Math.min(STRIP_MAX, shownIdx + 1))} highlightIndex={shownIdx} onClickIndex={setTokIdx} />
          ) : status === 'error' ? (
            <p className="text-sm text-ink-3">Токенизатор не загрузился, поэтому ленты нет; формулы выше от этого не зависят.</p>
          ) : (
            <TokenStripSkeleton count={30} />
          )}
          <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm @xl:grid-cols-3">
            <div className="grid gap-0.5">
              <dt className="text-ink-3">Весь ваш промпт</dt>
              <dd className="font-medium text-ink">
                {tokens ? `${fmt.tokens(tokens.length)} → ${fmt.bytes(promptBytes, { mode: 'binary' })}` : '…'}
              </dd>
            </div>
            <div className="grid gap-0.5">
              <dt className="text-ink-3">Контекст 32 768 токенов</dt>
              <dd className="font-medium text-ink">{fmt.bytes(ctx32k, { mode: 'binary' })} на одну последовательность</dd>
            </div>
            <div className="grid gap-0.5">
              <dt className="text-ink-3">Для сравнения: веса {model.name}</dt>
              <dd className="font-medium text-ink">{fmt.bytes(modelWeights(model), { mode: 'binary' })} в BF16</dd>
            </div>
          </dl>
          {layout.slidingLayers > 0 && <p className="text-sm text-ink-3">У модели есть слои со скользящим окном, подробнее в «Глубже».</p>}
        </div>
      </div>
      {tip.node}
    </Figure>
  )
}
