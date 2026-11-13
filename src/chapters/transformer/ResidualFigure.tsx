import { useMemo } from 'react'
import { BarChart, Figure, LineChart, Player, Stat, fmt, usePlayer, useElementSize } from '@/ui'
import { Barcode, BarcodeLegend } from '../embeddings/Barcode'
import { embeddingRow, norm, pseudoVector } from '../embeddings/vectors'
import { LENS_CONTEXT, lensAt, lensTop } from './lens'
import type { Arch } from './arch'

/** ID токена « —» в словаре Qwen3 (последний токен контекста «Столица Франции —»). */
const DASH_TOKEN_ID = 1959
const DIMS = 64

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

/** Условный поток: x₀ = эмбеддинг, xᵢ = xᵢ₋₁ + Δᵢ; в последней трети вклады тянут к «направлению ответа». */
function buildStream(layers: number) {
  const x0 = embeddingRow(DASH_TOKEN_ID, DIMS)
  const answer = pseudoVector(4242, DIMS, 9)
  const xs: number[][] = [x0]
  const deltas: number[][] = [x0.map(() => 0)]
  for (let i = 1; i <= layers; i++) {
    const f = i / layers
    const noise = pseudoVector(1000 + i, DIMS, 7)
    const pull = 0.55 * sigmoid((f - 0.62) / 0.08)
    const d = noise.map((v, j) => (0.32 + 0.2 * f) * v + pull * answer[j])
    deltas.push(d)
    xs.push(xs[i - 1].map((v, j) => v + d[j]))
  }
  return { xs, deltas }
}

function cosine(a: number[], b: number[]) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s / (norm(a) * norm(b))
}

export function ResidualFigure({ arch }: { arch: Arch }) {
  const L = arch.model.layers
  const p = usePlayer(L + 1, { intervalMs: (s) => (s === 0 ? 1400 : 380) })
  const layer = Math.min(p.step, L)
  const labels = useMemo(() => Array.from({ length: L + 1 }, (_, i) => (i === 0 ? 'эмбеддинг на входе' : `после блока ${i}`)), [L])
  const { xs, deltas } = useMemo(() => buildStream(L), [L])

  const x = xs[layer]
  const rms = norm(x) / Math.sqrt(DIMS)
  const shownX = x.map((v) => v / rms)
  const shownD = deltas[layer].map((v) => v / rms)
  const similarity = cosine(x, xs[0])
  const share = layer === 0 ? 0 : norm(deltas[layer]) / norm(xs[layer - 1])

  const series = useMemo(() => {
    const pts = (text: string) => Array.from({ length: L + 1 }, (_, i) => [i, lensAt(i / L).find((c) => c.text === text)!.p] as [number, number])
    return [
      { id: 'paris', label: '«·Париж»', color: 'decode', points: pts(' Париж') },
      { id: 'and', label: '«·и»', color: 'var(--ink-3)', points: pts(' и'), dashed: true },
    ]
  }, [L])
  const top = lensTop(layer, L, 3)

  return (
    <Figure
      id="transformer-figure-residual"
      title="Residual stream: каждый блок дописывает поправку к вектору токена"
      controls={<Player {...p.bind} labels={labels} aria-label="Проход по слоям" />}
      caption={`Попробуйте: запустите проигрывание или двигайте ползунок шагов. Вектор не заменяется, а накапливает вклады блоков, и предсказание проявляется к последним слоям. Иллюстрация: векторы условные, а вероятности показывают приём «logit lens» на условных числах для контекста «${LENS_CONTEXT}».`}
    >
      <div className="grid min-w-0 gap-5">
        <LayerStrip layers={L} current={layer} onPick={(i) => p.setStep(i)} />
        <div className="grid min-w-0 gap-6 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="grid min-w-0 content-start gap-4">
            <div className="grid gap-1.5">
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="font-medium text-ink">Вектор токена «·—» {layer === 0 ? 'на входе' : `после блока ${layer}`}</span>
                <span className="text-xs text-ink-3">{DIMS} из {fmt.int(arch.model.hidden)} чисел</span>
              </div>
              <Barcode values={shownX} height={30} label={`Условный вектор потока после блока ${layer}`} />
            </div>
            <div className="grid gap-1.5">
              <div className="text-sm font-medium text-ink">{layer === 0 ? 'Вклад блока: пока нет' : `Вклад блока ${layer}: attention + MLP`}</div>
              <Barcode values={shownD} height={18} label={`Условный вклад блока ${layer}`} />
            </div>
            <BarcodeLegend />
            <div className="grid grid-cols-2 gap-x-6 gap-y-3">
              <Stat label="Сходство со входом" value={fmt.num(similarity, 2)} hint="косинус угла с эмбеддингом, условно" />
              <Stat label="Размер вклада" value={layer === 0 ? '0' : fmt.pct(share)} hint="от длины вектора до блока" />
            </div>
          </div>
          <div className="grid min-w-0 content-start gap-4">
            <div className="grid gap-2">
              <div className="text-sm font-medium text-ink">Если остановиться здесь: топ-3 следующих токена (условно)</div>
              <BarChart
                data={top.map((c) => ({ label: c.text.replace(/^ /, '·'), value: c.p, id: c.text }))}
                max={1}
                valueFormat={(v) => fmt.pct(v)}
                labelFont="mono"
                color="decode"
                aria-label="Топ-3 кандидата на текущем слое"
              />
            </div>
            <LineChart
              height={150}
              xLabel="Слой"
              yLabel="Вероятность"
              series={series}
              yDomain={[0, 1]}
              yFormat={(v) => fmt.pct(v)}
              xFormat={(v) => fmt.int(v)}
              annotations={[{ kind: 'x', x: layer, label: layer === 0 ? 'вход' : `блок ${layer}` }]}
              aria-label="Вероятность ответа по слоям"
            />
          </div>
        </div>
      </div>
    </Figure>
  )
}

function LayerStrip({ layers, current, onPick }: { layers: number; current: number; onPick: (i: number) => void }) {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  const W = Math.max(200, width)
  const gap = layers > 60 ? 1 : 2
  const cell = (W - gap * (layers - 1)) / layers
  return (
    <div ref={ref} className="grid min-w-0 gap-1">
      <div className="flex justify-between text-xs text-ink-3">
        <span>блок 1</span>
        <span>
          {fmt.int(layers)} {fmt.plural(layers, ['блок', 'блока', 'блоков'])}, веса у каждого свои
        </span>
        <span>блок {layers}</span>
      </div>
      {width > 0 && (
        <svg width={W} height={26} role="img" aria-label={`Стопка из ${layers} блоков, пройдено ${current}`} className="block">
          {Array.from({ length: layers }, (_, i) => {
            const n = i + 1
            const on = n === current
            const done = n < current
            return (
              <rect
                key={i}
                x={i * (cell + gap)}
                y={on ? 0 : 4}
                width={Math.max(1, cell)}
                height={on ? 26 : 18}
                rx={Math.min(3, cell / 3)}
                onClick={() => onPick(n)}
                style={{
                  fill: on ? 'var(--accent)' : done ? 'var(--ink-3)' : 'var(--surface-3)',
                  stroke: done || on ? 'none' : 'var(--line-strong)',
                  strokeWidth: 0.75,
                  cursor: 'pointer',
                  transition: 'fill 120ms',
                }}
              />
            )
          })}
        </svg>
      )}
    </div>
  )
}
