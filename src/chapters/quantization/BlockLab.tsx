import { useMemo, useState } from 'react'
import { Figure, Segmented, Stat, Toggle, fmt, useElementSize } from '@/ui'
import { BLOCK_SCHEMES, quantizeBlocks, sampleWeights, type BlockQuantResult, type BlockScheme } from './blocks'

const PADL = 44
const PADR = 10
const TOP = 34
const BOTTOM = 26
/** Положительные значения FP4 E2M1: узлы сетки блока = масштаб × эти числа (и зеркально вниз). */
const FP4_LEVELS = [0.5, 1, 1.5, 2, 3, 4, 6]
const FONT = 'Onest Variable, sans-serif'

function num(v: number, d = 3): string {
  return fmt.num(v, d)
}

export function BlockLab() {
  const [scheme, setScheme] = useState<BlockScheme>('tensor')
  const [outlier, setOutlier] = useState(true)
  const [hover, setHover] = useState<number | null>(null)
  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  const weights = useMemo(() => sampleWeights({ outlier }), [outlier])
  const res = useMemo(() => quantizeBlocks(weights, scheme), [weights, scheme])
  const info = BLOCK_SCHEMES.find((s) => s.id === scheme)!
  const n = weights.length
  const regular = weights.filter((_, i) => !(outlier && i === 5))
  const yMax = Math.max(...regular.map(Math.abs)) * 1.18

  // ошибка обычных весов (без выброса): выброс сам по себе квантуется почти точно и прячет порчу остальных
  const small = weights.map((w, i) => ({ w, q: res.q[i], i })).filter((x) => !(outlier && x.i === 5))
  const smallErr = Math.sqrt(small.reduce((s, x) => s + (x.q - x.w) ** 2, 0) / Math.max(1e-12, small.reduce((s, x) => s + x.w * x.w, 0)))
  const bits = scheme === 'tensor' ? 4 : res.bitsPerValue
  const block = hover !== null ? res.blocks.find((b) => hover >= b.start && hover < b.end) : null

  // SVG по ширине фигуры (подписи 11 px на любом экране); на узком экране две строки по 32 веса
  const W = Math.max(300, Math.round(width || 660))
  const narrow = W < 520
  const rows: Array<[number, number]> = narrow ? [[0, n / 2], [n / 2, n]] : [[0, n]]

  return (
    <Figure
      id="quantization-figure-blocks"
      title="64 веса одной строки матрицы в FP4: зачем масштаб на блок"
      controls={
        <>
          <Segmented
            label="Масштаб"
            value={scheme}
            onChange={setScheme}
            options={BLOCK_SCHEMES.map((s) => ({ value: s.id, label: s.label }))}
          />
          <Toggle label="Выброс в первом блоке" checked={outlier} onChange={setOutlier} hint="один вес 0,9 при типичных ±0,05" />
        </>
      }
      caption="Попробуйте: с выбросом переключите «Один масштаб» на NVFP4. При одном масштабе на все 64 веса шаг сетки подстраивается под выброс, и почти все мелкие веса становятся нулём. С блоками выброс портит только свой блок. У MXFP4 масштаб только степень двойки, поэтому сетка подгоняется под блок грубо: по правилу OCP после деления на масштаб максимум блока оказывается между 4 и 8, а сетка FP4 кончается на 6, и всё выше 6 обрезается (здесь выброс 0,9 стал 0,75). У NVFP4 дробный масштаб ставит максимум блока почти точно на 6."
      source="Веса условные: нормальное распределение, seed фиксирован. Пунктир в каждом блоке: узлы его сетки FP4 (масштаб × 0,5, 1, 1,5, 2, 3, 4, 6)."
    >
      <div className="grid min-w-0 gap-5">
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @lg:grid-cols-4">
          <Stat label="Стали нулём" value={fmt.int(res.zeroed)} unit={`из ${n}`} />
          <Stat label="Ошибка обычных весов" value={fmt.pct(smallErr, smallErr < 0.1 ? 1 : 0)} hint="среднеквадратичная, без выброса" />
          <Stat label="Бит на вес" value={scheme === 'tensor' ? '≈ 4' : fmt.num(bits, 2)} hint={scheme === 'tensor' ? 'масштаб FP32 один на миллионы весов' : `4 бита FP4 + масштаб ${info.scaleName}`} />
          <Stat label="Масштабов" value={fmt.int(res.blocks.length)} hint={info.blockSize ? `по одному на ${fmt.count(info.blockSize, ['вес', 'веса', 'весов'])}` : 'один на весь тензор'} />
        </div>
        <div ref={wrapRef} className="grid min-w-0 gap-2" onPointerLeave={(e) => e.pointerType === 'mouse' && setHover(null)}>
          {rows.map(([start, end]) => (
            <WeightsPlot
              key={start}
              W={W}
              H={narrow ? 210 : 250}
              start={start}
              end={end}
              weights={weights}
              res={res}
              yMax={yMax}
              hover={hover}
              onHover={setHover}
              label={`${info.label}: веса ${start + 1}…${end}, стали нулём ${res.q.slice(start, end).filter((q, k) => q === 0 && weights[start + k] !== 0).length} из ${end - start}`}
            />
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[13px] text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-3 w-[3px] rounded-full bg-ink-3 opacity-60" />
            исходный вес
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-ink" />
            после FP4 и масштаба
          </span>
          <span className="inline-flex items-center gap-1.5">
            <svg width="10" height="10" aria-hidden>
              <circle cx="5" cy="5" r="3.5" fill="var(--surface)" stroke="var(--bad)" strokeWidth="1.6" />
            </svg>
            стал нулём
          </span>
          <span className="inline-flex items-center gap-1.5">
            <svg width="16" height="6" aria-hidden>
              <line x1="0" x2="16" y1="3" y2="3" stroke="var(--ink-3)" strokeDasharray="2 3" />
            </svg>
            сетка FP4 блока
          </span>
        </div>
        <p className="min-h-[1.5rem] font-mono text-[13px] text-ink-2 tabular-nums" aria-live="polite">
          {hover !== null && block
            ? `вес ${hover + 1}: ${num(weights[hover], 3)} → ${num(res.q[hover], 4)}, масштаб блока ${num(block.scale, 4)}, в сетке FP4: ${num(res.q[hover] / block.scale, 1)}`
            : 'наведите или нажмите на вес, чтобы увидеть, во что он превратился'}
        </p>
      </div>
    </Figure>
  )
}

function WeightsPlot({
  W,
  H,
  start,
  end,
  weights,
  res,
  yMax,
  hover,
  onHover,
  label,
}: {
  W: number
  H: number
  start: number
  end: number
  weights: number[]
  res: BlockQuantResult
  yMax: number
  hover: number | null
  onHover: (i: number) => void
  label: string
}) {
  const count = end - start
  const plotW = W - PADL - PADR
  const plotH = H - TOP - BOTTOM
  const cy = TOP + plotH / 2
  const xAt = (i: number) => PADL + ((i - start) / count) * plotW
  const sx = (i: number) => xAt(i + 0.5)
  const sy = (v: number) => cy - (Math.max(-yMax, Math.min(yMax, v)) / yMax) * (plotH / 2)
  const clipped = (v: number) => Math.abs(v) > yMax
  const blocks = res.blocks.filter((b) => b.end > start && b.start < end)
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block max-w-full touch-pan-y" role="img" aria-label={label}>
      {/* сетка значений */}
      {[-1, -0.5, 0, 0.5, 1].map((t) => (
        <g key={t}>
          <line x1={PADL} x2={W - PADR} y1={cy - t * (plotH / 2)} y2={cy - t * (plotH / 2)} stroke={t === 0 ? 'var(--line-strong)' : 'var(--grid)'} />
          <text x={PADL - 6} y={cy - t * (plotH / 2) + 4} textAnchor="end" fontSize={11} fill="var(--ink-3)" fontFamily={FONT}>
            {t === 0 ? '0' : fmt.num(t * yMax, 2)}
          </text>
        </g>
      ))}
      {/* блоки: граница, подпись масштаба и узлы сетки FP4 */}
      {blocks.map((b) => {
        const x0 = xAt(Math.max(b.start, start))
        const x1 = xAt(Math.min(b.end, end))
        const levels = FP4_LEVELS.flatMap((g) => [g * b.scale, -g * b.scale]).filter((v) => Math.abs(v) <= yMax)
        return (
          <g key={b.start}>
            {levels.map((v) => (
              <line key={v} x1={x0 + 2} x2={x1 - 2} y1={sy(v)} y2={sy(v)} stroke="var(--ink-3)" strokeOpacity={0.45} strokeDasharray="2 3" />
            ))}
            {b.start > start && <line x1={x0} x2={x0} y1={TOP - 22} y2={H - BOTTOM} stroke="var(--ink-3)" strokeDasharray="4 3" />}
            <text x={(x0 + x1) / 2} y={14} textAnchor="middle" fontSize={11.5} fill="var(--ink-2)" fontFamily={FONT}>
              {`масштаб ${num(b.scale, 4)}`}
            </text>
          </g>
        )
      })}
      {/* веса */}
      {weights.slice(start, end).map((w, k) => {
        const i = start + k
        const q = res.q[i]
        const x = sx(i)
        const zeroed = w !== 0 && q === 0
        return (
          <g key={i}>
            <rect
              x={x - plotW / count / 2}
              y={TOP - 4}
              width={plotW / count}
              height={plotH + 8}
              fill={hover === i ? 'var(--surface-3)' : 'transparent'}
              onPointerEnter={() => onHover(i)}
              onPointerDown={() => onHover(i)}
            />
            <line x1={x} x2={x} y1={cy} y2={sy(w)} stroke="var(--ink-3)" strokeWidth={3} strokeLinecap="round" opacity={0.45} pointerEvents="none" />
            {clipped(w) && (
              <g pointerEvents="none">
                <path d={`M ${x} ${sy(w) - 8} l -5 8 h 10 z`} fill="var(--ink-3)" transform={w < 0 ? `rotate(180 ${x} ${sy(w)})` : undefined} />
                <text x={x + 8} y={sy(w) + 2} fontSize={11} fill="var(--ink-2)" fontFamily={FONT}>
                  {num(w, 1)} → {num(q, 2)}
                </text>
              </g>
            )}
            {zeroed ? (
              <circle cx={x} cy={cy} r={3.5} fill="var(--surface)" stroke="var(--bad)" strokeWidth={1.6} pointerEvents="none" />
            ) : (
              !clipped(q) && <circle cx={x} cy={sy(q)} r={3.2} fill="var(--ink)" pointerEvents="none" />
            )}
          </g>
        )
      })}
      <text x={PADL} y={H - 6} fontSize={11} fill="var(--ink-3)" fontFamily={FONT}>
        номер веса {start + 1}…{end}
      </text>
    </svg>
  )
}
