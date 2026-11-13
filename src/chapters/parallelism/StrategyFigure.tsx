import { useMemo, useState, type ReactNode } from 'react'
import { Badge, FieldHint, Figure, Segmented, Select, fmt, measureText, truncateText, useElementSize, useFontsReady } from '@/ui'
import { useSettings } from '@/state/settings'
import { findModel, getModel } from '@/engine/data/models'
import { findGpu, getGpu } from '@/engine/data/gpus'
import type { DType } from '@/engine/types'
import { LegendItem, Swatch, useSvgId } from '../batching/chart'
import { GPUS_PER_NODE, IB_PER_GPU_GBS, LAUNCH_MS, plan, type LinkKind, type PlanResult, type Strategy } from './plan'

export const STRAT_MODELS = ['llama-3.1-70b', 'qwen3-235b-a22b', 'deepseek-v3', 'qwen3-32b', 'gpt-oss-120b', 'qwen3-30b-a3b', 'llama-3.1-405b', 'qwen3-8b'] as const
export const STRAT_GPUS = ['h100-sxm', 'h200', 'b200', 'a100-80g'] as const
const COUNTS = [1, 2, 4, 8, 16] as const
/** Длина последовательности в фигуре (текст главы считает с ней же). */
export const STRAT_CONTEXT = 2048
const CONTEXT = STRAT_CONTEXT
/** Гибрид TP × PP доступен, когда GPU больше, чем в одной ноде. */
const HYBRID_GPUS = 16

const STRAT_LABEL: Record<Strategy, string> = { tp: 'TP', pp: 'PP', ep: 'EP', dp: 'DP', tppp: 'TP8 × PP2' }
const LINK_NAME: Record<LinkKind, string> = { nvlink: 'NVLink', pcie: 'PCIe', ib: 'InfiniBand' }

export function StrategyFigure() {
  const [settings] = useSettings()
  const [strategy, setStrategy] = useState<Strategy>('tp')
  const [count, setCount] = useState(8)
  const [modelId, setModelId] = useState<string>('llama-3.1-70b')
  const [gpuId, setGpuId] = useState<string>((STRAT_GPUS as readonly string[]).includes(settings.gpuId) ? settings.gpuId : 'h100-sxm')
  const [dtype, setDtype] = useState<DType>('fp8')
  const [load, setLoad] = useState(64)
  const model = findModel(modelId) ?? getModel('llama-3.1-70b')
  const gpu = findGpu(gpuId) ?? getGpu('h100-sxm')
  const isMoe = Boolean(model.experts)
  const hybridOk = count === HYBRID_GPUS
  // Недоступный вариант откатывается к TP, как и FP8 на GPU без FP8 в тензорных ядрах откатывается к BF16.
  const strat: Strategy = (strategy === 'ep' && !isMoe) || (strategy === 'tppp' && !hybridOk) ? 'tp' : strategy
  const noFp8 = !gpu.tflops.fp8
  const dt: DType = dtype === 'fp8' && noFp8 ? 'bf16' : dtype

  const p = useMemo(() => plan({ model, gpu, dtype: dt, strategy: strat, gpus: count, concurrency: load, context: CONTEXT }), [model, gpu, dt, strat, count, load])
  const single = useMemo(() => plan({ model, gpu, dtype: dt, strategy: 'tp', gpus: 1, concurrency: load, context: CONTEXT }), [model, gpu, dt, load])
  // для 16 GPU сравниваем TP16 и TP8 × PP2 друг с другом
  const alt = useMemo(
    () => (count === HYBRID_GPUS && (strat === 'tp' || strat === 'tppp') ? plan({ model, gpu, dtype: dt, strategy: strat === 'tp' ? 'tppp' : 'tp', gpus: count, concurrency: load, context: CONTEXT }) : null),
    [model, gpu, dt, strat, count, load],
  )
  const stratHints: string[] = []
  if (!isMoe) stratHints.push('EP только для MoE-моделей.')
  if (!hybridOk) stratHints.push('TP8 × PP2 нужны две ноды: выберите 16 GPU.')

  return (
    <Figure
      id="parallelism-figure-strategy"
      width="full"
      title="Как модель раскладывается по GPU"
      controls={
        <>
          <div className="grid content-start gap-1.5 @min-[640px]/fig:col-span-2">
            <Segmented<Strategy>
              label="Стратегия"
              value={strat}
              onChange={setStrategy}
              wrap
              options={(['tp', 'pp', 'ep', 'dp', 'tppp'] as Strategy[]).map((s) => ({
                value: s,
                label: STRAT_LABEL[s],
                disabled: (s === 'ep' && !isMoe) || (s === 'tppp' && !hybridOk),
              }))}
            />
            {stratHints.length > 0 && <FieldHint>{stratHints.join(' ')}</FieldHint>}
          </div>
          <Segmented<number> label="Число GPU" value={count} onChange={setCount} options={COUNTS.map((c) => ({ value: c, label: String(c) }))} />
          <Select
            label="Модель"
            value={modelId}
            onChange={setModelId}
            options={STRAT_MODELS.map((id) => ({ value: id, label: getModel(id).name, group: getModel(id).experts ? 'MoE' : 'Плотные' }))}
          />
          <Select label="Модель GPU" value={gpuId} onChange={setGpuId} options={STRAT_GPUS.map((id) => ({ value: id, label: getGpu(id).name }))} />
          <div className="grid content-start gap-1.5">
            <Segmented<DType>
              label="Веса"
              value={dt}
              onChange={setDtype}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp8', label: 'FP8', disabled: noFp8 },
              ]}
            />
            {noFp8 && <FieldHint>У {gpu.name} нет FP8 в тензорных ядрах: считаем BF16.</FieldHint>}
          </div>
          <Segmented<number>
            label="Запросов одновременно"
            value={load}
            onChange={setLoad}
            options={[1, 16, 64, 256].map((v) => ({ value: v, label: String(v) }))}
          />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> Llama-3.1-70B, 8 GPU. При одном запросе сравните паузу между токенами у TP и
          PP: у PP токен проходит стадии по очереди, и он в разы медленнее. При 256 запросах PP выходит на полную загрузку, но TP всё равно выдаёт
          больше: каждая стадия PP перечитывает свои веса для каждой группы запросов (microbatch). Поставьте 16 GPU: TP16 гонит {fmt.int(2 * model.layers)} all-reduce на
          каждый шаг через InfiniBand, а TP8 × PP2 держит их внутри ноды и передаёт между нодами только активации. Для MoE-модели выберите EP.
          Упрощённая модель: roofline с MBU 70 % и MFU 50 %; у каждого обмена условная задержка запуска ({fmt.ms(LAUNCH_MS.nvlink)} по NVLink,{' '}
          {fmt.ms(LAUNCH_MS.ib)} по InfiniBand) плюс время по полосе канала; длина последовательностей {fmt.tokens(CONTEXT)}.
        </>
      }
    >
      <Diagram model={model.name} layers={model.layers} experts={model.experts?.total} strategy={strat} n={count} p={p} gpuName={gpu.name} />
      <Readout p={p} single={single} strategy={strat} n={count} load={load} gpuMem={gpu.memGB} />
      <p className="mt-3 max-w-[80ch] text-[15px] leading-relaxed text-ink">{explain(strat, count, p, single, alt, model.layers, model.experts?.total)}</p>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------
 * Схема
 * ------------------------------------------------------------------------- */

const CARD_H = 152
const ROWS = 16
/** Отступ рамки ноды от карточек по горизонтали и вертикали. */
const FRAME_PAD = 14
/** Полоса под рядом для кольца all-reduce (внутри рамки ноды). */
const RING_BAND = 30
/** Промежуток между нодами (там подпись InfiniBand). */
const NODE_GAP = 52

interface Seg {
  d: string
  dashed?: boolean
}

function Diagram({
  model,
  layers,
  experts,
  strategy,
  n,
  p,
  gpuName,
}: {
  model: string
  layers: number
  experts?: number
  strategy: Strategy
  n: number
  p: PlanResult
  gpuName: string
}) {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  useFontsReady() // подписи карточек выбираются по ширине текста: перерисовать, когда загрузится шрифт
  const arrow = useSvgId('par-arrow')
  const narrow = width > 0 && width < 600
  const perRow = narrow ? 4 : GPUS_PER_NODE
  const W = narrow ? Math.max(300, width) : Math.max(680, width || 1100)
  const gap = narrow ? 10 : 14
  const cols = Math.min(n, perRow)
  const cw = Math.min(124, (W - 2 * (FRAME_PAD + 4) - (cols - 1) * gap) / cols)
  const rowW = cols * cw + (cols - 1) * gap
  const x0 = (W - rowW) / 2
  const nodes = Math.ceil(n / GPUS_PER_NODE)
  const rowsPerNode = Math.ceil(Math.min(n, GPUS_PER_NODE) / perRow)
  const multi = n > 1
  const ring = multi && (strategy === 'tp' || strategy === 'tppp')
  const ep = multi && strategy === 'ep'
  const dp = multi && strategy === 'dp'
  const pp = multi && (strategy === 'pp' || strategy === 'tppp')
  const above = ep ? (narrow ? 32 : 42) : dp ? 30 : 0
  const rowGap = narrow ? 20 : 26
  const top = ep ? 26 : dp ? 36 : 14
  const small = narrow ? '11px "Onest Variable", sans-serif' : '12px "Onest Variable", sans-serif'

  // вертикальная раскладка: ноды, в каждой ряды карточек
  const rowY: number[] = []
  const rowNode: number[] = []
  const frames: Array<{ y0: number; y1: number }> = []
  let yy = top
  for (let k = 0; k < nodes; k++) {
    const y0 = yy
    yy += FRAME_PAD
    for (let r = 0; r < rowsPerNode; r++) {
      yy += above
      rowY.push(yy)
      rowNode.push(k)
      yy += CARD_H + (ring ? RING_BAND : 0)
      if (r < rowsPerNode - 1) yy += rowGap
    }
    yy += FRAME_PAD - (ring ? 6 : 0)
    frames.push({ y0, y1: yy })
    if (k < nodes - 1) yy += NODE_GAP
  }
  const bottomLabel = ring || pp
  const H = yy + (bottomLabel ? 30 : 8)

  const pos = (i: number) => {
    const node = Math.floor(i / GPUS_PER_NODE)
    const j = i % GPUS_PER_NODE
    const r = Math.floor(j / perRow)
    const c = j % perRow
    const row = node * rowsPerNode + r
    return { x: x0 + c * (cw + gap), y: rowY[row], node, row }
  }
  const cx = (i: number) => pos(i).x + cw / 2
  const rowsList: number[][] = []
  for (let i = 0; i < n; i++) {
    const r = pos(i).row
    if (!rowsList[r]) rowsList[r] = []
    rowsList[r].push(i)
  }
  const ok = p.weightsFit
  const memFrac = Math.min(1, p.weightsPerGpu / p.usablePerGpu)
  // подписи карточек: полные, если влезают все, иначе короткие у всех (ширина по шрифту, а не по ширине экрана)
  const labelFont = `${narrow ? 10.5 : 11.5}px "Onest Variable", sans-serif`
  const labelW = cw - 2 * (narrow ? 6 : 8)
  const allLabels = Array.from({ length: n }, (_, i) => cardLabels(strategy, i, n, layers, experts))
  const variant = allLabels.every((l) => measureText(l[0], labelFont) <= labelW) ? 0 : 1
  const cardLabel = (i: number) => {
    const l = allLabels[i][variant]
    return measureText(l, labelFont) <= labelW ? l : truncateText(l, labelW, labelFont)
  }
  const xL = x0 - 5
  const xLL = x0 - 10
  const xR = x0 + rowW + 5
  const gapMid = (k: number) => (frames[k].y1 + frames[k + 1].y0) / 2

  const lineStyle = (dashed?: boolean) => ({ fill: 'none', stroke: 'var(--ink-2)', strokeWidth: 1.6, strokeDasharray: dashed ? '5 4' : undefined })
  const comm: ReactNode[] = []

  /** Кольцо all-reduce через ряды rows (змейкой, замыкается слева); пунктир там, где оно переходит в другую ноду. */
  const drawRing = (rows: number[], key: string) => {
    const segs: Seg[] = []
    const ly = (r: number) => rowY[r] + CARD_H + 14
    const first = (r: number) => cx(rowsList[r][0])
    const last = (r: number) => cx(rowsList[r][rowsList[r].length - 1])
    if (rows.length === 1) {
      const r = rows[0]
      segs.push({ d: `M${first(r)},${ly(r)} H${last(r)} v10 H${first(r)} z` })
    } else {
      rows.forEach((r, j) => {
        segs.push({ d: `M${first(r)},${ly(r)} H${last(r)}` })
        if (j < rows.length - 1) {
          const r2 = rows[j + 1]
          const right = j % 2 === 0
          const xe = right ? xR : xL
          const a = right ? last(r) : first(r)
          const b = right ? last(r2) : first(r2)
          segs.push({ d: `M${a},${ly(r)} H${xe} V${ly(r2)} H${b}`, dashed: rowNode[r] !== rowNode[r2] })
        }
      })
      const rl = rows[rows.length - 1]
      const r0 = rows[0]
      const xc = rows.length > 2 ? xLL : xL
      segs.push({ d: `M${first(rl)},${ly(rl)} H${xc} V${ly(r0)} H${first(r0)}`, dashed: rowNode[rl] !== rowNode[r0] })
    }
    segs.forEach((s, k) => comm.push(<path key={`${key}-${k}`} d={s.d} style={lineStyle(s.dashed)} />))
    for (const r of rows) for (const i of rowsList[r]) comm.push(<circle key={`${key}-d${i}`} cx={cx(i)} cy={ly(r)} r={3.5} style={{ fill: 'var(--ink-2)' }} />)
  }

  /** Стрелка от карточки a к карточке b: в одном ряду напрямую, между рядами через промежуток под рядом a. */
  const stageArrow = (a: number, b: number, key: string) => {
    const A = pos(a)
    const B = pos(b)
    if (A.row === B.row) {
      const y = A.y + CARD_H / 2
      comm.push(<line key={key} x1={A.x + cw + 2} x2={B.x - 3} y1={y} y2={y} markerEnd={`url(#${arrow})`} style={{ stroke: 'var(--ink-2)', strokeWidth: 1.6 }} />)
      return
    }
    const y1 = A.y + CARD_H + (ring ? RING_BAND : 0) + 2
    const y2 = B.y - 4
    const mid = A.node !== B.node ? gapMid(A.node) : (A.y + CARD_H + (ring ? RING_BAND : 0) + B.y) / 2
    comm.push(
      <path key={key} d={`M${cx(a)},${y1} V${mid} H${cx(b)} V${y2}`} markerEnd={`url(#${arrow})`} style={{ fill: 'none', stroke: 'var(--ink-2)', strokeWidth: 1.6, strokeDasharray: A.node !== B.node ? '5 4' : undefined }} />,
    )
  }

  if (ring && strategy === 'tp') drawRing(rowsList.map((_, r) => r), 'ring')
  if (ring && strategy === 'tppp')
    for (let k = 0; k < nodes; k++)
      drawRing(
        rowsList.map((_, r) => r).filter((r) => rowNode[r] === k),
        `ring${k}`,
      )
  if (strategy === 'pp' && multi) for (let i = 0; i < n - 1; i++) stageArrow(i, i + 1, `pp${i}`)
  if (strategy === 'tppp') for (let k = 0; k < nodes - 1; k++) stageArrow((k + 1) * GPUS_PER_NODE - 1, (k + 1) * GPUS_PER_NODE, `hop${k}`)

  if (ep) {
    // all-to-all внутри каждого ряда: дуги над карточками
    rowsList.forEach((row, r) => {
      const y = rowY[r] - 2
      for (let a = 0; a < row.length; a++)
        for (let b = a + 1; b < row.length; b++) {
          const lift = 8 + ((b - a) / Math.max(1, row.length - 1)) * (above - 16)
          const xa = cx(row[a])
          const xb = cx(row[b])
          comm.push(<path key={`e${r}-${a}-${b}`} d={`M${xa},${y} Q${(xa + xb) / 2},${y - lift} ${xb},${y}`} style={{ fill: 'none', stroke: 'var(--ink-3)', strokeWidth: 1, opacity: 0.6 }} />)
        }
    })
    // и между рядами: пунктир там, где обмен идёт через InfiniBand
    for (let r = 0; r < rowsList.length - 1; r++) {
      const A = rowsList[r]
      const B = rowsList[r + 1]
      const pairs: Array<[number, number]> = [
        [Math.round((A.length - 1) * 0.2), Math.round((B.length - 1) * 0.75)],
        [Math.round((A.length - 1) * 0.8), Math.round((B.length - 1) * 0.3)],
      ]
      if (A.length > 4) pairs.push([Math.round((A.length - 1) * 0.45), Math.round((B.length - 1) * 0.55)])
      const cross = rowNode[r] !== rowNode[r + 1]
      pairs.forEach(([ia, ib], k) => {
        const xa = cx(A[ia])
        const xb = cx(B[ib])
        const ya = rowY[r] + CARD_H + 2
        const yb = rowY[r + 1] - 2
        const bend = (yb - ya) * 0.5
        comm.push(
          <path
            key={`ex${r}-${k}`}
            d={`M${xa},${ya} C${xa},${ya + bend} ${xb},${yb - bend} ${xb},${yb}`}
            style={{ fill: 'none', stroke: cross ? 'var(--ink-2)' : 'var(--ink-3)', strokeWidth: cross ? 1.4 : 1, opacity: cross ? 0.85 : 0.6, strokeDasharray: cross ? '5 4' : undefined }}
          />,
        )
      })
    }
    const text = narrow
      ? `all-to-all: ${fmt.int(2 * layers)} обменов на шаг`
      : `all-to-all: токены едут к GPU со своими экспертами и обратно, ${fmt.int(2 * layers)} обменов на шаг${nodes > 1 ? ', часть через InfiniBand (пунктир)' : ''}`
    comm.push(
      <text key="ept" x={W / 2} y={16} textAnchor="middle" style={{ font: small, fill: 'var(--ink-2)' }}>
        {text}
      </text>,
    )
  }

  if (dp) {
    // балансировщик: шина над каждым рядом и стрелка в каждую карточку
    const bw = narrow ? 200 : 230
    const busY = (r: number) => rowY[r] - 17
    comm.push(
      <line key="trunk0" x1={W / 2} x2={W / 2} y1={30} y2={busY(0)} style={{ stroke: 'var(--ink-3)', strokeWidth: 1.2 }} />,
      <rect key="lb" x={W / 2 - bw / 2} y={4} width={bw} height={26} rx={8} style={{ fill: 'var(--surface-2)', stroke: 'var(--line-strong)' }} />,
      <text key="lbt" x={W / 2} y={17} textAnchor="middle" dominantBaseline="central" style={{ font: `500 ${small}`, fill: 'var(--ink)' }}>
        балансировщик раздаёт запросы
      </text>,
    )
    if (rowsList.length > 1) comm.push(<line key="trunk" x1={xL} x2={xL} y1={busY(0)} y2={busY(rowsList.length - 1)} style={{ stroke: 'var(--ink-3)', strokeWidth: 1.2 }} />)
    rowsList.forEach((row, r) => {
      const xa = rowsList.length > 1 ? xL : cx(row[0])
      comm.push(<line key={`bus${r}`} x1={xa} x2={cx(row[row.length - 1])} y1={busY(r)} y2={busY(r)} style={{ stroke: 'var(--ink-3)', strokeWidth: 1.2 }} />)
      for (const i of row)
        comm.push(<line key={`lb${i}`} x1={cx(i)} x2={cx(i)} y1={busY(r)} y2={rowY[r] - 3} markerEnd={`url(#${arrow})`} style={{ stroke: 'var(--ink-3)', strokeWidth: 1.2 }} />)
    })
  }

  // подпись под схемой
  let bottom: string | null = null
  if (ring && strategy === 'tp')
    bottom = narrow
      ? `all-reduce: ${fmt.int(2 * layers)} обменов на шаг`
      : `all-reduce после attention и после MLP: ${fmt.int(2 * layers)} обменов на каждый шаг${nodes > 1 ? ', кольцо проходит через InfiniBand (пунктир)' : ''}`
  if (strategy === 'tppp')
    bottom = narrow ? 'all-reduce внутри нод, между нодами активации' : `внутри каждой ноды ${fmt.int(2 * layers)} all-reduce по NVLink, между нодами одна передача активаций по InfiniBand`
  if (strategy === 'pp' && multi)
    bottom = narrow ? 'между стадиями только активации' : `стадия передаёт следующей только активации токенов: ${n - 1} ${fmt.plural(n - 1, ['передача', 'передачи', 'передач'])} на шаг`

  return (
    <div ref={ref} className="min-w-0">
      <p className="mb-2 text-[13px] text-ink-2">
        {n === 1
          ? 'Один GPU: веса, KV cache и все вычисления в одной памяти.'
          : nodes > 1
            ? `Две ноды по ${GPUS_PER_NODE} GPU (пунктирные рамки): внутри ноды GPU связаны NVLink, между нодами InfiniBand, ${fmt.bandwidth(IB_PER_GPU_GBS)} на GPU.`
            : 'Одна нода (пунктирная рамка): GPU связаны NVLink.'}
      </p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        style={{ width: '100%', maxWidth: W, height: 'auto' }}
        role="img"
        aria-label={`${model} на ${n} × ${gpuName}, стратегия ${STRAT_LABEL[strategy]}: на каждом GPU ${p.holds || 'ничего'}`}
        className="block"
      >
        <defs>
          <marker id={arrow} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
          </marker>
        </defs>
        {frames.map((f, k) => (
          <rect key={`node${k}`} x={x0 - FRAME_PAD} y={f.y0} width={rowW + 2 * FRAME_PAD} height={f.y1 - f.y0} rx={12} style={{ fill: 'none', stroke: 'var(--line-strong)', strokeDasharray: '4 3' }} />
        ))}
        {frames.slice(0, -1).map((f, k) => (
          <line key={`ibl${k}`} x1={W / 2} x2={W / 2} y1={f.y1} y2={frames[k + 1].y0} style={{ stroke: 'var(--ink-3)', strokeWidth: 1.5, strokeDasharray: '4 3' }} />
        ))}
        {comm}
        {frames.slice(0, -1).map((_, k) => {
          const pw = narrow ? 150 : 172
          const mid = gapMid(k)
          return (
            <g key={`ib${k}`}>
              <rect x={W / 2 - pw / 2} y={mid - 11} width={pw} height={22} rx={11} style={{ fill: 'var(--surface-2)', stroke: 'var(--line-strong)' }} />
              <text x={W / 2} y={mid} textAnchor="middle" dominantBaseline="central" style={{ font: `600 ${small}`, fill: 'var(--ink)' }}>
                InfiniBand между нодами
              </text>
            </g>
          )
        })}
        {Array.from({ length: n }, (_, i) => (
          <Card
            key={i}
            i={i}
            n={n}
            x={pos(i).x}
            y={pos(i).y}
            w={cw}
            narrow={narrow}
            strategy={strategy}
            experts={experts}
            ok={ok}
            memFrac={memFrac}
            kvFrac={ok ? Math.max(0, 1 - memFrac) : 0}
            label={cardLabel(i)}
            labelFont={labelFont}
          />
        ))}
        {bottom && (
          <text x={W / 2} y={frames[frames.length - 1].y1 + 20} textAnchor="middle" style={{ font: small, fill: 'var(--ink-2)' }}>
            {bottom}
          </text>
        )}
      </svg>
      <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
        <LegendItem swatch={<Swatch color="var(--ink-2)" />}>лежит на этом GPU</LegendItem>
        {strategy === 'ep' && <LegendItem swatch={<Swatch color="color-mix(in oklab, var(--ink-3) 70%, var(--surface))" />}>копия на всех GPU</LegendItem>}
        {multi && strategy !== 'dp' && (
          <LegendItem swatch={<span className="inline-block h-3 w-4 rounded-[3px] bg-surface-3 shadow-[inset_0_0_0_1px_var(--line)]" />}>лежит на других GPU</LegendItem>
        )}
        <LegendItem swatch={<BarSwatch />}>полоска внизу: веса и место под KV cache</LegendItem>
      </ul>
    </div>
  )
}

function BarSwatch() {
  return (
    <span className="inline-flex h-2.5 w-6 overflow-hidden rounded-[2px] shadow-[inset_0_0_0_1px_var(--line)]">
      <span className="h-full w-1/2" style={{ background: 'var(--ink-2)' }} />
      <span className="flex h-full w-1/2 flex-col">
        <span className="flex-1" style={{ background: 'var(--c-k)' }} />
        <span className="flex-1" style={{ background: 'var(--c-v)' }} />
      </span>
    </span>
  )
}

function Card({
  i,
  n,
  x,
  y,
  w,
  narrow,
  strategy,
  experts,
  ok,
  memFrac,
  kvFrac,
  label,
  labelFont,
}: {
  i: number
  n: number
  x: number
  y: number
  w: number
  narrow: boolean
  strategy: Strategy
  experts?: number
  ok: boolean
  memFrac: number
  kvFrac: number
  label: string
  labelFont: string
}) {
  const pad = narrow ? 6 : 8
  const ix = x + pad
  const iw = w - 2 * pad
  const rowsTop = y + 26
  const held = 'var(--ink-2)'
  const empty = 'var(--surface-3)'
  const copy = 'color-mix(in oklab, var(--ink-3) 70%, var(--surface))'
  const rowsEls: ReactNode[] = []
  const attnRows = strategy === 'ep' ? 4 : ROWS
  const stages = strategy === 'tppp' ? Math.ceil(n / GPUS_PER_NODE) : n
  const tpGroup = strategy === 'tppp' ? Math.min(n, GPUS_PER_NODE) : n
  const stageOf = strategy === 'tppp' ? Math.floor(i / GPUS_PER_NODE) : i
  const rank = strategy === 'tppp' ? i % GPUS_PER_NODE : i
  for (let r = 0; r < attnRows; r++) {
    const ry = rowsTop + r * 5
    if (strategy === 'tp' || strategy === 'tppp') {
      const mine = strategy === 'tp' || Math.floor((r * stages) / ROWS) === stageOf
      rowsEls.push(<rect key={`b${r}`} x={ix} y={ry} width={iw} height={3.5} rx={1} style={{ fill: empty }} />)
      if (mine) rowsEls.push(<rect key={`s${r}`} x={ix + (rank * iw) / tpGroup} y={ry} width={Math.max(1.5, iw / tpGroup)} height={3.5} rx={1} style={{ fill: held }} />)
    } else if (strategy === 'pp') {
      const mine = Math.floor((r * n) / ROWS) === i
      rowsEls.push(<rect key={`b${r}`} x={ix} y={ry} width={iw} height={3.5} rx={1} style={{ fill: mine ? held : empty }} />)
    } else if (strategy === 'ep') {
      rowsEls.push(<rect key={`b${r}`} x={ix} y={ry} width={iw} height={3.5} rx={1} style={{ fill: copy }} />)
    } else {
      rowsEls.push(<rect key={`b${r}`} x={ix} y={ry} width={iw} height={3.5} rx={1} style={{ fill: held }} />)
    }
  }
  if (strategy === 'ep' && experts) {
    const ey = rowsTop + 4 * 5 + 4
    const colsE = experts > 128 ? 32 : 16
    const rowsE = Math.ceil(experts / colsE)
    const availH = y + CARD_H - 22 - 16 - ey
    const cell = Math.min(5, iw / colsE, availH / rowsE)
    const sz = Math.max(1.2, cell - 1)
    const per = experts / n
    for (let k = 0; k < experts; k++) {
      const mine = Math.floor(k / per) === i
      rowsEls.push(
        <rect key={`e${k}`} x={ix + (k % colsE) * cell} y={ey + Math.floor(k / colsE) * cell} width={sz} height={sz} rx={0.5} style={{ fill: mine ? held : empty }} />,
      )
    }
  }
  const barY = y + CARD_H - 22
  const kvX = ix + iw * memFrac
  return (
    <g>
      <rect x={x} y={y} width={w} height={CARD_H} rx={10} style={{ fill: 'var(--surface)', stroke: ok ? 'var(--line-strong)' : 'var(--bad)', strokeWidth: ok ? 1 : 1.5 }} />
      <text x={ix} y={y + 15} style={{ font: `600 ${narrow ? 11 : 12}px "Onest Variable", sans-serif`, fill: 'var(--ink)' }}>
        GPU {i + 1}
      </text>
      {rowsEls}
      <text x={ix} y={barY - 8} style={{ font: labelFont, fill: 'var(--ink-2)' }}>
        {label}
      </text>
      <rect x={ix} y={barY} width={iw} height={8} rx={2} style={{ fill: 'var(--surface-2)', stroke: 'var(--line)', strokeWidth: 1 }} />
      <rect x={ix} y={barY} width={iw * Math.min(1, memFrac)} height={8} rx={2} style={{ fill: ok ? held : 'var(--bad)' }} />
      {ok && kvFrac > 0 && (
        <>
          <rect x={kvX} y={barY} width={iw * kvFrac} height={4} style={{ fill: 'var(--c-k)' }} />
          <rect x={kvX} y={barY + 4} width={iw * kvFrac} height={4} style={{ fill: 'var(--c-v)' }} />
        </>
      )}
    </g>
  )
}

/** Подпись карточки: полная и короткая (если полные не влезают по ширине). */
function cardLabels(strategy: Strategy, i: number, n: number, layers: number, experts: number | undefined): [string, string] {
  const range = (a: number, b: number) => `${a}…${b}`
  if (n === 1 || strategy === 'dp') return ['вся модель', 'целиком']
  if (strategy === 'tp') return [`1/${n} каждого слоя`, `1/${n} слоя`]
  if (strategy === 'pp') {
    const r = range(Math.floor((i * layers) / n) + 1, Math.floor(((i + 1) * layers) / n))
    return [`слои ${r}`, r]
  }
  if (strategy === 'tppp') {
    const stages = Math.ceil(n / GPUS_PER_NODE)
    const st = Math.floor(i / GPUS_PER_NODE)
    const r = range(Math.floor((st * layers) / stages) + 1, Math.floor(((st + 1) * layers) / stages))
    return [`1/${Math.min(n, GPUS_PER_NODE)} слоёв ${r}`, r]
  }
  const e = fmtExperts((experts ?? 0) / n)
  return [`${e} экспертов`, `${e} эксп.`]
}

/** «2 раза», «8 раз». */
function times(n: number): string {
  return `${n} ${fmt.plural(n, ['раз', 'раза', 'раз'])}`
}

function fmtExperts(v: number): string {
  return Number.isInteger(v) ? String(v) : fmt.num(v, 1)
}

/* ---------------------------------------------------------------------------
 * Цифры и объяснение
 * ------------------------------------------------------------------------- */

function Readout({ p, single, strategy, n, load, gpuMem }: { p: PlanResult; single: PlanResult; strategy: Strategy; n: number; load: number; gpuMem: number }) {
  if (!p.feasible && !p.weightsFit && p.reason && !p.weightsPerGpu) {
    return (
      <div className="mt-3">
        <Badge tone="warn">{p.reason}</Badge>
      </div>
    )
  }
  const commHint =
    p.commMs > 0
      ? `из них связь ${fmt.ms(p.commMs)}${p.commIbMs > 0 && p.commIbMs < p.commMs * 0.98 ? `, по InfiniBand ${fmt.ms(p.commIbMs)}` : ''}`
      : 'связи между GPU нет'
  return (
    <div className="mt-3 grid grid-cols-2 gap-x-6 @md:grid-cols-3 @3xl:grid-cols-5">
      <Item label="Веса на GPU" hint={`из ${fmt.int(gpuMem)} ГБ памяти`}>
        {fmt.bytes(p.weightsPerGpu)}
      </Item>
      <Item label="Под KV на GPU" hint={p.weightsFit ? 'после весов и резерва' : undefined}>
        {p.weightsFit ? fmt.bytes(p.kvBudgetPerGpu) : <Badge tone="bad">веса не влезают</Badge>}
      </Item>
      <Item label="Влезет запросов" hint={`по ${fmt.tokens(CONTEXT)}${p.kvReplication > 1 ? `, KV ×${p.kvReplication}` : ''}`}>
        {p.weightsFit ? fmt.int(p.maxSeqs) : '0'}
      </Item>
      <Item label="Пауза между токенами" hint={p.weightsFit ? commHint : undefined}>
        {p.weightsFit ? fmt.ms(p.tokenMs) : 'н/д'}
      </Item>
      <Item
        label="Всего токенов в секунду"
        hint={p.weightsFit ? `при ${fmt.int(p.concurrencyUsed)} ${p.concurrencyUsed % 10 === 1 && p.concurrencyUsed % 100 !== 11 ? 'запросе' : 'запросах'}${n > 1 && single.weightsFit ? `; 1 GPU: ${fmt.rate(single.totalTokS)}` : ''}` : undefined}
      >
        {p.weightsFit ? fmt.int(p.totalTokS) : 'н/д'}
      </Item>
      {p.weightsFit && p.concurrencyUsed < load && (
        <div className="col-span-full pb-3">
          <Badge tone="warn">
            Под KV хватает только на {fmt.count(p.concurrencyUsed, ['последовательность', 'последовательности', 'последовательностей'])} из {fmt.int(load)}:
            остальные ждут в очереди.
          </Badge>
        </div>
      )}
      {(strategy === 'pp' || strategy === 'tppp') && n > 1 && p.weightsFit && (
        <div className="col-span-full text-sm text-ink-2">
          Скорость одного пользователя {fmt.rate(p.perUserTokS)}: токен проходит {p.stages === 2 ? 'обе стадии' : `все ${p.stages} ${fmt.plural(p.stages ?? n, ['стадию', 'стадии', 'стадий'])}`} по очереди.
        </div>
      )}
    </div>
  )
}

/** Ячейка сводки: подписи, значения и пояснения выровнены по рядам (subgrid), даже если подпись переносится. */
function Item({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="row-span-3 grid min-w-0 grid-rows-subgrid">
      <div className="self-end pb-0.5 text-sm leading-snug text-ink-2">{label}</div>
      <div className="text-[1.25rem] leading-tight font-semibold text-ink tabular-nums">{children}</div>
      <div className="pt-0.5 pb-3 text-[13px] leading-snug text-ink-3">{hint}</div>
    </div>
  )
}

function explain(s: Strategy, n: number, p: PlanResult, single: PlanResult, alt: PlanResult | null, layers: number, experts?: number): string {
  if (!p.weightsFit && p.reason && !p.weightsPerGpu) return p.reason
  if (n === 1) {
    return single.weightsFit
      ? 'Один GPU: вся модель и весь KV cache в его памяти, связи между GPU нет. Добавьте GPU, чтобы увидеть, как каждая стратегия делит работу.'
      : 'На одном GPU веса не помещаются. Нужно несколько GPU или формат весов поменьше.'
  }
  if (!p.weightsFit) {
    return s === 'dp'
      ? 'DP копирует модель целиком на каждый GPU, поэтому модель обязана помещаться в один GPU. Здесь она не помещается: сначала нужно разрезать её TP или PP, а уже потом размножать.'
      : 'Даже после деления веса не помещаются в память каждого GPU. Нужно больше GPU или квантизация.'
  }
  const linkName = p.commLink === 'none' ? 'NVLink' : LINK_NAME[p.commLink]
  if (s === 'tp') {
    const speed = single.weightsFit ? ` Пауза между токенами ${fmt.ms(p.tokenMs)} против ${fmt.ms(single.tokenMs)} на одном GPU.` : ''
    const launchShare = p.commMs > 0 ? p.commLaunchMs / p.commMs : 0
    const why = launchShare > 0.5 ? ', и это в основном задержка запуска каждого обмена, а не полоса' : ''
    const ib =
      p.commLink === 'ib'
        ? ` Через InfiniBand каждый обмен дороже, и связь съедает выигрыш от деления весов${alt?.weightsFit ? `: TP8 × PP2 на тех же 16 GPU даёт паузу ${fmt.ms(alt.tokenMs)}` : ''}. Поэтому TP держат внутри ноды.`
        : ''
    return `Каждый GPU хранит и читает 1/${n} весов каждого слоя, поэтому шаг decode короче (в идеале в ${times(n)}).${speed} Цена: после attention и после MLP каждого из ${layers} слоёв GPU складывают частичные результаты (all-reduce), это ${fmt.int(p.commOps)} обменов за шаг и ${fmt.ms(p.commMs)} по ${linkName}${why}.${ib}`
  }
  if (s === 'tppp') {
    const half = Math.floor(layers / 2)
    const vs = alt?.weightsFit ? ` Пауза между токенами ${fmt.ms(p.tokenMs)} против ${fmt.ms(alt.tokenMs)} у TP16, где все all-reduce идут через InfiniBand.` : ''
    return `Внутри каждой ноды работает TP${p.tpSize}: восемь GPU делят матрицы своих слоёв и складывают частичные результаты по NVLink. Между нодами PP${p.stages}: первая нода считает слои 1…${half}, вторая ${half + 1}…${layers}, и на каждый токен по InfiniBand едет одна пачка активаций (${fmt.ms(p.commIbMs)}) вместо ${fmt.int(2 * layers)} all-reduce.${vs} Так обычно запускают модель, которая не помещается в одну ноду.`
  }
  if (s === 'pp') {
    const idle =
      p.concurrencyUsed < n
        ? ` Сейчас запросов меньше, чем стадий: пока токен на одной стадии, остальные GPU ждут, и общий throughput не выше ${fmt.rate(p.totalTokS)}.`
        : ' При многих запросах стадии работают одновременно над разными группами запросов.'
    return `Модель режется по слоям на ${n} ${fmt.plural(n, ['стадию', 'стадии', 'стадий'])}. Чтобы выдать один токен, запрос проходит все стадии по очереди, поэтому один пользователь не ускоряется (пауза ${fmt.ms(p.tokenMs)}). Зато памяти в ${times(n)} больше.${idle} Между стадиями едут только активации, поэтому PP терпит медленную связь и годится для нескольких нод.`
  }
  if (s === 'ep') {
    return `Эксперты не режутся, а раздаются целиком: на каждом GPU ${fmtExperts((experts ?? 0) / n)} из ${experts}. Здесь attention и общие веса копируются на все GPU (DP-attention), и каждый GPU ведёт свои запросы; другой частый вариант режет attention по TP. Токен отправляется к GPU с нужными экспертами и возвращается (all-to-all), поэтому по скорости отдельного токена EP уступает TP, но в системе помещается больше запросов, и throughput растёт.`
  }
  return `Каждый GPU держит всю модель и обслуживает свою долю запросов, связи между GPU на каждом шаге нет. Throughput растёт почти в ${times(n)}, а пауза между токенами как у одного GPU с меньшим batch.`
}
