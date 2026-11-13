import { useMemo, useState } from 'react'
import {
  DEFAULT_MBU,
  DEFAULT_MFU,
  STEP_OVERHEAD_MS,
  decodeBatchSweep,
  decodeLowerBoundMs,
  kvBytesPerToken,
  maxConcurrentSeqs,
  stepWeightBytes,
  type PerfOptions,
} from '@/engine/perf'
import { Figure, Formula, LineChart, Segmented, Slider, Stat, StatusIcon, fmt, type LineAnnotation } from '@/ui'
import { niceScale } from '@/ui/chart-utils'
import { DTYPE_LABEL, DtypeSegmented, GpuSelect, ModelSelect, TpNote, VLLM_MEM_UTIL, type HardwareState, valueUnit } from '@/chapters/prefill-decode/shared/hardware'
import { BoundTag } from '@/chapters/prefill-decode/shared/Bound'
import { texBytes, texMs, texNum } from '@/chapters/prefill-decode/shared/tex'

const BATCHES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
/** Лог-шкала по batch: каждое удвоение одинаковой ширины, диапазон 1-64 занимает две трети графика. */
const X_TICKS = [1, 4, 16, 64, 512]
const SEQ_FORMS = ['последовательность', 'последовательности', 'последовательностей'] as const

type Mode = 'ideal' | 'real'

/**
 * Линейная ось Y от нуля с «красивым» верхом не ниже 1,25 × максимума данных: подписи отметок и серой зоны
 * стоят в верхних 30 px области построения (до двух строк), и линия с маркерами не должна их зачёркивать.
 */
function headroomScale(max: number): { domain: [number, number]; ticks: number[] } {
  const s = niceScale(0, Math.max(1e-9, max) * 1.25, 5)
  return { domain: s.domain, ticks: s.ticks }
}

/**
 * Калькулятор decode: шаг ≈ (веса + KV всех последовательностей) / полоса, пока не упрёмся в FLOPs.
 * Два отдельных графика (общий throughput и скорость на пользователя), не двойная ось.
 */
export function DecodeCalculator({ state }: { state: HardwareState }) {
  const hw = state.hw
  const { model, gpu, dtype, tp } = hw
  const [ctx, setCtx] = useState(1024)
  const [batch, setBatch] = useState(16)
  const [mode, setMode] = useState<Mode>('ideal')

  const r = useMemo(() => {
    const opts: PerfOptions = mode === 'ideal' ? { ...hw.perf, mbu: 1, mfu: 1, overheadMs: 0 } : { ...hw.perf }
    const sweep = decodeBatchSweep(model, gpu, BATCHES, { ...opts, avgContext: ctx, memUtil: VLLM_MEM_UTIL })
    const cur = decodeBatchSweep(model, gpu, [batch], { ...opts, avgContext: ctx, memUtil: VLLM_MEM_UTIL })[0]
    const maxSeqs = maxConcurrentSeqs(model, gpu, { contextLen: ctx, dtype, gpus: tp, memUtil: VLLM_MEM_UTIL })
    const lb = decodeLowerBoundMs(model, gpu, dtype, tp)
    const wBytes = stepWeightBytes(model, dtype, 1)
    const firstCompute = sweep.find((p) => p.bound === 'compute')?.batch ?? null
    // для подписи: сколько занимает KV одной длинной последовательности и сколько таких влезает
    const kv16k = kvBytesPerToken(model, 'bf16') * 16000
    const seqs16k = maxConcurrentSeqs(model, gpu, { contextLen: 16000, dtype, gpus: tp, memUtil: VLLM_MEM_UTIL })
    return { sweep, cur, maxSeqs, lb, wBytes, firstCompute, kv16k, seqs16k }
  }, [model, gpu, dtype, tp, ctx, batch, mode, hw.perf])

  const xMax = BATCHES[BATCHES.length - 1]
  const ann: LineAnnotation[] = [{ kind: 'x', x: batch, label: `batch ${fmt.int(batch)}` }]
  if (r.maxSeqs < xMax) ann.push({ kind: 'band', x: Math.max(1, r.maxSeqs), x2: xMax, label: 'не влезает' })
  else if (r.firstCompute !== null) ann.push({ kind: 'band', x: r.firstCompute, x2: xMax, label: 'упор в вычисления' })

  const totalPts = r.sweep.map((p) => [p.batch, p.totalTokS] as [number, number])
  const userPts = r.sweep.map((p) => [p.batch, p.perUserTokS] as [number, number])
  // ось Y с запасом сверху: полка кривой не должна доходить до строки подписей «batch …» и серой зоны
  const totalY = headroomScale(Math.max(...totalPts.map((p) => p[1])))
  const userY = headroomScale(Math.max(...userPts.map((p) => p[1])))
  const fits = batch <= r.maxSeqs
  const bwGBs = gpu.bwTBs * 1000 * tp
  const seqs = (n: number) => fmt.count(n, SEQ_FORMS)

  return (
    <Figure
      id="roofline-figure-decode"
      title="Калькулятор decode: скорость от размера batch"
      width="wide"
      controls={
        <div className="col-span-full grid gap-x-6 gap-y-4 sm:grid-cols-3">
          <ModelSelect state={state} />
          <GpuSelect state={state} />
          <DtypeSegmented state={state} />
        </div>
      }
      caption={
        <>
          Попробуйте: поднимите batch от 1 до 64 и сравните графики: общий throughput растёт в десятки раз, а скорость каждого пользователя падает
          гораздо меньше. Потом увеличьте контекст до 16 000 токенов: KV одной последовательности занимает {fmt.bytes(r.kv16k)}, и в память влезают
          только {r.seqs16k > 0 ? `около ${seqs(r.seqs16k)}` : 'считаные последовательности'} (серая зона); ещё до неё чтение KV съедает полосу, и рост
          throughput почти кончается.
        </>
      }
      source={`«Идеал»: 100 % полосы и пика, без накладных. «Реалистично»: ${fmt.pct(DEFAULT_MBU)} полосы, ${fmt.pct(DEFAULT_MFU)} пика и ${fmt.ms(STEP_OVERHEAD_MS)} накладных на шаг. Память под KV: gpu_memory_utilization ${fmt.num(VLLM_MEM_UTIL, 2)}, как в vLLM по умолчанию.`}
    >
      <div className="grid gap-5">
        <div className="overflow-x-auto">
          <Formula
            block
            tex={`t_{\\text{шаг}} \\ge \\frac{\\text{веса}}{\\text{полоса}} = \\frac{${texBytes(r.wBytes)}}{${texNum(bwGBs / 1000)}\\ \\text{ТБ/с}} = ${texMs(r.lb)} \\;\\Rightarrow\\; \\le ${texNum(1000 / r.lb, 0)}\\ \\text{ток/с на пользователя}`}
            narrowTex={`\\begin{aligned} t_{\\text{шаг}} &\\ge \\frac{\\text{веса}}{\\text{полоса}} = \\frac{${texBytes(r.wBytes)}}{${texNum(bwGBs / 1000)}\\ \\text{ТБ/с}} \\\\ &= ${texMs(r.lb)} \\\\ &\\Rightarrow\\; \\le ${texNum(1000 / r.lb, 0)}\\ \\text{ток/с на пользователя} \\end{aligned}`}
            label={`${model.name} в ${DTYPE_LABEL[dtype]} на ${tp > 1 ? `${tp} × ` : ''}${gpu.name}: batch 1, короткий контекст, 100 % полосы`}
          />
        </div>

        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
          <Stat
            label="Шаг decode (ITL)"
            value={valueUnit(fmt.ms(r.cur.stepMs))[0]}
            unit={valueUnit(fmt.ms(r.cur.stepMs))[1]}
            hint={<BoundTag bound={r.cur.bound} />}
          />
          <Stat label="На пользователя" value={fmt.int(r.cur.perUserTokS)} unit="ток/с" hint="1000 / шаг" />
          <Stat label="Всего" value={fmt.int(r.cur.totalTokS)} unit="ток/с" hint="batch × 1000 / шаг" />
          <Stat
            label="KV всего batch"
            value={valueUnit(fmt.bytes(r.cur.kvBytes))[0]}
            unit={valueUnit(fmt.bytes(r.cur.kvBytes))[1]}
            hint={r.maxSeqs > 0 ? `влезает ${fmt.count(r.maxSeqs, ['запрос', 'запроса', 'запросов'])}` : 'под KV не остаётся памяти'}
          />
        </div>
        {!fits && (
          <p className="flex items-start gap-2 text-[14px] leading-snug text-ink-2">
            <StatusIcon status="warn" size={18} className="mt-px" />
            <span>
              <span className="font-medium text-ink">Не влезает в память:</span> при контексте {fmt.tokens(ctx)} в KV помещается максимум {seqs(r.maxSeqs)}. Такой batch
              движок не соберёт: лишние запросы будут ждать в очереди, а числа выше показывают скорость, которой на этом GPU не бывает.
            </span>
          </p>
        )}

        {/* главный эксперимент: слайдеры рядом с графиками, которые они двигают */}
        <div className="grid gap-x-6 gap-y-4 border-y border-line py-4 sm:grid-cols-2 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
          <Slider label="Batch" value={batch} min={1} max={512} scale="log" onChange={setBatch} format={(v) => fmt.int(v)} hint="последовательностей в одном шаге" />
          <Slider label="Контекст на запрос" value={ctx} min={64} max={32768} scale="log" onChange={setCtx} format={(v) => fmt.tokens(v)} hint="токенов в KV каждой последовательности" />
          <Segmented<Mode>
            label="Расчёт"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'ideal', label: 'Идеал (100 % полосы)' },
              { value: 'real', label: 'Реалистично' },
            ]}
          />
        </div>

        <div className="grid gap-6 @3xl:grid-cols-2">
          <div className="min-w-0">
            <div className="mb-1 text-[14px] font-medium text-ink">Общий throughput</div>
            <LineChart
              series={[{ id: 'total', label: 'Всего, ток/с', color: 'decode', points: totalPts, markers: true }]}
              xLabel="Batch, логарифмическая шкала"
              yLabel="Токенов в секунду, всего"
              xScale="log"
              xDomain={[1, xMax]}
              xTicks={X_TICKS}
              yDomain={totalY.domain}
              yTicks={totalY.ticks}
              annotations={ann}
              height={210}
              xFormat={(v) => fmt.int(v)}
              yFormat={(v) => fmt.compact(v)}
              aria-label="Общий throughput decode в зависимости от batch"
            />
          </div>
          <div className="min-w-0">
            <div className="mb-1 text-[14px] font-medium text-ink">Скорость на пользователя</div>
            <LineChart
              series={[{ id: 'user', label: 'На пользователя, ток/с', color: 'decode', points: userPts, markers: true }]}
              xLabel="Batch, логарифмическая шкала"
              yLabel="Токенов в секунду на пользователя"
              xScale="log"
              xDomain={[1, xMax]}
              xTicks={X_TICKS}
              yDomain={userY.domain}
              yTicks={userY.ticks}
              annotations={ann}
              height={210}
              xFormat={(v) => fmt.int(v)}
              yFormat={(v) => fmt.compact(v)}
              aria-label="Скорость decode на одного пользователя в зависимости от batch"
            />
          </div>
        </div>
        <TpNote hw={hw} />
      </div>
    </Figure>
  )
}
