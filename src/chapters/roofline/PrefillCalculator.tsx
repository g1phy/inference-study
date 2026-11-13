import { useMemo, useState } from 'react'
import { DEFAULT_MBU, STEP_OVERHEAD_MS, peakTflops, prefillBreakdown, prefillFlopsBreakdown } from '@/engine/perf'
import { Figure, Formula, LineChart, Slider, Stat, fmt } from '@/ui'
import { DTYPE_LABEL, DtypeSegmented, GpuSelect, ModelSelect, TpNote, type HardwareState, valueUnit } from '@/chapters/prefill-decode/shared/hardware'
import { texMs, texNum, texSci } from '@/chapters/prefill-decode/shared/tex'
import { BoundTag } from '@/chapters/prefill-decode/shared/Bound'

/** Точки кривой «длина промпта → время», логарифмически от 16 до 131 072. */
const CURVE_N = Array.from({ length: 27 }, (_, i) => Math.round(16 * 2 ** (i / 2)))

/**
 * Калькулятор prefill: FLOPs ≈ 2·P·N (+ attention), время = FLOPs / (пик · MFU),
 * но не меньше чтения весов. Короткий промпт упирается в память, длинный в вычисления.
 */
export function PrefillCalculator({ state, userPrompt }: { state: HardwareState; userPrompt: number | null }) {
  const hw = state.hw
  const { model, gpu, dtype, tp } = hw
  const [n, setN] = useState(4096)
  const [mfu, setMfu] = useState(0.5)

  const r = useMemo(() => {
    const opts = { ...hw.perf, mfu }
    const fl = prefillFlopsBreakdown(model, n)
    const b = prefillBreakdown(model, gpu, n, opts)
    const peak = peakTflops(gpu, dtype) * tp
    const P = (model.activeMatmulParamsB ?? model.activeParamsB) * 1e9
    // LM head считает логиты только для последнего токена промпта (как в движках без prompt logprobs)
    const lm = model.lmHeadParamsB !== undefined ? model.lmHeadParamsB * 1e9 : model.vocab * model.hidden
    const curve = CURVE_N.map((k) => {
      const s = prefillBreakdown(model, gpu, k, opts)
      return { n: k, compute: s.computeMs, memory: s.memoryMs }
    })
    // граница: с какой длины вычисления дольше чтения весов
    let cross: number | null = null
    let crossMs = 0
    for (let k = 16; k <= 131072; k = Math.ceil(k * 1.05)) {
      const s = prefillBreakdown(model, gpu, k, opts)
      if (s.computeMs >= s.memoryMs) {
        cross = k
        crossMs = s.memoryMs
        break
      }
    }
    return { fl, b, peak, P, lm, curve, cross, crossMs }
  }, [model, gpu, dtype, tp, n, mfu, hw.perf])

  const attnShare = r.fl.attention / r.fl.total
  const tokPerSec = (n / r.b.ms) * 1000
  const computeWins = r.b.computeMs >= r.b.memoryMs
  // в max(…) цветом выделен член, который побеждает: вычисления (prefill) или чтение памяти (decode)
  const computeTerm = `\\frac{\\text{FLOPs}}{\\text{пик} \\cdot \\text{MFU}}`
  const memoryTerm = `\\frac{\\text{веса} + \\text{KV}}{\\text{полоса} \\cdot \\text{MBU}}`
  const timeMax = `\\max\\left(${computeWins ? `\\cprefill{${computeTerm}}` : computeTerm},\\ ${computeWins ? memoryTerm : `\\cdecode{${memoryTerm}}`}\\right)`
  const timeNum = `\\max(${texMs(r.b.computeMs)},\\ ${texMs(r.b.memoryMs)}) + ${texMs(r.b.overheadMs)}`
  const timeTex = `t \\approx ${timeMax} + \\text{накладные} = ${timeNum} = ${texMs(r.b.ms)}`
  // на телефоне та же формула по строкам: символы, числа, итог
  const timeTexNarrow = `\\begin{aligned} t &\\approx ${timeMax} \\\\ &\\quad + \\text{накладные} \\\\ &= ${timeNum} \\\\ &= ${texMs(r.b.ms)} \\end{aligned}`
  const flopsSym = `\\text{FLOPs} \\approx 2 \\cdot P \\cdot N + 2 \\cdot P_{\\text{LM}} + \\text{attention}`
  const flopsNum = `2 \\cdot ${texSci(r.P - r.lm)} \\cdot ${texNum(n)} + ${texSci(2 * r.lm)} + ${texSci(r.fl.attention)}`

  return (
    <Figure
      id="roofline-figure-prefill"
      title="Калькулятор prefill: сколько считается промпт"
      width="wide"
      controls={
        <>
          <div className="col-span-full grid gap-x-6 gap-y-4 sm:grid-cols-3">
            <ModelSelect state={state} />
            <GpuSelect state={state} />
            <DtypeSegmented state={state} />
          </div>
          <div className="col-span-full grid gap-x-6 gap-y-4 sm:grid-cols-3">
            <Slider
              label="Длина промпта"
              value={n}
              min={16}
              max={131072}
              scale="log"
              onChange={setN}
              format={(v) => fmt.tokens(v)}
              hint={
                userPrompt !== null ? (
                  <button type="button" className="text-accent underline decoration-dotted underline-offset-2" onClick={() => setN(userPrompt)}>
                    ваш запрос: {fmt.tokens(userPrompt)}
                  </button>
                ) : undefined
              }
            />
            <Slider label="MFU" value={mfu} min={0.2} max={0.9} step={0.05} onChange={setMfu} format={(v) => fmt.pct(v)} hint="доля пика, которую реально выжимают ядра" />
          </div>
        </>
      }
      caption={`Попробуйте: уменьшайте длину промпта и найдите место, где линия «Вычисления» опускается ниже линии «Чтение весов»: короче этой длины prefill упирается уже не в вычисления, а в чтение весов. Чтение весов считаем при ${fmt.pct(DEFAULT_MBU)} полосы, плюс ${fmt.ms(STEP_OVERHEAD_MS)} накладных на шаг.`}
    >
      <div className="grid gap-5">
        <div className="grid gap-2 overflow-x-auto">
          <Formula
            block
            tex={`${flopsSym} = ${flopsNum} \\approx ${texSci(r.fl.total)}`}
            narrowTex={`\\begin{aligned} \\text{FLOPs} &\\approx 2 \\cdot P \\cdot N + 2 \\cdot P_{\\text{LM}} \\\\ &\\quad + \\text{attention} \\\\ &= 2 \\cdot ${texSci(r.P - r.lm)} \\cdot ${texNum(n)} \\\\ &\\quad + ${texSci(2 * r.lm)} + ${texSci(r.fl.attention)} \\\\ &\\approx ${texSci(r.fl.total)} \\end{aligned}`}
            label={`P = ${fmt.num((r.P - r.lm) / 1e9)} млрд параметров умножаются на каждый токен промпта (без таблицы эмбеддингов и LM head); LM head (${fmt.num(r.lm / 1e9)} млрд) считает логиты только для последнего токена; attention растёт как N²`}
          />
          <Formula
            block
            tex={timeTex}
            narrowTex={timeTexNarrow}
            label={`пик ${tp > 1 ? `${tp} × ` : ''}${gpu.name} в ${DTYPE_LABEL[dtype]}: ${fmt.int(r.peak)} TFLOPS при MFU ${fmt.pct(mfu)}; полоса ${fmt.bandwidth(gpu.bwTBs * 1000 * tp)} при ${fmt.pct(DEFAULT_MBU)} (MBU); побеждает большее`}
          />
        </div>

        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
          <Stat
            label="Время prefill"
            value={valueUnit(fmt.ms(r.b.ms))[0]}
            unit={valueUnit(fmt.ms(r.b.ms))[1]}
            hint={<BoundTag bound={r.b.bound} memoryText="упор в чтение весов" />}
            swatch="var(--c-prefill)"
          />
          <Stat label="Скорость prefill" value={fmt.compact(tokPerSec)} unit="ток/с" hint="токенов промпта в секунду" />
          <Stat label="Доля attention" value={fmt.pct(attnShare)} hint="в FLOPs: растёт с длиной" />
          <Stat label="Граница memory / compute" value={r.cross !== null ? fmt.int(r.cross) : 'н/д'} unit="ток." hint="короче: упор в чтение весов" />
        </div>

        <LineChart
          series={[
            { id: 'compute', label: 'Вычисления', color: 'prefill', points: r.curve.map((c) => [c.n, c.compute]) },
            { id: 'memory', label: 'Чтение весов', color: 'decode', dashed: true, points: r.curve.map((c) => [c.n, c.memory]) },
          ]}
          xLabel="Длина промпта, токенов"
          yLabel="Время, мс"
          xScale="log"
          yScale="log"
          annotations={[
            { kind: 'x', x: n, label: fmt.tokens(n) },
            ...(r.cross !== null ? [{ kind: 'point' as const, x: r.cross, y: r.crossMs, label: `граница ≈ ${fmt.int(r.cross)}` }] : []),
          ]}
          height={240}
          xFormat={(v) => fmt.compact(v)}
          yFormat={(v) => fmt.compact(v)}
          aria-label="Время prefill от длины промпта: вычисления против чтения весов"
        />
        <TpNote hw={hw} />
      </div>
    </Figure>
  )
}
