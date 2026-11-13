import { useMemo, useState, type ReactNode } from 'react'
import { getInterconnect, KV_TRANSFER_LINK_IDS } from '@/engine/data/interconnects'
import {
  DEFAULT_MBU,
  DEFAULT_MFU,
  STEP_OVERHEAD_MS,
  decodeStepBreakdown,
  kvLayout,
  maxConcurrentSeqs,
  peakTflops,
  prefillFlopsBreakdown,
  prefillMs,
  ridgePoint,
  transferMs,
  weightBytes,
} from '@/engine/perf'
import { Figure, Formula, Select, Slider, fmt } from '@/ui'
import { DTYPE_LABEL, DtypeSegmented, GpuSelect, ModelSelect, TpNote, VLLM_MEM_UTIL, type HardwareState } from '@/chapters/prefill-decode/shared/hardware'
import { requestTiming } from '@/chapters/prefill-decode/shared/timing'
import { texBytes, texInt, texMs, texNum, texSci } from '@/chapters/prefill-decode/shared/tex'

const SEQ_FORMS = ['последовательность', 'последовательности', 'последовательностей'] as const

export function FormulasFigure({ state, userPrompt }: { state: HardwareState; userPrompt: number | null }) {
  const hw = state.hw
  const { model, gpu, dtype, tp } = hw
  const [promptRaw, setPrompt] = useState<number | null>(null)
  const prompt = promptRaw ?? userPrompt ?? 43
  const [answer, setAnswer] = useState(400)
  const [batch, setBatch] = useState(1)
  const [linkId, setLinkId] = useState('ib-ndr')

  const r = useMemo(() => {
    const L = kvLayout(model, 'bf16')
    const kvTok = L.bytesPerTokenNaive
    const total = prompt + answer
    const w = weightBytes(model, dtype)
    const peak = peakTflops(gpu, dtype)
    const ridge = ridgePoint(gpu, dtype)
    const ctx = prompt + Math.round(answer / 2)
    const ideal = decodeStepBreakdown(model, gpu, { ...hw.perf, batch, avgContext: ctx, mbu: 1, mfu: 1, overheadMs: 0 })
    // сколько последовательностей длины S вообще помещается в KV (как в калькуляторе decode главы 9)
    const maxSeqs = maxConcurrentSeqs(model, gpu, { contextLen: ctx, dtype, gpus: tp, memUtil: VLLM_MEM_UTIL })
    const fl = prefillFlopsBreakdown(model, prompt)
    const P = (model.activeMatmulParamsB ?? model.activeParamsB) * 1e9
    // LM head считает логиты только для последнего токена промпта
    const lm = model.lmHeadParamsB !== undefined ? model.lmHeadParamsB * 1e9 : model.vocab * model.hidden
    const preMs = prefillMs(model, gpu, prompt, hw.perf)
    const link = getInterconnect(linkId)
    const kvPrompt = kvTok * prompt
    const xfer = transferMs(kvPrompt, link.GBs)
    const t = requestTiming(hw, prompt, answer, 0)
    return { L, kvTok, total, w, peak, ridge, ctx, ideal, maxSeqs, fl, P, lm, preMs, link, kvPrompt, xfer, t }
  }, [model, gpu, dtype, tp, prompt, answer, batch, linkId, hw])

  const bw = gpu.bwTBs * tp
  // закон Литтла: с дробным числом «29,2 запроса», с целым по обычным формам
  const littleN = (10 * r.t.e2eMs) / 1000
  const littleWord = fmt.num(littleN).includes(',') ? 'запроса' : fmt.plural(Math.round(littleN), ['запрос', 'запроса', 'запросов'])
  const mla = model.attention === 'mla' && Boolean(model.mlaDim)

  return (
    <Figure
      id="cheatsheet-figure-formulas"
      title="Формулы на ваших числах"
      width="wide"
      controlsPosition="top"
      controls={
        <>
          <div className="col-span-full grid gap-x-6 gap-y-4 sm:grid-cols-3">
            <ModelSelect state={state} />
            <GpuSelect state={state} />
            <DtypeSegmented state={state} />
          </div>
          <div className="col-span-full grid gap-x-6 gap-y-4 sm:grid-cols-2 @3xl/fig:grid-cols-4">
            <Slider
              label="Промпт"
              value={prompt}
              min={16}
              max={32768}
              scale="log"
              onChange={setPrompt}
              format={(v) => fmt.tokens(v)}
              hint={
                promptRaw !== null && userPrompt !== null ? (
                  <button type="button" className="text-accent underline decoration-dotted underline-offset-2" onClick={() => setPrompt(null)}>
                    вернуть ваш запрос ({fmt.tokens(userPrompt)})
                  </button>
                ) : (
                  'ваш запрос после шаблона'
                )
              }
            />
            <Slider label="Ответ" value={answer} min={1} max={4096} scale="log" onChange={setAnswer} format={(v) => fmt.tokens(v)} />
            <Slider label="Batch в decode" value={batch} min={1} max={256} scale="log" onChange={setBatch} format={(v) => fmt.int(v)} />
            <Select
              label="Канал для передачи KV"
              value={linkId}
              onChange={setLinkId}
              options={KV_TRANSFER_LINK_IDS.map((id) => ({ value: id, label: linkLabel(id) }))}
            />
          </div>
        </>
      }
      caption="Попробуйте: поставьте промпт 32 000 токенов и посмотрите, как меняются KV на запрос, prefill и передача KV. Потом поднимите batch и сравните нижнюю границу шага decode с batch 1."
      source={`Ridge point, нижняя граница шага decode и передача KV идеальные: 100 % полосы, без накладных. Prefill, время ответа и закон Литтла реалистичные, как в главах 8-10: ${fmt.pct(DEFAULT_MBU)} полосы, ${fmt.pct(DEFAULT_MFU)} пика, ${fmt.ms(STEP_OVERHEAD_MS)} накладных на шаг.`}
    >
      <div className="grid gap-3 @2xl:grid-cols-2">
        <Card
          title="KV cache на токен"
          general={mla ? 'L \\cdot d_{latent} \\cdot b' : '2 \\cdot L \\cdot h_{kv} \\cdot d_{head} \\cdot b'}
          numbers={
            mla
              ? `${model.layers} \\cdot ${r.L.valuesPerLayerPerToken} \\cdot ${texNum(r.L.bytesPerElem)} = ${texInt(r.kvTok)}\\ \\text{Б}`
              : `2 \\cdot ${model.layers} \\cdot ${model.kvHeads} \\cdot ${model.headDim} \\cdot ${texNum(r.L.bytesPerElem)} = ${texInt(r.kvTok)}\\ \\text{Б}`
          }
          note={
            <>
              {fmt.bytes(r.kvTok, { mode: 'binary' })} на токен в BF16 ({mla ? 'MLA хранит один сжатый вектор на слой' : `${model.kvHeads} KV-голов, GQA`}).
              Ваш запрос с ответом ({fmt.tokens(r.total)}) держит {fmt.bytes(r.kvTok * r.total)} KV.
            </>
          }
        />
        <Card
          title="Веса в памяти"
          general="P_{\text{все}} \cdot b_{\text{вес}}"
          numbers={`${texSci(model.paramsB * 1e9)} \\cdot ${texNum(r.w / (model.paramsB * 1e9))} = ${texBytes(r.w)}`}
          note={
            <>
              {model.name} в {DTYPE_LABEL[dtype]}. FP8 занимает 1 байт на параметр, BF16 два: квантизация вдвое уменьшает и память, и чтение на каждом шаге
              decode.
            </>
          }
        />
        <Card
          title="Ridge point"
          general="\frac{\text{пик FLOPS}}{\text{полоса памяти}}"
          numbers={`\\frac{${texInt(r.peak)}\\ \\text{TFLOPS}}{${texNum(gpu.bwTBs)}\\ \\text{ТБ/с}} \\approx ${texInt(r.ridge)}\\ \\text{FLOP/байт}`}
          note={<>Операция с интенсивностью ниже этого числа упирается в память. Decode при batch 1 делает около 1 FLOP на байт.</>}
        />
        <Card
          title="Нижняя граница шага decode"
          general="t_{\text{шаг}} \ge \frac{W + B \cdot S \cdot \text{KV}_{\text{ток}}}{\text{полоса}}"
          numbers={`\\frac{${texBytes(r.ideal.weightBytes)} + ${batch} \\cdot ${texInt(r.ctx)} \\cdot ${texInt(r.kvTok)}\\ \\text{Б}}{${texNum(bw)}\\ \\text{ТБ/с}} = ${texMs(r.ideal.memoryMs)}`}
          numbersNarrow={`\\begin{aligned} &\\frac{${texBytes(r.ideal.weightBytes)} + ${batch} \\cdot ${texInt(r.ctx)} \\cdot ${texInt(r.kvTok)}\\ \\text{Б}}{${texNum(bw)}\\ \\text{ТБ/с}} \\\\ &= ${texMs(r.ideal.memoryMs)} \\end{aligned}`}
          note={
            <>
              {batch > 1
                ? `Не больше ${fmt.int(1000 / r.ideal.ms)} ток/с на пользователя и ${fmt.int((1000 * batch) / r.ideal.ms)} ток/с на весь batch из ${fmt.int(batch)}`
                : `Не больше ${fmt.int(1000 / r.ideal.ms)} ток/с на пользователя`}
              {r.ideal.bound === 'compute' ? '; здесь шаг уже упирается в вычисления' : ''}. S: средняя длина последовательности, {fmt.tokens(r.ctx)}.
              {batch > r.maxSeqs && (
                <>
                  {' '}
                  <span className="font-medium text-ink">Не влезает в память:</span> при S = {fmt.tokens(r.ctx)} в KV{' '}
                  {r.maxSeqs > 0 ? `помещается максимум ${fmt.count(r.maxSeqs, SEQ_FORMS)}` : 'не помещается ни одной последовательности'}, такой batch
                  движок не соберёт.
                </>
              )}
            </>
          }
        />
        <Card
          title="Prefill"
          general="\text{FLOPs} \approx 2 \cdot P \cdot N + 2 \cdot P_{\text{LM}}\,(+\,\text{attention}),\quad t \approx \frac{\text{FLOPs}}{\text{пик} \cdot \text{MFU}}"
          generalNarrow="\begin{aligned} \text{FLOPs} &\approx 2 \cdot P \cdot N + 2 \cdot P_{\text{LM}} \\ &\quad (+\,\text{attention}) \\ t &\approx \frac{\text{FLOPs}}{\text{пик} \cdot \text{MFU}} \end{aligned}"
          numbers={`2 \\cdot ${texSci(r.P - r.lm)} \\cdot ${texInt(prompt)} + ${texSci(2 * r.lm)} + ${texSci(r.fl.attention)} \\approx ${texSci(r.fl.total)}\\ \\text{FLOP}`}
          numbersNarrow={`\\begin{aligned} &2 \\cdot ${texSci(r.P - r.lm)} \\cdot ${texInt(prompt)} \\\\ &+ ${texSci(2 * r.lm)} + ${texSci(r.fl.attention)} \\\\ &\\approx ${texSci(r.fl.total)}\\ \\text{FLOP} \\end{aligned}`}
          note={
            <>
              {r.t.prefillBound === 'memory'
                ? `Около ${fmt.ms(r.preMs)} на ${tp > 1 ? `${tp} × ` : ''}${gpu.name}: промпт короткий, поэтому время задаёт чтение весов (${fmt.pct(DEFAULT_MBU)} полосы), а не счёт.`
                : `При MFU ${fmt.pct(DEFAULT_MFU)}: около ${fmt.ms(r.preMs)} на ${tp > 1 ? `${tp} × ` : ''}${gpu.name}.`}{' '}
              P: параметры, которые умножаются на каждый токен промпта; LM head считает логиты только для последнего токена.
            </>
          }
        />
        <Card
          title="Передача KV (при дезагрегации)"
          general="t = \frac{N \cdot \text{KV}_{\text{ток}}}{\text{полоса канала}}"
          numbers={`\\frac{${texInt(prompt)} \\cdot ${texInt(r.kvTok)}\\ \\text{Б}}{${texNum(r.link.GBs)}\\ \\text{ГБ/с}} = ${texMs(r.xfer)}`}
          numbersNarrow={`\\begin{aligned} &\\frac{${texInt(prompt)} \\cdot ${texInt(r.kvTok)}\\ \\text{Б}}{${texNum(r.link.GBs)}\\ \\text{ГБ/с}} \\\\ &= ${texMs(r.xfer)} \\end{aligned}`}
          note={
            <>
              {fmt.bytes(r.kvPrompt)} KV промпта по каналу «{r.link.name}». Сравните с prefill того же промпта: {fmt.ms(r.preMs)}.
            </>
          }
        />
        <Card
          title="Время ответа"
          general="\text{E2E} \approx \text{TTFT} + \text{ITL} \cdot (N_{\text{out}} - 1)"
          numbers={`${texMs(r.t.ttftMs)} + ${texNum(r.t.itlMs)}\\ \\text{мс} \\cdot ${texInt(answer - 1)} = ${texMs(r.t.e2eMs)}`}
          numbersNarrow={`\\begin{aligned} &${texMs(r.t.ttftMs)} + ${texNum(r.t.itlMs)}\\ \\text{мс} \\cdot ${texInt(answer - 1)} \\\\ &= ${texMs(r.t.e2eMs)} \\end{aligned}`}
          note={<>Один запрос на свободном GPU, без сети и очереди. Под нагрузкой оба слагаемых растут.</>}
        />
        <Card
          title="Закон Литтла"
          general="L = \lambda \cdot W"
          numbers={`10\\ \\text{запр/с} \\cdot ${texNum(r.t.e2eMs / 1000)}\\ \\text{с} = ${texNum((10 * r.t.e2eMs) / 1000)}\\ \\text{${littleWord}}`}
          note={<>Сколько запросов одновременно в работе при 10 запросах в секунду и таком E2E. Это минимум слотов в batch на всех репликах.</>}
        />
      </div>
      <div className="mt-3">
        <TpNote hw={hw} />
      </div>
    </Figure>
  )
}

function Card({
  title,
  general,
  generalNarrow,
  numbers,
  numbersNarrow,
  note,
}: {
  title: string
  general: string
  /** Та же формула по строкам для узкой карточки (телефон): показывается, когда строка не влезает. */
  generalNarrow?: string
  numbers: string
  numbersNarrow?: string
  note: ReactNode
}) {
  return (
    <section className="grid min-w-0 content-start gap-2 rounded-panel border border-line p-3.5">
      <h4 className="text-[15px] font-semibold text-ink">{title}</h4>
      <div className="min-w-0 py-0.5">
        <Formula block tex={general} narrowTex={generalNarrow} />
      </div>
      <div className="min-w-0 rounded-control bg-surface-2 px-2 py-1 text-[0.95em]">
        <Formula block tex={numbers} narrowTex={numbersNarrow} />
      </div>
      <p className="text-[13px] leading-snug text-ink-2">{note}</p>
    </section>
  )
}

/** Короткая подпись канала для выпадающего списка: без пояснения в скобках и гигабит, но с полосой в ГБ/с. */
function linkLabel(id: string): string {
  const l = getInterconnect(id)
  const name = l.name
    .replace(/\s*\(.*\)\s*/, '')
    .replace(/\s+\d+\s*Гбит\/с/, '')
    .trim()
  return `${name}, ${fmt.num(l.GBs, l.GBs < 10 ? 1 : 0)}\u00a0ГБ/с`
}
