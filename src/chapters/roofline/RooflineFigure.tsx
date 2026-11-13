import { useMemo, useState, type ReactNode } from 'react'
import { CpuIcon, MemoryIcon } from '@phosphor-icons/react'
import { arithmeticIntensity, attainableTflops, decodeAttentionIntensity, kvBytesPerToken, maxConcurrentSeqs, peakTflops, ridgePoint } from '@/engine/perf'
import { Figure, Slider, StatusIcon, TOKEN_FORMS, fmt, cssVar } from '@/ui'
import { DTYPE_LABEL, DtypeSegmented, GpuSelect, ModelSelect, VLLM_MEM_UTIL, type HardwareState } from '@/chapters/prefill-decode/shared/hardware'
import { RooflineChart, type RooflinePoint } from './RooflineChart'

interface Calc {
  flops: number
  bytes: number
  ai: number
  attain: number
  share: number
  bound: 'memory' | 'compute'
  /** Идеальное время шага на одном GPU: max(байты / полоса, FLOPs / пик), мс. */
  ms: number
}

const SEQ_FORMS = ['последовательность', 'последовательности', 'последовательностей'] as const

export function RooflineFigure({ state, userPrompt }: { state: HardwareState; userPrompt: number | null }) {
  const { model, gpu, dtype, tp } = state.hw
  const [batch, setBatch] = useState(1)
  const [ctx, setCtx] = useState(1024)
  const [prefillN, setPrefillN] = useState(2048)

  const r = useMemo(() => {
    const peak = peakTflops(gpu, dtype)
    const ridge = ridgePoint(gpu, dtype)
    const calc = (flops: number, bytes: number): Calc => {
      const ai = flops / bytes
      const attain = attainableTflops(gpu, dtype, ai)
      const memMs = (bytes / (gpu.bwTBs * 1e12)) * 1000
      const compMs = (flops / (peak * 1e12)) * 1000
      return { flops, bytes, ai, attain, share: attain / peak, bound: ai < ridge ? 'memory' : 'compute', ms: Math.max(memMs, compMs) }
    }
    const d = arithmeticIntensity(model, { phase: 'decode', batch, avgContext: ctx, dtype })
    const p = arithmeticIntensity(model, { phase: 'prefill', tokens: prefillN, dtype })
    const kvSeq = kvBytesPerToken(model, 'bf16') * ctx
    const attnAi = decodeAttentionIntensity(model, 'bf16')
    // attention decode: на каждую последовательность читается её KV целиком; FLOPs = AI × байты
    const attn = calc(attnAi * kvSeq * batch, kvSeq * batch)
    // сколько последовательностей такого контекста вообще помещается в KV (на всех GPU TP)
    const maxSeqs = maxConcurrentSeqs(model, gpu, { contextLen: ctx, dtype, gpus: tp, memUtil: VLLM_MEM_UTIL })
    return { peak, ridge, decode: calc(d.flops, d.bytes), prefill: calc(p.flops, p.bytes), attn, kvSeq, attnAi, maxSeqs }
  }, [model, gpu, dtype, tp, batch, ctx, prefillN])

  const peakText = `${fmt.int(r.peak)} TFLOPS`
  const bwText = fmt.bandwidth(gpu.bwTBs * 1000)
  const fits = batch <= r.maxSeqs
  const misfit = fits ? null : `не влезает в память: при контексте ${fmt.tokens(ctx)} максимум ${fmt.count(r.maxSeqs, SEQ_FORMS)}`

  const points: RooflinePoint[] = [
    {
      id: 'decode',
      label: `decode, batch ${fmt.int(batch)}`,
      shortLabel: 'decode',
      ai: r.decode.ai,
      role: 'decode',
      impossible: !fits,
      tip: (
        <CalcTip
          title={`Шаг decode: batch ${fmt.int(batch)}, контекст ${fmt.tokens(ctx)}`}
          c={r.decode}
          peakText={peakText}
          bwText={bwText}
          lines={[`FLOPs ≈ batch × 2 × параметры + attention = ${fmt.flops(r.decode.flops)}`, `байты = веса + KV всех последовательностей = ${fmt.bytes(r.decode.bytes)}`]}
          warning={misfit}
        />
      ),
      ariaLabel: `Decode, batch ${batch}: ${fmt.num(r.decode.ai)} FLOP на байт, достижимо ${fmt.num(r.decode.attain)} TFLOPS${misfit ? `; ${misfit}` : ''}`,
    },
    {
      id: 'attn',
      label: 'attention в decode',
      shortLabel: 'attention',
      ai: r.attn.ai,
      role: 'decode',
      hollow: true,
      tip: (
        <CalcTip
          title="Attention в шаге decode"
          c={r.attn}
          peakText={peakText}
          bwText={bwText}
          lines={[
            `каждая последовательность читает свой KV: ${fmt.bytes(r.kvSeq)}`,
            `≈ ${fmt.num(r.attnAi)} FLOP на байт KV при любом batch (${model.heads} Q-голов на ${model.kvHeads} KV-голов)`,
          ]}
        />
      ),
      ariaLabel: `Attention в decode: ${fmt.num(r.attn.ai)} FLOP на байт при любом batch`,
    },
    {
      id: 'prefill',
      label: `prefill, ${fmt.tokens(prefillN)}`,
      shortLabel: 'prefill',
      ai: r.prefill.ai,
      role: 'prefill',
      tip: (
        <CalcTip
          title={`Prefill промпта из ${fmt.tokens(prefillN)}`}
          c={r.prefill}
          peakText={peakText}
          bwText={bwText}
          lines={[`FLOPs ≈ 2 × параметры × токены + attention = ${fmt.flops(r.prefill.flops)}`, `байты ≈ веса один раз + KV = ${fmt.bytes(r.prefill.bytes)}`]}
        />
      ),
      ariaLabel: `Prefill ${fmt.tokens(prefillN)}: ${fmt.num(r.prefill.ai)} FLOP на байт, достижимо ${fmt.num(r.prefill.attain)} TFLOPS`,
    },
  ]

  return (
    <Figure
      id="roofline-figure-roofline"
      title={`Roofline: ${gpu.name}, ${DTYPE_LABEL[dtype]}, ${model.name}`}
      width="wide"
      controls={
        <div className="col-span-full grid gap-x-6 gap-y-4 sm:grid-cols-2 @3xl/fig:grid-cols-4">
          <GpuSelect state={state} />
          <DtypeSegmented state={state} label="Формат весов и вычислений" />
          <ModelSelect state={state} />
          <Slider label="Контекст в decode" value={ctx} min={64} max={32768} scale="log" onChange={setCtx} format={(v) => fmt.tokens(v)} hint="KV каждой последовательности" />
        </div>
      }
      caption="Попробуйте: двигайте batch от 1 до 512 и смотрите, как точка decode едет вправо, а полая точка attention стоит на месте. Потом поставьте в prefill ваш запрос: короткий промпт тоже оказывается слева от излома. Наведите или сфокусируйтесь на точке, чтобы увидеть расчёт."
    >
      <div className="grid gap-4">
        <RooflineChart gpu={gpu} dtype={dtype} points={points} />
        {/* главный эксперимент: слайдеры сразу под графиком, который они двигают */}
        <div className="grid gap-x-6 gap-y-4 border-y border-line py-4 sm:grid-cols-2">
          <Slider label="Batch в decode" value={batch} min={1} max={1024} scale="log" onChange={setBatch} format={(v) => fmt.int(v)} hint="сколько последовательностей в одном шаге" />
          <Slider
            label="Токенов в prefill"
            value={prefillN}
            min={16}
            max={32768}
            scale="log"
            onChange={setPrefillN}
            format={(v) => fmt.tokens(v)}
            hint={
              userPrompt !== null ? (
                <button type="button" className="text-accent underline decoration-dotted underline-offset-2" onClick={() => setPrefillN(userPrompt)}>
                  ваш запрос: {fmt.tokens(userPrompt)}
                </button>
              ) : undefined
            }
          />
        </div>
        <div className="grid grid-cols-3 gap-2 @2xl:gap-3">
          <ResultCard role="decode" title={`Decode, batch ${fmt.int(batch)}`} c={r.decode} warning={misfit} />
          <ResultCard role="decode" hollow title="Attention в decode" c={r.attn} />
          <ResultCard role="prefill" title={`Prefill, ${fmt.int(prefillN)}`} titleUnit={fmt.plural(prefillN, TOKEN_FORMS)} c={r.prefill} />
        </div>
      </div>
    </Figure>
  )
}

function CalcTip({
  title,
  c,
  lines,
  peakText,
  bwText,
  warning,
}: {
  title: string
  c: Calc
  lines: string[]
  peakText: string
  bwText: string
  warning?: string | null
}) {
  return (
    <div className="grid gap-1">
      <div className="font-medium text-ink">{title}</div>
      {lines.map((l) => (
        <div key={l} className="text-ink-2">
          {l}
        </div>
      ))}
      <div className="text-ink-2">
        интенсивность = {fmt.num(c.ai)} FLOP/байт
      </div>
      <div className="text-ink-2">
        достижимо = min({peakText}; {fmt.num(c.ai)} × {bwText}) = <span className="font-medium text-ink">{fmt.num(c.attain)} TFLOPS</span> (
        {fmt.pct(c.share)} пика)
      </div>
      <div className="text-ink-3">
        идеальное время ≥ {fmt.ms(c.ms)}, {c.bound === 'memory' ? 'упор в память' : 'упор в вычисления'}
      </div>
      {warning && (
        <div className="flex items-start gap-1.5 text-ink-2">
          <StatusIcon status="warn" className="mt-0.5" />
          <span>{warning}</span>
        </div>
      )}
    </div>
  )
}

function ResultCard({
  role,
  hollow = false,
  title,
  titleUnit,
  c,
  warning,
}: {
  role: 'prefill' | 'decode'
  hollow?: boolean
  title: string
  /** Единица после числа в заголовке («токенов»): только на широкой карточке, на узкой она не влезает. */
  titleUnit?: string
  c: Calc
  warning?: string | null
}) {
  const Icon = c.bound === 'memory' ? MemoryIcon : CpuIcon
  return (
    <div className="grid min-w-0 content-start gap-1 rounded-control border border-line px-2 py-2 @2xl:gap-1.5 @2xl:px-3 @2xl:py-2.5">
      <div className="flex min-w-0 items-center gap-1.5 text-[12px] leading-tight text-ink-2 @2xl:gap-2 @2xl:text-sm">
        <span
          aria-hidden
          className="h-2.5 w-2.5 shrink-0 rounded-full"
          style={
            warning
              ? { border: `2px dashed ${cssVar(role)}` }
              : hollow
                ? { border: `2px solid ${cssVar(role)}` }
                : { background: cssVar(role) }
          }
        />
        <span className="min-w-0 break-normal @2xl:truncate">
          {title}
          {titleUnit && <span className="hidden @2xl:inline">{` ${titleUnit}`}</span>}
        </span>
      </div>
      <div className="flex flex-wrap items-baseline gap-x-1.5 text-ink">
        <span className="text-[1.125rem] leading-tight font-semibold @2xl:text-[1.375rem]">{fmt.num(c.ai)}</span>
        <span className="text-[12px] text-ink-2 @2xl:text-[14px]">FLOP/байт</span>
      </div>
      <div className="hidden text-[13px] leading-snug text-ink-2 @2xl:block">
        {fmt.num(c.attain)} TFLOPS, {fmt.pct(c.share)} пика
      </div>
      <Verdict icon={<Icon size={16} />} text={c.bound === 'memory' ? 'упор в память' : 'упор в вычисления'} />
      {warning && (
        <div className="flex items-start gap-1.5 text-[12px] leading-snug text-ink-2 @2xl:text-[13px]">
          <StatusIcon status="warn" className="mt-px" />
          <span className="@2xl:hidden">не влезает в память</span>
          <span className="hidden @2xl:inline">{warning}</span>
        </div>
      )}
    </div>
  )
}

function Verdict({ icon, text }: { icon: ReactNode; text: string }) {
  return (
    <div className="inline-flex items-start gap-1.5 text-[12px] leading-snug font-medium text-ink @2xl:items-center @2xl:text-[13px]">
      <span aria-hidden className="mt-px text-ink-2 @2xl:mt-0">
        {icon}
      </span>
      {text}
    </div>
  )
}
