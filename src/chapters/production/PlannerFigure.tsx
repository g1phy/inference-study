import { useMemo, useState } from 'react'
import { WarningIcon } from '@phosphor-icons/react'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { findModel, getModel } from '@/engine/data/models'
import { fitsCheck } from '@/engine/perf'
import { useSettings } from '@/state/settings'
import { Figure, Slider, Stat, fmt, tint } from '@/ui'
import { planPools, type PoolPlan } from './planner'

const TTFT_SLA = 500
const ITL_SLA = 50

export function PlannerFigure() {
  const [settings] = useSettings()
  const { model, gpu } = useMemo(() => {
    const m = findModel(settings.modelId) ?? getModel('qwen3-8b')
    const g = findGpu(settings.gpuId) ?? getGpu('h100-sxm')
    // модель, которая не влезает даже в 8 GPU, заменяем на Qwen3-8B
    return fitsCheck(m, g, { contextLen: 4096, gpus: 8 }).fits ? { model: m, gpu: g } : { model: getModel('qwen3-8b'), gpu: g }
  }, [settings.modelId, settings.gpuId])
  const [rps, setRps] = useState(20)
  const [isl, setIsl] = useState(2048)
  const [osl, setOsl] = useState(300)
  const plan = useMemo(() => planPools(model, gpu, { rps, isl, osl, ttftMs: TTFT_SLA, itlMs: ITL_SLA }), [model, gpu, rps, isl, osl])
  const gpusP = plan.prefill.replicas * plan.prefill.gpusPerEngine
  const gpusD = plan.decode.replicas * plan.decode.gpusPerEngine

  return (
    <Figure
      id="production-figure-planner"
      title={`Prefill и decode масштабируются отдельно: ${model.name} на ${gpu.name}`}
      controls={
        <>
          <Slider label="Запросов в секунду" value={rps} min={1} max={100} onChange={setRps} format={(v) => fmt.int(v)} />
          <Slider label="Средняя длина промпта" value={isl} min={256} max={32768} step={64} scale="log" onChange={setIsl} format={(v) => fmt.tokens(v)} />
          <Slider label="Средняя длина ответа" value={osl} min={32} max={4096} step={1} scale="log" onChange={setOsl} format={(v) => fmt.tokens(v)} />
        </>
      }
      caption="Попробуйте: увеличьте длину промпта до 16 тысяч токенов. Сильнее всего растёт prefill-пул, но растёт и decode-пул: каждый ответ держит в памяти KV всего промпта, поэтому batch decode-реплики падает в разы. Затем верните промпт и увеличьте длину ответа: растёт только decode-пул."
      source={`Расчёт по формулам Dynamo Planner v1.0 с целями TTFT ${fmt.int(TTFT_SLA)} мс и ITL ${fmt.int(ITL_SLA)} мс (его значения по умолчанию; в текущем Planner они действуют в режиме sla). Производительность реплик из roofline-модели движка, без очередей и сети.`}
    >
      <div className="grid min-w-0 gap-5">
        <div className="grid min-w-0 gap-4 @2xl:grid-cols-2">
          <Pool title="Prefill-пул" role="prefill" plan={plan.prefill} gpus={gpusP} unit="промпт" ctx={isl + osl} />
          <Pool title="Decode-пул" role="decode" plan={plan.decode} gpus={gpusD} unit="ответ" ctx={isl + osl} />
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
          <Stat label="Нагрузка prefill" value={fmt.compact(plan.prefill.loadTokS)} unit="ток/с" swatch="var(--c-prefill)" hint="токенов промпта" />
          <Stat label="Нагрузка decode" value={fmt.compact(plan.decode.loadTokS)} unit="ток/с" swatch="var(--c-decode)" hint="токенов ответа" />
          <Stat label="GPU всего" value={fmt.int(gpusP + gpusD)} hint={`реплик: ${plan.prefill.replicas} prefill + ${plan.decode.replicas} decode`} />
          <Stat label="TP реплики" value={fmt.int(plan.decode.gpusPerEngine)} hint="веса не больше половины памяти" />
        </div>
      </div>
    </Figure>
  )
}

function Pool({ title, role, plan, gpus, unit, ctx }: { title: string; role: 'prefill' | 'decode'; plan: PoolPlan; gpus: number; unit: string; ctx: number }) {
  const shown = Math.min(gpus, 96)
  // соседние реплики чередуют насыщенность, только если в реплике несколько GPU: так видно, где кончается реплика
  const split = plan.gpusPerEngine > 1
  return (
    <div className="grid min-w-0 content-start gap-3 rounded-panel p-4 shadow-[inset_0_0_0_1px_var(--line)]" style={{ background: tint(role, 6) }}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="inline-flex items-center gap-2 text-[15px] font-semibold text-ink">
          <span className="h-3 w-3 rounded-[3px]" style={{ background: `var(--c-${role})` }} aria-hidden />
          {title}
        </span>
        <span className="text-[14px] text-ink-2">
          {fmt.count(plan.replicas, ['реплика', 'реплики', 'реплик'])} × {plan.gpusPerEngine} GPU
        </span>
      </div>
      <div className="flex flex-wrap gap-1" role="img" aria-label={`${title}: ${gpus} GPU`}>
        {Array.from({ length: shown }, (_, i) => (
          <span key={i} className="h-4 w-4 rounded-[4px]" style={{ background: `var(--c-${role})`, opacity: split && Math.floor(i / plan.gpusPerEngine) % 2 === 1 ? 0.6 : 1 }} />
        ))}
        {gpus > shown && <span className="self-center text-[12px] text-ink-3">ещё {gpus - shown}</span>}
      </div>
      <p className="text-[13px] leading-snug text-ink-2">
        {role === 'prefill'
          ? `Реплика считает ${fmt.count(plan.batch, ['промпт', 'промпта', 'промптов'])} за шаг ${fmt.ms(plan.stepMs)}: ${fmt.compact(plan.engineTokS)} ток/с.`
          : `Реплика держит batch ${fmt.int(plan.batch)} при ITL ${fmt.ms(plan.stepMs)}: ${fmt.compact(plan.engineTokS)} ток/с. ` +
            (plan.limit === 'kv'
              ? `Больше не помещается: каждый ответ держит в памяти KV своих ${fmt.tokens(ctx)} (промпт и ответ).`
              : `Больше нельзя: шаг вышел бы за ITL ${fmt.int(ITL_SLA)} мс.`)}
      </p>
      {plan.slaMissed && (
        <p className="flex gap-1.5 text-[13px] leading-snug text-ink">
          <WarningIcon size={16} weight="fill" className="mt-0.5 shrink-0 text-warn" aria-hidden />
          {role === 'prefill'
            ? `Один ${unit} такой длины считается дольше ${fmt.int(TTFT_SLA)} мс даже в одиночку: нужно больше GPU на реплику или мягче цель TTFT.`
            : `Даже один ${unit} не укладывается в ITL ${fmt.int(ITL_SLA)} мс: нужно больше GPU на реплику.`}
        </p>
      )}
    </div>
  )
}
