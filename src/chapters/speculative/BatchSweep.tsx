import { useMemo, useState } from 'react'
import { VLLM_MEM_UTIL, maxConcurrentSeqs } from '@/engine/perf'
import { Figure, LineChart, Segmented, Slider, fmt, mix } from '@/ui'
import { useRunningExample } from './runningExample'
import { batchSpeedupSweep } from './spec'

const KS = [2, 4, 8] as const
/** k это порядковая величина: одна гамма цвета decode, от светлого к тёмному, а не разные сущности. */
const COLORS: Record<(typeof KS)[number], string> = {
  2: mix('decode', 55),
  4: 'decode',
  8: 'color-mix(in oklab, var(--c-decode) 70%, var(--ink))',
}

export function BatchSweep() {
  const { model, gpu, fallback } = useRunningExample()
  const [alpha, setAlpha] = useState(0.8)
  const [c, setC] = useState(0.05)
  const [ctx, setCtx] = useState<number>(512)

  const data = useMemo(() => {
    const maxSeqs = Math.max(1, maxConcurrentSeqs(model, gpu, { contextLen: ctx, memUtil: VLLM_MEM_UTIL }))
    const batches: number[] = []
    for (let b = 1; b <= Math.min(1024, maxSeqs); b = Math.ceil(b * 1.25)) batches.push(b)
    if (batches[batches.length - 1] !== Math.min(1024, maxSeqs)) batches.push(Math.min(1024, maxSeqs))
    const series = KS.map((k) => {
      const pts = batchSpeedupSweep(model, gpu, { alpha, k, c, avgContext: ctx, batches })
      return { k, pts }
    })
    const summary = series.map(({ k, pts }) => {
      const peak = pts[0]?.speedup ?? 1
      // последний batch, где ускорение ещё почти полное (не ниже 98 % от batch 1), и где оно опускается ниже 1
      const full = pts.filter((p) => p.speedup >= 0.98 * peak).pop()?.batch ?? 1
      return { k, full, breakEven: pts.find((p) => p.speedup < 1)?.batch ?? null }
    })
    return { maxSeqs, series, summary }
  }, [model, gpu, alpha, c, ctx])

  const k4 = data.summary.find((b) => b.k === 4)!

  return (
    <Figure
      id="speculative-figure-batch"
      title={`Ускорение от спекуляции при разном batch: ${model.name} на ${gpu.name}`}
      controls={
        <>
          <Slider label="Acceptance rate α" value={alpha} min={0.4} max={0.95} step={0.05} onChange={setAlpha} format={(x) => fmt.num(x, 2)} />
          <Slider label="Цена черновика c" value={c} min={0} max={0.3} step={0.01} onChange={setC} format={(x) => fmt.num(x, 2)} />
          <Segmented
            label="Контекст запроса"
            value={ctx}
            onChange={setCtx}
            options={[
              { value: 512, label: '512' },
              { value: 2048, label: '2K' },
              { value: 8192, label: '8K' },
            ]}
          />
        </>
      }
      caption="Попробуйте: при контексте 512 найдите, где кривая k = 8 опускается ниже 1. Проверка k черновиков стоит как prefill-кусок из k + 1 токенов на каждую последовательность: пока шаг упирается в память, это почти бесплатно, а когда batch загрузил вычисления, проверка дорожает в k + 1 раз. При длинном контексте шаг дольше упирается в чтение KV, и запас вычислений держится дольше."
      source={`Идеализация: roofline движка с MFU 0,5, чистый decode без подмешанного prefill, цена черновика всегда c от шага target; в реальных движках выигрыш тает при заметно меньшем batch. Ось обрезана на ${fmt.int(data.maxSeqs)} последовательностях: дальше не влезает KV.${
        fallback === 'moe'
          ? ' В симуляторе выбрана MoE-модель, поэтому здесь, как и в первой фигуре главы, плотная Qwen3-8B.'
          : fallback === 'fit'
            ? ' Модель из симулятора не помещается в одну GPU, поэтому здесь Qwen3-8B.'
            : ''
      }`}
    >
      <LineChart
        series={data.series.map(({ k, pts }) => ({
          id: `k${k}`,
          label: `k = ${k}`,
          color: COLORS[k],
          points: pts.map((p) => [p.batch, p.speedup] as [number, number]),
        }))}
        xLabel="Последовательностей в batch"
        yLabel="Ускорение, ×"
        xScale="log"
        height={260}
        xFormat={(v) => fmt.int(v)}
        yFormat={(v) => fmt.num(v, 1)}
        annotations={[{ kind: 'y', y: 1, label: 'без спекуляции', labelAlign: 'start' }]}
        aria-label="Ускорение спекуляции в зависимости от размера batch для k = 2, 4, 8"
      />
      <p className="mt-3 text-[14px] leading-snug text-ink-2">
        {k4.breakEven !== null
          ? `В этой идеальной модели при k = 4 полный выигрыш держится до ${fmt.count(k4.full, ['последовательности', 'последовательностей', 'последовательностей'])} и пропадает к ${fmt.int(k4.breakEven)}. Это верхняя оценка: в реальных движках выигрыш тает раньше и постепенно. В статье EAGLE-3 (Llama-3.1-8B, SGLang, одна H100, черновик из 3 токенов) ускорение throughput 1,81× при batch 2, 1,48× при 16 и 1,38× при 64. Смотрите на форму кривой, а не на точное число.`
          : `В идеальной модели при k = 4 и этом контексте спекуляция окупается во всём диапазоне, который влезает в память (до ${fmt.count(data.maxSeqs, ['последовательности', 'последовательностей', 'последовательностей'])}); в реальных движках выигрыш с ростом batch всё равно тает.`}
      </p>
    </Figure>
  )
}
