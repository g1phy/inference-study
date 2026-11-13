import { useDeferredValue, useMemo, useState } from 'react'
import { prefillMs } from '@/engine/perf'
import { useSettings } from '@/state/settings'
import { Figure, LineChart, Select, Slider, Stat, cssVar, fmt } from '@/ui'
import { BackToSim, useFollowSim } from '../kv-cache/followSim'
import { gpuOr, logGrid, minGpus, modelOptions, modelOr, pickModelId, shortGpuName } from '../kv-cache/lib'

// все модели симулятора здесь есть: фигура следует за его выбором, пока читатель не выберет свою
const EFFECT_MODELS = ['qwen3-8b', 'qwen3-32b', 'qwen3-30b-a3b', 'qwen3-235b-a22b', 'llama-3.1-70b'] as const

export function EffectFigure() {
  const [settings] = useSettings()
  const simModelId = pickModelId(settings.modelId, EFFECT_MODELS)
  const modelSel = useFollowSim(simModelId)
  const modelId = modelSel.id
  const [prompt, setPrompt] = useState(8192)
  const [prefixShare, setPrefixShare] = useState(0.9)
  const [hitRate, setHitRate] = useState(0.8)
  const model = modelOr(modelId)
  const gpu = gpuOr(settings.gpuId)
  const gpus = useMemo(() => minGpus(model, gpu), [model, gpu])
  const dShare = useDeferredValue(prefixShare)
  const dHit = useDeferredValue(hitRate)

  const series = useMemo(() => {
    const xs = logGrid(256, 131072, 40)
    const miss: Array<[number, number]> = []
    const hit: Array<[number, number]> = []
    const avg: Array<[number, number]> = []
    for (const n of xs) {
      const m = prefillMs(model, gpu, n, { gpus })
      const h = prefillMs(model, gpu, n, { gpus, cachedTokens: Math.floor(n * dShare) })
      miss.push([n, m])
      hit.push([n, h])
      avg.push([n, (1 - dHit) * m + dHit * h])
    }
    return [
      { id: 'miss', label: 'Промах: весь промпт в prefill', color: 'prefill', points: miss },
      { id: 'hit', label: 'Попадание: только хвост', color: 'good', points: hit },
      { id: 'avg', label: `В среднем при hit rate ${fmt.pct(dHit)}`, color: 'var(--ink-2)', dashed: true, points: avg },
    ]
  }, [model, gpu, gpus, dShare, dHit])

  const cached = Math.floor(prompt * prefixShare)
  const missMs = prefillMs(model, gpu, prompt, { gpus })
  const hitMs = prefillMs(model, gpu, prompt, { gpus, cachedTokens: cached })
  const avgMs = (1 - hitRate) * missMs + hitRate * hitMs
  const gpuSecMiss = (1000 * missMs * gpus) / 1000
  const gpuSecAvg = (1000 * avgMs * gpus) / 1000
  const gain = missMs / avgMs
  const gpuLabel = `${gpus > 1 ? `${gpus} × ` : ''}${shortGpuName(gpu)}`
  const ratioAt = (share: number) => prefillMs(model, gpu, 32768, { gpus }) / prefillMs(model, gpu, 32768, { gpus, cachedTokens: Math.floor(32768 * share) })
  const r99 = ratioAt(0.99)
  const r90 = ratioAt(0.9)
  const times = (r: number) => {
    const v = r < 10 ? Number(r.toFixed(1)) : Math.round(r)
    return `${fmt.num(v, r < 10 && !Number.isInteger(v) ? 1 : 0)} ${fmt.plural(v, ['раз', 'раза', 'раз'])}`
  }
  const ratio = missMs / hitMs
  const caption = `Попробуйте: уведите длину промпта к 300 токенам: линии сливаются, короткий prefill упирается в чтение весов. Затем поставьте 32 тысячи токенов и общий префикс 99 %: попадание быстрее в ${times(Math.round(r99))}, а при 90 % только в ${times(Math.round(r90))}.`

  return (
    <Figure
      id="prefix-caching-figure-effect"
      title="Что даёт prefix cache: TTFT и GPU-время на prefill"
      width="wide"
      controlsPosition="side"
      controls={
        <>
          <Select
            label="Модель"
            value={modelId}
            onChange={modelSel.set}
            options={modelOptions(EFFECT_MODELS)}
            hint={
              modelSel.overridden ? (
                <>
                  на {gpuLabel} (GPU из симулятора)
                  <br />
                  <BackToSim what="модель" name={modelOr(simModelId).name} onClick={modelSel.reset} />
                </>
              ) : (
                `на ${gpuLabel}, модель и GPU из симулятора`
              )
            }
          />
          <Slider
            label="Длина промпта"
            value={prompt}
            min={256}
            max={131072}
            scale="log"
            onChange={(v) => setPrompt(Math.round(v))}
            format={(v) => fmt.tokens(Math.round(v))}
          />
          <Slider
            label="Общий префикс"
            value={prefixShare}
            min={0}
            max={0.99}
            step={0.01}
            onChange={setPrefixShare}
            format={(v) => fmt.pct(v)}
            hint="какая доля промпта одинакова у запросов"
          />
          <Slider label="Hit rate" value={hitRate} min={0} max={1} step={0.01} onChange={setHitRate} format={(v) => fmt.pct(v)} hint="как часто префикс реально в кэше" />
        </>
      }
      caption={caption}
    >
      <div className="grid min-w-0 gap-5">
        <LineChart
          xLabel="Длина промпта, токенов"
          yLabel="TTFT без очереди"
          xScale="log"
          yScale="log"
          height={250}
          series={series}
          directLabels={false}
          annotations={[{ kind: 'x', x: prompt, label: fmt.compact(prompt) }]}
          xFormat={(v) => fmt.compact(v)}
          yFormat={(v) => fmt.ms(v).replace(/ /g, ' ')}
          aria-label="TTFT при промахе, попадании и в среднем в зависимости от длины промпта"
        />
        <div className="grid grid-cols-2 gap-x-6 gap-y-5 @2xl:grid-cols-3">
          <Stat label="TTFT при промахе" value={fmt.ms(missMs)} swatch={cssVar('prefill')} hint={fmt.tokens(prompt)} />
          <Stat
            label="TTFT при попадании"
            value={fmt.ms(hitMs)}
            swatch={cssVar('good')}
            hint={`из кэша ${fmt.int(cached)} · ${ratio < 1.15 ? 'почти как промах' : `в ${times(ratio)} быстрее промаха`}`}
          />
          <Stat label="Средний TTFT" value={fmt.ms(avgMs)} hint={`при hit rate ${fmt.pct(hitRate)}`} />
          <Stat label="GPU-секунды prefill" value={`${fmt.num(gpuSecMiss, gpuSecMiss < 10 ? 1 : 0)} → ${fmt.num(gpuSecAvg, gpuSecAvg < 10 ? 1 : 0)}`} hint={`на ${fmt.int(1000)} запросов: без кэша → с кэшем`} />
          <Stat
            label="Ёмкость prefill"
            value={fmt.multiplier(gain, 1)}
            hint="во столько раз больше запросов на те же GPU"
            className="col-span-2 @2xl:col-span-1"
          />
        </div>
      </div>
    </Figure>
  )
}
